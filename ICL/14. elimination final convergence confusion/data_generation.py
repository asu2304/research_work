"""
Generate the EQUIPROBABLE-successor variant of the folder-12 dataset.

Same task, same vocab, same sequence length, same sizes, same thresholds as folder 12's
`data_generation.py`. The single change:

    folder 12 : each ruled state's two successors split p1 ~ U(0.1, 0.9),  p2 = 1 - p1
    HERE      : each ruled state's two successors split EXACTLY 0.5 / 0.5

That is the "toy" structure of the original project (§1 of CLAUDE.md) rather than the unequal
split folder 12 introduced, so results here are directly comparable to the earlier folders'
setting while sharing folder 12's 25k/10k sizes.

What the equal split changes downstream (worth knowing before analysing):
  * ideal-alpha now puts alpha/2 on each true successor, so its CE is a flat function of the
    table rather than of the split.
  * the context-Bayes predictor loses its q-integral: with the split known to be 0.5 the only
    thing left to infer is WHICH two tokens are the successors. The gap between ctx-count and
    Bayes therefore comes purely from the 2-of-4 sparsity prior.
  * the per-state entropy is maximal, so every threshold's CE floor is higher than folder 12's.

TRANSITION TABLES ARE BIT-IDENTICAL to folder 12's for the same threshold and split. The `order`
draw that picks each state's two successors happens FIRST in the RNG stream, and the p1 draw is
still consumed (then discarded) so the stream stays aligned up to that point. The emitted token
sequences do of course differ -- the successor coin now uses 0.5 instead of p1 -- but "same
chains, different split" makes the two datasets a clean controlled pair.

Output goes to a `dataset/` folder created NEXT TO THIS SCRIPT (anchored to the script's own
directory, not the cwd), with the same {train,test}/data_x_{thr}.pt layout every loader expects.
Approx disk usage: ~58 MB train + ~23 MB test per threshold, ~0.6 GB total for all 7.

Edit CONFIG below, then run:
    python data_generation.py
"""

import os
import time
import numpy as np
import torch

# ─── Config ───────────────────────────────────────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

CONFIG = {
    'output_dir':  os.path.join(_SCRIPT_DIR, 'dataset'),   # created next to this script
    'thresholds':  [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    'train_size':  25_000,
    'test_size':   10_000,
    'seq_len':     100,
    'vocab_set1':  4,          # ruled (signal) tokens  -> ids 0..3
    'total_vocab': 8,          # + noise tokens          -> ids 4..7
    # NO p1_low / p1_high: the successor split is fixed at 0.5/0.5 by construction.
    'seed':        42,         # base seed; threshold i uses seed + i*1000 (same as folder 12)
    'overwrite':   True,       # False -> skip files that already exist
    'stream_align': True,      # consume-and-discard the p1 draw so the transition tables match
                               # folder 12's exactly (see docstring). Set False to skip the draw.
}
# ─────────────────────────────────────────────────────────────────────────────


def generate_dataset(n, seq_len, thr, v_sig, v_total, rng, stream_align=True):
    """Vectorised generator with EQUIPROBABLE successors.

    Each ruled state gets 2 distinct successors drawn uniformly from the v_sig ruled tokens,
    each taken with probability exactly 1/2. At each step: w.p. thr emit a successor of the
    current ruled state (which updates the state); w.p. (1-thr) emit a uniform noise token
    (state unchanged).
    """
    B = int(n)
    order     = rng.random((B, v_sig, v_sig)).argsort(axis=2)
    trans_arr = order[:, :, :2].astype(np.int64)               # [B, v_sig, 2] successor ids
    if stream_align:
        rng.uniform(0.1, 0.9, size=(B, v_sig))                 # discarded; keeps the RNG stream
                                                               # aligned with folder 12's
    prob_arr = np.full((B, v_sig, 2), 0.5)                     # <-- the only real change

    seq  = np.empty((B, seq_len + 1), dtype=np.int64)
    last = np.full(B, -1, dtype=np.int64); bidx = np.arange(B)
    for t in range(seq_len + 1):
        is_sig = rng.random(B) < thr
        has    = last >= 0
        state  = np.where(has, last, 0)
        coin   = (rng.random(B) >= prob_arr[bidx, state, 0]).astype(np.int64)   # fair coin
        succ   = trans_arr[bidx, state, coin]
        first  = rng.integers(0, v_sig, B)
        noise  = rng.integers(v_sig, v_total, B)
        tok    = np.where(is_sig, np.where(has, succ, first), noise)
        last   = np.where(is_sig, tok, last)
        seq[:, t] = tok
    x = torch.from_numpy(seq[:, :-1]).contiguous().long()
    y = torch.from_numpy(seq[:, 1:]).contiguous().long()
    transitions = [{v: trans_arr[b, v] for v in range(v_sig)} for b in range(B)]
    trans_probs = [{v: prob_arr[b, v]  for v in range(v_sig)} for b in range(B)]
    return x, y, transitions, trans_probs


def main():
    C = CONFIG
    train_dir = os.path.join(C['output_dir'], 'train')
    test_dir  = os.path.join(C['output_dir'], 'test')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir,  exist_ok=True)

    # ~2.3 KB/sequence on disk (measured on folder 11's files: 20k train = 46 MB)
    approx_gb = 2.3e-6 * (C['train_size'] + C['test_size']) * len(C['thresholds'])
    print(f"output     : {os.path.abspath(C['output_dir'])}")
    print(f"approx size: ~{approx_gb:.2f} GB total "
          f"(~{2.3e-3 * C['train_size']:.0f} MB train + ~{2.3e-3 * C['test_size']:.0f} MB test "
          f"per threshold x {len(C['thresholds'])} thresholds)")
    print(f"splits     : train={C['train_size']}  test={C['test_size']}  seq_len={C['seq_len']}")
    print(f"vocab      : V_sig={C['vocab_set1']}  V={C['total_vocab']}")
    print(f"succ split : EQUIPROBABLE (0.5 / 0.5)   <-- differs from folder 12's p1~U(0.1,0.9)")
    print(f"stream algn: {C['stream_align']}  (transition tables match folder 12's when True)")
    print(f"thresholds : {C['thresholds']}\n")

    t0 = time.time()
    for i, thr in enumerate(C['thresholds']):
        seed_thr = C['seed'] + i * 1000
        for split, n, sub in [('train', C['train_size'], train_dir),
                               ('test',  C['test_size'],  test_dir)]:
            out = os.path.join(sub, f'data_x_{thr}.pt')
            if not C['overwrite'] and os.path.exists(out):
                print(f"  [skip] {out}")
                continue
            split_seed = seed_thr + (0 if split == 'train' else 1)
            rng = np.random.default_rng(split_seed)
            x, y, trans, probs = generate_dataset(
                n, C['seq_len'], thr, C['vocab_set1'], C['total_vocab'], rng,
                stream_align=C['stream_align'])
            torch.save({
                'x': x, 'y': y, 'transitions': trans, 'trans_probs': probs,
                'threshold':   float(thr),
                'seq_len':     C['seq_len'],
                'vocab_set1':  C['vocab_set1'],
                'total_vocab': C['total_vocab'],
                'num_samples': n,
                'seed':        split_seed,
                'split':       split,
                'succ_split':  'equiprobable',      # marks this dataset apart from folder 12's
            }, out)
            # sanity: the realised signal-token frequency should sit near thr
            sig_frac = float((x < C['vocab_set1']).float().mean())
            print(f"  [ok] {split:5s}  thr={thr}  N={n:>6d}  signal frac={sig_frac:.3f}  "
                  f"-> {out}", flush=True)

    print(f"\nDone in {time.time() - t0:.1f}s.")


if __name__ == '__main__':
    main()
