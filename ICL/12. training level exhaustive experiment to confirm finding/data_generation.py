"""
Generate train + test data for folder 13 (exhaustive experiments to confirm the
combined-vs-separate parametrization finding). Same task as folder 11: V_sig=4, n_noise=4,
seq_len=100, unequal per-state successor split p1~U(0.1,0.9). No validation split -- training
scripts train on the FULL train blob and checkpoint on best TRAIN loss.

Differences vs folder 11's dataset: train 25k (was 20k) and test 10k (was 2k) -- the larger
test split cuts the test-CE measurement noise to ~+-0.003 nats so the small gaps at
alpha=0.9-1.0 are resolvable. Everything else (vocab, seq_len, thresholds, p1 range, seeds)
is identical.

Output goes to a `dataset/` folder created NEXT TO THIS SCRIPT (anchored to the script's own
directory, not the cwd). Approx disk usage: ~58 MB train + ~23 MB test per threshold,
~0.6 GB total for all 7 thresholds.

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
    'p1_low':      0.1,        # per-state successor split p1~U(p1_low,p1_high), p2=1-p1
    'p1_high':     0.9,
    'seed':        42,         # base seed; threshold i uses seed + i*1000
    'overwrite':   True,       # False -> skip files that already exist
}
# ─────────────────────────────────────────────────────────────────────────────


def generate_dataset(n, seq_len, thr, v_sig, v_total, p_low, p_high, rng):
    """Vectorised altered-distribution generator: each ruled state has 2 (random,
    per-sequence) successors with an UNEQUAL split p1~U(p_low,p_high), p2=1-p1. At each
    step: w.p. thr emit a successor of the current ruled state (updates state); w.p.
    (1-thr) emit a uniform noise token (state unchanged)."""
    B = int(n)
    order     = rng.random((B, v_sig, v_sig)).argsort(axis=2)
    trans_arr = order[:, :, :2].astype(np.int64)               # [B, v_sig, 2] successor ids
    p1        = rng.uniform(p_low, p_high, size=(B, v_sig))
    prob_arr  = np.stack([p1, 1.0 - p1], axis=2)               # [B, v_sig, 2]
    seq  = np.empty((B, seq_len + 1), dtype=np.int64)
    last = np.full(B, -1, dtype=np.int64); bidx = np.arange(B)
    for t in range(seq_len + 1):
        is_sig = rng.random(B) < thr
        has    = last >= 0
        state  = np.where(has, last, 0)
        coin   = (rng.random(B) >= prob_arr[bidx, state, 0]).astype(np.int64)
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
    print(f"succ split : p1~U({C['p1_low']},{C['p1_high']})")
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
                n, C['seq_len'], thr, C['vocab_set1'], C['total_vocab'],
                C['p1_low'], C['p1_high'], rng)
            torch.save({
                'x': x, 'y': y, 'transitions': trans, 'trans_probs': probs,
                'threshold':   float(thr),
                'seq_len':     C['seq_len'],
                'vocab_set1':  C['vocab_set1'],
                'total_vocab': C['total_vocab'],
                'num_samples': n,
                'seed':        split_seed,
                'split':       split,
            }, out)
            print(f"  [ok] {split:5s}  thr={thr}  N={n:>6d}  -> {out}", flush=True)

    print(f"\nDone in {time.time() - t0:.1f}s.")


if __name__ == '__main__':
    main()
