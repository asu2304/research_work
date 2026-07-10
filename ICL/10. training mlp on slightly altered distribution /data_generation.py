"""
Generate train + test data for the *altered* Markov-noise ICL task.

Difference vs folder 6: the two successors of each ruled state are NO LONGER
equiprobable.  Each signal state independently draws  p1 ~ Uniform(0.25, 0.75),
p2 = 1 - p1  (so BOTH successors keep at least 0.25 of the signal mass and are
clearly unequal).  Emission:
    w.p. alpha :  emit successor s0 w.p. p1, s1 w.p. p2  (and advance state)
    w.p. 1-a  :  emit a uniform noise token            (state unchanged)

Why: with equal 0.5/0.5 successors GD could "cheat" by learning to split the
signal mass evenly across the two observed tokens WITHOUT reading their relative
counts.  Unequal (p1,p2) makes equalising provably sub-optimal, so the model is
forced to estimate the count *ratio* from the context.

The blob additionally stores 'trans_probs' (per-state (p1,p2), aligned with
'transitions') so the ideal-alpha Bayes floor can use the true probabilities.

Edit CONFIG below, then run:
    python data_generation.py
"""

import os
import time
import numpy as np
import torch

# ─── Config ──────────────────────────────────────────────────────────────────
CONFIG = {
    'output_dir':  'dataset',
    'thresholds':  [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    'train_size':  20_000,
    'test_size':   5_000,
    'seq_len':     200,
    'vocab_set1':  10,         # ruled (signal) tokens  →  ids 0..9
    'total_vocab': 20,         # noise tokens           →  ids 10..19
    'p1_low':      0.00,       # p1 ~ Uniform(p1_low, p1_high), p2 = 1 - p1
    'p1_high':     1.00,
    'seed':        42,         # base seed; threshold i uses seed + i*1000
    'overwrite':   True,
}

# ─────────────────────────────────────────────────────────────────────────────
def generate_dataset(num_samples, seq_len, threshold_x, vocab_set1, total_vocab,
                     p1_low, p1_high, rng):
    """Vectorised over sequences. Returns (x, y, transitions, trans_probs)."""
    B, V_sig, V = num_samples, vocab_set1, total_vocab
    # 2 DISTINCT successors per signal state
    order     = rng.random((B, V_sig, V_sig)).argsort(axis=2)
    trans_arr = order[:, :, :2].astype(np.int64)                  # [B, V_sig, 2]
    # per-state unequal successor probabilities
    p1        = rng.uniform(p1_low, p1_high, size=(B, V_sig))
    prob_arr  = np.stack([p1, 1.0 - p1], axis=2)                  # [B, V_sig, 2]

    seq  = np.empty((B, seq_len + 1), dtype=np.int64)
    last = np.full(B, -1, dtype=np.int64)
    bidx = np.arange(B)
    for t in range(seq_len + 1):
        is_sig = rng.random(B) < threshold_x
        has    = last >= 0
        state  = np.where(has, last, 0)
        u      = rng.random(B)
        coin   = (u >= prob_arr[bidx, state, 0]).astype(np.int64)  # 0 w.p. p1, else 1
        succ   = trans_arr[bidx, state, coin]
        first  = rng.integers(0, V_sig, B)                         # first signal token (no state yet)
        noise  = rng.integers(V_sig, V, B)
        tok    = np.where(is_sig, np.where(has, succ, first), noise)
        last   = np.where(is_sig, tok, last)                       # state advances only on a signal token
        seq[:, t] = tok

    x = torch.from_numpy(seq[:, :-1]).contiguous().long()
    y = torch.from_numpy(seq[:, 1:]).contiguous().long()
    trans = [{v: trans_arr[b, v] for v in range(V_sig)} for b in range(B)]
    probs = [{v: prob_arr[b, v]  for v in range(V_sig)} for b in range(B)]
    return x, y, trans, probs

def main():
    C = CONFIG
    train_dir = os.path.join(C['output_dir'], 'train')
    test_dir  = os.path.join(C['output_dir'], 'test')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir,  exist_ok=True)

    print(f"output     : {os.path.abspath(C['output_dir'])}")
    print(f"splits     : train={C['train_size']}  test={C['test_size']}  seq_len={C['seq_len']}")
    print(f"vocab      : V_sig={C['vocab_set1']}  V={C['total_vocab']}")
    print(f"p1 range   : U({C['p1_low']}, {C['p1_high']})   (p2 = 1 - p1, per state)")
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
                'p1_low':      C['p1_low'],
                'p1_high':     C['p1_high'],
                'num_samples': n,
                'seed':        split_seed,
                'split':       split,
            }, out)
            print(f"  [ok] {split:5s}  thr={thr:.1f}  N={n:>6d}  → {out}", flush=True)

    print(f"\nDone in {time.time() - t0:.1f}s.")

if __name__ == '__main__':
    main()
