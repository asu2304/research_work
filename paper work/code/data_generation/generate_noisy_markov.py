"""
Generate train + test data for the Markov-noise ICL task.
seq_len=200  train=20K  test=3K  (for D=220 comparison experiments)

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
    'train_size':  20_000,     # sequences per threshold for training
    'test_size':   3_000,      # sequences per threshold for testing
    'seq_len':     200,        # context length (matches D=220 hand construction)
    'vocab_set1':  10,         # ruled (signal) tokens  →  ids 0..9
    'total_vocab': 20,         # noise tokens           →  ids 10..19
    'seed':        42,         # base seed; threshold i uses seed + i*1000
    'overwrite':   True,       # False → skip files that already exist
}
# ─────────────────────────────────────────────────────────────────────────────

def generate_dataset(num_samples, seq_len, threshold_x, vocab_set1, total_vocab, rng):
    x_data, y_data, trans_list = [], [], []
    for _ in range(num_samples):
        transitions = {
            i: rng.choice(vocab_set1, size=2, replace=False).astype(np.int64)
            for i in range(vocab_set1)
        }
        seq, last_ruled = [], None
        for _ in range(seq_len + 1):
            if rng.random() < threshold_x:
                tok = (int(rng.integers(0, vocab_set1)) if last_ruled is None
                       else int(rng.choice(transitions[last_ruled])))
                last_ruled = tok
            else:
                tok = int(rng.integers(vocab_set1, total_vocab))
            seq.append(tok)
        x_data.append(seq[:-1])
        y_data.append(seq[1:])
        trans_list.append(transitions)
    return (
        torch.tensor(x_data, dtype=torch.long),
        torch.tensor(y_data, dtype=torch.long),
        trans_list,
    )

def main():
    C = CONFIG
    train_dir = os.path.join(C['output_dir'], 'train')
    test_dir  = os.path.join(C['output_dir'], 'test')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir,  exist_ok=True)

    print(f"output     : {os.path.abspath(C['output_dir'])}")
    print(f"splits     : train={C['train_size']}  test={C['test_size']}  seq_len={C['seq_len']}")
    print(f"vocab      : V_sig={C['vocab_set1']}  V={C['total_vocab']}")
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
            x, y, trans = generate_dataset(
                n, C['seq_len'], thr, C['vocab_set1'], C['total_vocab'], rng)
            torch.save({
                'x': x, 'y': y, 'transitions': trans,
                'threshold':   float(thr),
                'seq_len':     C['seq_len'],
                'vocab_set1':  C['vocab_set1'],
                'total_vocab': C['total_vocab'],
                'num_samples': n,
                'seed':        split_seed,
                'split':       split,
            }, out)
            print(f"  [ok] {split:5s}  thr={thr:.1f}  N={n:>6d}  → {out}", flush=True)

    print(f"\nDone in {time.time() - t0:.1f}s.")

if __name__ == '__main__':
    main()
