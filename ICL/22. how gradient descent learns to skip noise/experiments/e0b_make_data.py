"""
Generate the equiprobable Markov+noise data for alphas folder 14 does not have.

Uses folder 14's OWN generator function (imported read-only, nothing there is touched), same
sizes (25k train / 10k test, seq_len 100, V_sig 4, V 8). Only the seeds differ, and they are
chosen so they can never coincide with folder 14's (42 + i*1000 + {0,1}):

    seed = 900_000 + round(alpha * 10_000) * 10 + {0 train, 1 test}

Writes ONLY into 22/dataset/{train,test}/data_x_{alpha}.pt. Skips files that already exist.
usage: python e0b_make_data.py 0.2 0.25 0.3 ...
"""
import importlib.util, os, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import five as M

spec = importlib.util.spec_from_file_location(
    'gen14', os.path.join(M.ICL, '14. elimination final convergence confusion', 'data_generation.py'))
gen14 = importlib.util.module_from_spec(spec); spec.loader.exec_module(gen14)

for a in [float(s) for s in sys.argv[1:]]:
    for split, n, off in (('train', 25_000, 0), ('test', 10_000, 1)):
        out = os.path.join(M.DATA22, split, f'data_x_{M.afmt(a)}.pt')
        if os.path.exists(out):
            print('[skip]', out); continue
        os.makedirs(os.path.dirname(out), exist_ok=True)
        seed = 900_000 + int(round(a * 10_000)) * 10 + off
        t0 = time.time()
        x, y, tr, pr = gen14.generate_dataset(n, 100, a, 4, 8, np.random.default_rng(seed))
        torch.save({'x': x, 'y': y, 'transitions': tr, 'trans_probs': pr, 'threshold': a,
                    'seq_len': 100, 'vocab_set1': 4, 'total_vocab': 8, 'num_samples': n,
                    'seed': seed, 'split': split, 'succ_split': 'equiprobable',
                    'generator': "folder 14 data_generation.generate_dataset (imported)"}, out)
        print(f'[ok] {split} alpha={a} sig_frac={float((x < 4).float().mean()):.3f} '
              f'({time.time()-t0:.0f}s) -> {out}', flush=True)
