"""
E3 -- What pulls the state block c2 (and b2) up at the very start, before layer 1 has
learned anything?

At theta = 0 layer 1 attends uniformly, so r_i is just the token histogram of the prefix x_<i,
and layer 2 (also uniform) predicts that same histogram: the in-context UNIGRAM. Hypothesis:
the early pull on c2 rewards sharpening that unigram prediction and has nothing to do with
order, state or noise.

Test: recompute the gradient at theta = 0 (and at the first few trained steps) on
    real        the actual sequences
    shuffled    every sequence's tokens randomly permuted WITHIN the sequence: identical token
                counts per sequence, Markov order destroyed. y is rebuilt as the shifted x.
If the pull on c2 survives shuffling, Phase 1 is a frequency (unigram) effect.
"""
import json, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
import five as M

dev = torch.device(sys.argv[1] if len(sys.argv) > 1 else 'cuda:0')
g = torch.Generator().manual_seed(0)
out = {}
print(f"{'alpha':>5} {'step':>4} | {'-g b2 real':>11} {'-g b2 shuf':>11} | {'-g c2 real':>11} {'-g c2 shuf':>11} | {'-g a2 real':>11} {'-g a2 shuf':>11}")
for a in (0.2, 0.4, 0.6, 0.8, 1.0):
    tr = M.load(a, 'train')
    full = torch.cat([tr['x'], tr['y'][:, -1:]], 1)[:10000]          # the 101-token sequences
    perm = torch.argsort(torch.rand(full.shape, generator=g), dim=1)
    shuf = full.gather(1, perm)
    data = {'real': (full[:, :-1], full[:, 1:]), 'shuf': (shuf[:, :-1], shuf[:, 1:])}
    traj = os.path.join(M.F22, 'runs', 'main', f'traj_x_{M.afmt(a)}.npz')
    TH = np.load(traj)['theta'] if os.path.exists(traj) else np.zeros((1, 5))
    rows = []
    for k in (0, 3, 10):
        if k >= len(TH):
            continue
        res = {}
        for name, (x, y) in data.items():
            th = torch.tensor(TH[k], dtype=torch.float32, device=dev, requires_grad=True)
            L = M.ce(th, x.to(dev), y.to(dev), a)
            gr, = torch.autograd.grad(L, th)
            res[name] = (-gr).cpu().tolist()
        rows.append(dict(step=k, neg_grad_real=res['real'], neg_grad_shuffled=res['shuf']))
        print(f"{a:>5} {k:>4} | {res['real'][3]:>+11.2e} {res['shuf'][3]:>+11.2e} | "
              f"{res['real'][4]:>+11.2e} {res['shuf'][4]:>+11.2e} | "
              f"{res['real'][2]:>+11.2e} {res['shuf'][2]:>+11.2e}", flush=True)
    out[str(a)] = rows
json.dump(out, open(os.path.join(M.F22, 'runs', 'e3_phase1_origin.json'), 'w'), indent=1)
