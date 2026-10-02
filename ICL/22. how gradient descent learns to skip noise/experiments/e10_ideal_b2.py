"""
E10 -- Where is the induction weight b2 heading?

Unlike c2, b2 is still growing at step 1000 for most alpha. Idealised model as in E8 (perfect
layer 1, noise keys vetoed, a2 = -30) on the real 100-token sequences: scan (b2, c2) jointly for
the loss minimiser, and compare with GD's (b2, c2) at steps 200 / 500 / 1000.

Hypothesis: in an infinitely long context, larger b2 always helps (the prediction approaches the
true successor split). In a 100-token context each state has been seen only a few times, so
trusting those few examples completely is overconfident: the best b2 is FINITE and smaller when
noise is denser (fewer signal tokens in context).
"""
import json, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
import five as M

dev = torch.device(sys.argv[1] if len(sys.argv) > 1 else 'cuda:2')
B2 = np.arange(1, 40.01, 1.0)
C2 = np.arange(-3, 20.01, 0.5)
out = []
print(f"{'alpha':>6} | {'ideal best b2':>13} {'c2':>5} {'loss':>8} | GD (b2, c2) at steps 200 / 500 / 1000")
for a in (0.2, 0.4, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0):
    tr = M.load(a, 'train'); x, y = tr['x'][:5000].to(dev), tr['y'][:5000].to(dev)
    Ls = np.zeros((len(B2), len(C2)))
    with torch.no_grad():
        for i, b in enumerate(B2):
            for j, c in enumerate(C2):
                th = torch.tensor([0., 0., -30., float(b), float(c)], device=dev)
                Ls[i, j] = float(M.ce(th, x, y, a, perfect_layer1=True))
    i, j = np.unravel_index(np.argmin(Ls), Ls.shape)
    TH = np.load(os.path.join(M.F22, 'runs', 'main', f'traj_x_{M.afmt(a)}.npz'))['theta']
    gd = [(float(TH[s, 3]), float(TH[s, 4])) for s in (200, 500, 1000)]
    edge = ' (AT GRID EDGE)' if i == len(B2) - 1 else ''
    out.append(dict(alpha=a, best_b2=float(B2[i]), best_c2=float(C2[j]), best_loss=float(Ls[i, j]),
                    gd=gd, loss_vs_b2_at_best_c2=Ls[:, j].tolist(), b2_grid=B2.tolist()))
    print(f"{a:>6} | {B2[i]:>13.1f} {C2[j]:>5.1f} {Ls[i, j]:>8.5f} | " +
          '  '.join(f'({b:.2f}, {c:.2f})' for b, c in gd) + edge, flush=True)
json.dump(out, open(os.path.join(M.F22, 'runs', 'e10_ideal_b2.json'), 'w'), indent=1)
