"""
E8b -- Is GD's state weight c2 simply the BEST c2, given layer 1 has learned to find the state?

For each alpha: take GD's own induction weight b2 (step 1000 of runs/main), replace layer 1 by
PERFECT state recovery, veto noise keys (a2 = -30), and scan c2 on a 0.25 grid for the value
that minimises the loss. Compare with the c2 GD actually learned.
Result (2026-10-03): agreement within 0.25 logits at every alpha in 0.2 ... 1.0, including the
inverted-U shape and the slightly negative value at alpha = 1.
"""
import json, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
import five as M

dev = torch.device(sys.argv[1] if len(sys.argv) > 1 else 'cuda:2')
grid = np.round(np.arange(-3, 15.01, 0.25), 2)
rows = []
print(f"{'alpha':>6} {'GD b2':>6} {'GD c2':>6} | {'best c2 (ideal, GD b2)':>22}")
for a in (0.2, 0.4, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0):
    th = np.load(os.path.join(M.F22, 'runs', 'main', f'traj_x_{M.afmt(a)}.npz'))['theta'][-1]
    tr = M.load(a, 'train'); x, y = tr['x'][:6000].to(dev), tr['y'][:6000].to(dev)
    with torch.no_grad():
        L = [float(M.ce(torch.tensor([0., 0., -30., th[3], c], device=dev, dtype=torch.float32),
                        x, y, a, perfect_layer1=True)) for c in grid]
    best = float(grid[int(np.argmin(L))])
    rows.append(dict(alpha=a, gd_b2=float(th[3]), gd_c2=float(th[4]), ideal_best_c2=best, loss_curve=L))
    print(f"{a:>6} {th[3]:>6.2f} {th[4]:>6.2f} | {best:>22.2f}", flush=True)
json.dump(dict(grid=grid.tolist(), rows=rows),
          open(os.path.join(M.F22, 'runs', 'e8b_ideal_vs_gd_c2.json'), 'w'), indent=1)
