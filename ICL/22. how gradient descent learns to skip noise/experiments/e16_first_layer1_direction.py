"""
E16 -- The FIRST direction layer 1 moves in: does it point at 'skip noise'?

Gating (Prop. 1): grad_{(p,a1)} L = b2 * U(theta) + c2 * W(theta). Near theta = 0 (both layers
uniform), to first order in (b2, c2):
    grad_{(p,a1)} L  ~=  b2 * U0 + c2 * W0,     U0 = d/db2 grad_{(p,a1)} L |_{theta=0},  W0 likewise.
U0, W0 depend only on the data. Computed here as exact mixed second derivatives (autograd of
autograd) at theta = 0, and checked by finite differences at small b2, c2.

Skipping needs  -dL/dp > 0 (prefer recent keys)  and  -dL/da1 < 0 (veto noise keys).
GD's first step puts b2 = c2 = +0.096 (both > 0), a2 = -0.096; so the relevant direction is
-(U0 + W0) (a2 enters only at second order).
"""
import json, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
import five as M

dev = torch.device(sys.argv[1] if len(sys.argv) > 1 else 'cuda:2')
out = []
print(f"{'alpha':>6} | {'-U0 (via b2): p':>16} {'a1':>10} | {'-W0 (via c2): p':>16} {'a1':>10} | {'FD check':>9} | step-2 move of (p, a1)")
for a in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0):
    tr = M.load(a, 'train'); x, y = tr['x'][:8000].to(dev), tr['y'][:8000].to(dev)
    th = torch.zeros(5, device=dev, requires_grad=True)
    L = M.ce(th, x, y, a)
    g, = torch.autograd.grad(L, th, create_graph=True)
    U0 = torch.stack([torch.autograd.grad(g[0], th, retain_graph=True)[0][3],
                      torch.autograd.grad(g[1], th, retain_graph=True)[0][3]])      # d(g_p, g_a1)/db2
    W0 = torch.stack([torch.autograd.grad(g[0], th, retain_graph=True)[0][4],
                      torch.autograd.grad(g[1], th, retain_graph=True)[0][4]])      # d(g_p, g_a1)/dc2
    U0, W0 = U0.detach().cpu().numpy(), W0.detach().cpu().numpy()
    eps = 1e-2                                                                      # finite-difference check
    t2 = torch.tensor([0, 0, 0, eps, eps], dtype=torch.float32, device=dev, requires_grad=True)
    g2, = torch.autograd.grad(M.ce(t2, x, y, a), t2)
    fd = g2[:2].cpu().numpy() / eps
    rel = float(np.abs(fd - (U0 + W0)).max() / (np.abs(U0 + W0).max() + 1e-30))
    TH = np.load(os.path.join(M.F22, 'runs', 'main', f'traj_x_{M.afmt(a)}.npz'))['theta']
    r = dict(alpha=a, negU0=(-U0).tolist(), negW0=(-W0).tolist(), fd_rel_err=rel,
             step2_p=float(TH[2, 0]), step2_a1=float(TH[2, 1]))
    out.append(r)
    print(f"{a:>6} | {-U0[0]:>+16.3e} {-U0[1]:>+10.3e} | {-W0[0]:>+16.3e} {-W0[1]:>+10.3e} | {rel:>9.1e} | "
          f"({TH[2, 0]:+.3f}, {TH[2, 1]:+.3f})", flush=True)
json.dump(out, open(os.path.join(M.F22, 'runs', 'e16_first_layer1_direction.json'), 'w'), indent=1)
