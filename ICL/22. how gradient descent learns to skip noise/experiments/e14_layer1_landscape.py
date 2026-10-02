"""
E14 -- Does layer 1 have a finite best (p, a1), or does the loss keep falling as both grow?

Layer 2 held at GD's step-1000 values (a2, b2, c2). Two probes on real data:
  1. scale GD's layer-1 pair:   (p, a1) -> s * (p, a1),  s in 0.5 .. 4   (horizon fixed, sharper)
  2. move the horizon at fixed p:  a1 = -(H - 1) * p  for H in 2 .. 30     (sharpness fixed)
Also reports the fraction of scored positions whose gap g (distance back to the last signal)
exceeds GD's horizon H = 1 + |a1|/p: the positions layer 1 currently loses to noise.
"""
import json, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
import five as M

dev = torch.device(sys.argv[1] if len(sys.argv) > 1 else 'cuda:2')
S = (0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0)
HS = (2, 3, 4, 5, 6, 8, 10, 15, 20, 30)
out = []
for a in (0.2, 0.4, 0.6, 0.8, 0.9):
    th = np.load(os.path.join(M.F22, 'runs', 'main', f'traj_x_{M.afmt(a)}.npz'))['theta'][1000]
    p, a1, a2, b2, c2 = [float(v) for v in th]
    H = 1 + abs(a1) / p
    tr = M.load(a, 'train'); x, y = tr['x'][:5000].to(dev), tr['y'][:5000].to(dev)
    with torch.no_grad():
        Ls = [float(M.ce(torch.tensor([s * p, s * a1, a2, b2, c2], device=dev), x, y, a)) for s in S]
        Lh = [float(M.ce(torch.tensor([p, -(h - 1) * p, a2, b2, c2], device=dev), x, y, a)) for h in HS]
    idx = torch.arange(M.T, device=dev)
    last = torch.where(x < M.VS, idx, torch.full_like(idx, -1)).cummax(1).values
    lastb = torch.full_like(last, -1); lastb[:, 1:] = last[:, :-1]
    gap = (idx[M.TS:].view(1, -1) - lastb[:, M.TS:])[lastb[:, M.TS:] >= 0].float()
    lost = float((gap > H).float().mean())
    out.append(dict(alpha=a, theta=th.tolist(), H=H, frac_gap_beyond_H=lost, scale=S, L_scale=Ls,
                    H_grid=HS, L_horizon=Lh))
    base = Ls[S.index(1.0)]
    print(f"\nalpha={a}: GD p={p:.2f} a1={a1:.2f}  horizon H={H:.2f}   positions with gap > H: {100*lost:.2f}%")
    print('   scale s     : ' + '  '.join(f'{s}:{1000*(L-base):+.2f}' for s, L in zip(S, Ls)) + '   (millinats vs GD)')
    print('   horizon H   : ' + '  '.join(f'{h}:{1000*(L-base):+.2f}' for h, L in zip(HS, Lh)) + '   (p fixed)', flush=True)
json.dump(out, open(os.path.join(M.F22, 'runs', 'e14_layer1_landscape.json'), 'w'), indent=1)
