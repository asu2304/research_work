"""
E10b -- the loss as a function of b2 with c2 RE-OPTIMISED for every b2 (profile loss), in the
idealised model of E10 (perfect layer 1, noise vetoed). E10's loss-vs-b2 table held c2 at its
overall best value, which is unfair for small b2; this is the correct comparison.
Reports: profile loss minus its minimum (millinats), the 'knee' (smallest b2 within 1 millinat of
the plateau), and GD's b2 at steps 200 / 500 / 1000.
"""
import json, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
import five as M
dev = torch.device(sys.argv[1] if len(sys.argv) > 1 else 'cuda:2')
B2 = np.arange(2, 40.01, 1.0); C2 = np.arange(-4, 20.01, 0.5)
show = [4, 6, 8, 10, 12, 14, 16, 20, 25, 30, 40]
out = []
print('profile loss minus minimum (millinats); c2 re-optimised at each b2')
print(f"{'alpha':>6} " + ''.join(f'{b:>7}' for b in show) + '   knee | GD b2 @200/500/1000')
for a in (0.2, 0.4, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0):
    tr = M.load(a, 'train'); x, y = tr['x'][:5000].to(dev), tr['y'][:5000].to(dev)
    prof, argc = [], []
    with torch.no_grad():
        for b in B2:
            L = [float(M.ce(torch.tensor([0., 0., -30., float(b), float(c)], device=dev), x, y, a,
                            perfect_layer1=True)) for c in C2]
            prof.append(min(L)); argc.append(float(C2[int(np.argmin(L))]))
    prof = np.array(prof); rel = 1000 * (prof - prof.min())
    knee = float(B2[np.argmax(rel <= 1.0)])
    TH = np.load(os.path.join(M.F22, 'runs', 'main', f'traj_x_{M.afmt(a)}.npz'))['theta']
    gd = [float(TH[s, 3]) for s in (200, 500, 1000)]
    out.append(dict(alpha=a, b2=B2.tolist(), profile=prof.tolist(), best_c2_per_b2=argc,
                    knee=knee, gd_b2=gd))
    print(f"{a:>6} " + ''.join(f'{rel[B2 == b][0]:>7.2f}' for b in show) +
          f"  {knee:>5.0f} | {gd[0]:.2f} / {gd[1]:.2f} / {gd[2]:.2f}", flush=True)
json.dump(out, open(os.path.join(M.F22, 'runs', 'e10b_b2_profile.json'), 'w'), indent=1)
