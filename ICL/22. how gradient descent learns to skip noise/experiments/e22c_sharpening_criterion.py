"""
E22c -- Does 'layer 2 still wants to sharpen' predict whether the veto strengthens?

Gain lemma (E22b, exact): a1 only rescales layer 1's signal output at each position j by
g_j = 1 - m_j (m_j = layer-1 mass on noise keys). The induction term carries one gain (key j),
the state term two (query i and key j). If every g_j moved by the same factor, then
    dL/d(log g)  =  b2 * dL/db2  +  2 * c2 * dL/dc2   =: D
and since lowering a1 raises every g_j, the veto strengthens iff D < 0 ('sharpening layer 2 --
b2 -> s*b2, c2 -> s^2*c2 -- lowers the loss').
Real gains are NOT uniform (positions after long noise runs leak more), so this is a criterion to
test, not a theorem. Checked on random points of E22's box and along the training runs.
"""
import glob, json, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
import five as M

dev = torch.device(sys.argv[1] if len(sys.argv) > 1 else 'cuda:2')
rng = np.random.default_rng(1)
out = {}
for a in (0.2, 0.4, 0.6, 0.8, 0.9):
    tr = M.load(a, 'train'); x, y = tr['x'][:3000].to(dev), tr['y'][:3000].to(dev)
    agree = n = 0; dis = []
    for _ in range(400):
        th0 = np.array([rng.uniform(0, 8), rng.uniform(-30, 0), rng.uniform(-10, 0),
                        rng.uniform(0, 15), rng.uniform(0, 10)])
        th = torch.tensor(th0, dtype=torch.float32, device=dev, requires_grad=True)
        g, = torch.autograd.grad(M.ce(th, x, y, a), th)
        g = g.cpu().numpy()
        if abs(g[1]) < 1e-9:
            continue                                             # veto gradient numerically zero
        D = th0[3] * g[3] + 2 * th0[4] * g[4]
        n += 1
        if np.sign(D) == -np.sign(g[1]):
            agree += 1
        else:
            dis.append(np.round(th0, 2).tolist() + [float(D), float(g[1])])
    out[str(a)] = dict(n=n, agree=agree, disagreements=dis[:10])
    print(f"alpha={a}: sign(D) predicts the veto direction at {agree}/{n} random points "
          f"({100*agree/max(n,1):.1f}%)", flush=True)

for tag in ('main', 'gd_lr30'):
    agree = n = 0
    for f in sorted(glob.glob(os.path.join(M.F22, 'runs', tag, 'traj_x_*.npz'))):
        a = float(os.path.basename(f)[7:-4])
        if a >= 1:
            continue
        d = np.load(f); TH, G = d['theta'], d['grad']
        for k in range(2, len(G)):
            if abs(G[k, 1]) < 1e-12:
                continue
            D = TH[k, 3] * G[k, 3] + 2 * TH[k, 4] * G[k, 4]
            n += 1; agree += int(np.sign(D) == -np.sign(G[k, 1]))
    out[f'trajectory_{tag}'] = dict(n=n, agree=agree)
    print(f"{tag}: sign(D) predicts the veto direction at {agree}/{n} training steps ({100*agree/max(n,1):.1f}%)")
json.dump(out, open(os.path.join(M.F22, 'runs', 'e22c_sharpening_criterion.json'), 'w'), indent=1)
