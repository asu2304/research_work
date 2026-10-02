"""
E22 -- Does the noise veto keep strengthening throughout training? (the 'persistence' gap)

(1) Gain lemma (exact algebra, checked numerically): in layer 1 all noise keys get the same extra
    score a1, so the softmax split AMONG SIGNAL KEYS does not depend on a1. Hence
        r_j restricted to signal tokens = (1 - m_j) * rtilde_j,
    m_j = layer-1 attention mass on noise keys, rtilde_j independent of a1, and
    d m_j / d a1 = m_j (1 - m_j). Layer 2 only reads signal slots of r, so a1 acts purely as a
    per-position GAIN g_j = 1 - m_j on layer 2's induction and state features.
(2) Persistence along training: sign of dL/da1 at EVERY saved step of every trajectory
    (Adam runs/main, plain GD runs/gd_lr30). Veto strengthens iff dL/da1 > 0 (descent lowers a1).
(3) Beyond the trajectories: sign of dL/da1 at random points of the box
    p in [0, 8], a1 in [-30, 0], a2 in [-10, 0], b2 in [0, 15], c2 in [0, 10].
"""
import glob, json, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
import five as M

dev = torch.device(sys.argv[1] if len(sys.argv) > 1 else 'cuda:2')
out = {}

# (1) gain lemma
x = M.load(0.6, 'train')['x'][:500].to(dev)
noise = (x >= M.VS)
errs = []
for a1 in (0.0, -3.0, -10.0):
    th = torch.tensor([2.0, a1, 0, 0, 0], device=dev)
    _, P = M.forward(th, x, parts=True)
    rs = P['r'][..., :M.VS]
    m = (P['A1'] * noise.unsqueeze(1).float()).sum(-1)
    rt = rs / (1 - m).clamp_min(1e-12).unsqueeze(-1)
    errs.append(rt)
gain_err = float(max((errs[0] - errs[1]).abs().max(), (errs[0] - errs[2]).abs().max()))
print(f"(1) gain lemma: max change of the signal-key split across a1 in {{0,-3,-10}}: {gain_err:.1e}")
out['gain_lemma_max_err'] = gain_err

# (2) along the trajectories
for tag in ('main', 'gd_lr30'):
    tot = pos = 0; worst = []
    for f in sorted(glob.glob(os.path.join(M.F22, 'runs', tag, 'traj_x_*.npz'))):
        a = float(os.path.basename(f)[7:-4])
        if a >= 1:
            continue
        d = np.load(f); G = d['grad']; TH = d['theta']
        k = np.arange(2, len(G))                         # layer 1 has zero gradient at step 0, 1
        live = np.abs(G[k, 1]) > 0
        tot += int(live.sum()); pos += int((G[k, 1][live] > 0).sum())
        bad = k[live & (G[k, 1] <= 0)]
        if len(bad):
            worst.append((a, int(bad[0]), len(bad)))
    out[f'trajectory_{tag}'] = dict(steps=tot, veto_strengthens=pos, exceptions=worst)
    print(f"(2) {tag}: veto strengthens at {pos}/{tot} steps; exceptions (alpha, first step, count): {worst}")

# (3) random points
rng = np.random.default_rng(0)
res = {}
for a in (0.2, 0.6, 0.9):
    tr = M.load(a, 'train'); x, y = tr['x'][:3000].to(dev), tr['y'][:3000].to(dev)
    n = 300; neg = []
    for _ in range(n):
        th0 = np.array([rng.uniform(0, 8), rng.uniform(-30, 0), rng.uniform(-10, 0),
                        rng.uniform(0, 15), rng.uniform(0, 10)])
        th = torch.tensor(th0, dtype=torch.float32, device=dev, requires_grad=True)
        g, = torch.autograd.grad(M.ce(th, x, y, a), th)
        if g[1] <= 0:
            neg.append(np.round(th0, 2).tolist() + [float(g[1])])
    res[str(a)] = dict(n=n, veto_weakens=len(neg), examples=neg[:8])
    print(f"(3) alpha={a}: veto would WEAKEN at {len(neg)}/{n} random points" +
          (f"; e.g. (p,a1,a2,b2,c2,dL/da1) = {neg[:3]}" if neg else ''), flush=True)
out['random_box'] = res
json.dump(out, open(os.path.join(M.F22, 'runs', 'e22_veto_persistence.json'), 'w'), indent=1)
