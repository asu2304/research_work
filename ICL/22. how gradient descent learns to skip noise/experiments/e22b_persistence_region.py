"""
E22b -- (1) Gain lemma, checked correctly: the split of layer-1 attention AMONG SIGNAL KEYS does not
          depend on a1 (only positions with a non-negligible signal mass are compared; E22's check
          divided by ~0 at positions preceded only by noise).
       (2) Where does the veto weaken? Random points of the E22 box, classified by how far the state
          weight is above its balance point,  excess = c2 - c2*(b2),
          c2* = b2/2 + 1/2 log((1-alpha)/alpha) + kappa.
          Conjecture: dL/da1 > 0 (veto strengthens) whenever excess <= 0.
       (3) GD's own excess along training (Adam and plain GD): does it stay <= 0?
"""
import glob, json, math, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
import five as M

dev = torch.device(sys.argv[1] if len(sys.argv) > 1 else 'cuda:2')
kappa = json.load(open(os.path.join(M.F22, 'runs', 'e9_kappa_theory.json')))['kappa']
out = {}

# (1)
x = M.load(0.6, 'train')['x'][:500].to(dev); noise = (x >= M.VS)
split = []
for a1 in (0.0, -3.0, -10.0):
    _, P = M.forward(torch.tensor([2.0, a1, 0, 0, 0], device=dev), x, parts=True)
    A = P['A1'] * (~noise).unsqueeze(1).float()
    mass = A.sum(-1, keepdim=True)
    split.append((A / mass.clamp_min(1e-30), mass))
ok = (split[0][1] > 1e-6) & (split[1][1] > 1e-6) & (split[2][1] > 1e-6)
err = float(max(((split[0][0] - split[1][0]).abs() * ok).max(), ((split[0][0] - split[2][0]).abs() * ok).max()))
print(f"(1) gain lemma: max change of the attention split among signal keys across a1: {err:.1e}")
out['gain_lemma_max_err'] = err

# (2)
rng = np.random.default_rng(0)
res = {}
for a in (0.2, 0.4, 0.6, 0.8, 0.9):
    tr = M.load(a, 'train'); xx, yy = tr['x'][:3000].to(dev), tr['y'][:3000].to(dev)
    shift = 0.5 * math.log((1 - a) / a) + kappa
    rows = []
    for _ in range(400):
        th0 = np.array([rng.uniform(0, 8), rng.uniform(-30, 0), rng.uniform(-10, 0),
                        rng.uniform(0, 15), rng.uniform(0, 10)])
        th = torch.tensor(th0, dtype=torch.float32, device=dev, requires_grad=True)
        g, = torch.autograd.grad(M.ce(th, xx, yy, a), th)
        rows.append((th0[4] - (th0[3] / 2 + shift), float(g[1])))
    ex = np.array([r[0] for r in rows]); ga = np.array([r[1] for r in rows])
    below, above = ex <= 0, ex > 0
    res[str(a)] = dict(n_below=int(below.sum()), weaken_below=int((ga[below] <= 0).sum()),
                       n_above=int(above.sum()), weaken_above=int((ga[above] <= 0).sum()),
                       min_excess_where_weakens=float(ex[ga <= 0].min()) if (ga <= 0).any() else None)
    r = res[str(a)]
    print(f"(2) alpha={a}: c2 at/below balance: veto weakens at {r['weaken_below']}/{r['n_below']} points | "
          f"above balance: {r['weaken_above']}/{r['n_above']} | smallest excess where it weakens: "
          f"{r['min_excess_where_weakens']}", flush=True)
out['random_box'] = res

# (3)
for tag in ('main', 'gd_lr30'):
    mx = []
    for f in sorted(glob.glob(os.path.join(M.F22, 'runs', tag, 'traj_x_*.npz'))):
        a = float(os.path.basename(f)[7:-4])
        if a >= 1 or a < 0.2:
            continue
        TH = np.load(f)['theta']
        e = TH[2:, 4] - (TH[2:, 3] / 2 + 0.5 * math.log((1 - a) / a) + kappa)
        mx.append((a, round(float(e.max()), 3), int(np.argmax(e)) + 2))
    out[f'trajectory_max_excess_{tag}'] = mx
    print(f"(3) {tag}: largest excess of c2 over its balance along training (alpha, max, step): {mx}")
json.dump(out, open(os.path.join(M.F22, 'runs', 'e22b_persistence_region.json'), 'w'), indent=1)
