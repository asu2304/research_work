"""
E22e -- Test the rule found by E22d's trees:  c2 <= b2  =>  the noise veto strengthens (dL/da1 > 0).

Random points of E22's box, alpha in 0.2 .. 0.9. Points where (almost) no position sits on the
skip cliff (edge = mean m(1-m) < 1e-4) are reported separately: there dL/da1 is ~0 and its sign is
numerically meaningless. Also: the largest c2/b2 ratio GD ever visits (after step 3), Adam and plain GD.
"""
import glob, json, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
import five as M

dev = torch.device(sys.argv[1] if len(sys.argv) > 1 else 'cuda:2')
rng = np.random.default_rng(3)
out = {}
print(f"{'alpha':>6} | {'c2<=b2: weakens':>16} {'c2>b2: weakens':>15} | {'no-cliff points':>15}")
for a in (0.2, 0.4, 0.6, 0.8, 0.9):
    tr = M.load(a, 'train'); x, y = tr['x'][:2000].to(dev), tr['y'][:2000].to(dev)
    noise = (x >= M.VS).unsqueeze(1).float()
    c = {'le': [0, 0], 'gt': [0, 0]}; flat = 0; ex = []
    for _ in range(800):
        th0 = np.array([rng.uniform(0, 8), rng.uniform(-30, 0), rng.uniform(-10, 0),
                        rng.uniform(0, 15), rng.uniform(0, 10)])
        th = torch.tensor(th0, dtype=torch.float32, device=dev, requires_grad=True)
        g, = torch.autograd.grad(M.ce(th, x, y, a), th)
        with torch.no_grad():
            _, P = M.forward(th.detach(), x, parts=True)
            m = (P['A1'][:, M.TS:] * noise).sum(-1)
            edge = float((m * (1 - m)).mean())
        if edge < 1e-4:
            flat += 1; continue
        key = 'le' if th0[4] <= th0[3] else 'gt'
        c[key][0] += 1; w = float(g[1]) <= 0; c[key][1] += int(w)
        if w and key == 'le':
            ex.append(np.round(th0, 2).tolist() + [round(edge, 5), float(g[1])])
    out[str(a)] = dict(le=c['le'], gt=c['gt'], no_cliff=flat, exceptions=ex[:10])
    print(f"{a:>6} | {c['le'][1]:>7}/{c['le'][0]:<8} {c['gt'][1]:>7}/{c['gt'][0]:<7} | {flat:>15}", flush=True)
    for e in ex[:3]:
        print(f"         exception (p,a1,a2,b2,c2,edge,dL/da1): {e}")

for tag in ('main', 'gd_lr30'):
    mx = []
    for f in sorted(glob.glob(os.path.join(M.F22, 'runs', tag, 'traj_x_*.npz'))):
        a = float(os.path.basename(f)[7:-4])
        if a >= 1:
            continue
        TH = np.load(f)['theta'][3:]
        r = TH[:, 4] / np.maximum(TH[:, 3], 1e-9)
        mx.append((a, round(float(r.max()), 3)))
    out[f'max_c2_over_b2_{tag}'] = mx
    print(f"{tag}: largest c2/b2 along training: {mx}")
json.dump(out, open(os.path.join(M.F22, 'runs', 'e22e_rule_c2_le_b2.json'), 'w'), indent=1)
