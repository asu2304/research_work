"""
E22f -- Candidate theorem: once layer 1's split among signal keys points at the last signal
(p large enough) and the state weight is not above its balance point, the veto strengthens.

Random points of E22's box restricted to p >= 2, classified by
    (a) c2 <= c2*(b2) = b2/2 + 1/2 log((1-alpha)/alpha) + kappa      (at/below balance)
    (b) c2 <= b2                                                       (the tree's rule)
Points with no position on the skip cliff (edge < 1e-4) are excluded. Reports the fraction where
the veto would weaken, and the exceptions.
"""
import json, math, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
import five as M

dev = torch.device(sys.argv[1] if len(sys.argv) > 1 else 'cuda:2')
kappa = json.load(open(os.path.join(M.F22, 'runs', 'e9_kappa_theory.json')))['kappa']
rng = np.random.default_rng(4)
out = {}
print(f"{'alpha':>6} | {'p>=2, c2<=balance':>18} | {'p>=2, c2<=b2':>13} | {'p>=2, c2>b2':>12}")
for a in (0.2, 0.4, 0.6, 0.8, 0.9):
    tr = M.load(a, 'train'); x, y = tr['x'][:2000].to(dev), tr['y'][:2000].to(dev)
    noise = (x >= M.VS).unsqueeze(1).float()
    shift = 0.5 * math.log((1 - a) / a) + kappa
    cnt = {'bal': [0, 0], 'le': [0, 0], 'gt': [0, 0]}; ex = []
    n_try = 0
    while sum(v[0] for v in (cnt['le'], cnt['gt'])) < 600 and n_try < 3000:
        n_try += 1
        th0 = np.array([rng.uniform(2, 8), rng.uniform(-30, 0), rng.uniform(-10, 0),
                        rng.uniform(0, 15), rng.uniform(0, 10)])
        th = torch.tensor(th0, dtype=torch.float32, device=dev, requires_grad=True)
        g, = torch.autograd.grad(M.ce(th, x, y, a), th)
        with torch.no_grad():
            _, P = M.forward(th.detach(), x, parts=True)
            m = (P['A1'][:, M.TS:] * noise).sum(-1); edge = float((m * (1 - m)).mean())
        if edge < 1e-4:
            continue
        weak = int(float(g[1]) <= 0)
        if th0[4] <= th0[3] / 2 + shift:
            cnt['bal'][0] += 1; cnt['bal'][1] += weak
            if weak:
                ex.append(np.round(th0, 2).tolist() + [round(edge, 5)])
        k = 'le' if th0[4] <= th0[3] else 'gt'
        cnt[k][0] += 1; cnt[k][1] += weak
    out[str(a)] = dict(counts=cnt, exceptions_below_balance=ex[:10])
    f = lambda v: f"{v[1]}/{v[0]} ({100*v[1]/max(v[0],1):.1f}%)"
    print(f"{a:>6} | {f(cnt['bal']):>18} | {f(cnt['le']):>13} | {f(cnt['gt']):>12}", flush=True)
    for e in ex[:2]:
        print(f"         exception (p,a1,a2,b2,c2,edge): {e}")
json.dump(out, open(os.path.join(M.F22, 'runs', 'e22f_rule_localised.json'), 'w'), indent=1)
