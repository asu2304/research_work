"""
E12b -- E12's stylised b2 model, now WITH the state weight c2.

At a post-signal position (token v, previous state s', target y), with t = e^{b2}, s = e^{c2}:
    group v  (keys from v):          n  keys, k  of them y,   weight t
    group s' (keys from s', s' != v): n' keys, k' of them y,   weight s      (the rival)
    rest:                             R  keys, Ry of them y,   weight 1
    (if s' == v the two groups coincide and get weight t*s)
    f_y = (t k + s k' + Ry) / (t n + s n' + R),    loss = -log(f_y + b_sig)
Two versions of c2: (i) fixed at GD's step-1000 value; (ii) tied to b2 by the balance formula
c2 = b2/2 + 1/2 log((1-alpha)/alpha) + kappa (alpha < 1), i.e. as GD actually moves them together.
Best b2 of each vs GD's b2. Real counts, no model.
"""
import json, math, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
import five as M

B = M.B_SIG
KAPPA = json.load(open(os.path.join(M.F22, 'runs', 'e9_kappa_theory.json')))['kappa']
grid = np.arange(0, 40.01, 0.25)
out = []
print(f"{'alpha':>6} | {'best b2, c2=GD':>15} {'best b2, c2 tied':>17} | {'GD b2@1000':>11} {'GD growth/e-fold':>17}")
for a in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99):
    tr = M.load(a, 'train'); x, y = tr['x'][:5000], tr['y'][:5000]
    pv = M.prev_signal(x); sig = x < M.VS
    cols = {k: [] for k in ('n', 'k', 'n2', 'k2', 'R', 'Ry', 'same')}
    for i in range(M.TS, M.T):
        q = sig[:, i] & (y[:, i] < M.VS)
        sk = sig[:, :i]; is_y = x[:, :i] == y[:, i].unsqueeze(1)
        v = x[:, i]; sp = pv[:, i]
        gv = sk & (pv[:, :i] == v.unsqueeze(1))
        gs = sk & (pv[:, :i] == sp.unsqueeze(1)) & ~gv
        rest = sk & ~gv & ~gs
        for name, val in (('n', gv.sum(1)), ('k', (gv & is_y).sum(1)), ('n2', gs.sum(1)),
                          ('k2', (gs & is_y).sum(1)), ('R', rest.sum(1)), ('Ry', (rest & is_y).sum(1)),
                          ('same', (sp == v).long())):
            cols[name].append(val[q])
    c = {k_: torch.cat(v_).double().numpy() for k_, v_ in cols.items()}
    TH = np.load(os.path.join(M.F22, 'runs', 'main', f'traj_x_{M.afmt(a)}.npz'))['theta']
    c2_gd = float(TH[1000, 4])

    def loss(b2, c2):
        t, s = math.exp(b2), math.exp(c2)
        wv = np.where(c['same'] > 0, t * s, t)
        num = wv * c['k'] + s * c['k2'] + c['Ry']
        den = wv * c['n'] + s * c['n2'] + c['R']
        f = np.where(den > 0, num / np.maximum(den, 1e-300), 0.0)
        return float(np.mean(-np.log(f + B)))
    L1 = np.array([loss(b, c2_gd) for b in grid])
    L2 = np.array([loss(b, b / 2 + 0.5 * math.log((1 - a) / a) + KAPPA) for b in grid])
    def best(L):
        j = int(np.argmin(L)); return float(grid[j]) if j < len(grid) - 1 else float('inf')
    kk = float(np.polyfit(np.log(np.arange(200, 1001)), TH[200:1001, 3], 1)[0])
    # large-b2 criterion with c2 fixed: the rival group joins the background with weight e^{c2}
    s = math.exp(c2_gd)
    n_, k_ = c['n'], c['k']
    Reff = s * c['n2'] + c['R']; Ryeff = s * c['k2'] + c['Ry']
    ok = (n_ > 0) & (c['same'] == 0)                 # self-loop positions: c2 multiplies b2, excluded
    n_, k_, Reff, Ryeff = n_[ok], k_[ok], Reff[ok], Ryeff[ok]
    g = np.where(Reff > 0, Ryeff / np.maximum(Reff, 1e-300), 0.0)
    Kgain = float(np.mean(np.where(k_ > 0, (Reff / n_**2) * (k_ - n_ * g) / (k_ / n_ + B), 0.0)))
    Kcost = float(np.mean(np.where(k_ == 0, Reff * g / (n_ * B), 0.0)))
    r = dict(alpha=a, c2_gd=c2_gd, best_b2_c2_fixed=best(L1), best_b2_c2_tied=best(L2),
             gd_b2=float(TH[1000, 3]), gd_growth=kk, Kcost_over_Kgain=Kcost / Kgain,
             grid=grid.tolist(), L_fixed=L1.tolist(), L_tied=L2.tolist())
    out.append(r)
    print(f"{a:>6} | {r['best_b2_c2_fixed']:>15.2f} {r['best_b2_c2_tied']:>17.2f} | {r['gd_b2']:>11.2f} {kk:>17.2f}"
          f"   Kcost/Kgain {Kcost / Kgain:5.2f} -> {'finite' if Kcost > Kgain else 'drifts'}", flush=True)
json.dump(out, open(os.path.join(M.F22, 'runs', 'e12b_b2_with_c2.json'), 'w'), indent=1)
