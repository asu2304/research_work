"""
E12 -- Finite-sample theory of the induction weight b2.

Stylised model at a POST-SIGNAL position (perfect layer 1, noise vetoed, c2 = 0), t = e^{b2}:
    n = keys emitted from state v (= current token),  k = how many of them are the target y,
    R = all OTHER signal keys,  g = fraction of those that are y.
    f_y(t) = (t k + R g) / (t n + R),          loss = -log(f_y + b_sig)  (+ constant)
Post-noise positions do not depend on b2 at all.

Large-t expansion: d loss / d b2 = (1/t) * (Kcost - Kgain) + o(1/t), with
    Kgain = E[ 1{k>0} (R/n^2) (k - n g) / (k/n + b_sig) ]      (seen target: more trust helps)
    Kcost = E[ 1{k=0} R g / (n b_sig) ]                        (unseen target: more trust hurts)
=> Kcost > Kgain : the loss eventually RISES with b2 -> finite best b2 (GD settles)
   Kcost < Kgain : the loss keeps FALLING        -> no finite best (GD drifts like log t)
Checks: the ratio Kcost/Kgain across alpha vs where GD settled / drifted, and the exact stylised
loss minimised over b2 vs GD's b2 and E10b's idealised optimum. Real counts, no model.
"""
import json, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
import five as M

B = M.B_SIG
grid = np.arange(0, 40.01, 0.25)
e10b = {r['alpha']: r for r in json.load(open(os.path.join(M.F22, 'runs', 'e10b_b2_profile.json')))}
out = []
print(f"{'alpha':>6} {'Kcost/Kgain':>12} {'theory says':>12} | {'stylised best b2':>16} {'ideal (E10b)':>13} {'GD b2@1000':>11} {'GD growth/e-fold':>17}")
for a in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99, 1.0):
    tr = M.load(a, 'train'); x, y = tr['x'][:5000], tr['y'][:5000]
    pv = M.prev_signal(x); sig = x < M.VS
    N, K, R, G = [], [], [], []
    for i in range(M.TS, M.T):
        q = sig[:, i] & (y[:, i] < M.VS)
        sk = sig[:, :i]
        from_v = sk & (pv[:, :i] == x[:, i].unsqueeze(1))
        is_y = x[:, :i] == y[:, i].unsqueeze(1)
        n = from_v.sum(1); k = (from_v & is_y).sum(1)
        other = sk & ~from_v
        r = other.sum(1); gy = (other & is_y).sum(1)
        N.append(n[q]); K.append(k[q]); R.append(r[q]); G.append(gy[q])
    n, k, r, gy = [torch.cat(z).double().numpy() for z in (N, K, R, G)]
    ok = n > 0                                                     # n = 0: b2 has no effect
    n, k, r, gy = n[ok], k[ok], r[ok], gy[ok]
    g = np.where(r > 0, gy / np.maximum(r, 1), 0.0)
    seen = k > 0
    Kgain = float(np.mean(np.where(seen, (r / n**2) * (k - n * g) / (k / n + B), 0.0)))
    Kcost = float(np.mean(np.where(~seen, r * g / (n * B), 0.0)))
    ratio = Kcost / Kgain
    L = []
    for b2 in grid:
        t = np.exp(b2)
        f = (t * k + r * g) / (t * n + r)
        L.append(float(np.mean(-np.log(f + B))))
    L = np.array(L); kb = int(np.argmin(L))
    best = float(grid[kb]) if kb < len(grid) - 1 else float('inf')
    TH = np.load(os.path.join(M.F22, 'runs', 'main', f'traj_x_{M.afmt(a)}.npz'))['theta']
    t_ = np.arange(200, 1001); kk = float(np.polyfit(np.log(t_), TH[200:1001, 3], 1)[0])
    verdict = 'finite' if ratio > 1 else 'drifts'
    ideal = e10b[a]['knee'] if a in e10b else float('nan')
    ib = float(np.array(e10b[a]['b2'])[int(np.argmin(e10b[a]['profile']))]) if a in e10b else float('nan')
    out.append(dict(alpha=a, Kgain=Kgain, Kcost=Kcost, ratio=ratio, verdict=verdict,
                    stylised_best_b2=best, stylised_loss_curve=L.tolist(), grid=grid.tolist(),
                    ideal_best_b2=ib, gd_b2_1000=float(TH[1000, 3]), gd_growth_per_efold=kk))
    print(f"{a:>6} {ratio:>12.2f} {verdict:>12} | {best:>16.2f} {ib:>13.1f} {TH[1000, 3]:>11.2f} {kk:>17.2f}", flush=True)
json.dump(out, open(os.path.join(M.F22, 'runs', 'e12_b2_finite_sample.json'), 'w'), indent=1)
