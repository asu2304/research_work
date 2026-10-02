"""
E11 -- How many in-context examples does the induction head have to work with?

At each scored POST-SIGNAL position i (token v = x_i, target y = y_i in succ(v)):
    n   = number of earlier signal tokens emitted FROM state v (keys j < i with pv(j) = v)
    k   = how many of those were the target y
    unseen = (k == 0): the target successor has not appeared after v yet in this context
Only post-signal positions matter for b2: at a post-noise query the induction term is absent.
Pure counting on the real data; no model involved.
"""
import json, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
import five as M

out = []
print(f"{'alpha':>6} {'mean n':>7} {'median n':>9} {'P(n=0)':>7} {'P(target unseen)':>17} {'P(unseen | n>0)':>16}")
for a in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0):
    tr = M.load(a, 'train'); x, y = tr['x'][:5000], tr['y'][:5000]
    pv = M.prev_signal(x)                                    # source state of each key
    sig = x < M.VS
    ns, ks = [], []
    for i in range(M.TS, M.T):
        q = sig[:, i] & (y[:, i] < M.VS)                     # post-signal query, signal target
        v = x[:, i]
        keys = sig[:, :i] & (pv[:, :i] == v.unsqueeze(1))    # signal keys emitted from state v
        n = keys.sum(1)
        k = (keys & (x[:, :i] == y[:, i].unsqueeze(1))).sum(1)
        ns.append(n[q]); ks.append(k[q])
    n = torch.cat(ns).numpy(); k = torch.cat(ks).numpy()
    r = dict(alpha=a, mean_n=float(n.mean()), median_n=float(np.median(n)), p_n0=float((n == 0).mean()),
             p_unseen=float((k == 0).mean()), p_unseen_given_seen=float((k[n > 0] == 0).mean()),
             n_hist=np.bincount(np.minimum(n, 40), minlength=41).tolist())
    out.append(r)
    print(f"{a:>6} {r['mean_n']:>7.2f} {r['median_n']:>9.0f} {r['p_n0']:>7.3f} {r['p_unseen']:>17.3f} {r['p_unseen_given_seen']:>16.3f}", flush=True)
json.dump(out, open(os.path.join(M.F22, 'runs', 'e11_examples_per_state.json'), 'w'), indent=1)
