"""
E23 -- Quantitative checks of the limit theorems in proofs/7_complete_theorem.pdf (finite T = 100).

(a) Part (ii): pull(alpha) = kappa_T * alpha^3 * E[S(alpha)] -> pull / (alpha^3 E[S]) should be the SAME
    constant at every alpha. E[S] averages S over random tables and over the recurrent class each
    sequence ends in (simulated, uniform first state). Measured pulls from e17.
(b) Part (iii), recency: limit (-U0_p) = (alpha^3/2) E[S] while pull ~ alpha^3 E[S] * mean_i(H_{i-1}/i),
    so  (-U0_p) / pull  ->  (1/2) / mean_{i in [50,100)} (H_{i-1} / i). Compare with e17.
(c) Part (iii), veto: (-U0_a1) / pull -> -(1 - alpha). Compare with e17.
"""
import itertools, json, math, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
import five as M

pairs = list(itertools.combinations(range(4), 2))
rng = np.random.default_rng(0)


def S_of(succ, pi, alpha, b=M.B_SIG):
    vis = [s for s in range(4) if pi[s] > 0]
    k = len(vis); ix = {s: n for n, s in enumerate(vis)}
    P = np.zeros((k, k))
    for s in vis:
        for v in succ[s]:
            P[ix[s], ix[v]] += 0.5
    p = np.array([pi[s] for s in vis])
    Z = np.linalg.inv(np.eye(k) - P + np.outer(np.ones(k), p))
    D = Z - np.eye(k)
    return float(sum(p[v] ** 2 * P[v, y] * D[v, y] / (alpha * p[y] + b) for v in range(k) for y in range(k)))


# long-run in-class frequencies per (table, run): simulate once, reuse for every alpha
runs = []
for tab in itertools.product(pairs, repeat=4):
    succ = [list(t) for t in tab]
    for _ in range(4):
        s = int(rng.integers(4)); cnt = np.zeros(4)
        for t in range(3000):
            s = succ[s][int(rng.integers(2))]
            if t >= 1500:
                cnt[s] += 1
        runs.append((succ, cnt / cnt.sum()))
e17 = {r['alpha']: r for r in json.load(open(os.path.join(M.F22, 'runs', 'e17_veto_seed_formula.json')))}
Hmean = float(np.mean([sum(1.0 / j for j in range(1, i)) / i for i in range(50, 100)]))
print(f"(b) predicted recency/pull ratio at T=100: (1/2)/mean(H_(i-1)/i) = {0.5 / Hmean:.2f}\n")
print(f"{'alpha':>6} {'E[S]':>8} {'pull':>10} {'pull/(a^3 E[S])':>16} | {'recency/pull':>12} | {'veto/pull':>9} {'-(1-a)':>7}")
out = []
for a in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99):
    ES = float(np.mean([S_of(succ, pi, a) for succ, pi in runs]))
    r = e17[a]
    k = r['induction_pull'] / (a ** 3 * ES)
    rec = r['negU0_p_autograd'] / r['induction_pull']
    vet = r['negU0_a1_autograd'] / r['induction_pull']
    out.append(dict(alpha=a, ES=ES, pull=r['induction_pull'], kappa_T=k, recency_ratio=rec, veto_ratio=vet))
    print(f"{a:>6} {ES:>8.4f} {r['induction_pull']:>10.3e} {k:>16.4e} | {rec:>12.2f} | {vet:>9.3f} {-(1 - a):>7.2f}", flush=True)
json.dump(dict(predicted_recency_ratio=0.5 / Hmean, rows=out),
          open(os.path.join(M.F22, 'runs', 'e23_theory_checks.json'), 'w'), indent=1)
