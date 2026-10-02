"""
E9 -- Predict the constant kappa in   c2* = b2/2 + (1/2) log((1-alpha)/alpha) + kappa
from the table distribution ALONE (no training data, no GD).

Idealised long-context model: perfect layer 1, noise keys vetoed, each state u seen with
in-context frequency pi_u (its long-run frequency in this sequence), its two successors each
half the time. Leading-order balance (derivation in proofs/2_c2_b2_balance.tex):

  post-noise gain   = (1-alpha) * alpha * A * e^{-c2}       / (1/2 + b_sig)
  post-signal cost  = alpha * alpha * B * e^{c2 - b2}        / (1/2 + b_sig)

  A = sum_{s visited} [ 1/2 - (pi_{y1(s)} + pi_{y2(s)}) / 2 ]                 (y1, y2 = succ(s))
  B = 1/2 * sum_{s'} pi_{s'} * sum_{v in succ(s'), v != s'} 1/2 * (pi_{s'}/pi_v) * P_{y ~ succ(v)}(y not in succ(s'))

  balance  =>  c2* = b2/2 + 1/2 log((1-alpha)/alpha) + kappa,   kappa = 1/2 log( E[A] / E[B] ).

E[.] is over the random table (all 6^4 = 1296 tables enumerated, each equally likely) and over
which part of the chain a sequence gets trapped in (simulated: uniform first token, long run).
"""
import itertools, json, math, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
import five as M

rng = np.random.default_rng(0)
pairs = list(itertools.combinations(range(4), 2))            # the 6 possible successor sets
L_RUN, N_RUN = 4000, 8                                       # chain length, runs per table


def terms(succ, pi):
    A = sum(0.5 - (pi[succ[s][0]] + pi[succ[s][1]]) / 2 for s in range(4) if pi[s] > 0)
    B = 0.0
    for sp in range(4):
        if pi[sp] == 0:
            continue
        for v in succ[sp]:
            if v == sp or pi[v] == 0:
                continue
            p_out = np.mean([y not in succ[sp] for y in succ[v]])
            B += pi[sp] * 0.5 * (pi[sp] / pi[v]) * p_out
    return A, 0.5 * B


As, Bs = [], []
for tab in itertools.product(pairs, repeat=4):
    succ = [list(t) for t in tab]
    for _ in range(N_RUN):
        s = int(rng.integers(4)); cnt = np.zeros(4)
        for t in range(L_RUN):
            s = succ[s][int(rng.integers(2))]
            if t >= L_RUN // 2:                # burn-in: only the states the chain keeps visiting
                cnt[s] += 1
        pi = cnt / cnt.sum()
        a, b = terms(succ, pi); As.append(a); Bs.append(b)
EA, EB = float(np.mean(As)), float(np.mean(Bs))
kappa = 0.5 * math.log(EA / EB)
print(f"E[A] = {EA:.5f}   E[B] = {EB:.5f}   kappa (theory, long context) = {kappa:.3f}")

rows = json.load(open(os.path.join(M.F22, 'runs', 'e8b_ideal_vs_gd_c2.json')))['rows']
print(f"\n{'alpha':>6} {'GD b2':>6} {'GD c2':>6} {'predicted c2*':>14} {'error':>7}")
out = []
for r in rows:
    a = r['alpha']
    if a >= 1:
        continue
    pred = r['gd_b2'] / 2 + 0.5 * math.log((1 - a) / a) + kappa
    out.append(dict(alpha=a, gd_b2=r['gd_b2'], gd_c2=r['gd_c2'], predicted=pred))
    print(f"{a:>6} {r['gd_b2']:>6.2f} {r['gd_c2']:>6.2f} {pred:>14.2f} {r['gd_c2'] - pred:>+7.2f}")
json.dump(dict(EA=EA, EB=EB, kappa=kappa, rows=out),
          open(os.path.join(M.F22, 'runs', 'e9_kappa_theory.json'), 'w'), indent=1)
