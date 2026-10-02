"""
E19b -- (1) Extend E19's sign result from three alphas to EVERY alpha in (0, 1].
          S(alpha) = sum_y C_y / (alpha pi_y + b),   C_y = sum_v pi_v^2 P(v,y) (Z - I)(v,y).
          If every C_y >= 0, S >= 0 for all alpha at once. Otherwise S(alpha) is checked exactly:
          its numerator is a polynomial in alpha, whose sign on (0, 1] is settled by sympy.
       (2) Does S predict the measured pull? Compare with E18's per-shape means (irreducible shapes,
          where a single recurrent class makes S well defined), alpha = 0.8: rank correlation.
Exact arithmetic throughout part (1).
"""
import itertools, json, os, sys
import numpy as np
import sympy as sp
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
import five as M
from e19_pull_sign_all_tables import recurrent_classes, pairs

al = sp.symbols('alpha', positive=True)
b = sp.Rational(1, 100)


def parts(succ, cls):
    k = len(cls); ix = {s: n for n, s in enumerate(cls)}
    P = sp.zeros(k, k)
    for s in cls:
        for v in succ[s]:
            P[ix[s], ix[v]] += sp.Rational(1, 2)
    A = (P.T - sp.eye(k)).col_join(sp.ones(1, k))
    pi = A.solve_least_squares(sp.zeros(k, 1).col_join(sp.Matrix([1])))
    Z = (sp.eye(k) - P + sp.ones(k, 1) * pi.T).inv()
    D = Z - sp.eye(k)
    C = [sp.nsimplify(sum(pi[v] ** 2 * P[v, y] * D[v, y] for v in range(k))) for y in range(k)]
    return pi, C


n_all, n_cy_neg, bad = 0, 0, []
S_of = {}
for tab in itertools.product(pairs, repeat=4):
    succ = [list(t) for t in tab]
    for c in recurrent_classes(succ):
        n_all += 1
        pi, C = parts(succ, c)
        if any(cy < 0 for cy in C):
            n_cy_neg += 1
            S = sp.together(sum(C[y] / (al * pi[y] + b) for y in range(len(c))))
            num, den = sp.fraction(S)
            num = sp.Poly(sp.expand(num), al)
            # den > 0 on (0,1]; num's sign on (0,1]: check endpoints and real roots inside
            roots = [r for r in sp.real_roots(num) if 0 < r <= 1] if num.degree() > 0 else []
            if roots or num.eval(sp.Rational(1, 2)) < 0 or num.eval(1) < 0:
                bad.append((succ, c, [str(r) for r in roots]))
        if len(c) == 4:
            S_of[tuple(map(tuple, succ))] = float(sum(C[y] / (sp.Rational(4, 5) * pi[y] + b) for y in range(4)))
print(f"(1) {n_all} (table, class) pairs; some C_y < 0 in {n_cy_neg}; "
      f"of those, S(alpha) < 0 somewhere in (0,1]: {len(bad)}")
for x in bad[:5]:
    print('    counterexample:', x)

# (2) prediction vs measurement (E18, alpha = 0.8), irreducible shapes only
import itertools as it
PERMS = list(it.permutations(range(4)))
def canon(succ):
    best = None
    for pm in PERMS:
        t = [None] * 4
        for s in range(4):
            t[pm[s]] = tuple(sorted(pm[u] for u in succ[s]))
        t = tuple(t); best = t if best is None or t < best else best
    return best
e18 = json.load(open(os.path.join(M.F22, 'runs', 'e18_pull_by_table.json')))['0.8']['rows']
meas = {tuple(map(tuple, r['shape'])): (r['mean'], r['n']) for r in e18}
pred_by_shape = {}
for succ, s in S_of.items():
    pred_by_shape.setdefault(canon([list(t) for t in succ]), s)
common = [k for k in pred_by_shape if k in meas and meas[k][1] >= 100]
xs = np.array([pred_by_shape[k] for k in common]); ys = np.array([meas[k][0] for k in common])
from scipy.stats import spearmanr, pearsonr
rho = spearmanr(xs, ys).correlation; r = pearsonr(xs, ys)[0]
print(f"(2) irreducible shapes with >=100 sequences: {len(common)} | Spearman {rho:.3f} | Pearson {r:.3f}")
json.dump(dict(n_pairs=n_all, n_with_negative_Cy=n_cy_neg, n_bad=len(bad),
               bad=[str(x) for x in bad], spearman=rho, pearson=r, n_shapes_compared=len(common),
               pred=xs.tolist(), meas=ys.tolist()),
          open(os.path.join(M.F22, 'runs', 'e19b_pull_sign_all_alpha.json'), 'w'), indent=1)
