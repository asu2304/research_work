"""
E19 -- Long-context sign of the initial induction pull, checked on ALL 1296 transition tables.

Derivation (proofs/4_pull_sign.tex): at theta = 0 the induction feature of key j is h_j[v] =
(1/j) #{k < j : x_k = v}. Its covariance with hit_j = 1[x_j = y], summed over lags, involves
P(x_j = y | x_{j-lag} = v) for a signal token v; with m signal emissions in between,
m ~ Binomial(lag-1, alpha), and sum_lag Binom(lag-1, m; alpha) = 1/alpha, so the noise CANCELS:
    sum_lag [ P(x_j=y | x_{j-lag}=v) - P(x_j=y) ]  =  sum_{m>=1} (P^m - Pi)(v, y)  =  (Z - I)(v, y)
with Z = (I - P + Pi)^{-1} the fundamental matrix of the chain on its recurrent class. Weighting by
the query distribution (current token v, target y ~ P(.|v)) and w = 1/(f[y] + b), f[y] -> alpha pi_y:

    pull  ~  (const > 0) * S,     S = sum_v pi_v^2 sum_y P(v,y) (Z - I)(v, y) / (alpha pi_y + b_sig)

Claim to check: S >= 0 for every table and every recurrent class, with S = 0 exactly when the
current token is uninformative (every state of the class has the same successor set).
Exact rational arithmetic (fractions) is used for the sign; floats for the ordering check.
"""
import itertools, json, os, sys
from fractions import Fraction as Fr
import numpy as np
import sympy as sp
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
import five as M

pairs = list(itertools.combinations(range(4), 2))


def recurrent_classes(succ):
    reach = [set() for _ in range(4)]
    for s in range(4):
        stack, seen = [s], set()
        while stack:
            u = stack.pop()
            for v in succ[u]:
                if v not in seen:
                    seen.add(v); stack.append(v)
        reach[s] = seen
    cls = []
    for s in range(4):
        if s in reach[s] and all(s in reach[u] for u in reach[s]):
            c = tuple(sorted(reach[s]))
            if c not in cls:
                cls.append(c)
    return cls


def S_exact(succ, cls, alpha, bsig):
    k = len(cls); ix = {s: n for n, s in enumerate(cls)}
    P = sp.zeros(k, k)
    for s in cls:
        for v in succ[s]:
            P[ix[s], ix[v]] += sp.Rational(1, 2)
    # stationary distribution on the class (irreducible, possibly periodic)
    A = (P.T - sp.eye(k)).col_join(sp.ones(1, k))
    rhs = sp.zeros(k, 1).col_join(sp.Matrix([1]))
    pi = A.solve_least_squares(rhs)
    Pi = sp.ones(k, 1) * pi.T
    Z = (sp.eye(k) - P + Pi).inv()
    D = Z - sp.eye(k)
    a = sp.Rational(alpha).limit_denominator(1000); b = sp.Rational(bsig).limit_denominator(1000)
    S = 0
    for v in range(k):
        for y in range(k):
            if P[v, y] != 0:
                S += pi[v] ** 2 * P[v, y] * D[v, y] / (a * pi[y] + b)
    uninformative = all(set(succ[s]) == set(succ[cls[0]]) for s in cls)
    return sp.nsimplify(S), uninformative


res = {}
for alpha in (0.2, 0.6, 1.0):
    vals = []
    for tab in itertools.product(pairs, repeat=4):
        succ = [list(t) for t in tab]
        for c in recurrent_classes(succ):
            S, uninf = S_exact(succ, c, alpha, M.B_SIG)
            vals.append((float(S), bool(S == 0), uninf, c, succ))
    negs = [v for v in vals if v[0] < 0]
    zeros = [v for v in vals if v[1]]
    zero_iff_uninf = all((v[1] == v[2]) for v in vals)
    res[str(alpha)] = dict(n_class_tables=len(vals), n_negative=len(negs), n_zero=len(zeros),
                           zero_iff_uninformative=zero_iff_uninf,
                           min_positive=float(min(v[0] for v in vals if v[0] > 0)))
    print(f"alpha={alpha}: {len(vals)} (table, recurrent class) pairs | negative S: {len(negs)} | "
          f"S == 0 exactly: {len(zeros)} | 'S=0 iff uninformative' holds: {zero_iff_uninf} | "
          f"smallest positive S: {res[str(alpha)]['min_positive']:.3e}", flush=True)
json.dump(res, open(os.path.join(M.F22, 'runs', 'e19_pull_sign_all_tables.json'), 'w'), indent=1)
