"""
E18 -- Is the initial induction pull  -dL/db2|_0  positive for EVERY transition table, or only
on average? Decides what kind of proof is possible.

Per-sequence contribution (exact, from the covariance law at theta = 0):
    P_seq = sum_{i in [50,100)} w_i sigma_i Cov_{j<i}( 1[x_j = y_i], h_j[x_i] )
Sequences are grouped by the SHAPE of their table: tables equal up to relabelling the 4 signal
tokens (the data distribution is relabelling-invariant, so all tables of one shape have the same
expected pull). Reports each shape's mean pull with a standard error, at several alpha.
"""
import itertools, json, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
import five as M

dev = torch.device(sys.argv[1] if len(sys.argv) > 1 else 'cuda:2')
PERMS = list(itertools.permutations(range(4)))


def shape(trans):
    best = None
    for pm in PERMS:
        t = [None] * 4
        for s in range(4):
            t[pm[s]] = tuple(sorted(pm[u] for u in trans[s]))
        t = tuple(t)
        best = t if best is None or t < best else best
    return best


def per_seq_pull(x, y, a):
    B, T = x.shape
    oh = torch.nn.functional.one_hot(x, M.V).double()
    cnt = torch.cumsum(oh, 1)
    nk = torch.arange(T, device=dev).double().clamp_min(1)
    h = torch.zeros_like(oh); h[:, 1:] = cnt[:, :-1] / nk[1:].view(1, -1, 1); h[:, 0] = oh[:, 0]
    b = M.readout_bias(a, dev).double()
    P = torch.zeros(B, device=dev, dtype=torch.double)
    for i in range(M.TS, T):
        sig = (x[:, i] < M.VS).double()
        fy = h[:, i].gather(1, y[:, i:i + 1]).squeeze(1)
        w = 1.0 / (fy + b[y[:, i]])
        hit = (x[:, :i] == y[:, i:i + 1]).double()
        v = x[:, i].clamp_max(M.VS - 1)
        hx = h[:, :i].gather(2, v.view(-1, 1, 1).expand(-1, i, 1)).squeeze(-1)
        P += w * sig * ((hit - fy.unsqueeze(1)) * hx).mean(1)
    return (P / (T - M.TS)).cpu().numpy()


out = {}
for a in (0.4, 0.8, 1.0):
    tr = M.load(a, 'train')
    x, y = tr['x'].to(dev), tr['y'].to(dev)
    P = per_seq_pull(x, y, a)
    shapes = [shape([list(map(int, tr['transitions'][b][s])) for s in range(4)]) for b in range(len(P))]
    groups = {}
    for s, p in zip(shapes, P):
        groups.setdefault(s, []).append(p)
    rows = []
    for s, ps in groups.items():
        ps = np.array(ps)
        rows.append(dict(shape=[list(t) for t in s], n=len(ps), mean=float(ps.mean()),
                         se=float(ps.std(ddof=1) / np.sqrt(len(ps))) if len(ps) > 1 else float('nan')))
    rows.sort(key=lambda r: r['mean'])
    neg = [r for r in rows if r['mean'] + 2 * r['se'] < 0]
    out[str(a)] = dict(overall=float(P.mean()), n_shapes=len(rows), rows=rows)
    print(f"\nalpha={a}: overall pull {P.mean():+.3e} | {len(rows)} table shapes | "
          f"shapes clearly NEGATIVE (mean + 2se < 0): {len(neg)} covering {sum(r['n'] for r in neg)} of {len(P)} sequences")
    for r in rows[:5]:
        print(f"   lowest: mean {r['mean']:+.3e} +- {r['se']:.1e}  n={r['n']:>5}  succ={r['shape']}")
    for r in rows[-3:]:
        print(f"   highest: mean {r['mean']:+.3e} +- {r['se']:.1e}  n={r['n']:>5}  succ={r['shape']}", flush=True)
json.dump(out, open(os.path.join(M.F22, 'runs', 'e18_pull_by_table.json'), 'w'), indent=1)
