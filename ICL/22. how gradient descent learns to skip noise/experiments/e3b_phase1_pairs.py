"""
E3b -- Which (query, key) pairs carry the initial pull on c2 and b2?

At theta = 0 both layers attend uniformly over j < i, so r_i = f_i = histogram of x_<i and
q_i = (f_i + b) / (1 + sum b). For a score coefficient s2_ij = kappa * phi_ij (phi = <r_i, r_j>
for c2, r_j[x_i] for b2) the exact gradient of the mean loss is

    -dL/dkappa = (1/N) sum_i  1/(f_i[y_i] + b[y_i]) * (1/n_i) sum_{j<i} (1[x_j = y_i] - f_i[y_i]) * phi_ij

(n_i = number of keys; the normaliser of q is constant because sum f = 1). Each (i, j) term is
one pair's contribution. They are binned by
    distance d = i - j, and
    the key's PREDECESSOR x_{j-1}:  == x_i  /  == state(i) (the true current state)  /  other.
The formula is checked against autograd first.
"""
import json, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
import five as M

dev = torch.device(sys.argv[1] if len(sys.argv) > 1 else 'cuda:0')
out = {}
for a in (0.6, 1.0, 0.2):
    tr = M.load(a, 'train')
    x, y = tr['x'][:3000].to(dev), tr['y'][:3000].to(dev)
    B, Tn = x.shape
    b = M.readout_bias(a, dev)
    oh = torch.nn.functional.one_hot(x, M.V).float()
    cnt = torch.cumsum(oh, 1)
    nkeys = torch.arange(Tn, device=dev).clamp_min(1).float()          # keys j < i (pos 0: itself)
    r = torch.zeros_like(oh); r[:, 1:] = cnt[:, :-1] / nkeys[1:].view(1, -1, 1); r[:, 0] = oh[:, 0]
    f = r
    rs = r[..., :M.VS]
    st = M.state_of(x)
    i_idx = torch.arange(M.TS, Tn, device=dev)
    fy = f[:, M.TS:].gather(-1, y[:, M.TS:].unsqueeze(-1)).squeeze(-1)
    wq = 1.0 / (fy + b[y[:, M.TS:]])                                     # (B, 50)
    N = B * (Tn - M.TS)
    res = {}
    for name in ('c2', 'b2'):
        tot = 0.0
        by_d = torch.zeros(Tn, device=dev)
        by_pred = {'pred==x_i': 0.0, 'pred==state_i': 0.0, 'pred==both': 0.0, 'other': 0.0}
        for ii, i in enumerate(i_idx.tolist()):
            j = torch.arange(i, device=dev)
            if name == 'c2':
                phi = (rs[:, i:i + 1] * rs[:, j]).sum(-1)                     # (B, i)
            else:
                qs = (x[:, i] < M.VS)
                phi = r[:, j].gather(-1, x[:, i].clamp_max(M.VS - 1).view(-1, 1, 1).expand(-1, i, 1)).squeeze(-1)
                phi = phi * qs.float().unsqueeze(1)
            hit = (x[:, j] == y[:, i:i + 1]).float()
            term = wq[:, ii:ii + 1] * (hit - fy[:, ii:ii + 1]) * phi / float(i)   # (B, i)
            tot += float(term.sum())
            by_d[i - j] += term.sum(0)
            pred = torch.full_like(x[:, j], -1); pred[:, 1:] = x[:, j[1:] - 1]
            e1 = pred == x[:, i:i + 1]; e2 = pred == st[:, i:i + 1]
            by_pred['pred==both'] += float(term[e1 & e2].sum())
            by_pred['pred==x_i'] += float(term[e1 & ~e2].sum())
            by_pred['pred==state_i'] += float(term[~e1 & e2].sum())
            by_pred['other'] += float(term[~e1 & ~e2].sum())
        th = torch.zeros(5, device=dev, requires_grad=True)
        L = M.ce(th, x, y, a); g, = torch.autograd.grad(L, th)
        auto = -float(g[4 if name == 'c2' else 3])
        res[name] = dict(formula=tot / N, autograd=auto,
                         by_distance=[float(v) / N for v in by_d[1:16]],
                         by_predecessor={k: v / N for k, v in by_pred.items()})
        print(f"alpha={a} {name}: -dL/d{name} formula {tot/N:+.3e}  autograd {auto:+.3e}")
        print(f"   by distance d=1..8 : {[f'{float(v)/N:+.1e}' for v in by_d[1:9]]}")
        print(f"   by key predecessor : " + "  ".join(f"{k} {v/N:+.2e}" for k, v in by_pred.items()), flush=True)
    out[str(a)] = res
json.dump(out, open(os.path.join(M.F22, 'runs', 'e3b_phase1_pairs.json'), 'w'), indent=1)
