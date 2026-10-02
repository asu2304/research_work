"""
E5 -- Numerical check of the COVARIANCE LAW (1_exact_statements.pdf, Proposition 4), at trained weights.

For any layer-2 coefficient kappa in {a2, b2, c2} with feature phi^kappa_ij
    a2: nu_j = 1[x_j noise]     b2: sigma_i r_j[x_i]     c2: <r_i, r_j>_signal
the exact partial derivative of the mean loss is

    -dL/dkappa = (1/N) sum_i  w_i * Cov_{j ~ A2_i}( 1[x_j = y_i],  phi_ij ),
    w_i = 1 / (f_i[y_i] + b[y_i])

(layer 1 does not depend on kappa, so this is the whole derivative). Checked against autograd
at several (alpha, step), and the c2 term is split by query type with the favoured-key hit rates:
    hit_fav  = E_{j ~ A2_i * phi_ij}[ 1[x_j = y_i] ]  (where c2's feature points attention)
    hit_now  = f_i[y_i]                                (where attention already is)
Positive covariance <=> hit_fav > hit_now <=> descent raises c2.
"""
import json, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
import five as M

dev = torch.device(sys.argv[1] if len(sys.argv) > 1 else 'cuda:0')
out = []
for a, k in ((0.6, 0), (0.6, 30), (0.6, 1000), (0.2, 1000), (1.0, 10), (1.0, 1000)):
    TH = np.load(os.path.join(M.F22, 'runs', 'main', f'traj_x_{M.afmt(a)}.npz'))['theta']
    tr = M.load(a, 'train')
    x, y = tr['x'][:5000].to(dev), tr['y'][:5000].to(dev)
    th = torch.tensor(TH[k], dtype=torch.float32, device=dev, requires_grad=True)
    L = M.ce(th, x, y, a); g, = torch.autograd.grad(L, th)
    with torch.no_grad():
        b = M.readout_bias(a, dev)
        f, P = M.forward(th, x, parts=True)
        A2, r = P['A2'][:, M.TS:], P['r']
        rs = r[..., :M.VS]
        xs, ys = x[:, M.TS:], y[:, M.TS:]
        fy = f[:, M.TS:].gather(-1, ys.unsqueeze(-1)).squeeze(-1)
        w = 1.0 / (fy + b[ys])
        hit = (x.unsqueeze(1) == ys.unsqueeze(-1)).float()                 # (B, 50, T) 1[x_j = y_i]
        feats = {
            'a2': (x >= M.VS).float().unsqueeze(1).expand_as(hit),
            'b2': rs.gather(-1, xs.clamp_max(M.VS - 1).unsqueeze(1).expand(-1, rs.shape[1], -1)
                            ).transpose(1, 2) * (xs < M.VS).float().unsqueeze(-1),
            'c2': rs[:, M.TS:] @ rs.transpose(1, 2),
        }
        N = xs.numel()
        row = dict(alpha=a, step=k, theta=TH[k].tolist())
        for name, phi in feats.items():
            cov = (A2 * (hit - fy.unsqueeze(-1)) * phi).sum(-1)               # per query
            row[f'{name}_formula'] = float((w * cov).sum() / N)
            row[f'{name}_autograd'] = -float(g[M.NAMES.index(name)])
        phi = feats['c2']
        for typ, m in (('postnoise', xs >= M.VS), ('postsignal', xs < M.VS)):
            if m.sum() == 0:
                continue
            wt = A2 * phi
            hf = (wt * hit).sum(-1) / wt.sum(-1).clamp_min(1e-30)
            row[f'c2_hit_favoured_{typ}'] = float(hf[m].mean())
            row[f'c2_hit_now_{typ}'] = float(fy[m].mean())
            row[f'c2_term_{typ}'] = float((w * (A2 * (hit - fy.unsqueeze(-1)) * phi).sum(-1))[m].sum() / N)
    out.append(row)
    print(f"alpha={a} step={k}: " + "  ".join(f"{n}: formula {row[n+'_formula']:+.3e} autograd {row[n+'_autograd']:+.3e}" for n in ('a2', 'b2', 'c2')))
    for typ in ('postnoise', 'postsignal'):
        if f'c2_hit_favoured_{typ}' in row:
            print(f"     c2 at {typ:10s}: hit rate of keys c2 favours {row[f'c2_hit_favoured_{typ}']:.3f}"
                  f"  vs current f[y] {row[f'c2_hit_now_{typ}']:.3f}   term {row[f'c2_term_{typ}']:+.2e}", flush=True)
json.dump(out, open(os.path.join(M.F22, 'runs', 'e5_covariance_law.json'), 'w'), indent=1)
