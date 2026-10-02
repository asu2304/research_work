"""
E1 -- Is the five-number model a faithful SUMMARY of what GD learns with more freedom?

Take folder 18's GD solution (10,060 free entries, every block untied, the position block a
free 100x100 matrix) and project it onto the five-number family WITHOUT looking at the loss:

    a1  = mean over query rows q of [mean W1[q, noise keys] - mean W1[q, signal keys]]
    p   = least-squares slope of the position block, after removing each query row's
          constant (the softmax gauge), over the causal cells j < i only
    a2  = same as a1, on layer 2's token block
    b2  = mean(diag) - mean(off-diag) of the 4x4 induction block   (row-gauge invariant part)
    c2  = mean(diag) - mean(off-diag) of the 4x4 state block

all divided by sqrt(108) to put them in logits. Then evaluate the five-number model at those
values. If its CE stays close to folder 18's, the five numbers are a faithful coordinate system
for GD's own solution -- the reduction did not do GD's work for it.

Also reported: how much the untied entries actually spread (does GD keep the symmetry the tying
assumes?) and the SHAPE of GD's learned position profile vs the linear ramp.
"""
import json, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
import five as M

dev = torch.device(sys.argv[1] if len(sys.argv) > 1 else 'cuda:0')
V, VS, T, d1 = M.V, M.VS, M.T, M.V + M.T
res = {}
for a in (0.6, 0.8, 1.0):
    sd = torch.load(os.path.join(M.RUN18, f'model_final_seed0_x_{a}.pth'), map_location='cpu')
    W1 = sd['layer1.W_qk'].double().numpy() / M.SQ          # logits
    W2 = sd['layer2.W_qk'].double().numpy() / M.SQ

    # ---- token blocks: per-query-row noise-minus-signal contrast
    A1 = W1[0:V, 0:V]; A2 = W2[0:V, 0:V]
    a1_rows = A1[:, VS:].mean(1) - A1[:, :VS].mean(1)
    a2_rows = A2[:, VS:].mean(1) - A2[:, :VS].mean(1)
    # within-row spread across the 4 noise keys and 4 signal keys (tying assumes 0)
    a1_within = float(np.concatenate([A1[:, VS:] - A1[:, VS:].mean(1, keepdims=True),
                                      A1[:, :VS] - A1[:, :VS].mean(1, keepdims=True)], 1).std())
    a2_within = float(np.concatenate([A2[:, VS:] - A2[:, VS:].mean(1, keepdims=True),
                                      A2[:, :VS] - A2[:, :VS].mean(1, keepdims=True)], 1).std())

    # ---- position block: P[i, j] = W1[V+i, V+j], causal side j < i
    P = W1[V:, V:]
    rows, X, Y = [], [], []
    for i in range(1, T):
        js = np.arange(i)
        y_ = P[i, js]; d = (i - js).astype(float)
        y_ = y_ - y_.mean(); d = d - d.mean()               # remove the row's gauge constant
        X.append(d); Y.append(y_)
    X = np.concatenate(X); Y = np.concatenate(Y)
    p_ls = float(-(X @ Y) / (X @ X))
    resid = Y + p_ls * X
    r2 = 1 - float(resid @ resid) / float(Y @ Y)
    # profile relative to distance 1, averaged over rows that reach that distance
    prof = []
    for dd in range(1, 16):
        vals = [P[i, i - dd] - P[i, i - 1] for i in range(dd, T)]
        prof.append(float(np.mean(vals)))

    # ---- layer 2 4x4 blocks
    B = W2[0:VS, d1:d1 + VS]; C = W2[d1:d1 + VS, d1:d1 + VS]
    off = ~np.eye(VS, dtype=bool)
    b2 = float(np.diag(B).mean() - B[off].mean()); c2 = float(np.diag(C).mean() - C[off].mean())
    b_spread = float(np.diag(B).std()); c_spread = float(np.diag(C).std())

    th_proj = torch.tensor([p_ls, a1_rows.mean(), a2_rows.mean(), b2, c2], dtype=torch.float32)
    te = M.load(a, 'test'); x, y = te['x'].to(dev), te['y'].to(dev)
    with torch.no_grad():
        ce_proj = float(M.ce(th_proj.to(dev), x, y, a))
        th5 = M.f19_final(a)
        ce_5 = float(M.ce(th5.to(dev), x, y, a))
    m18 = json.load(open(os.path.join(M.RUN18, f'metrics_seed0_x_{a}.json')))
    r = dict(alpha=a, ce_folder18=m18['final_test'], ce_projected=ce_proj, ce_five_trained=ce_5,
             ideal=M.ideal_ce(a),
             theta_projected=dict(zip(M.NAMES, [round(float(v), 4) for v in th_proj])),
             theta_five_trained=dict(zip(M.NAMES, [round(float(v), 4) for v in th5])),
             a1_rowwise=np.round(a1_rows, 3).tolist(), a2_rowwise=np.round(a2_rows, 3).tolist(),
             a1_within_group_std=a1_within, a2_within_group_std=a2_within,
             b2_diag_std=b_spread, c2_diag_std=c_spread,
             position_ls_slope=p_ls, position_ls_R2=r2,
             position_profile_rel_d1=[round(v, 3) for v in prof])
    res[str(a)] = r
    print(f"\n=== alpha={a}")
    print(f"  CE: folder18 {r['ce_folder18']:.5f} | projected-to-5 {ce_proj:.5f} "
          f"| five-trained {ce_5:.5f} | ideal {r['ideal']:.5f}")
    print(f"  projected theta : {r['theta_projected']}")
    print(f"  five-trained    : {r['theta_five_trained']}")
    print(f"  a1 per query row: {r['a1_rowwise']}   within-group std {a1_within:.3f}")
    print(f"  a2 per query row: {r['a2_rowwise']}   within-group std {a2_within:.3f}")
    print(f"  B diag std {b_spread:.3f}   C diag std {c_spread:.3f}")
    print(f"  position block: LS slope {p_ls:.3f}  R^2 {r2:.3f}")
    print(f"  profile P[i,i-d]-P[i,i-1], d=1..15: {r['position_profile_rel_d1']}")
json.dump(res, open(os.path.join(M.F22, 'runs', 'e1_projection.json'), 'w'), indent=2)
