"""
E1b -- FUNCTIONAL projection of folder 18's solution onto the five numbers.

E1 found that a least-squares fit of the position block fails (CE 1.896 vs 1.698 at alpha 0.6)
because GD's free position profile is a CAPPED ramp: steep for ~5 positions, then flat. The far,
flat cells dominate a plain least-squares fit but carry essentially no attention.

So project by FUNCTION instead of by entry, and still without touching the loss:
    layer 1:  choose (p, a1) minimising  E_data KL( A1_folder18 || A1_five )   over queries
              in the scored window -- make the five-number layer 1 attend like GD's does.
    layer 2:  a2, b2, c2 as the block contrasts of E1 (their within-group spread is tiny).
Also reported: a near-field slope (distances 1..5 only) as a cruder entry-level alternative.
"""
import json, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
import five as M

dev = torch.device(sys.argv[1] if len(sys.argv) > 1 else 'cuda:0')
V, VS, T, d1 = M.V, M.VS, M.T, M.V + M.T
e1 = json.load(open(os.path.join(M.F22, 'runs', 'e1_projection.json')))
out = {}
for a in (0.6, 0.8, 1.0):
    sd = torch.load(os.path.join(M.RUN18, f'model_final_seed0_x_{a}.pth'), map_location='cpu')
    W1 = (sd['layer1.W_qk'].double() / M.SQ).float().to(dev)
    te = M.load(a, 'test'); x, y = te['x'].to(dev), te['y'].to(dev)
    xs = x[:1500]
    oh = torch.nn.functional.one_hot(xs, V).float()
    pos = torch.eye(T, device=dev)
    s = torch.cat([oh, pos.expand(xs.shape[0], T, T)], -1)            # (B, T, 108)
    mask = M.causal_mask(T, dev)
    with torch.no_grad():
        A18 = torch.softmax(s @ W1 @ s.transpose(1, 2) + mask, -1)   # folder 18's layer-1 attention
    q = slice(M.TS, T)                                               # scored queries only
    noise = (xs >= VS).float()
    D = M.dist(T, dev)
    th = torch.tensor([1.0, -5.0], device=dev, requires_grad=True)   # (p, a1)
    opt = torch.optim.Adam([th], lr=0.05)
    for it in range(3000):
        s5 = -th[0] * D.unsqueeze(0) + th[1] * noise.unsqueeze(1)
        logA5 = torch.log_softmax(s5 + mask, -1)
        P18 = A18[:, q]
        term = P18 * (torch.log(P18.clamp_min(1e-30)) - logA5[:, q])
        term = torch.where(P18 > 0, term, torch.zeros_like(term))     # 0 * log 0 := 0 (masked keys)
        kl = term.sum(-1).mean()
        opt.zero_grad(); kl.backward(); opt.step()
    p_f, a1_f = [float(v) for v in th.detach()]
    pr = e1[str(a)]
    near = -float(np.polyfit(np.arange(1, 6), pr['position_profile_rel_d1'][:5], 1)[0])
    tp = pr['theta_projected']
    with torch.no_grad():
        th_func = torch.tensor([p_f, a1_f, tp['a2'], tp['b2'], tp['c2']], device=dev)
        th_near = torch.tensor([near, tp['a1'], tp['a2'], tp['b2'], tp['c2']], device=dev)
        ce_func = float(M.ce(th_func, x, y, a)); ce_near = float(M.ce(th_near, x, y, a))
    r = dict(alpha=a, ce_folder18=pr['ce_folder18'], ce_five_trained=pr['ce_five_trained'],
             ce_naive_ls=pr['ce_projected'], ce_nearfield=ce_near, ce_functional=ce_func,
             kl_layer1=float(kl), p_functional=p_f, a1_functional=a1_f, p_nearfield=near,
             theta_functional=[round(v, 4) for v in th_func.tolist()],
             theta_five_trained=pr['theta_five_trained'])
    out[str(a)] = r
    print(f"alpha={a}: folder18 {r['ce_folder18']:.5f} | five-trained {r['ce_five_trained']:.5f} | "
          f"naive-LS {r['ce_naive_ls']:.5f} | near-field {ce_near:.5f} | FUNCTIONAL {ce_func:.5f}"
          f"   (p={p_f:.3f}, a1={a1_f:.3f}, layer-1 KL {float(kl):.4f})", flush=True)
json.dump(out, open(os.path.join(M.F22, 'runs', 'e1b_functional_projection.json'), 'w'), indent=2)
