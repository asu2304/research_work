"""
E17 -- Exact formula for the first veto push, and the law  'veto push ~ noise fraction x induction pull'.

At theta = 0 both layers attend uniformly: r_j = h_j = token histogram of x_<j (strict; r_0 = e_{x_0}),
f_i = h_i, w_i = 1/(f_i[y_i] + b[y_i]). First order in b2 (exact mixed derivative at 0):

    -U0_a1 = -(1/N) sum_i w_i sigma_i Cov_{j<i, uniform}( 1[x_j = y_i],  nubar_j * h_j[x_i] )
    -U0_p  = +(1/N) sum_i w_i sigma_i Cov_{j<i, uniform}( 1[x_j = y_i],  rec_j(x_i) )

where nubar_j = noise fraction of x_<j (because d r_j[v] / d a1 = -nubar_j h_j[v] for signal v), and
rec_j(v) = -(1/j) sum_{k<j} ((j-k) - dbar_j) 1[x_k = v]  (d r_j[v]/dp: how much v sits RECENTLY in x_<j).
Compare: the initial INDUCTION pull  -dL/db2|_0 = (1/N) sum_i w_i sigma_i Cov( 1[x_j = y_i], h_j[x_i] ).
If nubar_j were the same for every key, -U0_a1 = -nubar * (-dL/db2|_0) exactly: the veto is seeded in
proportion to the noise fraction times the induction signal.
Checked against autograd (E16).
"""
import json, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
import five as M

dev = torch.device(sys.argv[1] if len(sys.argv) > 1 else 'cuda:2')
e16 = {r['alpha']: r for r in json.load(open(os.path.join(M.F22, 'runs', 'e16_first_layer1_direction.json')))}
out = []
print(f"{'alpha':>6} | {'-U0_a1 formula':>15} {'autograd':>10} | {'-U0_p formula':>14} {'autograd':>10} | "
      f"{'induction pull':>14} {'mean nubar':>10} {'nubar x pull':>13}")
for a in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99):
    tr = M.load(a, 'train'); x, y = tr['x'][:8000].to(dev), tr['y'][:8000].to(dev)
    B, T = x.shape
    oh = torch.nn.functional.one_hot(x, M.V).double()
    nz = (x >= M.VS).double()
    cnt = torch.cumsum(oh, 1); ncnt = torch.cumsum(nz, 1)
    j = torch.arange(T, device=dev).double()
    nk = j.clamp_min(1)
    h = torch.zeros_like(oh); h[:, 1:] = cnt[:, :-1] / nk[1:].view(1, -1, 1); h[:, 0] = oh[:, 0]
    nub = torch.zeros(B, T, device=dev, dtype=torch.double); nub[:, 1:] = ncnt[:, :-1] / nk[1:]
    # rec_j(v) = -(1/j) sum_{k<j} ((j-k) - dbar_j) 1[x_k=v], dbar_j = (j+1)/2
    kc = torch.cumsum(oh * j.view(1, -1, 1), 1)            # sum_{k<=t} k 1[x_k=v]
    s_k = torch.zeros_like(oh); s_k[:, 1:] = kc[:, :-1]       # sum_{k<j} k 1[x_k=v]
    c_k = torch.zeros_like(oh); c_k[:, 1:] = cnt[:, :-1]      # sum_{k<j} 1[x_k=v]
    dbar = (j + 1) / 2
    rec = -((j.view(1, -1, 1) * c_k - s_k) - dbar.view(1, -1, 1) * c_k) / nk.view(1, -1, 1)
    rec[:, 0] = 0
    b = M.readout_bias(a, dev).double()
    N = B * (T - M.TS)
    tot_a1 = tot_p = tot_b2 = 0.0
    for i in range(M.TS, T):
        sig = (x[:, i] < M.VS).double()
        fy = h[:, i].gather(1, y[:, i:i + 1]).squeeze(1)
        w = 1.0 / (fy + b[y[:, i]])
        hit = (x[:, :i] == y[:, i:i + 1]).double()                         # (B, i)
        v = x[:, i].clamp_max(M.VS - 1)
        hx = h[:, :i].gather(2, v.view(-1, 1, 1).expand(-1, i, 1)).squeeze(-1)   # h_j[x_i]
        rx = rec[:, :i].gather(2, v.view(-1, 1, 1).expand(-1, i, 1)).squeeze(-1)
        def cov(phi):
            return ((hit - fy.unsqueeze(1)) * phi).mean(1)
        tot_a1 += float((w * sig * cov(nub[:, :i] * hx)).sum())
        tot_p += float((w * sig * cov(rx)).sum())
        tot_b2 += float((w * sig * cov(hx)).sum())
    negU_a1, negU_p, pull = -tot_a1 / N, tot_p / N, tot_b2 / N
    mean_nub = float(nz[:, :M.TS].mean())
    ag = e16[a]['negU0']
    r = dict(alpha=a, negU0_a1_formula=negU_a1, negU0_a1_autograd=ag[1], negU0_p_formula=negU_p,
             negU0_p_autograd=ag[0], induction_pull=pull, mean_nubar=mean_nub, approx=-mean_nub * pull)
    out.append(r)
    print(f"{a:>6} | {negU_a1:>+15.3e} {ag[1]:>+10.3e} | {negU_p:>+14.3e} {ag[0]:>+10.3e} | "
          f"{pull:>+14.3e} {mean_nub:>10.3f} {-mean_nub * pull:>+13.3e}", flush=True)
json.dump(out, open(os.path.join(M.F22, 'runs', 'e17_veto_seed_formula.json'), 'w'), indent=1)
