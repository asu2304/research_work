"""
E24 -- Does the noise-skipping mechanism depend on the hand-set readout bias b?

Check 1 (seconds): layer 1's first direction. At theta=0 layer 1 has zero gradient (Lemma 1); its first
direction is -(b2*U0 + c2*W0). We measure U0 = d/db2 grad_(p,a1) L and W0 = d/dc2 grad_(p,a1) L at theta=0
(central differences of autograd gradients), with (i) the designed bias and (ii) a flat bias 0.01 on every
token. "Skip" = -dL/dp > 0 (towards recency) and -dL/da1 < 0 (towards vetoing noise).
Also the gradient at the actual point after Adam's first step under each bias.

Check 2 (~6 min): train from zero with a LEARNABLE bias, b = exp(beta), beta initialised to log(0.01) on all
8 tokens (no knowledge of the noise rate), Adam lr 1.0 on the raw five (folder-19 convention) and lr 0.05 on
beta, 1000 full-batch steps. Compare with runs/main (fixed designed bias): five numbers, test CE, learned b,
layer-1 state mass, layer-2 successor mass after noise / after signal.

Writes runs/e24_bias_dependence.json and runs/learnable_bias/traj_x_<alpha>.npz. Touches nothing else.
usage: python e24_bias_dependence.py cuda:2
"""
import json, math, os, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
import five as M

dev = torch.device(sys.argv[1] if len(sys.argv) > 1 else 'cuda:2')
ALPHAS = (0.2, 0.4, 0.6, 0.8, 0.9)
STEPS, LR_BETA, CHUNK = 1000, 0.05, 5000
out_dir = os.path.join(M.F22, 'runs', 'learnable_bias'); os.makedirs(out_dir, exist_ok=True)
res = {'check1': {}, 'check2': {}}


def ce_b(theta, x, y, b):
    tot = 0.0
    for i in range(0, x.shape[0], CHUNK):
        f = M.forward(theta, x[i:i + CHUNK])[:, M.TS:]
        tot = tot + M.token_nll(f, y[i:i + CHUNK, M.TS:], b).sum()
    return tot / (x.shape[0] * (M.T - M.TS))


def grad_at(theta, x, y, b):
    th = torch.tensor(theta, dtype=torch.float64, device=dev, requires_grad=True)
    g, = torch.autograd.grad(ce_b(th, x, y, b.double()), th)
    return g.cpu().numpy()


@torch.no_grad()
def metrics(theta, x, y, succ):
    th = torch.tensor(theta, dtype=torch.float32, device=dev)
    st = M.state_of(x)[:, M.TS:]; ps = M.prev_signal(x)[:, M.TS:]
    isn = (x[:, M.TS:] >= M.VS)
    rec, okn, oks, nn_, ns = 0.0, 0.0, 0.0, 0, 0
    cnt = 0
    for i in range(0, x.shape[0], CHUNK):
        f, P = M.forward(th, x[i:i + CHUNK], parts=True)
        r, f = P['r'][:, M.TS:], f[:, M.TS:]
        p_ = ps[i:i + CHUNK]; ok = p_ >= 0
        rec += float(r.gather(-1, p_.clamp_min(0).unsqueeze(-1)).squeeze(-1)[ok].sum()); cnt += int(ok.sum())
        s_ = st[i:i + CHUNK].clamp_min(0)
        sm = torch.zeros(*s_.shape, M.V, dtype=torch.bool, device=dev)
        sm[..., :M.VS] = succ[i:i + CHUNK][torch.arange(s_.shape[0], device=dev).unsqueeze(1), s_]
        sm &= (st[i:i + CHUNK] >= 0).unsqueeze(-1)
        fm = (f * sm.float()).sum(-1); n_ = isn[i:i + CHUNK]
        okn += float(fm[n_].sum()); nn_ += int(n_.sum()); oks += float(fm[~n_].sum()); ns += int((~n_).sum())
    return dict(layer1_state_mass=rec / cnt, succ_mass_postnoise=okn / max(nn_, 1), succ_mass_postsignal=oks / ns)


def succ_table(trans):
    S = torch.zeros(len(trans), M.VS, M.VS, dtype=torch.bool)
    for k, t in enumerate(trans):
        for v in range(M.VS):
            S[k, v, torch.as_tensor(t[v])] = True
    return S.to(dev)


# ------------------------------------------------------------------ check 1
print('CHECK 1: layer-1 first direction (p, a1) | skip = -dL/dp > 0 and -dL/da1 < 0', flush=True)
for a in ALPHAS:
    d = M.load(a, 'train'); x, y = d['x'][:5000].to(dev), d['y'][:5000].to(dev)
    row = {}
    for name, b in (('designed', M.readout_bias(a, dev)), ('flat_0.01', torch.full((M.V,), 0.01, device=dev))):
        eps = 1e-3
        U = (grad_at([0, 0, 0, eps, 0], x, y, b) - grad_at([0, 0, 0, -eps, 0], x, y, b))[:2] / (2 * eps)
        W = (grad_at([0, 0, 0, 0, eps], x, y, b) - grad_at([0, 0, 0, 0, -eps], x, y, b))[:2] / (2 * eps)
        g0 = grad_at([0, 0, 0, 0, 0], x, y, b)
        th1 = [0.0, 0.0] + list(-np.sign(g0[2:]) / M.SQ)          # Adam's first step: -sign * lr/SQ logits
        g1 = grad_at(th1, x, y, b)
        row[name] = dict(neg_U0=(-U).tolist(), neg_W0=(-W).tolist(), first_grad=g0.tolist(),
                         theta_after_step1=th1, neg_grad_p_a1_after_step1=(-g1[:2]).tolist())
        print(f'  alpha={a} {name:9s}: -U0 (p,a1) = ({-U[0]:+.2e}, {-U[1]:+.2e})  -W0 = ({-W[0]:+.2e}, {-W[1]:+.2e})'
              f' | step-1 theta a2,b2,c2 = {np.round(th1[2:], 3)} -> -grad(p,a1) = ({-g1[0]:+.2e}, {-g1[1]:+.2e})',
              flush=True)
    res['check1'][str(a)] = row
json.dump(res, open(os.path.join(M.F22, 'runs', 'e24_bias_dependence.json'), 'w'), indent=1)

# ------------------------------------------------------------------ check 2
print('\nCHECK 2: train with a learnable bias from b = 0.01 everywhere', flush=True)
main = {r['alpha']: r for r in json.load(open(os.path.join(M.F22, 'runs', 'summary.json')))['main']}
for a in ALPHAS:
    tr, te = M.load(a, 'train'), M.load(a, 'test')
    xtr, ytr, xte, yte = tr['x'].to(dev), tr['y'].to(dev), te['x'].to(dev), te['y'].to(dev)
    model = M.FiveParam().to(dev)
    beta = torch.nn.Parameter(torch.full((M.V,), math.log(0.01), device=dev))
    opt = torch.optim.Adam([{'params': model.parameters(), 'lr': 1.0}, {'params': [beta], 'lr': LR_BETA}])
    TH, BB = np.zeros((STEPS + 1, 5)), np.zeros((STEPS + 1, M.V))
    t0 = time.time()
    for k in range(STEPS + 1):
        TH[k] = model.theta().detach().cpu().numpy(); BB[k] = beta.exp().detach().cpu().numpy()
        if k == STEPS:
            break
        opt.zero_grad(); L = ce_b(model.theta(), xtr, ytr, beta.exp()); L.backward(); opt.step()
        if k % 200 == 0:
            print(f'  alpha={a} step {k} train {float(L):.5f} theta {np.round(TH[k], 3)} '
                  f'b_sig {BB[k][:4].mean():.4f} b_noise {BB[k][4:].mean():.4f} ({time.time()-t0:.0f}s)', flush=True)
    with torch.no_grad():
        test = float(ce_b(model.theta(), xte, yte, beta.exp()))
    met = metrics(TH[-1], xte, yte, succ_table(te['transitions']))
    designed = M.readout_bias(a)
    m = dict(main.get(a, {}))
    if isinstance(m.get('theta_final'), dict):
        m['theta_final'] = [m['theta_final'][n] for n in M.NAMES]
    row = dict(alpha=a, theta_final=TH[-1].tolist(), test_ce=test, ideal=M.ideal_ce(a),
               b_learned=BB[-1].tolist(), b_designed=designed.tolist(), **met,
               main_theta_final=m.get('theta_final'), main_test_ce=m.get('test_ce_final'),
               main_layer1_state_mass=m.get('final_layer1_state_mass'),
               main_succ_mass_postnoise=m.get('final_succ_mass_postnoise'),
               main_succ_mass_postsignal=m.get('final_succ_mass_postsignal'))
    res['check2'][str(a)] = row
    np.savez(os.path.join(out_dir, f'traj_x_{M.afmt(a)}.npz'), theta=TH, b=BB, alpha=a)
    print(f'[done] alpha={a}: learnable-b  theta {np.round(TH[-1], 2)} test {test:.5f} | L1 state mass '
          f'{met["layer1_state_mass"]:.3f} succ|noise {met["succ_mass_postnoise"]:.3f} succ|signal '
          f'{met["succ_mass_postsignal"]:.3f} | b_noise learned {BB[-1][4:].mean():.4f} designed {designed[4]:.4f}',
          flush=True)
    print(f'        fixed-b (main) theta {np.round(m.get("theta_final"), 2) if m else None} test '
          f'{m.get("test_ce_final")} | L1 state mass {m.get("final_layer1_state_mass")} succ|noise '
          f'{m.get("final_succ_mass_postnoise")} succ|signal {m.get("final_succ_mass_postsignal")}', flush=True)
    json.dump(res, open(os.path.join(M.F22, 'runs', 'e24_bias_dependence.json'), 'w'), indent=1)
print('wrote runs/e24_bias_dependence.json', flush=True)
