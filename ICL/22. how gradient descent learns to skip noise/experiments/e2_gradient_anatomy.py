"""
E2 -- Anatomy of the gradient along GD's trajectory, as a function of noise density.

For each alpha's trajectory in runs/<tag>/traj_x_<alpha>.npz and each step k in a log-spaced
set, at theta_k:

  1. DECOMPOSE the full-batch gradient by the type of the query position i:
        post-noise   x_i is noise  -> the state must be recovered by skipping back
        post-signal  x_i is signal -> the state is x_i itself
     g = g_noise + g_signal exactly (each is that type's contribution to the mean loss).
     Also the per-type mean loss.

  2. LAYER 1 QUALITY: mass r_i[prev_signal_i] that layer 1 puts on the true last signal before
     i, overall and by the gap (how far back that signal is). This is "has it learned to skip".

  3. LAYER 2 QUALITY: mass of f_i on the true successors of state(i), split by query type.

  4. COUNTERFACTUAL: the same decomposition for (a2, b2, c2) at theta_k but with layer 1 replaced
     by PERFECT state recovery -- the pull the state block would feel if layer 1 were done.

Writes runs/<tag>/anatomy_x_<alpha>.json.
usage: python e2_gradient_anatomy.py --tag main --device cuda:0 [--alphas ...]
"""
import argparse, glob, json, os, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
import five as M

ap = argparse.ArgumentParser()
ap.add_argument('--tag', default='main')
ap.add_argument('--device', default='cuda:0')
ap.add_argument('--alphas', type=float, nargs='*', default=None)
ap.add_argument('--nseq', type=int, default=25000, help='train sequences used (full = 25000)')
a = ap.parse_args()
dev = torch.device(a.device)
rdir = os.path.join(M.F22, 'runs', a.tag)
STEPS = [0, 1, 2, 3, 4, 5, 7, 10, 15, 20, 30, 50, 75, 100, 150, 200, 300, 500, 750, 1000, 2000, 3000]


def succ_table(trans, n):
    S = torch.zeros(n, M.VS, M.VS, dtype=torch.bool)
    for b in range(n):
        for v in range(M.VS):
            S[b, v, torch.as_tensor(trans[b][v])] = True
    return S


def analyse(alpha):
    tr = M.load(alpha, 'train')
    n = min(a.nseq, tr['x'].shape[0])
    x, y = tr['x'][:n].to(dev), tr['y'][:n].to(dev)
    succ = succ_table(tr['transitions'], n).to(dev)
    st = M.state_of(x)[:, M.TS:]                    # state generating y_i
    ps = M.prev_signal(x)                           # last signal strictly before i
    xw = x[:, M.TS:]
    is_noise = (xw >= M.VS)
    w_noise, w_sig = is_noise.float(), (~is_noise).float()
    N = float(n * (M.T - M.TS))
    # gap: distance from i back to the last signal strictly before i
    idx = torch.arange(M.T, device=dev).expand(n, M.T)
    last_sig_idx = torch.where(x < M.VS, idx, torch.full_like(idx, -1)).cummax(1).values
    gap = torch.full_like(idx, -1); gap[:, 1:] = idx[:, 1:] - last_sig_idx[:, :-1]
    gap = torch.where(last_sig_idx.roll(1, 1) >= 0, gap, torch.full_like(gap, -1))
    gw, psw = gap[:, M.TS:], ps[:, M.TS:]
    # successor mask of the true state at each scored position: (n, 50, V)
    sv = st.clamp_min(0)
    smask = torch.zeros(n, M.T - M.TS, M.V, dtype=torch.bool, device=dev)
    smask[..., :M.VS] = succ[torch.arange(n, device=dev).unsqueeze(1), sv]
    smask &= (st >= 0).unsqueeze(-1)

    d = np.load(os.path.join(rdir, f'traj_x_{M.afmt(alpha)}.npz'))
    TH = d['theta']
    rows = []
    for k in [s for s in STEPS if s < len(TH)]:
        th = torch.tensor(TH[k], dtype=torch.float32, device=dev, requires_grad=True)
        g_type, L_type, gp = {}, {}, {}
        for name, w in (('noise', w_noise), ('signal', w_sig)):
            if float(w.sum()) == 0:                       # alpha = 1: no noise positions at all
                g_type[name] = gp[name] = np.zeros(5); L_type[name] = float('nan'); continue
            L = M.ce(th, x, y, alpha, weights=w) * (float(w.sum()) / N)   # contribution to mean
            g, = torch.autograd.grad(L, th)
            g_type[name] = g.cpu().double().numpy()
            L_type[name] = float(L) * N / float(w.sum())                 # per-position mean loss
            L = M.ce(th, x, y, alpha, weights=w, perfect_layer1=True) * (float(w.sum()) / N)
            g, = torch.autograd.grad(L, th)
            gp[name] = g.cpu().double().numpy()
        with torch.no_grad():
            rec, mass_ok = [], {'noise': 0.0, 'signal': 0.0}
            rec_by_gap = {}
            for i in range(0, n, 5000):
                f, P = M.forward(th, x[i:i + 5000], parts=True)
                r = P['r'][:, M.TS:]
                ok = (psw[i:i + 5000] >= 0)
                rr = r.gather(-1, psw[i:i + 5000].clamp_min(0).unsqueeze(-1)).squeeze(-1)
                rec.append(rr[ok])
                for gg in range(1, 7):
                    m_ = ok & (gw[i:i + 5000] == gg)
                    s_, c_ = rec_by_gap.get(gg, (0.0, 0))
                    rec_by_gap[gg] = (s_ + float(rr[m_].sum()), c_ + int(m_.sum()))
                fm = (f[:, M.TS:] * smask[i:i + 5000].float()).sum(-1)
                mass_ok['noise'] += float(fm[is_noise[i:i + 5000]].sum())
                mass_ok['signal'] += float(fm[~is_noise[i:i + 5000]].sum())
            rec = torch.cat(rec)
        rows.append(dict(
            step=k, theta=TH[k].tolist(),
            grad_noise=g_type['noise'].tolist(), grad_signal=g_type['signal'].tolist(),
            loss_noise=L_type['noise'], loss_signal=L_type['signal'],
            grad_perfectL1_noise=gp['noise'].tolist(), grad_perfectL1_signal=gp['signal'].tolist(),
            layer1_state_mass=float(rec.mean()),
            layer1_state_mass_by_gap={g_: s_ / max(c_, 1) for g_, (s_, c_) in rec_by_gap.items()},
            layer2_succ_mass_postnoise=mass_ok["noise"] / max(float(w_noise.sum()), 1.0),
            layer2_succ_mass_postsignal=mass_ok['signal'] / float(w_sig.sum())))
    out = dict(alpha=alpha, frac_postnoise=float(w_noise.mean()), n_seq=n, rows=rows)
    json.dump(out, open(os.path.join(rdir, f'anatomy_x_{M.afmt(alpha)}.json'), 'w'), indent=1)
    return out


alphas = a.alphas or sorted(float(os.path.basename(f)[7:-4]) for f in glob.glob(os.path.join(rdir, 'traj_x_*.npz')))
for al in alphas:
    t0 = time.time()
    o = analyse(al)
    r0, rl = o['rows'][0], o['rows'][-1]
    print(f"alpha={al}: step0 g_c2 noise {r0['grad_noise'][4]:+.2e} signal {r0['grad_signal'][4]:+.2e} | "
          f"final L1 state mass {rl['layer1_state_mass']:.3f}  L2 succ mass post-noise "
          f"{rl['layer2_succ_mass_postnoise']:.3f} post-signal {rl['layer2_succ_mass_postsignal']:.3f} "
          f"({time.time()-t0:.0f}s)", flush=True)
