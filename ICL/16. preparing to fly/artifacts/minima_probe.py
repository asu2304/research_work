"""How far are the converged solutions from a true local minimum, and can we reach it?

(A) exact line search along -grad, full train, all 3 configs x all 7 thresholds
(B) full-batch L-BFGS from each config at two thresholds, tracking loss / ||g|| / ||theta||

Writes _minima_probe.json.
"""
import importlib.util, json, os, time
import numpy as np
import torch
import torch.nn.functional as F

F14 = '../14. elimination final convergence confusion'
_s = importlib.util.spec_from_file_location('tw', os.path.join(F14, 'train_with_equiprobable.py'))
TW = importlib.util.module_from_spec(_s); _s.loader.exec_module(TW)
Model = TW.SimplifiedTransformerMLP

V, VS, T = 8, 4, 100
D = V + T
K_LOG = TW.CFG['freeze_hidden_k']
TS, TE = TW.CFG['token_start'], T
DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CDIR = os.path.join(F14, 'runs', 'equi_opt-adam_lr-0.003_b-256_init-normal_ep-2000')
DATA = os.path.join(F14, 'dataset')
THRS = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
LBFGS_THRS = [0.4, 0.8]
BATCH = 512

_cache = {}
def data(split, thr):
    k = (split, thr)
    if k not in _cache:
        _cache.clear()
        b = torch.load(os.path.join(DATA, split, f'data_x_{thr}.pt'), weights_only=False)
        _cache[k] = (b['x'], b['y'])
    return _cache[k]


def load_separate(thr):
    m = Model(V, T, D, VS, K_LOG, True, 'separate').to(DEV)
    m.load_state_dict(torch.load(os.path.join(CDIR, f'model_separate_seed0_x_{thr}.pth'),
                                 map_location=DEV, weights_only=True))
    return m


def load_combined(thr):
    m = Model(V, T, D, VS, K_LOG, True, 'combined').to(DEV)
    m.load_state_dict(torch.load(os.path.join(CDIR, f'model_combined_seed0_x_{thr}.pth'),
                                 map_location=DEV, weights_only=True))
    return m


def collapse(msep):
    m = Model(V, T, D, VS, K_LOG, True, 'combined').to(DEV)
    with torch.no_grad():
        for src, dst in ((msep.layer1, m.layer1), (msep.layer2, m.layer2)):
            dst.W_qk.copy_(src.W_q.weight.t() @ src.W_k.weight)
            dst.W_ov.copy_(src.W_v.weight.t() @ src.W_o.weight.t())
        for n in ('ff1', 'ff2', 'fc_out'):
            getattr(m, n).weight.copy_(getattr(msep, n).weight)
            getattr(m, n).bias.copy_(getattr(msep, n).bias)
    return m


def prep(m):
    for p in m.parameters():
        p.requires_grad_(False)
    ps = TW.trainable(m)
    for p in ps:
        p.requires_grad_(True)
    return ps


@torch.no_grad()
def loss_only(m, split, thr):
    x, y = data(split, thr)
    n = x.shape[0] * (TE - TS)
    tot = 0.0
    for i in range(0, x.shape[0], BATCH):
        xb = x[i:i + BATCH].to(DEV); yb = y[i:i + BATCH, TS:TE].to(DEV)
        tot += F.cross_entropy(m(xb)[:, TS:TE].reshape(-1, V), yb.reshape(-1),
                               reduction='sum').item()
    return tot / n


def loss_and_grad(m, ps, split, thr):
    x, y = data(split, thr)
    n = x.shape[0] * (TE - TS)
    for p in ps:
        p.grad = None
    tot = 0.0
    for i in range(0, x.shape[0], BATCH):
        xb = x[i:i + BATCH].to(DEV); yb = y[i:i + BATCH, TS:TE].to(DEV)
        l = F.cross_entropy(m(xb)[:, TS:TE].reshape(-1, V), yb.reshape(-1),
                            reduction='sum') / n
        l.backward(); tot += l.item()
    return tot


CONFIGS = [('separate @ separate', load_separate),
           ('combined @ combined', load_combined),
           ('combined @ collapsed', lambda t: collapse(load_separate(t)))]

OUT = {'line_search': [], 'lbfgs': []}

# ── (A) exact line search along -grad ────────────────────────────────────────
ETAS = [3e-4, 1e-3, 3e-3, 1e-2, 2e-2, 3e-2, 5e-2, 1e-1]
print('=== (A) line search along -grad, full train ===', flush=True)
print(f'{"config":22s} {"thr":>4s} {"L0":>8s} {"||g||":>9s} {"eta*":>8s} {"dL*":>11s} '
      f'{"gT H g":>10s} {"pred min dL":>12s}', flush=True)
for name, mk in CONFIGS:
    for thr in THRS:
        m = mk(thr); ps = prep(m)
        L0 = loss_and_grad(m, ps, 'train', thr)
        g = [p.grad.detach().clone() for p in ps]
        gn = float(torch.cat([t.reshape(-1) for t in g]).norm())
        d = [t / gn for t in g]
        base = [p.detach().clone() for p in ps]
        vals = {}
        with torch.no_grad():
            for e in ETAS:
                for p, b0, dd in zip(ps, base, d):
                    p.copy_(b0 - e * dd)
                vals[e] = loss_only(m, 'train', thr) - L0
            for p, b0 in zip(ps, base):
                p.copy_(b0)
        e_star = min(vals, key=vals.get)
        dL_star = vals[e_star]
        e1 = ETAS[0]
        curv = 2 * (vals[e1] + e1 * gn) / e1 ** 2
        pred = -gn ** 2 / (2 * curv) if curv > 0 else float('nan')
        OUT['line_search'].append(dict(config=name, thr=thr, L0=L0, gnorm=gn,
                                       etas=ETAS, dL=[vals[e] for e in ETAS],
                                       eta_star=e_star, dL_star=dL_star,
                                       curvature=curv, pred_min_dL=pred))
        print(f'{name:22s} {thr:4.1f} {L0:8.4f} {gn:9.3e} {e_star:8.3g} {dL_star:+11.3e} '
              f'{curv:10.3e} {pred:+12.3e}', flush=True)
        del m
    print(flush=True)
json.dump(OUT, open('_minima_probe.json', 'w'), indent=1)

# ── (B) full-batch L-BFGS ────────────────────────────────────────────────────
MAXIT = 60
print(f'=== (B) full-batch L-BFGS, {MAXIT} iterations ===', flush=True)
for name, mk in CONFIGS:
    for thr in LBFGS_THRS:
        m = mk(thr); ps = prep(m)
        L0 = loss_and_grad(m, ps, 'train', thr)
        g0 = float(torch.cat([p.grad.reshape(-1) for p in ps]).norm())
        te0 = loss_only(m, 'test', thr)
        th0 = float(torch.cat([p.detach().reshape(-1) for p in ps]).norm())
        opt = torch.optim.LBFGS(ps, lr=1.0, max_iter=1, history_size=10,
                                line_search_fn='strong_wolfe',
                                tolerance_grad=0.0, tolerance_change=0.0)
        hist = []
        t0 = time.time()
        for it in range(MAXIT):
            def closure():
                return loss_and_grad(m, ps, 'train', thr)
            opt.step(closure)
            L = loss_and_grad(m, ps, 'train', thr)
            gn = float(torch.cat([p.grad.reshape(-1) for p in ps]).norm())
            hist.append((L, gn))
            if it % 10 == 0 or it == MAXIT - 1:
                print(f'  {name:22s} thr={thr} it {it:3d}  L={L:.6f}  ||g||={gn:.3e}  '
                      f'({time.time() - t0:.0f}s)', flush=True)
        Lf = hist[-1][0]; gf = hist[-1][1]
        tef = loss_only(m, 'test', thr)
        thf = float(torch.cat([p.detach().reshape(-1) for p in ps]).norm())
        OUT['lbfgs'].append(dict(config=name, thr=thr, L0=L0, Lf=Lf, g0=g0, gf=gf,
                                 test0=te0, testf=tef, theta0=th0, thetaf=thf,
                                 hist=hist, iters=MAXIT))
        print(f'  -> {name} thr={thr}: train {L0:.6f} -> {Lf:.6f} ({Lf-L0:+.2e}), '
              f'||g|| {g0:.2e} -> {gf:.2e}, test {te0:.6f} -> {tef:.6f} ({tef-te0:+.2e}), '
              f'||theta|| {th0:.1f} -> {thf:.1f}\n', flush=True)
        json.dump(OUT, open('_minima_probe.json', 'w'), indent=1)
        del m, opt
print('done -> _minima_probe.json')
