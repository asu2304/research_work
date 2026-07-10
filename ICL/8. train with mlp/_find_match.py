"""Offline: two improved hand GELU readouts vs the GD minimal-MLP.
  logc : signal = log(freq+eps)  +  noise = CONSTANT exact floor log((1-a)/n_noise)   -> targets ctx-count (=mlp)
  eq   : signal = step->log(a/2) (observed) / OFF (unobserved) + noise = log((1-a)/n_noise) -> targets context-Bayes
Search the small floor/OFF params; compare CE to mlp/ctx/ideal. Dump _match.json."""
import json, os, glob, time, math
import numpy as np, torch, torch.nn.functional as F

nb = json.load(open('analyse.ipynb'))
src = ''.join(nb['cells'][0]['source'])
defs = src[:src.index('# ════════════════════════════════════════════════════════════════════════════════\n# LAZY')]
g = {}; exec(defs, g)
device = g['device']
load_data, _wlogits, _ce_logits = g['load_data'], g['_wlogits'], g['_ce_logits']
mlp_ce, alpha_ce, ideal_alpha_ce = g['mlp_ce'], g['alpha_ce'], g['ideal_alpha_ce']
DATA_DIR, TS, TE = g['DATA_DIR'], g['TOKEN_START'], g['TOKEN_END']
print('device:', device, flush=True)

def _gr(u, gain=200.0): return F.gelu(gain * u) / gain
def step01(c, tau, w):   return ((_gr(c - tau) - _gr(c - tau - w)) / w).clamp(0.0, 1.0)

@torch.no_grad()
def logc_ce(L, yw, V, Vs, alpha, eps_s):                       # ctx-count: log on signal, exact const floor on noise
    z = torch.empty_like(L)
    z[..., :Vs] = torch.log(L[..., :Vs].clamp_min(0.0) + eps_s)
    z[..., Vs:] = math.log((1.0 - alpha) / (V - Vs))
    return _ce_logits(z, yw)

@torch.no_grad()
def eq_ce(L, yw, V, Vs, alpha, tau, w, off):                   # context-Bayes: step + tuned OFF
    on = math.log(alpha / 2.0); noise = math.log((1.0 - alpha) / (V - Vs))
    s01 = step01(L[..., :Vs], tau, w)
    z = torch.empty_like(L)
    z[..., :Vs] = off + (on - off) * s01
    z[..., Vs:] = noise
    return _ce_logits(z, yw)

EPS = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2]
TAU, W = [0.004, 0.01, 0.02, 0.04], [0.004]
OFF = [-9., -7., -6., -5., -4., -3.5, -3.]

thrs = sorted(float(os.path.basename(f)[7:-3]) for f in glob.glob(os.path.join(DATA_DIR, 'test', 'data_x_*.pt')))
out = {}; t0 = time.time()
for thr in thrs:
    x, y, sl, V, blob = load_data('test', thr); yw = y[:, TS:TE]; Vs = int(blob['vocab_set1'])
    L = _wlogits(blob)
    bl = min((logc_ce(L, yw, V, Vs, thr, e), e) for e in EPS)
    be = min((eq_ce(L, yw, V, Vs, thr, t, w, o), (t, w, o)) for t in TAU for w in W for o in OFF)
    mlp = mlp_ce(thr, sl, V, x, y); ctx = alpha_ce(blob, thr); ideal = ideal_alpha_ce(blob, thr)
    out[round(thr, 3)] = dict(logc=round(bl[0], 4), logc_eps=bl[1],
                              eq=round(be[0], 4), eq_par=be[1],
                              mlp=(round(mlp, 4) if mlp is not None else None),
                              ctx=round(ctx, 4), ideal=round(ideal, 4))
    m = '' if mlp is None else f'{mlp:.4f}'
    print(f"thr={thr}  logc={bl[0]:.4f}  eq={be[0]:.4f}  mlp={m}  ctx={ctx:.4f}  ideal={ideal:.4f}  ({time.time()-t0:.0f}s)", flush=True)
json.dump(out, open('_match.json', 'w'), indent=1)
print('WROTE _match.json', flush=True)
