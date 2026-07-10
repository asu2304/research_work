"""Diagnose + close the count-readout -> ctx-count gap, genuinely and scalably.
Per alpha compare three CEs that share the SAME divide-by-sum readout:
  A  ctx-count            : exact integer counts, add-eps(=0.01) smoothing      [reference floor]
  C  oracle freq-readout  : EXACT counts, but normalised-then-fixed-floor delta  [perfect attention]
  B  construction         : W* attention freq f, normalised-then-fixed-floor     [what we have]
  D  construction + N-smooth: W* freq, but smoothing scaled by recovered count N  [the fix]
gap_attention = B-C   (how lossy the soft attention counting is)
gap_smoothing = C-A   (fixed-floor vs count-based add-eps)
"""
import json, os, glob, math
import numpy as np, torch

dev = 'cpu'
if torch.cuda.is_available():
    free = []
    for i in range(torch.cuda.device_count()):
        try: free.append((torch.cuda.mem_get_info(i)[0], i))
        except Exception: pass
    if free: dev = f'cuda:{max(free)[1]}'
print('device:', dev, flush=True)

nb = json.load(open('analyse.ipynb'))
src = ''.join(nb['cells'][0]['source'])
defs = src[:src.index('# ════════════════════════════════════════════════════════════════════════════════\n# LAZY')]
defs = defs.replace("DEVICE           = 'cuda:1'", f"DEVICE           = '{dev}'")
g = {}; exec(defs, g)
load_data, _wlogits = g['load_data'], g['_wlogits']
DATA_DIR, TS, TE = g['DATA_DIR'], g['TOKEN_START'], g['TOKEN_END']

def exact_counts(blob):
    """Return integer count tensor C[B, W, Vs] = #(signal token i was the next-signal after the
    governing state) and the matched-total N[B, W] (= sum over signal of C)."""
    x = blob['x'].numpy(); V = int(blob['total_vocab']); Vs = int(blob['vocab_set1']); B = x.shape[0]
    tc = np.zeros((B, V, Vs)); pend = np.zeros((B, V)); last = np.full(B, -1, np.int64)
    def step(c):
        t = x[:, c]; r = t < Vs; ri = np.where(r)[0]
        if len(ri):
            rt = t[ri]; oh = (np.arange(Vs) == rt[:, None]).astype(float)
            np.add.at(tc, ri, pend[ri, :, None] * oh[:, None, :]); pend[ri, :] = 0.0; pend[ri, rt] = 1.0; last[ri] = rt
        ni = np.where(~r)[0]
        if len(ni): pend[ni, t[ni]] += 1.0
    for i in range(TS): step(i)
    C = np.zeros((B, TE - TS, Vs))
    for j, t in enumerate(range(TS, TE)):
        v = x[:, t]; q = np.where(v < Vs, v, np.maximum(last, 0))
        C[:, j, :] = tc[np.arange(B), q]; step(t)
    return torch.from_numpy(C)

def ce_from_psig(psig, pn_each, yw, V, Vs):
    pn = torch.full(psig.shape[:-1] + (V - Vs,), pn_each)
    p = torch.cat([psig, pn], -1).reshape(-1, V).clamp_min(1e-12)
    yy = yw.reshape(-1)
    return (-torch.log(p[torch.arange(len(yy)), yy])).mean().item()

DELTA = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2]
thrs = sorted(float(os.path.basename(f)[7:-3]) for f in glob.glob(os.path.join(DATA_DIR, 'test', 'data_x_*.pt')))
print(f"{'a':>4} {'A ctx':>8} {'B constr':>9} {'C oracle':>9} {'D +Nsm':>8} {'gapAttn':>8} {'gapSm':>7}", flush=True)
for thr in thrs:
    x, y, sl, V, blob = load_data('test', thr); Vs = int(blob['vocab_set1']); yw = y[:, TS:TE]
    C = exact_counts(blob)                                   # exact counts [B,W,Vs]
    Csum = C.sum(-1, keepdim=True)
    f = _wlogits(blob)[..., :Vs].clamp_min(0.0)              # W* attention signal freq
    pn = (1 - thr) / (V - Vs)
    # A: ctx-count (the reference floor)
    A = ce_from_psig(thr * (C + 0.01) / (Csum + Vs * 0.01), pn, yw, V, Vs)
    # B: construction (W* freq, fixed floor; best delta)
    fn = f / f.sum(-1, keepdim=True).clamp_min(1e-9)
    B = min(ce_from_psig(thr * (fn + d) / (1 + Vs * d), pn, yw, V, Vs) for d in DELTA)
    # C: oracle (exact counts as freq, fixed floor; best delta) -> perfect attention, same readout as B
    cf = C / Csum.clamp_min(1e-9)
    Cor = min(ce_from_psig(thr * (cf + d) / (1 + Vs * d), pn, yw, V, Vs) for d in DELTA)
    # D: construction + count-scaled smoothing  -> recover N = matched total, smooth in COUNT domain
    #    p_sig = alpha (f*N + eps)/(N + Vs eps).  N estimated from the EXACT matched total (proxy for a
    #    counter head); eps searched.
    N = Csum
    D = min(ce_from_psig(thr * (fn * N + e) / (N + Vs * e), pn, yw, V, Vs) for e in [3e-3, 1e-2, 3e-2, 1e-1])
    print(f"{thr:>4} {A:>8.4f} {B:>9.4f} {Cor:>9.4f} {D:>8.4f} {B-Cor:>8.4f} {Cor-A:>7.4f}", flush=True)
print('DONE', flush=True)
