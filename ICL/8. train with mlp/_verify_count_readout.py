"""Non-cheating, SCALABLE construction: W* attention counts the successors, the readout
NORMALISES those counts (divide-by-sum) + pins the exact noise floor. No equalisation -> it
keeps the count magnitudes, so it reproduces ctx-count for ANY transition distribution.
Verify: (1) its CE overlaps ctx-count across alpha; (2) its per-position predictive distribution
overlaps ctx-count point-for-point. Saves two figures into this folder."""
import json, os, glob, math
import numpy as np, torch, torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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

EPS = 1e-3   # smoothing on the normalised counts

# ── the construction's per-position distribution: divide-by-sum on the W* successor freq ──
@torch.no_grad()
def count_readout_dist(L, V, Vs, alpha):
    f = L[..., :Vs].clamp_min(0.0)                       # W* empirical successor freq (signal block)
    ps = alpha * (f + EPS) / (f.sum(-1, keepdim=True) + Vs * EPS)   # normalise, scale by alpha
    pn = torch.full(L.shape[:-1] + (V - Vs,), (1.0 - alpha) / (V - Vs))
    return torch.cat([ps, pn], dim=-1)                   # [N, W, V], sums to 1

# ── ctx-count's per-position distribution (exact integer counting, the reference) ──
def ctx_count_dist(blob, alpha):
    x = blob['x'].numpy(); V = int(blob['total_vocab']); Vs = int(blob['vocab_set1']); B = x.shape[0]
    tc = np.zeros((B, V, Vs)); pend = np.zeros((B, V)); last = np.full(B, -1, np.int64)
    def step(c):
        t = x[:, c]; r = t < Vs; ri = np.where(r)[0]
        if len(ri):
            rt = t[ri]; oh = (np.arange(Vs) == rt[:, None]).astype(float)
            np.add.at(tc, ri, pend[ri, :, None] * oh[:, None, :])
            pend[ri, :] = 0.0; pend[ri, rt] = 1.0; last[ri] = rt
        ni = np.where(~r)[0]
        if len(ni): pend[ni, t[ni]] += 1.0
    for i in range(TS): step(i)
    P = np.zeros((B, TE - TS, V))
    for j, t in enumerate(range(TS, TE)):
        v = x[:, t]; q = np.where(v < Vs, v, np.maximum(last, 0))
        cnt = tc[np.arange(B), q] + 0.01; tot = cnt.sum(1, keepdims=True)
        P[:, j, :Vs] = cnt / tot * alpha; P[:, j, Vs:] = (1 - alpha) / (V - Vs)
        step(t)
    return torch.from_numpy(P)

def ce(p, y):
    p = p.reshape(-1, p.shape[-1]).clamp_min(1e-12); yy = y.reshape(-1)
    return (-torch.log(p[torch.arange(len(yy)), yy])).mean().item()

thrs = sorted(float(os.path.basename(f)[7:-3]) for f in glob.glob(os.path.join(DATA_DIR, 'test', 'data_x_*.pt')))
rows = []; scat = {}
for thr in thrs:
    x, y, sl, V, blob = load_data('test', thr); Vs = int(blob['vocab_set1'])
    yw = y[:, TS:TE]
    L = _wlogits(blob)
    p_con = count_readout_dist(L, V, Vs, thr)            # construction
    p_ctx = ctx_count_dist(blob, thr)                    # reference
    ce_con, ce_ctx = ce(p_con, yw), ce(p_ctx, yw)
    rows.append((thr, ce_con, ce_ctx));
    if thr in (0.3, 0.8): scat[thr] = (p_con.reshape(-1).numpy(), p_ctx.reshape(-1).numpy())
    print(f"alpha={thr}  count-readout CE={ce_con:.4f}   ctx-count CE={ce_ctx:.4f}   diff={ce_con-ce_ctx:+.4f}", flush=True)

# ── figure 1: CE vs alpha (overlap) ──
A = [r[0] for r in rows]
plt.figure(figsize=(7, 4.5))
plt.plot(A, [r[2] for r in rows], 'o-', color='firebrick', lw=2.5, ms=8, label='ctx-count (reference)')
plt.plot(A, [r[1] for r in rows], 'x--', color='black', lw=2.0, ms=9, label='count-readout construction')
plt.xlabel('threshold (alpha)'); plt.ylabel('Cross-Entropy (tokens 100-200)')
plt.title('Count-readout construction overlaps ctx-count (no equalisation)')
plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
plt.savefig('count_readout_vs_ctx_CE.png', dpi=150); plt.close()

# ── figure 2: per-position distribution overlap (scatter on y=x) ──
fig, axes = plt.subplots(1, len(scat), figsize=(5.2 * len(scat), 4.8))
if len(scat) == 1: axes = [axes]
for ax, thr in zip(axes, sorted(scat)):
    a, b = scat[thr]
    idx = np.random.default_rng(0).choice(len(a), size=min(40000, len(a)), replace=False)
    ax.scatter(b[idx], a[idx], s=3, alpha=0.25, color='steelblue')
    ax.plot([0, 1], [0, 1], 'r--', lw=1.2)
    r = np.corrcoef(a, b)[0, 1]
    ax.set_xlabel('ctx-count  P(token)'); ax.set_ylabel('construction  P(token)')
    ax.set_title(f'alpha={thr}   per-(pos,token) prob   r={r:.4f}'); ax.set_aspect('equal')
plt.tight_layout(); plt.savefig('count_readout_dist_overlap.png', dpi=150); plt.close()
print('WROTE count_readout_vs_ctx_CE.png  and  count_readout_dist_overlap.png', flush=True)
