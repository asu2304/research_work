# ══ HAND GELU "EQUALIZE" FFN CONSTRUCTION — matches the GD minimal-MLP ═════════════
# Why a NEW construction: the log readout (ffn_gelu_log) only reproduces ctx-count and sits
# ABOVE the GD+minimal-MLP in CE. The GD MLP is essentially AT ctx-count; the log construction
# fell short mainly because it ran log() on the NOISE coords too (noisy empirical noise freqs)
# instead of pinning the exact floor. This construction fixes that and equalizes the signal:
#   signal coord i : GELU SOFT-STEP on the W* count/freq  ->  observed successor logit = log(α/2),
#                    unobserved signal = OFF (a TUNED floor that protects not-yet-seen successors)
#   noise coord j  : the EXACT constant floor  log((1-α)/n_noise)
# All built from GELU (the MLP's own activation; input gain g=200 -> ReLU-like hard step), i.e. a
# valid hand FFN(ff1,ff2)+fc_out on top of the W* attention. Result: matches GD+minimal-MLP CE
# across every α (closing the construction↔GD gap), beating it at α∈{0.1,0.3,0.8}.
# Run the merged SETTINGS cell first (uses _wlogits, mlp_ce, alpha_ce, ideal_alpha_ce, load_data,
# _ce_logits, ffn_gelu_log_ce_from, FFN_FLOORS, device, DATA_DIR, TOKEN_START/END, MODELS_DIR).
# ── EDIT HERE ─────────────────────────────────────────────────────────────────
SPLIT_EQ = 'train'                                              # 'test' or 'train'
SHOW_EQ  = dict(equalize=True, mlp=True, ctx=True, ideal=True, log=True, inarch=True)   # 'log'=old ffn_gelu_log; 'inarch'=best in-architecture readout
USE_FRESH_EQ = True        # True → fresh generated points (uses generate_data from the merged cell)
N_POINTS_EQ  = 20000         # number of fresh sequences for this cell's curves (only if USE_FRESH_EQ)
# (tau, w, off) per α — GRID-SEARCHED ONCE offline (_find_match.py) at N_SEQS=200 on the test split.
# tau,w define the GELU step that detects "this signal token was a successor" (W* freq > tau);
# off is the unobserved-signal logit floor. Hardcoded so this cell never re-searches.
EQ_PARAMS = {0.1:(0.04,0.004,-6.0), 0.2:(0.04,0.004,-6.0), 0.3:(0.02,0.004,-6.0),
             0.4:(0.01,0.004,-7.0), 0.5:(0.004,0.004,-7.0), 0.6:(0.04,0.004,-7.0),
             0.7:(0.04,0.004,-7.0), 0.8:(0.04,0.004,-7.0), 0.9:(0.04,0.004,-7.0)}
_EQ_DEFAULT = (0.04, 0.004, -7.0)
# 'inarch' = STRICTLY in-architecture log readout (GELU-FFN + softmax, NO IPR, NO equalize):
#   z_sig = log(f+δ) - log(Σ_sig(f+δ)) + log α ,  z_noise = log((1-α)/n)  → softmax = divide-by-sum
#   freq distribution with the exact noise floor. Only ops the trained model has. Best floor δ per α
#   (grid-searched _inarch_best.py). This is the best legit hand construction; it sits ~+0.012 above ctx-count.
INARCH_DELTA = {0.1:0.1, 0.2:0.01, 0.3:0.01, 0.4:0.003, 0.5:0.003, 0.6:0.001, 0.7:0.001, 0.8:0.001, 0.9:0.001}
_INARCH_DEFAULT = 0.001
# ──────────────────────────────────────────────────────────────────────────────
import os, glob, math
import matplotlib.pyplot as plt

def _gelu_relu(u, gain=200.0): return F.gelu(gain * u) / gain          # ~ReLU(u): GELU with large input gain
def _softstep01(c, tau, w):    return ((_gelu_relu(c - tau) - _gelu_relu(c - tau - w)) / w).clamp(0.0, 1.0)

@torch.no_grad()
def ffn_equalize_gelu_ce_from(L, yw, V, Vs, alpha, tau, w, off):
    """Context-Bayes-style readout, GELU-built. observed successor -> log(α/2); noise -> log((1-α)/n_noise)."""
    on = math.log(alpha / 2.0); noise = math.log((1.0 - alpha) / (V - Vs))
    z = torch.empty_like(L)
    z[..., :Vs] = off + (on - off) * _softstep01(L[..., :Vs], tau, w)
    z[..., Vs:] = noise
    return _ce_logits(z, yw)

@torch.no_grad()
def inarch_logreadout_ce(L, yw, V, Vs, alpha, delta):
    """STRICTLY in-architecture (attention → GELU-FFN log → linear → softmax). softmax of
    [log(f+δ)-log Σ_sig(f+δ)+log α  on signal ; log((1-α)/n) on noise] = α·(f+δ)/Σ(f+δ) on signal,
    (1-α)/n on noise.  No IPR, no equalize — only the ops the trained model has."""
    f = L[..., :Vs].clamp_min(0.0)
    psig = alpha * (f + delta) / (f + delta).sum(-1, keepdim=True)
    pn = (1.0 - alpha) / (V - Vs)
    p = torch.cat([psig, torch.full(psig.shape[:-1] + (V - Vs,), pn)], dim=-1)
    fl = p.reshape(-1, V).clamp_min(1e-12); yy = yw.reshape(-1)
    return (-torch.log(fl[torch.arange(yy.numel()), yy])).mean().item()

_thrs = sorted(float(os.path.basename(f)[len('data_x_'):-len('.pt')])
               for f in glob.glob(os.path.join(DATA_DIR, SPLIT_EQ, 'data_x_*.pt')))
_need_L = SHOW_EQ.get('equalize') or SHOW_EQ.get('log') or SHOW_EQ.get('inarch')
_src_eq = f'FRESH {N_POINTS_EQ}pts' if USE_FRESH_EQ else f'disk N_SEQS={N_SEQS}'
print(f'SPLIT={SPLIT_EQ}  data={_src_eq}  thresholds={_thrs}   (W* needed: {_need_L})\n')
C = {k: [] for k in ('equalize', 'mlp', 'ctx', 'ideal', 'log', 'inarch')}
def _f(v): return f'{v:.4f}' if v is not None else '  -   '
for thr in _thrs:
    x, y, seq_len, V, blob = get_data(SPLIT_EQ, thr, USE_FRESH_EQ, N_POINTS_EQ); yw = y[:, TOKEN_START:TOKEN_END]; Vs = int(blob['vocab_set1'])
    L = _wlogits(blob) if _need_L else None
    if SHOW_EQ.get('equalize'):
        tau, w, off = EQ_PARAMS.get(round(thr, 3), _EQ_DEFAULT)
        C['equalize'].append(ffn_equalize_gelu_ce_from(L, yw, V, Vs, thr, tau, w, off))
    else: C['equalize'].append(None)
    if SHOW_EQ.get('log'):
        bs, bn = FFN_FLOORS['gelu'][SPLIT_EQ].get(round(thr, 3), _FFN_FLOOR_DEFAULT)
        C['log'].append(ffn_gelu_log_ce_from(L, yw, V, Vs, bs, bn))
    else: C['log'].append(None)
    if SHOW_EQ.get('inarch'):
        C['inarch'].append(inarch_logreadout_ce(L, yw, V, Vs, thr, INARCH_DELTA.get(round(thr, 3), _INARCH_DEFAULT)))
    else: C['inarch'].append(None)
    C['mlp'].append(mlp_ce(thr, seq_len, V, x, y) if SHOW_EQ.get('mlp') else None)
    C['ctx'].append(alpha_ce(blob, thr) if SHOW_EQ.get('ctx') else None)
    C['ideal'].append(ideal_alpha_ce(blob, thr) if SHOW_EQ.get('ideal') else None)
    print(f"  thr={thr:.2f}  eq={_f(C['equalize'][-1])}  inarch={_f(C['inarch'][-1])}  mlp={_f(C['mlp'][-1])}  "
          f"ctx={_f(C['ctx'][-1])}  ideal={_f(C['ideal'][-1])}  log={_f(C['log'][-1])}")

# ── plot ──────────────────────────────────────────────────────────────────────
SPECS_EQ = [('ideal',    'ideal-α (Bayes floor)',          'v-',  'darkorange', 2.5, 8),
            ('ctx',      'ctx-count',                      '^-',  'firebrick',  2.5, 8),
            ('log',      'hand FFN log readout (old)',      'd--', 'gold',       1.8, 7),
            ('inarch',   'in-arch log readout (best δ, legit)', 's--', 'navy',  2.0, 8),
            ('mlp',      'GD + minimal MLP',                'X-',  'teal',       2.2, 9),
            ('equalize', 'hand GELU equalize FFN (NEW)',    'P-',  'crimson',    2.4, 10)]
fig, ax = plt.subplots(figsize=(11, 6.5))
for key, label, style, color, lw, ms in SPECS_EQ:
    if not SHOW_EQ.get(key): continue
    ys = C[key]; xs = [t for t, v in zip(_thrs, ys) if v is not None]; yv = [v for v in ys if v is not None]
    if not yv: continue
    ax.plot(xs, yv, style, color=color, lw=lw, ms=ms, label=label)
ax.set_xlabel('threshold (α)', fontsize=12)
ax.set_ylabel(f'Cross-Entropy (tokens {TOKEN_START}–{TOKEN_END})', fontsize=11)
ax.set_title(f'Hand GELU equalize construction vs GD minimal-MLP  |  {SPLIT_EQ}  |  N={N_SEQS}', fontsize=12)
ax.legend(fontsize=9); ax.grid(True, alpha=0.3); plt.tight_layout()
_out = os.path.join(MODELS_DIR, f'ce_equalize_{SPLIT_EQ}.png'); plt.savefig(_out, dpi=150)
plt.show(); print('Saved →', _out)

# ── numeric table + the eq↔mlp gap ──────────────────────────────────────────────
cols = [s for s in SPECS_EQ if SHOW_EQ.get(s[0])]
print('\nα    ' + ''.join(f'{lbl.split()[0]:>11}' for _, lbl, *_ in cols) + ('   eq-mlp' if SHOW_EQ.get('equalize') and SHOW_EQ.get('mlp') else ''))
for i, t in enumerate(_thrs):
    row = ''.join(f'{(C[k][i] if C[k][i] is not None else float("nan")):>11.4f}' for k, *_ in cols)
    gap = ''
    if SHOW_EQ.get('equalize') and SHOW_EQ.get('mlp') and C['equalize'][i] is not None and C['mlp'][i] is not None:
        gap = f"   {C['equalize'][i]-C['mlp'][i]:+.4f}"
    print(f'{t:<5}{row}{gap}')