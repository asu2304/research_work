# Unified paper evaluation — reuses the exact definitions from ../constructions/
# (models_and_baselines.py / equalize_construction.py, themselves extracted verbatim
# from ICL/8/analyse.ipynb) so every number in the paper comes from the same harness
# the project has been using. READ-ONLY w.r.t. models.
#
# This IS the script that produced data/eval_test.json — the source for every CE
# and wrong-successor-rate table/figure in the main text (Table 3, Table 4, Figures
# 2/3/4/5/6). Table 1 (KL/cosine) and Table 5 (Appendix F) are produced separately
# by analysis/kl_divergence_calibration_table.py, analysis/gauge_invariant_
# structural_comparison.py, and analysis/unequal_successor_control_appendixF.py.
#
# Outputs: data/eval_test.json
#   curves[alpha][key]   CE (nats) on FRESH test data (5000 seqs, seed 13423), window [100:200)
#   postnoise[alpha][m]  wrong-successor rate at positions whose input token is NOISE
#   possig[alpha][m]     wrong-successor rate at positions whose input token is SIGNAL
#
# Before running: point MODELS_DIR_2CELL / g['MODELS_DIR_6'] / g['MODELS_DIR_7'] at
# wherever you trained the checkpoints (defaults below match this repo's original
# ICL/ layout; edit CANONICAL_ICL_ROOT if you cloned elsewhere).
#
# Run:  python eval_all.py
import os, sys, json, math

_TMP = os.path.join(os.path.expanduser('~'), '.cache', 'ashu_tmp')
os.makedirs(_TMP, exist_ok=True)
for _v in ('TMPDIR', 'TMP', 'TEMP', 'MPLCONFIGDIR'):
    os.environ[_v] = _TMP

HERE = os.path.dirname(os.path.abspath(__file__))
CONSTRUCTIONS = os.path.join(HERE, '..', 'constructions')
# Absolute paths to the ORIGINAL training-run output directories (folders 6/7/8 of
# the ICL/ tree). Edit this if your checkpoints live elsewhere; everything below
# derives from it so there is exactly one place to change.
CANONICAL_ICL_ROOT = '/home/hariguru/ashutosh/ICL'
DEVICE_OVERRIDE = 'cuda:2'          # near-idle per nvidia-smi at launch time
N_FRESH = 5000
SEED    = 13423                      # same fresh-eval seed as analyse.ipynb → cross-checkable

# ── exec the harness definitions (everything before its lazy-eval driver) ──
src0 = open(os.path.join(CONSTRUCTIONS, 'models_and_baselines.py')).read()
defs0 = src0[:src0.index('# LAZY EVALUATION')]
g = {}
exec(compile(defs0, 'models_and_baselines.py', 'exec'), g)

import torch, numpy as np
import torch.nn.functional as F

g['device'] = torch.device(DEVICE_OVERRIDE)
g['DATA_DIR'] = os.path.join(CANONICAL_ICL_ROOT,
    '6. understanding what gradient descent is firuring out', 'dataset')
g['MODELS_DIR_6'] = os.path.join(CANONICAL_ICL_ROOT,
    '6. understanding what gradient descent is firuring out', 'models')
g['MODELS_DIR_7'] = os.path.join(CANONICAL_ICL_ROOT,
    '7. unique_Wqk_Wok_test.ipynb', 'models')
g['MODELS_DIR_2CELL'] = os.path.join(CANONICAL_ICL_ROOT, '8. train with mlp', 'models')
g['USE_FRESH'] = True; g['N_POINTS'] = N_FRESH; g['FRESH_SEED'] = SEED
device = g['device']

# equalize + in-arch readouts (verbatim defs, lines up to the inarch fn)
src1 = open(os.path.join(CONSTRUCTIONS, 'equalize_construction.py')).read()
defs1 = src1[:src1.index('_thrs = sorted')]
exec(compile(defs1, 'equalize_construction.py', 'exec'), g)

EQ_PARAMS   = g['EQ_PARAMS'];   _EQ_DEFAULT   = g['_EQ_DEFAULT']
INARCH_DELTA = g['INARCH_DELTA']; _INARCH_DEFAULT = g['_INARCH_DEFAULT']
FFN_FLOORS  = g['FFN_FLOORS'];  _FFN_FLOOR_DEFAULT = g['_FFN_FLOOR_DEFAULT']
TOKEN_START, TOKEN_END = g['TOKEN_START'], g['TOKEN_END']

# ── batched W* logits (the notebook version runs the full blob at once) ────────
@torch.no_grad()
def wlogits_batched(blob, bs=500):
    V = int(blob['total_vocab']); T = int(blob['seq_len']); Vs = int(blob['vocab_set1'])
    m = g['_WSTAR_CACHE'].get((V, T, Vs))
    if m is None:
        m = g['PerfectInductionTransformer'](V, T, Vs).to(device).eval()
        g['_WSTAR_CACHE'][(V, T, Vs)] = m
    outs = []
    x = blob['x']
    for i in range(0, x.shape[0], bs):
        outs.append(m(x[i:i+bs].to(device))[:, TOKEN_START:TOKEN_END].cpu())
    return torch.cat(outs)

# ── context-Bayes posterior predictive (exact, under the 2-equiprobable prior) ──
# Counting mirrors alpha_ce's pending/last_ruled convention exactly; only the
# probability rule differs:
#   support size 2 -> alpha/2 on the two observed successors
#   support size 1 -> alpha/2 on it; alpha/2 * 1/(Vs-1) on each other signal token
#   support size 0 -> alpha/Vs on every signal token
#   noise          -> (1-alpha)/n_noise exactly
def context_bayes_and_ctx_argmax(blob, alpha):
    x_np = blob['x'].numpy(); y_np = blob['y'].numpy()
    V = int(blob['total_vocab']); V_sig = int(blob['vocab_set1'])
    n_noise = V - V_sig; B = x_np.shape[0]
    trans_counts = np.zeros((B, V, V_sig), dtype=np.float64)
    pending      = np.zeros((B, V), dtype=np.float64)
    last_ruled   = np.full(B, -1, dtype=np.int64)
    def _step(col):
        tok_col = x_np[:, col]; is_ruled = tok_col < V_sig
        ri = np.where(is_ruled)[0]
        if len(ri):
            rt = tok_col[ri]
            oh = (np.arange(V_sig) == rt[:, None]).astype(np.float64)
            np.add.at(trans_counts, ri, pending[ri, :, None] * oh[:, None, :])
            pending[ri, :] = 0.0; pending[ri, rt] = 1.0; last_ruled[ri] = rt
        ni = np.where(~is_ruled)[0]
        if len(ni): pending[ni, tok_col[ni]] += 1.0
    for i in range(TOKEN_START): _step(i)
    total_ce, n_tok = 0.0, 0
    ctx_argmax = np.zeros((B, TOKEN_END - TOKEN_START), dtype=np.int64)
    for t in range(TOKEN_START, TOKEN_END):
        v = x_np[:, t]; is_sig_q = v < V_sig
        query = np.where(is_sig_q, v, np.maximum(last_ruled, 0))
        cnt = trans_counts[np.arange(B), query]                       # [B, V_sig]
        ctx_argmax[:, t - TOKEN_START] = cnt.argmax(axis=1)           # most-seen successor
        sup = (cnt > 0)
        s2  = sup.sum(axis=1)                                         # support size (<=2 a.s.)
        p = np.zeros((B, V), dtype=np.float64)
        p[:, V_sig:] = (1.0 - alpha) / n_noise
        m2 = s2 >= 2
        if m2.any():
            p[np.ix_(m2, range(V_sig))] = sup[m2] * (alpha / 2.0)
        m1 = s2 == 1
        if m1.any():
            base = np.full((m1.sum(), V_sig), (alpha / 2.0) / (V_sig - 1))
            base[sup[m1]] = alpha / 2.0
            p[np.ix_(m1, range(V_sig))] = base
        m0 = s2 == 0
        if m0.any():
            p[np.ix_(m0, range(V_sig))] = alpha / V_sig
        total_ce += (-np.log(np.maximum(p[np.arange(B), y_np[:, t]], 1e-12))).sum(); n_tok += B
        _step(t)
    return total_ce / n_tok, ctx_argmax

# ── per-position state + successor sets (ground truth) ─────────────────────────
def state_and_successors(blob):
    x_np = blob['x'].numpy(); V_sig = int(blob['vocab_set1'])
    B, T = x_np.shape
    trans = blob['transitions']
    trans_arr = np.array([[trans[b][v] for v in range(V_sig)] for b in range(B)], dtype=np.int64)
    last_sig = np.zeros((B, T), dtype=np.int64); cur = np.zeros(B, dtype=np.int64)
    for t in range(T):
        is_sig = x_np[:, t] < V_sig
        cur = np.where(is_sig, x_np[:, t], cur)
        last_sig[:, t] = cur
    succ = trans_arr[np.arange(B)[:, None], last_sig]                 # [B, T, 2]
    return last_sig, succ

# ── model loading (read-only) using the harness classes/paths ──────────────────
def load_model(kind, thr, V, seq_len):
    _cfg = g['_cfg']; ATTN = g['ATTN_DIM']
    if kind == 'gd':
        mp = os.path.join(g['MODELS_DIR_6'], f'model_x_{thr}.pth')
        if not os.path.exists(mp): return None
        ad = int(_cfg(os.path.join(g['MODELS_DIR_6'], f'metrics_x_{thr}.json')).get('attn_dim', ATTN))
        m = g['DisentangledTransformer'](V, seq_len, ad, bias=True)
    elif kind == 'mlp':
        mp = os.path.join(g['MODELS_DIR_6'], f'model_mlp_x_{thr}.pth')
        if not os.path.exists(mp): return None
        c = _cfg(os.path.join(g['MODELS_DIR_6'], f'metrics_mlp_x_{thr}.json'))
        m = g['DisentangledTransformerMLP'](V, seq_len, int(c.get('attn_dim', ATTN)),
                                            int(c.get('mlp_hidden', 128)), bias=True,
                                            act=c.get('mlp_act', 'gelu'))
    else:
        return None
    m.load_state_dict(torch.load(mp, map_location=device, weights_only=True))
    return m.to(device).eval()

@torch.no_grad()
def wrong_successor_rates(model_logits_fn, x, succ, V_sig, bs=500):
    """rate of argmax-over-signal-coords NOT in the true successor set, split by
    whether the INPUT token at that position is noise (post-noise) or signal."""
    B, T = x.shape
    wrong_noise = tot_noise = wrong_sig = tot_sig = 0
    for i in range(0, B, bs):
        xb = x[i:i+bs].to(device)
        lg = model_logits_fn(xb)[:, TOKEN_START:TOKEN_END, :V_sig]
        pred = lg.argmax(-1).cpu().numpy()                            # [b, W]
        sc = succ[i:i+bs, TOKEN_START:TOKEN_END]                      # [b, W, 2]
        ok = (pred == sc[:, :, 0]) | (pred == sc[:, :, 1])
        is_noise = (x[i:i+bs, TOKEN_START:TOKEN_END].numpy() >= V_sig)
        wrong_noise += (~ok &  is_noise).sum(); tot_noise += is_noise.sum()
        wrong_sig   += (~ok & ~is_noise).sum(); tot_sig   += (~is_noise).sum()
    return float(wrong_noise) / max(tot_noise, 1), float(wrong_sig) / max(tot_sig, 1)

# ═══════════════════════════════ DRIVER ════════════════════════════════════════
thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
SIG_LIST, BN_LIST = g['SIG_LIST'], g['BN_LIST']
_par = {t: (float(SIG_LIST[i]), float(BN_LIST[i])) for i, t in enumerate(thresholds)}

out = {'meta': dict(n_fresh=N_FRESH, seed=SEED, window=[TOKEN_START, TOKEN_END],
                    device=DEVICE_OVERRIDE, note='fresh-generated eval data, same distribution & '
                    'generator as dataset; harness = ICL/8/analyse.ipynb definitions'),
       'curves': {}, 'postnoise': {}, 'possig': {}}

for thr in thresholds:
    x, y, seq_len, V, blob = g['generate_data']('test', thr, N_FRESH)
    Vs = int(blob['vocab_set1'])
    yw = y[:, TOKEN_START:TOKEN_END]
    L = wlogits_batched(blob)

    r = {}
    r['ideal'] = g['ideal_alpha_ce'](blob, thr)
    r['ctx']   = g['alpha_ce'](blob, thr)
    bayes_ce, ctx_am = context_bayes_and_ctx_argmax(blob, thr)
    r['bayes'] = bayes_ce
    for kind, fn in [('gd', g['gd_ce']), ('gd_comb', g['gd_comb_ce']), ('mlp', g['mlp_ce']),
                     ('twocell', g['twocell_ce']), ('twocell_comb', g['twocell_comb_ce'])]:
        r[kind] = fn(thr, seq_len, V, x, y)
    r['wstar']  = g['wstar_ce_from'](L, yw)
    sig, bn = _par[thr]
    r['wbetter'] = g['wbetter_ce_from'](L, yw, V, Vs, sig, bn)
    bs_, bn_ = FFN_FLOORS['gelu']['test'].get(round(thr, 3), _FFN_FLOOR_DEFAULT)
    r['log']    = g['ffn_gelu_log_ce_from'](L, yw, V, Vs, bs_, bn_)
    tau, w, off = EQ_PARAMS.get(round(thr, 3), _EQ_DEFAULT)
    r['eq']     = g['ffn_equalize_gelu_ce_from'](L, yw, V, Vs, thr, tau, w, off)
    r['inarch'] = g['inarch_logreadout_ce'](L, yw, V, Vs, thr,
                                            INARCH_DELTA.get(round(thr, 3), _INARCH_DEFAULT))
    out['curves'][str(thr)] = r

    # phase-transition diagnostic
    last_sig, succ = state_and_successors(blob)
    pn, ps = {}, {}
    for kind in ('gd', 'mlp'):
        m = load_model(kind, thr, V, seq_len)
        if m is None: continue
        pn[kind], ps[kind] = wrong_successor_rates(lambda xb: m(xb), x, succ, Vs)
        del m; torch.cuda.empty_cache()
    # ctx-count argmax (analytic predictor) — same metric, no model
    sc = succ[:, TOKEN_START:TOKEN_END]
    ok = (ctx_am == sc[:, :, 0]) | (ctx_am == sc[:, :, 1])
    is_noise = (x[:, TOKEN_START:TOKEN_END].numpy() >= Vs)
    pn['ctx'] = float((~ok &  is_noise).sum()) / max(is_noise.sum(), 1)
    ps['ctx'] = float((~ok & ~is_noise).sum()) / max((~is_noise).sum(), 1)
    # W*-better argmax
    pred_w = L[:, :, :Vs].argmax(-1).numpy()
    okw = (pred_w == sc[:, :, 0]) | (pred_w == sc[:, :, 1])
    pn['wstar'] = float((~okw &  is_noise).sum()) / max(is_noise.sum(), 1)
    ps['wstar'] = float((~okw & ~is_noise).sum()) / max((~is_noise).sum(), 1)
    out['postnoise'][str(thr)] = pn
    out['possig'][str(thr)]    = ps

    print(f"thr={thr:.1f}  " + "  ".join(f"{k}={v:.4f}" for k, v in r.items() if v is not None), flush=True)
    print(f"          postnoise: " + "  ".join(f"{k}={v:.4f}" for k, v in pn.items()), flush=True)

os.makedirs(os.path.join(HERE, 'data'), exist_ok=True)
with open(os.path.join(HERE, 'data', 'eval_test.json'), 'w') as f:
    json.dump(out, f, indent=1)
print('\nwrote data/eval_test.json')
