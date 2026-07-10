"""Assemble main.ipynb (3 cells) for the altered-distribution experiment. Run once."""
import json

CELL0 = r'''# ══ ALTERED DISTRIBUTION: settings + 3 predictors (ideal-alpha / ctx-count / in-arch log) ══
# Task is the SAME Markov-noise ICL task as folder 6, with ONE change: the two successors
# of each ruled state are UNEQUAL.  Each state draws p1 ~ U(0.25, 0.75), p2 = 1 - p1, so a
# signal emission picks successor s0 w.p. p1 and s1 w.p. p2 (both >= 0.25).  This removes the
# 0.5/0.5 symmetry that let GD "equalise" without reading the in-context count ratio.
#
# Three reference predictors (NO training):
#   ideal  : ideal-alpha   -- Bayes floor that KNOWS the true (s0,s1) AND true (p1,p2)
#                             p[s0]+=a*p1 ; p[s1]+=a*p2 ; noise=(1-a)/n_noise   (lowest possible CE)
#   ctx    : ctx-count      -- empirical in-context successor frequencies (a perfect count-reader);
#                             UNCHANGED from folder 6 -> it already estimates the unequal ratio
#   inarch : in-architecture log readout (IDENTICAL construction to folder 8): the W* induction
#                             circuit's frequency f read out as softmax[log(f+δ)-logΣ+logα ; log noise]
#                             = α·(f+δ)/Σ(f+δ).  The best LEGIT hand readout (only ops the model has).
#
# This cell only DEFINES + COMPUTES CEs; the next two cells plot (CE-vs-alpha, then per-position).

SPLIT = 'test'                                          # 'test' or 'train'
SHOW  = dict(ideal=True, ctx=True, inarch=True, trained=True)    # which curves to compute/plot
# 'trained' = the GD model (train_identical_arch.py): IDENTICAL architecture to the in-arch
# log readout (2 disentangled attn + residual GELU FFN head). Silently skipped until trained.

# in-arch log readout floors δ per α (STRICTLY in-architecture readout; see inarch_dist below).
# Carried over verbatim from folder 8 (grid-searched there via _inarch_best.py for the equal-prob
# task); re-tune for this altered distribution if you want the exact CE-optimal δ here.
INARCH_DELTA   = {0.1: 0.1, 0.2: 0.01, 0.3: 0.01, 0.4: 0.003, 0.5: 0.003,
                  0.6: 0.001, 0.7: 0.001, 0.8: 0.001, 0.9: 0.001}
_INARCH_DEFAULT = 0.001

# ── FRESH-DATA option (same altered distribution, generated on the fly) ────────
USE_FRESH  = True          # True -> fresh generated points; False -> load dataset/<split>/data_x_{thr}.pt
N_POINTS   = 5000          # fresh sequences per alpha (only if USE_FRESH)
FRESH_SEED = 13423         # base RNG seed (reproducible). None -> new draw each run.

DATA_DIR    = 'dataset'    # this folder's dataset/  (written by data_generation.py)
MODELS_DIR  = 'models'     # GD checkpoints (written by train_identical_arch.py)
TRAINED_TAG = 'idarch'     # model_{TAG}_x_{thr}.pth / metrics_{TAG}_x_{thr}.json
DEVICE      = 'cuda:2'
N_SEQS      = 3000         # cap when loading from disk
TOKEN_START = 100          # CE on tokens 100–200 (last half; seq_len=200)
TOKEN_END   = 200
P1_LOW, P1_HIGH = 0.25, 0.75    # successor-probability sampling range (must match data_generation.py)

import os, glob, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

device = torch.device(DEVICE if torch.cuda.is_available() else 'cpu')

# ── W* induction circuit (used by the in-arch log readout) ────────────────────
class _ConcatAttn(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.output_dim = output_dim
        self.W_q = nn.Linear(input_dim, output_dim, bias=False)
        self.W_k = nn.Linear(input_dim, output_dim, bias=False)
        self.W_v = nn.Linear(input_dim, output_dim, bias=False)
        self.W_o = nn.Linear(output_dim, output_dim, bias=False)
    def forward(self, x, mask=None):
        Q, K, V = self.W_q(x), self.W_k(x), self.W_v(x)
        s = torch.bmm(Q, K.transpose(1, 2)) / (self.output_dim ** 0.5)
        if mask is not None: s = s + mask
        return torch.cat([x, self.W_o(torch.bmm(F.softmax(s, dim=-1), V))], dim=-1)

class PerfectInductionTransformer(nn.Module):
    def __init__(self, total_vocab, seq_len, vocab_set1):
        super().__init__()
        self.total_vocab = total_vocab; self.seq_len = seq_len; self.vocab_set1 = vocab_set1
        self.initial_dim = total_vocab + seq_len; self.attn_dim = self.initial_dim
        self.layer1 = _ConcatAttn(self.initial_dim, self.attn_dim)
        self.layer2 = _ConcatAttn(self.initial_dim + self.attn_dim, self.attn_dim)
        self.fc_out = nn.Linear(self.initial_dim + 2 * self.attn_dim, total_vocab, bias=False)
        self._set_weights()
    def _set_weights(self):
        V = self.total_vocab; T = self.seq_len; D = self.attn_dim; V_r = self.vocab_set1
        self.layer1.W_q.weight.data = torch.eye(D) * (D ** 0.5)
        T1 = torch.zeros(D, D)
        for k in range(V_r, V): T1[k, :] = -1e9
        for tq in range(T):
            for tk in range(T):
                T1[V + tk, V + tq] = -1e9 if tq == tk else -20.0 * (tq - tk)
        self.layer1.W_k.weight.data = T1.t()
        Wv1 = torch.zeros(D, D); Wv1[:V, :V] = torch.eye(V)
        self.layer1.W_v.weight.data = Wv1
        self.layer1.W_o.weight.data = torch.eye(D)
        in2 = D + D
        Wq2 = torch.zeros(D, in2)
        Wq2[:V_r, :V_r]    = torch.eye(V_r) * (D ** 0.5) * 1000.0
        Wq2[:V_r, D:D+V_r] = torch.eye(V_r) * (D ** 0.5) * 50.0
        for v in range(V): Wq2[V_r, v] = (D ** 0.5) * 10.0
        self.layer2.W_q.weight.data = Wq2
        Wk2 = torch.zeros(D, in2); Wk2[:V, D:D+V] = torch.eye(V)
        for v in range(V_r, V): Wk2[V_r, v] = -50000.0
        self.layer2.W_k.weight.data = Wk2
        Wv2 = torch.zeros(D, in2); Wv2[:V, :V] = torch.eye(V)
        self.layer2.W_v.weight.data = Wv2
        self.layer2.W_o.weight.data = torch.eye(D)
        Wfc = torch.zeros(V, in2 + D); Wfc[:V, in2:in2+V] = torch.eye(V)
        self.fc_out.weight.data = Wfc
    def forward(self, x):
        B, T = x.shape
        tok = F.one_hot(x, num_classes=self.total_vocab).float()
        pos = F.one_hot(torch.arange(T, device=x.device),
                        num_classes=self.seq_len).float().unsqueeze(0).expand(B, T, -1)
        s = torch.cat([tok, pos], dim=-1)
        mask = torch.triu(torch.ones(T, T, device=x.device) * float('-inf'), diagonal=1)
        s = self.layer1(s, mask); s = self.layer2(s, mask)
        return self.fc_out(s)

_WSTAR_CACHE = {}

# ── GD model: DisentangledTransformerMLP (IDENTICAL to train_identical_arch.py / in-arch arch) ──
class ConcatAttentionBlock(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.output_dim = output_dim
        self.W_q = nn.Linear(input_dim, output_dim, bias=False)
        self.W_k = nn.Linear(input_dim, output_dim, bias=False)
        self.W_v = nn.Linear(input_dim, output_dim, bias=False)
        self.W_o = nn.Linear(output_dim, output_dim, bias=False)
    def forward(self, x, mask=None):
        Q, K, V = self.W_q(x), self.W_k(x), self.W_v(x)
        scores = torch.bmm(Q, K.transpose(1, 2)) / (self.output_dim ** 0.5)
        if mask is not None: scores = scores + mask
        attn_out = self.W_o(torch.bmm(F.softmax(scores, dim=-1), V))
        return torch.cat([x, attn_out], dim=-1), scores

class DisentangledTransformerMLP(nn.Module):
    def __init__(self, total_vocab, seq_len, attn_dim, mlp_hidden, bias=False, act='gelu'):
        super().__init__()
        self.total_vocab = total_vocab; self.seq_len = seq_len
        self.initial_dim = total_vocab + seq_len
        self.layer1 = ConcatAttentionBlock(self.initial_dim, attn_dim)
        self.layer2 = ConcatAttentionBlock(self.initial_dim + attn_dim, attn_dim)
        D = self.initial_dim + 2 * attn_dim
        self.ff1 = nn.Linear(D, mlp_hidden); self.ff2 = nn.Linear(mlp_hidden, D)
        self.act = nn.GELU() if act == 'gelu' else nn.ReLU()
        self.fc_out = nn.Linear(D, total_vocab, bias=bias)
    def forward(self, x):
        B, T = x.shape
        tok = F.one_hot(x, num_classes=self.total_vocab).float()
        pos = F.one_hot(torch.arange(T, device=x.device),
                        num_classes=self.seq_len).float().unsqueeze(0).expand(B, T, -1)
        s = torch.cat([tok, pos], dim=-1)
        mask = torch.triu(torch.ones(T, T, device=x.device) * float('-inf'), diagonal=1)
        s, _ = self.layer1(s, mask); s, _ = self.layer2(s, mask)
        s = s + self.ff2(self.act(self.ff1(s)))
        return self.fc_out(s)

_TRAINED_CACHE = {}
def _load_trained(thr, seq_len, V):
    """Load the GD checkpoint for this α (config read from its metrics json). None if missing."""
    key = (round(thr, 3), seq_len, V)
    if key in _TRAINED_CACHE: return _TRAINED_CACHE[key]
    mp = os.path.join(MODELS_DIR, f'model_{TRAINED_TAG}_x_{thr}.pth')
    cp = os.path.join(MODELS_DIR, f'metrics_{TRAINED_TAG}_x_{thr}.json')
    if not os.path.exists(mp):
        _TRAINED_CACHE[key] = None; return None
    cfg = json.load(open(cp)).get('config', {}) if os.path.exists(cp) else {}
    ad = int(cfg.get('attn_dim', V + seq_len)); mh = int(cfg.get('mlp_hidden', 128))
    act = cfg.get('mlp_act', 'gelu'); bias = bool(cfg.get('bias', True))
    m = DisentangledTransformerMLP(V, seq_len, ad, mh, bias=bias, act=act).to(device)
    m.load_state_dict(torch.load(mp, map_location=device, weights_only=True)); m.eval()
    _TRAINED_CACHE[key] = m; return m

@torch.no_grad()
def trained_dist(thr, seq_len, V, x):
    """GD model softmax distribution over the CE window [B,W,V]. None if no checkpoint."""
    m = _load_trained(thr, seq_len, V)
    if m is None: return None
    logits = m(x.to(device))[:, TOKEN_START:TOKEN_END]
    return F.softmax(logits, dim=-1)

def trained_ce(thr, seq_len, V, x, y):
    p = trained_dist(thr, seq_len, V, x)
    if p is None: return None
    yw = y[:, TOKEN_START:TOKEN_END].to(device)
    p = p.reshape(-1, V).clamp_min(1e-12); yy = yw.reshape(-1)
    return (-torch.log(p[torch.arange(len(yy), device=device), yy])).mean().item()

# ── Data: loader (disk) + fresh generator (same altered scheme as data_generation.py) ──
def load_data(split, thr, n_cap=N_SEQS):
    blob = torch.load(os.path.join(DATA_DIR, split, f'data_x_{thr}.pt'), weights_only=False)
    N = min(n_cap, blob['x'].shape[0])
    blob['x'] = blob['x'][:N]; blob['y'] = blob['y'][:N]
    for k in ('transitions', 'trans_probs'):
        if k in blob: blob[k] = blob[k][:N]
    return blob['x'], blob['y'], int(blob['seq_len']), int(blob['total_vocab']), blob

def generate_data(split, thr, n_points, seq_len=200, vocab_set1=10, total_vocab=20, seed='cfg'):
    """Fresh ALTERED-distribution sequences (unequal p1,p2 per state), vectorised."""
    base = FRESH_SEED if seed == 'cfg' else seed
    rng = (np.random.default_rng() if base is None
           else np.random.default_rng(base + (1 if split == 'train' else 2) + int(round(thr * 1000))))
    B, V_sig, V = int(n_points), vocab_set1, total_vocab
    order     = rng.random((B, V_sig, V_sig)).argsort(axis=2)
    trans_arr = order[:, :, :2].astype(np.int64)                  # [B, V_sig, 2]
    p1        = rng.uniform(P1_LOW, P1_HIGH, size=(B, V_sig))
    prob_arr  = np.stack([p1, 1.0 - p1], axis=2)                  # [B, V_sig, 2]
    seq  = np.empty((B, seq_len + 1), dtype=np.int64)
    last = np.full(B, -1, dtype=np.int64); bidx = np.arange(B)
    for t in range(seq_len + 1):
        is_sig = rng.random(B) < thr
        has    = last >= 0
        state  = np.where(has, last, 0)
        u      = rng.random(B)
        coin   = (u >= prob_arr[bidx, state, 0]).astype(np.int64)  # 0 w.p. p1 else 1
        succ   = trans_arr[bidx, state, coin]
        first  = rng.integers(0, V_sig, B)
        noise  = rng.integers(V_sig, V, B)
        tok    = np.where(is_sig, np.where(has, succ, first), noise)
        last   = np.where(is_sig, tok, last)
        seq[:, t] = tok
    x = torch.from_numpy(seq[:, :-1]).contiguous().long()
    y = torch.from_numpy(seq[:, 1:]).contiguous().long()
    trans = [{v: trans_arr[b, v] for v in range(V_sig)} for b in range(B)]
    probs = [{v: prob_arr[b, v]  for v in range(V_sig)} for b in range(B)]
    blob = {'x': x, 'y': y, 'seq_len': seq_len, 'total_vocab': V, 'vocab_set1': V_sig,
            'transitions': trans, 'trans_probs': probs, 'threshold': float(thr), 'split': split}
    return x, y, seq_len, V, blob

def get_data(split, thr, use_fresh=False, n_points=None):
    if use_fresh: return generate_data(split, thr, n_points or N_POINTS)
    return load_data(split, thr)

# Every predictor exposes a *_dist producer returning the per-position distribution over the
# CE window [TOKEN_START,TOKEN_END) as [B, W, V]; the *_ce wrappers just cross-entropy it.
# The diagnostic cell reuses the SAME producers, so plotted bars == the CE distributions.
def _ce_from_win(p_win, y_np):                                  # p_win [B,W,V] (numpy), y_np [B,T]
    yw = y_np[:, TOKEN_START:TOKEN_END]; B, W = yw.shape
    bb = np.arange(B)[:, None]; ww = np.arange(W)[None, :]
    return float((-np.log(np.maximum(p_win[bb, ww, yw], 1e-12))).mean())

# ── PREDICTOR 1: ideal-alpha (knows true table AND true unequal probs) ─────────
def ideal_alpha_dist(blob, alpha):
    x_np = blob['x'].numpy(); trans = blob['transitions']; probs = blob['trans_probs']
    V = int(blob['total_vocab']); V_sig = int(blob['vocab_set1']); n_noise = V - V_sig
    B = x_np.shape[0]; W = TOKEN_END - TOKEN_START
    trans_arr = np.array([[trans[b][v] for v in range(V_sig)] for b in range(B)], dtype=np.int64)
    prob_arr  = np.array([[probs[b][v] for v in range(V_sig)] for b in range(B)], dtype=np.float64)
    p_win = np.zeros((B, W, V), dtype=np.float64)
    last_sig = np.zeros(B, dtype=np.int64)
    for t in range(TOKEN_END):
        is_sig = x_np[:, t] < V_sig
        last_sig = np.where(is_sig, x_np[:, t], last_sig)
        if t < TOKEN_START: continue
        s0 = trans_arr[np.arange(B), last_sig, 0]; s1 = trans_arr[np.arange(B), last_sig, 1]
        q0 = prob_arr[np.arange(B), last_sig, 0];  q1 = prob_arr[np.arange(B), last_sig, 1]
        p = np.zeros((B, V), dtype=np.float64)
        p[:, V_sig:] = (1.0 - alpha) / n_noise
        p[np.arange(B), s0] += alpha * q0
        p[np.arange(B), s1] += alpha * q1
        p_win[:, t - TOKEN_START] = p
    return p_win

def ideal_alpha_ce(blob, alpha):
    return _ce_from_win(ideal_alpha_dist(blob, alpha), blob['y'].numpy())

# ── PREDICTOR 2: ctx-count (empirical successor frequencies; unchanged logic) ──
def ctx_count_dist(blob, alpha):
    x_np = blob['x'].numpy()
    V = int(blob['total_vocab']); V_sig = int(blob['vocab_set1'])
    n_noise = V - V_sig; B = x_np.shape[0]; W = TOKEN_END - TOKEN_START
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
    eps = 0.01; p_win = np.zeros((B, W, V), dtype=np.float64)
    for t in range(TOKEN_START, TOKEN_END):
        v = x_np[:, t]; is_sig_q = v < V_sig
        query = np.where(is_sig_q, v, np.maximum(last_ruled, 0))
        cnt = trans_counts[np.arange(B), query]; cnt_s = cnt + eps
        tot_s = cnt_s.sum(axis=1, keepdims=True)
        p = np.zeros((B, V), dtype=np.float64)
        p[:, :V_sig] = cnt_s / tot_s * alpha; p[:, V_sig:] = (1.0 - alpha) / n_noise
        p_win[:, t - TOKEN_START] = p
        _step(t)
    return p_win

def alpha_ce(blob, alpha):
    return _ce_from_win(ctx_count_dist(blob, alpha), blob['y'].numpy())

# ── PREDICTOR 3: in-architecture log readout (IDENTICAL construction to folder 8) ──
@torch.no_grad()
def _wstar_window(seq_len, V, Vs, x):
    """W* induction-circuit output over the CE window [B,W,V] = empirical successor freq f
    (signal cols), via the hand-built W* PerfectInductionTransformer (identical to folder 8)."""
    m = _WSTAR_CACHE.get((V, seq_len, Vs))
    if m is None:
        m = PerfectInductionTransformer(V, seq_len, Vs).to(device).eval(); _WSTAR_CACHE[(V, seq_len, Vs)] = m
    return m(x.to(device))[:, TOKEN_START:TOKEN_END]

@torch.no_grad()
def inarch_dist(thr, seq_len, V, Vs, x, delta=None):
    """STRICTLY in-architecture log readout (attention -> GELU-FFN log -> linear -> softmax):
       softmax[ log(f+δ) - log Σ_sig(f+δ) + log α  (signal) ;  log((1-α)/n)  (noise) ]
       = α·(f+δ)/Σ_sig(f+δ) on signal, (1-α)/n on noise.  NO IPR, NO equalize — only ops the
       trained model has.  This is the best LEGIT hand readout; sits a hair above ctx-count."""
    if delta is None: delta = INARCH_DELTA.get(round(thr, 3), _INARCH_DEFAULT)
    f = _wstar_window(seq_len, V, Vs, x)[..., :Vs].clamp_min(0.0)
    psig = thr * (f + delta) / (f + delta).sum(-1, keepdim=True)
    pn = (1.0 - thr) / (V - Vs)
    return torch.cat([psig, torch.full(psig.shape[:-1] + (V - Vs,), pn, device=device)], dim=-1)

def inarch_ce(thr, seq_len, V, Vs, x, y):
    p = inarch_dist(thr, seq_len, V, Vs, x)
    yw = y[:, TOKEN_START:TOKEN_END].to(device)
    p = p.reshape(-1, V).clamp_min(1e-12); yy = yw.reshape(-1)
    return (-torch.log(p[torch.arange(len(yy), device=device), yy])).mean().item()

# ════════════════════════════════════════════════════════════════════════════════
# COMPUTE the toggled CEs over all alphas (used by the plot in the next cell)
# ════════════════════════════════════════════════════════════════════════════════
if USE_FRESH:
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
else:
    thresholds = sorted(float(os.path.basename(f)[len('data_x_'):-len('.pt')])
                        for f in glob.glob(os.path.join(DATA_DIR, SPLIT, 'data_x_*.pt')))

_src = f'FRESH {N_POINTS}pts (seed={FRESH_SEED})' if USE_FRESH else f'disk N_SEQS={N_SEQS}'
print(f'SPLIT={SPLIT}  data={_src}  window=[{TOKEN_START}:{TOKEN_END}]  device={device}')
print(f'p1~U({P1_LOW},{P1_HIGH}) per state, p2=1-p1   thresholds={thresholds}')
print(f'computing: {[k for k,v in SHOW.items() if v]}\n')

results = {}
for thr in thresholds:
    x, y, seq_len, V, blob = get_data(SPLIT, thr, USE_FRESH, N_POINTS)
    Vs = int(blob['vocab_set1']); r = {}
    if SHOW.get('ideal'):       r['ideal']       = ideal_alpha_ce(blob, thr)
    if SHOW.get('ctx'):         r['ctx']         = alpha_ce(blob, thr)
    if SHOW.get('inarch'):      r['inarch']      = inarch_ce(thr, seq_len, V, Vs, x, y)
    if SHOW.get('trained'):
        _tc = trained_ce(thr, seq_len, V, x, y)
        if _tc is not None: r['trained'] = _tc          # silently skipped until the model is trained
    results[thr] = r
    print(f"  thr={thr:.2f}  " + "  ".join(f"{k}={v:.4f}" for k, v in r.items()))
print('\nDONE. Run the next cell for the CE-vs-alpha plot.')
'''

CELL1 = r'''# ══ PLOT 1: CE vs alpha  (ideal-alpha / ctx-count / in-arch log readout) ═══════
# Uses `results`, `thresholds`, `SHOW`, `SPLIT` from the cell above.
_STYLE = {
    'ideal':   ('ideal-alpha (true table+probs)', 'tab:green',  'o', '-'),
    'ctx':     ('ctx-count (empirical freq)',      'tab:blue',   's', '-'),
    'inarch':  ('in-arch log readout (best δ)',    'navy',       'D', '--'),
    'trained': ('GD trained (identical arch)',     'tab:purple', 'P', '-'),
}
plt.figure(figsize=(8, 5))
for key, (label, color, mk, ls) in _STYLE.items():
    if not SHOW.get(key): continue
    xs = [t for t in thresholds if results[t].get(key) is not None]
    ys = [results[t][key] for t in xs]
    if xs: plt.plot(xs, ys, ls, marker=mk, color=color, label=label, lw=2, ms=6)
plt.xlabel(r'$\alpha$ (signal probability)'); plt.ylabel('cross-entropy (nats)')
plt.title(f'Altered distribution (unequal $p_1,p_2$) — CE vs $\\alpha$  [{SPLIT}]')
plt.grid(alpha=0.3); plt.legend()
plt.tight_layout()
plt.savefig(f'ce_vs_alpha_{SPLIT}.png', dpi=130)
plt.show()
print(f'saved ce_vs_alpha_{SPLIT}.png')
'''

CELL2 = r'''# ══ PLOT 2: 50-token window + per-position distributions (EXACT CE distributions) ══
# Top strip: x[t]/y[t] over a 50-token window (blue=signal, gray=noise; orange=focus pos).
# Below: the next-token distribution at the focus position for each of the three curves.
# The bars are produced by the SAME functions used for the CE curve above
#   (ideal_alpha_dist / ctx_count_dist / inarch_dist / trained_dist), restricted to ONE
#   sequence — so what you see here is exactly what enters the cross-entropy up there.
#   The GD-trained bar appears only once train_identical_arch.py has saved a checkpoint.
# Green bars mark the two TRUE successors; the "Nx" label = how many times each was actually
# emitted FROM this state earlier in the context (the count evidence the model can use).
INSPECT_THRESH = 0.7          # which alpha to inspect
INSPECT_SPLIT  = SPLIT        # same data source as the curve
SEQ_IDX        = 0            # which sequence
RANGE_START    = 150          # 50-token window [150,200); must lie inside [TOKEN_START,TOKEN_END)
RANGE_END      = 200
FOCUS_POS      = 196          # position for the bar charts
# ──────────────────────────────────────────────────────────────────────────────
x, y, T, V, blob = get_data(INSPECT_SPLIT, INSPECT_THRESH, USE_FRESH, N_POINTS)
Vs = int(blob['vocab_set1']); n_noise = V - Vs
assert TOKEN_START <= RANGE_START < RANGE_END <= TOKEN_END, 'window must lie inside the CE window'
assert RANGE_START <= FOCUS_POS < RANGE_END, 'focus must lie inside the window'

# single-sequence blob -> identical code path as the curve's CE
one = {k: blob[k] for k in ('seq_len', 'total_vocab', 'vocab_set1', 'threshold')}
one['x'] = blob['x'][SEQ_IDX:SEQ_IDX+1]; one['y'] = blob['y'][SEQ_IDX:SEQ_IDX+1]
one['transitions'] = [blob['transitions'][SEQ_IDX]]; one['trans_probs'] = [blob['trans_probs'][SEQ_IDX]]
x_np = one['x'][0].numpy(); y_np = one['y'][0].numpy()
ta = blob['transitions'][SEQ_IDX]; pr = blob['trans_probs'][SEQ_IDX]

# EXACT CE distributions for this one sequence (window [TOKEN_START,TOKEN_END))
ideal_w = ideal_alpha_dist(one, INSPECT_THRESH)[0]                          # [W,V]
ctx_w   = ctx_count_dist(one, INSPECT_THRESH)[0]
ina_w   = inarch_dist(INSPECT_THRESH, T, V, Vs, one['x'])[0].cpu().numpy()
_tr     = trained_dist(INSPECT_THRESH, T, V, one['x'])                     # None until trained
tr_w    = _tr[0].cpu().numpy() if _tr is not None else None
def _at(arr, t): return arr[t - TOKEN_START]                               # window index helper

# last ruled token (state) at each position
lr_arr = np.full(T, -1); cur = -1
for t in range(T):
    if x_np[t] < Vs: cur = int(x_np[t])
    lr_arr[t] = cur

fp = FOCUS_POS; state_fp = int(lr_arr[fp])
exp = (int(ta[state_fp][0]), int(ta[state_fp][1])) if state_fp >= 0 else None
# context counts: how many times each token was emitted FROM state_fp before fp
ctx_cnt = np.zeros(V, int)
for _t in range(fp):
    if lr_arr[_t] == state_fp: ctx_cnt[int(y_np[_t])] += 1

PRED = [('ideal-alpha (oracle)',         ideal_w, 'tab:green'),
        ('ctx-count (empirical freq)',   ctx_w,   'tab:blue'),
        ('in-arch log readout (best δ)', ina_w,   'navy')]
if tr_w is not None:
    PRED.append(('GD trained (identical arch)', tr_w, 'tab:purple'))

# ── printout ──
print(f"seq {SEQ_IDX} | {INSPECT_SPLIT} | alpha={INSPECT_THRESH} | window [{RANGE_START},{RANGE_END}) | focus {fp}")
if exp:
    q0, q1 = float(pr[state_fp][0]), float(pr[state_fp][1])
    print(f"state {state_fp} -> successors {exp[0]} (p1={q0:.2f}), {exp[1]} (p2={q1:.2f}); target y[{fp}]={int(y_np[fp])}")
    print(f"context evidence from state {state_fp}: {exp[0]}={ctx_cnt[exp[0]]}x, {exp[1]}={ctx_cnt[exp[1]]}x, "
          f"noise={int(ctx_cnt[Vs:].sum())}x")
    _tr_py = f"  trained={_at(tr_w,fp)[int(y_np[fp])]:.3f}" if tr_w is not None else ""
    print(f"p(y) at focus:  ideal={_at(ideal_w,fp)[int(y_np[fp])]:.3f}  "
          f"ctx={_at(ctx_w,fp)[int(y_np[fp])]:.3f}  inarch={_at(ina_w,fp)[int(y_np[fp])]:.3f}{_tr_py}")

# ── figure: token strip + 3 distribution bars ──
BLUE='#1f6feb'; ORANGE='#e07a18'; GRAY='#bfc4cb'; RED='#d62828'; GREEN='#2a9d4a'
rs, re_ = RANGE_START, RANGE_END; cols = re_ - rs; n = len(PRED)
fig = plt.figure(figsize=(max(11, cols*0.24), 3.0 + 2.3*n))
gs = fig.add_gridspec(1+n, 1, height_ratios=[1.2]+[2.0]*n, hspace=0.7)
ax = fig.add_subplot(gs[0])
for j, t in enumerate(range(rs, re_)):
    ax.text(j, 1, str(int(x_np[t])), ha='center', va='center', fontsize=7, color=(BLUE if x_np[t] < Vs else GRAY))
    ax.text(j, 0, str(int(y_np[t])), ha='center', va='center', fontsize=7, color=(BLUE if y_np[t] < Vs else GRAY))
ax.axvspan(fp-rs-0.5, fp-rs+0.5, color=ORANGE, alpha=0.18, zorder=0)
ax.set_yticks([0, 1]); ax.set_yticklabels(['y[t]', 'x[t]'], fontsize=8)
ax.set_xticks(range(0, cols, max(1, cols//12))); ax.set_xticklabels([str(rs+i) for i in range(0, cols, max(1, cols//12))], fontsize=7)
ax.set_xlim(-0.5, cols-0.5); ax.set_ylim(-0.6, 1.6)
ax.set_title(f"seq {SEQ_IDX}: tokens [{rs},{re_})  (blue=signal, gray=noise; orange=focus {fp})", fontsize=10)
for sp in ['top', 'right']: ax.spines[sp].set_visible(False)

chain = f"state {state_fp} -> {{{exp[0]},{exp[1]}}}" if exp else "(no state yet)"
for r, (lbl, arr, col) in enumerate(PRED):
    axb = fig.add_subplot(gs[1+r]); p1 = _at(arr, fp)
    bc = [BLUE if k < Vs else GRAY for k in range(V)]
    if exp:
        for k in exp: bc[k] = GREEN
    axb.bar(range(V), p1, color=bc, edgecolor='white', linewidth=0.6)
    if exp:
        for k in exp: axb.text(k, p1[k], f"{int(ctx_cnt[k])}x", ha='center', va='bottom', fontsize=8, color=GREEN, fontweight='bold')
    axb.axvline(int(y_np[fp]), color=ORANGE, lw=1.2, ls='--', label=f'target y={int(y_np[fp])}')
    axb.axvline(int(p1.argmax()), color=RED, lw=1.0, ls=':', label=f'argmax={int(p1.argmax())}')
    axb.set_xticks(range(V)); axb.set_ylabel('P(tok|ctx)', fontsize=8)
    axb.set_xlabel('token id (0-9 signal, 10-19 noise)', fontsize=8)
    axb.legend(fontsize=7, loc='upper right'); axb.grid(axis='y', alpha=0.3)
    axb.set_title(f"{lbl}   t={fp}   {chain}   p(y)={p1[int(y_np[fp])]:.3f}", fontsize=9, color=col)
    for sp in ['top', 'right']: axb.spines[sp].set_visible(False)
plt.tight_layout()
plt.savefig(f'dist_pos_{INSPECT_SPLIT}.png', dpi=150); plt.show()
print(f'saved dist_pos_{INSPECT_SPLIT}.png')
'''

nb = {
    "cells": [
        {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": CELL0.splitlines(keepends=True)},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": CELL1.splitlines(keepends=True)},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": CELL2.splitlines(keepends=True)},
    ],
    "metadata": {"language_info": {"name": "python"}},
    "nbformat": 4, "nbformat_minor": 5,
}
json.dump(nb, open('main.ipynb', 'w'), indent=1)
print('wrote main.ipynb with', len(nb['cells']), 'cells')
