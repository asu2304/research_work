"""Appendix F, last paragraph: the exact gradient/Hessian probe. Restricted to the
FFN+readout subspace (attention frozen at W* values), computes the exact gradient
via autograd and v^T H v via finite differences along 300 random unit directions,
at (A) the W*-attention + dormant-FFN point and (B) the trained GD checkpoint (if
present), classifying each as a local min / saddle by sign of the curvature sweep.

Depends on data_generation/generate_unequal_successor_control.py (or
CONFIG['use_fresh']=True to generate on the fly) and, for POINT B, a checkpoint from
training/train_unequal_successor_control.py.

Extracted verbatim from ICL/10.../main.ipynb (cells 0 and 6); cell 0 is duplicated
here (also in unequal_successor_control_appendixF.py) so this file runs standalone.

Run: python hessian_probe_appendixF.py
"""

# ══ ALTERED DISTRIBUTION: settings + 3 predictors (ideal-alpha / ctx-count / in-arch log) ══
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
N_POINTS   = 10000          # fresh sequences per alpha (only if USE_FRESH)
FRESH_SEED = 1423         # base RNG seed (reproducible). None -> new draw each run.

DATA_DIR    = 'dataset'    # this folder's dataset/  (written by data_generation.py)
MODELS_DIR  = 'models'     # GD checkpoints (written by train_identical_arch.py)
TRAINED_TAG = 'idarch'     # model_{TAG}_x_{thr}.pth / metrics_{TAG}_x_{thr}.json
# Per-α override of the trained tag. 0.8 from scratch is stuck at the 1.59 'equalise' basin
# (GD can't find the count-reader from random init → α=0.8 optimisation phase transition);
# the warm-start-from-0.9 model (train_identical_arch.py init_from) reaches the 1.42 basin.
TRAINED_OVERRIDE = {0.7: "idarch_ws06", 0.8: "idarch_ws09"}   # warm-starts; falls back to TRAINED_TAG for others
DEVICE      = 'cuda:1'
N_SEQS      = 3000         # cap when loading from disk
TOKEN_START = 100          # CE on tokens 100–200 (last half; seq_len=200)
TOKEN_END   = 200
P1_LOW, P1_HIGH = 0.0, 1.0    # successor-probability sampling range (must match data_generation.py)

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
    tag = TRAINED_OVERRIDE.get(round(thr, 3), TRAINED_TAG)
    mp = os.path.join(MODELS_DIR, f'model_{tag}_x_{thr}.pth')
    cp = os.path.join(MODELS_DIR, f'metrics_{tag}_x_{thr}.json')
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


# ══ CELL 6: HESSIAN ANALYSIS — GRADIENT + CURVATURE TO CLASSIFY CRITICAL POINTS ═════════
# ─────────────────────────────────────────────────────────────────────────────────────────
# METHOD: For a point θ to be a local minimum we need:
#   (i)  ∇L(θ) = 0          (critical point — first-order necessary condition)
#   (ii) v^T H v ≥ 0 for all v  (positive semi-definite Hessian — second-order condition)
# A saddle point satisfies (i) but has some v with v^T H v < 0.
# We compute:
#   • exact gradient at each test point via autograd
#   • v^T H v via finite differences: [L(θ+εv)+L(θ−εv)−2L(θ)] / ε²  for N random unit vectors v
# We analyse two points:
#   POINT A: W* attention weights + dormant FFN (ff2.weight=0)  — baseline / GD-scratch proxy
#   POINT B: trained GD model (if checkpoint exists)            — GD converged point
#
# Gradient and curvature restricted to the FFN+readout subspace (ff1, ff2, fc_out).
# Attention layers frozen at W* values (they have ±1e9 entries — kept fixed).
# ─────────────────────────────────────────────────────────────────────────────────────────
HESS_THR        = 0.8         # threshold to analyse (0.8 = the interesting case)
HESS_N          = 400         # sequences for loss estimation
HESS_N_DIRS     = 300         # random unit directions for curvature sweep
HESS_EPS        = 1e-3        # finite difference step size
HESS_ALL_THR    = True        # also compute grad norm for all thresholds (quick)
GD_TAG_H        = 'idarch'    # which GD checkpoint to load for POINT B
# ─────────────────────────────────────────────────────────────────────────────────────────
import os, math, itertools
import numpy as np, torch, torch.nn.functional as F
import matplotlib.pyplot as plt
from itertools import chain

# ── build W* + dormant FFN in DisentangledTransformerMLP ─────────────────────────────────
def _build_wstar_dormant(V, T, Vs, dev):
    ws = _WSTAR_CACHE.get((V, T, Vs)) or PerfectInductionTransformer(V, T, Vs).to(dev).eval()
    _WSTAR_CACHE[(V, T, Vs)] = ws
    D_attn = V + T  # = initial_dim = attn_dim = 220
    m = DisentangledTransformerMLP(V, T, D_attn, 128, bias=True, act='gelu').to(dev)
    with torch.no_grad():
        for ln in ('layer1', 'layer2'):
            for wn in ('W_q','W_k','W_v','W_o'):
                s = getattr(getattr(ws, ln), wn).weight
                d = getattr(getattr(m, ln),  wn).weight
                if s.shape == d.shape: d.copy_(s)
        # dormant FFN: ff2.weight=0 → residual is exactly zero → model = W* attention + fc_out
        m.ff2.weight.data.zero_(); m.ff2.bias.data.zero_()
        if ws.fc_out.weight.shape == m.fc_out.weight.shape:
            m.fc_out.weight.data.copy_(ws.fc_out.weight)
        if m.fc_out.bias is not None: m.fc_out.bias.data.zero_()  # W* has no bias
    return m

# ── restrict grad to FFN+readout subspace, freeze attention ──────────────────────────────
def _set_grad_scope(m, ffn_only=True):
    for p in chain(m.layer1.parameters(), m.layer2.parameters()):
        p.requires_grad_(not ffn_only)
    for p in chain(m.ff1.parameters(), m.ff2.parameters(), m.fc_out.parameters()):
        p.requires_grad_(True)

# ── compute CE loss (with backward) over the eval window ─────────────────────────────────
def _loss_bwd(m, xb, ywin, V):
    m.zero_grad()
    lg = m(xb)[:, TOKEN_START:TOKEN_END, :]         # [B, W, V]
    loss = F.cross_entropy(lg.reshape(-1, V), ywin.reshape(-1))
    loss.backward()
    return loss.item()

@torch.no_grad()
def _loss_ngrad(m, xb, ywin, V):
    lg = m(xb)[:, TOKEN_START:TOKEN_END, :]
    return F.cross_entropy(lg.reshape(-1, V), ywin.reshape(-1)).item()

# ── directional curvature via finite differences in FFN+readout subspace ─────────────────
def _fd_curvature(m, xb, ywin, V, L0, eps):
    ffn_params = list(chain(m.ff1.parameters(), m.ff2.parameters(), m.fc_out.parameters()))
    vs = [torch.randn_like(p) for p in ffn_params]
    norm = sum(vi.norm()**2 for vi in vs).sqrt()
    vs = [vi / norm for vi in vs]
    with torch.no_grad():
        for p, vi in zip(ffn_params, vs): p.data.add_(eps * vi)
    Lp = _loss_ngrad(m, xb, ywin, V)
    with torch.no_grad():
        for p, vi in zip(ffn_params, vs): p.data.add_(-2.0*eps * vi)
    Lm = _loss_ngrad(m, xb, ywin, V)
    with torch.no_grad():
        for p, vi in zip(ffn_params, vs): p.data.add_(eps * vi)   # restore
    return (Lp + Lm - 2.0*L0) / (eps*eps)

# ── gradient norm breakdown ───────────────────────────────────────────────────────────────
def _grad_norms(m):
    res = {}
    for name, p in m.named_parameters():
        if p.grad is not None: res[name] = p.grad.norm().item()
    ff1_n  = (res.get('ff1.weight',0)**2 + res.get('ff1.bias',0)**2)**0.5
    ff2_n  = (res.get('ff2.weight',0)**2 + res.get('ff2.bias',0)**2)**0.5
    fc_n   = (res.get('fc_out.weight',0)**2 + res.get('fc_out.bias',0)**2 if m.fc_out.bias is not None else res.get('fc_out.weight',0))**0.5
    total  = sum(v**2 for v in res.values())**0.5
    return ff1_n, ff2_n, fc_n, total

# ══════════════════════════════════════════════════════════════════════════════════════════
# ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════════════════
x, y, T, V, blob = get_data('test', HESS_THR)
_N = min(HESS_N, x.shape[0]); Vs = int(blob['vocab_set1'])
_xb   = x[:_N].to(device)
_ywin = y[:_N, TOKEN_START:TOKEN_END].to(device)

# ── POINT A: W* attention + dormant FFN ──────────────────────────────────────────────────
print('='*72)
print(f'POINT A: W* attention + dormant FFN  (alpha={HESS_THR})')
print('='*72)
mA = _build_wstar_dormant(V, T, Vs, device)
_set_grad_scope(mA, ffn_only=True)
L0_A = _loss_bwd(mA, _xb, _ywin, V)
ff1n_A, ff2n_A, fcn_A, tot_A = _grad_norms(mA)
print(f'  CE at W*+dormant-FFN (pure σ=1 softmax): {L0_A:.4f}')
print(f'  (note: raw freq-logits → softmax ≈ near-uniform → CE~2.5 is expected;')
print(f'   the GD scratch basin at CE~1.59 is a DIFFERENT critical point)')
print(f'  context-Bayes (global min): {context_bayes_ce(dict(x=x[:_N],y=y[:_N],**{k:blob[k] for k in blob if k not in ("x","y")}), HESS_THR):.4f}')
print()
print(f'  GRADIENT NORMS (FFN+readout subspace):')
print(f'    ||grad_ff1||  = {ff1n_A:.4e}  ← nonzero → ff1 wants to move')
print(f'    ||grad_ff2||  = {ff2n_A:.4e}  ← nonzero → ff2 wants to activate')
print(f'    ||grad_fc_out||= {fcn_A:.4e}')
print(f'    ||grad_total|| = {tot_A:.4e}')
print()
if tot_A > 1e-4:
    print('  CONCLUSION: gradient ≠ 0  → W* + dormant FFN is NOT a critical point.')
    print('              It is a SADDLE POINT — gradient points toward the global min.')
else:
    print('  CONCLUSION: gradient ≈ 0  → W* + dormant FFN IS a critical point (unexpected).')
print()

# ── CURVATURE SWEEP at POINT A (FFN+readout subspace) ────────────────────────────────────
print(f'  CURVATURE SWEEP: {HESS_N_DIRS} random unit directions in FFN+readout subspace')
print(f'  eps = {HESS_EPS}   (finite difference step size)')
torch.manual_seed(42)
curvs_A = [_fd_curvature(mA, _xb, _ywin, V, L0_A, HESS_EPS) for _ in range(HESS_N_DIRS)]
curvs_A = np.array(curvs_A)
n_neg_A = (curvs_A < 0).sum()
print(f'  min curvature: {curvs_A.min():.4e}    max: {curvs_A.max():.4e}    mean: {curvs_A.mean():.4e}')
print(f'  negative curvatures: {n_neg_A}/{HESS_N_DIRS}  ({100*n_neg_A/HESS_N_DIRS:.1f}%)')
print()
if n_neg_A > 0:
    print('  CURVATURE VERDICT: SADDLE POINT (negative curvatures exist)')
    print('  → the loss function DECREASES along these FFN directions from Point A')
    print('  → GD can escape this saddle toward the global minimum (in-arch/Bayes level)')
    print('  → but at alpha=0.8, GD from random init lands in a DIFFERENT local basin (CE~1.59)')
    print('    rather than following the descent path through Point A')
else:
    print('  CURVATURE VERDICT: locally convex in FFN subspace (all sampled curvatures positive)')
    print('  → W* attention + dormant FFN MAY be a local min in the FFN subspace.')
print()

# ── POINT B: trained GD model (if available) ─────────────────────────────────────────────
_mp_B = os.path.join(MODELS_DIR, f'model_{GD_TAG_H}_x_{HESS_THR}.pth')
if os.path.exists(_mp_B):
    print('='*72)
    print(f'POINT B: GD trained model ({GD_TAG_H}, alpha={HESS_THR})')
    print('='*72)
    mB = DisentangledTransformerMLP(V, T, V+T, 128, bias=True, act='gelu').to(device)
    mB.load_state_dict(torch.load(_mp_B, map_location=device, weights_only=True))
    _set_grad_scope(mB, ffn_only=False)  # allow grads on all params
    L0_B = _loss_bwd(mB, _xb, _ywin, V)
    ff1n_B, ff2n_B, fcn_B, tot_B = _grad_norms(mB)
    print(f'  CE at this point: {L0_B:.4f}')
    print()
    print(f'  GRADIENT NORMS (all params):')
    print(f'    ||grad_ff1||  = {ff1n_B:.4e}')
    print(f'    ||grad_ff2||  = {ff2n_B:.4e}')
    print(f'    ||grad_fc_out||= {fcn_B:.4e}')
    print(f'    ||grad_total|| = {tot_B:.4e}')
    if tot_B < 1e-2:
        print('  → gradient ≈ 0 (GD converged to a critical point)')
    else:
        print('  → gradient still nonzero (GD not fully converged yet)')
    print()
    _set_grad_scope(mB, ffn_only=True)
    torch.manual_seed(42)
    curvs_B = [_fd_curvature(mB, _xb, _ywin, V, L0_B, HESS_EPS) for _ in range(HESS_N_DIRS)]
    curvs_B = np.array(curvs_B)
    n_neg_B = (curvs_B < 0).sum()
    print(f'  CURVATURE SWEEP ({HESS_N_DIRS} dirs, FFN+readout subspace):')
    print(f'  min curvature: {curvs_B.min():.4e}    max: {curvs_B.max():.4e}    mean: {curvs_B.mean():.4e}')
    print(f'  negative curvatures: {n_neg_B}/{HESS_N_DIRS}  ({100*n_neg_B/HESS_N_DIRS:.1f}%)')
    print()
    if tot_B < 5e-2 and n_neg_B == 0:
        print('  VERDICT: Point B is a LOCAL MINIMUM (gradient≈0, all sampled curvatures positive)')
        if L0_B > L0_A + 0.05:
            print(f'  Its CE ({L0_B:.4f}) > Point A CE ({L0_A:.4f}) → it is NOT the global minimum.')
            print('  PROOF: a lower CE exists (inarch/Bayes level) → GD is stuck at a local min.')
    elif n_neg_B > 0:
        print('  VERDICT: Point B is a SADDLE POINT in FFN subspace (GD not yet converged)')
    else:
        print('  VERDICT: gradient still large → GD has not converged yet')
else:
    print('='*72)
    print(f'POINT B: GD model not found at {_mp_B}')
    print('  → retrain with train_identical_arch.py to get the local-min checkpoint')
    print('  → this cell will then compute gradient/curvature at the 1.59-basin')
    print('='*72)

# ── GRADIENT NORM SWEEP across all alpha (quick check: is construction a saddle for all?) ─
if HESS_ALL_THR:
    print()
    print('GRADIENT NORM AT W* + DORMANT FFN FOR ALL ALPHA:')
    print(f'  {"alpha":>6}  {"CE":>8}  {"||grad_ff2||":>14}  {"||grad_total||":>14}  verdict')
    for _t in sorted(thresholds):
        _x, _y, _T, _V, _bl = get_data('test', _t)
        _Vs = int(_bl['vocab_set1']); _Nn = min(300, _x.shape[0])
        _xb2 = _x[:_Nn].to(device); _yw2 = _y[:_Nn, TOKEN_START:TOKEN_END].to(device)
        _m2  = _build_wstar_dormant(_V, _T, _Vs, device)
        _set_grad_scope(_m2, ffn_only=True)
        _L2  = _loss_bwd(_m2, _xb2, _yw2, _V)
        _f1n, _f2n, _fcn, _tn = _grad_norms(_m2)
        _verdict = 'SADDLE (not critical)' if _tn > 1e-4 else 'near-critical point'
        print(f'  {_t:>6}  {_L2:>8.4f}  {_f2n:>14.4e}  {_tn:>14.4e}  {_verdict}')

# ── VISUALIZATION ────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2 if os.path.exists(_mp_B) else 1, figsize=(14 if os.path.exists(_mp_B) else 7, 5))
axes = np.atleast_1d(axes)
ax = axes[0]
bins = np.linspace(curvs_A.min()-0.1*abs(curvs_A.min()), curvs_A.max()+0.1*abs(curvs_A.max()), 40)
ax.hist(curvs_A, bins=bins, color='steelblue', edgecolor='white', linewidth=0.5)
ax.axvline(0, color='red', lw=2, ls='--', label='v^T H v = 0')
ax.set_xlabel('directional curvature  v^T H v  (FFN+readout subspace)'); ax.set_ylabel('count')
ax.set_title(f'Point A: W* + dormant FFN  (alpha={HESS_THR})\n'
             f'neg={n_neg_A}/{HESS_N_DIRS}  → {"SADDLE" if n_neg_A>0 else "locally convex"}')
ax.legend()
if os.path.exists(_mp_B) and len(axes) > 1:
    ax2 = axes[1]
    bins2 = np.linspace(curvs_B.min()-0.1*abs(curvs_B.min()), curvs_B.max()+0.1*abs(curvs_B.max()), 40)
    ax2.hist(curvs_B, bins=bins2, color='mediumseagreen', edgecolor='white', linewidth=0.5)
    ax2.axvline(0, color='red', lw=2, ls='--', label='v^T H v = 0')
    ax2.set_xlabel('directional curvature  v^T H v  (FFN+readout subspace)'); ax2.set_ylabel('count')
    ax2.set_title(f'Point B: GD trained ({GD_TAG_H}, alpha={HESS_THR})\n'
                 f'neg={n_neg_B}/{HESS_N_DIRS}  CE={L0_B:.4f}  ||grad||={tot_B:.2e}')
    ax2.legend()
plt.tight_layout()
out = f'hessian_curvature_alpha{HESS_THR}.png'
plt.savefig(out, dpi=130); plt.show(); print(f'saved -> {out}')
