"""Appendix F: the unequal-successor control benchmark (Table 5) and the Hessian/
perturbation stability probe cited in Appendix F's last paragraph. Generates CE for
ideal-alpha / ctx-count / in-arch-log-readout / GD on data from
data_generation/generate_unequal_successor_control.py, trained by
training/train_unequal_successor_control.py, plus a perturbation test around the
W*-attention-plus-dormant-FFN point in DisentangledTransformerMLP's parameter space.

NOTE: the in-code comment below says p1~U(0.25,0.75); the actual CONFIG used to
generate the paper's data (and shipped in generate_unequal_successor_control.py) is
p1~U(0,1) per state -- matching the paper text. This is a stale comment from an
earlier draft of the notebook cell, left as in the original for fidelity.

Extracted verbatim from ICL/10.../main.ipynb (cells 0 and 5). Run after generating
data and training the control model:  python unequal_successor_control_appendixF.py
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
# ══ CELL 5: IS THE IN-ARCH CONSTRUCTION AT THE GLOBAL MINIMUM? ══════════════════════════
# The global minimum of population CE for ANY context-conditioned predictor is the
# context-Bayes predictor: E[p1|n1,n2] under a Beta(1,1) prior on the unequal p1.
# For our altered distribution (p1~U[0,1] per state), this is Laplace-smoothed
# empirical counts (add 1 to each successor count = Beta(1,1) pseudo-count).
#
# We check for each alpha:
#   construction gap = inarch_CE - context_Bayes_CE  (0 → construction IS at global min)
#   GD gap           = gd_CE     - inarch_CE          (0 → GD found the construction)
# Then: perturbation stability test at the W* init point in DisentangledTransformerMLP space.
# ──────────────────────────────────────────────────────────────────────────────────────────
MIN_SPLIT    = 'test'
MIN_N        = 2000
GD_TAG       = 'idarch'
PERTURB_STDS = [1e-4, 5e-4, 1e-3, 5e-3, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0]
# ──────────────────────────────────────────────────────────────────────────────────────────
import os, math, numpy as np, torch, torch.nn.functional as F
import matplotlib.pyplot as plt

# ── context-Bayes predictor (Beta(1,1) prior = Laplace eps=1 smoothing) ──────────────────
def context_bayes_dist(blob, alpha):
    x_np = blob['x'].numpy()
    V = int(blob['total_vocab']); Vs = int(blob['vocab_set1']); n_noise = V - Vs
    B = x_np.shape[0]; W = TOKEN_END - TOKEN_START
    trans_counts = np.zeros((B, V, Vs), dtype=np.float64)
    pending      = np.zeros((B, V),     dtype=np.float64)
    last_ruled   = np.full(B, -1, dtype=np.int64)
    def _step(col):
        tok = x_np[:, col]; is_sig = tok < Vs
        ri = np.where(is_sig)[0]
        if len(ri):
            rt = tok[ri]
            oh = (np.arange(Vs) == rt[:, None]).astype(np.float64)
            np.add.at(trans_counts, ri, pending[ri, :, None] * oh[:, None, :])
            pending[ri, :] = 0.0; pending[ri, rt] = 1.0; last_ruled[ri] = rt
        ni = np.where(~is_sig)[0]
        if len(ni): pending[ni, tok[ni]] += 1.0
    for i in range(TOKEN_START): _step(i)
    eps = 1.0; p_win = np.zeros((B, W, V), dtype=np.float64)
    for t in range(TOKEN_START, TOKEN_END):
        v = x_np[:, t]; is_sig_q = v < Vs
        q   = np.where(is_sig_q, v, np.maximum(last_ruled, 0))
        cnt = trans_counts[np.arange(B), q] + eps
        tot = cnt.sum(axis=1, keepdims=True)
        p   = np.zeros((B, V), dtype=np.float64)
        p[:, :Vs] = cnt / tot * alpha; p[:, Vs:] = (1.0 - alpha) / n_noise
        p_win[:, t - TOKEN_START] = p
        _step(t)
    return p_win

def context_bayes_ce(blob, alpha):
    return _ce_from_win(context_bayes_dist(blob, alpha), blob['y'].numpy())

# ── compute full hierarchy for all thresholds ────────────────────────────────────────────
print('Computing CE hierarchy for all alpha...')
print(f'  {"alpha":>5}  {"ideal":>8}  {"Bayes":>8}  {"ctx":>8}  {"inarch":>8}  {"GD":>8}  |  constr-gap  GD-gap')
rows = []
for thr in sorted(thresholds):
    x, y, T, V, blob = get_data(MIN_SPLIT, thr)
    N = min(MIN_N, x.shape[0]); Vs = int(blob['vocab_set1'])
    bn = {k: blob[k] for k in blob if k not in ('x','y','transitions','trans_probs')}
    bn['x'] = blob['x'][:N]; bn['y'] = blob['y'][:N]
    bn['transitions'] = blob['transitions'][:N]; bn['trans_probs'] = blob['trans_probs'][:N]
    ce_i  = ideal_alpha_ce(bn, thr)
    ce_b  = context_bayes_ce(bn, thr)
    ce_c  = alpha_ce(bn, thr)
    ce_ia = inarch_ce(thr, T, V, Vs, bn['x'], bn['y'])
    mp = os.path.join(MODELS_DIR, f'model_{GD_TAG}_x_{thr}.pth')
    ce_gd = trained_ce(thr, T, V, bn['x'], bn['y']) if os.path.exists(mp) else None
    rows.append((thr, ce_i, ce_b, ce_c, ce_ia, ce_gd))
    gd_s  = f'{ce_gd:.4f}' if ce_gd is not None else '   N/A'
    cg    = ce_ia - ce_b
    gg    = (ce_gd - ce_ia) if ce_gd is not None else float('nan')
    print(f'  {thr:>5}  {ce_i:>8.4f}  {ce_b:>8.4f}  {ce_c:>8.4f}  {ce_ia:>8.4f}  {gd_s:>8}  |  {cg:>+10.4f}  {gg:>+8.4f}')

# ── verdict table ───────────────────────────────────────────────────────────────────────
print()
print('VERDICT — α > 0.5 focus:')
for thr, ce_i, ce_b, ce_c, ce_ia, ce_gd in rows:
    if thr < 0.5: continue
    cg = ce_ia - ce_b
    gg = (ce_gd - ce_ia) if ce_gd is not None else float('nan')
    at_glob  = 'YES' if cg < 0.01 else f'NO  (+{cg:.4f} above Bayes)'
    gd_finds = ('N/A' if math.isnan(gg) else
                'YES' if gg < 0.01 else
                f'LOCAL MIN  (+{gg:.4f} above construction)')
    print(f'  α={thr}  construction at global min? {at_glob}  |  GD finds it? {gd_finds}')

# ── CE hierarchy plot ───────────────────────────────────────────────────────────────────
alphas  = [r[0] for r in rows]
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
ax = axes[0]
ax.plot(alphas, [r[1] for r in rows], 'o-',  color='tab:green',  lw=2, label='ideal-alpha (oracle floor)')
ax.plot(alphas, [r[2] for r in rows], 's--', color='gold',        lw=2, ms=8, label='context-Bayes (global min for any context predictor)')
ax.plot(alphas, [r[3] for r in rows], '^:',  color='tab:blue',   lw=1.5, label='ctx-count (eps=0.01)')
ax.plot(alphas, [r[4] for r in rows], 'D-',  color='navy',        lw=2,  label='in-arch log readout (construction)')
gd_vals = [r[5] if r[5] is not None else float('nan') for r in rows]
if any(not math.isnan(v) for v in gd_vals):
    ax.plot(alphas, gd_vals, 'P-', color='tab:purple', lw=2, label=f'GD trained ({GD_TAG})')
ax.set_xlabel('alpha'); ax.set_ylabel('cross-entropy (nats)')
ax.set_title(f'CE hierarchy with context-Bayes (global min)  [{MIN_SPLIT}]')
ax.legend(fontsize=8); ax.grid(alpha=0.3)
ax = axes[1]
gap_constr = [r[4] - r[2] for r in rows]
gap_gd_arr = [(r[5] - r[4]) if r[5] is not None else float('nan') for r in rows]
xp = np.array(alphas); w = 0.04
ax.bar(xp - w, gap_constr, 2*w, color='navy', alpha=0.8, label='inarch - Bayes  (construction gap from global min)')
if any(not math.isnan(v) for v in gap_gd_arr):
    ax.bar(xp + w, gap_gd_arr, 2*w, color='tab:purple', alpha=0.8, label='GD - inarch  (GD gap from construction)')
ax.axhline(0, color='k', lw=0.8, ls='--')
ax.set_xlabel('alpha'); ax.set_ylabel('CE gap (nats)')
ax.set_title('Gap from global min: construction vs GD scratch')
ax.legend(fontsize=8); ax.grid(alpha=0.3, axis='y')
plt.tight_layout()
out = f'global_min_analysis_{MIN_SPLIT}.png'
plt.savefig(out, dpi=130); plt.show(); print(f'saved -> {out}')

# ── perturbation test at W* init in DisentangledTransformerMLP space ────────────────────
# Init: copy W* attention weights into DisentangledTransformerMLP, set ff2.weight=0 (dormant FFN)
# → measures stability of the attention-circuit solution in the full parameter space
print()
print('PERTURBATION TEST at alpha=0.8 (W* attention init, ff2=0 dormant FFN)')
_thr = 0.8
x, y, T, V, blob = get_data(MIN_SPLIT, _thr)
_N = min(500, x.shape[0]); Vs = int(blob['vocab_set1'])
_bn = {k: blob[k] for k in blob if k not in ('x','y','transitions','trans_probs')}
_bn['x'] = blob['x'][:_N]; _bn['y'] = blob['y'][:_N]
_bn['transitions'] = blob['transitions'][:_N]; _bn['trans_probs'] = blob['trans_probs'][:_N]
_xb = _bn['x'].to(device); _ywin = _bn['y'][:, TOKEN_START:TOKEN_END].to(device)
_ws = _WSTAR_CACHE.get((V, T, Vs)) or PerfectInductionTransformer(V, T, Vs).to(device).eval()
_WSTAR_CACHE[(V, T, Vs)] = _ws
_D = V + T  # initial_dim = 220
_pm = DisentangledTransformerMLP(V, T, _D, 128, bias=True, act='gelu').to(device)
with torch.no_grad():
    for ln in ('layer1', 'layer2'):
        for wn in ('W_q','W_k','W_v','W_o'):
            s = getattr(getattr(_ws, ln), wn).weight
            d = getattr(getattr(_pm, ln), wn).weight
            if s.shape == d.shape: d.copy_(s)
    _pm.ff2.weight.data.zero_(); _pm.ff2.bias.data.zero_()
    if _ws.fc_out.weight.shape == _pm.fc_out.weight.shape:
        _pm.fc_out.weight.data.copy_(_ws.fc_out.weight)

@torch.no_grad()
def _pm_ce(m):
    lg = m(_xb)[:, TOKEN_START:TOKEN_END, :]
    p  = F.softmax(lg, -1).clamp_min(1e-12)
    return float((-torch.log(p.gather(-1, _ywin.unsqueeze(-1)).squeeze(-1))).mean())

_base_ce = _pm_ce(_pm)
_ref_sd  = {k: v.clone() for k, v in _pm.state_dict().items()}
_n_params = sum(v.numel() for v in _ref_sd.values())
print(f'  CE at W* init + dormant FFN: {_base_ce:.4f}  (params={_n_params:,})')
print(f'  context-Bayes (global min) : {context_bayes_ce(_bn, _thr):.4f}')
print(f'  in-arch construction CE    : {inarch_ce(_thr, T, V, Vs, _bn["x"], _bn["y"]):.4f}')
print()
print(f'  sigma        CE      delta   direction')
_perturb_ces = []
for _sigma in PERTURB_STDS:
    with torch.no_grad():
        for k, v in _ref_sd.items(): _pm.state_dict()[k].copy_(v + _sigma * torch.randn_like(v))
    _ce = _pm_ce(_pm); _delta = _ce - _base_ce; _perturb_ces.append((_sigma, _ce, _delta))
    _dir = 'increases CE (stable toward this basin)' if _delta > 0 else 'DECREASES CE → saddle point'
    print(f'  sigma={_sigma:<8.4f}  CE={_ce:.4f}  d={_delta:+.4f}  {_dir}')

print()
_any_decrease = any(d < 0 for _, _, d in _perturb_ces)
if not _any_decrease:
    print('  RESULT: W* attention + dormant FFN is a LOCAL MINIMUM of the attention weights.')
    print('  The attention circuit is stable, but the FFN head needs to ACTIVATE to reach the')
    print('  global min (construction/Bayes level). GD at alpha=0.8 is stuck because the FFN')
    print('  activation barrier is too high for GD to cross from random init at this alpha.')
else:
    print('  RESULT: W* init point is a SADDLE POINT — noise can escape to lower CE.')
    print('  GD should in principle be able to find the global min from nearby init.')
