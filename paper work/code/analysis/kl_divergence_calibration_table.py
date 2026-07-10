"""Table 1 (LEFT half, KL columns) and Section 4 claim (2): attention-only GD is a
temperature-calibrated count reader. Computes per-position KL(GD || ctx-count),
KL(GD || W* fixed sigma), KL(GD || W*-better tuned sigma,b_N) at N_SEQS=200 for
every alpha, on the trained attention-only models from training/train_attention_only.py.

Extracted verbatim from ICL/6.../opening_up_trained_model.ipynb (cells 4, 15, 16).
Expects: DATA_DIR='dataset', MODELS_DIR='models' relative to cwd (point them at the
folder-6 dataset/ and models/ produced by generate_noisy_markov.py + train_attention_only.py,
or edit the two paths below).

Run: python kl_divergence_calibration_table.py
"""

# ── SETTINGS ──────────────────────────────────────────────────────────────────
DATA_DIR    = 'dataset'
MODELS_DIR  = 'models'
DEVICE      = 'cuda:3'

N_SEQS      = 200         # sequences per split used for CE evaluation
TOKEN_START = 100         # evaluate CE on tokens 100–200 (last half, seq_len=200)
TOKEN_END   = 200
WS_SIGMA    = 5.5         # temperature scale on W* logits
ATTN_DIM    = 220          # must match train.py attn_dim
# ──────────────────────────────────────────────────────────────────────────────

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

device = torch.device(DEVICE)

# ── Trained model (DisentangledTransformer, must match train.py) ──────────────
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

class DisentangledTransformer(nn.Module):
    def __init__(self, total_vocab, seq_len, attn_dim, bias=False):
        super().__init__()
        self.total_vocab = total_vocab
        self.seq_len     = seq_len
        self.initial_dim = total_vocab + seq_len
        self.layer1 = ConcatAttentionBlock(self.initial_dim, attn_dim)
        self.layer2 = ConcatAttentionBlock(self.initial_dim + attn_dim, attn_dim)
        self.fc_out = nn.Linear(self.initial_dim + 2 * attn_dim, total_vocab, bias=bias)
    def forward(self, x, return_attn=False):
        B, T = x.shape
        tok = F.one_hot(x, num_classes=self.total_vocab).float()
        pos = F.one_hot(torch.arange(T, device=x.device),
                        num_classes=self.seq_len).float().unsqueeze(0).expand(B, T, -1)
        s = torch.cat([tok, pos], dim=-1)
        mask = torch.triu(torch.ones(T, T, device=x.device) * float('-inf'), diagonal=1)
        s, a1 = self.layer1(s, mask)
        s, a2 = self.layer2(s, mask)
        return (self.fc_out(s), [a1, a2]) if return_attn else self.fc_out(s)

# ── PerfectInductionTransformer W* (seq_len=200 matches hand construction) ────
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

# ── Helpers ───────────────────────────────────────────────────────────────────
def load_data(split, thr, n_cap=N_SEQS):
    blob = torch.load(os.path.join(DATA_DIR, split, f'data_x_{thr}.pt'), weights_only=False)
    N    = min(n_cap, blob['x'].shape[0])
    blob['x'] = blob['x'][:N]; blob['y'] = blob['y'][:N]
    if 'transitions' in blob: blob['transitions'] = blob['transitions'][:N]
    return blob['x'], blob['y'], int(blob['seq_len']), int(blob['total_vocab']), blob

def load_model(thr, seq_len, total_vocab):
    attn_dim = ATTN_DIM
    model = DisentangledTransformer(total_vocab, seq_len, attn_dim, bias=True).to(device)
    model.load_state_dict(torch.load(os.path.join(MODELS_DIR, f'model_x_{thr}.pth'),
                                     map_location=device, weights_only=True))
    model.eval(); return model

@torch.no_grad()
def compute_ce(model, x, y, total_vocab, batch_size=256):
    total_loss, n_tokens = 0.0, 0
    for i in range(0, x.shape[0], batch_size):
        xb = x[i:i+batch_size].to(device); yb = y[i:i+batch_size].to(device)
        logits = model(xb)[:, TOKEN_START:TOKEN_END, :]
        yb_w   = yb[:, TOKEN_START:TOKEN_END]
        total_loss += F.cross_entropy(logits.reshape(-1, total_vocab),
                                      yb_w.reshape(-1), reduction='sum').item()
        n_tokens += yb_w.numel()
    return total_loss / n_tokens

@torch.no_grad()
def wstar_ce(blob, sigma=WS_SIGMA, batch_size=128):
    x = blob['x']; y = blob['y']
    V = int(blob['total_vocab']); T = int(blob['seq_len'])
    model = PerfectInductionTransformer(V, T, int(blob['vocab_set1'])).to(device).eval()
    total_loss, n_tok = 0.0, 0
    for i in range(0, x.shape[0], batch_size):
        xb = x[i:i+batch_size].to(device); yb = y[i:i+batch_size].to(device)
        logits = model(xb)[:, TOKEN_START:TOKEN_END, :] * sigma
        yb_w   = yb[:, TOKEN_START:TOKEN_END]
        total_loss += F.cross_entropy(logits.reshape(-1, V), yb_w.reshape(-1), reduction='sum').item()
        n_tok += yb_w.numel()
    del model
    if device.type == 'cuda': torch.cuda.empty_cache()
    return total_loss / n_tok

def alpha_ce(blob, alpha):
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
    eps = 0.01; total_ce, n_tok = 0.0, 0
    for t in range(TOKEN_START, TOKEN_END):
        v = x_np[:, t]; is_sig_q = v < V_sig
        query = np.where(is_sig_q, v, np.maximum(last_ruled, 0))
        cnt = trans_counts[np.arange(B), query]; cnt_s = cnt + eps
        tot_s = cnt_s.sum(axis=1, keepdims=True)
        p = np.zeros((B, V), dtype=np.float64)
        p[:, :V_sig] = cnt_s / tot_s * alpha; p[:, V_sig:] = (1.0 - alpha) / n_noise
        p = np.maximum(p, 1e-12)
        total_ce += (-np.log(p[np.arange(B), y_np[:, t]])).sum(); n_tok += B
        _step(t)
    return total_ce / n_tok

def ideal_alpha_ce(blob, alpha):
    x_np = blob['x'].numpy(); y_np = blob['y'].numpy(); trans = blob['transitions']
    V = int(blob['total_vocab']); V_sig = int(blob['vocab_set1'])
    n_noise = V - V_sig; B = x_np.shape[0]
    trans_arr = np.array([[trans[b][v] for v in range(V_sig)] for b in range(B)], dtype=np.int64)
    total_ce, n_tok = 0.0, 0
    last_sig = np.zeros(B, dtype=np.int64)
    for t in range(TOKEN_END):
        is_sig = x_np[:, t] < V_sig
        last_sig = np.where(is_sig, x_np[:, t], last_sig)
        if t < TOKEN_START: continue
        s0 = trans_arr[np.arange(B), last_sig, 0]; s1 = trans_arr[np.arange(B), last_sig, 1]
        p = np.zeros((B, V), dtype=np.float64)
        p[:, V_sig:] = (1.0 - alpha) / n_noise
        p[np.arange(B), s0] += alpha / 2; p[np.arange(B), s1] += alpha / 2
        total_ce += (-np.log(np.maximum(p[np.arange(B), y_np[:, t]], 1e-12))).sum(); n_tok += B
    return total_ce / n_tok

# ── Discover thresholds ───────────────────────────────────────────────────────
thresholds = sorted(
    float(f[len('model_x_'):-len('.pth')])
    for f in os.listdir(MODELS_DIR)
    if f.startswith('model_x_') and f.endswith('.pth')
)
print(f'Thresholds found : {thresholds}')
print(f'N_SEQS={N_SEQS}  window=[{TOKEN_START}:{TOKEN_END}]  device={DEVICE}  attn_dim={ATTN_DIM}\n')

# ── Evaluate all baselines for both train and test splits ─────────────────────
results = {'train': {}, 'test': {}}
for split in ('train', 'test'):
    print(f'--- {split} ---')
    for thr in thresholds:
        x, y, seq_len, total_vocab, blob = load_data(split, thr)
        model   = load_model(thr, seq_len, total_vocab)
        ce_gd   = compute_ce(model, x, y, total_vocab)
        ce_ws   = wstar_ce(blob)
        ce_alp  = alpha_ce(blob, thr)
        ce_ialp = ideal_alpha_ce(blob, thr)
        results[split][thr] = dict(gd=ce_gd, ws=ce_ws, alp=ce_alp, ialp=ce_ialp)
        print(f'  thr={thr:.2f}  GD={ce_gd:.4f}  W*={ce_ws:.4f}  ctx={ce_alp:.4f}  ideal={ce_ialp:.4f}')

print('\nEvaluation complete.')

# ═══ notebook cell 15 ═══
# ── Setup: load one GD model + the hand-coded W* at the same threshold ────────
THR   = 0.9    # inspect a well-trained threshold in detail
N_CMP = 128     # sequences for the detailed (single-threshold) comparisons

x_c, y_c, seq_len, V, blob_c = load_data('test', THR, n_cap=N_CMP)
V_sig    = int(blob_c['vocab_set1'])
gd_model = load_model(THR, seq_len, V)
ws_model = PerfectInductionTransformer(V, seq_len, V_sig).to(device).eval()
print(f'THR={THR}  seq_len={seq_len}  V={V}  V_sig={V_sig}  N={x_c.shape[0]}')

# ═══ notebook cell 16 ═══
# ── Rung 1 (FUNCTIONAL): is GD computing the count-α distribution? ────────────
# KL(GD ‖ ctx-count)  vs  KL(GD ‖ W* σ=5.5)  vs  KL(GD ‖ W*-better)  per threshold.
# W*-better = same W* logits, but readout = σ·logits + bN·(noise mask), with the
# per-α (σ, bN) calibrated in the "W*-better" construction cell above.
def ctx_count_probs(blob, alpha, eps=0.01):
    """Laplace-smoothed empirical count-α predictive distribution → [B, window, V]."""
    x_np = blob['x'].numpy(); Vv = int(blob['total_vocab']); Vs = int(blob['vocab_set1'])
    n_noise = Vv - Vs; B = x_np.shape[0]
    tc_ = np.zeros((B, Vv, Vs)); pend = np.zeros((B, Vv)); lastr = np.full(B, -1, np.int64)
    def step(col):
        c = x_np[:, col]; ruled = c < Vs; ri = np.where(ruled)[0]
        if len(ri):
            rt = c[ri]; oh = (np.arange(Vs) == rt[:, None]).astype(float)
            np.add.at(tc_, ri, pend[ri, :, None] * oh[:, None, :])
            pend[ri, :] = 0.0; pend[ri, rt] = 1.0; lastr[ri] = rt
        ni = np.where(~ruled)[0]
        if len(ni): pend[ni, c[ni]] += 1.0
    for i in range(TOKEN_START): step(i)
    Wn = TOKEN_END - TOKEN_START; out = np.zeros((B, Wn, Vv))
    for j, t in enumerate(range(TOKEN_START, TOKEN_END)):
        v = x_np[:, t]; q = np.where(v < Vs, v, np.maximum(lastr, 0))
        cs = tc_[np.arange(B), q] + eps; ts = cs.sum(1, keepdims=True)
        p = np.zeros((B, Vv)); p[:, :Vs] = cs / ts * alpha; p[:, Vs:] = (1 - alpha) / n_noise
        out[:, j, :] = np.maximum(p, 1e-12); step(t)
    return out

def mean_kl(p, q):                       # KL(p ‖ q), mean over batch and window
    p = np.clip(p, 1e-12, 1); q = np.clip(q, 1e-12, 1)
    return float((p * (np.log(p) - np.log(q))).sum(-1).mean())

def _sig_bN_for(thr, Lws_train, y_train, V, V_sig):
    """Reuse the construction cell's calibrated (σ, bN); else fit inline on train."""
    if 'better_par' in globals() and thr in better_par:
        return better_par[thr]
    nmask = torch.zeros(V); nmask[V_sig:] = 1.0
    best = (float('inf'), (WS_SIGMA, 0.0))
    for sig in np.arange(0.5, 12.01, 0.5):
        for bN in np.arange(-3.0, 6.01, 0.5):
            ce = F.cross_entropy((Lws_train * float(sig) + nmask * float(bN)).reshape(-1, V),
                                 y_train.reshape(-1)).item()
            if ce < best[0]: best = (ce, (float(sig), float(bN)))
    return best[1]

kl_ctx, kl_ws, kl_bet = [], [], []
for thr in thresholds:
    xb_, _, sl, Vv, bl = load_data('test', thr, n_cap=N_SEQS)
    V_sig = int(bl['vocab_set1']); nmask = torch.zeros(Vv); nmask[V_sig:] = 1.0
    m  = load_model(thr, sl, Vv)
    ws = PerfectInductionTransformer(Vv, sl, V_sig).to(device).eval()
    with torch.no_grad():
        pg   = F.softmax(m(xb_.to(device))[:, TOKEN_START:TOKEN_END], -1).cpu().numpy()
        Lws  = ws(xb_.to(device))[:, TOKEN_START:TOKEN_END].cpu()                 # raw W* logits
    pw = F.softmax(Lws * WS_SIGMA, -1).numpy()
    # calibrated (σ, bN) for W*-better
    if 'better_par' in globals() and thr in better_par:
        sig, bN = better_par[thr]
    else:                                                                          # inline fallback fit
        xtr, ytr, sl2, _, bltr = load_data('train', thr, n_cap=N_SEQS)
        wstr = PerfectInductionTransformer(Vv, sl2, int(bltr['vocab_set1'])).to(device).eval()
        with torch.no_grad():
            Ltr = wstr(xtr.to(device))[:, TOKEN_START:TOKEN_END].cpu()
        sig, bN = _sig_bN_for(thr, Ltr, ytr[:, TOKEN_START:TOKEN_END], Vv, V_sig); del wstr
    pbet = F.softmax(Lws * sig + nmask * bN, -1).numpy()                            # W*-better probs
    pc = ctx_count_probs(bl, thr)
    kl_ctx.append(mean_kl(pg, pc)); kl_ws.append(mean_kl(pg, pw)); kl_bet.append(mean_kl(pg, pbet))
    del m, ws

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(thresholds, kl_ctx, 'o-', color='firebrick', label='KL(GD ‖ ctx-count)')
ax.plot(thresholds, kl_ws,  's-', color='seagreen',  label=f'KL(GD ‖ W* σ={WS_SIGMA})')
ax.plot(thresholds, kl_bet, 'D-', color='purple',    label='KL(GD ‖ W*-better, per-α σ+noise)')
ax.set_xlabel('threshold (α)'); ax.set_ylabel('mean KL over window')
ax.set_title("Which construction does GD's predictive distribution match?")
ax.grid(alpha=.3); ax.legend(); plt.tight_layout(); plt.show()
for t, kc, kw, kb in zip(thresholds, kl_ctx, kl_ws, kl_bet):
    print(f'  α={t:.1f}   KL→ctx={kc:.3f}   KL→W*={kw:.3f}   KL→W*-better={kb:.3f}')