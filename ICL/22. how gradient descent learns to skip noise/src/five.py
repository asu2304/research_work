"""
The five-number model of folder 19, rewritten DIRECTLY in its five scalars.

Folder 19 (`train_five_params.py`) stores the model as full W_qk matrices with masks and tie
groups. Every free number there is one of five shared values, so the forward pass can be
written straight in terms of them. That is what this file does -- no masks, no tying hooks, no
216x216 matrices -- which makes the model fast to evaluate and, more importantly, easy to read
and to differentiate by hand.

THE FIVE NUMBERS  (all in LOGIT units, i.e. folder 19's raw W_qk entry divided by sqrt(108))

    p    position slope     layer 1:  score(i, j) += -p * (i - j)
    a1   L1 noise veto      layer 1:  score(i, j) += a1   if key token x_j is noise
    a2   L2 noise veto      layer 2:  score(i, j) += a2   if key token x_j is noise
    b2   induction          layer 2:  score(i, j) += b2 * r_j[x_i]        (x_i a signal token)
    c2   state              layer 2:  score(i, j) += c2 * <r_i, r_j>      (signal slots only)

where r_t = layer 1's output at position t, the attention-weighted mix of the one-hot tokens
strictly before t (W_ov1 = I). When layer 1 works, r_t is the one-hot of the LAST SIGNAL TOKEN
BEFORE t -- the Markov state, recovered past any amount of noise.

Layer 2's output f_i is the attention-weighted mix of the one-hot key tokens (W_ov2 = I), and
the prediction is p_i = (f_i + b) / sum(f_i + b) with the fixed readout bias b of folder 18/19.

Mask: STRICTLY causal (a position never attends to itself), except position 0, which attends
to itself because it has nothing else.

UNITS. Folder 19 divides scores by sqrt(108) and trains the raw entries with Adam lr 1.0.
`FiveParam` keeps that convention internally (raw parameters, scale SQ) so training here
reproduces folder 19 exactly; everything reported is in logits = raw / SQ.
"""
import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

V, VS, NN, T, TS = 8, 4, 4, 100, 50            # vocab, signal, noise, seq_len, scored from
SQ = math.sqrt(V + T)                          # sqrt(108): folder 19's score scale
B_SIG, NOISE_FLOOR, LOG_FLOOR = 0.01, 1e-9, 1e-30
NAMES = ('p', 'a1', 'a2', 'b2', 'c2')

ICL = '/home/hariguru/ashutosh/ICL'
F22 = os.path.join(ICL, '22. how gradient descent learns to skip noise')
DATA14 = os.path.join(ICL, '14. elimination final convergence confusion', 'dataset')
DATA22 = os.path.join(F22, 'dataset')
RUN19 = os.path.join(ICL, '19. reducing even more', 'runs',
                     'tied5_opt-adam_lr-1.0_init-zeros_wov-fixed_blocks-relevant_readout-bias_ep-200')
RUN18 = os.path.join(ICL, '18. reducing_parameters', 'runs',
                     'master_opt-adam_lr-1.0_init-zeros_wov-fixed_blocks-relevant-strict_readout-bias_ep-500')


# ------------------------------------------------------------------------------------- data

def afmt(alpha):
    """file-name form of alpha: 0.6 -> '0.6', 1.0 -> '1.0', 0.65 -> '0.65'"""
    return repr(float(round(alpha, 4)))


def data_path(alpha, split):
    """folder 22's own dataset first (new alphas), then folder 14's (the original seven)"""
    for root in (DATA22, DATA14):
        f = os.path.join(root, split, f'data_x_{afmt(alpha)}.pt')
        if os.path.exists(f):
            return f
    raise FileNotFoundError(f'no {split} data for alpha={alpha} in {DATA22} or {DATA14}')


_CACHE = {}


def load(alpha, split):
    k = (afmt(alpha), split)
    if k not in _CACHE:
        if len(_CACHE) > 6:
            _CACHE.clear()
        _CACHE[k] = torch.load(data_path(alpha, split), weights_only=False)
    return _CACHE[k]


def readout_bias(alpha, device='cpu'):
    """b_sig on signal slots; b_noise = (1-a)(1+VS*b_sig)/(NN*a) on noise slots (folder 18)"""
    b = torch.full((V,), B_SIG, device=device)
    b[VS:] = NOISE_FLOOR if alpha >= 1.0 else (1 - alpha) * (1 + VS * B_SIG) / (NN * alpha)
    return b


def ideal_ce(alpha):
    """knows the true table: alpha/2 on each successor, (1-alpha)/NN on each noise token"""
    c = -alpha * math.log(alpha / 2)
    if alpha < 1:
        c += -(1 - alpha) * math.log((1 - alpha) / NN)
    return c


def state_of(x):
    """state[b, t] = last SIGNAL token in x[b, 0..t] (inclusive); -1 if none yet.
    This is the Markov state that generates y[b, t] = x[b, t+1]."""
    B, Tn = x.shape
    st = torch.full((B, Tn), -1, dtype=torch.long, device=x.device)
    cur = torch.full((B,), -1, dtype=torch.long, device=x.device)
    for t in range(Tn):
        sig = x[:, t] < VS
        cur = torch.where(sig, x[:, t], cur)
        st[:, t] = cur
    return st


def prev_signal(x):
    """prev[b, t] = last SIGNAL token in x[b, 0..t-1] (strictly before t); -1 if none.
    This is what a perfect layer 1 writes at position t (it cannot see t itself)."""
    st = state_of(x)
    out = torch.full_like(st, -1)
    out[:, 1:] = st[:, :-1]
    return out


def run_length(x):
    """number of consecutive noise tokens ending at t (0 if x[t] is signal)"""
    B, Tn = x.shape
    rl = torch.zeros((B, Tn), dtype=torch.long, device=x.device)
    for t in range(Tn):
        isn = x[:, t] >= VS
        rl[:, t] = torch.where(isn, (rl[:, t - 1] if t else 0) + 1, torch.zeros_like(rl[:, t]))
    return rl


# ------------------------------------------------------------------------------------ model

_MASK = {}


def causal_mask(Tn, device):
    k = (Tn, str(device))
    if k not in _MASK:
        m = torch.triu(torch.full((Tn, Tn), float('-inf'), device=device), diagonal=0)
        m[0, 0] = 0.0
        _MASK[k] = m
    return _MASK[k]


_DIST = {}


def dist(Tn, device):
    """D[i, j] = i - j  (only the causal side i > j is ever used)"""
    k = (Tn, str(device))
    if k not in _DIST:
        i = torch.arange(Tn, device=device).view(-1, 1)
        j = torch.arange(Tn, device=device).view(1, -1)
        _DIST[k] = (i - j).float().clamp_min(0)
    return _DIST[k]


def perfect_r(x):
    """what an ideal layer 1 writes: r_t = one-hot of the last signal token strictly before t
    (all-zero where there is none yet)"""
    ps = prev_signal(x)
    r = F.one_hot(ps.clamp_min(0), V).float()
    return r * (ps >= 0).float().unsqueeze(-1)


def forward(theta, x, parts=False, r_override=None):
    """theta: tensor (5,) in LOGITS [p, a1, a2, b2, c2].  x: (B, T) long.

    Returns f (B, T, V), layer 2's output, as a distribution over tokens.
    parts=True also returns the layer-1/2 attention and layer 1's output r.
    r_override: use this (B, T, V) as layer 1's output instead of computing it (counterfactual,
    e.g. perfect_r(x) = a layer 1 that recovers the state exactly); p and a1 are then unused.
    """
    p, a1, a2, b2, c2 = theta[0], theta[1], theta[2], theta[3], theta[4]
    B, Tn = x.shape
    dev = x.device
    oh = F.one_hot(x, V).float()                                   # (B, T, V)
    noise = (x >= VS).float()                                      # (B, T)
    mask = causal_mask(Tn, dev)

    if r_override is None:
        s1 = -p * dist(Tn, dev).unsqueeze(0) + a1 * noise.unsqueeze(1)  # key-indexed veto
        A1 = torch.softmax(s1 + mask, dim=-1)                      # (B, T, T)
        r = A1 @ oh                                                # (B, T, V)
    else:
        s1 = A1 = None
        r = r_override
    rs = r[..., :VS]                                               # signal slots

    qtok = oh[..., :VS]                                            # one-hot of x_i if signal
    s2 = (a2 * noise.unsqueeze(1)
          + b2 * (qtok @ rs.transpose(1, 2))                       # r_j[x_i]
          + c2 * (rs @ rs.transpose(1, 2)))                        # <r_i, r_j>
    A2 = torch.softmax(s2 + mask, dim=-1)
    f = A2 @ oh
    if parts:
        return f, dict(A1=A1, r=r, A2=A2, s1=s1, s2=s2)
    return f


def token_nll(f, y, b):
    """per-token -log p[y] with p = (f+b)/sum(f+b)"""
    q = f + b
    q = q / q.sum(-1, keepdim=True)
    return -torch.log(q.gather(-1, y.unsqueeze(-1)).squeeze(-1).clamp_min(LOG_FLOOR))


def ce(theta, x, y, alpha, chunk=4096, weights=None, perfect_layer1=False):
    """mean CE over tokens [TS:T). weights (B, T-TS) optional per-token weights (sum-normalised).
    Differentiable in theta (call under torch.no_grad() for evaluation).
    perfect_layer1=True replaces layer 1 by exact state recovery (see perfect_r)."""
    b = readout_bias(alpha, x.device)
    tot = 0.0
    n = x.shape[0] * (T - TS) if weights is None else float(weights.sum())
    for i in range(0, x.shape[0], chunk):
        xb, yb = x[i:i + chunk], y[i:i + chunk, TS:]
        ro = perfect_r(xb) if perfect_layer1 else None
        nll = token_nll(forward(theta, xb, r_override=ro)[:, TS:], yb, b)
        if weights is not None:
            nll = nll * weights[i:i + chunk]
        tot = tot + nll.sum()
    return tot / n


class FiveParam(nn.Module):
    """Five raw scalars, trained exactly like folder 19 (raw = logit * SQ, Adam lr 1.0)."""

    def __init__(self, theta_logit=None):
        super().__init__()
        t = torch.zeros(5) if theta_logit is None else torch.as_tensor(theta_logit, dtype=torch.float32)
        self.raw = nn.Parameter(t * SQ)

    def theta(self):
        return self.raw / SQ


# ---------------------------------------------------------------------- folder 19 / 18 readers

def theta_from_f19_state(sd):
    """the five logits from a folder-19 state_dict"""
    W1, W2, d1 = sd['layer1.W_qk'], sd['layer2.W_qk'], V + T
    raw = [float(sd['p_ramp']), float(W1[0, VS]), float(W2[0, VS]),
           float(W2[0, d1]), float(W2[d1, d1])]
    return torch.tensor(raw) / SQ


def f19_final(alpha):
    sd = torch.load(os.path.join(RUN19, f'model_final_seed0_x_{afmt(alpha)}.pth'),
                    map_location='cpu', weights_only=False)
    return theta_from_f19_state(sd)


def f19_snapshots(alpha):
    """(W, G): W[k] = the five logits before step k+1; G[k] = dL/d(logit) at W[k].

    Folder 19 stores raw values and dL/d(raw) (a tied group's gradient is the SUM over its
    members = dL/d(shared raw value)). logit = raw / SQ  =>  dL/d(logit) = SQ * dL/d(raw)."""
    import json
    a = afmt(alpha)
    meta = json.load(open(os.path.join(RUN19, f'snapmeta_x_{a}.json')))
    nw, ng = meta['w_rows_written'], meta['g_rows_written']
    def col(kind, param, gname=None):
        arr = np.load(os.path.join(RUN19, f'{kind}_{param}_x_{a}.npy'), mmap_mode='r')
        idx = 0 if gname is None else meta['tie_groups'][param][gname]
        return np.asarray(arr[:, idx], dtype=np.float64)
    cols = [('p_ramp', None), ('layer1.W_qk', 'L1.A.knoi'), ('layer2.W_qk', 'L2.A.knoi'),
            ('layer2.W_qk', 'L2.B.induction.diag'), ('layer2.W_qk', 'L2.C.state.diag')]
    W = np.stack([col('w', p, g) for p, g in cols], 1)[:nw] / SQ
    G = np.stack([col('g', p, g) for p, g in cols], 1)[:ng] * SQ
    return W, G
