"""Training script -- two-layer transformer with SEPARATE W_q / W_k, W_ov frozen.

Folder 20: the separate-parameterisation detour.

WHAT IS THE SAME AS FOLDERS 17 / 18 / 19
----------------------------------------
Everything except the query-key parametrisation.

    x            one-hot token (8) + one-hot position (100)        -> 108 = d1
    layer 1      causal attention, returns cat[x, attn_out]        -> 216 = d2
    layer 2      causal attention, returns attn_out only           -> 108
    f = layer2 attn_out[..., :8]                                   -> the raw count vector
    loss         CE on (f + b) / sum(f + b)                        -> no unembedding,
                                                                      no log, no softmax

  * W_ov is FROZEN: W_ov[:8, :8] = I, everything else 0, requires_grad = False.
    That is what makes f a convex combination of token one-hots, hence f >= 0 and sum(f) = 1
    exactly, hence f already IS a distribution and the old log -> softmax pair (which was only
    ever x / sum(x), since softmax(log x) = x / sum(x)) has nothing left to do.
    With --train-wov that guarantee is gone: f can go negative, sum(f + b) can cross zero and
    -log p is unbounded below. The loop detects a negative CE and stops.

  * The readout bias b is FIXED from alpha, never trained:

        b_sig   = 0.01                                         (add-beta smoothing)
        b_noise = (1 - alpha)(1 + V_sig*b_sig) / (n_noise*alpha)

    A good induction head drives the NOISE slots of f to zero -- that is the noise filter. But
    noise really occurs (1 - alpha) of the time, so without the bias a noise target lands on
    f = 0 and costs 69 nats, and GD is forced to park (1 - alpha) of layer 2's attention on
    noise instead of filtering it. The /alpha is the part people drop: the renormalisation
    shrinks whatever you put in, so you have to put in more.

  * STRICTLY causal mask, triu(diagonal=0): a position can never attend to itself, so the
    circuit does not have to spend a self-veto on it. mask[0, 0] is reopened because position 0
    has no earlier key at all and a softmax over a row of -inf is nan.

  * Full batch (25,000 train sequences accumulated in chunks), 1 epoch = 1 optimizer step,
    Adam, 500 steps, loss and eval on tokens [50:100).

  * ONE setting had to change: the learning rate. Folders 17-19 use Adam lr 1.0 on the
    COMBINED W_qk, where a step moves the score linearly. Factored, both halves move at once
    and the score moves QUADRATICALLY, so lr 1.0 overshoots on the first step and never
    recovers. Measured at alpha = 0.8, 100 steps, --init eps (best test CE):

        lr 1.0    3.206   diverged; |W_qk1|max runs to 2.7e4 and the loss never comes back
        lr 0.3    1.506   still falling, slowly
        lr 0.1    1.385   <- the folder-18/19 reference is 1.38435, reached by step ~50
        lr 0.03   1.387   still falling at step 100
        lr 0.01   1.520   still falling

    So the default here is lr 0.1. Everything else -- optimizer, batching, epochs, seed, data,
    mask, readout, freezing -- is the folder-19 setting unchanged. Pass --lr 1.0 to see the
    divergence for yourself.

WHAT IS NEW -- W_qk IS FACTORED
-------------------------------
Folders 18 and 19 carry ONE parameter per layer, the combined W_qk, and score a pair by

    s(q, k) = x_q W_qk x_k^T / sqrt(output_dim).

That removes the within-head rotation gauge by construction. Here we put it back: each layer
carries TWO parameters, W_q and W_k of shape (input_dim, head_dim), and

    s(q, k) = (x_q W_q) (x_k W_k)^T / sqrt(output_dim)        i.e.   W_qk = W_q W_k^T.

Nothing else changes -- same denominator sqrt(output_dim) = sqrt(108), same mask, same loss.
The question the run asks: with the output/value circuit nailed down, does gradient descent
rediscover the hand construction's QK circuit when it is free to split it across two matrices?

Display convention (the one the PDF uses). nn.Linear stores W in (out, in) and computes
x W^T. Every matrix here is stored the way x multiplies it FROM THE LEFT -- shape (in, head) --
so rows are input dims and columns are head dims, and

    W_qk = W_q @ W_k.T                (in x in)

matches `W_qk = Wq.weight.T @ Wk.weight` in the construction code exactly.

ZERO INIT IS A DEAD SADDLE HERE -- READ THIS BEFORE PASSING --init zeros
-----------------------------------------------------------------------
Folders 17/18/19 all start from W_qk = 0 and that is fine: dL/dW_qk is nonzero there. Under the
factorisation it is NOT. With L a function of W_qk = W_q W_k^T,

    dL/dW_q = (dL/dW_qk) W_k        and        dL/dW_k = (dL/dW_qk)^T W_q,

so at W_q = W_k = 0 BOTH gradients are exactly zero. The point is a strict saddle: Adam's
update is 0/sqrt(0) -> 0, the weights never leave, and the loss curve is flat forever. This is
not a tuning problem, it is the geometry of a product parametrisation, and it is the one thing
that cannot carry over from the combined runs.

    --init eps      [DEFAULT]  W_q, W_k ~ N(0, init_scale) with init_scale = 1e-3.
                    As close to zero as a factored parametrisation can start while still
                    having a gradient: |W_qk| at init is ~1e-6 * 108 ~ 1e-4, i.e. the initial
                    attention is uniform to five decimal places, exactly as a zero init would
                    be. This is the honest stand-in for "zero init" here.
    --init zeros    the literal thing. Allowed, and instrumented: the script prints ||g|| at
                    init and, if the weights are still exactly 0 after step 1, stops the
                    threshold with stop_reason 'zero-init saddle' instead of burning 500 steps
                    on a flat curve. Use it to SHOW the saddle, not to train through it.
    --init normal   uniform(-s, s), s = fan_in^-0.5 -- the same rule train.py uses in folder 19.
    --init orth     scaled semi-orthogonal (W_q^T W_q = init_scale^2 I).

THE FACTORISATION ADDS A GAUGE -- THE SEPARATE MATRICES ARE NOT COMPARABLE ON THEIR OWN
--------------------------------------------------------------------------------------
For ANY invertible M (head_dim x head_dim),

    W_q -> W_q M,     W_k -> W_k M^-T      leaves W_q W_k^T unchanged.

So the learned W_q on its own carries no gauge-invariant information whatsoever: it can be
rotated, scaled and sheared into almost anything without touching a single logit. Only the
product W_qk is meaningful, and even that only up to the softmax row-shift gauge of
CLAUDE.md section 8.7.

That is why this script logs the cosine against the construction on the PRODUCT (gauge-fixed),
not on W_q or W_k, and why the companion notebook plots the separate matrices only after an
explicit alignment step. Plotting raw W_q against the construction's W_q is reporting the
notation, not the model.

THE CONSTRUCTION
----------------
`construction()` below builds the hand-built counting induction head from
`seperate_parameterization_weight_handconstruction.pdf` in exactly this display convention,
in both magnitude regimes:

    paper=True    the PDF's own numbers: alpha_rec = 20, P1 = 1000, P2 = 50, eta = 10,
                  gamma = -5e4, BIG = -1e9, uncapped ramp, explicit self-veto.
    paper=False   the reduced magnitudes of folder 18 (CLAUDE.md section 8.6):
                  lam = 20, K = 100, B = 100, cap = 5, self-veto dropped (the strict causal
                  mask already does it), future half zeroed (the mask discards it anyway).
                  Costs 0.004 nats and brings |W_qk|max from 1e10 down to ~1039.

It is importable -- the notebook does `from train_separate_qk import construction`.

Usage:
    python train_separate_qk.py                              # defaults below
    python train_separate_qk.py --thresholds 0.6 0.8 1.0
    python train_separate_qk.py --init zeros                 # demonstrate the saddle
    python train_separate_qk.py --lr 1.0                      # watch it diverge
    python train_separate_qk.py --runs-dir /tmp/scratch --epochs 3 --no-wandb   # smoke test
"""

import os
import sys
import glob
import json
import math
import time
import argparse
import tempfile

# /tmp is unusable on this box -- route every temp path to ~/.cache/ashu_tmp
_TMP = os.path.expanduser('~/.cache/ashu_tmp')
os.makedirs(_TMP, exist_ok=True)
for _v in ('TMPDIR', 'TMP', 'TEMP'):
    os.environ[_v] = _TMP
tempfile.tempdir = _TMP
os.environ.setdefault('MPLCONFIGDIR', _TMP)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_DATA_DIR = os.path.abspath(os.path.join(
    _SCRIPT_DIR, '..', '14. elimination final convergence confusion', 'dataset'))

LOG_FLOOR = 1e-30
SEED = 0
# The headline triple of CLAUDE.md section 8. The dataset also holds 0.4, 0.5, 0.7, 0.9 --
# pass --thresholds to use them.
THRESHOLDS = (0.6, 0.8, 1.0)

WANDB_PROJECT = 'separate_qk_detour'
WANDB_ENTITY = None
USE_WANDB = True

CFG = {
    'data_dir':  _DATA_DIR,
    'runs_dir':  os.path.join(_SCRIPT_DIR, 'runs'),

    # --- task ---
    'v_sig':       4,          # signal tokens 0..3
    'n_noise':     4,          # noise tokens 4..7
    'seq_len':     100,
    'token_start': 50,         # loss and eval on tokens [50:100)

    # --- optimisation ---
    'optimizer':    'adam',    # 'adam' | 'sgd'
    'lr':           0.05,       # NOT folder 19's 1.0 -- the factored score moves quadratically
                               # and lr 1.0 diverges. See the docstring for the probe table.
     'weight_decay': 0.0,
    'epochs':       250,       # == optimizer steps (full batch)
    'init':         'eps',     # 'eps' | 'zeros' | 'normal' | 'orth'   (W_q / W_k only)
    'init_scale':   1e-3,      # std for 'eps', scale for 'orth'
    'head_dim':     None,      # None -> output_dim (108), what the construction assumes
    'train_wov':    False,     # False -> W_ov top-left 8x8 = identity, frozen
    'blocks':       'relevant',
    'readout':      'bias',
    'b_sig':        0.01,
    'noise_floor':  1e-9,      # b_noise at alpha=1, where (1-alpha)/(n_noise*alpha) = 0

    # NO per-step snapshots. Storing the 48,384 free cells' weights AND gradients every step
    # cost ~185 MB per threshold (~555 MB for the default three), and the .npy files are
    # preallocated, so the cost landed whether or not the run finished. What survives a run:
    # model_best / model_final .pth and metrics_*.json -- and the latter already carries the
    # per-step gauge-fixed cosines in `history`, which is what the analysis actually plots.

    # --- batching / evaluation ---
    # Chunking is a MEMORY device only: each chunk's mean loss is reweighted by its share of
    # the scored tokens, so the accumulated gradient equals one 25,000-sequence backward.
    'grad_chunk':  8192,
    'eval_chunk':  8192,
    'log_every':   0,          # 0 -> every 5% of the run; set N to print every N steps
    'device':      None,       # None -> least-loaded visible GPU, else cpu (see pick_device)
}


def pick_device(spec=None):
    """Least-loaded visible GPU, or what the caller asked for.

    torch.cuda.is_available() only says a GPU EXISTS -- on this shared box all four A100s are
    usually near-saturated by other users, so defaulting to cuda:0 can queue behind a 71 GB
    job or OOM. This picks the device with the most free memory instead.
    """
    if spec:
        return spec
    if not torch.cuda.is_available():
        return 'cpu'
    free = []
    for i in range(torch.cuda.device_count()):
        try:
            f, _ = torch.cuda.mem_get_info(i)
            free.append((f, i))
        except Exception:
            free.append((0, i))
    f, i = max(free)
    print(f'device     : auto-picked cuda:{i} ({f / 2**30:.0f} GB free of '
          f'{torch.cuda.get_device_properties(i).total_memory / 2**30:.0f} GB) -- '
          f'override with --device')
    return f'cuda:{i}'


def cell_tag():
    hd = '' if CFG['head_dim'] is None else f"_hd-{CFG['head_dim']}"
    return (f"sepqk_opt-{CFG['optimizer']}_lr-{CFG['lr']}_init-{CFG['init']}"
            f"_wov-{'train' if CFG['train_wov'] else 'fixed'}"
            f"_blocks-{CFG['blocks']}{hd}"
            f"_readout-{CFG['readout']}_ep-{CFG['epochs']}")


def run_dir():
    return os.path.join(CFG['runs_dir'], cell_tag())


def metrics_path(thr):
    return os.path.join(run_dir(), f'metrics_seed{SEED}_x_{thr}.json')


# ------------------------------------------------------------------------------- readout bias

def noise_bias(alpha):
    """b_noise chosen so that, after renormalising, the noise slots hold EXACTLY (1-alpha).

    With a perfect noise filter the signal slots of f sum to 1 and the noise slots to 0, so the
    signal side of (f+b) totals S = 1 + V_sig*b_sig and the noise side n_noise*b_noise. Solving

        n_noise*b_noise / (S + n_noise*b_noise) = 1 - alpha

    puts alpha in the DENOMINATOR -- collecting the b terms is what does it. The naive
    (1-alpha)/n_noise forgets that the renormalisation shrinks whatever you put in.
    """
    if alpha >= 1.0:
        return CFG['noise_floor']          # no noise exists; just keep the slot off zero
    return ((1.0 - alpha) * (1.0 + CFG['v_sig'] * CFG['b_sig'])
            / (CFG['n_noise'] * alpha))


def readout_bias(alpha, device):
    """The 8-vector b: b_sig on the signal slots, noise_bias(alpha) on the noise slots."""
    b = torch.full((CFG['v_sig'] + CFG['n_noise'],), CFG['b_sig'], device=device)
    b[CFG['v_sig']:] = noise_bias(alpha)
    return b


# --------------------------------------------------------------------------------------- data

_DATA = {}


def load_data(split, thr):
    """Cache one threshold's splits at a time; drop the previous threshold's when it changes."""
    key = (split, thr)
    if key not in _DATA:
        if any(k[1] != thr for k in _DATA):
            _DATA.clear()
        _DATA[key] = torch.load(os.path.join(CFG['data_dir'], split, f'data_x_{thr}.pt'),
                                weights_only=False)
    return _DATA[key]


def check_dataset(thresholds):
    """Every file present, and stamped as the equiprobable dataset."""
    for thr in thresholds:
        for split in ('train', 'test'):
            p = os.path.join(CFG['data_dir'], split, f'data_x_{thr}.pt')
            if not os.path.exists(p):
                sys.exit(f'missing {p}\ngenerate it in folder 14: python data_generation.py')
        if load_data('test', thr).get('succ_split') != 'equiprobable':
            sys.exit(f'{CFG["data_dir"]}/test/data_x_{thr}.pt is not the equiprobable dataset')
    _DATA.clear()


# -------------------------------------------------------------------------------------- model

class SeparateAttentionBlock(nn.Module):
    """One causal attention layer with SEPARATE W_q / W_k and a combined, frozen W_ov.

        scores = (x W_q)(x W_k)^T / sqrt(output_dim) + causal mask
        out    = softmax(scores) (x W_ov)

    W_q and W_k have shape (input_dim, head_dim) in DISPLAY convention -- x multiplies them
    from the left -- so W_qk = W_q @ W_k.T, matching the construction PDF cell for cell.

    W_ov stays combined and frozen. It is the value/output circuit that makes f a distribution;
    splitting it would buy nothing here and would reintroduce a second gauge on the one part of
    the model this experiment is deliberately holding still.

    `concat` controls whether the residual stream is carried forward: layer 1 needs it (layer 2
    reads [x | attn_out]), layer 2 does not.
    """

    def __init__(self, input_dim, output_dim, concat, head_dim=None):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.head_dim = output_dim if head_dim is None else head_dim
        self.concat = concat
        self.W_q = nn.Parameter(torch.zeros(input_dim, self.head_dim))
        self.W_k = nn.Parameter(torch.zeros(input_dim, self.head_dim))
        self.W_ov = nn.Parameter(torch.zeros(input_dim, output_dim))

    def W_qk(self):
        """The combined circuit the softmax actually sees. The only gauge-invariant object."""
        return self.W_q @ self.W_k.T

    def forward(self, x, mask):
        # (x W_q)(x W_k)^T is algebraically x W_q W_k^T x^T but costs O(T * head_dim) instead
        # of materialising the (in x in) product every forward.
        q = x @ self.W_q
        k = x @ self.W_k
        scores = torch.bmm(q, k.transpose(1, 2)) / (self.output_dim ** 0.5)
        out = torch.bmm(F.softmax(scores + mask, dim=-1), x @ self.W_ov)
        return torch.cat([x, out], dim=-1) if self.concat else out


class SeparateQKTransformer(nn.Module):
    def __init__(self, v_sig, n_noise, seq_len, head_dim=None):
        super().__init__()
        self.V = v_sig + n_noise
        self.seq_len = seq_len
        self.d1 = self.V + seq_len                  # 108
        self.d2 = 2 * self.d1                       # 216
        self.layer1 = SeparateAttentionBlock(self.d1, self.d1, concat=True, head_dim=head_dim)
        self.layer2 = SeparateAttentionBlock(self.d2, self.d1, concat=False, head_dim=head_dim)

    def forward(self, x):
        B, T = x.shape
        tok = F.one_hot(x, num_classes=self.V).float()
        pos = F.one_hot(torch.arange(T, device=x.device),
                        num_classes=self.seq_len).float().unsqueeze(0).expand(B, T, -1)
        s = torch.cat([tok, pos], dim=-1)
        # STRICTLY causal: diagonal=0 masks j == i too, so a position can never attend to
        # itself and the circuit does not have to spend a self-veto on it.
        mask = torch.triu(torch.ones(T, T, device=x.device) * float('-inf'), diagonal=0)
        mask[0, 0] = 0.0        # position 0 has no earlier key; without this its softmax is nan
        out = self.layer2(self.layer1(s, mask), mask)
        return out[..., :self.V]                   # the predicted distribution, as produced


# ------------------------------------------------------------------- the hand construction

def construction(v_sig=4, n_noise=4, seq_len=100, paper=False,
                 lam=20.0, K=100.0, B=100.0, cap=5, drop_col4=True, as_numpy=True):
    """The hand-built counting induction head, in SEPARATE W_q / W_k display form.

    Returns a dict with the four attention matrices in (in x head) display convention, the two
    value/output circuits, and the two combined products:

        Wq1 (108,108)  Wk1 (108,108)  W_qk1 = Wq1 @ Wk1.T   (108,108)
        Wq2 (216,108)  Wk2 (216,108)  W_qk2 = Wq2 @ Wk2.T   (216,216)
        W_ov1 (108,108)  W_ov2 (216,108)

    paper=True reproduces seperate_parameterization_weight_handconstruction.pdf verbatim:

        alpha_rec = 20      layer-1 recency slope, uncapped
        P1 = 1000           layer-2 query: own signal token
        P2 = 50             layer-2 query: layer-1 predecessor    (P1/P2 = 20, the priority)
        eta = 10            layer-2 noise-filter trigger
        gamma = -5e4        layer-2 noise-key kill
        BIG = -1e9          layer-1 noise-key kill and self-veto

    paper=False is the reduced form of CLAUDE.md section 8.6 -- same circuit, magnitudes cut to
    what GD actually uses (lam, K, B, cap) at a cost of 0.004 nats:

        P1 = K, P2 = K/20, eta = 1, gamma = -B, BIG = -B, ramp capped at `cap` steps,
        no self-veto (the strict causal mask already forbids it), future half left at 0
        (softmax(s + mask) discards it every pass).

    drop_col4 removes column 108+V of W_qk2, the '+103.92' artefact of the Wq^T Wk
    factorisation. Ablating it changes the CE by exactly 0.00e+00 (CLAUDE.md section 8.6) --
    it is there only because eta rides on the same query column as the noise trigger.
    """
    V, Vs, T = v_sig + n_noise, v_sig, seq_len
    d1, d2 = V + T, 2 * (V + T)
    sd = d1 ** 0.5                     # every score is divided by sqrt(output_dim) = sqrt(108)

    if paper:
        alpha_rec, P1, P2, eta, gamma, BIG = 20.0, 1000.0, 50.0, 10.0, -5e4, -1e9
        # verbatim: uncapped ramp, explicit self-veto, and the future half left at
        # -alpha_rec*(tq - tk) > 0 rather than zeroed. nnz(Wk1) = 432 + 10,000 = 10,432, which
        # is the figure in the PDF's summary table. The causal mask discards the future half
        # every pass, which is exactly why paper=False may drop it.
        cap, self_veto, zero_future = T, True, False
    else:
        alpha_rec, P1, P2, eta, gamma, BIG = lam, K, K / 20.0, 1.0, -B, -B
        self_veto, zero_future = False, True

    # ---- layer 1: the strict previous-signal retriever -------------------------------------
    # Wq1 is a scaled identity; the whole layer-1 pattern is decided by Wk1 alone, and the
    # sqrt(D) cancels the score denominator exactly.
    Wq1 = np.eye(d1) * sd
    Wk1 = np.zeros((d1, d1))
    Wk1[Vs:V, :] = BIG                                    # noise KEY token rows: kill any noise key
    dd = np.arange(T)[None, :] - np.arange(T)[:, None]    # dd[tk, tq] = tq - tk
    ramp = np.where(dd > 0, -alpha_rec * np.minimum(dd, cap), 0.0)
    if not zero_future:
        ramp = np.where(dd < 0, -alpha_rec * dd, ramp)    # masked anyway
    if self_veto:
        ramp[np.arange(T), np.arange(T)] = BIG            # no self-attention
    Wk1[V:d1, V:d1] = ramp

    Wv1 = np.zeros((d1, d1)); Wv1[:V, :V] = np.eye(V)     # only the token one-hot passes
    W_ov1 = Wv1                                           # Wo1 = I

    # ---- layer 2: the counting induction head ----------------------------------------------
    Wq2 = np.zeros((d2, d1))
    Wq2[:Vs, :Vs] = np.eye(Vs) * sd * P1                  # P1: own signal token
    Wq2[d1:d1 + Vs, :Vs] = np.eye(Vs) * sd * P2           # P2: layer-1 predecessor
    Wq2[:V, Vs] = sd * eta                                # noise-filter trigger row

    Wk2 = np.zeros((d2, d1))
    Wk2[d1:d1 + V, :V] = np.eye(V)                        # key = the key's own predecessor
    Wk2[Vs:V, Vs] = gamma                                 # noise key kill

    Wv2 = np.zeros((d2, d1)); Wv2[:V, :V] = np.eye(V)     # copies the ORIGINAL token one-hot,
    W_ov2 = Wv2                                           # which is what turns an average into
                                                          # a frequency. Wo2 = I.

    W_qk1 = Wq1 @ Wk1.T
    W_qk2 = Wq2 @ Wk2.T
    if drop_col4:
        W_qk2[:V, d1 + Vs] = 0.0                          # the inert factorisation artefact

    out = {'Wq1': Wq1, 'Wk1': Wk1, 'W_qk1': W_qk1, 'W_ov1': W_ov1,
           'Wq2': Wq2, 'Wk2': Wk2, 'W_qk2': W_qk2, 'W_ov2': W_ov2,
           'params': {'paper': paper, 'alpha_rec': alpha_rec, 'P1': P1, 'P2': P2, 'eta': eta,
                      'gamma': gamma, 'BIG': BIG, 'cap': cap, 'self_veto': self_veto,
                      'drop_col4': drop_col4, 'sqrt_D': sd}}
    if not as_numpy:
        out = {k: (torch.as_tensor(v, dtype=torch.float32) if isinstance(v, np.ndarray) else v)
               for k, v in out.items()}
    return out


# ------------------------------------------------------------------------------ gauge fixing

def key_groups(layer, v_sig=4, n_noise=4, seq_len=100):
    """The COMPLETE key groups of each layer, and which query rows are live.

    softmax(s + c*1) = softmax(s), so adding one constant to a query row across a complete
    group of keys changes nothing. A key contributes exactly one entry from each one-hot block
    of the residual stream, so each block is one such group:

        layer 1   keys: token 0:8 (sums to 1), position 8:108 (sums to 1)
        layer 2   keys: token 0:8, position 8:108, layer-1 output 108:116 (sums to ~0.998)
                        -- dims 116:216 are identically zero, so they are not a group at all

    Query rows behave the same way: rows 116:216 of layer 2 multiply a coordinate of x that is
    always 0, so they are dead and are excluded rather than compared against noise.

    Returns (live_rows, [slice, ...]).
    """
    V, T = v_sig + n_noise, seq_len
    d1 = V + T
    if layer == 1:
        return slice(0, d1), [slice(0, V), slice(V, d1)]
    return slice(0, d1 + V), [slice(0, V), slice(V, d1), slice(d1, d1 + V)]


def gauge_fix(W, layer, **kw):
    """Subtract, from each query row, its mean over each complete key group separately.

    This is the projection onto the quotient by the exact softmax gauge. Any metric that moves
    under a transformation which provably cannot change the model is reporting notation, not
    the model -- CLAUDE.md section 8.7. Raw cosine between two writings of the SAME circuit
    came out as 0.7169 / 0.8631 / 0.8631 / 0.8794; gauge-fixed it is 0.8646 every time.
    """
    rows, groups = key_groups(layer, **kw)
    W = np.asarray(W, dtype=np.float64)[rows]
    out = np.zeros_like(W)
    for g in groups:
        blk = W[:, g]
        out[:, g] = blk - blk.mean(axis=1, keepdims=True)
    return out


def gauge_cos(A, Bm, layer, **kw):
    """Gauge-fixed cosine between two W_qk matrices of the same layer."""
    a, b = gauge_fix(A, layer, **kw).ravel(), gauge_fix(Bm, layer, **kw).ravel()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 0 and nb > 0 else 0.0


# ------------------------------------------------------------------------------------- masks

def build_masks(model):
    """Boolean masks marking the only entries allowed to be nonzero. Everything else stays 0.

    'full'      every entry of W_q and W_k is free.
    'relevant'  the same, minus layer 2's input rows 116:216. Layer 1's output lives entirely
                in its first V coordinates (W_ov1 is nonzero only on [0:V, 0:V]), so those 100
                coordinates of layer 2's input are identically zero: their rows of W_q2 and
                W_k2 have identically zero gradient and can never affect a score. Dropping them
                is provably free and cuts the snapshot size by ~45%.

    Note this is NOT the block sparsity of folders 18/19. Those masks live on the COMBINED
    W_qk, where a block is a (query dim, key dim) pair; a factored parametrisation has no such
    coordinates -- a single row of W_q touches every key at once. Imposing the folder-18 block
    pattern here would be a different, and much stronger, restriction than it was there.
    """
    V, d1, d2 = model.V, model.d1, model.d2
    m = {n: torch.ones_like(p, dtype=torch.bool) for n, p in model.named_parameters()}

    if CFG['blocks'] == 'relevant':
        qk2 = torch.zeros(d2, model.layer2.head_dim, dtype=torch.bool)
        qk2[0:d1 + V, :] = True
        m['layer2.W_q'] = qk2.clone()
        m['layer2.W_k'] = qk2.clone()
    elif CFG['blocks'] != 'full':
        sys.exit(f"blocks must be 'relevant' or 'full', got {CFG['blocks']!r}")

    # W_ov is the frozen selector whatever --blocks says
    ov1 = torch.zeros(d1, d1, dtype=torch.bool); ov1[0:V, 0:V] = True
    ov2 = torch.zeros(d2, d1, dtype=torch.bool); ov2[0:V, 0:V] = True
    m['layer1.W_ov'] = ov1
    m['layer2.W_ov'] = ov2
    return m


def to_device_masks(d, device):
    """Move every boolean mask onto the device ONCE -- reapply_masks() runs every step."""
    return {k: v.to(device) for k, v in d.items()}


def install(model, masks, device):
    """Initialise inside the masks, zero everything outside, pin gradients to the masks."""
    V = model.V
    params = dict(model.named_parameters())
    torch.manual_seed(SEED)
    scale = CFG['init_scale']

    with torch.no_grad():
        for name, p in params.items():
            p.zero_()
            mask = masks[name].to(device)
            if name.endswith(('W_q', 'W_k')):
                if CFG['init'] == 'zeros':
                    pass                                   # a strict saddle -- see the docstring
                elif CFG['init'] == 'eps':
                    p.copy_(torch.randn_like(p) * scale * mask)
                elif CFG['init'] == 'normal':
                    s = p.shape[0] ** -0.5                 # uniform(-s, s), fan_in ** -0.5
                    p.copy_(torch.empty_like(p).uniform_(-s, s) * mask)
                elif CFG['init'] == 'orth':
                    w = torch.empty(p.shape[1], p.shape[0], device=device)
                    nn.init.orthogonal_(w)
                    p.copy_(w.T * scale * mask)
                else:
                    sys.exit(f"init must be one of zeros/eps/normal/orth, got {CFG['init']!r}")
            else:
                # W_ov: identity on the 8x8 selector, trained or not. Starting it at zero would
                # make f == 0 everywhere and kill every gradient.
                p[0:V, 0:V] = torch.eye(V, device=device)

    trainable, names, n_free = [], [], 0
    for name, p in params.items():
        train_this = name.endswith(('W_q', 'W_k')) or CFG['train_wov']
        p.requires_grad_(train_this)
        if not train_this:
            continue
        mask = masks[name].to(device)
        p.register_hook(lambda g, mask=mask: g * mask)
        trainable.append(p)
        names.append(name)
        n_free += int(mask.sum())
    return trainable, n_free, names


@torch.no_grad()
def reapply_masks(model, masks):
    """Belt and braces: re-zero outside the masks after each step. Masks already on device."""
    for name, p in model.named_parameters():
        p.mul_(masks[name])


# --------------------------------------------------------------------------------------- loss

def ce_loss(p, y, b=None):
    """Cross entropy on p, the V-vector layer 2 puts out.

    readout 'bias':  p <- (p + b) / sum(p + b). No log, no softmax, no unembedding -- the old
    log->softmax pair was exactly this division, since softmax(log x) = x / sum(x).

    f >= 0 and b > 0  =>  p strictly in (0,1)  =>  loss >= 0 and LOG_FLOOR never binds. That
    chain needs W_ov frozen to the identity; with train_wov the loss is unbounded below.
    """
    if b is not None:
        p = p + b
    s = p.sum(-1, keepdim=True)
    p = p / torch.where(s.abs() < LOG_FLOOR, torch.ones_like(s), s)
    q = p.gather(-1, y.unsqueeze(-1)).squeeze(-1)
    return -torch.log(q.clamp_min(LOG_FLOOR)).mean()


@torch.no_grad()
def eval_ce(model, x, y, device, b=None):
    """Mean CE over the scored window, in chunks. No gradient."""
    model.eval()
    ts, te = CFG['token_start'], CFG['seq_len']
    tot = cnt = 0.0
    for i in range(0, x.shape[0], CFG['eval_chunk']):
        xb = x[i:i + CFG['eval_chunk']].to(device)
        yb = y[i:i + CFG['eval_chunk'], ts:te].to(device)
        tot += float(ce_loss(model(xb)[:, ts:te], yb, b)) * yb.numel()
        cnt += yb.numel()
    return tot / cnt


def full_batch_grad(model, x, y, b=None):
    """Exact gradient of the mean loss over ALL of x, accumulated in chunks.

    Each chunk's mean loss is reweighted by its share of the scored tokens, so the accumulated
    .grad equals what a single 25,000-sequence backward would have produced.
    """
    model.train()
    for p in model.parameters():
        p.grad = None
    ts, te = CFG['token_start'], CFG['seq_len']
    n_tok = x.shape[0] * (te - ts)
    tot = 0.0
    for i in range(0, x.shape[0], CFG['grad_chunk']):
        xb, yb = x[i:i + CFG['grad_chunk']], y[i:i + CFG['grad_chunk'], ts:te]
        loss = ce_loss(model(xb)[:, ts:te], yb, b)
        (loss * (yb.numel() / n_tok)).backward()
        tot += float(loss) * yb.numel()
    return tot / n_tok


@torch.no_grad()
def qk_report(model, ref):
    """Norms of the factored and combined circuits, and gauge-fixed cosine vs the construction.

    The cosine is taken on the PRODUCT only. W_q and W_k individually carry the GL(head_dim)
    gauge W_q -> W_q M, W_k -> W_k M^-T, so their own cosine against the construction is not a
    property of the model at all.
    """
    out = {}
    for i, (lay, blk) in enumerate(((1, model.layer1), (2, model.layer2)), start=1):
        qk = blk.W_qk().detach().float().cpu().numpy()
        out[f'L{lay}_cos'] = gauge_cos(qk, ref[f'W_qk{lay}'], lay)
        out[f'L{lay}_qk_absmax'] = float(np.abs(qk).max())
        out[f'L{lay}_qk_fro'] = float(np.linalg.norm(qk))
        # gauge-fixed norm: the scale that is actually comparable with the construction's,
        # and the one the analysis plots against step. Recorded in `history` so the evolution
        # can be reconstructed without per-step weight snapshots.
        out[f'L{lay}_qk_fro_gf'] = float(np.linalg.norm(gauge_fix(qk, lay)))
        out[f'L{lay}_q_fro'] = float(blk.W_q.detach().norm())
        out[f'L{lay}_k_fro'] = float(blk.W_k.detach().norm())
    return out


# ----------------------------------------------------------------------------------- training

def train_one(thr, device, ref):
    wb = None
    if USE_WANDB:
        try:
            import wandb
            wb = wandb.init(project=WANDB_PROJECT, entity=WANDB_ENTITY,
                            name=f'{cell_tag()}_x_{thr}', group=cell_tag(),
                            config={**CFG, 'threshold': thr, 'seed': SEED,
                                    'parametrisation': 'separate W_q / W_k'}, reinit=True)
        except Exception as e:                       # a dead wandb must not kill a run
            print(f'    [wandb off: {e}]', flush=True)

    model = SeparateQKTransformer(CFG['v_sig'], CFG['n_noise'], CFG['seq_len'],
                                  CFG['head_dim']).to(device)
    masks = build_masks(model)
    params, n_free, tnames = install(model, masks, device)
    dmasks = to_device_masks(masks, device)        # moved once, used every step

    tr = load_data('train', thr)
    te = load_data('test', thr)
    tr_x, tr_y = tr['x'].to(device), tr['y'].to(device)
    te_x, te_y = te['x'].to(device), te['y'].to(device)
    b = readout_bias(thr, device) if CFG['readout'] == 'bias' else None

    if CFG['optimizer'] == 'adam':
        opt = torch.optim.Adam(params, lr=CFG['lr'], weight_decay=CFG['weight_decay'])
    elif CFG['optimizer'] == 'sgd':
        opt = torch.optim.SGD(params, lr=CFG['lr'], weight_decay=CFG['weight_decay'])
    else:
        sys.exit(f"optimizer must be 'adam' or 'sgd', got {CFG['optimizer']!r}")

    tr0 = eval_ce(model, tr_x, tr_y, device, b)
    te0 = eval_ce(model, te_x, te_y, device, b)
    history = {'step': [0], 'train_loss': [tr0], 'test_loss': [te0], 'train_loss_pre': [tr0],
               'L1_cos': [], 'L2_cos': [], 'L1_fro_gf': [], 'L2_fro_gf': []}
    best_train, best_test, best_state = tr0, te0, None

    n_all = sum(p.numel() for p in model.parameters())
    print(f'  alpha={thr}  free entries {n_free:,} of {n_all:,}  '
          f'(W_q + W_k only; W_ov frozen)')

    # The saddle check, done once and reported rather than assumed.
    _ = full_batch_grad(model, tr_x, tr_y, b)
    g0 = {n: float(p.grad.norm()) for n, p in model.named_parameters() if p.grad is not None}
    print('    grad norms at init: ' + '  '.join(f'{n}={v:.3e}' for n, v in g0.items()))
    if CFG['init'] == 'zeros' and max(g0.values(), default=0.0) == 0.0:
        print('    [!] every gradient is EXACTLY zero -- W_q = W_k = 0 is a strict saddle of '
              'the factored\n        parametrisation (dL/dW_q = (dL/dW_qk) W_k = 0). Nothing '
              'will move. Use --init eps.', flush=True)

    rep0 = qk_report(model, ref)
    print(f'    init: train {tr0:.5f}  test {te0:.5f}   '
          f'gauge-fixed cos vs construction  L1 {rep0["L1_cos"]:+.4f}  '
          f'L2 {rep0["L2_cos"]:+.4f}', flush=True)

    t0 = time.time()
    stop_reason = None
    interrupted = False
    try:
        for ep in range(1, CFG['epochs'] + 1):
            tr_pre = full_batch_grad(model, tr_x, tr_y, b)     # loss at the CURRENT weights
            opt.step()
            reapply_masks(model, dmasks)

            # both measured AFTER the step -- history[k] is one set of weights throughout
            tr_loss = eval_ce(model, tr_x, tr_y, device, b)
            te_loss = eval_ce(model, te_x, te_y, device, b)
            rep = qk_report(model, ref)

            if ep == 1 and CFG['init'] == 'zeros' and rep['L1_qk_fro'] == 0.0 \
                    and rep['L2_qk_fro'] == 0.0:
                stop_reason = ('zero-init saddle: after one step W_q and W_k are still exactly '
                               '0. dL/dW_q = (dL/dW_qk) W_k vanishes when W_k = 0, so the '
                               'factored parametrisation cannot leave the origin. Use '
                               '--init eps.')
                print(f'    [STOP] {stop_reason}', flush=True)
                break

            # A negative or non-finite CE is a correctness violation, not a bad hyperparameter:
            # it means the output stopped being a distribution (p[target] > 1, so -log p < 0).
            if not math.isfinite(tr_loss) or tr_loss < 0:
                stop_reason = (f'loss not a valid CE at step {ep}: train={tr_loss:.6g}. '
                               f'p[target] > 1 or non-finite -> the output is not a '
                               f'distribution (expected when train_wov=True lets f go negative)')
                print(f'    [STOP] {stop_reason}', flush=True)
                break

            history['step'].append(ep)
            history['train_loss'].append(tr_loss)
            history['test_loss'].append(te_loss)
            history['train_loss_pre'].append(tr_pre)
            history['L1_cos'].append(rep['L1_cos'])
            history['L2_cos'].append(rep['L2_cos'])
            history['L1_fro_gf'].append(rep['L1_qk_fro_gf'])
            history['L2_fro_gf'].append(rep['L2_qk_fro_gf'])
            best_train = min(best_train, tr_loss)
            if te_loss < best_test:
                best_test = te_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            if wb is not None:
                wb.log({'train_loss': tr_loss, 'test_loss': te_loss, 'train_loss_pre': tr_pre,
                        'best_train': best_train, 'best_test': best_test, **rep}, step=ep)
            every = CFG['log_every'] or max(1, CFG['epochs'] // 20)   # default: every 5%
            if ep % every == 0 or ep == CFG['epochs'] or ep == 1:
                print(f'    step {ep:>5}/{CFG["epochs"]}  train {tr_loss:.5f}  '
                      f'test {te_loss:.5f}  best_test {best_test:.5f}  '
                      f'cos L1 {rep["L1_cos"]:+.4f} L2 {rep["L2_cos"]:+.4f}  '
                      f'|W_qk|max {rep["L1_qk_absmax"]:.3g}/{rep["L2_qk_absmax"]:.3g}',
                      flush=True)
    except KeyboardInterrupt:
        interrupted = True
        print('\n    [interrupted -- saving what exists and moving to the next threshold]',
              flush=True)

    d = run_dir()
    os.makedirs(d, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(d, f'model_final_seed{SEED}_x_{thr}.pth'))
    if best_state is not None:
        torch.save(best_state, os.path.join(d, f'model_best_seed{SEED}_x_{thr}.pth'))

    final_rep = qk_report(model, ref)
    out = {'cell': cell_tag(), 'threshold': thr, 'seed': SEED,
           'config': {**CFG, 'trainable_entries': n_free,
                      'parametrisation': 'separate W_q / W_k',
                      'head_dim': model.layer1.head_dim,
                      'b_noise': noise_bias(thr) if CFG['readout'] == 'bias' else None},
           'grad_norm_at_init': g0,
           'history': history, 'best_train': best_train, 'best_test': best_test,
           'final_train': history['train_loss'][-1], 'final_test': history['test_loss'][-1],
           'final_qk': final_rep,
           'steps_run': history['step'][-1], 'seconds': time.time() - t0,
           'stopped_early': stop_reason is not None or interrupted,
           'stop_reason': stop_reason or ('interrupted' if interrupted else None)}
    with open(metrics_path(thr), 'w') as fh:
        json.dump(out, fh, indent=2)
    if wb is not None:
        wb.summary.update({k: out[k] for k in ('best_train', 'best_test', 'final_train',
                                               'final_test', 'steps_run')})
        wb.summary.update(final_rep)
        wb.finish()
    print(f'    done in {(time.time()-t0)/60:.1f}m  best_train {best_train:.5f}  '
          f'best_test {best_test:.5f}  cos L1 {final_rep["L1_cos"]:+.4f} '
          f'L2 {final_rep["L2_cos"]:+.4f}  -> {metrics_path(thr)}', flush=True)
    return out


def rebuild_summary():
    """Re-glob every metrics file on disk, so retraining a subset never shrinks the summary."""
    rows = []
    for p in sorted(glob.glob(os.path.join(CFG['runs_dir'], '*', 'metrics_seed*_x_*.json'))):
        try:
            rows.append(json.load(open(p)))
        except Exception:
            pass
    os.makedirs(CFG['runs_dir'], exist_ok=True)
    with open(os.path.join(CFG['runs_dir'], 'summary.json'), 'w') as fh:
        json.dump(rows, fh, indent=2)
    return len(rows)


def main():
    global USE_WANDB, THRESHOLDS
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--optimizer', choices=('adam', 'sgd'), default=CFG['optimizer'])
    ap.add_argument('--lr', type=float, default=CFG['lr'])
    ap.add_argument('--epochs', type=int, default=CFG['epochs'])
    ap.add_argument('--init', choices=('eps', 'zeros', 'normal', 'orth'), default=CFG['init'],
                    help="'eps' = zeros + N(0, init_scale); 'zeros' is a strict saddle here")
    ap.add_argument('--init-scale', type=float, default=CFG['init_scale'])
    ap.add_argument('--head-dim', type=int, default=None,
                    help='inner dimension of W_q / W_k (default 108, what the construction uses)')
    ap.add_argument('--blocks', choices=('relevant', 'full'), default=CFG['blocks'],
                    help="'relevant' drops layer 2's dead input rows 116:216; 'full' keeps them")
    ap.add_argument('--readout', choices=('none', 'bias'), default=CFG['readout'])
    ap.add_argument('--train-wov', action='store_true',
                    help='UNFREEZE W_ov -- the loss stops being a CE; see the docstring')
    ap.add_argument('--no-wandb', action='store_true')
    ap.add_argument('--thresholds', type=float, nargs='+', default=list(THRESHOLDS))
    ap.add_argument('--runs-dir', default=None,
                    help='where to write runs/ -- point at a scratch dir for smoke tests')
    ap.add_argument('--device', default=None,
                    help='cuda:N or cpu; default picks the GPU with most free memory')
    ap.add_argument('--log-every', type=int, default=CFG['log_every'],
                    help='print every N steps (default 0 = every 5%% of the run)')
    ap.add_argument('--grad-chunk', type=int, default=CFG['grad_chunk'],
                    help='sequences per forward/backward chunk -- memory only')
    a = ap.parse_args()

    USE_WANDB = not a.no_wandb
    THRESHOLDS = tuple(a.thresholds)
    CFG.update({'optimizer': a.optimizer, 'lr': a.lr, 'epochs': a.epochs, 'init': a.init,
                'init_scale': a.init_scale, 'head_dim': a.head_dim,
                'blocks': a.blocks, 'readout': a.readout, 'train_wov': a.train_wov,
                'device': pick_device(a.device), 'log_every': a.log_every,
                'grad_chunk': a.grad_chunk, 'eval_chunk': a.grad_chunk})
    if a.runs_dir:
        CFG['runs_dir'] = os.path.abspath(a.runs_dir)

    check_dataset(THRESHOLDS)
    device = torch.device(CFG['device'])

    probe = SeparateQKTransformer(CFG['v_sig'], CFG['n_noise'], CFG['seq_len'], CFG['head_dim'])
    pm = build_masks(probe)
    _, n_free, pnames = install(probe, pm, torch.device('cpu'))
    del probe

    ref = construction(CFG['v_sig'], CFG['n_noise'], CFG['seq_len'], paper=False)

    print(f'cell       : {cell_tag()}')
    print(f'device     : {device}')
    print(f'thresholds : {list(THRESHOLDS)}')
    print(f'params     : SEPARATE W_q / W_k per layer  (W_qk = W_q @ W_k.T), '
          f'W_ov frozen to the 8x8 identity')
    print(f'free cells : {n_free:,}   over {pnames}')
    print(f'init       : {CFG["init"]}'
          + (f' (scale {CFG["init_scale"]:g})' if CFG['init'] in ('eps', 'orth') else '')
          + ('   [!] a strict saddle -- every gradient is exactly 0 there; the run will stop '
             'after step 1' if CFG['init'] == 'zeros' else ''))
    print(f'batching   : FULL BATCH in chunks of {CFG["grad_chunk"]} -> 1 epoch = 1 step')
    print(f'reference  : reduced hand construction '
          f'(lam=20, K=100, B=100, cap=5) -- gauge-fixed cosine logged every step')
    print(f'on disk    : model_best / model_final .pth and metrics_*.json only '
          f'(~1 MB per threshold).\n'
          f'             NO per-step weight or gradient snapshots -- at {n_free:,} free cells '
          f'those cost\n             ~185 MB per threshold. The per-step cosines live in '
          f'metrics_*.json `history`.')
    print()

    for thr in THRESHOLDS:
        train_one(thr, device, ref)
    n = rebuild_summary()
    print(f'\nsummary rebuilt from {n} metrics files -> '
          f'{os.path.join(CFG["runs_dir"], "summary.json")}')


if __name__ == '__main__':
    main()
