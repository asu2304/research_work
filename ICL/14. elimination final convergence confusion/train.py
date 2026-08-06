"""
COMBINED vs SEPARATE, re-run with a frozen readout that is ACTUALLY a log.

Why this run exists
-------------------
Every run in folders 12 and 14 up to now was trained with a broken frozen readout. The old
`fit_log_core` warm-started from a closed-form lstsq solution (~3e-2 MSE) and then ran 2000
Adam steps at lr=1e-2 on top of it. Those steps DIVERGED: the installed core had MSE 4.0e+02
and max|err| 114.7 nats, was non-monotone, and implied a "log" running from +9.833 at f=0 down
to -111.006 at f=40. It was not a log; it was noise with the right tensor shapes. Since the
identical broken core went into every run, comparisons *within* those projects stayed
apples-to-apples, but nothing mechanistic could be read off them.

This script fixes the readout and re-runs the core comparison from scratch.

What the readout is supposed to do
----------------------------------
The FFN block is RESIDUAL:  s = s + ff2(GELU(ff1(s))).  Layer-2 attention deposits the
successor statistic f_v at residual coordinate `in2 + v`. For that coordinate to END UP holding
log(f_v + delta), the FFN must emit

    ff2_out[in2+v]  =  log(f_v + delta) - f_v          <-- the '- f' is why it is a residual

so that  f + (log(f+delta) - f) = log(f+delta).  `fc_out` then reads that coordinate and the
model's own softmax exponentiates it, giving  p(v) proportional to (f_v + delta): the softmax
UNDOES the log and returns the counts. Without the log, softmax would return e^{f_v}, i.e.
exponential in the count -- that distortion is precisely what the readout exists to cancel.

The new fit (see fit_log_core)
------------------------------
  * knots GEOMETRIC in (f + delta), not uniform: log has all its curvature at small f, so
    uniform knots waste the basis on the flat part (uniform bottoms out at ~3.4e-2 max error,
    geometric reaches ~4e-4).
  * gain_j = cgain / (local knot spacing), so each GELU ramp turns on over its own local scale.
  * NO gradient refinement. With knots and gains fixed the problem is LINEAR in the output
    weights, so lstsq is already the global optimum -- there is nothing left for Adam to do
    except leave it. (`freeze_fit_steps` is gone.)
  * delta = 0.01, not 1.0. delta is the smoothing that survives into the prediction
    (p ~ f + delta). With f in the frequency regime, delta=1.0 would swamp it and make the
    output near-uniform.
  * fitted over f in [0, 40] so the readout stays a log whichever scale GD picks for W_ov.
    Measured max|err| in float32: 4.1e-4 nats on [0,1], 2.1e-3 at the far end of [20,40].
    A global scale g on the attention output is harmless by construction:
    log(g f + delta) = log g + log(f + delta/g), and the constant log g cancels in the
    softmax -- it only rescales the effective smoothing. What is NOT harmless is leaving the
    fitted range, so every run records the observed range of those coordinates
    (`readout_range_*` in the metrics json and on wandb) to make that visible instead of silent.

Expected behaviour of that diagnostic
-------------------------------------
At init W_ov is a random Gaussian, so the layer-2 output is symmetric about zero and ~50% of
the values entering the readout are NEGATIVE, where a log does not exist. This is expected, not
a defect. Below f=0 every GELU in the basis is off, so the core emits ~c and the coordinate
becomes f + c -- a monotone near-identity passthrough. That keeps the gradient w.r.t. W_ov
alive so GD can push the values positive, which a hard clamp to log(delta) would not. The ideal
induction circuit has W_ov copying token one-hots, giving a convex combination in [0,1], so a
converged model SHOULD end up inside the calibrated range. `readout_range_*` is how you check
whether it actually did; a run that ends with a large frac_negative has not learned a
count-like layer 2 and its readout is not acting as a log.

This run's grid
---------------
    optimizer = adam    lr = 0.003    batch_size = 256    epochs = 500
    init      = 'normal' EVERYWHERE, unembedding included (see apply_init / _init_fc_out)
    archs     = separate AND combined        seeds = [0]
    thresholds = 0.4, 0.6, 0.8, 1.0          -> 8 runs

Early stopping is OFF: the full 500-epoch trajectory is the evidence.

Data: read IN PLACE from folder 12's dataset (25k train / 10k test per threshold). No copy.
Logging: wandb AND local json/checkpoints. Local files are the source of truth.
Loss curves START AT THE RANDOM INIT: epoch 0 = train/test CE before any update.
Resume-safe: runs whose metrics json already exists are skipped (SKIP_DONE). Ctrl+C during a
run saves its best-so-far and advances to the next run; Ctrl+C again within ~1s exits.

Run in tmux:
    python -u train.py
"""

import glob
import json
import math
import os
import time
import tempfile

# This box's system temp dirs are unusable -> route temp to $HOME.
_TMP = os.path.join(os.path.expanduser('~'), '.cache', 'ashu_tmp')
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
_ICL_DIR    = os.path.dirname(_SCRIPT_DIR)
_DATA_DIR   = os.path.join(
    _ICL_DIR, '12. training level exhaustive experiment to confirm finding', 'dataset')

# ─── Config ───────────────────────────────────────────────────────────────────
BASE = {
    'data_dir':  _DATA_DIR,                          # folder 12's 25k train / 10k test
    'runs_dir':  os.path.join(_SCRIPT_DIR, 'runs'),
    'device':    'cuda:1',

    'v_sig':       4,
    'n_noise':     4,
    'seq_len':     100,
    'token_start': 50,

    'bias': True,

    # ── the frozen log readout ────────────────────────────────────────────────
    # These four define the function ff1/GELU/ff2 is pinned to. See fit_log_core: the core
    # represents  g(f) = log(f + delta) - f  on [0, fmax], installed once and frozen.
    'freeze_readout':   True,
    'freeze_delta':     0.01,      # the smoothing that survives into the prediction
                                   # (softmax(log(f+delta)) ~ f + delta). WAS 1.0, which swamps
                                   # a frequency-scale f and flattens the output.
    'freeze_hidden_k':  128,       # GELU units per signal state (v_sig * k = ff1's width)
    'freeze_fit_fmax':  40.0,      # fit range; geometric knots keep small f accurate anyway
    'freeze_fit_cgain': 32.0,      # gain_j = cgain / (local knot spacing); grid-searched
    'freeze_fit_tol':   5e-3,      # HARD CHECK: refuse to train if max|err| exceeds this.
                                   # The old core would have failed this at 114.7.

    'train_params': ['attn', 'fc_out'],
    'weight_decay': 0.0,
    'clip_grad':    False,
    'grad_clip':    0.0,
    'eval_batch':   256,

    # ── the pinned cell for this experiment ───────────────────────────────────
    'optimizer':  'adam',
    'lr':         0.003,
    'batch_size': 256,
    'init':       'normal',        # N(0, 1/sqrt(fan_in)) on each arch's own matrices AND on the
                                   # unembedding (see init_fc_out below).
                                   # 'default' (uniform) and 'matched' also available.
    'init_fc_out': True,           # NEW: also initialize fc_out (weight + bias) explicitly, from
                                   # a DEDICATED generator, so both architectures start from the
                                   # identical readout. Previously fc_out kept nn.Linear's default
                                   # drawn from the GLOBAL rng, whose position depends on how many
                                   # tensors the architecture had already allocated -- so combined
                                   # and separate silently got different unembeddings.

    'epochs':     2000,
    'early_stop': False,           # DISABLED: keep the whole trajectory (see docstring)
    'patience':   50,              # only used when early_stop=True
}

# The comparison this experiment is about. 'separate' first so the reference arm exists early if
# the run is interrupted. ('qk_comb_ov_sep' / 'qk_sep_ov_comb' remain available in _ATTN_BLOCKS.)
ARCHS      = ['separate', 'combined']
SEEDS      = [0]
THRESHOLDS = [0.4, 0.6, 0.8, 1.0]

SKIP_DONE     = True
USE_WANDB     = True
RUN_NAME      = 'training with large number of epochs with smaller learning rate'
WANDB_PROJECT = 'long-train-small-lr'
WANDB_ENTITY  = None         # None -> default entity (set WANDB_ENTITY env var to override)
# ─────────────────────────────────────────────────────────────────────────────


# ─── LR sweep ────────────────────────────────────────────────────────────────
# When LR_SWEEP is non-empty each learning rate gets its OWN cell directory (the metrics
# filename carries only arch/seed/threshold, so two lrs in one directory would collide).
# LR_SWEEP = None -> the single BASE['lr'], which is what this experiment wants.
LR_SWEEP  = None
CELL_PREFIX = 'logfix'         # groups this experiment's cells together on disk

def cell_tag_for(lr):
    return (f"{CELL_PREFIX}_opt-{BASE['optimizer']}_lr-{lr}_b-{BASE['batch_size']}"
            f"_init-{BASE['init']}_ep-{BASE['epochs']}")

LRS = LR_SWEEP if LR_SWEEP else [BASE['lr']]


# ─── MODE ────────────────────────────────────────────────────────────────────
#   'sweep'     : the high-lr sweep configured above (combined, lr in LR_SWEEP)
#   'factorize' : TRANSPLANT experiment -- take finished COMBINED solutions, factorize each
#                 W_qk -> Wq^T Wk and W_ov -> Wv^T Wo^T into balanced factors, load them into
#                 a SEPARATE model (same function, more coordinates), and keep training.
#                 Tests whether combined's solution is a genuine minimum or only a minimum
#                 in combined's coordinates: if separate escapes and improves from the very
#                 same point, the deficit is the parametrization, not the loss surface.
MODE = 'sweep'

FACTORIZE = {
    'source_cell': 'opt-adam_lr-0.006_b-128_init-matched_ep-1000',
    'source_arch': 'combined',
    'target_arch': 'separate',
    'cell':        'factorized-from-combined_ep-100',   # its own directory + run names
    'epochs':      100,                                  # "100 more epochs from that point"
    'lr':          0.006,                                # same lr the source runs used
    'project':     'ICL-1000ep-matched-init',            # land in the SAME wandb project
}


# ─── Model (identical to folder 12's train.py) ────────────────────────────────
class CombinedAttentionBlock(nn.Module):
    """QK/OV combined parametrization: scores = x @ W_qk @ x^T ; out = softmax @ (x @ W_ov)."""
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.output_dim = output_dim
        self.W_qk = nn.Parameter(torch.empty(input_dim, input_dim))
        self.W_ov = nn.Parameter(torch.empty(input_dim, output_dim))
        # nn.Linear-default method (U(+-1/sqrt(fan_in))) -- same as the separate block's
        # Linears. apply_init(scheme) re-initializes before training anyway.
        s = input_dim ** -0.5
        nn.init.uniform_(self.W_qk, -s, s)
        nn.init.uniform_(self.W_ov, -s, s)

    def forward(self, x, mask=None):
        scores = torch.bmm(x @ self.W_qk, x.transpose(1, 2)) / (self.output_dim ** 0.5)
        if mask is not None:
            scores = scores + mask
        attn_out = torch.bmm(F.softmax(scores, dim=-1), x @ self.W_ov)
        return torch.cat([x, attn_out], dim=-1), scores


class SeparateAttentionBlock(nn.Module):
    """Separate Wq,Wk,Wv,Wo parametrization -- keeps the within-head rotation gauge."""
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.output_dim = output_dim
        self.W_q = nn.Linear(input_dim, output_dim, bias=False)
        self.W_k = nn.Linear(input_dim, output_dim, bias=False)
        self.W_v = nn.Linear(input_dim, output_dim, bias=False)
        self.W_o = nn.Linear(output_dim, output_dim, bias=False)

    def forward(self, x, mask=None):
        Q, K, V_ = self.W_q(x), self.W_k(x), self.W_v(x)
        scores = torch.bmm(Q, K.transpose(1, 2)) / (self.output_dim ** 0.5)
        if mask is not None:
            scores = scores + mask
        attn_out = self.W_o(torch.bmm(F.softmax(scores, dim=-1), V_))
        return torch.cat([x, attn_out], dim=-1), scores


class QKCombinedOVSeparateBlock(nn.Module):
    """HYBRID: QK collapsed to a single W_qk, but V/O left factorized as Wv, Wo.
    Isolates the QK circuit -- if the separate-vs-combined effect is driven by the OV
    factorization alone, this arch should behave like fully-combined."""
    
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.output_dim = output_dim
        self.W_qk = nn.Parameter(torch.empty(input_dim, input_dim))
        s = input_dim ** -0.5
        nn.init.uniform_(self.W_qk, -s, s)
        self.W_v = nn.Linear(input_dim, output_dim, bias=False)
        self.W_o = nn.Linear(output_dim, output_dim, bias=False)

    def forward(self, x, mask=None):
        scores = torch.bmm(x @ self.W_qk, x.transpose(1, 2)) / (self.output_dim ** 0.5)
        if mask is not None:
            scores = scores + mask
        attn_out = self.W_o(torch.bmm(F.softmax(scores, dim=-1), self.W_v(x)))
        return torch.cat([x, attn_out], dim=-1), scores


class QKSeparateOVCombinedBlock(nn.Module):
    """HYBRID: Q/K left factorized as Wq, Wk, but OV collapsed to a single W_ov.
    The complement of the block above -- together the two isolate which circuit's
    factorization actually carries the optimization advantage."""
    
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.output_dim = output_dim
        self.W_q = nn.Linear(input_dim, output_dim, bias=False)
        self.W_k = nn.Linear(input_dim, output_dim, bias=False)
        self.W_ov = nn.Parameter(torch.empty(input_dim, output_dim))
        s = input_dim ** -0.5
        nn.init.uniform_(self.W_ov, -s, s)

    def forward(self, x, mask=None):
        Q, K = self.W_q(x), self.W_k(x)
        scores = torch.bmm(Q, K.transpose(1, 2)) / (self.output_dim ** 0.5)
        if mask is not None:
            scores = scores + mask
        attn_out = torch.bmm(F.softmax(scores, dim=-1), x @ self.W_ov)
        return torch.cat([x, attn_out], dim=-1), scores


# All four are the SAME function class (W_qk == Wq^T Wk, W_ov == Wv^T Wo^T exactly);
# they differ only in how many redundant coordinates GD gets to move in.
_ATTN_BLOCKS = {
    'combined':       CombinedAttentionBlock,        # QK combined, OV combined
    'separate':       SeparateAttentionBlock,        # QK separate, OV separate
    'qk_comb_ov_sep': QKCombinedOVSeparateBlock,     # QK combined, OV separate
    'qk_sep_ov_comb': QKSeparateOVCombinedBlock,     # QK separate, OV combined
}


class SimplifiedTransformerMLP(nn.Module):
    def __init__(self, total_vocab, seq_len, attn_dim, v_sig, ffn_hidden_per_state, bias=True,
                 attn_param='combined'):
        super().__init__()
        self.total_vocab = total_vocab
        self.seq_len      = seq_len
        self.v_sig        = v_sig
        self.attn_param   = attn_param
        self.initial_dim  = total_vocab + seq_len
        Block = _ATTN_BLOCKS[attn_param]
        self.layer1 = Block(self.initial_dim, attn_dim)
        self.layer2 = Block(self.initial_dim + attn_dim, attn_dim)
        self.D   = self.initial_dim + 2 * attn_dim
        self.in2 = self.initial_dim + attn_dim
        self.ff1 = nn.Linear(self.D, v_sig * ffn_hidden_per_state)
        self.ff2 = nn.Linear(v_sig * ffn_hidden_per_state, self.D)
        self.act = nn.GELU()
        self.fc_out = nn.Linear(self.D, total_vocab, bias=bias)

    def forward(self, x):
        B, T = x.shape
        tok = F.one_hot(x, num_classes=self.total_vocab).float()
        pos = F.one_hot(torch.arange(T, device=x.device),
                        num_classes=self.seq_len).float().unsqueeze(0).expand(B, T, -1)
        s = torch.cat([tok, pos], dim=-1)
        mask = torch.triu(torch.ones(T, T, device=x.device) * float('-inf'), diagonal=1)
        s, _ = self.layer1(s, mask)
        s, _ = self.layer2(s, mask)
        s = s + self.ff2(self.act(self.ff1(s)))
        return self.fc_out(s)


def _attn_mats_with_fanin(model):
    """(param, fan_in) for every attention matrix, for ANY of the four parametrizations.
    Combined mats (W_qk, W_ov) are applied as x @ W, so fan_in = shape[0]; separate mats are
    nn.Linear weights of shape [out, in], so fan_in = shape[1]. Detection is per-circuit
    (QK and OV independently), which is what makes the two hybrids work."""
    out = []
    for layer in (model.layer1, model.layer2):
        if hasattr(layer, 'W_qk'):
            out.append((layer.W_qk, layer.W_qk.shape[0]))
        else:
            out.append((layer.W_q.weight, layer.W_q.weight.shape[1]))
            out.append((layer.W_k.weight, layer.W_k.weight.shape[1]))
        if hasattr(layer, 'W_ov'):
            out.append((layer.W_ov, layer.W_ov.shape[0]))
        else:
            out.append((layer.W_v.weight, layer.W_v.weight.shape[1]))
            out.append((layer.W_o.weight, layer.W_o.weight.shape[1]))
    return out


def _layer_dims(layer):
    """(input_dim, output_dim) of an attention block, for any of the four parametrizations."""
    if hasattr(layer, 'W_qk'):
        d_in = layer.W_qk.shape[0]
    elif hasattr(layer, 'W_q'):
        d_in = layer.W_q.weight.shape[1]
    else:
        d_in = layer.W_ov.shape[0]
    return d_in, layer.output_dim


def _init_fc_out(model, scheme, seed):
    """Initialize the unembedding fc_out (weight AND bias) from a DEDICATED generator.

    Why a dedicated generator rather than the global RNG: the architectures allocate different
    numbers of parameter tensors during construction (combined creates 4 attention tensors,
    separate creates 8), so by the time nn.Linear reaches fc_out the global RNG sits at a
    different position for each. Under torch.manual_seed(seed) alone the two archs therefore got
    DIFFERENT unembeddings -- a real confound that showed up as epoch-0 losses differing by up to
    ~0.08 nats even in the supposedly matched-init runs. Drawing fc_out from a generator seeded
    only by `seed` makes every architecture start from the identical readout.

    fan_in = D (the residual width), matching nn.Linear's convention:
      'normal'            -> N(0, 1/sqrt(D))     on weight and bias
      'default'/'matched' -> U(+-1/sqrt(D))      on weight and bias (= nn.Linear's own method)
    """
    gen = torch.Generator(device='cpu').manual_seed(int(seed) * 1000 + 7)
    dev = next(model.parameters()).device
    fan_in = model.fc_out.weight.shape[1]
    s = 1.0 / math.sqrt(fan_in)

    def draw(shape):
        if scheme == 'normal':
            return torch.randn(shape, generator=gen) * s
        if scheme in ('default', 'matched'):
            return (torch.rand(shape, generator=gen) * 2.0 - 1.0) * s
        raise ValueError(f'unknown init scheme {scheme!r}')

    with torch.no_grad():
        model.fc_out.weight.copy_(draw(model.fc_out.weight.shape).to(dev))
        if model.fc_out.bias is not None:
            model.fc_out.bias.copy_(draw(model.fc_out.bias.shape).to(dev))


def apply_init(model, scheme, seed=0, init_fc_out=False):
    """Initialize the attention matrices.

    'default' : U(+-1/sqrt(fan_in)) on each arch's own matrices  (= nn.Linear's default)
    'normal'  : N(0, 1/sqrt(fan_in)) on each arch's own matrices
    'matched' : MATCHED-PRODUCT init -- the control that removes the initialization-statistics
                confound. Under 'normal', separate's *effective* W_qk = Wq^T Wk is a product of
                two Gaussians while combined's W_qk is a single Gaussian draw: same method, but
                different scale/spectrum, so the two architectures do NOT start from the same
                function (epoch-0 losses differed by up to ~0.1 nats). Here BOTH archs draw the
                same Wq,Wk,Wv,Wo ~ N(0, 1/sqrt(fan_in)) from a dedicated generator seeded only by
                `seed`; separate installs them directly, combined installs their exact products
                W_qk = Wq^T Wk and W_ov = Wv^T Wo^T. The two models are then the SAME FUNCTION at
                step 0 (verified: identical logits), so any subsequent divergence is pure
                optimization dynamics, not a head start.

                The generator is independent of the global RNG on purpose: model construction
                consumes a different number of global draws per architecture, so a shared
                torch.manual_seed would NOT give the two archs the same matrices.

    When init_fc_out is True the unembedding is initialized too (see _init_fc_out) instead of
    being left at nn.Linear's default.
    """
    if init_fc_out:
        _init_fc_out(model, scheme, seed)
    if scheme == 'matched':
        gen = torch.Generator(device='cpu').manual_seed(int(seed))
        dev = next(model.parameters()).device
        with torch.no_grad():
            for layer in (model.layer1, model.layer2):
                d_in, d_out = _layer_dims(layer)
                # canonical draw order q,k,v,o -- identical for ALL parametrizations
                Wq = torch.randn(d_out, d_in,  generator=gen) / math.sqrt(d_in)
                Wk = torch.randn(d_out, d_in,  generator=gen) / math.sqrt(d_in)
                Wv = torch.randn(d_out, d_in,  generator=gen) / math.sqrt(d_in)
                Wo = torch.randn(d_out, d_out, generator=gen) / math.sqrt(d_out)
                # each circuit installs either the factors or their exact algebraic collapse,
                # so all four architectures are the SAME function at step 0
                if hasattr(layer, 'W_qk'):
                    layer.W_qk.copy_((Wq.t() @ Wk).to(dev))
                else:
                    layer.W_q.weight.copy_(Wq.to(dev)); layer.W_k.weight.copy_(Wk.to(dev))
                if hasattr(layer, 'W_ov'):
                    layer.W_ov.copy_((Wv.t() @ Wo.t()).to(dev))
                else:
                    layer.W_v.weight.copy_(Wv.to(dev)); layer.W_o.weight.copy_(Wo.to(dev))
        return

    with torch.no_grad():
        for p, fan_in in _attn_mats_with_fanin(model):
            s = 1.0 / math.sqrt(fan_in)
            if scheme == 'default':
                p.uniform_(-s, s)
            elif scheme == 'normal':
                p.normal_(0.0, s)
            else:
                raise ValueError(f'unknown init scheme {scheme!r}')


# ─── Frozen log-readout ───────────────────────────────────────────────────────
def fit_log_core(delta, k, fmax, cgain, n_grid=40001):
    """Least-squares GELU representation of the RESIDUAL log target

        g(f) = log(f + delta) - f        for f in [0, fmax]

    so that, after the residual add, coordinate in2+v holds log(f_v + delta) and the model's
    softmax returns p(v) proportional to (f_v + delta).

    Returns (a, b, w, c, max_abs_err, mse) with  g(f) ~ GELU(f*a + b) @ w + c,
    a, b, w each [k] and c a scalar, all float32; the fit itself is done in float64.

    Design (all three points are where the old implementation went wrong):
      * knots geometric in (f + delta) -- g'' = -(f+delta)^-2, so all the curvature is at small
        f and uniform knots waste the basis. Geometric knots make the local error scale-free.
      * gain a_j = cgain / h_j with h_j the local knot spacing, so each ramp turns on over its
        own scale rather than a single global width.
      * NO gradient refinement. With a and b fixed the problem is linear in (w, c) and lstsq is
        the global optimum. The old code ran 2000 Adam steps on top of that optimum and they
        diverged (installed MSE 4.0e+02, max|err| 114.7, non-monotone).
    """
    dt = torch.float64
    kn = torch.exp(torch.linspace(math.log(delta), math.log(fmax + delta), k, dtype=dt)) - delta
    h  = torch.diff(kn, prepend=(2 * kn[:1] - kn[1:2])).clamp_min(1e-12)
    a  = cgain / h
    b  = -a * kn
    fg = (torch.exp(torch.linspace(math.log(delta), math.log(fmax + delta), n_grid,
                                   dtype=dt)) - delta).clamp_min(0.0)
    basis  = F.gelu(fg.unsqueeze(1) * a.unsqueeze(0) + b.unsqueeze(0))
    basis1 = torch.cat([basis, torch.ones(n_grid, 1, dtype=dt)], dim=1)
    target = torch.log(fg + delta) - fg
    sol    = torch.linalg.lstsq(basis1, target.unsqueeze(1)).solution.squeeze(1)
    w, c   = sol[:k], sol[k]

    # Judge the core in float32 -- that is how the network will actually evaluate it.
    a32, b32, w32, c32 = a.float(), b.float(), w.float(), float(c)
    f32  = fg.float()
    pred = F.gelu(f32.unsqueeze(1) * a32 + b32) @ w32 + c32
    err  = (pred + f32 - torch.log(f32 + delta)).abs()      # error of the IMPLIED log
    return a32, b32, w32, c32, float(err.max()), float((err ** 2).mean())


def check_log_core(a, b, w, c, delta, fmax, tol, verbose=True):
    """Independent audit of an installed core: max|err| of the implied log in several bands,
    monotonicity where it matters, and the endpoint values. Raises if it is not a log."""
    def implied(f):
        f = torch.as_tensor(f, dtype=torch.float32).reshape(-1)
        return F.gelu(f.unsqueeze(1) * a + b) @ w + c + f
    bands = [(0.0, 1.0), (1.0, 5.0), (5.0, max(5.0, fmax))]
    worst = 0.0
    for lo, hi in bands:
        if hi <= lo: continue
        g = torch.linspace(lo, hi, 20001)
        e = float((implied(g) - torch.log(g + delta)).abs().max())
        worst = max(worst, e)
        if verbose:
            print(f'    band f in [{lo:5.1f},{hi:5.1f}] : max|err| = {e:.2e} nats')
    g = torch.linspace(0.0, 1.0, 20001)
    mono = bool((torch.diff(implied(g)) > 0).all())
    if verbose:
        pts = [0.0, delta, 0.1, 1.0, fmax]
        print('    implied log at f = ' +
              '  '.join(f'{p:g}->{float(implied([p])[0]):+.4f}' for p in pts))
        print(f'    target        at f = ' +
              '  '.join(f'{p:g}->{math.log(p + delta):+.4f}' for p in pts))
        print(f'    monotone on [0,1]: {mono}')
    if worst > tol or not mono:
        raise RuntimeError(
            f'frozen readout is NOT a usable log: max|err|={worst:.3e} (tol {tol:.1e}), '
            f'monotone={mono}. Refusing to train -- this is exactly the failure that '
            f'invalidated the earlier runs.')
    return worst, mono


_LOG_CORE_CACHE = {}   # fit depends only on (delta,k,fmax,cgain) -- fit ONCE, reuse everywhere

def build_frozen_ffn(model, delta, k, fmax, cgain, tol):
    """Install the log core into ff1/ff2 and freeze them.

    Wiring: signal state v owns hidden rows [v*k, (v+1)*k). Those rows read ONE column of the
    residual stream (in2 + v) and write back to that same coordinate -- no cross-talk between
    states and no other coordinate touched, so the residual add overwrites exactly the count
    slots and leaves the rest of the stream bit-identical.
    """
    dev = next(model.parameters()).device
    key = (delta, k, fmax, cgain)
    if key not in _LOG_CORE_CACHE:
        core = fit_log_core(delta, k, fmax, cgain)
        print(f'  [frozen readout] fit  delta={delta}  k={k}  fmax={fmax}  cgain={cgain}')
        print(f'                   max|err|={core[4]:.2e} nats   mse={core[5]:.2e}')
        check_log_core(*core[:4], delta, fmax, tol)
        print(f'                   PASSED (tol {tol:.1e}) -- cached for all runs')
        _LOG_CORE_CACHE[key] = core
    a, b, w, c, _, _ = _LOG_CORE_CACHE[key]
    a, b, w = a.to(dev), b.to(dev), w.to(dev)
    Vs, D, in2 = model.v_sig, model.D, model.in2
    assert model.ff1.out_features == Vs * k, \
        f'ff1 width {model.ff1.out_features} != v_sig*k {Vs * k}'
    W1 = torch.zeros(Vs * k, D, device=dev); B1 = torch.zeros(Vs * k, device=dev)
    W2 = torch.zeros(D, Vs * k, device=dev); B2 = torch.zeros(D, device=dev)
    for v in range(Vs):
        rows = slice(v * k, (v + 1) * k)
        W1[rows, in2 + v] = a          # read f_v ...
        B1[rows]          = b
        W2[in2 + v, rows] = w          # ... write log(f_v+delta) - f_v back into the same slot
        B2[in2 + v]       = c
    with torch.no_grad():
        model.ff1.weight.copy_(W1); model.ff1.bias.copy_(B1)
        model.ff2.weight.copy_(W2); model.ff2.bias.copy_(B2)
    for p in (model.ff1.weight, model.ff1.bias, model.ff2.weight, model.ff2.bias):
        p.requires_grad_(False)


@torch.no_grad()
def signal_slots(model, x):
    """The layer-2 attention output at the count coordinates in2 : in2+v_sig, i.e. the f the
    frozen readout is about to take the log of -- read BEFORE the FFN."""
    B, T = x.shape
    tok = F.one_hot(x, num_classes=model.total_vocab).float()
    pos = F.one_hot(torch.arange(T, device=x.device),
                    num_classes=model.seq_len).float().unsqueeze(0).expand(B, T, -1)
    s = torch.cat([tok, pos], dim=-1)
    mask = torch.triu(torch.ones(T, T, device=x.device) * float('-inf'), diagonal=1)
    s, _ = model.layer1(s, mask)
    s, _ = model.layer2(s, mask)
    return s[..., model.in2:model.in2 + model.v_sig]


@torch.no_grad()
def readout_range(model, x, token_start, token_end, device, fmax, batch=256, n_max=2048):
    """Is the model still feeding the frozen readout inputs it was CALIBRATED for?

    The core is a log only on [0, fmax]; below 0 it flattens out and above fmax it continues
    roughly linearly. A global rescale of the attention output is harmless (log(g f + delta) =
    log g + log(f + delta/g), and the constant cancels in the softmax) -- leaving the fitted
    range is not. This is recorded every eval so that failure mode can never again be silent.
    """
    was_training = model.training
    model.eval()
    lo, hi, n_out, n_neg, tot = float('inf'), float('-inf'), 0, 0, 0
    for i in range(0, min(x.shape[0], n_max), batch):
        f = signal_slots(model, x[i:i + batch].to(device))[:, token_start:token_end]
        lo = min(lo, float(f.min())); hi = max(hi, float(f.max()))
        n_out += int((f > fmax).sum()); n_neg += int((f < 0).sum()); tot += f.numel()
    if was_training:
        model.train()
    return {'min': lo, 'max': hi,
            'frac_above_fmax': n_out / max(tot, 1), 'frac_negative': n_neg / max(tot, 1)}


# ─── Data / training infra ────────────────────────────────────────────────────
_DATA_CACHE = {}   # (split,thr) -> blob; holds ONE threshold's train+test at a time

def load_data(split, thr):
    key = (split, thr)
    if key not in _DATA_CACHE:
        if any(k[1] != thr for k in _DATA_CACHE):
            _DATA_CACHE.clear()
        _DATA_CACHE[key] = torch.load(
            os.path.join(BASE['data_dir'], split, f'data_x_{thr}.pt'), weights_only=False)
    return _DATA_CACHE[key]


def _attn_params(model):
    return [p for p, _ in _attn_mats_with_fanin(model)]


def get_trainable_params(model, groups):
    params = []
    if 'attn' in groups:
        params += _attn_params(model)
    if 'ffn' in groups:
        params += list(model.ff1.parameters()) + list(model.ff2.parameters())
    if 'fc_out' in groups:
        params += list(model.fc_out.parameters())
    return params


def set_trainable(model, groups, freeze_readout):
    for p in _attn_params(model):
        p.requires_grad_('attn' in groups)
    for p in list(model.ff1.parameters()) + list(model.ff2.parameters()):
        p.requires_grad_(('ffn' in groups) and not freeze_readout)
    for p in model.fc_out.parameters():
        p.requires_grad_('fc_out' in groups)


def make_optimizer(params, kind, lr, wd):
    if kind == 'sgd':
        return torch.optim.SGD(params, lr=lr, momentum=0.0, weight_decay=wd)
    if kind == 'adam':
        return torch.optim.Adam(params, lr=lr, weight_decay=wd)
    raise ValueError(f'unknown optimizer={kind!r}')


@torch.no_grad()
def eval_ce(model, x, y, V, token_start, token_end, device, batch):
    model.eval()
    total_nll, total_n = 0.0, 0
    for i in range(0, x.shape[0], batch):
        xb = x[i:i + batch].to(device)
        yb = y[i:i + batch, token_start:token_end].to(device)
        logits = model(xb)[:, token_start:token_end]
        loss = F.cross_entropy(logits.reshape(-1, V), yb.reshape(-1), reduction='sum')
        total_nll += loss.item(); total_n += yb.numel()
        del xb, yb, logits, loss
        if device.type == 'cuda': torch.cuda.empty_cache()
    model.train()
    return total_nll / total_n


# ─── wandb (mirror; local json is the source of truth) ────────────────────────
_WANDB_OK = None

def wandb_start(run_cfg, name, group, tags, project=None):
    global _WANDB_OK
    if not USE_WANDB or _WANDB_OK is False:
        return None
    try:
        import wandb
        run = wandb.init(project=project or WANDB_PROJECT, entity=WANDB_ENTITY,
                         name=name, group=group,
                         tags=tags, config=run_cfg, reinit='finish_previous',
                         dir=os.path.join(_SCRIPT_DIR, 'wandb_local'))
        _WANDB_OK = True
        return run
    except Exception as e:
        if _WANDB_OK is None:
            print(f'  [wandb] disabled ({type(e).__name__}: {e}) -- local logging only')
        _WANDB_OK = False
        return None


# ─── One run = (arch, seed, threshold) ────────────────────────────────────────
def balanced_factorize(M, inner, mode):
    """Split M into two balanced factors whose product is M again.

    mode='qk': returns (Wq, Wk) with Wq^T @ Wk == M      (both [inner, D_in])
    mode='ov': returns (Wv, Wo) with Wv^T @ Wo^T == M    (Wv [inner, D_in], Wo [D_out, inner])

    Via the SVD M = U S V^T, both factors carry S^(1/2), which makes them *balanced*
    (Wq Wq^T == Wk Wk^T == S) -- the standard balanced factorization, and the state
    gradient flow on a factorized parametrization conserves.

    Returns (A, B, rel_err). rel_err > 0 means M's rank exceeded `inner` and the split is a
    best rank-`inner` approximation rather than exact -- see the note in build_factorized().
    """
    U, S, Vh = torch.linalg.svd(M, full_matrices=False)
    U, S, Vh = U[:, :inner], S[:inner], Vh[:inner]
    root = torch.diag(torch.sqrt(S))
    if mode == 'qk':
        A = root @ U.t()          # [inner, D_in]   -> Wq
        B = root @ Vh             # [inner, D_in]   -> Wk
        recon = A.t() @ B
    else:
        A = root @ U.t()          # [inner, D_in]   -> Wv   (so Wv^T == U @ root)
        B = Vh.t() @ root         # [D_out, inner]  -> Wo   (so Wo^T == root @ Vh)
        recon = A.t() @ B.t()     # == (U root)(root Vh) == U S Vh == M
    rel = (recon - M).norm().item() / max(M.norm().item(), 1e-12)
    return A, B, rel


def build_factorized(thr, device):
    """Load a finished source (combined) run, factorize its attention into balanced factors,
    and return an equivalent model in the target (separate) parametrization -- fc_out and the
    frozen readout carried over unchanged, so training genuinely resumes from that solution."""
    C = BASE; F = FACTORIZE
    V = C['v_sig'] + C['n_noise']; attn_dim = V + C['seq_len']
    src = os.path.join(C['runs_dir'], F['source_cell'],
                       f"model_{F['source_arch']}_seed{SEEDS[0]}_x_{thr}.pth")
    sd = torch.load(src, map_location='cpu', weights_only=True)

    model = SimplifiedTransformerMLP(V, C['seq_len'], attn_dim, C['v_sig'],
                                     C['freeze_hidden_k'], bias=C['bias'],
                                     attn_param=F['target_arch']).to(device)
    # The frozen log-readout is COPIED from the source, never refit -- the target must differ
    # from the source in exactly one respect, the attention factorization. This also matters
    # historically: the source runs were trained with the OLD (broken) core, so refitting would
    # hand the transplanted model a different readout than the solution it is resuming from.
    with torch.no_grad():
        for name in ('ff1', 'ff2'):
            getattr(model, name).weight.copy_(sd[f'{name}.weight'].to(device))
            getattr(model, name).bias.copy_(sd[f'{name}.bias'].to(device))
        for p in (model.ff1.weight, model.ff1.bias, model.ff2.weight, model.ff2.bias):
            p.requires_grad_(not C['freeze_readout'])
    errs = {}
    with torch.no_grad():
        for i, layer in enumerate((model.layer1, model.layer2), start=1):
            inner = layer.output_dim
            Wq, Wk, e1 = balanced_factorize(sd[f'layer{i}.W_qk'].float(), inner, 'qk')
            Wv, Wo, e2 = balanced_factorize(sd[f'layer{i}.W_ov'].float(), inner, 'ov')
            layer.W_q.weight.copy_(Wq.to(device)); layer.W_k.weight.copy_(Wk.to(device))
            layer.W_v.weight.copy_(Wv.to(device)); layer.W_o.weight.copy_(Wo.to(device))
            errs[f'layer{i}_qk'] = e1; errs[f'layer{i}_ov'] = e2
        # the readout is part of the learned solution -- carry it over verbatim
        model.fc_out.weight.copy_(sd['fc_out.weight'].to(device))
        if model.fc_out.bias is not None and 'fc_out.bias' in sd:
            model.fc_out.bias.copy_(sd['fc_out.bias'].to(device))
    return model, errs, src


def metrics_path_for(arch, seed, thr, lr):
    return os.path.join(BASE['runs_dir'], cell_tag_for(lr),
                        f'metrics_{arch}_seed{seed}_x_{thr}.json')


def train_one(arch, seed, thr, lr, device, preset_model=None, cell_tag=None,
              epochs=None, project=None, extra=None):
    C = BASE
    V = C['v_sig'] + C['n_noise']
    seq_len = C['seq_len']; token_start = C['token_start']; token_end = seq_len
    attn_dim = V + seq_len

    cell_tag = cell_tag or cell_tag_for(lr)
    n_epochs = epochs or C['epochs']
    out_dir = os.path.join(C['runs_dir'], cell_tag)
    os.makedirs(out_dir, exist_ok=True)
    stem         = f'{arch}_seed{seed}_x_{thr}'
    metrics_path = os.path.join(out_dir, f'metrics_{stem}.json')
    model_path   = os.path.join(out_dir, f'model_{stem}.pth')
    if SKIP_DONE and os.path.exists(metrics_path):
        print(f'  [skip done] {stem}')
        return False

    run_cfg = {**{k: v for k, v in C.items() if k not in ('runs_dir', 'data_dir')},
               'attn_param': arch, 'seed': seed, 'threshold': thr, 'cell': cell_tag,
               'lr': lr, 'epochs': n_epochs, **(extra or {})}

    torch.manual_seed(seed); np.random.seed(seed)
    train_blob = load_data('train', thr)
    test_blob  = load_data('test',  thr)
    tr_x, tr_y = train_blob['x'], train_blob['y']
    te_x, te_y = test_blob['x'], test_blob['y']

    if preset_model is not None:
        model = preset_model          # already carries the transplanted weights + frozen readout
    else:
        model = SimplifiedTransformerMLP(V, seq_len, attn_dim, C['v_sig'], C['freeze_hidden_k'],
                                          bias=C['bias'], attn_param=arch).to(device)
        apply_init(model, C['init'], seed=seed, init_fc_out=C.get('init_fc_out', False))
        if C['freeze_readout']:
            build_frozen_ffn(model, C['freeze_delta'], C['freeze_hidden_k'],
                              C['freeze_fit_fmax'], C['freeze_fit_cgain'], C['freeze_fit_tol'])
    set_trainable(model, C['train_params'], C['freeze_readout'])
    params = get_trainable_params(model, C['train_params'])
    opt = make_optimizer(params, C['optimizer'], lr, C['weight_decay'])

    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(tr_x, tr_y), batch_size=C['batch_size'], shuffle=True)

    wb = wandb_start(run_cfg, name=f'{cell_tag}__{stem}', group=f'lr{lr}',
                     project=project,
                     tags=[arch, f'thr{thr}', f'seed{seed}', C['optimizer'], C['init'],
                           f'ep{n_epochs}', f'lr{lr}'])

    # epoch 0 = the RANDOM INIT: first point of both curves, before any update
    tr0 = eval_ce(model, tr_x, tr_y, V, token_start, token_end, device, C['eval_batch'])
    te0 = eval_ce(model, te_x, te_y, V, token_start, token_end, device, C['eval_batch'])
    history = {'train_loss': [tr0], 'test_loss': [te0]}

    # Is the frozen readout being fed inputs it is calibrated for? Recorded every epoch so the
    # "the log silently stopped being a log" failure can never go unnoticed again.
    track_readout = C['freeze_readout']
    rr0 = readout_range(model, te_x, token_start, token_end, device,
                        C['freeze_fit_fmax']) if track_readout else None
    history['readout_min'] = [rr0['min']] if rr0 else []
    history['readout_max'] = [rr0['max']] if rr0 else []
    if wb:
        # Declare the running-best metrics up front. Without this wandb only decides how to
        # summarise a metric once the run FINISHES, so an in-progress (or Ctrl+C'd) run shows
        # nothing for them in the runs table -- which is exactly what happened before.
        # summary='min' also keeps the table column live while the run is still going.
        for _mname in ('train_ce', 'test_ce'):
            wb.define_metric(_mname, summary='min')
        for _mname in ('best_train', 'best_test'):
            wb.define_metric(_mname, summary='last')
        # log them at step 0 as well, so both curves start at the random init rather than
        # popping into existence at step 1
        wb.log({'train_ce': tr0, 'test_ce': te0,
                'best_train': tr0, 'best_test': te0,
                **({f'readout_{k}': v for k, v in rr0.items()} if rr0 else {})}, step=0)

    n_params = sum(p.numel() for p in params)
    print(f'  [lr={lr}] {stem}  trainable={n_params:,}  init: tr={tr0:.4f} te={te0:.4f}',
          flush=True)
    if rr0:
        print(f'    readout input range at init: [{rr0["min"]:.3f}, {rr0["max"]:.3f}]  '
              f'(calibrated for [0, {C["freeze_fit_fmax"]:g}])', flush=True)

    # best_train drives the CHECKPOINT and so still starts at inf (epoch 0 is the random init and
    # was never a checkpoint candidate). best_test has no checkpoint role, so it is seeded with
    # the epoch-0 value and is a true "best so far" from the very first point.
    best_train = float('inf'); test_at_best = None
    best_test  = te0
    best_state = None; stale = 0
    t0 = time.time(); ep = 0
    try:
        for ep in range(1, n_epochs + 1):
            model.train()
            ep_loss, ep_n = 0.0, 0
            for xb, yb in loader:
                xb = xb.to(device); yb = yb[:, token_start:token_end].to(device)
                opt.zero_grad()
                logits = model(xb)[:, token_start:token_end]
                loss = F.cross_entropy(logits.reshape(-1, V), yb.reshape(-1))
                loss.backward()
                if C['clip_grad']: torch.nn.utils.clip_grad_norm_(params, C['grad_clip'])
                opt.step()
                ep_loss += loss.item() * yb.numel(); ep_n += yb.numel()
            tr_loss = ep_loss / ep_n
            te_loss = eval_ce(model, te_x, te_y, V, token_start, token_end, device, C['eval_batch'])
            history['train_loss'].append(tr_loss)
            history['test_loss'].append(te_loss)
            rr = readout_range(model, te_x, token_start, token_end, device,
                               C['freeze_fit_fmax']) if track_readout else None
            if rr:
                history['readout_min'].append(rr['min'])
                history['readout_max'].append(rr['max'])

            if tr_loss < best_train - 1e-6:
                best_train = tr_loss; test_at_best = te_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                stale = 0
            else:
                stale += 1
            best_test = min(best_test, te_loss)
            if wb: wb.log({'train_ce': tr_loss, 'test_ce': te_loss,
                            'best_train': best_train, 'best_test': best_test,
                            **({f'readout_{k}': v for k, v in rr.items()} if rr else {})},
                           step=ep)
            if ep % 100 == 0:
                extra_s = (f'  f in [{rr["min"]:.2f},{rr["max"]:.2f}]'
                           + (f' OUT-OF-RANGE {100 * rr["frac_above_fmax"]:.1f}%'
                              if rr['frac_above_fmax'] > 1e-4 else '')) if rr else ''
                print(f'    ep {ep:5d}/{n_epochs}  tr={tr_loss:.4f}  te={te_loss:.4f}  '
                      f'best_tr={best_train:.4f}  best_te={best_test:.4f}{extra_s}', flush=True)
            if C['early_stop'] and stale >= C['patience']:
                print(f'    [early stop] ep={ep}  best_train={best_train:.4f}')
                break
    except KeyboardInterrupt:
        # A SECOND Ctrl+C used to land inside this sleep, escape train_one, and skip both the
        # metrics json and wb.finish() -- leaving an unsynced wandb run with no summary (which is
        # why best_train/best_test appeared to be missing) and an empty run directory on disk.
        # The sleep is now interruptible without losing the run: the second Ctrl+C is caught here,
        # the results are still written below, and `interrupted_twice` re-raises afterwards so the
        # sweep still exits as intended.
        interrupted_twice = False
        print('    [INTERRUPTED] saving best-so-far, moving to next run '
              '(Ctrl+C again within ~1s exits the sweep)', flush=True)
        try:
            time.sleep(1)
        except KeyboardInterrupt:
            interrupted_twice = True
            print('    [INTERRUPTED x2] finishing this run cleanly, then exiting', flush=True)
    else:
        interrupted_twice = False

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    torch.save(model.state_dict(), model_path)
    rr_end = readout_range(model, te_x, token_start, token_end, device,
                           C['freeze_fit_fmax']) if track_readout else None
    result = {'cell': cell_tag, 'arch': arch, 'seed': seed, 'threshold': float(thr),
              'history': history,
              'best_train': best_train, 'test_at_best': test_at_best,
              'best_test': best_test,
              'final_train': history['train_loss'][-1], 'final_test': history['test_loss'][-1],
              'epochs_run': ep, 'time_s': round(time.time() - t0, 1), 'config': run_cfg,
              'readout_range_best': rr_end}
    with open(metrics_path, 'w') as fh:
        json.dump(result, fh, indent=2)
    if wb:
        wb.summary.update({'best_train': best_train, 'test_at_best': test_at_best,
                            'best_test': best_test,
                            'final_train': result['final_train'],
                            'final_test': result['final_test'], 'epochs_run': ep,
                            **({f'readout_final_{k}': v for k, v in rr_end.items()}
                               if rr_end else {})})
        try:
            wb.finish()          # must run, or the summary never syncs and the web run
        except Exception as e:   # stays stuck in "running" with no best_* columns
            print(f'    [wandb] finish failed ({type(e).__name__}: {e})', flush=True)
    print(f'    done in {result["time_s"]/60:.1f}m  best_train={best_train:.4f}  '
          f'best_test={best_test:.4f}  test_at_best={test_at_best:.4f}  '
          f'final_tr={result["final_train"]:.4f}  -> {metrics_path}', flush=True)
    if rr_end and (rr_end['frac_above_fmax'] > 1e-4 or rr_end['min'] < -0.5):
        print(f'    WARNING: the readout was fed inputs outside its calibrated range '
              f'[0, {C["freeze_fit_fmax"]:g}] -- observed [{rr_end["min"]:.3f}, '
              f'{rr_end["max"]:.3f}], {100 * rr_end["frac_above_fmax"]:.2f}% above fmax, '
              f'{100 * rr_end["frac_negative"]:.2f}% negative. The frozen core is only a log '
              f'inside that range.', flush=True)
    if interrupted_twice:        # results are safely written -- now honour the second Ctrl+C
        raise KeyboardInterrupt
    return True


def rebuild_summary():
    """Scan ALL on-disk metrics into one summary json (re-running never shrinks it)."""
    rows = []
    for mp in sorted(glob.glob(os.path.join(BASE['runs_dir'], '*', 'metrics_*.json'))):
        try:
            m = json.load(open(mp))
            row = {k: m[k] for k in
                   ('cell', 'arch', 'seed', 'threshold', 'best_train', 'test_at_best',
                    'final_train', 'final_test', 'epochs_run')}
            row['best_test'] = m.get('best_test')   # .get: earlier runs predate this field
            rows.append(row)
        except (ValueError, KeyError):
            pass
    out = os.path.join(BASE['runs_dir'], 'summary.json')
    with open(out, 'w') as fh:
        json.dump(rows, fh, indent=2)
    print(f'\nSummary ({len(rows)} runs) -> {out}')


# ─── Main ─────────────────────────────────────────────────────────────────────
def main_factorize():
    """TRANSPLANT: factorize each finished COMBINED solution into balanced separate factors and
    keep training. If separate escapes a point combined had settled at, the deficit lives in the
    parametrization rather than in the loss surface."""
    C = BASE; F = FACTORIZE
    device = torch.device(C['device'] if torch.cuda.is_available() else 'cpu')
    V = C['v_sig'] + C['n_noise']
    print(f"MODE       : factorize (transplant)")
    print(f"source     : {F['source_cell']}  [{F['source_arch']}]")
    print(f"target     : {F['target_arch']}  ->  cell {F['cell']}")
    print(f"epochs     : {F['epochs']} more, lr={F['lr']}")
    print(f"wandb      : {F['project'] if USE_WANDB else 'OFF'}")
    print(f"thresholds : {THRESHOLDS}\n")

    os.makedirs(os.path.join(C['runs_dir'], F['cell']), exist_ok=True)
    t_start = time.time(); n_run = 0
    for thr in THRESHOLDS:
        seed = SEEDS[0]
        mp = os.path.join(C['runs_dir'], F['cell'],
                          f"metrics_{F['target_arch']}_seed{seed}_x_{thr}.json")
        if SKIP_DONE and os.path.exists(mp):
            print(f'  [skip done] thr={thr}'); continue

        model, errs, src = build_factorized(thr, device)
        set_trainable(model, C['train_params'], C['freeze_readout'])

        # How faithful was the transplant?  A factorization is exact only when the source
        # matrix's rank fits inside the factor's inner dimension. layer2's W_qk is [216,216]
        # while the separate form caps rank at output_dim=108, so that one is a best rank-108
        # approximation -- report it rather than let it pass silently.
        worst = max(errs.values())
        print(f"  thr={thr}  transplanted from {os.path.basename(src)}")
        print(f"    factorization rel. error: " +
              "  ".join(f'{k}={v:.2e}' for k, v in errs.items()))
        if worst > 1e-4:
            print(f"    NOTE: worst={worst:.2e} -- not an exact reparametrization "
                  f"(rank of the source exceeds the factor's inner dim); the transplanted "
                  f"model is the closest reachable point, not an identical function.")

        # CE of the source solution vs the transplanted one -- the size of the jump, if any
        blob_tr = load_data('train', thr)
        ce_tr = eval_ce(model, blob_tr['x'], blob_tr['y'], V, C['token_start'], C['seq_len'],
                        device, C['eval_batch'])
        src_metrics = json.load(open(os.path.join(
            C['runs_dir'], F['source_cell'],
            f"metrics_{F['source_arch']}_seed{seed}_x_{thr}.json")))
        print(f"    source best_train={src_metrics['best_train']:.4f}  ->  "
              f"transplanted train CE={ce_tr:.4f}  (jump {ce_tr - src_metrics['best_train']:+.4f})")

        train_one(F['target_arch'], seed, thr, F['lr'], device,
                  preset_model=model, cell_tag=F['cell'], epochs=F['epochs'],
                  project=F['project'],
                  extra={'transplant': True, 'source_cell': F['source_cell'],
                         'source_arch': F['source_arch'],
                         'source_best_train': src_metrics['best_train'],
                         'transplant_train_ce': ce_tr,
                         'factorization_rel_err': errs})
        n_run += 1
        print(f'  [{n_run}] elapsed={(time.time() - t_start) / 60:.1f}m\n', flush=True)
    rebuild_summary()


def main():
    device = torch.device(BASE['device'] if torch.cuda.is_available() else 'cpu')
    # threshold OUTERMOST so each threshold's train+test blobs load once for both archs
    # THRESHOLD is the outermost axis: finish every learning rate for one threshold before
    # moving to the next, so a partial sweep still yields a complete lr-vs-CE curve at the
    # thresholds it reached (rather than one lr across all thresholds and nothing else).
    # It also maximises the data cache -- _DATA_CACHE holds one threshold's blobs at a time,
    # so this loads each threshold's train+test exactly once instead of once per lr.
    runs = [(arch, seed, thr, lr)
            for thr in THRESHOLDS for lr in LRS for arch in ARCHS for seed in SEEDS]
    n_total = len(runs)
    n_done = sum(1 for r in runs if os.path.exists(metrics_path_for(*r)))

    print(f'device     : {device}')
    print(f'experiment : {RUN_NAME}')
    print(f'cells      : {[cell_tag_for(lr) for lr in LRS]}')
    print(f'optimizer  : {BASE["optimizer"]}  lr={LRS if len(LRS) > 1 else LRS[0]}  '
          f'batch={BASE["batch_size"]}  init={BASE["init"]}'
          f'{" (unembedding included)" if BASE.get("init_fc_out") else ""}')
    print(f'archs      : {ARCHS}   seeds: {SEEDS}   thresholds: {THRESHOLDS}')
    print(f'epochs     : {BASE["epochs"]}   early_stop={BASE["early_stop"]}'
          f'{"" if BASE["early_stop"] else "  (full trajectory kept -- the point of this run)"}')
    print(f'readout    : frozen log,  delta={BASE["freeze_delta"]}  k={BASE["freeze_hidden_k"]}'
          f'  fmax={BASE["freeze_fit_fmax"]}  cgain={BASE["freeze_fit_cgain"]}'
          f'  (verified below, tol {BASE["freeze_fit_tol"]:.0e})')
    print(f'data       : {BASE["data_dir"]}')
    print(f'runs       : {n_total} total  ({n_done} already done, will skip)')
    print(f'wandb      : {"ON, project=" + WANDB_PROJECT if USE_WANDB else "OFF"} '
          f'(local json/checkpoints always written)')
    est = len([r for r in runs if not os.path.exists(metrics_path_for(*r))]) \
          * BASE['epochs'] * 1.15 / 3600
    print(f'rough ETA  : ~{est:.1f}h at ~1.15 s/epoch (folder-12 measured rate)\n')

    os.makedirs(BASE['runs_dir'], exist_ok=True)
    os.makedirs(os.path.join(_SCRIPT_DIR, 'wandb_local'), exist_ok=True)
    t_start = time.time(); n_run = 0
    try:
        for i, (arch, seed, thr, lr) in enumerate(runs):
            did = train_one(arch, seed, thr, lr, device)
            if did:
                n_run += 1
                avg = (time.time() - t_start) / n_run
                remaining = sum(1 for r in runs[i + 1:]
                                if not os.path.exists(metrics_path_for(*r)))
                print(f'  [{i + 1}/{n_total}]  avg={avg/60:.1f}m/run  '
                      f'ETA={remaining * avg / 3600:.1f}h\n', flush=True)
    except KeyboardInterrupt:
        # re-raised by train_one after it has written its results -- stop the sweep, but still
        # rebuild the summary so the completed runs are indexed
        print('\n[sweep stopped by Ctrl+C]', flush=True)
    rebuild_summary()


if __name__ == '__main__':
    main_factorize() if MODE == 'factorize' else main()
