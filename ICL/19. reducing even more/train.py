"""Training script -- two-layer QK/OV model, block-sparse AND weight-tied, no unembedding.

Written from scratch. Nothing is imported from folder 18; hyperparameters that carry over are
kept identical so the runs are comparable.

WHAT IS NEW HERE -- WEIGHT TYING
--------------------------------
Folder 18 reduced which ENTRIES may move. This folder reduces how many INDEPENDENT VALUES those
entries may take: inside each tied group every entry is forced to share one value, so a 64-cell
block can carry a single degree of freedom.

    layer 1  W_qk[0:8, 0:8]        four 4x4 quadrants, one value each        64 cells ->  4 dof
    layer 2  W_qk[0:8, 0:8]    (A) left 8x4 and right 8x4, one value each    64 cells ->  2 dof
    layer 2  W_qk[0:4, 108:112](B) diagonal / off-diagonal                   16 cells ->  2 dof
    layer 2  W_qk[108:112, 108:112](C) diagonal / off-diagonal               16 cells ->  2 dof

B and C are NOT tied to each other -- 2 dof each, 4 between them.
Layer 1's position block W_qk[8:108, 8:108] is untouched: every entry moves on its own.

    degrees of freedom:  4 + 9,900 (position) + 2 + 2 + 2  =  9,910
    nonzero entries    :  64 + 9,900 + 64 + 16 + 16        = 10,060   (folder 18's count)

HOW THE TYING IS ENFORCED
-------------------------
Not by reparametrising the model -- W_qk stays a plain Parameter, so the snapshots, the masks
and every downstream analysis keep working on the full matrices. Instead, two things happen:

  1. a gradient hook replaces each group's gradient by its SUM over the group. If W[i] = theta
     for all i in a group G, then dL/dtheta = sum_{i in G} dL/dW[i] -- that sum IS the tied
     parameter's gradient, so every member then takes the same step the single parameter would.
     (Under Adam the choice of sum vs mean barely matters, since the update is scale-invariant
     per coordinate; under SGD it matters by a factor of |G|, so the sum is what is correct.)
  2. `retie()` after every optimizer step sets each group back to its own mean. With a zero
     init and identical gradients the members never actually diverge; this only mops up float
     drift, and it makes the invariant checkable rather than assumed.

`--check-ties` asserts, every step, that the spread inside each group is 0. Cheap, and it turns
a silent bug into a loud one.

ARCHITECTURE
------------
    x            one-hot token (8) + one-hot position (100)        -> 108
    layer 1      attention, returns cat[x, attn_out]               -> 216
    layer 2      attention, returns attn_out only                  -> 108
    f = layer2 attn_out[..., :8]                                   -> the raw count vector

W_ov[:8,:8] = I frozen, so f is a convex combination of token one-hots: f >= 0 and sum(f) = 1
exactly. f already IS a distribution, which is why there is no unembedding, no log and no output
softmax -- there is nothing left for them to do.

The causal mask is STRICTLY causal, triu(diagonal=0): a position can never attend to itself, so
W_qk does not have to spend a self-veto on it. mask[0,0] is reopened because position 0 has no
earlier key at all and softmax over a row of -inf is nan.

READOUT -- `--readout {none,bias}`
----------------------------------
    'none'   p = f / sum(f)
    'bias'   p = (f + b) / sum(f + b)        b FIXED from alpha, never trained   [default]

    b_noise = (1-alpha)(1 + V_sig*b_sig) / (n_noise*alpha)
        A good induction head drives the NOISE slots of f to zero -- that is what the noise
        filter is. But noise really occurs (1-alpha) of the time, so under 'none' a noise target
        lands on f = 0 and costs -log(LOG_FLOOR) = 69 nats. 'none' therefore FORBIDS the noise
        filter. The bias supplies that mass as a known constant instead. The /alpha is the part
        people drop: the renormalisation shrinks whatever you put in, so you have to put in more.
    b_sig = 0.01
        add-beta smoothing, so a successor the context never showed is not assigned exactly 0.

LOSS LOGGING -- fixed here
-------------------------
Folder 18 logged the train loss returned by full_batch_grad, which is computed BEFORE
opt.step(), while the test loss is evaluated AFTER. The two columns were therefore one step out
of phase: train_loss[k] = L(w_{k-1}) but test_loss[k] = L(w_k), and train_loss[1] repeated the
init value, so the train curve looked flat over the first step while test had already dropped.

Here the train loss is re-evaluated AFTER the step, so `train_loss[k]` and `test_loss[k]` are
the same weights. The pre-step value is still recorded as `train_loss_pre` (it is free -- the
backward pass computed it anyway) so the two can be compared. Cost: one extra forward pass over
the train split per epoch, no backward.

Usage:
    python train.py                                  # defaults below
    python train.py --epochs 500 --init zeros
    python train.py --no-tie                         # same masks, tying off (folder 18 behaviour)
    python train.py --check-ties                     # assert the invariant every step
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
THRESHOLDS = (0.5, 0.6, 0.7, 0.8, 0.9, 1.0)

WANDB_PROJECT = 'reducing_even_more_tied_blocks'
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
    'lr':           1.0,
    'weight_decay': 0.0,
    'epochs':       500,       # == optimizer steps (full batch)
    'init':         'zeros',   # 'zeros' | 'normal'   (W_qk only)
    'train_wov':    False,     # False -> W_ov top-left 8x8 = identity, frozen
    'blocks':       'relevant',
    'readout':      'bias',
    'b_sig':        0.01,
    'noise_floor':  1e-9,      # b_noise at alpha=1, where (1-alpha)/(n_noise*alpha) = 0

    # --- tying ---
    'tie':         True,
    'check_ties':  False,

    # --- snapshots ---
    'snapshots':   True,
    'snap_grads':  True,
    'snap_dtype':  'float32',

    # --- batching / evaluation ---
    # Chunking is a MEMORY device only: each chunk's mean loss is reweighted by its share of
    # the scored tokens, so the accumulated gradient equals one 25,000-sequence backward.
    # Bigger is faster where it fits -- measured on an A100, full_batch_grad took 3.33s at
    # 2048 and 1.14s at 25,000 (the whole split in one pass). 8192 is the safe middle.
    'grad_chunk':  8192,
    'eval_chunk':  8192,
    'meta_every':  25,         # rewrite snapmeta every N snapshot rows, not every row
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
    return (f"tied_opt-{CFG['optimizer']}_lr-{CFG['lr']}_init-{CFG['init']}"
            f"_wov-{'train' if CFG['train_wov'] else 'fixed'}"
            f"_blocks-{CFG['blocks']}{'' if CFG['tie'] else '-untied'}"
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


def check_dataset():
    """Every file present, and stamped as the equiprobable dataset."""
    for thr in THRESHOLDS:
        for split in ('train', 'test'):
            p = os.path.join(CFG['data_dir'], split, f'data_x_{thr}.pt')
            if not os.path.exists(p):
                sys.exit(f'missing {p}\ngenerate it in folder 14: python data_generation.py')
        if load_data('test', thr).get('succ_split') != 'equiprobable':
            sys.exit(f'{CFG["data_dir"]}/test/data_x_{thr}.pt is not the equiprobable dataset')
    _DATA.clear()


# -------------------------------------------------------------------------------------- model

class AttentionBlock(nn.Module):
    """One causal attention layer in combined W_qk / W_ov form.

    scores = (x W_qk x^T) / sqrt(output_dim) + causal mask,  out = softmax(scores) (x W_ov).
    `concat` controls whether the residual stream is carried forward: layer 1 needs it (layer 2
    reads [x | attn_out]), layer 2 does not.
    """

    def __init__(self, input_dim, output_dim, concat):
        super().__init__()
        self.output_dim = output_dim
        self.concat = concat
        self.W_qk = nn.Parameter(torch.zeros(input_dim, input_dim))
        self.W_ov = nn.Parameter(torch.zeros(input_dim, output_dim))

    def forward(self, x, mask):
        scores = torch.bmm(x @ self.W_qk, x.transpose(1, 2)) / (self.output_dim ** 0.5)
        out = torch.bmm(F.softmax(scores + mask, dim=-1), x @ self.W_ov)
        return torch.cat([x, out], dim=-1) if self.concat else out


class ReducedTransformer(nn.Module):
    def __init__(self, v_sig, n_noise, seq_len):
        super().__init__()
        self.V = v_sig + n_noise
        self.seq_len = seq_len
        self.d1 = self.V + seq_len                  # 108
        self.d2 = 2 * self.d1                       # 216
        self.layer1 = AttentionBlock(self.d1, self.d1, concat=True)
        self.layer2 = AttentionBlock(self.d2, self.d1, concat=False)

    def forward(self, x):
        B, T = x.shape
        tok = F.one_hot(x, num_classes=self.V).float()
        pos = F.one_hot(torch.arange(T, device=x.device),
                        num_classes=self.seq_len).float().unsqueeze(0).expand(B, T, -1)
        s = torch.cat([tok, pos], dim=-1)
        # STRICTLY causal: diagonal=0 masks j == i too, so a position can never attend to
        # itself and W_qk does not have to spend a self-veto on it.
        mask = torch.triu(torch.ones(T, T, device=x.device) * float('-inf'), diagonal=0)
        mask[0, 0] = 0.0        # position 0 has no earlier key; without this its softmax is nan
        out = self.layer2(self.layer1(s, mask), mask)
        return out[..., :self.V]                   # the predicted distribution, as produced


# ------------------------------------------------------------- block sparsity and weight tying

def build_masks(model):
    """Boolean masks marking the only entries allowed to be nonzero. Everything else stays 0."""
    V, Vs, d1, d2 = model.V, CFG['v_sig'], model.d1, model.d2

    if CFG['blocks'] == 'full':
        return {n: torch.ones_like(p, dtype=torch.bool) for n, p in model.named_parameters()}
    if CFG['blocks'] != 'relevant':
        sys.exit(f"blocks must be 'relevant' or 'full', got {CFG['blocks']!r}")

    m = {}
    qk1 = torch.zeros(d1, d1, dtype=torch.bool)
    qk1[0:V, 0:V] = True                 # token x token  (tied into four quadrants)
    qk1[V:d1, V:d1] = True               # position x position, fully free
    qk1.diagonal()[V:d1] = False         # self-attention is -inf in the causal mask, so these
                                         # 100 cells have identically zero gradient
    m['layer1.W_qk'] = qk1

    # Layer 1's output lives entirely in its first V coordinates (W_ov1 is nonzero only on
    # [0:V, 0:V]), so columns d1+V .. d2 of layer 2's input are identically zero. Two of the
    # three blocks shrink further to their SIGNAL corner: noise tokens are never a state and
    # never a successor, so 'own token == key's state' and 'own state == key's state' are only
    # ever asked about signal tokens.
    qk2 = torch.zeros(d2, d2, dtype=torch.bool)
    qk2[0:V, 0:V] = True                            # A  noise suppression   V  x V  = 64
    qk2[0:Vs, d1:d1 + Vs] = True                    # B  induction head      Vs x Vs = 16
    qk2[d1:d1 + Vs, d1:d1 + Vs] = True              # C  L1 out x L1 out     Vs x Vs = 16
    m['layer2.W_qk'] = qk2

    ov1 = torch.zeros(d1, d1, dtype=torch.bool); ov1[0:V, 0:V] = True
    ov2 = torch.zeros(d2, d1, dtype=torch.bool); ov2[0:V, 0:V] = True
    m['layer1.W_ov'] = ov1
    m['layer2.W_ov'] = ov2
    return m


def build_ties(model):
    """Groups of entries forced to share one value.

    Returns {param_name: [(group_name, bool_mask), ...]}. Entries not in any group move freely.
    Every mask here is a subset of the corresponding block mask -- asserted in install().
    """
    V, Vs, d1, d2 = model.V, CFG['v_sig'], model.d1, model.d2
    if not CFG['tie'] or CFG['blocks'] != 'relevant':
        return {n: [] for n, _ in model.named_parameters()}

    def m1():
        return torch.zeros(d1, d1, dtype=torch.bool)

    def m2():
        return torch.zeros(d2, d2, dtype=torch.bool)

    t = {n: [] for n, _ in model.named_parameters()}

    # layer 1, token x token: four 4x4 quadrants, one value each  (64 cells -> 4 dof)
    for rname, rs in (('sig', slice(0, Vs)), ('noi', slice(Vs, V))):
        for cname, cs in (('sig', slice(0, Vs)), ('noi', slice(Vs, V))):
            g = m1(); g[rs, cs] = True
            t['layer1.W_qk'].append((f'L1.tok.q{rname}_k{cname}', g))

    # layer 2 A, noise suppression: left 8x4 and right 8x4  (64 cells -> 2 dof)
    for cname, cs in (('sig', slice(0, Vs)), ('noi', slice(Vs, V))):
        g = m2(); g[0:V, cs] = True
        t['layer2.W_qk'].append((f'L2.A.k{cname}', g))

    # layer 2 B and C: diagonal / off-diagonal, each block on its own  (16 cells -> 2 dof each)
    for bname, rs, cs in (('B.induction', slice(0, Vs), slice(d1, d1 + Vs)),
                          ('C.state',     slice(d1, d1 + Vs), slice(d1, d1 + Vs))):
        blk = m2(); blk[rs, cs] = True
        dg = m2(); dg[torch.arange(rs.start, rs.stop), torch.arange(cs.start, cs.stop)] = True
        t['layer2.W_qk'].append((f'L2.{bname}.diag', dg))
        t['layer2.W_qk'].append((f'L2.{bname}.off', blk & ~dg))
    return t


def to_device_masks(d, device):
    """Move every boolean mask onto the device ONCE.

    reapply_masks() and retie() run every step; calling .to(device) inside them copied the
    masks host->device 500 times per threshold for nothing.
    """
    if isinstance(next(iter(d.values())), list):          # tie groups
        return {k: [(g, m.to(device)) for g, m in v] for k, v in d.items()}
    return {k: v.to(device) for k, v in d.items()}


@torch.no_grad()
def retie(model, ties, device):
    """Set every tied group back to its own mean.

    A no-op when the invariant already holds -- which it does, because the members start equal
    (zero init) and receive identical gradients. This exists to mop up float drift and to make
    the invariant something the code enforces rather than something the reader has to trust.
    """
    params = dict(model.named_parameters())
    for name, groups in ties.items():
        if not groups:
            continue
        p = params[name]
        for _, gm in groups:                              # already on the device
            p[gm] = p[gm].mean()


def tie_gradient_hook(groups, device):
    """dL/dtheta for a tied group is the SUM of its members' gradients.

    If W[i] = theta for all i in G, then dL/dtheta = sum_{i in G} dL/dW[i]. Writing that sum
    back to every member makes each one take exactly the step the single parameter would, and
    keeps them identical. Under Adam sum vs mean barely matters (the update is scale-invariant
    per coordinate); under SGD it matters by a factor of |G|, so the sum is what is correct.
    """
    masks = [g.to(device) for _, g in groups]

    def hook(grad):
        if not masks:
            return grad
        out = grad.clone()
        for gm in masks:
            out[gm] = grad[gm].sum()
        return out
    return hook


@torch.no_grad()
def tie_spread(model, ties, device):
    """max |value - group mean| over every tied group -- 0 iff the invariant holds."""
    params = dict(model.named_parameters())
    worst, where = 0.0, None
    for name, groups in ties.items():
        for gname, g in groups:
            v = params[name][g if g.device == params[name].device else g.to(device)]
            d = float((v - v.mean()).abs().max()) if v.numel() else 0.0
            if d > worst:
                worst, where = d, f'{name}:{gname}'
    return worst, where


def install(model, masks, ties, device):
    """Initialise inside the masks, zero everything outside, pin gradients to mask and ties."""
    V = model.V
    params = dict(model.named_parameters())
    torch.manual_seed(SEED)

    for name, groups in ties.items():                 # ties must live inside their block
        for gname, g in groups:
            assert bool((g & ~masks[name]).any()) is False, f'{gname} escapes its block mask'

    with torch.no_grad():
        for name, p in params.items():
            p.zero_()
            mask = masks[name].to(device)
            if name.endswith('W_qk'):
                if CFG['init'] == 'normal':
                    s = p.shape[0] ** -0.5            # uniform(-s, s), fan_in ** -0.5
                    p.copy_(torch.empty_like(p).uniform_(-s, s) * mask)
                elif CFG['init'] != 'zeros':
                    sys.exit(f"init must be 'zeros' or 'normal', got {CFG['init']!r}")
            else:
                # W_ov: identity on the 8x8 selector, trained or not. Starting it at zero would
                # make f == 0 everywhere and kill every gradient.
                p[0:V, 0:V] = torch.eye(V, device=device)

    if CFG['init'] == 'normal':
        retie(model, ties, device)       # a random init breaks the ties; average them back

    trainable, names, n_free, n_dof = [], [], 0, 0
    for name, p in params.items():
        train_this = name.endswith('W_qk') or CFG['train_wov']
        p.requires_grad_(train_this)
        if not train_this:
            continue
        mask = masks[name].to(device)
        groups = ties.get(name, [])
        hook = tie_gradient_hook(groups, device)
        p.register_hook(lambda g, mask=mask, hook=hook: hook(g * mask))
        trainable.append(p)
        names.append(name)
        n_free += int(mask.sum())
        tied_cells = torch.zeros_like(mask, dtype=torch.bool)
        for _, g in groups:
            tied_cells |= g.to(device)
        n_dof += int((mask & ~tied_cells).sum()) + len(groups)
    return trainable, n_free, n_dof, names


@torch.no_grad()
def reapply_masks(model, masks):
    """Belt and braces: re-zero outside the masks after each step. Masks already on device."""
    for name, p in model.named_parameters():
        p.mul_(masks[name])


# ---------------------------------------------------------------------------------- snapshots

class BlockRecorder:
    """Per-step snapshots of only the entries that can move.

    Storing whole matrices costs 233 KB/row for the two W_qk together; the masks leave ~10,060
    free entries, so a row is 40 KB. Nothing is lost -- every other entry is identically zero.

    Per trainable parameter, per threshold:
        w_<param>_x_<thr>.npy      (rows, n_free)  float32   weights
        g_<param>_x_<thr>.npy      (rows, n_free)  float32   dL/dW          [if snap_grads]
        mask_<param>_x_<thr>.npy   full shape      bool      where they live
        snapmeta_x_<thr>.json                                shapes, order, rows, tie groups

    Reconstruct a full matrix from row k with   W = np.zeros(shape); W[mask] = w[k]

    Row convention: w[k] is the state BEFORE step k+1 (w[0] is the init) and g[k] is the
    gradient evaluated AT w[k], so w[k] -> w[k+1] is the optimizer's response to g[k].
    """

    def __init__(self, model, masks, ties, thr, n_rows, names, meta):
        d = run_dir()
        os.makedirs(d, exist_ok=True)
        dt = np.dtype(CFG['snap_dtype'])
        self.names = list(names)
        self.idx, self.w, self.g = {}, {}, {}
        self.meta_path = os.path.join(d, f'snapmeta_x_{thr}.json')
        shapes, nfree, tie_index = {}, {}, {}
        for n in self.names:
            m = masks[n].cpu().numpy()
            np.save(os.path.join(d, f'mask_{n}_x_{thr}.npy'), m)
            flat = np.flatnonzero(m.ravel())
            self.idx[n] = torch.from_numpy(flat)
            shapes[n], nfree[n] = list(m.shape), int(m.sum())
            # where each tie group lands inside the flattened row, so an analysis can group them
            pos = {int(v): i for i, v in enumerate(flat)}
            tie_index[n] = {gname: [pos[int(v)] for v in np.flatnonzero(g.cpu().numpy().ravel())]
                            for gname, g in ties.get(n, [])}
            self.w[n] = np.lib.format.open_memmap(
                os.path.join(d, f'w_{n}_x_{thr}.npy'), mode='w+', dtype=dt,
                shape=(n_rows, nfree[n]))
            if CFG['snap_grads']:
                self.g[n] = np.lib.format.open_memmap(
                    os.path.join(d, f'g_{n}_x_{thr}.npy'), mode='w+', dtype=dt,
                    shape=(n_rows, nfree[n]))
        self.nw = self.ng = 0
        self.meta_every = CFG['meta_every']
        self.meta = {**meta, 'params': self.names, 'shapes': shapes, 'n_free': nfree,
                     'tie_groups': tie_index, 'dtype': CFG['snap_dtype'],
                     'rows_allocated': n_rows, 'snap_grads': bool(CFG['snap_grads']),
                     'w_rows_written': 0, 'g_rows_written': 0, 'w_steps': [], 'g_steps': [],
                     'layout': 'each row holds the masked entries in row-major order; '
                               'W = zeros(shape); W[mask] = row',
                     'convention': 'w[k] is before step k+1 (w[0] = init); '
                                   'g[k] is the gradient at w[k]'}

    def _flush(self, force=False):
        """Rewrite the meta JSON. Doing this on every row cost ~30 ms/step and grew with the
        run, because w_steps/g_steps keep lengthening. Every META_EVERY rows is plenty -- a
        reader only needs it to know how far the run has got."""
        self.meta['w_rows_written'], self.meta['g_rows_written'] = self.nw, self.ng
        if not force and (self.nw + self.ng) % self.meta_every:
            return
        with open(self.meta_path, 'w') as fh:
            json.dump(self.meta, fh, indent=2)

    def _take(self, p, n):
        return p.detach().reshape(-1)[self.idx[n].to(p.device)].to('cpu', torch.float32).numpy()

    @torch.no_grad()
    def write_w(self, model, step):
        if self.nw >= self.meta['rows_allocated']:
            return
        params = dict(model.named_parameters())
        for n in self.names:
            self.w[n][self.nw] = self._take(params[n], n)
        self.meta['w_steps'].append(int(step))
        self.nw += 1
        self._flush()

    @torch.no_grad()
    def write_g(self, model, step):
        if not CFG['snap_grads'] or self.ng >= self.meta['rows_allocated']:
            return
        params = dict(model.named_parameters())
        for n in self.names:
            if params[n].grad is None:
                return
            self.g[n][self.ng] = self._take(params[n].grad, n)
        self.meta['g_steps'].append(int(step))
        self.ng += 1
        self._flush()

    def close(self):
        for a in list(self.w.values()) + list(self.g.values()):
            a.flush()
        self._flush(force=True)
        self.w, self.g = {}, {}

    def nbytes(self):
        n = np.dtype(CFG['snap_dtype']).itemsize
        per_row = sum(self.meta['n_free'].values()) * n
        return per_row * self.meta['rows_allocated'] * (2 if CFG['snap_grads'] else 1)


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


# ----------------------------------------------------------------------------------- training

def train_one(thr, device):
    wb = None
    if USE_WANDB:
        try:
            import wandb
            wb = wandb.init(project=WANDB_PROJECT, entity=WANDB_ENTITY,
                            name=f'{cell_tag()}_x_{thr}', group=cell_tag(),
                            config={**CFG, 'threshold': thr, 'seed': SEED}, reinit=True)
        except Exception as e:                       # a dead wandb must not kill a run
            print(f'    [wandb off: {e}]', flush=True)

    model = ReducedTransformer(CFG['v_sig'], CFG['n_noise'], CFG['seq_len']).to(device)
    masks = build_masks(model)
    ties = build_ties(model)
    params, n_free, n_dof, tnames = install(model, masks, ties, device)
    dmasks = to_device_masks(masks, device)        # moved once, used every step
    dties = to_device_masks(ties, device)

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
    # train_loss and test_loss are BOTH evaluated after the step, so history[k] is one set of
    # weights throughout. train_loss_pre is the value the backward pass happened to compute at
    # the PREVIOUS weights -- kept because it is free, and because folder 18 logged only that.
    history = {'step': [0], 'train_loss': [tr0], 'test_loss': [te0], 'train_loss_pre': [tr0]}
    best_train, best_test, best_state = tr0, te0, None

    n_all = sum(p.numel() for p in model.parameters())
    print(f'  alpha={thr}  free entries {n_free:,} of {n_all:,}   '
          f'degrees of freedom {n_dof:,}'
          + (f'  (tying folds {n_free - n_dof:,} of them away)' if CFG['tie'] else ''))

    rec = None
    if CFG['snapshots']:
        rec = BlockRecorder(model, masks, ties, thr, CFG['epochs'] + 1, tnames,
                            {'cell': cell_tag(), 'threshold': float(thr), 'seed': SEED,
                             'init': CFG['init'], 'blocks': CFG['blocks'], 'tie': CFG['tie'],
                             'readout': CFG['readout'], 'train_wov': CFG['train_wov']})
        rec.write_w(model, 0)                       # row 0 = the initialisation
        print(f'    snapshots: {tnames} every step, '
              f'{"weights + gradients" if CFG["snap_grads"] else "weights only"}, '
              f'{rec.nbytes() / 2**20:.0f} MB -> {run_dir()}')
    print(f'    init: train {tr0:.5f}  test {te0:.5f}', flush=True)

    t0 = time.time()
    stop_reason = None
    interrupted = False
    try:
        for ep in range(1, CFG['epochs'] + 1):
            tr_pre = full_batch_grad(model, tr_x, tr_y, b)     # loss at the CURRENT weights
            if rec is not None:
                rec.write_g(model, ep - 1)                     # gradient at w-row (ep-1)
            opt.step()
            reapply_masks(model, dmasks)
            if CFG['tie']:
                retie(model, dties, device)
            if rec is not None:
                rec.write_w(model, ep)

            # both measured AFTER the step -- this is the folder-18 logging bug, fixed
            tr_loss = eval_ce(model, tr_x, tr_y, device, b)
            te_loss = eval_ce(model, te_x, te_y, device, b)

            if CFG['check_ties'] and CFG['tie']:
                spread, where = tie_spread(model, dties, device)
                if spread > 1e-6:
                    stop_reason = f'tie broken at step {ep}: spread {spread:.3g} in {where}'
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
            best_train = min(best_train, tr_loss)
            if te_loss < best_test:
                best_test = te_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            if wb is not None:
                wb.log({'train_loss': tr_loss, 'test_loss': te_loss, 'train_loss_pre': tr_pre,
                        'best_train': best_train, 'best_test': best_test}, step=ep)
            every = CFG['log_every'] or max(1, CFG['epochs'] // 20)   # default: every 5%
            if ep % every == 0 or ep == CFG['epochs'] or ep == 1:
                print(f'    step {ep:>5}/{CFG["epochs"]}  train {tr_loss:.5f}  test {te_loss:.5f}'
                      f'  best_train {best_train:.5f}  best_test {best_test:.5f}', flush=True)
    except KeyboardInterrupt:
        interrupted = True
        print('\n    [interrupted -- saving what exists and moving to the next threshold]',
              flush=True)

    d = run_dir()
    os.makedirs(d, exist_ok=True)
    if rec is not None:
        rec.close()
    torch.save(model.state_dict(), os.path.join(d, f'model_final_seed{SEED}_x_{thr}.pth'))
    if best_state is not None:
        torch.save(best_state, os.path.join(d, f'model_best_seed{SEED}_x_{thr}.pth'))

    spread, where = tie_spread(model, dties, device) if CFG['tie'] else (0.0, None)
    out = {'cell': cell_tag(), 'threshold': thr, 'seed': SEED,
           'config': {**CFG, 'trainable_entries': n_free, 'degrees_of_freedom': n_dof,
                      'b_noise': noise_bias(thr) if CFG['readout'] == 'bias' else None},
           'history': history, 'best_train': best_train, 'best_test': best_test,
           'final_train': history['train_loss'][-1], 'final_test': history['test_loss'][-1],
           'steps_run': history['step'][-1], 'seconds': time.time() - t0,
           'tie_spread': spread, 'tie_spread_where': where,
           'stopped_early': stop_reason is not None or interrupted,
           'stop_reason': stop_reason or ('interrupted' if interrupted else None),
           'snapshots': None if rec is None else
           {'params': rec.names, 'w_rows_written': rec.meta['w_rows_written'],
            'g_rows_written': rec.meta['g_rows_written'], 'meta': f'snapmeta_x_{thr}.json'}}
    with open(metrics_path(thr), 'w') as fh:
        json.dump(out, fh, indent=2)
    if wb is not None:
        wb.summary.update({k: out[k] for k in ('best_train', 'best_test', 'final_train',
                                               'final_test', 'steps_run', 'tie_spread')})
        wb.finish()
    print(f'    done in {(time.time()-t0)/60:.1f}m  best_train {best_train:.5f}  '
          f'best_test {best_test:.5f}'
          + (f'  tie spread {spread:.2e}' if CFG['tie'] else '')
          + f'  -> {metrics_path(thr)}', flush=True)
    return out


def rebuild_summary():
    """Re-glob every metrics file on disk, so retraining a subset never shrinks the summary."""
    rows = []
    for p in sorted(glob.glob(os.path.join(CFG['runs_dir'], '*', 'metrics_seed*_x_*.json'))):
        try:
            rows.append(json.load(open(p)))
        except Exception:
            pass
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
    ap.add_argument('--init', choices=('zeros', 'normal'), default=CFG['init'])
    ap.add_argument('--blocks', choices=('relevant', 'full'), default=CFG['blocks'],
                    help="'relevant' = the block-sparse masks; 'full' = every entry free")
    ap.add_argument('--readout', choices=('none', 'bias'), default=CFG['readout'])
    ap.add_argument('--train-wov', action='store_true')
    ap.add_argument('--no-tie', action='store_true',
                    help='same masks, tying off -- reproduces folder 18')
    ap.add_argument('--check-ties', action='store_true',
                    help='assert every step that each tied group has zero spread')
    ap.add_argument('--no-snapshots', action='store_true')
    ap.add_argument('--no-snap-grads', action='store_true')
    ap.add_argument('--no-wandb', action='store_true')
    ap.add_argument('--thresholds', type=float, nargs='+', default=list(THRESHOLDS))
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
                'blocks': a.blocks, 'readout': a.readout, 'train_wov': a.train_wov,
                'device': pick_device(a.device), 'tie': not a.no_tie,
                'check_ties': a.check_ties, 'log_every': a.log_every,
                'grad_chunk': a.grad_chunk, 'eval_chunk': a.grad_chunk,
                'snapshots': CFG['snapshots'] and not a.no_snapshots,
                'snap_grads': CFG['snap_grads'] and not a.no_snap_grads})

    check_dataset()
    device = torch.device(CFG['device'])

    probe = ReducedTransformer(CFG['v_sig'], CFG['n_noise'], CFG['seq_len'])
    pm, pt = build_masks(probe), build_ties(probe)
    _, n_free, n_dof, _ = install(probe, pm, pt, torch.device('cpu'))
    snap_mb = ((CFG['epochs'] + 1) * n_free * np.dtype(CFG['snap_dtype']).itemsize
               * (2 if CFG['snap_grads'] else 1) / 2 ** 20) if CFG['snapshots'] else 0
    del probe

    print(f'cell       : {cell_tag()}')
    print(f'device     : {device}')
    print(f'thresholds : {list(THRESHOLDS)}')
    print(f'free cells : {n_free:,}   degrees of freedom : {n_dof:,}'
          + (f'   (tying folds {n_free - n_dof:,} away)' if CFG['tie'] else '   (tying OFF)'))
    if CFG['tie']:
        for nm, groups in pt.items():
            if groups:
                print(f'    {nm}: ' + ', '.join(g for g, _ in groups))
    print(f'batching   : FULL BATCH in chunks of {CFG["grad_chunk"]} -> 1 epoch = 1 step')
    print(f'logging    : every '
          f'{CFG["log_every"] or max(1, CFG["epochs"] // 20)} steps'
          f'{"" if CFG["log_every"] else " (5% of the run)"}, plus step 1')
    print(f'loss log   : train and test BOTH evaluated after the step '
          f'(train_loss_pre keeps the pre-step value)')
    if CFG['snapshots']:
        print(f'snapshots  : the {n_free:,} free cells every step, '
              f'{"weights + gradients" if CFG["snap_grads"] else "weights only"}  '
              f'~= {snap_mb:.0f} MB per threshold ({snap_mb * len(THRESHOLDS):.0f} MB total)')
    else:
        print('snapshots  : OFF')
    print()

    for thr in THRESHOLDS:
        train_one(thr, device)
    n = rebuild_summary()
    print(f'\nsummary rebuilt from {n} metrics files -> '
          f'{os.path.join(CFG["runs_dir"], "summary.json")}')


if __name__ == '__main__':
    main()
