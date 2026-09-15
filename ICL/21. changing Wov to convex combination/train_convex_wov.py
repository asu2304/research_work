"""Folder 21 -- W_ov as a CONVEX COMBINATION, W_qk trained, 3 x 2 sweep.

Continues the folder-19 flow: COMBINED W_qk (one matrix per layer, not a factored W_q/W_k),
zero init, Adam, full batch, and the folder-17/19 readout with no unembedding, no log and no
output softmax.

WHAT CHANGES FROM FOLDER 19
---------------------------
Folders 17-19 pin W_ov[:8,:8] = I_8. An identity row SELECTS: an attended token contributes its
own one-hot, so layer 2's output is the empirical successor frequency. Here that block is
replaced by a CONVEX COMBINATION -- every row non-negative and summing to 1, but mixing mass
across tokens instead of selecting one. Five blocks are implemented; the sweep runs the three
that train, and the two that cannot are kept as documented null controls.

    key       block                                  rank  distinct rows  trains
    --------  -------------------------------------  ----  -------------  ------
    smooth    (1-eps) I + eps * (1/8) J,  eps = 0.5     8         8        yes
    mix       circulant 2-term blend,   gamma = 0.3     8         8        yes
    pair      fixed pairwise averaging                  4         4        yes, lossy
    ruled     every row = uniform over the 4 signal     2         1        NO
    all       every row = uniform over all 8            2         1        NO

Written out (rows = attended token v, columns = output slot):

  smooth  eps=0.5                             mix  gamma=0.3
        o0   o1   o2   o3   o4 .. o7               o0  o1  o2  o3   o4  o5  o6  o7
  v=0 [ .56  .06  .06  .06  .06 .06 .06 .06 ] v=0 [ .7  .3   0   0    0   0   0   0 ]
  v=1 [ .06  .56  .06  .06  .06 .06 .06 .06 ] v=1 [  0  .7  .3   0    0   0   0   0 ]
  v=2 [ .06  .06  .56  .06  .06 .06 .06 .06 ] v=2 [  0   0  .7  .3    0   0   0   0 ]
  v=3 [ .06  .06  .06  .56  .06 .06 .06 .06 ] v=3 [ .3   0   0  .7    0   0   0   0 ]
  v=4 [ .06  .06  .06  .06  .56 .06 .06 .06 ] v=4 [  0   0   0   0   .7  .3   0   0 ]
   ..   (symmetric, doubly stochastic)        v=5 [  0   0   0   0    0  .7  .3   0 ]
                                              v=6 [  0   0   0   0    0   0  .7  .3 ]
                                              v=7 [  0   0   0   0   .3   0   0  .7 ]

  pair                                        ruled / all   (NULL CONTROLS)
        o0  o1  o2  o3  o4  o5  o6  o7              every row identical:
  v=0 [ .5  .5   0   0   0   0   0   0 ]      ruled: [ .25 .25 .25 .25  0  0  0  0 ]
  v=1 [ .5  .5   0   0   0   0   0   0 ]      all  : [ .125 x 8 ]
  v=2 [  0   0  .5  .5   0   0   0   0 ]
  v=3 [  0   0  .5  .5   0   0   0   0 ]
   ..   (likewise 4,5 and 6,7)

THE CONDITION FOR TRAINING -- DISTINCT ROWS
-------------------------------------------
The value path reads only the token one-hot, so an attended key contributes row v_k of the
block. Writing c for the attention-weighted count vector over tokens, f = C^T c exactly.

If ALL rows equal some u then f = sum_k softmax(s)_k * u = u * 1 = u, identically, for every
context and every attention pattern -- so dL/dW_qk = 0 and nothing moves. This is arithmetic,
not tuning: no learning rate, step count or init fixes it. Measured, alpha = 0.6:

    ov         ||dL/dW_qk2|| at init    CE step 0    CE step 60    verdict
    smooth               2.29e-03         2.11051       1.90909    TRAINS
    mix                  3.44e-03         2.08284       1.89063    TRAINS
    pair                 3.39e-03         2.13152       2.01023    TRAINS
    ruled                8.12e-11         2.05992       2.05992    frozen
    all                  1.93e-11         2.23414       2.23414    frozen
    identity (ref)              --        2.04045       1.74219    folder-19 baseline

The gradient moves eight orders of magnitude the moment the rows become distinct. Random init
does NOT rescue the null cells -- scrambling W_qk with N(0, 50) changes f by 2.8e-07, because
the attention is redistributing weight among identical vectors. `probe_degeneracy()` runs once
per threshold and prints a [!] DEGENERATE line if it is handed one of them.

    The frozen values are exactly the closed form for a constant output,
    CE = -[alpha*log p_sig + (1-alpha)*log p_noise] with p = (u+b)/sum(u+b):
    ruled at alpha 0.6 predicts 2.05931 and the run sits at 2.05937.

WHAT THE BLEND COSTS, AND WHY IT IS NOT ZERO
--------------------------------------------
f = C^T c is a linear image of the counts, so an INVERTIBLE C destroys no information. But the
readout is fixed -- there is no trainable matrix to undo the mixing -- so the set of outputs
the model can reach is conv(rows of C), a sub-simplex of the full one. The CE floor therefore
sits above the identity's by whatever that shrinkage costs. At alpha = 0.6, step 60:

    mix      +0.148 nats        full rank
    smooth   +0.167 nats        full rank
    pair     +0.268 nats        rank 4 -- f says which PAIR was attended, never which member

The two full-rank blends cost about the same and largely duplicate each other. `pair` is the
informative one: it is the only block that actually destroys information, and its extra
~0.12 nats is the price of collapsing eight tokens onto four directions. (These are step-60
numbers on a still-falling curve -- the identity reference is 1.742 here against 1.698 at
convergence -- so treat them as an upper bound on the converged gap.)

Both blends are tunable: --eps for smooth, --gamma for mix. smooth is invertible for every
eps < 1; mix is invertible whenever gamma != 0.5.

THE SECOND AXIS -- WHICH W_qk ENTRIES MAY MOVE
----------------------------------------------
    --qk-mask full        every entry of both W_qk matrices. 11,664 + 46,656 = 58,320 cells.

    --qk-mask relevant    layer 1:  [0:8, 0:8]        token x token          64
                                    [8:108, 8:108]    position x position    10,000
                          layer 2:  [0:8, 0:8]        A, noise suppression    64
                                    [0:8, 108:116]    B, induction head       64
                                    [108:116, 108:116] C, state x state       64
                                                                     total  10,256 cells

    Note these are the FULL 8x8 blocks for layer 2, not the 4x4 signal corners folder 18 cut
    them down to. The 100 diagonal cells of layer 1's position block are inside the mask but
    have identically zero gradient -- a strictly causal mask makes self-attention -inf -- so
    they stay at their zero init and cost nothing.

THE SWEEP
---------
`--ov sweep` (the default) covers the three TRAINABLE blocks; `--qk-mask sweep` (also the
default) covers both masks. So the default run is 3 x 2 = 6 cells, each over every threshold:

    smooth x full      mix x full      pair x full
    smooth x relevant  mix x relevant  pair x relevant

At 3 thresholds and 500 steps that is 18 runs, roughly 2 hours on a free GPU and ~15 MB on
disk. The null controls are NOT in the sweep -- ask for them by name, e.g. --ov ruled.

Each cell writes its own runs/<tag>/ directory, and every cell's metrics are re-globbed into
runs/summary.json at the end, so re-running one cell never shrinks the summary.

ARCHITECTURE (otherwise unchanged from folder 19)
-------------------------------------------------
    x            one-hot token (8) + one-hot position (100)        -> 108 = d1
    layer 1      attention, returns cat[x, attn_out]               -> 216 = d2
    layer 2      attention, returns attn_out only                  -> 108
    f            layer2 attn_out[..., :8]
    loss         CE on (f + b) / sum(f + b), tokens [50:100)

    scores = x W_qk x^T / sqrt(output_dim) + causal mask       COMBINED parametrisation

Strictly causal mask, triu(diagonal=0), with mask[0,0] reopened because position 0 has no
earlier key and a softmax over a row of -inf is nan. Full batch, 1 epoch = 1 optimizer step,
checkpoint on best TEST loss, Ctrl+C saves and moves to the next threshold.

ON DISK: model_best / model_final .pth and metrics_*.json only. NO per-step weight or gradient
snapshots -- those are off in this folder until asked for.

Usage:
    python train_convex_wov.py                       # the 3 x 2 sweep, 6 cells
    python train_convex_wov.py --ov pair             # the lossy rank-4 block, both masks
    python train_convex_wov.py --ov mix --gamma 0.45 # a harder blend (gamma -> 0.5 is singular)
    python train_convex_wov.py --ov smooth --eps 0.2 # a gentler blend
    python train_convex_wov.py --qk-mask relevant    # all three blocks, one mask
    python train_convex_wov.py --ov ruled            # a null control: prints [!] DEGENERATE

Full configuration surface:
    --ov {smooth,mix,pair,ruled,all,sweep}   which convex block  (default sweep)
    --qk-mask {full,relevant,sweep}          which W_qk entries  (default sweep)
    --gamma FLOAT   mix blend weight, default 0.3     --eps FLOAT  smooth blend, default 0.5
    --optimizer {adam,sgd}  --lr FLOAT  --epochs INT  --init {zeros,normal}
    --thresholds F [F ...]  --device STR  --runs-dir STR  --log-every INT  --grad-chunk INT
    --no-wandb
"""

import os
import sys
import glob
import json
import math
import time
import argparse
import tempfile
import itertools

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
THRESHOLDS = (0.6, 0.8, 1.0)

WANDB_PROJECT = 'convex_wov'
WANDB_ENTITY = None
USE_WANDB = True

OV_KINDS = ('smooth', 'mix', 'pair', 'ruled', 'all')   # 'ruled'/'all' are the
                                       # untrainable null controls (equal rows)
QK_MASKS = ('full', 'relevant')

CFG = {
    'data_dir':  _DATA_DIR,
    'runs_dir':  os.path.join(_SCRIPT_DIR, 'runs'),

    # --- task ---
    'v_sig':       4,          # signal tokens 0..3   ("ruled" -- governed by the chain)
    'n_noise':     4,          # noise tokens 4..7
    'seq_len':     100,
    'token_start': 50,         # loss and eval on tokens [50:100)

    # --- the two sweep axes ---
    'ov':          'ruled',    # 'ruled' | 'all'
    'qk_mask':     'relevant', # 'full'  | 'relevant'

    # --- optimisation ---
    'optimizer':    'adam',
    'lr':           1.0,       # folder 19's value: combined W_qk, zero init, full batch
    'weight_decay': 0.0,
    'epochs':       500,       # == optimizer steps (full batch)
    'init':         'zeros',   # 'zeros' | 'normal'   (W_qk only)

    # --- readout (fixed, never trained) ---
    'b_sig':        0.01,
    'noise_floor':  1e-9,      # b_noise at alpha=1, where (1-alpha)/(n_noise*alpha) = 0

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
    return (f"cvx_ov-{CFG['ov']}_qk-{CFG['qk_mask']}_opt-{CFG['optimizer']}"
            f"_lr-{CFG['lr']}_init-{CFG['init']}_ep-{CFG['epochs']}")


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

    Under ov='ruled' this is exactly the folder-19 situation: f's noise slots are identically
    zero, so b_noise carries all of the noise probability. Under ov='noise' the model can also
    place mass there itself, so the same number acts as a floor instead.
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


# ------------------------------------------------------------------ the convex-combination OV

def ov_block(kind, v_sig=4, n_noise=4, gamma=0.3, eps=0.5):
    """The 8x8 value/output block. Row v = what an attended token v contributes.

    Every variant here is a CONVEX COMBINATION: each row is non-negative and sums to 1, so
    layer 2's output f = sum_k softmax(s)_k * (row v_k) is a convex combination of probability
    vectors, hence f >= 0 and sum(f) = 1 exactly, and the readout stays (f + b)/sum(f + b)
    with no unembedding, no log and no output softmax.

    THE CONDITION FOR TRAINING AT ALL. The value path reads only the token one-hot, so an
    attended key contributes row v_k. Writing c for the attention-weighted count vector over
    tokens, f = C^T c exactly. If two rows of C are equal, the two tokens are indistinguishable
    in f; if ALL rows are equal to u then f = u for every context, dL/dW_qk = 0 and nothing
    moves. So the rows must be DISTINCT. Rank tells you how much survives: full rank loses
    nothing, rank r < 8 collapses the token space onto r directions.

    Null controls (rows all equal -> provably untrainable, kept to document the negative):
        'ruled'   every row = [1/4, 1/4, 1/4, 1/4, 0, 0, 0, 0]
        'all'     every row = [1/8] * 8

    Trainable convex combinations, none of them the identity or a scalar multiple of it:
        'smooth'  C = (1-eps) I + eps * (1/8) J     -- each token contributes eps of the global
                  average and (1-eps) of itself. Eigenvalues (1-eps) and 1, so invertible for
                  every eps < 1, and doubly stochastic. eps = 0.5 by default: half blend.
        'mix'     a circulant two-term blend INSIDE each group -- row v puts (1-gamma) on
                  itself and gamma on the next token of its own group (signal wraps 0->1->2->
                  3->0, noise wraps 4->5->6->7->4). Doubly stochastic, invertible whenever
                  gamma != 0.5. Every token contributes to exactly two slots.
        'pair'    tokens are averaged in fixed PAIRS: rows 0,1 both give (e0+e1)/2; rows 2,3
                  give (e2+e3)/2; likewise 4,5 and 6,7. Rank 4, deliberately lossy -- f can
                  say which PAIR was attended but not which member, so the successor is only
                  ever resolved to a pair. The interesting middle case: it trains, but it
                  cannot reach the identity's floor.

    NOTE what invertibility does and does not buy. f = C^T c is a linear image of the counts,
    so an invertible C destroys no information -- but the readout is FIXED, with no trainable
    matrix to undo the mixing. The set of outputs the model can reach is conv(rows of C), a
    sub-simplex of the full one. So these variants train, and their CE floor is above the
    identity's by however much that shrinkage costs. Measuring that cost is the point.
    """
    V = v_sig + n_noise
    M = torch.zeros(V, V)
    if kind == 'ruled':
        M[:, :v_sig] = 1.0 / v_sig
    elif kind == 'all':
        M[:, :] = 1.0 / V
    elif kind == 'smooth':
        M = (1.0 - eps) * torch.eye(V) + eps / V
    elif kind == 'mix':
        for lo, hi in ((0, v_sig), (v_sig, V)):
            n = hi - lo
            for i in range(n):
                M[lo + i, lo + i] = 1.0 - gamma
                M[lo + i, lo + (i + 1) % n] += gamma
    elif kind == 'pair':
        for lo in range(0, V, 2):
            M[lo:lo + 2, lo:lo + 2] = 0.5
    else:
        sys.exit(f"ov must be one of {OV_KINDS}, got {kind!r}")
    rs = M.sum(1)
    assert bool((M >= 0).all()) and torch.allclose(rs, torch.ones(V)), \
        f'W_ov block for {kind!r} must be non-negative with row sums 1: {rs.tolist()}'
    return M


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

class AttentionBlock(nn.Module):
    """One causal attention layer in COMBINED W_qk / W_ov form -- the folder-19 parametrisation.

    scores = (x W_qk x^T) / sqrt(output_dim) + causal mask,  out = softmax(scores) (x W_ov).

    W_ov is registered as a buffer, not a parameter: it is a fixed construction here, and a
    buffer makes that structural rather than a requires_grad flag someone can flip by accident.
    It still travels in the state dict, so a checkpoint records which variant produced it.
    """

    def __init__(self, input_dim, output_dim, concat):
        super().__init__()
        self.output_dim = output_dim
        self.concat = concat
        self.W_qk = nn.Parameter(torch.zeros(input_dim, input_dim))
        self.register_buffer('W_ov', torch.zeros(input_dim, output_dim))

    def forward(self, x, mask):
        scores = torch.bmm(x @ self.W_qk, x.transpose(1, 2)) / (self.output_dim ** 0.5)
        out = torch.bmm(F.softmax(scores + mask, dim=-1), x @ self.W_ov)
        return torch.cat([x, out], dim=-1) if self.concat else out


class ConvexOVTransformer(nn.Module):
    def __init__(self, v_sig, n_noise, seq_len, ov_kind):
        super().__init__()
        self.V = v_sig + n_noise
        self.v_sig, self.n_noise = v_sig, n_noise
        self.seq_len = seq_len
        self.d1 = self.V + seq_len                  # 108
        self.d2 = 2 * self.d1                       # 216
        self.layer1 = AttentionBlock(self.d1, self.d1, concat=True)
        self.layer2 = AttentionBlock(self.d2, self.d1, concat=False)
        self.set_ov(ov_kind)

    @torch.no_grad()
    def set_ov(self, kind):
        """Write the block into BOTH layers' W_ov. Every other cell of W_ov is zero.

        So the position coordinates are never carried through the value path, as in folders
        17-19. Both layers get the SAME averaging block, so layer 1's output is the constant u
        too -- it stops carrying the previous signal token's identity, which is what layer 2's
        induction match reads. That is the second way this configuration is a null model.
        """
        V = self.V
        M = ov_block(kind, self.v_sig, self.n_noise)
        self.layer1.W_ov.zero_()
        self.layer2.W_ov.zero_()
        self.layer1.W_ov[:V, :V] = M
        self.layer2.W_ov[:V, :V] = M
        self.ov_kind = kind

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
        return out[..., :self.V]                   # already a distribution -- see the docstring


# -------------------------------------------------------------------------------- W_qk masks

def build_masks(model):
    """Boolean masks marking the only W_qk entries allowed to be nonzero.

    'full'      every entry of both matrices: 11,664 + 46,656 = 58,320.
    'relevant'  layer 1  [0:8, 0:8]          token x token           64
                         [8:108, 8:108]      position x position     10,000
                layer 2  [0:8, 0:8]          A, noise suppression    64
                         [0:8, 108:116]      B, induction head       64
                         [108:116, 108:116]  C, state x state        64
                                                            total    10,256

    Layer 2's blocks are the FULL 8x8, not the 4x4 signal corners of folder 18. The 100
    diagonal cells of layer 1's position block sit inside the mask but have identically zero
    gradient under a strictly causal mask, so they never leave zero.
    """
    V, d1, d2 = model.V, model.d1, model.d2
    if CFG['qk_mask'] == 'full':
        return {'layer1.W_qk': torch.ones(d1, d1, dtype=torch.bool),
                'layer2.W_qk': torch.ones(d2, d2, dtype=torch.bool)}
    if CFG['qk_mask'] != 'relevant':
        sys.exit(f"qk-mask must be 'full' or 'relevant', got {CFG['qk_mask']!r}")

    qk1 = torch.zeros(d1, d1, dtype=torch.bool)
    qk1[0:V, 0:V] = True                            # token x token
    qk1[V:d1, V:d1] = True                          # position x position, all 100x100
    qk2 = torch.zeros(d2, d2, dtype=torch.bool)
    qk2[0:V, 0:V] = True                            # A  noise suppression
    qk2[0:V, d1:d1 + V] = True                      # B  induction head
    qk2[d1:d1 + V, d1:d1 + V] = True                # C  state x state
    return {'layer1.W_qk': qk1, 'layer2.W_qk': qk2}


def install(model, masks, device):
    """Initialise W_qk inside the masks, zero outside, and pin every gradient to the mask.

    W_ov is a buffer, so it carries no gradient at all and needs no freezing flag -- it is not
    in named_parameters() and the optimizer never sees it.
    """
    torch.manual_seed(SEED)
    trainable, names, n_free = [], [], 0
    with torch.no_grad():
        for name, p in model.named_parameters():
            p.zero_()
            mask = masks[name].to(device)
            if CFG['init'] == 'normal':
                s = p.shape[0] ** -0.5                # uniform(-s, s), fan_in ** -0.5
                p.copy_(torch.empty_like(p).uniform_(-s, s) * mask)
            elif CFG['init'] != 'zeros':
                sys.exit(f"init must be 'zeros' or 'normal', got {CFG['init']!r}")
    for name, p in model.named_parameters():
        mask = masks[name].to(device)
        p.requires_grad_(True)
        p.register_hook(lambda g, mask=mask: g * mask)
        trainable.append(p)
        names.append(name)
        n_free += int(mask.sum())
    return trainable, n_free, names


def to_device_masks(d, device):
    return {k: v.to(device) for k, v in d.items()}


@torch.no_grad()
def reapply_masks(model, masks):
    """Belt and braces: re-zero outside the masks after each step. Masks already on device."""
    for name, p in model.named_parameters():
        p.mul_(masks[name])


# --------------------------------------------------------------------------------------- loss

def ce_loss(p, y, b):
    """Cross entropy on p, the V-vector layer 2 puts out.

    p <- (p + b)/sum(p + b). No log, no softmax, no unembedding: the old log->softmax pair was
    exactly this division, since softmax(log x) = x/sum(x).

    The load-bearing property is f >= 0, which a non-negative W_ov gives. Both variants here
    are also row-stochastic, so sum(f) = 1 exactly and f is a genuine distribution. With b > 0,
    p is strictly in (0,1), the loss is >= 0 and LOG_FLOOR never binds.
    """
    p = p + b
    s = p.sum(-1, keepdim=True)
    p = p / torch.where(s.abs() < LOG_FLOOR, torch.ones_like(s), s)
    q = p.gather(-1, y.unsqueeze(-1)).squeeze(-1)
    return -torch.log(q.clamp_min(LOG_FLOOR)).mean()


@torch.no_grad()
def eval_ce(model, x, y, device, b):
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


def full_batch_grad(model, x, y, b):
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
def probe_degeneracy(model, x, device, n=256):
    """Does layer 2's output actually depend on the context? Returns the spread of f.

    The value path reads only the token one-hot (W_ov's position rows are zero), so an attended
    key contributes row v_k of the 8x8 block. If all 8 rows are EQUAL to some u, then

        f = sum_k softmax(s)_k * u = u,

    identically, for every context and every attention pattern -- and therefore dL/dW_qk = 0.
    Any W_ov whose rows do not differ destroys token identity and makes the model a constant
    predictor. This is cheap, so it runs once per threshold and says so out loud instead of
    letting an hour of sweep discover it.
    """
    model.eval()
    f = model(x[:n].to(device))[:, CFG['token_start']:CFG['seq_len']].reshape(-1, model.V)
    return float((f - f.mean(0)).abs().max())


@torch.no_grad()
def check_output_cone(model, x, device, b, n=512):
    """Assert what the readout actually rests on: f >= 0, and sum(f) <= 1.

    Non-negativity is the load-bearing half -- with b > 0 it is what makes (f+b)/sum(f+b) a
    probability and the loss a real cross entropy. Both blocks in this folder are also
    row-stochastic, so sum(f) should come back as exactly 1; the check allows anything up to 1
    so that a future sub-stochastic block (zeroed rows rather than uniform ones) still passes.
    Cheap, run once per threshold, so a future edit that breaks non-negativity fires here
    instead of quietly producing a loss that is not a CE.
    """
    model.eval()
    f = model(x[:n].to(device))[:, CFG['token_start']:CFG['seq_len']]
    smin, ssum = float(f.min()), f.sum(-1)
    lo, hi = float(ssum.min()), float(ssum.max())
    assert smin >= -1e-6 and hi <= 1.0 + 1e-5, \
        f'layer 2 output left the cone: min {smin:.3g}, max sum {hi:.6g}'
    return smin, lo, hi


# ----------------------------------------------------------------------------------- training

def train_one(thr, device):
    wb = None
    if USE_WANDB:
        try:
            import wandb
            wb = wandb.init(project=WANDB_PROJECT, entity=WANDB_ENTITY,
                            name=f'{cell_tag()}_x_{thr}', group=cell_tag(),
                            config={**CFG, 'threshold': thr, 'seed': SEED,
                                    'b_noise': noise_bias(thr)}, reinit=True)
        except Exception as e:                       # a dead wandb must not kill a run
            print(f'    [wandb off: {e}]', flush=True)

    model = ConvexOVTransformer(CFG['v_sig'], CFG['n_noise'], CFG['seq_len'],
                                CFG['ov']).to(device)
    masks = build_masks(model)
    params, n_free, tnames = install(model, masks, device)
    dmasks = to_device_masks(masks, device)        # moved once, used every step

    tr = load_data('train', thr)
    te = load_data('test', thr)
    tr_x, tr_y = tr['x'].to(device), tr['y'].to(device)
    te_x, te_y = te['x'].to(device), te['y'].to(device)
    b = readout_bias(thr, device)

    if CFG['optimizer'] == 'adam':
        opt = torch.optim.Adam(params, lr=CFG['lr'], weight_decay=CFG['weight_decay'])
    elif CFG['optimizer'] == 'sgd':
        opt = torch.optim.SGD(params, lr=CFG['lr'], weight_decay=CFG['weight_decay'])
    else:
        sys.exit(f"optimizer must be 'adam' or 'sgd', got {CFG['optimizer']!r}")

    smin, slo, shi = check_output_cone(model, tr_x, device, b)
    spread = probe_degeneracy(model, tr_x, device)
    tr0 = eval_ce(model, tr_x, tr_y, device, b)
    te0 = eval_ce(model, te_x, te_y, device, b)
    history = {'step': [0], 'train_loss': [tr0], 'test_loss': [te0], 'train_loss_pre': [tr0]}
    best_train, best_test, best_state = tr0, te0, None

    n_all = sum(p.numel() for p in model.parameters())
    print(f'  alpha={thr}  W_qk free {n_free:,} of {n_all:,}   b_noise {noise_bias(thr):.6f}   '
          f'f >= 0 (min {smin:+.2e}), sum(f) in [{slo:.4f}, {shi:.4f}]')
    if spread < 1e-5:
        print(f'    [!] DEGENERATE: max|f - mean f| = {spread:.2e} -- layer 2\'s output is the '
              f'SAME vector for\n        every context, because every row of the W_ov block is '
              f'equal, so the token\n        identity is destroyed. dL/dW_qk is 0 and nothing '
              f'will train. See ov_block().', flush=True)
    else:
        print(f'    f varies with context: max|f - mean f| = {spread:.3f}')
    print(f'    init: train {tr0:.5f}  test {te0:.5f}', flush=True)

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

            # A negative or non-finite CE would mean the output stopped being a distribution.
            # Row-stochastic W_ov makes that impossible, so this is a tripwire, not a policy.
            if not math.isfinite(tr_loss) or tr_loss < 0:
                stop_reason = (f'loss not a valid CE at step {ep}: train={tr_loss:.6g} -- '
                               f'layer 2 stopped emitting a distribution')
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
                wb.log({'train_loss': tr_loss, 'test_loss': te_loss,
                        'best_train': best_train, 'best_test': best_test}, step=ep)
            every = CFG['log_every'] or max(1, CFG['epochs'] // 20)   # default: every 5%
            if ep % every == 0 or ep == CFG['epochs'] or ep == 1:
                print(f'    step {ep:>5}/{CFG["epochs"]}  train {tr_loss:.5f}  '
                      f'test {te_loss:.5f}  best_train {best_train:.5f}  '
                      f'best_test {best_test:.5f}', flush=True)
    except KeyboardInterrupt:
        interrupted = True
        print('\n    [interrupted -- saving what exists and moving to the next threshold]',
              flush=True)

    d = run_dir()
    os.makedirs(d, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(d, f'model_final_seed{SEED}_x_{thr}.pth'))
    if best_state is not None:
        torch.save(best_state, os.path.join(d, f'model_best_seed{SEED}_x_{thr}.pth'))

    out = {'cell': cell_tag(), 'threshold': thr, 'seed': SEED,
           'config': {**CFG, 'trainable_entries': n_free, 'b_noise': noise_bias(thr),
                      'ov_block': ov_block(CFG['ov'], CFG['v_sig'], CFG['n_noise']).tolist()},
           'history': history, 'best_train': best_train, 'best_test': best_test,
           'final_train': history['train_loss'][-1], 'final_test': history['test_loss'][-1],
           'steps_run': history['step'][-1], 'seconds': time.time() - t0,
           'stopped_early': stop_reason is not None or interrupted,
           'stop_reason': stop_reason or ('interrupted' if interrupted else None)}
    with open(metrics_path(thr), 'w') as fh:
        json.dump(out, fh, indent=2)
    if wb is not None:
        wb.summary.update({k: out[k] for k in ('best_train', 'best_test',
                                               'final_train', 'final_test')})
        wb.finish()
    print(f'    done in {(time.time()-t0)/60:.1f}m  best_train {best_train:.5f}  '
          f'best_test {best_test:.5f}  -> {metrics_path(thr)}', flush=True)
    return out


def rebuild_summary():
    """Re-glob every metrics file on disk, so re-running one cell never shrinks the summary."""
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
    ap.add_argument('--ov', choices=OV_KINDS + ('sweep',), default='sweep',
                    help="convex W_ov block: 'smooth'/'mix'/'pair' train; 'ruled'/'all' have "
                         "equal rows and provably cannot. Default sweeps the trainable three")
    ap.add_argument('--gamma', type=float, default=0.3, help="blend weight for --ov mix")
    ap.add_argument('--eps', type=float, default=0.5, help="blend weight for --ov smooth")
    ap.add_argument('--qk-mask', choices=('full', 'relevant', 'sweep'), default='sweep',
                    help="'full' = every W_qk entry; 'relevant' = the five blocks. "
                         "Default sweeps both")
    ap.add_argument('--optimizer', choices=('adam', 'sgd'), default=CFG['optimizer'])
    ap.add_argument('--lr', type=float, default=CFG['lr'])
    ap.add_argument('--epochs', type=int, default=CFG['epochs'])
    ap.add_argument('--init', choices=('zeros', 'normal'), default=CFG['init'])
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
    CFG.update({'optimizer': a.optimizer, 'lr': a.lr,
                'epochs': a.epochs, 'init': a.init, 'device': pick_device(a.device),
                'log_every': a.log_every, 'grad_chunk': a.grad_chunk,
                'eval_chunk': a.grad_chunk})
    if a.runs_dir:
        CFG['runs_dir'] = os.path.abspath(a.runs_dir)

    ovs = ['smooth', 'mix', 'pair'] if a.ov == 'sweep' else [a.ov]
    qks = list(QK_MASKS) if a.qk_mask == 'sweep' else [a.qk_mask]
    cells = list(itertools.product(ovs, qks))

    check_dataset(THRESHOLDS)
    device = torch.device(CFG['device'])

    print(f'sweep      : {len(cells)} cell(s) x {len(THRESHOLDS)} threshold(s) '
          f'= {len(cells) * len(THRESHOLDS)} runs')
    print(f'device     : {device}')
    print(f'thresholds : {list(THRESHOLDS)}')
    print(f'params     : COMBINED W_qk (one per layer), zero init, {CFG["optimizer"]} '
          f'lr {CFG["lr"]}, {CFG["epochs"]} full-batch steps')
    print(f'W_ov       : FROZEN construction (a buffer, not a parameter), same block in '
          f'BOTH layers, every other cell zero')
    print(f'readout    : (f + b) / sum(f + b), b FIXED from alpha -- no unembedding, no log, '
          f'no output softmax')
    print(f'on disk    : model_best / model_final .pth and metrics_*.json only. '
          f'NO weight or gradient snapshots.')
    print()

    for ov, qk in cells:
        CFG.update({'ov': ov, 'qk_mask': qk})
        probe = ConvexOVTransformer(CFG['v_sig'], CFG['n_noise'], CFG['seq_len'], ov)
        pm = build_masks(probe)
        _, n_free, _ = install(probe, pm, torch.device('cpu'))
        M = ov_block(ov, CFG['v_sig'], CFG['n_noise'])
        del probe

        print('=' * 100)
        print(f'CELL  ov={ov}  qk-mask={qk}   ->  {cell_tag()}')
        print(f'  W_ov 8x8 block (rows = attended token, cols = output slot):')
        for v in range(CFG['v_sig'] + CFG['n_noise']):
            kind = 'sig ' if v < CFG['v_sig'] else 'noi '
            print('      ' + kind + f'v={v}  [' +
                  ' '.join(f'{x:5.2f}' for x in M[v].tolist()) + ' ]')
        print(f'  W_qk trainable entries: {n_free:,}')
        print('=' * 100, flush=True)

        for thr in THRESHOLDS:
            train_one(thr, device)

    n = rebuild_summary()
    print(f'\nsummary rebuilt from {n} metrics files -> '
          f'{os.path.join(CFG["runs_dir"], "summary.json")}')


if __name__ == '__main__':
    main()
