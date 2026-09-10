"""Master training script -- minimal two-layer QK/OV model, block-sparse, no unembedding.

Written from scratch; nothing is imported from folders 6/7/8/14/17. Hyperparameters that carry
over from those scripts are kept identical. The unembedding, the log and the final softmax are
gone for good; the readout bias is optional and is what `--readout` selects.

ARCHITECTURE
------------
    x            one-hot token (8) + one-hot position (100)        -> 108
    layer 1      attention, returns cat[x, attn_out]               -> 216
    layer 2      attention, returns attn_out only                  -> 108
    f = layer2 attn_out[..., :8]                                   -> the raw count vector

With W_ov[:8,:8] = I frozen, f is a convex combination of token one-hots (attention weights sum
to 1), so f >= 0 and sum(f) = 1 exactly -- f already IS a distribution. That is what lets the
unembedding, the log and the output softmax go: there is nothing left for them to do.

READOUT -- `--readout {none,bias}`
----------------------------------
    'none'   p = f / sum(f)                  (the division is a no-op when W_ov is frozen)
    'bias'   p = (f + b) / sum(f + b)        b FIXED from alpha, never trained   [default]

No log and no softmax in either. They were never doing any work: softmax(log(x)) = x / sum(x)
exactly, so the old log->softmax pair IS this division. Measured difference: 5.6e-17.

But the BIAS was doing two jobs, and dropping it cost ~0.2 nats:

  b_noise = (1-alpha)(1 + V_sig*b_sig) / (n_noise*alpha)
      A good induction head drives the NOISE slots of f to zero -- that is what the noise filter
      is. But noise really occurs (1-alpha) of the time, so under 'none' a noise target lands on
      f = 0 and costs -log(LOG_FLOOR) = 69 nats. 'none' therefore FORBIDS the noise filter: GD is
      forced to park (1-alpha) of its attention on noise positions (measured signal_mass 0.598 at
      alpha=0.6, versus 1.0 for the construction), so one attention distribution has to estimate
      the successor counts AND carry the noise mass. The bias supplies that mass as a known
      constant instead, freeing the whole attention budget for the induction task.

  b_sig = 0.01
      Pseudo-count smoothing the signal slots, so a successor the context never happened to show
      is not assigned probability exactly 0.

Measured at alpha=0.6, perfect induction head: 'none' 28.33  /  'bias' 1.6936  (ideal-alpha
1.6434). Trained end to end, 'none' plateaus around 1.847.

Because b > 0 and f >= 0, 'bias' makes p strictly inside (0,1): the loss is bounded below by 0
and LOG_FLOOR never binds (measured margin ~1e28). Under 'none' the floor is load-bearing --
it binds on ~0.098% of positions at alpha=0.6.

CAVEAT -- both of those guarantees need f >= 0, i.e. W_ov FROZEN to the identity. With
--train-wov, f is a convex combination of arbitrary vectors: it can go negative, sum(f+b) can
pass through zero, and -log p becomes unbounded below. That is not hypothetical -- it produced
best_train = -1.05 before the guard in train_one() existed. A small positive b does NOT rescue
it (b_sig = 0.01 cannot lift f = -0.91 above zero). If you want a trainable value path, the fix
is to constrain W_ov's rows to be distributions, not to add a bias.

BLOCK SPARSITY -- the point of this folder
------------------------------------------
Only these blocks are trainable. Every other entry of every W is exactly zero, at init and forever
(gradients are masked, and the masks are re-applied after each step):

    W_qk layer 1   (108 x 108)   [0:8,   0:8]        token   x token
                                 [8:108, 8:108]      position x position
    W_qk layer 2   (216 x 216)   [0:8,     0:8]      noise suppression
                                 [0:8,   108:116]    induction head  (own token x prev token)
                                 [108:116, 108:116]  layer-1 output x layer-1 output
    W_ov layer 1   (108 x 108)   [0:8,   0:8]        input selector      (only if TRAIN_WOV)
    W_ov layer 2   (216 x 108)   [0:8,   0:8]        input selector      (only if TRAIN_WOV)

    TRAIN_WOV = False -> both W_ov top-left blocks are the identity, frozen. That reproduces the
                         hand construction's value path: attention copies the token one-hot.
    TRAIN_WOV = True  -> those two 8x8 blocks are trainable, starting from the identity.

Layer 2's three blocks are all V x V because layer 1's output occupies only its first V
coordinates -- columns 116..216 of layer 2's input are identically zero, so nothing there could
ever matter. Those 192 cells are exactly the hand construction's layer-2 support.

Trainable parameter count: 8*8 + 100*100 (layer 1 QK) + 3 * 8*8 (layer 2 QK)
                         = 64 + 10,000 + 192 = 10,256   (+128 if TRAIN_WOV)
against 11,664 + 46,656 + 11,664 + 23,328 = 93,312 entries in the four matrices -- 11.0%.
`--blocks full` (below) removes this restriction entirely.

TRAINING
--------
Full batch: the whole 25,000-sequence training split is one optimizer step, so 1 epoch = 1 step.
The gradient is accumulated over chunks of `grad_chunk` sequences (each chunk's mean loss reweighted
by its share of the scored tokens) -- mathematically identical to one 25,000-sequence backward,
at a fraction of the memory.

`INIT` controls W_qk only ('zeros' or 'normal'), which is what the zero-init line of work means by
it. W_ov's block always starts at the identity, trained or not: starting it at zero would make
f == 0 everywhere, and under readout='none' the floor would then flatten every gradient.

FULL-MATRIX MODE
----------------
`--blocks full` lifts the sparsity entirely: every entry of all four matrices is trainable
(93,312 instead of 10,256). The ARCHITECTURE is untouched -- only which entries GD may move; the
readout is whatever `--readout` says. Use it to ask whether GD actually needs the blocks the
construction leaves out, or simply never uses them.

`--blocks` and `--train-wov` are independent, so all four combinations exist:

    --blocks relevant                 10,256   W_qk blocks only
    --blocks relevant --train-wov     10,384   + the two 8x8 W_ov blocks
    --blocks full                     58,320   both W_qk full, W_ov frozen identity
    --blocks full --train-wov         93,312   everything

The two modes are NOT nested: layer 2's three V x V blocks are justified by W_ov1 being
restricted, which confines layer 1's output to its first V coordinates. Under 'full', W_ov1 can
write anywhere, so columns 116..216 stop being dead and genuinely need the full block.

Usage:
    python master_train.py                            # defaults below
    python master_train.py --readout none             # the no-bias variant, for comparison
    python master_train.py --blocks full              # full W_qk, both layers
    python master_train.py --epochs 500 --init normal --optimizer sgd
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

import torch
import torch.nn as nn
import torch.nn.functional as F

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_DATA_DIR = os.path.abspath(os.path.join(
    _SCRIPT_DIR, '..', '14. elimination final convergence confusion', 'dataset'))


CFG = {
    'data_dir': _DATA_DIR,                                  # the equiprobable dataset
    'runs_dir': os.path.join(_SCRIPT_DIR, 'runs'),
    'device':   'cuda:3',

    # --- task, unchanged from folders 14/17 ---
    'v_sig':       4,          # signal tokens 0..3
    'n_noise':     4,          # noise tokens 4..7
    'seq_len':     100,
    'token_start': 50,         # loss and eval on tokens [50:100]

    # --- optimisation ---
    'optimizer':    'adam',    # 'adam' | 'sgd'
    'lr':           1.0,
    'weight_decay': 0.0,
    'epochs':       1000,       # == optimizer steps (full batch)
    'init':         'zeros',   # 'zeros' | 'normal'   (applies to W_qk only)
    'train_wov':    False,     # False -> W_ov top-left 8x8 = identity, frozen
    'blocks':       'relevant',  # 'relevant' -> the block-sparse masks below
                                 # 'full'     -> every entry of all four matrices is free
    'readout':      'bias',    # 'none' -> p = f            (f must already be a distribution)
                               # 'bias' -> p = (f+b)/sum(f+b), b a FIXED buffer from alpha
    'b_sig':        0.01,      # pseudo-count on the signal slots (smoothing)
    'noise_floor':  1e-9,      # b_noise at alpha=1, where (1-alpha)/(n_noise*alpha) = 0

    # --- batching / evaluation ---
    'grad_chunk':  2048,       # memory only; does not change the gradient
    'eval_batch':  256,
}

LOG_FLOOR  = 1e-30             # p[target] is clamped here before the log
SEED       = 0
THRESHOLDS = [0.6, 0.8, 1.0]

USE_WANDB     = True
WANDB_PROJECT = 'training_Wqk_relevant_fixing_Wov_and_bias'
WANDB_ENTITY  = None
SKIP_DONE     = True


def cell_tag():
    return (f"master_opt-{CFG['optimizer']}_lr-{CFG['lr']}_init-{CFG['init']}"
            f"_wov-{'train' if CFG['train_wov'] else 'fixed'}"
            f"_blocks-{CFG['blocks']}_readout-{CFG['readout']}_ep-{CFG['epochs']}")


def noise_bias(alpha):
    """b_noise chosen so that, after renormalising, the noise slots hold EXACTLY (1-alpha).

    With a perfect noise filter the signal slots of f sum to 1 and the noise slots to 0, so the
    signal side of (f+b) totals S = 1 + V_sig*b_sig and the noise side n_noise*b_noise. Solving

        n_noise*b_noise / (S + n_noise*b_noise) = 1 - alpha

    gives the expression below -- the /alpha is what makes it 'over_alpha' rather than the naive
    (1-alpha)/n_noise, which forgets that the renormalisation shrinks whatever you put in.
    """
    if alpha >= 1.0:
        return CFG['noise_floor']          # no noise exists; just keep the slot off zero
    return ((1.0 - alpha) * (1.0 + CFG['v_sig'] * CFG['b_sig'])
            / (CFG['n_noise'] * alpha))


def readout_bias(alpha, device):
    """The 8-vector b: b_sig on the signal slots, noise_bias(alpha) on the noise slots.

    Not a parameter -- two constants computed from alpha, never trained.
    """
    b = torch.full((CFG['v_sig'] + CFG['n_noise'],), CFG['b_sig'], device=device)
    b[CFG['v_sig']:] = noise_bias(alpha)
    return b


def run_dir():
    return os.path.join(CFG['runs_dir'], cell_tag())


def metrics_path(thr):
    return os.path.join(run_dir(), f'metrics_seed{SEED}_x_{thr}.json')


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
    """Every file present, and stamped as the equiprobable dataset (not folder 12's)."""
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
    reads [x | attn_out]), layer 2 does not (its attention output is the model's output).
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
    """Two attention layers; layer 2's first V attention coordinates are the output distribution."""

    def __init__(self, v_sig, n_noise, seq_len):
        super().__init__()
        self.V = v_sig + n_noise
        self.v_sig = v_sig
        self.seq_len = seq_len
        self.d1 = self.V + seq_len                 # 108
        self.d2 = 2 * self.d1                      # 216
        self.layer1 = AttentionBlock(self.d1, self.d1, concat=True)
        self.layer2 = AttentionBlock(self.d2, self.d1, concat=False)

    def forward(self, x):
        B, T = x.shape
        tok = F.one_hot(x, num_classes=self.V).float()
        pos = F.one_hot(torch.arange(T, device=x.device),
                        num_classes=self.seq_len).float().unsqueeze(0).expand(B, T, -1)
        s = torch.cat([tok, pos], dim=-1)
        mask = torch.triu(torch.ones(T, T, device=x.device) * float('-inf'), diagonal=1)
        out = self.layer2(self.layer1(s, mask), mask)
        return out[..., :self.V]                   # the predicted distribution, as produced


# ------------------------------------------------------------------------- block sparsity

def build_masks(model):
    """Boolean masks marking the only entries allowed to be nonzero. Everything else stays 0.

    CFG['blocks'] == 'full' lifts the restriction entirely: every entry of all four matrices is
    free. The architecture is unchanged either way -- no unembedding, no log, no final softmax,
    and whatever CFG['readout'] says; only which entries gradient descent may move.

    Note the two modes are NOT nested. Layer 2's three V x V blocks are justified by W_ov1 being
    restricted, which confines layer 1's output to its first V coordinates. Under 'full', W_ov1
    can write anywhere, so columns d1+V..d2 stop being dead and genuinely need the full block.
    """
    V, d1, d2 = model.V, model.d1, model.d2

    if CFG['blocks'] == 'full':
        return {n: torch.ones_like(p, dtype=torch.bool) for n, p in model.named_parameters()}
    if CFG['blocks'] != 'relevant':
        sys.exit(f"blocks must be 'relevant' or 'full', got {CFG['blocks']!r}")

    m = {}
    qk1 = torch.zeros(d1, d1, dtype=torch.bool)
    qk1[0:V, 0:V] = True                 # token x token
    qk1[V:d1, V:d1] = True               # position x position
    m['layer1.W_qk'] = qk1

    # layer 1's attention output lives entirely in its first V coordinates (W_ov1 is nonzero
    # only on [0:V, 0:V]), so columns d1+V .. d2 of layer 2's input are identically zero.
    # All three trainable blocks are therefore V x V.
    qk2 = torch.zeros(d2, d2, dtype=torch.bool)
    qk2[0:V, 0:V] = True                         # noise suppression
    qk2[0:V, d1:d1 + V] = True                   # induction head: own token x prev token
    qk2[d1:d1 + V, d1:d1 + V] = True             # layer-1 output x layer-1 output
    m['layer2.W_qk'] = qk2

    ov1 = torch.zeros(d1, d1, dtype=torch.bool); ov1[0:V, 0:V] = True
    ov2 = torch.zeros(d2, d1, dtype=torch.bool); ov2[0:V, 0:V] = True
    m['layer1.W_ov'] = ov1
    m['layer2.W_ov'] = ov2
    return m


def install(model, masks, device):
    """Initialise inside the masks, zero everything outside, and pin the gradients to the masks.

    Returns the list of trainable parameters.
    """
    V = model.V
    params = dict(model.named_parameters())
    torch.manual_seed(SEED)

    with torch.no_grad():
        for name, p in params.items():
            p.zero_()
            mask = masks[name].to(device)
            if name.endswith('W_qk'):
                if CFG['init'] == 'normal':
                    # same scheme the folder-17 scripts label 'normal': uniform(-s, s),
                    # s = fan_in ** -0.5, applied only inside the trainable blocks
                    s = p.shape[0] ** -0.5
                    p.copy_(torch.empty_like(p).uniform_(-s, s) * mask)
                elif CFG['init'] != 'zeros':
                    sys.exit(f"init must be 'zeros' or 'normal', got {CFG['init']!r}")
            else:
                # W_ov: identity on the 8x8 selector, whether or not it will be trained.
                # Starting it at zero would make p == 0 everywhere and kill every gradient.
                p[0:V, 0:V] = torch.eye(V, device=device)

    trainable, n_free = [], 0
    for name, p in params.items():
        train_this = name.endswith('W_qk') or CFG['train_wov']
        p.requires_grad_(train_this)
        if train_this:
            mask = masks[name].to(device)
            p.register_hook(lambda g, mask=mask: g * mask)     # gradients never leave the blocks
            trainable.append(p)
            n_free += int(mask.sum())          # entries that can actually move
    return trainable, n_free


@torch.no_grad()
def reapply_masks(model, masks, device):
    """Belt and braces: re-zero outside the masks after each step."""
    for name, p in model.named_parameters():
        p.mul_(masks[name].to(device))


# --------------------------------------------------------------------------------------- loss

def ce_loss(p, y, b=None):
    """Cross entropy on p, the V-vector layer 2 puts out.

    readout 'none' (b is None)
        p is used as-is, renormalised by its own sum. Exact when W_ov is the frozen identity,
        because then p is a convex combination of token one-hots. But the noise slots of a good
        induction head go to ZERO, and a noise target there costs -log(LOG_FLOOR) = 69 nats --
        so this readout forbids the noise filter and caps how low the loss can go.

    readout 'bias'
        p <- (p + b) / sum(p + b),  b fixed from alpha. No log, no softmax, no unembedding: the
        old log->softmax pair was exactly this division, since softmax(log x) = x / sum(x). The
        bias supplies the (1-alpha) noise mass as a KNOWN constant instead of something attention
        has to estimate, and b_sig smooths signal slots the context never saw.

        f >= 0 and b > 0  =>  p strictly in (0,1)  =>  loss >= 0, and LOG_FLOOR never binds.
        That chain needs f >= 0, i.e. W_ov frozen to the identity. With W_ov trainable f can go
        negative, the sum can pass through zero, and the loss becomes unbounded below -- see the
        check in train_one().
    """
    if b is not None:
        p = p + b
    s = p.sum(-1, keepdim=True)
    p = p / torch.where(s.abs() < LOG_FLOOR, torch.ones_like(s), s)
    q = p.gather(-1, y.unsqueeze(-1)).squeeze(-1)
    return -torch.log(q.clamp_min(LOG_FLOOR)).mean()


def full_batch_grad(model, x, y, b=None):
    """Exact gradient of the mean loss over ALL of x, accumulated in chunks.

    Each chunk's mean loss is reweighted by its share of the scored tokens, so the accumulated
    .grad equals what a single 25,000-sequence backward would have produced.
    """
    ts, te = CFG['token_start'], CFG['seq_len']
    chunk = CFG['grad_chunk']
    n_tok = x.shape[0] * (te - ts)
    total = 0.0
    for p in model.parameters():
        p.grad = None
    for i in range(0, x.shape[0], chunk):
        xb, yb = x[i:i + chunk], y[i:i + chunk, ts:te]
        w = yb.numel() / n_tok
        loss = ce_loss(model(xb)[:, ts:te], yb, b)
        (loss * w).backward()
        total += float(loss) * w
    return total


@torch.no_grad()
def eval_ce(model, x, y, device, b=None):
    model.eval()
    ts, te = CFG['token_start'], CFG['seq_len']
    total, n = 0.0, 0
    for i in range(0, x.shape[0], CFG['eval_batch']):
        xb = x[i:i + CFG['eval_batch']].to(device)
        yb = y[i:i + CFG['eval_batch'], ts:te].to(device)
        total += float(ce_loss(model(xb)[:, ts:te], yb, b)) * yb.numel()
        n += yb.numel()
    model.train()
    return total / n


# -------------------------------------------------------------------------------------- wandb

_WANDB_OK = None


def wandb_start(run_cfg, name, group, tags):
    global _WANDB_OK
    if not USE_WANDB or _WANDB_OK is False:
        return None
    try:
        import wandb
        wdir = os.path.join(_SCRIPT_DIR, 'wandb_local')
        os.makedirs(wdir, exist_ok=True)          # wandb will not create it for us
        run = wandb.init(project=WANDB_PROJECT, entity=WANDB_ENTITY, name=name, group=group,
                         tags=tags, config=run_cfg, reinit='finish_previous', dir=wdir)
        _WANDB_OK = True
        return run
    except Exception as e:
        if _WANDB_OK is None:
            print(f'  [wandb] disabled ({type(e).__name__}: {e})', flush=True)
        _WANDB_OK = False
        return None


# ------------------------------------------------------------------------------------ training

def train_one(thr, device):
    tag = cell_tag()
    d = run_dir()
    os.makedirs(d, exist_ok=True)
    mpath = metrics_path(thr)
    if SKIP_DONE and os.path.exists(mpath):
        print(f'  [skip] alpha={thr} already done -> {mpath}', flush=True)
        return

    tr, te = load_data('train', thr), load_data('test', thr)
    tr_x, tr_y = tr['x'].to(device), tr['y'].to(device)
    te_x, te_y = te['x'], te['y']

    model = ReducedTransformer(CFG['v_sig'], CFG['n_noise'], CFG['seq_len']).to(device)
    masks = build_masks(model)
    params, n_free = install(model, masks, device)

    opt = (torch.optim.Adam(params, lr=CFG['lr'], weight_decay=CFG['weight_decay'])
           if CFG['optimizer'] == 'adam' else
           torch.optim.SGD(params, lr=CFG['lr'], weight_decay=CFG['weight_decay']))

    b = readout_bias(thr, device) if CFG['readout'] == 'bias' else None
    if b is not None and CFG['train_wov']:
        print(f'    [warn] readout=bias assumes f >= 0, which needs W_ov frozen to the identity.'
              f'\n           With train_wov=True f can go negative and the loss is unbounded'
              f'\n           below -- watch for a negative train/test loss.', flush=True)

    run_cfg = {**{k: v for k, v in CFG.items() if k not in ('runs_dir', 'data_dir')},
               'threshold': float(thr), 'seed': SEED, 'log_floor': LOG_FLOOR,
               'n_train': int(tr_x.shape[0]), 'n_test': int(te_x.shape[0]),
               'trainable_entries': n_free,
               'tensor_entries': sum(p.numel() for p in params),
               'b_noise': None if b is None else float(noise_bias(thr))}
    wb = wandb_start(run_cfg, name=f'{tag}__x_{thr}', group=f'thr{thr}',
                     tags=[f'thr{thr}', CFG['init'], CFG['optimizer']])
    if wb:
        for m in ('train_loss', 'test_loss'):
            wb.define_metric(m, summary='min')
        for m in ('best_train', 'best_test'):
            wb.define_metric(m, summary='last')

    tr0, te0 = eval_ce(model, tr_x, tr_y, device, b), eval_ce(model, te_x, te_y, device, b)
    history = {'step': [0], 'train_loss': [tr0], 'test_loss': [te0]}
    best_train, best_test, best_state = tr0, te0, None
    if wb:
        wb.log({'train_loss': tr0, 'test_loss': te0,
                'best_train': tr0, 'best_test': te0}, step=0)

    n_all = sum(p.numel() for p in model.parameters())
    print(f'  alpha={thr}  free entries {n_free:,} of {n_all:,} '
          f'({n_free / n_all:.2%} of the four matrices)')
    print(f'    init: train {tr0:.5f}  test {te0:.5f}', flush=True)

    t0, ep, interrupted, stop_reason = time.time(), 0, False, None
    try:
        for ep in range(1, CFG['epochs'] + 1):
            model.train()
            tr_loss = full_batch_grad(model, tr_x, tr_y, b)
            opt.step()
            reapply_masks(model, masks, device)
            te_loss = eval_ce(model, te_x, te_y, device, b)

            # A negative or non-finite CE is a correctness violation, not a bad hyperparameter:
            # it means the output stopped being a distribution (p[target] > 1, so -log p < 0).
            # Stop at the step it happens instead of grinding on for hundreds more.
            if not math.isfinite(tr_loss) or tr_loss < 0:
                stop_reason = (f'loss not a valid CE at step {ep}: train={tr_loss:.6g}. '
                               f'p[target] > 1 or non-finite -> the output is not a distribution '
                               f'(expected when train_wov=True lets f go negative).')
                print(f'    [STOP] {stop_reason}', flush=True)
                break

            history['step'].append(ep)
            history['train_loss'].append(tr_loss)
            history['test_loss'].append(te_loss)
            if tr_loss < best_train - 1e-9:
                best_train = tr_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_test = min(best_test, te_loss)

            if wb:
                wb.log({'train_loss': tr_loss, 'test_loss': te_loss,
                        'best_train': best_train, 'best_test': best_test}, step=ep)
            if ep <= 10 or ep % 25 == 0 or ep == CFG['epochs']:
                print(f'    step {ep:5d}/{CFG["epochs"]}  train {tr_loss:.5f}  test {te_loss:.5f}'
                      f'  best_train {best_train:.5f}  best_test {best_test:.5f}', flush=True)
    except KeyboardInterrupt:
        interrupted = True
        print('    [INTERRUPTED] saving best-so-far', flush=True)

    torch.save(model.state_dict(), os.path.join(d, f'model_final_seed{SEED}_x_{thr}.pth'))
    if best_state is not None:
        torch.save(best_state, os.path.join(d, f'model_best_seed{SEED}_x_{thr}.pth'))

    result = {'cell': tag, 'threshold': float(thr), 'seed': SEED, 'config': run_cfg,
              'history': history, 'best_train': best_train, 'best_test': best_test,
              'final_train': history['train_loss'][-1], 'final_test': history['test_loss'][-1],
              'steps_run': ep, 'time_s': round(time.time() - t0, 1),
              'stopped_early': stop_reason is not None or interrupted,
              'stop_reason': stop_reason or ('interrupted' if interrupted else None)}
    with open(mpath, 'w') as fh:
        json.dump(result, fh, indent=2)
    if wb:
        wb.summary.update({'best_train': best_train, 'best_test': best_test,
                           'final_train': result['final_train'],
                           'final_test': result['final_test'], 'steps_run': ep})
        try:
            wb.finish()
        except Exception:
            pass
    print(f'    done in {result["time_s"]/60:.1f}m  best_train {best_train:.5f}  '
          f'best_test {best_test:.5f}  -> {mpath}', flush=True)
    if interrupted:
        raise KeyboardInterrupt


def rebuild_summary():
    rows = []
    for mp in sorted(glob.glob(os.path.join(CFG['runs_dir'], 'master_*', 'metrics_*.json'))):
        m = json.load(open(mp))
        rows.append({k: m.get(k) for k in ('cell', 'threshold', 'seed', 'best_train',
                                           'best_test', 'final_train', 'final_test',
                                           'steps_run')})
    out = os.path.join(CFG['runs_dir'], 'master_summary.json')
    os.makedirs(CFG['runs_dir'], exist_ok=True)
    with open(out, 'w') as fh:
        json.dump(rows, fh, indent=2)
    print(f'\nSummary ({len(rows)} runs) -> {out}')


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--epochs', type=int, default=CFG['epochs'],
                    help='optimizer steps (full batch, so 1 epoch = 1 step)')
    ap.add_argument('--lr', type=float, default=CFG['lr'])
    ap.add_argument('--optimizer', choices=('adam', 'sgd'), default=CFG['optimizer'])
    ap.add_argument('--init', choices=('zeros', 'normal'), default=CFG['init'],
                    help='W_qk initialisation; W_ov always starts at the identity')
    ap.add_argument('--train-wov', action='store_true', default=CFG['train_wov'],
                    help='train W_ov as well as W_qk (default: frozen identity selector)')
    ap.add_argument('--blocks', choices=('relevant', 'full'), default=CFG['blocks'],
                    help="'relevant' = the block-sparse masks (10,256 entries); "
                         "'full' = every entry of all four matrices free (93,312)")
    ap.add_argument('--readout', choices=('none', 'bias'), default=CFG['readout'],
                    help="'none' = p used as-is; 'bias' = p <- (p+b)/sum(p+b) with b fixed "
                         "from alpha (no log, no softmax, no unembedding)")
    ap.add_argument('--device', default=CFG['device'])
    ap.add_argument('--thresholds', type=float, nargs='+', default=THRESHOLDS)
    ap.add_argument('--no-wandb', action='store_true')
    return ap.parse_args()


def main():
    global THRESHOLDS, USE_WANDB
    a = parse_args()
    CFG.update({'epochs': a.epochs, 'lr': a.lr, 'optimizer': a.optimizer, 'init': a.init,
                'train_wov': a.train_wov, 'device': a.device, 'blocks': a.blocks,
                'readout': a.readout})
    THRESHOLDS = a.thresholds
    USE_WANDB = USE_WANDB and not a.no_wandb

    device = torch.device(CFG['device'] if torch.cuda.is_available() else 'cpu')
    check_dataset()

    probe = ReducedTransformer(CFG['v_sig'], CFG['n_noise'], CFG['seq_len'])
    masks = build_masks(probe)
    n_all = sum(p.numel() for p in probe.parameters())
    n_qk = int(masks['layer1.W_qk'].sum() + masks['layer2.W_qk'].sum())
    n_ov = int(masks['layer1.W_ov'].sum() + masks['layer2.W_ov'].sum())
    n_free = n_qk + (n_ov if CFG['train_wov'] else 0)
    del probe

    print(f'device     : {device}')
    print(f'data       : {CFG["data_dir"]}  [equiprobable]')
    print(f'cell       : {cell_tag()}')
    print(f'runs       : {run_dir()}')
    V = CFG['v_sig'] + CFG['n_noise']
    print(f'model      : 2 attention layers, no unembedding / no log / no final softmax;')
    if CFG['readout'] == 'bias':
        print(f'             p = (f + b) / sum(f + b)   with f = layer2 attn_out[..., :{V}]')
        print(f'             b fixed from alpha: b_sig={CFG["b_sig"]}, '
              f'b_noise=(1-a)(1+{CFG["v_sig"]}*b_sig)/({CFG["n_noise"]}*a)')
        for t in THRESHOLDS:
            print(f'               alpha={t}: b_noise = {noise_bias(t):.6f}')
    else:
        print(f'             p = layer2 attn_out[..., :{V}] used as-is, '
              f'loss = -log p[target] floored at {LOG_FLOOR:g}')
    what = 'FULL matrices' if CFG['blocks'] == 'full' else 'block-sparse'
    print(f'trainable  : {what}  {n_free:,} entries of {n_all:,} ({n_free / n_all:.2%})  '
          f'= W_qk {n_qk:,}'
          + (f' + W_ov {n_ov:,}' if CFG['train_wov']
             else f'   (W_ov frozen to identity)'))
    print(f'optimiser  : {CFG["optimizer"]}  lr={CFG["lr"]}  wd={CFG["weight_decay"]}  '
          f'init={CFG["init"]}  steps={CFG["epochs"]}')
    print(f'batching   : FULL BATCH in chunks of {CFG["grad_chunk"]} -> 1 epoch = 1 step')
    print(f'thresholds : {THRESHOLDS}\n')

    for thr in THRESHOLDS:
        try:
            train_one(thr, device)
        except KeyboardInterrupt:
            print('\nstopping')
            break
    rebuild_summary()


if __name__ == '__main__':
    main()
