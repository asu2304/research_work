"""Training script -- NOTHING frozen, NOTHING masked, softmax readout.

Folder 20, the unconstrained control. Same task, same data, same two-layer shape as folders
17/18/19 and as `train_separate_qk.py`; the point is to remove every restriction those runs
imposed and see what the loss can reach when the model is allowed everything.

WHAT IS FREE HERE THAT WAS NOT FREE BEFORE
------------------------------------------
    W_ov, both layers   FULL and TRAINABLE. Folders 17-19 and train_separate_qk.py pin it to
                        W_ov[:8,:8] = I and freeze it. Here EVERY entry of the whole (in x out)
                        matrix carries a gradient and moves from step 1 -- 11,664 cells in
                        layer 1 and 23,328 in layer 2, not just the 8x8 corner.
                        --ov-init only says where it STARTS. It cannot start at zero: W_ov2 = 0
                        makes layer 2's output identically zero and kills every W_qk gradient
                        with it. 'corner' (the default) starts at the value folders 17-19 froze
                        it to, so step 0 is exactly their model; 'full' starts at the identity
                        on the whole leading diagonal; 'normal' at uniform(-s, s).
    W_qk, both layers   FULL and TRAINABLE -- every entry, no block masks at all. Folder 18
                        allowed 10.8% of the entries, folder 19 tied those down to 5 numbers.
    the readout bias    TRAINABLE. Everywhere else it is computed from alpha and held fixed.

    degrees of freedom: 93,320  (--qk combined)   or   104,984  (--qk separate)
                        against folder 19's 9,910, and folder 18's 10,060.

WHY THE SOFTMAX COMES BACK
--------------------------
This is the whole reason the readout has to change, and it is worth being precise about.

With W_ov FROZEN to the 8x8 identity, layer 2's output f is a convex combination of token
one-hots, so f >= 0 and sum(f) = 1 exactly: f already IS a distribution, and the readout
(f + b)/sum(f + b) is an honest renormalisation with b > 0 guaranteeing p in (0,1). That chain
is what let folders 17-19 delete the unembedding, the log and the output softmax -- see
CLAUDE.md section 8.2.

Unfreeze W_ov and every link of it breaks. f can go negative, sum(f + b) can cross zero, and
-log p is then unbounded BELOW -- folder 19 measured best_train = -1.05 doing exactly this. A
negative cross entropy is not a good result, it is the model having stopped emitting a
distribution. So here layer 2's output is read as LOGITS instead:

        z = layer2_out[..., :8] + b,        p = softmax(z),        loss = -log p[y]

which is `F.cross_entropy` and is bounded below by 0 for any real z, however negative. The
guard that folder 19 needed (stop when the CE goes negative) is structurally unnecessary now.

Note this is NOT the old log->softmax pair coming back. That pair was the identity map
x/sum(x) on a nonneg vector (measured difference 5.6e-17); this is a genuine exponential
readout, and layer 2's output means something different because of it -- log-counts rather
than counts.

THE BIAS, AND WHY ITS INITIALISATION IS THE PDF'S beta
------------------------------------------------------
In count space the fixed bias was b_noise = (1-alpha)(1 + V_sig*b_sig)/(n_noise*alpha). In
LOGIT space the same target is its logarithm, and that is exactly the unembedding bias of
`seperate_parameterization_weight_handconstruction.pdf` section 6.1:

        b = [0, 0, 0, 0, beta, beta, beta, beta],
        beta = log( (1 + V_sig*delta)(1 - alpha) / (alpha * n_noise) )      (PDF eq. 34)

which for a perfect noise filter puts exactly (1-alpha) of the mass on the four noise tokens
and alpha on the signal side. `--bias-init logalpha` (the default) starts there; the bias is
then TRAINABLE, so the run also answers whether GD keeps that value or prefers another.
At alpha = 1 the exact beta is -inf and -30 is used, as in the PDF's table (it moves the
signal mass by 4e-13, far below float32 resolution).

WHAT THIS RUN CAN AND CANNOT BE COMPARED TO
-------------------------------------------
CAN: the CE numbers. Both readouts emit a distribution over the same 8 tokens scored on the
same window, so best_test sits on the same axis as folder 18's 1.69823 / 1.38435 / 0.74498 and
as ideal-alpha's 1.64342 / 1.33218 / 0.69315. This run's hypothesis class strictly contains
those, so it should not do worse; how much better is the measurement.

MEASURED, alpha = 0.8, --qk separate, 400 full-batch steps:

    lr     best test   final train / final test        what happens
    0.3      1.82122   1.87915 / 1.87900              overshoots at step ~20, partly recovers
    0.1      1.46128   1.44619 / 1.46767              <- the default
    0.03     1.46258   1.45147 / 1.46397              same place, slower
    0.01     1.56811   (at step 100, still falling)

lr 0.1 and 0.03 land within 0.0013 nats of each other, so the default stays at 0.1 -- the same
value `train_separate_qk.py` uses, which keeps the two families comparable.

    THE RESULT IS THAT MORE FREEDOM MAKES IT WORSE. The constrained run reaches 1.38435 at
    alpha = 0.8 with 10,060 free entries; this one plateaus at ~1.461 with 104,984, and by
    step 400 its train loss (1.446) has dropped BELOW its test loss (1.468). It is overfitting
    25,000 sequences. The block sparsity and the frozen W_ov were not merely a simplification
    of the hypothesis class -- they were a correct inductive bias, and removing them costs
    ~0.077 nats rather than buying anything.

    Checkpointing is on best TEST loss, so `model_best_*.pth` and `best_test` already report
    the pre-overfit point; `final_test` is where it ended up. Compare the two.

CANNOT, at least not naively: the weights, against the hand construction. Freeing W_ov adds a
gauge the earlier runs did not have. Layer 1 writes its output through W_ov1 and layer 2 reads
it through W_q/W_k/W_ov2, so for any invertible A

        W_ov1 -> W_ov1 A,   (layer 2's rows 108:216) -> A^-1 (those rows)

leaves every logit identical. The basis of layer 1's output channel is no longer pinned to
"the token one-hot", so comparing layer 2's L1-output blocks with the construction's cell by
cell is meaningless without first solving for A.

What IS still comparable is the part of the circuit that reads the fixed input embedding:
layer 1's whole W_qk, and layer 2's [0:108, 0:108] corner (token and position queries against
token and position keys). Those are what the script logs -- `L1_cos` and `L2_cos_xx`. The full
layer-2 cosine is logged too, as `L2_cos_all`, but it is basis-dependent; do not quote it.

The softmax row-shift gauge also shrinks. softmax(s + c*1) = softmax(s) needs the shifted key
group to sum to a constant, which holds exactly for the token one-hot and the position one-hot
but NOT for layer 1's output once W_ov1 is free (its coordinate sum depends on the column sums
of W_ov1 and on which keys were attended). So the gauge fix here centres the token and position
blocks only, and leaves the rest alone.

--blocks relevant IS REFUSED WHEN W_ov IS TRAINABLE
---------------------------------------------------
The 'relevant' mask of `train_separate_qk.py` drops layer 2's input rows 116:216 on the grounds
that layer 1's output lives entirely in its first 8 coordinates -- true only because W_ov1 was
pinned to the 8x8 corner. With W_ov1 free those 100 coordinates carry real signal, and masking
them would silently amputate the model. The script exits rather than let that happen.

ARCHITECTURE (unchanged otherwise)
----------------------------------
    x            one-hot token (8) + one-hot position (100)        -> 108 = d1
    layer 1      causal attention, returns cat[x, attn_out]        -> 216 = d2
    layer 2      causal attention, returns attn_out only           -> 108
    z = layer2_out[..., :8] + b                                    -> logits
    loss         F.cross_entropy(z, y)  on tokens [50:100)

    scores =  x W_qk x^T      / sqrt(output_dim)     [--qk combined, the DEFAULT --
                                                     folders 17/18/19's parametrisation]
           = (x W_q)(x W_k)^T / sqrt(output_dim)     [--qk separate, this folder's sibling
                                                     train_separate_qk.py]

Strictly causal mask, triu(diagonal=0), mask[0,0] reopened. Full batch, 1 epoch = 1 step.

--qk combined (the default) / separate
--------------------------------------
`combined` is the folder-17/18/19 parametrisation -- ONE W_qk per layer -- with every
restriction lifted. It is the default and the configuration this script is for. `separate`
is kept because the sibling `train_separate_qk.py` uses it and the flag makes the two
directly comparable from one codebase; the two differ in exactly one way that matters for
training, the origin:

    combined   W_qk = 0 has a NONZERO gradient. Zero init works, as in folders 17-19, and is
               the default here. Layer 1's gradient IS zero at step 0 -- dS2/dZ1 carries W_qk2
               as a multiplicative factor -- so layer 1 cannot move until layer 2 has left the
               origin. That is the folder-18 gating of CLAUDE.md section 8.9, and it is
               reproduced here: measured ||g|| at init is 0.000e+00 for layer1.W_qk and
               2.431e-04 for layer2.W_qk, and layer 1 starts moving at step 2.
    separate   W_q = W_k = 0 is a strict saddle -- dL/dW_q = (dL/dW_qk) W_k = 0 -- so BOTH
               gradients vanish and nothing moves at all. Pass --init eps with it.

The script measures this rather than asserting it: it prints ||g|| at init and, under
--qk separate, stops the threshold if the weights have not left the origin after one step.

Usage:
    python train_full_free.py                                  # combined W_qk, zero init
    python train_full_free.py --qk separate --init eps         # the factored form instead
    python train_full_free.py --freeze-bias                    # ablate the trainable bias
    python train_full_free.py --runs-dir /tmp/scratch --epochs 3 --no-wandb   # smoke test
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
sys.path.insert(0, _SCRIPT_DIR)
from train_separate_qk import construction        # one source of truth for the hand circuit

_DATA_DIR = os.path.abspath(os.path.join(
    _SCRIPT_DIR, '..', '14. elimination final convergence confusion', 'dataset'))

SEED = 0
THRESHOLDS = (0.6, 0.8, 1.0)

WANDB_PROJECT = 'full_free_softmax'       # everything trainable, softmax readout
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
    'lr':           0.01,       # see the probe table in train_separate_qk.py's docstring
    'weight_decay': 0.0,
    'epochs':       500,       # == optimizer steps (full batch)
    'qk':           'combined',   # one W_qk per layer, as in folders 17/18/19
    'init':         'normal',   # 'zeros' | 'eps' | 'normal' | 'orth'   (the QK parameters)
    'init_scale':   1e-3,
    'head_dim':     None,      # None -> output_dim (108)
    'blocks':       'full',    # 'relevant' is REFUSED while W_ov is trainable
    'ov_init':      'corner',  # 'corner' | 'full' | 'normal'
    'train_wov':    True,      # the point of this script
    'bias_init':    'logalpha',
    'train_bias':   True,      # the other point of this script
    'b_sig':        0.01,      # the delta of the PDF's beta
    'beta_at_one': -30.0,      # alpha = 1: exact beta is -inf; the PDF uses -30

    # NO per-step snapshots. This model has 93,320 free cells, so storing weights AND
    # gradients every step cost ~357 MB per threshold (~1.05 GB for the default three) and
    # the .npy files are preallocated, so the cost lands whether or not the run finishes.
    # What survives a run: model_best / model_final .pth and metrics_*.json.

    # --- batching / evaluation ---
    'grad_chunk':  8192,
    'eval_chunk':  8192,
    'log_every':   0,          # 0 -> every 5% of the run
    'device':      None,
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
    return (f"free_qk-{CFG['qk']}_opt-{CFG['optimizer']}_lr-{CFG['lr']}_init-{CFG['init']}"
            f"_wov-{'train' if CFG['train_wov'] else 'fixed'}-{CFG['ov_init']}"
            f"_bias-{'train' if CFG['train_bias'] else 'fixed'}-{CFG['bias_init']}"
            f"_blocks-{CFG['blocks']}{hd}_readout-softmax_ep-{CFG['epochs']}")


def run_dir():
    return os.path.join(CFG['runs_dir'], cell_tag())


def metrics_path(thr):
    return os.path.join(run_dir(), f'metrics_seed{SEED}_x_{thr}.json')


# ------------------------------------------------------------------------------- readout bias

def beta(alpha):
    """The PDF's unembedding bias (eq. 34) -- the LOGIT-space version of the count-space bias.

        beta = log( (1 + V_sig*delta)(1 - alpha) / (alpha * n_noise) )

    With a perfect noise filter the signal logits are log(f_v + delta) and the noise logits are
    all beta; the softmax then puts exactly (1-alpha) on the noise side and alpha on the signal
    side, with no external renormalisation. It is the exact logarithm of the count-space
    b_noise used by folders 17-19, which is the sense in which the two readouts agree.
    """
    if alpha >= 1.0:
        return CFG['beta_at_one']          # exact beta is -inf; -30 moves the mass by 4e-13
    return math.log((1.0 + CFG['v_sig'] * CFG['b_sig']) * (1.0 - alpha)
                    / (alpha * CFG['n_noise']))


def init_bias(alpha):
    """b: zeros on the signal logits, beta(alpha) on the noise logits."""
    b = torch.zeros(CFG['v_sig'] + CFG['n_noise'])
    if CFG['bias_init'] == 'logalpha':
        b[CFG['v_sig']:] = beta(alpha)
    elif CFG['bias_init'] != 'zeros':
        sys.exit(f"bias-init must be 'logalpha' or 'zeros', got {CFG['bias_init']!r}")
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

class FreeAttentionBlock(nn.Module):
    """One causal attention layer with a FULL, TRAINABLE W_ov and either QK parametrisation.

    Display convention throughout -- x multiplies from the left, so W_q, W_k are (in x head),
    W_ov is (in x out), and W_qk = W_q @ W_k.T lines up with the construction cell for cell.
    """

    def __init__(self, input_dim, output_dim, concat, qk='separate', head_dim=None):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.head_dim = output_dim if head_dim is None else head_dim
        self.concat = concat
        self.qk = qk
        if qk == 'separate':
            self.W_q = nn.Parameter(torch.zeros(input_dim, self.head_dim))
            self.W_k = nn.Parameter(torch.zeros(input_dim, self.head_dim))
        elif qk == 'combined':
            self.W_qk = nn.Parameter(torch.zeros(input_dim, input_dim))
        else:
            sys.exit(f"qk must be 'separate' or 'combined', got {qk!r}")
        self.W_ov = nn.Parameter(torch.zeros(input_dim, output_dim))

    def qk_matrix(self):
        """The combined circuit the softmax sees -- the only QK object worth comparing."""
        return self.W_q @ self.W_k.T if self.qk == 'separate' else self.W_qk

    def forward(self, x, mask):
        if self.qk == 'separate':
            # (x W_q)(x W_k)^T is algebraically x W_q W_k^T x^T but costs O(T * head_dim)
            # instead of materialising the (in x in) product every forward.
            scores = torch.bmm(x @ self.W_q, (x @ self.W_k).transpose(1, 2))
        else:
            scores = torch.bmm(x @ self.W_qk, x.transpose(1, 2))
        scores = scores / (self.output_dim ** 0.5)
        out = torch.bmm(F.softmax(scores + mask, dim=-1), x @ self.W_ov)
        return torch.cat([x, out], dim=-1) if self.concat else out


class FreeTransformer(nn.Module):
    """Two attention layers, a trainable readout bias, and a softmax over the first V outputs."""

    def __init__(self, v_sig, n_noise, seq_len, qk='separate', head_dim=None):
        super().__init__()
        self.V = v_sig + n_noise
        self.seq_len = seq_len
        self.d1 = self.V + seq_len                  # 108
        self.d2 = 2 * self.d1                       # 216
        self.layer1 = FreeAttentionBlock(self.d1, self.d1, True, qk, head_dim)
        self.layer2 = FreeAttentionBlock(self.d2, self.d1, False, qk, head_dim)
        self.b = nn.Parameter(torch.zeros(self.V))  # filled by install() from alpha

    def forward(self, x):
        """Returns LOGITS. The softmax lives in the loss (F.cross_entropy), which is the
        numerically stable place for it."""
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
        return out[..., :self.V] + self.b


# ------------------------------------------------------------------------------ gauge fixing

def free_key_groups(layer, v_sig=4, n_noise=4, seq_len=100):
    """The key blocks whose softmax row-shift is still an EXACT gauge once W_ov is free.

    softmax(s + c*1) = softmax(s) lets a constant be added to a query row across a key group
    only if that group's contribution is the same for every key -- i.e. if the group's
    coordinates sum to a constant. Two blocks of the residual stream always do:

        token one-hot     dims 0:8      sums to 1
        position one-hot  dims 8:108    sums to 1

    Layer 1's output block (dims 108:216 of layer 2's input) used to be a third, because
    W_ov1 was pinned to the 8x8 identity and the block was a convex combination of one-hots.
    With W_ov1 trainable its coordinate sum is (attn @ x) . colsum(W_ov1), which varies with
    the key. So it is NOT a gauge group here and is left uncentred.

    Returns (live_query_rows, [slice, ...]).
    """
    V, T = v_sig + n_noise, seq_len
    d1 = V + T
    return (slice(0, d1) if layer == 1 else slice(0, 2 * d1)), [slice(0, V), slice(V, d1)]


def free_gauge_fix(W, layer, **kw):
    """Centre each query row within each exact key group; leave every other column untouched.

    Unlike train_separate_qk.gauge_fix this KEEPS the non-group columns rather than dropping
    them, because here they are live rather than identically zero.
    """
    rows, groups = free_key_groups(layer, **kw)
    out = np.array(W, dtype=np.float64, copy=True)[rows]
    for g in groups:
        blk = out[:, g]
        out[:, g] = blk - blk.mean(axis=1, keepdims=True)
    return out


def free_gauge_cos(A, Bm, layer, sub=None, **kw):
    """Gauge-fixed cosine between two W_qk matrices; `sub` restricts to a sub-block first."""
    a, b = free_gauge_fix(A, layer, **kw), free_gauge_fix(Bm, layer, **kw)
    if sub is not None:
        a, b = a[sub, sub], b[sub, sub]
    a, b = a.ravel(), b.ravel()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 0 and nb > 0 else 0.0


# ------------------------------------------------------------------------------------- masks

def build_masks(model):
    """Boolean masks marking the entries allowed to be nonzero.

    'full' -- every entry of every parameter. That is the point of this script, and it is the
    default. 'relevant' exists only so the flag parses; it is REFUSED while W_ov is trainable,
    because it drops layer 2's input rows 116:216 on the grounds that layer 1's output lives in
    its first 8 coordinates -- true only while W_ov1 is pinned to the 8x8 corner.
    """
    if CFG['blocks'] == 'full':
        return {n: torch.ones_like(p, dtype=torch.bool) for n, p in model.named_parameters()}
    if CFG['blocks'] != 'relevant':
        sys.exit(f"blocks must be 'full' or 'relevant', got {CFG['blocks']!r}")
    if CFG['train_wov']:
        sys.exit("--blocks relevant is not valid while W_ov is trainable: it would mask away "
                 "layer 2's input rows 116:216, which carry real signal once W_ov1 is free.\n"
                 "Use --blocks full, or train_separate_qk.py if you want W_ov frozen.")

    V, d1, d2 = model.V, model.d1, model.d2
    m = {n: torch.ones_like(p, dtype=torch.bool) for n, p in model.named_parameters()}
    live = torch.zeros(d2, dtype=torch.bool)
    live[0:d1 + V] = True
    for n, p in model.named_parameters():
        if n.startswith('layer2.') and p.dim() == 2 and p.shape[0] == d2:
            mm = torch.zeros_like(p, dtype=torch.bool)
            mm[live] = True
            if p.shape[1] == d2:                    # W_qk is square: mask columns too
                mm &= live.unsqueeze(0)
            m[n] = mm
    return m


def to_device_masks(d, device):
    return {k: v.to(device) for k, v in d.items()}


def install(model, masks, alpha, device):
    """Initialise every parameter, then pin gradients to the masks."""
    V = model.V
    params = dict(model.named_parameters())
    torch.manual_seed(SEED)
    scale = CFG['init_scale']

    with torch.no_grad():
        model.b.copy_(init_bias(alpha).to(device))
        for name, p in params.items():
            if name == 'b':
                continue
            p.zero_()
            mask = masks[name].to(device)
            if name.endswith('W_ov'):
                # W_ov is trainable now, but it still has to START somewhere that has a
                # gradient: W_ov2 = 0 makes layer 2's output identically 0 and kills every QK
                # gradient with it. 'corner' is the value folders 17-19 froze it to.
                if CFG['ov_init'] == 'corner':
                    p[0:V, 0:V] = torch.eye(V, device=device)
                elif CFG['ov_init'] == 'full':
                    n = min(p.shape)
                    p[torch.arange(n), torch.arange(n)] = 1.0
                elif CFG['ov_init'] == 'normal':
                    s = p.shape[0] ** -0.5
                    p.copy_(torch.empty_like(p).uniform_(-s, s))
                else:
                    sys.exit(f"ov-init must be corner/full/normal, got {CFG['ov_init']!r}")
                p.mul_(mask)
            else:                                     # W_q / W_k / W_qk
                if CFG['init'] == 'zeros':
                    pass          # fine for --qk combined; a strict saddle for --qk separate
                elif CFG['init'] == 'eps':
                    p.copy_(torch.randn_like(p) * scale * mask)
                elif CFG['init'] == 'normal':
                    s = p.shape[0] ** -0.5            # uniform(-s, s), fan_in ** -0.5
                    p.copy_(torch.empty_like(p).uniform_(-s, s) * mask)
                elif CFG['init'] == 'orth':
                    w = torch.empty(p.shape[1], p.shape[0], device=device)
                    nn.init.orthogonal_(w)
                    p.copy_(w.T * scale * mask)
                else:
                    sys.exit(f"init must be one of zeros/eps/normal/orth, got {CFG['init']!r}")

    trainable, names, n_free = [], [], 0
    for name, p in params.items():
        if name == 'b':
            p.requires_grad_(CFG['train_bias'])
            if CFG['train_bias']:
                trainable.append(p)
                names.append(name)
                n_free += p.numel()
            continue
        train_this = (not name.endswith('W_ov')) or CFG['train_wov']
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
    """Re-zero outside the masks after each step. A no-op under --blocks full."""
    for name, p in model.named_parameters():
        if name in masks and masks[name].shape == p.shape:
            p.mul_(masks[name])


# --------------------------------------------------------------------------------------- loss

def ce_loss(logits, y):
    """Softmax cross entropy on the 8 output logits.

    This is where the readout differs from folders 17-19 and from train_separate_qk.py. There,
    W_ov frozen to the identity made layer 2's output a distribution already and the readout
    was the renormalisation (f + b)/sum(f + b). Here W_ov is free, f can be negative, and that
    expression is not a probability -- so the output is read as logits and exponentiated.
    F.cross_entropy does the log-sum-exp in a stable way and is >= 0 for any real input, which
    is exactly the guarantee the frozen-W_ov runs got from f >= 0 instead.
    """
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))


@torch.no_grad()
def eval_ce(model, x, y, device):
    """Mean CE over the scored window, in chunks. No gradient."""
    model.eval()
    ts, te = CFG['token_start'], CFG['seq_len']
    tot = cnt = 0.0
    for i in range(0, x.shape[0], CFG['eval_chunk']):
        xb = x[i:i + CFG['eval_chunk']].to(device)
        yb = y[i:i + CFG['eval_chunk'], ts:te].to(device)
        tot += float(ce_loss(model(xb)[:, ts:te], yb)) * yb.numel()
        cnt += yb.numel()
    return tot / cnt


def full_batch_grad(model, x, y):
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
        loss = ce_loss(model(xb)[:, ts:te], yb)
        (loss * (yb.numel() / n_tok)).backward()
        tot += float(loss) * yb.numel()
    return tot / n_tok


@torch.no_grad()
def report(model, ref):
    """Norms, bias, and the cosines against the hand construction that are still meaningful.

    L1_cos      layer 1's whole W_qk. Layer 1 reads the FIXED input embedding, so its basis is
                pinned and this number means the same thing it did in folders 18/19.
    L2_cos_xx   layer 2's [0:108, 0:108] corner -- token and position queries against token and
                position keys. Also read in the fixed input basis, so also meaningful.
    L2_cos_all  the full 216x216. Logged for completeness and NOT to be quoted: with W_ov1 free,
                layer 1's output channel has an arbitrary invertible basis (W_ov1 -> W_ov1 A,
                layer 2's rows 108:216 -> A^-1 ...), so these cells are only comparable after
                solving for A.
    """
    d1 = model.d1
    out = {}
    for lay, blk in ((1, model.layer1), (2, model.layer2)):
        qk = blk.qk_matrix().detach().float().cpu().numpy()
        out[f'L{lay}_qk_absmax'] = float(np.abs(qk).max())
        out[f'L{lay}_qk_fro'] = float(np.linalg.norm(qk))
        out[f'L{lay}_ov_fro'] = float(blk.W_ov.detach().norm())
        # how far W_ov has moved from the identity corner it used to be frozen at
        ov = blk.W_ov.detach().float().cpu().numpy()
        eye = np.zeros_like(ov); eye[:model.V, :model.V] = np.eye(model.V)
        out[f'L{lay}_ov_off_identity'] = float(np.linalg.norm(ov - eye))
        if lay == 1:
            out['L1_cos'] = free_gauge_cos(qk, ref['W_qk1'], 1)
        else:
            out['L2_cos_xx'] = free_gauge_cos(qk, ref['W_qk2'], 2, sub=slice(0, d1))
            out['L2_cos_all'] = free_gauge_cos(qk, ref['W_qk2'], 2)
    b = model.b.detach().float().cpu().numpy()
    out['b_sig_mean'] = float(b[:CFG['v_sig']].mean())
    out['b_noise_mean'] = float(b[CFG['v_sig']:].mean())
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
                                    'readout': 'softmax', 'beta_init': beta(thr)}, reinit=True)
            # config only -- it is what tells two runs apart in the UI, not a metric
        except Exception as e:                       # a dead wandb must not kill a run
            print(f'    [wandb off: {e}]', flush=True)

    model = FreeTransformer(CFG['v_sig'], CFG['n_noise'], CFG['seq_len'],
                            CFG['qk'], CFG['head_dim']).to(device)
    masks = build_masks(model)
    params, n_free, tnames = install(model, masks, thr, device)
    dmasks = to_device_masks(masks, device)

    tr = load_data('train', thr)
    te = load_data('test', thr)
    tr_x, tr_y = tr['x'].to(device), tr['y'].to(device)
    te_x, te_y = te['x'].to(device), te['y'].to(device)

    if CFG['optimizer'] == 'adam':
        opt = torch.optim.Adam(params, lr=CFG['lr'], weight_decay=CFG['weight_decay'])
    elif CFG['optimizer'] == 'sgd':
        opt = torch.optim.SGD(params, lr=CFG['lr'], weight_decay=CFG['weight_decay'])
    else:
        sys.exit(f"optimizer must be 'adam' or 'sgd', got {CFG['optimizer']!r}")

    tr0 = eval_ce(model, tr_x, tr_y, device)
    te0 = eval_ce(model, te_x, te_y, device)
    history = {'step': [0], 'train_loss': [tr0], 'test_loss': [te0], 'train_loss_pre': [tr0],
               'L1_cos': [], 'L2_cos_xx': []}
    best_train, best_test, best_state = tr0, te0, None

    n_all = sum(p.numel() for p in model.parameters())
    print(f'  alpha={thr}  trainable entries {n_free:,} of {n_all:,}   '
          f'beta init {beta(thr):+.6f}')

    # The saddle check, measured once and reported rather than assumed.
    _ = full_batch_grad(model, tr_x, tr_y)
    g0 = {n: float(p.grad.norm()) for n, p in model.named_parameters() if p.grad is not None}
    print('    grad norms at init: ' + '  '.join(f'{n}={v:.3e}' for n, v in g0.items()))
    qk_g = [v for n, v in g0.items() if not n.endswith('W_ov') and n != 'b']
    if CFG['qk'] == 'separate' and CFG['init'] == 'zeros' and max(qk_g, default=0.0) == 0.0:
        print('    [!] every QK gradient is EXACTLY zero -- W_q = W_k = 0 is a strict saddle '
              'of the\n        factored parametrisation. Use --init eps, or --qk combined.',
              flush=True)

    r0 = report(model, ref)
    print(f'    init: train {tr0:.5f}  test {te0:.5f}   cos L1 {r0["L1_cos"]:+.4f}  '
          f'L2(xx) {r0["L2_cos_xx"]:+.4f}', flush=True)

    t0 = time.time()
    stop_reason = None
    interrupted = False
    try:
        for ep in range(1, CFG['epochs'] + 1):
            tr_pre = full_batch_grad(model, tr_x, tr_y)        # loss at the CURRENT weights
            opt.step()
            reapply_masks(model, dmasks)

            # both measured AFTER the step -- history[k] is one set of weights throughout
            tr_loss = eval_ce(model, tr_x, tr_y, device)
            te_loss = eval_ce(model, te_x, te_y, device)
            rep = report(model, ref)

            if ep == 1 and CFG['qk'] == 'separate' and CFG['init'] == 'zeros' \
                    and rep['L1_qk_fro'] == 0.0 and rep['L2_qk_fro'] == 0.0:
                stop_reason = ('zero-init saddle: after one step W_q and W_k are still exactly '
                               '0. dL/dW_q = (dL/dW_qk) W_k vanishes when W_k = 0. Use '
                               '--init eps, or --qk combined.')
                print(f'    [STOP] {stop_reason}', flush=True)
                break

            # Under a softmax readout the CE is >= 0 by construction, so this can only fire on
            # a genuine numerical blow-up -- unlike the frozen-W_ov scripts, where a negative
            # CE was the expected symptom of unfreezing W_ov.
            if not math.isfinite(tr_loss):
                stop_reason = f'loss not finite at step {ep}: train={tr_loss:.6g}'
                print(f'    [STOP] {stop_reason}', flush=True)
                break

            history['step'].append(ep)
            history['train_loss'].append(tr_loss)
            history['test_loss'].append(te_loss)
            history['train_loss_pre'].append(tr_pre)
            history['L1_cos'].append(rep['L1_cos'])
            history['L2_cos_xx'].append(rep['L2_cos_xx'])
            best_train = min(best_train, tr_loss)
            if te_loss < best_test:
                best_test = te_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            # wandb gets the four loss curves and nothing else. Everything the report()
            # computes -- cosines, norms, |W_ov - I|, the bias means -- still goes to stdout
            # and into metrics_*.json, which is where it is actually read.
            if wb is not None:
                wb.log({'train_loss': tr_loss, 'test_loss': te_loss,
                        'best_train': best_train, 'best_test': best_test}, step=ep)
            every = CFG['log_every'] or max(1, CFG['epochs'] // 20)   # default: every 5%
            if ep % every == 0 or ep == CFG['epochs'] or ep == 1:
                print(f'    step {ep:>5}/{CFG["epochs"]}  train {tr_loss:.5f}  '
                      f'test {te_loss:.5f}  best_test {best_test:.5f}  '
                      f'cos L1 {rep["L1_cos"]:+.4f} L2xx {rep["L2_cos_xx"]:+.4f}  '
                      f'|Wov-I| {rep["L1_ov_off_identity"]:.3g}/'
                      f'{rep["L2_ov_off_identity"]:.3g}  '
                      f'b_noise {rep["b_noise_mean"]:+.3f}', flush=True)
    except KeyboardInterrupt:
        interrupted = True
        print('\n    [interrupted -- saving what exists and moving to the next threshold]',
              flush=True)

    d = run_dir()
    os.makedirs(d, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(d, f'model_final_seed{SEED}_x_{thr}.pth'))
    if best_state is not None:
        torch.save(best_state, os.path.join(d, f'model_best_seed{SEED}_x_{thr}.pth'))

    final = report(model, ref)
    out = {'cell': cell_tag(), 'threshold': thr, 'seed': SEED,
           'config': {**CFG, 'trainable_entries': n_free, 'readout': 'softmax',
                      'head_dim': model.layer1.head_dim, 'beta_init': beta(thr)},
           'grad_norm_at_init': g0,
           'history': history, 'best_train': best_train, 'best_test': best_test,
           'final_train': history['train_loss'][-1], 'final_test': history['test_loss'][-1],
           'final_report': final,
           'final_bias': model.b.detach().cpu().tolist(),
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
          f'best_test {best_test:.5f}  cos L1 {final["L1_cos"]:+.4f} '
          f'L2xx {final["L2_cos_xx"]:+.4f}  -> {metrics_path(thr)}', flush=True)
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
    ap.add_argument('--qk', choices=('combined', 'separate'), default=CFG['qk'],
                    help="'combined' (default) = one W_qk per layer, as in folders 17/18/19; "
                         "'separate' = W_q / W_k as in train_separate_qk.py")
    ap.add_argument('--optimizer', choices=('adam', 'sgd'), default=CFG['optimizer'])
    ap.add_argument('--lr', type=float, default=CFG['lr'])
    ap.add_argument('--epochs', type=int, default=CFG['epochs'])
    ap.add_argument('--init', choices=('eps', 'zeros', 'normal', 'orth'), default=CFG['init'],
                    help="QK init. 'zeros' is a strict saddle under --qk separate")
    ap.add_argument('--init-scale', type=float, default=CFG['init_scale'])
    ap.add_argument('--head-dim', type=int, default=None,
                    help='inner dimension of W_q / W_k (default 108); --qk separate only')
    ap.add_argument('--ov-init', choices=('corner', 'full', 'normal'), default=CFG['ov_init'],
                    help="where the FULL trainable W_ov starts; 'corner' = the 8x8 identity "
                         "folders 17-19 froze it to. It cannot start at zero")
    ap.add_argument('--freeze-wov', action='store_true',
                    help='ablation: keep W_ov frozen after all (the softmax readout stays)')
    ap.add_argument('--bias-init', choices=('logalpha', 'zeros'), default=CFG['bias_init'],
                    help="'logalpha' = the PDF's beta on the noise logits, 0 on the signal ones")
    ap.add_argument('--freeze-bias', action='store_true',
                    help='ablation: hold the readout bias at its initial value')
    ap.add_argument('--blocks', choices=('full', 'relevant'), default=CFG['blocks'],
                    help="'full' = every entry free; 'relevant' is refused while W_ov trains")
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
    CFG.update({'qk': a.qk, 'optimizer': a.optimizer, 'lr': a.lr, 'epochs': a.epochs,
                'init': a.init, 'init_scale': a.init_scale, 'head_dim': a.head_dim,
                'ov_init': a.ov_init, 'train_wov': not a.freeze_wov,
                'bias_init': a.bias_init, 'train_bias': not a.freeze_bias,
                'blocks': a.blocks, 'device': pick_device(a.device), 'log_every': a.log_every,
                'grad_chunk': a.grad_chunk, 'eval_chunk': a.grad_chunk})
    if a.runs_dir:
        CFG['runs_dir'] = os.path.abspath(a.runs_dir)
    if CFG['qk'] == 'combined' and CFG['head_dim'] is not None:
        sys.exit('--head-dim only applies to --qk separate')

    check_dataset(THRESHOLDS)
    device = torch.device(CFG['device'])

    probe = FreeTransformer(CFG['v_sig'], CFG['n_noise'], CFG['seq_len'],
                            CFG['qk'], CFG['head_dim'])
    pm = build_masks(probe)
    _, n_free, pnames = install(probe, pm, THRESHOLDS[0], torch.device('cpu'))
    n_all = sum(p.numel() for p in probe.parameters())
    del probe

    # drop_col4=False so that W_qk2 == Wq2 @ Wk2.T exactly -- see train_separate_qk.py
    ref = construction(CFG['v_sig'], CFG['n_noise'], CFG['seq_len'], paper=False,
                       drop_col4=False)

    print(f'cell       : {cell_tag()}')
    print(f'device     : {device}')
    print(f'thresholds : {list(THRESHOLDS)}')
    print(f'params     : QK {CFG["qk"]}   W_ov '
          f'{"TRAINABLE (full)" if CFG["train_wov"] else "frozen"} init {CFG["ov_init"]}   '
          f'bias {"TRAINABLE" if CFG["train_bias"] else "frozen"} init {CFG["bias_init"]}')
    print(f'readout    : SOFTMAX over the first 8 outputs + bias -- required once W_ov is free, '
          f'since f can go negative and (f+b)/sum(f+b) stops being a probability')
    print(f'free cells : {n_free:,} of {n_all:,}   over {pnames}')
    print(f'             (folder 18 allowed 10,060; folder 19 tied those to 9,910 dof)')
    print(f'init       : {CFG["init"]}'
          + (f' (scale {CFG["init_scale"]:g})' if CFG['init'] in ('eps', 'orth') else '')
          + ('   [!] a strict saddle under --qk separate; the run will stop after step 1'
             if CFG['init'] == 'zeros' and CFG['qk'] == 'separate' else ''))
    print(f'batching   : FULL BATCH in chunks of {CFG["grad_chunk"]} -> 1 epoch = 1 step')
    print(f'reference  : reduced hand construction -- L1_cos and L2_cos_xx are meaningful; '
          f'L2_cos_all is NOT (free W_ov1 leaves layer 1\'s output basis unpinned)')
    print(f'on disk    : model_best / model_final .pth and metrics_*.json only '
          f'(~3 MB per threshold).\n'
          f'             NO per-step weight or gradient snapshots -- at {n_free:,} free cells '
          f'those cost\n             ~357 MB per threshold and are not written here.')
    print()

    for thr in THRESHOLDS:
        train_one(thr, device, ref)
    n = rebuild_summary()
    print(f'\nsummary rebuilt from {n} metrics files -> '
          f'{os.path.join(CFG["runs_dir"], "summary.json")}')


if __name__ == '__main__':
    main()
