"""Master training script -- minimal two-layer QK/OV model, block-sparse, no readout head.

Written from scratch; nothing is imported from folders 6/7/8/14/17. Hyperparameters that carry
over from those scripts are kept identical; everything the new architecture makes meaningless
(readout bias b_sig/b_noise, the unembedding, the log, the final softmax) is gone.

ARCHITECTURE
------------
    x            one-hot token (8) + one-hot position (100)        -> 108
    layer 1      attention, returns cat[x, attn_out]               -> 216
    layer 2      attention, returns attn_out only                  -> 108
    p = layer2 attn_out[..., :8]                                   -> the predicted distribution

There is no unembedding, no bias, no log-readout and no final softmax. Layer 2's attention output
already IS a distribution over the 8 tokens: its rows are a convex combination (attention weights,
which sum to 1) of token one-hots, so with W_ov[:8,:8] = I it is non-negative and sums to exactly 1.
The loss is cross-entropy read straight off it:

    loss = -log( p[target] )        floored at LOG_FLOOR to keep log(0) finite

When W_ov is trainable the output is no longer guaranteed normalised, so p is divided by its own
sum first (a no-op in the frozen-identity case).

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

TRAINING
--------
Full batch: the whole 25,000-sequence training split is one optimizer step, so 1 epoch = 1 step.
The gradient is accumulated over chunks of `grad_chunk` sequences (each chunk's mean loss reweighted
by its share of the scored tokens) -- mathematically identical to one 25,000-sequence backward,
at a fraction of the memory.

`INIT` controls W_qk only ('zeros' or 'normal'), which is what the zero-init line of work means by
it. W_ov's trainable block always starts at the identity: starting it at zero would make p == 0
everywhere, the log floor would flatten the gradient, and nothing would train at all.

Usage:
    python master_train.py                       # defaults below
    python master_train.py --epochs 500 --init normal --optimizer sgd --train-wov
"""

import os
import sys
import glob
import json
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

    # --- batching / evaluation ---
    'grad_chunk':  2048,       # memory only; does not change the gradient
    'eval_batch':  256,
}

LOG_FLOOR  = 1e-30             # p[target] is clamped here before the log
SEED       = 0
THRESHOLDS = [0.6, 0.8, 1.0]

USE_WANDB     = True
WANDB_PROJECT = 'ICL-reduced-params'
WANDB_ENTITY  = None
SKIP_DONE     = True


def cell_tag():
    return (f"master_opt-{CFG['optimizer']}_lr-{CFG['lr']}_init-{CFG['init']}"
            f"_wov-{'train' if CFG['train_wov'] else 'fixed'}_ep-{CFG['epochs']}")


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
    """Boolean masks marking the only entries allowed to be nonzero. Everything else stays 0."""
    V, d1, d2 = model.V, model.d1, model.d2
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

def ce_loss(p, y):
    """Cross entropy reading p directly as a distribution over the V tokens.

    p is normalised by its own sum first -- a no-op when W_ov is the frozen identity (attention
    weights sum to 1 and each row of x @ W_ov is a token one-hot), and the thing that keeps the
    loss meaningful when W_ov is trainable and the output need not sum to 1.
    """
    s = p.sum(-1, keepdim=True)
    p = p / torch.where(s.abs() < LOG_FLOOR, torch.ones_like(s), s)
    q = p.gather(-1, y.unsqueeze(-1)).squeeze(-1)
    return -torch.log(q.clamp_min(LOG_FLOOR)).mean()


def full_batch_grad(model, x, y):
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
        loss = ce_loss(model(xb)[:, ts:te], yb)
        (loss * w).backward()
        total += float(loss) * w
    return total


@torch.no_grad()
def eval_ce(model, x, y, device):
    model.eval()
    ts, te = CFG['token_start'], CFG['seq_len']
    total, n = 0.0, 0
    for i in range(0, x.shape[0], CFG['eval_batch']):
        xb = x[i:i + CFG['eval_batch']].to(device)
        yb = y[i:i + CFG['eval_batch'], ts:te].to(device)
        total += float(ce_loss(model(xb)[:, ts:te], yb)) * yb.numel()
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

    run_cfg = {**{k: v for k, v in CFG.items() if k not in ('runs_dir', 'data_dir')},
               'threshold': float(thr), 'seed': SEED, 'log_floor': LOG_FLOOR,
               'n_train': int(tr_x.shape[0]), 'n_test': int(te_x.shape[0]),
               'trainable_entries': n_free,
               'tensor_entries': sum(p.numel() for p in params)}
    wb = wandb_start(run_cfg, name=f'{tag}__x_{thr}', group=f'thr{thr}',
                     tags=[f'thr{thr}', CFG['init'], CFG['optimizer']])
    if wb:
        for m in ('train_loss', 'test_loss'):
            wb.define_metric(m, summary='min')
        for m in ('best_train', 'best_test'):
            wb.define_metric(m, summary='last')

    tr0, te0 = eval_ce(model, tr_x, tr_y, device), eval_ce(model, te_x, te_y, device)
    history = {'step': [0], 'train_loss': [tr0], 'test_loss': [te0]}
    best_train, best_test, best_state = tr0, te0, None
    if wb:
        wb.log({'train_loss': tr0, 'test_loss': te0,
                'best_train': tr0, 'best_test': te0}, step=0)

    n_all = sum(p.numel() for p in model.parameters())
    print(f'  alpha={thr}  free entries {n_free:,} of {n_all:,} '
          f'({n_free / n_all:.2%} of the four matrices)')
    print(f'    init: train {tr0:.5f}  test {te0:.5f}', flush=True)

    t0, ep, interrupted = time.time(), 0, False
    try:
        for ep in range(1, CFG['epochs'] + 1):
            model.train()
            tr_loss = full_batch_grad(model, tr_x, tr_y)
            opt.step()
            reapply_masks(model, masks, device)
            te_loss = eval_ce(model, te_x, te_y, device)

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
              'steps_run': ep, 'time_s': round(time.time() - t0, 1)}
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
                    help='train the two 8x8 W_ov selector blocks (default: frozen identity)')
    ap.add_argument('--device', default=CFG['device'])
    ap.add_argument('--thresholds', type=float, nargs='+', default=THRESHOLDS)
    ap.add_argument('--no-wandb', action='store_true')
    return ap.parse_args()


def main():
    global THRESHOLDS, USE_WANDB
    a = parse_args()
    CFG.update({'epochs': a.epochs, 'lr': a.lr, 'optimizer': a.optimizer, 'init': a.init,
                'train_wov': a.train_wov, 'device': a.device})
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
    print(f'model      : 2 attention layers, no unembedding / bias / log / final softmax;')
    print(f'             p = layer2 attn_out[..., :{CFG["v_sig"] + CFG["n_noise"]}], '
          f'loss = -log p[target] floored at {LOG_FLOOR:g}')
    print(f'trainable  : {n_free:,} entries of {n_all:,} ({n_free / n_all:.2%})  '
          f'= W_qk blocks {n_qk:,}'
          + (f' + W_ov blocks {n_ov:,}' if CFG['train_wov']
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
