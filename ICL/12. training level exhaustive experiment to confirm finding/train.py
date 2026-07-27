"""
Training-level sweep to confirm the combined-vs-separate parametrization finding (folder 11).

Same model/task as folder 11's train.py (frozen log-readout FFN K=128/state, train attn+fc_out,
checkpoint on best TRAIN loss, patience=50 on train loss), run as a FULL CROSS PRODUCT over
training-method axes. Every run is logged to wandb AND to local json/checkpoints -- local files
are the source of truth; wandb is a mirror for browsing/plotting.

Sweep grid (full cross product):
  (optimizer, lr) : adam@0.01, adam@0.003, sgd@0.1, sgd@0.03   (4)
  batch_size      : 128, 256                                    (2)
  init            : default, normal                             (2)
  -> 16 cells; each cell x 2 architectures x 2 seeds x 4 thresholds = 256 runs.

Within any single run both architectures ALWAYS share the identical lr (the core control:
only the parametrization changes). SGD's lr grid is 10x adam's because plain SGD steps by
lr * raw_gradient and needs ~10x larger lr to move at all (folder 11's sgd default was 0.1).

Init schemes (same METHOD applied to both architectures' own native matrices -- no product
matching; that's an architecture-level experiment done separately):
  - 'default': U(-1/sqrt(fan_in), +1/sqrt(fan_in)) on every attention matrix -- exactly
    nn.Linear's default init.
  - 'normal':  N(0, 1/sqrt(fan_in)) on every attention matrix.

Loss curves START AT THE RANDOM INIT: epoch 0 = train/test CE evaluated before any update.

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

# ─── Fixed config (matches folder 11's train.py unless noted) ─────────────────
BASE = {
    'data_dir':  os.path.join(_SCRIPT_DIR, 'dataset'),   # 25k train / 10k test per threshold
    'runs_dir':  os.path.join(_SCRIPT_DIR, 'runs'),
    'device':    'cuda:1',

    'v_sig':       4,
    'n_noise':     4,
    'seq_len':     100,
    'token_start': 50,

    'bias': True,

    'freeze_readout':   True,
    'freeze_delta':     1.0,
    'freeze_hidden_k':  128,
    'freeze_fit_fmax':  40.0,
    'freeze_fit_steps': 2000,

    'train_params': ['attn', 'fc_out'],
    'weight_decay': 0.0,
    'clip_grad':    False,
    'grad_clip':    0.0,
    'epochs':       400,
    'patience':     50,      # on TRAIN loss
    'eval_batch':   256,
}

# ─── Sweep axes (full cross product) ──────────────────────────────────────────
OPT_LR       = [('adam', 0.01), ('adam', 0.003), ('sgd', 0.1), ('sgd', 0.03)]
BATCH_SIZES  = [128, 256]
INIT_SCHEMES = ['default', 'normal']
ARCHS        = ['combined', 'separate']
SEEDS        = [0, 1]
THRESHOLDS   = [0.4, 0.6, 0.8, 1.0]

SKIP_DONE     = True         # skip runs whose metrics json already exists
USE_WANDB     = True
WANDB_PROJECT = 'ICL-exhaustive-experiment-to-verify-claim'
WANDB_ENTITY  = None         # None -> your default wandb entity
# ─────────────────────────────────────────────────────────────────────────────


def cell_tag(opt, lr, bs, init):
    return f'opt-{opt}_lr-{lr}_b-{bs}_init-{init}'


def build_cells():
    return [dict(optimizer=opt, lr=lr, batch_size=bs, init=init)
            for opt, lr in OPT_LR
            for bs in BATCH_SIZES
            for init in INIT_SCHEMES]


# ─── Model (identical to folder 11's train.py) ────────────────────────────────
class CombinedAttentionBlock(nn.Module):
    """QK/OV combined parametrization: scores = x @ W_qk @ x^T ; out = softmax @ (x @ W_ov)."""
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.output_dim = output_dim
        self.W_qk = nn.Parameter(torch.empty(input_dim, input_dim))
        self.W_ov = nn.Parameter(torch.empty(input_dim, output_dim))
        # nn.Linear-default method (U(+-1/sqrt(fan_in))) -- same as the separate block's
        # Linears. apply_init(scheme) re-initializes before training anyway; this just
        # guarantees no code path ever has the two archs initialized by different methods.
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


_ATTN_BLOCKS = {'combined': CombinedAttentionBlock, 'separate': SeparateAttentionBlock}


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
    """(param, fan_in) for every attention matrix. Combined mats are used as x @ W, so
    fan_in = shape[0]; separate mats are nn.Linear weights [out,in], so fan_in = shape[1]."""
    if model.attn_param == 'combined':
        return [(p, p.shape[0]) for layer in (model.layer1, model.layer2)
                for p in (layer.W_qk, layer.W_ov)]
    return [(lin.weight, lin.weight.shape[1])
            for layer in (model.layer1, model.layer2)
            for lin in (layer.W_q, layer.W_k, layer.W_v, layer.W_o)]


def apply_init(model, scheme):
    """Re-initialize all attention matrices with the SAME method for both architectures.
    'default' = U(+-1/sqrt(fan_in)) (exactly nn.Linear's default); 'normal' = N(0, 1/sqrt(fan_in)).
    NOTE for combined this differs from folder 11's xavier_uniform (sqrt(3) smaller bound) --
    the point of this axis is method parity across architectures. fc_out keeps nn.Linear's
    default in both archs (already identical method)."""
    with torch.no_grad():
        for p, fan_in in _attn_mats_with_fanin(model):
            s = 1.0 / math.sqrt(fan_in)
            if scheme == 'default':
                p.uniform_(-s, s)
            elif scheme == 'normal':
                p.normal_(0.0, s)
            else:
                raise ValueError(f'unknown init scheme {scheme!r}')


# ─── Frozen log-readout (identical to folder 11) ──────────────────────────────
def fit_log_core(delta, k, fmax, steps, device='cpu', seed=0):
    torch.manual_seed(seed)
    f = torch.linspace(0, fmax, 400, device=device).unsqueeze(1)
    target = torch.log(f + delta) - f
    breakpoints = torch.linspace(0, fmax, k, device=device)
    a0 = torch.full((1, k), 6.0 / max(fmax, 1e-6), device=device)
    b0 = (-a0.squeeze(0) * breakpoints).unsqueeze(0)
    a = nn.Parameter(a0.clone()); b = nn.Parameter(b0.clone())
    with torch.no_grad():
        basis  = F.gelu(f * a + b)
        basis1 = torch.cat([basis, torch.ones(basis.shape[0], 1, device=device)], dim=1)
        sol    = torch.linalg.lstsq(basis1, target).solution
        w0, c0 = sol[:k], sol[k:k + 1].squeeze(0)
    w = nn.Parameter(w0.clone()); c = nn.Parameter(c0.clone())
    opt = torch.optim.Adam([a, b, w, c], lr=1e-2)
    for _ in range(steps):
        opt.zero_grad()
        pred = F.gelu(f * a + b) @ w + c
        loss = F.mse_loss(pred, target)
        loss.backward(); opt.step()
    return a.detach(), b.detach(), w.detach(), c.detach(), float(loss.item())


_LOG_CORE_CACHE = {}   # fit depends only on (delta,k,fmax,steps) -- fit ONCE, reuse in all runs

def build_frozen_ffn(model, delta, k, fmax, steps):
    dev = next(model.parameters()).device
    key = (delta, k, fmax, steps)
    if key not in _LOG_CORE_CACHE:
        _LOG_CORE_CACHE[key] = fit_log_core(delta, k, fmax, steps, device=dev)
        print(f'  [frozen readout] fit MSE={_LOG_CORE_CACHE[key][4]:.2e}  (cached for all runs)')
    a, b, w, c, _ = _LOG_CORE_CACHE[key]
    a, b, w, c = a.to(dev), b.to(dev), w.to(dev), c.to(dev)
    Vs, D, in2 = model.v_sig, model.D, model.in2
    W1 = torch.zeros(Vs * k, D, device=dev); B1 = torch.zeros(Vs * k, device=dev)
    W2 = torch.zeros(D, Vs * k, device=dev); B2 = torch.zeros(D, device=dev)
    for v in range(Vs):
        rows = slice(v * k, (v + 1) * k)
        W1[rows, in2 + v] = a.squeeze(0)
        B1[rows]          = b.squeeze(0)
        W2[in2 + v, rows] = w.squeeze(1)
        B2[in2 + v]       = c
    with torch.no_grad():
        model.ff1.weight.copy_(W1); model.ff1.bias.copy_(B1)
        model.ff2.weight.copy_(W2); model.ff2.bias.copy_(B2)
    for p in (model.ff1.weight, model.ff1.bias, model.ff2.weight, model.ff2.bias):
        p.requires_grad_(False)


# ─── Data / training infra ────────────────────────────────────────────────────
_DATA_CACHE = {}   # (split,thr) -> blob; holds ONE threshold's train+test at a time

def load_data(split, thr):
    key = (split, thr)
    if key not in _DATA_CACHE:
        if any(k[1] != thr for k in _DATA_CACHE):   # new threshold -> drop the old one's blobs
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


# ─── wandb (optional mirror; local json is the source of truth) ───────────────
_WANDB_OK = None

def wandb_start(run_cfg, name, group, tags):
    global _WANDB_OK
    if not USE_WANDB or _WANDB_OK is False:
        return None
    try:
        import wandb
        run = wandb.init(project=WANDB_PROJECT, entity=WANDB_ENTITY, name=name, group=group,
                         tags=tags, config=run_cfg, reinit='finish_previous',
                         dir=os.path.join(_SCRIPT_DIR, 'wandb_local'))
        _WANDB_OK = True
        return run
    except Exception as e:
        if _WANDB_OK is None:
            print(f'  [wandb] disabled ({type(e).__name__}: {e}) -- local logging only')
        _WANDB_OK = False
        return None


# ─── One run = (cell, arch, seed, threshold) ──────────────────────────────────
def metrics_path_for(cell, arch, seed, thr):
    tag = cell_tag(cell['optimizer'], cell['lr'], cell['batch_size'], cell['init'])
    return os.path.join(BASE['runs_dir'], tag, f'metrics_{arch}_seed{seed}_x_{thr}.json')


def train_one(cell, arch, seed, thr, device):
    C = BASE
    V = C['v_sig'] + C['n_noise']
    seq_len = C['seq_len']; token_start = C['token_start']; token_end = seq_len
    attn_dim = V + seq_len
    tag = cell_tag(cell['optimizer'], cell['lr'], cell['batch_size'], cell['init'])

    out_dir = os.path.join(C['runs_dir'], tag)
    os.makedirs(out_dir, exist_ok=True)
    stem         = f'{arch}_seed{seed}_x_{thr}'
    metrics_path = os.path.join(out_dir, f'metrics_{stem}.json')
    model_path   = os.path.join(out_dir, f'model_{stem}.pth')
    if SKIP_DONE and os.path.exists(metrics_path):
        print(f'  [skip done] {tag}/{stem}')
        return False

    run_cfg = {**{k: v for k, v in C.items() if k not in ('runs_dir', 'data_dir')},
               'optimizer': cell['optimizer'], 'lr': cell['lr'],
               'batch_size': cell['batch_size'], 'init': cell['init'],
               'attn_param': arch, 'seed': seed, 'threshold': thr, 'cell': tag}

    torch.manual_seed(seed); np.random.seed(seed)
    train_blob = load_data('train', thr)
    test_blob  = load_data('test',  thr)
    tr_x, tr_y = train_blob['x'], train_blob['y']
    te_x, te_y = test_blob['x'], test_blob['y']

    model = SimplifiedTransformerMLP(V, seq_len, attn_dim, C['v_sig'], C['freeze_hidden_k'],
                                      bias=C['bias'], attn_param=arch).to(device)
    apply_init(model, cell['init'])
    if C['freeze_readout']:
        build_frozen_ffn(model, C['freeze_delta'], C['freeze_hidden_k'],
                          C['freeze_fit_fmax'], C['freeze_fit_steps'])
    set_trainable(model, C['train_params'], C['freeze_readout'])
    params = get_trainable_params(model, C['train_params'])
    opt = make_optimizer(params, cell['optimizer'], cell['lr'], C['weight_decay'])

    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(tr_x, tr_y), batch_size=cell['batch_size'], shuffle=True)

    wb = wandb_start(run_cfg, name=f'{tag}__{stem}', group=tag,
                     tags=[arch, f'thr{thr}', f'seed{seed}', cell['optimizer'], cell['init'],
                           f'b{cell["batch_size"]}'])

    # epoch 0 = the RANDOM INIT: first point of both curves, before any update
    tr0 = eval_ce(model, tr_x, tr_y, V, token_start, token_end, device, C['eval_batch'])
    te0 = eval_ce(model, te_x, te_y, V, token_start, token_end, device, C['eval_batch'])
    history = {'train_loss': [tr0], 'test_loss': [te0]}
    if wb: wb.log({'train_ce': tr0, 'test_ce': te0}, step=0)

    n_params = sum(p.numel() for p in params)
    print(f'  {tag}/{stem}  trainable={n_params:,}  init: tr={tr0:.4f} te={te0:.4f}', flush=True)

    best_train = float('inf'); test_at_best = None; best_state = None; stale = 0
    t0 = time.time(); ep = 0
    try:
        for ep in range(1, C['epochs'] + 1):
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

            if tr_loss < best_train - 1e-6:
                best_train = tr_loss; test_at_best = te_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                stale = 0
            else:
                stale += 1
            if wb: wb.log({'train_ce': tr_loss, 'test_ce': te_loss,
                            'best_train': best_train}, step=ep)
            if ep % 25 == 0:
                print(f'    ep {ep:4d}/{C["epochs"]}  tr={tr_loss:.4f}  te={te_loss:.4f}  '
                      f'best={best_train:.4f}', flush=True)
            if stale >= C['patience']:
                print(f'    [early stop] ep={ep}  best_train={best_train:.4f}')
                break
    except KeyboardInterrupt:
        print('    [INTERRUPTED] saving best-so-far, moving to next run '
              '(Ctrl+C again within ~1s exits the sweep)', flush=True)
        time.sleep(1)

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    torch.save(model.state_dict(), model_path)
    result = {'cell': tag, 'arch': arch, 'seed': seed, 'threshold': float(thr),
              'history': history, 'best_train': best_train, 'test_at_best': test_at_best,
              'epochs_run': ep, 'time_s': round(time.time() - t0, 1), 'config': run_cfg}
    with open(metrics_path, 'w') as fh:
        json.dump(result, fh, indent=2)
    if wb:
        wb.summary.update({'best_train': best_train, 'test_at_best': test_at_best,
                            'epochs_run': ep})
        wb.finish()
    print(f'    done in {result["time_s"]/60:.1f}m  best_train={best_train:.4f}  '
          f'test_at_best={test_at_best:.4f}  -> {metrics_path}', flush=True)
    return True


def rebuild_summary():
    """Scan ALL on-disk metrics into one summary json (re-running never shrinks it)."""
    rows = []
    for mp in sorted(glob.glob(os.path.join(BASE['runs_dir'], '*', 'metrics_*.json'))):
        try:
            m = json.load(open(mp))
            rows.append({k: m[k] for k in
                         ('cell', 'arch', 'seed', 'threshold', 'best_train',
                          'test_at_best', 'epochs_run')})
        except (ValueError, KeyError):
            pass
    out = os.path.join(BASE['runs_dir'], 'sweep_summary.json')
    with open(out, 'w') as fh:
        json.dump(rows, fh, indent=2)
    print(f'\nSummary ({len(rows)} runs) -> {out}')


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    device = torch.device(BASE['device'] if torch.cuda.is_available() else 'cpu')
    cells = build_cells()
    # threshold OUTERMOST so each threshold's train+test blobs load once for all 16 cells
    runs = [(cell, arch, seed, thr)
            for thr in THRESHOLDS
            for cell in cells
            for arch in ARCHS
            for seed in SEEDS]
    n_total = len(runs)
    n_done = sum(1 for r in runs if os.path.exists(metrics_path_for(*r[:3], r[3])))

    print(f'device : {device}')
    print(f'cells  : {len(cells)} = (opt,lr){len(OPT_LR)} x batch{len(BATCH_SIZES)} '
          f'x init{len(INIT_SCHEMES)}')
    for c in cells:
        print(f'  {cell_tag(c["optimizer"], c["lr"], c["batch_size"], c["init"])}')
    print(f'runs   : {n_total} = cells x {len(ARCHS)} archs x {len(SEEDS)} seeds '
          f'x {len(THRESHOLDS)} thresholds   ({n_done} already done, will skip)')
    print(f'wandb  : {"ON, project=" + WANDB_PROJECT if USE_WANDB else "OFF"} '
          f'(local json/checkpoints always written)\n')

    os.makedirs(BASE['runs_dir'], exist_ok=True)
    os.makedirs(os.path.join(_SCRIPT_DIR, 'wandb_local'), exist_ok=True)
    t_start = time.time(); n_run = 0
    for i, (cell, arch, seed, thr) in enumerate(runs):
        did = train_one(cell, arch, seed, thr, device)
        if did:
            n_run += 1
            avg = (time.time() - t_start) / n_run
            remaining = sum(1 for r in runs[i + 1:]
                            if not os.path.exists(metrics_path_for(*r[:3], r[3])))
            print(f'  [{i + 1}/{n_total}]  avg={avg/60:.1f}m/run  '
                  f'ETA={remaining * avg / 3600:.1f}h\n', flush=True)
    rebuild_summary()


if __name__ == '__main__':
    main()


