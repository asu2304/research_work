"""
Supporting experiment: does the COMBINED parametrization keep improving past its usual budget,
and does a LARGER learning rate help it reach a lower minimum?

Reproduces exactly the folder-13 sweep cell that looked like it was still descending at 500
epochs on wandb:

    opt-adam_lr-0.01_b-256_init-normal   __   combined   seed1   x_0.4
    (adam, batch=256, init='normal', arch='combined', seed=1, threshold=0.4)

...but with wandb removed, epochs pushed to 500, EARLY STOPPING DISABLED (we WANT the whole
trajectory, not a patience cutoff), and the learning rate swept over LR_SWEEP so we can see
whether a bigger step gets combined to a lower loss.

Everything else (model, frozen K=128 log-readout, train attn+fc_out, checkpoint on best TRAIN
loss, epoch-0 point at the random init, the 25k/10k dataset) is identical to folder 13's
train.py. Uses folder 13's dataset in place (no copy). Saves per-lr checkpoints + metrics json
NEXT TO THIS SCRIPT; the companion notebook plots from those.

All three lr runs start from the SAME init and SAME batch order (re-seeded per run, loader given
its own seeded generator), so lr is the only thing that varies -- the point of the experiment.

Run in tmux:
    python -u train_increasing_epochs_for_combined.py
"""

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
    _ICL_DIR, '13. training level exhaustive experiment to confirm finding', 'dataset')

# ─── The fixed cell being reproduced + what we vary ───────────────────────────
CONFIG = {
    'data_dir':    _DATA_DIR,           # 25k train / 10k test per threshold (folder 13)
    'device':      'cuda:0',

    'v_sig':       4,
    'n_noise':     4,
    'seq_len':     100,
    'token_start': 50,
    'bias':        True,

    # ── the reproduced sweep cell (everything below is FIXED) ─────────────────
    'attn_param':  'combined',
    'init':        'normal',            # N(0, 1/sqrt(fan_in)) on every attention matrix
    'optimizer':   'adam',
    'batch_size':  256,
    'seed':        1,
    'threshold':   0.4,

    # ── frozen log-readout (identical to folder 13) ───────────────────────────
    'freeze_readout':   True,
    'freeze_delta':     1.0,
    'freeze_hidden_k':  128,
    'freeze_fit_fmax':  40.0,
    'freeze_fit_steps': 2000,
    'train_params': ['attn', 'fc_out'],
    'weight_decay': 0.0,
    'clip_grad':    False,
    'grad_clip':    0.0,
    'eval_batch':   256,

    # ── what THIS experiment changes ──────────────────────────────────────────
    'epochs':      500,                 # was 300/400 in the sweep
    'early_stop':  False,               # DISABLED: keep the full 500-epoch trajectory
    'patience':    50,                  # unused when early_stop=False (kept for the record)
    'lr_sweep':    [0.02, 0.03, 0.04],   # the whole point: does a bigger adam step help combined?
}
# ─────────────────────────────────────────────────────────────────────────────


# ─── Model (identical to folder 13's train.py) ────────────────────────────────
class CombinedAttentionBlock(nn.Module):
    """QK/OV combined parametrization: scores = x @ W_qk @ x^T ; out = softmax @ (x @ W_ov)."""
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.output_dim = output_dim
        self.W_qk = nn.Parameter(torch.empty(input_dim, input_dim))
        self.W_ov = nn.Parameter(torch.empty(input_dim, output_dim))
        s = input_dim ** -0.5                    # nn.Linear-default method; apply_init overwrites
        nn.init.uniform_(self.W_qk, -s, s)
        nn.init.uniform_(self.W_ov, -s, s)

    def forward(self, x, mask=None):
        scores = torch.bmm(x @ self.W_qk, x.transpose(1, 2)) / (self.output_dim ** 0.5)
        if mask is not None:
            scores = scores + mask
        attn_out = torch.bmm(F.softmax(scores, dim=-1), x @ self.W_ov)
        return torch.cat([x, attn_out], dim=-1), scores


class SeparateAttentionBlock(nn.Module):
    """Separate Wq,Wk,Wv,Wo parametrization (kept for parity; unused when attn_param='combined')."""
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
    if model.attn_param == 'combined':
        return [(p, p.shape[0]) for layer in (model.layer1, model.layer2)
                for p in (layer.W_qk, layer.W_ov)]
    return [(lin.weight, lin.weight.shape[1])
            for layer in (model.layer1, model.layer2)
            for lin in (layer.W_q, layer.W_k, layer.W_v, layer.W_o)]


def apply_init(model, scheme):
    """Same method for both archs. 'default' = U(+-1/sqrt(fan_in)); 'normal' = N(0, 1/sqrt(fan_in))."""
    with torch.no_grad():
        for p, fan_in in _attn_mats_with_fanin(model):
            s = 1.0 / math.sqrt(fan_in)
            if scheme == 'default':
                p.uniform_(-s, s)
            elif scheme == 'normal':
                p.normal_(0.0, s)
            else:
                raise ValueError(f'unknown init scheme {scheme!r}')


# ─── Frozen log-readout (identical to folder 13) ──────────────────────────────
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


_LOG_CORE_CACHE = {}

def build_frozen_ffn(model, delta, k, fmax, steps):
    dev = next(model.parameters()).device
    key = (delta, k, fmax, steps)
    if key not in _LOG_CORE_CACHE:
        _LOG_CORE_CACHE[key] = fit_log_core(delta, k, fmax, steps, device=dev)
        print(f'  [frozen readout] fit MSE={_LOG_CORE_CACHE[key][4]:.2e}  (cached)')
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


# ─── Data / infra ─────────────────────────────────────────────────────────────
def load_data(data_dir, split, thr):
    return torch.load(os.path.join(data_dir, split, f'data_x_{thr}.pt'), weights_only=False)


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


# ─── One lr run (fixed cell, this lr) ─────────────────────────────────────────
def train_one_lr(lr, device, tr_x, tr_y, te_x, te_y):
    C = CONFIG
    V = C['v_sig'] + C['n_noise']
    seq_len = C['seq_len']; token_start = C['token_start']; token_end = seq_len
    attn_dim = V + seq_len

    # identical init + identical batch order across every lr -> lr is the only difference
    torch.manual_seed(C['seed']); np.random.seed(C['seed'])
    model = SimplifiedTransformerMLP(V, seq_len, attn_dim, C['v_sig'], C['freeze_hidden_k'],
                                      bias=C['bias'], attn_param=C['attn_param']).to(device)
    apply_init(model, C['init'])
    if C['freeze_readout']:
        build_frozen_ffn(model, C['freeze_delta'], C['freeze_hidden_k'],
                          C['freeze_fit_fmax'], C['freeze_fit_steps'])
    set_trainable(model, C['train_params'], C['freeze_readout'])
    params = get_trainable_params(model, C['train_params'])
    opt = torch.optim.Adam(params, lr=lr, weight_decay=C['weight_decay'])

    gen = torch.Generator().manual_seed(C['seed'])   # deterministic, lr-independent batch order
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(tr_x, tr_y), batch_size=C['batch_size'],
        shuffle=True, generator=gen)

    stem         = f'combined_seed{C["seed"]}_x_{C["threshold"]}_lr-{lr}'
    model_path   = os.path.join(_SCRIPT_DIR, f'model_{stem}.pth')
    metrics_path = os.path.join(_SCRIPT_DIR, f'metrics_{stem}.json')

    # epoch 0 = the random init (first point of both curves)
    tr0 = eval_ce(model, tr_x, tr_y, V, token_start, token_end, device, C['eval_batch'])
    te0 = eval_ce(model, te_x, te_y, V, token_start, token_end, device, C['eval_batch'])
    history = {'train_loss': [tr0], 'test_loss': [te0]}
    print(f"\n{'='*72}\n  lr={lr}  init: tr={tr0:.4f}  te={te0:.4f}  "
          f"trainable={sum(p.numel() for p in params):,}\n{'='*72}", flush=True)

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
            if ep == 1 or ep % 25 == 0:
                print(f'  ep {ep:4d}/{C["epochs"]}  tr={tr_loss:.4f}  te={te_loss:.4f}  '
                      f'best_train={best_train:.4f}', flush=True)
            if C['early_stop'] and stale >= C['patience']:
                print(f'  [early stop] ep={ep}  best_train={best_train:.4f}')
                break
    except KeyboardInterrupt:
        print('  [INTERRUPTED] saving best-so-far, moving to next lr', flush=True)
        time.sleep(1)

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    torch.save(model.state_dict(), model_path)
    result = {'lr': lr, 'history': history, 'best_train': best_train,
              'test_at_best': test_at_best, 'epochs_run': ep,
              'final_train': history['train_loss'][-1], 'final_test': history['test_loss'][-1],
              'time_s': round(time.time() - t0, 1),
              'config': {k: v for k, v in C.items() if k != 'lr_sweep'}}
    with open(metrics_path, 'w') as fh:
        json.dump(result, fh, indent=2)
    print(f'  done in {result["time_s"]/60:.1f}m  best_train={best_train:.4f}  '
          f'test_at_best={test_at_best:.4f}  final_tr={result["final_train"]:.4f}  '
          f'-> {metrics_path}', flush=True)
    return result


def main():
    C = CONFIG
    device = torch.device(C['device'] if torch.cuda.is_available() else 'cpu')
    print(f'device      : {device}')
    print(f'reproducing : opt-adam_lr-<sweep>_b-{C["batch_size"]}_init-{C["init"]}__'
          f'{C["attn_param"]}_seed{C["seed"]}_x_{C["threshold"]}')
    print(f'lr sweep    : {C["lr_sweep"]}   epochs={C["epochs"]}   early_stop={C["early_stop"]}')
    print(f'data        : {C["data_dir"]}')

    train_blob = load_data(C['data_dir'], 'train', C['threshold'])
    test_blob  = load_data(C['data_dir'], 'test',  C['threshold'])
    tr_x, tr_y = train_blob['x'], train_blob['y']
    te_x, te_y = test_blob['x'], test_blob['y']
    print(f'shapes      : train={tuple(tr_x.shape)}  test={tuple(te_x.shape)}\n')

    for lr in C['lr_sweep']:
        train_one_lr(lr, device, tr_x, tr_y, te_x, te_y)

    print('\nAll lr runs done. Load metrics_*.json in the notebook to plot.')


if __name__ == '__main__':
    main()
