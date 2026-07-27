"""
Train the simplified reduced-state model (combined-QK/OV attention + frozen log-readout FFN)
on pre-generated data (run data_generation.py first). This script does ONLY training -- safe to
run in a tmux session on the shared box so a dropped connection doesn't kill the run. The
companion notebook simplified_setting.ipynb LOADS the checkpoints/metrics this script writes for
plotting and baseline comparison (it does not retrain).

CONFIG below must stay in sync with simplified_setting.ipynb's CONFIG cell (same values --
V_SIG/N_NOISE/SEQ_LEN/ATTN_DIM/FREEZE_* must match at load time, since the model's shape depends
on them). Also mirrors data_generation.py's V_SIG/N_NOISE/SEQ_LEN.

Ctrl+C during a threshold's epoch loop saves that threshold's best-so-far checkpoint and moves on
to the next threshold (same convention as folder 10's train_identical_arch.py).

Edit CONFIG below, then run in tmux:
    python -u train.py
"""

import json
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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ─── Config (must match simplified_setting.ipynb's CONFIG cell) ──────────────
CONFIG = {
    'data_dir':   'dataset',
    'models_dir': None,       # resolved below to models_{attn_param} -- keeps combined/separate
                              # runs from overwriting each other's checkpoints
    'device':     'cuda:0',

    # ── task size (must match data_generation.py) ─────────────────────────────
    'v_sig':       4,
    'n_noise':     4,
    'seq_len':     100,
    'token_start': 50,        # loss/eval window: last (seq_len - token_start) tokens

    'thresholds': [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],

    # No held-out validation split: trains on the FULL train blob, checkpoints on best TRAIN loss
    # (test loss is still tracked/saved each epoch for monitoring, just not used for selection).
    'seed': 7,

    'bias': True,

    # 'combined' -> W_qk/W_ov (folder 7 style, gauge-free) ; 'separate' -> Wq,Wk,Wv,Wo
    # (folders 6/8/10 style, keeps the within-head rotation gauge)
    'attn_param': 'combined',

    # ── frozen log-readout construction ────────────────────────────────────────
    'freeze_readout':   True,
    'freeze_delta':     1.0,
    'freeze_hidden_k':  128,   # GELU hidden units per signal state approximating log(f+delta);
                               # matches this project's standard mlp_hidden=128 convention
                               # (folders 6/8/10); raised from 12 for a much tighter fit.
    'freeze_fit_fmax':  40.0,
    'freeze_fit_steps': 2000,

    # 'attn' -> attention params (whichever attn_param uses) ; 'ffn' -> ff1/ff2 (forced-frozen
    # if freeze_readout) ; 'fc_out' -> fc_out.weight/.bias
    'train_params': ['attn', 'fc_out'],

    # ── optimization ─────────────────────────────────────────────────────────────
    'optimizer':    'adam',   # 'adam' or 'sgd'
    'lr':           None,     # None -> per-optimizer default (1e-3 adam / 0.1 sgd)
    'momentum':     0.9,      # only used when optimizer='sgd'
    'weight_decay': 0.0,
    'clip_grad':    False,     # toggle gradient clipping on/off
    'grad_clip':    0.00,      # max grad norm, only applied when clip_grad=True
    'epochs':       100,
    'batch_size':   256,
    'patience':     50,
    'eval_batch':   256,      # bounds peak GPU memory on a contested shared box (CLAUDE.md §6)
}

CONFIG['lr'] = CONFIG['lr'] or {'adam': 1e-2, 'sgd': 0.1}[CONFIG['optimizer']]
CONFIG['models_dir'] = CONFIG['models_dir'] or f"models_{CONFIG['attn_param']}"
TAG = f"simplified_{CONFIG['optimizer']}"
# ─────────────────────────────────────────────────────────────────────────────


# ─── Model: combined- OR separate-QK/OV attention + (frozen) log-readout FFN ──
class CombinedAttentionBlock(nn.Module):
    """QK/OV combined parametrization (folder 7 style): scores = x @ W_qk @ x^T ;
    out = softmax(scores) @ (x @ W_ov). Removes the within-head Q/K, V/O rotation gauge --
    W_qk, W_ov ARE the only free parameters (exact reparametrization, not an approximation)."""
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.output_dim = output_dim
        self.W_qk = nn.Parameter(torch.empty(input_dim, input_dim))
        self.W_ov = nn.Parameter(torch.empty(input_dim, output_dim))
        nn.init.xavier_uniform_(self.W_qk)
        nn.init.xavier_uniform_(self.W_ov)

    def forward(self, x, mask=None):
        scores = torch.bmm(x @ self.W_qk, x.transpose(1, 2)) / (self.output_dim ** 0.5)
        if mask is not None:
            scores = scores + mask
        attn_out = torch.bmm(F.softmax(scores, dim=-1), x @ self.W_ov)
        return torch.cat([x, attn_out], dim=-1), scores


class SeparateAttentionBlock(nn.Module):
    """Separate Wq,Wk,Wv,Wo parametrization (folders 6/8/10 style). Unlike the combined block,
    this keeps the within-head Q/K and V/O rotation gauge -- Wq,Wk individually free even though
    only WqᵀWk affects the output (same for Wo·Wv). Strictly more parameters than the combined
    form for the same functional capacity; useful for comparing whether GD's solution under the
    two parametrizations differs beyond that gauge freedom."""
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
    """Two attention layers (combined W_qk/W_ov OR separate Wq/Wk/Wv/Wo, per attn_param) + a
    position-wise GELU FFN head + linear unembedding. `in2` marks where layer-2's attention-output
    block begins -- the coordinate convention the project's hand-built W* construction has always
    used for "where the frequency counts live"; this is architecture-level (concat structure), so
    it holds regardless of which attention parametrization is chosen."""
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


# ─── Frozen log-readout: fit + install ────────────────────────────────────────
def fit_log_core(delta, k, fmax, steps, device='cpu', seed=0):
    """Fit CORE(f) = GELU(f*a+b) @ w + c to the residual target log(f+delta) - f, via a
    closed-form spline warm-start (GELU breakpoints spread over [0,fmax], linear least-squares
    for w,c) followed by a short Adam refinement."""
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


def build_frozen_ffn(model, delta, k, fmax, steps):
    """Overwrite model.ff1/ff2 block-diagonally: state v's K hidden units read/write ONLY
    coordinate (in2+v). Noise floor is left for the trainable fc_out.bias to learn directly."""
    dev = next(model.parameters()).device
    a, b, w, c, mse = fit_log_core(delta, k, fmax, steps, device=dev)
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
    print(f'  [frozen readout] fit MSE={mse:.2e}  over f in [0,{fmax}]  (delta={delta}, '
          f'K={k}/state, {Vs} states)')
    return mse


# ─── Data (pre-generated by data_generation.py) ───────────────────────────────
def load_data(data_dir, split, thr):
    return torch.load(os.path.join(data_dir, split, f'data_x_{thr}.pt'), weights_only=False)


# ─── Training infra ────────────────────────────────────────────────────────────
def _attn_params(model):
    """Return the list of attention parameters, whichever parametrization the model uses."""
    if model.attn_param == 'combined':
        return [model.layer1.W_qk, model.layer1.W_ov, model.layer2.W_qk, model.layer2.W_ov]
    return [p for layer in (model.layer1, model.layer2)
            for p in (layer.W_q.weight, layer.W_k.weight, layer.W_v.weight, layer.W_o.weight)]


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


def make_optimizer(params, kind, lr, momentum, wd):
    if kind == 'sgd':
        return torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=wd)
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


# ─── Train one threshold ──────────────────────────────────────────────────────
def train_one(thr, thr_idx, n_thrs, device):
    C = CONFIG
    V = C['v_sig'] + C['n_noise']
    seq_len = C['seq_len']; token_start = C['token_start']; token_end = seq_len
    attn_dim = V + seq_len

    torch.manual_seed(C['seed']); np.random.seed(C['seed'])
    train_blob = load_data(C['data_dir'], 'train', thr)
    test_blob  = load_data(C['data_dir'], 'test',  thr)
    tr_x, tr_y = train_blob['x'], train_blob['y']       # full train set, no held-out split
    te_x, te_y = test_blob['x'], test_blob['y']

    model = SimplifiedTransformerMLP(V, seq_len, attn_dim, C['v_sig'], C['freeze_hidden_k'],
                                      bias=C['bias'], attn_param=C['attn_param']).to(device)
    if C['freeze_readout']:
        build_frozen_ffn(model, C['freeze_delta'], C['freeze_hidden_k'],
                          C['freeze_fit_fmax'], C['freeze_fit_steps'])
    set_trainable(model, C['train_params'], C['freeze_readout'])
    params = get_trainable_params(model, C['train_params'])
    opt = make_optimizer(params, C['optimizer'], C['lr'], C['momentum'], C['weight_decay'])

    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(tr_x, tr_y), batch_size=C['batch_size'], shuffle=True)

    print(f"\n{'='*72}")
    n_params = sum(p.numel() for p in params)
    print(f"  [{thr_idx+1}/{n_thrs}] thr={thr}  optimizer={C['optimizer']}  lr={C['lr']}  "
          f"attn_param={C['attn_param']}  train_params={C['train_params']}  "
          f"freeze_readout={C['freeze_readout']}  clip_grad={C['clip_grad']}  "
          f"trainable_params={n_params:,}")
    print(f"  data: train={tr_x.shape[0]:,}  test={te_x.shape[0]:,}  "
          f"loss_window=[{token_start}:{token_end}]")
    print(f"{'='*72}")

    best_train = float('inf'); test_at_best = None; best_state = None; stale = 0
    history = {'train_loss': [], 'test_loss': []}
    t0 = time.time()
    ep = 0
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
                best_train = tr_loss
                test_at_best = te_loss    # test CE recorded AT the train-selected checkpoint
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                stale = 0; tag = ' *best'
            else:
                stale += 1; tag = f' [stale {stale}/{C["patience"]}]'
            if ep == 1 or ep % 10 == 0 or tag == ' *best':
                print(f'  ep {ep:4d}/{C["epochs"]}  tr={tr_loss:.4f}  te={te_loss:.4f}{tag}',
                      flush=True)
            if stale >= C['patience']:
                print(f'\n  [EARLY STOP] no train-loss improvement for {C["patience"]} epochs  '
                      f'best_train={best_train:.4f}  test_at_best={test_at_best:.4f}')
                break
    except KeyboardInterrupt:
        print(f'\n  [INTERRUPTED] Ctrl+C after {len(history["train_loss"])} epochs this run -- '
              f'saving best-so-far (best_train={best_train:.4f}) and moving to the next threshold',
              flush=True)

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    dt = time.time() - t0
    print(f'\n  [DONE] thr={thr}  time={dt/60:.1f}m  best_train={best_train:.4f}  '
          f'test_at_best={test_at_best:.4f}  epochs_run={ep}')

    os.makedirs(C['models_dir'], exist_ok=True)
    model_path   = os.path.join(C['models_dir'], f'model_{TAG}_x_{thr}.pth')
    metrics_path = os.path.join(C['models_dir'], f'metrics_{TAG}_x_{thr}.json')
    torch.save(model.state_dict(), model_path)
    with open(metrics_path, 'w') as fh:
        json.dump({'threshold': float(thr), 'history': history, 'best_train': best_train,
                    'test_at_best': test_at_best, 'epochs_run': ep, 'config': C}, fh, indent=2)
    print(f'  saved -> {model_path}')
    return history, best_train, test_at_best


# ─── Main ──────────────────────────────────────────────────────────────────────
def main():
    C = CONFIG
    device = torch.device(C['device'] if torch.cuda.is_available() else 'cpu')
    V = C['v_sig'] + C['n_noise']
    print(f"device      : {device}")
    print(f"model       : SimplifiedTransformerMLP (attn_param={C['attn_param']}, "
          f"freeze_readout={C['freeze_readout']}, freeze_hidden_k={C['freeze_hidden_k']}/state)")
    print(f"data        : {os.path.abspath(C['data_dir'])}")
    print(f"vocab       : v_sig={C['v_sig']}  n_noise={C['n_noise']}  V={V}")
    print(f"seq_len     : {C['seq_len']}   loss_window=[{C['token_start']}:{C['seq_len']}]")
    print(f"optimizer   : {C['optimizer']}  lr={C['lr']}  train_params={C['train_params']}")
    print(f"thresholds  : {C['thresholds']}\n")

    os.makedirs(C['models_dir'], exist_ok=True)
    t_start = time.time()
    all_history = {}
    for i, thr in enumerate(C['thresholds']):
        history, best_train, test_at_best = train_one(thr, i, len(C['thresholds']), device)
        all_history[thr] = {'history': history, 'best_train': best_train, 'test_at_best': test_at_best}
        elapsed = time.time() - t_start
        print(f"  [{i+1}/{len(C['thresholds'])} done]  elapsed={elapsed/60:.1f}m")

    # ── rebuild the combined summary + training-curves plot from ALL on-disk metrics ──
    # (so re-running a subset of thresholds doesn't shrink the summary -- CLAUDE.md §6)
    disk = {}
    import glob
    for mp in glob.glob(os.path.join(C['models_dir'], f'metrics_{TAG}_x_*.json')):
        try:
            thr = float(os.path.basename(mp)[len(f'metrics_{TAG}_x_'):-len('.json')])
            disk[thr] = json.load(open(mp))
        except (ValueError, KeyError):
            pass

    summary_path = os.path.join(C['models_dir'], f'training_summary_{TAG}.json')
    with open(summary_path, 'w') as f:
        json.dump({'thresholds': sorted(disk), 'config': C,
                    'best_train':   {t: disk[t]['best_train']   for t in disk},
                    'test_at_best': {t: disk[t]['test_at_best'] for t in disk}}, f, indent=2)
    print(f"\nSummary -> {summary_path}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for thr in sorted(disk):
        h = disk[thr]['history']
        axes[0].plot(h['test_loss'], label=f'a={thr}')
        axes[1].plot(h['train_loss'], label=f'a={thr}')
    axes[0].set_title(f'Test CE  [{TAG}]'); axes[0].set_xlabel('epoch')
    axes[1].set_title(f'Train CE  [{TAG}]'); axes[1].set_xlabel('epoch')
    for ax in axes:
        ax.legend(fontsize=7); ax.grid(alpha=0.3)
    plt.tight_layout()
    fig_path = os.path.join(C['models_dir'], f'training_curves_{TAG}.png')
    plt.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"Curves  -> {fig_path}")


if __name__ == '__main__':
    main()
