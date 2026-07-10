"""
Train the in-arch-log-readout architecture with GD on the ALTERED distribution.

Architecture (IDENTICAL to the in-arch log readout's home model = folder 6/8's
DisentangledTransformerMLP): two DISENTANGLED ConcatAttention layers followed by a
single residual position-wise FFN head with GELU, then a linear unembedding:

    s -> layer1(attn) -> layer2(attn) -> s + FFN(s)  (GELU, hidden=mlp_hidden) -> fc_out

This is the model whose hand-built readout we call "in-arch log readout". Training it
with GD on the unequal-successor data tests whether GD discovers a count-ratio reader
(it CAN'T cheat with the old 0.5/0.5 "equalise" trick here, since p1≠p2).

Model selection: 10% of the TRAIN split is held out as VALIDATION; the checkpoint
saved is the one with the lowest VALIDATION cross-entropy. Each epoch reports
train / val / test loss and train / val / test accuracy (on the last suffix_len tokens).

Data source (CONFIG['use_fresh']):
    use_fresh=False : load this folder's saved dataset/  ('train' split for training,
                      'test' split for the test metric)  ← from data_generation.py
    use_fresh=True  : generate fresh altered-distribution sequences on the fly
                      (same p1~U(p1_low,p1_high) scheme), fresh_train / fresh_test counts.

Artifacts (into this folder's models/):
    models/model_idarch_x_{thr}.pth, models/metrics_idarch_x_{thr}.json,
    models/training_summary_idarch.json, models/training_curves_idarch.png

Edit CONFIG below, then run:
    python -u train_identical_arch.py
"""

import json
import os
import tempfile
import time

# This box's system temp dirs are unusable → route temp files under $HOME.
_TMP = os.path.join(os.path.expanduser('~'), '.cache', 'ashu_tmp')
os.makedirs(_TMP, exist_ok=True)
for _v in ('TMPDIR', 'TMP', 'TEMP'):
    os.environ[_v] = _TMP
tempfile.tempdir = _TMP
os.environ.setdefault('MPLCONFIGDIR', _TMP)

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data

# ─── Config ──────────────────────────────────────────────────────────────────
# α=0.7 warm-start from model_idarch_x_0.6.pth.
# History:
#   - Fresh retrain (263 ep, lr=1e-3): stuck at val≈2.26 — wrong basin.
#   - Warm-start from 0.8 ws09 (wd=0):   best_val=1.7212 at ep5, immediate overfit.
#   - Warm-start from 0.9 base (wd=1e-4): best_val=1.7244 at ep12, immediate overfit.
# Both above-threshold warm-starts converge to ~1.72 and stall — possibly the 0.8/0.9
# attention is too specialized for high-α. Trying from 0.6 (below phase transition,
# converged cleanly to 1.9766 from scratch) to approach from the other direction.
# Saves → models/model_idarch_ws06_x_0.7.pth
# main.ipynb loads via TRAINED_OVERRIDE={0.7:'idarch_ws06'}
# ─────────────────────────────────────────────────────────────────────────────
CONFIG = {
    'data_dir':   'dataset',
    'models_dir': 'models',
    'thresholds':  [0.7],
    'device':      'cuda:1',
    'resume':      False,
    'init_from':   None,

    # ── data source ───────────────────────────────────────────────────────────
    'use_fresh':   False,
    'fresh_train': 20_000,
    'fresh_test':  5_000,
    'fresh_seed':  123,
    'p1_low':      0.00,
    'p1_high':     1.00,

    # ── validation split ──────────────────────────────────────────────────────
    'val_frac':    0.15,

    # ── model ─────────────────────────────────────────────────────────────────
    'attn_dim':    220,
    'bias':        True,
    'mlp_hidden':  128,
    'mlp_act':     'gelu',

    # ── training defaults (overridden per-threshold by PER_THR below) ─────────
    'suffix_len':  100,
    'epochs':      200,
    'batch_size':  256,
    'lr':          1e-3,
    'weight_decay': 1e-4,
    'grad_clip':   0.5,
    'patience':    80,
    'eval_subset': 1000,
    'num_workers': 0,
    'seed':        7,

    # ── per-threshold overrides ───────────────────────────────────────────────
    # Each entry may override: tag, init_from, resume, epochs, weight_decay.
    # Anything not listed falls back to the defaults above.
    'PER_THR': {
        # α=0.7: warm-start from 0.6 (below phase transition, approach from other direction).
        0.7: {
            'tag':          'idarch_ws06',
            'init_from':    'models/model_idarch_x_0.6.pth',
            'resume':       False,
            'epochs':       50,
            'weight_decay': 1e-2,
        },
    },
}
TAG = 'idarch'   # used only for the combined summary/plot; per-threshold saves use PER_THR tags
# ─────────────────────────────────────────────────────────────────────────────


# ─── Model (IDENTICAL architecture to the in-arch log readout) ─────────────────
class ConcatAttentionBlock(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.output_dim = output_dim
        self.W_q = nn.Linear(input_dim, output_dim, bias=False)
        self.W_k = nn.Linear(input_dim, output_dim, bias=False)
        self.W_v = nn.Linear(input_dim, output_dim, bias=False)
        self.W_o = nn.Linear(output_dim, output_dim, bias=False)

    def forward(self, x, mask=None):
        Q, K, V = self.W_q(x), self.W_k(x), self.W_v(x)
        scores = torch.bmm(Q, K.transpose(1, 2)) / (self.output_dim ** 0.5)
        if mask is not None:
            scores = scores + mask
        attn_out = self.W_o(torch.bmm(F.softmax(scores, dim=-1), V))
        return torch.cat([x, attn_out], dim=-1), scores


class DisentangledTransformerMLP(nn.Module):
    """Two disentangled ConcatAttention layers + one residual GELU FFN head + linear readout."""
    def __init__(self, total_vocab, seq_len, attn_dim, mlp_hidden, bias=False, act='gelu'):
        super().__init__()
        self.total_vocab = total_vocab
        self.seq_len = seq_len
        self.initial_dim = total_vocab + seq_len
        self.layer1 = ConcatAttentionBlock(self.initial_dim, attn_dim)
        self.layer2 = ConcatAttentionBlock(self.initial_dim + attn_dim, attn_dim)
        D = self.initial_dim + 2 * attn_dim
        self.ff1 = nn.Linear(D, mlp_hidden)
        self.ff2 = nn.Linear(mlp_hidden, D)
        self.act = nn.GELU() if act == 'gelu' else nn.ReLU()
        self.fc_out = nn.Linear(D, total_vocab, bias=bias)

    def forward(self, x, return_attn=False):
        B, T = x.shape
        tok = F.one_hot(x, num_classes=self.total_vocab).float()
        pos = F.one_hot(torch.arange(T, device=x.device),
                        num_classes=self.seq_len).float().unsqueeze(0).expand(B, T, -1)
        s = torch.cat([tok, pos], dim=-1)
        mask = torch.triu(torch.ones(T, T, device=x.device) * float('-inf'), diagonal=1)
        s, a1 = self.layer1(s, mask)
        s, a2 = self.layer2(s, mask)
        s = s + self.ff2(self.act(self.ff1(s)))
        logits = self.fc_out(s)
        return (logits, [a1, a2]) if return_attn else logits


# ─── Data (altered distribution: unequal per-state p1, p2) ─────────────────────
def _generate_blob(split, thr, n, seq_len, vocab_set1, total_vocab, p1_low, p1_high, base_seed):
    """Fresh altered-distribution sequences (same scheme as data_generation.py). Returns blob."""
    rng = np.random.default_rng(base_seed + (1 if split == 'train' else 2) + int(round(thr * 1000)))
    B, V_sig, V = int(n), vocab_set1, total_vocab
    order     = rng.random((B, V_sig, V_sig)).argsort(axis=2)
    trans_arr = order[:, :, :2].astype(np.int64)
    p1        = rng.uniform(p1_low, p1_high, size=(B, V_sig))
    prob_arr  = np.stack([p1, 1.0 - p1], axis=2)
    seq  = np.empty((B, seq_len + 1), dtype=np.int64)
    last = np.full(B, -1, dtype=np.int64); bidx = np.arange(B)
    for t in range(seq_len + 1):
        is_sig = rng.random(B) < thr
        has    = last >= 0
        state  = np.where(has, last, 0)
        u      = rng.random(B)
        coin   = (u >= prob_arr[bidx, state, 0]).astype(np.int64)
        succ   = trans_arr[bidx, state, coin]
        first  = rng.integers(0, V_sig, B)
        noise  = rng.integers(V_sig, V, B)
        tok    = np.where(is_sig, np.where(has, succ, first), noise)
        last   = np.where(is_sig, tok, last)
        seq[:, t] = tok
    x = torch.from_numpy(seq[:, :-1]).contiguous().long()
    y = torch.from_numpy(seq[:, 1:]).contiguous().long()
    return {'x': x, 'y': y, 'seq_len': seq_len, 'vocab_set1': vocab_set1, 'total_vocab': total_vocab,
            'threshold': float(thr), 'split': split}


def get_data(split, thr):
    """split='train' → generated/saved train data;  split='test' → saved test data;
       use_fresh=True → fresh generated points instead.  (Mirrors the notebook's get_data.)"""
    C = CONFIG
    if C['use_fresh']:
        n = C['fresh_train'] if split == 'train' else C['fresh_test']
        # vocab/seq_len from any saved file if present, else defaults (20/200)
        return _generate_blob(split, thr, n, seq_len=200, vocab_set1=10, total_vocab=20,
                              p1_low=C['p1_low'], p1_high=C['p1_high'], base_seed=C['fresh_seed'])
    return torch.load(os.path.join(C['data_dir'], split, f'data_x_{thr}.pt'), weights_only=False)


# ─── Helpers ─────────────────────────────────────────────────────────────────
def _discover_thresholds(data_dir, use_fresh):
    if use_fresh:
        return [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    train_dir = os.path.join(data_dir, 'train')
    thrs = []
    for name in sorted(os.listdir(train_dir)):
        if name.startswith('data_x_') and name.endswith('.pt'):
            try:
                thrs.append(float(name[len('data_x_'):-len('.pt')]))
            except ValueError:
                pass
    return sorted(thrs)


@torch.no_grad()
def _evaluate(model, x, y, batch_size, device, suffix_len, total_vocab):
    """Mean per-token CE loss and accuracy on the last suffix_len tokens."""
    model.eval()
    correct, total, loss_sum = 0, 0, 0.0
    for i in range(0, x.shape[0], batch_size):
        xb = x[i:i + batch_size].to(device, non_blocking=True)
        yb = y[i:i + batch_size].to(device, non_blocking=True)
        logits = model(xb)[:, -suffix_len:, :]
        yb_s   = yb[:, -suffix_len:]
        loss_sum += F.cross_entropy(logits.reshape(-1, total_vocab), yb_s.reshape(-1),
                                    reduction='sum').item()
        correct  += (logits.argmax(-1) == yb_s).sum().item()
        total    += yb_s.numel()
    return loss_sum / max(total, 1), correct / max(total, 1)


def _fmt(seconds):
    if seconds >= 3600: return f"{seconds / 3600:.1f}h"
    if seconds >= 60:   return f"{seconds / 60:.0f}m"
    return f"{seconds:.0f}s"


def _set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ─── Training ────────────────────────────────────────────────────────────────
def train_one(thr, thr_idx, n_thrs, device):
    C = CONFIG
    suffix_len = C['suffix_len']

    # ── resolve per-threshold overrides ──────────────────────────────────────
    _per        = C.get('PER_THR', {}).get(round(thr, 3), {})
    local_tag   = _per.get('tag',          TAG)
    local_init  = _per.get('init_from',    C.get('init_from'))
    local_resume= _per.get('resume',       C.get('resume', False))
    local_epochs= _per.get('epochs',       C['epochs'])
    local_wd    = _per.get('weight_decay', C.get('weight_decay', 0.0))

    train_blob = get_data('train', thr)
    test_blob  = get_data('test',  thr)

    seq_len     = int(train_blob['seq_len'])
    total_vocab = int(train_blob['total_vocab'])
    attn_dim    = C['attn_dim'] if C['attn_dim'] else (total_vocab + seq_len)

    # ── 90/10 train/val split (validation used for model selection) ──
    full_x, full_y = train_blob['x'], train_blob['y']
    n_full = full_x.shape[0]
    g = torch.Generator().manual_seed(C['seed'] + 99)
    perm = torch.randperm(n_full, generator=g)
    n_val = max(1, int(round(C['val_frac'] * n_full)))
    val_idx, tr_idx = perm[:n_val], perm[n_val:]
    train_x, train_y = full_x[tr_idx], full_y[tr_idx]
    val_x,   val_y   = full_x[val_idx], full_y[val_idx]

    test_x,  test_y  = test_blob['x'], test_blob['y']
    es = min(C['eval_subset'], test_x.shape[0])
    eval_x, eval_y = test_x[:es], test_y[:es]

    loader = data.DataLoader(
        data.TensorDataset(train_x, train_y),
        batch_size=C['batch_size'], shuffle=True,
        num_workers=C['num_workers'], pin_memory=(device.type == 'cuda'),
    )

    model = DisentangledTransformerMLP(
        total_vocab, seq_len, attn_dim, C['mlp_hidden'], bias=C['bias'], act=C['mlp_act'],
    ).to(device)
    opt = optim.AdamW(model.parameters(), lr=C['lr'], weight_decay=local_wd)

    # warm-start init from another checkpoint (only when NOT resuming this thr's own ckpt)
    if local_init and not local_resume:
        sd = torch.load(local_init, map_location=device, weights_only=True)
        model.load_state_dict(sd)
        print(f"  [INIT] warm-started from {local_init}", flush=True)

    model_path   = os.path.join(C['models_dir'], f'model_{local_tag}_x_{thr}.pth')
    metrics_path = os.path.join(C['models_dir'], f'metrics_{local_tag}_x_{thr}.json')

    # checkpoint selection is by VALIDATION loss
    prev_history = {'train_loss': [], 'val_loss': [], 'test_loss': [],
                    'train_acc': [], 'val_acc': [], 'test_acc': []}
    best_loss  = float('inf')
    best_state = None
    resumed    = False
    if local_resume:
        if not (os.path.exists(model_path) and os.path.exists(metrics_path)):
            raise FileNotFoundError(f"resume=True but checkpoint/metrics missing for thr={thr}: "
                                    f"{model_path} / {metrics_path}")
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        prev = json.load(open(metrics_path))
        prev_history = prev['history']
        best_loss  = min(prev_history.get('val_loss') or [float('inf')])
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        resumed    = True

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n{'='*72}", flush=True)
    mode = (f"RESUME (+{local_epochs} on top of {len(prev_history['train_loss'])}, "
            f"best_val={best_loss:.4f})") if resumed else "FRESH"
    print(f"  [{thr_idx+1}/{n_thrs}] thr={thr}  tag={local_tag}  {mode}  "
          f"attn_dim={attn_dim}  mlp_hidden={C['mlp_hidden']}"
          f"  act={C['mlp_act']}  params={n_params:,}", flush=True)
    print(f"  data: train={train_x.shape[0]:,}  val={val_x.shape[0]:,}  test={test_x.shape[0]:,}"
          f"  (eval test subset={es})  loss_suffix={suffix_len}/{seq_len}", flush=True)
    print(f"{'='*72}", flush=True)

    history = {'train_loss': [], 'val_loss': [], 'test_loss': [],
               'train_acc': [], 'val_acc': [], 'test_acc': []}
    stale_epochs = 0
    t0 = time.time()

    try:
        for epoch in range(local_epochs):
            model.train()
            ep_t0 = time.time()
            l_sum, n_steps, ep_correct, ep_total = 0.0, 0, 0, 0

            for xb, yb in loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                logits = model(xb)
                logits_s = logits[:, -suffix_len:, :]
                yb_s     = yb[:, -suffix_len:]
                loss = F.cross_entropy(logits_s.reshape(-1, total_vocab), yb_s.reshape(-1))
                loss.backward()
                if C['grad_clip'] is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), C['grad_clip'])
                opt.step()
                l_sum      += loss.item(); n_steps += 1
                ep_correct += (logits_s.detach().argmax(-1) == yb_s).sum().item()
                ep_total   += yb_s.numel()

            train_loss = l_sum / n_steps
            train_acc  = ep_correct / ep_total
            val_loss,  val_acc  = _evaluate(model, val_x,  val_y,  C['batch_size'], device, suffix_len, total_vocab)
            test_loss, test_acc = _evaluate(model, eval_x, eval_y, C['batch_size'], device, suffix_len, total_vocab)
            ep_time = time.time() - ep_t0

            history['train_loss'].append(train_loss); history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss);     history['val_acc'].append(val_acc)
            history['test_loss'].append(test_loss);   history['test_acc'].append(test_acc)

            # checkpoint on best VALIDATION loss
            if val_loss < best_loss:
                best_loss    = val_loss
                stale_epochs = 0
                best_state   = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                stale_epochs += 1

            elapsed = time.time() - t0
            eta = (elapsed / (epoch + 1)) * (local_epochs - epoch - 1)
            stale_tag = f"  [stale {stale_epochs}/{C['patience']}]" if stale_epochs > 0 else ""
            best_tag  = "  *best" if stale_epochs == 0 else ""
            print(f"  ep {epoch+1:4d}/{local_epochs}  tr_loss={train_loss:.4f}  va_loss={val_loss:.4f}  "
                  f"te_loss={test_loss:.4f}  tr={train_acc:.3f}  va={val_acc:.3f}  te={test_acc:.3f}  "
                  f"{ep_time:.1f}s/ep  ETA={_fmt(eta)}{stale_tag}{best_tag}", flush=True)

            if stale_epochs >= C['patience']:
                print(f"\n  [EARLY STOP] no val-loss improvement for {C['patience']} epochs  "
                      f"best_val_loss={best_loss:.4f}", flush=True)
                break
    except KeyboardInterrupt:
        print(f"\n  [INTERRUPTED] Ctrl+C after {len(history['train_loss'])} epochs this run — "
              f"saving best-so-far (best_val_loss={best_loss:.4f}) and exiting cleanly", flush=True)

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    full_history = {k: list(prev_history.get(k, [])) + list(history[k]) for k in history}

    os.makedirs(C['models_dir'], exist_ok=True)
    torch.save(model.state_dict(), model_path)
    with open(metrics_path, 'w') as f:
        json.dump({
            'threshold': float(thr),
            'history':   full_history,
            'config': {
                'attn_dim':      attn_dim,
                'mlp_hidden':    C['mlp_hidden'],
                'mlp_act':       C['mlp_act'],
                'bias':          C['bias'],
                'head':          'mlp',          # single residual FFN head (== in-arch readout arch)
                'arch':          'DisentangledTransformerMLP',
                'seq_len':       seq_len,
                'suffix_len':    suffix_len,
                'total_vocab':   total_vocab,
                'val_frac':      C['val_frac'],
                'use_fresh':     C['use_fresh'],
                'train_size':    int(train_x.shape[0]),
                'val_size':      int(val_x.shape[0]),
                'test_size':     int(test_x.shape[0]),
                'epochs_run':    len(full_history['train_loss']),
                'epochs_added':  len(history['train_loss']) if resumed else 0,
                'resumed':       resumed,
                'lr':            C['lr'],
                'weight_decay':  local_wd,
                'schedule':      'constant',
                'select_by':     'val_loss',
                'best_val_loss': best_loss,
            },
        }, f, indent=2)

    total = time.time() - t0
    final_te = full_history['test_acc'][-1] if full_history['test_acc'] else float('nan')
    print(f"\n  [DONE] thr={thr}  time={_fmt(total)}  best_val_loss={best_loss:.4f}  "
          f"final_te_acc={final_te:.3f}  total_epochs={len(full_history['train_loss'])}", flush=True)
    print(f"  saved → {model_path}", flush=True)
    return full_history


# ─── Main ────────────────────────────────────────────────────────────────────
def main():
    C = CONFIG
    _set_seed(C['seed'])
    device     = torch.device(C['device'])
    thresholds = C['thresholds'] or _discover_thresholds(C['data_dir'], C['use_fresh'])

    print(f"device     : {device}")
    print(f"model      : DisentangledTransformerMLP (2 disentangled attn + residual {C['mlp_act']} FFN head, "
          f"mlp_hidden={C['mlp_hidden']})   [identical to in-arch log readout arch]")
    print(f"data       : {'FRESH altered points' if C['use_fresh'] else os.path.abspath(C['data_dir'])}")
    print(f"val split  : {C['val_frac']:.0%} of train  (model selected by VALIDATION loss)")
    print(f"attn_dim   : {C['attn_dim'] or 'auto (= total_vocab + seq_len = 220)'}")
    print(f"suffix_len : {C['suffix_len']}  (loss & eval on last {C['suffix_len']} tokens only)")
    print(f"lr         : {C['lr']} (constant)  weight_decay={C.get('weight_decay', 0.0)}  "
          f"grad_clip={C['grad_clip']}  patience={C['patience']}")
    print(f"epochs     : {C['epochs']}  batch={C['batch_size']}")
    print(f"thresholds : {thresholds}\n")

    t_all = time.time()
    all_history = {}
    for i, thr in enumerate(thresholds):
        all_history[thr] = train_one(thr, i, len(thresholds), device)
        done = i + 1
        elapsed = time.time() - t_all
        eta_all = (elapsed / done) * (len(thresholds) - done)
        print(f"\n  [{done}/{len(thresholds)} done]  elapsed={_fmt(elapsed)}  "
              f"est remaining={_fmt(eta_all)}", flush=True)

    summary_path = os.path.join(C['models_dir'], f'training_summary_{TAG}.json')
    os.makedirs(C['models_dir'], exist_ok=True)
    with open(summary_path, 'w') as f:
        json.dump({'thresholds': thresholds, 'history': all_history, 'config': C}, f, indent=2)
    print(f"\nSummary → {summary_path}")

    # plot every threshold that has metrics on disk (resume-safe: don't drop other curves)
    # For thresholds with a PER_THR tag override, prefer their warm-start metrics file.
    import glob
    disk = {}
    for mp in glob.glob(os.path.join(C['models_dir'], f'metrics_{TAG}_x_*.json')):
        try:
            thr = float(os.path.basename(mp)[len(f'metrics_{TAG}_x_'):-len('.json')])
            disk[thr] = json.load(open(mp))['history']
        except (ValueError, KeyError):
            pass
    # Override with PER_THR warm-start metrics where they exist (e.g. ws09 for 0.8)
    for thr, per in C.get('PER_THR', {}).items():
        ws_tag = per.get('tag')
        if ws_tag and ws_tag != TAG:
            ws_mp = os.path.join(C['models_dir'], f'metrics_{ws_tag}_x_{thr}.json')
            if os.path.exists(ws_mp):
                disk[thr] = json.load(open(ws_mp))['history']
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for thr in sorted(disk):
        h = disk[thr]
        axes[0].plot(h['val_loss'], label=f'x={thr}')
        axes[1].plot(h['test_acc'], label=f'x={thr}')
    axes[0].set_title(f'Validation loss (last {C["suffix_len"]} tokens)  [identical arch]')
    axes[0].set_xlabel('epoch')
    axes[1].set_title(f'Test accuracy (last {C["suffix_len"]} tokens)  [identical arch]')
    axes[1].set_xlabel('epoch')
    for ax in axes:
        ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig_path = os.path.join(C['models_dir'], f'training_curves_{TAG}.png')
    plt.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"Curves  → {fig_path}")


if __name__ == '__main__':
    main()





