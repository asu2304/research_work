"""
Train a TWO-CELL DisentangledTransformer on the Markov-noise ICL task.

Same data + same base attention as train_with_mlp.py, but now BOTH attention
layers are followed by their own residual position-wise FFN — i.e. each layer is
a complete transformer cell (attention sub-block + FFN sub-block):

    s -> [cell 1: attn1 -> s + FFN1(s)] -> [cell 2: attn2 -> s + FFN2(s)] -> fc_out

The only change vs train_with_mlp.py is the NEW FFN1 inserted after layer 1.
After layer 1 the state is cat[s, attn_out] = 220 + 220 = 440 dim; FFN1 projects
it down to mlp_hidden and back to 440 with a residual connection (exactly the same
shape of block as the existing FFN2, which acts on the 660-dim post-layer-2 state).

Optional LayerNorm is available per sub-block via two top-level toggles
(use_norm_attn, use_norm_ffn). Both default OFF, so the default run is identical
to train_with_mlp.py + the extra FFN1; flip either on to get a normed transformer.

lr, weight_decay, suffix_len, batch, epochs, etc. are all kept the same as
train_with_mlp.py. The dataset is the SAME one (folder 6's dataset/). Artifacts are
saved with the same names as before, but into this folder's own models/ dir:
    models/model_mlp_x_{thr}.pth, models/metrics_mlp_x_{thr}.json,
    models/training_summary_mlp.json, models/training_curves_mlp.png

Edit CONFIG below, then run:
    python -u train_two_cell.py
"""

import json
import os
import tempfile
import time

# This box's /tmp, /var/tmp, /usr/tmp are unusable → route all temp files
# (DataLoader workers, matplotlib, torch) to a writable dir under $HOME.
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
CONFIG = {
    # same dataset as train_with_mlp.py (folder 6) — absolute path so this script
    # can live in folder 8 and still read the identical data.
    'data_dir':   '/home/hariguru/ashutosh/ICL/6. understanding what gradient descent is firuring out/dataset',
    'models_dir': 'models',          # new folder INSIDE "8. train with mlp"
    'thresholds':  None,            # None → auto-discover from dataset/train/data_x_*.pt
    'device':      'cuda:1',
    'resume':      False,            # True → load models/model_mlp_x_{thr}.pth and train `epochs` MORE,
                                     #        appending to the existing curve (continue, don't restart)

    # model
    'attn_dim':    220,        # None → total_vocab + seq_len = 220
    'bias':        True,       # bias in the unembedding head
    'mlp_hidden':  128,        # hidden width of BOTH residual FFNs (the new capacity)
    'mlp_act':     'gelu',     # 'gelu' or 'relu'

    # normalization — independent LayerNorm toggles per sub-block (pre-LN style).
    # Both False → identical to train_with_mlp.py + the extra FFN1 (no norm).
    'use_norm_attn': False,    # LayerNorm on each attention sub-block's input
    'use_norm_ffn':  False,    # LayerNorm on each FFN sub-block's input

    # training (loss / accuracy on the last suffix_len tokens) — same as train_with_mlp.py
    'suffix_len':  100,
    'epochs':      100,
    'batch_size':  256,
    'lr':          5e-3,       # CONSTANT — same as train_with_mlp.py
    'weight_decay':1e-3,
    'grad_clip':   1.0,
    'patience':    200,

    'eval_subset': 1000,
    'num_workers': 0,          # /tmp is unusable on this box → no DataLoader multiprocessing
    'seed':        7,
}
# ─────────────────────────────────────────────────────────────────────────────


# ─── Model ───────────────────────────────────────────────────────────────────
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


class DisentangledTransformerTwoCell(nn.Module):
    """Two complete transformer cells: each is a ConcatAttentionBlock followed by
    a residual position-wise FFN (project down to mlp_hidden, nonlinearity, project
    back, add). Optional pre-LayerNorm on each sub-block via use_norm_attn /
    use_norm_ffn. The FFN after cell 1 is the new piece vs train_with_mlp.py."""
    def __init__(self, total_vocab, seq_len, attn_dim, mlp_hidden, bias=False,
                 act='gelu', use_norm_attn=False, use_norm_ffn=False):
        super().__init__()
        self.total_vocab   = total_vocab
        self.seq_len       = seq_len
        self.attn_dim      = attn_dim
        self.mlp_hidden    = mlp_hidden
        self.use_norm_attn = use_norm_attn
        self.use_norm_ffn  = use_norm_ffn
        self.initial_dim   = total_vocab + seq_len            # 220

        act_layer = nn.GELU() if act == 'gelu' else nn.ReLU()

        # ── Cell 1 ──  attn over initial_dim(220) → cat → D1 = 220 + attn_dim(220) = 440
        D1 = self.initial_dim + attn_dim
        self.layer1 = ConcatAttentionBlock(self.initial_dim, attn_dim)
        self.ln_attn1 = nn.LayerNorm(self.initial_dim) if use_norm_attn else nn.Identity()
        self.ln_ffn1  = nn.LayerNorm(D1)               if use_norm_ffn  else nn.Identity()
        self.ff1_1 = nn.Linear(D1, mlp_hidden)
        self.ff1_2 = nn.Linear(mlp_hidden, D1)
        self.act1  = act_layer

        # ── Cell 2 ──  attn over D1(440) → cat → D2 = 440 + attn_dim(220) = 660
        D2 = D1 + attn_dim
        self.layer2 = ConcatAttentionBlock(D1, attn_dim)
        self.ln_attn2 = nn.LayerNorm(D1) if use_norm_attn else nn.Identity()
        self.ln_ffn2  = nn.LayerNorm(D2) if use_norm_ffn  else nn.Identity()
        self.ff2_1 = nn.Linear(D2, mlp_hidden)
        self.ff2_2 = nn.Linear(mlp_hidden, D2)
        self.act2  = act_layer

        self.fc_out = nn.Linear(D2, total_vocab, bias=bias)

    def forward(self, x, return_attn=False):
        B, T = x.shape
        tok = F.one_hot(x, num_classes=self.total_vocab).float()
        pos = F.one_hot(torch.arange(T, device=x.device),
                        num_classes=self.seq_len).float().unsqueeze(0).expand(B, T, -1)
        s = torch.cat([tok, pos], dim=-1)
        mask = torch.triu(torch.ones(T, T, device=x.device) * float('-inf'), diagonal=1)

        # ── Cell 1: attention sub-block + residual FFN sub-block ──
        s, a1 = self.layer1(self.ln_attn1(s), mask)
        s = s + self.ff1_2(self.act1(self.ff1_1(self.ln_ffn1(s))))

        # ── Cell 2: attention sub-block + residual FFN sub-block ──
        s, a2 = self.layer2(self.ln_attn2(s), mask)
        s = s + self.ff2_2(self.act2(self.ff2_1(self.ln_ffn2(s))))

        logits = self.fc_out(s)
        return (logits, [a1, a2]) if return_attn else logits


# ─── Helpers ─────────────────────────────────────────────────────────────────
def _discover_thresholds(data_dir):
    train_dir = os.path.join(data_dir, 'train')
    thrs = []
    for name in sorted(os.listdir(train_dir)):
        if name.startswith('data_x_') and name.endswith('.pt'):
            try:
                thrs.append(float(name[len('data_x_'):-len('.pt')]))
            except ValueError:
                pass
    return sorted(thrs)


def _load(data_dir, split, thr):
    return torch.load(os.path.join(data_dir, split, f'data_x_{thr}.pt'),
                      weights_only=False)


@torch.no_grad()
def _evaluate(model, x, y, batch_size, device, suffix_len, total_vocab):
    """Return (mean per-token test CE loss, test accuracy) on the last suffix_len tokens."""
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

    train_blob = _load(C['data_dir'], 'train', thr)
    test_blob  = _load(C['data_dir'], 'test',  thr)

    seq_len     = int(train_blob['seq_len'])
    total_vocab = int(train_blob['total_vocab'])
    attn_dim    = C['attn_dim'] if C['attn_dim'] else (total_vocab + seq_len)

    train_x, train_y = train_blob['x'], train_blob['y']
    test_x,  test_y  = test_blob['x'],  test_blob['y']
    es = min(C['eval_subset'], test_x.shape[0])
    eval_x, eval_y = test_x[:es], test_y[:es]

    loader = data.DataLoader(
        data.TensorDataset(train_x, train_y),
        batch_size=C['batch_size'], shuffle=True,
        num_workers=C['num_workers'], pin_memory=(device.type == 'cuda'),
    )

    model = DisentangledTransformerTwoCell(
        total_vocab, seq_len, attn_dim, C['mlp_hidden'],
        bias=C['bias'], act=C['mlp_act'],
        use_norm_attn=C['use_norm_attn'], use_norm_ffn=C['use_norm_ffn'],
    ).to(device)
    opt   = optim.AdamW(model.parameters(), lr=C['lr'],
                        weight_decay=C.get('weight_decay', 0.0))

    model_path   = os.path.join(C['models_dir'], f'model_mlp_x_{thr}.pth')
    metrics_path = os.path.join(C['models_dir'], f'metrics_mlp_x_{thr}.json')

    # ── checkpoint selection is by TEST loss (best_loss tracks test CE, not train) ──
    prev_history = {'loss': [], 'train_acc': [], 'test_acc': [], 'test_loss': []}
    best_loss    = float('inf')
    best_state   = None
    resumed      = False
    if C.get('resume'):
        if not (os.path.exists(model_path) and os.path.exists(metrics_path)):
            raise FileNotFoundError(f"resume=True but checkpoint/metrics missing for thr={thr}: "
                                    f"{model_path} / {metrics_path}")
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        prev = json.load(open(metrics_path))
        prev_history = prev['history']
        # never save worse than the loaded checkpoint (by test loss if available)
        best_loss  = min(prev_history.get('test_loss') or [float('inf')])
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        resumed    = True

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n{'='*68}", flush=True)
    mode = (f"RESUME  (+{C['epochs']} epochs on top of {len(prev_history['loss'])}, "
            f"loaded best_loss={best_loss:.4f})") if resumed else "FRESH"
    print(f"  [{thr_idx+1}/{n_thrs}] thr={thr}  {mode}  attn_dim={attn_dim}  mlp_hidden={C['mlp_hidden']}  "
          f"norm(attn={C['use_norm_attn']},ffn={C['use_norm_ffn']})  "
          f"params={n_params:,}  train={train_x.shape[0]:,}  "
          f"loss_suffix={suffix_len}/{seq_len}", flush=True)
    print(f"{'='*68}", flush=True)

    history = {'loss': [], 'train_acc': [], 'test_acc': [], 'test_loss': []}
    stale_epochs = 0
    t0 = time.time()

    try:
        for epoch in range(C['epochs']):
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
                torch.nn.utils.clip_grad_norm_(model.parameters(), C['grad_clip'])
                opt.step()
                l_sum      += loss.item(); n_steps += 1
                ep_correct += (logits_s.detach().argmax(-1) == yb_s).sum().item()
                ep_total   += yb_s.numel()

            train_loss = l_sum / n_steps
            train_acc  = ep_correct / ep_total
            test_loss, test_acc = _evaluate(model, eval_x, eval_y, C['batch_size'],
                                            device, suffix_len, total_vocab)
            ep_time    = time.time() - ep_t0

            history['loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['test_acc'].append(test_acc)
            history['test_loss'].append(test_loss)

            # checkpoint on best TEST loss (generalization), not train loss
            if test_loss < best_loss:
                best_loss    = test_loss
                stale_epochs = 0
                best_state   = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                stale_epochs += 1

            elapsed = time.time() - t0
            avg_ep  = elapsed / (epoch + 1)
            eta     = avg_ep * (C['epochs'] - epoch - 1)
            stale_tag = f"  [stale {stale_epochs}/{C['patience']}]" if stale_epochs > 0 else ""
            best_tag  = "  *best" if stale_epochs == 0 else ""
            print(f"  ep {epoch+1:4d}/{C['epochs']}  tr_loss={train_loss:.4f}  te_loss={test_loss:.4f}  "
                  f"tr={train_acc:.3f}  te={test_acc:.3f}  "
                  f"{ep_time:.1f}s/ep  ETA={_fmt(eta)}{stale_tag}{best_tag}", flush=True)

            if stale_epochs >= C['patience']:
                print(f"\n  [EARLY STOP] no test-loss improvement for {C['patience']} epochs  "
                      f"best_te_loss={best_loss:.4f}", flush=True)
                break
    except KeyboardInterrupt:
        # Ctrl+C: stop cleanly and fall through to saving the best checkpoint so far,
        # instead of aborting and losing all progress since the last full run.
        print(f"\n  [INTERRUPTED] Ctrl+C after {len(history['loss'])} epochs this run — "
              f"saving best-so-far (best_te_loss={best_loss:.4f}) and exiting cleanly", flush=True)

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    # append the new epochs onto the prior curve (resume), else just the new run
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
                'head':          'mlp2',          # two-cell (FFN after both layers)
                'use_norm_attn': C['use_norm_attn'],
                'use_norm_ffn':  C['use_norm_ffn'],
                'seq_len':       seq_len,
                'suffix_len':    suffix_len,
                'total_vocab':   total_vocab,
                'train_size':    int(train_x.shape[0]),
                'test_size':     int(test_x.shape[0]),
                'epochs_run':    len(full_history['loss']),
                'epochs_added':  len(history['loss']) if resumed else 0,
                'resumed':       resumed,
                'lr':            C['lr'],
                'weight_decay':  C.get('weight_decay', 0.0),
                'schedule':      'constant',
                'select_by':     'test_loss',
            },
        }, f, indent=2)

    total    = time.time() - t0
    final_te = full_history['test_acc'][-1] if full_history['test_acc'] else float('nan')
    print(f"\n  [DONE] thr={thr}  time={_fmt(total)}  "
          f"best_te_loss={best_loss:.4f}  final_te_acc={final_te:.3f}  "
          f"total_epochs={len(full_history['loss'])}", flush=True)
    print(f"  saved → {model_path}", flush=True)
    return full_history


# ─── Main ────────────────────────────────────────────────────────────────────
def main():
    C = CONFIG
    _set_seed(C['seed'])
    device     = torch.device(C['device'])
    thresholds = C['thresholds'] or _discover_thresholds(C['data_dir'])

    print(f"device     : {device}")
    print(f"model      : DisentangledTransformer TWO-CELL  (FFN after both layers, mlp_hidden={C['mlp_hidden']}, act={C['mlp_act']})")
    print(f"norm       : attn={C['use_norm_attn']}  ffn={C['use_norm_ffn']}")
    print(f"attn_dim   : {C['attn_dim'] or 'auto (= total_vocab + seq_len = 220)'}")
    print(f"data_dir   : {C['data_dir']}")
    print(f"suffix_len : {C['suffix_len']}  (loss & eval on last {C['suffix_len']} tokens only)")
    print(f"lr         : {C['lr']} (constant)  weight_decay={C.get('weight_decay', 0.0)}  grad_clip={C['grad_clip']}  patience={C['patience']}")
    print(f"epochs     : {C['epochs']}  batch={C['batch_size']}")
    print(f"thresholds : {thresholds}\n")

    t_all = time.time()
    all_history = {}
    for i, thr in enumerate(thresholds):
        all_history[thr] = train_one(thr, i, len(thresholds), device)
        done    = i + 1
        elapsed = time.time() - t_all
        eta_all = (elapsed / done) * (len(thresholds) - done)
        print(f"\n  [{done}/{len(thresholds)} done]  "
              f"elapsed={_fmt(elapsed)}  est remaining={_fmt(eta_all)}", flush=True)

    summary_path = os.path.join(C['models_dir'], 'training_summary_mlp.json')
    with open(summary_path, 'w') as f:
        json.dump({'thresholds': thresholds, 'history': all_history, 'config': C}, f, indent=2)
    print(f"\nSummary → {summary_path}")

    # plot EVERY threshold that has metrics on disk (so a single-threshold resume
    # run still shows the full picture and doesn't drop the other curves)
    import glob
    disk = {}
    for mp in glob.glob(os.path.join(C['models_dir'], 'metrics_mlp_x_*.json')):
        try:
            thr = float(os.path.basename(mp)[len('metrics_mlp_x_'):-len('.json')])
            disk[thr] = json.load(open(mp))['history']
        except (ValueError, KeyError):
            pass
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for thr in sorted(disk):
        h = disk[thr]
        axes[0].plot(h['loss'],     label=f'x={thr}')
        axes[1].plot(h['test_acc'], label=f'x={thr}')
    axes[0].set_title(f'Training loss (last {C["suffix_len"]} tokens)  [two-cell]')
    axes[0].set_xlabel('epoch')
    axes[1].set_title(f'Test accuracy (last {C["suffix_len"]} tokens)  [two-cell]')
    axes[1].set_xlabel('epoch')
    for ax in axes:
        ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig_path = os.path.join(C['models_dir'], 'training_curves_mlp.png')
    plt.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"Curves  → {fig_path}")


if __name__ == '__main__':
    main()
