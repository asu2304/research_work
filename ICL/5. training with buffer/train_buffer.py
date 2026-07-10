"""
Train a DisentangledTransformer on each threshold's dataset,
optimising cross-entropy loss on the LAST `suffix_len` tokens only
(default 250 of 500).

Edit CONFIG below, then run:
    python -u train_buffer.py
"""

import json
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data

# ─── Config ──────────────────────────────────────────────────────────────────
CONFIG = {
    'data_dir':   '../4.training_for_oracle/dataset',
    'models_dir': 'buffer models',
    'thresholds':  None,       # None → auto-discover from dataset/train/data_x_*.pt
    'device':      'cuda:3',

    # model
    'attn_dim':    None,       # None → total_vocab + seq_len  (matches W* construction)
    'bias':        True,       # bias in the unembedding head

    # training — only the final suffix_len tokens contribute to loss / accuracy
    'suffix_len':  250,        # train & eval on last 250 of 500 tokens
    'epochs':      20,         # max epochs (early stopping will usually cut this short)
    'batch_size':  256,
    'lr':          1e-3,       # higher than full-seq variant to compensate for halved gradient signal
    'grad_clip':   1.0,        # clip gradient norm (prevents explosions)
    'patience':    5,          # stop if train loss does not improve for this many epochs

    # mi
    'eval_subset': 500,        # test sequences used for per-epoch accuracy estimate
    'num_workers': 4,
    'seed':        42,
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


class DisentangledTransformer(nn.Module):
    def __init__(self, total_vocab, seq_len, attn_dim, bias=False):
        super().__init__()
        self.total_vocab = total_vocab
        self.seq_len     = seq_len
        self.attn_dim    = attn_dim
        self.initial_dim = total_vocab + seq_len
        self.layer1 = ConcatAttentionBlock(self.initial_dim, attn_dim)
        self.layer2 = ConcatAttentionBlock(self.initial_dim + attn_dim, attn_dim)
        self.fc_out = nn.Linear(self.initial_dim + 2 * attn_dim, total_vocab, bias=bias)

    def forward(self, x, return_attn=False):
        B, T = x.shape
        tok = F.one_hot(x, num_classes=self.total_vocab).float()
        pos = F.one_hot(torch.arange(T, device=x.device),
                        num_classes=self.seq_len).float().unsqueeze(0).expand(B, T, -1)
        s = torch.cat([tok, pos], dim=-1)
        mask = torch.triu(torch.ones(T, T, device=x.device) * float('-inf'), diagonal=1)
        s, a1 = self.layer1(s, mask)
        s, a2 = self.layer2(s, mask)
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
def _test_acc(model, x, y, batch_size, device, suffix_len):
    model.eval()
    correct, total = 0, 0
    for i in range(0, x.shape[0], batch_size):
        xb = x[i:i + batch_size].to(device, non_blocking=True)
        yb = y[i:i + batch_size].to(device, non_blocking=True)
        pred = model(xb)[:, -suffix_len:, :].argmax(-1)
        yb_s = yb[:, -suffix_len:]
        correct += (pred == yb_s).sum().item()
        total   += yb_s.numel()
    return correct / max(total, 1)


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

    model = DisentangledTransformer(total_vocab, seq_len, attn_dim, bias=C['bias']).to(device)
    opt   = optim.Adam(model.parameters(), lr=C['lr'])

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n{'='*68}", flush=True)
    print(f"  [{thr_idx+1}/{n_thrs}] thr={thr}  attn_dim={attn_dim}  "
          f"params={n_params:,}  train={train_x.shape[0]:,}  "
          f"loss_suffix={suffix_len}/{seq_len}", flush=True)
    print(f"{'='*68}", flush=True)

    history = {'loss': [], 'train_acc': [], 'test_acc': []}
    best_loss    = float('inf')
    stale_epochs = 0
    best_state   = None
    t0 = time.time()

    for epoch in range(C['epochs']):
        model.train()
        ep_t0 = time.time()
        l_sum, n_steps, ep_correct, ep_total = 0.0, 0, 0, 0

        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            logits = model(xb)

            # loss on last suffix_len tokens only
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
        test_acc   = _test_acc(model, eval_x, eval_y, C['batch_size'], device, suffix_len)
        ep_time    = time.time() - ep_t0

        history['loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_acc'].append(test_acc)

        if train_loss < best_loss:
            best_loss    = train_loss
            stale_epochs = 0
            best_state   = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale_epochs += 1

        elapsed = time.time() - t0
        avg_ep  = elapsed / (epoch + 1)
        eta     = avg_ep * (C['epochs'] - epoch - 1)
        stale_tag = f"  [stale {stale_epochs}/{C['patience']}]" if stale_epochs > 0 else ""
        print(f"  ep {epoch+1:4d}/{C['epochs']}  loss={train_loss:.4f}  "
              f"tr={train_acc:.3f}  te={test_acc:.3f}  "
              f"{ep_time:.1f}s/ep  ETA={_fmt(eta)}{stale_tag}", flush=True)

        if stale_epochs >= C['patience']:
            print(f"\n  [EARLY STOP] no improvement for {C['patience']} epochs  "
                  f"best_loss={best_loss:.4f}", flush=True)
            break

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    os.makedirs(C['models_dir'], exist_ok=True)
    model_path   = os.path.join(C['models_dir'], f'model_x_{thr}.pth')
    metrics_path = os.path.join(C['models_dir'], f'metrics_x_{thr}.json')
    torch.save(model.state_dict(), model_path)
    with open(metrics_path, 'w') as f:
        json.dump({
            'threshold': float(thr),
            'history':   history,
            'config': {
                'attn_dim':    attn_dim,
                'bias':        C['bias'],
                'seq_len':     seq_len,
                'suffix_len':  suffix_len,
                'total_vocab': total_vocab,
                'train_size':  int(train_x.shape[0]),
                'test_size':   int(test_x.shape[0]),
                'epochs_run':  len(history['loss']),
                'lr':          C['lr'],
            },
        }, f, indent=2)

    total = time.time() - t0
    print(f"\n  [DONE] thr={thr}  time={_fmt(total)}  "
          f"best_loss={best_loss:.4f}  final_te={test_acc:.3f}", flush=True)
    print(f"  saved → {model_path}", flush=True)
    return history


# ─── Main ────────────────────────────────────────────────────────────────────
def main():
    C = CONFIG
    _set_seed(C['seed'])
    device     = torch.device(C['device'])
    thresholds = C['thresholds'] or _discover_thresholds(C['data_dir'])

    print(f"device     : {device}")
    print(f"attn_dim   : {C['attn_dim'] or 'auto (= total_vocab + seq_len, matches W*)'}")
    print(f"suffix_len : {C['suffix_len']}  (loss & eval on last {C['suffix_len']} tokens only)")
    print(f"lr         : {C['lr']}  grad_clip={C['grad_clip']}  patience={C['patience']}")
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

    summary_path = os.path.join(C['models_dir'], 'training_summary.json')
    with open(summary_path, 'w') as f:
        json.dump({'thresholds': thresholds, 'history': all_history, 'config': C}, f, indent=2)
    print(f"\nSummary → {summary_path}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for thr, h in all_history.items():
        axes[0].plot(h['loss'],     label=f'x={thr}')
        axes[1].plot(h['test_acc'], label=f'x={thr}')
    axes[0].set_title(f'Training loss (last {C["suffix_len"]} tokens)')
    axes[0].set_xlabel('epoch')
    axes[1].set_title(f'Test accuracy (last {C["suffix_len"]} tokens)')
    axes[1].set_xlabel('epoch')
    for ax in axes:
        ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig_path = os.path.join(C['models_dir'], 'training_curves.png')
    plt.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"Curves  → {fig_path}")


if __name__ == '__main__':
    main()
