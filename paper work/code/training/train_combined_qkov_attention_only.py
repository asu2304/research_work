"""
Train a DisentangledTransformer on the Markov-noise ICL task, but with the
attention REPARAMETRIZED into combined circuits:

    Standard block : separate W_q, W_k, W_v, W_o  (4 matrices / layer)
    This block      : combined W_qk  and  W_ov     (2 matrices / layer)

Because attention only ever uses W_q,W_k through the product (W_qᵀ W_k) and
W_v,W_o through the product (W_o W_v), we parametrize those products directly:

        scores   = (x · W_qk) · xᵀ / √out          W_qk : [in , in]
        attn_out =  A · (x · W_ovᵀ)                 W_ov : [out, in]
        h        =  cat[x, attn_out]                (same concat residual)

Everything else is identical to folder 6's train.py (softmax head, suffix-100
loss, fc_out unembedding with bias).  This lets us ask: does GD in the (W_qk,
W_ov) parametrization converge to the SAME solution we hand-constructed
(PerfectInductionTransformer collapsed to W_qk/W_ov) — or a different one?

Artifacts (consistent with the other folders) land in this folder's models/:
    models/model_x_{thr}.pth   models/metrics_x_{thr}.json
    models/training_summary.json   models/training_curves.png

Edit CONFIG below, then run:
    python -u train_combined.py
"""

import json
import os
import tempfile
import time

# This box's /tmp, /var/tmp, /usr/tmp are unusable → route temp under $HOME.
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
    # same dataset as folders 6/8 (absolute path so this script can live here)
    'data_dir':   '/home/hariguru/ashutosh/ICL/6. understanding what gradient descent is firuring out/dataset',
    'models_dir': 'models',          # new folder INSIDE "7. unique_Wqk_Wok_test.ipynb"
    'thresholds':  None,             # None → auto-discover all from dataset/train/data_x_*.pt
    'device':      'cuda:1',
    'resume':      True,             # True → load models/model_x_{thr}.pth + its metrics, train
                                     #        `epochs` MORE, appending to the existing curve
                                     #        (continue, don't restart). False → fresh run.

    # model
    'attn_dim':    220,        # output_dim of each attention block (None → total_vocab+seq_len=220)
    'bias':        True,       # bias in the unembedding head (fc_out)

    # training — only the final suffix_len tokens contribute to loss / accuracy
    'suffix_len':  100,
    'epochs':      30,         # epochs THIS RUN (with resume=True: 30 more on top of the checkpoint)
    'batch_size':  256,
    'lr':          1e-4,       # same proven LR as train.py (constant)
    'weight_decay':0.0,        # 0: matches the softmax baseline in folder 6
    'grad_clip':   1.0,
    'patience':    30,        # effectively off (long pre-grok plateau); Ctrl+C to bank early

    'eval_subset': 1000,
    'num_workers': 0,          # /tmp unusable on this box → no DataLoader workers
    'seed':        42,
    # If a threshold won't climb on seed 42, try another seed here (the full-rank
    # W_qk is MORE expressive than separate Wq,Wk, so convergence is usually easier).
}
# ─────────────────────────────────────────────────────────────────────────────


# ─── Model ───────────────────────────────────────────────────────────────────
class CombinedAttentionBlock(nn.Module):
    """Concat-attention with the QK and OV circuits as single matrices.
        scores   = (x · W_qk) · xᵀ / √out          (W_qk collapses W_qᵀW_k)
        attn_out =  softmax(scores) · (x · W_ovᵀ)   (W_ov collapses W_oW_v)
        return cat[x, attn_out]
    """
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.input_dim  = input_dim
        self.output_dim = output_dim
        self.W_qk = nn.Parameter(torch.empty(input_dim, input_dim))
        self.W_ov = nn.Parameter(torch.empty(output_dim, input_dim))
        nn.init.xavier_uniform_(self.W_qk)
        nn.init.xavier_uniform_(self.W_ov)

    def forward(self, x, mask=None):
        scores = torch.bmm(x @ self.W_qk, x.transpose(1, 2)) / (self.output_dim ** 0.5)
        if mask is not None:
            scores = scores + mask
        attn_out = torch.bmm(F.softmax(scores, dim=-1), x @ self.W_ov.t())
        return torch.cat([x, attn_out], dim=-1), scores


class DisentangledTransformerCombined(nn.Module):
    """Identical wiring to folder 6's DisentangledTransformer, but each attention
    layer is a CombinedAttentionBlock (W_qk, W_ov) instead of W_q/W_k/W_v/W_o."""
    def __init__(self, total_vocab, seq_len, attn_dim, bias=False):
        super().__init__()
        self.total_vocab = total_vocab
        self.seq_len     = seq_len
        self.attn_dim    = attn_dim
        self.initial_dim = total_vocab + seq_len
        self.layer1 = CombinedAttentionBlock(self.initial_dim, attn_dim)
        self.layer2 = CombinedAttentionBlock(self.initial_dim + attn_dim, attn_dim)
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
def _evaluate(model, x, y, batch_size, device, suffix_len, total_vocab):
    """Return (mean per-token test CE, test accuracy) on the last suffix_len tokens."""
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

    model = DisentangledTransformerCombined(total_vocab, seq_len, attn_dim, bias=C['bias']).to(device)
    opt   = optim.AdamW(model.parameters(), lr=C['lr'], weight_decay=C.get('weight_decay', 0.0))

    model_path   = os.path.join(C['models_dir'], f'model_x_{thr}.pth')
    metrics_path = os.path.join(C['models_dir'], f'metrics_x_{thr}.json')

    best_loss, best_state, stale = float('inf'), None, 0
    prev_history = {}
    resumed      = False
    if C.get('resume'):
        if not (os.path.exists(model_path) and os.path.exists(metrics_path)):
            raise FileNotFoundError(f"resume=True but checkpoint/metrics missing for thr={thr}: "
                                    f"{model_path} / {metrics_path}")
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        prev_history = json.load(open(metrics_path))['history']
        # never save worse than the loaded checkpoint (by test loss)
        best_loss  = min(prev_history.get('test_loss') or [float('inf')])
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        resumed    = True

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n{'='*68}", flush=True)
    mode = (f"RESUME (+{C['epochs']} ep on top of {len(prev_history['loss'])}, "
            f"loaded best_te_loss={best_loss:.4f})") if resumed else "FRESH"
    print(f"  [{thr_idx+1}/{n_thrs}] thr={thr}  COMBINED(W_qk,W_ov)  {mode}  attn_dim={attn_dim}  "
          f"params={n_params:,}  train={train_x.shape[0]:,}  loss_suffix={suffix_len}/{seq_len}", flush=True)
    print(f"{'='*68}", flush=True)

    history = {'loss': [], 'train_acc': [], 'test_acc': [], 'test_loss': []}
    interrupted = False
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
                logits_s = model(xb)[:, -suffix_len:, :]
                yb_s     = yb[:, -suffix_len:]
                loss = F.cross_entropy(logits_s.reshape(-1, total_vocab), yb_s.reshape(-1))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), C['grad_clip'])
                opt.step()
                l_sum += loss.item(); n_steps += 1
                ep_correct += (logits_s.detach().argmax(-1) == yb_s).sum().item()
                ep_total   += yb_s.numel()

            train_loss = l_sum / n_steps
            train_acc  = ep_correct / ep_total
            test_loss, test_acc = _evaluate(model, eval_x, eval_y, C['batch_size'],
                                            device, suffix_len, total_vocab)
            history['loss'].append(train_loss); history['train_acc'].append(train_acc)
            history['test_acc'].append(test_acc); history['test_loss'].append(test_loss)

            if test_loss < best_loss:
                best_loss, stale = test_loss, 0
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                stale += 1

            elapsed = time.time() - t0
            eta = (elapsed / (epoch + 1)) * (C['epochs'] - epoch - 1)
            tag = f"  [stale {stale}/{C['patience']}]" if stale > 0 else "  *best"
            print(f"  ep {epoch+1:4d}/{C['epochs']}  tr_loss={train_loss:.4f}  te_loss={test_loss:.4f}  "
                  f"tr={train_acc:.3f}  te={test_acc:.3f}  {time.time()-ep_t0:.1f}s/ep  "
                  f"ETA={_fmt(eta)}{tag}", flush=True)

            if stale >= C['patience']:
                print(f"\n  [EARLY STOP] no test-loss gain for {C['patience']} epochs  "
                      f"best_te_loss={best_loss:.4f}", flush=True)
                break
    except KeyboardInterrupt:
        interrupted = True
        print(f"\n  [INTERRUPTED] Ctrl+C after {len(history['loss'])} epochs — "
              f"saving best-so-far (best_te_loss={best_loss:.4f}) and moving on", flush=True)

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    # append the new epochs onto the prior curve (resume), else just this run
    full_history = {k: list(prev_history.get(k, [])) + list(history[k]) for k in history}

    os.makedirs(C['models_dir'], exist_ok=True)
    torch.save(model.state_dict(), model_path)
    with open(metrics_path, 'w') as f:
        json.dump({
            'threshold': float(thr),
            'history':   full_history,
            'config': {
                'attn_dim':    attn_dim,
                'bias':        C['bias'],
                'head':        'softmax',
                'param':       'combined_qk_ov',     # tag: W_qk / W_ov parametrization
                'seq_len':     seq_len,
                'suffix_len':  suffix_len,
                'total_vocab': total_vocab,
                'train_size':  int(train_x.shape[0]),
                'test_size':   int(test_x.shape[0]),
                'epochs_run':  len(full_history['loss']),
                'epochs_added': len(history['loss']) if resumed else 0,
                'resumed':     resumed,
                'lr':          C['lr'],
                'weight_decay':C.get('weight_decay', 0.0),
                'select_by':   'test_loss',
                'seed':        C['seed'],
            },
        }, f, indent=2)

    total    = time.time() - t0
    final_te = full_history['test_acc'][-1] if full_history['test_acc'] else float('nan')
    print(f"\n  [DONE] thr={thr}  time={_fmt(total)}  best_te_loss={best_loss:.4f}  "
          f"final_te_acc={final_te:.3f}  total_epochs={len(full_history['loss'])}", flush=True)
    print(f"  saved → {model_path}", flush=True)
    return history, interrupted


# ─── Main ────────────────────────────────────────────────────────────────────
def main():
    C = CONFIG
    _set_seed(C['seed'])
    device     = torch.device(C['device'])
    thresholds = C['thresholds'] or _discover_thresholds(C['data_dir'])

    print(f"device     : {device}")
    print(f"mode       : {'RESUME (+' + str(C['epochs']) + ' epochs per threshold)' if C.get('resume') else 'FRESH'}")
    print(f"model      : DisentangledTransformer COMBINED (W_qk, W_ov)  softmax head")
    print(f"attn_dim   : {C['attn_dim'] or 'auto (= total_vocab + seq_len = 220)'}")
    print(f"data_dir   : {C['data_dir']}")
    print(f"suffix_len : {C['suffix_len']}  (loss & eval on last {C['suffix_len']} tokens only)")
    print(f"lr         : {C['lr']} (constant)  weight_decay={C.get('weight_decay', 0.0)}  "
          f"grad_clip={C['grad_clip']}  patience={C['patience']}")
    print(f"epochs     : {C['epochs']}  batch={C['batch_size']}  seed={C['seed']}")
    print(f"thresholds : {thresholds}\n")

    t_all = time.time()
    all_history = {}
    for i, thr in enumerate(thresholds):
        all_history[thr], interrupted = train_one(thr, i, len(thresholds), device)
        done    = i + 1
        elapsed = time.time() - t_all
        eta_all = (elapsed / done) * (len(thresholds) - done)
        print(f"\n  [{done}/{len(thresholds)} done]  elapsed={_fmt(elapsed)}  "
              f"est remaining={_fmt(eta_all)}", flush=True)
        if interrupted:    # Ctrl+C saved this threshold; stop before starting the next
            print(f"  [STOP] interrupted on thr={thr} — saved it; skipping the rest. "
                  f"Re-run with the remaining thresholds in CONFIG.", flush=True)
            break

    # Build summary + curves from ALL on-disk metrics (not just this run's thresholds),
    # so retraining a subset (e.g. [0.8, 0.9]) keeps the full picture instead of
    # shrinking training_summary.json / training_curves.png to only the retrained ones.
    import glob
    disk = {}
    for mp in glob.glob(os.path.join(C['models_dir'], 'metrics_x_*.json')):
        try:
            thr = float(os.path.basename(mp)[len('metrics_x_'):-len('.json')])
            disk[thr] = json.load(open(mp))['history']
        except (ValueError, KeyError):
            pass

    summary_path = os.path.join(C['models_dir'], 'training_summary.json')
    with open(summary_path, 'w') as f:
        json.dump({'thresholds': sorted(disk), 'history': disk,
                   'this_run': sorted(all_history), 'config': C}, f, indent=2)
    print(f"\nSummary → {summary_path}  (all thresholds on disk: {sorted(disk)})")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for thr in sorted(disk):
        h = disk[thr]
        axes[0].plot(h['loss'],     label=f'x={thr}')
        axes[1].plot(h['test_acc'], label=f'x={thr}')
    axes[0].set_title(f'Training loss (last {C["suffix_len"]} tokens)  [combined W_qk,W_ov]')
    axes[0].set_xlabel('epoch')
    axes[1].set_title(f'Test accuracy (last {C["suffix_len"]} tokens)  [combined]')
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
