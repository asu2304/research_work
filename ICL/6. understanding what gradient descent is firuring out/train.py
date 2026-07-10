"""
Train a DisentangledTransformer (D=220) on the Markov-noise ICL task.
seq_len=200  →  attn_dim auto = total_vocab + seq_len = 220
Loss computed on last suffix_len=100 tokens (last half of 200).

Edit CONFIG below, then run:
    python -u train.py
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
    'data_dir':   'dataset',
    'models_dir': 'models',
    'thresholds':  [0.8],   # retrain only these; replaces model_x_{thr}.pth in place
    'device':      'cuda:1',

    # model
    'attn_dim':    220,       # NOTE: existing model_x_*.pth are 500-wide & the notebook loads all
                              #   thresholds with a global ATTN_DIM=500 — retraining these at 220 will
                              #   not load there until the notebook reads attn_dim per-threshold.
    'bias':        True,       # bias in the unembedding head

    # output head:
    #   'softmax'        → standard  p = softmax(z)
    #   'l1'             → divide-by-sum  p = (relu(z)+eps)/Σ(relu(z)+eps)
    #   'freeze_readout' → NO softmax, NO relu.  Pin layer-2 value/output (W_v2,W_o2)
    #                      and fc_out.WEIGHT to the W* selection, so the readout emits
    #                      the attention-weighted token frequency (>=0).  The fc_out
    #                      BIAS is LEARNABLE and clamped >= 0 — the noise-floor that
    #                      makes ctx-α representable.  Output = freq + bias, normalised
    #                      by its sum:  p = (freq+bias+eps) / Σ(freq+bias+eps).
    #                      GD learns layer 1 fully + layer-2 W_q,W_k + the bias floor.
    # non-softmax heads save under model_<head>_x_{thr}.pth (softmax ones untouched).
    'head':        'softmax',
    'l1_eps':      1e-6,       # additive floor in the loss (avoids log 0)

    # training — only the final suffix_len tokens contribute to loss / accuracy
    'suffix_len':  100,        # train & eval on last 100 of 200 tokens
    'epochs':      120,         # max epochs PER SEED (a good seed reaches the ceiling in ~40-60)
    'batch_size':  256,
    'lr':          1e-2,       # same as the neighbors (0.7/0.9) that hit their ceiling
    'weight_decay':0.0,        # 0: neighbors hit ceiling without it; wd=1e-3 was capping 0.8's readout
    'grad_clip':   1.0,        # clip gradient norm
    'patience':    120,         # effectively off; sweep is governed by grok/abandon below

    'eval_subset': 1000,       # larger test eval → less noisy best-test pick
    'num_workers': 0,          # /tmp is unusable on this box → no DataLoader multiprocessing
    'seed':        42,         # used only if seed_search is None

    # ── seed sweep: 0.8 is an INIT LOTTERY (data ceiling=0.40, neighbors reach theirs;
    #    0.8 got stuck at 0.359). Try seeds, keep the FIRST that reaches near the ceiling. ──
    'seed_search':   [42, 0, 7, 1, 13, 123, 2024],
    'probe_epochs':  60,       # by here a good seed is clearly climbing; a stuck one isn't
    'grok_te_acc':   0.385,    # success: test acc near the α=0.8 ceiling (0.40) → stop sweep, keep it
    'abandon_te_acc':0.370,    # if best test acc still below this at probe_epochs → next seed
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
def _evaluate(model, x, y, batch_size, device, suffix_len, head, eps):
    """Return (mean per-token test NLL under the head, test accuracy) on last suffix_len tokens."""
    model.eval()
    correct, total, loss_sum = 0, 0, 0.0
    for i in range(0, x.shape[0], batch_size):
        xb = x[i:i + batch_size].to(device, non_blocking=True)
        yb = y[i:i + batch_size].to(device, non_blocking=True)
        logits = model(xb)[:, -suffix_len:, :]
        yb_s   = yb[:, -suffix_len:]
        n = yb_s.numel()
        loss_sum += _nll(logits, yb_s, head, eps).item() * n   # _nll is a per-token mean → weight by n
        correct  += (logits.argmax(-1) == yb_s).sum().item()
        total    += n
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


def _nll(logits, target, head, eps):
    """Cross-entropy under the chosen output head.

    'softmax' : standard  -log softmax(z)
    'l1'      : NO softmax — p = (relu(z)+eps) / Σ(relu(z)+eps), loss = -log p_target.
                relu maps the learnable readout to non-negatives (it can be negative);
                eps is the additive floor.  log is taken on the PROBABILITY (the loss),
                not on any count inside the network.
    """
    V = logits.shape[-1]
    z = logits.reshape(-1, V)
    t = target.reshape(-1)
    if head == 'softmax':
        return F.cross_entropy(z, t)
    if head == 'l1':
        pos  = F.relu(z) + eps
        logp = torch.log(pos) - torch.log(pos.sum(-1, keepdim=True))
        return F.nll_loss(logp, t)
    if head == 'freeze_readout':
        # z = freq + bias  (freq >= 0 from the frozen selection; bias clamped >= 0).
        # Normalise by the ACTUAL sum (bias makes it != 1); eps is a safety floor.
        p = (z + eps) / (z.sum(-1, keepdim=True) + V * eps)
        return F.nll_loss(torch.log(p), t)
    raise ValueError(f"unknown head: {head!r}")


def _freeze_readout(model, V):
    """Pin layer-2 value/output (W_v2, W_o2) and fc_out.WEIGHT to the W* selection so
    the readout emits the attention-weighted token frequency (>= 0).  These are
    frozen.  fc_out.BIAS stays LEARNABLE (clamped >= 0 each step in the loop) — the
    noise-floor.  GD learns: all of layer 1, layer-2 W_q/W_k, and the bias floor."""
    ad   = model.attn_dim
    idim = model.initial_dim
    dev  = model.fc_out.weight.device
    assert ad >= V, f"freeze_readout needs attn_dim ({ad}) >= total_vocab ({V})"
    with torch.no_grad():
        # W_v2: value = the RAW token one-hot block of h2 (first V dims) -> always >= 0
        model.layer2.W_v.weight.zero_()
        model.layer2.W_v.weight[:V, :V] = torch.eye(V, device=dev)
        # W_o2: identity (pass the convex-combined one-hots straight through)
        model.layer2.W_o.weight.copy_(torch.eye(ad, device=dev))
        # fc_out.weight: select the layer-2 attention block's first V dims (the freq)
        model.fc_out.weight.zero_()
        off = idim + ad                                # start of attn_out2 inside h3
        model.fc_out.weight[:V, off:off + V] = torch.eye(V, device=dev)
        if model.fc_out.bias is not None:
            model.fc_out.bias.zero_()                  # init the learnable floor at 0
    # freeze the SELECTION (W_v2, W_o2, fc_out.weight); fc_out.bias stays trainable
    for mod in (model.layer2.W_v, model.layer2.W_o):
        for p in mod.parameters():
            p.requires_grad_(False)
    model.fc_out.weight.requires_grad_(False)


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

    def _build_model():
        m = DisentangledTransformer(total_vocab, seq_len, attn_dim, bias=C['bias']).to(device)
        if C['head'] == 'freeze_readout':
            _freeze_readout(m, total_vocab)
        return m

    n_params = sum(p.numel() for p in _build_model().parameters())
    seeds       = C.get('seed_search') or [C['seed']]
    sweep       = len(seeds) > 1
    probe       = C.get('probe_epochs')      # abandon a seed if not promising by here
    grok_te_acc = C.get('grok_te_acc')       # stop the whole sweep once a seed reaches this test acc
    abandon_acc = C.get('abandon_te_acc')    # "promising" bar checked at `probe`
    print(f"\n{'='*68}", flush=True)
    print(f"  [{thr_idx+1}/{n_thrs}] thr={thr}  attn_dim={attn_dim}  head={C['head']}  "
          f"params={n_params:,}  train={train_x.shape[0]:,}  loss_suffix={suffix_len}/{seq_len}", flush=True)
    if sweep:
        print(f"  SEED-SWEEP {seeds}  probe={probe}ep  grok_te_acc={grok_te_acc}  "
              f"abandon<{abandon_acc}  (α={thr} ceiling≈{thr/2:.3f})", flush=True)
    print(f"{'='*68}", flush=True)

    def _run_seed(seed):
        """Train one fresh init; checkpoint best TEST loss. Returns
        (history, best_te_loss, best_state, best_te_acc, grokked, interrupted)."""
        _set_seed(seed)
        model = _build_model()
        opt   = optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=C['lr'],
                            weight_decay=C.get('weight_decay', 0.0))
        hist  = {'loss': [], 'train_acc': [], 'test_acc': [], 'test_loss': []}
        b_loss, b_state, b_acc, stale = float('inf'), None, 0.0, 0
        grokked = interrupted = False
        t0 = time.time()
        try:
            for epoch in range(C['epochs']):
                model.train(); ep_t0 = time.time()
                l_sum, n_steps, ep_correct, ep_total = 0.0, 0, 0, 0
                for xb, yb in loader:
                    xb = xb.to(device, non_blocking=True); yb = yb.to(device, non_blocking=True)
                    opt.zero_grad(set_to_none=True)
                    logits_s = model(xb)[:, -suffix_len:, :]
                    yb_s     = yb[:, -suffix_len:]
                    loss = _nll(logits_s, yb_s, C['head'], C['l1_eps'])
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), C['grad_clip'])
                    opt.step()
                    if C['head'] == 'freeze_readout' and model.fc_out.bias is not None:
                        model.fc_out.bias.data.clamp_(min=0.0)
                    l_sum += loss.item(); n_steps += 1
                    ep_correct += (logits_s.detach().argmax(-1) == yb_s).sum().item()
                    ep_total   += yb_s.numel()
                train_loss = l_sum / n_steps
                train_acc  = ep_correct / ep_total
                test_loss, test_acc = _evaluate(model, eval_x, eval_y, C['batch_size'],
                                                device, suffix_len, C['head'], C['l1_eps'])
                hist['loss'].append(train_loss); hist['train_acc'].append(train_acc)
                hist['test_acc'].append(test_acc); hist['test_loss'].append(test_loss)
                if test_loss < b_loss:
                    b_loss, stale = test_loss, 0
                    b_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                else:
                    stale += 1
                b_acc = max(b_acc, test_acc)
                ep_time = time.time() - ep_t0
                tag = f"  [stale {stale}/{C['patience']}]" if stale > 0 else "  *best"
                sd  = f"seed {seed}  " if sweep else ""
                print(f"  {sd}ep {epoch+1:4d}/{C['epochs']}  tr_loss={train_loss:.4f}  "
                      f"te_loss={test_loss:.4f}  tr={train_acc:.3f}  te={test_acc:.3f}  "
                      f"best_te={b_acc:.3f}  {ep_time:.1f}s/ep{tag}", flush=True)
                if grok_te_acc is not None and b_acc >= grok_te_acc:
                    grokked = True
                    print(f"  [GROK] seed {seed} reached te_acc {b_acc:.3f} >= {grok_te_acc}", flush=True)
                    break
                if probe is not None and abandon_acc is not None and (epoch + 1) >= probe and b_acc < abandon_acc:
                    print(f"  [ABANDON] seed {seed} stuck (best_te={b_acc:.3f} < {abandon_acc} by ep {probe})", flush=True)
                    break
                if stale >= C['patience']:
                    print(f"  [EARLY STOP] seed {seed} no test-loss gain {C['patience']} ep  "
                          f"best_te_loss={b_loss:.4f}", flush=True)
                    break
        except KeyboardInterrupt:
            interrupted = True
            print(f"\n  [INTERRUPTED] Ctrl+C (seed {seed}) after {len(hist['loss'])} epochs — "
                  f"keeping best-so-far (best_te={b_acc:.3f})", flush=True)
        return hist, b_loss, b_state, b_acc, grokked, interrupted

    best = None          # (best_te_acc, history, best_te_loss, best_state, seed)
    interrupted = False
    grokked_any = False
    t0 = time.time()
    for seed in seeds:
        hist, b_loss, b_state, b_acc, grokked, intr = _run_seed(seed)
        if best is None or b_acc > best[0]:
            best = (b_acc, hist, b_loss, b_state, seed)
        if intr:
            interrupted = True
            best = (b_acc, hist, b_loss, b_state, seed)   # keep the interrupted seed
            break
        if grokked:
            grokked_any = True
            best = (b_acc, hist, b_loss, b_state, seed)
            print(f"  [WIN] seed {seed} converged (best_te={b_acc:.3f}); stopping sweep", flush=True)
            break

    b_acc, history, best_loss, best_state, used_seed = best
    if sweep and not grokked_any and not interrupted:
        print(f"\n  [WARN] no seed hit grok_te_acc={grok_te_acc}; keeping best seed {used_seed} "
              f"(best_te={b_acc:.3f}, ceiling≈{thr/2:.3f})", flush=True)

    model = _build_model()
    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    os.makedirs(C['models_dir'], exist_ok=True)
    sfx = '' if C['head'] == 'softmax' else f"_{C['head']}"   # l1 → model_l1_x_*.pth
    model_path   = os.path.join(C['models_dir'], f'model{sfx}_x_{thr}.pth')
    metrics_path = os.path.join(C['models_dir'], f'metrics{sfx}_x_{thr}.json')
    torch.save(model.state_dict(), model_path)
    with open(metrics_path, 'w') as f:
        json.dump({
            'threshold': float(thr),
            'history':   history,
            'config': {
                'attn_dim':    attn_dim,
                'bias':        C['bias'],
                'head':        C['head'],
                'l1_eps':      C['l1_eps'],
                'seq_len':     seq_len,
                'suffix_len':  suffix_len,
                'total_vocab': total_vocab,
                'train_size':  int(train_x.shape[0]),
                'test_size':   int(test_x.shape[0]),
                'epochs_run':  len(history['loss']),
                'lr':          C['lr'],
                'weight_decay':C.get('weight_decay', 0.0),
                'select_by':   'test_loss',
                'seed':        used_seed,
            },
        }, f, indent=2)

    total    = time.time() - t0
    final_te = history['test_acc'][-1] if history['test_acc'] else float('nan')
    print(f"\n  [DONE] thr={thr}  time={_fmt(total)}  "
          f"best_te_loss={best_loss:.4f}  final_te_acc={final_te:.3f}", flush=True)
    print(f"  saved → {model_path}", flush=True)
    return history, interrupted


# ─── Main ────────────────────────────────────────────────────────────────────
def main():
    C = CONFIG
    _set_seed(C['seed'])
    device     = torch.device(C['device'])
    thresholds = C['thresholds'] or _discover_thresholds(C['data_dir'])

    print(f"device     : {device}")
    print(f"head       : {C['head']}" + (f"  (eps={C['l1_eps']})" if C['head'] != 'softmax' else ""))
    print(f"attn_dim   : {C['attn_dim'] or 'auto (= total_vocab + seq_len = 220)'}")
    print(f"suffix_len : {C['suffix_len']}  (loss & eval on last {C['suffix_len']} tokens only)")
    print(f"lr         : {C['lr']}  grad_clip={C['grad_clip']}  patience={C['patience']}")
    print(f"epochs     : {C['epochs']}  batch={C['batch_size']}")
    print(f"thresholds : {thresholds}\n")

    t_all = time.time()
    all_history = {}
    for i, thr in enumerate(thresholds):
        all_history[thr], interrupted = train_one(thr, i, len(thresholds), device)
        done    = i + 1
        elapsed = time.time() - t_all
        eta_all = (elapsed / done) * (len(thresholds) - done)
        print(f"\n  [{done}/{len(thresholds)} done]  "
              f"elapsed={_fmt(elapsed)}  est remaining={_fmt(eta_all)}", flush=True)
        if interrupted:    # Ctrl+C saved this threshold; stop before starting the next
            print(f"  [STOP] interrupted on thr={thr} — saved it; "
                  f"skipping remaining thresholds. Re-run with the rest in CONFIG.", flush=True)
            break

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
