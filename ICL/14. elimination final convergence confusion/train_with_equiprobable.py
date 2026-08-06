"""Combined vs separate on the equiprobable-successor dataset (this folder's data_generation.py).
Same model, frozen log readout, init and optimizer settings as train.py."""

import glob
import json
import math
import os
import sys
import tempfile
import time

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

CFG = {
    'data_dir': os.path.join(_SCRIPT_DIR, 'dataset'),
    'runs_dir': os.path.join(_SCRIPT_DIR, 'runs'),
    'device':   'cuda:1',

    'v_sig':       4,
    'n_noise':     4,
    'seq_len':     100,
    'token_start': 50,
    'bias':        True,

    'freeze_delta':     0.01,
    'freeze_hidden_k':  128,
    'freeze_fit_fmax':  40.0,
    'freeze_fit_cgain': 32.0,
    'freeze_fit_tol':   5e-3,

    'optimizer':   'adam',
    'lr':          0.003,
    'batch_size':  256,
    'init':        'normal',
    'weight_decay': 0.0,
    'eval_batch':  256,
    'epochs':      2000,
    'early_stop':  False,
    'patience':    50,
}

ARCHS      = ['separate', 'combined']
SEEDS      = [0]
THRESHOLDS = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

CELL_PREFIX   = 'equi'
SKIP_DONE     = True
USE_WANDB     = True
WANDB_PROJECT = 'ICL-equiprobable-successors'
WANDB_ENTITY  = None


def cell_tag():
    return (f"{CELL_PREFIX}_opt-{CFG['optimizer']}_lr-{CFG['lr']}_b-{CFG['batch_size']}"
            f"_init-{CFG['init']}_ep-{CFG['epochs']}")


class CombinedAttentionBlock(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.output_dim = output_dim
        self.W_qk = nn.Parameter(torch.empty(input_dim, input_dim))
        self.W_ov = nn.Parameter(torch.empty(input_dim, output_dim))
        s = input_dim ** -0.5
        nn.init.uniform_(self.W_qk, -s, s)
        nn.init.uniform_(self.W_ov, -s, s)

    def forward(self, x, mask=None):
        scores = torch.bmm(x @ self.W_qk, x.transpose(1, 2)) / (self.output_dim ** 0.5)
        if mask is not None:
            scores = scores + mask
        attn_out = torch.bmm(F.softmax(scores, dim=-1), x @ self.W_ov)
        return torch.cat([x, attn_out], dim=-1)


class SeparateAttentionBlock(nn.Module):
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
        return torch.cat([x, attn_out], dim=-1)


_BLOCKS = {'combined': CombinedAttentionBlock, 'separate': SeparateAttentionBlock}


class SimplifiedTransformerMLP(nn.Module):
    def __init__(self, total_vocab, seq_len, attn_dim, v_sig, hidden_k, bias, attn_param):
        super().__init__()
        self.total_vocab = total_vocab
        self.seq_len     = seq_len
        self.v_sig       = v_sig
        self.attn_param  = attn_param
        self.initial_dim = total_vocab + seq_len
        Block = _BLOCKS[attn_param]
        self.layer1 = Block(self.initial_dim, attn_dim)
        self.layer2 = Block(self.initial_dim + attn_dim, attn_dim)
        self.D   = self.initial_dim + 2 * attn_dim
        self.in2 = self.initial_dim + attn_dim
        self.ff1 = nn.Linear(self.D, v_sig * hidden_k)
        self.ff2 = nn.Linear(v_sig * hidden_k, self.D)
        self.act = nn.GELU()
        self.fc_out = nn.Linear(self.D, total_vocab, bias=bias)

    def stream(self, x):
        B, T = x.shape
        tok = F.one_hot(x, num_classes=self.total_vocab).float()
        pos = F.one_hot(torch.arange(T, device=x.device),
                        num_classes=self.seq_len).float().unsqueeze(0).expand(B, T, -1)
        s = torch.cat([tok, pos], dim=-1)
        mask = torch.triu(torch.ones(T, T, device=x.device) * float('-inf'), diagonal=1)
        return self.layer2(self.layer1(s, mask), mask)

    def forward(self, x):
        s = self.stream(x)
        s = s + self.ff2(self.act(self.ff1(s)))
        return self.fc_out(s)


def attn_mats(model):
    out = []
    for layer in (model.layer1, model.layer2):
        if hasattr(layer, 'W_qk'):
            out.append((layer.W_qk, layer.W_qk.shape[0]))
            out.append((layer.W_ov, layer.W_ov.shape[0]))
        else:
            for lin in (layer.W_q, layer.W_k, layer.W_v, layer.W_o):
                out.append((lin.weight, lin.weight.shape[1]))
    return out


def apply_init(model, seed):
    with torch.no_grad():
        for p, fan_in in attn_mats(model):
            p.normal_(0.0, 1.0 / math.sqrt(fan_in))
        gen = torch.Generator(device='cpu').manual_seed(int(seed) * 1000 + 7)
        dev = next(model.parameters()).device
        s = 1.0 / math.sqrt(model.fc_out.weight.shape[1])
        model.fc_out.weight.copy_(
            (torch.randn(model.fc_out.weight.shape, generator=gen) * s).to(dev))
        model.fc_out.bias.copy_(
            (torch.randn(model.fc_out.bias.shape, generator=gen) * s).to(dev))


def fit_log_core(delta, k, fmax, cgain, n_grid=40001):
    dt = torch.float64
    kn = torch.exp(torch.linspace(math.log(delta), math.log(fmax + delta), k, dtype=dt)) - delta
    h  = torch.diff(kn, prepend=(2 * kn[:1] - kn[1:2])).clamp_min(1e-12)
    a  = cgain / h
    b  = -a * kn
    fg = (torch.exp(torch.linspace(math.log(delta), math.log(fmax + delta), n_grid,
                                   dtype=dt)) - delta).clamp_min(0.0)
    basis  = F.gelu(fg.unsqueeze(1) * a.unsqueeze(0) + b.unsqueeze(0))
    basis1 = torch.cat([basis, torch.ones(n_grid, 1, dtype=dt)], dim=1)
    sol = torch.linalg.lstsq(basis1, (torch.log(fg + delta) - fg).unsqueeze(1)).solution.squeeze(1)
    w, c = sol[:k], float(sol[k])
    a32, b32, w32 = a.float(), b.float(), w.float()
    f32 = fg.float()
    err = (F.gelu(f32.unsqueeze(1) * a32 + b32) @ w32 + c + f32
           - torch.log(f32 + delta)).abs()
    return a32, b32, w32, c, float(err.max())


def verify_log_core(a, b, w, c, delta, fmax, tol):
    def implied(f):
        f = torch.as_tensor(f, dtype=torch.float32).reshape(-1)
        return F.gelu(f.unsqueeze(1) * a + b) @ w + c + f
    worst = 0.0
    for lo, hi in [(0.0, 1.0), (1.0, 5.0), (5.0, fmax)]:
        g = torch.linspace(lo, hi, 20001)
        e = float((implied(g) - torch.log(g + delta)).abs().max())
        worst = max(worst, e)
        print(f'    f in [{lo:5.1f},{hi:5.1f}] : max|err| = {e:.2e}')
    g = torch.linspace(0.0, 1.0, 20001)
    mono = bool((torch.diff(implied(g)) > 0).all())
    print(f'    monotone on [0,1]: {mono}')
    if worst > tol or not mono:
        sys.exit(f'frozen readout is not a usable log (max|err|={worst:.3e}, monotone={mono})')


_CORE = {}


def build_frozen_ffn(model):
    key = (CFG['freeze_delta'], CFG['freeze_hidden_k'], CFG['freeze_fit_fmax'],
           CFG['freeze_fit_cgain'])
    if key not in _CORE:
        core = fit_log_core(*key)
        print(f'  [frozen readout] delta={key[0]} k={key[1]} fmax={key[2]} cgain={key[3]}'
              f'  max|err|={core[4]:.2e}')
        verify_log_core(*core[:4], key[0], key[2], CFG['freeze_fit_tol'])
        _CORE[key] = core
    a, b, w, c, _ = _CORE[key]
    dev = next(model.parameters()).device
    a, b, w = a.to(dev), b.to(dev), w.to(dev)
    k, Vs, D, in2 = CFG['freeze_hidden_k'], model.v_sig, model.D, model.in2
    W1 = torch.zeros(Vs * k, D, device=dev); B1 = torch.zeros(Vs * k, device=dev)
    W2 = torch.zeros(D, Vs * k, device=dev); B2 = torch.zeros(D, device=dev)
    for v in range(Vs):
        rows = slice(v * k, (v + 1) * k)
        W1[rows, in2 + v] = a
        B1[rows]          = b
        W2[in2 + v, rows] = w
        B2[in2 + v]       = c
    with torch.no_grad():
        model.ff1.weight.copy_(W1); model.ff1.bias.copy_(B1)
        model.ff2.weight.copy_(W2); model.ff2.bias.copy_(B2)
    for p in (model.ff1.weight, model.ff1.bias, model.ff2.weight, model.ff2.bias):
        p.requires_grad_(False)


@torch.no_grad()
def readout_range(model, x, device, batch=256, n_max=2048):
    was_training = model.training
    model.eval()
    lo, hi, n_out, n_neg, tot = float('inf'), float('-inf'), 0, 0, 0
    for i in range(0, min(x.shape[0], n_max), batch):
        s = model.stream(x[i:i + batch].to(device))
        f = s[:, CFG['token_start']:CFG['seq_len'], model.in2:model.in2 + model.v_sig]
        lo = min(lo, float(f.min())); hi = max(hi, float(f.max()))
        n_out += int((f > CFG['freeze_fit_fmax']).sum()); n_neg += int((f < 0).sum())
        tot += f.numel()
    if was_training:
        model.train()
    return {'min': lo, 'max': hi,
            'frac_above_fmax': n_out / max(tot, 1), 'frac_negative': n_neg / max(tot, 1)}


_DATA = {}


def load_data(split, thr):
    key = (split, thr)
    if key not in _DATA:
        if any(k[1] != thr for k in _DATA):
            _DATA.clear()
        _DATA[key] = torch.load(os.path.join(CFG['data_dir'], split, f'data_x_{thr}.pt'),
                                weights_only=False)
    return _DATA[key]


def trainable(model):
    return [p for p, _ in attn_mats(model)] + list(model.fc_out.parameters())


@torch.no_grad()
def eval_ce(model, x, y, V, device):
    model.eval()
    ts, te = CFG['token_start'], CFG['seq_len']
    total, n = 0.0, 0
    for i in range(0, x.shape[0], CFG['eval_batch']):
        xb = x[i:i + CFG['eval_batch']].to(device)
        yb = y[i:i + CFG['eval_batch'], ts:te].to(device)
        logits = model(xb)[:, ts:te]
        total += F.cross_entropy(logits.reshape(-1, V), yb.reshape(-1),
                                 reduction='sum').item()
        n += yb.numel()
    model.train()
    return total / n


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
            print(f'  [wandb] disabled ({type(e).__name__}: {e})')
        _WANDB_OK = False
        return None


def metrics_path(arch, seed, thr):
    return os.path.join(CFG['runs_dir'], cell_tag(), f'metrics_{arch}_seed{seed}_x_{thr}.json')


def train_one(arch, seed, thr, device):
    V = CFG['v_sig'] + CFG['n_noise']
    ts, tend = CFG['token_start'], CFG['seq_len']
    tag = cell_tag()
    out_dir = os.path.join(CFG['runs_dir'], tag)
    os.makedirs(out_dir, exist_ok=True)
    stem = f'{arch}_seed{seed}_x_{thr}'
    mpath = os.path.join(out_dir, f'metrics_{stem}.json')
    cpath = os.path.join(out_dir, f'model_{stem}.pth')
    if SKIP_DONE and os.path.exists(mpath):
        print(f'  [skip done] {stem}')
        return False

    run_cfg = {**{k: v for k, v in CFG.items() if k not in ('runs_dir', 'data_dir')},
               'attn_param': arch, 'seed': seed, 'threshold': thr, 'cell': tag,
               'succ_split': 'equiprobable'}

    torch.manual_seed(seed); np.random.seed(seed)
    tr_blob, te_blob = load_data('train', thr), load_data('test', thr)
    tr_x, tr_y = tr_blob['x'], tr_blob['y']
    te_x, te_y = te_blob['x'], te_blob['y']

    model = SimplifiedTransformerMLP(V, CFG['seq_len'], V + CFG['seq_len'], CFG['v_sig'],
                                     CFG['freeze_hidden_k'], CFG['bias'], arch).to(device)
    apply_init(model, seed)
    build_frozen_ffn(model)
    params = trainable(model)
    opt = torch.optim.Adam(params, lr=CFG['lr'], weight_decay=CFG['weight_decay'])

    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(tr_x, tr_y), batch_size=CFG['batch_size'], shuffle=True)

    wb = wandb_start(run_cfg, name=f'{tag}__{stem}', group=f'thr{thr}',
                     tags=[arch, f'thr{thr}', f'seed{seed}', 'equiprobable'])

    tr0 = eval_ce(model, tr_x, tr_y, V, device)
    te0 = eval_ce(model, te_x, te_y, V, device)
    rr0 = readout_range(model, te_x, device)
    history = {'train_loss': [tr0], 'test_loss': [te0],
               'readout_min': [rr0['min']], 'readout_max': [rr0['max']]}
    if wb:
        for m in ('train_ce', 'test_ce'):
            wb.define_metric(m, summary='min')
        for m in ('best_train', 'best_test'):
            wb.define_metric(m, summary='last')
        wb.log({'train_ce': tr0, 'test_ce': te0, 'best_train': tr0, 'best_test': te0,
                **{f'readout_{k}': v for k, v in rr0.items()}}, step=0)

    print(f'  {stem}  trainable={sum(p.numel() for p in params):,}  '
          f'init: tr={tr0:.4f} te={te0:.4f}  f in [{rr0["min"]:.3f}, {rr0["max"]:.3f}]',
          flush=True)

    best_train = float('inf'); test_at_best = None
    best_test = te0
    best_state = None; stale = 0
    t0 = time.time(); ep = 0
    interrupted_twice = False
    try:
        for ep in range(1, CFG['epochs'] + 1):
            model.train()
            ep_loss, ep_n = 0.0, 0
            for xb, yb in loader:
                xb = xb.to(device); yb = yb[:, ts:tend].to(device)
                opt.zero_grad()
                logits = model(xb)[:, ts:tend]
                loss = F.cross_entropy(logits.reshape(-1, V), yb.reshape(-1))
                loss.backward()
                opt.step()
                ep_loss += loss.item() * yb.numel(); ep_n += yb.numel()
            tr_loss = ep_loss / ep_n
            te_loss = eval_ce(model, te_x, te_y, V, device)
            rr = readout_range(model, te_x, device)
            history['train_loss'].append(tr_loss)
            history['test_loss'].append(te_loss)
            history['readout_min'].append(rr['min'])
            history['readout_max'].append(rr['max'])

            if tr_loss < best_train - 1e-6:
                best_train = tr_loss; test_at_best = te_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                stale = 0
            else:
                stale += 1
            best_test = min(best_test, te_loss)

            if wb:
                wb.log({'train_ce': tr_loss, 'test_ce': te_loss,
                        'best_train': best_train, 'best_test': best_test,
                        **{f'readout_{k}': v for k, v in rr.items()}}, step=ep)
            if ep % 100 == 0:
                print(f'    ep {ep:5d}/{CFG["epochs"]}  tr={tr_loss:.4f}  te={te_loss:.4f}  '
                      f'best_tr={best_train:.4f}  best_te={best_test:.4f}  '
                      f'f in [{rr["min"]:.2f},{rr["max"]:.2f}]', flush=True)
            if CFG['early_stop'] and stale >= CFG['patience']:
                print(f'    [early stop] ep={ep}')
                break
    except KeyboardInterrupt:
        print('    [INTERRUPTED] saving best-so-far (Ctrl+C again within ~1s exits)', flush=True)
        try:
            time.sleep(1)
        except KeyboardInterrupt:
            interrupted_twice = True
            print('    [INTERRUPTED x2] finishing this run cleanly, then exiting', flush=True)

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    torch.save(model.state_dict(), cpath)
    rr_end = readout_range(model, te_x, device)
    result = {'cell': tag, 'arch': arch, 'seed': seed, 'threshold': float(thr),
              'history': history, 'best_train': best_train, 'test_at_best': test_at_best,
              'best_test': best_test,
              'final_train': history['train_loss'][-1], 'final_test': history['test_loss'][-1],
              'epochs_run': ep, 'time_s': round(time.time() - t0, 1), 'config': run_cfg,
              'readout_range_best': rr_end}
    with open(mpath, 'w') as fh:
        json.dump(result, fh, indent=2)
    if wb:
        wb.summary.update({'best_train': best_train, 'test_at_best': test_at_best,
                           'best_test': best_test, 'final_train': result['final_train'],
                           'final_test': result['final_test'], 'epochs_run': ep,
                           **{f'readout_final_{k}': v for k, v in rr_end.items()}})
        try:
            wb.finish()
        except Exception as e:
            print(f'    [wandb] finish failed ({type(e).__name__}: {e})', flush=True)
    print(f'    done in {result["time_s"]/60:.1f}m  best_train={best_train:.4f}  '
          f'best_test={best_test:.4f}  -> {mpath}', flush=True)
    if interrupted_twice:
        raise KeyboardInterrupt
    return True


def rebuild_summary():
    rows = []
    for mp in sorted(glob.glob(os.path.join(CFG['runs_dir'], f'{CELL_PREFIX}_*',
                                            'metrics_*.json'))):
        m = json.load(open(mp))
        rows.append({k: m.get(k) for k in
                     ('cell', 'arch', 'seed', 'threshold', 'best_train', 'test_at_best',
                      'best_test', 'final_train', 'final_test', 'epochs_run')})
    out = os.path.join(CFG['runs_dir'], f'{CELL_PREFIX}_summary.json')
    with open(out, 'w') as fh:
        json.dump(rows, fh, indent=2)
    print(f'\nSummary ({len(rows)} runs) -> {out}')


def check_dataset():
    for thr in THRESHOLDS:
        for split in ('train', 'test'):
            p = os.path.join(CFG['data_dir'], split, f'data_x_{thr}.pt')
            if not os.path.exists(p):
                sys.exit(f'missing {p}\nrun: python data_generation.py')
        blob = load_data('test', thr)
        if blob.get('succ_split') != 'equiprobable':
            sys.exit(f'{CFG["data_dir"]}/test/data_x_{thr}.pt is not the equiprobable dataset')
    _DATA.clear()


def main():
    device = torch.device(CFG['device'] if torch.cuda.is_available() else 'cpu')
    check_dataset()
    runs = [(arch, seed, thr) for thr in THRESHOLDS for arch in ARCHS for seed in SEEDS]
    n_done = sum(1 for r in runs if os.path.exists(metrics_path(*r)))

    print(f'device     : {device}')
    print(f'data       : {CFG["data_dir"]}  [equiprobable]')
    print(f'cell       : {cell_tag()}')
    print(f'optimizer  : {CFG["optimizer"]} lr={CFG["lr"]} b={CFG["batch_size"]} '
          f'init={CFG["init"]} epochs={CFG["epochs"]} early_stop={CFG["early_stop"]}')
    print(f'archs      : {ARCHS}   seeds: {SEEDS}   thresholds: {THRESHOLDS}')
    print(f'runs       : {len(runs)} ({n_done} done, will skip)')
    print(f'wandb      : {WANDB_PROJECT if USE_WANDB else "OFF"}\n')

    os.makedirs(CFG['runs_dir'], exist_ok=True)
    os.makedirs(os.path.join(_SCRIPT_DIR, 'wandb_local'), exist_ok=True)
    t_start = time.time(); n_run = 0
    try:
        for i, (arch, seed, thr) in enumerate(runs):
            if train_one(arch, seed, thr, device):
                n_run += 1
                avg = (time.time() - t_start) / n_run
                left = sum(1 for r in runs[i + 1:] if not os.path.exists(metrics_path(*r)))
                print(f'  [{i + 1}/{len(runs)}]  avg={avg/60:.1f}m/run  '
                      f'ETA={left * avg / 3600:.1f}h\n', flush=True)
    except KeyboardInterrupt:
        print('\n[sweep stopped by Ctrl+C]', flush=True)
    rebuild_summary()


if __name__ == '__main__':
    main()
