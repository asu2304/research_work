"""The first 5 epochs at STEP resolution -- one snapshot per optimizer step.

`train_Wov_fixing_rest_all.py` records W_qk once per epoch, which is far too coarse to see what
happens at initialization: by the end of epoch 1 the matrices have already grown ~60% in norm.
This script re-runs the identical setup for 5 epochs and writes W_qk after **every optimizer
step**.

    25,000 train sequences / batch 256 = 98 batches per epoch (the last one holds 168)
    5 epochs x 98 = 490 steps  ->  491 snapshot rows (row 0 is the random init)

Everything else is bit-identical to train_Wov_fixing_rest_all.py: same model, same frozen
circuits, same readout, same hyperparameters, same seed, same dataset.

Reproducing the same trajectory
-------------------------------
The DataLoader's shuffle draws from the global torch RNG, so the batch order depends on how much
RNG was consumed before the loader starts iterating. The construction sequence below --
manual_seed -> build model (the Block constructors call nn.init.uniform_) -> apply_init
(re-seeds, then two normal_ calls) -> install_fixed -> DataLoader -- is copied verbatim from the
epoch-level script so that the RNG is in the identical state and the batch order matches.

If the epoch-level run is on disk, `VERIFY_AGAINST` makes this script check that claim: after
training it compares its step-490k snapshots against rows 1..5 of
`artifacts/<epoch cell>/wqk{1,2}_x_{alpha}.npy` and prints max|diff|, which should be 0 (or
float-noise) if the two runs really followed the same path.

Where things land
-----------------
    runs/<cell>/metrics_combined_seed0_x_{alpha}.json      per-step + per-epoch curves
    runs/<cell>/model_final_combined_seed0_x_{alpha}.pth
    runs/wqkinit_summary.json
    wandb_local/

    artifacts/<cell>/wqk1_steps_x_{alpha}.npy   float32 [491, 108, 108]   <- StepRecorder
    artifacts/<cell>/wqk2_steps_x_{alpha}.npy   float32 [491, 216, 216]   <- StepRecorder
    artifacts/<cell>/stepmeta_x_{alpha}.json    step -> (epoch, batch, batch loss)

Size: 491 rows x (108^2 + 216^2) x 4 B = 112 MB per threshold, 0.33 GB for all three.

Run in tmux:
    python -u inquiring_at_init.py
"""

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
_DATA_DIR = os.path.abspath(os.path.join(
    _SCRIPT_DIR, '..', '14. elimination final convergence confusion', 'dataset'))

CFG = {
    'data_dir': _DATA_DIR,
    'runs_dir': os.path.join(_SCRIPT_DIR, 'runs'),
    'art_dir':  os.path.join(_SCRIPT_DIR, 'artifacts'),
    'device':   'cuda:3',

    'v_sig':       4,
    'n_noise':     4,
    'seq_len':     100,
    'token_start': 50,

    'b_sig':        0.01,
    'b_noise_rule': 'over_alpha',
    'noise_floor':  1e-9,
    'log_floor':    1e-30,

    'optimizer':    'adam',
    'lr':           0.005,
    'batch_size':   256,
    'init':         'normal',
    'weight_decay': 0.0,
    'eval_batch':   256,
    'epochs':       5,             # <- the only hyperparameter that differs

    'snap_dtype':   'float32',
}

ARCH       = 'combined'
SEEDS      = [0]
THRESHOLDS = [0.6, 0.8, 1.0]

CELL_PREFIX   = 'wqkinit'
SKIP_DONE     = True
USE_WANDB     = True
WANDB_PROJECT = 'ICL-Wqk-init-steps'
WANDB_ENTITY  = None

# the epoch-level run to cross-check against; None disables the check
VERIFY_AGAINST = 'wqkevo_opt-adam_lr-0.005_b-256_init-normal_ep-1000'


def cell_tag():
    return (f"{CELL_PREFIX}_opt-{CFG['optimizer']}_lr-{CFG['lr']}_b-{CFG['batch_size']}"
            f"_init-{CFG['init']}_ep-{CFG['epochs']}")


def run_dir():
    return os.path.join(CFG['runs_dir'], cell_tag())


def snap_dir():
    return os.path.join(CFG['art_dir'], cell_tag())


def noise_bias(alpha):
    if alpha >= 1.0:
        return CFG['noise_floor']
    if CFG['b_noise_rule'] == 'plain':
        return (1.0 - alpha) / CFG['n_noise']
    return (1.0 - alpha) * (1.0 + CFG['v_sig'] * CFG['b_sig']) / (CFG['n_noise'] * alpha)


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


class QKOnlyTransformer(nn.Module):
    def __init__(self, total_vocab, seq_len, attn_dim, v_sig, alpha):
        super().__init__()
        self.total_vocab = total_vocab
        self.seq_len     = seq_len
        self.v_sig       = v_sig
        self.attn_param  = 'combined'
        self.initial_dim = total_vocab + seq_len
        self.layer1 = CombinedAttentionBlock(self.initial_dim, attn_dim)
        self.layer2 = CombinedAttentionBlock(self.initial_dim + attn_dim, attn_dim)
        self.D   = self.initial_dim + 2 * attn_dim
        self.in2 = self.initial_dim + attn_dim
        self.fc_out = nn.Linear(self.D, total_vocab, bias=False)
        b = torch.full((total_vocab,), CFG['b_sig'])
        b[v_sig:] = noise_bias(alpha)
        self.register_buffer('readout_bias', b)

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
        sl = slice(self.in2, self.in2 + self.total_vocab)
        z = s.clone()
        z[..., sl] = torch.log((s[..., sl] + self.readout_bias).clamp_min(CFG['log_floor']))
        return self.fc_out(z)


def qk_params(model):
    return [model.layer1.W_qk, model.layer2.W_qk]


def install_fixed(model):
    V = model.total_vocab
    dev = next(model.parameters()).device
    with torch.no_grad():
        for layer in (model.layer1, model.layer2):
            layer.W_ov.zero_()
            layer.W_ov[:V, :V] = torch.eye(V, device=dev)
        model.fc_out.weight.zero_()
        for v in range(V):
            model.fc_out.weight[v, model.in2 + v] = 1.0
    for p in model.parameters():
        p.requires_grad_(False)
    ps = qk_params(model)
    for p in ps:
        p.requires_grad_(True)
    return ps


def apply_init(model, seed):
    torch.manual_seed(seed)
    with torch.no_grad():
        for layer in (model.layer1, model.layer2):
            layer.W_qk.normal_(0.0, 1.0 / math.sqrt(layer.W_qk.shape[0]))


class StepRecorder:
    """W_qk for both layers, one row per optimizer step. Row 0 is the random init.

    Flushed every 25 rows and on close, so an interrupted run still leaves a readable file;
    `stepmeta` records how many rows are valid and what each row is.
    """

    def __init__(self, model, thr, n_rows, dtype, meta):
        d = snap_dir()
        os.makedirs(d, exist_ok=True)
        self.paths = [os.path.join(d, f'wqk1_steps_x_{thr}.npy'),
                      os.path.join(d, f'wqk2_steps_x_{thr}.npy')]
        self.meta_path = os.path.join(d, f'stepmeta_x_{thr}.json')
        shapes = [tuple(p.shape) for p in qk_params(model)]
        self.arrs = [np.lib.format.open_memmap(p, mode='w+', dtype=np.dtype(dtype),
                                               shape=(n_rows,) + s)
                     for p, s in zip(self.paths, shapes)]
        self.meta = {**meta, 'shapes': [list(s) for s in shapes], 'dtype': dtype,
                     'n_rows_allocated': n_rows, 'rows_written': 0,
                     'step': [], 'epoch': [], 'batch': [], 'batch_loss': []}
        self.n = 0

    def write(self, model, step, epoch, batch, loss):
        if self.n >= len(self.arrs[0]):
            return
        for arr, p in zip(self.arrs, qk_params(model)):
            arr[self.n] = p.detach().to('cpu', torch.float32).numpy()
        self.meta['step'].append(int(step))
        self.meta['epoch'].append(int(epoch))
        self.meta['batch'].append(int(batch))
        self.meta['batch_loss'].append(None if loss is None else float(loss))
        self.n += 1
        self.meta['rows_written'] = self.n
        if self.n % 25 == 0:
            self.flush()

    def flush(self):
        for a in self.arrs:
            a.flush()
        with open(self.meta_path, 'w') as fh:
            json.dump(self.meta, fh, indent=2)

    def close(self):
        self.flush()
        self.arrs = []

    def nbytes(self):
        return sum(a.nbytes for a in self.arrs)


_DATA = {}


def load_data(split, thr):
    key = (split, thr)
    if key not in _DATA:
        if any(k[1] != thr for k in _DATA):
            _DATA.clear()
        _DATA[key] = torch.load(os.path.join(CFG['data_dir'], split, f'data_x_{thr}.pt'),
                                weights_only=False)
    return _DATA[key]


@torch.no_grad()
def eval_ce(model, x, y, V, device):
    model.eval()
    ts, te = CFG['token_start'], CFG['seq_len']
    total, n = 0.0, 0
    for i in range(0, x.shape[0], CFG['eval_batch']):
        xb = x[i:i + CFG['eval_batch']].to(device)
        yb = y[i:i + CFG['eval_batch'], ts:te].to(device)
        total += F.cross_entropy(model(xb)[:, ts:te].reshape(-1, V), yb.reshape(-1),
                                 reduction='sum').item()
        n += yb.numel()
    model.train()
    return total / n


@torch.no_grad()
def slot_stats(model, x, device, batch=256, n_max=2048):
    was = model.training
    model.eval()
    V, VS = model.total_vocab, model.v_sig
    ts, te = CFG['token_start'], CFG['seq_len']
    sig = noi = zer = 0.0
    n = 0
    for i in range(0, min(x.shape[0], n_max), batch):
        f = model.stream(x[i:i + batch].to(device))[:, ts:te, model.in2:model.in2 + V]
        sig += float(f[..., :VS].sum(-1).sum())
        noi += float(f[..., VS:].sum(-1).sum())
        zer += float((f[..., :VS] == 0).float().sum())
        n += f.shape[0] * f.shape[1]
    if was:
        model.train()
    return {'signal_mass': sig / n, 'noise_mass': noi / n, 'frac_zero': zer / (n * VS)}


@torch.no_grad()
def qk_stats(model):
    V, VS = model.total_vocab, model.v_sig
    D = model.initial_dim
    W1, W2 = model.layer1.W_qk.detach(), model.layer2.W_qk.detach()
    s = W2[:V, D:D + V][:VS, :VS]
    dg = s.diagonal().abs().mean()
    off = (s - torch.diag(s.diagonal())).abs().sum() / (s.numel() - VS)
    return {'wqk1_fro': float(W1.norm()), 'wqk2_fro': float(W2.norm()),
            'induction_diag_off': float(dg / off.clamp_min(1e-30)),
            'induction_diag': float(dg)}


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
            import wandb
            try:
                api = wandb.Api()
                who = f'username={api.viewer.username} default_entity={api.default_entity}'
            except Exception:
                who = 'could not query wandb.Api()'
            print(f'  [wandb] disabled ({type(e).__name__}: {e})\n'
                  f'          {who}  attempted project={WANDB_PROJECT} entity={WANDB_ENTITY}')
        _WANDB_OK = False
        return None


def metrics_path(seed, thr):
    return os.path.join(run_dir(), f'metrics_{ARCH}_seed{seed}_x_{thr}.json')


def verify_against_epoch_run(thr, rec, epoch_rows):
    """Compare the epoch-boundary step snapshots to the epoch-level run's own snapshots.

    Both runs claim to be the same trajectory; if the RNG replay is right the two agree exactly.
    """
    if not VERIFY_AGAINST:
        return None
    d = os.path.join(CFG['art_dir'], VERIFY_AGAINST)
    p1 = os.path.join(d, f'wqk1_x_{thr}.npy')
    p2 = os.path.join(d, f'wqk2_x_{thr}.npy')
    if not (os.path.exists(p1) and os.path.exists(p2)):
        print(f'    [verify] {VERIFY_AGAINST} has no snapshots for alpha={thr}, skipping')
        return None
    A1 = np.load(p1, mmap_mode='r'); A2 = np.load(p2, mmap_mode='r')
    S1 = np.load(rec.paths[0], mmap_mode='r'); S2 = np.load(rec.paths[1], mmap_mode='r')
    out = {}
    for ep, row in epoch_rows.items():
        if ep >= A1.shape[0] or row >= rec.meta['rows_written']:
            continue
        d1 = float(np.abs(np.asarray(S1[row], dtype=np.float64) -
                          np.asarray(A1[ep], dtype=np.float64)).max())
        d2 = float(np.abs(np.asarray(S2[row], dtype=np.float64) -
                          np.asarray(A2[ep], dtype=np.float64)).max())
        out[ep] = (d1, d2)
        print(f'    [verify] epoch {ep}: max|diff| vs {VERIFY_AGAINST}  '
              f'W_qk1 {d1:.3e}   W_qk2 {d2:.3e}')
    return out


def train_one(seed, thr, device):
    V = CFG['v_sig'] + CFG['n_noise']
    ts, tend = CFG['token_start'], CFG['seq_len']
    tag = cell_tag()
    d = run_dir()
    os.makedirs(d, exist_ok=True)
    os.makedirs(snap_dir(), exist_ok=True)
    stem = f'{ARCH}_seed{seed}_x_{thr}'
    mpath = os.path.join(d, f'metrics_{stem}.json')
    if SKIP_DONE and os.path.exists(mpath):
        print(f'  [skip done] {stem}')
        return False

    b_n = noise_bias(thr)

    # --- identical construction order to train_Wov_fixing_rest_all.py (see the docstring) ---
    torch.manual_seed(seed); np.random.seed(seed)
    tr_blob, te_blob = load_data('train', thr), load_data('test', thr)
    tr_x, tr_y = tr_blob['x'], tr_blob['y']
    te_x, te_y = te_blob['x'], te_blob['y']

    model = QKOnlyTransformer(V, CFG['seq_len'], V + CFG['seq_len'], CFG['v_sig'],
                              thr).to(device)
    apply_init(model, seed)
    params = install_fixed(model)
    opt = torch.optim.Adam(params, lr=CFG['lr'], weight_decay=CFG['weight_decay'])

    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(tr_x, tr_y), batch_size=CFG['batch_size'], shuffle=True)
    # ---------------------------------------------------------------------------------------

    steps_per_epoch = len(loader)
    n_rows = CFG['epochs'] * steps_per_epoch + 1
    run_cfg = {**{k: v for k, v in CFG.items()
                  if k not in ('runs_dir', 'art_dir', 'data_dir')},
               'attn_param': ARCH, 'seed': seed, 'threshold': thr, 'cell': tag,
               'b_noise': b_n, 'trainable': 'W_qk layer1 + layer2 only',
               'succ_split': 'equiprobable', 'steps_per_epoch': steps_per_epoch,
               'snap_rows': n_rows, 'snapshot_granularity': 'every optimizer step'}

    wb = wandb_start(run_cfg, name=f'{tag}__{stem}', group=f'thr{thr}',
                     tags=[ARCH, f'thr{thr}', f'seed{seed}', 'qk-only', 'step-level', 'init'])

    rec = StepRecorder(model, thr, n_rows, CFG['snap_dtype'],
                       {'cell': tag, 'arch': ARCH, 'seed': seed, 'threshold': float(thr),
                        'steps_per_epoch': steps_per_epoch,
                        'row0': 'random init, before any step',
                        'row_meaning': 'row s>0 holds W_qk AFTER optimizer step s'})
    rec.write(model, 0, 0, -1, None)          # row 0 = the random initialization

    tr0 = eval_ce(model, tr_x, tr_y, V, device)
    te0 = eval_ce(model, te_x, te_y, V, device)
    st0 = slot_stats(model, te_x, device)
    qs0 = qk_stats(model)
    history = {'epoch': [0], 'train_loss': [tr0], 'test_loss': [te0],
               'signal_mass': [st0['signal_mass']], 'frac_zero': [st0['frac_zero']],
               'wqk1_fro': [qs0['wqk1_fro']], 'wqk2_fro': [qs0['wqk2_fro']],
               'induction_diag_off': [qs0['induction_diag_off']],
               'step': [0], 'step_loss': [], 'step_wqk1_fro': [qs0['wqk1_fro']],
               'step_wqk2_fro': [qs0['wqk2_fro']]}
    if wb:
        for m in ('train_ce', 'test_ce'):
            wb.define_metric(m, summary='min')
        for m in ('best_train', 'best_test'):
            wb.define_metric(m, summary='last')
        wb.log({'train_ce': tr0, 'test_ce': te0, 'best_train': tr0, 'best_test': te0,
                **st0, **qs0}, step=0)

    print(f'  {stem}  trainable={sum(p.numel() for p in params):,}  '
          f'b_sig={CFG["b_sig"]}  b_noise={b_n:.6f}')
    print(f'    {steps_per_epoch} steps/epoch x {CFG["epochs"]} epochs = '
          f'{n_rows - 1} steps -> {n_rows} rows, {rec.nbytes()/2**20:.0f} MB  -> {snap_dir()}')
    print(f'    init (row 0): tr={tr0:.4f} te={te0:.4f}  '
          f'|Wqk1|={qs0["wqk1_fro"]:.3f} |Wqk2|={qs0["wqk2_fro"]:.3f}', flush=True)

    best_train = float('inf'); test_at_best = None
    best_test = te0
    epoch_rows = {}
    t0 = time.time(); gstep = 0; ep = 0
    interrupted_twice = False
    try:
        for ep in range(1, CFG['epochs'] + 1):
            model.train()
            ep_loss, ep_n = 0.0, 0
            for bi, (xb, yb) in enumerate(loader):
                xb = xb.to(device); yb = yb[:, ts:tend].to(device)
                opt.zero_grad()
                loss = F.cross_entropy(model(xb)[:, ts:tend].reshape(-1, V), yb.reshape(-1))
                loss.backward()
                opt.step()
                gstep += 1
                lv = loss.item()
                rec.write(model, gstep, ep, bi, lv)
                qs_s = qk_stats(model)
                history['step'].append(gstep)
                history['step_loss'].append(lv)
                history['step_wqk1_fro'].append(qs_s['wqk1_fro'])
                history['step_wqk2_fro'].append(qs_s['wqk2_fro'])
                if wb:
                    wb.log({'batch_loss': lv, 'step_wqk1_fro': qs_s['wqk1_fro'],
                            'step_wqk2_fro': qs_s['wqk2_fro'],
                            'step_induction_diag_off': qs_s['induction_diag_off']},
                           step=gstep)
                ep_loss += lv * yb.numel(); ep_n += yb.numel()

            epoch_rows[ep] = gstep
            tr_loss = ep_loss / ep_n
            te_loss = eval_ce(model, te_x, te_y, V, device)
            st = slot_stats(model, te_x, device)
            qs = qk_stats(model)
            history['epoch'].append(ep)
            history['train_loss'].append(tr_loss)
            history['test_loss'].append(te_loss)
            history['signal_mass'].append(st['signal_mass'])
            history['frac_zero'].append(st['frac_zero'])
            history['wqk1_fro'].append(qs['wqk1_fro'])
            history['wqk2_fro'].append(qs['wqk2_fro'])
            history['induction_diag_off'].append(qs['induction_diag_off'])
            best_train = min(best_train, tr_loss)
            if best_train == tr_loss:
                test_at_best = te_loss
            best_test = min(best_test, te_loss)
            if wb:
                wb.log({'train_ce': tr_loss, 'test_ce': te_loss, 'best_train': best_train,
                        'best_test': best_test, **st, **qs}, step=gstep)
            print(f'    ep {ep}/{CFG["epochs"]}  step {gstep}  tr={tr_loss:.4f}  '
                  f'te={te_loss:.4f}  |Wqk1|={qs["wqk1_fro"]:.2f}  '
                  f'|Wqk2|={qs["wqk2_fro"]:.2f}  ind_d/o={qs["induction_diag_off"]:.2f}',
                  flush=True)
    except KeyboardInterrupt:
        print('    [INTERRUPTED] flushing snapshots (Ctrl+C again within ~1s exits)', flush=True)
        try:
            time.sleep(1)
        except KeyboardInterrupt:
            interrupted_twice = True
            print('    [INTERRUPTED x2] finishing this run cleanly, then exiting', flush=True)

    torch.save(model.state_dict(), os.path.join(d, f'model_final_{stem}.pth'))
    st_end = slot_stats(model, te_x, device)
    qs_end = qk_stats(model)
    rec.close()
    ver = verify_against_epoch_run(thr, rec, epoch_rows)

    result = {'cell': tag, 'arch': ARCH, 'seed': seed, 'threshold': float(thr),
              'history': history, 'best_train': best_train, 'test_at_best': test_at_best,
              'best_test': best_test,
              'final_train': history['train_loss'][-1], 'final_test': history['test_loss'][-1],
              'epochs_run': ep, 'steps_run': gstep, 'steps_per_epoch': steps_per_epoch,
              'time_s': round(time.time() - t0, 1), 'config': run_cfg,
              'slot_stats_final': st_end, 'qk_stats_final': qs_end,
              'epoch_rows': epoch_rows,
              'verify_vs_epoch_run': ({str(k): v for k, v in ver.items()} if ver else None),
              'snapshots': {'dir': os.path.relpath(snap_dir(), _SCRIPT_DIR),
                            'wqk1': f'wqk1_steps_x_{thr}.npy',
                            'wqk2': f'wqk2_steps_x_{thr}.npy',
                            'rows_written': rec.meta['rows_written'],
                            'row0': 'random init, before any step'}}
    with open(mpath, 'w') as fh:
        json.dump(result, fh, indent=2)
    if wb:
        wb.summary.update({'best_train': best_train, 'test_at_best': test_at_best,
                           'best_test': best_test, 'final_train': result['final_train'],
                           'final_test': result['final_test'], 'epochs_run': ep,
                           'steps_run': gstep,
                           'snap_rows_written': rec.meta['rows_written'],
                           **{f'final_{k}': v for k, v in {**st_end, **qs_end}.items()}})
        try:
            wb.finish()
        except Exception as e:
            print(f'    [wandb] finish failed ({type(e).__name__}: {e})', flush=True)
    print(f'    done in {result["time_s"]/60:.1f}m  {gstep} steps  '
          f'final_train={result["final_train"]:.4f}  final_test={result["final_test"]:.4f}  '
          f'-> {mpath}', flush=True)
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
                      'best_test', 'final_train', 'final_test', 'epochs_run', 'steps_run')})
    out = os.path.join(CFG['runs_dir'], f'{CELL_PREFIX}_summary.json')
    with open(out, 'w') as fh:
        json.dump(rows, fh, indent=2)
    print(f'\nSummary ({len(rows)} runs) -> {out}')


def check_dataset():
    for thr in THRESHOLDS:
        for split in ('train', 'test'):
            p = os.path.join(CFG['data_dir'], split, f'data_x_{thr}.pt')
            if not os.path.exists(p):
                sys.exit(f'missing {p}\ngenerate it in folder 14: python data_generation.py')
        if load_data('test', thr).get('succ_split') != 'equiprobable':
            sys.exit(f'{CFG["data_dir"]}/test/data_x_{thr}.pt is not the equiprobable dataset')
    _DATA.clear()


def main():
    device = torch.device(CFG['device'] if torch.cuda.is_available() else 'cpu')
    check_dataset()
    runs = [(seed, thr) for thr in THRESHOLDS for seed in SEEDS]
    n_done = sum(1 for r in runs if os.path.exists(metrics_path(*r)))

    V = CFG['v_sig'] + CFG['n_noise']
    n_train = load_data('train', THRESHOLDS[0])['x'].shape[0]
    _DATA.clear()
    spe = math.ceil(n_train / CFG['batch_size'])
    n_rows = CFG['epochs'] * spe + 1
    probe = QKOnlyTransformer(V, CFG['seq_len'], V + CFG['seq_len'], CFG['v_sig'], 0.6)
    n_tr = sum(p.numel() for p in install_fixed(probe))
    n_all = sum(p.numel() for p in probe.parameters())
    mb = n_rows * (probe.layer1.W_qk.numel() + probe.layer2.W_qk.numel()) * \
        np.dtype(CFG['snap_dtype']).itemsize / 2 ** 20
    del probe

    print(f'device     : {device}')
    print(f'data       : {CFG["data_dir"]}  [equiprobable]  {n_train:,} train sequences')
    print(f'cell       : {cell_tag()}')
    print(f'runs       : {run_dir()}')
    print(f'snapshots  : {snap_dir()}')
    print(f'model      : W_qk layer1 + layer2 trainable ({n_tr:,} of {n_all:,} params); '
          f'W_ov + unembedding frozen to the construction')
    print(f'readout    : z = log(f + b), bias BEFORE the log, no FFN, no residual')
    for t in THRESHOLDS:
        print(f'               alpha={t}: b_noise = {noise_bias(t):.6f}')
    print(f'recorder   : W_qk both layers after EVERY optimizer step')
    print(f'             {n_train:,}/{CFG["batch_size"]} -> {spe} steps/epoch x '
          f'{CFG["epochs"]} epochs = {n_rows - 1} steps, {n_rows} rows')
    print(f'             {mb:.0f} MB/run, {mb*len(runs)/1024:.2f} GB total')
    print(f'optimizer  : {CFG["optimizer"]} lr={CFG["lr"]} b={CFG["batch_size"]} '
          f'init={CFG["init"]} epochs={CFG["epochs"]}')
    print(f'arch       : {ARCH} only   thresholds: {THRESHOLDS}   seeds: {SEEDS}')
    print(f'verify vs  : {VERIFY_AGAINST or "OFF"}')
    print(f'to run     : {len(runs)} ({n_done} done, will skip)')
    print(f'wandb      : {WANDB_PROJECT if USE_WANDB else "OFF"}\n')

    os.makedirs(run_dir(), exist_ok=True)
    os.makedirs(snap_dir(), exist_ok=True)
    os.makedirs(os.path.join(_SCRIPT_DIR, 'wandb_local'), exist_ok=True)
    t_start = time.time(); n_run = 0
    try:
        for i, (seed, thr) in enumerate(runs):
            if train_one(seed, thr, device):
                n_run += 1
                avg = (time.time() - t_start) / n_run
                left = sum(1 for r in runs[i + 1:] if not os.path.exists(metrics_path(*r)))
                print(f'  [{i + 1}/{len(runs)}]  avg={avg/60:.1f}m/run  '
                      f'ETA={left * avg / 60:.1f}m\n', flush=True)
    except KeyboardInterrupt:
        print('\n[sweep stopped by Ctrl+C]', flush=True)
    rebuild_summary()


if __name__ == '__main__':
    main()
