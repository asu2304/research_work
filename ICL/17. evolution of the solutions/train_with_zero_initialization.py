"""W_qk evolution from a ZERO initialization.

Identical to `train_Wov_fixing_rest_all.py` in every respect -- same model, same frozen
circuits, same readout, same hyperparameters, same snapshotting -- EXCEPT that W_qk for both
layers starts at exactly zero instead of N(0, 1/sqrt(fan_in)).

    W_qk for BOTH layers is written to disk after every epoch, starting from the
    ZERO initialization (row 0 = the all-zero matrix, before any gradient step).

The question: with no random asymmetry to exploit, does gradient descent still build the
induction circuit, and does it land where the random-init run landed?
Compare against  wqkevo_opt-adam_lr-0.005_b-256_init-normal_ep-1000  (identical but for the
init); the analysis lives in `zero_init_analysis.ipynb`.

READ THIS BEFORE INTERPRETING THE FIRST EPOCH
---------------------------------------------
At W_qk = 0 the layer-1 gradient is EXACTLY zero. Measured, not feared:

    ||dL/dW_qk1|| = 0.000000e+00        ||dL/dW_qk2|| = 3.83e-03

The reason is structural. Layer 1's output can reach the loss by two routes and at step 0
both are shut:
  * the VALUE route -- W_ov2 reads rows 0:8 (the position's own token) and never rows
    108:116, so layer 1's output is not in the value path at all;
  * the SCORE route -- W_qk2 = 0, so layer 1's output contributes nothing to any score.
No path to the loss means no gradient.

This is NOT a permanent dead start. As soon as W_qk2 is non-zero -- one Adam step -- layer 1
reconnects and its gradient switches on. Measured over the first steps:

    step 0: ||g1|| = 0.00e+00   step 1: 1.66e-06   step 2: 3.42e-06   step 5: 8.05e-06

So layer 1 is frozen for exactly one step, then trains normally. Rows 0 and 1 of the layer-1
snapshots are therefore identical by construction -- expected, not a bug. (For contrast the
random init has ||g1|| = 1.18e-05 already at step 0.)

What is frozen (identical to train_qk_only.py)
----------------------------------------------
  * W_ov, both layers  -> I_V block  (copy the attended position's token one-hot)
  * fc_out             -> identity selector on the eight layer-2 count slots, no bias
  * the readout        -> log() applied directly, bias added BEFORE the log,
                          no FFN, no residual across the log
  * only W_qk layer1 (108x108) and W_qk layer2 (216x216) train: 58,320 params

Readout bias (unchanged, see train_qk_only.py for the derivation)
-----------------------------------------------------------------
    z_v = log(f_v + b_v)  ->  softmax  ->  p(v) = (f_v + b_v) / sum_u (f_u + b_u)
    b_sig   = 0.01
    b_noise = (1 - alpha)(1 + 4 b_sig) / (4 alpha)      [NOT (1-alpha)/4]

Where things land (folder-14 layout, plus the snapshots)
--------------------------------------------------------
    runs/<cell>/metrics_combined_seed0_x_{alpha}.json      curves + config
    runs/<cell>/model_final_combined_seed0_x_{alpha}.pth   endpoint of the trajectory
    runs/<cell>/model_best_combined_seed0_x_{alpha}.pth    best train CE
    runs/wqkevo_summary.json
    wandb_local/                                           wandb run dirs

    artifacts/<cell>/wqk1_x_{alpha}.npy   float32 [epochs+1, 108, 108]   <- QKRecorder
    artifacts/<cell>/wqk2_x_{alpha}.npy   float32 [epochs+1, 216, 216]   <- QKRecorder
    artifacts/<cell>/snapmeta_x_{alpha}.json

Row 0 is the random init; row e is the matrix at the END of epoch e. Written through a
np.lib.format.open_memmap and flushed every epoch, so a Ctrl+C'd run leaves a readable file
(`snapmeta` records how many rows are valid). Size: 233 MB per threshold at 1000 epochs,
~0.7 GB for all three.

Curves
------
history['epoch'] starts at 0 and history[...][0] is measured at the zero init, BEFORE the
first optimizer step -- so every plotted curve starts from the initialization, not from the
first epoch. wandb gets the same value at step 0.

Run in tmux:
    python -u train_with_zero_initialization.py
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
    
    'data_dir': _DATA_DIR,                                  # the equiprobable dataset
    'runs_dir': os.path.join(_SCRIPT_DIR, 'runs'),          # metrics + checkpoints
    'art_dir':  os.path.join(_SCRIPT_DIR, 'artifacts'),     # extra artifacts (the snapshots)
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
    'init':         'zeros',
    'weight_decay': 0.0,
    'eval_batch':   256,
    'epochs':       1000,

    'snap_every':   1,             # epochs between snapshots; 1 = every epoch
    'snap_dtype':   'float32',
}

ARCH       = 'combined'
SEEDS      = [0]
THRESHOLDS = [0.6, 0.8, 1.0]

CELL_PREFIX   = 'wqkzero'
SKIP_DONE     = True
USE_WANDB     = True
WANDB_PROJECT = 'ICL-Wqk-zero-init'
WANDB_ENTITY  = None


# the random-init twin, for the side-by-side in zero_init_analysis.ipynb
COMPARE_WITH  = 'wqkevo_opt-adam_lr-0.005_b-256_init-normal_ep-1000'


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
    """W_ov <- I_V block (both layers); fc_out <- identity selector. Freeze everything, then
    re-enable the two W_qk matrices."""
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
    """ZERO on both W_qk matrices -- the one change this script makes.

    `seed` is still consumed and torch.manual_seed still called, so the RNG is left in exactly
    the state the random-init script leaves it in and the DataLoader shuffle draws the same
    batch order. Only the starting weights differ between the two runs.
    """
    torch.manual_seed(seed)
    with torch.no_grad():
        for layer in (model.layer1, model.layer2):
            layer.W_qk.zero_()


class QKRecorder:
    """Writes W_qk layer1/layer2 to two .npy memmaps, one row per snapshot.

    Row 0 is the random initialization. `flush(meta)` after every write keeps the files
    readable if the run is interrupted.
    """

    def __init__(self, model, thr, n_rows, dtype, meta):
        d = snap_dir()
        os.makedirs(d, exist_ok=True)
        self.paths = [os.path.join(d, f'wqk1_x_{thr}.npy'),
                      os.path.join(d, f'wqk2_x_{thr}.npy')]
        self.meta_path = os.path.join(d, f'snapmeta_x_{thr}.json')
        shapes = [tuple(p.shape) for p in qk_params(model)]
        self.arrs = [np.lib.format.open_memmap(p, mode='w+', dtype=np.dtype(dtype),
                                               shape=(n_rows,) + s)
                     for p, s in zip(self.paths, shapes)]
        self.meta = {**meta, 'shapes': [list(s) for s in shapes], 'dtype': dtype,
                     'n_rows_allocated': n_rows, 'rows_written': 0, 'epochs': []}
        self.n = 0

    def write(self, model, epoch):
        if self.n >= len(self.arrs[0]):
            return
        for arr, p in zip(self.arrs, qk_params(model)):
            arr[self.n] = p.detach().to('cpu', torch.float32).numpy()
            arr.flush()
        self.meta['epochs'].append(int(epoch))
        self.n += 1
        self.meta['rows_written'] = self.n
        with open(self.meta_path, 'w') as fh:
            json.dump(self.meta, fh, indent=2)

    def close(self):
        for a in self.arrs:
            a.flush()
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
    """signal_mass -> 1 and frac_zero -> ~0.27 is the signature of the learned noise filter."""
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
    """Scalars that summarise where the two W_qk matrices are, logged every epoch so the
    trajectory is visible live as well as in the snapshots."""
    V, VS = model.total_vocab, model.v_sig
    D = model.initial_dim
    W1, W2 = model.layer1.W_qk.detach(), model.layer2.W_qk.detach()
    blk = W2[:V, D:D + V]
    s = blk[:VS, :VS]
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
                  f'          {who}  attempted project={WANDB_PROJECT} '
                  f'entity={WANDB_ENTITY}')
        _WANDB_OK = False
        return None


def metrics_path(seed, thr):
    return os.path.join(run_dir(), f'metrics_{ARCH}_seed{seed}_x_{thr}.json')


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
    n_rows = CFG['epochs'] // CFG['snap_every'] + 1
    run_cfg = {**{k: v for k, v in CFG.items()
                  if k not in ('runs_dir', 'art_dir', 'data_dir')},
               'attn_param': ARCH, 'seed': seed, 'threshold': thr, 'cell': tag,
               'b_noise': b_n, 'trainable': 'W_qk layer1 + layer2 only',
               'succ_split': 'equiprobable', 'snap_rows': n_rows}

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

    wb = wandb_start(run_cfg, name=f'{tag}__{stem}', group=f'thr{thr}',
                     tags=[ARCH, f'thr{thr}', f'seed{seed}', 'qk-only', 'wqk-evolution'])

    rec = QKRecorder(model, thr, n_rows, CFG['snap_dtype'],
                     {'cell': tag, 'arch': ARCH, 'seed': seed, 'threshold': float(thr),
                      'snap_every': CFG['snap_every'], 'row0': 'random init, before any step'})
    rec.write(model, 0)                       # row 0 = the random initialization

    # every curve starts at the initialization, measured before the first optimizer step
    tr0 = eval_ce(model, tr_x, tr_y, V, device)
    te0 = eval_ce(model, te_x, te_y, V, device)
    st0 = slot_stats(model, te_x, device)
    qs0 = qk_stats(model)
    history = {'epoch': [0], 'train_loss': [tr0], 'test_loss': [te0],
               'signal_mass': [st0['signal_mass']], 'frac_zero': [st0['frac_zero']],
               'wqk1_fro': [qs0['wqk1_fro']], 'wqk2_fro': [qs0['wqk2_fro']],
               'induction_diag_off': [qs0['induction_diag_off']]}
    if wb:
        for m in ('train_ce', 'test_ce'):
            wb.define_metric(m, summary='min')
        for m in ('best_train', 'best_test'):
            wb.define_metric(m, summary='last')
        wb.log({'train_ce': tr0, 'test_ce': te0, 'best_train': tr0, 'best_test': te0,
                **st0, **qs0}, step=0)

    print(f'  {stem}  trainable={sum(p.numel() for p in params):,}  '
          f'b_sig={CFG["b_sig"]}  b_noise={b_n:.6f}')
    print(f'    snapshots: {n_rows} rows x (108x108 + 216x216) {CFG["snap_dtype"]} '
          f'= {rec.nbytes()/2**20:.0f} MB  -> {snap_dir()}')
    assert qs0['wqk1_fro'] == 0.0 and qs0['wqk2_fro'] == 0.0, \
        f'zero init failed: |Wqk1|={qs0["wqk1_fro"]} |Wqk2|={qs0["wqk2_fro"]}'
    print(f'    init (row 0): tr={tr0:.4f} te={te0:.4f}  '
          f'signal_mass={st0["signal_mass"]:.4f} (target 1.0)  '
          f'|Wqk1|=0.000 |Wqk2|=0.000  [both exactly zero, as intended]', flush=True)

    best_train = float('inf'); test_at_best = None
    best_test = te0
    best_state = None
    t0 = time.time(); ep = 0
    interrupted_twice = False
    try:
        for ep in range(1, CFG['epochs'] + 1):
            model.train()
            ep_loss, ep_n = 0.0, 0
            for xb, yb in loader:
                xb = xb.to(device); yb = yb[:, ts:tend].to(device)
                opt.zero_grad()
                loss = F.cross_entropy(model(xb)[:, ts:tend].reshape(-1, V), yb.reshape(-1))
                loss.backward()
                opt.step()
                ep_loss += loss.item() * yb.numel(); ep_n += yb.numel()
            tr_loss = ep_loss / ep_n
            te_loss = eval_ce(model, te_x, te_y, V, device)
            st = slot_stats(model, te_x, device)
            qs = qk_stats(model)

            if ep % CFG['snap_every'] == 0:
                rec.write(model, ep)

            history['epoch'].append(ep)
            history['train_loss'].append(tr_loss)
            history['test_loss'].append(te_loss)
            history['signal_mass'].append(st['signal_mass'])
            history['frac_zero'].append(st['frac_zero'])
            history['wqk1_fro'].append(qs['wqk1_fro'])
            history['wqk2_fro'].append(qs['wqk2_fro'])
            history['induction_diag_off'].append(qs['induction_diag_off'])

            if tr_loss < best_train - 1e-6:
                best_train = tr_loss; test_at_best = te_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_test = min(best_test, te_loss)

            if wb:
                wb.log({'train_ce': tr_loss, 'test_ce': te_loss, 'best_train': best_train,
                        'best_test': best_test, **st, **qs}, step=ep)
            if ep % 100 == 0:
                print(f'    ep {ep:5d}/{CFG["epochs"]}  tr={tr_loss:.4f}  te={te_loss:.4f}  '
                      f'best_tr={best_train:.4f}  best_te={best_test:.4f}  '
                      f'sig_mass={st["signal_mass"]:.4f}  '
                      f'ind_d/o={qs["induction_diag_off"]:.2f}', flush=True)
    except KeyboardInterrupt:
        print('    [INTERRUPTED] saving best-so-far (Ctrl+C again within ~1s exits)', flush=True)
        try:
            time.sleep(1)
        except KeyboardInterrupt:
            interrupted_twice = True
            print('    [INTERRUPTED x2] finishing this run cleanly, then exiting', flush=True)

    # the LAST state -- this is the endpoint the snapshot trajectory arrives at
    torch.save(model.state_dict(), os.path.join(d, f'model_final_{stem}.pth'))
    st_end = slot_stats(model, te_x, device)
    qs_end = qk_stats(model)
    rec.close()

    # and the best-train state, for comparability with the folder-14 runs
    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        torch.save(model.state_dict(), os.path.join(d, f'model_best_{stem}.pth'))

    result = {'cell': tag, 'arch': ARCH, 'seed': seed, 'threshold': float(thr),
              'history': history, 'best_train': best_train, 'test_at_best': test_at_best,
              'best_test': best_test,
              'final_train': history['train_loss'][-1], 'final_test': history['test_loss'][-1],
              'epochs_run': ep, 'time_s': round(time.time() - t0, 1), 'config': run_cfg,
              'slot_stats_final': st_end, 'qk_stats_final': qs_end,
              'snapshots': {'dir': os.path.relpath(snap_dir(), _SCRIPT_DIR),
                            'wqk1': f'wqk1_x_{thr}.npy', 'wqk2': f'wqk2_x_{thr}.npy',
                            'rows_written': rec.meta['rows_written'],
                            'row0': 'random init, before any step'}}
    with open(mpath, 'w') as fh:
        json.dump(result, fh, indent=2)
    if wb:
        wb.summary.update({'best_train': best_train, 'test_at_best': test_at_best,
                           'best_test': best_test, 'final_train': result['final_train'],
                           'final_test': result['final_test'], 'epochs_run': ep,
                           'snap_rows_written': rec.meta['rows_written'],
                           **{f'final_{k}': v for k, v in {**st_end, **qs_end}.items()}})
        try:
            wb.finish()
        except Exception as e:
            print(f'    [wandb] finish failed ({type(e).__name__}: {e})', flush=True)
    print(f'    done in {result["time_s"]/60:.1f}m  best_train={best_train:.4f}  '
          f'best_test={best_test:.4f}  snapshots={rec.meta["rows_written"]}  -> {mpath}',
          flush=True)
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
    probe = QKOnlyTransformer(V, CFG['seq_len'], V + CFG['seq_len'], CFG['v_sig'], 0.6)
    n_tr = sum(p.numel() for p in install_fixed(probe))
    n_all = sum(p.numel() for p in probe.parameters())
    n_rows = CFG['epochs'] // CFG['snap_every'] + 1
    mb = n_rows * (probe.layer1.W_qk.numel() + probe.layer2.W_qk.numel()) * \
        np.dtype(CFG['snap_dtype']).itemsize / 2 ** 20
    del probe

    print(f'device     : {device}')
    print(f'data       : {CFG["data_dir"]}  [equiprobable]')
    print(f'cell       : {cell_tag()}')
    print(f'runs       : {run_dir()}')
    print(f'snapshots  : {snap_dir()}')
    print(f'model      : W_qk layer1 + layer2 trainable ({n_tr:,} of {n_all:,} params); '
          f'W_ov + unembedding frozen to the construction')
    print(f'readout    : z = log(f + b), bias BEFORE the log, no FFN, no residual')
    print(f'             b_sig = {CFG["b_sig"]}   b_noise rule = {CFG["b_noise_rule"]}')
    for t in THRESHOLDS:
        print(f'               alpha={t}: b_noise = {noise_bias(t):.6f}')
    print(f'recorder   : W_qk both layers every {CFG["snap_every"]} epoch(s), '
          f'{n_rows} rows (row 0 = random init), {mb:.0f} MB/run, '
          f'{mb*len(runs)/1024:.2f} GB total')
    print(f'optimizer  : {CFG["optimizer"]} lr={CFG["lr"]} b={CFG["batch_size"]} '
          f'init={CFG["init"]} epochs={CFG["epochs"]}')
    print(f'arch       : {ARCH} only   thresholds: {THRESHOLDS}   seeds: {SEEDS}')
    print(f'init       : W_qk = 0 for BOTH layers   (twin run: {COMPARE_WITH})')
    print(f'             ||dL/dW_qk1|| = 0 at step 0 only, by design -- see the docstring')
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
                      f'ETA={left * avg / 3600:.1f}h\n', flush=True)
    except KeyboardInterrupt:
        print('\n[sweep stopped by Ctrl+C]', flush=True)
    rebuild_summary()


if __name__ == '__main__':
    main()


