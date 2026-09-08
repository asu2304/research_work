"""Full-batch zero-init training of the two W_qk matrices, recording GRADIENTS as well as weights.

Why this script exists
----------------------
`train_with_zero_initialization.py` trains with batch 256 and records only the PARAMETERS
(`QKRecorder` writes `p.detach()`; `.grad` appears nowhere in that file). Two consequences for
the step-level analysis in `zero_init_qk_evolution.ipynb`:

  * every recorded gradient is a 256-sequence *estimate* of the population gradient, so small
    structures sit under sampling noise (measured: cos(g_256, g_population) = 0.9797, and the
    induction block still has sign errors at that batch size);
  * one epoch is 98 optimizer steps, so the per-epoch snapshots cannot resolve the first steps
    at all -- exactly the window where layer 1 is frozen.

This script fixes both:

  * **batch = the whole training set (25,000 sequences), so 1 epoch == 1 optimizer step**, and
    every gradient recorded is the exact population gradient, not an estimate;
  * **both W_qk and dL/dW_qk are written every step** for the first `RECORD_GRAD_STEPS` steps
    (default 50), and W_qk every step for the whole run.

Full batch without the memory bill
----------------------------------
A single 25,000-sequence forward+backward needs ~17 GB of activations. Instead the gradient is
accumulated over chunks of `grad_chunk` sequences, each chunk's mean loss reweighted by its share
of the scored tokens. That is *mathematically identical* to one full-batch backward -- same
gradient to floating-point -- at ~1.3 GB peak. Measured on this box: 2.05 s per full-batch step
on cuda:3 (chunk 2048), 22.4 s on CPU. So the default 300 steps is ~10 min/threshold on a GPU.

The model, the init, the frozen W_ov / unembedding and all the metrics are IMPORTED from
`train_with_zero_initialization.py` rather than copied, so the architecture is guaranteed
identical and any difference between the two runs is attributable to the batch size alone.

Outputs (cell tag `wqkzero25k_opt-adam_lr-0.005_b-25000_init-zeros_ep-<EPOCHS>`):
    runs/<tag>/metrics_combined_seed0_x_<thr>.json      history + config + summary
    runs/<tag>/model_final_combined_seed0_x_<thr>.pth   last state (endpoint of the trajectory)
    runs/<tag>/model_best_combined_seed0_x_<thr>.pth    best-train state
    artifacts/<tag>/wqk1_x_<thr>.npy   (steps+1, 108, 108)   W_qk layer 1, row k = BEFORE step k+1
    artifacts/<tag>/wqk2_x_<thr>.npy   (steps+1, 216, 216)   W_qk layer 2
    artifacts/<tag>/gqk1_x_<thr>.npy   (G, 108, 108)         dL/dW_qk1 evaluated AT wqk1 row k
    artifacts/<tag>/gqk2_x_<thr>.npy   (G, 216, 216)         dL/dW_qk2
    artifacts/<tag>/snapmeta_x_<thr>.json                    shapes, dtypes, rows actually written

Row convention (the notebook relies on it):  wqk*[0] is the zero init; gqk*[k] is the gradient
computed at wqk*[k]; the update from wqk*[k] to wqk*[k+1] is Adam's response to gqk*[k].

Ctrl+C saves the best-so-far and moves to the next threshold, as in the sibling scripts.
"""

import os
import sys
import glob
import json
import time
import tempfile

# /tmp is unusable on this box -- route everything to ~/.cache/ashu_tmp (same as the siblings)
_TMP = os.path.expanduser('~/.cache/ashu_tmp')
os.makedirs(_TMP, exist_ok=True)
for _v in ('TMPDIR', 'TMP', 'TEMP'):
    os.environ[_v] = _TMP
tempfile.tempdir = _TMP
os.environ.setdefault('MPLCONFIGDIR', _TMP)

import importlib.util
import numpy as np
import torch
import torch.nn.functional as F

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ---- the model, the init, the frozen circuit and the metrics, imported not copied -----------
_BASE = os.path.join(_SCRIPT_DIR, 'train_with_zero_initialization.py')
if not os.path.exists(_BASE):
    sys.exit(f'missing {_BASE} -- this script imports the model definition from it')
_spec = importlib.util.spec_from_file_location('zi_base', _BASE)
ZI = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ZI)

QKOnlyTransformer = ZI.QKOnlyTransformer
apply_init        = ZI.apply_init
install_fixed     = ZI.install_fixed
qk_params         = ZI.qk_params
qk_stats          = ZI.qk_stats
slot_stats        = ZI.slot_stats
eval_ce           = ZI.eval_ce
noise_bias        = ZI.noise_bias
load_data         = ZI.load_data


CFG = {
    'data_dir': ZI.CFG['data_dir'],                          # the equiprobable dataset
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
    'lr':           1.0,
    'batch_size':   25000,     # the WHOLE training set: 1 epoch == 1 optimizer step
    'grad_chunk':   2048,      # accumulation chunk -- memory only, does not change the gradient
    'init':         'zeros',
    'weight_decay': 0.0,
    'eval_batch':   256,
    'epochs':       100,       # == optimizer steps here (full batch: 1 epoch = 1 step)

    'eval_every':   1,         # steps between test-CE / slot-stats evaluations
    'snap_dtype':   'float32',
}

ARCH       = 'combined'
SEEDS      = [0]
THRESHOLDS = [0.6, 0.8, 1.0]

RECORD_GRAD_STEPS = 50         # gradients stored for this many steps (weights: every step)

CELL_PREFIX   = 'wqkzero25k'
SKIP_DONE     = True
USE_WANDB     = True
WANDB_PROJECT = 'ICL-Wqk-zero-init-fullbatch-lr5-100-epochs'
WANDB_ENTITY  = None

# the batch-256 twin, for the side-by-side in the analysis notebook
COMPARE_WITH  = 'wqkzero_opt-adam_lr-0.005_b-256_init-zeros_ep-1000'


def cell_tag():
    return (f"{CELL_PREFIX}_opt-{CFG['optimizer']}_lr-{CFG['lr']}_b-{CFG['batch_size']}"
            f"_init-{CFG['init']}_ep-{CFG['epochs']}")


def run_dir():
    return os.path.join(CFG['runs_dir'], cell_tag())


def snap_dir():
    return os.path.join(CFG['art_dir'], cell_tag())


def metrics_path(seed, thr):
    return os.path.join(run_dir(), f'metrics_{ARCH}_seed{seed}_x_{thr}.json')


class StepRecorder:
    """Four .npy memmaps: W_qk layer1/layer2 every step, dL/dW_qk layer1/layer2 for the first
    `n_grad` steps. `flush(meta)` after every write keeps the files readable if the run dies."""

    def __init__(self, model, thr, n_w_rows, n_g_rows, dtype, meta):
        d = snap_dir()
        os.makedirs(d, exist_ok=True)
        shapes = [tuple(p.shape) for p in qk_params(model)]
        self.meta_path = os.path.join(d, f'snapmeta_x_{thr}.json')

        def mk(names, rows):
            return [np.lib.format.open_memmap(os.path.join(d, f'{nm}_x_{thr}.npy'), mode='w+',
                                              dtype=np.dtype(dtype), shape=(rows,) + s)
                    for nm, s in zip(names, shapes)]

        self.w = mk(['wqk1', 'wqk2'], n_w_rows)
        self.g = mk(['gqk1', 'gqk2'], n_g_rows)
        self.nw = self.ng = 0
        self.meta = {**meta, 'shapes': [list(s) for s in shapes], 'dtype': dtype,
                     'w_rows_allocated': n_w_rows, 'g_rows_allocated': n_g_rows,
                     'w_rows_written': 0, 'g_rows_written': 0,
                     'w_steps': [], 'g_steps': []}

    def _flush(self):
        self.meta['w_rows_written'], self.meta['g_rows_written'] = self.nw, self.ng
        with open(self.meta_path, 'w') as fh:
            json.dump(self.meta, fh, indent=2)

    def write_w(self, model, step):
        if self.nw >= len(self.w[0]):
            return
        for arr, p in zip(self.w, qk_params(model)):
            arr[self.nw] = p.detach().to('cpu', torch.float32).numpy()
            arr.flush()
        self.meta['w_steps'].append(int(step))
        self.nw += 1
        self._flush()

    def write_g(self, model, step):
        if self.ng >= len(self.g[0]):
            return
        for arr, p in zip(self.g, qk_params(model)):
            if p.grad is None:
                return
            arr[self.ng] = p.grad.detach().to('cpu', torch.float32).numpy()
            arr.flush()
        self.meta['g_steps'].append(int(step))
        self.ng += 1
        self._flush()

    def close(self):
        for a in self.w + self.g:
            a.flush()
        self.w, self.g = [], []

    def nbytes(self):
        n = np.dtype(CFG['snap_dtype']).itemsize
        s = [int(np.prod(x)) for x in self.meta['shapes']]
        return (self.meta['w_rows_allocated'] * sum(s) +
                self.meta['g_rows_allocated'] * sum(s)) * n


def full_batch_grad(model, x, y, V):
    """Exact gradient of the mean CE over ALL of x, accumulated in chunks.

    Each chunk's mean loss is reweighted by (tokens in chunk)/(tokens total), so the accumulated
    .grad equals what a single 25,000-sequence backward would have produced. Returns the loss.
    """
    ts, te = CFG['token_start'], CFG['seq_len']
    chunk = CFG['grad_chunk']
    n_tok = x.shape[0] * (te - ts)
    total = 0.0
    for p in qk_params(model):
        p.grad = None
    for i in range(0, x.shape[0], chunk):
        xb, yb = x[i:i + chunk], y[i:i + chunk, ts:te]
        w = yb.numel() / n_tok
        loss = F.cross_entropy(model(xb)[:, ts:te].reshape(-1, V), yb.reshape(-1))
        (loss * w).backward()
        total += float(loss) * w
    return total


def train_one(seed, thr, device):
    V = CFG['v_sig'] + CFG['n_noise']
    tag, stem = cell_tag(), f'{ARCH}_seed{seed}_x_{thr}'
    d = run_dir()
    os.makedirs(d, exist_ok=True)
    mpath = metrics_path(seed, thr)
    if SKIP_DONE and os.path.exists(mpath):
        print(f'  [skip] {stem} already done -> {mpath}', flush=True)
        return False

    tr_blob, te_blob = load_data('train', thr), load_data('test', thr)
    # 25k x 100 int64 = 20 MB per tensor: park the whole thing on the device once
    tr_x, tr_y = tr_blob['x'].to(device), tr_blob['y'].to(device)
    te_x, te_y = te_blob['x'], te_blob['y']

    model = QKOnlyTransformer(V, CFG['seq_len'], V + CFG['seq_len'], CFG['v_sig'],
                              thr).to(device)
    apply_init(model, seed)
    params = install_fixed(model)
    opt = torch.optim.Adam(params, lr=CFG['lr'], weight_decay=CFG['weight_decay'])

    steps = CFG['epochs']
    n_g = min(RECORD_GRAD_STEPS, steps)
    b_n = noise_bias(thr)
    run_cfg = {**{k: v for k, v in CFG.items() if k not in ('runs_dir', 'art_dir', 'data_dir')},
               'succ_split': 'equiprobable', 'record_grad_steps': n_g,
               'full_batch': True, 'steps_per_epoch': 1, 'n_train': int(tr_x.shape[0])}

    wb = None
    if USE_WANDB:
        ZI.USE_WANDB, ZI.WANDB_PROJECT, ZI.WANDB_ENTITY = True, WANDB_PROJECT, WANDB_ENTITY
        wb = ZI.wandb_start(run_cfg, name=f'{tag}__{stem}', group=f'thr{thr}',
                            tags=[ARCH, f'thr{thr}', f'seed{seed}', 'qk-only', 'full-batch'])
        if wb:
            for _m in ('train_ce', 'test_ce'):
                wb.define_metric(_m, summary='min')
            for _m in ('best_train', 'best_test'):
                wb.define_metric(_m, summary='last')

    rec = StepRecorder(model, thr, steps + 1, n_g, CFG['snap_dtype'],
                       {'cell': tag, 'arch': ARCH, 'seed': seed, 'threshold': float(thr),
                        'full_batch': True, 'n_train': int(tr_x.shape[0]),
                        'row0': 'zero init, before any step',
                        'convention': 'gqk*[k] is the gradient evaluated at wqk*[k]'})
    rec.write_w(model, 0)                       # row 0 = the zero init

    tr0 = eval_ce(model, tr_x, tr_y, V, device)
    te0 = eval_ce(model, te_x, te_y, V, device)
    st0, qs0 = slot_stats(model, te_x, device), qk_stats(model)
    history = {'step': [0], 'train_loss': [tr0], 'test_loss': [te0],
               'signal_mass': [st0['signal_mass']], 'frac_zero': [st0['frac_zero']],
               'wqk1_fro': [qs0['wqk1_fro']], 'wqk2_fro': [qs0['wqk2_fro']],
               'induction_diag_off': [qs0['induction_diag_off']],
               'gqk1_fro': [], 'gqk2_fro': []}

    if wb:
        wb.log({'train_ce': tr0, 'test_ce': te0, 'best_train': tr0, 'best_test': te0,
                **st0, **qs0}, step=0)

    print(f'  {stem}  trainable={sum(p.numel() for p in params):,}  '
          f'b_sig={CFG["b_sig"]}  b_noise={b_n:.6f}')
    print(f'    full batch : {tr_x.shape[0]:,} sequences in chunks of {CFG["grad_chunk"]} '
          f'-> 1 epoch = 1 optimizer step')
    print(f'    recording  : W_qk every step ({steps + 1} rows), grad for the first {n_g} steps, '
          f'{rec.nbytes() / 2 ** 20:.0f} MB -> {snap_dir()}')
    assert qs0['wqk1_fro'] == 0.0 and qs0['wqk2_fro'] == 0.0, \
        f'zero init failed: |Wqk1|={qs0["wqk1_fro"]} |Wqk2|={qs0["wqk2_fro"]}'
    print(f'    init (row 0): tr={tr0:.4f} te={te0:.4f}  '
          f'|Wqk1|=0.000 |Wqk2|=0.000  [both exactly zero, as intended]', flush=True)

    best_train, test_at_best, best_test, best_state = float('inf'), None, te0, None
    te_loss, st = te0, st0          # carried forward on steps where eval_every skips the eval
    t0, ep, interrupted_twice = time.time(), 0, False
    try:
        for ep in range(1, steps + 1):
            model.train()
            tr_loss = full_batch_grad(model, tr_x, tr_y, V)     # exact population gradient
            g1 = float(model.layer1.W_qk.grad.norm())
            g2 = float(model.layer2.W_qk.grad.norm())
            if ep - 1 < n_g:
                rec.write_g(model, ep - 1)                      # gradient AT w-row (ep-1)
            opt.step()
            rec.write_w(model, ep)

            if ep % CFG['eval_every'] == 0 or ep == steps:
                te_loss = eval_ce(model, te_x, te_y, V, device)
                st = slot_stats(model, te_x, device)
            qs = qk_stats(model)

            history['step'].append(ep)
            history['train_loss'].append(tr_loss)
            history['test_loss'].append(te_loss)
            history['signal_mass'].append(st['signal_mass'])
            history['frac_zero'].append(st['frac_zero'])
            history['wqk1_fro'].append(qs['wqk1_fro'])
            history['wqk2_fro'].append(qs['wqk2_fro'])
            history['induction_diag_off'].append(qs['induction_diag_off'])
            history['gqk1_fro'].append(g1)
            history['gqk2_fro'].append(g2)

            if tr_loss < best_train - 1e-6:
                best_train, test_at_best = tr_loss, te_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_test = min(best_test, te_loss)

            if wb:
                wb.log({'train_ce': tr_loss, 'test_ce': te_loss, 'best_train': best_train,
                        'best_test': best_test, 'gqk1_fro': g1, 'gqk2_fro': g2,
                        **st, **qs}, step=ep)
            if ep <= 10 or ep % 25 == 0:
                print(f'    step {ep:4d}/{steps}  tr={tr_loss:.5f}  te={te_loss:.5f}  '
                      f'|g1|={g1:.3e} |g2|={g2:.3e}  '
                      f'|W1|={qs["wqk1_fro"]:.4f} |W2|={qs["wqk2_fro"]:.4f}  '
                      f'ind_d/o={qs["induction_diag_off"]:.3f}', flush=True)
    except KeyboardInterrupt:
        print('    [INTERRUPTED] saving best-so-far (Ctrl+C again within ~1s exits)', flush=True)
        try:
            time.sleep(1)
        except KeyboardInterrupt:
            interrupted_twice = True
            print('    [INTERRUPTED x2] finishing this run cleanly, then exiting', flush=True)

    torch.save(model.state_dict(), os.path.join(d, f'model_final_{stem}.pth'))
    st_end, qs_end = slot_stats(model, te_x, device), qk_stats(model)
    rec.close()

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        torch.save(model.state_dict(), os.path.join(d, f'model_best_{stem}.pth'))

    result = {'cell': tag, 'arch': ARCH, 'seed': seed, 'threshold': float(thr),
              'history': history, 'best_train': best_train, 'test_at_best': test_at_best,
              'best_test': best_test,
              'final_train': history['train_loss'][-1], 'final_test': history['test_loss'][-1],
              'epochs_run': ep, 'steps_run': ep, 'time_s': round(time.time() - t0, 1),
              'config': run_cfg, 'slot_stats_final': st_end, 'qk_stats_final': qs_end,
              'compare_with': COMPARE_WITH,
              'snapshots': {'dir': os.path.relpath(snap_dir(), _SCRIPT_DIR),
                            'wqk1': f'wqk1_x_{thr}.npy', 'wqk2': f'wqk2_x_{thr}.npy',
                            'gqk1': f'gqk1_x_{thr}.npy', 'gqk2': f'gqk2_x_{thr}.npy',
                            'w_rows_written': rec.meta['w_rows_written'],
                            'g_rows_written': rec.meta['g_rows_written'],
                            'row0': 'zero init, before any step',
                            'convention': 'gqk*[k] is the gradient evaluated at wqk*[k]'}}
    with open(mpath, 'w') as fh:
        json.dump(result, fh, indent=2)
    if wb:
        wb.summary.update({'best_train': best_train, 'best_test': best_test,
                           'final_train': result['final_train'],
                           'final_test': result['final_test'], 'steps_run': ep})
        try:
            wb.finish()
        except Exception as e:
            print(f'    [wandb] finish failed ({type(e).__name__}: {e})', flush=True)
    print(f'    done in {result["time_s"] / 60:.1f}m  best_train={best_train:.5f}  '
          f'best_test={best_test:.5f}  w_rows={rec.meta["w_rows_written"]} '
          f'g_rows={rec.meta["g_rows_written"]}  -> {mpath}', flush=True)
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
                      'best_test', 'final_train', 'final_test', 'steps_run')})
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
    ZI._DATA.clear()


def main():
    device = torch.device(CFG['device'] if torch.cuda.is_available() else 'cpu')
    check_dataset()
    runs = [(seed, thr) for thr in THRESHOLDS for seed in SEEDS]
    n_done = sum(1 for r in runs if os.path.exists(metrics_path(*r)))

    V = CFG['v_sig'] + CFG['n_noise']
    probe = QKOnlyTransformer(V, CFG['seq_len'], V + CFG['seq_len'], CFG['v_sig'], 0.6)
    n_tr = sum(p.numel() for p in install_fixed(probe))
    n_all = sum(p.numel() for p in probe.parameters())
    per_row = sum(p.numel() for p in qk_params(probe)) * \
        np.dtype(CFG['snap_dtype']).itemsize
    n_g = min(RECORD_GRAD_STEPS, CFG['epochs'])
    mb = (CFG['epochs'] + 1 + n_g) * per_row / 2 ** 20
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
    print(f'batching   : FULL BATCH {CFG["batch_size"]:,} in chunks of {CFG["grad_chunk"]} '
          f'-> 1 epoch = 1 optimizer step, gradient is exact (not a 256-sample estimate)')
    print(f'recorder   : W_qk both layers every step ({CFG["epochs"] + 1} rows) + '
          f'grad both layers for the first {n_g} steps  ~= {mb:.0f} MB per threshold')
    print(f'runs       : {len(runs)} ({n_done} already done, SKIP_DONE={SKIP_DONE})\n')

    for seed, thr in runs:
        print(f'[thr {thr}  seed {seed}]', flush=True)
        try:
            train_one(seed, thr, device)
        except KeyboardInterrupt:
            print('\nstopping after the current run (Ctrl+C x2)')
            break
    rebuild_summary()


if __name__ == '__main__':
    main()
