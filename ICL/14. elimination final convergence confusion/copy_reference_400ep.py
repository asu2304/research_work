"""
Build the 400-epoch REFERENCE cell for the "testing combined with high learning rate" experiment.

The high-lr sweep (train.py) runs combined at lr in {0.008 ... 0.030} for 400 epochs. To judge
whether a bigger Adam step lets combined reach what separate reaches, those runs need a baseline
on the SAME budget -- so this script takes the existing lr=0.006 runs (the cell wandb called
ICL-1000ep-convergence-check2, i.e. long-1000ep-b: combined AND separate, 1000 epochs) and
copies them into a reference cell with every curve TRUNCATED to the first 400 epochs.

Truncation is honest: best_train / test_at_best / final_* / epochs_run are all RECOMPUTED from
the truncated history, so the reference reports what those runs had actually achieved by epoch
400 -- not their eventual 1000-epoch values. The checkpoints are NOT copied, because the saved
.pth is the best-over-1000-epochs model and would not correspond to the truncated curve.

It also pushes the truncated runs to wandb (replaying their stored history), so the project
shows the baseline alongside the swept learning rates. Runs already present by name are skipped.

Usage:
    python copy_reference_400ep.py
"""

import glob
import json
import os

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ─── Config ───────────────────────────────────────────────────────────────────
SRC_CELL  = 'opt-adam_lr-0.006_b-128_init-normal_ep-1000'   # = long-1000ep-b
REF_CELL  = 'hi-lr_reference_opt-adam_lr-0.006_b-128_init-normal_ep-400'
TRUNC_EPOCHS = 400        # keep history[0 .. TRUNC_EPOCHS] (index 0 = the random init)

# Only the baseline architectures belong in the reference. The source cell may also hold stray
# runs from other experiments that happened to share its hyperparameters; those are not part of
# this comparison and are filtered out here rather than silently carried in.
ARCHS_TO_COPY = ['combined', 'separate']

WANDB_PROJECT = 'testing-combined-with-high-learning-rate'   # must match train.py
WANDB_ENTITY  = None
PUSH_TO_WANDB = True
DRY_RUN       = False
# ─────────────────────────────────────────────────────────────────────────────

SRC_DIR = os.path.join(_SCRIPT_DIR, 'runs', SRC_CELL)
DST_DIR = os.path.join(_SCRIPT_DIR, 'runs', REF_CELL)


def truncate(m):
    """Return a copy of the metrics dict with history cut to TRUNC_EPOCHS and every derived
    scalar recomputed from that shorter history."""
    tr = m['history']['train_loss'][:TRUNC_EPOCHS + 1]     # +1: index 0 is the init point
    te = m['history']['test_loss'][:TRUNC_EPOCHS + 1]
    # best over the TRAINED epochs only (epoch 0 is the init, never a checkpoint candidate)
    best_i = min(range(1, len(tr)), key=lambda i: tr[i])
    out = dict(m)
    out['history']      = {'train_loss': tr, 'test_loss': te}
    out['best_train']   = tr[best_i]
    out['test_at_best'] = te[best_i]
    out['final_train']  = tr[-1]
    out['final_test']   = te[-1]
    out['epochs_run']   = len(tr) - 1
    out['cell']         = REF_CELL
    out['truncated_from'] = {'cell': m['cell'], 'epochs': m['epochs_run']}
    cfg = dict(out.get('config') or {})
    cfg['cell'] = REF_CELL
    cfg['epochs'] = TRUNC_EPOCHS
    out['config'] = cfg
    return out


def main():
    src = sorted(glob.glob(os.path.join(SRC_DIR, 'metrics_*.json')))
    if not src:
        print(f'no metrics found in {SRC_DIR}'); return
    os.makedirs(DST_DIR, exist_ok=True)

    print(f'source : {SRC_CELL}  ({len(src)} runs)')
    print(f'target : {REF_CELL}  (truncated to {TRUNC_EPOCHS} epochs)\n')

    written = []
    for mp in src:
        m = json.load(open(mp))
        if m['arch'] not in ARCHS_TO_COPY:
            print(f"  [skip] {m['arch']} thr={m['threshold']} -- not a reference architecture")
            continue
        if len(m['history']['train_loss']) < TRUNC_EPOCHS + 1:
            print(f"  [skip] {os.path.basename(mp)} has only "
                  f"{len(m['history']['train_loss']) - 1} epochs"); continue
        t = truncate(m)
        print(f"  {t['arch']:9s} thr={t['threshold']}  "
              f"best_train {m['best_train']:.4f} @1000ep -> {t['best_train']:.4f} @400ep   "
              f"test_at_best {m['test_at_best']:.4f} -> {t['test_at_best']:.4f}")
        if not DRY_RUN:
            json.dump(t, open(os.path.join(DST_DIR, os.path.basename(mp)), 'w'), indent=2)
        written.append(t)
    print(f'\nwrote {len(written)} truncated runs -> runs/{REF_CELL}/'
          f'{"  (DRY_RUN: nothing written)" if DRY_RUN else ""}')
    print('note: checkpoints are intentionally NOT copied -- the saved .pth is the '
          'best-over-1000-epochs model and would not match the truncated curve.')

    if not (PUSH_TO_WANDB and written) or DRY_RUN:
        return

    import wandb
    api = wandb.Api()
    entity = WANDB_ENTITY or api.default_entity
    path = f'{entity}/{WANDB_PROJECT}' if entity else WANDB_PROJECT
    try:
        present = {r.name for r in api.runs(path)}
    except Exception as e:
        print(f'  [warn] could not list existing runs ({type(e).__name__}) -- assuming empty')
        present = set()

    print(f'\npushing to wandb project {WANDB_PROJECT} (entity {entity})')
    for t in written:
        stem = f"{t['arch']}_seed{t['seed']}_x_{t['threshold']}"
        name = f'{REF_CELL}__{stem}'
        if name in present:
            print(f'  [skip, present] {name}'); continue
        run = wandb.init(project=WANDB_PROJECT, entity=WANDB_ENTITY, name=name,
                         group=f"reference_lr-{t['config']['lr']}",
                         tags=[t['arch'], f"thr{t['threshold']}", f"seed{t['seed']}",
                               'reference', 'truncated400', f"lr{t['config']['lr']}"],
                         config=t['config'], reinit='finish_previous',
                         dir=os.path.join(_SCRIPT_DIR, 'wandb_local'))
        tr, te = t['history']['train_loss'], t['history']['test_loss']
        best = float('inf')
        for ep, (a, b) in enumerate(zip(tr, te)):
            if ep > 0 and a < best - 1e-6:
                best = a
            run.log({'train_ce': a, 'test_ce': b,
                     **({'best_train': best} if ep > 0 else {})}, step=ep)
        run.summary.update({'best_train': t['best_train'], 'test_at_best': t['test_at_best'],
                            'final_train': t['final_train'], 'final_test': t['final_test'],
                            'epochs_run': t['epochs_run'], 'reference': True})
        run.finish()
        print(f'  [uploaded] {name}')


if __name__ == '__main__':
    main()
