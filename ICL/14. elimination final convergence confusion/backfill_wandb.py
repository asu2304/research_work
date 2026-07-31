"""
Push ALREADY-TRAINED runs to wandb by replaying their stored history -- no GPU, no retraining.

Why this exists: train.py's SKIP_DONE (correctly) skips any run whose metrics json is already on
disk, and a skipped run never calls wandb.init, so it never appears on the web. When runs are
copied into a new cell directory from an earlier experiment, they are therefore missing from
that cell's wandb project even though the local files are complete. This script closes that gap.

Everything logged here comes from the metrics json train.py already wrote -- the per-epoch
train/test CE curves (index 0 = the random init, exactly as train.py logs it), the config, and
the summary scalars. The run name / group / tags are constructed with the SAME rules train.py
uses, so backfilled runs sit alongside natively-logged ones and group identically in the UI.

Runs that would be duplicates are skipped by default: a run is considered already present if the
project contains one with the same name (override with FORCE=True).

Usage -- edit CELL below if needed, then:
    python backfill_wandb.py
"""

import glob
import json
import os

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ─── Config ───────────────────────────────────────────────────────────────────
CELL = 'qk-ov-hybrids_opt-adam_lr-0.006_b-128_init-normal_ep-1000'
RUNS_DIR = os.path.join(_SCRIPT_DIR, 'runs', CELL)

WANDB_PROJECT = 'ICL-1000ep-qk-ov-hybrids'   # must match train.py
WANDB_ENTITY  = None                          # None -> default entity / $WANDB_ENTITY

# Only these architectures are backfilled -- the ones copied in from an earlier experiment.
# The others are trained (and logged) natively by train.py, so backfilling them would duplicate.
ARCHS_TO_BACKFILL = ['combined', 'separate']

FORCE = False        # True -> upload even if a run with the same name already exists
DRY_RUN = False      # True -> print what would be uploaded, contact wandb only to list runs
# ─────────────────────────────────────────────────────────────────────────────


def existing_run_names(api, entity, project):
    """Names already present in the project, so we never create duplicates."""
    path = f'{entity}/{project}' if entity else project
    try:
        return {r.name for r in api.runs(path)}
    except Exception as e:
        print(f'  [warn] could not list existing runs ({type(e).__name__}: {e}) -- '
              f'assuming the project is empty')
        return set()


def main():
    import wandb

    metrics = sorted(glob.glob(os.path.join(RUNS_DIR, 'metrics_*.json')))
    todo = []
    for mp in metrics:
        m = json.load(open(mp))
        if m['arch'] in ARCHS_TO_BACKFILL:
            todo.append(m)

    print(f'cell     : {CELL}')
    print(f'project  : {WANDB_PROJECT}')
    print(f'found    : {len(metrics)} metrics files, {len(todo)} match '
          f'ARCHS_TO_BACKFILL={ARCHS_TO_BACKFILL}')
    if not todo:
        print('nothing to do.'); return

    api = wandb.Api()
    entity = WANDB_ENTITY or api.default_entity
    print(f'entity   : {entity}')
    present = existing_run_names(api, entity, WANDB_PROJECT)
    print(f'already on wandb: {len(present)} run(s)\n')

    n_up = n_skip = 0
    for m in todo:
        cfg   = m['config']
        stem  = f"{m['arch']}_seed{m['seed']}_x_{m['threshold']}"
        name  = f'{CELL}__{stem}'                     # same rule train.py uses
        group = f"thr{m['threshold']}"
        tags  = [m['arch'], f"thr{m['threshold']}", f"seed{m['seed']}",
                 cfg['optimizer'], cfg['init'], f"ep{cfg['epochs']}", 'backfilled']

        if name in present and not FORCE:
            print(f'  [skip, already present] {name}'); n_skip += 1; continue
        if DRY_RUN:
            print(f'  [dry-run would upload]  {name}  '
                  f'({len(m["history"]["train_loss"])} epochs)'); n_up += 1; continue

        run = wandb.init(project=WANDB_PROJECT, entity=WANDB_ENTITY, name=name, group=group,
                         tags=tags, config=cfg, reinit='finish_previous',
                         dir=os.path.join(_SCRIPT_DIR, 'wandb_local'))
        tr, te = m['history']['train_loss'], m['history']['test_loss']
        best = float('inf')
        for ep, (a, b) in enumerate(zip(tr, te)):        # ep 0 == the random init
            if ep > 0 and a < best - 1e-6:
                best = a
            run.log({'train_ce': a, 'test_ce': b,
                     **({'best_train': best} if ep > 0 else {})}, step=ep)
        run.summary.update({'best_train': m['best_train'], 'test_at_best': m['test_at_best'],
                            'final_train': m.get('final_train', tr[-1]),
                            'final_test': m.get('final_test', te[-1]),
                            'epochs_run': m['epochs_run'], 'backfilled': True})
        run.finish()
        print(f'  [uploaded] {name}  ({len(tr)} epochs)')
        n_up += 1

    print(f'\ndone: {n_up} uploaded, {n_skip} skipped'
          f'{"  (DRY_RUN -- nothing actually sent)" if DRY_RUN else ""}')


if __name__ == '__main__':
    main()
