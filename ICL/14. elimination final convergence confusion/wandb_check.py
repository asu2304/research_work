"""Diagnose which wandb account/entity this shell can actually write to.

Run in the SAME terminal where WANDB_API_KEY is exported:
    python wandb_check.py
"""
import os, traceback
import wandb

print('WANDB_API_KEY set :', 'yes' if os.environ.get('WANDB_API_KEY') else 'no (using ~/.netrc)')
print('WANDB_ENTITY      :', os.environ.get('WANDB_ENTITY', '<unset>'))
api = wandb.Api()
v = api.viewer
print('username          :', v.username)
print('default entity    :', api.default_entity)
print('teams             :', getattr(v, 'teams', None))

for ent in [None, v.username] + list(getattr(v, 'teams', []) or []):
    label = ent if ent else '<default>'
    try:
        r = wandb.init(project='wandb-write-test', entity=ent, mode='online',
                       reinit='finish_previous', dir='/tmp')
        r.log({'ok': 1}); url = r.url; r.finish()
        print(f'  WRITE OK   entity={label}   {url}')
    except Exception as e:
        print(f'  DENIED     entity={label}   {type(e).__name__}: {e}')
