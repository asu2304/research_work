"""One-time: grid-search the CE-optimal FFN floors per threshold/split; dump to _floors.json.

Paper provenance: produced the (b_sig, b_noise) values hardcoded as FFN_FLOORS in
models_and_baselines.py / eval_all.py, used for the 'log' curve in Table 3 and
Figure 4 ("log(count+beta) FFN" rung). Re-run only if you retrain W* or change the grid.
"""
import json, time, os, glob, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # find models_and_baselines.py
import numpy as np, torch, torch.nn.functional as F

torch.set_num_threads(max(1, (os.cpu_count() or 8)))   # use all cores; W* forward is the cost
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        'models_and_baselines.py')).read()
defs = src[:src.index('# LAZY EVALUATION')]
defs = defs.replace("DEVICE           = 'cuda:1'", "DEVICE           = 'cpu'")   # CPU: uncontended, predictable
g = {}; exec(compile(defs, 'models_and_baselines.py', 'exec'), g)
load_data, _wlogits, _ce_logits = g['load_data'], g['_wlogits'], g['_ce_logits']
_pl_log_relu, _pl_log_gelu = g['_pl_log_relu'], g['_pl_log_gelu']
DATA_DIR, TS, TE, K = g['DATA_DIR'], g['TOKEN_START'], g['TOKEN_END'], g['RELU_LOG_K']

GRID = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1]

@torch.no_grad()
def argmin_floor(kind, L, yw, V, Vs):
    best = (float('inf'), None, None)
    for bs in GRID:
        for bn in GRID:
            if kind == 'log':
                b = torch.zeros(V); b[:Vs] = bs; b[Vs:] = bn
                ce = _ce_logits(torch.log(L + b), yw)
            else:
                z = torch.empty_like(L); fn = _pl_log_relu if kind == 'relu' else _pl_log_gelu
                z[..., :Vs] = fn(L[..., :Vs], bs, K); z[..., Vs:] = fn(L[..., Vs:], bn, K)
                ce = _ce_logits(z, yw)
            if ce < best[0]:
                best = (ce, bs, bn)
    return best

thrs = sorted(float(os.path.basename(f)[7:-3]) for f in glob.glob(os.path.join(DATA_DIR, 'test', 'data_x_*.pt')))
out = {k: {'test': {}, 'train': {}} for k in ('log', 'relu', 'gelu')}
t0 = time.time()
for split in ('test', 'train'):
    for thr in thrs:
        x, y, sl, V, blob = load_data(split, thr); yw = y[:, TS:TE]; Vs = int(blob['vocab_set1'])
        L = _wlogits(blob)
        for kind in ('log', 'relu', 'gelu'):
            ce, bs, bn = argmin_floor(kind, L, yw, V, Vs)
            out[kind][split][round(thr, 3)] = [bs, bn, round(ce, 4)]
        print(f'  {split} thr={thr}  ({time.time()-t0:.0f}s)', flush=True)
    json.dump(out, open('_floors.json', 'w'), indent=1)        # dump after EACH split (incremental)
    print(f'== wrote _floors.json after {split} ==', flush=True)
print('DONE', flush=True)
