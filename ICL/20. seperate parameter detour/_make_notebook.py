"""Regenerates `analysis_separate_qk.ipynb` from this file.

Provenance script, in the same spirit as folder 8's `_find_floors.py`: the notebook is the
deliverable, this is where its source lives so a cell can be edited without hand-patching JSON.

    python _make_notebook.py                      # -> analysis_separate_qk.ipynb
    python _make_notebook.py other_name.ipynb

Running it OVERWRITES the target notebook, outputs and all. Edit here, not there.
"""

import json
import sys

C = []          # (kind, source)


def md(s):
    C.append(('markdown', s.strip('\n')))


def co(s):
    C.append(('code', s.strip('\n')))


# ----------------------------------------------------------------------------------------- 1
md(r"""
# Separate $W_q$ / $W_k$ — does GD still find the construction?

Companion notebook to **`train_separate_qk.py`** (folder 20).

Folders 17/18/19 carry **one** parameter per layer, the combined $W_{qk}$, and score a pair by
$s(q,k) = x_q W_{qk} x_k^\top/\sqrt{108}$. That parametrisation removes the within-head rotation
gauge by construction. This run puts it back: each layer carries $W_q$ and $W_k$ of shape
$(\text{in}\times\text{head})$ and

$$s(q,k) \;=\; (x_q W_q)(x_k W_k)^\top / \sqrt{108}, \qquad W_{qk} = W_q W_k^\top .$$

Everything else is held still — $W_{ov}$ frozen to the $8\times8$ identity, no unembedding, no
log, no output softmax, CE on $(f+b)/\sum(f+b)$ with $b$ fixed from $\alpha$, strictly causal
mask, full batch, Adam, loss on tokens $[50{:}100)$.

**Two things change that are not choices.**

1. **Zero init is a strict saddle.** $\partial L/\partial W_q = (\partial L/\partial W_{qk})W_k$
   and $\partial L/\partial W_k = (\partial L/\partial W_{qk})^\top W_q$, so at $W_q=W_k=0$
   *both* gradients are exactly zero and nothing moves — measured, not argued: the training
   script prints $\|g\|$ at init and stops the threshold if the weights have not left the
   origin after one step. The run uses `--init eps` ($\mathcal N(0,10^{-3})$), which puts
   $|W_{qk}|$ at $\sim\!10^{-4}$ — uniform attention to five decimals, i.e. a zero init in
   every way that matters except the geometry.
2. **The factorisation adds a gauge.** For any invertible $M$,
   $W_q \to W_q M$, $W_k \to W_k M^{-\top}$ leaves every logit untouched. So a raw plot of the
   learned $W_q$ against the construction's $W_q$ is a picture of the notation, not the model.

This notebook therefore plots **both**: the separate matrices as learned (§5), and the combined
product $W_qW_k^\top$ (§6), which is the object that actually has to match. §9 shows how much of
the separate-matrix disagreement is pure gauge, by fixing it and re-measuring.

> **The construction** is built by `construction()` inside `train_separate_qk.py`, transcribed
> from `seperate_parameterization_weight_handconstruction.pdf` in the same
> (in $\times$ head) display convention, so `W_qk = Wq @ Wk.T` cell for cell.
""")

# ----------------------------------------------------------------------------------------- 2
md("## 1. Setup — find the run, read the metrics")

co(r"""
import os, re, sys, json, glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm

sys.path.insert(0, os.path.abspath('.'))
from train_separate_qk import construction, gauge_fix, gauge_cos, key_groups

V, Vs, T = 8, 4, 100
D1, D2   = V + T, 2 * (V + T)          # 108, 216
SDq      = D1 ** 0.5                   # entry = logit * sqrt(108)

RUNS_DIR = 'runs'      # point this at a scratch dir to inspect a smoke test
RUN_NAME = None        # None -> the sepqk_ run with the most epochs

# Reference numbers for the COMBINED-W_qk runs, CLAUDE.md section 8.5 (folder 18, Adam lr 1.0,
# zero init, 500 full-batch steps). 'gd' is what gradient descent reached there, 'con' the hand
# construction evaluated directly, 'ideal' the alpha-oracle floor.
REF = {0.6: dict(gd=1.69823, con=1.70100, ideal=1.64342),
       0.8: dict(gd=1.38435, con=1.38383, ideal=1.33218),
       1.0: dict(gd=0.74498, con=0.74730, ideal=0.69315)}
# Gauge-fixed cosine reached by those combined runs, CLAUDE.md section 8.8.
REF_COS = {0.6: (0.9209, 0.8646), 0.8: (0.9564, 0.8352), 1.0: (0.5477, 0.4105)}


def discover_run():
    cands = [d.rstrip('/') for d in glob.glob(os.path.join(RUNS_DIR, 'sepqk_*/'))
             if glob.glob(os.path.join(d, 'metrics_seed*_x_*.json'))]
    if not cands:
        raise FileNotFoundError(
            f'no {RUNS_DIR}/sepqk_* with metrics yet -- run:\n'
            f'    python train_separate_qk.py\n  (present: '
            f'{sorted(os.path.basename(d.rstrip("/")) for d in glob.glob(RUNS_DIR + "/*/"))})')

    def eps(d):
        m = re.search(r'_ep-(\d+)', d)
        return int(m.group(1)) if m else -1
    return max(cands, key=lambda d: (eps(d), os.path.getmtime(d)))


RUN = os.path.join(RUNS_DIR, RUN_NAME) if RUN_NAME else discover_run()
MET = {}
for p in sorted(glob.glob(os.path.join(RUN, 'metrics_seed*_x_*.json'))):
    m = json.load(open(p))
    MET[float(m['threshold'])] = m
THRS = sorted(MET)
CFG0 = MET[THRS[0]]['config']

print(f'run        : {RUN}')
print(f'thresholds : {THRS}')
print(f'optimiser  : {CFG0["optimizer"]}  lr {CFG0["lr"]}  {CFG0["epochs"]} full-batch steps  '
      f'init {CFG0["init"]}'
      + (f' (scale {CFG0["init_scale"]:g})' if CFG0['init'] in ('eps', 'orth') else ''))
print(f'head dim   : {CFG0["head_dim"]}   trainable entries {CFG0["trainable_entries"]:,} '
      f'(W_q + W_k; W_ov frozen)')
print()
hdr = (f'{"alpha":>6}{"best test":>11}{"final test":>12}{"steps":>7}'
       f'{"cos L1":>9}{"cos L2":>9}   |  combined-W_qk run (folder 18)')
print(hdr)
print('-' * len(hdr))
for t in THRS:
    m, q = MET[t], MET[t]['final_qk']
    r = REF.get(t)
    ref = f'gd {r["gd"]:.5f}  con {r["con"]:.5f}  ideal {r["ideal"]:.5f}' if r else ''
    print(f'{t:>6}{m["best_test"]:>11.5f}{m["final_test"]:>12.5f}{m["steps_run"]:>7}'
          f'{q["L1_cos"]:>+9.4f}{q["L2_cos"]:>+9.4f}   |  {ref}')
    if m['stopped_early']:
        print(f'       [stopped early] {m["stop_reason"]}')
""")

# ----------------------------------------------------------------------------------------- 3
md(r"""
## 2. The construction, in separate form

`construction()` returns the four attention matrices in display convention
($x$ multiplies from the left, rows = input dims, columns = head dims) plus the two products.
Two magnitude regimes:

| | `paper=True` | `paper=False` (the reference used here) |
|---|---|---|
| layer-1 recency | $\alpha_{\rm rec}=20$, uncapped | $\lambda=20$, capped at 5 steps |
| layer-2 priority | $P_1=1000,\ P_2=50$ | $K=100,\ K/20=5$ |
| noise trigger | $\eta=10$ | $1$ |
| noise kill | $\gamma=-5\times10^4$, $\mathrm{BIG}=-10^9$ | $-B=-100$ |
| self-veto | explicit $-10^9$ on the diagonal | dropped — the strict causal mask does it |
| $\|W_{qk}\|_\max$ | $1.04\times10^{10}$ / $5.2\times10^6$ | $1039$ / $1039$ |

The reduced form costs **0.004 nats** (CLAUDE.md §8.6) and is the fair comparison, because GD
never has a reason to reach $10^{10}$.

`drop_col4=False` here, unlike folder 18: that column of $W_{qk}^{(2)}$ is inert (ablating it
changes the CE by exactly `0.00e+00`) but it *is* what the factorisation produces, and in a
notebook about the factorisation the reference has to actually be a factorisation.
""")

co(r"""
CON  = construction(paper=False, drop_col4=False)    # the reference everywhere below
CONP = construction(paper=True,  drop_col4=False)    # the PDF's own magnitudes

for nm, c in (('reduced', CON), ('paper', CONP)):
    e1 = np.abs(c['Wq1'] @ c['Wk1'].T - c['W_qk1']).max()
    e2 = np.abs(c['Wq2'] @ c['Wk2'].T - c['W_qk2']).max()
    print(f'{nm:8s}  Wq @ Wk.T == W_qk :  L1 {e1:.3e}   L2 {e2:.3e}')

print()
print(f'{"matrix":10s}{"shape":>14}{"nnz":>9}{"|max|":>14}{"rank":>7}')
for k in ('Wq1', 'Wk1', 'W_qk1', 'Wq2', 'Wk2', 'W_qk2'):
    A = CON[k]
    print(f'{k:10s}{str(A.shape):>14}{int((A != 0).sum()):>9}{np.abs(A).max():>14.4f}'
          f'{np.linalg.matrix_rank(A, tol=1e-8):>7}')
print('\nLayer 2 is rank 5: four P1 columns plus the one noise-trigger column. Everything the\n'
      'induction head does lives in a 5-dimensional slice of a 108-dimensional head.')


def heat(ax, A, title, annot=False, unit=SDq, fmt='{:+.0f}'):
    '''SymLog heatmap. Annotations are in LOGIT units (entry / sqrt(108)) -- the construction's
       constants are round numbers there, and the raw entries are too wide to print.'''
    A = np.asarray(A, dtype=float)
    v = float(np.abs(A).max())
    im = ax.imshow(A, cmap='RdBu_r', interpolation='nearest',
                   norm=SymLogNorm(linthresh=max(v / 1e3, 1e-30), linscale=0.6,
                                   vmin=-v, vmax=v, base=10) if v > 0 else None)
    if annot and A.size <= 400:
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                z = A[i, j] / unit
                ax.text(j, i, '0' if z == 0 else fmt.format(z), ha='center', va='center',
                        fontsize=6.5, color='white' if abs(A[i, j]) > v / 3 else 'black')
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f'{title}\n|max| {v:.4g}   $\\|\\cdot\\|_F$ {np.linalg.norm(A):.4g}',
                 fontsize=9)
    return im


fig, ax = plt.subplots(2, 4, figsize=(19, 9))
heat(ax[0, 0], CON['Wq1'], '$W_q^{(1)}$   (108, 108)\n$\\sqrt{D}\\,I$ — a no-op scaling')
heat(ax[0, 1], CON['Wk1'], '$W_k^{(1)}$   (108, 108)\nnoise kill + recency ramp')
heat(ax[0, 2], CON['Wk1'][V:V + 14, V:V + 14],
     '$W_k^{(1)}$ position corner [8:22, 8:22]\nthe capped ramp, cell by cell',
     annot=True, unit=1.0)
heat(ax[0, 3], CON['W_qk1'][0:V, 0:V], '$W_{qk}^{(1)}$[0:8, 0:8]   token $\\times$ token',
     annot=True)
heat(ax[1, 0], CON['Wq2'], '$W_q^{(2)}$   (216, 108)\n$P_1$, $P_2$, noise trigger')
heat(ax[1, 1], CON['Wk2'], '$W_k^{(2)}$   (216, 108)\npredecessor readout + noise kill')
heat(ax[1, 2], CON['W_qk2'][0:V, 0:V], '$W_{qk}^{(2)}$[0:8, 0:8]\nA — noise suppression',
     annot=True)
heat(ax[1, 3], CON['W_qk2'][0:Vs, D1:D1 + V], '$W_{qk}^{(2)}$[0:4, 108:116]\nB — induction',
     annot=True)
fig.suptitle('The hand construction, reduced magnitudes   '
             '(annotated cells are in LOGIT units = entry / $\\sqrt{108}$)', fontsize=13)
plt.tight_layout(); plt.show()
""")

# ----------------------------------------------------------------------------------------- 4
md(r"""
## 3. Did it train? — loss, and agreement with the construction step by step

The gauge-fixed cosine against the construction is logged by the training script every step
(on the **product**, never on $W_q$ or $W_k$ alone). Dashed lines are the combined-$W_{qk}$
run's numbers from CLAUDE.md §8.5 / §8.8.
""")

co(r"""
fig, ax = plt.subplots(2, len(THRS), figsize=(5.0 * len(THRS), 8), squeeze=False)
for j, t in enumerate(THRS):
    h = MET[t]['history']
    s = np.asarray(h['step'])
    a = ax[0, j]
    a.plot(s, h['train_loss'], lw=1.6, color='#2f6fc2', label='train')
    a.plot(s, h['test_loss'], lw=1.6, color='#c2492f', label='test')
    if t in REF:
        a.axhline(REF[t]['gd'], ls='--', lw=1.1, color='0.35',
                  label=f'combined-$W_{{qk}}$ GD  {REF[t]["gd"]:.4f}')
        a.axhline(REF[t]['ideal'], ls=':', lw=1.1, color='#1a8f4a',
                  label=f'ideal-$\\alpha$  {REF[t]["ideal"]:.4f}')
    a.set_title(f'$\\alpha$ = {t}   best test {MET[t]["best_test"]:.5f}', fontsize=11)
    a.set_xlabel('step'); a.grid(alpha=.3); a.legend(fontsize=8)
    if j == 0:
        a.set_ylabel('cross entropy (nats)')

    b = ax[1, j]
    sc = s[1:len(h['L1_cos']) + 1]
    b.plot(sc, h['L1_cos'][:len(sc)], lw=1.6, color='#2f6fc2', label='layer 1')
    b.plot(sc, h['L2_cos'][:len(sc)], lw=1.6, color='#c2492f', label='layer 2')
    if t in REF_COS:
        b.axhline(REF_COS[t][0], ls='--', lw=1.0, color='#2f6fc2', alpha=.6,
                  label=f'combined run L1  {REF_COS[t][0]:.3f}')
        b.axhline(REF_COS[t][1], ls='--', lw=1.0, color='#c2492f', alpha=.6,
                  label=f'combined run L2  {REF_COS[t][1]:.3f}')
    b.axhline(0, color='0.7', lw=.8)
    b.set_ylim(-0.3, 1.0); b.set_xlabel('step'); b.grid(alpha=.3); b.legend(fontsize=8)
    b.set_title('gauge-fixed cosine vs the construction', fontsize=10)
    if j == 0:
        b.set_ylabel('cosine')
fig.suptitle(os.path.basename(RUN), fontsize=12)
plt.tight_layout(); plt.show()
""")

# ----------------------------------------------------------------------------------------- 5
md(r"""
## 4. Loading the learned matrices

`model_best_seed0_x_{alpha}.pth` is checkpointed on **best test loss**, `model_final_...` is the
last step. Both are plain state dicts with keys `layer{1,2}.W_{q,k,ov}`, already in display
convention — so `W_qk = W_q @ W_k.T` with no transposes to get wrong.

The split of magnitude between $\|W_q\|$ and $\|W_k\|$ is pure gauge ($W_q \to cW_q$,
$W_k \to W_k/c$ changes nothing), so only the product's norm is worth comparing.
""")

co(r"""
import torch

WHICH = 'best'          # 'best' (checkpointed on test loss) or 'final'


def load_run(thr, which=WHICH):
    p = os.path.join(RUN, f'model_{which}_seed0_x_{thr}.pth')
    sd = torch.load(p, map_location='cpu', weights_only=True)
    d = {k: v.numpy().astype(np.float64) for k, v in sd.items()}
    d['W_qk1'] = d['layer1.W_q'] @ d['layer1.W_k'].T
    d['W_qk2'] = d['layer2.W_q'] @ d['layer2.W_k'].T
    return d


GD = {t: load_run(t) for t in THRS}

print(f'{"alpha":>6}{"|Wq1|_F":>11}{"|Wk1|_F":>11}{"|Wqk1|max":>12}'
      f'{"|Wq2|_F":>11}{"|Wk2|_F":>11}{"|Wqk2|max":>12}')
for t in THRS:
    g = GD[t]
    print(f'{t:>6}{np.linalg.norm(g["layer1.W_q"]):>11.4g}'
          f'{np.linalg.norm(g["layer1.W_k"]):>11.4g}{np.abs(g["W_qk1"]).max():>12.4g}'
          f'{np.linalg.norm(g["layer2.W_q"]):>11.4g}'
          f'{np.linalg.norm(g["layer2.W_k"]):>11.4g}{np.abs(g["W_qk2"]).max():>12.4g}')
print(f'{"con":>6}{np.linalg.norm(CON["Wq1"]):>11.4g}{np.linalg.norm(CON["Wk1"]):>11.4g}'
      f'{np.abs(CON["W_qk1"]).max():>12.4g}{np.linalg.norm(CON["Wq2"]):>11.4g}'
      f'{np.linalg.norm(CON["Wk2"]):>11.4g}{np.abs(CON["W_qk2"]).max():>12.4g}')

print('\nBalance |Wq|_F / |Wk|_F  (1.0 = the balanced gauge; the construction is far from it,')
print('which is a fact about how it was written down, not about the circuit):')
for lbl, d, kq, kk in ([(f'GD  a={t}', GD[t], 'layer1.W_q', 'layer1.W_k') for t in THRS]
                       + [('construction', CON, 'Wq1', 'Wk1')]):
    r1 = np.linalg.norm(d[kq]) / np.linalg.norm(d[kk])
    kq2, kk2 = (kq.replace('1', '2'), kk.replace('1', '2'))
    r2 = np.linalg.norm(d[kq2]) / np.linalg.norm(d[kk2])
    print(f'   {lbl:>14}   layer 1 {r1:>10.4g}   layer 2 {r2:>10.4g}')
""")

# ----------------------------------------------------------------------------------------- 6
md(r"""
## 5. The learned $W_q$ and $W_k$, as they come out

Plotted raw, next to the construction's. **Read these with the gauge in mind.** Any invertible
$M$ acting as $W_q \to W_q M$, $W_k \to W_k M^{-\top}$ leaves every logit of the model
identical, so there is no reason for a converged GD run to land on the construction's particular
writing of the pair even when it has found exactly the same circuit.

The one thing that *is* safe to read off directly is the **row support** — which residual-stream
coordinates each head reads at all. $M$ acts on columns, so row norms are gauge-invariant. The
construction's $W_q^{(2)}$ is nonzero only on rows 0:4 (own signal token), row 4 (the noise
trigger) and 108:112 (the layer-1 predecessor); its $W_k^{(2)}$ only on rows 4:8 and 108:116.
""")

co(r"""
def rowmass(A):
    return np.linalg.norm(np.asarray(A, dtype=float), axis=1)


for t in THRS:
    g = GD[t]
    fig, ax = plt.subplots(2, 4, figsize=(19, 8.4))
    for r, lay in enumerate((1, 2)):
        heat(ax[r, 0], g[f'layer{lay}.W_q'], f'GD  $W_q^{{({lay})}}$')
        heat(ax[r, 1], CON[f'Wq{lay}'], f'construction  $W_q^{{({lay})}}$')
        heat(ax[r, 2], g[f'layer{lay}.W_k'], f'GD  $W_k^{{({lay})}}$')
        heat(ax[r, 3], CON[f'Wk{lay}'], f'construction  $W_k^{{({lay})}}$')
    fig.suptitle(f'$\\alpha$ = {t}   learned vs constructed $W_q$, $W_k$ — RAW, gauge not fixed '
                 f'(disagreement here means little; see §6 and §9)', fontsize=13)
    plt.tight_layout(); plt.show()

    fig, ax = plt.subplots(2, 2, figsize=(15, 6.6))
    for r, lay in enumerate((1, 2)):
        for c, (short, ckey) in enumerate((('W_q', f'Wq{lay}'), ('W_k', f'Wk{lay}'))):
            a = ax[r, c]
            gr = rowmass(g[f'layer{lay}.{short}'])
            cr = rowmass(CON[ckey])
            a.semilogy(np.maximum(gr, 1e-12), lw=1.2, color='#c2492f', label='GD')
            scale = max(gr.max(), 1e-12) / max(cr.max(), 1e-12)
            a.semilogy(np.maximum(cr * scale, 1e-12), lw=1.2, color='#2f6fc2', alpha=.75,
                       label='construction (rescaled to GD)')
            for x in ([V, D1] if lay == 1 else [V, D1, D1 + V]):
                a.axvline(x - .5, color='0.6', lw=.8)
            a.set_title(f'layer {lay}   ${short[0]}_{{{short[-1]}}}$ row norms   '
                        f'(dividers: token | position' + (' | L1-out | dead)' if lay == 2
                                                          else ')'), fontsize=9)
            a.set_xlabel('input dimension'); a.grid(alpha=.3); a.legend(fontsize=8)
    fig.suptitle(f'$\\alpha$ = {t}   row support is gauge-INVARIANT — $M$ acts on columns',
                 fontsize=12)
    plt.tight_layout(); plt.show()
""")

# ----------------------------------------------------------------------------------------- 7
md(r"""
## 6. The combined $W_{qk} = W_q W_k^\top$ — the object that has to match

Same six cuts the folder-18/19 notebooks use, GD on top, construction below:

| cut | what it is |
|---|---|
| L1 $[0{:}8,0{:}8]$ | token $\times$ token — the layer-1 noise veto |
| L1 $[8{:}108,8{:}108]$ | position $\times$ position — the recency ramp |
| L2 $[0{:}8,0{:}8]$ | **A**, noise suppression |
| L2 $[0{:}4,108{:}112]$ | **B**, the induction match |
| L2 $[108{:}112,108{:}112]$ | **C**, state $\times$ state |

A constant added to a query row across a complete key group is invisible to the softmax, so
compare **contrasts within a row**, not absolute cell values.
""")

co(r"""
ZOOM_AT, ZOOM_N = 45, 10
ZOOM = slice(V + ZOOM_AT, V + ZOOM_AT + ZOOM_N)
CUTS = [('W_qk1', slice(0, V), slice(0, V), 'L1 [0:8, 0:8]\ntoken x token', True),
        ('W_qk1', slice(V, D1), slice(V, D1), 'L1 [8:108, 8:108]\nposition x position', False),
        ('W_qk1', ZOOM, ZOOM,
         f'L1 pos x pos [{ZOOM_AT}:{ZOOM_AT + ZOOM_N}]\nthe recency ramp', True),
        ('W_qk2', slice(0, V), slice(0, V), 'L2 [0:8, 0:8]\nA — noise suppression', True),
        ('W_qk2', slice(0, Vs), slice(D1, D1 + Vs), 'L2 [0:4, 108:112]\nB — induction', True),
        ('W_qk2', slice(D1, D1 + Vs), slice(D1, D1 + Vs),
         'L2 [108:112, 108:112]\nC — state x state', True)]

for t in THRS:
    fig, ax = plt.subplots(2, len(CUTS), figsize=(3.1 * len(CUTS), 7.6))
    for j, (key, rs, cs, ttl, ann) in enumerate(CUTS):
        heat(ax[0, j], GD[t][key][rs, cs], f'GD\n{ttl}', annot=ann)
        heat(ax[1, j], CON[key][rs, cs], f'construction\n{ttl}', annot=ann)
    fig.suptitle(f'$\\alpha$ = {t}   $W_{{qk}} = W_qW_k^\\top$, the gauge-invariant object   '
                 f'(annotations in logit units)', fontsize=13)
    plt.tight_layout(); plt.show()
""")

# ----------------------------------------------------------------------------------------- 8
md(r"""
## 7. Where does GD put mass when nothing forces it into the blocks?

Folders 18/19 **masked** $W_{qk}$ down to the construction's blocks, so "GD matches the
construction" there was partly enforced. Here $W_q$ and $W_k$ are dense and a single row of
$W_q$ touches every key at once — the block pattern is not imposed at all. So this run can ask a
question the earlier ones could not: *given the freedom, does GD stay inside the construction's
blocks?*

The table splits each layer's gauge-fixed energy $\|\cdot\|_F^2$ over blocks, and the
construction's own row is the ceiling — **but that ceiling is not 100%, and the reason is the
factorisation itself.** A single row of $W_k^{(1)}$ vetoes noise keys for *every* query at once,
so the construction's layer-1 noise veto necessarily spills into the position-query rows. Folder
18 could mask that spill away (it ablated those 400 cells at a cost of ~0 nats, CLAUDE.md §8.6);
a factored parametrisation cannot express the masked version at all. Quote each row against the
construction's own figure, not against 100.

Layer 2 is the interesting one: the construction lives entirely in A, B and C, with *nothing* on
position queries or position keys. Whatever GD parks there is something the construction does
not do.
""")

co(r"""
def blocks_of(layer):
    if layer == 1:
        return {'tok x tok': (slice(0, V), slice(0, V)),
                'pos x pos': (slice(V, D1), slice(V, D1)),
                'tok x pos': (slice(0, V), slice(V, D1)),
                'pos x tok': (slice(V, D1), slice(0, V))}
    return {'A tok x tok': (slice(0, V), slice(0, V)),
            'B tok x L1out': (slice(0, V), slice(D1, D1 + V)),
            'C L1o x L1o': (slice(D1, D1 + V), slice(D1, D1 + V)),
            'L1out x tok': (slice(D1, D1 + V), slice(0, V)),
            'pos as query': (slice(V, D1), slice(0, D1 + V)),
            'pos as key': (slice(0, D1 + V), slice(V, D1))}


CON_BLOCKS = {1: ['tok x tok', 'pos x pos'],
              2: ['A tok x tok', 'B tok x L1out', 'C L1o x L1o']}


def energy_split(W, layer):
    '''gauge-fixed energy per block, as a fraction of the layer total'''
    G = gauge_fix(W, layer)                       # rows already trimmed to the live ones
    rows, _ = key_groups(layer)
    off = rows.start
    tot = float((G ** 2).sum())
    out = {}
    for nm, (rs, cs) in blocks_of(layer).items():
        r = slice(max(rs.start - off, 0), min(rs.stop - off, G.shape[0]))
        c = slice(cs.start, min(cs.stop, G.shape[1]))
        out[nm] = float((G[r, c] ** 2).sum()) / tot if tot > 0 else 0.0
    return out


for layer in (1, 2):
    names = list(blocks_of(layer))
    print(f'--- layer {layer} ---    * marks a block the construction uses')
    w = 16
    print(f'{"":>14}' + ''.join(f'{n + (" *" if n in CON_BLOCKS[layer] else ""):>{w}}'
                                for n in names) + f'{"in con blocks":>16}')
    rows_out = ([('construction', CON[f'W_qk{layer}'])]
                + [(f'GD a={t}', GD[t][f'W_qk{layer}']) for t in THRS])
    for lbl, W in rows_out:
        e = energy_split(W, layer)
        inside = sum(e[n] for n in CON_BLOCKS[layer])
        print(f'{lbl:>14}' + ''.join(f'{100 * e[n]:>{w - 1}.2f}%' for n in names)
              + f'{100 * inside:>15.2f}%')
    print()
print('The "pos as query" / "pos as key" rows overlap the others by design, so the columns do\n'
      'not sum to 100. The last column is the number to read.')
""")

# ----------------------------------------------------------------------------------------- 9
md(r"""
## 8. Gauge — fixing it, and what is left

Two separate gauges are in play, and they need different treatment.

**(a) The softmax row-shift gauge** (CLAUDE.md §8.7), which the combined runs already had.
$\mathrm{softmax}(s + c\mathbf 1) = \mathrm{softmax}(s)$, and a key contributes exactly one
entry from each one-hot block of the residual stream, so adding a constant to a query row across
a complete key block changes nothing. `gauge_fix` projects it out by subtracting each query
row's mean over each complete key group, on both sides. This is the one that made raw cosine
report 0.7169 / 0.8631 / 0.8631 / 0.8794 for four writings of the *same* circuit, and 0.8646
every time once fixed.

**(b) The factorisation gauge** $W_q \to W_qM$, $W_k \to W_kM^{-\top}$, new to this folder.
It does **not** touch $W_{qk}$ at all, so it is invisible in any product-level metric — which is
exactly why the product-level metric is the one to quote.

Note also that **global scale is not a gauge**: $\times 2$ and $\times 0.25$ both change the
loss. Cosine quotients it out by choice, not necessity, so the table prints $\|\cdot\|_F$ next
to it.
""")

co(r"""
print('Sanity check: the softmax gauge really is exact for this construction.')
for layer in (1, 2):
    W = CON[f'W_qk{layer}']
    rows, groups = key_groups(layer)
    W2 = W.copy()
    W2[rows, groups[0]] += 137.0          # one constant added to one complete key group
    W2[rows, groups[1]] += -412.0
    raw = float(W.ravel() @ W2.ravel() / (np.linalg.norm(W) * np.linalg.norm(W2)))
    print(f'  layer {layer}:  raw cosine after the shift {raw:>8.4f}      '
          f'gauge-fixed {gauge_cos(W, W2, layer):.6f}')


def rawcos(A, B):
    a, b = np.asarray(A, dtype=float).ravel(), np.asarray(B, dtype=float).ravel()
    n = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / n) if n > 0 else 0.0


print()
hdr = (f'{"alpha":>6}{"L1 raw":>10}{"L1 fixed":>10}{"L1 |F| gd / con":>26}'
       f'{"L2 raw":>10}{"L2 fixed":>10}{"L2 |F| gd / con":>26}')
print(hdr); print('-' * len(hdr))
COS = {}
for t in THRS:
    row, COS[t] = [f'{t:>6}'], {}
    for layer in (1, 2):
        g, c = GD[t][f'W_qk{layer}'], CON[f'W_qk{layer}']
        COS[t][layer] = gauge_cos(g, c, layer)
        row.append(f'{rawcos(g, c):>+10.4f}{COS[t][layer]:>+10.4f}'
                   f'{np.linalg.norm(gauge_fix(g, layer)):>14.4g} /'
                   f'{np.linalg.norm(gauge_fix(c, layer)):>11.4g}')
    print(''.join(row))

print('\nThe COMBINED-W_qk runs of folder 18, for comparison (CLAUDE.md 8.8):')
for t, (c1, c2) in REF_COS.items():
    print(f'   alpha={t}   layer 1 {c1:.4f}   layer 2 {c2:.4f}')
print('\nalpha=1.0 is the control: with no noise the suppression block is unconstrained, so a\n'
      'low score there means "not comparable", not "poor agreement".')
""")

# ---------------------------------------------------------------------------------------- 10
md(r"""
## 9. Comparing $W_q$ and $W_k$ *properly* — the balanced canonical form

The only gauge-free way to name $W_q$ and $W_k$ separately is to pick one representative of the
$GL$ orbit. Take the SVD of the product, $W_{qk} = U\Sigma V^\top$, and set

$$\widehat W_q = U\Sigma^{1/2}, \qquad \widehat W_k = V\Sigma^{1/2} \qquad\Longrightarrow\qquad
\widehat W_q\widehat W_k^\top = W_{qk}.$$

This is the *balanced* representative ($\widehat W_q^\top\widehat W_q = \widehat W_k^\top
\widehat W_k = \Sigma$). It removes the entire factorisation gauge except for rotations inside
equal-singular-value subspaces and column sign flips, which one orthogonal Procrustes step mops
up:

$$R^\star = \arg\min_{R^\top R = I}\ \|\widehat W_q^{\rm gd}R - \widehat W_q^{\rm con}\|_F^2
+ \|\widehat W_k^{\rm gd}R - \widehat W_k^{\rm con}\|_F^2 ,$$

solved by the SVD of $(\widehat W_q^{\rm gd})^\top \widehat W_q^{\rm con} +
(\widehat W_k^{\rm gd})^\top \widehat W_k^{\rm con}$.

**What to expect, and why it is not circular.** Once the gauge is fixed the separate matrices
carry *no information beyond the product*, so aligned agreement on $\widehat W_q$/$\widehat W_k$
should track product agreement — and if it does, that is the point being demonstrated rather
than a bug. The informative, alignment-free output of this section is the **singular spectrum**:
the construction's layer-2 circuit is rank 5, and how much of GD's spectral energy sits in that
top-5 subspace is a real statement about whether GD found a low-rank induction head or spread
it out.
""")

co(r"""
def canonical(W_qk, r=None, tol=1e-9):
    '''balanced representative: W_qk = U S V^T -> (U sqrt(S), V sqrt(S), S)'''
    A = np.asarray(W_qk, dtype=np.float64)
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    if r is None:
        r = int((S > tol * max(S[0], 1e-300)).sum())
    r = max(int(r), 1)
    rt = np.sqrt(S[:r])
    return U[:, :r] * rt, Vt[:r].T * rt, S


def procrustes(Aq, Ak, Bq, Bk):
    '''orthogonal R minimising ||Aq R - Bq||^2 + ||Ak R - Bk||^2'''
    U, _, Vt = np.linalg.svd(Aq.T @ Bq + Ak.T @ Bk)
    return U @ Vt


hdr = (f'{"alpha":>6}{"layer":>6}{"rank(con)":>11}{"GD energy in top-r":>21}'
       f'{"cos Wq raw":>12}{"cos Wq aligned":>16}{"cos Wk aligned":>16}{"cos W_qk fixed":>16}')
print(hdr); print('-' * len(hdr))
ALIGNED = {}
for t in THRS:
    for layer in (1, 2):
        cq, ck, cs = canonical(CON[f'W_qk{layer}'])
        r = cq.shape[1]
        gq, gk, gs = canonical(GD[t][f'W_qk{layer}'], r=r)
        R = procrustes(gq, gk, cq, ck)
        gqa, gka = gq @ R, gk @ R
        ALIGNED[(t, layer)] = (gqa, gka, cq, ck, gs, cs)
        top = float((gs[:r] ** 2).sum() / (gs ** 2).sum())
        h = min(GD[t][f'layer{layer}.W_q'].shape[1], CON[f'Wq{layer}'].shape[1])
        raw = rawcos(GD[t][f'layer{layer}.W_q'][:, :h], CON[f'Wq{layer}'][:, :h])
        print(f'{t:>6}{layer:>6}{r:>11}{100 * top:>20.2f}%'
              f'{raw:>+12.4f}{rawcos(gqa, cq):>+16.4f}{rawcos(gka, ck):>+16.4f}'
              f'{COS[t][layer]:>+16.4f}')
print('\n"GD energy in top-r" is the share of GD\'s squared singular values inside the\n'
      'construction\'s own rank. It is gauge-free and needs no alignment at all -- but it only\n'
      'says something for LAYER 2, whose construction is rank 5 out of 216. Layer 1\'s\n'
      'construction is rank 100 (the full position ramp), so "inside the top 100" is vacuous.')
print('\n"cos Wq raw" is the number the gauge makes meaningless: it compares GD\'s W_q with the\n'
      'construction\'s in whatever basis each happened to land in. Landing near 0 is the\n'
      'expected result, not a finding. The aligned columns are the honest version.')
""")

co(r"""
for t in THRS:
    fig, ax = plt.subplots(2, 4, figsize=(18, 8.2))
    for r, layer in enumerate((1, 2)):
        gqa, gka, cq, ck, gs, cs = ALIGNED[(t, layer)]
        heat(ax[r, 0], gqa, f'GD  $\\widehat W_q^{{({layer})}}$ aligned')
        heat(ax[r, 1], cq, f'construction  $\\widehat W_q^{{({layer})}}$')
        heat(ax[r, 2], gka, f'GD  $\\widehat W_k^{{({layer})}}$ aligned')
        heat(ax[r, 3], ck, f'construction  $\\widehat W_k^{{({layer})}}$')
    fig.suptitle(f'$\\alpha$ = {t}   balanced canonical form, gauge fixed and Procrustes '
                 f'aligned   (rows = input dims, columns = singular directions)', fontsize=13)
    plt.tight_layout(); plt.show()

fig, ax = plt.subplots(1, 2, figsize=(14, 4.6))
for layer, a in zip((1, 2), ax):
    cs = ALIGNED[(THRS[0], layer)][5]
    a.semilogy(np.maximum(cs / cs[0], 1e-18), 'k--', lw=1.8, label='construction')
    for t in THRS:
        gs = ALIGNED[(t, layer)][4]
        a.semilogy(np.maximum(gs / gs[0], 1e-18), lw=1.4, label=f'GD $\\alpha$={t}')
    a.set_title(f'layer {layer} — singular spectrum of $W_{{qk}}$, normalised', fontsize=11)
    a.set_xlabel('index'); a.set_ylabel('$\\sigma_i/\\sigma_0$')
    a.set_xlim(-0.5, 24.5); a.set_ylim(1e-8, 2); a.grid(alpha=.3); a.legend(fontsize=8)
plt.tight_layout(); plt.show()
""")

# ---------------------------------------------------------------------------------------- 11
md(r"""
## 10. Evolution — when does each layer form?

Rebuilt from `metrics_*.json` -- the training script records the gauge-fixed cosine and the
gauge-fixed $\|W_{qk}\|_F$ at every step into `history`, so this needs no weight snapshots.
(It used to read per-step `w_layer*.npy` arrays; those cost ~185 MB per threshold and were
removed, since the two scalars below are all the section ever plotted from them.)

In the combined runs layer 1
had **identically zero** gradient at $W_{qk}=0$ and could not move until layer 2 had left zero
(CLAUDE.md §8.9). Under the factorisation the origin is a saddle for *both* layers at once, so
the ordering has to be re-measured rather than assumed.

The combined runs also showed layer 2 peaking at $\approx 0.99$ cosine by step 12 and then
*falling* to 0.83 while the loss kept improving — the open question of CLAUDE.md §9. The same
curve is plotted here.
""")

co(r"""
fig, ax = plt.subplots(2, len(THRS), figsize=(5.0 * len(THRS), 8), squeeze=False)
for j, t in enumerate(THRS):
    h = MET[t]['history']
    cos = {1: np.asarray(h['L1_cos']), 2: np.asarray(h['L2_cos'])}
    nrm = {1: np.asarray(h.get('L1_fro_gf', [])), 2: np.asarray(h.get('L2_fro_gf', []))}
    # history['step'] starts at 0 (the init); the per-step series start after step 1
    steps = np.asarray(h['step'])[1:len(cos[1]) + 1]

    a = ax[0, j]
    for layer, col in ((1, '#2f6fc2'), (2, '#c2492f')):
        a.plot(steps, cos[layer][:len(steps)], lw=1.6, color=col, label=f'layer {layer}')
        kb = int(np.argmax(cos[layer][:len(steps)]))
        a.plot(steps[kb], cos[layer][kb], 'o', ms=5, color=col)
        a.annotate(f'  peak {cos[layer][kb]:.3f} @ {steps[kb]}',
                   (steps[kb], cos[layer][kb]), fontsize=8, color=col)
    a.axhline(0, color='0.7', lw=.8); a.grid(alpha=.3); a.legend(fontsize=8)
    a.set_title(f'$\\alpha$ = {t}   gauge-fixed cosine vs step', fontsize=11)
    a.set_xlabel('step')
    if j == 0:
        a.set_ylabel('cosine')

    b = ax[1, j]
    if nrm[1].size:
        for layer, col in ((1, '#2f6fc2'), (2, '#c2492f')):
            b.semilogy(steps, np.maximum(nrm[layer][:len(steps)], 1e-18), lw=1.6, color=col,
                       label=f'layer {layer}')
            b.axhline(np.linalg.norm(gauge_fix(CON[f'W_qk{layer}'], layer)), ls='--', lw=1.0,
                      color=col, alpha=.6)
        b.legend(fontsize=8)
        b.set_title('gauge-fixed $\\|W_{qk}\\|_F$   (dashed = construction)', fontsize=10)
    else:
        b.set_title('no L1_fro_gf / L2_fro_gf in history -- rerun the training script',
                    fontsize=10)
    b.grid(alpha=.3); b.set_xlabel('step')
plt.tight_layout(); plt.show()
""")

# ---------------------------------------------------------------------------------------- 12
md("## 11. Summary")

co(r"""
print(f'run: {RUN}\n')
hdr = (f'{"alpha":>6}{"best test":>11}{"folder-18 GD":>14}{"ideal":>9}{"gap":>9}   '
       f'{"cos L1":>8}{"cos L2":>8}   {"combined run L1 / L2":>22}')
print(hdr); print('-' * len(hdr))
for t in THRS:
    m, r = MET[t], REF.get(t, {})
    gap = m['best_test'] - r['ideal'] if r else float('nan')
    rc = REF_COS.get(t, (float('nan'), float('nan')))
    print(f'{t:>6}{m["best_test"]:>11.5f}{r.get("gd", float("nan")):>14.5f}'
          f'{r.get("ideal", float("nan")):>9.5f}{gap:>9.4f}   '
          f'{COS[t][1]:>+8.4f}{COS[t][2]:>+8.4f}   {f"{rc[0]:.4f} / {rc[1]:.4f}":>22}')

print('''
How to read this
----------------
* 'best test' vs 'folder-18 GD' answers: does the factored parametrisation reach the same
  solution QUALITY as the combined one? Same architecture, same data, same loss, same frozen
  W_ov -- only the parametrisation and the learning rate differ.
* 'cos L1 / L2' vs 'combined run L1 / L2' answers the harder question: does it reach the same
  PLACE. A matching loss with a lower cosine means GD found a different circuit of equal
  quality, which is a stronger statement than either number on its own.
* alpha = 1.0 is the control -- no noise exists, so layer 2's suppression block is
  unconstrained and its cosine is not comparable to the other two.
* Section 7 is the part folders 18/19 could not ask: with the block mask gone, the fraction of
  gauge-fixed energy that GD still leaves inside the construction's blocks.
''')
""")

nb = {'cells': [{'cell_type': k, 'metadata': {},
                 'source': (s + '\n').splitlines(keepends=True),
                 **({'execution_count': None, 'outputs': []} if k == 'code' else {})}
                for k, s in C],
      'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python',
                                  'name': 'python3'},
                   'language_info': {'name': 'python', 'version': '3'}},
      'nbformat': 4, 'nbformat_minor': 5}

out = sys.argv[1] if len(sys.argv) > 1 else 'analysis_separate_qk.ipynb'
with open(out, 'w') as fh:
    json.dump(nb, fh, indent=1)
print(f'wrote {out}   ({len(C)} cells: '
      f'{sum(1 for k, _ in C if k == "code")} code, '
      f'{sum(1 for k, _ in C if k == "markdown")} markdown)')
