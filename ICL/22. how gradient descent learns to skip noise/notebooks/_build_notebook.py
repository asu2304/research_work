"""Builds notebooks/analysis.ipynb from the outputs in runs/ and figures/, then executes it.
Re-run after any experiment:  python notebooks/_build_notebook.py"""
import os
import nbformat as nbf
from nbclient import NotebookClient

HERE = os.path.dirname(os.path.abspath(__file__))
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
cells = [
md("""# How gradient descent learns to skip noise — analysis

Everything here reads finished outputs in `../runs/` and `../figures/`; nothing is retrained.
The claims and their status (exact / verified / refuted / open) are in `../README.md`; the proofs
in `../proofs/1_exact_statements.pdf`. Re-generate with `python notebooks/_build_notebook.py`."""),
code("""import json, os, sys, glob
import numpy as np
from IPython.display import Image, display, Markdown
F22 = os.path.abspath('..'); R = os.path.join(F22, 'runs'); FIG = os.path.join(F22, 'figures')
sys.path.insert(0, os.path.join(F22, 'src'))
import five as M
S = json.load(open(os.path.join(R, 'summary.json')))
def table(rows, cols, fmt):
    print('  '.join(f'{c:>{w}}' for c, w in zip(cols, [len(c) + 2 for c in cols])))
    for r in rows:
        print('  '.join(f(r).rjust(len(c) + 2) for c, f in zip(cols, fmt)))"""),
md("""## 1. Is the five-number model faithful? (E0, E1)

E0: the direct five-scalar rewrite reproduces folder 19 exactly. E1: folder 18's 10,060-entry
solution, projected onto the five numbers **without using the loss**, keeps its loss when the
position slope is read from the near field — but not with a global least-squares slope, because
GD's free position profile is a capped ramp."""),
code("""e0 = json.load(open(os.path.join(R, 'e0_verify.json')))
table(e0, ['alpha', 'CE folder19 class', 'CE direct', '|diff|'],
      [lambda r: f"{r['alpha']}", lambda r: f"{r['ce_f19_class']:.6f}", lambda r: f"{r['ce_direct']:.6f}", lambda r: f"{r['abs_diff']:.1e}"])
print()
e1 = json.load(open(os.path.join(R, 'e1b_functional_projection.json')))
table(list(e1.values()), ['alpha', 'folder18', 'five-trained(200)', 'naive LS', 'near-field', 'KL-matched'],
      [lambda r: f"{r['alpha']}", lambda r: f"{r['ce_folder18']:.5f}", lambda r: f"{r['ce_five_trained']:.5f}",
       lambda r: f"{r['ce_naive_ls']:.5f}", lambda r: f"{r['ce_nearfield']:.5f}", lambda r: f"{r['ce_functional']:.5f}"])
display(Image(os.path.join(FIG, 'f7_position_profile.png')))"""),
md("""## 2. GD learns to skip at every noise density (E4)

19 values of α, 1000 full-batch Adam steps each from zero."""),
code("""table(S['main'], ['alpha', 'test CE', 'gap to ideal', 'L1 state mass', 'succ|noise', 'succ|signal'],
      [lambda r: f"{r['alpha']}", lambda r: f"{r['test_ce_final']:.5f}", lambda r: f"{r['gap']:.4f}",
       lambda r: f"{r.get('final_layer1_state_mass', float('nan')):.3f}",
       lambda r: f"{r.get('final_succ_mass_postnoise', float('nan')):.3f}" if r['alpha'] < 1 else '   -',
       lambda r: f"{r.get('final_succ_mass_postsignal', float('nan')):.3f}"])
display(Image(os.path.join(FIG, 'f2_loss_gap_and_success.png')))
display(Image(os.path.join(FIG, 'f1_five_numbers_vs_alpha.png')))"""),
md("""## 3. How the state weight `c2` ends up positive (E2, E3, E5)

**Phase 1** (layer 1 uniform): `c2` is pulled up at every α — by keys that follow the query's
current state (a diluted state-matching signal; an *order* effect, killed by shuffling).
**Phase 2** (layer 1 learns recency): post-signal queries turn against `c2`.
**Phase 3**: the two pulls cancel. With no noise (α=1) nothing balances them and `c2` returns to ≈0."""),
code("""display(Image(os.path.join(FIG, 'f3_c2_and_layer1_over_training.png')))
display(Image(os.path.join(FIG, 'f4_c2_tug_of_war.png')))
for a in ('0.6', '1.0'):
    d = json.load(open(os.path.join(R, 'main', f'anatomy_x_{a}.json')))
    print(f'\\nalpha={a}:  step   c2   pull(noise)  pull(signal)  L1 state mass')
    for r in d['rows']:
        if r['step'] in (0, 3, 7, 10, 20, 30, 100, 1000):
            print(f"        {r['step']:>5} {r['theta'][4]:5.2f}  {-r['grad_noise'][4]:+.2e}    {-r['grad_signal'][4]:+.2e}     {r['layer1_state_mass']:.3f}")"""),
code("""e3 = json.load(open(os.path.join(R, 'e3_phase1_origin.json')))
print('-dL/dc2 at init: real sequences vs tokens shuffled within each sequence')
for a, rows in e3.items():
    r = rows[0]; print(f"  alpha={a}: real {r['neg_grad_real'][4]:+.2e}   shuffled {r['neg_grad_shuffled'][4]:+.2e}"
                      f"   | a2: real {r['neg_grad_real'][2]:+.2e} shuffled {r['neg_grad_shuffled'][2]:+.2e}")
display(Image(os.path.join(FIG, 'f6_initial_pull_on_c2.png')))"""),
code("""e5 = json.load(open(os.path.join(R, 'e5_covariance_law.json')))
print('Covariance law (Prop. 3): formula vs autograd, and the hit rates behind the c2 tug-of-war')
for r in e5:
    print(f"alpha={r['alpha']} step={r['step']}: c2 formula {r['c2_formula']:+.3e} autograd {r['c2_autograd']:+.3e}")
    for t in ('postnoise', 'postsignal'):
        if f'c2_hit_favoured_{t}' in r:
            print(f"     {t:10s} keys c2 favours are right {r[f'c2_hit_favoured_{t}']:.3f} vs current {r[f'c2_hit_now_{t}']:.3f}")"""),
md("""## 4. Who does what (freeze ablations)

`a1`+`p` **skip** (recover the state); `c2` **uses** it after noise; `b2` **teaches** layer 1 at
every α, while `c2` alone teaches it only when noise is dense."""),
code("""for tag in ('freeze_c2', 'freeze_b2', 'freeze_a1', 'freeze_p'):
    print(f'\\n{tag}')
    table(S[tag], ['alpha', 'test CE', 'L1 state mass', 'succ|noise', 'succ|signal'],
          [lambda r: f"{r['alpha']}", lambda r: f"{r['test_ce_final']:.5f}",
           lambda r: f"{r.get('final_layer1_state_mass', float('nan')):.3f}",
           lambda r: f"{r.get('final_succ_mass_postnoise', float('nan')):.3f}",
           lambda r: f"{r.get('final_succ_mass_postsignal', float('nan')):.3f}"])
display(Image(os.path.join(FIG, 'f5_freeze_ablations.png')))"""),
md("""## 5. The skip horizon tracks the noise

`H = 1 + |a1|/p` is the farthest a signal key can be and still beat an adjacent noise key.
At any fixed step, `alpha * H` is nearly constant across alpha: the horizon is a fixed multiple of
the mean distance back to the last signal. The multiple grows slowly with training. Not derived."""),
code("""steps = (50, 100, 200, 500, 1000)
print('alpha ' + ''.join(f'{s:>8}' for s in steps))
for f in sorted(glob.glob(os.path.join(R, 'main', 'traj_x_*.npz')), key=lambda f: float(os.path.basename(f)[7:-4])):
    a = float(os.path.basename(f)[7:-4])
    if a >= 1: continue
    th = np.load(f)['theta']
    print(f'{a:<6}' + ''.join(f'{a * (1 + abs(th[s, 1]) / th[s, 0]):8.2f}' for s in steps))"""),
]
nb = nbf.v4.new_notebook(); nb['cells'] = cells
nb['metadata']['kernelspec'] = {'name': 'python3', 'display_name': 'Python 3', 'language': 'python'}
out = os.path.join(HERE, 'analysis.ipynb')
NotebookClient(nb, timeout=600, kernel_name='python3', resources={'metadata': {'path': HERE}}).execute()
nbf.write(nb, out)
print('wrote', out)
