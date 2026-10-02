"""
E6 -- Aggregate everything in runs/ into one summary JSON and the figures in figures/.

Reads only folder-22 outputs (and folder 18's checkpoint for the projection profile).
Figures are static PNGs for the paper/README. Palette: the dataviz reference palette's
categorical slots in fixed order (blue, orange, aqua, yellow) and its sequential blue ramp for
alpha; every multi-series panel has a legend; one y-axis per panel.
"""
import glob, json, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
os.environ.setdefault('MPLCONFIGDIR', os.path.expanduser('~/.cache/ashu_tmp'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import five as M

R = os.path.join(M.F22, 'runs'); FIG = os.path.join(M.F22, 'figures')
os.makedirs(FIG, exist_ok=True)
SLOT = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4']   # categorical, fixed order
SEQ = ['#86b6ef', '#6da7ec', '#5598e7', '#3987e5', '#2a78d6', '#256abf', '#1c5cab', '#184f95', '#104281', '#0d366b']
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e4e3df'
plt.rcParams.update({'font.size': 10, 'axes.edgecolor': INK2, 'axes.labelcolor': INK,
                     'xtick.color': INK2, 'ytick.color': INK2, 'axes.grid': True,
                     'grid.color': GRID, 'grid.linewidth': 0.6, 'axes.spines.top': False,
                     'axes.spines.right': False, 'figure.facecolor': '#fcfcfb',
                     'axes.facecolor': '#fcfcfb', 'lines.linewidth': 2, 'legend.frameon': False})


def seqcol(k, n):
    return SEQ[int(round(k * (len(SEQ) - 1) / max(n - 1, 1)))]


def load_tag(tag):
    T, A = {}, {}
    for f in glob.glob(os.path.join(R, tag, 'traj_x_*.npz')):
        a = float(os.path.basename(f)[7:-4]); T[a] = np.load(f)
    for f in glob.glob(os.path.join(R, tag, 'anatomy_x_*.json')):
        a = float(os.path.basename(f)[10:-5]); A[a] = json.load(open(f))
    return T, A


T, A = load_tag('main')
alphas = sorted(T)
summary = {'main': []}
for a in alphas:
    th = T[a]['theta']; te = T[a]['test']
    row = dict(alpha=a, test_ce_final=float(te[-1]), ideal=M.ideal_ce(a),
               gap=float(te[-1] - M.ideal_ce(a)),
               theta_step200=dict(zip(M.NAMES, np.round(th[200], 4).tolist())),
               theta_final=dict(zip(M.NAMES, np.round(th[-1], 4).tolist())),
               c2_peak_first100=float(th[:100, 4].max()), c2_peak_step=int(th[:100, 4].argmax()))
    if a < 1:
        row['skip_horizon_step200'] = float(1 + abs(th[200, 1]) / th[200, 0])
        row['skip_horizon_final'] = float(1 + abs(th[-1, 1]) / th[-1, 0])
    if a in A:
        rows = A[a]['rows']
        sig = [(r['step'], -r['grad_signal'][4]) for r in rows]
        flip = next((s for s, v in sig if s > 0 and v < 0), None)
        row['c2_signal_pull_flip_step'] = flip
        if flip is not None:
            row['layer1_state_mass_at_flip'] = next(r['layer1_state_mass'] for r in rows if r['step'] == flip)
        last = rows[-1]
        row['final_layer1_state_mass'] = last['layer1_state_mass']
        row['final_succ_mass_postnoise'] = last['layer2_succ_mass_postnoise']
        row['final_succ_mass_postsignal'] = last['layer2_succ_mass_postsignal']
        if a < 1:
            row['final_c2_pull_noise'] = -last['grad_noise'][4]
            row['final_c2_pull_signal'] = -last['grad_signal'][4]
            row['final_c2_pull_ratio'] = -last['grad_noise'][4] / last['grad_signal'][4]
    summary['main'].append(row)

for tag in ('freeze_c2', 'freeze_b2', 'freeze_a1', 'freeze_p'):
    Tf, Af = load_tag(tag)
    summary[tag] = []
    for a in sorted(Tf):
        r = dict(alpha=a, test_ce_final=float(Tf[a]['test'][-1]),
                 theta_final=dict(zip(M.NAMES, np.round(Tf[a]['theta'][-1], 4).tolist())))
        if a in Af:
            last = Af[a]['rows'][-1]
            r.update(final_layer1_state_mass=last['layer1_state_mass'],
                     final_succ_mass_postnoise=last['layer2_succ_mass_postnoise'],
                     final_succ_mass_postsignal=last['layer2_succ_mass_postsignal'])
        summary[tag].append(r)
json.dump(summary, open(os.path.join(R, 'summary.json'), 'w'), indent=1)

# ------------------------------------------------------------------- figure 1: five numbers vs alpha
fig, axs = plt.subplots(1, 5, figsize=(17, 3.4))
lab = {'p': 'p  (recency slope)', 'a1': 'a1  (layer-1 noise veto)', 'a2': 'a2  (layer-2 noise veto)',
       'b2': 'b2  (induction)', 'c2': 'c2  (state)'}
for k, n in enumerate(M.NAMES):
    ax = axs[k]
    for s, col, nm in ((200, SLOT[1], 'step 200'), (1000, SLOT[0], 'step 1000')):
        ax.plot(alphas, [T[a]['theta'][s, k] for a in alphas], 'o-', color=col, ms=4, label=nm)
    ax.set_title(lab[n], fontsize=10); ax.set_xlabel(r'$\alpha$ (signal density)')
    ax.axhline(0, color=INK2, lw=0.8)
axs[0].set_ylabel('value (logits)'); axs[0].legend()
fig.suptitle('The five numbers GD learns, as a function of noise density (1 - alpha)', x=0.01, ha='left')
fig.tight_layout(); fig.savefig(os.path.join(FIG, 'f1_five_numbers_vs_alpha.png'), dpi=150); plt.close(fig)

# ------------------------------------------- figure 2: loss gap and post-noise / post-signal success
fig, axs = plt.subplots(1, 2, figsize=(11, 3.6))
axs[0].plot(alphas, [r['gap'] for r in summary['main']], 'o-', color=SLOT[0], ms=4)
axs[0].set_title('test CE minus ideal-alpha (step 1000)', fontsize=10)
axs[0].set_xlabel(r'$\alpha$'); axs[0].set_ylabel('nats')
an = [r for r in summary['main'] if 'final_succ_mass_postnoise' in r and r['alpha'] < 1]
axs[1].plot([r['alpha'] for r in an], [r['final_succ_mass_postnoise'] for r in an], 'o-', color=SLOT[0], ms=4, label='after a noise token')
axs[1].plot([r['alpha'] for r in an], [r['final_succ_mass_postsignal'] for r in an], 'o-', color=SLOT[1], ms=4, label='after a signal token')
axs[1].set_title("layer-2 mass on the true state's successors", fontsize=10)
axs[1].set_xlabel(r'$\alpha$'); axs[1].set_ylabel('probability mass'); axs[1].legend(loc='lower left')
fig.tight_layout(); fig.savefig(os.path.join(FIG, 'f2_loss_gap_and_success.png'), dpi=150); plt.close(fig)

# ------------------------------------------------------------------ figure 3: c2 trajectories
sel = [a for a in (0.2, 0.4, 0.6, 0.8, 0.9, 0.99, 1.0) if a in T]
fig, axs = plt.subplots(1, 2, figsize=(12, 3.8))
for k, a in enumerate(sel):
    th = T[a]['theta']; st = np.arange(len(th))
    axs[0].plot(st + 1, th[:, 4], color=seqcol(k, len(sel)), label=f'alpha={a}')
    if a in A:
        rr = A[a]['rows']
        axs[1].plot([r['step'] + 1 for r in rr], [r['layer1_state_mass'] for r in rr], 'o-', ms=3,
                    color=seqcol(k, len(sel)), label=f'alpha={a}')
axs[0].set_xscale('log'); axs[0].set_xlabel('step + 1'); axs[0].set_ylabel('c2 (logits)')
axs[0].set_title('state weight c2 over training', fontsize=10); axs[0].axhline(0, color=INK2, lw=0.8)
axs[0].legend(fontsize=8, ncol=2)
axs[1].set_xscale('log'); axs[1].set_xlabel('step + 1'); axs[1].set_ylabel('mass on last signal before i')
axs[1].set_title('layer 1 learning to skip noise (state recovery)', fontsize=10)
fig.tight_layout(); fig.savefig(os.path.join(FIG, 'f3_c2_and_layer1_over_training.png'), dpi=150); plt.close(fig)

# ------------------------------------------------------------- figure 4: the tug-of-war on c2
sel = [a for a in (0.4, 0.6, 0.8, 1.0) if a in A]
fig, axs = plt.subplots(1, len(sel), figsize=(4.2 * len(sel), 3.6), sharey=True)
for ax, a in zip(np.atleast_1d(axs), sel):
    rr = A[a]['rows']; s = np.array([r['step'] for r in rr]) + 1
    ax.plot(s, [-r['grad_noise'][4] for r in rr], 'o-', ms=3, color=SLOT[0], label='pull from post-noise queries')
    ax.plot(s, [-r['grad_signal'][4] for r in rr], 'o-', ms=3, color=SLOT[1], label='pull from post-signal queries')
    ax.set_xscale('log'); ax.set_yscale('symlog', linthresh=1e-5); ax.axhline(0, color=INK2, lw=0.8)
    ax.set_title(f'alpha = {a}', fontsize=10); ax.set_xlabel('step + 1')
    m = [r['layer1_state_mass'] for r in rr]
    cross = next((r['step'] + 1 for r in rr if r['layer1_state_mass'] >= 0.5), None)
    if cross:
        ax.axvline(cross, color=INK2, lw=0.8, ls='--')
        ax.text(cross * 1.1, ax.get_ylim()[1] * 0.3, 'layer 1\nstate mass 0.5', fontsize=7, color=INK2)
np.atleast_1d(axs)[0].set_ylabel('-dL/dc2 contribution (symlog)')
np.atleast_1d(axs)[0].legend(fontsize=8, loc='lower left')
fig.suptitle('Tug-of-war on the state weight: post-noise queries pull c2 up, post-signal queries pull it down', x=0.01, ha='left')
fig.tight_layout(); fig.savefig(os.path.join(FIG, 'f4_c2_tug_of_war.png'), dpi=150); plt.close(fig)

# ---------------------------------------------------------- figure 5: freeze ablations
tags = [('main', 'nothing frozen'), ('freeze_c2', 'c2 frozen at 0'), ('freeze_b2', 'b2 frozen at 0'),
        ('freeze_a1', 'a1 frozen at 0'), ('freeze_p', 'p frozen at 0')]
fa = [0.2, 0.4, 0.6, 0.8]
fig, axs = plt.subplots(1, 3, figsize=(15, 3.8), sharey=True)
for j, (key, ttl) in enumerate((('final_layer1_state_mass', 'layer 1: mass on true last signal (the skip)'),
                                ('final_succ_mass_postnoise', 'layer 2 after NOISE: mass on correct successors'),
                                ('final_succ_mass_postsignal', 'layer 2 after SIGNAL: mass on correct successors'))):
    for k, (tag, nm) in enumerate(tags):
        rows = summary['main'] if tag == 'main' else summary.get(tag, [])
        d = {r['alpha']: r.get(key) for r in rows}
        xs = [a for a in fa if d.get(a) is not None]
        if xs:
            axs[j].plot(xs, [d[a] for a in xs], 'o-', ms=4, color=SLOT[k] if k < 4 else '#4a3aa7', label=nm)
    axs[j].set_title(ttl, fontsize=9.5); axs[j].set_xlabel(r'$\alpha$')
axs[0].set_ylabel('probability mass'); axs[0].legend(fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(FIG, 'f5_freeze_ablations.png'), dpi=150); plt.close(fig)

# ------------------------------------------------- figure 6: what carries the very first pull on c2
p1 = os.path.join(R, 'e3b_phase1_pairs.json')
if os.path.exists(p1):
    d = json.load(open(p1))
    cats = ['pred==both', 'pred==state_i', 'pred==x_i', 'other']
    nice = ["key's predecessor = state = x_i", "key's predecessor = state", "key's predecessor = x_i", 'other keys']
    sel = [a for a in ('0.2', '0.6', '1.0') if a in d]
    fig, axs = plt.subplots(1, len(sel), figsize=(4.6 * len(sel), 3.4))
    for ax, a in zip(np.atleast_1d(axs), sel):
        v = [d[a]['c2']['by_predecessor'][c] for c in cats]
        ax.barh(range(4), v, color=SLOT[0], height=0.6)
        ax.set_yticks(range(4)); ax.set_yticklabels(nice, fontsize=8); ax.axvline(0, color=INK2, lw=0.8)
        ax.set_title(f'alpha={a}: -dL/dc2 at init = {d[a]["c2"]["autograd"]:+.1e}', fontsize=9)
        ax.set_xlabel('contribution')
    fig.suptitle('At initialisation, the pull on c2 comes from keys that follow the current state', x=0.01, ha='left')
    fig.tight_layout(); fig.savefig(os.path.join(FIG, 'f6_initial_pull_on_c2.png'), dpi=150); plt.close(fig)

# ------------------------------------------- figure 7: folder 18's free position profile vs linear
p2 = os.path.join(R, 'e1_projection.json')
if os.path.exists(p2):
    d = json.load(open(p2))
    fig, ax = plt.subplots(figsize=(6, 3.6))
    for k, a in enumerate(('0.6', '0.8')):
        prof = d[a]['position_profile_rel_d1']
        ax.plot(range(1, len(prof) + 1), prof, 'o-', ms=3, color=SLOT[k], label=f'GD, 10,060 free entries, alpha={a}')
        pf = T[float(a)]['theta'][200, 0] if float(a) in T else None
    th = M.f19_final(0.6)
    ax.plot(range(1, 16), [-float(th[0]) * (dd - 1) for dd in range(1, 16)], '--', color=INK2, lw=1.2,
            label='five-number linear ramp (alpha=0.6, step 200)')
    ax.set_xlabel('distance i - j'); ax.set_ylabel('score relative to distance 1 (logits)')
    ax.set_title("GD's free position profile is a capped ramp", fontsize=10); ax.legend(fontsize=8)
    ax.set_ylim(-25, 2)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, 'f7_position_profile.png'), dpi=150); plt.close(fig)

print(json.dumps({k: (v if k != 'main' else f'{len(v)} alphas') for k, v in summary.items()}, indent=1)[:3000])
for r in summary['main']:
    print(f"alpha={r['alpha']:<6} gap {r['gap']:.4f}  c2 peak {r['c2_peak_first100']:.2f}@{r['c2_peak_step']}  "
          f"flip {r.get('c2_signal_pull_flip_step')}  L1mass@flip {r.get('layer1_state_mass_at_flip', float('nan')):.3f}  "
          f"c2 final {r['theta_final']['c2']:.2f}  ratio {r.get('final_c2_pull_ratio', float('nan')):.3f}  "
          f"H200 {r.get('skip_horizon_step200', float('nan')):.2f}")
