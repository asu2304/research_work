#!/usr/bin/env python3
"""Render the weight/gradient evolution of a master_train.py run as an MP4 video.

Standalone: reads only the per-step snapshots the run wrote (w_*.npy, g_*.npy, mask_*.npy,
snapmeta_*.json) plus metrics_*.json for the loss curve. Nothing is imported from the
notebook or from master_train.py.

Each frame shows, for one optimizer step:

    top row      the trainable blocks of W_qk        (layer 1 first, then layer 2)
    bottom row   the exact full-batch gradient at those same weights
    right        the loss over the animated window, with the current step marked

The four small blocks are ROW-CENTERED for display -- each query row's mean subtracted.
Adding a constant to a whole query row is an exact softmax invariance, so that offset carries
no behaviour, and it is what otherwise hides the structure: early in training the
state x state block sits at -6.6 on the diagonal and -14.2 off it, so every cell is negative
and the panel reads as one flat blue square. Centered, the diagonal preference is visible in
every frame. The 100x100 position block is left raw -- its structure shows without help.

Colour scales are FIXED across frames (per panel, computed over the whole window), so growth
you see is real and not the scale rescaling under you.

Writes H.264 MP4 via ffmpeg (yuv420p, faststart) so it plays in browsers, VLC and slides.
Pass --format gif for an animated GIF instead -- much larger for the same length.

Usage
-----
    python make_evolution_video.py                                # alpha 0.6, 50 steps, mp4
    python make_evolution_video.py --threshold 0.8 --steps 30
    python make_evolution_video.py --run runs/<other-run> --fps 6 --dpi 100
    python make_evolution_video.py --format gif                   # if you really want a GIF
    python make_evolution_video.py --list                         # what runs have snapshots
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')                       # no display on this box
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.colors import SymLogNorm

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_RUN = ('runs/master_opt-adam_lr-1.0_init-zeros_wov-fixed_blocks-relevant-strict'
               '_readout-bias_ep-500')
V, VS, T = 8, 4, 100
D1, D2 = V + T, 2 * (V + T)                 # 108, 216

# (parameter, row slice, col slice, label, annotate?)
BLOCKS = [
    ('layer1.W_qk', slice(0, V),       slice(0, V),        'L1  token x token',    True),
    ('layer1.W_qk', slice(V, D1),      slice(V, D1),       'L1  position x position', False),
    ('layer2.W_qk', slice(0, V),       slice(0, V),        'L2  noise suppression', True),
    ('layer2.W_qk', slice(0, VS),      slice(D1, D1 + VS), 'L2  induction head',   True),
    ('layer2.W_qk', slice(D1, D1 + VS), slice(D1, D1 + VS), 'L2  state x state',   True),
]

INK, GRID, MUTED = '#1a1a1a', '#d8d8d8', '#6b6b6b'
TRAIN_C, TEST_C = '#1f6fb4', '#e07b39'


def find_runs():
    out = []
    for d in sorted(glob.glob(os.path.join(HERE, 'runs', '*'))):
        if glob.glob(os.path.join(d, 'snapmeta_*.json')):
            out.append(os.path.relpath(d, HERE))
    return out


def load(run_dir, thr):
    """snapshots + loss history for one threshold, sliced to the rows actually written"""
    meta_p = os.path.join(run_dir, f'snapmeta_x_{thr}.json')
    if not os.path.exists(meta_p):
        sys.exit(f'no snapshots for alpha={thr} in {run_dir}\n'
                 f'  available: {sorted(os.path.basename(p) for p in glob.glob(os.path.join(run_dir, "snapmeta_*.json")))}')
    meta = json.load(open(meta_p))
    nw, ng = meta['w_rows_written'], meta['g_rows_written']
    W, G, MK = {}, {}, {}
    for p in meta['params']:
        MK[p] = np.load(os.path.join(run_dir, f'mask_{p}_x_{thr}.npy'))
        W[p] = np.load(os.path.join(run_dir, f'w_{p}_x_{thr}.npy'), mmap_mode='r')[:nw]
        G[p] = np.load(os.path.join(run_dir, f'g_{p}_x_{thr}.npy'), mmap_mode='r')[:ng]
    his = json.load(open(os.path.join(run_dir, f'metrics_seed0_x_{thr}.json')))['history']
    return meta, W, G, MK, his, nw, ng


def stack(meta, flat, mask, param, rs, cs, n, center):
    """rows 0..n-1 of one block, put back into full matrices; optionally row-centered"""
    shape = meta['shapes'][param]
    out = np.empty((n, rs.stop - rs.start, cs.stop - cs.start))
    for k in range(n):
        A = np.zeros(shape)
        A[mask] = flat[k]
        out[k] = A[rs, cs]
    if center:
        out -= out.mean(-1, keepdims=True)
    return out


def draw_block(ax, A, title, vmax, annot, nrm_cache={}):
    v = vmax if vmax > 0 else 1.0
    ax.imshow(A, cmap='RdBu_r', interpolation='nearest',
              norm=SymLogNorm(linthresh=v / 1e3, linscale=0.6, vmin=-v, vmax=v, base=10))
    if annot:
        lab = [['0' if z == 0 else (f'{z:.3g}' if abs(z) >= 0.01 else f'{z:.1e}') for z in row]
               for row in A]
        wid = max(len(x) for row in lab for x in row)
        # a 4x4 draws cells twice the width of an 8x8's, so it can carry a much larger font
        fs = (12.5 if A.shape[0] <= VS else 6.4) * min(1.0, 5.0 / wid)
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                ax.text(j, i, lab[i][j], ha='center', va='center', fontsize=fs,
                        fontweight='bold' if i == j else 'normal',
                        color='white' if abs(A[i, j]) > v / 3 else INK)
        if A.shape[0] <= VS:                                  # room to label the 4x4s
            ax.set_xticks(range(A.shape[1])); ax.set_yticks(range(A.shape[0]))
            ax.set_xticklabels([f'k{q}' for q in range(A.shape[1])], fontsize=8, color=MUTED)
            ax.set_yticklabels([f'q{q}' for q in range(A.shape[0])], fontsize=8, color=MUTED)
            ax.tick_params(length=0)
        else:
            ax.set_xticks([]); ax.set_yticks([])
            ax.axhline(VS - .5, color=INK, lw=.7, alpha=.55)
            ax.axvline(VS - .5, color=INK, lw=.7, alpha=.55)
    else:
        ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_edgecolor(GRID)
    ax.set_title(title, fontsize=10.5, color=INK, pad=6)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run', default=DEFAULT_RUN, help='run directory, relative to this file')
    ap.add_argument('--threshold', type=float, default=0.6)
    ap.add_argument('--steps', type=int, default=50, help='frames are steps 0..STEPS')
    ap.add_argument('--fps', type=float, default=4.0)
    ap.add_argument('--dpi', type=int, default=100)
    ap.add_argument('--format', choices=('mp4', 'gif'), default='mp4')
    ap.add_argument('--crf', type=int, default=20,
                    help='H.264 quality, lower is better (18 near-lossless, 28 small)')
    ap.add_argument('--outdir', default='artifacts')
    ap.add_argument('--name', default=None, help='output filename (default is built from args)')
    ap.add_argument('--list', action='store_true', help='list runs that have snapshots and exit')
    a = ap.parse_args()

    if a.list:
        rs = find_runs()
        print('runs with per-step snapshots:' if rs else 'no run has snapshots yet')
        for r in rs:
            print('   ', r)
        return

    run_dir = a.run if os.path.isabs(a.run) else os.path.join(HERE, a.run)
    thr = a.threshold
    meta, W, G, MK, his, nw, ng = load(run_dir, thr)
    n = min(a.steps + 1, nw, ng + 1)
    if n < 2:
        sys.exit(f'only {nw} weight rows and {ng} gradient rows recorded -- nothing to animate')
    print(f'run        : {os.path.basename(run_dir)}')
    print(f'alpha      : {thr}   recorded w={nw} g={ng}   animating steps 0..{n-1}')

    Wst = [stack(meta, W[p], MK[p], p, rs, cs, n, an) for p, rs, cs, _, an in BLOCKS]
    Gst = [stack(meta, G[p], MK[p], p, rs, cs, n, an) for p, rs, cs, _, an in BLOCKS]
    vW = [float(np.abs(x).max()) for x in Wst]
    vG = [float(np.abs(x).max()) for x in Gst]

    step = np.array(his['step'][:n])
    ltr = np.array(his['train_loss'][:n])
    lte = np.array(his['test_loss'][:n])
    lfin = his['test_loss'][-1]

    nb = len(BLOCKS)
    fig = plt.figure(figsize=(4.6 * nb + 5.0, 10.6), facecolor='white')
    gs = fig.add_gridspec(2, nb + 2, width_ratios=[1] * nb + [0.10, 0.95],
                          wspace=0.10, hspace=0.16,
                          left=0.025, right=0.985, top=0.90, bottom=0.055)
    axes = np.array([[fig.add_subplot(gs[r, c]) for c in range(nb)] for r in range(2)])
    ax_loss = fig.add_subplot(gs[:, nb + 1])

    def loss_panel(k):
        ax_loss.clear()
        ax_loss.plot(step, ltr, color=TRAIN_C, lw=2.0, label='train', zorder=3)
        ax_loss.plot(step, lte, color=TEST_C, lw=1.5, ls='--', label='test', zorder=3)
        ax_loss.axvline(k, color=MUTED, lw=1.1, alpha=0.8, zorder=2)
        ax_loss.plot([k], [ltr[k]], 'o', color=TRAIN_C, ms=9, zorder=5)
        ax_loss.plot([k], [lte[k]], 'o', color=TEST_C, ms=9, zorder=5)
        ax_loss.annotate(f'step {k}\ntrain {ltr[k]:.5f}\ntest  {lte[k]:.5f}',
                         xy=(0.96, 0.95), xycoords='axes fraction', ha='right', va='top',
                         fontsize=9, family='monospace', color=INK,
                         bbox=dict(boxstyle='round,pad=0.45', fc='white', ec=GRID, alpha=0.95))
        ax_loss.set_xlim(-0.5, n - 0.5)
        ax_loss.set_xlabel('optimizer step  (= epoch, full batch)', fontsize=9.5, color=INK)
        ax_loss.set_ylabel('cross entropy', fontsize=9.5, color=INK)
        ax_loss.set_title(f'loss over these {n-1} steps\n'
                          f'(the run ends at {lfin:.5f} after 500)', fontsize=10, color=INK)
        ax_loss.grid(alpha=0.28, color=GRID)
        ax_loss.tick_params(labelsize=8.5, colors=MUTED)
        for sp in ax_loss.spines.values():
            sp.set_edgecolor(GRID)
        ax_loss.legend(fontsize=8.5, loc='lower left', framealpha=0.9)

    def frame(k):
        for ax in axes.ravel():
            ax.clear()
        for j, (_, _, _, ttl, an) in enumerate(BLOCKS):
            tag = '   (row-centered)' if an else ''
            draw_block(axes[0][j], Wst[j][k], f'$W$   {ttl}{tag}', vW[j], an)
            draw_block(axes[1][j], Gst[j][k], f'$g$   {ttl}{tag}', vG[j], an)
        loss_panel(k)
        note = '        zero init: every weight is 0, and layer 1 has no gradient yet' if k == 0 else ''
        fig.suptitle(f'$W_{{qk}}$ and its gradient, step {k} / {n-1}      '
                     f'$\\alpha$={thr}      '
                     f'top: $W$ before update {k+1}      bottom: $g[{k}]$ at those weights'
                     f'{note}', fontsize=13.5, color=INK, y=0.965)
        return []

    outdir = a.outdir if os.path.isabs(a.outdir) else os.path.join(HERE, a.outdir)
    os.makedirs(outdir, exist_ok=True)
    name = a.name or f'evolution_x_{thr}_steps0-{n-1}.{a.format}'
    out = os.path.join(outdir, name)

    anim = animation.FuncAnimation(fig, frame, frames=n, blit=False)
    if a.format == 'mp4':
        if not animation.FFMpegWriter.isAvailable():
            sys.exit('ffmpeg not found -- install it, or pass --format gif')
        # yuv420p + even dimensions: what browsers and QuickTime actually decode.
        # faststart puts the index first so it streams instead of buffering the whole file.
        writer = animation.FFMpegWriter(
            fps=a.fps, codec='libx264', bitrate=-1,
            extra_args=['-crf', str(a.crf), '-pix_fmt', 'yuv420p', '-preset', 'slow',
                        '-movflags', '+faststart',
                        '-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2'])
    else:
        writer = animation.PillowWriter(fps=a.fps)
    anim.save(out, writer=writer, dpi=a.dpi, savefig_kwargs={'facecolor': 'white'})
    plt.close(fig)
    mb = os.path.getsize(out) / 2 ** 20
    print(f'wrote      : {os.path.relpath(out, HERE)}   {mb:.1f} MB   '
          f'{n} frames at {a.fps} fps, dpi {a.dpi}'
          + (f', crf {a.crf}' if a.format == 'mp4' else ''))


if __name__ == '__main__':
    main()
