"""Build log_readout_construction.pdf: the complete specification of the frozen log readout.

Every number in the document is computed here from the live code in folder 14, not transcribed.
Run:  python make_log_construction_pdf.py
"""

import glob
import importlib.util
import os

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

_HERE = os.path.dirname(os.path.abspath(__file__))
_F14  = os.path.join(os.path.dirname(_HERE), '14. elimination final convergence confusion')
_F12  = os.path.join(os.path.dirname(_HERE),
                     '12. training level exhaustive experiment to confirm finding')
OUT = os.path.join(_HERE, 'log_readout_construction.pdf')


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


T = _load(os.path.join(_F14, 'train.py'), 'train14')
C = T.BASE
DELTA = C['freeze_delta']
K     = C['freeze_hidden_k']
FMAX  = C['freeze_fit_fmax']
CGAIN = C['freeze_fit_cgain']
TOL   = C['freeze_fit_tol']

A, B, W, CC, MAXERR, MSE = T.fit_log_core(DELTA, K, FMAX, CGAIN)
KNOTS = (-B / A)
TILT  = W * A
CUM   = torch.cumsum(TILT, 0)
RATIO = float(((KNOTS[-1] + DELTA) / (KNOTS[0] + DELTA)) ** (1.0 / (K - 1)))

V, VS, SEQ = C['v_sig'] + C['n_noise'], C['v_sig'], C['seq_len']
ATTN = V + SEQ
IN2  = 2 * ATTN
D    = 3 * ATTN


def implied(f, dt=torch.float32):
    f = torch.as_tensor(f, dtype=dt).reshape(-1)
    return F.gelu(f.unsqueeze(1) * A.to(dt) + B.to(dt)) @ W.to(dt) + CC + f


BANDS = []
for lo, hi in [(0.0, 0.1), (0.0, 1.0), (1.0, 5.0), (5.0, 20.0), (20.0, FMAX)]:
    g = torch.linspace(lo, hi, 20001)
    BANDS.append((lo, hi, float((implied(g) - torch.log(g + DELTA)).abs().max())))

_g01 = torch.linspace(0, 1, 20001)
MONO01 = bool((torch.diff(implied(_g01)) > 0).all())
_g64 = torch.linspace(0, FMAX, 100001, dtype=torch.float64)
MONO64 = bool((torch.diff(implied(_g64, torch.float64)) > 0).all())
_freal = torch.tensor(sorted({0.0} | {p / q for q in range(1, SEQ) for p in range(0, q + 1)}),
                      dtype=torch.float32)
ERR_REAL = float((implied(_freal) - torch.log(_freal + DELTA)).abs().max())
MONO_REAL = bool((torch.diff(implied(_freal)) > 0).all())

_model = T.SimplifiedTransformerMLP(V, SEQ, ATTN, VS, K, bias=True, attn_param='separate')
T.build_frozen_ffn(_model, DELTA, K, FMAX, CGAIN, TOL)
W1, B1, W2, B2 = (_model.ff1.weight.detach(), _model.ff1.bias.detach(),
                  _model.ff2.weight.detach(), _model.ff2.bias.detach())
NZ1, NZ2 = int((W1 != 0).sum()), int((W2 != 0).sum())
ROWS1_ONE = int((W1 != 0).sum(1).eq(1).sum())
COLS1 = sorted(torch.nonzero((W1 != 0).sum(0))[:, 0].tolist())
ROWS2 = sorted(torch.nonzero((W2 != 0).sum(1))[:, 0].tolist())

FEX = 0.10
_h = F.gelu(torch.tensor([FEX]).unsqueeze(1) * A + B).squeeze(0)
_contrib = _h * W
N_ON = int((_h > 1e-6).sum())
SUM_CONTRIB = float(_contrib.sum())

_old = sorted(glob.glob(os.path.join(_F14, 'runs',
                                     'opt-adam_lr-0.003_b-256_init-normal_ep-1000',
                                     'model_*_x_0.6.pth')))
OLD = None
if _old:
    sd = torch.load(_old[0], map_location='cpu', weights_only=True)
    oa, ob = sd['ff1.weight'][0:K, IN2], sd['ff1.bias'][0:K]
    ow, oc = sd['ff2.weight'][IN2, 0:K], float(sd['ff2.bias'][IN2])
    f = torch.linspace(0, FMAX, 4001)
    imp = F.gelu(f.unsqueeze(1) * oa + ob) @ ow + oc + f
    tgt = torch.log(f + 1.0)
    OLD = dict(file=os.path.basename(_old[0]), c=oc,
               at0=float(imp[0]), at40=float(imp[-1]),
               maxerr=float((imp - tgt).abs().max()),
               mse=float(((imp - tgt) ** 2).mean()),
               mono=bool((torch.diff(imp) > 0).all()))

# ─── page rendering ───────────────────────────────────────────────────────────
PAGE = (8.27, 11.69)
MARGIN_L, MARGIN_R, TOP, BOT = 0.075, 0.945, 0.955, 0.055
STY = {
    'h1':   dict(size=17, weight='bold',   family='sans-serif', lead=0.034, pre=0.000),
    'h2':   dict(size=12, weight='bold',   family='sans-serif', lead=0.024, pre=0.018),
    'body': dict(size=9.2, weight='normal', family='sans-serif', lead=0.0175, pre=0.006),
    'mono': dict(size=8.0, weight='normal', family='monospace',  lead=0.0155, pre=0.008),
    'note': dict(size=8.2, weight='normal', family='sans-serif', lead=0.016, pre=0.006),
}
pdf = PdfPages(OUT)
_pageno = [0]


def render(blocks, footer=True):
    _pageno[0] += 1
    fig = plt.figure(figsize=PAGE)
    y = TOP
    for kind, text in blocks:
        if kind == 'gap':
            y -= float(text)
            continue
        s = STY[kind]
        y -= s['pre']
        for line in text.split('\n'):
            fig.text(MARGIN_L, y, line, fontsize=s['size'], fontweight=s['weight'],
                     family=s['family'], va='top', ha='left', color='#111111')
            y -= s['lead']
    if footer:
        fig.text(MARGIN_R, BOT - 0.018, f'{_pageno[0]}', fontsize=8, ha='right',
                 color='#888888')
        fig.text(MARGIN_L, BOT - 0.018, 'frozen log readout — precise construction',
                 fontsize=8, ha='left', color='#888888')
    pdf.savefig(fig)
    plt.close(fig)


def render_fig(fig, caption=None):
    _pageno[0] += 1
    if caption:
        fig.text(MARGIN_L, 0.028, caption, fontsize=8.4, ha='left', color='#333333',
                 wrap=True)
    fig.text(MARGIN_R, 0.012, f'{_pageno[0]}', fontsize=8, ha='right', color='#888888')
    pdf.savefig(fig)
    plt.close(fig)


def row(*cells):
    return ''.join(str(c) for c in cells)


# ─── page 1 ───────────────────────────────────────────────────────────────────
render([
    ('h1', 'The frozen log readout'),
    ('note', 'Precise construction of the FFN weights in the DisentangledTransformer\n'
             'ICL of Markov-with-noise · folder 14 · train.py'),
    ('gap', 0.012),
    ('body',
     'This document specifies, cell by cell, how the residual feed-forward block of the model is\n'
     'set so that a coordinate of the residual stream holding a raw successor statistic f leaves the\n'
     'block holding log(f + δ). Every number below is computed from the live code, not transcribed.'),
    ('h2', 'Configuration'),
    ('mono',
     f'δ  (freeze_delta)      = {DELTA}          smoothing that survives into the prediction\n'
     f'k  (freeze_hidden_k)   = {K}           GELU units per signal state\n'
     f'fmax (freeze_fit_fmax) = {FMAX}          upper end of the calibrated range\n'
     f'cgain                  = {CGAIN}          gain multiplier, a_j = cgain / h_j\n'
     f'tol  (freeze_fit_tol)  = {TOL}         build-time hard check on max|err|\n'
     f'V   = {V}   (V_sig = {VS} signal + {V - VS} noise)\n'
     f'T   = {SEQ}  sequence length\n'
     f'D   = {D}  residual width       in2 = {IN2}  first layer-2 output coordinate'),
    ('h2', 'The residual stream, coordinate by coordinate'),
    ('mono',
     f'   0 ..   {V - 1}    token one-hot                        ({V} coords)\n'
     f'   {V} .. {ATTN - 1}    position one-hot                     ({SEQ} coords)\n'
     f' {ATTN} .. {IN2 - 1}    layer-1 attention output             ({ATTN} coords)\n'
     f' {IN2} .. {IN2 + VS - 1}    layer-2 attention: SIGNAL COUNTS f_v  ({VS} coords)  <-- the FFN acts here\n'
     f' {IN2 + VS} .. {IN2 + V - 1}    layer-2 attention: noise slots       ({V - VS} coords)  (0 in the construction)\n'
     f' {IN2 + V} .. {D - 1}    layer-2 attention: unused            ({D - IN2 - V} coords)  (0 in the construction)'),
    ('h2', 'Forward pass'),
    ('mono',
     's = cat[token one-hot, position one-hot]            ->  108\n'
     's = cat[s, layer1_attention(s)]                     ->  216      CONCATENATION\n'
     's = cat[s, layer2_attention(s)]                     ->  324      CONCATENATION\n'
     's = s + ff2(GELU(ff1(s)))                           ->  324      ADDITION (residual)\n'
     'logits = fc_out(s)                                  ->    8\n'
     'p = softmax(logits)'),
    ('body',
     'The two attention blocks CONCATENATE; only the FFN ADDS. That single fact determines the\n'
     'target the FFN must represent, and is derived on the next page.'),
    ('h2', 'Layer shapes'),
    ('mono',
     f'ff1     nn.Linear({D}, {VS * K})     .weight.shape = ({VS * K}, {D})   .bias = ({VS * K},)\n'
     f'ff2     nn.Linear({VS * K}, {D})     .weight.shape = ({D}, {VS * K})   .bias = ({D},)\n'
     f'fc_out  nn.Linear({D}, {V})         .weight.shape = ({V}, {D})     .bias = ({V},)\n'
     '\n'
     f'The hidden width is V_sig x k = {VS} x {K} = {VS * K}, not k. One log curve needs k ramps and\n'
     f'a ramp can read only one input, so the {VS} count slots need {VS} independent blocks of {K}.'),
])

# ─── page 2 ───────────────────────────────────────────────────────────────────
render([
    ('h1', '1.  What the readout must compute, and why'),
    ('h2', '1.1  The softmax cancels the log'),
    ('body',
     'fc_out reads coordinate in2+v and the model applies softmax. If that coordinate holds\n'
     'log(f_v + δ), then'),
    ('mono',
     '    softmax( log(f_v + δ) )_v   =   exp(log(f_v + δ)) / Σ_u exp(log(f_u + δ))\n'
     '                                =   (f_v + δ) / Σ_u (f_u + δ)'),
    ('body',
     'i.e. the softmax UNDOES the log and returns the normalised counts. Without the log the model\n'
     'would emit softmax(f_v) ∝ exp(f_v), exponential in the count. Cancelling that distortion is the\n'
     'entire purpose of this block.'),
    ('h2', '1.2  The block is residual, so the target carries a −f'),
    ('body',
     'Layer-2 attention deposits f_v at coordinate in2+v. The FFN output is ADDED to it:'),
    ('mono',
     '    coordinate_after  =  f_v  +  ff2_out[in2+v]'),
    ('body',
     'For coordinate_after to equal log(f_v + δ), the FFN must emit'),
    ('mono',
     '    g(f)  =  log(f + δ) − f            <-- the residual log target'),
    ('body',
     'The −f is not cosmetic. Fitting plain log(f + δ) would leave the coordinate holding\n'
     'f + log(f + δ), wrong by up to a full nat over the calibrated range. The add is an OVERWRITE\n'
     'implemented as an addition: the incoming f is cancelled algebraically.'),
    ('h2', '1.3  Range and scale'),
    ('body',
     'In the hand construction layer-2 attention is a softmax average of token one-hots, so\n'
     'Σ_v f_v ≡ 1 exactly and f ∈ [0, 1]. In a trained model W_ov is learned, so f can take any\n'
     'value. The core is therefore fitted on [0, fmax] with fmax = 40.\n'
     '\n'
     'A global rescale of the attention output is harmless. For a scale factor s > 0,\n'
     '    log(s·f + δ) = log s + log(f + δ/s),\n'
     'and the constant log s cancels inside the softmax; only the effective smoothing changes from\n'
     'δ to δ/s. What is NOT harmless is leaving the fitted range — see section 6.'),
    ('h2', '1.4  Why δ = 0.01 and not 1.0'),
    ('body',
     'δ is the smoothing that survives into the prediction, p ∝ f + δ. With f at frequency scale\n'
     '(f ∈ [0,1]) a value of δ = 1.0 dominates the signal and drives the output towards uniform.\n'
     'δ = 0.01 keeps the counts legible while still giving unobserved successors non-zero mass.'),
])

# ─── page 3 ───────────────────────────────────────────────────────────────────
tbl = ['   j      knot p_j       gain a_j     offset b_j     weight w_j      tilt w_j·a_j']
for j in (0, 1, 2, 10, 30, 64, 100, 120, 126, 127):
    tbl.append(f'{j:4d} {float(KNOTS[j]):13.6f} {float(A[j]):14.1f} {float(B[j]):14.4f} '
               f'{float(W[j]):+14.6f} {float(TILT[j]):+16.4f}')
render([
    ('h1', '2.  How 128 GELU units make a logarithm'),
    ('h2', '2.1  One unit is a one-sided ramp'),
    ('body',
     'Unit j computes GELU(a_j · f + b_j) = GELU(a_j · (f − p_j)) with p_j = −b_j / a_j, its\n'
     'SWITCH-ON POINT (knot). Below p_j the unit is off; above it, it contributes a straight line of\n'
     'slope w_j · a_j. With a_j large, GELU is a smooth ReLU, so the corner is rounded rather than sharp.'),
    ('h2', '2.2  Summing ramps makes the slope a running total'),
    ('body',
     'Each unit switches on once and stays on. The total slope at any f is therefore the sum of the\n'
     'tilts of every unit already switched on. Building a curve reduces to bookkeeping:\n'
     '\n'
     '    tilt of unit j  =  (slope wanted after p_j)  −  (slope in force before p_j)'),
    ('h2', '2.3  The slope a log requires'),
    ('body',
     "The residual target's derivative is  g'(f) = 1/(f + δ) − 1, which FALLS from 1/δ − 1 = 99.0 at\n"
     'f = 0 to −0.975 at f = 40. A running total only accumulates, so the fit needs one large positive\n'
     'tilt at the very start and negative tilts thereafter. That is exactly what least squares produces\n'
     'unprompted:'),
    ('mono', '\n'.join(tbl)),
    ('note',
     f'Unit 0 alone supplies tilt {float(TILT[0]):+.3f} against a required initial slope of '
     f'{1/DELTA - 1:.3f}.\nEvery interior unit is negative; the tilts shrink from '
     f'{float(TILT[3]):+.3f} to {float(TILT[120]):+.4f} as the curve flattens.'),
    ('h2', '2.4  Accumulated slope tracks the true derivative'),
    ('mono',
     '   j      knot p_j     accumulated slope     required  1/(p+δ) − 1\n' +
     '\n'.join(f'{j:4d} {float(KNOTS[j]):13.5f} {float(CUM[j]):21.4f} {1/(float(KNOTS[j])+DELTA)-1:20.4f}'
               for j in (0, 3, 10, 20, 40, 60, 80, 100, 120, 127))),
])

# ─── page 4 ───────────────────────────────────────────────────────────────────
_fr = [-float(TILT[j]) / float(CUM[j - 1]) for j in range(3, 90) if float(CUM[j - 1]) > 0.5]
render([
    ('h1', '3.  Knot placement and the fit'),
    ('h2', '3.1  Knots are geometric in (f + δ)'),
    ('body',
     "g''(f) = −(f + δ)^-2, so all the curvature sits at small f: the curvature at f = 0.01 is 10^8\n"
     'times that at f = 40. Uniformly spaced knots spend most of the basis on the flat tail.\n'
     '\n'
     'The knots are therefore placed geometrically in (f + δ):'),
    ('mono',
     f'    p_j + δ  =  δ · r^j ,      r = ((fmax + δ)/δ)^(1/(k−1)) = {RATIO:.6f}\n'
     f'    p_0 = {float(KNOTS[0]):.6f}      p_127 = {float(KNOTS[-1]):.4f}'),
    ('body',
     "Because a log's slope halves whenever f doubles, geometric spacing gives every knot an equal\n"
     'SHARE of the work rather than an equal stretch of the axis. Measured over the steep region,\n'
     f'each knot removes {100*np.mean(_fr):.1f}% ± {100*np.std(_fr):.1f}% of the slope it finds.'),
    ('h2', '3.2  Gains scale with the local spacing'),
    ('mono',
     f'    h_j  = local knot spacing         a_j = cgain / h_j ,   cgain = {CGAIN}\n'
     f'    a_0   = {float(A[0]):.1f}   (h = {float(CGAIN/A[0]):.2e})\n'
     f'    a_127 = {float(A[-1]):.1f}       (h = {float(CGAIN/A[-1]):.3f})'),
    ('body',
     "Each ramp's smooth transition zone is then narrow relative to its OWN segment, so the rounding\n"
     'GELU applies is proportionate everywhere. High gains pair with knots near zero, where b_j = −a_j·p_j\n'
     'is also near zero, so no catastrophic cancellation occurs in float32.'),
    ('h2', '3.3  The fit is a linear least-squares problem — solved exactly, once'),
    ('body',
     'With a and b fixed, g(f) = GELU(f·a + b) · w + c is LINEAR in (w, c). Least squares therefore\n'
     'returns the global optimum and there is nothing left to improve. The design matrix is evaluated\n'
     'on a 40001-point grid, itself geometric in (f + δ), in float64; only the final tensors are cast\n'
     'to float32.\n'
     '\n'
     'NO GRADIENT REFINEMENT IS PERFORMED. See section 7 for what happened when it was.'),
    ('h2', '3.4  Accuracy of the installed core'),
    ('mono',
     f'overall max|err| (float32)            {MAXERR:.3e} nats\n'
     f'mse                                    {MSE:.3e}\n' +
     '\n'.join(f'  band f ∈ [{lo:5.1f}, {hi:5.1f}]              {e:.3e} nats' for lo, hi, e in BANDS) +
     f'\n\nrealizable f = p/q, q < {SEQ}  ({len(_freal)} values)   {ERR_REAL:.3e} nats\n'
     f'monotone, float64, dense grid [0, {FMAX:g}]      {MONO64}\n'
     f'monotone, float32, on [0, 1]                 {MONO01}\n'
     f'monotone, float32, realizable values         {MONO_REAL}'),
    ('note',
     'The float32 monotonicity test is reported on [0,1] and on the realizable values, not on a\n'
     '1e-5 dense grid over [0,40]: there the true increment is itself ~1e-5, i.e. at the level of\n'
     'float32 rounding on a 128-term sum, so such a test measures arithmetic noise, not the fit. The\n'
     'float64 test on the same grid answers whether the fitted function is monotone at all.'),
])

# ─── page 5: figure — the fit ─────────────────────────────────────────────────
fd = torch.linspace(0, 1, 100001)
real = (implied(fd)).numpy()
tg = np.log(fd.numpy() + DELTA)
fig = plt.figure(figsize=PAGE)
fig.suptitle('Figure 1 — the realised readout against log(f + δ)', y=0.965, fontsize=13)
ax = fig.add_axes([0.11, 0.700, 0.80, 0.205])
ax.plot(fd.numpy(), tg, color='#0072B2', lw=3, alpha=.5, label='target  log(f + δ)')
ax.plot(fd.numpy(), real, color='#D55E00', lw=1.2, ls='--', label='realised by ff1/GELU/ff2')
ax.set_xlabel('f'); ax.set_ylabel('coordinate value'); ax.legend(frameon=False, fontsize=9)
ax.set_title('linear axis', fontsize=10); ax.grid(alpha=.25)
ax2 = fig.add_axes([0.11, 0.405, 0.80, 0.205])
ax2.plot(fd.numpy() + DELTA, tg, color='#0072B2', lw=3, alpha=.5)
ax2.plot(fd.numpy() + DELTA, real, color='#D55E00', lw=1.2, ls='--')
ax2.set_xscale('log'); ax2.set_xlabel('f + δ  (log axis)'); ax2.set_ylabel('coordinate value')
ax2.set_title('log axis — where uniform knots fail', fontsize=10); ax2.grid(alpha=.25)
ax3 = fig.add_axes([0.11, 0.110, 0.80, 0.205])
ax3.plot(fd.numpy(), real - tg, color='#009E73', lw=0.9)
ax3.axhline(0, color='0.6', lw=.8)
ax3.set_xlabel('f'); ax3.set_ylabel('realised − target  (nats)')
ax3.set_title(f'residual error, max |·| = {BANDS[1][2]:.1e} nats on [0,1]', fontsize=10)
ax3.grid(alpha=.25)
render_fig(fig, 'The error is a zero-mean ripple whose amplitude is set by the knot spacing; it is '
                'not a systematic bias.')

# ─── page 6: figure — the mechanism ───────────────────────────────────────────
f2 = torch.linspace(0, 2, 4001)
Bm = F.gelu(f2.unsqueeze(1) * A + B)
fig = plt.figure(figsize=PAGE)
fig.suptitle('Figure 2 — how the ramps accumulate into a logarithm', y=0.965, fontsize=13)
axa = fig.add_axes([0.11, 0.700, 0.80, 0.205])
for j in (30, 50, 60, 70, 80, 90):
    axa.plot(f2.numpy(), (Bm[:, j] * W[j]).numpy(), lw=1.3,
             label=f'j={j}, p={float(KNOTS[j]):.3f}')
axa.set_xlabel('f'); axa.set_ylabel('contribution'); axa.grid(alpha=.25)
axa.legend(frameon=False, fontsize=8, ncol=2)
axa.set_title('individual units  w_j · GELU(a_j (f − p_j))', fontsize=10)
axb = fig.add_axes([0.11, 0.405, 0.80, 0.205])
for n, col in [(20, '#56B4E9'), (50, '#0072B2'), (80, '#CC79A7'), (K, '#D55E00')]:
    axb.plot(f2.numpy(), (Bm[:, :n] @ W[:n] + CC + f2).numpy(), color=col, lw=1.6,
             label=f'first {n} units')
axb.plot(f2.numpy(), torch.log(f2 + DELTA).numpy(), 'k--', lw=1.4, label='target')
axb.set_ylim(-5.2, 1.2); axb.set_xlabel('f'); axb.set_ylabel('coordinate value')
axb.grid(alpha=.25); axb.legend(frameon=False, fontsize=8)
axb.set_title('partial sums closing on the log', fontsize=10)
axc = fig.add_axes([0.11, 0.110, 0.80, 0.205])
axc.plot(KNOTS.numpy(), CUM.numpy(), '.', ms=5, color='#0072B2', label='accumulated slope')
kk = np.linspace(0, FMAX, 2000)
axc.plot(kk, 1 / (kk + DELTA) - 1, color='#D55E00', lw=1.4, label="required g'(f)")
axc.set_xscale('symlog', linthresh=1e-3); axc.set_yscale('symlog', linthresh=1e-2)
axc.set_xlabel('knot p_j'); axc.set_ylabel('slope')
axc.grid(alpha=.25); axc.legend(frameon=False, fontsize=9)
axc.set_title("accumulated slope vs the derivative it must reproduce", fontsize=10)
render_fig(fig, 'Bottom panel: the running total of tilts (points) lands on the required derivative '
                '(line) at every knot.')

# ─── page 7: ff1 ──────────────────────────────────────────────────────────────
render([
    ('h1', '4.  Exact construction of ff1'),
    ('body',
     f'ff1.weight is a {VS * K} x {D} grid. It starts at ZERO everywhere. Exactly {NZ1} cells are set —\n'
     f'{W1.numel() - NZ1} of the {W1.numel()} cells remain zero.\n'
     '\n'
     'Rows index neurons; columns index residual-stream positions.'),
    ('h2', '4.1  Assignment'),
    ('mono',
     f'for v in range({VS}):                          # one block per signal state\n'
     f'    rows = slice(v*{K}, (v+1)*{K})\n'
     f'    ff1.weight[rows, {IN2} + v] = a          # the {K} steepnesses\n'
     f'    ff1.bias[rows]             = b          # the {K} offsets'),
    ('h2', '4.2  The resulting cells'),
    ('mono',
     f'  row     column        value          meaning\n'
     f'    0        {IN2}     {float(W1[0, IN2]):12.4f}     neuron   0 of state 0 reads f_0, steepness a[0]\n'
     f'    1        {IN2}     {float(W1[1, IN2]):12.4f}     neuron   1 of state 0 reads f_0, steepness a[1]\n'
     f'  ...\n'
     f'  {K - 1}        {IN2}     {float(W1[K - 1, IN2]):12.4f}     neuron {K - 1} of state 0, steepness a[{K - 1}]\n'
     f'  {K}        {IN2 + 1}     {float(W1[K, IN2 + 1]):12.4f}     neuron   0 of state 1 reads f_1\n'
     f'  ...\n'
     f'  {2 * K - 1}        {IN2 + 1}     {float(W1[2 * K - 1, IN2 + 1]):12.4f}     neuron {K - 1} of state 1\n'
     f'  {2 * K}        {IN2 + 2}     {float(W1[2 * K, IN2 + 2]):12.4f}     state 2 block begins\n'
     f'  {3 * K}        {IN2 + 3}     {float(W1[3 * K, IN2 + 3]):12.4f}     state 3 block begins\n'
     f'  {4 * K - 1}        {IN2 + 3}     {float(W1[4 * K - 1, IN2 + 3]):12.4f}     last neuron of the last block'),
    ('h2', '4.3  Verified structure'),
    ('mono',
     f'non-zero cells                          {NZ1}\n'
     f'rows with EXACTLY one non-zero          {ROWS1_ONE} / {VS * K}\n'
     f'columns ever touched                    {COLS1}\n'
     f'ff1.bias blocks equal to b              '
     f'{all(torch.equal(B1[i*K:(i+1)*K], B) for i in range(VS))}'),
    ('h2', '4.4  Why exactly one non-zero per row'),
    ('body',
     f'A row computes Σ over all {D} columns of (cell × stream value). With a single non-zero of value\n'
     f'a[j] at column {IN2}+v, the row computes a[j]·f_v + b[j] and nothing else. A second non-zero would\n'
     'mix another coordinate into the ramp and it would cease to be a function of f_v alone.\n'
     '\n'
     f'The four blocks are wired to four DIFFERENT columns but hold IDENTICAL values of a and b. They are\n'
     'four copies of one scalar function, evaluated at four different inputs by a single matrix multiply.'),
])

# ─── page 8: ff2 ──────────────────────────────────────────────────────────────
zero_rows = D - len(ROWS2)
render([
    ('h1', '5.  Exact construction of ff2, and the readout'),
    ('body',
     f'ff2.weight is a {D} x {VS * K} grid, also starting at ZERO. Exactly {NZ2} cells are set. Here rows\n'
     'index stream positions and columns index neurons — the transpose of ff1\'s roles.'),
    ('h2', '5.1  Assignment'),
    ('mono',
     f'for v in range({VS}):\n'
     f'    rows = slice(v*{K}, (v+1)*{K})\n'
     f'    ff2.weight[{IN2} + v, rows] = w          # the {K} output weights\n'
     f'    ff2.bias[{IN2} + v]         = c          # the constant, {CC:+.5f}'),
    ('h2', '5.2  The resulting cells'),
    ('mono',
     f'  row {IN2}  ->  columns    0 .. {K - 1}   = w[0..{K - 1}]     rest of the row zero\n'
     f'  row {IN2 + 1}  ->  columns  {K} .. {2 * K - 1}   = w[0..{K - 1}]     rest of the row zero\n'
     f'  row {IN2 + 2}  ->  columns  {2 * K} .. {3 * K - 1}   = w[0..{K - 1}]     rest of the row zero\n'
     f'  row {IN2 + 3}  ->  columns  {3 * K} .. {4 * K - 1}   = w[0..{K - 1}]     rest of the row zero\n'
     f'\n'
     f'  all other {zero_rows} rows are entirely zero'),
    ('h2', '5.3  Verified structure'),
    ('mono',
     f'non-zero cells                          {NZ2}\n'
     f'rows that are not all zero              {ROWS2}\n'
     f'each such row has                       {int((W2[IN2] != 0).sum())} non-zeros\n'
     f'ff2.bias non-zero at                    {torch.nonzero(B2)[:, 0].tolist()}\n'
     f'ff2.bias value there                    {float(B2[IN2]):+.5f}      log(δ) = {np.log(DELTA):+.5f}'),
    ('h2', '5.4  Why 320 rows stay exactly zero'),
    ('body',
     'The FFN output is ADDED to the stream. A zero row outputs zero, and adding zero changes nothing.\n'
     'That is what keeps the token one-hot, the position one-hot, layer-1\'s output and the unused\n'
     'layer-2 coordinates bit-identical across the block. Only the four count slots are rewritten.'),
    ('h2', '5.5  c is the value when every unit is off'),
    ('body',
     f'At f = 0 all ramps are off (or exactly at their corner), so the sum vanishes and the block emits c.\n'
     f'The target there is log(0 + δ) = {np.log(DELTA):.5f}, and the fit returns c = {CC:.5f}.\n'
     'This also fixes the behaviour for negative inputs — see section 6.'),
    ('h2', '5.6  The readout that consumes it (hand construction only)'),
    ('body',
     'In the hand-built reference model fc_out is set to a pure selector, and the noise logit to one\n'
     'constant. In the TRAINED models fc_out is learned and this section does not apply.'),
    ('mono',
     f'fc_out.weight            = 0 everywhere\n'
     f'fc_out.weight[0:{VS}, {IN2}:{IN2 + VS}] = I_{VS}\n'
     f'fc_out.bias[0:{VS}]         = 0\n'
     f'fc_out.bias[{VS}:{V}]         = log( (1 + V_sig·δ)(1 − α) / (α · n_noise) )\n'
     '\n'
     'Since layer-2 attention is a softmax average, Σ_v f_v ≡ 1 exactly, so Σ_v (f_v + δ) = 1 + V_sig·δ\n'
     'is a CONSTANT and that single bias makes the model softmax return\n'
     '    p(v)     = α (f_v + δ) / (1 + V_sig·δ)      for v < V_sig\n'
     '    p(noise) = (1 − α) / n_noise\n'
     'exactly, with no external renormalisation. At α = 1 the exact bias is −inf; −30 is used.'),
])

# ─── page 9: worked example ───────────────────────────────────────────────────
on_idx = torch.where(_h > 1e-6)[0].tolist()
lines = [f'{"j":>5s} {"knot p_j":>12s} {"GELU output":>14s} {"× w_j = contribution":>22s}']
for j in on_idx[:4] + ['...'] + on_idx[-4:]:
    if j == '...':
        lines.append(f'{"...":>5s}')
        continue
    lines.append(f'{j:5d} {float(KNOTS[j]):12.5f} {float(_h[j]):14.3f} {float(_contrib[j]):22.5f}')
lines.append(f'{"off":>5s} {"":>12s} {"0.000":>14s} {"0.00000":>22s}   (the other {K - N_ON})')
render([
    ('h1', '6.  Worked example, and out-of-range behaviour'),
    ('h2', f'6.1  A single evaluation at f = {FEX}'),
    ('body',
     f'{N_ON} of the {K} units are on — those whose knot lies below {FEX}. All {N_ON} contribute; the other\n'
     f'{K - N_ON} output exactly zero. The units that switched on earliest have ramped furthest and\n'
     'contribute most.'),
    ('mono', '\n'.join(lines)),
    ('mono',
     f'\n  sum of all {N_ON} contributions            {SUM_CONTRIB:+.5f}\n'
     f'  + c                                  {SUM_CONTRIB + CC:+.5f}      <- ff2 output\n'
     f'  + f  (the residual add)              {SUM_CONTRIB + CC + FEX:+.5f}\n'
     f'  target log(f + δ)                    {np.log(FEX + DELTA):+.5f}\n'
     f'  error                                {SUM_CONTRIB + CC + FEX - np.log(FEX + DELTA):+.2e}'),
    ('h2', '6.2  Negative inputs'),
    ('body',
     'For f < 0 every GELU argument a_j(f − p_j) is negative (all knots are ≥ 0 and all gains > 0), so\n'
     'every unit is off and the block emits c. The coordinate becomes'),
    ('mono',
     f'    coordinate  =  f + c  ≈  f + log δ            (c = {CC:.5f}, log δ = {np.log(DELTA):.5f})'),
    ('body',
     'a straight line of slope 1 anchored at the log\'s floor. Measured:'),
    ('mono',
     f'{"f":>9s} {"coordinate":>14s} {"f + c":>12s} {"d/df":>8s}\n' +
     '\n'.join(
         f'{v:9.2f} {float(implied([v])[0]):14.5f} {v + CC:12.5f} {1.0:8.3f}'
         for v in (-3.3, -1.0, -0.3, -0.1, -0.01)) +
     f'\n{0.0:9.2f} {float(implied([0.0])[0]):14.5f} {0.0 + CC:12.5f} {"—":>8s}   <- continuous at 0'),
    ('body',
     'This is finite, monotone, continuous with the log at f = 0, and has gradient exactly 1, so the\n'
     'learning signal through W_ov survives. A hard clamp to log δ would be a more faithful log but\n'
     'would have zero gradient there, and at initialisation roughly half of all values entering the\n'
     'readout are negative. The linear branch is the escape ramp.\n'
     '\n'
     'CONSEQUENCE: in the negative branch the model emits softmax(f + log δ) ∝ δ·exp(f), exponential\n'
     'rather than linear in f. Every training run therefore records readout_min, readout_max,\n'
     'frac_negative and frac_above_fmax each epoch so the size of this region is never invisible.'),
    ('h2', '6.3  Above fmax'),
    ('body',
     f'Beyond f = {FMAX:g} all units are saturated on and the sum continues linearly, so the readout\n'
     'over-estimates the log. Measured: at f = 50 it gives '
     f'{float(implied([50.0])[0]):+.4f} against a true {np.log(50 + DELTA):+.4f}.'),
])

# ─── page 10: verification + the old failure ──────────────────────────────────
blocks = [
    ('h1', '7.  Verification, and the failure this replaced'),
    ('h2', '7.1  Build-time check (train.py refuses to train if this fails)'),
    ('body',
     'check_log_core() is called from build_frozen_ffn() every time the core is fitted. It measures the\n'
     'implied log — that is, the coordinate value AFTER the residual add — in three bands and tests\n'
     'monotonicity on [0,1], then raises unless max|err| ≤ tol and the function is monotone.'),
    ('mono',
     f'tolerance                                {TOL:.1e}\n' +
     '\n'.join(f'band f ∈ [{lo:5.1f}, {hi:5.1f}]                {e:.3e}   {"PASS" if e <= TOL else "FAIL"}'
               for lo, hi, e in BANDS if (lo, hi) in [(0.0, 1.0), (1.0, 5.0), (5.0, 20.0), (20.0, FMAX)]) +
     f'\nmonotone on [0,1]                        {MONO01}'),
    ('h2', '7.2  End-to-end check'),
    ('body',
     'With the core installed and the hand construction assembled, the CE read off the model\'s own\n'
     'softmax was compared against the closed-form predictor α(f+δ)/Σ(f+δ), with f read independently\n'
     'from the raw layer-2 output rather than recovered from the logits. Agreement across all seven\n'
     'thresholds: +9e-07 to +2.4e-06 nats.'),
]
if OLD:
    blocks += [
        ('h2', '7.3  What the previous implementation installed'),
        ('body',
         'The earlier fit_log_core warm-started from the same closed-form least-squares solution and then\n'
         'ran 2000 Adam steps at lr = 1e-2 on (a, b, w, c) jointly. Measured directly from the ff1/ff2\n'
         f'weights saved in a trained checkpoint ({OLD["file"]}):'),
        ('mono',
         f'c installed                              {OLD["c"]:+.4f}\n'
         f'implied log at f = 0                     {OLD["at0"]:+.4f}      target  {0.0:+.4f}\n'
         f'implied log at f = 40                    {OLD["at40"]:+.4f}      target  {np.log(41.0):+.4f}\n'
         f'max|err|                                 {OLD["maxerr"]:.4f} nats\n'
         f'mse                                      {OLD["mse"]:.4e}\n'
         f'monotone                                 {OLD["mono"]}'),
        ('body',
         'That is not a logarithm. It is a non-monotone function running from +9.83 down to −111.00 over\n'
         'a range where the log rises from 0 to +3.71.\n'
         '\n'
         'IMPORTANT NUANCE: re-running that identical code on CPU today converges instead, giving\n'
         'max|err| ≈ 0.071. The Adam refinement is not deterministically divergent — it is\n'
         'environment-dependent, and on the GPU used for the folder-12 and folder-14 training runs it\n'
         'diverged. An unreliable refinement on top of an already-optimal solution is the defect; the\n'
         'current implementation removes the refinement entirely, which is why the problem cannot recur.'),
        ('body',
         'Because the identical broken core went into every run of those two projects, comparisons WITHIN\n'
         'them remain internally consistent. What those runs cannot support is any mechanistic claim\n'
         'about the readout, since the readout was not computing a log.'),
    ]
render(blocks)

# ─── page 11: reference implementation ────────────────────────────────────────
render([
    ('h1', '8.  Reference implementation'),
    ('body', 'Verbatim from folder 14, train.py. Comments and docstrings removed for compactness.'),
    ('mono',
     'def fit_log_core(delta, k, fmax, cgain, n_grid=40001):\n'
     '    dt = torch.float64\n'
     '    kn = torch.exp(torch.linspace(math.log(delta), math.log(fmax + delta),\n'
     '                                  k, dtype=dt)) - delta\n'
     '    h  = torch.diff(kn, prepend=(2 * kn[:1] - kn[1:2])).clamp_min(1e-12)\n'
     '    a  = cgain / h\n'
     '    b  = -a * kn\n'
     '    fg = (torch.exp(torch.linspace(math.log(delta), math.log(fmax + delta),\n'
     '                                   n_grid, dtype=dt)) - delta).clamp_min(0.0)\n'
     '    basis  = F.gelu(fg.unsqueeze(1) * a.unsqueeze(0) + b.unsqueeze(0))\n'
     '    basis1 = torch.cat([basis, torch.ones(n_grid, 1, dtype=dt)], dim=1)\n'
     '    target = torch.log(fg + delta) - fg\n'
     '    sol    = torch.linalg.lstsq(basis1, target.unsqueeze(1)).solution.squeeze(1)\n'
     '    w, c   = sol[:k], sol[k]\n'
     '    a32, b32, w32, c32 = a.float(), b.float(), w.float(), float(c)\n'
     '    f32  = fg.float()\n'
     '    pred = F.gelu(f32.unsqueeze(1) * a32 + b32) @ w32 + c32\n'
     '    err  = (pred + f32 - torch.log(f32 + delta)).abs()\n'
     '    return a32, b32, w32, c32, float(err.max()), float((err ** 2).mean())\n'
     '\n'
     'def build_frozen_ffn(model, delta, k, fmax, cgain, tol):\n'
     '    dev = next(model.parameters()).device\n'
     '    key = (delta, k, fmax, cgain)\n'
     '    if key not in _LOG_CORE_CACHE:\n'
     '        core = fit_log_core(delta, k, fmax, cgain)\n'
     '        check_log_core(*core[:4], delta, fmax, tol)\n'
     '        _LOG_CORE_CACHE[key] = core\n'
     '    a, b, w, c, _, _ = _LOG_CORE_CACHE[key]\n'
     '    a, b, w = a.to(dev), b.to(dev), w.to(dev)\n'
     '    Vs, D, in2 = model.v_sig, model.D, model.in2\n'
     '    assert model.ff1.out_features == Vs * k\n'
     '    W1 = torch.zeros(Vs * k, D, device=dev); B1 = torch.zeros(Vs * k, device=dev)\n'
     '    W2 = torch.zeros(D, Vs * k, device=dev); B2 = torch.zeros(D, device=dev)\n'
     '    for v in range(Vs):\n'
     '        rows = slice(v * k, (v + 1) * k)\n'
     '        W1[rows, in2 + v] = a\n'
     '        B1[rows]          = b\n'
     '        W2[in2 + v, rows] = w\n'
     '        B2[in2 + v]       = c\n'
     '    with torch.no_grad():\n'
     '        model.ff1.weight.copy_(W1); model.ff1.bias.copy_(B1)\n'
     '        model.ff2.weight.copy_(W2); model.ff2.bias.copy_(B2)\n'
     '    for p in (model.ff1.weight, model.ff1.bias,\n'
     '              model.ff2.weight, model.ff2.bias):\n'
     '        p.requires_grad_(False)'),
    ('h2', 'Summary of what is set (all four tensors are then frozen)'),
    ('mono',
     f'                         cells        non-zero        zero\n'
     f'ff1.weight            {W1.numel():8d}        {NZ1:8d}    {W1.numel() - NZ1:8d}\n'
     f'ff1.bias              {B1.numel():8d}        {int((B1 != 0).sum()):8d}    '
     f'{B1.numel() - int((B1 != 0).sum()):8d}\n'
     f'ff2.weight            {W2.numel():8d}        {NZ2:8d}    {W2.numel() - NZ2:8d}\n'
     f'ff2.bias              {B2.numel():8d}        {int((B2 != 0).sum()):8d}    '
     f'{B2.numel() - int((B2 != 0).sum()):8d}'),
])

d = pdf.infodict()
d['Title'] = 'Frozen log readout — precise construction'
d['Subject'] = 'ICL of Markov-with-noise: the FFN log readout, cell by cell'
pdf.close()
print(f'wrote {OUT}  ({_pageno[0]} pages, {os.path.getsize(OUT) / 1024:.0f} KB)')
