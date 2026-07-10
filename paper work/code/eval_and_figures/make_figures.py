# Publication figures for the ICLR draft (Figures 1-6 + Appendix figures). Reads
# data/eval_test.json (from eval_all.py, run first) + training histories from the
# ICL folders. Writes vector PDFs to figures/ (this script's own subfolder — copy
# into ../../figures/ to update the versions embedded in main.tex).
# Palette: Okabe-Ito subset validated CVD-safe; identity fixed across all figures;
# linestyle/marker as secondary encoding; gray = de-emphasized reference.
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

HERE = os.path.dirname(os.path.abspath(__file__))
FIG  = os.path.join(HERE, "figures"); os.makedirs(FIG, exist_ok=True)
ICL  = "/home/hariguru/ashutosh/ICL"   # edit if your checkpoints live elsewhere
F6   = os.path.join(ICL, "6. understanding what gradient descent is firuring out/models")
F7   = os.path.join(ICL, "7. unique_Wqk_Wok_test.ipynb/models")
F8   = os.path.join(ICL, "8. train with mlp/models")

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8.5,
    "legend.fontsize": 7, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "font.family": "serif", "mathtext.fontset": "stix", "font.serif": ["STIXGeneral"],
    "axes.linewidth": 0.6, "grid.alpha": 0.25, "grid.linewidth": 0.5,
    "lines.linewidth": 1.4, "figure.dpi": 150, "savefig.bbox": "tight",
})

E = json.load(open(os.path.join(HERE, "data", "eval_test.json")))
AL = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
def curve(key): return np.array([E["curves"][str(a)][key] for a in AL])
def pn(key):    return np.array([E["postnoise"][str(a)][key] for a in AL])

# fixed identity map ------------------------------------------------------------
S = {
 "ideal":  dict(c="#D55E00", ls="--", m="v",  lbl=r"ideal-$\alpha$ (oracle floor)"),
 "bayes":  dict(c="#E69F00", ls="-.", m="^",  lbl="context-Bayes"),
 "ctx":    dict(c="#CC79A7", ls=":",  m="s",  lbl="ctx-count"),
 "gd":     dict(c="#0072B2", ls="-",  m="o",  lbl="GD attention-only"),
 "gd_comb":dict(c="#0072B2", ls="--", m="o",  lbl=r"GD attn-only, $(W_{QK},W_{OV})$", open=True),
 "mlp":    dict(c="#56B4E9", ls="-",  m="P",  lbl="GD + one FFN"),
 "twocell":dict(c="#009E73", ls="-",  m="X",  lbl="GD two-cell (FFN$\\times$2)"),
 "twocell_comb": dict(c="#009E73", ls="--", m="X", lbl=r"GD two-cell, $(W_{QK},W_{OV})$", open=True),
}
def plot_c(ax, key, y, **kw):
    st = S[key]
    mfc = "white" if st.get("open") else st["c"]
    ax.plot(AL, y, st["ls"], marker=st["m"], color=st["c"], mfc=mfc, mec=st["c"],
            ms=3.6, label=kw.pop("lbl", st["lbl"]), **kw)

# ═ Figure 1: task schematic ════════════════════════════════════════════════════
rng = np.random.default_rng(5)
V_sig, V, T_show, alpha_s = 10, 20, 21, 0.7
order = rng.random((V_sig, V_sig)).argsort(axis=1); trans = order[:, :2]
seq, last = [], -1
for t in range(T_show):
    if rng.random() < alpha_s:
        tok = int(trans[last, rng.integers(0, 2)]) if last >= 0 else int(rng.integers(0, V_sig))
        last = tok
    else:
        tok = int(rng.integers(V_sig, V))
    seq.append(tok)

fig, ax = plt.subplots(figsize=(7.0, 1.95))
ax.set_xlim(-0.6, T_show + 7.6); ax.set_ylim(-2.3, 2.55); ax.axis("off")
sig_c, noi_c = "#0072B2", "#BBBBBB"
prev_sig_i = None
for i, tok in enumerate(seq):
    is_sig = tok < V_sig
    fc = sig_c if is_sig else noi_c
    ax.add_patch(FancyBboxPatch((i - 0.38, -0.42), 0.76, 0.84,
                 boxstyle="round,pad=0.02", fc=fc, ec="none", alpha=0.92 if is_sig else 0.55))
    ax.text(i, 0, str(tok), ha="center", va="center", fontsize=7,
            color="white" if is_sig else "#333333", fontweight="bold")
    if is_sig:
        if prev_sig_i is not None:
            ax.add_patch(FancyArrowPatch((prev_sig_i, -0.55), (i, -0.55),
                         connectionstyle=f"arc3,rad={-0.16 - 0.02*(i-prev_sig_i)}",
                         arrowstyle="-|>", mutation_scale=6, lw=0.7, color="#D55E00"))
        prev_sig_i = i
ax.text((T_show - 1) / 2, 2.0, (r"w.p. $\alpha$: emit a successor of the current state "
        r"(each w.p. $\alpha/2$) and advance;" "\n" r"w.p. $1-\alpha$: emit a uniform noise "
        "token (state unchanged)"), ha="center", va="center", fontsize=7.5)
ax.text((T_show - 1) / 2, -1.65, "state chain (last signal token = current state); "
        "noise tokens leave the state unchanged", ha="center", fontsize=7, color="#D55E00")
tx = T_show + 0.9
ax.text(tx + 3.2, 2.25, "per-sequence table (resampled\nfor every sequence)",
        ha="center", va="center", fontsize=7, fontweight="bold")
for r in range(5):
    ax.text(tx, 1.15 - r * 0.72, f"${r}\\to\\{{{trans[r,0]},{trans[r,1]}\\}}$",
            fontsize=6.6, ha="left", va="center")
    ax.text(tx + 3.4, 1.15 - r * 0.72, f"${r+5}\\to\\{{{trans[r+5,0]},{trans[r+5,1]}\\}}$",
            fontsize=6.6, ha="left", va="center")
ax.plot([T_show + 0.25, T_show + 0.25], [-1.9, 2.4], lw=0.5, color="#999999")
fig.savefig(os.path.join(FIG, "fig_task.pdf")); plt.close(fig)

# ═ Figure 2: predictor hierarchy — absolute + excess CE ═════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.45))
ax = axes[0]
for k in ["ideal", "bayes", "ctx", "gd", "twocell"]:
    plot_c(ax, k, curve(k))
ax.set_xlabel(r"signal density $\alpha$"); ax.set_ylabel("test CE (nats)")
ax.set_title("(a) cross-entropy, all predictors"); ax.grid(True)
ax.legend(frameon=False, handlelength=1.8)
ax = axes[1]
ideal = curve("ideal")
for k in ["bayes", "ctx", "gd", "mlp", "twocell"]:
    plot_c(ax, k, curve(k) - ideal)
ax.axhline(0, color="#D55E00", ls="--", lw=1.0)
ax.text(0.55, 0.015, r"ideal-$\alpha$", color="#D55E00", fontsize=7, ha="center")
ax.set_xlabel(r"signal density $\alpha$"); ax.set_ylabel(r"excess CE over ideal-$\alpha$ (nats)")
ax.set_title("(b) excess cross-entropy (gap to the oracle floor)"); ax.grid(True)
ax.set_ylim(-0.01, 0.62)
ax.legend(frameon=False, handlelength=1.8)
fig.savefig(os.path.join(FIG, "fig_hierarchy.pdf")); plt.close(fig)

# ═ Figure 3: phase transition — rate + dynamics ═════════════════════════════════
def hist(path):
    d = json.load(open(path)); return d["history"]
h_gd07 = hist(os.path.join(F6, "metrics_x_0.7.json"))
h_gd08 = hist(os.path.join(F6, "metrics_x_0.8.json"))
h_ml08 = hist(os.path.join(F6, "metrics_mlp_x_0.8.json.bak400grok"))

fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.45))
ax = axes[0]
for k in ["gd", "mlp", "ctx"]:
    st = S.get(k, dict(c="#CC79A7", ls=":", m="s", lbl="ctx-count (analytic)"))
    ax.semilogy(AL, pn(k), st["ls"], marker=st["m"], color=st["c"], ms=3.6,
                label=S[k]["lbl"] if k in S else st["lbl"])
ax.set_xlabel(r"signal density $\alpha$")
ax.set_ylabel("post-noise wrong-successor rate")
ax.set_title("(a) state recovery after a noise token"); ax.grid(True, which="both")
ax.axvspan(0.75, 0.92, color="#D55E00", alpha=0.06)
ax.text(0.835, 0.13, "no noise-skipping\ncircuit", fontsize=6.5, ha="center", color="#8a4a00")
ax.legend(frameon=False, loc="center left")
ax = axes[1]
h_ml08fast = hist(os.path.join(F6, "metrics_mlp_x_0.8.json"))
def rel(h, a): return np.array(h["test_acc"]) / (a / 2.0)
ax.plot(np.arange(1, len(h_gd07["test_acc"]) + 1), rel(h_gd07, 0.7), "-",
        color="#0072B2", label=r"attn-only, $\alpha=0.7$")
ax.plot(np.arange(1, len(h_gd08["test_acc"]) + 1), rel(h_gd08, 0.8), "--",
        color="#0072B2", label=r"attn-only, $\alpha=0.8$ (best of 7 seeds)")
ax.plot(np.arange(1, len(h_ml08fast["test_acc"]) + 1), rel(h_ml08fast, 0.8), "-",
        color="#56B4E9", label=r"+FFN, $\alpha=0.8$ (fast run)")
ax.plot(np.arange(1, len(h_ml08["test_acc"]) + 1), rel(h_ml08, 0.8), ":",
        color="#56B4E9", lw=1.2, label=r"+FFN, $\alpha=0.8$ (slow run)")
ax.axhline(1.0, color="#767676", lw=0.6, ls=":")
ax.text(395, 1.008, r"argmax ceiling $\alpha/2$", fontsize=6.3, ha="right", color="#555555")
ax.set_xlabel("epoch"); ax.set_ylabel(r"test accuracy $/\ (\alpha/2)$")
ax.set_ylim(0.42, 1.06)
ax.set_title("(b) training dynamics at the transition"); ax.grid(True)
ax.legend(frameon=False, loc="lower right", fontsize=6.3)
fig.savefig(os.path.join(FIG, "fig_phase.pdf")); plt.close(fig)

# ═ Figure 4: calibration ladder (hand readouts on the SAME W* attention) ════════
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.5))
fig.subplots_adjust(wspace=0.58)
stages = ["wstar", "wbetter", "log", "inarch", "eq"]
stage_lbl = [r"softmax, fixed $\sigma$", r"softmax, tuned $(\sigma,b_N)$",
             r"$\log(\mathrm{count}+\beta)$ FFN", r"in-arch normalised log FFN",
             "equalize FFN (GELU)"]
blues = ["#c6dbef", "#9ecae1", "#6baed6", "#3182bd", "#08519c"]
for ax, a in zip(axes, [0.5, 0.8]):
    r = E["curves"][str(a)]
    ex = [r[k] - r["ideal"] for k in stages]
    bars = ax.bar(range(len(stages)), ex, color=blues, width=0.62, zorder=3)
    for ln_k, ln_v, ln_c, ls in [("ctx-count", r["ctx"] - r["ideal"], S["ctx"]["c"], ":"),
                                 ("GD + one FFN", r["mlp"] - r["ideal"], "#2b7fa8", "-"),
                                 ("context-Bayes", r["bayes"] - r["ideal"], "#8a6100", "-.")]:
        ax.axhline(ln_v, color=ln_c, ls=ls, lw=1.1, zorder=4, label=ln_k)
    for i, v in enumerate(ex):
        ax.text(i, v + 0.004, f"{v:.3f}", ha="center", fontsize=6)
    ax.legend(frameon=False, loc="upper right", fontsize=6.3, handlelength=1.7)
    ax.set_xticks(range(len(stages)))
    ax.set_xticklabels(stage_lbl, rotation=22, ha="right", fontsize=6.3)
    ax.set_ylabel(r"excess CE over ideal-$\alpha$ (nats)")
    ax.set_title(rf"$\alpha={a}$"); ax.grid(True, axis="y", zorder=0)
fig.savefig(os.path.join(FIG, "fig_ffn.pdf")); plt.close(fig)

# ═ Figure 5: separate vs combined parametrisation ═══════════════════════════════
h_tc05  = hist(os.path.join(F8, "metrics_mlp_x_0.5.json"))
h_tcc05 = hist(os.path.join(F8, "metrics_mlp_comb_x_0.5.json"))
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.45))
ax = axes[0]
for k in ["gd", "gd_comb", "twocell", "twocell_comb"]:
    plot_c(ax, k, curve(k) - ideal)
ax.axhline(curve("ctx")[4] - ideal[4], visible=False)  # spacing no-op
ax.set_xlabel(r"signal density $\alpha$"); ax.set_ylabel(r"excess CE over ideal-$\alpha$ (nats)")
ax.set_title("(a) combined parametrisation trains to worse CE"); ax.grid(True)
ax.legend(frameon=False, handlelength=1.9)
ax = axes[1]
a05 = 0.5
ax.plot(np.arange(1, len(h_tc05["test_acc"]) + 1),
        np.array(h_tc05["test_acc"]) / (a05 / 2), "-",
        color="#009E73", label=r"separate $W_Q,W_K,W_V,W_O$")
ax.plot(np.arange(1, len(h_tcc05["test_acc"]) + 1),
        np.array(h_tcc05["test_acc"]) / (a05 / 2), "--",
        color="#009E73", label=r"combined $W_{QK},W_{OV}$")
ax.axhline(1.0, color="#767676", lw=0.6, ls=":")
ax.text(228, 0.985, r"argmax ceiling $\alpha/2$", fontsize=6.3, ha="right",
        va="top", color="#555555")
ax.set_xlabel("epoch"); ax.set_ylabel(r"test accuracy $/\ (\alpha/2)$")
ax.set_ylim(top=1.035)
ax.set_title(r"(b) two-cell training at $\alpha=0.5$"); ax.grid(True)
ax.legend(frameon=False, loc="lower right")
fig.savefig(os.path.join(FIG, "fig_comb.pdf")); plt.close(fig)

# ═ Appendix: all-predictor curves + copy per-position diagnostic ════════════════
fig, ax = plt.subplots(figsize=(5.6, 3.4))
for k in ["ideal", "bayes", "ctx", "gd", "gd_comb", "mlp", "twocell", "twocell_comb"]:
    plot_c(ax, k, curve(k))
ax.set_xlabel(r"signal density $\alpha$"); ax.set_ylabel("test CE (nats)")
ax.set_title("all predictors, fresh evaluation data (5{,}000 sequences)")
ax.grid(True); ax.legend(frameon=False, ncol=2, fontsize=6.5)
fig.savefig(os.path.join(FIG, "fig_all_curves.pdf")); plt.close(fig)

import shutil
shutil.copy(os.path.join(F8, "inspect5_seq0_thr0.8_fp196.png"),
            os.path.join(FIG, "fig_inspect.png"))
print("figures written:", sorted(os.listdir(FIG)))
