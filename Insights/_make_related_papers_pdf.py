# Provenance script: generates related_papers_top20_2026-07-04.pdf (matplotlib only; no LaTeX on box).
# Papers verified against arXiv API on 2026-07-04. Re-run: python3 _make_related_papers_pdf.py
import textwrap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

OUT = "/home/hariguru/ashutosh/Insights/related_papers_top20_2026-07-04.pdf"

PAGE_W, PAGE_H = 8.27, 11.69  # A4 inches
MARGIN_L, MARGIN_R, MARGIN_T, MARGIN_B = 0.75, 0.75, 0.7, 0.7

INK = "#1a1a1a"
MUTED = "#555555"
LINK = "#1a56b0"
ACCENT = "#7a2f2f"

def wrap_chars(size):
    return int(6.77 * 72 / (size * 0.52))

class Doc:
    def __init__(self, pdf):
        self.pdf = pdf
        self.fig = None
        self.y = None
        self.pageno = 0
        self._new_page()

    def _new_page(self):
        if self.fig is not None:
            self._footer()
            self.pdf.savefig(self.fig)
            plt.close(self.fig)
        self.fig = plt.figure(figsize=(PAGE_W, PAGE_H))
        self.pageno += 1
        self.y = MARGIN_T

    def _footer(self):
        self.fig.text(0.5, MARGIN_B / 2 / PAGE_H, str(self.pageno),
                      ha="center", va="center", fontsize=8, color=MUTED)

    def need(self, inches):
        if self.y + inches > PAGE_H - MARGIN_B:
            self._new_page()

    def space(self, inches):
        self.y += inches

    def text_line(self, s, size=9.5, weight="normal", color=INK, url=None,
                  indent=0.0, style="normal"):
        line_h = size * 1.45 / 72
        self.need(line_h)
        t = self.fig.text((MARGIN_L + indent) / PAGE_W, 1 - (self.y + line_h * 0.8) / PAGE_H,
                          s, fontsize=size, fontweight=weight, color=color,
                          ha="left", va="baseline", fontstyle=style)
        if url:
            t.set_url(url)
        self.y += line_h

    def para(self, s, size=9.5, weight="normal", color=INK, indent=0.0, style="normal"):
        width = wrap_chars(size) - int(indent * 12)
        lines = []
        for chunk in s.split("\n"):
            lines.extend(textwrap.wrap(chunk, width=width) or [""])
        line_h = size * 1.45 / 72
        self.need(min(len(lines), 2) * line_h)
        for ln in lines:
            self.text_line(ln, size=size, weight=weight, color=color,
                           indent=indent, style=style)

    def close(self):
        self._footer()
        self.pdf.savefig(self.fig)
        plt.close(self.fig)

def entry(d, rank, title, meta, url, summary, why, delta):
    d.space(0.10)
    d.need(1.0)
    d.para(f"{rank}.  {title}", size=10.5, weight="bold")
    d.para(meta, size=8.8, color=MUTED, indent=0.22)
    d.text_line(url, size=8.8, color=LINK, url=url, indent=0.22)
    d.space(0.03)
    d.para(summary, size=9.3, indent=0.22)
    d.space(0.02)
    d.para("Why relevant:  " + why, size=9.3, indent=0.22, color="#2f4f2f")
    d.space(0.02)
    d.para("Vs. your work:  " + delta, size=9.3, indent=0.22, color=MUTED)
    d.space(0.06)

with PdfPages(OUT) as pdf:
    d = Doc(pdf)

    d.space(0.1)
    d.para("Closest Related Papers - Top 20, Ranked", size=17, weight="bold")
    d.space(0.05)
    d.para("For: In-Context Learning of Markov-chains-with-noise - statistical induction heads, "
           "noise-driven phase transitions, GD-vs-hand-construction (gauge-aware), combined "
           "W_QK/W_OV parametrization, FFN as Bayesian count-calibration head.", size=9.5, color=MUTED)
    d.space(0.06)
    d.para("Compiled 4 July 2026. All arXiv IDs, titles, authors and dates verified against the arXiv "
           "API on this date; summaries grounded in the papers' abstracts. Ranked #1 = closest to the "
           "research program overall, recency-weighted.", size=8.8, color=MUTED, style="italic")
    d.space(0.15)

    d.para("PART I - Top 10 closest papers", size=13, weight="bold", color=ACCENT)
    d.space(0.05)

    entry(d, 1,
          "What One Cannot, Two Can: Two-Layer Transformers Provably Represent Induction Heads on Any-Order Markov Chains",
          "Ekbote, Bondaschi, Rajaraman, Lee, Gastpar, Makkuva, Liang - arXiv 2508.07208, Aug 2025 (v2 Nov 2025), NeurIPS 2025",
          "https://arxiv.org/abs/2508.07208",
          "Proves a two-layer, single-head transformer can exactly represent the induction head / in-context "
          "conditional k-gram estimator for ANY-order Markov chains (prior constructions needed 3+ layers), and "
          "complements this with a gradient-descent analysis on a reduced two-layer variant showing GD learns the "
          "induction head on first-order chains.",
          "The theoretical mirror of your entire W*-vs-GD program at exactly your architecture depth; latest paper "
          "in the Makkuva line you already reproduce (themes: statistical induction heads, construction vs "
          "learnability, training dynamics).",
          "Clean Markov data only - no noise sub-vocabulary or alpha sweep, no FFN-calibration analysis, no "
          "gauge-aware weight-space comparison. Exactly where your work is novel.")

    entry(d, 2,
          "The Bayesian Geometry of Transformer Attention  (+ companion: Gradient Dynamics of Attention, arXiv 2512.22473)",
          "Agarwal et al. - arXiv 2512.22471, Dec 2025 (v5 May 2026)",
          "https://arxiv.org/abs/2512.22471",
          "Builds 'Bayesian wind tunnels' - controlled tasks (incl. HMM state tracking) with closed-form posteriors "
          "and provably no memorization - and shows small transformers reproduce the Bayes posterior to 1e-3 to "
          "1e-4 bits while capacity-matched MLPs fail. Mechanism: residual stream = belief state, FEED-FORWARD "
          "performs the posterior update, attention does content-addressable routing. Companion paper analyzes "
          "when cross-entropy GD reaches the Bayes-optimal manifold.",
          "Independent confirmation of your folder-8 result that the FFN is the posterior/calibration head "
          "(log-readout toward context-Bayes) on top of the attention count vector; companion touches your "
          "GD-convergence question.",
          "HMM/bijection tasks, not per-sequence random transition tables with swept noise density; no "
          "hand-constructed W* weight comparison. Cite and differentiate before submitting.")

    entry(d, 3,
          "Learning In-context n-grams with Transformers: Sub-n-grams Are Near-stationary Points",
          "Varre et al. - arXiv 2508.12837, Aug 2025",
          "https://arxiv.org/abs/2508.12837",
          "Proves hand-constructed sub-n-gram estimators are near-stationary points of the population cross-entropy "
          "loss, explaining why training proceeds through long plateaus with discrete jumps between statistical "
          "solutions of increasing order.",
          "Supplies the loss-landscape machinery for your alpha=0.8 non-convergence - GD stuck near a sub-optimal "
          "near-stationary point where the noise-skipping circuit never forms; generalizes the unigram-trap plateau.",
          "Plateaus arise from n-gram order structure, not from a noise-density knob; no noise tokens, no Bayes "
          "predictor hierarchy.")

    entry(d, 4,
          "Phase Transitions in Attention: A Bayesian Theory of Copy Head Emergence",
          "Lavie et al. - arXiv 2606.12058, June 2026 (freshest on this list)",
          "https://arxiv.org/abs/2606.12058",
          "Derives a closed-form posterior over the attention matrix for the copy subcircuit of an induction head, "
          "reduces it to a low-dimensional order-parameter space, and proves emergence is a FIRST-ORDER phase "
          "transition in training-data amount for softmax attention (second-order + crossover for linear "
          "attention); verified with Bayesian sampling and Adam training.",
          "First-principles formal language for your noise-driven phase transition - your alpha is a "
          "data-distribution knob controlling exactly such a transition in the copy/noise-skipping subcircuit.",
          "Control parameter is dataset size on a clean copy task, not noise density; no W* comparison, no "
          "noise sub-vocabulary.")

    entry(d, 5,
          "From Shortcut to Induction Head: How Data Diversity Shapes Algorithm Selection in Transformers",
          "Kawata et al. - arXiv 2512.18634, Dec 2025, NeurIPS 2025",
          "https://arxiv.org/abs/2512.18634",
          "Rigorous analysis of GD on a trigger-copy task proving a transition in the learned mechanism: sufficient "
          "diversity of trigger-to-trigger distances (low 'max-sum' ratio) yields a generalizing induction head; "
          "low diversity yields a positional shortcut that fails OOD. Derives the optimal pretraining distribution.",
          "Unusually tight connection: your noise tokens are exactly what stretch/randomize the distance from a "
          "state to its previous occurrence, so their max-sum distance ratio is a candidate formal predictor for "
          "your alpha<=0.7 forms / alpha>=0.8 fails boundary.",
          "Diversity, not noise density, is the knob; single layer; no Bayes hierarchy or noise floor.")

    entry(d, 6,
          "Unveiling Induction Heads: Provable Training Dynamics and Feature Learning in Transformers",
          "Chen, Sheen, Wang, Yang - arXiv 2409.10559, Sep 2024, NeurIPS 2024",
          "https://arxiv.org/abs/2409.10559",
          "First proof that GD dynamics on n-gram Markov data produce a generalized induction head in a "
          "two-attention-layer transformer; multi-phase analysis in which attention, the FFN and normalization "
          "co-adapt in a specific order (a chi-squared mutual-information criterion selects the parent token).",
          "One of the few theory papers assigning the FFN a provable functional role next to induction attention - "
          "the theoretical analogue of your log-readout/count-denoising finding - plus stage-wise learning order.",
          "No noise tokens, no formation-failure regime, no hand-built W* baseline.")

    entry(d, 7,
          "The Emergence of Sparse Attention: Impact of Data Distribution and Benefits of Repetition",
          "Zucchet et al. - arXiv 2505.17863, May 2025 (v2 Dec 2025), NeurIPS 2025",
          "https://arxiv.org/abs/2505.17863",
          "Shows theoretically and empirically how training-data statistics control WHEN an induction-like "
          "sparse-attention circuit emerges, with measured power laws for plateau-escape time. Mechanism: "
          "irrelevant/distractor tokens dilute the learning signal (~1/T) until attention concentrates via a "
          "feedback loop with the feed-forward pathway.",
          "The signal-to-noise-bottleneck mechanism is a plausible CAUSE of your phase transition - noise tokens "
          "are exactly the signal-diluting distractors; their emergence-time scalings are directly testable "
          "against your alpha sweep.",
          "Generic emergence-timing analysis; no per-sequence Markov tables, no Bayes hierarchy, no gauge analysis.")

    entry(d, 8,
          "Towards Understanding Transformers in Learning Random Walks",
          "Shi, Cao - arXiv 2511.23239, Nov 2025",
          "https://arxiv.org/abs/2511.23239",
          "Proves a one-layer transformer trained by GD reaches optimal prediction of random walks on circles, with "
          "an interpretable solution (attention = parent-state selector, value matrix = one-step transition); shows "
          "the conditions are tight via edge cases where GD with small initialization provably FAILS to converge "
          "to a good solution even on simple tasks.",
          "Another rigorous instance of 'the optimum is hand-constructible but GD sometimes cannot find it' - your "
          "alpha=0.8 story, and consistent with your observation that better initialization rescued convergence.",
          "Single structured chain (circle), not per-sequence random tables; no noise mixing; one layer.")

    entry(d, 9,
          "On the Role of Transformer Feed-Forward Layers in Nonlinear In-Context Learning",
          "Sun et al. - arXiv 2501.18187, Jan 2025 (v3 Oct 2025)",
          "https://arxiv.org/abs/2501.18187",
          "First dedicated theory of what position-wise FFN layers contribute to ICL: attention-only transformers "
          "are essentially limited to linear/kernel ICL, while interleaved FFNs enable genuinely nonlinear "
          "in-context algorithms.",
          "Formal counterpart of your folder-8 result that the FFN's entire value is the nonlinear count-domain "
          "step (log + noise floor) that a plain softmax readout cannot perform.",
          "Regression-style ICL; no connection to Bayesian smoothing under structured priors - your finite-sample "
          "/ structured-prior characterization of WHEN the FFN helps goes beyond it.")

    entry(d, 10,
          "In-Context Learning Is Provably Bayesian Inference: A Generalization Theory for Meta-Learning",
          "Wakayama, Imaizumi - arXiv 2510.10981, Oct 2025 (v3 June 2026)",
          "https://arxiv.org/abs/2510.10981",
          "Formalizes the Bayes-optimal in-context predictor and defines a 'Bayes Gap' measuring a trained "
          "transformer's distance from it, with non-asymptotic bounds in number of pretraining tasks and context "
          "length.",
          "The general statistical theory underneath your predictor hierarchy (ideal-alpha <= context-Bayes <= "
          "ctx-count) and your result that the FFN's finite-sample gain decays with context length.",
          "Purely statistical - no architecture-level attribution (attention vs FFN), no Markov/noise structure; "
          "your mechanistic account is the complement.")

    d.space(0.15)
    d.para("PART II - Next 10 (ranks 11-20): the current frontier in this exact field", size=13,
           weight="bold", color=ACCENT)
    d.space(0.02)
    d.para("Chosen for scoop-risk awareness: what people are trying right now (five are from 2026).",
           size=9, color=MUTED, style="italic")
    d.space(0.05)

    entry(d, 11,
          "Pre-trained Large Language Models Learn Hidden Markov Models In-context",
          "Dai et al. - arXiv 2506.07298, Jun 2025 (v3 Apr 2026)",
          "https://arxiv.org/abs/2506.07298",
          "Shows LLMs in-context-learn HMM-generated sequences, achieving next-token prediction near the "
          "theoretical optimum across sweeps of transition/emission entropies, degrading gracefully as mixing "
          "gets harder.",
          "Your noisy Markov chain IS an HMM (latent signal state, emissions that sometimes leave the state "
          "unchanged); their entropy sweep parallels your alpha sweep.",
          "Behavioral study of large pretrained models - no mechanism, no circuits, no constructions.")

    entry(d, 12,
          "Transformers Learn Latent Mixture Models In-Context via Mirror Descent",
          "D'Angelo, Flammarion - arXiv 2604.10848, Apr 2026",
          "https://arxiv.org/abs/2604.10848",
          "Formalizes estimating token importance as ICL of latent mixture models and shows attention implements a "
          "mirror-descent-like algorithm for inferring which past tokens are causally relevant and how much they "
          "matter.",
          "Same authors as Selective Induction Heads (the group working closest to you); 'which context tokens are "
          "signal vs irrelevant' is the abstract version of your signal/noise-token relevance problem.",
          "Latent mixture / token-importance framing, no Markov transition tables, no noise-density phase "
          "transition, no weight-level construction comparison.")

    entry(d, 13,
          "Incremental Learning of Sparse Attention Patterns in Transformers",
          "Yuksel et al. - arXiv 2602.19143, Feb 2026 (cites Edelman et al. 2024)",
          "https://arxiv.org/abs/2602.19143",
          "Introduces a high-order Markov chain task where past positions carry varying statistical significance "
          "and shows transformers learn it INCREMENTALLY - each training stage is the acquisition of a specific "
          "sparse attention pattern.",
          "Directly extends the statistical-induction-heads line you build on; stage-wise acquisition on Markov "
          "data is your pre-grok plateau story with a different difficulty knob.",
          "Difficulty comes from Markov order / position significance, not from a noise sub-vocabulary; no Bayes "
          "floor comparison or FFN analysis.")

    entry(d, 14,
          "Bayesian Optimality of In-Context Learning with Selective State Spaces",
          "Zhang et al. - arXiv 2602.17744, Feb 2026",
          "https://arxiv.org/abs/2602.17744",
          "Proposes Bayes-optimal sequential prediction as the organizing principle of ICL (meta-learning over "
          "latent sequence tasks); proves selective SSMs approximate the optimal Bayesian filter and converge "
          "faster to Bayes-optimal risk than linear transformers in structured-noise settings.",
          "The 'ICL = Bayes-optimal filtering under noise' thesis is your predictor-hierarchy worldview; also an "
          "architecture-comparison angle (SSM vs attention) for the same computation.",
          "SSMs rather than softmax transformers; LG-SSM tasks; no circuit-level or construction analysis.")

    entry(d, 15,
          "On the Emergence of Induction Heads for In-Context Learning",
          "Musat et al. - arXiv 2511.01033, Nov 2025 (v2 Jan 2026)",
          "https://arxiv.org/abs/2511.01033",
          "In a minimal transformer, shows induction-head training dynamics live in a 19-dimensional subspace of "
          "which only 3 principal directions (match + copy) matter; derives emergence-time bounds - quadratic in "
          "context length - matching the abrupt ICL phase transition.",
          "A reduced-coordinates account of exactly the circuit your W* hand-codes; gives analytic emergence-time "
          "scalings you could compare against your alpha-dependent formation times.",
          "Clean copy/induction task; no noise, no Bayes baselines, no readout-calibration analysis.")

    entry(d, 16,
          "How Transformers Get Rich: Approximation and Dynamics Analysis",
          "Wang et al. - arXiv 2410.11474, Oct 2024 (v3 Jan 2025)",
          "https://arxiv.org/abs/2410.11474",
          "Approximation + gradient-dynamics theory of how two-layer transformers implement induction heads, "
          "including sequential learning (2-gram before induction) and time-to-formation scaling quadratically "
          "with context length.",
          "Rigorous account of the unigram-to-induction learning order and of what the trained solution converges "
          "to relative to the ideal construction.",
          "Clean data; cannot capture a noise-driven formation failure.")

    entry(d, 17,
          "In-context Denoising with One-layer Transformers: Connections between Attention and Associative Memory Retrieval",
          "Smart et al. - arXiv 2502.05164, Feb 2025, ICML 2025",
          "https://arxiv.org/abs/2502.05164",
          "Frames ICL as in-context DENOISING: a single attention head is proven expressive enough to implement "
          "the Bayes-optimal denoiser for certain priors, and GD training from random init empirically recovers "
          "that optimal construction.",
          "Construction-vs-learnability with a denoising-with-a-prior view of ICL - the continuous cousin of your "
          "count-denoising FFN story.",
          "Continuous associative-memory setting; optimum realized in attention alone, not an FFN calibration "
          "stage.")

    entry(d, 18,
          "Symmetry Breaking in Transformers for Efficient and Interpretable Training",
          "Silverstein et al. - arXiv 2601.22257, Jan 2026",
          "https://arxiv.org/abs/2601.22257",
          "Analyzes training through the continuous rotational symmetries of the QK and VO sectors: tracks "
          "gauge-orbit motion (angular momentum / rotational fraction) of Wq,Wk vs Wv,Wo during training and "
          "tests symmetry-broken parametrizations; finds the VO sector substantially more rotational than QK.",
          "Directly your folder-7 question - how much of GD's motion is pure gauge vs meaningful - and your "
          "identifiability-vs-trainability trade-off for the combined parametrization.",
          "LLM-scale training diagnostics, not a Markov-ICL circuit identification against a hand construction.")

    entry(d, 19,
          "Weight Decay Induces Low-rank Attention Layers",
          "Kobayashi et al. - arXiv 2410.23819, Oct 2024, NeurIPS 2024",
          "https://arxiv.org/abs/2410.23819",
          "Proves that with L2 on the factors, local minima of L(AB^T) coincide with minima of the nuclear-norm "
          "regularized loss on the product, and the losses converge during training; empirically the attention "
          "products Wk^T Wq and P Wv acquire reduced rank.",
          "The cleanest formal statement of the implicit low-rank bias that the factored Wq,Wk,Wv,Wo "
          "parametrization has and your combined W_QK/W_OV parametrization loses - your theme-4 trainability gap.",
          "General regularization theory; no ICL/Markov task, no circuit comparison.")

    entry(d, 20,
          "Transformers as Multi-task Learners: Decoupling Features in Hidden Markov Models",
          "Hao et al. - arXiv 2506.01919, Jun 2025",
          "https://arxiv.org/abs/2506.01919",
          "Theoretical layerwise analysis of transformers on sequence tasks generated by hidden Markov models: "
          "lower layers decouple/extract latent features that upper layers combine for multiple downstream tasks.",
          "HMM-generated sequences with latent state extraction is the general version of your "
          "recover-state-from-last-signal (noise-skipping) sub-task.",
          "Multi-task representation angle; no noise-density sweep, constructions, or induction-circuit "
          "comparison.")

    d.space(0.15)
    d.para("ALREADY ON YOUR RADAR (would otherwise rank in the top 10)", size=11, weight="bold", color=ACCENT)
    d.space(0.03)
    for s, u in [
        ("Edelman et al. 2024, The Evolution of Statistical Induction Heads (arXiv 2402.11004, NeurIPS 2024) - still the closest single twin.",
         "https://arxiv.org/abs/2402.11004"),
        ("D'Angelo, Croce, Flammarion - Selective Induction Heads (now on arXiv as 2509.08184, Sep 2025; you have the PDF + reproduction notebook).",
         "https://arxiv.org/abs/2509.08184"),
        ("Makkuva et al. - Attention with Markov (arXiv 2402.04161, ICLR 2025) and Local to Global (arXiv 2406.03072, NeurIPS 2024) - you reproduce these.",
         "https://arxiv.org/abs/2402.04161"),
        ("Nichani, Damian, Lee 2024 (arXiv 2402.14735); Bietti et al. 2023 (2306.00802); Reddy ICLR 2024; Olsson et al. 2022; Elhage et al. 2021.",
         "https://arxiv.org/abs/2402.14735"),
        ("Zhou et al. - Transformers' ICL of Variable-Order Markov Chains (arXiv 2410.05493) - matches your VOMC_explained PDF; NOTE: updated v3 on 31 Mar 2026.",
         "https://arxiv.org/abs/2410.05493"),
    ]:
        d.para("- " + s, size=8.8, color=MUTED, indent=0.1)
        d.text_line("  " + u, size=8.2, color=LINK, url=u, indent=0.1)
        d.space(0.02)

    d.space(0.12)
    d.para("ALSO ON THE RADAR (verified, just below the cut)", size=11, weight="bold", color=ACCENT)
    d.space(0.03)
    for s, u in [
        ("What Happens During the Loss Plateau? (2506.13688, NeurIPS 2025) - plateau anatomy: repetition bias, representation collapse, attention-map bottleneck.", "https://arxiv.org/abs/2506.13688"),
        ("Beyond Induction Heads: Multi-Phase Circuit Emergence (2505.16694, ICML 2025) - sequential circuits: non-context -> semi-context -> full-context.", "https://arxiv.org/abs/2505.16694"),
        ("Predicting the Emergence of Induction Heads in LM Pretraining (2511.16893, v2 Feb 2026) - distributional predictors of formation (candidate covariates for your alpha boundary).", "https://arxiv.org/abs/2511.16893"),
        ("Decomposing Prediction Mechanisms for In-Context Recall (2507.01414, v2 Jun 2026) - competing recall-vs-prediction mechanisms on per-sequence random dynamics.", "https://arxiv.org/abs/2507.01414"),
        ("Measuring Affinity between Attention-Head Weight Subspaces via the Projection Kernel (2601.10266, Jan 2026) - gauge-invariant head-similarity metric useful for folder 7.", "https://arxiv.org/abs/2601.10266"),
        ("Transformer Models are Gauge Invariant (2412.14543, Dec 2024) - the invariance formalism behind weight-level comparisons.", "https://arxiv.org/abs/2412.14543"),
        ("What Needs to Go Right for an Induction Head? (2404.07129, ICML 2024) - causal clamping; phase transition from co-adapting subcircuits.", "https://arxiv.org/abs/2404.07129"),
        ("Transformers Simulate MLE for Sequence Generation in Bayesian Networks (2501.02547, v2 Jul 2025).", "https://arxiv.org/abs/2501.02547"),
        ("Exploring the Robustness of ICL with Noisy Labels (2404.18191) - label-noise (not token-noise) robustness of ICL.", "https://arxiv.org/abs/2404.18191"),
        ("Universal Response and Emergence of Induction in LLMs (2411.07071).", "https://arxiv.org/abs/2411.07071"),
    ]:
        d.para("- " + s, size=8.8, color=MUTED, indent=0.1)
        d.text_line("  " + u, size=8.2, color=LINK, url=u, indent=0.1)
        d.space(0.02)

    d.space(0.15)
    d.para("SCOOP-RISK ASSESSMENT (as of 4 July 2026)", size=11, weight="bold", color=ACCENT)
    d.space(0.03)
    d.para("No paper found combines your three pillars: (i) an explicit noise-token sub-vocabulary with swept "
           "density alpha driving a circuit-formation phase transition, (ii) a gauge-aware, weight-level "
           "comparison of GD solutions against an exact hand construction in the combined W_QK/W_OV "
           "parametrization, and (iii) the FFN characterized as a finite-sample Bayesian count-calibrator "
           "(log-readout + noise floor, gain vanishing for unstructured Dirichlet priors or infinite context). "
           "Closest scoop pressure: 2512.22471 (FFN = posterior update - cite and differentiate), 2512.18634 and "
           "2606.12058 (formal phase-transition boundaries - your noise-density version remains open), and "
           "2508.07208 (two-layer construction + GD, clean data). The field is moving fast on themes 2 and 6 "
           "(five 2026 papers above); the themes with least competition remain your gauge-aware W* comparison "
           "and the structured-prior FFN-gain characterization.", size=9.3)

    d.close()

print("written:", OUT)
