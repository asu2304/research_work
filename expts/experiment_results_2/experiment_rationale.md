# Mechanistic Interpretability Experiments Report (V2)
... [Exp 1-3 content omitted for brevity] ...

---

### Experiment 4: Accuracy vs. Signal Count (Normalizing Noise)
* **File:** `exp4_signal_efficiency.png`
* **Why it was performed:** This experiment "normalizes" for the noise density. In Exp 1, low $x$ looks "slower" simply because signal tokens are farther apart in time. Exp 4 asks: "If we only look at the amount of useful information (signals) seen, does the model learn at the same speed regardless of noise?"
* **Implementation Strategy:** We use a cumulative sum on the ruled-mask to assign each token a "Signal Index." We then group accuracy by this index rather than by position $t$.
* **What to look for:** If all lines overlap, the model is perfectly noise-robust. If low $x$ lines are lower than high $x$ lines on this plot, it proves that "distractor" noise tokens cause catastrophic forgetting or structural degradation in the attention mechanism.
