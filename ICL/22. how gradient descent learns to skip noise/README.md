# 22 — How gradient descent learns to skip noise

**Read this first in any new session.** This folder is the project's final line of work. This README
is the complete record: what was done, in what order, what was found, and what is still open.

**Central question.** In-context learning of a Markov chain interleaved with noise tokens: the Markov
state can sit any number of noise tokens back. How does gradient descent, from zero, learn the
circuit that skips the noise and recovers the state?

**Where things stand (2026-10-03):** a complete six-part theorem with proofs,
`proofs/7_complete_theorem.pdf`, every part checked against training. The next planned step is
writing the paper (see §6).

---

## 0. Working rules for any session

- **Never modify or delete anything outside this folder.** Other folders are read-only inputs
  (folder 14 data, folder 18/19 checkpoints). `rm` is blocked by a hook (`~/.claude/hooks/guard_bash.py`).
  Rename instead of deleting, and never overwrite results.
- **Consult the user before launching new experiments or training runs** (standing preference).
  Analysis of existing files is fine.
- The user prefers **short, plain-language answers**, avoids jargon, and is not a theory specialist:
  explain proofs in easy words.
- Python: `/home/hariguru/miniconda3/envs/ashu_env/bin/python`, with `TMPDIR=~/.cache/ashu_tmp`
  (`/tmp` is unreliable on this box). Shared A100s: check `nvidia-smi`; cuda:2/3 were usually free.
- LaTeX: `~/.local/bin/tectonic file.tex` (escape `_` inside `\texttt{}`).
- **Units:** the five numbers are reported in **logits** = folder 19's raw `W_qk` entry / √108.
  Adam uses lr 1.0 on raw values (= 0.096 logits per sign-step); plain GD (`--opt sgd`) takes lr in logits.

## 1. Layout and numbering

| path | contents |
|---|---|
| `src/five.py` | the five-number model written directly in its scalars; data loaders; folder 18/19 readers |
| `src/f18.py` | folder 18's 10,060-entry model as a function of its two `W_qk` matrices |
| `experiments/eN_*.py` | **numbered in the order they were run**; letters (`e1b`, `e22c`…) are follow-ups to the same question |
| `proofs/N_*.tex/.pdf` | **numbered in the order they were written**; **7 is the final, complete document**, 1–6 are the steps that led to it |
| `runs/` | all outputs. `main/` = Adam sweep (19 α × 1000 steps, θ and gradient every step); `gd_lr30/` = plain GD sweep; `freeze_*/` = ablations; `eN_*.json` = experiment results; `logs/` = stdout |
| `dataset/` | 12 extra α values (folder 14's generator, disjoint seeds); the other 7 α come from folder 14 |
| `simplified_proofs/N_*.tex/.pdf` | **step-by-step versions of the proofs for the user** (undergraduate level, every intermediate step, worked examples). The originals in `proofs/` stay unchanged. Done so far: `1_covariance_law` |
| `figures/f1…f8` | f4 (tug-of-war on `c2`) and f8 (transfer to folder 18) are the key figures |
| `notebooks/analysis.ipynb` | walkthrough of night-1 results; rebuild with `notebooks/_build_notebook.py` |

**Proofs, in order:**
1. `1_exact_statements` — first exact results: gating, covariance law, symmetry ⇒ ties, what `c2` compares
2. `2_c2_b2_balance` — the `c2` formula; §6 adds when `b2` settles
3. `3_skip_seed` — the first layer-1 direction (veto ∝ noise × induction)
4. `4_pull_sign` — the initial induction pull is never negative (all 1296 tables)
5. `5_middle` — the middle stage under Adam (straight line, first turn)
6. `6_headline_theorem` — the first combined statement, with a status table
7. **`7_complete_theorem` — the final theorem: six parts, all proofs, finite-T checks**

## 2. The model and data

- **Data:** signal tokens 0–3, noise tokens 4–7, T = 100, loss on positions [50, 100). Each sequence draws
  its own table (every state has 2 equiprobable successors). Each step emits a successor w.p. α, else a
  uniform noise token (state unchanged).
- **Model (five numbers, all logits):** layer 1 score `−p·(i−j) + a1·[key is noise]`, giving `r_i` = mix of
  earlier tokens. When it works, `r_i` is the last signal token before i. Layer 2 score
  `a2·[key noise] + b2·r_j[x_i] + c2·<r_i, r_j>`, then copy the attended tokens (`W_ov = I`), and readout
  `(f+b)/sum` with the fixed bias `b`.
  - `p` recency, `a1` layer-1 noise veto (together: **skip / recover the state**);
  - `b2` induction;
  - `c2` state weight (**uses** the recovered state after noise);
  - `a2` layer-2 noise veto (stops copying noise; separable).
- **Assumed, not derived:** copy readout `W_ov = I`, the block layout, the linear ramp, one shared veto.
  Tying is justified by relabelling symmetry (exact). The reduction's faithfulness is checked in E1 and E7.

## 3. Chronological log

### Night 1 (2026-10-02 → 03)
- **E0** `e0_verify_reimplementation`: `src/five.py` reproduces folder 19 (loss Δ ≤ 1.2e-7, gradients ≤ 1.2e-6).
- **E0b** `e0b_make_data`: 12 new α (0.2–0.99). **E0c** `e0c_train_five`: training script (Adam/SGD, `--freeze`).
  Sweep: 19 α × 1000 steps → `runs/main`.
- **E1/E1b** projection of folder 18's solution onto five numbers. A global least-squares slope fails
  (1.896 vs 1.698): GD's free position profile is a **capped ramp**. The near-field slope recovers the loss
  within 0.0013. The tying symmetry holds (spread 0.1–0.2 logits).
- **E2** `e2_gradient_anatomy`: splits the gradient by query type (post-noise vs post-signal). **Three-phase `c2`
  story:** early, both types push `c2` up; once layer 1 localises, post-signal pushes it down; then they balance.
- **E3/E3b**: the early pull on `c2`/`b2` is an **order** effect (killed by shuffling), carried by keys that follow
  the current state. The `a2` pull is a frequency effect.
- **E5** `e5_covariance_law`: exact covariance law verified against autograd.
- **E6**: summary + figures f1–f7. Freeze ablations: `a1`+`p` skip, `c2` uses, `b2` teaches layer 1 at every α,
  `c2` teaches it only under dense noise. Learns to skip at every α in 0.2–0.99 (no trap).
- **Proof 1** written.

### Day 2 (2026-10-03)
- **E7** `e7_transfer_f18`: the `c2` tug-of-war (split, flip at layer-1 localisation, balance) **also holds in folder
  18** (10,060 free entries); the clean early phase does not. Figure f8.
- **E8/E8b/E9 → proof 2**: with layer 1 perfect, GD's `c2` equals the loss-optimal `c2` (within 0.25). Formula
  **c2\* = b2/2 + ½·log((1−α)/α) + κ, κ = 0.514 from the table distribution alone**. Matches GD within
  0.07 for α ∈ [0.6, 0.99], also during training. Fails for α ≤ 0.4.
- **E10/E10b**: `b2`'s loss has a knee (~7–11) then is flat. Settles under dense noise, drifts like log t otherwise.
- **E11/E12/E12b → proof 2 §6**: `b2` settles iff the cost of not-yet-seen successors beats the gain. The count
  model matches GD's settled `b2` within 0.1–0.4. `c2` must be included (without it, off by 2–4).
- **E13/E14**: layer 1 leaks to noise like a **cliff** at the horizon H = 1 + |a1|/p; there is no finite best
  layer 1 (flat, drifts).
- **E15**: `a2` is necessary (+0.08–0.16 nats without it) but **separable**: freezing it leaves the other four unchanged.
- **E16/E17 → proof 3**: first layer-1 direction = recency + veto; **veto ≈ (1−α) × induction pull** (exact formula).
- **E18/E19/E19b → proof 4**: initial induction pull ≥ 0 for **every** table (exact enumeration); = 0 iff the
  table is uninformative.
- **E20 → proof 5**: under Adam, the middle stage is a straight line (sign steps) on which the horizon = 2;
  the first turn is predicted within 1 step (`p` under dense noise, `c2` under sparse noise).
- **E21** `e21_gd_vs_adam` (plain GD, lr 30, 3000 steps): start, end, `b2` values, `c2` balance and the horizon law
  α·H ≈ 4 (α ≤ 0.6) all survive. The middle path and the slow veto at rare noise are Adam-specific.
- **Proof 6** (headline theorem draft).
- **E22–E22f → persistence**: gain lemma (`a1` only scales layer 1's signal output; exact). The veto strengthens at
  **every** training step (44,946/44,946). The rule: layer 1 localised + `c2` ≤ `c2*` ⇒ veto strengthens (0
  exceptions in 689 random points at α ≥ 0.6).
- **E23 → proof 7 (complete theorem)**: all parts as exact limits with proofs. Finite-T checks: accurate for
  α ≳ 0.6, drifting under dense noise.

## 4. The theorem (`proofs/7_complete_theorem.pdf`)

From zero, gradient descent:
1. **moves layer 2 first** — exact;
2. **always switches on the induction head** (off iff the table is uninformative) — exact limit + exact enumeration;
3. **seeds the skip:** pushes layer 1 to recency and to vetoing noise, the veto ∝ (1−α) × induction — exact limits;
4. **settles the state weight** at c2\* = b2/2 + ½·log((1−α)/α) + 0.514 — exact asymptotics (late regime);
5. **keeps strengthening the veto** while c2·g < c2\* on cliff positions (so c2 ≤ c2\* suffices) — exact reduction + asymptotics;
6. **settles the induction weight** iff unseen-successor cost > gain — exact expansion.

**Not covered:** middle-stage timing (Adam-specific), the skip-horizon law α·H ≈ 4 (measured), finite-T error
constants, models without the copy readout, other table families.

## 5. Hypotheses that turned out wrong

- the first gradient on `c2` scales with noise → no, it is largest with no noise;
- the early phase is a unigram effect → only for `a2`;
- rare noise makes GD fail to learn skipping → not in this model (that failure is folder 6's V=20 model);
- the linear ramp is GD's own position code → no, GD learns a capped ramp;
- the veto strengthens iff layer 2 wants to sharpen overall → no (28–71%); the right rule is per-position (E22f);
- `c2` ≤ balance alone ⇒ veto strengthens → needs layer 1 localised too.

## 6. What is next

- **Agreed next step:** write an A\* **workshop** paper from what exists (4 pages + appendix): lead with the `c2`
  prediction and figure f4, the theorem in plain words, proofs in the appendix, and an honest limitations
  section. Estimated acceptance with good writing: roughly 60–75% (a guess).
- **For a main conference (~2–3 weeks more):** error bars (data resampling via `e0b_make_data.py`); a 6–8 signal-token
  check; a transfer test on the 58k (blocks-full) model; positioning against Edelman et al. 2024 and
  2512.18634 (data diversity selects the algorithm); finite-T error control.
- The workshop or venue choice is pending from the user.

## 7. Reproduce

```bash
PY=/home/hariguru/miniconda3/envs/ashu_env/bin/python; export TMPDIR=~/.cache/ashu_tmp
cd "22. how gradient descent learns to skip noise"
$PY experiments/e0_verify_reimplementation.py cuda:2
$PY experiments/e0b_make_data.py 0.2 0.25 0.3 0.35 0.45 0.55 0.65 0.75 0.85 0.95 0.975 0.99
$PY experiments/e0c_train_five.py --alphas <...> --steps 1000 --device cuda:2 --tag main
$PY experiments/e0c_train_five.py --alphas 0.2 0.4 0.6 0.8 1.0 --steps 1000 --tag freeze_c2 --freeze c2   # also b2 a1 p a2
$PY experiments/e0c_train_five.py --alphas <...> --steps 3000 --opt sgd --lr 30 --tag gd_lr30
# then any experiments/eN_*.py (most take a device argument); e6 redraws figures f1-f7
cd proofs && ~/.local/bin/tectonic 7_complete_theorem.tex
```
One 1000-step run takes ~1.2 min on an A100. Most analyses take seconds to minutes.
