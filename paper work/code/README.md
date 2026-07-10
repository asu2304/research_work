# Code supplement — "In-Context Learning of Markov Chains under Noise"

This is the code that produced every number, table, and figure in `main.pdf`
(`paper work/main.tex`). It is organized by pipeline stage, not by the original
`ICL/1`–`ICL/10` folder numbering — see the provenance table below for the mapping.

Every script that trains or evaluates a model has a `CONFIG` dict near the top
(paths, device, hyperparameters); edit that, then run the script directly. No CLI
argument parsing is used anywhere, by design — matching how the experiments were
actually run.

## Directory layout

```
data_generation/
  generate_noisy_markov.py                 the main task (Sec. 3): per-sequence random
                                            2-successor tables + swept noise density alpha
  generate_unequal_successor_control.py    Appendix F control: same task, successor
                                            probabilities unequal (p1~U(0,1), p2=1-p1)

training/
  train_attention_only.py                  "attn" family (Sec. 4-5): plain softmax,
                                            separate W_Q,W_K,W_V,W_O, no FFN
  train_combined_qkov_attention_only.py    "attn, (W_QK,W_OV)" family (Sec. 7): same,
                                            gauge-free combined parametrization
  train_one_ffn.py                         "+FFN" family (Sec. 5-6): attn-only + one
                                            residual GELU FFN before the unembedding
  train_two_cell.py                        "two-cell" family (Sec. 6-7): FFN after
                                            BOTH attention layers, separate params
  train_two_cell_combined_qkov.py          "two-cell, (W_QK,W_OV)" family (Sec. 7)
  train_unequal_successor_control.py       Appendix F: identical arch to train_one_ffn.py,
                                            trained on the unequal-successor data

constructions/
  models_and_baselines.py                  THE shared harness: every architecture class
                                            (DisentangledTransformer and its 5 variants),
                                            the hand-built W* (PerfectInductionTransformer,
                                            Appendix D), the ideal-alpha / ctx-count analytic
                                            baselines, and all hand-set readout constructions
                                            (log, ReLU-log, GELU-log). Imported by everything
                                            in eval_and_figures/ and constructions/search_*.py.
  equalize_construction.py                 the GELU "equalize" FFN construction targeting
                                            context-Bayes (Sec. 6, Table 3 "eq" column) and the
                                            in-arch normalised-log readout ("inarch" column)
  search_ffn_log_floors.py                 offline grid search -> _floors.json (produces the
                                            FFN_FLOORS constants hardcoded in models_and_baselines.py)
  search_equalize_params.py                offline grid search -> _match.json (produces the
                                            EQ_PARAMS / INARCH_DELTA constants hardcoded in
                                            equalize_construction.py)
  _floors.json, _match.json                the search outputs already used to produce the paper

analysis/
  kl_divergence_calibration_table.py       Table 1 LEFT half (KL columns); Sec. 4 claim that
                                            attn-only GD is a temperature-calibrated count reader
  gauge_invariant_structural_comparison.py Table 1 RIGHT half (cosine columns); Appendix D
                                            (W* collapse, max|delta|=0) and Appendix E (gauge
                                            metrics); Sec. 7 identifiability discussion
  unequal_successor_control_appendixF.py   Table 5 + the finite-difference perturbation
                                            stability probe (Appendix F)
  hessian_probe_appendixF.py               Appendix F's exact gradient/Hessian probe (autograd
                                            gradient + finite-difference curvature along 300
                                            random directions in the FFN+readout subspace)

eval_and_figures/
  eval_all.py                              THE evaluation harness: computes every CE curve
                                            and post-noise wrong-successor rate in the paper
                                            on 5,000 freshly generated held-out sequences
                                            (seed 13423). Writes data/eval_test.json.
                                            -> Tables 3, 4; Figures 2, 3, 4, 5, 6, 7 (Appendix)
  make_figures.py                          reads data/eval_test.json + the training-history
                                            JSONs and renders all 7 vector-PDF figures used
                                            in main.tex

requirements.txt
```

## Provenance: original folder -> this package

| original ICL/ folder | script(s) here |
|---|---|
| `6. understanding what gradient descent is firuring out/data_generation.py` | `data_generation/generate_noisy_markov.py` |
| `6.../train.py` | `training/train_attention_only.py` |
| `6.../train_with_mlp.py` | `training/train_one_ffn.py` |
| `6.../opening_up_trained_model.ipynb` (cells 4, 15, 16) | `analysis/kl_divergence_calibration_table.py` |
| `7. unique_Wqk_Wok_test.ipynb/train_combined.py` | `training/train_combined_qkov_attention_only.py` |
| `7.../comparing_gd_softmax_vs_constructed.ipynb` (cells 1,2,7,8,9,10) | `analysis/gauge_invariant_structural_comparison.py` |
| `8. train with mlp/train_two_cell.py` | `training/train_two_cell.py` |
| `8.../train_two_cell_combined.py` | `training/train_two_cell_combined_qkov.py` |
| `8.../analyse.ipynb` (cell 0, cell 1) | `constructions/models_and_baselines.py`, `constructions/equalize_construction.py` |
| `8.../_find_floors.py`, `_find_match.py` | `constructions/search_ffn_log_floors.py`, `search_equalize_params.py` |
| `10. training mlp on slightly altered distribution/data_generation.py` | `data_generation/generate_unequal_successor_control.py` |
| `10.../train_identical_arch.py` | `training/train_unequal_successor_control.py` |
| `10.../main.ipynb` (cells 0, 5) | `analysis/unequal_successor_control_appendixF.py` |
| `10.../main.ipynb` (cells 0, 6) | `analysis/hessian_probe_appendixF.py` |
| (new, written for the paper) | `eval_and_figures/eval_all.py`, `eval_and_figures/make_figures.py` |

## Reproducing from scratch

Each `train_*.py` and `data_generation/*.py` has a `CONFIG['data_dir']` /
`CONFIG['models_dir']` pointing at a local `dataset/` / `models/` folder by
convention (matching how they were originally run, each from inside its own
project folder) — a few (`train_combined_qkov_attention_only.py`,
`train_two_cell.py`, `train_two_cell_combined_qkov.py`) instead point
`CONFIG['data_dir']` at an absolute path to the canonical dataset (all trained
model families share ONE dataset, generated once). Adjust these paths for your
layout, or recreate the original `ICL/6`, `ICL/7`, `ICL/8` directory structure and
run each script from inside the corresponding folder.

1. **Generate data** (`data_generation/generate_noisy_markov.py`): produces
   `dataset/{train,test}/data_x_{alpha}.pt` for alpha in {0.1,...,0.9}
   (20k/3k sequences, seq_len=200). Also run
   `generate_unequal_successor_control.py` if you want to reproduce Appendix F.
2. **Train the five model families** (`training/*.py`, one script per family,
   `CONFIG['thresholds']` controls which alpha to train — set to all nine, or
   loop the script per value). Each is checkpointed by best held-out CE and
   writes `metrics_*.json` + `training_summary*.json` alongside the `.pth`.
   Expect single-GPU training times of minutes-to-tens-of-minutes per
   (family, alpha) pair; the attention-only alpha=0.8 run additionally sweeps
   7 seeds (`CONFIG['seed_search']`) as described in the paper's Limitations.
3. **Offline construction searches** (`constructions/search_*.py`, optional —
   their outputs are already checked in as `_floors.json` / `_match.json` and
   hardcoded into `models_and_baselines.py` / `equalize_construction.py`; only
   re-run if you retrain W* or change the search grid).
4. **Evaluate everything** (`eval_and_figures/eval_all.py`): loads every
   checkpoint from step 2, evaluates all analytic baselines and hand
   constructions from step 3, on 5,000 freshly generated held-out sequences.
   Writes `eval_and_figures/data/eval_test.json` — every number in Tables 3-4
   and Figures 2-6 traces back to this one file.
5. **Render figures** (`eval_and_figures/make_figures.py`): reads step 4's
   JSON plus the raw training-history JSONs from step 2, writes 7 vector PDFs
   to `eval_and_figures/figures/`. Copy into `../../figures/` to refresh the
   versions embedded in `main.tex`.
6. **Table 1** (`analysis/kl_divergence_calibration_table.py` +
   `analysis/gauge_invariant_structural_comparison.py`) and **Table 5**
   (`analysis/unequal_successor_control_appendixF.py`) are computed
   separately, directly from the trained checkpoints — they are not part of
   the `eval_all.py` JSON.

## Honest notes on reproducibility

- **Not everything is single-seed.** Only the attention-only alpha=0.8 sweep
  used multiple seeds (7, with probe-and-abandon); every other (family, alpha)
  cell in the paper's tables is one training run. Exact millinat-level CE
  values may shift a little on a re-run with a different seed; every
  qualitative claim in the paper is checked to hold across multiple alpha and
  model families, not on a single number.
- **The hand-construction parameters** (`_floors.json`, `_match.json`, and the
  `SIG_LIST`/`BN_LIST`/`INARCH_DELTA` tables inside `models_and_baselines.py`
  and `equalize_construction.py`) were grid-searched once and are hardcoded —
  this is intentional (see the comments in those files), not a hidden
  per-run search. Re-run `search_ffn_log_floors.py` / `search_equalize_params.py`
  only if you change the architecture or the grid.
- **GPU device strings** (`cuda:1`, `cuda:2`, etc.) throughout are whatever GPU
  was free on the original shared box at the time — edit `CONFIG['device']` /
  `DEVICE_OVERRIDE` to match your hardware.
- All temp-file routing (`~/.cache/ashu_tmp`) is a workaround for the original
  box's unusable `/tmp`; harmless to keep, safe to delete on a normal machine.
