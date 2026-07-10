# ICLR 2027 draft — "In-Context Learning of Markov Chains under Noise"

First full draft, generated 2026-07-09. Target: ICLR 2027 (abstract deadline
**Sep 19, 2026**, full paper **Sep 24, 2026**).

## Files

| file | what |
|---|---|
| `main.tex` | the paper (main text + appendix), ICLR style |
| `refs.bib` | bibliography — every arXiv ID verified against the arXiv API on 2026-07-04/09 |
| `iclr2026_conference.{sty,bst}`, `fancyhdr.sty`, `natbib.sty` | ICLR template (2026; swap in the 2027 files when released — usually only the year string changes) |
| `figures/` | all paper figures (vector PDFs) + `fig_inspect.png` (copied from folder 8) |
| `eval_all.py` | the evaluation harness that produced every number in the paper (kept here as the exact script that was run; the packaged/documented copy is `code/eval_and_figures/eval_all.py`). Reuses the exact model/baseline definitions from `ICL/8. train with mlp/analyse.ipynb` (extracted verbatim into `_nb_cell0.py` / `_nb_cell1.py`). Read-only w.r.t. model checkpoints. |
| `data/eval_test.json` | its output: all CE curves + post-noise wrong-successor rates, computed on 5,000 freshly generated sequences (seed 13423), disjoint from training/selection data |
| `make_figures.py` | regenerates every figure from `data/eval_test.json` + training-history JSONs |
| `code/` | **the full code supplement for submission** — every data-generation, training, hand-construction, offline-search, and analysis script that produced the paper, reorganized by pipeline stage with a per-script provenance table. See `code/README.md`. Self-contained (404K, 22 files); zip this folder for OpenReview's supplementary-material upload. |

## Compile

- **Overleaf** (recommended): upload the whole folder; compiler pdfLaTeX; `main.tex`.
- **Locally**: no TeX on this box; a standalone tectonic binary lives at
  `~/.cache/ashu_tmp/tectonic/tectonic` → `cd 'paper work' && tectonic main.tex`.
  Note: tectonic (XeTeX) falls back from Times to Computer Modern (harmless font
  warnings); Overleaf/pdfLaTeX uses real Times, so page breaks may shift slightly.

## Before submitting — checklist

1. **Anonymize**: comment out `\iclrfinalcopy` in `main.tex` (ICLR is double-blind).
2. **Style**: replace with `iclr2027_conference.{sty,bst}` when the CFP template is out;
   update `\usepackage{iclr2026_conference,times}` and `\bibliographystyle`.
3. **Seeds**: most α were trained with a single seed (stated in Limitations). If time
   permits, rerun 3 seeds/α for error bars — the Limitations paragraph can then be relaxed.
4. **Numbers provenance**: if any model is retrained, rerun `eval_all.py` then
   `make_figures.py`, and update Tables 2–5 in `main.tex` from `data/eval_test.json`
   (tables are hardcoded).
5. **KL / cosine table (Table 1)**: taken from the folder-6/7 notebook outputs at
   N=200 sequences (noted in caption + appendix). Rerun those notebook cells if models change.
6. **Folder-10 table (Table 5)**: from `ICL/10.../main.ipynb` cell 5 outputs (2,000 seqs).
7. **Related work**: re-run a literature sweep close to the deadline — this field moves
   fast (see `Insights/related_papers_top20_2026-07-04.pdf` for the July snapshot).
8. Advisor pass: the claims in §5 ("fails to form") are deliberately budget-relative —
   phrased in Limitations; keep that framing.

## Submitting code alongside the paper

`code/` is ready to zip and attach as OpenReview supplementary material, or to push
to a (still-anonymous, for the double-blind period) repo linked from the paper. It
does NOT include model checkpoints or datasets (both are regenerable from the
scripts + fixed seeds; checkpoints total several hundred MB and don't belong in a
code submission). Before zipping:
1. Skim `code/README.md`'s provenance table against the final `main.tex` — if you
   change which model/analysis produces a table between now and submission, update
   both.
2. If double-blind, grep `code/` for the box's hostname/username in comments
   (`ashu_tmp`, `hariguru` cache paths) — these are harmless (just a workaround for
   a shared machine's broken /tmp) but you may prefer to scrub them for anonymity.

## Known honest caveats already stated in the paper

- Attention-only "GD tracks ctx-count" is NOT claimed — the unified eval showed a
  calibration gap (0.10–0.28 nats); the paper's claim is "temperature-calibrated count
  reader", supported by the KL table. FFN models are the ones at/below ctx-count.
- The α=0.8 phase-transition claim is scoped to the training protocol (7-seed sweep).
- Folder-10's "Bayes" notebook column is an add-1 Laplace proxy, not the exact
  posterior — the paper does not cite it as Bayes.
