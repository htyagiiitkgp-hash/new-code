# DP Healthcare Pipeline — Master Script

Single, complete pipeline accompanying the paper *"Privacy-Utility
Trade-offs in Healthcare Machine Learning: A Differential Privacy and
Ensemble Averaging Approach."* Running `master_pipeline.py` top-to-bottom
reproduces **every table and figure in the paper**, across all three
datasets, in one run.

| Section | Dataset | Reproduces |
|---|---|---|
| A | UCI Heart Disease (N=920, primary benchmark) | Table II, Table III, Figures 1–4, 8, 9 |
| B | Pima Indians Diabetes (N=768) | Table IV (Section 5.5) |
| C | Breast Cancer Wisconsin (N=569) | Table V (Section 5.5) |

## Requirements

```bash
pip install -r requirements.txt
```

## Datasets

Two of the three datasets are **not bundled** in this repo (standard
practice for UCI/Kaggle-sourced data) and must be supplied via environment
variable. The third ships automatically with scikit-learn.

```bash
export HEART_DATASET_PATH=/path/to/heart_disease_uci.csv   # UCI Heart Disease, N=920
export PIMA_DATASET_PATH=/path/to/pima-indians-diabetes.csv # Pima Diabetes, N=768 (header or not — auto-detected)
python master_pipeline.py
```

Breast Cancer Wisconsin needs no setup — it loads automatically via
`sklearn.datasets.load_breast_cancer`.

`HEART_DATASET_PATH` also accepts a `.xlsx` or a `.zip` containing one such
file. In Google Colab, Section A instead prompts an interactive upload
widget for the Heart Disease file (Pima and Breast Cancer are unaffected).

## Running

```bash
python master_pipeline.py
```

Runs Sections A → B → C in sequence and prints a final summary. Each
section is self-contained (its own function, own variable scope), so
nothing from one dataset leaks into another.

## Output

- `table2_privacy_accounting.csv`, `table3_heart_disease_performance.csv`,
  `epsilon_sweep_heart_disease.csv` — Heart Disease results (Section A)
- `table4_pima_diabetes.csv` — Pima Diabetes results (Section B)
- `table5_breast_cancer.csv` — Breast Cancer Wisconsin results (Section C)
- `figure1..4, 8, 9 _*.pdf` (+ matching `.png`) — Heart Disease figures,
  vector PDF
- `table4_pima_barchart.pdf/.png`, `table5_breast_cancer_barchart.pdf/.png`
  — supporting bar charts for the two generalization datasets

## Notes on reproducibility

- All differentially-private runs are stochastic (the DP models in
  Sections A and B are not seeded), which is why results are reported as
  means ± SD over repeated draws — 5-fold CV or 20 independent runs for
  Heart Disease, 10 independent trials for Breast Cancer Wisconsin — rather
  than a single-shot number. Re-running the script will reproduce the
  non-private baselines and the Rényi-DP accounting table (Table II)
  exactly, and will land the DP-based numbers in the same range reported
  in the paper, but not bit-for-bit identical for the Heart Disease and
  Pima single-run figures.
- `data_norm` (the L2 sensitivity bound used by diffprivlib) is computed
  per-dataset from the standardized training features; changes to
  preprocessing will change this value and should be re-verified.
- The Breast Cancer Wisconsin pipeline uses the *median* (not max) L2 norm
  as a more robust sensitivity bound, and fixes a `random_state` per trial
  — this is why its results reproduce exactly on re-run, unlike the other
  two datasets. See the comments in `run_breast_cancer_pipeline()` for why.

## Standalone per-dataset scripts

If you only need one dataset, the equivalent standalone scripts are also
included: `dp_healthcare_pipeline.py` (Heart Disease only),
`pima_diabetes_pipeline.py` (Pima only), `breast_cancer_pipeline.py`
(Breast Cancer only) — functionally identical to the corresponding
sections of `master_pipeline.py`.
