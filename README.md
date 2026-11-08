# Student Performance — Classifier Comparison

A Streamlit app that trains and compares 5 classic classification algorithms
to predict whether a student passes their course, using the UCI Student
Performance dataset.

## a. Problem Statement

Predict whether a student will **pass or fail** a course based on
demographic, social, and study-habit attributes (family background, study
time, prior failures, alcohol consumption, free time, internet access,
etc.) — *without* using their period grades (G1/G2), so the model has to
learn from behavioral/contextual factors rather than just extrapolating
from earlier grades. This is a binary classification problem, trained and
compared across 5 different algorithms.

## b. Dataset Description

Source: [UCI Student Performance Dataset](https://archive.ics.uci.edu/dataset/320/student+performance)
(Cortez & Silva, 2008) — the Portuguese-language-course file
(`student-por.csv`): **649 students, 30 features** (well above the
assignment's minimum of 12 features / 500 instances).

- **Target**: `pass` — derived as `1` if final grade `G3 >= 10` (out of
  20), else `0`. Test-split pass rate: **84.6%** (imbalanced).
- **Features (30)**: demographics (`age`, `sex`, `address`, `famsize`,
  `Pstatus`, parents' education/jobs), study habits (`studytime`,
  `failures`, `schoolsup`, `famsup`, `paid`, `activities`), social factors
  (`famrel`, `freetime`, `goout`, `Dalc`, `Walc`, `romantic`,
  `internet`), `health`, `absences`, etc.
- **Dropped**: `G1`, `G2`, `G3` — the three grade columns. `G3` is the
  target source; `G1`/`G2` (earlier-period grades) are excluded too since
  they're highly correlated with `G3` and would make the task close to
  trivial (this is the "hardest" of the three prediction settings
  described in the dataset's own documentation).

Prepared via `prepare_student_data.py` (downloads the UCI zip, derives
`pass`, drops the grade columns). Split: stratified 80/20 train/test
(`random_state=42`).

## c. GitHub Repository Link

https://github.com/thirumalakiran/ML_ASSIGNMENT_2

## d. Models Used

Each model is a scikit-learn `Pipeline` (one-hot encode the 17 categorical
features, standard-scale the 13 numeric features, then classify) — no
separate preprocessing artifact is needed since it's bundled into each
saved `.joblib`.

### Comparison Table (test split)

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.7923 | 0.6227 | 0.8673 | 0.8909 | 0.8789 | 0.1508 |
| Decision Tree | 0.7769 | 0.5409 | 0.8584 | 0.8818 | 0.8700 | 0.0876 |
| KNN | 0.8077 | 0.6066 | 0.8571 | 0.9273 | 0.8908 | 0.1002 |
| Naive Bayes | 0.6308 | 0.4950 | 0.8690 | 0.6636 | 0.7526 | 0.0857 |
| Support Vector Machine | 0.8308 | 0.6691 | 0.8607 | 0.9545 | 0.9052 | 0.1570 |

For reference, the test split's pass rate (i.e. the accuracy of a trivial
"always predict pass" baseline) is **84.6%** — every model's raw accuracy
sits at or below that, which is exactly why MCC matters here: it's the
one metric that isn't flattered by the class imbalance.

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Accuracy (79.2%) is actually below the 84.6% majority-class baseline, but it has the 2nd-highest MCC (0.151) and a genuinely above-chance AUC (0.623) — it's finding a real, if weak, linear signal in the behavioral features rather than just defaulting to the majority class. |
| Decision Tree | Middling everywhere (MCC 0.088, AUC 0.541 — close to random). Without the strongly predictive G1/G2 grades, there isn't a clean set of thresholds in the remaining features for it to split on, so it tends to overfit noise in the training split. |
| KNN | Highest recall (92.7%) but that's mostly it defaulting to the majority class in a soft-imbalanced neighborhood — MCC (0.100) shows limited real separating power. |
| Naive Bayes | Clearly weakest (accuracy 63.1%, MCC 0.086) — its independence assumption is a poor fit for correlated social/family features (e.g. `Dalc`/`Walc`, `Medu`/`Fedu`), and it ends up flagging far more false "fail" predictions than the other models. |
| Support Vector Machine | Best across the board — highest accuracy (83.1%), highest AUC (0.669), highest recall (95.5%), and highest MCC (0.157). The RBF kernel's non-linear decision boundary captures interactions between behavioral features (e.g. `failures` × `studytime`) that the linear/tree/distance-based models miss. |
| **Overall, Winner for your dataset?** | **Support Vector Machine** — highest MCC (0.157) and highest AUC (0.669) of the five. That said, every model's MCC is fairly low (0.09–0.16), which honestly reflects that pass/fail is hard to predict from behavioral/demographic data alone once the grade columns are removed — a realistic result, not a data-leakage artifact. |

## Project Structure

```
project-folder/
│-- app.py                      # Streamlit app
│-- train_models.py             # Generic trainer: any CSV + target column
│-- prepare_student_data.py     # Downloads & prepares this specific dataset
│-- requirements.txt
│-- README.md
│-- test_data.csv               # Held-out test split (generated by train_models.py)
│-- model/                      # Saved trained models + comparison metrics
│   │-- logistic_regression.joblib
│   │-- decision_tree.joblib
│   │-- knn.joblib
│   │-- naive_bayes.joblib
│   │-- svm.joblib
│   └-- metrics.csv
└-- data/                       # Cached dataset download (gitignored)
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

1. Prepare the dataset:

   ```bash
   python prepare_student_data.py
   ```

2. Train all 5 models (saves them to `model/`, writes `test_data.csv` and
   `model/metrics.csv`):

   ```bash
   python train_models.py --data data/student_performance.csv --target pass
   ```

   (`train_models.py` itself is dataset-agnostic — it takes `--data`,
   `--target`, and optional `--drop` arguments and auto-detects categorical
   vs numeric features, so it can be pointed at any CSV.)

3. Launch the app:

   ```bash
   streamlit run app.py
   ```

4. In the app, upload a test CSV (the committed `test_data.csv`, or the
   "Download sample test data" button in the app), pick the target/feature
   columns, and the app evaluates all 5 saved models against it (inference
   only — nothing is retrained) to populate the comparison table,
   observations, and winner. Pick one model from the dropdown to inspect
   its confusion matrix and classification report.
