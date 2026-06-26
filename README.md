# 🎮 Video Game Hit-or-Flop Prediction

Predict whether a game will be a commercial **Hit** or **Flop** using only information available
**before release** (genre, platform, publisher reputation). Supervised binary classification, framed
along **CRISP-DM**.

---

## Locked decisions (the foundation — already done, do not redo)

| Area | Decision | Why |
|---|---|---|
| **Target** | `Hit = sales in the top X% of its release year` (primary X = **20%**; 20% vs 30% compared later) | year-relative = fair across a growing market; corroborated by the industry million-seller / *Greatest Hits* badge |
| **Scope** | `scope_2000_2018` (primary) + `scope_all` (≤2018, incl. pre-2000) | post-2018 is empty (data quality); pre-2000 kept only to *justify the cutoff empirically* |
| **Features** | `genre` (15) · `platform_family` (5) · `publisher_tier` (3) — all known pre-release | 81 consoles collapsed to families; small genres folded into *Misc*; tier = a publisher's past million-seller count |
| **Metric** | **ROC-AUC + F-measure (F1)** + confusion matrix | data is ~80/20 imbalanced → accuracy alone is misleading (Lec 5: "rare disease") |

---

## Repo

```
Phase_A_Evidence.ipynb     ← evidence for every decision above (DONE)
Phase_B_Pipeline.ipynb     ← leakage-safe preprocessing → data/processed/ (DONE)
Phase_C_Modeling.ipynb     ← modeling + comparison + synthesis (TODO)
data/raw/                  ← raw CSV
data/processed/<scope>/    ← X_train X_test y_train y_test  (+ meta_* )  ← train on THIS
src/  evidence/  results/
```

**Status:** Phases **A & B done**. Phase **C** = build the course's classifiers and compare them.

---

## Phase C — how to work on it (shared rules, so the 6 results stay comparable)

The models in scope are the course toolbox: **kNN, Naive Bayes, Decision Tree, Random Forest,
ANN (MLP), Logistic Regression** — each as its **own notebook**. *How* you build/tune your model is
your call (research it); the only shared rules are:

1. **Load** `data/processed/<scope>/` — **do not rebuild preprocessing.** The train/test split and the
   leakage-safe `publisher_tier` are already done. (`meta_*` is for synthesis only — **not** a feature.)
2. **Encoding** (your method, but mind the data type):
   - `genre`, `platform_family` are **nominal → one-hot**.
   - `publisher_tier` is **ordinal** (0 < 1 < 2) → numeric or one-hot both fine.
4. **Run each model twice**: no resampling, and with a **50/50 train downsample** — to measure the
   imbalance effect (Lec 5: balance classes during training).
5. **Evaluate** on the test set with **ROC-AUC + F-measure/f-score + confusion matrix**; always beat the majority
   baseline. Save to `results/<model>.csv` with columns: `model, resample, roc_auc, f1, accuracy`.

The results are then collected for the Phase C synthesis (model comparison, feature ablation, and the
scope `2000` vs `all` comparison — scored on identical test rows).

---

## Note on the dominant feature

`publisher_tier` carries most of the signal (a publisher's track record predicts new releases). This is
legitimate predictive signal, not leakage: it is computed **train-only**, and a temporal check
(train ≤2014 → test 2015–2018) confirms performance is not inflated. Expect a modest, honest
**ROC-AUC ≈ 0.78** — the ceiling reflects drivers we can't see pre-release (marketing, hype), which is
a finding, not a failure.
