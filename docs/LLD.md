# 📐 Low-Level Design — The Invisible Risk

> Feature Specifications, Schema Definitions, Missing Value Strategy & Module Contracts

**Project:** Financial Default Intelligence — Home Credit Dataset  
**Version:** 1.0 | April 2026

---

## Table of Contents

1. [Data Layer](#1-data-layer)
2. [Feature Engineering Modules](#2-feature-engineering-modules)
3. [Categorical Encoding Module](#3-categorical-encoding-module)
4. [NLP Embeddings Module](#4-nlp-embeddings-module)
5. [Model Training Module](#5-model-training-module)
6. [Evaluation Module](#6-evaluation-module)
7. [Missing Value Strategy](#7-missing-value-strategy)
8. [Feature Catalog (210 Features)](#8-feature-catalog)
9. [Inference API](#9-inference-api)

---

## 1. Data Layer

### 1.1 Source Tables

| Table | Rows (approx.) | Grain | Join Key |
|---|---|---|---|
| `application_train.csv` | 307,511 | 1 row per loan application | `SK_ID_CURR` |
| `application_test.csv` | 48,744 | Same as above | `SK_ID_CURR` |
| `bureau.csv` | 1,716,428 | 1 row per prior credit record | `SK_ID_CURR` |
| `bureau_balance.csv` | 27,299,925 | Monthly snapshot per bureau record | `SK_ID_BUREAU` |
| `previous_application.csv` | 1,670,214 | 1 row per past application | `SK_ID_CURR` |
| `installments_payments.csv` | 13,605,401 | 1 row per installment payment | `SK_ID_PREV` |
| `POS_CASH_balance.csv` | 10,001,358 | Monthly POS/cash loan snapshot | `SK_ID_PREV` |
| `credit_card_balance.csv` | 3,840,312 | Monthly credit card snapshot | `SK_ID_PREV` |

### 1.2 `loader.py` — Contract

**Input:** Path to `data/raw/`  
**Output:** Dictionary of DataFrames keyed by table name  
**Behavior:**
- Reads all 8 CSVs with `pd.read_csv()`, low memory mode disabled
- Returns `{"application_train": df, "bureau": df, ...}`
- Raises `FileNotFoundError` if any required file is absent
- Does **not** perform any transformations — raw pass-through only

---

## 2. Feature Engineering Modules

All modules share the same interface contract:

```
Input:  dict of raw DataFrames (from loader.py)
Output: pd.DataFrame — one row per SK_ID_CURR, engineered features only
```

Results from all modules are merged on `SK_ID_CURR` in `engineering_main.py`.

---

### 2.1 `engineering_app.py` — Application Table Features

**Source table:** `application_train/test.csv`

#### Feature Specifications

| Feature Name | Formula / Logic | Dtype | Null Strategy |
|---|---|---|---|
| `CREDIT_INCOME_RATIO` | `AMT_CREDIT / AMT_INCOME_TOTAL` | float64 | Fill with median |
| `ANNUITY_INCOME_RATIO` | `AMT_ANNUITY / AMT_INCOME_TOTAL` | float64 | Fill with median |
| `CREDIT_ANNUITY_RATIO` | `AMT_CREDIT / AMT_ANNUITY` | float64 | Fill with median |
| `GOODS_CREDIT_RATIO` | `AMT_GOODS_PRICE / AMT_CREDIT` | float64 | Fill with 1.0 (no goods) |
| `EMPLOYED_RATIO` | `DAYS_EMPLOYED / DAYS_BIRTH` | float64 | Clip then fill median |
| `AGE_YEARS` | `abs(DAYS_BIRTH) / 365.25` | float64 | No nulls expected |
| `EMPLOYMENT_YEARS` | `abs(DAYS_EMPLOYED) / 365.25` | float64 | Replace 365243 → NaN first |
| `EXT_SOURCE_MEAN` | `mean(EXT_SOURCE_1, 2, 3)` | float64 | Mean of available, else -1 |
| `EXT_SOURCE_PROD` | `EXT_SOURCE_1 × EXT_SOURCE_2 × EXT_SOURCE_3` | float64 | Replace NaN → 0 before product |
| `EXT_SOURCE_STD` | `std(EXT_SOURCE_1, 2, 3)` | float64 | Fill with 0 if only 1 available |
| `EXT_SOURCE_MISSING_COUNT` | Count of nulls across EXT_SOURCE_1/2/3 | int8 | Always computable |

**Edge case — `DAYS_EMPLOYED`:** Value `365243` is a sentinel for "unemployed/retired". Replace with `NaN` before computing `EMPLOYMENT_YEARS` and `EMPLOYED_RATIO`.

**Clipping:** `EMPLOYED_RATIO` clipped to `[0, 1]` to handle edge cases where employment exceeds age.

---

### 2.2 `engineering_bureau.py` — Bureau History Features

**Source tables:** `bureau.csv`, `bureau_balance.csv`

#### Aggregation Logic

Bureau records are grouped by `SK_ID_CURR`. The following aggregates are computed:

| Feature Name | Aggregation | Source Column | Filter |
|---|---|---|---|
| `BUREAU_LOAN_COUNT` | count | `SK_ID_BUREAU` | All records |
| `BUREAU_ACTIVE_COUNT` | count | `CREDIT_ACTIVE == 'Active'` | Active only |
| `BUREAU_CLOSED_COUNT` | count | `CREDIT_ACTIVE == 'Closed'` | Closed only |
| `BUREAU_ACTIVE_DEBT_SUM` | sum | `AMT_CREDIT_SUM_DEBT` | Active only |
| `BUREAU_ACTIVE_DEBT_MEAN` | mean | `AMT_CREDIT_SUM_DEBT` | Active only |
| `BUREAU_OVERDUE_COUNT` | count | `CREDIT_DAY_OVERDUE > 0` | All records |
| `BUREAU_OVERDUE_DEBT_SUM` | sum | `AMT_CREDIT_SUM_OVERDUE` | All records |
| `BUREAU_MAX_OVERDUE_DAYS` | max | `CREDIT_DAY_OVERDUE` | All records |
| `BUREAU_AVG_PROLONGED` | mean | `CNT_CREDIT_PROLONG` | All records |
| `BUREAU_CREDIT_TYPES` | nunique | `CREDIT_TYPE` | All records |

**Bureau Balance Sub-aggregation:**  
For each `SK_ID_BUREAU`, compute `STATUS_DPD_COUNT` (count of months where `STATUS` ∈ `{1, 2, 3, 4, 5}`), then aggregate to `SK_ID_CURR` level as `BUREAU_BALANCE_DPD_MONTHS_MEAN` and `BUREAU_BALANCE_DPD_MONTHS_MAX`.

**Null handling for bureau features:** Applicants with no bureau records get all bureau features filled with `0`. Rationale: absence of bureau history is a distinct, informative state — not missing data.

---

### 2.3 `engineering_prev.py` — Previous Applications Features

**Source table:** `previous_application.csv`

| Feature Name | Aggregation | Filter |
|---|---|---|
| `PREV_APP_COUNT` | count total | All |
| `PREV_APPROVED_COUNT` | count | `NAME_CONTRACT_STATUS == 'Approved'` |
| `PREV_REFUSED_COUNT` | count | `NAME_CONTRACT_STATUS == 'Refused'` |
| `PREV_APPROVAL_RATE` | `APPROVED / COUNT` | All |
| `PREV_AMT_CREDIT_MEAN` | mean | Approved only |
| `PREV_AMT_CREDIT_MAX` | max | Approved only |
| `PREV_AMT_REQUESTED_MEAN` | mean | `AMT_APPLICATION` all |
| `PREV_REQUEST_VS_APPROVED_RATIO` | `mean(AMT_APPLICATION) / mean(AMT_CREDIT)` | Approved |
| `PREV_DAYS_LAST_DUE_MEAN` | mean | `DAYS_LAST_DUE` all |
| `PREV_DAYS_TERMINATION_MEAN` | mean | `DAYS_TERMINATION` all |
| `PREV_CONSUMER_LOAN_COUNT` | count | `NAME_CONTRACT_TYPE == 'Consumer loans'` |
| `PREV_CASH_LOAN_COUNT` | count | `NAME_CONTRACT_TYPE == 'Cash loans'` |

**Null handling:** Fill all `prev_` features with `0` for applicants with no prior application history.

---

### 2.4 `engineering_installments.py` — Installment Payment Behavior

**Source table:** `installments_payments.csv`

#### Computed Per-Row Fields (before aggregation)

```
DPD  = max(DAYS_ENTRY_PAYMENT - DAYS_INSTALMENT, 0)   # Days past due
DBD  = max(DAYS_INSTALMENT - DAYS_ENTRY_PAYMENT, 0)   # Days before due
PAYMENT_RATIO  = AMT_PAYMENT / AMT_INSTALMENT          # Fraction paid
PAYMENT_DIFF   = AMT_INSTALMENT - AMT_PAYMENT          # Underpayment amount
```

#### Aggregated Features (grouped by `SK_ID_CURR`)

| Feature Name | Aggregation of |
|---|---|
| `INSTAL_DPD_MEAN` | `DPD` — mean |
| `INSTAL_DPD_MAX` | `DPD` — max |
| `INSTAL_DBD_MEAN` | `DBD` — mean |
| `INSTAL_DBD_MAX` | `DBD` — max |
| `INSTAL_PAYMENT_RATIO_MEAN` | `PAYMENT_RATIO` — mean |
| `INSTAL_PAYMENT_RATIO_MIN` | `PAYMENT_RATIO` — min |
| `INSTAL_PAYMENT_DIFF_MEAN` | `PAYMENT_DIFF` — mean |
| `INSTAL_PAYMENT_DIFF_MAX` | `PAYMENT_DIFF` — max |
| `INSTAL_COUNT` | total installment records |
| `INSTAL_DPD_NONZERO_RATE` | count(DPD > 0) / COUNT |

**Null handling:** Fill all installment features with `0` for applicants with no installment history.

---

### 2.5 `engineering_pos_cc.py` — POS Cash & Credit Card Features

**Source tables:** `POS_CASH_balance.csv`, `credit_card_balance.csv`

#### POS Cash Aggregates (grouped by `SK_ID_CURR`)

| Feature Name | Aggregation |
|---|---|
| `POS_MONTHS_BALANCE_MEAN` | mean of `MONTHS_BALANCE` |
| `POS_SK_DPD_MEAN` | mean of `SK_DPD` |
| `POS_SK_DPD_MAX` | max of `SK_DPD` |
| `POS_SK_DPD_DEF_MEAN` | mean of `SK_DPD_DEF` |
| `POS_ACTIVE_MONTHS` | count where `NAME_CONTRACT_STATUS == 'Active'` |
| `POS_COMPLETED_MONTHS` | count where `NAME_CONTRACT_STATUS == 'Completed'` |

#### Credit Card Aggregates (grouped by `SK_ID_CURR`)

| Feature Name | Aggregation |
|---|---|
| `CC_BALANCE_MEAN` | mean of `AMT_BALANCE` |
| `CC_BALANCE_MAX` | max of `AMT_BALANCE` |
| `CC_CREDIT_LIMIT_MEAN` | mean of `AMT_CREDIT_LIMIT_ACTUAL` |
| `CC_UTILIZATION_MEAN` | mean of `AMT_BALANCE / AMT_CREDIT_LIMIT_ACTUAL` |
| `CC_UTILIZATION_MAX` | max of utilization ratio |
| `CC_DRAWINGS_ATM_MEAN` | mean of `AMT_DRAWINGS_ATM_CURRENT` |
| `CC_DRAWINGS_TOTAL_MEAN` | mean of `AMT_DRAWINGS_CURRENT` |
| `CC_ATM_RELIANCE_RATIO` | `CC_DRAWINGS_ATM_MEAN / CC_DRAWINGS_TOTAL_MEAN` |
| `CC_DPD_MEAN` | mean of `SK_DPD` |
| `CC_DPD_MAX` | max of `SK_DPD` |
| `CC_ACTIVE_MONTHS` | count of active month records |

**ATM reliance flag:** `CC_HIGH_ATM_RELIANCE` = 1 if `CC_ATM_RELIANCE_RATIO > 0.5`, else 0. Cash-dominant card usage is associated with liquidity stress.

**Null handling:** Fill all POS and CC features with `0` for applicants without history.

---

## 3. Categorical Encoding Module

**File:** `categorical_encoding.py`

### Strategy by Cardinality

| Cardinality | Strategy | Threshold | Example Columns |
|---|---|---|---|
| Low (≤ 10 unique) | One-Hot Encoding | ≤ 10 | `NAME_CONTRACT_TYPE`, `FLAG_OWN_CAR` |
| Medium (11–50) | One-Hot Encoding | 11–50 | `OCCUPATION_TYPE`, `NAME_EDUCATION_TYPE` |
| High (> 50) | Target + Frequency Encoding | > 50 | `ORGANIZATION_TYPE`, `CITY_REG` |

### One-Hot Encoding

- `pd.get_dummies()` with `drop_first=False`
- `dtype=np.uint8` to minimize memory
- Column naming convention: `{ORIGINAL_COL}_{VALUE}` (spaces replaced with `_`)
- Unknown categories at inference time: all dummy columns set to 0

### Target Encoding

- Compute mean of `TARGET` per category on **training fold only** (never on full train to prevent leakage)
- Applied inside cross-validation loop in `train_lgbm.py`
- Smoothing formula: `(n_i × mean_i + alpha × global_mean) / (n_i + alpha)` where `alpha = 10`
- Categories unseen at inference: fallback to `global_mean`
- **Implementation class:** `category_encoders.TargetEncoder` with smoothing

### Frequency Encoding

- Compute value counts normalized on training data
- Replace category with frequency fraction
- Unseen at inference: frequency = 0

---

## 4. NLP Embeddings Module

**File:** `nlp_embeddings.py`

### Pipeline

```
Raw text column(s)
      ↓
SentenceBERT  [model: 'all-MiniLM-L6-v2']
      ↓
768-dim dense embedding per record
      ↓
PCA (n_components=32, fit on train only)
      ↓
32 float32 features named embed_0 … embed_31
```

### Text Columns Used

Text is constructed by concatenating:
- `OCCUPATION_TYPE`
- `ORGANIZATION_TYPE`
- `NAME_INCOME_TYPE`
- `NAME_HOUSING_TYPE`

Format: `"{OCCUPATION_TYPE} works at {ORGANIZATION_TYPE}, income type {NAME_INCOME_TYPE}, housing {NAME_HOUSING_TYPE}"`

Null values in source columns are replaced with `"unknown"` before concatenation.

### PCA Fit/Transform Split

The PCA transformer is **fit only on training data** and persisted to `models/pca_embeddings.pkl`. At inference the saved transformer is loaded and only `transform()` is called — never `fit_transform()`.

### Variance Retained

Target: ≥ 85% cumulative explained variance with 32 components. Verified at training time; logged to stdout.

---

## 5. Model Training Module

**File:** `train_lgbm.py`

### Cross-Validation Setup

```
StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
Metric optimized: ROC-AUC
```

### Hyperparameter Search — Optuna

| Parameter | Search Space | Type |
|---|---|---|
| `num_leaves` | [20, 300] | int |
| `max_depth` | [3, 12] | int |
| `learning_rate` | [0.005, 0.3] | log-uniform float |
| `n_estimators` | [100, 2000] | int |
| `min_child_samples` | [10, 100] | int |
| `subsample` | [0.5, 1.0] | float |
| `colsample_bytree` | [0.5, 1.0] | float |
| `reg_alpha` | [1e-8, 10.0] | log-uniform float |
| `reg_lambda` | [1e-8, 10.0] | log-uniform float |

**Trials:** 100  
**Pruner:** `MedianPruner` (stops unpromising trials early)  
**Objective:** Maximize mean CV ROC-AUC

### Class Imbalance Handling

Target class distribution: ~8% defaults. Strategy:

```python
scale_pos_weight = n_negatives / n_positives   # ≈ 11.5
```

Set as a fixed LightGBM parameter (not tuned) to maintain stable calibration during hyperparameter search.

### Output Artifacts

| Artifact | Path | Format |
|---|---|---|
| Best model | `models/lgbm_best.pkl` | joblib pickle |
| Optuna study | `models/optuna_study.pkl` | joblib pickle |
| OOF predictions | `models/oof_predictions.npy` | numpy array |
| Best params | `models/best_params.json` | JSON |
| PCA transformer | `models/pca_embeddings.pkl` | joblib pickle |
| Target encoder maps | `models/target_encoders.pkl` | joblib pickle |

---

## 6. Evaluation Module

**File:** `evaluate_lgbm.py`

### Metrics Computed

| Metric | Notes |
|---|---|
| ROC-AUC | Primary ranking metric |
| PR-AUC | Better for imbalanced classes |
| F1-Score | At F1-optimal threshold |
| Recall (defaults) | Priority metric — missed defaults are costly |
| Precision (defaults) | Tradeoff partner to recall |
| Brier Score | Probability calibration quality |

### Threshold Tuning

Default threshold of 0.5 is suboptimal for imbalanced credit risk. Procedure:

1. Compute F1 across thresholds in `np.arange(0.05, 0.95, 0.01)`
2. Select `threshold*` = argmax F1
3. Report all classification metrics at `threshold*`
4. Save `threshold*` to `models/optimal_threshold.json`

At inference, the API loads `optimal_threshold.json` and applies it.

### Variant — `evaluate_lgbm_pca.py`

Identical pipeline but the feature set replaces raw EXT_SOURCE and bureau aggregates with their first 10 principal components. Used for ablation study only.

### SVM Baseline — `evaluate_svm.py`

- `sklearn.svm.SVC(kernel='rbf', probability=True, C=1.0, gamma='scale')`
- Features standardized with `StandardScaler`
- Trained on a 50K subsample due to SVM quadratic complexity
- Reported metrics are on held-out 20% split (no CV — baseline only)

---

## 7. Missing Value Strategy

A core design principle of this project: **missingness is informative.**

### Rule 1 — Never drop sparse columns

A column with high null rate is retained if missingness carries signal. Bureau and credit card features are null for applicants with no prior history — a strong default signal in itself.

### Rule 2 — Binary missingness indicators

For application-level columns with >5% nulls, a binary flag `{COL}_MISSING` is added:

```
Columns flagged: OCCUPATION_TYPE, EXT_SOURCE_1, EXT_SOURCE_2, EXT_SOURCE_3,
                 AMT_GOODS_PRICE, AMT_ANNUITY, CNT_FAM_MEMBERS, DAYS_LAST_PHONE_CHANGE
```

### Rule 3 — Fill strategy by column type

| Column Category | Fill Strategy | Rationale |
|---|---|---|
| Ratio features (computed) | Median of training set | Robust to outliers |
| Aggregated bureau/prev/install counts | `0` | Absent history = zero activity |
| Aggregated bureau/prev/install sums | `0` | Same rationale |
| EXT_SOURCE missing | `-1` | Out-of-distribution flag; model learns -1 is special |
| Employment sentinel `365243` | `NaN` then median | Sentinel is not a real value |
| Categorical columns | `"XNA"` or `"Unknown"` | Preserves as a learnable category |
| Continuous app-level columns | Median (training) | Standard imputation |

### Rule 4 — Train/test consistency

All fill values (medians, global means, encoder maps) are computed **exclusively on training data** and stored. At inference, the stored values are applied to test/live data — no recalculation.

---

## 8. Feature Catalog

Total: **210 features** post-engineering and encoding.

### Feature Groups by Origin

| Group | Count | Module |
|---|---|---|
| Application ratios & flags | ~25 | `engineering_app.py` |
| Bureau aggregates | ~20 | `engineering_bureau.py` |
| Previous application patterns | ~15 | `engineering_prev.py` |
| Installment payment behavior | ~10 | `engineering_installments.py` |
| POS Cash balance | ~8 | `engineering_pos_cc.py` |
| Credit card balance | ~12 | `engineering_pos_cc.py` |
| One-hot encoded categoricals | ~80 | `categorical_encoding.py` |
| Target/frequency encoded | ~8 | `categorical_encoding.py` |
| NLP PCA embeddings | 32 | `nlp_embeddings.py` |

### Key Features by Predictive Importance (from LGBM feature importance)

1. `EXT_SOURCE_MEAN` — External credit score composite
2. `EXT_SOURCE_PROD` — Interaction of all three sources
3. `CREDIT_INCOME_RATIO` — Credit burden relative to income
4. `INSTAL_DPD_MEAN` — Historical payment lateness
5. `BUREAU_ACTIVE_DEBT_SUM` — Current outstanding obligations
6. `AGE_YEARS` — Applicant age (older → lower default risk on average)
7. `DAYS_LAST_PHONE_CHANGE` — Recency of phone change (instability signal)
8. `PREV_APPROVAL_RATE` — Prior lending relationships
9. `CC_UTILIZATION_MAX` — Peak credit card stress
10. `ANNUITY_INCOME_RATIO` — Monthly repayment burden

---

## 9. Inference API

**File:** `app/app.py` _(WIP)_

### Endpoint

```
POST /predict
Content-Type: application/json
```

### Request Schema

```json
{
  "SK_ID_CURR": 100001,
  "AMT_INCOME_TOTAL": 135000,
  "AMT_CREDIT": 406597,
  "AMT_ANNUITY": 24700,
  "AMT_GOODS_PRICE": 351000,
  "DAYS_BIRTH": -12005,
  "DAYS_EMPLOYED": -1213,
  "EXT_SOURCE_1": 0.502,
  "EXT_SOURCE_2": 0.731,
  "EXT_SOURCE_3": null,
  "NAME_CONTRACT_TYPE": "Cash loans",
  "OCCUPATION_TYPE": "Laborers",
  "ORGANIZATION_TYPE": "Business Entity Type 3"
}
```

Minimum required fields: `AMT_INCOME_TOTAL`, `AMT_CREDIT`, `AMT_ANNUITY`, `DAYS_BIRTH`. All others default to null and are imputed.

### Response Schema

```json
{
  "SK_ID_CURR": 100001,
  "default_probability": 0.143,
  "risk_label": "LOW",
  "threshold_used": 0.31,
  "top_risk_factors": [
    "CREDIT_INCOME_RATIO: 3.01 (high)",
    "EXT_SOURCE_MEAN: 0.41 (below average)",
    "INSTAL_DPD_MEAN: missing"
  ]
}
```

### Risk Label Mapping

| Probability Range | Label |
|---|---|
| 0.00 – 0.15 | `LOW` |
| 0.15 – 0.35 | `MEDIUM` |
| 0.35 – 0.60 | `HIGH` |
| > 0.60 | `VERY HIGH` |

### Inference Pipeline (inside `/predict`)

```
Raw JSON → validate schema
        → apply missing value fills (from stored medians)
        → compute engineered features (same logic as training)
        → apply target/frequency encoders (from models/target_encoders.pkl)
        → apply PCA transform (from models/pca_embeddings.pkl)
        → load lgbm_best.pkl → predict_proba
        → apply optimal_threshold from models/optimal_threshold.json
        → return response JSON
```

---

## Appendix — Key Design Decisions

**Why LightGBM over XGBoost?** Histogram-based splitting handles the high-cardinality and null-heavy feature space more efficiently. Native support for `scale_pos_weight` and leaf-wise tree growth yields better minority class recall.

**Why SentenceBERT over TF-IDF?** Occupation and organization type combinations form semantic clusters (e.g., "doctor at hospital" ≈ "physician at clinic") that bag-of-words methods miss. BERT embeddings capture these latent groupings with no manual taxonomy.

**Why 32 PCA components?** Beyond 32, incremental explained variance per component drops below 1%. 32 components retain ≥85% variance while keeping the feature matrix tractable for tree models (which are sensitive to dimensionality in split search).

**Why not a neural network?** Tabular data with mixed types, high cardinality, and sparse bureau history favors gradient boosting. The interpretability requirement (top risk factors in API response) also favors SHAP-compatible tree models.

---

*Document maintained by the project team. Last updated April 2026.*
