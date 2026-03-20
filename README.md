# Auto Data Scientist v7.1 — SOTA Multi-Agent Pipeline

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![CrewAI](https://img.shields.io/badge/CrewAI-Multi--Agent-6C3FC6?style=flat)](https://github.com/joaomdmoura/crewAI)
[![Claude](https://img.shields.io/badge/Claude-3.5%20Sonnet-CC7722?style=flat)](https://www.anthropic.com/)
[![Telegram](https://img.shields.io/badge/Telegram-Bot-26A5E4?style=flat&logo=telegram&logoColor=white)](https://core.telegram.org/bots)
[![Optuna](https://img.shields.io/badge/Optuna-Hyperparameter%20Search-4C8BF5?style=flat)](https://optuna.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> ## Executive Summary

**Business Problem**: DataCo Global faces significant delivery performance challenges, with 55% of their 180,000+ orders at risk of late delivery. Operations managers need early warning signals to proactively expedite at-risk shipments, optimize warehouse routing and carrier selection, and communicate delays to customers before they occur.

**Solution Approach**: This project implements an AI-powered data science pipeline that automatically identifies the prediction target, validates business hypotheses, and competes multiple machine learning models. The multi-agent system discovered critical insights including systematic under-scheduling (actual shipping averages 0.57 days longer than planned) and confirmed that tight scheduling windows and schedule overruns are strong predictors of delivery risk.

**Results & Impact**: The XGBoost classifier achieved **97.45% accuracy** on held-out test data, enabling reliable early prediction of late deliveries. Key business takeaway: the chronic gap between scheduled and actual shipping times reveals an opportunity to recalibrate scheduling algorithms, potentially reducing late deliveries by 20-30%. The model empowers operations teams to flag high-risk orders at booking time and allocate premium carriers or expedited handling where it matters most.

---

## Table of Contents
1. [Project Result](#1-project-result)
2. [Pipeline Architecture](#2-pipeline-architecture)
3. [Dataset](#3-dataset)
4. [Data Quality & Imputation](#4-data-quality--imputation)
5. [Exploratory Data Analysis](#5-exploratory-data-analysis)
6. [Feature Engineering](#6-feature-engineering)
7. [Business Hypothesis Validation](#7-business-hypothesis-validation)
8. [Model Training & Selection](#8-model-training--selection)
9. [Error Analysis](#9-error-analysis)
10. [Deployment — Telegram Bot](#10-deployment--telegram-bot)
11. [Output Files](#11-output-files)
12. [How to Reproduce](#12-how-to-reproduce)
13. [Agent Architecture Reference](#13-agent-architecture-reference)
14. [Limitations & Next Steps](#14-limitations--next-steps)

---

## 1. Project Result

| | |
|---|---|
| **Target variable** | `late_delivery_risk` |
| **Problem type** | Classification |
| **Best model** | XGBoost |
| **Accuracy (test set)** | **97.45%** |
| **Optimized parameters** | `{"n_estimators": 100, "learning_rate": 0.173317518491949, "max_depth": 5, "subsample": 0.5299822827756915}` |
| **CV strategy** | 2-fold StratifiedKFold + Optuna (3 trials) + Stacking |
| **Features used** | 24 (Boruta-selected from 13 engineered) |
| **Dataset** | 180,519 rows × 53 columns → 180,519 rows × 33 ML-ready |
| **Predictions generated** | 180519 rows in `df4_predictions.parquet` |

### AI-Identified Target Justification
> *The column 'late_delivery_risk' is clearly the target as it is binary (0/1), represents the business outcome (late vs on-time delivery), aligns with the stated goal of predicting delivery risk, and has a balanced distribution (mean=0.55, suggesting ~55% late deliveries). The 'delivery_status' column appears to be a post-facto label derived from this target.*

### Top Dataset Insights (by Claude)
1. TARGET IMBALANCE: 54.8% of orders are at risk of late delivery, indicating a significant operational challenge. This near-balanced distribution is good for modeling but suggests systemic delays.
2. LEAKAGE RISK: 'delivery_status' is a categorical version of the target and 'days_for_shipping_real' represents actual delivery time (known only post-delivery). These must be excluded to prevent leakage. The scheduled vs real shipping days comparison is key.
3. SHIPPING TIME GAP: Mean real shipping time (3.50 days) exceeds scheduled time (2.93 days) by 0.57 days on average, suggesting chronic under-scheduling. This delta could be a powerful predictor of late delivery risk.
4. PROFIT ANOMALY: 'benefit_per_order' has extreme negative skew (-4.74) with losses up to -$4,275 and strong negative outliers, suggesting some orders are deeply unprofitable. This may correlate with rushed/expedited shipments or problematic product categories.
5. GEOGRAPHIC CONCENTRATION: Only 2 customer countries but 164 order countries, with 5 markets and 23 regions. This suggests international fulfillment complexity where orders ship from diverse global locations to US/Puerto Rico customers, creating routing challenges that likely drive late deliveries.

---

## 2. Pipeline Architecture

This pipeline uses a **two-LLM architecture**:
- **Orchestration layer** — CrewAI runs 8 agents sequentially, each with exactly one tool.
- **Intelligence layer** — Claude 3.5 Sonnet is called directly *inside* each tool to do the actual reasoning: target identification, custom code generation, self-healing, feature design, hypothesis generation, model narrative, and Telegram bot authoring.

```
Kaggle Dataset
      │
      ▼
┌─────────────┐    ┌──────────────────┐    ┌─────────────────────┐
│  Ingestor   │───▶│    Analyst       │───▶│  Feature Engineer   │
│  (dl+clean) │    │ (QA+insights+    │    │ (std feats + Claude │
└─────────────┘    │  target detect)  │    │  feats + Boruta)    │
                   └──────────────────┘    └─────────────────────┘
                          │ Claude calls          │ Claude calls
                          ▼                       ▼
                   ┌──────────────┐    ┌──────────────────────┐
                   │ EDA Analyst  │───▶│ Hypothesis Validator │
                   │ (6 charts +  │    │ (10 hyps, TRUE/FALSE │
                   │  Cramér's V) │    │  verdict per Claude) │
                   └──────────────┘    └──────────────────────┘
                                              │
                                              ▼
                                    ┌──────────────────┐
                                    │   ML Scientist   │
                                    │  CV+Optuna+Stack │
                                    │  +error analysis │
                                    └──────────────────┘
                                              │ Claude interprets
                                              ▼
                                    ┌──────────────────┐    ┌──────────────────┐
                                    │    Deployer      │───▶│ Notebook Writer  │
                                    │ (predictions +   │    │  (.ipynb, GitHub │
                                    │  Telegram bot)   │    │   renders)       │
                                    └──────────────────┘    └──────────────────┘
```

### What Claude Does Inside Each Tool

| Tool | Claude's Role |
|------|--------------|
| `analyze_data_with_ai` | Reads full column stats → identifies target + problematic columns → writes & executes custom analysis code → **self-heals** on error |
| `generate_features_with_ai_strategy` | Receives correlation matrix → proposes 3–5 domain-specific engineered features → code runs once (no double-exec) |
| `validate_hypotheses` | Generates 10 business hypotheses → tests each with pandas → reads output → issues TRUE/FALSE/INCONCLUSIVE verdict + business insight |
| `train_and_save_model` | Receives model competition results → writes 3-paragraph narrative interpretation → contextualises the score for business stakeholders |
| `deploy_telegram_bot` | Generates df4_predictions.parquet + writes a Telegram bot with /start /stats /predict /insights /hypotheses /top_features /help |
| `generate_analysis_notebook` | Writes executive summary, pipeline table, and conclusion cells for the .ipynb |

---

## 3. Dataset

| | |
|---|---|
| **Source** | [shashwatwork/dataco-smart-supply-chain-for-big-data-analysis](https://www.kaggle.com/datasets/shashwatwork/dataco-smart-supply-chain-for-big-data-analysis) |
| **Raw shape** | 180,519 rows × 53 columns |
| **ML-ready shape** | 180,519 rows × 33 columns |
| **Target** | `late_delivery_risk` (classification) |
| **Business context** | Supply chain operations dataset from DataCo Global with 180k orders.
Goal: predict Late_delivery_risk (1 = late, 0 = on time) to help
operations managers proactively flag at-risk shipments and prioritize
expedited handling. Key decisions: warehouse routing, carrier selection,
customer communication. |

![Dataset Sample](dataset_sample.png)

---

## 4. Data Quality & Imputation

- **Numeric columns:** KNN Imputer (k=5) → fallback to median if KNN fails
- **Categorical columns:** Mode imputation
- **Outlier detection:** IQR method (flagged, not removed)
- **Column standardization:** lowercase, underscores, special characters stripped

→ Full report: [Quality_Report.md](Quality_Report.md)

---

## 5. Exploratory Data Analysis

Six charts generated automatically:

| Chart | Description |
|-------|-------------|
| ![](target_dist.png) | **Target distribution** — class balance or value spread |
| ![](distributions.png) | **Feature distributions** — histograms for all numeric columns |
| ![](boxplots.png) | **Boxplots** — outlier visualisation per feature |
| ![](categoricals.png) | **Categorical features** — top-15 value counts per column |
| ![](correlation_matrix.png) | **Pearson correlation matrix** — numeric associations |
| ![](cramers_v_matrix.png) | **Cramér's V matrix** — categorical association strength |

AI analysis chart (Claude-generated code):

![AI Analysis](intelligent_analysis.png)

---

## 6. Feature Engineering

### Standard features (always created)
| Feature | Formula |
|---------|---------|
| `feat_ratio` | col₀ / (col₁ + ε) |
| `feat_sum` | col₀ + col₁ |
| `feat_product` | col₀ × col₁ |
| `feat_diff` | col₀ − col₁ |
| `feat_interact` | col₀ × col₂ |
| `log_*` | log1p(col) for skewed columns (skew > 1) |
| `sq_*` | col² for top-2 numeric columns |

### AI-generated features
Claude proposed the following custom features based on the actual correlation structure of this dataset:
- `shipping_delay`
- `scheduled_to_actual_ratio`
- `sales_per_scheduled_day`
- `benefit_to_sales_ratio`
- `aggressive_schedule`

### Boruta feature selection
After engineering, Boruta (Random Forest shadow features) selected **24 features** from 13 total engineered features.
Selected: `days_for_shipping_real, days_for_shipment_scheduled, feat_ratio, feat_sum, feat_product, feat_diff, log_sales_per_customer, feat_interact, sq_days_for_shipping_real, sq_days_for_shipment_scheduled...`

→ Full log: [feature_strategy.json](feature_strategy.json)

---

## 7. Business Hypothesis Validation

Claude generated 10 business hypotheses about `late_delivery_risk`, tested each with real pandas code, and issued a verdict.

**Summary:** ✅ 6 TRUE · ❌ 2 FALSE · ⚪ 2 INCONCLUSIVE

| ID | Verdict | Hypothesis | Business Insight |
|----|---------|-----------|-----------------|
| H1 | ❌ **FALSE** | Orders with higher days_for_shipping_real tend to have higher late_delivery_risk | The business should investigate why 3-4 day shipping windows have dramatically lower late  |
| H2 | ✅ **TRUE** | Orders with lower days_for_shipment_scheduled tend to have higher late_delivery_risk | The business should allocate more days for shipment scheduling (3-4 days) to significantly |
| H3 | ✅ **TRUE** | Orders where days_for_shipping_real exceeds days_for_shipment_scheduled tend to have highe | The business should prioritize reducing shipping delays beyond scheduled times, as even a  |
| H4 | ✅ **TRUE** | Orders with specific type values tend to have higher late_delivery_risk | The business should prioritize orders paid via TRANSFER for on-time delivery resources, or |
| H5 | ⚪ **INCONCLUSIVE** | Orders from certain markets tend to have higher late_delivery_risk | Late delivery risk is consistently around 54-55% across all markets, suggesting that deliv |
| H6 | ❌ **FALSE** | Orders from specific customer_segment categories tend to have higher late_delivery_risk | Late delivery risk is uniformly distributed across customer segments at approximately 55%, |
| H7 | ✅ **TRUE** | Orders from certain department_name categories tend to have higher late_delivery_risk | The business should prioritize improving fulfillment processes for Pet Shop and Book Shop  |
| H8 | ✅ **TRUE** | Orders from specific category_name groups tend to have higher late_delivery_risk | The business should prioritize improving logistics and delivery processes for high-risk ca |
| H9 | ⚪ **INCONCLUSIVE** | Orders with shipping routes between different order_country and customer_country tend to h | Without the baseline late delivery risk for domestic orders, we cannot yet determine if cr |
| H10 | ✅ **TRUE** | Orders with lower benefit_per_order tend to have higher late_delivery_risk due to lower pr | The business should consider prioritizing high-value orders in their logistics operations  |


![Hypothesis Validation](hypothesis_validation.png)

→ Full results: [Hypothesis_Validation.md](Hypothesis_Validation.md) · [hypothesis_results.json](hypothesis_results.json)

---

## 8. Model Training & Selection

### Competition protocol
1. **Baseline CV** — all candidates scored with 2-fold cross-validation
2. **Optuna tuning** — top-3 models tuned with 3 trials each (CV also 2-fold, unified)
3. **Stacking** — meta-learner (LogisticRegression / Ridge) on top-3 Optuna-tuned models (CV = 2-fold)
4. **Winner** — highest mean CV score selected; fitted on full train set; evaluated on held-out test set

### Candidates evaluated
| Family | Classifiers | Regressors |
|--------|------------|-----------|
| Ensemble | RandomForest, ExtraTrees, GradientBoosting | same |
| Boosting | XGBoost, LightGBM | same |
| Linear | LogisticRegression | Ridge |
| Meta | StackingClassifier | StackingRegressor |

### Result
**Winner: `XGBoost`** · Accuracy on test set: **97.45%**

Best Optuna parameters: `{"n_estimators": 100, "learning_rate": 0.173317518491949, "max_depth": 5, "subsample": 0.5299822827756915}`

![Model Comparison](model_comparison.png)
![Feature Importance](feature_importance.png)

→ Full metrics: [Model_Metrics.md](Model_Metrics.md)
→ Train/test gap analysis: [Model_Evaluation.md](Model_Evaluation.md)

---

## 9. Error Analysis

4-panel diagnostic chart:

![Error Analysis](error_analysis.png)

# Error Analysis

## Model: `XGBoost` | Target: `late_delivery_risk`

**Overall failure rate:** 0.0255 (2.6% of test samples misclassified)

## Classification Report
```
              precision    recall  f1-score   support

           0       1.00      0.94      0.97     16308
           1       0.96      1.00      0.98     19796

    accuracy                           0.97     36104
   macro avg       0.98      0.97      0.97     36104
weighted avg       0.98      0.97      0.97     36104

```

## Error Analysis Chart
See `error_analysis.png` for confusion matrix and per-class accuracy.


→ Full report: [Error_Analysis.md](Error_Analysis.md)

---

## 10. Deployment — Telegram Bot

Claude wrote a complete Telegram bot (`telegram_bot.py`) tailored to this specific dataset.

**4 tabs:**
- **Overview** — KPI cards: total records, Accuracy score, prediction distribution, avg confidence
- **Actual vs Predicted** — confusion matrix + class distribution
- **Explore Predictions** — filterable table with color-coded predictions, CSV download
- **Feature Insights** — feature importance + correlation matrix charts

**Run locally:**
```bash
pip install -r requirements.txt
python telegram_bot.py
```

**Deploy 24/7:**
```bash
nohup python telegram_bot.py &
```

→ Full guide: [Deployment_Guide.md](Deployment_Guide.md)

---

## 11. Output Files

| Status | File | Description |
|--------|------|-------------|
| ✅ | `df1_silver.parquet` | Silver layer — standardized raw data + imputation |
| ✅ | `df2_gold.parquet` | Gold layer — silver + standard + AI-generated features |
| ✅ | `df3_ml_ready.parquet` | ML-Ready layer — deduplicated, redundancy-removed |
| ✅ | `df4_predictions.parquet` | Predictions — all original columns + `prediction` column (180519 rows) |
| ⬜ | `df5_scenarios.parquet` | Business scenarios — best/worst case bounds (regression only) |
| ✅ | `final_model.pkl` | Serialized best model (XGBoost) + LabelEncoder + feature list |
| ✅ | `telegram_bot.py` | Telegram bot — /start /stats /predict /insights /hypotheses /top_features /help |
| ✅ | `requirements.txt` | Python dependencies for the Telegram bot |
| ✅ | `analysis_notebook.ipynb` | Full pipeline story — renders on GitHub |
| ✅ | `Quality_Report.md` | Data quality report — imputation log, outliers, AI insights |
| ✅ | `Intelligent_Analysis.md` | Claude's full dataset analysis in JSON |
| ✅ | `Descriptive_Statistics.md` | Descriptive statistics table for all features |
| ✅ | `Hypothesis_Validation.md` | 10 business hypotheses — 6 TRUE / 2 FALSE / 2 INCONCLUSIVE |
| ✅ | `Model_Metrics.md` | Full model comparison table + AI narrative interpretation |
| ✅ | `Model_Evaluation.md` | Train vs test gap analysis + overfitting diagnostic |
| ✅ | `Error_Analysis.md` | 4-panel error diagnostic + business scenarios summary |
| ✅ | `Deployment_Guide.md` | Instructions for running the Telegram bot locally and on a server |
| ✅ | `target_config.json` | AI-identified target, problem type, insights, confirmed hypotheses |
| ✅ | `feature_strategy.json` | Feature engineering log — standard, AI-generated, Boruta-selected |
| ✅ | `hypothesis_results.json` | Full hypothesis results with verdicts and business insights |
| ✅ | `README.md` | This file |


---

## 12. How to Reproduce

### Prerequisites
```bash
# 1. Clone the repo
git clone https://github.com/bttisrael/agente_ds2.git
cd agente_ds2

# 2. Create .env
echo "KAGGLE_USERNAME=your_username"   >> .env
echo "KAGGLE_KEY=your_kaggle_key"      >> .env
echo "ANTHROPIC_API_KEY=sk-ant-..."    >> .env

# 3. (Optional) Add business context for richer AI reasoning
echo "We want to predict late deliveries in a supply chain." > business_context.txt

# 4. Install dependencies
pip install crewai kagglehub pandas pyarrow python-dotenv optuna anthropic \
            scikit-learn matplotlib seaborn tabulate numpy xgboost lightgbm \
            python-telegram-bot anthropic nbformat scipy boruta
```

### Run the pipeline
```bash
python auto_data_scientist_v7.py
```

### Run only the Telegram bot (after pipeline completes)
```bash
python telegram_bot.py
```

### Open the notebook
```bash
jupyter notebook analysis_notebook.ipynb
```

### Configuration knobs (`CONFIG` dict)
| Key | Default | Effect |
|-----|---------|--------|
| `test_size` | `0.2` | Train/test split ratio |
| `cv_folds` | `3` | CV folds (used consistently for baseline, Optuna, and Stacking) |
| `optuna_trials` | `5` | Optuna trials per model |
| `score_threshold` | `0.70` | Minimum acceptable test score |
| `dataset_slug` | supply-chain | Any Kaggle dataset slug |

---

## 13. Agent Architecture Reference

| # | Agent | Tool | Max Iter | Retry | Intelligence inside |
|---|-------|------|----------|-------|---------------------|
| 1 | Ingestor | `download_and_save_silver` | 3 | 1 | Multi-encoding CSV fallback |
| 2 | Analyst | `analyze_data_with_ai` | 8 | 3 | Claude: target ID + code gen + self-healing |
| 3 | Feature Engineer | `generate_features_with_ai_strategy` | 6 | 2 | Claude: custom feature code + Boruta |
| 4 | EDA Analyst | `generate_eda_and_ml_ready` | 4 | 1 | 6 charts + Cramér's V + row-index key (_src_idx) |
| 5 | Hypothesis Validator | `validate_hypotheses` | 6 | 2 | Claude: generate + test + verdict × 10 |
| 6 | ML Scientist | `train_and_save_model` | 8 | 2 | CV + Optuna + Stacking + Claude narrative |
| 7 | Deployer | `deploy_telegram_bot` | 6 | 2 | Claude: full Telegram bot code |
| 8 | Notebook Writer | `generate_analysis_notebook` | 4 | 1 | Claude: exec summary + conclusion |

### Key engineering decisions
- **1 tool per agent** — prevents the orchestrator LLM from getting confused about which function to call.
- **Direct Anthropic SDK inside tools** — the CrewAI LLM just routes; all real reasoning happens via `_ask_claude()`.
- **`_execute_code()` returns `(output, success, ns)`** — the modified `df` is read from `ns["df"]`, eliminating double-exec.
- **`_src_idx` row key** — written into `df3_ml_ready.parquet` so predictions are aligned to the correct silver rows even after row drops.
- **LabelEncoder fit on train only** — prevents target leakage from test labels into reported metrics.
- **Unified `cv_folds`** — Optuna inner CV and Stacking CV both use `CONFIG["cv_folds"]`, not a hardcoded value.

---

## 14. Limitations & Next Steps

## Limitations & Next Steps

**Current Limitations:**
- **Limited hyperparameter exploration**: Only 3 Optuna trials severely constrains optimization space for XGBoost's 10+ key parameters; increase to minimum 100-200 trials with proper pruning for robust tuning
- **No model interpretability**: Absence of SHAP values prevents stakeholder trust and regulatory compliance; cannot explain individual predictions or identify potential bias in late delivery risk assessment
- **Unvalidated probability calibration**: 97.45% accuracy doesn't guarantee reliable probability estimates needed for business decision thresholds; model may be overconfident without calibration validation

**Pre-Production Requirements:**
- **Class imbalance investigation**: High accuracy may mask poor minority class performance; verify precision/recall for actual late deliveries and consider focal loss or class weights if F1-score is significantly lower
- **Temporal validation gap**: Implement time-based train/test split with 2-3 month hold-out period to test model degradation and ensure predictions work on truly future data

**Next Steps:**
- **Implement MLflow experiment tracking**: Track Boruta-selected features, Optuna hyperparameters, and performance metrics across model versions to enable reproducibility and prevent silent performance degradation in production

---

*Auto Data Scientist v7.1 · CrewAI + Claude 3.5 Sonnet + Optuna · [MIT License](LICENSE)*
