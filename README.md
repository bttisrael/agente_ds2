# Auto Data Scientist v7 — SOTA Multi-Agent Pipeline

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Google Cloud](https://img.shields.io/badge/Google_Cloud-4285F4?style=flat&logo=google-cloud&logoColor=white)](https://cloud.google.com/)
[![BigQuery](https://img.shields.io/badge/BigQuery-SQL-4285F4?style=flat&logo=google-cloud&logoColor=white)](https://cloud.google.com/bigquery)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Gurobi](https://img.shields.io/badge/Gurobi-Optimization-007f00?style=flat&logo=gurobi&logoColor=white)](https://www.gurobi.com/)
[![Google Colab](https://img.shields.io/badge/Colab-Notebook-F9AB00?style=flat&logo=google-colab&logoColor=white)](https://colab.research.google.com/)

> # Executive Summary

This project develops a machine learning solution to predict delivery delays using DataCo Global's supply chain dataset of 180,000 orders. By forecasting **late_delivery_risk** (binary: late vs. on-time), the model enables operations managers to proactively identify at-risk shipments before delays occur, supporting critical decisions around warehouse routing, carrier selection, and customer communication.

The analysis revealed that shipping mode, delivery days, and order priority are key predictors of delivery risk. The final model achieved **[X]% accuracy** with **[Y]% recall** on late deliveries, successfully flagging high-risk orders for expedited handling. This predictive capability can reduce late deliveries by enabling preemptive intervention, improving customer satisfaction, and optimizing logistics resource allocation. The solution is deployable as a real-time scoring API to integrate with existing supply chain management systems.

---
*Note: Insert your actual model performance metrics where indicated.*

---
## Architecture

**Orchestration Layer:** CrewAI (stable, sequential, 1 tool per agent)
**Intelligence Layer:** Claude 3.5 Sonnet called inside each tool

| What AI Actually Does |
|-----------------------|
| Analyzes the dataset and automatically identifies the target |
| Writes and executes custom Python analysis code |
| Detects code errors and self-corrects (self-healing) |
| Decides which features to create based on real data |
| Interprets model results in natural language |
| Writes a narrative performance diagnostic |

---
## Target Selection by AI
**Target identified by AI:** `late_delivery_risk`  
**Justification:** late_delivery_risk is binary (0/1) with 54.83% being late deliveries. The business context explicitly states the goal is to predict Late_delivery_risk to help operations managers flag at-risk shipments. This matches perfectly with the column name and distribution.  
**Type:** `classification`

---
## Data Quality
- KNN imputation (numeric) + Mode (categorical)
- Intelligent analysis by Claude with business insights

[Quality_Report.md](Quality_Report.md)

---
## Medallion Architecture

| Layer | File | Description |
|-------|------|-------------|
| Silver | df1_silver.parquet | Standardized raw data + imputed |
| Gold | df2_gold.parquet | Standard features + AI-generated features |
| ML-Ready | df3_ml_ready.parquet | No redundancies or IDs |

![Sample](dataset_sample.png)

---
## EDA
![Distributions](distributions.png)
![Boxplots](boxplots.png)
![Categoricals](categoricals.png)
![Target](target_dist.png)
![Correlation](correlation_matrix.png)
![AI Analysis](intelligent_analysis.png)

---
## Modeling — CV + Optuna + Stacking + AI Interpretation



![Comparison](model_comparison.png)
![Features](feature_importance.png)

---
## Agent Architecture

| Agent | Tool | AI Intelligence |
|-------|------|----------------|
| Ingestor | download_and_save_silver | Not required |
| Analyst | analyze_data_with_ai | Claude analyzes dataset, identifies target, writes and executes code, self-healing |
| Feature Engineer | generate_features_with_ai_strategy | Claude decides and writes custom feature code |
| EDA Analyst | generate_eda_and_ml_ready | Pure Python (visualizations) |
| ML Scientist | train_and_save_model | Claude interprets results and writes narrative |

---
## How to Reproduce
```bash
git clone <repo>
echo "KAGGLE_USERNAME=x" >> .env
echo "KAGGLE_KEY=y" >> .env
echo "ANTHROPIC_API_KEY=sk-ant-..." >> .env
# Optional:
echo "We want to predict whether candidates will be hired." > business_context.txt
pip install crewai kagglehub pandas pyarrow python-dotenv optuna anthropic \
            scikit-learn matplotlib seaborn tabulate numpy xgboost lightgbm
python auto_data_scientist_v7.py
```

---
*Auto Data Scientist v7 — CrewAI + Claude 3.5 Sonnet + Optuna*
