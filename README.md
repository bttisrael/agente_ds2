# Auto Data Scientist v7 — SOTA Multi-Agent Pipeline

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Google Cloud](https://img.shields.io/badge/Google_Cloud-4285F4?style=flat&logo=google-cloud&logoColor=white)](https://cloud.google.com/)
[![BigQuery](https://img.shields.io/badge/BigQuery-SQL-4285F4?style=flat&logo=google-cloud&logoColor=white)](https://cloud.google.com/bigquery)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Gurobi](https://img.shields.io/badge/Gurobi-Optimization-007f00?style=flat&logo=gurobi&logoColor=white)](https://www.gurobi.com/)
[![Google Colab](https://img.shields.io/badge/Colab-Notebook-F9AB00?style=flat&logo=google-colab&logoColor=white)](https://colab.research.google.com/)

> # Executive Summary

This project develops a predictive model to identify at-risk shipments in DataCo Global's supply chain operations using a dataset of 180,000 orders. By predicting `Late_delivery_risk` (binary: 1=late, 0=on-time), the model enables operations managers to proactively flag problematic orders before delays occur, allowing timely intervention through optimized warehouse routing, strategic carrier selection, and preemptive customer communication.

The analysis employed multiple machine learning algorithms including Logistic Regression, Random Forest, and XGBoost to classify delivery risk. The best-performing model achieved [X%] accuracy with [Y%] recall on late deliveries, successfully identifying the majority of at-risk shipments while maintaining acceptable false-positive rates. Feature importance analysis revealed that [key factors such as shipping mode, order priority, and geographic distance] were the strongest predictors of delivery delays. This solution provides actionable insights that can reduce late deliveries, improve customer satisfaction, and optimize operational resource allocation.

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
