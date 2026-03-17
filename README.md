# Auto Data Scientist v7 — SOTA Multi-Agent Pipeline

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Google Cloud](https://img.shields.io/badge/Google_Cloud-4285F4?style=flat&logo=google-cloud&logoColor=white)](https://cloud.google.com/)
[![BigQuery](https://img.shields.io/badge/BigQuery-SQL-4285F4?style=flat&logo=google-cloud&logoColor=white)](https://cloud.google.com/bigquery)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Gurobi](https://img.shields.io/badge/Gurobi-Optimization-007f00?style=flat&logo=gurobi&logoColor=white)](https://www.gurobi.com/)
[![Google Colab](https://img.shields.io/badge/Colab-Notebook-F9AB00?style=flat&logo=google-colab&logoColor=white)](https://colab.research.google.com/)

> # Executive Summary

This project develops a machine learning solution to predict late delivery risk using DataCo Global's supply chain dataset of 180,000 orders. By identifying shipments likely to arrive late, operations managers can proactively intervene through expedited handling, alternative carrier selection, and early customer communication, reducing delays and improving satisfaction.

Multiple classification algorithms were evaluated, with the **[best model]** achieving **[X% accuracy/F1-score]** on test data. The model identifies key risk factors including shipping mode, geographic distance, and order processing time. Feature importance analysis reveals actionable insights for warehouse routing optimization and carrier performance evaluation. This predictive system enables data-driven prioritization of at-risk shipments, providing operations teams with a scalable tool to minimize late deliveries and enhance supply chain efficiency. The complete implementation, including data preprocessing, model training, and evaluation notebooks, is available in this repository.

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
**Target identified by AI:** `hired`  
**Justification:** The 'hired' column is a binary variable (0/1) with 70.61% positive rate, representing a hiring outcome. Despite the business context mentioning supply chain and Late_delivery_risk, the actual dataset contains recruitment/hiring data (age, education_level, university_tier, cgpa, internships, projects, programming_languages, certifications, experience_years, hackathons, research_papers, skills_score, soft_skills_score, resume_length_words, company_type). This is clearly a candidate hiring prediction dataset, not supply chain data. The 'hired' column is the only binary outcome variable suitable as a target.  
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
