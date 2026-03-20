# Streamlit Deployment Guide

## Files Generated
| File | Description |
|------|-------------|
| `streamlit_app.py` | Stakeholder-facing dashboard |
| `df4_predictions.parquet` | Full dataset + predictions |
| `requirements.txt` | Python dependencies |

## Run Locally
```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Deploy to Streamlit Cloud
1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repo → set **Main file**: `streamlit_app.py`
4. Click **Deploy**

## What the App Shows
- **Overview**: KPI cards — total records, model Accuracy (0.9745), prediction distribution
- **Actual vs Predicted**: Confusion matrix + class distribution comparison
- **Explore Predictions**: Filterable table with color-coded predictions + CSV download
- **Feature Insights**: Feature importance and correlation matrix charts

## Prediction Data Schema
- All 55 original columns preserved
- `prediction` — model output (class label)
- `prediction_proba` — model confidence (0–1)
- `late_delivery_risk` — actual value (ground truth for comparison)
- Row alignment: guaranteed via _src_idx (FIX-5)
