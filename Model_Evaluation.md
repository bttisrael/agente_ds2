# Model Evaluation

## `XGBoost`
**Type:** classification | **Target:** `late_delivery_risk`

| Dataset   | Accuracy |
|-----------|-------|
| Train     | 0.9758 |
| Test      | 0.9745 |
| Gap       | 0.0013  |

## AI Diagnostic

## Model Diagnosis: Well-Fitted

This XGBoost classification model demonstrates **excellent fit** with no significant overfitting or underfitting issues. The training accuracy of 97.58% and test accuracy of 97.45% are nearly identical, with only a 0.13% gap between them. This minimal difference indicates the model generalizes very well to unseen data and hasn't memorized the training set. The high performance on both datasets suggests the model has successfully learned the underlying patterns for predicting late delivery risk.

The model is production-ready from a bias-variance perspective. However, you should verify that this high accuracy isn't misleading due to class imbalance in your target variable. If late deliveries are rare (or very common), ensure you're also evaluating precision, recall, and F1-score for both classes. Additionally, confirm the model's performance on recent data to ensure these patterns remain stable over time.

## Optimized Parameters (Optuna)
```json
{
  "n_estimators": 100,
  "learning_rate": 0.173317518491949,
  "max_depth": 5,
  "subsample": 0.5299822827756915
}
```
