# Model Evaluation

## `GradientBoosting`
**Type:** classification | **Target:** `late_delivery_risk`

| Dataset   | Accuracy |
|-----------|-------|
| Train     | 0.9758 |
| Test      | 0.9745 |
| Gap       | 0.0013  |

## AI Diagnostic

This GradientBoosting classification model is **well-fitted**. The training accuracy of 97.58% and test accuracy of 97.45% are both high and nearly identical, with only a 0.13% gap between them. This minimal difference indicates the model generalizes excellently to unseen data without memorizing the training set. There are no signs of overfitting (which would show a large gap with much higher training accuracy) or underfitting (which would show poor performance on both sets).

The model is performing strongly and is ready for deployment to predict late delivery risk. The near-equal performance across train and test sets suggests robust learning of the underlying patterns. However, you should still validate the model on recent production data, monitor for concept drift over time, and ensure the 97.45% accuracy meets your business requirements for this particular use case (considering the costs of false positives vs false negatives in delivery predictions).

## Optimized Parameters (Optuna)
```json
{
  "n_estimators": 68,
  "learning_rate": 0.02735885933605079,
  "max_depth": 5,
  "subsample": 0.6473392896634209
}
```
