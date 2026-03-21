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

This XGBoost classification model is **well-fitted** and performing excellently. The training accuracy of 97.58% and test accuracy of 97.45% are both very high, with only a 0.13% gap between them. This minimal difference indicates the model generalizes well to unseen data without memorizing the training set. There are no signs of overfitting (which would show a large gap with much higher training accuracy) or underfitting (which would show poor performance on both sets).

The model is production-ready for predicting late delivery risk. The negligible performance gap suggests robust learning of actual patterns rather than noise. You should proceed with standard validation steps like checking performance across different customer segments or time periods, but from a fit perspective, this model shows healthy balance between bias and variance. No immediate architectural changes or regularization adjustments are needed.

## Optimized Parameters (Optuna)
```json
{
  "n_estimators": 104,
  "learning_rate": 0.018507754954953916,
  "max_depth": 7,
  "subsample": 0.7464645250826458
}
```
