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
