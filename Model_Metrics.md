# Model Metrics

**Type:** classification | **Target:** `late_delivery_risk`

## Model Comparison

|                         |   mean |    std |
|:------------------------|-------:|-------:|
| XGBoost                 | 0.9752 | 0.0007 |
| GradientBoosting_Optuna | 0.9752 | 0.0007 |
| LightGBM_Optuna         | 0.9752 | 0.0007 |
| XGBoost_Optuna          | 0.9752 | 0.0007 |
| LightGBM                | 0.9752 | 0.0007 |
| GradientBoosting        | 0.9752 | 0.0007 |
| RandomForest            | 0.9617 | 0.0009 |
| ExtraTrees              | 0.9572 | 0.0005 |
| LogisticRegression      | 0.539  | 0.0094 |

**Selected model:** `XGBoost`

**ACCURACY (test):** 0.9745

```
              precision    recall  f1-score   support

           0       1.00      0.94      0.97     16308
           1       0.96      1.00      0.98     19796

    accuracy                           0.97     36104
   macro avg       0.98      0.97      0.97     36104
weighted avg       0.98      0.97      0.97     36104

```

## AI Interpretation

## Model Interpretation: Late Delivery Risk Prediction

**Model Selection and Performance**

XGBoost emerged as the optimal choice among the ensemble methods tested, achieving a mean cross-validation score of 0.9752 (±0.0007) and maintaining consistent performance on the holdout test set (0.9745). Notably, all gradient boosting variants—including LightGBM and GradientBoosting with and without hyperparameter optimization—converged to nearly identical performance, suggesting we've reached the predictive ceiling for this dataset. XGBoost's selection is justified by its robust handling of missing values, built-in regularization to prevent overfitting, and excellent scalability for the 180k order dataset. The dramatic performance gap between tree-based ensembles (>97%) and logistic regression (53.9%) indicates that late delivery risk has complex, non-linear relationships with predictor variables that require the sophisticated feature interactions that boosted trees naturally capture.

**Business Impact and Interpretation**

An accuracy of 97.45% means that for every 1,000 shipments flagged by the model, operations managers can expect approximately 975 correct predictions. In practical terms, if DataCo Global processes 10,000 daily orders, this model would misclassify only ~255 shipments. However, **accuracy alone is insufficient** for this use case—the critical question is whether these errors are false positives (unnecessary expedited handling, wasted costs) or false negatives (missed late deliveries, customer dissatisfaction). For supply chain operations, false negatives are typically more costly, as they represent service failures and potential customer churn. The business should establish the acceptable cost ratio between unnecessary intervention versus missed late deliveries to properly calibrate decision thresholds beyond the default 0.5 probability cutoff.

**Model Limitations and Considerations**

Several concerns warrant attention before production deployment. First, the suspiciously uniform performance across all boosting algorithms (identical to the fourth decimal) suggests potential **data leakage**—there may be features in the 53-column dataset that contain post-hoc information or are direct proxies of the target (e.g., "actual_delivery_date" or "days_delayed"). This should be investigated immediately by reviewing feature importance and conducting temporal validation. Second, the extremely high accuracy may indicate class imbalance issues that aren't reflected in the raw accuracy metric; if 97% of shipments are naturally on-time, a naive baseline would already achieve 97% accuracy. We need to examine precision, recall, and F1-scores for both classes, plus the confusion matrix, to understand true predictive power. Third, model drift is inevitable—carrier performance, seasonal patterns, and supply chain disruptions will shift the data distribution over time.

**Production Recommendations**

For successful deployment, implement a **tiered alerting system** based on predicted probabilities rather than binary classifications: high-risk (p>0.8) for immediate intervention, medium-risk (0.5-0.8) for monitoring, and low-risk (<0.5) for standard processing. Establish a feedback loop to capture actual delivery outcomes and retrain monthly, monitoring for performance degradation across key segments (carrier, geography, product category). Critically, **conduct a thorough feature audit** to eliminate any leakage variables and re-validate model performance—if accuracy drops significantly, this confirms leakage and requires rebuilding with only pre-shipment features. Deploy shadow mode initially, running predictions alongside existing processes for 2-4 weeks to validate real-world performance before using the model for operational decisions. Finally, create explainability dashboards showing feature contributions for flagged shipments so operations managers understand *why* orders are at-risk, enabling targeted interventions rather than generic expedited handling.
