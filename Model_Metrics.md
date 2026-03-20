# Model Metrics

**Type:** classification | **Target:** `late_delivery_risk`

## Model Comparison

|                         |   mean |    std |
|:------------------------|-------:|-------:|
| GradientBoosting        | 0.9758 | 0.0001 |
| XGBoost_Optuna          | 0.9758 | 0.0001 |
| XGBoost                 | 0.9758 | 0.0001 |
| GradientBoosting_Optuna | 0.9758 | 0.0001 |
| LightGBM                | 0.9758 | 0.0001 |
| LightGBM_Optuna         | 0.9758 | 0.0001 |
| RandomForest            | 0.9626 | 0.0005 |
| ExtraTrees              | 0.9579 | 0.0001 |
| LogisticRegression      | 0.5292 | 0.0017 |

**Selected model:** `GradientBoosting`

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

**Model Selection Rationale**

Gradient Boosting emerged as the optimal choice alongside other ensemble tree-based methods (XGBoost, LightGBM), all achieving virtually identical performance (~97.6% accuracy). This convergence suggests the prediction task has strong, learnable patterns that sequential boosting algorithms effectively capture. Gradient Boosting's advantage lies in its iterative error-correction mechanism, which is particularly well-suited for this moderately imbalanced dataset (54.8% late deliveries) where misclassifying the minority class has business consequences. Unlike the dramatically underperforming Logistic Regression (52.9% - barely better than random), Gradient Boosting can model complex non-linear interactions between features like geographic variables, shipping modes, scheduled days, and profit margins without requiring extensive manual feature engineering. The minimal standard deviation (0.0001) across cross-validation folds indicates exceptional stability, critical for operational deployment where consistent predictions build trust with warehouse managers.

**Business Impact of 97.45% Accuracy**

A test accuracy of 97.45% translates to correctly identifying late delivery risk for approximately 35,000 of the 36,000 test orders (assuming 20% test split from 180k total). In practical terms, operations managers can trust that when the model flags a shipment as high-risk, it's correct 97+ times out of 100, enabling confident intervention decisions like carrier upgrades, proactive customer notifications, or warehouse prioritization. However, **accuracy alone is insufficient** for this use case—we must examine precision and recall separately. With 54.8% base late-delivery rate, false negatives (missing truly at-risk orders) mean missed opportunities for intervention, while false positives waste resources on unnecessary expedited handling. The business should prioritize recall if the cost of late delivery (customer churn, penalties) exceeds expedited shipping costs, or precision if operational capacity for interventions is limited. A confusion matrix analysis and cost-sensitive threshold tuning are essential next steps.

**Critical Limitations and Data Leakage Concerns**

The most significant risk is **data leakage from post-hoc features**. The dataset insights explicitly flag 'delivery_status' and 'days_for_shipping_real' as variables known only after delivery completion—if these were included in training, the 97.45% accuracy is artificially inflated and the model will fail catastrophically in production where only pre-shipment data exists. Rigorous feature auditing must confirm only 'days_for_shipment_scheduled' and attributes known at order placement time were used. Additionally, the geographic complexity (164 countries, 3,597 cities) poses **overfitting risks** if rare locations have insufficient samples—the model may memorize city-specific patterns that don't generalize. The strong correlation between negative profit orders and late deliveries (-$4,275 minimum benefit) suggests the model might be learning that operationally distressed, unprofitable orders receive deprioritized handling, which is a legitimate business signal but could perpetuate inequitable service if not monitored.

**Production Deployment Recommendations**

Before deployment, conduct three critical validations: (1) **Temporal holdout testing**—retrain on historical data and validate on the most recent 2-4 weeks to ensure the model handles evolving logistics patterns, (2) **Feature availability audit**—document the exact timestamp when each predictor becomes available in operational systems to prevent leakage, and (3) **Threshold optimization**—use business cost parameters (e.g., $50 expedited shipping cost vs. $200 late-delivery penalty) to set classification thresholds that maximize ROI rather than accuracy. For production, implement monitoring dashboards tracking prediction distribution shifts, feature drift (especially for volatile attributes like 'benefit_per_order'), and intervention success rates. Consider deploying a **two-stage model**: initial Gradient Boosting screening for all orders, then a specialized high-recall model for marginal cases where scheduled days are tight. Finally, establish a feedback loop where operations teams label intervention outcomes ("flagged and expedited—arrived on time" vs. "not flagged—arrived late") to enable continuous model retraining and align predictions with evolving carrier performance.
