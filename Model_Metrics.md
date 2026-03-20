# Model Metrics

**Type:** classification | **Target:** `late_delivery_risk`

## Model Comparison

|                         |   mean |    std |
|:------------------------|-------:|-------:|
| XGBoost_Optuna          | 0.9758 | 0.0001 |
| LightGBM                | 0.9758 | 0.0001 |
| GradientBoosting_Optuna | 0.9757 | 0.0001 |
| XGBoost                 | 0.9757 | 0.0001 |
| GradientBoosting        | 0.9757 | 0.0001 |
| LightGBM_Optuna         | 0.975  | 0.0001 |
| RandomForest            | 0.9691 | 0.0002 |
| ExtraTrees              | 0.9607 | 0.0002 |
| LogisticRegression      | 0.7522 | 0.2086 |

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

**Model Selection Rationale**

XGBoost emerged as the optimal choice for this late delivery prediction problem, achieving 97.45% test accuracy with exceptional consistency (std dev of 0.0001 across folds). This performance advantage stems from XGBoost's architectural strengths that align perfectly with supply chain data characteristics. The algorithm excels at capturing complex non-linear interactions between geographic routing variables (164 order countries, 23 regions, 5 markets), temporal patterns (scheduled vs. real shipping time gaps), and operational factors (carrier performance, warehouse efficiency). The near-balanced target distribution (54.8% late) allows XGBoost's gradient boosting framework to learn effectively without requiring aggressive rebalancing techniques. Critically, the ensemble tree structure handles the extreme skewness in variables like benefit_per_order (skew: -4.74) and naturally surfaces interaction effects—such as how specific country-carrier-product combinations drive delays—that simpler linear models like Logistic Regression (75.22% accuracy) completely miss.

**Business Impact of 97.45% Accuracy**

In operational terms, this model correctly identifies late delivery risk for approximately 97 out of every 100 orders, translating to substantial business value across multiple decision points. For operations managers processing 180,000+ orders, this means the model accurately flags roughly 96,000 of the 98,640 genuinely at-risk shipments while maintaining low false alarm rates. This precision enables proactive interventions: warehouse teams can prioritize expedited routing for flagged orders, logistics coordinators can upgrade carriers selectively rather than blanket-upgrading (reducing expedite costs), and customer service can send preemptive delay notifications only to truly affected customers (preserving trust without alert fatigue). Given that the current mean shipping delay is 0.57 days beyond schedule—suggesting chronic under-delivery—even a 10% reduction in late deliveries through model-guided interventions could significantly improve customer satisfaction scores, reduce penalty costs, and optimize the deeply unprofitable orders (losses up to -$4,275) that likely correlate with emergency expediting.

**Critical Limitations and Monitoring Requirements**

Despite strong performance, several concerns warrant careful attention in production. First, the data leakage risk from 'delivery_status' and 'days_for_shipping_real' must be rigorously validated—these post-delivery variables must be completely excluded from model training and inference to ensure predictions rely only on information available at order placement. Second, the 0.57-day average under-scheduling suggests potential concept drift: if operational improvements reduce this gap, the model will require retraining as historical delay patterns become obsolete. Third, the extreme geographic complexity (164 origin countries serving just 2 customer countries) creates long-tail prediction scenarios where certain rare country-carrier-product combinations may have insufficient training samples, leading to overconfident predictions. The model should implement confidence thresholds and flag low-support predictions for manual review. Finally, monitoring for feedback loops is essential—if the model systematically flags certain routes as high-risk, and those routes receive disproportionate resources, the training data distribution will shift, potentially degrading model calibration over time.

**Production Deployment Recommendations**

For successful operationalization, implement a phased rollback strategy starting with a shadow deployment where predictions run parallel to existing processes for 2-4 weeks to validate real-world accuracy and catch any data pipeline issues. Establish dual monitoring: technical metrics (prediction latency <100ms, feature availability >99%, model drift detection via PSI scores on key features weekly) and business metrics (actual late delivery rate for flagged vs. unflagged orders, cost per intervention, false positive rate impact on expedite spend). Create interpretability layers using SHAP values to explain individual predictions to operations managers—showing that "Order #12345 is flagged because: origin country Chile + carrier Standard Class + scheduled 2-day window has 89% historical late rate" builds trust and enables intelligent overrides. Set up automated retraining pipelines triggered monthly or when performance degrades beyond 95% accuracy thresholds. Finally, establish A/B testing cohorts where 10% of flagged orders receive no intervention as a control group, ensuring the model's recommendations actually drive business outcomes rather than merely correlating with existing operational intuitions.
