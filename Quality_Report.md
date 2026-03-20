# Quality Report — AI-Powered Analysis

**Context:** Supply chain operations dataset from DataCo Global with 180k orders.
Goal: predict Late_delivery_risk (1 = late, 0 = on time) to help
operations managers proactively flag at-risk shipments and prioritize
expedited handling. Key decisions: warehouse routing, carrier selection,
customer communication.
**Shape:** 180519 x 53

## Applied Imputation
- Median imputer applied (dataset has 180,519 rows — KNN skipped to avoid RAM spike, threshold=50,000).
- Mode applied to 'customer_lname'.

## Detected Outliers (IQR)
{
  "benefit_per_order": 18942,
  "sales_per_customer": 1943,
  "customer_id": 1198,
  "department_id": 362,
  "latitude": 9,
  "longitude": 1414,
  "order_customer_id": 1198,
  "order_item_discount": 7537,
  "order_item_product_price": 2048,
  "order_item_profit_ratio": 17300,
  "sales": 488,
  "order_item_total": 1943,
  "order_profit_per_order": 18942,
  "order_zipcode": 24818,
  "product_price": 2048
}

## Intelligent Analysis by Claude

### Identified Target
**Column:** `late_delivery_risk`
**Justification:** The column 'late_delivery_risk' is clearly the target as it is binary (0/1), represents the business outcome (late vs on-time delivery), aligns with the stated goal of predicting delivery risk, and has a balanced distribution (mean=0.55, suggesting ~55% late deliveries). The 'delivery_status' column appears to be a post-facto label derived from this target.

### Problematic Columns
['delivery_status', 'customer_email', 'customer_password', 'product_description', 'product_status', 'order_customer_id', 'customer_id', 'order_id', 'order_item_id', 'customer_street', 'customer_fname', 'customer_lname', 'order_date_dateorders', 'shipping_date_dateorders']

### Top Dataset Insights
1. TARGET IMBALANCE: 54.8% of orders are at risk of late delivery, indicating a significant operational challenge. This near-balanced distribution is good for modeling but suggests systemic delays.
2. LEAKAGE RISK: 'delivery_status' is a categorical version of the target and 'days_for_shipping_real' represents actual delivery time (known only post-delivery). These must be excluded to prevent leakage. The scheduled vs real shipping days comparison is key.
3. SHIPPING TIME GAP: Mean real shipping time (3.50 days) exceeds scheduled time (2.93 days) by 0.57 days on average, suggesting chronic under-scheduling. This delta could be a powerful predictor of late delivery risk.
4. PROFIT ANOMALY: 'benefit_per_order' has extreme negative skew (-4.74) with losses up to -$4,275 and strong negative outliers, suggesting some orders are deeply unprofitable. This may correlate with rushed/expedited shipments or problematic product categories.
5. GEOGRAPHIC CONCENTRATION: Only 2 customer countries but 164 order countries, with 5 markets and 23 regions. This suggests international fulfillment complexity where orders ship from diverse global locations to US/Puerto Rico customers, creating routing challenges that likely drive late deliveries.

### Recommended Feature Engineering Strategy
1. **Shipping Time Features**: Create 'shipping_delay' (real - scheduled), 'delay_ratio' (real/scheduled), and interaction with shipping_mode. Engineer binary flags for 'scheduled_too_optimistic' (scheduled < 3 days) and 'weekend_delivery' from parsed dates. 2. **Geographic Complexity**: Calculate 'order_customer_distance' using haversine formula on lat/long pairs, create 'cross_border' flag (order_country != customer_country), and encode 'market' + 'region' combinations to capture fulfillment complexity. 3. **Product/Order Economics**: Create 'is_profitable' (benefit_per_order > 0), 'discount_to_price_ratio', 'quantity_category' bins, and 'high_value_order' flag (sales > 300). Aggregate customer-level features like 'customer_avg_order_value' and 'customer_order_count'. 4. **Categorical Encoding**: Target-encode high-cardinality geography (order_country, order_city, customer_city) using late_delivery_risk mean with regularization. One-hot encode low-cardinality (shipping_mode, customer_segment, type, market). 5. **Temporal Features**: Parse order_date to extract 'day_of_week', 'month', 'hour', 'is_weekend', 'is_holiday_season' (Nov-Dec), and 'days_since_first_order' per customer. 6. **Remove Leakage**: Drop delivery_status, days_for_shipping_real (use only scheduled), shipping_date, and all IDs/PII columns before modeling.

### Analysis Execution Output
```
=== Statistics by Shipping Mode ===
               late_delivery_risk         shipping_delay days_for_shipping_real benefit_per_order
                             mean   count           mean                   mean              mean
shipping_mode                                                                                    
First Class                 0.953   27814          1.000                  2.000            23.122
Same Day                    0.457    9737          0.478                  0.478            20.850
Second Class                0.766   35216          1.991                  3.991            21.306
Standard Class              0.381  107752         -0.004                  3.996            21.999

=== Target Distribution ===
On-time (0): 45.17%
Late (1): 54.83%

=== Top Correlations with Late Delivery Risk ===
shipping_delay              0.777644
days_for_shipping_real      0.401415
customer_zipcode            0.003151
category_id                 0.001752
product_category_id         0.001752
department_id               0.001077
latitude                    0.000679
order_item_discount_rate    0.000404
order_item_quantity        -0.000139
order_item_discount        -0.000750
Name: late_delivery_risk, dtype: float64

Plot saved successfully.

```

---
*Analysis generated by Claude 3.5 Sonnet*
