# Quality Report — AI-Powered Analysis

**Context:** Supply chain operations dataset from DataCo Global with 180k orders.
Goal: predict Late_delivery_risk (1 = late, 0 = on time) to help
operations managers proactively flag at-risk shipments and prioritize
expedited handling. Key decisions: warehouse routing, carrier selection,
customer communication.
**Shape:** 180519 x 53

## Applied Imputation
- KNN imputer applied to numeric columns.
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
  "product_price": 2048
}

## Intelligent Analysis by Claude

### Identified Target
**Column:** `late_delivery_risk`
**Justification:** late_delivery_risk is binary (0/1) with 54.83% being late deliveries. The business context explicitly states the goal is to predict Late_delivery_risk to help operations managers flag at-risk shipments. This matches perfectly with the column name and distribution.

### Problematic Columns
['delivery_status', 'days_for_shipping_real', 'customer_email', 'customer_password', 'product_description', 'product_status', 'order_customer_id', 'customer_id', 'order_id', 'order_item_id', 'customer_street']

### Top Dataset Insights
1. Target imbalance: 54.83% of deliveries are late, suggesting systemic issues rather than random delays - this is actionable for operations improvement
2. Critical leakage risk: 'delivery_status' has 4 categories including 'Late delivery' which directly reveals the target - must be excluded. 'days_for_shipping_real' is actual shipping time (known only after delivery) vs 'days_for_shipment_scheduled' (known upfront)
3. Geographic concentration: Only 2 customer countries (Puerto Rico and USA) but 164 order countries across 5 markets - suggests B2B wholesale model where US/PR customers order from global suppliers
4. Profit challenges: Mean benefit_per_order is only $21.98 with high negative skew (-4.74) and minimum of -$4,275, indicating 25% of transactions lose money - late deliveries likely correlate with these losses
5. Shipping mode and schedule mismatch: days_for_shipment_scheduled has only 4 unique values (0-4 days, mean 2.93) while actual delivery takes 0-6 days (mean 3.50), creating a 0.57 day average delay that operations can optimize

### Recommended Feature Engineering Strategy
1) REMOVE LEAKAGE: Drop 'delivery_status' and 'days_for_shipping_real' (post-delivery data). 2) TEMPORAL: Extract features from 'order_date_dateorders' and 'shipping_date_dateorders' - hour of day, day of week, month, quarter, holiday flags, lead time patterns. 3) GEOGRAPHIC: Create distance features using lat/long between customer and order locations; aggregate historical late delivery rates by customer_city/state, order_city/country, and market/region combinations. 4) SHIPPING EFFICIENCY: Create 'scheduled_vs_actual_gap' = days_for_shipment_scheduled as predictor; interaction terms between shipping_mode and customer_segment; shipping_mode frequency by region. 5) FINANCIAL INDICATORS: Profit margin categories from benefit_per_order and order_item_profit_ratio; order value bands from sales; discount impact ratio. 6) CUSTOMER BEHAVIOR: Aggregate customer history - lifetime orders, average order value, historical late delivery rate, preferred shipping modes by customer_id. 7) PRODUCT COMPLEXITY: Category risk scores based on historical late rates by category_name/department_name. 8) CARDINALITY REDUCTION: Target encode high-cardinality categoricals (customer_city, order_city, order_state) using late delivery rates with smoothing. 9) DROP: Remove IDs, constant columns (product_status, customer_email/password), and redundant pairs (customer_id/order_customer_id).

### Analysis Execution Output
```

=== Statistics by Shipping Mode ===
               late_delivery_risk         days_for_shipping_real days_for_shipment_scheduled benefit_per_order
                             mean   count                   mean                        mean              mean
shipping_mode                                                                                                 
First Class                 0.953   27814                  2.000                         1.0            23.122
Same Day                    0.457    9737                  0.478                         0.0            20.850
Second Class                0.766   35216                  3.991                         2.0            21.306
Standard Class              0.381  107752                  3.996                         4.0            21.999

=== Target Distribution ===
On-time (0): 45.17%
Late (1): 54.83%

=== Top 10 Correlations with Late Delivery Risk ===
days_for_shipping_real    0.401415
customer_zipcode          0.003141
category_id               0.001752
product_category_id       0.001752
order_item_cardprod_id    0.001490
product_card_id           0.001490
order_customer_id         0.001484
customer_id               0.001484
department_id             0.001077
latitude                  0.000679
dtype: float64

=== Bottom 10 Correlations with Late Delivery Risk ===
order_item_product_price      -0.002175
order_item_profit_ratio       -0.002316
sales                         -0.003564
benefit_per_order             -0.003727
order_profit_per_order        -0.003727
sales_per_customer            -0.003791
order_item_total              -0.003791
days_for_shipment_scheduled   -0.369352
product_description                 NaN
product_status                      NaN
dtype: float64

Plot saved to C:\Users\israb\Documents\Agente_RPA\intelligent_analysis.png

```

---
*Analysis generated by Claude 3.5 Sonnet*
