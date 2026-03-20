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
**Justification:** The column 'late_delivery_risk' is binary (0.0 and 1.0), has a mean of 0.5483 indicating ~55% positive class rate, and directly aligns with the business goal stated: 'predict Late_delivery_risk (1 = late, 0 = on time)'. The 'delivery_status' column appears to be the post-hoc outcome that would not be available at prediction time.

### Problematic Columns
['delivery_status', 'customer_email', 'customer_password', 'product_description', 'product_status', 'order_customer_id', 'customer_id', 'customer_street', 'order_date_dateorders', 'shipping_date_dateorders', 'order_id', 'order_item_id', 'customer_fname', 'customer_lname', 'product_image']

### Top Dataset Insights
1. Target class imbalance (54.8% late deliveries) is moderate, suggesting predictive modeling is viable without extreme resampling. The shipping performance indicates systemic delays across the supply chain.
2. Critical leakage risk: 'delivery_status' is a direct post-hoc label of the outcome, and 'days_for_shipping_real' (actual shipping time) would only be known after delivery. Only 'days_for_shipment_scheduled' should be used for prediction.
3. Geographic complexity is high with 164 countries, 3,597 order cities, and 1,089 order states, indicating strong potential for location-based feature engineering (market aggregations, regional risk scores).
4. Profitability concerns: 'benefit_per_order' has mean $21.98 but ranges to -$4,275, with strong negative skew (-4.74). Late deliveries may correlate with negative profit orders, suggesting operational pressure on low-margin shipments.
5. Shipping mode has only 4 categories and scheduled days only 4 unique values (0-4), while actual shipping takes 0-6 days. The gap between scheduled (mean=2.93) and actual (mean=3.50) days reveals consistent underestimation of delivery times.

### Recommended Feature Engineering Strategy
1) **Temporal Features**: Extract order_date components (hour, day_of_week, month, quarter) and create time-to-ship gap ('days_for_shipment_scheduled' only). Calculate days_since_last_order per customer. 2) **Geographic Aggregations**: Create market/region-level late_delivery_risk rates, average shipping days, and order volumes as features to encode location risk without high cardinality. Use target encoding for order_country/order_city with regularization. 3) **Customer Behavior**: Aggregate customer_id features (total orders, average order value, historical late rate, preferred shipping mode). Calculate customer segment risk scores. 4) **Product/Category Risk**: Create category-level and product-level aggregations of late delivery rates, average shipping days, and profit margins. 5) **Interaction Features**: Create scheduled_days × shipping_mode, market × customer_segment, and benefit_per_order × shipping_mode interactions. 6) **Remove Leakage**: Drop delivery_status, days_for_shipping_real, shipping_date, and all customer PII. Drop constant columns (product_status, customer_email/password) and high-cardinality IDs without aggregation value.

### Analysis Execution Output
```
=== Shipping Mode Analysis ===
               late_delivery_risk         days_for_shipping_real days_for_shipment_scheduled benefit_per_order
                             mean   count                   mean                        mean              mean
shipping_mode                                                                                                 
First Class                 0.953   27814                  2.000                         1.0            23.122
Same Day                    0.457    9737                  0.478                         0.0            20.850
Second Class                0.766   35216                  3.991                         2.0            21.306
Standard Class              0.381  107752                  3.996                         4.0            21.999

=== Market Analysis ===
             late_delivery_risk        days_for_shipping_real benefit_per_order
                           mean  count                   mean              mean
market                                                                         
Africa                    0.546  11614                  3.511            21.704
Europe                    0.552  50252                  3.488            23.272
LATAM                     0.544  51594                  3.509            21.772
Pacific Asia              0.550  41260                  3.501            20.789
USCA                      0.548  25799                  3.485            21.873

=== Top 15 Correlations with Late Delivery Risk ===
late_delivery_risk          1.000000
days_for_shipping_real      0.401415
customer_zipcode            0.003151
category_id                 0.001752
product_category_id         0.001752
order_item_cardprod_id      0.001490
product_card_id             0.001490
order_customer_id           0.001484
customer_id                 0.001484
department_id               0.001077
latitude                    0.000679
order_item_discount_rate    0.000404
order_item_quantity        -0.
```

---
*Analysis generated by Claude 3.5 Sonnet*
