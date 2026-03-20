# Hypothesis Validation

**Target:** `late_delivery_risk` | TRUE: 2 | FALSE: 3 | INCONCLUSIVE: 5

| ID | Hypothesis | Verdict | Business Insight |
|----|-----------|---------|-----------------|
| H1 | Orders with higher days_for_shipping_real tend to have higher late_delivery_risk | **FALSE** | The business should investigate why extremely short shipping windows (5-6 days)  |
| H2 | Orders with lower days_for_shipment_scheduled tend to have higher late_delivery_ | **FALSE** | The business should investigate why 1-day shipments are failing at such high rat |
| H3 | Orders where days_for_shipping_real exceeds days_for_shipment_scheduled tend to  | **TRUE** | The business should prioritize identifying and addressing root causes of shippin |
| H4 | Orders of specific transaction types tend to have higher late_delivery_risk rate | **TRUE** | The business should prioritize improving delivery processes for PAYMENT, DEBIT,  |
| H5 | Orders from specific markets tend to have higher late_delivery_risk due to geogr | **INCONCLUSIVE** | Late delivery risk is essentially uniform across all markets (54-55%), suggestin |
| H6 | Orders from specific customer_country tend to have higher late_delivery_risk due | **INCONCLUSIVE** | The similar risk rates between EE. UU. and Puerto Rico suggest that country-spec |
| H7 | Orders in specific category_name tend to have higher late_delivery_risk due to h | **INCONCLUSIVE** | Categories like Golf Bags & Carts, Lacrosse, and Pet Supplies show moderately el |
| H8 | Orders from specific department_name tend to have higher late_delivery_risk due  | **INCONCLUSIVE** | The relatively uniform late delivery risk across all departments (54-59%) sugges |
| H9 | Orders where order_country differs from customer_country tend to have higher lat | **INCONCLUSIVE** | International orders show a concerning 54.8% late delivery risk rate, suggesting |
| H10 | Orders from specific customer_segment tend to have different late_delivery_risk  | **FALSE** | Service level agreements appear to be applied consistently across customer segme |
