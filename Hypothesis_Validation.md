# Hypothesis Validation

**Target:** `late_delivery_risk` | TRUE: 6 | FALSE: 2 | INCONCLUSIVE: 2

| ID | Hypothesis | Verdict | Business Insight |
|----|-----------|---------|-----------------|
| H1 | Orders with higher actual shipping days (days_for_shipping_real) tend to have hi | **FALSE** | The business should focus on keeping shipping times at 3-4 days maximum, as dela |
| H2 | Orders with lower scheduled shipping days (days_for_shipment_scheduled) tend to  | **TRUE** | The business should either avoid offering 1-2 day shipping options or invest hea |
| H3 | Orders where actual shipping days exceed scheduled days tend to have higher late | **TRUE** | The business should prioritize reducing shipping delays as even a 1-day delay is |
| H4 | Orders of specific transaction types (type) tend to have higher late_delivery_ri | **TRUE** | The business should prioritize resource allocation and handling procedures for P |
| H5 | Orders from specific markets (market) tend to have higher late_delivery_risk due | **INCONCLUSIVE** | The consistently high late delivery risk across all markets (54-55%) indicates a |
| H6 | Orders in specific product categories (category_name) tend to have higher late_d | **TRUE** | The business should prioritize improving logistics and inventory management for  |
| H7 | Orders from specific customer segments (customer_segment) tend to have higher la | **FALSE** | Since all customer segments face similar late delivery risks, the business shoul |
| H8 | Orders from specific departments (department_name) tend to have higher late_deli | **TRUE** | The business should investigate operational processes in Pet Shop and Book Shop  |
| H9 | Orders shipped to specific countries (order_country) tend to have higher late_de | **INCONCLUSIVE** | The business should collect more data from these countries before making shippin |
| H10 | Orders with lower benefit per order (benefit_per_order) tend to have slightly hi | **TRUE** | The business should consider implementing priority fulfillment processes for hig |
