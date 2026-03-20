# Hypothesis Validation

**Target:** `late_delivery_risk` | TRUE: 6 | FALSE: 2 | INCONCLUSIVE: 2

| ID | Hypothesis | Verdict | Business Insight |
|----|-----------|---------|-----------------|
| H1 | Orders with higher days_for_shipping_real tend to have higher late_delivery_risk | **FALSE** | The business should investigate why 3-4 day shipping windows have dramatically l |
| H2 | Orders with lower days_for_shipment_scheduled tend to have higher late_delivery_ | **TRUE** | The business should allocate more days for shipment scheduling (3-4 days) to sig |
| H3 | Orders where days_for_shipping_real exceeds days_for_shipment_scheduled tend to  | **TRUE** | The business should prioritize reducing shipping delays beyond scheduled times,  |
| H4 | Orders with specific type values tend to have higher late_delivery_risk | **TRUE** | The business should prioritize orders paid via TRANSFER for on-time delivery res |
| H5 | Orders from certain markets tend to have higher late_delivery_risk | **INCONCLUSIVE** | Late delivery risk is consistently around 54-55% across all markets, suggesting  |
| H6 | Orders from specific customer_segment categories tend to have higher late_delive | **FALSE** | Late delivery risk is uniformly distributed across customer segments at approxim |
| H7 | Orders from certain department_name categories tend to have higher late_delivery | **TRUE** | The business should prioritize improving fulfillment processes for Pet Shop and  |
| H8 | Orders from specific category_name groups tend to have higher late_delivery_risk | **TRUE** | The business should prioritize improving logistics and delivery processes for hi |
| H9 | Orders with shipping routes between different order_country and customer_country | **INCONCLUSIVE** | Without the baseline late delivery risk for domestic orders, we cannot yet deter |
| H10 | Orders with lower benefit_per_order tend to have higher late_delivery_risk due t | **TRUE** | The business should consider prioritizing high-value orders in their logistics o |
