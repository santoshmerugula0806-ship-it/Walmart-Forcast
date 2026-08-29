# Supplier Rules

## Standard lead time
The default supplier lead time used for replenishment planning is **2 weeks**
(14 days) from purchase order to store receipt, for standard grocery and
general-merchandise categories.

## Expedited lead time
An expedited order can cut lead time to **5 business days**, but carries a
premium freight surcharge and should only be used when a store is flagged
HIGH risk (see inventory_policy.md escalation rule) and standard replenishment
would arrive too late to avoid a stockout.

## Holiday-period lead times
During the 3 weeks immediately before a federal holiday in the holiday
calendar, supplier lead times increase by approximately **3–5 additional
days** due to network-wide order volume. Replenishment planning during these
windows should add this buffer on top of the standard 2-week lead time.

## Minimum order quantities
Orders below 500 units per SKU may be consolidated with a neighboring store's
order to meet supplier minimums, which can add up to 2 days to the delivery
timeline for the smaller of the two stores.

## Supplier reliability
Historical on-time delivery rate for standard orders is approximately 94%.
When computing safety stock for a store with a recent history of late
deliveries, consider using the upper end of the safety-stock range (20–25%)
rather than the 15% baseline.
