# Stock Thresholds

## Default planning parameters
These are the defaults used across the app unless a page-level override is
supplied:

- Safety stock: **15%** of forecasted demand over the lead-time window
- Supplier lead time: **2 weeks** (14 days)
- Minimum inventory floor: **10%** of trailing 12-week average weekly sales

## Risk bands
The automated daily check (see automation) classifies each store into a risk
band based on projected shortfall as a percentage of forecasted demand for
the lead-time window:

| Band | Projected shortfall | Action |
|---|---|---|
| LOW | Inventory covers demand + safety stock | No action needed |
| MEDIUM | Shortfall up to 20% of forecasted demand | Add to next scheduled order |
| HIGH | Shortfall over 20% of forecasted demand | Same-day review, consider expedited order |

## Product-level thresholds
Store-level thresholds above apply uniformly across product categories in
this system, since the underlying dataset is store-level weekly sales rather
than SKU-level. If SKU-level inventory data is integrated later, thresholds
should be recomputed per product category rather than reused at the store
level, since fast-moving and slow-moving SKUs have very different safety
stock needs.
