# Product / Dataset Information

## What this dataset contains
Weekly sales for 45 stores, from 2010-02-05 to 2012-10-26, with these fields:

- `Store` — store ID (1–45)
- `Date` — week-ending date
- `Weekly_Sales` — total sales for that store that week
- `Holiday_Flag` — 1 if the week contains a major holiday, else 0
- `Temperature` — average regional temperature that week
- `Fuel_Price` — average regional fuel price
- `CPI` — Consumer Price Index for the region
- `Unemployment` — regional unemployment rate

## What this dataset does NOT contain
This is store-level, not SKU/product-level, data — there is no per-product
breakdown of what was sold. "Inventory" figures shown in the Stock Plan,
Alerts, and AI Assistant pages are **derived from forecasted store-level
demand**, not a live inventory feed — treat them as planning estimates, not
as ground truth for what's physically on a shelf. Connecting a real
inventory/POS system would let every tool in this app (forecast, stock plan,
agent, alerts) operate on real current-inventory numbers instead of the
demand-derived estimate used here.

## Store identity
Store IDs are anonymized (1–45) — there's no mapping to a real Walmart store
number, region, or address in this dataset.
