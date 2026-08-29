# Inventory Policy

## Safety stock
Every store carries safety stock equal to **15% of forecasted demand** over the
replenishment window, by default. This buffer exists to absorb forecast error
and short-term demand spikes without triggering a stockout. Store managers may
request an adjusted safety-stock percentage (via the Stock Plan page) for
categories with higher volatility — for example, seasonal or promotional
items may warrant 20–25% instead of the 15% baseline.

## Reorder trigger
A store should reorder when:

    projected_inventory_at_delivery = current_inventory - demand_during_lead_time

is **less than** the safety stock requirement. Projected inventory at delivery
accounts for the fact that stock keeps depleting during the supplier's lead
time, before the new shipment arrives.

## Minimum inventory threshold
No store should be allowed to fall below **10% of its trailing 12-week average
weekly sales** in on-hand inventory at any point, regardless of the safety
stock calculation above — this is a hard floor, not a target.

## Replenishment cadence
Standard stores are reviewed weekly. High-volume stores (trailing 12-week
average weekly sales above $2M) are reviewed twice weekly, since demand
volatility at that scale can exhaust a week's safety stock faster.

## Escalation
If a store's projected shortfall exceeds 20% of forecasted demand for the
replenishment window, the recommendation is escalated to "HIGH" risk and
should trigger a same-day review by the regional inventory manager rather
than waiting for the next scheduled cycle.
