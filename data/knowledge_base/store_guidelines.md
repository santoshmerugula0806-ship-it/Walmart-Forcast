# Store Guidelines

## Store review cadence
Store managers should review the Forecast and Stock Plan pages every Monday
morning before placing the week's replenishment order. For stores flagged
HIGH risk by the automated daily check, review is expected same-day.

## What-if scenarios
Before a known local event (a store closure nearby, a local festival, a
weather event) the Forecast page's scenario overrides should be used to model
the likely effect on Temperature, Holiday_Flag, or other exogenous inputs
rather than relying on the default "carry forward last known value" behavior,
since the default assumes conditions stay the same as the most recent week.

## Interpreting forecast confidence
Forecasts are generated recursively — each predicted week feeds into the
next. Accuracy degrades gradually as the horizon grows. Treat forecasts
beyond 8 weeks as directional guidance for planning purposes, not as
a precise number to order against. Forecasts inside the 1–4 week window
are the most reliable and are what the automated reorder recommendations
are based on.

## Escalation contacts
HIGH risk alerts should be acknowledged within 4 business hours. If a store
manager cannot resolve a HIGH risk alert (e.g. supplier cannot expedite),
escalate to the regional inventory manager per inventory_policy.md.

## Data quality
If a store's recent sales history looks anomalous (e.g. a data entry error,
a system outage that suppressed recorded sales), flag it before trusting the
forecast — the model has no way to distinguish a genuine demand drop from a
data quality issue.
