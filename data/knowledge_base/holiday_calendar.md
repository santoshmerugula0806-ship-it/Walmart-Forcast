# Holiday Calendar

The dataset's Holiday_Flag marks weeks containing one of the following
federal holidays, which historically show a measurable sales lift (see the
Dashboard's holiday-effect chart):

- **Super Bowl week** — early February
- **Labor Day** — early September
- **Thanksgiving** — late November
- **Christmas** — late December

## Planning implications
- Forecast demand for these weeks tends to run above the non-holiday average
  — use the Forecast page's "Force holiday week" override when forecasting
  into a known holiday week beyond the model's recent-history carry-forward.
- Supplier lead times increase in the 3 weeks before a holiday (see
  supplier_rules.md) — factor this into reorder timing, not just quantity.
- Thanksgiving and Christmas show the largest historical lift and are the
  two windows most likely to produce a HIGH risk alert if safety stock isn't
  increased ahead of time.
