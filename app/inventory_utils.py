"""
Inventory + supplier "tools" for the AI Agent.

This dataset is store-level weekly SALES data — it has no real inventory feed.
To make the agent's reorder logic meaningfully demoable, current inventory is
DERIVED deterministically from each store's recent demand (see
product_information.md in the knowledge base, which documents this limitation
plainly to the user). Swap get_inventory() for a real POS/inventory API call
and everything downstream (agent, alerts, stock plan) keeps working unchanged.
"""
import hashlib

STANDARD_LEAD_TIME_DAYS = 14
EXPEDITED_LEAD_TIME_DAYS = 5


def get_inventory(store_id, recent_avg_weekly_sales, lead_time_weeks=2):
    """
    Deterministic pseudo-current-inventory, seeded by store id so it's stable
    across calls (not random noise on every request). Centered around what a
    store carrying roughly one lead-time window of stock would have, with
    +/- variance so some stores land under threshold and some over it —
    same spread of situations a real inventory feed would show.
    """
    seed = int(hashlib.sha256(f"inv-{store_id}".encode()).hexdigest(), 16)
    factor = 0.70 + (seed % 900) / 1000.0  # 0.70 .. 1.60, mean ~1.15
    inventory = recent_avg_weekly_sales * lead_time_weeks * factor
    return {
        "store": store_id,
        "current_inventory_units_equiv": round(inventory, 2),
        "note": "Simulated from recent demand — no live inventory feed in this dataset.",
    }


def get_supplier_lead_time(expedited=False):
    days = EXPEDITED_LEAD_TIME_DAYS if expedited else STANDARD_LEAD_TIME_DAYS
    return {
        "lead_time_days": days,
        "lead_time_weeks": round(days / 7, 2),
        "expedited": expedited,
    }
