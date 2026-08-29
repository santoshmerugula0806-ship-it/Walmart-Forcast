"""
Agentic AI layer.

The agent has four tools:
  - forecast_tool     -> model_utils.forecast_store() (XGBoost, production model)
  - inventory_tool     -> inventory_utils.get_inventory()  (simulated, see that module)
  - supplier_tool       -> inventory_utils.get_supplier_lead_time()
  - policy_tool (RAG)  -> rag_utils.retrieve()

Decision logic is deterministic/rule-based (see stock_thresholds.md for the
exact bands), so the agent always produces a reproducible, auditable
recommendation even with no LLM configured. If ANTHROPIC_API_KEY is set,
the same tool outputs are additionally handed to Claude purely to phrase the
final explanation in natural language — Claude never invents the numbers,
it only narrates numbers the deterministic logic already computed.
"""
import os

import model_utils as mu
import inventory_utils as iu
import rag_utils as rag

SAFETY_STOCK_PCT = 0.15
LEAD_TIME_WEEKS = 2


def _risk_band(shortfall, forecasted_demand):
    if forecasted_demand <= 0:
        return "LOW"
    pct = shortfall / forecasted_demand
    if shortfall <= 0:
        return "LOW"
    if pct <= 0.20:
        return "MEDIUM"
    return "HIGH"


def check_store(store_id, lead_time_weeks=LEAD_TIME_WEEKS, safety_stock_pct=SAFETY_STOCK_PCT):
    """
    Runs the full agent flow for one store:
      forecast -> inventory -> supplier lead time -> policy -> decision
    Returns a structured result usable by both the API and the automation job.
    """
    # --- Tool 1: forecast ---
    forecast = mu.forecast_store(store_id, weeks=lead_time_weeks)
    if forecast is None:
        return None
    forecasted_demand = sum(w["predicted_sales"] for w in forecast["forecast"])

    # --- Tool 2: inventory (simulated) ---
    stores = {s["store"]: s for s in mu.get_stores()}
    recent_avg = stores.get(store_id, {}).get("avg_weekly_sales", forecasted_demand / max(lead_time_weeks, 1))
    inventory = iu.get_inventory(store_id, recent_avg, lead_time_weeks=lead_time_weeks)
    current_inventory = inventory["current_inventory_units_equiv"]

    # --- Tool 3: supplier lead time ---
    supplier = iu.get_supplier_lead_time(expedited=False)

    # --- Tool 4: policy (RAG) ---
    policy_hits = rag.retrieve("safety stock reorder threshold", k=2)

    # --- Deterministic decision ---
    safety_stock = forecasted_demand * safety_stock_pct
    required = forecasted_demand + safety_stock
    shortfall = max(required - current_inventory, 0.0)
    risk = _risk_band(shortfall, forecasted_demand)

    recommend_expedite = risk == "HIGH"
    if recommend_expedite:
        supplier = iu.get_supplier_lead_time(expedited=True)

    decision = {
        "store": store_id,
        "lead_time_weeks": lead_time_weeks,
        "forecasted_demand": round(forecasted_demand, 2),
        "current_inventory": round(current_inventory, 2),
        "safety_stock_pct": safety_stock_pct,
        "safety_stock": round(safety_stock, 2),
        "required_stock": round(required, 2),
        "shortfall": round(shortfall, 2),
        "risk": risk,
        "recommend_reorder": shortfall > 0,
        "recommend_expedite": recommend_expedite,
        "supplier_lead_time_days": supplier["lead_time_days"],
        "policy_sources": [{"doc": h["doc"], "heading": h["heading"]} for h in policy_hits],
    }

    decision["explanation"] = _explain(decision)
    return decision


def _explain(decision):
    """Deterministic narrative. Used as-is unless Claude is configured (see ask_agent)."""
    s = decision
    if s["risk"] == "LOW":
        return (f"Store {s['store']} is on track. Forecasted demand over the next "
                f"{s['lead_time_weeks']} weeks is ~{s['forecasted_demand']:,.0f}, current "
                f"inventory (~{s['current_inventory']:,.0f}) covers that plus the "
                f"{int(s['safety_stock_pct']*100)}% safety-stock buffer. No action needed.")
    if s["risk"] == "MEDIUM":
        return (f"Store {s['store']} has a projected shortfall of ~{s['shortfall']:,.0f} units "
                f"against forecasted demand of ~{s['forecasted_demand']:,.0f} over the next "
                f"{s['lead_time_weeks']} weeks. This is within the MEDIUM band — add it to the "
                f"next scheduled order (standard {s['supplier_lead_time_days']}-day lead time).")
    return (f"Store {s['store']} has a projected shortfall of ~{s['shortfall']:,.0f} units, "
            f"over 20% of forecasted demand (~{s['forecasted_demand']:,.0f}). This is HIGH risk per "
            f"policy — recommend same-day review and an expedited order "
            f"({s['supplier_lead_time_days']}-day lead time) rather than waiting for the next cycle.")


def ask_agent(question, store_id=None, lead_time_weeks=LEAD_TIME_WEEKS):
    """
    Natural-language entry point, e.g. "Check Store 10 and tell me whether we
    need to reorder." Runs the same deterministic flow, then optionally uses
    Claude to phrase the final answer (grounded strictly in the computed
    decision — see rag_utils._claude_answer for the same pattern).
    """
    if store_id is None:
        return {"error": "Please specify a store (e.g. 'Check Store 10')."}

    decision = check_store(store_id, lead_time_weeks=lead_time_weeks)
    if decision is None:
        return {"error": f"Store {store_id} not found."}

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if api_key:
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=api_key)
            context = (
                f"forecast_tool -> forecasted_demand={decision['forecasted_demand']} over "
                f"{decision['lead_time_weeks']} weeks\n"
                f"inventory_tool -> current_inventory={decision['current_inventory']}\n"
                f"supplier_tool -> lead_time_days={decision['supplier_lead_time_days']}\n"
                f"policy -> safety_stock_pct={decision['safety_stock_pct']}, "
                f"required_stock={decision['required_stock']}, shortfall={decision['shortfall']}, "
                f"risk={decision['risk']}"
            )
            resp = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=300,
                system=("You are the Walmart AI Agent. Phrase a short, decisive answer to the "
                        "manager's question using ONLY the tool outputs given. Do not invent numbers."),
                messages=[{"role": "user", "content": f"Tool outputs:\n{context}\n\nQuestion: {question}"}],
            )
            decision["explanation"] = "".join(b.text for b in resp.content if b.type == "text")
            decision["generated_by"] = "claude"
        except Exception:
            decision["generated_by"] = "deterministic"
    else:
        decision["generated_by"] = "deterministic"

    return decision


def check_all_stores(lead_time_weeks=LEAD_TIME_WEEKS, safety_stock_pct=SAFETY_STOCK_PCT):
    stores = mu.get_stores()
    results = []
    for s in stores:
        d = check_store(s["store"], lead_time_weeks=lead_time_weeks, safety_stock_pct=safety_stock_pct)
        if d:
            results.append(d)
    return results
