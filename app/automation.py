"""
Automation layer.

Runs agent.check_all_stores() on a schedule (default: daily at 08:00 server
time) and persists any MEDIUM/HIGH risk stores as alerts to model/alerts.json,
so the /alerts page and /api/alerts endpoint always have the latest sweep
without recomputing on every page load. Also exposed for a manual trigger
(POST /api/alerts/run) for demoing without waiting for the schedule.
"""
import os
import json
from datetime import datetime, timezone

import agent

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ALERTS_PATH = os.path.join(BASE_DIR, "model", "alerts.json")

_scheduler = None


def run_daily_check():
    all_results = agent.check_all_stores()
    alerts = [r for r in all_results if r["risk"] in ("MEDIUM", "HIGH")]
    alerts.sort(key=lambda r: (r["risk"] != "HIGH", -r["shortfall"]))

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "stores_checked": len(all_results),
        "alert_count": len(alerts),
        "high_count": sum(1 for a in alerts if a["risk"] == "HIGH"),
        "medium_count": sum(1 for a in alerts if a["risk"] == "MEDIUM"),
        "alerts": alerts,
    }
    with open(ALERTS_PATH, "w") as f:
        json.dump(payload, f, indent=2)
    return payload


def get_latest_alerts():
    if not os.path.exists(ALERTS_PATH):
        return None
    with open(ALERTS_PATH) as f:
        return json.load(f)


def start_scheduler():
    """Best-effort: starts a background daily job. If APScheduler isn't
    installed or fails to start, the app still works fine — alerts just need
    a manual trigger via POST /api/alerts/run instead."""
    global _scheduler
    try:
        from apscheduler.schedulers.background import BackgroundScheduler
    except ImportError:
        print("[automation] APScheduler not installed — daily auto-run disabled; use POST /api/alerts/run.")
        return

    if _scheduler is not None:
        return

    _scheduler = BackgroundScheduler(daemon=True)
    _scheduler.add_job(run_daily_check, "cron", hour=8, minute=0, id="daily_stock_check")
    _scheduler.start()
    print("[automation] Daily stock-risk check scheduled for 08:00.")
