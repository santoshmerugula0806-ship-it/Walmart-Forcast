from flask import Flask, render_template, jsonify, request
import model_utils as mu
import rag_utils as rag
import agent
import automation

app = Flask(__name__)

mu.load_everything()
rag.build_index()
automation.start_scheduler()


# ---------------------------------------------------------------- pages ----

@app.route("/")
def home():
    return render_template("index.html", meta=mu.get_metadata())


@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")


@app.route("/stores")
def stores_page():
    return render_template("stores.html")


@app.route("/forecast")
def forecast_page():
    return render_template("forecast.html", lstm_available=mu.lstm_is_available())


@app.route("/stock-plan")
def stock_plan_page():
    return render_template("stock_plan.html")


@app.route("/explain")
def explain_page():
    return render_template("explain.html")


@app.route("/ai-assistant")
def ai_assistant_page():
    return render_template("ai_assistant.html")


@app.route("/alerts")
def alerts_page():
    return render_template("alerts.html")


@app.route("/about")
def about_page():
    return render_template("about.html", meta=mu.get_metadata())


# ------------------------------------------------------------------ api ----

@app.route("/api/model-metrics")
def api_model_metrics():
    return jsonify({
        "metadata": mu.get_metadata(),
        "comparison": mu.get_model_comparison(),
        "lstm_available": mu.lstm_is_available(),
    })


@app.route("/api/feature-importance")
def api_feature_importance():
    return jsonify(mu.get_feature_importance())


@app.route("/api/actual-vs-predicted")
def api_actual_vs_predicted():
    return jsonify(mu.get_actual_vs_predicted())


@app.route("/api/holiday-effect")
def api_holiday_effect():
    return jsonify(mu.get_holiday_effect())


@app.route("/api/stores")
def api_stores():
    return jsonify(mu.get_stores())


@app.route("/api/stores/<int:store_id>/history")
def api_store_history(store_id):
    data = mu.get_store_history(store_id)
    if data is None:
        return jsonify({"error": "store not found"}), 404
    return jsonify(data)


@app.route("/api/stores/<int:store_id>/actual-vs-predicted")
def api_store_avp(store_id):
    data = mu.get_store_actual_vs_predicted(store_id)
    if data is None:
        return jsonify({"error": "no test-set predictions for this store"}), 404
    return jsonify(data)


@app.route("/api/stores/<int:store_id>/latest-state")
def api_store_latest_state(store_id):
    data = mu.get_latest_known_state(store_id)
    if data is None:
        return jsonify({"error": "store not found"}), 404
    return jsonify(data)


@app.route("/api/forecast", methods=["POST"])
def api_forecast():
    body = request.get_json(force=True) or {}
    store_id = int(body.get("store"))
    weeks = int(body.get("weeks", 4))
    weeks = max(1, min(weeks, 26))
    model_choice = (body.get("model") or "xgboost").lower()

    overrides = {}
    for key in ["Holiday_Flag", "Temperature", "Fuel_Price", "CPI", "Unemployment"]:
        if key in body and body[key] not in (None, ""):
            overrides[key] = float(body[key]) if key != "Holiday_Flag" else int(body[key])

    if model_choice == "lstm":
        if not mu.lstm_is_available():
            return jsonify({"error": "LSTM model not available on this server"}), 503
        result = mu.lstm_forecast_store(store_id, weeks=weeks, overrides=overrides)
    else:
        result = mu.forecast_store(store_id, weeks=weeks, overrides=overrides)
        if result is not None:
            result["model"] = "XGBoost"

    if result is None:
        return jsonify({"error": "store not found"}), 404
    return jsonify(result)


@app.route("/api/stock-plan")
def api_stock_plan():
    safety_stock_pct = float(request.args.get("safety_stock_pct", 0.15))
    lead_time_weeks = int(request.args.get("lead_time_weeks", 2))
    lead_time_weeks = max(1, min(lead_time_weeks, 12))
    plan = mu.compute_stock_plan(safety_stock_pct=safety_stock_pct, lead_time_weeks=lead_time_weeks)
    return jsonify({
        "safety_stock_pct": safety_stock_pct,
        "lead_time_weeks": lead_time_weeks,
        "plan": plan,
    })


# --------------------------------------------------- explainable ai (xai) --

@app.route("/api/explain")
def api_explain():
    store_id = int(request.args.get("store", 1))
    result = mu.explain_forecast(store_id)
    if result is None:
        return jsonify({"error": "store not found"}), 404
    return jsonify(result)


# ------------------------------------------------------------- rag / ask --

@app.route("/api/ask", methods=["POST"])
def api_ask():
    body = request.get_json(force=True) or {}
    question = (body.get("question") or "").strip()
    if not question:
        return jsonify({"error": "question is required"}), 400
    result = rag.answer_question(question)
    return jsonify(result)


# ----------------------------------------------------------- agentic ai --

@app.route("/api/agent", methods=["POST"])
def api_agent():
    body = request.get_json(force=True) or {}
    store_id = body.get("store")
    question = body.get("question", "").strip()
    lead_time_weeks = int(body.get("lead_time_weeks", agent.LEAD_TIME_WEEKS))

    if store_id is None:
        return jsonify({"error": "store is required"}), 400
    store_id = int(store_id)

    result = agent.ask_agent(question or f"Check Store {store_id} and tell me whether we need to reorder.",
                              store_id=store_id, lead_time_weeks=lead_time_weeks)
    if "error" in result:
        return jsonify(result), 404
    return jsonify(result)


# -------------------------------------------------------- automation ------

@app.route("/api/alerts")
def api_alerts():
    payload = automation.get_latest_alerts()
    if payload is None:
        payload = automation.run_daily_check()
    return jsonify(payload)


@app.route("/api/alerts/run", methods=["POST"])
def api_alerts_run():
    payload = automation.run_daily_check()
    return jsonify(payload)


if __name__ == "__main__":
    import os
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    app.run(host="0.0.0.0", port=5000, debug=debug)
