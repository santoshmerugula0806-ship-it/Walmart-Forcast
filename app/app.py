from flask import Flask, render_template, jsonify, request
import model_utils as mu

app = Flask(__name__)

mu.load_everything()


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
    return render_template("forecast.html")


@app.route("/stock-plan")
def stock_plan_page():
    return render_template("stock_plan.html")


@app.route("/about")
def about_page():
    return render_template("about.html", meta=mu.get_metadata())


# ------------------------------------------------------------------ api ----

@app.route("/api/model-metrics")
def api_model_metrics():
    return jsonify({
        "metadata": mu.get_metadata(),
        "comparison": mu.get_model_comparison(),
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

    overrides = {}
    for key in ["Holiday_Flag", "Temperature", "Fuel_Price", "CPI", "Unemployment"]:
        if key in body and body[key] not in (None, ""):
            overrides[key] = float(body[key]) if key != "Holiday_Flag" else int(body[key])

    result = mu.forecast_store(store_id, weeks=weeks, overrides=overrides)
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


if __name__ == "__main__":
    import os
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    app.run(host="0.0.0.0", port=5000, debug=debug)
