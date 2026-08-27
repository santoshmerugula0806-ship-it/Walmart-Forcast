import os
import json
import joblib
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "model")

_model = None
_metadata = None
_full_raw = None       # engineered features incl. NaN rows, all history, per store
_full_clean = None      # engineered features, NaN-free (train+test)
_model_comparison = None
_feature_importance = None
_actual_vs_predicted = None
_row_predictions = None
_production_plan = None


def load_everything():
    global _model, _metadata, _full_raw, _full_clean, _model_comparison
    global _feature_importance, _actual_vs_predicted, _row_predictions, _production_plan

    with open(os.path.join(MODEL_DIR, "metadata.json")) as f:
        _metadata = json.load(f)

    model_format = _metadata.get("model_format", "joblib")
    if model_format == "xgboost_json":
        _model = XGBRegressor(n_jobs=1)
        _model.load_model(os.path.join(MODEL_DIR, "best_model.json"))
    else:
        _model = joblib.load(os.path.join(MODEL_DIR, "best_model.joblib"))

    _full_raw = pd.read_csv(os.path.join(DATA_DIR, "full_features_raw.csv"), parse_dates=["Date"])
    _full_clean = pd.read_csv(os.path.join(DATA_DIR, "full_features.csv"), parse_dates=["Date"])

    _model_comparison = pd.read_csv(os.path.join(MODEL_DIR, "model_comparison.csv"))
    _feature_importance = pd.read_csv(os.path.join(MODEL_DIR, "feature_importance.csv"), index_col=0)
    _actual_vs_predicted = pd.read_csv(os.path.join(MODEL_DIR, "actual_vs_predicted_weekly.csv"))
    _row_predictions = pd.read_csv(os.path.join(MODEL_DIR, "row_predictions.csv"))
    _production_plan = pd.read_csv(os.path.join(MODEL_DIR, "production_plan.csv"))


def get_metadata():
    return _metadata


def get_model_comparison():
    return _model_comparison.to_dict(orient="records")


def get_feature_importance():
    return [{"feature": idx, "importance": float(row["importance"])}
            for idx, row in _feature_importance.iterrows()]


def get_actual_vs_predicted():
    return _actual_vs_predicted.to_dict(orient="records")


def get_holiday_effect():
    grp = _full_clean.groupby("Holiday_Flag")["Weekly_Sales"].mean()
    non_holiday = float(grp.get(0, 0.0))
    holiday = float(grp.get(1, 0.0))
    lift_pct = ((holiday - non_holiday) / non_holiday * 100) if non_holiday else 0.0
    return {
        "non_holiday_avg": round(non_holiday, 2),
        "holiday_avg": round(holiday, 2),
        "lift_pct": round(lift_pct, 2),
    }


def get_stores():
    stores = []
    for store_id, grp in _full_clean.groupby("Store"):
        stores.append({
            "store": int(store_id),
            "avg_weekly_sales": round(float(grp["Weekly_Sales"].mean()), 2),
            "latest_date": grp["Date"].max().strftime("%Y-%m-%d"),
            "n_weeks": int(len(grp)),
        })
    stores.sort(key=lambda s: s["store"])
    return stores


def get_store_history(store_id):
    grp = _full_clean[_full_clean["Store"] == store_id].sort_values("Date")
    if grp.empty:
        return None
    return {
        "store": store_id,
        "dates": grp["Date"].dt.strftime("%Y-%m-%d").tolist(),
        "actual_sales": grp["Weekly_Sales"].round(2).tolist(),
        "holiday_flag": grp["Holiday_Flag"].tolist(),
    }


def get_store_actual_vs_predicted(store_id):
    grp = _row_predictions[_row_predictions["Store"] == store_id].sort_values("Date")
    if grp.empty:
        return None
    return {
        "store": store_id,
        "dates": grp["Date"].tolist(),
        "actual": grp["actual"].round(2).tolist(),
        "predicted": grp["predicted"].round(2).tolist(),
    }


def get_latest_known_state(store_id):
    """Latest fully engineered (non-null) row for a store - the jumping-off
    point for a forward forecast."""
    grp = _full_raw[(_full_raw["Store"] == store_id)].dropna(
        subset=["sales_lag_12", "rolling_mean_12"]
    ).sort_values("Date")
    if grp.empty:
        return None
    last = grp.iloc[-1].to_dict()
    last["Date"] = pd.Timestamp(last["Date"]).strftime("%Y-%m-%d")
    return last


def _row_to_feature_vector(row):
    cols = _metadata["feature_cols"]
    return pd.DataFrame([[row[c] for c in cols]], columns=cols)


def forecast_store(store_id, weeks=4, overrides=None):
    """
    Recursive multi-step forecast for a single store, starting from the last
    known engineered state in the dataset.

    overrides: optional dict applied to EVERY future week for exogenous vars
    the model can't know in advance, e.g. {"Holiday_Flag": 1, "Temperature": 55,
    "Fuel_Price": 3.1, "CPI": 215.0, "Unemployment": 7.5}. Anything not
    overridden is carried forward from the last known value (a reasonable
    naive assumption for a short horizon).
    """
    overrides = overrides or {}
    grp = _full_raw[_full_raw["Store"] == store_id].dropna(
        subset=["sales_lag_12", "rolling_mean_12"]
    ).sort_values("Date").reset_index(drop=True)

    if grp.empty:
        return None

    # Rolling window of recent actual/predicted weekly sales (need last 12 for lag/rolling features)
    recent_sales = grp["Weekly_Sales"].tolist()[-12:]
    last_row = grp.iloc[-1]
    cur_date = pd.Timestamp(last_row["Date"])

    exogenous_defaults = {
        "Holiday_Flag": int(last_row["Holiday_Flag"]),
        "Temperature": float(last_row["Temperature"]),
        "Fuel_Price": float(last_row["Fuel_Price"]),
        "CPI": float(last_row["CPI"]),
        "Unemployment": float(last_row["Unemployment"]),
    }
    exogenous_defaults.update(overrides)

    results = []
    for step in range(weeks):
        cur_date = cur_date + pd.Timedelta(weeks=1)

        sales_lag_1 = recent_sales[-1]
        sales_lag_4 = recent_sales[-4] if len(recent_sales) >= 4 else recent_sales[0]
        sales_lag_8 = recent_sales[-8] if len(recent_sales) >= 8 else recent_sales[0]
        sales_lag_12 = recent_sales[-12] if len(recent_sales) >= 12 else recent_sales[0]

        window4 = recent_sales[-4:]
        window8 = recent_sales[-8:] if len(recent_sales) >= 8 else recent_sales
        window12 = recent_sales[-12:] if len(recent_sales) >= 12 else recent_sales

        feat = {
            "sales_lag_1": sales_lag_1,
            "sales_lag_4": sales_lag_4,
            "sales_lag_8": sales_lag_8,
            "sales_lag_12": sales_lag_12,
            "rolling_mean_4": float(np.mean(window4)),
            "rolling_mean_8": float(np.mean(window8)),
            "rolling_mean_12": float(np.mean(window12)),
            "rolling_std_4": float(np.std(window4, ddof=1)) if len(window4) > 1 else 0.0,
            "rolling_std_8": float(np.std(window8, ddof=1)) if len(window8) > 1 else 0.0,
            "rolling_std_12": float(np.std(window12, ddof=1)) if len(window12) > 1 else 0.0,
            "Holiday_Flag": exogenous_defaults["Holiday_Flag"],
            "Temperature": exogenous_defaults["Temperature"],
            "Fuel_Price": exogenous_defaults["Fuel_Price"],
            "CPI": exogenous_defaults["CPI"],
            "Unemployment": exogenous_defaults["Unemployment"],
            "month": cur_date.month,
            "week_of_year": int(cur_date.isocalendar().week),
            "quarter": cur_date.quarter,
        }

        X = _row_to_feature_vector(feat)
        pred = float(_model.predict(X)[0])
        pred = max(pred, 0.0)

        results.append({
            "date": cur_date.strftime("%Y-%m-%d"),
            "predicted_sales": round(pred, 2),
        })

        recent_sales.append(pred)
        recent_sales = recent_sales[-12:]

    return {
        "store": store_id,
        "last_known_date": last_row["Date"] if isinstance(last_row["Date"], str) else pd.Timestamp(last_row["Date"]).strftime("%Y-%m-%d"),
        "last_known_sales": round(float(last_row["Weekly_Sales"]), 2),
        "assumptions": exogenous_defaults,
        "forecast": results,
    }


def compute_stock_plan(safety_stock_pct=0.15, lead_time_weeks=2):
    """Recompute the stock/production plan from stored test-set predictions,
    using an adjustable safety-stock % and lead-time window (in weeks)."""
    df = _row_predictions.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    latest_date = df["Date"].max()
    horizon_start = latest_date - pd.Timedelta(weeks=lead_time_weeks - 1)
    horizon = df[df["Date"] >= horizon_start]

    plan = horizon.groupby("Store")["predicted"].sum().reset_index()
    plan = plan.rename(columns={"predicted": "forecasted_demand"})
    plan["forecasted_demand"] = plan["forecasted_demand"].clip(lower=0).round().astype(int)
    plan["safety_stock"] = (plan["forecasted_demand"] * safety_stock_pct).round().astype(int)
    plan["recommended_stock"] = plan["forecasted_demand"] + plan["safety_stock"]
    plan = plan.sort_values("Store")
    return plan.to_dict(orient="records")
