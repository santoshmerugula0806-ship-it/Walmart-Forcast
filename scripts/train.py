"""
Replicates Walmart_Model_Training.ipynb:
- trains Naive baseline, Linear Regression, Random Forest, XGBoost
- evaluates with MAE / RMSE / MAPE / WAPE
- picks the best model by WAPE
- saves the trained model + metrics + feature importance + a production/stock plan
"""
import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "model")

FEATURE_COLS = [
    "sales_lag_1", "sales_lag_4", "sales_lag_8", "sales_lag_12",
    "rolling_mean_4", "rolling_mean_8", "rolling_mean_12",
    "rolling_std_4", "rolling_std_8", "rolling_std_12",
    "Holiday_Flag", "Temperature", "Fuel_Price", "CPI", "Unemployment",
    "month", "week_of_year", "quarter",
]
TARGET = "Weekly_Sales"

SAFETY_STOCK_PCT = 0.15
LEAD_TIME_WEEKS = 2


def evaluate(y_true, y_pred, model_name):
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)

    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))

    nonzero = y_true != 0
    mape = np.mean(np.abs((y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero])) * 100

    wape = np.sum(np.abs(y_true - y_pred)) / np.sum(y_true) * 100

    return {"Model": model_name, "MAE": round(mae, 2), "RMSE": round(rmse, 2),
            "MAPE (%)": round(mape, 2), "WAPE (%)": round(wape, 2)}


def run():
    os.makedirs(MODEL_DIR, exist_ok=True)

    train = pd.read_csv(os.path.join(DATA_DIR, "train_features.csv"), parse_dates=["Date"])
    test = pd.read_csv(os.path.join(DATA_DIR, "test_features.csv"), parse_dates=["Date"])

    X_train, y_train = train[FEATURE_COLS], train[TARGET]
    X_test, y_test = test[FEATURE_COLS], test[TARGET]

    naive_pred = test["sales_lag_1"].values
    naive_results = evaluate(y_test, naive_pred, "Naive (last week's sales)")

    lr = LinearRegression()
    lr.fit(X_train, y_train)
    lr_pred = lr.predict(X_test)

    rf = RandomForestRegressor(
        n_estimators=200, max_depth=12, min_samples_leaf=5,
        random_state=42, n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)

    xgb = XGBRegressor(
        n_estimators=400, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1,
    )
    xgb.fit(X_train, y_train)
    xgb_pred = xgb.predict(X_test)

    results_df = pd.DataFrame([
        naive_results,
        evaluate(y_test, lr_pred, "Linear Regression"),
        evaluate(y_test, rf_pred, "Random Forest"),
        evaluate(y_test, xgb_pred, "XGBoost"),
    ])

    best_model_name = results_df.loc[results_df["WAPE (%)"].idxmin(), "Model"]
    print("Best model:", best_model_name)

    models = {"Linear Regression": lr, "Random Forest": rf, "XGBoost": xgb}
    best_model = models.get(best_model_name, xgb)

    preds_by_model = {
        "Naive (last week's sales)": naive_pred,
        "Linear Regression": lr_pred,
        "Random Forest": rf_pred,
        "XGBoost": xgb_pred,
    }
    best_pred = preds_by_model[best_model_name]

    # Feature importance (best_model may not have it, e.g. Linear Regression -> use coef)
    if hasattr(best_model, "feature_importances_"):
        importances = pd.Series(best_model.feature_importances_, index=FEATURE_COLS)
    else:
        importances = pd.Series(np.abs(best_model.coef_), index=FEATURE_COLS)
    importances = (importances / importances.sum()).sort_values(ascending=False)

    # Actual vs predicted, aggregated by date (for chart)
    chart_df = test[["Date"]].copy()
    chart_df["actual"] = y_test.values
    chart_df["predicted"] = best_pred
    weekly = chart_df.groupby("Date")[["actual", "predicted"]].sum().reset_index()
    weekly["Date"] = weekly["Date"].dt.strftime("%Y-%m-%d")

    # Per-row predictions (for store-level lookups / scatter plot, sampled)
    row_preds = test[["Store", "Date"]].copy()
    row_preds["actual"] = y_test.values
    row_preds["predicted"] = best_pred
    row_preds["Date"] = row_preds["Date"].dt.strftime("%Y-%m-%d")

    # Production / stock plan
    plan = test[["Store", "Date"]].copy()
    plan["predicted_sales"] = np.clip(best_pred, a_min=0, a_max=None)
    latest_date = plan["Date"].max()
    horizon_start = latest_date - pd.Timedelta(weeks=LEAD_TIME_WEEKS - 1)
    horizon = plan[plan["Date"] >= horizon_start]

    production_plan = horizon.groupby("Store")["predicted_sales"].sum().reset_index()
    production_plan = production_plan.rename(columns={"predicted_sales": "forecasted_demand"})
    production_plan["forecasted_demand"] = production_plan["forecasted_demand"].round().astype(int)
    production_plan["safety_stock"] = (production_plan["forecasted_demand"] * SAFETY_STOCK_PCT).round().astype(int)
    production_plan["recommended_stock"] = production_plan["forecasted_demand"] + production_plan["safety_stock"]

    # ---- Save everything the app needs ----
    if isinstance(best_model, XGBRegressor):
        # Native XGBoost format - portable across xgboost/sklearn/numpy versions,
        # unlike a joblib/pickle dump of the sklearn wrapper.
        best_model.save_model(os.path.join(MODEL_DIR, "best_model.json"))
        model_format = "xgboost_json"
    else:
        joblib.dump(best_model, os.path.join(MODEL_DIR, "best_model.joblib"))
        model_format = "joblib"

    with open(os.path.join(MODEL_DIR, "metadata.json"), "w") as f:
        json.dump({
            "feature_cols": FEATURE_COLS,
            "target": TARGET,
            "best_model_name": best_model_name,
            "model_format": model_format,
            "safety_stock_pct": SAFETY_STOCK_PCT,
            "lead_time_weeks": LEAD_TIME_WEEKS,
        }, f, indent=2)

    results_df.to_csv(os.path.join(MODEL_DIR, "model_comparison.csv"), index=False)
    importances.rename("importance").to_csv(os.path.join(MODEL_DIR, "feature_importance.csv"))
    weekly.to_csv(os.path.join(MODEL_DIR, "actual_vs_predicted_weekly.csv"), index=False)
    row_preds.to_csv(os.path.join(MODEL_DIR, "row_predictions.csv"), index=False)
    production_plan.to_csv(os.path.join(MODEL_DIR, "production_plan.csv"), index=False)

    print("Saved model + metrics to", MODEL_DIR)
    print(results_df)


if __name__ == "__main__":
    run()
