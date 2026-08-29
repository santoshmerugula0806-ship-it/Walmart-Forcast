"""
Deep Learning forecaster (Phase 1): an LSTM sequence model that supplements
the XGBoost baseline in train.py.

Unlike the XGBoost model (which consumes hand-engineered lag/rolling features
for a single week), the LSTM consumes the raw last SEQ_LEN weeks of history
per store as a sequence and learns temporal patterns itself.

Outputs (consumed by app/model_utils.py):
  model/lstm_model.pt        - PyTorch state_dict
  model/lstm_meta.json       - architecture + feature/scaling config
  model/model_comparison.csv - XGBoost row (from train.py) + LSTM row appended
"""
import os
import sys
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "model")
sys.path.insert(0, os.path.join(BASE_DIR, "app"))
from lstm_model import LSTMForecaster  # shared architecture, keeps train/inference in sync

SEQ_LEN = 12
TEST_SPLIT_FRACTION = 0.2
SEQ_FEATURE_COLS = [
    "Weekly_Sales_log", "Holiday_Flag", "Temperature", "Fuel_Price", "CPI",
    "Unemployment", "month_sin", "month_cos", "week_sin", "week_cos",
]
TARGET = "Weekly_Sales"

torch.manual_seed(42)
np.random.seed(42)


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


def build_frame():
    df = pd.read_csv(os.path.join(DATA_DIR, "Walmart.csv"))
    df = df.drop_duplicates()
    df["Date"] = pd.to_datetime(df["Date"], format="%d-%m-%Y")
    df = df.sort_values(["Store", "Date"]).reset_index(drop=True)
    df["Weekly_Sales"] = df["Weekly_Sales"].clip(lower=0)
    df["Weekly_Sales_log"] = np.log1p(df["Weekly_Sales"])
    df["month"] = df["Date"].dt.month
    df["week_of_year"] = df["Date"].dt.isocalendar().week.astype(int)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["week_sin"] = np.sin(2 * np.pi * df["week_of_year"] / 52)
    df["week_cos"] = np.cos(2 * np.pi * df["week_of_year"] / 52)
    return df


def make_sequences(df, split_date):
    """Sliding windows of SEQ_LEN weeks -> next week's Weekly_Sales, per store."""
    X_train, y_train, X_test, y_test, test_meta = [], [], [], [], []
    for store, grp in df.groupby("Store"):
        grp = grp.sort_values("Date").reset_index(drop=True)
        vals = grp[SEQ_FEATURE_COLS].values.astype(np.float32)
        sales = grp["Weekly_Sales"].values.astype(np.float32)
        dates = grp["Date"].values
        for i in range(SEQ_LEN, len(grp)):
            seq = vals[i - SEQ_LEN:i]
            target = sales[i]
            target_date = pd.Timestamp(dates[i])
            if target_date < split_date:
                X_train.append(seq)
                y_train.append(target)
            else:
                X_test.append(seq)
                y_test.append(target)
                test_meta.append((int(store), target_date.strftime("%Y-%m-%d")))
    return (np.array(X_train), np.array(y_train),
            np.array(X_test), np.array(y_test), test_meta)


def run():
    os.makedirs(MODEL_DIR, exist_ok=True)
    df = build_frame()

    min_date, max_date = df["Date"].min(), df["Date"].max()
    split_date = min_date + (max_date - min_date) * (1 - TEST_SPLIT_FRACTION)

    X_train, y_train, X_test, y_test, test_meta = make_sequences(df, split_date)
    print(f"Train sequences: {X_train.shape}, Test sequences: {X_test.shape}")

    # Standardize continuous features (index 0 = log sales, 2..5 = exogenous)
    scale_idx = [0, 2, 3, 4, 5]
    flat_train = X_train.reshape(-1, X_train.shape[-1])
    mean = flat_train[:, scale_idx].mean(axis=0)
    std = flat_train[:, scale_idx].std(axis=0)
    std[std == 0] = 1.0

    def scale(X):
        X = X.copy()
        X[:, :, scale_idx] = (X[:, :, scale_idx] - mean) / std
        return X

    X_train_s = scale(X_train)
    X_test_s = scale(X_test)

    # Scale target (log sales) using same mean/std as the log-sales feature channel
    y_mean, y_std = mean[0], std[0]
    y_train_log = np.log1p(y_train)
    y_train_s = (y_train_log - y_mean) / y_std

    device = torch.device("cpu")
    model = LSTMForecaster(n_features=len(SEQ_FEATURE_COLS)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    loss_fn = nn.MSELoss()

    ds = TensorDataset(torch.tensor(X_train_s), torch.tensor(y_train_s))
    dl = DataLoader(ds, batch_size=64, shuffle=True)

    EPOCHS = 60
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0.0
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            total_loss += loss.item() * xb.size(0)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"epoch {epoch+1}/{EPOCHS}  train_mse(scaled)={total_loss/len(ds):.4f}")

    model.eval()
    with torch.no_grad():
        pred_s = model(torch.tensor(X_test_s)).numpy()
    pred_log = pred_s * y_std + y_mean
    pred_sales = np.clip(np.expm1(pred_log), 0, None)

    lstm_results = evaluate(y_test, pred_sales, "LSTM")
    print(lstm_results)

    torch.save(model.state_dict(), os.path.join(MODEL_DIR, "lstm_model.pt"))

    with open(os.path.join(MODEL_DIR, "lstm_meta.json"), "w") as f:
        json.dump({
            "seq_len": SEQ_LEN,
            "feature_cols": SEQ_FEATURE_COLS,
            "scale_idx": scale_idx,
            "mean": mean.tolist(),
            "std": std.tolist(),
            "y_mean": float(y_mean),
            "y_std": float(y_std),
            "hidden_size": 64,
            "num_layers": 2,
            "dropout": 0.2,
        }, f, indent=2)

    # Row-level predictions on the test set (for store lookups / accuracy charts)
    row_preds = pd.DataFrame(test_meta, columns=["Store", "Date"])
    row_preds["actual"] = y_test
    row_preds["predicted"] = pred_sales
    row_preds.to_csv(os.path.join(MODEL_DIR, "lstm_row_predictions.csv"), index=False)

    # Append/replace the LSTM row in model_comparison.csv (written by train.py)
    comp_path = os.path.join(MODEL_DIR, "model_comparison.csv")
    if os.path.exists(comp_path):
        comp = pd.read_csv(comp_path)
        comp = comp[comp["Model"] != "LSTM"]
        comp = pd.concat([comp, pd.DataFrame([lstm_results])], ignore_index=True)
    else:
        comp = pd.DataFrame([lstm_results])
    comp.to_csv(comp_path, index=False)

    print("Saved LSTM model + metrics to", MODEL_DIR)


if __name__ == "__main__":
    run()
