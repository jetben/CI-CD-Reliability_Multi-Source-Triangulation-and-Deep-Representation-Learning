import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.calibration import calibration_curve

import torch
import torch.nn as nn
import torch.optim as optim

DATA_PATH = Path("output") / "dataset_supervised.csv"
OUT_DIR = Path("paper_outputs")
OUT_DIR.mkdir(exist_ok=True)

LABEL_COL = "y_degrade"


def time_split(df: pd.DataFrame, time_col="month"):
    months = sorted(df[time_col].dropna().unique())
    train_months = set(months[:2])
    val_months = set(months[2:3])
    test_months = set(months[3:4])
    return (
        df[df[time_col].isin(train_months)].copy(),
        df[df[time_col].isin(val_months)].copy(),
        df[df[time_col].isin(test_months)].copy(),
        months
    )


def get_numeric_feature_cols(df: pd.DataFrame, exclude):
    exclude = set(exclude)
    cols = []
    for c in df.columns:
        if c in exclude:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return cols


def best_threshold_by_f1(y_true, y_prob):
    best_thr, best_f1 = 0.5, -1.0
    for thr in np.linspace(0.05, 0.95, 37):
        f1 = f1_score(y_true, (y_prob >= thr).astype(int), zero_division=0)
        if f1 > best_f1:
            best_thr, best_f1 = float(thr), float(f1)
    return best_thr, best_f1


# -----------------------------
# PyTorch Deep Model
# -----------------------------
class DeepModel(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.30),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x)


def train_deep_model(X_train, y_train, X_val, y_val, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = DeepModel(X_train.shape[1])
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    pos = max(float(y_train.sum()), 1.0)
    neg = max(float(len(y_train) - y_train.sum()), 1.0)
    pos_weight = torch.tensor([neg / pos], dtype=torch.float32)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    Xtr = torch.tensor(X_train, dtype=torch.float32)
    ytr = torch.tensor(y_train, dtype=torch.float32)
    Xva = torch.tensor(X_val, dtype=torch.float32)
    yva = torch.tensor(y_val, dtype=torch.float32)

    best_state = None
    best_val_auc = -1
    patience = 20
    wait = 0

    for epoch in range(200):
        model.train()
        optimizer.zero_grad()
        logits = model(Xtr).squeeze()
        loss = criterion(logits, ytr)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_probs = torch.sigmoid(model(Xva).squeeze()).cpu().numpy()
            if len(np.unique(y_val)) > 1:
                val_auc = roc_auc_score(y_val, val_probs)
            else:
                val_auc = np.nan

        if np.isfinite(val_auc) and val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    model.load_state_dict(best_state)
    return model


def deep_predict_proba(model, X):
    model.eval()
    with torch.no_grad():
        X_t = torch.tensor(X, dtype=torch.float32)
        probs = torch.sigmoid(model(X_t).squeeze()).cpu().numpy()
    return probs


def plot_save(fig, name):
    png = OUT_DIR / f"{name}.png"
    pdf = OUT_DIR / f"{name}.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", png.resolve())
    print("Saved:", pdf.resolve())


def main():
    df = pd.read_csv(DATA_PATH)
    df["month"] = pd.to_datetime(df["month"], errors="coerce")

    exclude = [
        "repository_name", "month", "month_next",
        "p_fail_EB_next", "q_thr", "runs_next",
        LABEL_COL
    ]
    feature_cols = get_numeric_feature_cols(df, exclude)

    df_train, df_val, df_test, months = time_split(df)

    # Impute + scale
    med = df_train[feature_cols].median(numeric_only=True)
    for d in [df_train, df_val, df_test]:
        d[feature_cols] = d[feature_cols].fillna(med)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(df_train[feature_cols].values)
    X_val = scaler.transform(df_val[feature_cols].values)
    X_test = scaler.transform(df_test[feature_cols].values)

    y_train = df_train[LABEL_COL].astype(int).values
    y_val = df_val[LABEL_COL].astype(int).values
    y_test = df_test[LABEL_COL].astype(int).values

    # Train models once and keep test probabilities fixed
    probs: Dict[str, np.ndarray] = {}

    # Heuristic
    probs["Heuristic(p_fail_EB)"] = df_test["p_fail_EB"].values

    # LogReg
    lr = LogisticRegression(max_iter=4000, class_weight="balanced")
    lr.fit(X_train, y_train)
    probs["LogReg(balanced)"] = lr.predict_proba(X_test)[:, 1]

    # RF
    rf = RandomForestClassifier(
        n_estimators=500,
        min_samples_leaf=5,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    probs["RandForest(balanced)"] = rf.predict_proba(X_test)[:, 1]

    # Deep
    deep = train_deep_model(X_train, y_train, X_val, y_val, seed=42)
    probs["DeepModel(DNN 64-32)"] = deep_predict_proba(deep, X_test)

    # Thresholds tuned on VAL (to be consistent, tune using val probabilities)
    thr = {}
    thr["Heuristic(p_fail_EB)"] = best_threshold_by_f1(y_val, df_val["p_fail_EB"].values)[0]
    thr["LogReg(balanced)"] = best_threshold_by_f1(y_val, lr.predict_proba(X_val)[:, 1])[0]
    thr["RandForest(balanced)"] = best_threshold_by_f1(y_val, rf.predict_proba(X_val)[:, 1])[0]
    thr["DeepModel(DNN 64-32)"] = best_threshold_by_f1(y_val, deep_predict_proba(deep, X_val))[0]

    # -----------------------------
    # Bootstrap distributions on TEST
    # -----------------------------
    B = 1000
    rng = np.random.default_rng(42)
    n = len(y_test)
    idxs = rng.integers(0, n, size=(B, n))  # bootstrap indices

    auc_dist = {k: [] for k in probs.keys()}
    f1_dist = {k: [] for k in probs.keys()}

    for b in range(B):
        ids = idxs[b]
        yb = y_test[ids]
        # ensure both classes appear; otherwise skip to avoid undefined AUC
        if len(np.unique(yb)) < 2:
            continue

        for name, p in probs.items():
            pb = p[ids]
            auc_dist[name].append(roc_auc_score(yb, pb))

            ypred = (pb >= thr[name]).astype(int)
            f1_dist[name].append(f1_score(yb, ypred, zero_division=0))

    # -----------------------------
    # Boxplot AUC
    # -----------------------------
    labels = list(auc_dist.keys())
    data_auc = [auc_dist[k] for k in labels]

    fig = plt.figure(figsize=(8, 4))
    ax = fig.add_subplot(111)
    ax.boxplot(data_auc, labels=labels, showfliers=False)
    ax.set_ylabel("AUC (bootstrap over test set)")
    ax.set_title("Model Comparison: AUC Distribution (Bootstrap)")
    plt.xticks(rotation=20, ha="right")
    plot_save(fig, "fig_boxplot_auc_bootstrap")

    # -----------------------------
    # Boxplot F1
    # -----------------------------
    data_f1 = [f1_dist[k] for k in labels]

    fig = plt.figure(figsize=(8, 4))
    ax = fig.add_subplot(111)
    ax.boxplot(data_f1, labels=labels, showfliers=False)
    ax.set_ylabel("F1 (bootstrap over test set)")
    ax.set_title("Model Comparison: F1 Distribution (Bootstrap)")
    plt.xticks(rotation=20, ha="right")
    plot_save(fig, "fig_boxplot_f1_bootstrap")

    # Optional: save summary stats table
    rows = []
    for name in labels:
        rows.append({
            "Model": name,
            "AUC_median": float(np.median(auc_dist[name])),
            "AUC_IQR": float(np.percentile(auc_dist[name], 75) - np.percentile(auc_dist[name], 25)),
            "F1_median": float(np.median(f1_dist[name])),
            "F1_IQR": float(np.percentile(f1_dist[name], 75) - np.percentile(f1_dist[name], 25)),
            "Boot_samples": len(auc_dist[name]),
        })
    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "table_bootstrap_boxplot_summary.csv", index=False)
    print("Saved:", (OUT_DIR / "table_bootstrap_boxplot_summary.csv").resolve())

    print("\nDONE ✅ Bootstrap boxplots created in:", OUT_DIR.resolve())


if __name__ == "__main__":
    main()
