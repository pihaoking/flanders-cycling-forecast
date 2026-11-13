"""Run model comparison and residual diagnostics."""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
from scipy.stats import skew, kurtosis

from split import temporal_split


ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)

FCOLS = [
    "site_id", "hour", "day_of_week", "month",
    "is_holiday", "days_to_nearest_holiday", "is_in",
    "temp_c", "humidity", "dewpoint_c", "precip_mm", "rain_mm", "snowfall_cm",
    "windspeed_kmh", "winddir_deg", "windgust_kmh",
    "lat", "long", "road_bearing", "wind_impact",
    "lag_24h_count",
]

RF_KW = dict(n_estimators=100, max_depth=20, max_samples=0.3,
             random_state=42, n_jobs=-1)
XGB_KW = dict(n_estimators=1051, learning_rate=0.05, max_depth=6,
              random_state=42, n_jobs=-1)


def load_filtered():
    panel = pd.read_parquet(ROOT / "data" / "panel_merged.parquet")
    panel = add_lag_24h(panel)
    train, val, test = temporal_split(panel)
    train_sites = set(train["site_id"].unique())
    val = val[val["site_id"].isin(train_sites)]
    test = test[test["site_id"].isin(train_sites)]
    return train, val, test


def add_lag_24h(panel):
    lag_src = panel[["site_id", "richting", "datetime", "cyclist_count"]].copy()
    lag_src["datetime"] = lag_src["datetime"] + pd.Timedelta(days=1)
    lag_src = lag_src.rename(columns={"cyclist_count": "lag_24h_count"})
    out = panel.merge(lag_src, on=["site_id", "richting", "datetime"], how="left")
    out["lag_24h_count"] = out["lag_24h_count"].fillna(0)
    return out


def prep(df):
    out = df.dropna(subset=["cyclist_count"]).copy()
    out["road_bearing"] = out["road_bearing"].fillna(-1)
    out["wind_impact"] = out["wind_impact"].fillna(0)
    return out


def fit_rf(X, y):
    m = RandomForestRegressor(**RF_KW)
    m.fit(X, y)
    return m


def fit_xgb(X, y):
    m = XGBRegressor(**XGB_KW)
    m.fit(X, y)
    return m


def cv_diagnostics(X_tr, y_tr):
    print("=" * 70)
    print("1. Train-internal 5-fold CV (overfitting vs distribution shift)")
    print("=" * 70)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    rf_scores, xgb_scores = [], []
    for fold, (i_tr, i_va) in enumerate(kf.split(X_tr), 1):
        rf = fit_rf(X_tr.iloc[i_tr], y_tr.iloc[i_tr])
        xgb = fit_xgb(X_tr.iloc[i_tr], y_tr.iloc[i_tr])
        rf_r2 = r2_score(y_tr.iloc[i_va], rf.predict(X_tr.iloc[i_va]))
        xgb_r2 = r2_score(y_tr.iloc[i_va], xgb.predict(X_tr.iloc[i_va]))
        rf_scores.append(rf_r2)
        xgb_scores.append(xgb_r2)
        print(f"  fold {fold}: RF={rf_r2:+.3f}   XGB={xgb_r2:+.3f}")
    print(f"  mean   : RF={np.mean(rf_scores):+.3f} (sd {np.std(rf_scores):.3f})  "
          f"XGB={np.mean(xgb_scores):+.3f} (sd {np.std(xgb_scores):.3f})")


def segment_rmse(test_df, y_te, rf_pred, xgb_pred):
    print()
    print("=" * 70)
    print("2. Per-segment RMSE on test")
    print("=" * 70)
    df = pd.DataFrame({
        "y": y_te.values,
        "rf": rf_pred,
        "xgb": xgb_pred,
        "hour": test_df["hour"].values,
        "dow": test_df["day_of_week"].values,
        "month": test_df["month"].values,
        "is_weekend": test_df["is_weekend"].values,
        "site_id": test_df["site_id"].values,
    })
    df["rf_se"] = (df["rf"] - df["y"]) ** 2
    df["xgb_se"] = (df["xgb"] - df["y"]) ** 2

    site_means = test_df.groupby("site_id")["cyclist_count"].mean()
    site_q = pd.qcut(site_means, 4, labels=["Q1 low", "Q2", "Q3", "Q4 high"])
    df["site_q"] = df["site_id"].map(site_q.to_dict()).values

    def rmse_table(group_col, label):
        g = df.groupby(group_col).agg(
            n=("y", "size"),
            rf_rmse=("rf_se", lambda s: np.sqrt(s.mean())),
            xgb_rmse=("xgb_se", lambda s: np.sqrt(s.mean())),
        ).round(2)
        g["winner"] = np.where(g["rf_rmse"] < g["xgb_rmse"], "RF", "XGB")
        print(f"\n  by {label}:")
        print(g.to_string())

    rmse_table("hour", "hour of day")
    rmse_table("dow", "day of week (0 = Mon)")
    rmse_table("month", "month")
    rmse_table("is_weekend", "weekend (1 = Sat/Sun)")
    rmse_table("site_q", "station volume quartile")


def residual_diagnostics(y_te, rf_pred, xgb_pred):
    print()
    print("=" * 70)
    print("3. Residual diagnostics on test")
    print("=" * 70)
    rf_res = y_te.values - rf_pred
    xgb_res = y_te.values - xgb_pred
    rows = []
    for name, r in [("RF", rf_res), ("XGB", xgb_res)]:
        rows.append({
            "model": name,
            "mean (bias)": r.mean(),
            "std": r.std(),
            "skew": skew(r),
            "kurtosis": kurtosis(r),
            "q05": np.quantile(r, 0.05),
            "q50": np.quantile(r, 0.50),
            "q95": np.quantile(r, 0.95),
        })
    print(pd.DataFrame(rows).round(3).to_string(index=False))

    plt.rcParams.update({
        'font.size': 11,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': True,
        'grid.color': '#E0E0E0',
        'grid.linestyle': '--',
        'grid.alpha': 0.7,
        'axes.facecolor': 'white',
        'figure.facecolor': 'white'
    })

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    bins = np.linspace(-100, 100, 80)
    axes[0, 0].hist(np.clip(rf_res, -100, 100), bins=bins, alpha=0.85,
                    label="RF", color="#2F6F9F", edgecolor="white", linewidth=0.8)
    axes[0, 0].hist(np.clip(xgb_res, -100, 100), bins=bins, alpha=0.85,
                    label="XGB", color="#D98B32", edgecolor="white", linewidth=0.8)
    axes[0, 0].axvline(0, color="black", lw=0.5)
    axes[0, 0].set_xlabel("Residual (actual - predicted)")
    axes[0, 0].set_title("Residual distribution (clipped to +/- 100)")
    axes[0, 0].legend()

    p95 = int(np.percentile(y_te, 95))
    axes[0, 1].scatter(y_te.iloc[:50000], rf_pred[:50000],
                       alpha=0.05, s=3, color="#2F6F9F")
    axes[0, 1].plot([0, p95], [0, p95], linestyle="--", color="#C95D63", lw=1)
    axes[0, 1].set_xlim(0, p95); axes[0, 1].set_ylim(0, p95)
    axes[0, 1].set_xlabel("Actual"); axes[0, 1].set_ylabel("RF predicted")
    axes[0, 1].set_title("RF: actual vs predicted")

    axes[1, 0].scatter(y_te.iloc[:50000], xgb_pred[:50000],
                       alpha=0.05, s=3, color="#D98B32")
    axes[1, 0].plot([0, p95], [0, p95], linestyle="--", color="#C95D63", lw=1)
    axes[1, 0].set_xlim(0, p95); axes[1, 0].set_ylim(0, p95)
    axes[1, 0].set_xlabel("Actual"); axes[1, 0].set_ylabel("XGB predicted")
    axes[1, 0].set_title("XGBoost: actual vs predicted")

    axes[1, 1].scatter(rf_pred[:50000], rf_res[:50000],
                       alpha=0.05, s=3, color="#2F6F9F", label="RF")
    axes[1, 1].scatter(xgb_pred[:50000], xgb_res[:50000],
                       alpha=0.05, s=3, color="#D98B32", label="XGB")
    axes[1, 1].axhline(0, color="black", lw=0.5)
    axes[1, 1].set_xlim(0, p95); axes[1, 1].set_ylim(-150, 150)
    axes[1, 1].set_xlabel("Predicted"); axes[1, 1].set_ylabel("Residual")
    axes[1, 1].set_title("Residual vs predicted (heteroscedasticity check)")
    axes[1, 1].legend()

    plt.tight_layout()
    out = FIG / "model_diagnostics.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    print(f"\n  saved {out}")


def main():
    print("loading panel and filtering to train stations only")
    train, val, test = load_filtered()
    tr, te = prep(train), prep(test)
    X_tr, y_tr = tr[FCOLS], tr["cyclist_count"]
    X_te, y_te = te[FCOLS], te["cyclist_count"]
    print(f"  train={len(X_tr):,}  test={len(X_te):,}\n")

    cv_diagnostics(X_tr, y_tr)

    print("\nfitting full models on train for test-set diagnostics")
    rf_full = fit_rf(X_tr, y_tr)
    xgb_full = fit_xgb(X_tr, y_tr)
    rf_pred = np.clip(rf_full.predict(X_te), 0, None)
    xgb_pred = np.clip(xgb_full.predict(X_te), 0, None)

    print(f"  RF  in-sample R^2 = {r2_score(y_tr, rf_full.predict(X_tr)):+.3f}   "
          f"test R^2 = {r2_score(y_te, rf_pred):+.3f}")
    print(f"  XGB in-sample R^2 = {r2_score(y_tr, xgb_full.predict(X_tr)):+.3f}   "
          f"test R^2 = {r2_score(y_te, xgb_pred):+.3f}")

    segment_rmse(te, y_te, rf_pred, xgb_pred)
    residual_diagnostics(y_te, rf_pred, xgb_pred)


if __name__ == "__main__":
    main()
