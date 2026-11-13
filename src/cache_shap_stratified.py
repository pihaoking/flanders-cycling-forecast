"""Cache RF SHAP values for a cluster-stratified test sample."""

import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import shap
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestRegressor

from split import temporal_split

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = DATA / "shap_values_stratified.pkl"
SEED = 42
N_PER_CLUSTER = 1500
SHAP_JOBS = 4

FCOLS = [
    "site_id", "hour", "day_of_week", "month",
    "is_holiday", "days_to_nearest_holiday", "is_in",
    "temp_c", "humidity", "dewpoint_c", "precip_mm", "rain_mm", "snowfall_cm",
    "windspeed_kmh", "winddir_deg", "windgust_kmh",
    "lat", "long", "road_bearing", "wind_impact",
    "lag_24h_count",
]


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


def shap_for_chunk(model, chunk):
    explainer = shap.TreeExplainer(model)
    return explainer.shap_values(chunk)


def main():
    print("loading panel", flush=True)
    panel = pd.read_parquet(DATA / "panel_merged.parquet")
    panel = add_lag_24h(panel)
    train, _, test = temporal_split(panel)

    print("building stratified sample", flush=True)
    clusters = pd.read_csv(
        ROOT / "Cluster" / "station_profiles_clustered.csv",
        usecols=["site_id", "cluster"],
    )
    test = test.merge(clusters, on="site_id", how="left")
    test_sample = (
        test.dropna(subset=["cluster"])
        .groupby("cluster", group_keys=False)
        .sample(n=N_PER_CLUSTER, random_state=SEED)
        .reset_index(drop=True)
    )
    print(f"  shape: {test_sample.shape}", flush=True)
    print(f"  per-cluster counts: {test_sample['cluster'].value_counts().to_dict()}", flush=True)

    tr = prep(train)
    X_tr, y_tr = tr[FCOLS], tr["cyclist_count"]

    print("training RF", flush=True)
    t0 = time.time()
    model = RandomForestRegressor(
        n_estimators=100, max_depth=20, max_samples=0.3,
        random_state=SEED, n_jobs=-1,
    )
    model.fit(X_tr, y_tr)
    print(f"  done in {time.time() - t0:.1f}s", flush=True)

    sample_prep = prep(test_sample)
    print(f"sample after prep: {sample_prep.shape}", flush=True)
    X_sample = sample_prep[FCOLS]

    print("computing SHAP", flush=True)
    explainer = shap.TreeExplainer(model)
    t0 = time.time()
    chunks = [c for c in np.array_split(X_sample, SHAP_JOBS) if len(c)]
    shap_values = np.vstack(
        Parallel(n_jobs=SHAP_JOBS)(
            delayed(shap_for_chunk)(model, chunk) for chunk in chunks
        )
    )
    print(f"  done in {time.time() - t0:.1f}s "
          f"({(time.time() - t0) / 60:.1f} min)", flush=True)

    payload = {
        "sample": sample_prep,
        "shap_values": shap_values,
        "expected_value": explainer.expected_value,
        "n_samples": len(X_sample),
        "n_per_cluster": N_PER_CLUSTER,
        "seed": SEED,
        "feature_columns": FCOLS,
        "clusters": sample_prep["cluster"].values,
    }
    with open(OUT, "wb") as f:
        pickle.dump(payload, f)
    print(f"saved {OUT}  ({OUT.stat().st_size / 1e6:.2f} MB)", flush=True)


if __name__ == "__main__":
    main()
