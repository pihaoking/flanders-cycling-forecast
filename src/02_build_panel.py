"""
Build the merged hourly panel from raw inputs.

Inputs  (data/): sites.csv, richtingen.csv, road_bearings.csv,
                 data-YYYY-MM.csv, weather_multi_region.csv, event_features.csv
Output (data/):  panel_merged.parquet

Notes:
- Weather region name `far_east` must match weather_multi_region.csv exactly,
  otherwise the merge silently produces NaN.
- road_bearing is per-(site, richting): OUT direction is IN bearing + 180 (mod 360).
"""

from pathlib import Path
import glob
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def load_sites():
    return pd.read_csv(
        DATA / "sites.csv",
        header=None,
        names=["site_id", "site_nr", "long", "lat", "naam", "domein",
               "wegnr", "district", "gemeente", "interval", "datum_van"],
    )


def load_richtingen():
    return pd.read_csv(
        DATA / "richtingen.csv",
        header=None,
        names=["site_id", "richting", "direction_naam"],
    )


def load_counts():
    files = sorted(glob.glob(str(DATA / "data-*.csv")))
    print(f"loading {len(files)} monthly count files")
    df = pd.concat(
        (pd.read_csv(f, header=None,
                     names=["site_id", "richting", "type", "van", "tot", "aantal"])
         for f in files),
        ignore_index=True,
    )
    df = df[df["type"] == "FIETSERS"].copy()
    df["van"] = pd.to_datetime(df["van"])
    df["datetime"] = df["van"].dt.floor("h")
    return df


def assign_weather_region(sites, weather):
    centers = weather.groupby("region")[["region_lat", "region_lon"]].first()
    out = sites.copy()
    idx = cdist(out[["lat", "long"]].values,
                centers[["region_lat", "region_lon"]].values).argmin(axis=1)
    out["region"] = centers.index[idx]
    return out


def build_directional_bearings(rb_per_site, richtingen):
    """One row per (site_id, richting). OUT direction is IN bearing + 180 (mod 360)."""
    site_to_bearing = dict(zip(rb_per_site["site_id"], rb_per_site["road_bearing"]))
    rows = []
    for _, r in richtingen.iterrows():
        bearing = site_to_bearing.get(r["site_id"], np.nan)
        if r["richting"] == "OUT" and pd.notna(bearing):
            bearing = (bearing + 180.0) % 360.0
        rows.append({"site_id": r["site_id"],
                     "richting": r["richting"],
                     "road_bearing": bearing})
    return pd.DataFrame(rows)


def main():
    print("loading inputs")
    sites = load_sites()
    richtingen = load_richtingen()
    rb_per_site = pd.read_csv(DATA / "road_bearings.csv")
    weather = pd.read_csv(DATA / "weather_multi_region.csv")
    weather["datetime"] = pd.to_datetime(weather["datetime"])
    counts = load_counts()
    events = pd.read_csv(DATA / "event_features.csv")
    events["date"] = pd.to_datetime(events["date"]).dt.date
    print(f"sites={len(sites)} richtingen={len(richtingen)} "
          f"weather_rows={len(weather):,} count_rows={len(counts):,}")

    print("aggregating counts to hourly")
    hourly = (counts
              .groupby(["site_id", "richting", "datetime"], as_index=False)["aantal"]
              .sum()
              .rename(columns={"aantal": "cyclist_count"}))
    print(f"hourly_rows={len(hourly):,}")

    print("assigning sites to nearest weather region")
    sites_r = assign_weather_region(sites, weather)
    print(sites_r.groupby("region")["site_id"].count().to_string())

    print("expanding road bearings per direction (OUT = IN + 180)")
    rb_dir = build_directional_bearings(rb_per_site, richtingen)
    n_missing = int(rb_dir["road_bearing"].isna().sum())
    if n_missing:
        miss = rb_dir.loc[rb_dir["road_bearing"].isna(), "site_id"].unique().tolist()
        print(f"{n_missing} (site, richting) rows have no bearing, sites {miss}. "
              f"Re-run OSMnx for these and update road_bearings.csv.")

    print("merging")
    panel = (hourly
             .merge(sites_r[["site_id", "region", "lat", "long", "naam"]],
                    on="site_id", how="left")
             .merge(weather, on=["datetime", "region"], how="left")
             .merge(rb_dir, on=["site_id", "richting"], how="left"))

    panel["date"] = panel["datetime"].dt.date
    panel["hour"] = panel["datetime"].dt.hour
    panel["day_of_week"] = panel["datetime"].dt.dayofweek
    panel["month"] = panel["datetime"].dt.month
    panel["year"] = panel["datetime"].dt.year
    panel["is_weekend"] = (panel["day_of_week"] >= 5).astype(int)
    panel["is_in"] = (panel["richting"] == "IN").astype(int)
    panel["wind_impact"] = np.cos(np.radians(panel["winddir_deg"] - panel["road_bearing"]))

    panel = panel.merge(events, on="date", how="left")
    panel["is_holiday"] = (
        (panel["is_weekend"] == 1)
        | (panel["is_public_holiday"] == 1)
        | (panel["is_school_holiday"] == 1)
    ).astype(int)

    print("\nsanity checks")
    print(f"panel_rows={len(panel):,}")
    print(f"weather missing rate: {panel['temp_c'].isna().mean()*100:.2f}%")
    print(f"road_bearing missing rate: {panel['road_bearing'].isna().mean()*100:.2f}%")
    print(f"event missing rate: {panel['is_public_holiday'].isna().mean()*100:.2f}%")
    print("region distribution:")
    print(panel["region"].value_counts(dropna=False).to_string())

    out_path = DATA / "panel_merged.parquet"
    panel.to_parquet(out_path, index=False)
    print(f"\nsaved {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
