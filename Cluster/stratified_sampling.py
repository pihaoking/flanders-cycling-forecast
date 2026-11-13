# This script draws a stratified sample of 1500 rows per cluster (4500 total) from the test set into test_sample.
# To use it, import it with `from stratified_sampling import test_sample`.

import sys
import pandas as pd

sys.path.append("../src")
from split import temporal_split

df = pd.read_parquet("../data/panel_merged.parquet")

train, val, test = temporal_split(df)

clusters = pd.read_csv("station_profiles_clustered.csv", usecols=["site_id", "cluster"])
test = test.merge(clusters, on="site_id", how="left")

test_sample = (
    test.groupby("cluster", group_keys=False)
    .apply(lambda g: g.sample(n=1500, random_state=42))
    .reset_index(drop=True)
)
