"""Mean observed vs predicted count by hour of day on the 2025 test set."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

cols = ["split", "hour", "is_holiday", "cyclist_count", "predicted"]
df = pd.read_parquet(ROOT / "data" / "predictions.parquet", columns=cols)
df = df[df["split"] == "test"]

fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), sharey=True)
for ax, flag, title in [(axes[0], 0, "Working days"), (axes[1], 1, "Weekends and holidays")]:
    g = df[df["is_holiday"] == flag].groupby("hour")[["cyclist_count", "predicted"]].mean()
    ax.plot(g.index, g["cyclist_count"], color="black", label="observed")
    ax.plot(g.index, g["predicted"], color="tab:blue", linestyle="--", label="predicted")
    ax.set_title(title)
    ax.set_xlabel("hour of day")
axes[0].set_ylabel("cyclists per hour, per direction")
axes[0].legend(frameon=False)
fig.tight_layout()
fig.savefig(ROOT / "figures" / "hourly_profile_test.png", dpi=150)
