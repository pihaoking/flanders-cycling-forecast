# Cluster Folder

This folder contains the clustering analysis and interactive dashboard for the cycling traffic stations in Flanders.

## Files

### `clustering.ipynb`
Jupyter notebook that performs the cluster analysis on the station feature profiles. It reads the engineered features (avg hourly volume, peak ratio, off-day/workday ratio, weather sensitivity, seasonal amplitude), applies K-Means clustering, evaluates the optimal number of clusters, and writes the results to `station_profiles_clustered.csv`. 

### `station_profiles_clustered.csv`
Output of `clustering.ipynb`. Each row is one monitoring station with its five engineered features, assigned cluster label, and geographic coordinates (longitude, latitude). This file will be used by the dashboard.

| Column | Description |
|---|---|
| `site_id` | Unique station identifier |
| `avg_hourly_volume` | Mean hourly cyclist count |
| `peak_ratio` | Share of counts in peak hours |
| `offday_workday_ratio` | Off-day vs. workday mean hourly counts |
| `weather_sensitivity` | Rainy day vs dry day mean hourly counts |
| `seasonal_amplitude` | Summer vs winter mean hourly counts |
| `cluster` | Assigned cluster (0, 1, 2) |
| `long` / `lat` | Station coordinates |

### `dashboard.py`
Interactive Shiny for Python dashboard. Displays all valid stations in 2023 on an OpenStreetMap of Flanders, coloured by cluster. Clicking a station opens its profile in the right panel: a radar chart of the five normalised features and a table of raw values.

**To run:**
```bash
shiny run Cluster/dashboard.py
```
Then open `http://127.0.0.1:8000` in your browser.

**Dependencies:** `shiny`, `shinywidgets`, `plotly`, `pandas`

## Clusters

| ID | Name |
|---|---|
| 0 | Peripheral & Local Connectors (Low Volume) |
| 1 | Urban Hubs (High Volume) |
| 2 | Leisure & Tourism |
