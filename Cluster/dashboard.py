from shiny import App, reactive, render, ui
from shinywidgets import output_widget, render_widget
import plotly.graph_objects as go
import pandas as pd
from pathlib import Path


### Data ###

HERE = Path(__file__).parent
df = pd.read_csv(HERE / "station_profiles_clustered.csv")

FEATURES = [
    "avg_hourly_volume",
    "peak_ratio",
    "offday_workday_ratio",
    "weather_sensitivity",
    "seasonal_amplitude",
]
FEATURE_LABELS = [
    "Avg Hourly\nVolume",
    "Peak\nRatio",
    "Off-day /\nWorkday",
    "Weather\nSensitivity",
    "Seasonal\nAmplitude",
]

# Per-feature min-max normalisation for radar chart
feat_min = df[FEATURES].min()
feat_max = df[FEATURES].max()
df_norm = (df[FEATURES] - feat_min) / (feat_max - feat_min)

# Colours per cluster, aligned with the report's muted academic palette.
CLUSTER_LINE  = {0: "#2F6F9F", 1: "#D98B32", 2: "#5F9E6E"}
CLUSTER_FILL  = {
    0: "rgba(47,111,159,0.25)",
    1: "rgba(217,139,50,0.25)",
    2: "rgba(95,158,110,0.25)",
}
CLUSTER_NAMES = {0: "Peripheral & Local Connectors (Low Volume)", 1: "Urban Hubs (High Volume)", 2: "Leisure & Tourism"}


### UI ###

app_ui = ui.page_fluid(
    ui.tags.style("""
        body { background: #f4f6f9; font-family: sans-serif; }
        h2   { margin: 18px 0 6px; }
        .card-header { font-weight: 600; }
        .hint { color: #999; font-size: 13px; margin-top: 6px; }
    """),
    ui.h2("Cycling Traffic Station Clusters – Flanders"),
    ui.p("Click a station on the map to inspect its profile.", class_="hint"),
    ui.layout_columns(
        # Map panel
        ui.card(
            ui.card_header("Station Map"),
            output_widget("map_plot"),
            full_screen=True,
        ),
        # Profile panel 
        ui.card(
            ui.card_header("Station Profile"),
            ui.output_ui("site_header"),
            output_widget("radar_plot"),
            ui.output_ui("feature_table"),
        ),
        col_widths=[7, 5],
    ),
)


### Server ###

def server(input, output, session):
    selected_id = reactive.Value(None)

    # Map with clickable stations
    @render_widget
    def map_plot():
        fig = go.FigureWidget()

        for cid in sorted(df["cluster"].unique()):
            sub = df[df["cluster"] == cid]
            fig.add_trace(
                go.Scattermapbox(
                    lat=sub["lat"],
                    lon=sub["long"],
                    mode="markers",
                    marker=dict(size=10, color=CLUSTER_LINE[cid], opacity=0.85),
                    name=CLUSTER_NAMES[cid],
                    customdata=sub[["site_id", "naam"]].values,
                    hovertemplate="<b>%{customdata[0]}</b> – %{customdata[1]}<extra>"
                    + CLUSTER_NAMES[cid]
                    + "</extra>",
                )
            )

        fig.update_layout(
            mapbox=dict(
                style="open-street-map",
                center=dict(lat=50.98, lon=4.50),
                zoom=7.2,
            ),
            margin=dict(r=0, t=0, l=0, b=0),
            height=500,
            legend=dict(
                title="",
                bgcolor="rgba(255,255,255,0.85)",
                bordercolor="#ccc",
                borderwidth=1,
                x=0.01,
                y=0.99,
                font=dict(size=11),
            ),
        )

        # Register click callback for every trace
        def on_click(trace, points, _state):
            if points.point_inds:
                idx = points.point_inds[0]
                selected_id.set(int(trace.customdata[idx][0]))

        for trace in fig.data:
            trace.on_click(on_click)

        return fig

    # Site header with name and id
    @render.ui
    def site_header():
        sid = selected_id.get()
        if sid is None:
            return ui.p("← Select a station on the map.", class_="hint")
        row = df[df["site_id"] == sid].iloc[0]
        cid = int(row["cluster"])
        color = CLUSTER_LINE[cid]
        return ui.div(
            ui.h4(f"{row['site_id']} – {row['naam']}", style=f"color:{color}; margin:8px 0 2px;"),
            ui.p(CLUSTER_NAMES[cid], style="color:#666; font-size:13px; margin:0 0 4px;"),
        )

    # Radar chart of normalised feature values, coloured by cluster
    @render_widget
    def radar_plot():
        sid = selected_id.get()

        if sid is None:
            fig = go.Figure()
            fig.add_annotation(
                text="No station selected",
                showarrow=False,
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                font=dict(size=13, color="#bbb"),
            )
            fig.update_layout(
                height=300,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
            )
            return fig

        norm_vals = df_norm.loc[df["site_id"] == sid, FEATURES].iloc[0].tolist()
        cid = int(df[df["site_id"] == sid]["cluster"].iloc[0])

        # Close the loop
        r_vals = norm_vals + [norm_vals[0]]
        theta_vals = FEATURE_LABELS + [FEATURE_LABELS[0]]

        fig = go.Figure(
            go.Scatterpolar(
                r=r_vals,
                theta=theta_vals,
                fill="toself",
                fillcolor=CLUSTER_FILL[cid],
                line=dict(color=CLUSTER_LINE[cid], width=2.5),
                mode="lines+markers",
                marker=dict(size=7, color=CLUSTER_LINE[cid]),
            )
        )
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1],
                    tickfont=dict(size=9),
                    tickvals=[0.25, 0.5, 0.75, 1.0],
                ),
                angularaxis=dict(tickfont=dict(size=10)),
            ),
            showlegend=False,
            height=320,
            margin=dict(t=20, b=20, l=50, r=50),
            paper_bgcolor="rgba(0,0,0,0)",
        )
        return fig

    # Feature table 
    @render.ui
    def feature_table():
        sid = selected_id.get()
        if sid is None:
            return ui.div()

        row = df[df["site_id"] == sid].iloc[0]
        feat_display = [
            "avg_hourly_volume",
            "peak_ratio",
            "offday_workday_ratio",
            "weather_sensitivity",
            "seasonal_amplitude",
        ]
        label_display = [
            "Avg hourly volume",
            "Peak ratio",
            "Off-day / workday ratio",
            "Weather sensitivity",
            "Seasonal amplitude",
        ]

        rows = "".join(
            f"<tr><td style='padding:4px 10px;color:#555;'>{lbl}</td>"
            f"<td style='padding:4px 10px;text-align:right;font-weight:600;'>{row[feat]:.4f}</td></tr>"
            for lbl, feat in zip(label_display, feat_display)
        )

        table_html = (
            "<table style='width:100%;border-collapse:collapse;font-size:13px;'>"
            "<thead><tr>"
            "<th style='padding:6px 10px;background:#f0f0f0;text-align:left;'>Feature</th>"
            "<th style='padding:6px 10px;background:#f0f0f0;text-align:right;'>Value (raw)</th>"
            "</tr></thead>"
            f"<tbody>{rows}</tbody>"
            "</table>"
        )
        return ui.HTML(table_html)


app = App(app_ui, server)
