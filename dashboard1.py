from shiny import App, reactive, render, ui
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from shiny.ui import HTML
import pandas as pd
from pathlib import Path

# Shared muted, colourblind-aware report palette.
BLUE = "#2F6F9F"
LIGHT_BLUE = "#73A9CF"
ORANGE = "#D98B32"
RED = "#C95D63"
GREY = "#857A70"

# ---------------------------------------------------------------------------
# Data  
# ---------------------------------------------------------------------------
HERE = Path(__file__).parent

# Load pre-computed predictions (joined with actual counts)
pred_path = HERE / "data" / "predictions.parquet"
if pred_path.exists():
    df_panel = pd.read_parquet(pred_path)
    df_panel["datetime"] = pd.to_datetime(df_panel["datetime"])
    df_panel["date"] = df_panel["datetime"].dt.normalize()
    HAS_PREDICTIONS = True
else:
    df_panel = pd.read_parquet(HERE / "data" / "panel_merged.parquet")
    df_panel["date"] = pd.to_datetime(df_panel["datetime"]).dt.normalize()
    df_panel["datetime"] = pd.to_datetime(df_panel["datetime"])
    HAS_PREDICTIONS = False

# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
app_ui = ui.page_fluid(
    ui.tags.style("""
        body { background: #f4f6f9; font-family: sans-serif; }
        h2   { margin: 18px 0 6px; }
        .card-header { font-weight: 600; }
        .hint { color: #999; font-size: 13px; margin-top: 6px; }
    """),
    ui.h2("Cycling Traffic: Prediction and Anomaly Detection"),
    ui.layout_sidebar(
        ui.sidebar(
            ui.h4("Controls"),
        # Dropdown to select station
        ui.input_select("site_id", "Select station:", choices={str(s): f"Station {s}" for s in sorted(df_panel["site_id"].unique())}),
        # Date range for the timeline
        ui.input_date_range("dates", "Date Range", start="2023-01-01", end="2023-01-14"),
        ui.hr(),
        ui.h5("Alarm Configuration"),
        ui.p("Adjust sensitivity to flag unexpected traffic patterns.", class_="hint", style="font-size: 12px; color: #666;"),
        ui.input_slider("threshold", "Anomaly Threshold (Cyclists)", min=0, max=200, value=50, step=1),
    ui.hr(),
    ui.h5("Policy Simulation"),
    ui.input_switch("no_holiday", value=False, label="Simulate no holidays"),
        ),
        ui.card(
            ui.card_header("Predictive Line and Alarms"),
            ui.output_ui("prediction_plot"),
            ui.markdown("""
    **Legend:**
    * **Solid muted blue:** Actual counts.
    * **Dashed muted orange:** Predicted counts.
    * **Muted red 'X':** Automated alarm (prediction error exceeds threshold).
    * **Light blue bars:** Precipitation.
    """)
        ),
    )
)


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------
def server(input, output, session):

    @reactive.calc
    def filtered_data():
        sid = int(input.site_id())
        start_date, end_date = input.dates()
        mask = (
            (df_panel["site_id"] == sid) &
            (df_panel["date"].dt.date >= start_date) &
            (df_panel["date"].dt.date <= end_date)
        )
        return df_panel.loc[mask].sort_values("datetime").copy()
    
    @render.ui
    def prediction_plot():
        df = filtered_data()
        if df.empty:
            fig = go.Figure().add_annotation(text="No data found for selected criteria", showarrow=False)
            return ui.HTML(fig.to_html(full_html=False, include_plotlyjs='cdn'))

        # Base Prediction
        if HAS_PREDICTIONS and 'predicted' in df.columns:
            df['error'] = (df['cyclist_count'] - df['predicted']).abs()
            threshold = input.threshold()
            anomalies = df[df['error'] > threshold]
        else:
            anomalies = pd.DataFrame()

        # Build Figure
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        # Actual Line
        fig.add_trace(go.Scatter(x=df['datetime'], y=df['cyclist_count'], 
                                 name="Actual", line=dict(color=BLUE, width=2.5)))
        
        # Predicted Line
        if HAS_PREDICTIONS and 'predicted' in df.columns:
            fig.add_trace(go.Scatter(x=df['datetime'], y=df['predicted'],
                                     name="Predicted", line=dict(color=ORANGE, dash='dot')))

        # Alarm Markers
        fig.add_trace(go.Scatter(x=anomalies['datetime'], y=anomalies['cyclist_count'],
                                 mode='markers', name="ALARM",
                                 marker=dict(color=RED, size=12, symbol='x-open', line=dict(width=2))))

        # Weather Context (Secondary Axis)
        fig.add_trace(go.Bar(x=df['datetime'], y=df['precip_mm'], 
                             name="Rain (mm)", opacity=0.3, marker_color=LIGHT_BLUE),
                      secondary_y=True)

        # Formatting
        fig.update_layout(
            hovermode="x unified",
            template="plotly_white",
            height=600,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        fig.update_yaxes(title_text="<b>Cyclists per Hour</b>", secondary_y=False)
        fig.update_yaxes(title_text="<b>Precipitation (mm)</b>", secondary_y=True, range=[0, 20])
        
        return ui.HTML(fig.to_html(full_html=False, include_plotlyjs='cdn'))

app = App(app_ui, server)

