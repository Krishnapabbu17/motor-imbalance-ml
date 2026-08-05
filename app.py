"""Local web interface for motor-imbalance prediction."""

from __future__ import annotations

import os
from pathlib import Path

import gradio as gr
import pandas as pd

from src.predict import predict_trial


MODEL_PATH = Path("results/models/development_candidate.joblib")
CONFIG_PATH = Path("results/tables/locked_model_config.json")
THEME = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="amber",
    neutral_hue="slate",
    radius_size="lg",
)

CSS = """
:root {
  --motor-primary: #2563eb;
  --motor-foreground: #0f172a;
  --motor-muted: #475569;
  --motor-surface: #ffffff;
  --motor-border: #dbeafe;
}
.gradio-container {
  max-width: 1080px !important;
  margin: 0 auto !important;
  color: var(--motor-foreground);
}
.hero {
  padding: 28px 4px 14px;
}
.hero h1 {
  font-size: clamp(2rem, 5vw, 3.25rem);
  line-height: 1.08;
  letter-spacing: -0.035em;
  margin-bottom: 12px;
}
.hero p {
  color: var(--motor-muted);
  font-size: 1.05rem;
  line-height: 1.65;
  max-width: 720px;
}
.panel {
  background: var(--motor-surface);
  border: 1px solid var(--motor-border);
  border-radius: 18px;
  padding: 18px;
  box-shadow: 0 10px 30px rgba(37, 99, 235, 0.08);
}
#predict-button {
  min-height: 48px;
  font-weight: 650;
  transition: transform 180ms ease, box-shadow 180ms ease;
}
#predict-button:hover {
  transform: translateY(-1px);
  box-shadow: 0 8px 20px rgba(37, 99, 235, 0.22);
}
#prediction-result h2 {
  font-size: 1.55rem;
  margin-bottom: 6px;
}
#prediction-result strong {
  color: var(--motor-primary);
}
.privacy-note {
  color: var(--motor-muted);
  font-size: 0.92rem;
}
@media (prefers-reduced-motion: reduce) {
  #predict-button { transition: none; }
  #predict-button:hover { transform: none; }
}
"""


def format_prediction(
    result: dict[str, object],
) -> tuple[str, pd.DataFrame, pd.DataFrame]:
    """Convert a prediction dictionary into clear UI outputs."""
    predicted_mass = float(result["predicted_mass_g"])
    window_values = list(result["window_predictions_g"])
    result_markdown = (
        "## Prediction complete\n\n"
        f"Predicted imbalance: **{predicted_mass:.2f} g**\n\n"
        "The result is the majority vote across five non-overlapping two-second windows."
    )
    windows = pd.DataFrame(
        {
            "Window": [f"{index + 1} ({index * 2}-{(index + 1) * 2} s)" for index in range(5)],
            "Predicted imbalance (g)": [f"{float(value):.2f}" for value in window_values],
        }
    )
    scores = pd.DataFrame(
        [
            {"Imbalance level": f"{float(label):.2f} g", "Model score": float(score)}
            for label, score in result.get("mean_model_scores", {}).items()
        ]
    )
    return result_markdown, windows, scores


def predict_from_upload(
    uploaded_path: str | None,
) -> tuple[str, pd.DataFrame, pd.DataFrame]:
    """Run the locked local predictor for one uploaded CSV."""
    if not uploaded_path:
        return (
            "### Select a CSV first\nChoose an approximately 10-second file containing `time, ax, ay, az`.",
            pd.DataFrame(columns=["Window", "Predicted imbalance (g)"]),
            pd.DataFrame(columns=["Imbalance level", "Model score"]),
        )
    try:
        result = predict_trial(Path(uploaded_path), MODEL_PATH, CONFIG_PATH)
    except (FileNotFoundError, ValueError) as exc:
        return (
            f"### Could not analyze this file\n{exc}",
            pd.DataFrame(columns=["Window", "Predicted imbalance (g)"]),
            pd.DataFrame(columns=["Imbalance level", "Model score"]),
        )
    return format_prediction(result)


def build_app() -> gr.Blocks:
    """Create the local, single-screen Gradio application."""
    with gr.Blocks(
        title="Motor Imbalance Detector",
        fill_width=True,
    ) as demo:
        gr.Markdown(
            "# Motor Imbalance Detector\n"
            "Upload a new MPU6050 recording to estimate the motor's imbalance level. "
            "The recording should be approximately 10 seconds and contain "
            "`time`, `ax`, `ay`, and `az` columns.",
            elem_classes="hero",
        )
        with gr.Row(equal_height=False):
            with gr.Column(scale=5, min_width=300, elem_classes="panel"):
                gr.Markdown("## 1. Choose a recording")
                upload = gr.File(
                    label="MPU6050 CSV file",
                    file_types=[".csv"],
                    type="filepath",
                    height=170,
                )
                gr.Markdown(
                    "Your file is processed locally and is not uploaded to a public service.",
                    elem_classes="privacy-note",
                )
                predict_button = gr.Button(
                    "Predict imbalance",
                    variant="primary",
                    size="lg",
                    elem_id="predict-button",
                )
            with gr.Column(scale=6, min_width=320, elem_classes="panel"):
                gr.Markdown("## 2. Review the result")
                result = gr.Markdown(
                    "Select a CSV and press **Predict imbalance**.",
                    elem_id="prediction-result",
                )
                window_table = gr.Dataframe(
                    headers=["Window", "Predicted imbalance (g)"],
                    datatype=["str", "str"],
                    interactive=False,
                    label="Two-second window votes",
                    wrap=True,
                )
        with gr.Accordion("Model score details", open=False):
            gr.Markdown(
                "These scores show the random forest's average preference across the five "
                "windows. They are not calibrated probabilities or a guarantee of correctness."
            )
            score_table = gr.Dataframe(
                headers=["Imbalance level", "Model score"],
                datatype=["str", "number"],
                interactive=False,
                label="Average model scores",
                wrap=True,
            )
        gr.Markdown(
            "**Important:** This research model achieved 7/10 correct trial predictions "
            "on its locked final test set. Use it as an experimental diagnostic aid, not "
            "as a safety control system.",
            elem_classes="privacy-note",
        )
        predict_button.click(
            fn=predict_from_upload,
            inputs=upload,
            outputs=[result, window_table, score_table],
            show_progress="full",
            scroll_to_output=True,
            concurrency_limit=1,
        )
    return demo


if __name__ == "__main__":
    build_app().launch(
        server_name="127.0.0.1",
        inbrowser=os.environ.get("MOTOR_APP_NO_BROWSER") != "1",
        share=False,
        show_error=True,
        theme=THEME,
        css=CSS,
    )
