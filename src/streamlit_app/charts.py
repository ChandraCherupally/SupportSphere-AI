"""
Plotly chart factory functions for SupportSphere AI Streamlit UI.

All functions are pure — they receive data and return Plotly figures.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go

from evaluation.models import EvaluationResult, EvaluationSummary


# =============================================================================
# Placeholder Charts (empty state)
# =============================================================================

def get_placeholder_charts():
    """Generate placeholder Plotly charts for empty state rendering.

    Returns:
        Tuple of (fig_bar, fig_radar, fig_hist, fig_cm).
    """
    acc_df = __import__("pandas").DataFrame({
        "Category": ["Request Type", "Product Area", "Status"],
        "Accuracy": [0.0, 0.0, 0.0],
    })
    fig_bar = px.bar(
        acc_df, x="Category", y="Accuracy", color="Category",
        text_auto=".1%", range_y=[0, 1.05],
        title="Classification Accuracies (No Data Loaded)"
    )
    fig_bar.update_layout(showlegend=False)

    categories = ["Context Precision", "Context Recall", "Faithfulness", "Answer Correctness", "Answer Relevancy"]
    fig_radar = go.Figure(data=go.Scatterpolar(
        r=[0.0, 0.0, 0.0, 0.0, 0.0], theta=categories, fill="toself", line_color="#a0a0a0"
    ))
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=False, title="RAGAS Performance Shape (No Data Loaded)"
    )

    fig_hist = px.histogram(
        x=[0.0], nbins=10, labels={"x": "Faithfulness Score"},
        title="Faithfulness Score Distribution (No Data Loaded)"
    )

    fig_cm = ff.create_annotated_heatmap(
        z=[[0, 0], [0, 0]], x=["Class A", "Class B"], y=["Class A", "Class B"],
        colorscale="Greys", showscale=True
    )
    fig_cm.update_layout(title="Confusion Matrix (No Data Loaded)", xaxis_title="Predicted", yaxis_title="Ground Truth")

    return fig_bar, fig_radar, fig_hist, fig_cm


# =============================================================================
# Actual Charts (with data)
# =============================================================================

def get_actual_charts(summary: EvaluationSummary, results: List[EvaluationResult]):
    """Generate active Plotly charts from evaluation data.

    Args:
        summary: Aggregated EvaluationSummary.
        results: List of individual EvaluationResult objects.

    Returns:
        Tuple of (fig_bar, fig_radar, fig_hist, fig_cm).
    """
    import pandas as pd

    acc_df = pd.DataFrame({
        "Category": ["Request Type", "Product Area", "Status"],
        "Accuracy": [
            summary.request_type_metrics.accuracy,
            summary.product_area_metrics.accuracy,
            summary.status_metrics.accuracy,
        ],
    })
    fig_bar = px.bar(
        acc_df, x="Category", y="Accuracy", color="Category",
        text_auto=".1%", range_y=[0, 1.05], title="Classification Accuracies"
    )
    fig_bar.update_layout(showlegend=False)

    categories = ["Context Precision", "Context Recall", "Faithfulness", "Answer Correctness", "Answer Relevancy"]
    r_values = [
        summary.avg_context_precision or 0.0,
        summary.avg_context_recall or 0.0,
        summary.avg_faithfulness or 0.0,
        summary.avg_answer_correctness or 0.0,
        summary.avg_answer_relevancy or 0.0,
    ]
    fig_radar = go.Figure(data=go.Scatterpolar(
        r=r_values, theta=categories, fill="toself", line_color="#4F8BF9"
    ))
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=False, title="RAGAS Performance Shape"
    )

    hist_data = [r.ragas_metrics.faithfulness for r in results if r.ragas_metrics.faithfulness is not None]
    if not hist_data:
        hist_data = [0.0]
    fig_hist = px.histogram(x=hist_data, nbins=10, labels={"x": "Faithfulness Score"}, title="Faithfulness Score Distribution")

    y_true = [r.sample.expected_request_type for r in results]
    y_pred = [r.prediction.predicted_request_type for r in results]
    labels = sorted(list(set(y_true) | set(y_pred)))
    if labels:
        z = summary.request_type_metrics.confusion_matrix
        x = labels
        y = labels
    else:
        z = [[0, 0], [0, 0]]
        x = ["A", "B"]
        y = ["A", "B"]

    fig_cm = ff.create_annotated_heatmap(z=z, x=x, y=y, colorscale="Blues", showscale=True)
    fig_cm.update_layout(title="Confusion Matrix (Request Type)", xaxis_title="Predicted", yaxis_title="Ground Truth")

    return fig_bar, fig_radar, fig_hist, fig_cm


# =============================================================================
# Comparison Charts
# =============================================================================

def get_comparison_charts(exp_a: Dict[str, Any], exp_b: Dict[str, Any]):
    """Generate side-by-side comparison charts for two experiments.

    Args:
        exp_a: Baseline experiment dict from ExperimentManager.list_experiments().
        exp_b: Optimized experiment dict from ExperimentManager.list_experiments().

    Returns:
        Tuple of (fig_radar, fig_bar).
    """
    categories = ["Context Precision", "Context Recall", "Faithfulness", "Answer Correctness", "Answer Relevancy"]
    a_vals = [exp_a.get(k) or 0.0 for k in ["context_precision", "context_recall", "faithfulness", "answer_correctness", "answer_relevancy"]]
    b_vals = [exp_b.get(k) or 0.0 for k in ["context_precision", "context_recall", "faithfulness", "answer_correctness", "answer_relevancy"]]

    fig_radar = go.Figure()
    fig_radar.add_trace(go.Scatterpolar(r=a_vals, theta=categories, fill="toself", name=f"Previous ({exp_a['id']})"))
    fig_radar.add_trace(go.Scatterpolar(r=b_vals, theta=categories, fill="toself", name=f"Current ({exp_b['id']})"))
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True, title="RAGAS Performance Radar Comparison"
    )

    fig_bar = go.Figure(data=[
        go.Bar(name=f"Previous ({exp_a['id']})", x=categories, y=a_vals),
        go.Bar(name=f"Current ({exp_b['id']})", x=categories, y=b_vals),
    ])
    fig_bar.update_layout(barmode="group", title="Grouped RAGAS Scores", yaxis=dict(range=[0, 1.05]))

    return fig_radar, fig_bar


# =============================================================================
# Timeline Chart
# =============================================================================

def get_timeline_chart(experiments: List[Dict[str, Any]]):
    """Generate a timeline line chart of avg faithfulness across experiments.

    Args:
        experiments: List of experiment dicts from ExperimentManager.

    Returns:
        Plotly figure.
    """
    if not experiments:
        return go.Figure()

    ids = [e["id"] for e in experiments]
    faith_vals = [e.get("faithfulness") or 0.0 for e in experiments]
    correctness_vals = [e.get("answer_correctness") or 0.0 for e in experiments]
    precision_vals = [e.get("context_precision") or 0.0 for e in experiments]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ids, y=faith_vals, mode="lines+markers", name="Faithfulness", line=dict(color="#4F8BF9")))
    fig.add_trace(go.Scatter(x=ids, y=correctness_vals, mode="lines+markers", name="Answer Correctness", line=dict(color="#F97B4F")))
    fig.add_trace(go.Scatter(x=ids, y=precision_vals, mode="lines+markers", name="Context Precision", line=dict(color="#4FF9A5")))
    fig.update_layout(
        title="📈 RAGAS Metrics Timeline Across Experiments",
        xaxis_title="Experiment ID",
        yaxis=dict(range=[0, 1.05]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=350,
    )
    return fig


# =============================================================================
# Cost Comparison Chart
# =============================================================================

def get_cost_comparison_chart(experiments: List[Dict[str, Any]]):
    """Generate a bar chart of avg cost per ticket across experiments.

    Args:
        experiments: List of experiment dicts.

    Returns:
        Plotly figure.
    """
    if not experiments:
        return go.Figure()

    ids = [e["id"] for e in experiments]
    costs = [e.get("avg_cost_per_ticket") or 0.0 for e in experiments]

    fig = px.bar(x=ids, y=costs, labels={"x": "Experiment ID", "y": "Avg Cost / Ticket (USD)"},
                 title="💰 Average Cost Per Ticket", color=costs, color_continuous_scale="Viridis")
    fig.update_layout(coloraxis_showscale=False)
    return fig
