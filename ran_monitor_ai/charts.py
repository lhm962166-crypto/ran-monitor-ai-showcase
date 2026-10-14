"""Reusable KPI charts with explicit window-mean reference semantics."""
from __future__ import annotations

import plotly.graph_objects as go


def kpi_figure(name: str, values, baseline: dict | None = None, comparison=None) -> go.Figure:
    figure = go.Figure()
    if baseline:
        mean, std = float(baseline["mean"]), float(baseline["std"])
        figure.add_hrect(y0=mean - 3 * std, y1=mean + 3 * std,
                         fillcolor="#42c6d9", opacity=.11, line_width=0,
                         annotation_text="正常窗口均值 ±3σ（参考）", annotation_position="top left")
        figure.add_hline(y=mean, line_dash="dash", line_color="#7b91aa",
                         annotation_text="正常均值", annotation_position="bottom right")
    if comparison is not None:
        figure.add_trace(go.Scatter(x=list(range(len(comparison))), y=list(comparison), mode="lines",
                                   name="正常参考窗口", line={"color": "#087f8c", "width": 1.5, "dash": "dot"}))
    figure.add_trace(go.Scatter(x=list(range(len(values))), y=list(values), mode="lines",
                               name="当前窗口", line={"color": "#1876c9", "width": 2.5}))
    figure.update_layout(title=name, xaxis_title="采样点", yaxis_title="原始指标值", height=320,
                         paper_bgcolor="white", plot_bgcolor="white", font_color="#38516a",
                         margin=dict(l=12, r=12, t=48, b=12), hovermode="x unified",
                         legend=dict(orientation="h", y=-.2, x=0))
    figure.update_xaxes(showgrid=False, linecolor="#dce5ec")
    figure.update_yaxes(gridcolor="#e9eff4", zeroline=False)
    return figure
