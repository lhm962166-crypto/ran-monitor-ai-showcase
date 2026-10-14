"""Shared dashboard labels, formatting and root-cause probability views."""
from __future__ import annotations
from datetime import datetime, timedelta, timezone
import pandas as pd
import plotly.express as px
import streamlit as st
from .settings import CAUSE_ZH

SCORE_COLUMNS = {
    "异常评分": st.column_config.NumberColumn(format="%.4f"),
    "异常分数": st.column_config.NumberColumn(format="%.4f"),
    "阈值": st.column_config.NumberColumn(format="%.4f"),
    "检测阈值": st.column_config.NumberColumn(format="%.4f"),
    "原因原始输出（未校准）": st.column_config.NumberColumn(format="percent"),
}

def severity_label(value: str) -> str:
    return {"P1": "P1 · 紧急", "P2": "P2 · 严重", "P3": "P3 · 一般", "P4": "P4 · 提示"}.get(value, value)


def alert_status_label(value: str) -> str:
    return {"OPEN": "待处理", "ACKNOWLEDGED": "已确认", "RESOLVED": "已解决"}.get(value, value)


def display_time(value: str | None) -> str:
    if not value:
        return "—"
    try:
        parsed = datetime.fromisoformat(value)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone(timedelta(hours=8))).strftime("%Y-%m-%d %H:%M:%S")
    except (TypeError, ValueError):
        return str(value)


def history_frame(alerts: list[dict]) -> pd.DataFrame:
    return pd.DataFrame([
        {"告警编号": a["alert_id"], "事件编号": a.get("incident_id", "—"), "窗口": a["sample_id"],
         "级别": severity_label(a["severity"]), "原因": CAUSE_ZH.get(a["cause"], a["cause"]),
         "异常分数": a["score"], "状态": alert_status_label(a["status"]),
         "时间（北京时间）": display_time(a["created_at"])} for a in alerts
    ])


def render_cause_probabilities(alert: dict, key: str) -> None:
    probabilities = alert.get("calibrated_cause_probabilities") or alert.get("cause_probabilities") or {}
    if not probabilities:
        return
    if alert.get("requires_review"):
        st.warning("根因分类存在不确定性，请结合指标曲线和其他网络证据复核。")
    candidates = alert.get("cause_candidates") or []
    if candidates:
        st.caption("候选原因：" + "、".join(item.get("cause_zh", item.get("cause", "未知")) for item in candidates))
    calibration_status = alert.get("cause_set_status", "unavailable")
    calibration_status_zh = {
        "experimental": "实验性校准",
        "insufficient_independent_calibration_data": "校准证据不足",
        "unavailable": "暂无校准数据",
    }.get(calibration_status, calibration_status)
    with st.expander("查看原因分布与候选集合", expanded=bool(alert.get("requires_review"))):
        distribution = pd.DataFrame([
            {"故障类型": CAUSE_ZH.get(name, name), "温度缩放实验值": probability,
             "原始模型输出": (alert.get("cause_probabilities") or {}).get(name)}
            for name, probability in probabilities.items()
        ]).sort_values("温度缩放实验值", ascending=True)
        figure = px.bar(distribution, x="温度缩放实验值", y="故障类型", orientation="h",
                        text="温度缩放实验值", color="故障类型",
                        color_discrete_map={"无线干扰": "#7048aa", "网络拥塞": "#df7627",
                                            "设备故障": "#d83d4d", "移动性 / 切换问题": "#167f87"})
        figure.update_traces(texttemplate="%{x:.1%}", textposition="outside")
        figure.update_layout(height=250, showlegend=False, xaxis_range=[0, 1.12],
                             xaxis_tickformat=".0%", paper_bgcolor="white", plot_bgcolor="white",
                             margin=dict(l=12, r=28, t=12, b=12), xaxis_title="", yaxis_title="")
        st.plotly_chart(figure, width="stretch", key=key, config={
            "displayModeBar": "hover", "displaylogo": False, "responsive": True, "scrollZoom": False,
        })
        st.dataframe(distribution.rename(columns={"温度缩放实验值": "温度缩放实验值", "原始模型输出": "原始未校准输出"}),
                     width="stretch", hide_index=True)
        st.caption(f"温度缩放与 APS 候选集合来自 {alert.get('cause_calibration_samples', 0)} 个验证故障窗口、"
                   f"{alert.get('cause_calibration_groups', 0)} 个不重叠时间组；当前校准状态："
                   f"{calibration_status_zh}。{alert.get('cause_set_caveat', '')}"
                   f"温度参数另用 {alert.get('temperature_calibration_samples', 0)} 个窗口、"
                   f"{alert.get('temperature_calibration_groups', 0)} 个不重叠组拟合。"
                   "分数不等同于真实发生率，需结合 KPI 曲线和现场证据确认。")
        ranking = alert.get("cause_ranking_shadow")
        if isinstance(ranking, dict) and ranking.get("candidates"):
            st.markdown("#### KPI 原型辅助排序（影子实验）")
            st.caption("分类器名次与训练 KPI 原型距离的辅助排序；排序分数不是概率，也不替代人工核实。")
            st.dataframe(pd.DataFrame([{
                "排名": item["rank"],
                "候选原因": CAUSE_ZH.get(item["cause"], item["cause"]),
                "分类器原始分数": round(item["classifier_probability"], 4),
                "原型距离（越小越接近）": round(item["kpi_prototype_distance"], 3),
                "实验排序分数": round(item["ranking_score"], 3),
            } for item in ranking["candidates"]]), width="stretch", hide_index=True)
