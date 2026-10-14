from __future__ import annotations

import json
import os
import sqlite3
from time import perf_counter
from pathlib import Path
from datetime import datetime, timezone

from ran_monitor_ai.runtime import configure_threads
configure_threads()

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st
from sklearn.metrics import precision_recall_curve

from ran_monitor_ai.charts import kpi_figure
from ran_monitor_ai.configuration import load_env_file
from ran_monitor_ai.engine import POLICY_ZH, baseline_for_record, bundle_for_policy, explain_record, score_record, score_records
from ran_monitor_ai.exporting import dataframe_csv_bytes
from ran_monitor_ai.features import extract_features, parse_json_payload, true_cause, validate_batch
from ran_monitor_ai.notify import channels_ready, deliver_digest
from ran_monitor_ai.monitoring import update_score_drift
from ran_monitor_ai.model_benchmark import compare_with_isolation_forest
from ran_monitor_ai.reporting import detection_report_html
from ran_monitor_ai.replay import ReplayRun, replay_windows
from ran_monitor_ai.settings import ACTION, CAUSE_ZH, KPIS
from ran_monitor_ai.store import (recent, reset_persisted_score_drift, save_batch,
                                  update_persisted_score_drift)
from ran_monitor_ai.train import ARTIFACT_PATH, DATA_PATH, ROOT, calibrate_threshold, get_data_path, train
from ran_monitor_ai.ui_common import SCORE_COLUMNS, display_time, history_frame, render_cause_probabilities, severity_label
from ran_monitor_ai.ui_alerts import render_alert_center

PLOTLY_CONFIG = {
    "displayModeBar": "hover",
    "displaylogo": False,
    "responsive": True,
    "scrollZoom": False,
}
pio.templates["ran_ops"] = go.layout.Template(
    layout=go.Layout(
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(family="Microsoft YaHei UI, PingFang SC, sans-serif", color="#38516a", size=12),
        colorway=["#1876c9", "#087f8c", "#7559b6", "#d47923", "#16805d", "#c54345"],
        hoverlabel=dict(bgcolor="#102a43", bordercolor="#102a43", font=dict(color="#f6fbff", size=12)),
        xaxis=dict(gridcolor="#e9eff4", linecolor="#dce5ec", zerolinecolor="#dce5ec"),
        yaxis=dict(gridcolor="#e9eff4", linecolor="#dce5ec", zerolinecolor="#dce5ec"),
    )
)
pio.templates.default = "ran_ops"

st.set_page_config(page_title="脉序 · 无线网络智能运维", page_icon="🛰️", layout="wide", initial_sidebar_state="collapsed")
DEMO_MODE = os.getenv("RAN_MONITOR_DEMO") == "1"
if not DEMO_MODE:
    try:
        load_env_file(ROOT / ".env")
    except (ValueError, OSError) as error:
        st.warning(f"本地配置文件未加载：{error}。请修正配置后重新启动。")
DATA_PATH = get_data_path()
if not DATA_PATH.is_file():
    st.error(
        "尚未配置本地训练数据。此代码仓库不包含 TelecomTS 样本；请从 "
        "https://huggingface.co/datasets/AliMaatouk/TelecomTS 单独获取并转换为项目 JSONL 格式，"
        "再在 .env 中设置 RAN_MONITOR_DATA_PATH，或放到 data/telecomts_demo.jsonl。"
    )
    st.stop()
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_BATCH_RECORDS = 100

st.markdown("<style>" + (ROOT / "assets" / "theme.css").read_text(encoding="utf-8") + "</style>", unsafe_allow_html=True)
if DEMO_MODE:
    st.info("合成演示模式：数据由固定随机种子生成，仅用于体验检测、告警和回放流程，不代表真实网络或模型现网效果。")


@st.cache_resource
def load_model(revision):
    return train()


@st.cache_data
def load_records(revision):
    with DATA_PATH.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


revision_paths = [DATA_PATH, ROOT / "ran_monitor_ai" / "features.py", ROOT / "ran_monitor_ai" / "train.py",
                  ROOT / "ran_monitor_ai" / "settings.py", ROOT / "ran_monitor_ai" / "evaluation.py",
                  ROOT / "ran_monitor_ai" / "data_audit.py", ROOT / "ran_monitor_ai" / "governance.py",
                  ROOT / "ran_monitor_ai" / "uncertainty.py", ROOT / "ran_monitor_ai" / "structure_detector.py",
                  ARTIFACT_PATH]
revision = tuple((str(path), path.stat().st_mtime_ns, path.stat().st_size) if path.exists()
                 else (str(path), None, None) for path in revision_paths)
with st.spinner("正在加载本地模型与网络窗口…"):
    bundle, metrics = load_model(revision)
    records = load_records(revision[0])
sample_map = {int(r.get("sample_id", i)): r for i, r in enumerate(records)}
test_records = [records[i] for i in bundle["test_indices"]]


def infer_test_rows():
    rows = []
    predictions = score_records(
        test_records, bundle, feature_rows=all_feature_rows(revision[0]).iloc[bundle["test_indices"]]
    )
    for record, result in zip(test_records, predictions):
        true_label = true_cause(record)
        alert = result["alert"]
        rows.append({
            "窗口编号": record.get("sample_id"),
            "参考标签": CAUSE_ZH.get(true_label, true_label),
            "模型判定": "异常" if result["detected"] else "正常",
            "异常评分": result["score"],
            "告警级别": alert["severity"] if alert else "—",
            "可能原因": alert["cause_zh"] if alert else "—",
            "原因原始输出（未校准）": alert["confidence"] if alert else None,
            "候选原因数": len(alert.get("cause_candidates", [])) if alert else 0,
            "结构影子分数": (result.get("structure_shadow") or {}).get("score"),
            "结构影子提示": "影子信号" if (result.get("structure_shadow") or {}).get("shadow_flag") else "—",
            "异常指标数": len(alert["affected_kpis"]) if alert else 0,
        })
    return pd.DataFrame(rows)


@st.cache_data
def dashboard_rows(model_revision):
    return infer_test_rows()


@st.cache_data
def all_feature_rows(data_revision):
    return pd.DataFrame([
        extract_features(record) | {"cause": true_cause(record)}
        for record in records
    ])


@st.cache_data
def explain_window(record, model_revision):
    return explain_record(record, bundle)


def render_explanation(record: dict, key: str) -> None:
    with st.expander("模型为什么给出这个异常分数？"):
        st.caption("正贡献推高异常分数，负贡献降低异常分数。贡献解释模型的判断，不等同于故障的因果证据。")
        if st.button("计算特征贡献", key=f"{key}_explain"):
            st.session_state[f"{key}_explanation"] = {
                "sample_id": record.get("sample_id"),
                "model_id": bundle.get("metadata", {}).get("model_id"),
                "data": explain_window(record, bundle.get("metadata", {}).get("fingerprint")),
            }
        explanation = st.session_state.get(f"{key}_explanation", {})
        if (explanation.get("sample_id") == record.get("sample_id")
                and explanation.get("model_id") == bundle.get("metadata", {}).get("model_id")
                and explanation.get("data")):
            data = explanation["data"]
            frame = pd.DataFrame(data["groups"][:10]).sort_values("contribution")
            figure = px.bar(frame, x="contribution", y="name", orientation="h",
                            color=frame["contribution"] > 0,
                            color_discrete_map={True: "#c54345", False: "#16805d"})
            figure.update_layout(height=360, showlegend=False, title="对异常分数影响最大的 KPI 特征组",
                                 xaxis_title="对数几率贡献", yaxis_title="", paper_bgcolor="white", plot_bgcolor="white",
                                 margin=dict(l=12, r=12, t=55, b=12))
            st.plotly_chart(figure, width="stretch", key=f"{key}_contributions", config=PLOTLY_CONFIG)
            st.dataframe(pd.DataFrame(data["features"][:12]).rename(columns={
                "feature": "特征", "value": "当前值", "contribution": "对数几率贡献",
            }), hide_index=True, width="stretch")


def observe_score_drift(scores, window_ids, stream_source: str, stream_id: str | None = None) -> dict:
    calibration = bundle.get("calibration", {})
    model_id = bundle.get("metadata", {}).get("model_id", "unknown")
    if stream_id:
        return update_persisted_score_drift(
        stream_id, scores, window_ids, calibration.get("normal_scores", calibration.get("scores", [])),
        model_id=model_id,
        )
    state, summary = update_score_drift(
        st.session_state.get("score_drift_state"), scores,
        calibration.get("normal_scores", calibration.get("scores", [])),
        model_id=model_id, stream_source=stream_source, window_ids=window_ids,
    )
    st.session_state["score_drift_state"] = state
    return summary


def render_score_drift(summary: dict, title: str = "模型输出分数漂移观察") -> None:
    st.markdown(f"#### {title}")
    if summary.get("status") == "multi_stream":
        labels = {"warming_up": "积累中", "stable": "未发现显著偏移", "watch": "观察", "drift": "显著偏移"}
        st.dataframe(pd.DataFrame([{
            "数据流 ID": stream_id,
            "状态": labels.get(item.get("status"), item.get("status")),
            "滚动窗口": item.get("current_samples"),
            "累计唯一窗口": item.get("observed_total"),
            "本批跳过重复": item.get("duplicate_samples_ignored", 0),
            "KS 差异": item.get("ks_statistic"),
            "p 值": item.get("p_value"),
        } for stream_id, item in summary.get("streams", {}).items()]), width="stretch", hide_index=True)
        st.caption("各数据流分别持久化监测；KS 变化只提示分布偏移，不是准确率下降或序贯误报率保证。")
        return
    if summary["status"] == "warming_up":
        st.info(f"样本积累中：已处理 {summary.get('observed_total', summary['current_samples'])} 个，"
                f"滚动窗口 {summary['current_samples']} / {summary['minimum_samples']}。达到最小样本数后开始检验。")
    else:
        status = summary["status"]
        if status == "drift":
            st.error("检测到模型输出分数分布显著偏移，建议检查近期输入并复核模型。")
        elif status == "watch":
            st.warning("发现分数分布变化迹象，继续观察后续窗口。")
        else:
            st.success("当前滚动窗口未发现显著分数分布偏移。")
        columns = st.columns(4)
        columns[0].metric("滚动 / 累积窗口",
                          f"{summary['current_samples']} / {summary.get('observed_total', summary['current_samples'])}")
        columns[1].metric("KS 差异", f"{summary['ks_statistic']:.3f}" if summary.get("ks_statistic") is not None else "—")
        columns[2].metric("检验 p 值", f"{summary['p_value']:.4f}" if summary.get("p_value") is not None else "—")
        columns[3].metric("近期 / 参照中位数",
                          f"{summary['recent_median']:.3f} / {summary['reference_median']:.3f}")
    lifetime = "状态已写入本机数据库，可跨页面刷新恢复。" if summary.get("persistence") == "persistent" else "未提供稳定数据流 ID，状态仅保留在本次会话中。"
    st.caption(f"比较最近最多 100 个唯一窗口的模型分数与正常验证窗口参照；本批跳过重复窗口 {summary.get('duplicate_samples_ignored', 0)} 个。"
               "滚动 KS 是分布观察信号，多次查看不构成序贯误报率保证，也不代表准确率下降，更不会自动改动模型。" + lifetime)


st.markdown(f"""
<header class="brand-header">
  <div class="brand-left">
    <div class="brand-mark" aria-hidden="true">
      <svg viewBox="0 0 32 32" fill="none" xmlns="http://www.w3.org/2000/svg">
        <path d="M3 17h6l3.2-8 5.2 15 3.5-10 2.3 3H29" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/>
        <circle cx="16" cy="16" r="14.2" stroke="currentColor" stroke-opacity=".32"/>
      </svg>
    </div>
    <div class="brand-copy">
      <div class="brand-kicker">RAN OBSERVABILITY · LOCAL LAB</div>
      <div class="brand-title">脉序</div>
      <div class="brand-subtitle">无线网络智能运维 · 运行态势与异常分析</div>
    </div>
  </div>
  <div class="brand-meta">
    <div class="status-pill"><span class="status-dot"></span>本地模型已就绪</div>
    <div class="environment-pill">离线演示环境</div>
  </div>
</header>
""", unsafe_allow_html=True)
st.caption(
    f"当前运行独立训练的本地模型。演示集共 {metrics.get('samples', len(records)):,} 个窗口，"
    "来自本机配置的数据文件；指标只描述该数据切分，不代表现网性能。"
)

alerts_df = dashboard_rows(bundle.get("metadata", {}).get("fingerprint", revision))
st.sidebar.markdown("<div class='side-brand'><strong>筛选条件</strong><span>应用于测试集告警列表</span></div>", unsafe_allow_html=True)
severity_values = ["P1", "P2", "P3", "P4"]
severity_options = [
    ("P1", "P1 · 紧急"), ("P2", "P2 · 严重"), ("P3", "P3 · 一般"), ("P4", "P4 · 提示"),
]
selected_severity = []
st.sidebar.markdown("<div style='font-size:.82rem;font-weight:700;margin:4px 0 8px'>告警级别</div>", unsafe_allow_html=True)
with st.sidebar.container(key="severity-filter"):
    for level, label in severity_options:
        if st.checkbox(label, value=True, key=f"severity_{level}"):
            selected_severity.append(level)
cause_options = [
    ("interference", "无线干扰"), ("congestion", "网络拥塞"),
    ("equipment", "设备故障"), ("mobility", "移动性 / 切换问题"),
]
selected_causes = []
st.sidebar.markdown("<div style='font-size:.82rem;font-weight:700;margin:6px 0 8px'>可能原因</div>", unsafe_allow_html=True)
with st.sidebar.container(key="cause-filter"):
    for cause_key, cause in cause_options:
        if st.checkbox(cause, value=True, key=f"cause_{cause_key}"):
            selected_causes.append(cause)
st.sidebar.caption("概览基于留出测试窗口；演示数据小，统计仅作功能预览。")

tabs = st.tabs(["网络概览", "故障分析", "窗口检查", "实时检测演示", "告警中心", "模型评估", "场景回放"])

with tabs[0]:
    selected = alerts_df[
        alerts_df["告警级别"].isin(selected_severity)
        & alerts_df["可能原因"].isin(selected_causes)
    ]
    st.markdown("<div class='section-head'><div><div class='section-title'>网络运行总览</div><div class='section-note'>模型在留出测试集上的检测与告警概况</div></div></div>", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    anomaly_rate = (alerts_df["模型判定"] == "异常").mean() * 100 if len(alerts_df) else 0
    overview_metrics = [
        (c1, "overview_windows", "留出集窗口", len(alerts_df)),
        (c2, "overview_anomalies", "模型检出的异常", int((alerts_df["模型判定"] == "异常").sum())),
        (c3, "overview_rate", "异常占比", f"{anomaly_rate:.1f}%"),
        (c4, "overview_p1", "P1 紧急告警", int((selected["告警级别"] == "P1").sum())),
    ]
    for column, key, label, value in overview_metrics:
        with column:
            with st.container(key=key):
                st.metric(label, value)
    st.caption(f"当前检测阈值：{bundle['threshold']:.3f}　·　告警数量受左侧级别和原因筛选影响")
    st.markdown("<div class='section-head'><div><div class='section-title'>告警态势</div><div class='section-note'>按故障类型和告警等级查看留出集结果</div></div></div>", unsafe_allow_html=True)
    st.caption(
        f"留出集：Precision {metrics.get('test_precision', 0):.3f} · "
        f"Recall {metrics.get('test_recall', 0):.3f} · F1 {metrics.get('test_f1', 0):.3f} · "
        f"ROC-AUC {metrics.get('test_auc', 0):.3f} · AP {metrics.get('test_average_precision', 0):.3f} · "
        f"真实异常上的根因分类 Macro-F1 {metrics.get('cause_test_macro_f1', 0):.3f}。"
        f"{metrics.get('split_note', '演示数据仅用于功能验证。')}"
    )
    if not selected.empty:
        counts = selected.groupby(["可能原因", "告警级别"]).size().reset_index(name="告警数")
        left_chart, right_chart = st.columns([1.65, 1])
        with left_chart:
            fig = px.bar(counts, x="可能原因", y="告警数", color="告警级别", barmode="stack",
                         category_orders={"告警级别": severity_values},
                         color_discrete_map={"P1":"#c54345","P2":"#d47923","P3":"#c49a20","P4":"#3583ac"})
            fig.update_layout(title="故障类别与告警等级", paper_bgcolor="white", plot_bgcolor="white",
                              font_color="#526259", margin=dict(l=8,r=8,t=52,b=8), height=340,
                              legend_title_text="告警等级", xaxis_title="", yaxis_title="告警数量")
            fig.update_xaxes(showgrid=False, linecolor="#e1e6dc")
            fig.update_yaxes(gridcolor="#edf0e9", zeroline=False)
            st.plotly_chart(fig, width="stretch", config=PLOTLY_CONFIG)
        with right_chart:
            severity_counts = selected["告警级别"].value_counts().reindex(severity_values).dropna().reset_index()
            severity_counts.columns = ["告警级别", "数量"]
            fig2 = px.pie(severity_counts, names="告警级别", values="数量", hole=.68,
                          color="告警级别", color_discrete_map={"P1":"#c54345","P2":"#d47923","P3":"#c49a20","P4":"#3583ac"})
            fig2.update_layout(title="告警等级占比", paper_bgcolor="white", font_color="#526259",
                               margin=dict(l=8,r=8,t=52,b=8), height=340, showlegend=True, legend_title_text="等级")
            st.plotly_chart(fig2, width="stretch", config=PLOTLY_CONFIG)
    st.markdown("<div class='section-head'><div><div class='section-title'>告警事件</div><div class='section-note'>模型判定异常的窗口明细，可用于筛选与导出</div></div></div>", unsafe_allow_html=True)
    st.dataframe(selected, width="stretch", hide_index=True, column_config=SCORE_COLUMNS)
    if not selected.empty:
        st.download_button(
            "导出当前筛选结果 CSV",
            data=dataframe_csv_bytes(selected),
            file_name=f"ran-alerts-{datetime.now():%Y%m%d-%H%M%S}.csv",
            mime="text/csv",
            key="export_filtered_alerts",
        )
    else:
        st.info("当前筛选条件下没有告警，请调整左侧级别或原因筛选。")
    try:
        saved = recent(10)
    except (sqlite3.Error, OSError, ValueError) as error:
        saved = []
        st.warning(f"暂时无法读取本机告警历史：{error}")
    with st.expander("最近保存的实时告警（最多 10 条）"):
        if saved:
            st.dataframe(history_frame(saved), width="stretch", hide_index=True, column_config=SCORE_COLUMNS)
            st.caption("完整历史、筛选和处置操作请打开“告警中心”。")
        else:
            st.caption("运行实时检测后，异常结果会写入本机数据库；之后可以确认、解决或重新打开告警。")

with tabs[1]:
    st.markdown("<div class='section-head'><div><div class='section-title'>故障根因分析</div><div class='section-note'>对比各类故障相对正常 KPI 基线的偏离程度</div></div></div>", unsafe_allow_html=True)
    feature_df = all_feature_rows(revision[0])
    mean_cols = [f"{k}_mean" for k in KPIS]
    normal = feature_df[feature_df["cause"] == "Normal"]
    base_mean = normal[mean_cols].mean()
    base_std = normal[mean_cols].std().replace(0, 1e-6)
    matrix = {}
    for cause, zh in (("Interference", "无线干扰"), ("Congestion", "网络拥塞"),
                      ("Equipment", "设备故障"), ("Mobility", "移动性 / 切换问题")):
        group = feature_df[feature_df["cause"] == cause]
        if len(group):
            matrix[zh] = ((group[mean_cols].mean() - base_mean) / base_std).to_numpy()
    if matrix:
        heatmap = pd.DataFrame(matrix, index=KPIS)
        st.plotly_chart(px.imshow(heatmap, color_continuous_scale=["#1876c9", "#f4f8fb", "#c54345"], zmin=-5, zmax=5,
                                  aspect="auto", labels={"color": "相对正常值偏离"}),
                        width="stretch", config=PLOTLY_CONFIG)
    st.markdown("<div class='section-head'><div><div class='section-title'>处置建议</div><div class='section-note'>根据故障分类查看建议操作与负责团队</div></div></div>", unsafe_allow_html=True)
    chosen_cause = st.selectbox("查看处理建议", list(ACTION), format_func=lambda c: CAUSE_ZH[c])
    advice, team = ACTION[chosen_cause]
    st.success(f"建议：{advice}。负责团队：{team}。")
    st.caption("颜色是演示数据中已知标签的均值差异，模型输出的是概率分类，需结合其他网络证据核实。")

with tabs[2]:
    st.markdown("<div class='section-head'><div><div class='section-title'>KPI 窗口检查</div><div class='section-note'>查看单个 128 采样点窗口内的指标时间序列</div></div></div>", unsafe_allow_html=True)
    chosen_id = st.selectbox("选择窗口", list(sample_map),
                             format_func=lambda sid: f"窗口 {sid} · {CAUSE_ZH.get(true_cause(sample_map[sid]), '未知')}",
                             key="inspector_sample_id")
    chosen = sample_map[chosen_id]
    result = score_record(chosen, bundle)
    baseline, baseline_scope = baseline_for_record(chosen, bundle)
    window_alert = result["alert"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("模型判定", "异常" if result["detected"] else "正常")
    c2.metric("异常评分", f"{result['score']:.4f}")
    c3.metric("检测阈值", f"{result['threshold']:.4f}")
    c4.metric("可能原因", window_alert["cause_zh"] if window_alert else "—")
    st.caption(f"数据参考标签：{CAUSE_ZH.get(true_cause(chosen), '未知')}。模型判定只依据 KPI 特征。")
    if window_alert:
        st.warning(f"{severity_label(window_alert['severity'])} · {window_alert['severity_reason']}")
        advice_label = "候选原因建议（需复核）" if window_alert.get("requires_review") else "建议"
        st.caption(f"原因原始输出（未校准）：{window_alert['confidence']:.1%} · "
                   f"{advice_label}：{window_alert['action']}（{window_alert['team']}）")
        render_cause_probabilities(window_alert, key="inspector_cause_distribution")
        if window_alert["affected_kpis"]:
            st.dataframe(pd.DataFrame(window_alert["affected_kpis"]).rename(columns={
                "name": "指标", "current": "当前均值", "baseline": "正常基线",
                "zscore": "偏离标准差", "direction": "偏离方向",
            }), width="stretch", hide_index=True)
    else:
        st.success("该窗口低于异常检测阈值，模型判定为正常。")
        shadow = result.get("structure_shadow") or {}
        if shadow.get("shadow_flag"):
            st.info(f"影子通道提示关系结构偏离（分数 {shadow['score']:.3f} / 阈值 {shadow['threshold']:.3f}）。"
                    "该信号尚未晋级，不会触发告警。")
            st.dataframe(pd.DataFrame(shadow.get("top_relation_shifts", [])),
                         width="stretch", hide_index=True)
    render_explanation(chosen, "inspector")
    picked_kpis = st.multiselect("选择指标", list(KPIS),
                                 default=["UL_SNR", "DL_BLER", "RX_Bytes", "PRB_Utilization_DL"], key="inspector_kpis")
    labels = chosen.get("labels") or {}
    normal_references = {sid: record for sid, record in sample_map.items() if true_cause(record) == "Normal"
                         and all((record.get("labels") or {}).get(field) == labels.get(field)
                                 for field in ("zone", "application"))}
    if not normal_references:
        normal_references = {sid: record for sid, record in sample_map.items() if true_cause(record) == "Normal"}
    comparison = None
    compare_enabled = st.checkbox("叠加正常参考窗口", value=False, key="compare_normal_window")
    if compare_enabled and normal_references:
        reference_id = st.selectbox("正常参考窗口", list(normal_references), key="normal_reference_id",
                                    format_func=lambda sid: f"正常窗口 {sid}")
        comparison = normal_references[reference_id]
    st.caption(f"均值基线：{baseline_scope}，来自 {baseline.get('__metadata__', {}).get('normal_windows', '—')} 个正常训练窗口。"
               "阴影是正常窗口均值的 ±3σ 参考范围，不是单个采样点的告警阈值；参考曲线按采样点对齐。")
    chart_columns = st.columns(2)
    for index, name in enumerate(picked_kpis):
        values = chosen["KPIs"][name]
        figure = kpi_figure(name, values, baseline.get(f"{name}_mean"),
                            comparison["KPIs"][name] if comparison else None)
        with chart_columns[index % 2]:
            st.plotly_chart(figure, width="stretch", config=PLOTLY_CONFIG)

with tabs[3]:
    st.markdown("<div class='section-head'><div><div class='section-title'>实时检测工作台</div><div class='section-note'>提交单个窗口或批量 JSON，查看异常评分、根因和告警等级</div></div></div>", unsafe_allow_html=True)
    st.caption("选一个演示窗口，或上传一条 TelecomTS JSON 窗口 / 窗口数组。数组按文件顺序逐条处理。")
    st.download_button("下载可上传的 JSON 样例", data=(ROOT / "sample_window.json").read_bytes(),
                       file_name="sample_window.json", mime="application/json", key="download_window_template")
    source = st.radio("数据来源", ["项目样例", "上传 JSON"], horizontal=True, key="data_source")
    upload = st.file_uploader(
        "选择 TelecomTS JSON 文件", type=["json"], disabled=source != "上传 JSON",
        help=f"单文件不超过 {MAX_UPLOAD_BYTES // (1024 * 1024)} MB；窗口数组最多 {MAX_BATCH_RECORDS} 条。",
    )
    sample_id = st.selectbox("样例窗口", list(sample_map),
                             format_func=lambda sid: f"窗口 {sid} · {CAUSE_ZH.get(true_cause(sample_map[sid]), '未知')}",
                             disabled=source != "项目样例", key="live_sample_id")
    available_policies = list(POLICY_ZH)
    if "custom_detection_threshold" in st.session_state:
        available_policies.append("custom")
    detection_policy = st.selectbox(
        "检测策略", available_policies,
        format_func=lambda value: "自定义校准" if value == "custom" else POLICY_ZH[value],
        key="detection_policy",
    )
    if detection_policy == "custom":
        custom = st.session_state["custom_detection_threshold"]
        policy_bundle = bundle | {"threshold": custom["threshold"], "detection_policy": "custom"}
        profile = custom
    else:
        policy_bundle = bundle_for_policy(bundle, detection_policy)
        profile = bundle.get("threshold_profiles", {}).get(detection_policy, {})
    st.caption(f"当前阈值 {policy_bundle['threshold']:.4f} · 验证集精确率 {profile.get('validation_precision', profile.get('precision', 0)):.1%} · "
               f"验证集召回率 {profile.get('validation_recall', profile.get('recall', 0)):.1%}。策略目标仅在验证数据上校准，不保证现网表现。")
    stream_id_input = st.text_input(
        "补录稳定数据流 ID（可选）", max_chars=200, key="monitor_stream_id",
        help="仅为缺少 ID 的窗口补入基站、仿真轨道或采集流标识；文件已有的流 ID 会保留。不要用文件名或临时会话号代替稳定身份。",
    ).strip()
    st.caption("告警级别按观测到的 KPI 影响计算；模型原因只用于排查建议。结构关系分数处于影子模式，不会触发或抑制告警。")
    if profile and not profile.get("target_met", True):
        st.warning("验证集无法达到该策略目标，当前沿用均衡阈值。")
    persist_alerts = st.checkbox("保存异常结果到本机告警中心", value=True, key="persist_alerts")
    with st.expander("可选告警通知"):
        ready = {"email": False, "whatsapp": False} if DEMO_MODE else channels_ready()
        notify_email = st.checkbox("检测后发送邮件" + ("（已配置）" if ready["email"] else "（未配置）"),
                                    disabled=DEMO_MODE or not ready["email"])
        notify_wa = st.checkbox("检测后发送 WhatsApp" + ("（已配置）" if ready["whatsapp"] else "（未配置）"),
                                disabled=DEMO_MODE or not ready["whatsapp"])
        notify_policy_label = st.selectbox("发送级别", ["仅 P1", "P1 和 P2", "所有异常"])
        if DEMO_MODE:
            st.caption("演示模式已关闭所有外部通知；告警仅保存在本次演示的临时本机数据库中。")
        else:
            st.caption("需要在本地 .env 中配置收件人。每次检测按渠道发送一条汇总通知，避免批量检测逐条发送。")
    if st.button("运行检测", type="primary", key="run_detection"):
        try:
            if source == "上传 JSON":
                if upload is None:
                    raise ValueError("请先选择 JSON 文件。")
                if upload.size > MAX_UPLOAD_BYTES:
                    raise ValueError(f"文件超过 {MAX_UPLOAD_BYTES // (1024 * 1024)} MB 限制。")
                payload = parse_json_payload(upload.getvalue())
                batch = validate_batch(payload, max_records=MAX_BATCH_RECORDS)
            else:
                batch = [sample_map[sample_id]]
            if stream_id_input:
                enriched = []
                for record in batch:
                    labels = record.get("labels") or {}
                    metadata = record.get("metadata") or {}
                    has_identity = record.get("stream_id") or labels.get("stream_id") or metadata.get("stream_id")
                    enriched.append(dict(record) if has_identity else dict(record) | {"stream_id": stream_id_input})
                batch = enriched
            record_stream_ids = {
                str(record.get("stream_id") or (record.get("labels") or {}).get("stream_id")
                    or (record.get("metadata") or {}).get("stream_id")).strip()
                for record in batch
                if record.get("stream_id") or (record.get("labels") or {}).get("stream_id")
                or (record.get("metadata") or {}).get("stream_id")
            }
            monitor_stream_id = next(iter(record_stream_ids)) if len(record_stream_ids) == 1 else None
            started = perf_counter()
            results = score_records(batch, policy_bundle)
            elapsed = perf_counter() - started
            stream_source = f"上传 JSON · {upload.name}" if source == "上传 JSON" else "项目样例"
            if monitor_stream_id:
                score_drift = observe_score_drift(
                    [result["score"] for result in results], [result["window_id"] for result in results],
                    stream_source, monitor_stream_id,
                )
            elif record_stream_ids:
                stream_summaries = {}
                for identity in sorted(record_stream_ids):
                    stream_indices = [index for index, result in enumerate(results) if result.get("stream_id") == identity]
                    stream_summaries[identity] = observe_score_drift(
                        [results[index]["score"] for index in stream_indices],
                        [results[index]["window_id"] for index in stream_indices],
                        stream_source, identity,
                    )
                score_drift = (next(iter(stream_summaries.values())) if len(stream_summaries) == 1
                               else {"status": "multi_stream", "streams": stream_summaries,
                                     "persistence": "persistent"})
            else:
                score_drift = observe_score_drift(
                    [result["score"] for result in results], [result["window_id"] for result in results],
                    stream_source,
                )
            st.session_state["last_batch"] = results
            st.session_state["last_batch_inputs"] = batch
            st.session_state["last_detection_seconds"] = elapsed
            st.session_state["last_detection_meta"] = {
                "source": upload.name if source == "上传 JSON" else f"项目样例窗口 {sample_id}",
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "policy": detection_policy,
                "threshold": policy_bundle["threshold"],
                "score_drift": score_drift,
                "stream_id": monitor_stream_id,
                "stream_ids": sorted(record_stream_ids) if not stream_id_input else [stream_id_input],
            }
            st.session_state.pop("result_detail_index", None)
            st.session_state.pop("detection_explanation", None)
            stored = 0
            if persist_alerts:
                try:
                    stored = save_batch(results)
                except (sqlite3.Error, OSError, ValueError) as error:
                    stored = None
                    st.warning(f"模型检测已完成，但告警保存失败：{error}。结果仍可下载。")
            levels = {"仅 P1": {"P1"}, "P1 和 P2": {"P1", "P2"},
                      "所有异常": {"P1", "P2", "P3", "P4"}}[notify_policy_label]
            channels = ({"email"} if notify_email else set()) | ({"whatsapp"} if notify_wa else set())
            if channels and not DEMO_MODE:
                for channel, ok, detail in deliver_digest(results, channels, levels):
                    (st.success if ok else st.warning)(f"{channel}：{detail}")
            st.success(f"完成：处理 {len(results)} 个窗口，检出 {sum(r['detected'] for r in results)} 个异常。")
            st.caption(f"模型检测用时 {elapsed:.3f} 秒；批量窗口按输入顺序处理。")
            if persist_alerts and stored is not None:
                st.caption(f"已保存 {stored} 条新告警，可在“告警中心”确认或解决。")
        except Exception as error:
            st.error(f"检测失败：{error}")
            st.session_state.pop("last_batch", None)
    if st.session_state.get("last_batch"):
        batch_results = st.session_state["last_batch"]
        meta = st.session_state.get("last_detection_meta", {})
        st.caption(f"最近检测任务：{meta.get('source', '—')} · "
                   f"完成于 {display_time(meta.get('completed_at'))}（北京时间） · "
                   f"检测耗时 {st.session_state.get('last_detection_seconds', 0):.3f} 秒。")
        st.caption(f"本次策略：{('自定义校准' if meta.get('policy') == 'custom' else POLICY_ZH.get(meta.get('policy'), '均衡'))}。修改上方策略后，需重新运行检测才会生成新结果。")
        if meta.get("score_drift"):
            render_score_drift(meta["score_drift"])
            if st.button("重置当前漂移观察窗口", key="reset_live_score_drift"):
                for identity in meta.get("stream_ids", []):
                    reset_persisted_score_drift(identity, bundle.get("metadata", {}).get("model_id", "unknown"))
                st.session_state.pop("score_drift_state", None)
                meta["score_drift"] = None
                st.rerun()
        if st.button("清空当前检测结果", key="clear_detection_results"):
            for name in ("last_batch", "last_batch_inputs", "last_detection_meta", "last_detection_seconds", "result_detail_index", "detection_explanation"):
                st.session_state.pop(name, None)
            st.rerun()
        rows = []
        for index, result in enumerate(batch_results, start=1):
            alert = result.get("alert")
            rows.append({
                "窗口序号": index, "窗口编号": result.get("sample_id", "上传"),
                "检测结果": "异常" if result["detected"] else "正常",
                "异常评分": result["score"], "阈值": result["threshold"],
                "告警级别": severity_label(alert["severity"]) if alert else "—",
                "级别说明": alert["severity_reason"] if alert else "—",
                "可能原因": alert["cause_zh"] if alert else "—",
                "原因原始输出（未校准）": alert["confidence"] if alert else None,
                "候选原因数": len(alert.get("cause_candidates", [])) if alert else 0,
                "结构影子分数": (result.get("structure_shadow") or {}).get("score"),
                "结构影子提示": "影子信号" if (result.get("structure_shadow") or {}).get("shadow_flag") else "—",
                "需要复核": "是" if alert and alert.get("requires_review") else "否",
                "异常指标数": len(alert["affected_kpis"]) if alert else 0,
            })
        results_df = pd.DataFrame(rows)
        st.dataframe(results_df, width="stretch", hide_index=True, column_config=SCORE_COLUMNS)
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        export_csv, export_json, export_html = st.columns(3)
        export_csv.download_button(
            "下载检测摘要 CSV", data=dataframe_csv_bytes(results_df),
            file_name=f"ran-detection-{timestamp}.csv", mime="text/csv", width="stretch",
        )
        export_json.download_button(
            "下载完整结果 JSON",
            data=json.dumps(batch_results, ensure_ascii=False, indent=2),
            file_name=f"ran-detection-{timestamp}.json", mime="application/json", width="stretch",
        )
        export_html.download_button(
            "下载离线检测报告", data=detection_report_html(batch_results, metadata=meta,
                                                       model_metadata=bundle.get("metadata")),
            file_name=f"ran-report-{timestamp}.html", mime="text/html", width="stretch", key="download_html_report",
        )
        detail_index = st.selectbox(
            "查看窗口检测详情", range(len(batch_results)), key="result_detail_index",
            format_func=lambda i: f"第 {i + 1} 条 · 窗口 {batch_results[i].get('sample_id', '上传')} · "
                                  f"{'异常' if batch_results[i]['detected'] else '正常'}",
        )
        index = detail_index + 1
        result = batch_results[detail_index]
        alert = result.get("alert")
        inputs = st.session_state.get("last_batch_inputs", [])
        if detail_index < len(inputs):
            render_explanation(inputs[detail_index], "detection")
        if not alert:
            st.info(f"窗口 {index}：模型判断正常（{result['score']:.4f} < {result['threshold']:.4f}）。")
        else:
            st.warning(f"窗口 {index}：{severity_label(alert['severity'])} · {alert['cause_zh']} · "
                       f"原因原始输出（未校准）{alert['confidence']:.1%}")
            st.caption(f"等级说明：{alert['severity_reason']}")
            shadow = result.get("structure_shadow") or {}
            st.caption(f"结构关系影子分数：{shadow.get('score', 0):.3f} / {shadow.get('threshold', 0):.3f} · "
                       f"{'提示结构偏离' if shadow.get('shadow_flag') else '未超过影子阈值'}（不参与告警判定）")
            if shadow.get("top_relation_shifts"):
                st.dataframe(pd.DataFrame(shadow["top_relation_shifts"]), width="stretch", hide_index=True)
            advice_label = "候选原因建议（需复核）" if alert.get("requires_review") else "处理建议"
            st.write(f"{advice_label}：{alert['action']}（{alert['team']}）")
            render_cause_probabilities(alert, key=f"result_cause_distribution_{index}")
            if alert["affected_kpis"]:
                st.dataframe(pd.DataFrame(alert["affected_kpis"]).rename(columns={
                    "name": "指标", "current": "当前均值", "baseline": "正常基线", "zscore": "偏离标准差",
                }), width="stretch", hide_index=True)
            with st.expander(f"查看窗口 {index} 的告警 JSON"):
                st.json(result)

with tabs[4]:
    render_alert_center()

with tabs[5]:
    st.markdown("<div class='section-head'><div><div class='section-title'>模型评估与版本</div><div class='section-note'>查看当前留出集诊断、候选原因和影子通道表现</div></div></div>", unsafe_allow_html=True)
    metadata = bundle.get("metadata", {})
    st.caption(f"模型版本：{metadata.get('model_id', '—')} · 训练时间：{display_time(metadata.get('trained_at'))}（北京时间）")
    st.info("当前留出集已用于功能诊断。校准与影子通道指标用于开发比较，最终结论需要新的冻结测试集。")
    metric_columns = st.columns(6)
    for column, label, name in zip(metric_columns,
            ["检测精确率", "异常召回率", "检测 F1", "ROC-AUC", "平均精度 AP", "真实异常分类 Macro-F1"],
            ["test_precision", "test_recall", "test_f1", "test_auc", "test_average_precision", "cause_test_macro_f1"]):
        column.metric(label, f"{metrics.get(name, 0):.3f}")
    uncertainty = metrics.get("test_metric_uncertainty", {})
    uncertainty_metrics = uncertainty.get("metrics", {})
    if uncertainty_metrics:
        interval_text = " · ".join(
            f"{label} {uncertainty_metrics[key]['estimate']:.3f} "
            f"[95% CI {uncertainty_metrics[key]['ci95'][0]:.3f}–{uncertainty_metrics[key]['ci95'][1]:.3f}]"
            for key, label in (("precision", "Precision"), ("recall", "Recall"),
                               ("f1", "F1"), ("roc_auc", "ROC-AUC"),
                               ("average_precision", "AP"))
            if uncertainty_metrics.get(key, {}).get("ci95") is not None
        )
        if interval_text:
            st.caption(
                f"分组自助法不确定性：{interval_text}。"
                f"按 {uncertainty.get('independent_groups', 0)} 个不重叠时间组重采样；"
                "区间反映当前演示留出集的抽样不确定性，不代表现网置信度。"
            )
    pr_labels = (alerts_df["参考标签"] != CAUSE_ZH["Normal"]).astype(int).to_numpy()
    pr_scores = alerts_df["异常评分"].to_numpy(dtype=float)
    pr_precision, pr_recall, _ = precision_recall_curve(pr_labels, pr_scores)
    pr_figure = go.Figure(go.Scatter(
        x=pr_recall, y=pr_precision, mode="lines",
        name=f"PR 曲线 · AP {metrics.get('test_average_precision', 0):.3f}",
        line={"color": "#1769aa", "width": 3},
    ))
    pr_figure.add_trace(go.Scatter(
        x=[metrics.get("test_recall", 0)], y=[metrics.get("test_precision", 0)],
        mode="markers", name="验证集校准阈值",
        marker={"color": "#d25a4b", "size": 11, "symbol": "diamond"},
    ))
    pr_figure.add_hline(y=float(pr_labels.mean()), line_dash="dot", line_color="#94a3b8",
                        annotation_text=f"异常占比基线 {pr_labels.mean():.1%}")
    pr_figure.update_layout(
        title="留出集 Precision–Recall 曲线", xaxis_title="Recall", yaxis_title="Precision",
        xaxis_range=[0, 1], yaxis_range=[0, 1.04], height=340,
        paper_bgcolor="white", plot_bgcolor="white", font_color="#526259",
        margin=dict(l=12, r=18, t=52, b=12), legend=dict(orientation="h", y=-.2),
    )
    st.plotly_chart(pr_figure, width="stretch", key="test_pr_curve", config=PLOTLY_CONFIG)
    st.caption("标记点使用验证集校准的默认阈值；测试集曲线只用于独立展示，没有据此再调阈值。")
    st.info(metrics.get("split_note", "指标用于演示验证。"))
    st.caption("上方总览和混淆矩阵按默认“均衡”策略统计；下面的调优区会用所选验证集阈值重新复核测试集。")
    detector_calibration = metrics.get("detector_probability_diagnostics", {})
    cause_calibration = metrics.get("cause_probability_diagnostics", {})
    if detector_calibration and cause_calibration:
        st.markdown("#### 预测概率与置信度校准")
        calibrated_diagnostics = metrics.get("cause_calibrated_probability_diagnostics", {})
        calibration_columns = st.columns(6)
        calibration_columns[0].metric("检测 Brier 分数", f"{detector_calibration['brier_score']:.3f}")
        calibration_columns[1].metric("检测 Top-label ECE", f"{detector_calibration['top_label_ece']:.3f}")
        calibration_columns[2].metric("原因原始 Brier", f"{cause_calibration['brier_score']:.3f}")
        calibration_columns[3].metric("温度缩放 Brier", f"{calibrated_diagnostics.get('brier_score', float('nan')):.3f}")
        calibration_columns[4].metric("原因原始 ECE", f"{cause_calibration['top_label_ece']:.3f}")
        calibration_columns[5].metric("温度缩放 ECE", f"{calibrated_diagnostics.get('top_label_ece', float('nan')):.3f}")
        calibration_rows = []
        for task_name, diagnostic in (("异常检测", detector_calibration),
                                      ("真实异常原因·原始", cause_calibration),
                                      ("真实异常原因·温度缩放", calibrated_diagnostics)):
            for row in diagnostic["bins"]:
                calibration_rows.append({
                    "任务": task_name,
                    "置信度区间": f"{row['lower']:.1f}–{row['upper']:.1f}",
                    "窗口数": row["count"],
                    "平均置信度": row["mean_confidence"],
                    "实际准确率": row["accuracy"],
                })
        st.dataframe(pd.DataFrame(calibration_rows), hide_index=True, width="stretch",
                     column_config={key: st.column_config.NumberColumn(format="%.3f")
                                    for key in ("平均置信度", "实际准确率")})
        st.caption(
            f"检测校准基于 {detector_calibration['samples']} 个测试窗口；"
            f"故障分类校准基于 {cause_calibration['samples']} 个真实异常测试窗口。"
            "ECE 与 Brier 是小样本诊断，不用于自动修改置信度或阈值。"
        )
    cause_set = metrics.get("cause_prediction_set_diagnostics", {})
    calibration_fit = metrics.get("cause_calibration", {})
    if cause_set:
        st.markdown("#### 原因候选集合与拒绝判断")
        set_columns = st.columns(4)
        set_columns[0].metric("测试样本经验覆盖", f"{cause_set.get('empirical_coverage', 0):.1%}")
        set_columns[1].metric("平均候选数", f"{cause_set.get('mean_set_size', 0):.2f}")
        set_columns[2].metric("单一候选比例", f"{cause_set.get('singleton_rate', 0):.1%}")
        set_columns[3].metric("需要复核比例", f"{cause_set.get('review_rate', 0):.1%}")
        st.caption(f"校准方法：独立分组温度缩放 + APS 式集合 · APS 校准 {calibration_fit.get('calibration_samples', 0)} 个窗口 / "
                   f"{calibration_fit.get('independent_time_groups', 0)} 组 · 温度拟合 {calibration_fit.get('temperature_calibration_samples', 0)} 个窗口 / "
                   f"{calibration_fit.get('temperature_calibration_groups', 0)} 组 · "
                   f"状态：{'实验性校准' if calibration_fit.get('status') == 'experimental' else '校准证据不足'}。{calibration_fit.get('caveat', '')} "
                   "覆盖率是在已查看测试集上的经验值，没有未来覆盖率保证。")
    ranking = metrics.get("cause_candidate_ranking", {})
    if ranking:
        st.markdown(f"#### 固定 Top-{ranking.get('candidate_budget', 2)} 候选覆盖与排序")
        ranking_columns = st.columns(3)
        ranking_columns[0].metric("候选覆盖率", f"{ranking.get('candidate_coverage', 0):.1%}")
        ranking_columns[1].metric("Top-1 原因准确率", f"{ranking.get('top1_accuracy', 0):.1%}")
        ranking_columns[2].metric("命中候选后的 Top-1 排序率",
                                  f"{ranking.get('top1_accuracy_given_retrieved', 0):.1%}"
                                  if ranking.get('top1_accuracy_given_retrieved') is not None else "—")
        st.caption(ranking.get("interpretation", ""))
    shadow = metrics.get("structure_shadow_test", {})
    if shadow:
        st.markdown("#### 结构异常通道（影子模式）")
        shadow_columns = st.columns(5)
        for column, label, name in zip(shadow_columns,
                ["经验阈值", "Precision", "Recall", "F1", "ROC-AUC"],
                ["threshold", "precision", "recall", "f1", "roc_auc"]):
            value = shadow.get(name)
            column.metric(label, f"{value:.3f}" if value is not None else "—")
        st.caption(f"正常训练原型：{metrics.get('structure_shadow', {}).get('prototype_count', 0)} 个 · "
                   f"仅由正常训练窗口拟合，验证正常窗口设阈值。该通道不影响线上告警，需经过留出故障家族、正常工况变化与同误报预算消融后才可晋级。")
    confusion = metrics.get("detector_confusion")
    if confusion:
        tn, fp, fn, tp = np.asarray(confusion).ravel()
        st.caption(f"正常判对 {tn} · 误报 {fp} · 漏检 {fn} · 异常检出 {tp}。阈值只从验证集选取，当前值 {bundle['threshold']:.4f}。")
        columns = st.columns(2)
        short_cause_names = {"Congestion": "拥塞", "Equipment": "设备", "Interference": "干扰", "Mobility": "移动性"}
        with columns[0]:
            detector_plot = px.imshow(np.asarray(confusion), x=["预测正常", "预测异常"],
                                       y=["实际正常", "实际异常"], text_auto=True,
                                       color_continuous_scale=["#f3f4ee", "#cadcc9", "#7da38a", "#315c49"], aspect="auto", labels={"color": "窗口数"})
            detector_plot.update_layout(title="异常检测混淆矩阵", height=360, margin=dict(t=55, b=20, l=20, r=20))
            st.plotly_chart(detector_plot, width="stretch", key="detector_confusion_chart", config=PLOTLY_CONFIG)
        with columns[1]:
            cause_names = [CAUSE_ZH[name] for name in bundle["classes"]]
            cause_plot = px.imshow(np.asarray(metrics["cause_confusion"]),
                                    x=[short_cause_names[name] for name in bundle["classes"]],
                                    y=cause_names, text_auto=True,
                                    color_continuous_scale=["#f3f4ee", "#cadcc9", "#7da38a", "#315c49"], aspect="auto", labels={"color": "窗口数"})
            cause_plot.update_layout(title="故障原因分类混淆矩阵", xaxis_title="预测原因（简称）", yaxis_title="实际原因",
                                     xaxis_tickangle=0,
                                     height=360, margin=dict(t=55, b=20, l=20, r=20))
            st.plotly_chart(cause_plot, width="stretch", key="cause_confusion_chart", config=PLOTLY_CONFIG)
        st.caption("根因矩阵使用测试集内所有真实异常窗口单独评估第二阶段分类器，包含第一阶段未检出的异常。")
        pipeline = metrics.get("pipeline")
        if pipeline:
            st.markdown("#### 两阶段端到端结果")
            pipeline_cols = st.columns(5)
            pipeline_cols[0].metric("端到端故障 Macro-F1", f"{pipeline['fault_macro_f1']:.3f}")
            pipeline_cols[1].metric("原因正确预测", pipeline["correctly_classified_faults"])
            pipeline_cols[2].metric("原因误分类", pipeline["misclassified_detected_faults"])
            pipeline_cols[3].metric("第一阶段漏检", pipeline["missed_faults"])
            pipeline_cols[4].metric("正常窗口误报", pipeline["false_alarms"])
            pipeline_plot = px.imshow(np.asarray(pipeline["confusion"]),
                                      x=[short_cause_names.get(name, name) for name in pipeline["classes"]],
                                      y=[CAUSE_ZH.get(name, name) for name in pipeline["classes"]],
                                      text_auto=True, color_continuous_scale=["#f3f4ee", "#cadcc9", "#7da38a", "#315c49"], aspect="auto",
                                      labels={"color": "窗口数"})
            pipeline_plot.update_layout(title="两阶段端到端混淆矩阵（测试集）",
                                        xaxis_title="最终预测（故障类别用简称）", yaxis_title="实际类别",
                                        xaxis_tickangle=0, height=390, margin=dict(t=55, b=30, l=20, r=20))
            st.plotly_chart(pipeline_plot, width="stretch", key="pipeline_confusion_chart",
                            config=PLOTLY_CONFIG)
            st.dataframe(pd.DataFrame([
                {"实际类别": CAUSE_ZH.get(name, name), "精确率": row["precision"],
                 "召回率": row["recall"], "F1": row["f1"], "测试窗口数": row["support"]}
                for name, row in pipeline["per_class"].items()
            ]), width="stretch", hide_index=True,
                column_config={key: st.column_config.NumberColumn(format="%.3f")
                               for key in ("精确率", "召回率", "F1")})
            st.caption("这张矩阵按实际运行顺序统计：第一阶段漏检计为正常，只有检出的窗口才会调用根因分类器；测试集未参与阈值选择。")
    partitions = metrics.get("partitions", {})
    if partitions:
        st.markdown("#### 数据划分与类别覆盖")
        st.dataframe(pd.DataFrame([
            {"集合": {"train": "训练集", "validation": "验证集", "test": "测试集"}[name],
             "窗口数": partition["size"], "不重叠时间组": partition["time_groups"],
             **{CAUSE_ZH[cause]: count for cause, count in partition["classes"].items()}}
            for name, partition in partitions.items()
        ]), hide_index=True, width="stretch")
    temporal_audit = metrics.get("temporal_coverage_audit", {})
    if temporal_audit:
        st.markdown("#### 时间覆盖与未来评估适用性")
        contract_rows = temporal_audit.get("data_contract_coverage", {})
        if contract_rows:
            st.dataframe(pd.DataFrame([
                {"数据契约字段": {"sampling_rate": "采样率", "measurement_units": "测量单位",
                                  "schema_version": "模式版本", "data_source": "来源标注"}.get(name, name),
                 "覆盖窗口": row["windows"], "覆盖率": row["fraction"],
                 "完整": "是" if row["complete"] else "否"}
                for name, row in contract_rows.items()
            ]), hide_index=True, width="stretch",
                column_config={"覆盖率": st.column_config.NumberColumn(format="percent")})
        if not temporal_audit.get("forward_evaluation_data_eligible"):
            st.warning(
                "当前数据缺少完整的基站/数据流 ID 或异常事件 ID，无法证明时间组对应独立事件，"
                "也无法把全局时间切分解释为同一网络实体的未来泛化测试。"
                "上方测试集采用不重叠时间组的分层留出，适合功能验证，不应表述为现网未来预测成绩。"
            )
        else:
            st.info(
                "数据包含前向评测所需的实体/事件标识和最新时间块异常；"
                "当前模型仍采用按重叠时间组的分层留出，须另跑时间向前留出的评测后才能用于晋级审查。"
            )
        blocks = temporal_audit.get("chronological_blocks", [])
        if blocks:
            causes = sorted({cause for block in blocks for cause in block["class_windows"]})
            st.dataframe(pd.DataFrame([
                {"时间块": f"第 {block['block']} 块", "开始时间": block["time_start"],
                 "结束时间": block["time_end"], "重叠窗口组": block["time_groups"],
                 "窗口数": block["windows"],
                 **{CAUSE_ZH.get(cause, cause): block["class_windows"].get(cause, 0) for cause in causes}}
                for block in blocks
            ]), hide_index=True, width="stretch")
        st.caption(
            f"{temporal_audit.get('overlap_groups', 0)} 个不重叠时间组；"
            f"单窗口中位时长 {temporal_audit.get('median_window_duration_seconds', 0):.1f} 秒。"
            "时间块仅用于检查类别覆盖；分组置信区间不替代按基站和事件 ID 切分的前向评估。"
        )
    readiness = metrics.get("promotion_readiness", {})
    if readiness:
        st.markdown("#### 模型晋级证据门禁")
        if readiness.get("promotion_allowed"):
            st.success("评测证据达到当前晋级评审门槛；这仍需业务审批和灰度验证。")
        else:
            st.warning("当前评测证据不足，建议暂缓模型晋级。")
        st.dataframe(pd.DataFrame([
            {"检查项": row["key"], "结果": "通过" if row["passed"] else "未通过",
             "说明": row["message"]}
            for row in readiness.get("checks", [])
        ]), hide_index=True, width="stretch")
        st.caption("门禁只检查数据与评测证据是否完整，不替代业务风险评审、灰度发布或人工批准。")
    origin_audit = metrics.get("origin_audit", {})
    if origin_audit:
        st.markdown("#### 测试集：按异常来源审计")
        if origin_audit.get("status") == "missing_origin":
            st.warning(
                f"测试集 {origin_audit.get('unknown_windows', 0)} 个窗口都缺少来源标签，"
                "因此当前无法比较合成数据、受控真实数据和现网数据上的效果。"
                "请在每条记录中增加 anomaly_origin（synthetic、controlled_real 或 real）；系统不会根据故障类别推断来源。"
            )
        origin_names = {"synthetic": "合成/模拟", "controlled_real": "受控真实", "real": "真实现场/现网", "unknown": "来源未知"}
        origin_rows = []
        for group in origin_audit.get("groups", []):
            origin_rows.append({
                "来源": origin_names.get(group["origin"], group["origin"]),
                "窗口数": group["windows"], "正常窗口": group["normal_windows"],
                "异常窗口": group["anomaly_windows"], "精确率": group["precision"],
                "召回率": group["recall"], "F1": group["f1"], "误报率": group["false_positive_rate"],
            })
        if origin_rows:
            st.dataframe(pd.DataFrame(origin_rows), hide_index=True, width="stretch",
                         column_config={key: st.column_config.NumberColumn(format="%.3f")
                                        for key in ("精确率", "召回率", "F1", "误报率")})
            st.caption("来源分组仅使用数据中显式提供的 provenance 字段；来源未知的样本单独列出，不计入已知来源结论。")
    st.download_button("导出模型评估 JSON", data=json.dumps(metrics | {"model_metadata": metadata}, ensure_ascii=False, indent=2),
                       file_name="ran-model-evaluation.json", mime="application/json", key="export_evaluation")
    st.markdown("#### 验证集阈值调优与独立测试")
    profiles = bundle.get("threshold_profiles", {})
    if profiles:
        st.dataframe(pd.DataFrame([
            {"策略": POLICY_ZH[name], "阈值": profile["threshold"],
             "验证精确率": profile["validation_precision"], "验证召回率": profile["validation_recall"],
             "目标达成": "是" if profile["target_met"] else "否"}
            for name, profile in profiles.items()
        ]), width="stretch", hide_index=True)
    tuner_columns = st.columns([2, 1, 1])
    calibration_objective = tuner_columns[0].selectbox(
        "调优目标", ["fpr", "recall"], format_func=lambda value: "限制误报率" if value == "fpr" else "保证异常召回率",
        key="calibration_goal",
    )
    if calibration_objective == "fpr":
        target_percent = tuner_columns[1].selectbox("目标误报率上限", [1, 2, 5, 10, 20], index=2,
                                                    format_func=lambda value: f"{value}%", key="calibration_target_fpr")
        target = target_percent / 100
    else:
        target_percent = tuner_columns[1].selectbox("最低召回率", [80, 90, 95, 98], index=1,
                                                    format_func=lambda value: f"{value}%", key="calibration_target_recall")
        target = target_percent / 100
    calibration_data = bundle.get("calibration", {})
    calibrated = calibrate_threshold(calibration_data["labels"], calibration_data["scores"],
                                     calibration_objective, target)
    target_description = "误报率上限" if calibration_objective == "fpr" else "最低召回率"
    status = "已达到目标" if calibrated["target_met"] else "验证集无法达到目标，以下为最接近结果"
    st.caption(f"基于验证集 {len(calibration_data['labels'])} 个窗口校准：阈值 {calibrated['threshold']:.4f} · "
               f"精确率 {calibrated['precision']:.1%} · 召回率 {calibrated['recall']:.1%} · "
               f"误报率 {calibrated['false_positive_rate']:.1%} · {status}（{target_description} {target:.0%}）。")
    test_threshold = calibrated["threshold"]
    actual = (alerts_df["参考标签"] != "正常").to_numpy()
    test_scores = alerts_df["异常评分"].to_numpy(dtype=float)
    predicted = test_scores >= test_threshold
    with st.expander("查看按该阈值复核独立测试集", expanded=True):
        true_positive = int(np.sum(actual & predicted))
        false_positive = int(np.sum(~actual & predicted))
        false_negative = int(np.sum(actual & ~predicted))
        true_negative = int(np.sum(~actual & ~predicted))
        test_precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
        test_recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
        test_f1 = 2 * test_precision * test_recall / (test_precision + test_recall) if test_precision + test_recall else 0.0
        test_fpr = false_positive / (false_positive + true_negative) if false_positive + true_negative else 0.0
        preview_cols = st.columns(7)
        for column, label, value in zip(preview_cols,
                ["测试精确率", "测试召回率", "测试 F1", "测试误报率", "异常检出", "漏检", "误报"],
                [f"{test_precision:.3f}", f"{test_recall:.3f}", f"{test_f1:.3f}", f"{test_fpr:.3f}",
                 str(true_positive), str(false_negative), str(false_positive)]):
            column.metric(label, value)
        st.caption(f"独立测试集共 {len(actual)} 个窗口；该集合只用于复核验证集选出的阈值，没有参与阈值选择。测试阈值 {test_threshold:.4f}。")
        apply_columns = st.columns(2)
        if apply_columns[0].button("应用到实时检测策略", key="apply_calibrated_threshold", type="primary"):
            st.session_state["custom_detection_threshold"] = calibrated | {
                "validation_precision": calibrated["precision"],
                "validation_recall": calibrated["recall"],
            }
            st.session_state["calibrated_threshold_notice"] = True
            st.rerun()
        if st.session_state.pop("calibrated_threshold_notice", False):
            st.success("已保存到本次会话。“实时检测演示”现在可以选择“自定义校准”。")
        if "custom_detection_threshold" in st.session_state and apply_columns[1].button(
                "清除本次自定义策略", key="clear_custom_threshold"):
            del st.session_state["custom_detection_threshold"]
            st.rerun()
        mistakes = alerts_df.loc[actual != predicted, ["窗口编号", "参考标签", "异常评分"]].copy()
        if not mistakes.empty:
            mistakes["问题"] = np.where((mistakes["参考标签"] == "正常"), "误报", "漏检")
            st.dataframe(mistakes, hide_index=True, width="stretch", column_config=SCORE_COLUMNS)
            st.caption("可在“窗口检查”中按上述编号查看 KPI 曲线与模型特征贡献。")
        else:
            st.success("该阈值下，这个测试子集没有误报或漏检。")

    st.markdown("#### 与无监督基线比较")
    st.caption("同一训练/验证/测试分组；两个检测器都只按验证集目标召回率选阈值，测试集只用于最终比较。")
    if st.button("运行 XGBoost 与 Isolation Forest 对比", key="run_detector_comparison"):
        with st.spinner("正在拟合正常样本基线并计算分组置信区间…"):
            st.session_state["detector_comparison"] = compare_with_isolation_forest(records, bundle)
    comparison = st.session_state.get("detector_comparison")
    if comparison and comparison.get("model_id") == metadata.get("model_id"):
        comparison_rows = []
        model_labels = {
            "xgboost_supervised": "XGBoost（监督式）",
            "isolation_forest_normal_only": "Isolation Forest（仅正常样本训练）",
        }
        for model_name, report in comparison["models"].items():
            test = report["test"]
            confusion_row = np.asarray(test["confusion"])
            interval = test["group_bootstrap_95ci"]["metrics"]["f1"]["ci95"]
            comparison_rows.append({
                "检测器": model_labels[model_name],
                "验证阈值": report["calibration"]["threshold"],
                "测试精确率": test["precision"], "测试召回率": test["recall"],
                "测试 F1": test["f1"], "ROC-AUC": test["roc_auc"], "AP": test["average_precision"],
                "F1 95% 分组 CI": f"{interval[0]:.3f}–{interval[1]:.3f}" if interval else "—",
                "误报": int(confusion_row[0, 1]), "漏检": int(confusion_row[1, 0]),
                "推理毫秒/窗口": report["test_inference_ms_per_window"],
            })
        st.dataframe(pd.DataFrame(comparison_rows), hide_index=True, width="stretch",
                     column_config={key: st.column_config.NumberColumn(format="%.3f")
                                    for key in ("验证阈值", "测试精确率", "测试召回率", "测试 F1",
                                                "ROC-AUC", "AP", "推理毫秒/窗口")})
        st.caption(
            f"Isolation Forest 仅使用 {comparison['models']['isolation_forest_normal_only']['fit_normal_windows']} 个正常训练窗口拟合；"
            f"阈值目标召回率 {comparison['target_validation_recall']:.0%}。"
            "推理时间为本机 CPU 上 152 个测试窗口、预热后重复测量的中位数；样本很小，比较结果仅作实验参考。"
        )

with tabs[6]:
    st.markdown("<div class='section-head'><div><div class='section-title'>连续检测场景回放</div><div class='section-note'>按固定间隔重放项目窗口，观察检测评分与告警变化</div></div></div>", unsafe_allow_html=True)
    st.caption("回放使用项目公开样例，按窗口逐个推理；它是演示数据流。回放结束后停止，可随时暂停或继续。")
    replay = st.session_state.get("replay_run")
    running = bool(replay and replay.state == "running")
    setting_columns = st.columns(3)
    replay_cause = setting_columns[0].selectbox("回放场景", [None, "Normal"] + list(ACTION),
                                              format_func=lambda cause: "混合场景" if cause is None else CAUSE_ZH[cause],
                                              key="replay_cause", disabled=running,
                                              placeholder="混合场景")
    replay_count = setting_columns[1].selectbox("回放窗口数", [10, 20, 50, 100], index=1, key="replay_count", disabled=running)
    replay_interval = setting_columns[2].selectbox("窗口间隔（秒）", [2, 5, 10], index=1, key="replay_interval", disabled=running)
    replay_persist = st.checkbox("把回放中的异常保存到告警中心", value=False, key="replay_persist", disabled=running)
    button_columns = st.columns(3)
    if button_columns[0].button("开始新回放", key="replay_start", type="primary", disabled=running):
        replay = ReplayRun(replay_windows(records, replay_count, replay_cause), interval=float(replay_interval), persist=replay_persist)
        replay.resume()
        st.session_state["replay_run"] = replay
        st.rerun()
    if button_columns[1].button("暂停回放", key="replay_pause", disabled=not running):
        replay.pause()
        st.rerun()
    if button_columns[2].button("继续回放", key="replay_resume", disabled=not replay or replay.state != "paused"):
        replay.resume()
        st.rerun()

    @st.fragment(run_every=1 if replay and replay.state == "running" else None)
    def replay_panel():
        current = st.session_state.get("replay_run")
        if not current:
            st.info("选择场景后点击“开始新回放”。默认只预览结果；勾选保存后才写入告警中心。")
            return
        try:
            fresh = current.advance(bundle, perf_counter())
        except (ValueError, RuntimeError) as error:
            st.error(f"回放已停止：{error}")
            return
        if fresh:
            st.session_state["replay_score_drift"] = observe_score_drift(
                [fresh["score"]], [fresh["window_id"]], "连续场景回放"
            )
        if fresh and current.persist:
            try:
                current.saved += save_batch([fresh])
            except (OSError, sqlite3.Error, ValueError) as error:
                current.save_error = str(error)
        status = {"idle": "待开始", "running": "运行中", "paused": "已暂停", "completed": "已完成", "failed": "已停止"}[current.state]
        cols = st.columns(4)
        cols[0].metric("回放状态", status)
        cols[1].metric("已处理窗口", f"{current.cursor} / {len(current.records)}")
        cols[2].metric("检出异常", sum(result["detected"] for result in current.results))
        cols[3].metric("已保存告警", current.saved)
        st.progress(current.cursor / len(current.records))
        replay_drift = st.session_state.get("replay_score_drift")
        if replay_drift:
            render_score_drift(replay_drift, "回放分数漂移观察")
        if current.save_error:
            st.warning(f"部分告警未保存：{current.save_error}。回放结果仍可导出。")
        if current.results:
            frame = pd.DataFrame([
                {"序号": index + 1, "窗口": result["sample_id"], "异常评分": result["score"],
                 "检测阈值": result["threshold"], "判定": "异常" if result["detected"] else "正常",
                 "级别": (result.get("alert") or {}).get("severity", "—"),
                 "原因": (result.get("alert") or {}).get("cause_zh", "—"),
                 "推理用时（毫秒）": round(current.durations[index] * 1000, 1)}
                for index, result in enumerate(current.results)
            ])
            curve = go.Figure()
            curve.add_trace(go.Scatter(x=frame["序号"], y=frame["异常评分"], mode="lines+markers", name="异常评分", line_color="#548c70"))
            curve.add_hline(y=bundle["threshold"], line_dash="dash", line_color="#d83d4d", annotation_text="检测阈值")
            curve.update_layout(height=300, xaxis_title="回放窗口序号", yaxis_title="异常评分", yaxis_range=[0, 1.05],
                                paper_bgcolor="white", plot_bgcolor="white", margin=dict(t=20, b=20, l=20, r=20))
            st.plotly_chart(curve, width="stretch", key="replay_score_curve", config=PLOTLY_CONFIG)
            st.dataframe(frame.iloc[::-1], hide_index=True, width="stretch", column_config=SCORE_COLUMNS)
            st.download_button("下载当前回放结果 JSON", data=json.dumps(current.results, ensure_ascii=False, indent=2, allow_nan=False),
                               file_name="ran-replay-results.json", mime="application/json", key="download_replay_results")
        if fresh and current.state == "completed":
            st.rerun()

    replay_panel()
