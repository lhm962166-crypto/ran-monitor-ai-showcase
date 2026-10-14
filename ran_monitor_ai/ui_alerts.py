"""Alert center UI; database failures do not interrupt other pages."""
from __future__ import annotations
import json
import sqlite3
from datetime import datetime
import pandas as pd
import streamlit as st
from .engine import POLICY_ZH
from .exporting import dataframe_csv_bytes
from .settings import ACTION, CAUSE_ZH
from .store import (add_feedback, add_note, alert_events, backup_database, export_alerts,
                    feedback_rows, history_summary, incident_events, query_alerts,
                    query_incident, update_incident_status, update_status)
from .train import ROOT
from .ui_common import SCORE_COLUMNS, alert_status_label, display_time, history_frame, render_cause_probabilities, severity_label


def render_alert_center():
    try:
        _render_alert_center()
    except (ValueError, OSError, sqlite3.Error) as error:
        st.error(f"告警中心暂不可用：{error}。可继续使用检测、模型评估与回放。")


def _render_alert_center():
    st.markdown("<div class='section-head'><div><div class='eyebrow'>ALERT OPERATIONS</div><div class='section-title'>告警中心</div><div class='section-note'>管理本机保存的告警，记录确认、解决和重新打开的操作</div></div></div>", unsafe_allow_html=True)
    try:
        summary = history_summary()
    except (sqlite3.Error, OSError, ValueError) as error:
        st.error(f"告警数据库暂不可用：{error}。实时检测仍可使用，结果可直接下载。")
        return
    metric_columns = st.columns(8)
    for column, label, value in zip(metric_columns,
            ["窗口告警", "待处理", "已确认", "已解决", "未解决 P1", "活跃事件", "事件已确认", "已恢复事件"],
            [summary["total"], summary["OPEN"], summary["ACKNOWLEDGED"], summary["RESOLVED"],
             summary["critical_unresolved"], summary["open_incidents"], summary["acknowledged_incidents"],
             summary["recovered_incidents"]]):
        column.metric(label, value)
    if st.button("刷新告警历史", key="refresh_history"):
        st.session_state.pop("history_export", None)
        st.rerun()
    if st.session_state.get("history_feedback"):
        st.success(st.session_state.pop("history_feedback"))
    with st.expander("备份本机告警数据"):
        st.caption("备份包含所有告警和处置记录；使用数据库快照，保存期间的新记录也不会影响备份文件的完整性。")
        if st.button("生成数据库备份", key="prepare_database_backup"):
            name = f"ran-alerts-backup-{datetime.now():%Y%m%d-%H%M%S}.sqlite3"
            target = backup_database(ROOT / "artifacts" / "backups" / name)
            st.session_state["database_backup"] = {"name": name, "data": target.read_bytes()}
        backup = st.session_state.get("database_backup")
        if backup:
            st.download_button("下载告警数据库备份", data=backup["data"], file_name=backup["name"],
                               mime="application/octet-stream", key="download_database_backup")
            st.caption("下载文件是生成备份时的快照。需要更新数据时请重新生成。")
    filter1, filter2, filter3 = st.columns([1.5, 1, 1.5])
    search = filter1.text_input("搜索告警编号或窗口", key="history_search", max_chars=100)
    status = filter2.selectbox("处置状态", [None, "OPEN", "ACKNOWLEDGED", "RESOLVED"],
                              format_func=lambda value: "全部状态" if value is None else alert_status_label(value),
                              key="history_status")
    history_levels = filter3.multiselect("历史告警级别", ["P1", "P2", "P3", "P4"],
                                         default=["P1", "P2", "P3", "P4"], key="history_levels")
    causes = st.multiselect("历史故障原因", list(ACTION), default=list(ACTION),
                            format_func=lambda value: CAUSE_ZH[value], key="history_causes")
    page_size = st.selectbox("每页显示", [20, 50, 100], key="history_page_size")
    signature = (search, status, tuple(history_levels), tuple(causes), page_size)
    if st.session_state.get("history_filter_signature") != signature:
        st.session_state["history_filter_signature"] = signature
        st.session_state["history_page"] = 1
        st.session_state.pop("history_selection", None)
    filters = {"status": status, "severities": history_levels, "causes": causes, "search": search}
    _, count = query_alerts(limit=1, **filters)
    page_count = max(1, (count + page_size - 1) // page_size)
    st.session_state["history_page"] = min(st.session_state.get("history_page", 1), page_count)
    page = st.number_input("页码", min_value=1, max_value=page_count, step=1, key="history_page")
    saved, count = query_alerts(limit=page_size, offset=(page - 1) * page_size, **filters)
    st.caption(f"匹配 {count} 条告警 · 第 {page} / {page_count} 页 · 时间均为北京时间。")
    export_signature = signature + (summary["total"], summary["OPEN"], summary["ACKNOWLEDGED"], summary["RESOLVED"])
    if st.session_state.get("history_export", {}).get("signature") != export_signature:
        st.session_state.pop("history_export", None)
    if count and st.button("准备完整筛选结果导出", key="prepare_history_export"):
        exported, matched = export_alerts(**filters)
        st.session_state["history_export"] = {"signature": export_signature,
                                               "data": dataframe_csv_bytes(history_frame(exported)),
                                               "size": len(exported), "matched": matched}
    prepared = st.session_state.get("history_export")
    if prepared:
        if prepared["matched"] > prepared["size"]:
            st.warning(f"匹配 {prepared['matched']} 条；本次导出最近 {prepared['size']} 条。可进一步筛选缩小范围。")
        st.download_button(f"下载筛选结果 CSV（{prepared['size']} 条）", data=prepared["data"],
                           file_name=f"ran-history-filtered-{datetime.now():%Y%m%d-%H%M%S}.csv",
                           mime="text/csv", key="download_history_filtered")
    if saved:
        frame = history_frame(saved)
        st.dataframe(frame, width="stretch", hide_index=True, column_config=SCORE_COLUMNS)
        st.download_button("导出当前页告警 CSV", data=dataframe_csv_bytes(frame),
                           file_name=f"ran-history-{datetime.now():%Y%m%d-%H%M%S}.csv",
                           mime="text/csv", key="export_history_page")
        by_id = {alert["alert_id"]: alert for alert in saved}
        selected_id = st.selectbox("选择告警查看详情与处置", list(by_id), key="history_selection",
                                   format_func=lambda value: f"{value[:18]}… · 窗口 {by_id[value]['sample_id']} · "
                                                             f"{alert_status_label(by_id[value]['status'])}")
        chosen_alert = by_id[selected_id]
        payload = chosen_alert["payload"]
        st.caption(f"告警编号：{selected_id} · 当前状态：{alert_status_label(chosen_alert['status'])}")
        if chosen_alert.get("incident_id"):
            incident = query_incident(chosen_alert["incident_id"])
            incident_status = incident.get("status") if incident else "未知"
            status_label = {"OPEN": "待确认", "ACKNOWLEDGED": "已确认", "RECOVERED": "连续正常窗口确认恢复", "RESOLVED": "已关闭"}.get(incident_status, incident_status)
            st.caption(f"持续事件：{chosen_alert['incident_id']} · 事件状态：{status_label} · "
                       f"关联窗口：{incident.get('window_count', 0) if incident else 0} · "
                       f"数据流：{chosen_alert.get('stream_id') or '未提供；该告警独立记录'}")
            if incident:
                incident_actions = st.columns(2)
                if incident_actions[0].button("确认持续事件", key=f"incident_ack_{chosen_alert['incident_id']}",
                                               disabled=incident_status != "OPEN"):
                    try:
                        update_incident_status(chosen_alert["incident_id"], "ACKNOWLEDGED")
                        st.session_state["history_feedback"] = "持续事件已确认。"
                        st.rerun()
                    except (ValueError, OSError, sqlite3.Error) as error:
                        st.error(str(error))
                if incident_actions[1].button("关闭持续事件", key=f"incident_resolve_{chosen_alert['incident_id']}",
                                               disabled=incident_status not in {"OPEN", "ACKNOWLEDGED", "RECOVERED"}):
                    try:
                        update_incident_status(chosen_alert["incident_id"], "RESOLVED")
                        st.session_state["history_feedback"] = "持续事件已关闭；窗口告警的处置状态仍独立保留。"
                        st.rerun()
                    except (ValueError, OSError, sqlite3.Error) as error:
                        st.error(str(error))
        if payload.get("detection_threshold") is not None:
            st.caption(f"生成时策略：{POLICY_ZH.get(payload.get('detection_policy'), '均衡')} · "
                       f"阈值 {payload['detection_threshold']:.4f} · 模型版本 {payload.get('model_id', '—')}")
        if payload.get("severity_reason"):
            st.info(f"{severity_label(chosen_alert['severity'])} · {payload['severity_reason']}")
        if payload.get("action"):
            advice_label = "候选原因建议（需复核）" if payload.get("requires_review") else "处置建议"
            st.write(f"{advice_label}：{payload['action']}（{payload.get('team', '—')}）")
        render_cause_probabilities(payload, key="history_cause_distribution")
        if st.session_state.pop("history_note_reset", None) == selected_id:
            st.session_state.pop(f"history_note_{selected_id}", None)
        note = st.text_area("处置备注（可选）", max_chars=2000, key=f"history_note_{selected_id}",
                             placeholder="记录已核实的情况、处理步骤或重新打开的原因")
        if st.button("保存排查备注", key="history_add_note", disabled=not note.strip()):
            try:
                if add_note(selected_id, note):
                    st.session_state["history_feedback"] = "排查备注已保存，告警状态保持不变。"
                    st.session_state["history_note_reset"] = selected_id
                    st.rerun()
                else:
                    st.warning("该告警已不存在，请刷新后重试。")
            except (ValueError, OSError, sqlite3.Error) as error:
                st.error(f"备注未保存：{error}")
        action_columns = st.columns(3)
        actions = [("确认告警", "ACKNOWLEDGED", chosen_alert["status"] != "OPEN"),
                   ("标记已解决", "RESOLVED", chosen_alert["status"] == "RESOLVED"),
                   ("重新打开", "OPEN", chosen_alert["status"] != "RESOLVED")]
        for column, (label, target, disabled) in zip(action_columns, actions):
            if column.button(label, key=f"history_action_{target}", disabled=disabled):
                try:
                    if update_status(selected_id, target, note=note):
                        st.session_state["history_feedback"] = f"告警已更新为{alert_status_label(target)}，操作已记录。"
                        st.session_state["history_note_reset"] = selected_id
                        st.rerun()
                    else:
                        st.warning("该告警已不存在，请刷新后重试。")
                except (ValueError, OSError, sqlite3.Error) as error:
                    st.error(str(error))
        events = alert_events(selected_id)
        st.markdown("#### 处置记录")
        if events:
            st.dataframe(pd.DataFrame([
                {"时间（北京时间）": display_time(event["created_at"]),
                 "操作": {"CREATED": "生成告警", "STATUS_CHANGED": "变更状态", "NOTE": "排查备注",
                          "FEEDBACK": "人工核实原因"}.get(event["event_type"], event["event_type"]),
                 "原状态": alert_status_label(event["previous_status"]) if event["previous_status"] else "生成",
                 "新状态": alert_status_label(event["status"]), "备注": event["note"]}
                for event in events
            ]), width="stretch", hide_index=True)
        else:
            st.caption("这是旧版本保存的告警，尚无处置操作记录。")
        if chosen_alert.get("incident_id"):
            event_rows = incident_events(chosen_alert["incident_id"])
            with st.expander("持续事件时间线"):
                if event_rows:
                    st.dataframe(pd.DataFrame([{
                        "时间（北京时间）": display_time(event["created_at"]),
                        "事件": {"WINDOW_DETECTED": "检测到异常窗口", "RECOVERY_CONFIRMED": "连续正常窗口确认恢复",
                                 "STATUS_CHANGED": "事件状态变更"}.get(event["event_type"], event["event_type"]),
                        "详情": json.dumps(event["payload"], ensure_ascii=False),
                    } for event in event_rows]), width="stretch", hide_index=True)
                else:
                    st.caption("暂无事件时间线。")
        with st.expander("记录人工核实原因"):
            st.caption("反馈追加到本机审计库并关联模型版本；不会直接修改模型或把处置完成等同于原因正确。")
            options = ["undetermined", "Congestion", "Equipment", "Interference", "Mobility"]
            confirmed = st.selectbox("核实原因", options,
                                     format_func=lambda value: "无法确认" if value == "undetermined" else CAUSE_ZH[value],
                                     key=f"feedback_cause_{selected_id}")
            confidence_value = st.selectbox("反馈可信度", ["high", "medium", "low", "undetermined"],
                                            format_func=lambda value: {"high": "高", "medium": "中", "low": "低", "undetermined": "无法评估"}[value],
                                            disabled=confirmed == "undetermined", key=f"feedback_confidence_{selected_id}")
            reviewer = st.text_input("复核人（可选）", max_chars=120, key=f"feedback_reviewer_{selected_id}")
            feedback_note = st.text_area("核实依据", max_chars=2000, key=f"feedback_note_{selected_id}")
            if st.button("保存核实记录", key=f"save_feedback_{selected_id}", disabled=not feedback_note.strip()):
                try:
                    confidence_value = "undetermined" if confirmed == "undetermined" else confidence_value
                    if add_feedback(selected_id, confirmed, confidence_value, reviewer, feedback_note):
                        st.session_state["history_feedback"] = "人工核实记录已追加；模型版本与复核信息已留档。"
                        st.rerun()
                    st.warning("该告警已不存在，请刷新后重试。")
                except (ValueError, OSError, sqlite3.Error) as error:
                    st.error(f"核实记录未保存：{error}")
            current_feedback = [item for item in feedback_rows() if item["alert_id"] == selected_id]
            if current_feedback:
                st.dataframe(pd.DataFrame(current_feedback).drop(columns=["alert_id"]),
                             width="stretch", hide_index=True)
        with st.expander("查看完整告警 JSON"):
            st.json(payload)
        st.download_button("下载告警与处置记录 JSON", data=json.dumps(chosen_alert | {"events": events}, ensure_ascii=False, indent=2),
                           file_name=f"{selected_id}.json", mime="application/json", key="export_alert_audit")
    else:
        st.info("当前没有匹配的告警。可调整筛选，或到实时检测页勾选保存并运行一个异常窗口。")

    all_feedback = feedback_rows()
    with st.expander("导出人工核实数据"):
        st.caption(f"共 {len(all_feedback)} 条追加式原因反馈，可作为后续标注审查材料；导出内容不会自动进入训练集。")
        if all_feedback:
            st.download_button("下载人工反馈 JSON", data=json.dumps(all_feedback, ensure_ascii=False, indent=2),
                               file_name=f"ran-feedback-{datetime.now():%Y%m%d-%H%M%S}.json",
                               mime="application/json", key="download_feedback_json")
