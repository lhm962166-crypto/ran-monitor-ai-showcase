"""Self-contained, escaped HTML reports that work offline."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from html import escape
from .settings import CAUSE_ZH, POLICY_ZH


COLORS = {"P1": "#c23445", "P2": "#c96d20", "P3": "#967009", "P4": "#2867b2"}


def _text(value) -> str:
    return escape(str(value if value is not None else "—"), quote=True)


def detection_report_html(results: list[dict], *, metadata: dict | None = None,
                          model_metadata: dict | None = None) -> bytes:
    metadata, model_metadata = metadata or {}, model_metadata or {}
    now = datetime.now(timezone(timedelta(hours=8))).strftime("%Y-%m-%d %H:%M:%S")
    abnormal = sum(bool(result.get("detected")) for result in results)
    critical = sum((result.get("alert") or {}).get("severity") == "P1" for result in results)
    review = sum(bool((result.get("alert") or {}).get("requires_review")) for result in results)
    policy = metadata.get("policy") or (results[0].get("detection_policy", "balanced") if results else "balanced")
    rows, details = [], []
    for index, result in enumerate(results, start=1):
        alert = result.get("alert") or {}
        severity = alert.get("severity", "—")
        badge = f'<span class="badge" style="background:{COLORS.get(severity, "#637792")}">{_text(severity)}</span>'
        confidence = f'{float(alert["confidence"]):.1%}' if alert.get("confidence") is not None else "—"
        rows.append(f'<tr><td>{index}</td><td>{_text(result.get("sample_id"))}</td>'
                    f'<td>{"异常" if result.get("detected") else "正常"}</td>'
                    f'<td>{float(result["score"]):.4f}</td><td>{float(result["threshold"]):.4f}</td>'
                    f'<td>{badge}</td><td>{_text(alert.get("cause_zh"))}</td>'
                    f'<td>{confidence}</td></tr>')
        if not alert:
            continue
        affected_rows = "".join(
            f'<tr><td>{_text(item["name"])}</td><td>{_text(item["current"])}</td>'
            f'<td>{_text(item["baseline"])}</td><td>{_text(item["zscore"])}</td>'
            f'<td>{_text(item.get("direction"))}</td></tr>' for item in alert.get("affected_kpis", [])
        )
        evidence = ('<table><thead><tr><th>KPI</th><th>当前均值</th><th>正常均值</th><th>偏离 σ</th><th>方向</th></tr></thead>'
                    f'<tbody>{affected_rows}</tbody></table>') if affected_rows else '<p>没有窗口均值达到 3σ 偏离。</p>'
        probabilities = alert.get("cause_probabilities") or {}
        calibrated = alert.get("calibrated_cause_probabilities") or {}
        probability_rows = "".join(
            f'<tr><td>{_text(CAUSE_ZH.get(name, name))}</td><td>{float(calibrated.get(name, value)):.1%}</td><td>{float(value):.1%}</td></tr>'
            for name, value in sorted(probabilities.items(), key=lambda item: item[1], reverse=True))
        distribution = ('<h3>原因分类分数</h3><table><thead><tr><th>可能原因</th><th>温度缩放实验值</th><th>原始未校准输出</th></tr></thead>'
                        f'<tbody>{probability_rows}</tbody></table>') if probability_rows else ""
        ranking = alert.get("cause_ranking_shadow") or {}
        ranking_rows = "".join(
            f'<tr><td>{int(item["rank"])}</td><td>{_text(CAUSE_ZH.get(item["cause"], item["cause"]))}</td>'
            f'<td>{float(item["classifier_probability"]):.1%}</td>'
            f'<td>{float(item["kpi_prototype_distance"]):.3f}</td>'
            f'<td>{float(item["ranking_score"]):.3f}</td></tr>'
            for item in ranking.get("candidates", [])
        )
        ranking_section = (
            '<h3>KPI 原型辅助排序（影子实验）</h3>'
            '<p class="muted">排序分数不是概率；KPI 原型距离越小越接近训练窗口的特征原型。</p>'
            '<table><thead><tr><th>排名</th><th>候选原因</th><th>分类器原始分数</th>'
            '<th>KPI 原型距离</th><th>实验排序分数（非概率）</th></tr></thead>'
            f'<tbody>{ranking_rows}</tbody></table>'
        ) if ranking_rows else ""
        candidate_text = "、".join(item.get("cause_zh", item.get("cause", "未知"))
                                    for item in alert.get("cause_candidates", [])) or "—"
        shadow = alert.get("structure_shadow") or {}
        relation_text = "；".join(f'{_text(item.get("relation"))}（偏离 {_text(round(float(item.get("robust_deviation", 0)), 2))}）'
                                  for item in shadow.get("top_relation_shifts", []))
        details.append(f'<section><h2>窗口 {_text(result.get("sample_id"))} · {badge} {_text(alert.get("cause_zh"))}</h2>'
                       f'<p class="muted">告警编号：{_text(alert.get("alert_id"))} · 持续事件：{_text(alert.get("incident_id"))}</p>'
                       f'<p>{_text(alert.get("severity_reason"))}</p>'
                       f'<p>原因候选：{_text(candidate_text)} · 集合状态：{_text(alert.get("cause_set_status"))}</p>'
                       f'<p><strong>{"候选原因建议（需复核）" if alert.get("requires_review") else "建议"}：</strong>{_text(alert.get("action"))} · {_text(alert.get("team"))}</p>'
                       f'<p>基线：{_text(alert.get("baseline_scope"))} · 正常窗口数：{_text(alert.get("baseline_sample_count"))}</p>'
                       f'<p>结构影子分数：{_text(shadow.get("score"))} / 阈值 {_text(shadow.get("threshold"))}；不参与告警决策。{_text(relation_text)}</p>'
                       f'{"<p class=review>根因分类存在不确定性，需要人工复核。</p>" if alert.get("requires_review") else ""}'
                       f'<p class=muted>{_text(alert.get("cause_set_caveat"))}</p>{distribution}{ranking_section}'
                       f'<h3>指标偏离证据</h3>{evidence}</section>')
    html = f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>RAN Monitor AI 检测报告</title>
<style>body{{margin:0;background:#f2f5f9;color:#132b52;font-family:Arial,"Microsoft YaHei",sans-serif;line-height:1.6}}
main{{max-width:1200px;margin:30px auto;padding:0 20px}}header{{background:#0b46a0;color:white;padding:24px;border-radius:8px}}
h1{{margin:0;font-size:26px}}h2{{font-size:18px}}.muted{{color:#637792;font-size:13px}}.cards{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:20px 0}}
.card,section{{background:white;border:1px solid #dce5f0;padding:18px;border-radius:8px;margin-bottom:18px}}.card strong{{font-size:30px;display:block}}
.badge{{display:inline-block;color:white;padding:2px 8px;border-radius:4px;font-size:13px}}table{{border-collapse:collapse;width:100%;font-size:13px;background:white}}
td,th{{padding:9px 12px;border-bottom:1px solid #dce5f0;text-align:left}}th{{background:#eaf1fb}}.scroll{{overflow:auto}}.review{{color:#967009;background:#fff4d8;padding:10px}}
@media(max-width:700px){{.cards{{grid-template-columns:repeat(2,1fr)}}main{{padding:0 12px}}}}
@media print{{body{{background:white}}main{{margin:0;max-width:none}}section{{break-inside:avoid}}}}</style></head>
<body><main><header><h1>RAN Monitor AI · 网络窗口检测报告</h1><p>来源：{_text(metadata.get("source", "本地检测"))} · 生成时间：{now}（北京时间）</p></header>
<div class="cards"><div class="card">窗口总数<strong>{len(results)}</strong></div><div class="card">检出异常<strong>{abnormal}</strong></div>
<div class="card">P1 紧急<strong>{critical}</strong></div><div class="card">需复核<strong>{review}</strong></div></div>
<p class="muted">模型版本：{_text(model_metadata.get("model_id"))} · 分数为模型输出，故障根因需结合现场证据确认。公开演示数据的表现不代表现网性能。</p>
<p class="muted">检测策略：{_text(POLICY_ZH.get(policy, policy))} · 阈值在每个窗口的摘要中记录；策略通过验证集校准，不保证现网准确率。</p>
<section><h2>检测摘要</h2><div class="scroll"><table><thead><tr><th>序号</th><th>窗口</th><th>判定</th><th>评分</th><th>阈值</th><th>级别</th><th>原因</th><th>原因原始输出（未校准）</th></tr></thead><tbody>{"".join(rows)}</tbody></table></div></section>
{"".join(details)}</main></body></html>'''
    return html.encode("utf-8")
