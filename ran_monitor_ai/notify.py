"""Optional email and WhatsApp delivery using only Python's standard library."""
from __future__ import annotations

import os
import smtplib
import ssl
import urllib.parse
import urllib.request
from email.message import EmailMessage


def channels_ready() -> dict[str, bool]:
    return {
        "email": bool(os.getenv("SMTP_USER") and os.getenv("SMTP_PASSWORD")
                      and os.getenv("ALERT_EMAIL_TO")),
        "whatsapp": bool(os.getenv("TWILIO_ACCOUNT_SID") and os.getenv("TWILIO_AUTH_TOKEN")
                         and os.getenv("TWILIO_WHATSAPP_FROM") and os.getenv("TWILIO_WHATSAPP_TO")),
    }


def send_email(alert: dict, score: float) -> None:
    advice_label = "候选原因建议（需复核）" if alert.get("requires_review") else "处理建议"
    _send_email_message(f"[{alert['severity']}] 5G RAN 告警：{alert['cause_zh']}",
        f"告警编号：{alert['alert_id']}\n异常分数：{score:.4f}\n"
        f"可能原因：{alert['cause_zh']}（模型输出概率，未校准 {alert['confidence']:.3f}）\n"
        f"级别：{alert['severity']}\n{advice_label}：{alert['action']}\n负责团队：{alert['team']}")


def _send_email_message(subject: str, body: str) -> None:
    message = EmailMessage()
    message["Subject"] = subject
    message["From"] = os.environ["SMTP_USER"]
    message["To"] = os.environ["ALERT_EMAIL_TO"]
    message.set_content(body)
    host = os.getenv("SMTP_HOST", "smtp.gmail.com")
    port = int(os.getenv("SMTP_PORT", "587"))
    with smtplib.SMTP(host, port, timeout=15) as server:
        server.starttls(context=ssl.create_default_context())
        server.login(os.environ["SMTP_USER"], os.environ["SMTP_PASSWORD"])
        server.send_message(message)


def send_whatsapp(alert: dict, score: float) -> None:
    advice_label = "候选原因建议（需复核）" if alert.get("requires_review") else "建议"
    body = (f"[{alert['severity']}] {alert['cause_zh']}\n告警 {alert['alert_id']}，"
            f"分数 {score:.4f}\n根因模型输出概率（未校准）{alert['confidence']:.3f}\n"
            f"{advice_label}：{alert['action']}（{alert['team']}）")
    _send_whatsapp_body(body)


def _send_whatsapp_body(body: str) -> None:
    sid = os.environ["TWILIO_ACCOUNT_SID"]
    token = os.environ["TWILIO_AUTH_TOKEN"]
    endpoint = f"https://api.twilio.com/2010-04-01/Accounts/{sid}/Messages.json"
    data = urllib.parse.urlencode({
        "From": os.environ["TWILIO_WHATSAPP_FROM"],
        "To": os.environ["TWILIO_WHATSAPP_TO"], "Body": body,
    }).encode()
    password_mgr = urllib.request.HTTPPasswordMgrWithDefaultRealm()
    password_mgr.add_password(None, endpoint, sid, token)
    opener = urllib.request.build_opener(urllib.request.HTTPBasicAuthHandler(password_mgr))
    request = urllib.request.Request(endpoint, data=data, method="POST")
    with opener.open(request, timeout=15) as response:
        if response.status >= 300:
            raise RuntimeError(f"Twilio returned HTTP {response.status}")


def deliver(alert: dict, score: float, channels: set[str]) -> list[tuple[str, bool, str]]:
    senders = {"email": send_email, "whatsapp": send_whatsapp}
    results = []
    for channel in channels:
        try:
            senders[channel](alert, score)
            results.append((channel, True, "发送成功"))
        except Exception as error:
            results.append((channel, False, str(error)))
    return results


def notification_digest(results: list[dict], levels: set[str]) -> tuple[str, str, str] | None:
    eligible = [result for result in results if result.get("detected")
                and (result.get("alert") or {}).get("severity") in levels]
    if not eligible:
        return None
    counts = {level: sum(result["alert"]["severity"] == level for result in eligible)
              for level in ("P1", "P2", "P3", "P4")}
    highest = next(level for level, count in counts.items() if count)
    title = f"[{highest}] RAN Monitor AI 批量告警汇总（{len(eligible)} 条）"
    overview = f"本次检测 {len(results)} 个窗口，符合通知级别的告警 {len(eligible)} 条。\n" + " / ".join(
        f"{level}：{count}" for level, count in counts.items() if count
    )
    details = [f"[{result['alert']['severity']}] {result['alert']['cause_zh']} · 窗口 {result.get('sample_id', '—')}\n"
               f"告警 {result['alert']['alert_id']} · 分数 {result['score']:.4f}\n"
               f"{'候选原因建议（需复核）' if result['alert'].get('requires_review') else '建议'}："
               f"{result['alert']['action']}（{result['alert']['team']}）" for result in eligible[:20]]
    body = overview + "\n\n" + "\n\n".join(details)
    if len(eligible) > len(details):
        body += f"\n\n另有 {len(eligible) - len(details)} 条，请查看本机告警中心或检测报告。"
    short = overview + "\n" + "\n".join(
        f"[{result['alert']['severity']}] {result['alert']['cause_zh']} · 窗口 {str(result.get('sample_id', '—'))[:40]}"
        for result in eligible[:8]
    )
    if len(eligible) > 8:
        short += f"\n另有 {len(eligible) - 8} 条，请查看完整报告。"
    return title, body, short


def deliver_digest(results: list[dict], channels: set[str], levels: set[str]) -> list[tuple[str, bool, str]]:
    digest = notification_digest(results, levels)
    if digest is None:
        return []
    title, body, short = digest
    outcome = []
    for channel in sorted(channels):
        try:
            if channel == "email":
                _send_email_message(title, body)
            elif channel == "whatsapp":
                _send_whatsapp_body(short)
            else:
                raise ValueError("通知渠道无效。")
            outcome.append((channel, True, "汇总通知发送成功"))
        except Exception as error:
            outcome.append((channel, False, str(error)))
    return outcome
