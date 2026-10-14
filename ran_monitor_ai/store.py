"""Transactional SQLite alert history, filtering, and disposition audit trail."""
from __future__ import annotations

import json
import math
import os
import sqlite3
import tempfile
import uuid
from datetime import datetime, timezone, timedelta
from pathlib import Path

DB_PATH = Path(__file__).resolve().parents[1] / "alerts.sqlite3"
STATUSES = {"OPEN", "ACKNOWLEDGED", "RESOLVED"}
INCIDENT_TRANSITIONS = {
    "OPEN": {"ACKNOWLEDGED", "RESOLVED"},
    "ACKNOWLEDGED": {"OPEN", "RESOLVED"},
    "RECOVERED": {"OPEN", "RESOLVED"},
    "RESOLVED": {"OPEN"},
}
SEVERITIES = {"P1", "P2", "P3", "P4"}
TRANSITIONS = {
    "OPEN": {"ACKNOWLEDGED", "RESOLVED"},
    "ACKNOWLEDGED": {"OPEN", "RESOLVED"},
    "RESOLVED": {"OPEN"},
}
ALERT_COLUMNS = ("alert_id", "sample_id", "severity", "cause", "score", "payload", "created_at",
                 "status", "acknowledged_at", "resolved_at", "request_id", "window_id",
                 "incident_id", "stream_id")
CAUSES = {"Congestion", "Equipment", "Interference", "Mobility", "Unknown", "undetermined"}


def _decode_alert_rows(rows) -> list[dict]:
    records = []
    for row in rows:
        try:
            payload = json.loads(row[5])
            if not isinstance(payload, dict):
                raise ValueError("payload must be an object")
        except (ValueError, TypeError) as error:
            raise ValueError(f"告警 {str(row[0])[:64]} 的详情数据无效，请检查数据库或恢复备份。") from error
        records.append(dict(zip(ALERT_COLUMNS, row)) | {"payload": payload})
    return records


def database_path(path: Path | str | None = None) -> Path:
    configured = path if path is not None else (os.getenv("RAN_MONITOR_DB_PATH") or DB_PATH)
    return Path(configured).expanduser().resolve()


def backup_database(destination: Path | str, path: Path | str | None = None) -> Path:
    """Create a consistent SQLite snapshot, including committed WAL data."""
    destination = Path(destination).expanduser().resolve()
    if destination == database_path(path):
        raise ValueError("备份文件须与正在使用的数据库不同。")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=destination.parent, prefix=".ran-backup-", suffix=".sqlite3", delete=False) as handle:
        temporary = Path(handle.name)
    source = target = None
    try:
        source = connect(path)
        target = sqlite3.connect(temporary)
        source.backup(target)
        target.close()
        target = None
        source.close()
        source = None
        os.replace(temporary, destination)
        return destination
    finally:
        if target is not None:
            target.close()
        if source is not None:
            source.close()
        temporary.unlink(missing_ok=True)


def connect(path: Path | str | None = None) -> sqlite3.Connection:
    target = database_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(target, timeout=10)
    try:
        connection.execute("PRAGMA busy_timeout=10000")
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA journal_mode=WAL")
        # Serialize schema upgrades across processes as well as across threads.
        connection.execute("BEGIN IMMEDIATE")
        connection.execute("""CREATE TABLE IF NOT EXISTS alerts (
            alert_id TEXT PRIMARY KEY, sample_id TEXT, severity TEXT NOT NULL,
            cause TEXT NOT NULL, score REAL NOT NULL, payload TEXT NOT NULL,
            created_at TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'OPEN',
            acknowledged_at TEXT, resolved_at TEXT
        )""")
        existing = {row[1] for row in connection.execute("PRAGMA table_info(alerts)")}
        for column, definition in (
            ("status", "TEXT NOT NULL DEFAULT 'OPEN'"),
            ("acknowledged_at", "TEXT"), ("resolved_at", "TEXT"),
            ("request_id", "TEXT"), ("window_id", "TEXT"),
            ("incident_id", "TEXT"), ("stream_id", "TEXT"),
        ):
            if column not in existing:
                connection.execute(f"ALTER TABLE alerts ADD COLUMN {column} {definition}")
        connection.execute("CREATE INDEX IF NOT EXISTS idx_alerts_created ON alerts(created_at DESC)")
        connection.execute("CREATE INDEX IF NOT EXISTS idx_alerts_status ON alerts(status, created_at DESC)")
        connection.execute("CREATE INDEX IF NOT EXISTS idx_alerts_incident ON alerts(incident_id, created_at DESC)")
        connection.execute("""CREATE TABLE IF NOT EXISTS detection_requests (
            request_id TEXT PRIMARY KEY, created_at TEXT NOT NULL
        )""")
        connection.execute("""CREATE TABLE IF NOT EXISTS incidents (
            incident_id TEXT PRIMARY KEY, stream_id TEXT, model_id TEXT,
            first_seen TEXT NOT NULL, last_seen TEXT NOT NULL,
            first_start TEXT, last_end TEXT, status TEXT NOT NULL DEFAULT 'OPEN',
            window_count INTEGER NOT NULL DEFAULT 0, cause TEXT, severity TEXT,
            normal_streak INTEGER NOT NULL DEFAULT 0,
            updated_at TEXT NOT NULL
        )""")
        incident_columns = {row[1] for row in connection.execute("PRAGMA table_info(incidents)")}
        if "normal_streak" not in incident_columns:
            connection.execute("ALTER TABLE incidents ADD COLUMN normal_streak INTEGER NOT NULL DEFAULT 0")
        connection.execute("CREATE INDEX IF NOT EXISTS idx_incidents_stream ON incidents(stream_id, model_id, status, last_seen DESC)")
        connection.execute("""CREATE TABLE IF NOT EXISTS incident_windows (
            incident_id TEXT NOT NULL REFERENCES incidents(incident_id),
            window_id TEXT NOT NULL, first_seen TEXT NOT NULL,
            PRIMARY KEY(incident_id, window_id)
        )""")
        connection.execute("""CREATE TABLE IF NOT EXISTS incident_events (
            event_id INTEGER PRIMARY KEY AUTOINCREMENT,
            incident_id TEXT NOT NULL REFERENCES incidents(incident_id),
            alert_id TEXT REFERENCES alerts(alert_id),
            event_type TEXT NOT NULL, payload TEXT NOT NULL, created_at TEXT NOT NULL
        )""")
        connection.execute("""CREATE TABLE IF NOT EXISTS monitoring_streams (
            stream_id TEXT NOT NULL, model_id TEXT NOT NULL, reference_id TEXT NOT NULL,
            scores TEXT NOT NULL, window_ids TEXT NOT NULL,
            observed_total INTEGER NOT NULL DEFAULT 0, updated_at TEXT NOT NULL,
            PRIMARY KEY(stream_id, model_id, reference_id)
        )""")
        connection.execute("""CREATE TABLE IF NOT EXISTS alert_feedback (
            feedback_id INTEGER PRIMARY KEY AUTOINCREMENT,
            alert_id TEXT NOT NULL REFERENCES alerts(alert_id),
            confirmed_cause TEXT NOT NULL, confidence TEXT NOT NULL,
            reviewer TEXT NOT NULL, note TEXT NOT NULL,
            model_id TEXT, feedback_version TEXT NOT NULL, created_at TEXT NOT NULL
        )""")
        connection.execute("""CREATE TABLE IF NOT EXISTS alert_events (
            event_id INTEGER PRIMARY KEY AUTOINCREMENT,
            alert_id TEXT NOT NULL REFERENCES alerts(alert_id),
            previous_status TEXT, status TEXT NOT NULL, note TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL, event_type TEXT NOT NULL DEFAULT 'STATUS_CHANGED'
        )""")
        event_columns = {row[1] for row in connection.execute("PRAGMA table_info(alert_events)")}
        if "event_type" not in event_columns:
            connection.execute("ALTER TABLE alert_events ADD COLUMN event_type TEXT NOT NULL DEFAULT 'STATUS_CHANGED'")
            connection.execute("UPDATE alert_events SET event_type='CREATED' WHERE previous_status IS NULL")
        connection.execute("CREATE INDEX IF NOT EXISTS idx_alert_events ON alert_events(alert_id, event_id)")
        connection.execute("""CREATE TABLE IF NOT EXISTS monitoring_observations (
            stream_id TEXT NOT NULL, model_id TEXT NOT NULL, window_id TEXT NOT NULL,
            detected INTEGER NOT NULL, score REAL NOT NULL, start_time TEXT, end_time TEXT,
            created_at TEXT NOT NULL, PRIMARY KEY(stream_id, model_id, window_id)
        )""")
        connection.execute("""CREATE TABLE IF NOT EXISTS incident_recovery (
            stream_id TEXT NOT NULL, model_id TEXT NOT NULL, incident_id TEXT NOT NULL,
            last_window_end TEXT, normal_started TEXT, normal_streak INTEGER NOT NULL DEFAULT 0,
            PRIMARY KEY(stream_id, model_id)
        )""")
        connection.commit()
        return connection
    except Exception:
        connection.rollback()
        connection.close()
        raise


def save_batch(results: list[dict], path: Path | str | None = None) -> int:
    """Persist a request atomically and attach windows to explicit-stream incidents."""
    if not results:
        return 0
    request_ids = {str(result.get("request_id")) for result in results if result.get("request_id")}
    if len(request_ids) > 1:
        raise ValueError("同一次保存不能混合多个检测请求编号。")
    request_id = next(iter(request_ids), None)
    db = connect(path)
    inserted = 0
    now = datetime.now(timezone.utc).isoformat()
    try:
        with db:
            if request_id:
                cursor = db.execute("INSERT OR IGNORE INTO detection_requests(request_id, created_at) VALUES (?, ?)",
                                    (request_id, now))
                if not cursor.rowcount:
                    return 0
            ordered = sorted(results, key=lambda result: (
                str(result.get("stream_id") or ""), _time_key(result.get("start_time")),
            ))
            for result in ordered:
                if not result.get("detected") or not result.get("alert"):
                    _observe_recovery(db, result, now)
                    continue
                alert = result["alert"]
                if not isinstance(alert.get("alert_id"), str) or not alert["alert_id"].strip():
                    raise ValueError("告警编号须为非空文本。")
                if alert["severity"] not in SEVERITIES:
                    raise ValueError("告警级别无效。")
                if not isinstance(alert.get("cause"), str) or not alert["cause"].strip():
                    raise ValueError("告警原因须为非空文本。")
                if not math.isfinite(float(result["score"])) or not 0 <= float(result["score"]) <= 1:
                    raise ValueError("异常评分须为 0–1 的有限数值。")
                if db.execute("SELECT 1 FROM alerts WHERE alert_id=?", (alert["alert_id"],)).fetchone():
                    continue
                incident_id = _attach_incident(db, result, alert, now)
                alert["incident_id"] = incident_id
                cursor = db.execute("""INSERT INTO alerts
                    (alert_id, sample_id, severity, cause, score, payload, created_at,
                     request_id, window_id, incident_id, stream_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) ON CONFLICT(alert_id) DO NOTHING""", (
                        alert["alert_id"], str(result.get("sample_id", "")), alert["severity"],
                        alert["cause"], result["score"], json.dumps(alert, ensure_ascii=False, allow_nan=False), now,
                        request_id or alert.get("request_id"), result.get("window_id") or alert.get("window_id"),
                        incident_id, result.get("stream_id") or alert.get("stream_id"),
                    ))
                if cursor.rowcount:
                    inserted += 1
                    db.execute("""INSERT INTO alert_events
                        (alert_id, previous_status, status, note, created_at, event_type) VALUES (?, NULL, 'OPEN', ?, ?, 'CREATED')""",
                               (alert["alert_id"], "检测生成告警", now))
                    db.execute("""INSERT INTO incident_events
                        (incident_id, alert_id, event_type, payload, created_at) VALUES (?, ?, 'WINDOW_DETECTED', ?, ?)""",
                               (incident_id, alert["alert_id"], json.dumps({
                                   "window_id": result.get("window_id"), "severity": alert["severity"],
                                   "cause": alert["cause"], "score": result["score"],
                               }, ensure_ascii=False, allow_nan=False), now))
                    _reset_recovery(db, result)
        return inserted
    finally:
        db.close()


def _normalized_time(value) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _time_key(value) -> str:
    parsed = _normalized_time(value)
    return parsed.isoformat() if parsed else "9999"


def _attach_incident(db: sqlite3.Connection, result: dict, alert: dict, now: str) -> str:
    stream_id = result.get("stream_id") or alert.get("stream_id")
    stream_id = str(stream_id).strip()[:200] if stream_id not in (None, "") else None
    start = _normalized_time(result.get("start_time") or alert.get("start_time"))
    end = _normalized_time(result.get("end_time") or alert.get("end_time"))
    if start and end and end < start:
        start = end = None
    model_id = str(alert.get("model_id") or "unknown")[:200]
    incident_id = None
    matched = None
    # Merge only when an explicit stream identity and valid time interval support continuity.
    if stream_id and start and end:
        duration = max((end - start).total_seconds(), 0.5)
        max_gap = timedelta(seconds=2 * duration)
        open_rows = db.execute("""SELECT incident_id, first_start, last_end
            FROM incidents WHERE stream_id=? AND model_id=? AND status IN ('OPEN', 'ACKNOWLEDGED')
            ORDER BY last_seen DESC LIMIT 50""", (stream_id, model_id)).fetchall()
        for row in open_rows:
            old_start, old_end = _normalized_time(row[1]), _normalized_time(row[2])
            if old_start and old_end and start <= old_end + max_gap and end >= old_start - max_gap:
                incident_id, matched = row[0], row
                break
    if incident_id is None:
        incident_id = f"INC-{uuid.uuid4().hex.upper()}"
        seen = now
        db.execute("""INSERT INTO incidents
            (incident_id, stream_id, model_id, first_seen, last_seen, first_start, last_end,
             status, window_count, cause, severity, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, 'OPEN', 0, ?, ?, ?)""",
                   (incident_id, stream_id, model_id, seen, seen,
                    start.isoformat() if start else None, end.isoformat() if end else None,
                    alert.get("cause"), alert.get("severity"), now))
    else:
        first_start = _normalized_time(matched[1]) if matched else start
        last_end = _normalized_time(matched[2]) if matched else end
        if start and first_start:
            first_start = min(start, first_start)
        if end and last_end:
            last_end = max(end, last_end)
        rank = {"P1": 1, "P2": 2, "P3": 3, "P4": 4}
        previous = db.execute("SELECT severity FROM incidents WHERE incident_id=?", (incident_id,)).fetchone()
        severity = alert.get("severity")
        if previous and rank.get(previous[0], 9) < rank.get(severity, 9):
            severity = previous[0]
        previous_cause = db.execute("SELECT cause FROM incidents WHERE incident_id=?", (incident_id,)).fetchone()
        cause = previous_cause[0] if previous_cause else alert.get("cause")
        if previous_cause and previous_cause[0] != alert.get("cause"):
            cause = "MULTIPLE_CANDIDATES"
        previous_seen = _normalized_time(db.execute(
            "SELECT last_seen FROM incidents WHERE incident_id=?", (incident_id,)
        ).fetchone()[0])
        latest_seen = max(last_end, previous_seen) if last_end and previous_seen else (last_end or previous_seen)
        db.execute("""UPDATE incidents SET first_start=?, last_end=?, last_seen=?,
            window_count=window_count, normal_streak=0, cause=?, severity=?, updated_at=? WHERE incident_id=?""",
                   (first_start.isoformat() if first_start else None,
                    last_end.isoformat() if last_end else None,
                    latest_seen.isoformat() if latest_seen else now,
                    cause, severity, now, incident_id))
    window_id = str(result.get("window_id") or alert.get("window_id") or f"ALERT:{alert['alert_id']}")
    cursor = db.execute("INSERT OR IGNORE INTO incident_windows(incident_id, window_id, first_seen) VALUES (?, ?, ?)",
                        (incident_id, window_id, now))
    if cursor.rowcount:
        db.execute("UPDATE incidents SET window_count=window_count+1 WHERE incident_id=?", (incident_id,))
    return incident_id


def _reset_recovery(db: sqlite3.Connection, result: dict) -> None:
    stream_id = result.get("stream_id")
    if not stream_id:
        return
    model_id = str(result.get("model_id") or "unknown")
    stream_id = str(stream_id).strip()[:200]
    db.execute("DELETE FROM incident_recovery WHERE stream_id=? AND model_id=?", (stream_id, model_id))


def _observe_recovery(db: sqlite3.Connection, result: dict, now: str) -> None:
    """Close an incident after three distinct, advancing normal windows."""
    stream_id = result.get("stream_id")
    start = _normalized_time(result.get("start_time"))
    end = _normalized_time(result.get("end_time"))
    if not stream_id or not start or not end or end < start:
        return
    stream_id = str(stream_id).strip()[:200]
    model_id = str(result.get("model_id") or "unknown")
    window_id = str(result.get("window_id") or "")
    if not window_id:
        return
    cursor = db.execute("""INSERT OR IGNORE INTO monitoring_observations
        (stream_id, model_id, window_id, detected, score, start_time, end_time, created_at)
        VALUES (?, ?, ?, 0, ?, ?, ?, ?)""",
                       (stream_id, model_id, window_id, float(result.get("score", 0)),
                        start.isoformat(), end.isoformat(), now))
    if not cursor.rowcount:
        return
    incident = db.execute("""SELECT incident_id, last_end FROM incidents
        WHERE stream_id=? AND model_id=? AND status IN ('OPEN', 'ACKNOWLEDGED')
        ORDER BY last_seen DESC LIMIT 1""", (stream_id, model_id)).fetchone()
    if not incident:
        db.execute("DELETE FROM incident_recovery WHERE stream_id=? AND model_id=?", (stream_id, model_id))
        return
    incident_id, incident_end_value = incident
    incident_end = _normalized_time(incident_end_value)
    if incident_end and start < incident_end:
        return
    previous = db.execute("""SELECT incident_id, last_window_end, normal_started, normal_streak
        FROM incident_recovery WHERE stream_id=? AND model_id=?""", (stream_id, model_id)).fetchone()
    duration = max((end - start).total_seconds(), 0.5)
    streak, normal_started = 1, start
    if previous and previous[0] == incident_id:
        previous_end = _normalized_time(previous[1])
        if previous_end and end <= previous_end:
            return
        if previous_end and start <= previous_end + timedelta(seconds=2 * duration):
            streak = int(previous[3]) + 1
            normal_started = _normalized_time(previous[2]) or start
    db.execute("""INSERT INTO incident_recovery
        (stream_id, model_id, incident_id, last_window_end, normal_started, normal_streak)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(stream_id, model_id) DO UPDATE SET incident_id=excluded.incident_id,
        last_window_end=excluded.last_window_end, normal_started=excluded.normal_started,
        normal_streak=excluded.normal_streak""",
               (stream_id, model_id, incident_id, end.isoformat(), normal_started.isoformat(), streak))
    db.execute("UPDATE incidents SET normal_streak=?, updated_at=? WHERE incident_id=?",
               (streak, now, incident_id))
    if streak >= 3:
        db.execute("UPDATE incidents SET status='RECOVERED', updated_at=? WHERE incident_id=?",
                   (now, incident_id))
        db.execute("""INSERT INTO incident_events
            (incident_id, alert_id, event_type, payload, created_at)
            VALUES (?, NULL, 'RECOVERY_CONFIRMED', ?, ?)""",
                   (incident_id, json.dumps({"consecutive_normal_windows": streak,
                                             "recovery_started": normal_started.isoformat(),
                                             "last_normal_end": end.isoformat()}, ensure_ascii=False), now))


def save(result: dict, path: Path | str | None = None) -> None:
    save_batch([result], path)


def _where(status=None, severities=None, causes=None, search: str = "") -> tuple[str, list]:
    clauses, parameters = [], []
    if status is not None:
        if status not in STATUSES:
            raise ValueError("告警状态无效。")
        clauses.append("status=?")
        parameters.append(status)
    for column, values in (("severity", severities), ("cause", causes)):
        if values is not None:
            if not values:
                clauses.append("0=1")
            else:
                clauses.append(f"{column} IN ({','.join('?' for _ in values)})")
                parameters.extend(values)
    if search.strip():
        query = search.strip().replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        clauses.append("(alert_id LIKE ? ESCAPE '\\' OR sample_id LIKE ? ESCAPE '\\' OR incident_id LIKE ? ESCAPE '\\' OR window_id LIKE ? ESCAPE '\\')")
        parameters.extend([f"%{query}%"] * 4)
    return (" WHERE " + " AND ".join(clauses) if clauses else ""), parameters


def query_alerts(limit: int = 50, offset: int = 0, *, status=None, severities=None,
                 causes=None, search: str = "", path: Path | str | None = None) -> tuple[list[dict], int]:
    if not 1 <= int(limit) <= 500 or int(offset) < 0:
        raise ValueError("分页范围无效；每页须为 1–500 条。")
    where, parameters = _where(status, severities, causes, search)
    db = connect(path)
    try:
        db.execute("BEGIN")
        total = db.execute("SELECT COUNT(*) FROM alerts" + where, parameters).fetchone()[0]
        rows = db.execute("SELECT " + ",".join(ALERT_COLUMNS) + " FROM alerts" + where
                          + " ORDER BY created_at DESC, alert_id DESC LIMIT ? OFFSET ?",
                          parameters + [int(limit), int(offset)]).fetchall()
    finally:
        db.close()
    return _decode_alert_rows(rows), total


def recent(limit: int = 200, path: Path | str | None = None) -> list[dict]:
    return query_alerts(limit, path=path)[0]


def export_alerts(*, status=None, severities=None, causes=None, search: str = "",
                  limit: int = 10000, path: Path | str | None = None) -> tuple[list[dict], int]:
    """Read a consistent filtered snapshot with a bounded export size."""
    if not 1 <= limit <= 10000:
        raise ValueError("导出上限须为 1–10000 条。")
    where, parameters = _where(status, severities, causes, search)
    db = connect(path)
    try:
        db.execute("BEGIN")
        total = db.execute("SELECT COUNT(*) FROM alerts" + where, parameters).fetchone()[0]
        rows = db.execute("SELECT " + ",".join(ALERT_COLUMNS) + " FROM alerts" + where
                          + " ORDER BY created_at DESC, alert_id DESC LIMIT ?", parameters + [limit]).fetchall()
    finally:
        db.close()
    return _decode_alert_rows(rows), total


def history_summary(path: Path | str | None = None) -> dict:
    db = connect(path)
    try:
        counts = dict(db.execute("SELECT status, COUNT(*) FROM alerts GROUP BY status"))
        critical = db.execute("SELECT COUNT(*) FROM alerts WHERE severity='P1' AND status!='RESOLVED'").fetchone()[0]
        incident_counts = db.execute("SELECT status, COUNT(*) FROM incidents GROUP BY status").fetchall()
    finally:
        db.close()
    incident_counts = dict(incident_counts)
    active_incidents = incident_counts.get("OPEN", 0) + incident_counts.get("ACKNOWLEDGED", 0)
    return {"total": sum(counts.values()), "critical_unresolved": critical,
            "incident_total": sum(incident_counts.values()),
            "open_incidents": active_incidents,
            "acknowledged_incidents": incident_counts.get("ACKNOWLEDGED", 0),
            "recovered_incidents": incident_counts.get("RECOVERED", 0),
            **{status: counts.get(status, 0) for status in STATUSES}}


def update_persisted_score_drift(stream_id: str, new_scores, window_ids, reference_scores, *,
                                 model_id: str, path: Path | str | None = None) -> dict:
    """Atomically persist bounded score drift state and ignore repeated windows."""
    stream_id = str(stream_id).strip()
    model_id = str(model_id).strip()
    if not stream_id or len(stream_id) > 200:
        raise ValueError("跨会话漂移监测需要 1–200 字的数据流 ID。")
    if not model_id:
        raise ValueError("持久化漂移监测需要有效模型版本。")
    from .monitoring import update_score_drift

    db = connect(path)
    try:
        db.execute("BEGIN IMMEDIATE")
        row = db.execute("""SELECT scores, window_ids, observed_total
            FROM monitoring_streams WHERE stream_id=? AND model_id=? AND reference_id=?""",
                         (stream_id, model_id, model_id)).fetchone()
        state = None
        if row:
            state = {"model_id": model_id, "stream_source": stream_id,
                     "scores": json.loads(row[0]), "window_ids": json.loads(row[1]),
                     "observed_total": int(row[2])}
        updated, summary = update_score_drift(
            state, new_scores, reference_scores, model_id=model_id,
            stream_source=stream_id, window_ids=window_ids,
        )
        db.execute("""INSERT INTO monitoring_streams
            (stream_id, model_id, reference_id, scores, window_ids, observed_total, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(stream_id, model_id, reference_id) DO UPDATE SET
            scores=excluded.scores, window_ids=excluded.window_ids,
            observed_total=excluded.observed_total, updated_at=excluded.updated_at""",
                   (stream_id, model_id, model_id,
                    json.dumps(updated["scores"], allow_nan=False),
                    json.dumps(updated["window_ids"], ensure_ascii=False),
                    updated["observed_total"], datetime.now(timezone.utc).isoformat()))
        db.commit()
        summary["persistence"] = "persistent"
        return summary
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def reset_persisted_score_drift(stream_id: str, model_id: str, path: Path | str | None = None) -> int:
    db = connect(path)
    try:
        with db:
            cursor = db.execute("DELETE FROM monitoring_streams WHERE stream_id=? AND model_id=?",
                                (str(stream_id).strip(), str(model_id).strip()))
            return cursor.rowcount
    finally:
        db.close()


def add_feedback(alert_id: str, confirmed_cause: str, confidence: str, reviewer: str,
                 note: str, path: Path | str | None = None) -> bool:
    """Append a versioned, auditable human cause review without changing alert state."""
    if confirmed_cause not in CAUSES:
        raise ValueError("核实原因不在支持的标签范围内。")
    if confidence not in {"high", "medium", "low", "undetermined"}:
        raise ValueError("反馈可信度选项无效。")
    reviewer = reviewer.strip()[:120]
    note = note.strip()
    if not note:
        raise ValueError("请填写核实依据；无法确认时也请说明原因。")
    if len(note) > 2000:
        raise ValueError("复核备注最多 2000 字。")
    db = connect(path)
    try:
        db.execute("BEGIN IMMEDIATE")
        row = db.execute("SELECT payload FROM alerts WHERE alert_id=?", (alert_id,)).fetchone()
        if row is None:
            db.rollback()
            return False
        try:
            payload = json.loads(row[0])
        except (ValueError, TypeError) as error:
            raise ValueError("告警详情无法解析，不能附加复核反馈。") from error
        db.execute("""INSERT INTO alert_feedback
            (alert_id, confirmed_cause, confidence, reviewer, note, model_id,
             feedback_version, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                   (alert_id, confirmed_cause, confidence, reviewer, note,
                    payload.get("model_id"), "cause-review-v1",
                    datetime.now(timezone.utc).isoformat()))
        db.execute("""INSERT INTO alert_events
            (alert_id, previous_status, status, note, created_at, event_type)
            SELECT alert_id, status, status, ?, ?, 'FEEDBACK' FROM alerts WHERE alert_id=?""",
                   (f"人工核实原因：{confirmed_cause}；可信度：{confidence}；复核人：{reviewer or '未填写'}",
                    datetime.now(timezone.utc).isoformat(), alert_id))
        db.commit()
        return True
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def feedback_rows(path: Path | str | None = None) -> list[dict]:
    db = connect(path)
    try:
        rows = db.execute("""SELECT f.alert_id, f.confirmed_cause, f.confidence, f.reviewer, f.note,
            f.model_id, f.feedback_version, f.created_at, a.window_id, a.sample_id, a.stream_id, a.cause
            FROM alert_feedback AS f JOIN alerts AS a ON a.alert_id=f.alert_id
            ORDER BY f.feedback_id DESC""").fetchall()
    finally:
        db.close()
    columns = ("alert_id", "confirmed_cause", "confidence", "reviewer", "note",
               "model_id", "feedback_version", "created_at", "window_id", "sample_id",
               "stream_id", "predicted_cause")
    return [dict(zip(columns, row)) for row in rows]


def update_status(alert_id: str, status: str, path: Path | str | None = None,
                  *, note: str = "") -> bool:
    if status not in STATUSES:
        raise ValueError(f"Unsupported alert status: {status}")
    note = note.strip()
    if len(note) > 2000:
        raise ValueError("处置备注最多 2000 字。")
    now = datetime.now(timezone.utc).isoformat()
    db = connect(path)
    try:
        # Lock before reading the current state to serialize concurrent updates.
        db.execute("BEGIN IMMEDIATE")
        row = db.execute("SELECT status FROM alerts WHERE alert_id=?", (alert_id,)).fetchone()
        if row is None:
            db.rollback()
            return False
        previous = row[0]
        if previous == status:
            db.rollback()
            return True
        if status not in TRANSITIONS.get(previous, set()):
            raise ValueError(f"不能从 {previous} 直接转为 {status}。")
        if status == "OPEN":
            db.execute("UPDATE alerts SET status=?, acknowledged_at=NULL, resolved_at=NULL WHERE alert_id=?",
                       (status, alert_id))
        elif status == "ACKNOWLEDGED":
            db.execute("UPDATE alerts SET status=?, acknowledged_at=?, resolved_at=NULL WHERE alert_id=?",
                       (status, now, alert_id))
        else:
            db.execute("UPDATE alerts SET status=?, resolved_at=? WHERE alert_id=?", (status, now, alert_id))
        db.execute("""INSERT INTO alert_events
            (alert_id, previous_status, status, note, created_at) VALUES (?, ?, ?, ?, ?)""",
                   (alert_id, previous, status, note, now))
        db.commit()
        return True
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def alert_events(alert_id: str, path: Path | str | None = None) -> list[dict]:
    db = connect(path)
    try:
        rows = db.execute("""SELECT previous_status, status, note, created_at, event_type
            FROM alert_events WHERE alert_id=? ORDER BY event_id DESC""", (alert_id,)).fetchall()
    finally:
        db.close()
    return [dict(zip(("previous_status", "status", "note", "created_at", "event_type"), row)) for row in rows]


def incident_events(incident_id: str, path: Path | str | None = None) -> list[dict]:
    db = connect(path)
    try:
        rows = db.execute("""SELECT event_type, payload, created_at, alert_id
            FROM incident_events WHERE incident_id=? ORDER BY event_id DESC""", (incident_id,)).fetchall()
    finally:
        db.close()
    return [{"event_type": row[0], "payload": json.loads(row[1]), "created_at": row[2], "alert_id": row[3]}
            for row in rows]


def query_incident(incident_id: str, path: Path | str | None = None) -> dict | None:
    db = connect(path)
    try:
        row = db.execute("""SELECT incident_id, stream_id, model_id, first_seen, last_seen,
            first_start, last_end, status, window_count, cause, severity, normal_streak
            FROM incidents WHERE incident_id=?""", (incident_id,)).fetchone()
    finally:
        db.close()
    columns = ("incident_id", "stream_id", "model_id", "first_seen", "last_seen", "first_start",
               "last_end", "status", "window_count", "cause", "severity", "normal_streak")
    return dict(zip(columns, row)) if row else None


def update_incident_status(incident_id: str, status: str, note: str = "",
                           path: Path | str | None = None) -> bool:
    if status not in INCIDENT_TRANSITIONS:
        raise ValueError("持续事件状态无效。")
    note = note.strip()
    if len(note) > 2000:
        raise ValueError("事件处置备注最多 2000 字。")
    now = datetime.now(timezone.utc).isoformat()
    db = connect(path)
    try:
        db.execute("BEGIN IMMEDIATE")
        row = db.execute("SELECT status FROM incidents WHERE incident_id=?", (incident_id,)).fetchone()
        if row is None:
            db.rollback()
            return False
        previous = row[0]
        if previous == status:
            db.rollback()
            return True
        if status not in INCIDENT_TRANSITIONS.get(previous, set()):
            raise ValueError(f"持续事件不能从 {previous} 直接转为 {status}。")
        db.execute("UPDATE incidents SET status=?, normal_streak=0, updated_at=? WHERE incident_id=?",
                   (status, now, incident_id))
        db.execute("DELETE FROM incident_recovery WHERE incident_id=?", (incident_id,))
        db.execute("""INSERT INTO incident_events
            (incident_id, alert_id, event_type, payload, created_at)
            VALUES (?, NULL, 'STATUS_CHANGED', ?, ?)""",
                   (incident_id, json.dumps({"previous_status": previous, "status": status,
                                             "note": note}, ensure_ascii=False), now))
        db.commit()
        return True
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def add_note(alert_id: str, note: str, path: Path | str | None = None) -> bool:
    note = note.strip()
    if not note or len(note) > 2000:
        raise ValueError("备注须为 1–2000 字。")
    db = connect(path)
    try:
        db.execute("BEGIN IMMEDIATE")
        row = db.execute("SELECT status FROM alerts WHERE alert_id=?", (alert_id,)).fetchone()
        if row is None:
            db.rollback()
            return False
        db.execute("""INSERT INTO alert_events
            (alert_id, previous_status, status, note, created_at, event_type) VALUES (?, ?, ?, ?, ?, 'NOTE')""",
                   (alert_id, row[0], row[0], note, datetime.now(timezone.utc).isoformat()))
        db.commit()
        return True
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()
