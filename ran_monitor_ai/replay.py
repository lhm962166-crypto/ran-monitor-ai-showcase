"""Deterministic demo replay; it never connects to a network data source."""
from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter

from .engine import score_record
from .features import true_cause


def replay_windows(records: list[dict], limit: int = 20, cause: str | None = None) -> list[dict]:
    if not 1 <= limit <= 100:
        raise ValueError("回放窗口数须为 1–100。")
    if cause:
        return [record for record in records if true_cause(record) == cause][:limit]
    buckets = {label: [record for record in records if true_cause(record) == label]
               for label in ("Normal", "Interference", "Congestion", "Equipment", "Mobility")}
    output, offset = [], 0
    while len(output) < limit:
        added = False
        for bucket in buckets.values():
            if offset < len(bucket):
                output.append(bucket[offset])
                added = True
                if len(output) == limit:
                    break
        if not added:
            break
        offset += 1
    return output


@dataclass
class ReplayRun:
    records: list[dict]
    interval: float = 5.
    persist: bool = False
    state: str = "idle"
    cursor: int = 0
    next_due: float = 0.
    results: list[dict] = field(default_factory=list)
    durations: list[float] = field(default_factory=list)
    saved: int = 0
    save_error: str | None = None

    def __post_init__(self):
        if not self.records or len(self.records) > 100 or self.interval < 1:
            raise ValueError("回放需要 1–100 个窗口，间隔至少 1 秒。")

    def resume(self):
        if self.cursor >= len(self.records):
            return
        self.state = "running"
        self.next_due = 0.

    def pause(self):
        if self.state == "running":
            self.state = "paused"

    def advance(self, bundle: dict, now: float) -> dict | None:
        if self.state != "running" or now < self.next_due:
            return None
        started = perf_counter()
        try:
            result = score_record(self.records[self.cursor], bundle)
        except Exception:
            self.state = "failed"
            raise
        self.durations.append(perf_counter() - started)
        self.results.append(result)
        self.cursor += 1
        self.next_due = now + self.interval
        if self.cursor == len(self.records):
            self.state = "completed"
        return result
