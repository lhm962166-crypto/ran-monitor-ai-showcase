"""Deterministic synthetic windows used only by the local product demo."""
from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from .settings import KPIS, WINDOW_SIZE

CAUSES = ("Normal", "Interference", "Congestion", "Equipment", "Mobility")
ANOMALY_TYPES = {
    "Interference": "Jamming",
    "Congestion": "High Network Congestion (Sudden Spike)",
    "Equipment": "Antenna Failure",
    "Mobility": "Doppler Shift (Severe)",
}
BASELINES = {
    "RSRP": -88.0, "DL_BLER": .08, "DL_MCS": 12.0, "UL_BLER": .07,
    "UL_MCS": 11.0, "UL_NPRB": 40.0, "UL_SNR": 18.0, "TX_Bytes": 120_000.0,
    "RX_Bytes": 150_000.0, "Estimated_UL_Buffer": 50_000.0,
    "PRBs_DL_Current": 40.0, "PRBs_UL_Current": 35.0,
    "PRB_Utilization_DL": .35, "PRB_Utilization_UL": .30,
    "UL_NumberOfPackets": 180.0, "DL_NumberOfPackets": 200.0,
}
NOISE = {
    "RSRP": 1.4, "DL_BLER": .012, "DL_MCS": .5, "UL_BLER": .012, "UL_MCS": .5,
    "UL_NPRB": 3.0, "UL_SNR": 1.6, "TX_Bytes": 5_000.0, "RX_Bytes": 5_500.0,
    "Estimated_UL_Buffer": 4_000.0, "PRBs_DL_Current": 2.8, "PRBs_UL_Current": 2.6,
    "PRB_Utilization_DL": .035, "PRB_Utilization_UL": .035,
    "UL_NumberOfPackets": 10.0, "DL_NumberOfPackets": 11.0,
}


def synthetic_records(*, per_cause: int = 35, seed: int = 42) -> list[dict]:
    """Create realistic-shaped but entirely synthetic samples for exercising the app."""
    if isinstance(per_cause, bool) or not isinstance(per_cause, int) or per_cause < 10:
        raise ValueError("每类至少生成 10 个窗口，才能覆盖时间切分所需的类别。")
    rng = np.random.default_rng(seed)
    total = per_cause * len(CAUSES)
    start = datetime(2026, 1, 1)
    records = []
    for index in range(total):
        cause = CAUSES[index % len(CAUSES)]
        phase = np.linspace(0, 2 * np.pi, WINDOW_SIZE, endpoint=False)
        common = np.sin(phase + (index % 7) * .2)
        kpis = {}
        for name in KPIS:
            values = BASELINES[name] + rng.normal(0, NOISE[name], WINDOW_SIZE) + common * NOISE[name] * .3
            if cause == "Interference":
                if name == "UL_SNR": values -= 8.0
                if name in {"DL_BLER", "UL_BLER"}: values += .27
                if name in {"DL_MCS", "UL_MCS"}: values -= 3.0
            elif cause == "Congestion":
                if name in {"PRB_Utilization_DL", "PRB_Utilization_UL"}: values += .48
                if name in {"PRBs_DL_Current", "PRBs_UL_Current", "UL_NPRB"}: values += 35.0
                if name == "Estimated_UL_Buffer": values += 55_000.0
                if name in {"TX_Bytes", "RX_Bytes"}: values -= 30_000.0
                if name in {"UL_NumberOfPackets", "DL_NumberOfPackets"}: values -= 35.0
            elif cause == "Equipment":
                if name == "RSRP": values -= 14.0
                if name == "UL_SNR": values -= 5.0
                if name in {"TX_Bytes", "RX_Bytes"}: values -= 42_000.0
                if name in {"DL_BLER", "UL_BLER"}: values += .16
            elif cause == "Mobility":
                if name == "RSRP": values += np.linspace(10.0, -12.0, WINDOW_SIZE)
                if name == "UL_SNR": values += np.linspace(4.0, -7.0, WINDOW_SIZE)
                if name in {"DL_BLER", "UL_BLER"}:
                    values += (np.arange(WINDOW_SIZE) % 32 < 5) * .22
            kpis[name] = np.round(values, 5).tolist()

        timestamp = start + timedelta(minutes=index)
        record = {
            "sample_id": index + 1,
            "stream_id": f"demo-cell-{index % 5:02d}",
            "start_time": timestamp.isoformat(),
            "end_time": (timestamp + timedelta(seconds=12.7)).isoformat(),
            "sampling_rate": 10,
            "data_source": "synthetic_generated_demo",
            "schema_version": "ran-monitor-ai-synthetic-v1",
            "anomaly_origin": "synthetic",
            "KPIs": kpis,
            "labels": {"zone": "DEMO", "application": "Telemetry",
                       "mobility": "Yes" if cause == "Mobility" else "No",
                       "congestion": "Yes" if cause == "Congestion" else "No"},
            "anomalies": {"exists": cause != "Normal", "type": ANOMALY_TYPES.get(cause)},
        }
        if cause != "Normal":
            record["event_id"] = f"synthetic-event-{index + 1:04d}"
        records.append(record)
    return records


def write_synthetic_dataset(path: Path | str, *, per_cause: int = 35, seed: int = 42) -> Path:
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    records = synthetic_records(per_cause=per_cause, seed=seed)
    target.write_text("".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n"
                                      for row in records), encoding="utf-8")
    return target
