"""Monte Carlo checks for the score-distribution drift monitor.

The benchmark resamples held-out validation scores and applies controlled shifts.
It measures the monitor's operating behavior; it is not evidence of field accuracy.
"""
from __future__ import annotations

import math

import numpy as np

from .monitoring import update_score_drift


SCENARIOS = ("stable", "mean_shift", "spread_shift", "surge")


def _wilson_interval(successes: int, trials: int, z: float = 1.96) -> list[float]:
    if trials <= 0:
        return [0.0, 0.0]
    proportion = successes / trials
    denominator = 1 + z * z / trials
    center = (proportion + z * z / (2 * trials)) / denominator
    radius = z * math.sqrt(proportion * (1 - proportion) / trials + z * z / (4 * trials * trials)) / denominator
    return [max(0.0, center - radius), min(1.0, center + radius)]


def _make_stream(reference: np.ndarray, scenario: str, horizon: int,
                 change_point: int, rng: np.random.Generator) -> np.ndarray:
    stream = rng.choice(reference, size=horizon, replace=True)
    post = np.arange(horizon) >= change_point
    if scenario == "stable":
        return stream
    if scenario == "mean_shift":
        stream[post] = np.clip(stream[post] + 0.30, 0.0, 1.0)
    elif scenario == "spread_shift":
        center = float(np.median(reference))
        stream[post] = np.clip(center + 2.5 * (stream[post] - center), 0.0, 1.0)
    elif scenario == "surge":
        post_indices = np.flatnonzero(post)
        stream[post_indices] = rng.choice(reference, size=len(post_indices), replace=True)
        surge = rng.random(len(post_indices)) < 0.35
        stream[post_indices[surge]] = rng.uniform(0.85, 1.0, int(surge.sum()))
    else:
        raise ValueError(f"未知漂移场景：{scenario}")
    return stream


def _run_stream(reference: np.ndarray, stream: np.ndarray, *, model_id: str,
                batch_size: int) -> list[int]:
    state = None
    alarm_episodes = []
    was_drift = False
    for offset in range(0, len(stream), batch_size):
        batch = stream[offset:offset + batch_size]
        state, result = update_score_drift(
            state, batch, reference, model_id=model_id,
            stream_source="monte_carlo_benchmark", window_size=100,
        )
        is_drift = result["status"] == "drift"
        if is_drift and not was_drift:
            alarm_episodes.append(min(offset + len(batch), len(stream)))
        was_drift = is_drift
    return alarm_episodes


def benchmark_score_drift(reference_scores, *, model_id: str = "unknown",
                          replicates: int = 200, horizon: int = 200,
                          batch_size: int = 10, seed: int = 42) -> dict:
    """Estimate false-alert and shift-detection rates using reproducible streams."""
    reference = np.asarray(reference_scores, dtype=float)
    if reference.ndim != 1 or len(reference) < 30:
        raise ValueError("漂移基准至少需要 30 个验证集参照分数。")
    if not np.isfinite(reference).all() or np.any((reference < 0) | (reference > 1)):
        raise ValueError("参照分数须为 0–1 的有限数值。")
    if not isinstance(replicates, int) or not 20 <= replicates <= 5000:
        raise ValueError("重复次数须为 20–5000 的整数。")
    if not isinstance(horizon, int) or horizon < 60 or horizon > 10000:
        raise ValueError("流长度须为 60–10000 个窗口。")
    if not isinstance(batch_size, int) or not 1 <= batch_size <= min(100, horizon):
        raise ValueError("批大小须为 1 到 min(100, 流长度) 之间的整数。")
    if not isinstance(seed, int):
        raise ValueError("随机种子须为整数。")

    change_point = max(30, horizon // 2)
    if change_point >= horizon:
        change_point = horizon // 2
    rng = np.random.default_rng(seed)
    output = {}
    for scenario in SCENARIOS:
        stable_alarms = 0
        pre_change_alarms = 0
        detected_after_change = 0
        detection_delays = []
        total_alerts = 0
        for _ in range(replicates):
            stream = _make_stream(reference, scenario, horizon, change_point, rng)
            alarms = _run_stream(reference, stream, model_id=model_id, batch_size=batch_size)
            total_alerts += len(alarms)
            if scenario == "stable":
                stable_alarms += bool(alarms)
                continue
            pre_change_alarms += any(alarm <= change_point for alarm in alarms)
            post_alarms = [alarm for alarm in alarms if alarm > change_point]
            if post_alarms:
                detected_after_change += 1
                detection_delays.append(post_alarms[0] - change_point)

        summary = {
            "streams": replicates,
            "drift_alert_episodes": total_alerts,
            "change_point_window": None if scenario == "stable" else change_point,
        }
        if scenario == "stable":
            summary.update({
                "false_alert_streams": stable_alarms,
                "false_alert_probability": stable_alarms / replicates,
                "false_alert_probability_95ci": _wilson_interval(stable_alarms, replicates),
                "false_alert_episodes_per_100_windows": total_alerts * 100 / (replicates * horizon),
            })
        else:
            summary.update({
                "pre_change_false_alert_streams": pre_change_alarms,
                "pre_change_false_alert_probability": pre_change_alarms / replicates,
                "post_change_detected_streams": detected_after_change,
                "post_change_detection_probability": detected_after_change / replicates,
                "post_change_detection_probability_95ci": _wilson_interval(detected_after_change, replicates),
                "median_detection_delay_windows": (float(np.median(detection_delays))
                                                   if detection_delays else None),
                "p90_detection_delay_windows": (float(np.percentile(detection_delays, 90))
                                                if detection_delays else None),
            })
        output[scenario] = summary

    return {
        "protocol": "fixed-validation-reference bootstrap; controlled shifts; batched streaming updates",
        "scope_note": "Synthetic score streams estimate monitor behavior only; they do not measure RAN field accuracy.",
        "model_id": str(model_id),
        "reference_samples": int(len(reference)),
        "replicates_per_scenario": replicates,
        "stream_horizon_windows": horizon,
        "batch_size_windows": batch_size,
        "seed": seed,
        "scenarios": output,
    }
