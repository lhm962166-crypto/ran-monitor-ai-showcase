"""Experimental features excluded from production after grouped CV."""
from __future__ import annotations
import numpy as np
from ran_monitor_ai.features import validate_window, correlation


def extract_temporal_features(record: dict) -> dict[str, float]:
    """Measured temporal shape features, independent of all labels and targets."""
    series = validate_window(record)
    matrix = np.array(list(series.values()))
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            centered = matrix - matrix.mean(axis=1, keepdims=True)
            deviation = matrix.std(axis=1)
            scale = np.where(deviation > 1e-9, deviation, 1.)
            normalized = centered / scale[:, None]
            power = np.abs(np.fft.rfft(normalized, axis=1))[:, 1:] ** 2
            total_power = power.sum(axis=1)
            fractions = np.divide(power, total_power[:, None], out=np.zeros_like(power),
                                  where=total_power[:, None] > 1e-12)
            entropy = -(fractions * np.log(np.maximum(fractions, 1e-15))).sum(axis=1) / np.log(power.shape[1])
            half = matrix.shape[1] // 2
            shape = {
                "half_mean_shift": (matrix[:, half:].mean(axis=1) - matrix[:, :half].mean(axis=1)) / scale,
                "normalized_jump": np.abs(np.diff(normalized, axis=1)).max(axis=1),
                "diff_std_ratio": np.diff(normalized, axis=1).std(axis=1),
                "spectral_entropy": entropy,
                "high_frequency_share": fractions[:, power.shape[1] // 4:].sum(axis=1),
                "dominant_frequency_share": fractions.max(axis=1),
                "peak_to_rms": np.abs(normalized).max(axis=1),
            }
            row = {f"{name}_{suffix}": float(values[index]) for index, name in enumerate(series)
                   for suffix, values in shape.items()}
            for name, values in series.items():
                row[f"{name}_lag1corr"] = correlation(values[:-1], values[1:])
                row[f"{name}_lag8corr"] = correlation(values[:-8], values[8:])
    except FloatingPointError as error:
        raise ValueError("KPI 数值范围过大，无法计算时序特征。") from error
    if not all(np.isfinite(value) for value in row.values()):
        raise ValueError("时序特征出现非有限值，请检查输入范围。")
    return row
