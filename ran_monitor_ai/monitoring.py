"""Session-scoped monitoring for shifts in detector score streams."""
from __future__ import annotations

import numpy as np
from scipy.stats import ks_2samp


def assess_score_drift(reference_scores, recent_scores, *, minimum_samples: int = 30,
                       alpha: float = .01, minimum_effect: float = .25,
                       watch_alpha: float = .05, watch_effect: float = .15) -> dict:
    """Compare recent model scores with a fixed validation-score reference."""
    reference = np.asarray(reference_scores, dtype=float)
    recent = np.asarray(recent_scores, dtype=float)
    if (reference.ndim != 1 or recent.ndim != 1 or len(reference) < 2
            or not len(recent)):
        raise ValueError("漂移监测需要至少 2 个参照分数和 1 个近期分数。")
    if (not np.isfinite(reference).all() or not np.isfinite(recent).all()
            or np.any((reference < 0) | (reference > 1))
            or np.any((recent < 0) | (recent > 1))):
        raise ValueError("漂移监测分数须为 0–1 的有限数值。")
    if not isinstance(minimum_samples, int) or minimum_samples < 2:
        raise ValueError("漂移监测的最小样本数须为至少 2 的整数。")
    criteria = (alpha, minimum_effect, watch_alpha, watch_effect)
    if (not np.isfinite(criteria).all() or not 0 <= alpha <= watch_alpha <= 1
            or not 0 <= watch_effect <= minimum_effect <= 1):
        raise ValueError("漂移监测检验阈值配置无效。")

    result = {
        "current_samples": int(len(recent)),
        "minimum_samples": minimum_samples,
        "reference_samples": int(len(reference)),
        "reference_median": float(np.median(reference)),
        "recent_median": float(np.median(recent)),
        "ks_statistic": None,
        "p_value": None,
    }
    if len(recent) < minimum_samples:
        return result | {"status": "warming_up"}

    test = ks_2samp(reference, recent, alternative="two-sided", method="auto")
    statistic, p_value = float(test.statistic), float(test.pvalue)
    if p_value <= alpha and statistic >= minimum_effect:
        status = "drift"
    elif p_value <= watch_alpha and statistic >= watch_effect:
        status = "watch"
    else:
        status = "stable"
    return result | {"status": status, "ks_statistic": statistic, "p_value": p_value}


def update_score_drift(state: dict | None, new_scores, reference_scores, *,
                       model_id: str, stream_source: str,
                       window_size: int = 100, minimum_samples: int = 30,
                       window_ids=None) -> tuple[dict, dict]:
    """Append unique windows to a bounded stream and reset on model/source changes."""
    scores = np.asarray(new_scores, dtype=float)
    if scores.ndim != 1 or not len(scores):
        raise ValueError("新一批漂移监测分数须为非空一维序列。")
    if not isinstance(window_size, int) or window_size < minimum_samples:
        raise ValueError("滚动窗口大小不能小于最小监测样本数。")
    if not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
        raise ValueError("新一批漂移监测分数须为 0–1 的有限数值。")

    compatible = (isinstance(state, dict)
                  and state.get("model_id") == model_id
                  and state.get("stream_source") == stream_source)
    previous = list(state.get("scores", [])) if compatible else []
    previous_ids = list(state.get("window_ids", [])) if compatible else []
    incoming_count = len(scores)
    used_ids = window_ids is not None
    if window_ids is not None:
        ids = [str(value) for value in window_ids]
        if len(ids) != len(scores):
            raise ValueError("漂移窗口 ID 与分数行数不一致。")
        seen = set(previous_ids)
        keep = []
        for index, value in enumerate(ids):
            if value not in seen:
                keep.append(index)
                seen.add(value)
        scores = scores[keep]
        window_ids = [ids[index] for index in keep]
    else:
        window_ids = []
    combined = (previous + scores.astype(float).tolist())[-window_size:]
    combined_ids = (previous_ids + list(window_ids))[-max(window_size * 10, 1000):]
    if len(scores):
        observation = assess_score_drift(reference_scores, combined, minimum_samples=minimum_samples)
    elif combined:
        observation = assess_score_drift(reference_scores, combined, minimum_samples=minimum_samples)
    else:
        raise ValueError("漂移状态没有可用窗口。")
    updated = {
        "model_id": str(model_id),
        "stream_source": str(stream_source),
        "scores": combined,
        "window_ids": combined_ids,
        "observed_total": (int(state.get("observed_total", 0)) if compatible else 0) + len(scores),
    }
    observation["observed_total"] = updated["observed_total"]
    observation["stream_source"] = str(stream_source)
    observation["new_unique_samples"] = int(len(scores))
    observation["duplicate_samples_ignored"] = int(incoming_count - len(scores)) if used_ids else 0
    return updated, observation
