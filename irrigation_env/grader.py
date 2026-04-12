"""Deterministic episode graders for precision irrigation.

Each grader takes an ``episode_log`` (list of per-step dicts from ``IrrigationEnv``)
and returns a ``float`` score strictly between **0.0** and **1.0** (endpoints are never
returned; values are clipped to **[0.01, 0.99]**).

Public entry points (referenced from ``openenv.yaml`` and ``api.main``):

- ``grade_easy``   — stress containment & recovery (Don't Kill the Crop)
- ``grade_medium`` — rain-aware irrigation restraint (Respect the Rain)
- ``grade_hard``   — yield vs water efficiency (Do More With Less)
- ``grade_episode`` — legacy weighted aggregate (same clipping rules)
"""

from __future__ import annotations

from typing import Any

import numpy as np

# Open interval (0.0, 1.0): implementation never emits exactly 0.0 or 1.0.
_SCORE_LO = 0.01
_SCORE_HI = 0.99
_EPS = 1e-12


def _clip_unit(x: float) -> float:
    """Clamp a component score to [0, 1] before weighting."""
    return float(np.clip(x, 0.0, 1.0))


def _clip_final(raw: float) -> float:
    """Final score in (0, 1); enforced as [_SCORE_LO, _SCORE_HI]."""
    return float(np.clip(raw, _SCORE_LO, _SCORE_HI))


def _empty_log_score() -> float:
    return _SCORE_LO


def _mean_stress(step: dict[str, Any]) -> float:
    arr = np.asarray(step.get("stress_index", [0.0]), dtype=float)
    return float(np.mean(arr)) if arr.size else 0.0


def _n_zones(episode_log: list[Any]) -> int:
    if not episode_log:
        return 1
    return int(np.asarray(episode_log[0].get("stress_index", np.zeros(1))).size)


def grade_episode(episode_log: list[dict[str, Any]]) -> float:
    """Weighted aggregate over yield, water efficiency, stress events, completion."""
    if not episode_log:
        return _empty_log_score()

    total_steps = len(episode_log)
    final_stress = np.asarray(episode_log[-1].get("stress_index", [0.0]), dtype=float)
    yield_score = _clip_unit(1.0 - float(np.mean(final_stress)))

    water_used = sum(float(s.get("water_used_step", 0.0)) for s in episode_log)
    water_budget = float(episode_log[0].get("water_budget", 1.0))
    water_efficiency = _clip_unit(1.0 - (water_used / max(water_budget, 1.0)))

    stress_steps = sum(1 for s in episode_log if s.get("stress_increased", False))
    stress_events_score = _clip_unit(1.0 - (stress_steps / max(total_steps, 1)))

    last_step = episode_log[-1]
    completed = bool(last_step.get("terminated", False))
    budget_exhausted = bool(last_step.get("budget_exhausted", False))
    completion_score = 1.0 if (completed and not budget_exhausted) else 0.0

    raw = (
        0.4 * yield_score
        + 0.3 * water_efficiency
        + 0.2 * stress_events_score
        + 0.1 * completion_score
    )
    return _clip_final(raw)


def grade_easy(episode_log: list[Any]) -> float:
    """Don't Kill the Crop: stress containment, spike recovery, completion."""
    if not episode_log:
        return _empty_log_score()

    total_steps = len(episode_log)
    means = [_mean_stress(s) for s in episode_log]

    containment = sum(1 for m in means if m < 0.35) / max(total_steps, 1)
    containment = _clip_unit(containment)

    spike_indices: list[int] = []
    for i in range(1, total_steps):
        if means[i - 1] < 0.35 <= means[i]:
            spike_indices.append(i)

    if not spike_indices:
        recovery_score = 1.0
    else:
        recovered = 0
        for i in spike_indices:
            window_end = min(i + 5, total_steps - 1)
            if any(means[j] < 0.35 for j in range(i + 1, window_end + 1)):
                recovered += 1
        recovery_score = recovered / len(spike_indices)

    last_step = episode_log[-1]
    completion_score = 1.0 if bool(last_step.get("terminated", False)) else 0.0

    raw = 0.6 * containment + 0.3 * recovery_score + 0.1 * completion_score
    return _clip_final(raw)


def grade_medium(episode_log: list[Any]) -> float:
    """Respect the Rain: hold irrigation when heavy rain is forecast; drought & water."""
    if not episode_log:
        return _empty_log_score()

    total_steps = len(episode_log)
    n_z = _n_zones(episode_log)

    rain_steps = [s for s in episode_log if float(s.get("rain_forecast_mm", 0.0)) > 10.0]
    if not rain_steps:
        rain_restraint = 0.5
    else:
        # Reward restraint if mean water per zone is under 30% of max (flood) per zone.
        max_mean_per_zone = 600.0
        threshold = 0.3 * max_mean_per_zone
        correct = 0
        for s in rain_steps:
            w = float(s.get("water_used_step", 0.0))
            mean_per_zone = w / max(n_z, 1)
            if mean_per_zone < threshold:
                correct += 1
        rain_restraint = correct / len(rain_steps)

    drought_ok = sum(1 for s in episode_log if _mean_stress(s) < 0.4) / max(
        total_steps, 1
    )

    total_water = sum(float(s.get("water_used_step", 0.0)) for s in episode_log)
    denom = max(total_steps * n_z * 150.0, _EPS)
    water_saved = _clip_unit(1.0 - (total_water / denom))

    raw = 0.5 * rain_restraint + 0.3 * drought_ok + 0.2 * water_saved
    return _clip_final(raw)


def grade_hard(episode_log: list[Any]) -> float:
    """Do More With Less: yield per water, consistency, second-half efficiency."""
    if not episode_log:
        return _empty_log_score()

    total_steps = len(episode_log)
    n_z = _n_zones(episode_log)

    final_stress = np.asarray(episode_log[-1].get("stress_index", [0.0]), dtype=float)
    mean_final = float(np.mean(final_stress))

    total_water = sum(float(s.get("water_used_step", 0.0)) for s in episode_log)
    water_fraction = total_water / max(total_steps * n_z * 150.0, _EPS)
    raw_yield_eff = (1.0 - mean_final) / max(water_fraction, 0.01)
    yield_eff = _clip_unit(raw_yield_eff / 5.0)

    consistency_hits = 0
    for s in episode_log:
        m = _mean_stress(s)
        w = float(s.get("water_used_step", 0.0))
        if 0.0 <= m <= 0.3 and w < (n_z * 200.0):
            consistency_hits += 1
    consistency = consistency_hits / max(total_steps, 1)

    half = total_steps // 2
    second = episode_log[half:]
    if not second:
        no_waste_pressure = 0.0
    else:
        lean_threshold = n_z * 50.0
        nw = sum(
            1
            for s in second
            if float(s.get("water_used_step", 0.0)) < lean_threshold
            and _mean_stress(s) < 0.25
        )
        no_waste_pressure = nw / len(second)

    raw = 0.4 * yield_eff + 0.35 * consistency + 0.25 * no_waste_pressure
    return _clip_final(raw)
