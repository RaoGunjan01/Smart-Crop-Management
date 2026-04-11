"""Episode grader: scores a completed episode from 0.0 to 1.0."""

from __future__ import annotations

from typing import Any

import numpy as np


def grade_episode(episode_log: list[dict[str, Any]]) -> float:
    """Score a complete episode on a 0.0–1.0 scale.

    The episode_log is a list of step dicts, each expected to contain:
        - "stress_index":      np.ndarray per-zone stress at this step
        - "water_used_step":   litres used this step
        - "water_budget":      total budget in litres
        - "stress_increased":  bool, whether any zone had stress increase
        - "terminated":        bool, whether this was the terminal step

    Args:
        episode_log: List of per-step state/info dicts (one per env step).

    Returns:
        Weighted score in [0.0, 1.0].
    """
    if not episode_log:
        print("Grade: empty episode log → 0.0")
        return 0.0

    total_steps = len(episode_log)

    # ------------------------------------------------------------------
    # yield_score: 1.0 - mean(final stress indices)   weight 0.4
    # ------------------------------------------------------------------
    final_stress = np.asarray(episode_log[-1].get("stress_index", [0.0]), dtype=float)
    yield_score = float(np.clip(1.0 - np.mean(final_stress), 0.0, 1.0))

    # ------------------------------------------------------------------
    # water_efficiency: 1.0 - (water_used / water_budget)  weight 0.3
    # ------------------------------------------------------------------
    water_used = sum(float(s.get("water_used_step", 0.0)) for s in episode_log)
    water_budget = float(episode_log[0].get("water_budget", 1.0))
    water_efficiency = float(np.clip(1.0 - (water_used / max(water_budget, 1.0)), 0.0, 1.0))

    # ------------------------------------------------------------------
    # stress_events: 1.0 - (stress_steps / total_steps)   weight 0.2
    # ------------------------------------------------------------------
    stress_steps = sum(1 for s in episode_log if s.get("stress_increased", False))
    stress_events_score = float(np.clip(1.0 - (stress_steps / max(total_steps, 1)), 0.0, 1.0))

    # ------------------------------------------------------------------
    # completion: 1.0 if finished without budget exhaustion  weight 0.1
    # ------------------------------------------------------------------
    last_step = episode_log[-1]
    completed = bool(last_step.get("terminated", False))
    budget_exhausted = bool(last_step.get("budget_exhausted", False))
    completion_score = 1.0 if (completed and not budget_exhausted) else 0.0

    # ------------------------------------------------------------------
    # Weighted sum
    # ------------------------------------------------------------------
    score = (
        0.4 * yield_score
        + 0.3 * water_efficiency
        + 0.2 * stress_events_score
        + 0.1 * completion_score
    )
    score = float(np.clip(score, 0.0, 1.0))

    # Print breakdown
    print("\n" + "=" * 50)
    print("EPISODE GRADE BREAKDOWN")
    print("=" * 50)
    print(f"  Yield Score        (×0.4): {yield_score:.4f}  → {0.4 * yield_score:.4f}")
    print(f"  Water Efficiency   (×0.3): {water_efficiency:.4f}  → {0.3 * water_efficiency:.4f}")
    print(f"  Stress Events      (×0.2): {stress_events_score:.4f}  → {0.2 * stress_events_score:.4f}")
    print(f"  Completion         (×0.1): {completion_score:.4f}  → {0.1 * completion_score:.4f}")
    print(f"  TOTAL GRADE              : {score:.4f}")
    print("=" * 50 + "\n")

    return score


def _mean_stress(step: dict[str, Any]) -> float:
    return float(np.mean(np.asarray(step.get("stress_index", [0.0]), dtype=float)))


def grade_easy(episode_log: list[Any]) -> float:
    """Universal stress-control test ("Don't Kill the Crop")."""
    if not episode_log:
        print("Easy Grade: empty episode log -> 0.0")
        return 0.0

    total_steps = len(episode_log)
    means = [_mean_stress(s) for s in episode_log]

    containment = sum(1 for m in means if m < 0.35) / max(total_steps, 1)
    containment = float(np.clip(containment, 0.0, 1.0))

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
            ok = any(means[j] < 0.35 for j in range(i + 1, window_end + 1))
            if ok:
                recovered += 1
        recovery_score = recovered / len(spike_indices)

    last_step = episode_log[-1]
    completion_score = 1.0 if bool(last_step.get("terminated", False)) else 0.0

    final_score = float(
        np.clip(
            0.6 * containment + 0.3 * recovery_score + 0.1 * completion_score,
            0.0,
            1.0,
        )
    )

    print("\n" + "=" * 50)
    print("EASY GRADE BREAKDOWN (Don't Kill the Crop)")
    print("=" * 50)
    print(f"  Stress containment (x0.6): {containment:.4f}  -> {0.6 * containment:.4f}")
    print(f"  Recovery after spike (x0.3): {recovery_score:.4f}  -> {0.3 * recovery_score:.4f}")
    print(f"  Completion         (x0.1): {completion_score:.4f}  -> {0.1 * completion_score:.4f}")
    print(f"  TOTAL GRADE              : {final_score:.4f}")
    print("=" * 50 + "\n")

    return final_score


def grade_medium(episode_log: list[Any]) -> float:
    """Universal rain-awareness test ("Respect the Rain")."""
    if not episode_log:
        print("Medium Grade: empty episode log -> 0.0")
        return 0.0

    total_steps = len(episode_log)
    n_zones = int(
        np.asarray(episode_log[0].get("stress_index", np.zeros(1))).size
    )

    rain_steps = [
        s
        for s in episode_log
        if float(s.get("rain_forecast_mm", 0.0)) > 10.0
    ]
    if not rain_steps:
        rain_restraint = 0.5
    else:
        correct = 0
        for s in rain_steps:
            za = s.get("zone_actions", [])
            if isinstance(za, np.ndarray):
                za = za.tolist()
            if not za:
                continue
            if all(int(a) == 0 for a in za):
                correct += 1
        rain_restraint = correct / len(rain_steps)

    drought_ok = sum(1 for s in episode_log if _mean_stress(s) < 0.4) / max(
        total_steps, 1
    )

    total_water = sum(float(s.get("water_used_step", 0.0)) for s in episode_log)
    denom = max(total_steps * n_zones * 150.0, 1e-9)
    water_saved = float(np.clip(1.0 - (total_water / denom), 0.0, 1.0))

    final_score = float(
        np.clip(
            0.5 * rain_restraint + 0.3 * drought_ok + 0.2 * water_saved,
            0.0,
            1.0,
        )
    )

    print("\n" + "=" * 50)
    print("MEDIUM GRADE BREAKDOWN (Respect the Rain)")
    print("=" * 50)
    print(f"  Rain restraint     (x0.5): {rain_restraint:.4f}  -> {0.5 * rain_restraint:.4f}")
    print(f"  No drought         (x0.3): {drought_ok:.4f}  -> {0.3 * drought_ok:.4f}")
    print(f"  Water saved        (x0.2): {water_saved:.4f}  -> {0.2 * water_saved:.4f}")
    print(f"  TOTAL GRADE              : {final_score:.4f}")
    print("=" * 50 + "\n")

    return final_score


def grade_hard(episode_log: list[Any]) -> float:
    """Universal efficiency test ("Do More With Less")."""
    if not episode_log:
        print("Hard Grade: empty episode log -> 0.0")
        return 0.0

    total_steps = len(episode_log)
    n_zones = int(
        np.asarray(episode_log[0].get("stress_index", np.zeros(1))).size
    )

    final_stress = np.asarray(episode_log[-1].get("stress_index", [0.0]), dtype=float)
    mean_final = float(np.mean(final_stress))

    total_water = sum(float(s.get("water_used_step", 0.0)) for s in episode_log)
    water_fraction = total_water / max(total_steps * n_zones * 150.0, 1e-9)
    raw_yield_eff = (1.0 - mean_final) / max(water_fraction, 0.01)
    yield_eff = float(np.clip(raw_yield_eff / 5.0, 0.0, 1.0))

    consistency_hits = 0
    for s in episode_log:
        m = _mean_stress(s)
        w = float(s.get("water_used_step", 0.0))
        if 0.0 <= m <= 0.3 and w < (n_zones * 200.0):
            consistency_hits += 1
    consistency = consistency_hits / max(total_steps, 1)

    half = total_steps // 2
    second = episode_log[half:]
    if not second:
        no_waste_pressure = 0.0
    else:
        nw = sum(
            1
            for s in second
            if float(s.get("water_used_step", 0.0)) == 0.0
            and _mean_stress(s) < 0.25
        )
        no_waste_pressure = nw / len(second)

    final_score = float(
        np.clip(
            0.4 * yield_eff + 0.35 * consistency + 0.25 * no_waste_pressure,
            0.0,
            1.0,
        )
    )

    print("\n" + "=" * 50)
    print("HARD GRADE BREAKDOWN (Do More With Less)")
    print("=" * 50)
    print(f"  Yield / water      (x0.4): {yield_eff:.4f}  -> {0.4 * yield_eff:.4f}")
    print(f"  Low stress + lean  (x0.35): {consistency:.4f}  -> {0.35 * consistency:.4f}")
    print(f"  No waste 2nd half  (x0.25): {no_waste_pressure:.4f}  -> {0.25 * no_waste_pressure:.4f}")
    print(f"  TOTAL GRADE              : {final_score:.4f}")
    print("=" * 50 + "\n")

    return final_score
