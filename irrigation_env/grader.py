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


def grade_easy(episode_log: list[Any]) -> float:
    """Grade easy-task episodes on a 0.0-1.0 scale."""
    if not episode_log:
        print("Easy Grade: empty episode log -> 0.0")
        return 0.0

    total_steps = len(episode_log)
    last_step = episode_log[-1]

    final_stress_raw = last_step.get("stress_index", 1.0)
    final_stress_arr = np.asarray(final_stress_raw, dtype=float)
    final_stress = float(np.mean(final_stress_arr))

    if final_stress < 0.3:
        survival_score = 1.0
    elif final_stress < 0.6:
        survival_score = 0.5
    else:
        survival_score = 0.0

    stress_increased_steps = sum(1 for step in episode_log if step.get("stress_increased", False))
    consistency_score = 1.0 - (stress_increased_steps / max(total_steps, 1))
    consistency_score = float(np.clip(consistency_score, 0.0, 1.0))

    completion_score = 1.0 if (last_step.get("terminated", False) and not last_step.get("budget_exhausted", False)) else 0.0

    final_score = (
        0.6 * survival_score
        + 0.3 * consistency_score
        + 0.1 * completion_score
    )
    final_score = float(np.clip(final_score, 0.0, 1.0))

    print("\n" + "=" * 50)
    print("EASY GRADE BREAKDOWN")
    print("=" * 50)
    print(f"  Survival Score     (x0.6): {survival_score:.4f}  -> {0.6 * survival_score:.4f}")
    print(f"  Consistency Score  (x0.3): {consistency_score:.4f}  -> {0.3 * consistency_score:.4f}")
    print(f"  Completion Score   (x0.1): {completion_score:.4f}  -> {0.1 * completion_score:.4f}")
    print(f"  TOTAL GRADE              : {final_score:.4f}")
    print("=" * 50 + "\n")

    return final_score


def grade_medium(episode_log: list[Any]) -> float:
    """Grade medium-task episodes on a 0.0-1.0 scale."""
    if not episode_log:
        print("Medium Grade: empty episode log -> 0.0")
        return 0.0

    total_steps = len(episode_log)
    last_step = episode_log[-1]

    final_stress = np.asarray(last_step.get("stress_index", [1.0]), dtype=float)
    yield_score = float(np.clip(1.0 - np.mean(final_stress), 0.0, 1.0))

    total_water_used = sum(float(step.get("water_used_step", 0.0)) for step in episode_log)
    total_water_budget = float(episode_log[0].get("water_budget", 1.0))
    water_fraction = total_water_used / max(total_water_budget, 1.0)

    efficiency_score = (yield_score / max(water_fraction, 1e-9)) / 10.0
    efficiency_score = float(np.clip(efficiency_score, 0.0, 1.0))

    rain_waste_steps = sum(
        1
        for step in episode_log
        if float(step.get("reward_components", {}).get("rain_waste_penalty", 0.0)) < 0.0
    )
    no_waste_score = 1.0 - (rain_waste_steps / max(total_steps, 1))
    no_waste_score = float(np.clip(no_waste_score, 0.0, 1.0))

    final_score = (
        0.5 * efficiency_score
        + 0.3 * yield_score
        + 0.2 * no_waste_score
    )
    final_score = float(np.clip(final_score, 0.0, 1.0))

    print("\n" + "=" * 50)
    print("MEDIUM GRADE BREAKDOWN")
    print("=" * 50)
    print(f"  Efficiency Score   (x0.5): {efficiency_score:.4f}  -> {0.5 * efficiency_score:.4f}")
    print(f"  Yield Score        (x0.3): {yield_score:.4f}  -> {0.3 * yield_score:.4f}")
    print(f"  No Waste Score     (x0.2): {no_waste_score:.4f}  -> {0.2 * no_waste_score:.4f}")
    print(f"  TOTAL GRADE              : {final_score:.4f}")
    print("=" * 50 + "\n")

    return final_score


def grade_hard(episode_log: list[Any]) -> float:
    """Grade hard-task episodes on a 0.0-1.0 scale."""
    if not episode_log:
        print("Hard Grade: empty episode log -> 0.0")
        return 0.0

    total_steps = len(episode_log)

    recovery_lags: list[int] = []
    in_spike = False
    spike_start_idx = 0

    for idx, step in enumerate(episode_log):
        avg_stress = float(np.mean(np.asarray(step.get("stress_index", [0.0]), dtype=float)))

        if not in_spike and avg_stress > 0.4:
            in_spike = True
            spike_start_idx = idx
        elif in_spike and avg_stress < 0.3:
            recovery_lags.append(idx - spike_start_idx)
            in_spike = False

    if recovery_lags:
        avg_recovery_lag = float(np.mean(recovery_lags))
        resilience_score = float(np.clip(1.0 - (avg_recovery_lag / 10.0), 0.0, 1.0))
    else:
        resilience_score = 1.0

    final_stress = np.asarray(episode_log[-1].get("stress_index", [0.0]), dtype=float)
    zone_balance_score = float(np.clip(1.0 - np.std(final_stress), 0.0, 1.0))

    last_step = episode_log[-1]
    completed = bool(last_step.get("terminated", False))
    budget_exhausted = bool(last_step.get("budget_exhausted", False))
    if completed and not budget_exhausted:
        endurance_score = 1.0
    else:
        endurance_score = float(np.clip(total_steps / 360.0, 0.0, 0.9))

    final_score = (
        0.4 * resilience_score
        + 0.3 * zone_balance_score
        + 0.3 * endurance_score
    )
    final_score = float(np.clip(final_score, 0.0, 1.0))

    print("\n" + "=" * 50)
    print("HARD GRADE BREAKDOWN")
    print("=" * 50)
    print(f"  Resilience Score   (x0.4): {resilience_score:.4f}  -> {0.4 * resilience_score:.4f}")
    print(f"  Zone Balance Score (x0.3): {zone_balance_score:.4f}  -> {0.3 * zone_balance_score:.4f}")
    print(f"  Endurance Score    (x0.3): {endurance_score:.4f}  -> {0.3 * endurance_score:.4f}")
    print(f"  TOTAL GRADE              : {final_score:.4f}")
    print("=" * 50 + "\n")

    return final_score
