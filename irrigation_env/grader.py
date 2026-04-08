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
