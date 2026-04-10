"""Isolated, testable reward function for the Irrigation RL environment."""

from __future__ import annotations

from typing import Any

import numpy as np

from irrigation_env.tasks import TaskConfig

# Baseline water use per step per zone in litres (IRRIGATE_MED amount)
BASELINE_WATER_PER_STEP_PER_ZONE: float = 150.0


def compute_reward(
    state: dict[str, Any],
    action: dict[str, Any],
    next_state: dict[str, Any],
    task_config: TaskConfig,
    episode_done: bool = False,
    accumulated_stress: list[float] | None = None,
) -> tuple[float, dict[str, Any]]:
    """Compute the step reward and return a breakdown info dict.

    Args:
        state: Environment state *before* this step.
        action: Action taken this step (keys: 'zone_actions', 'global_action').
        next_state: Environment state *after* this step.
        task_config: The current task configuration.
        episode_done: Whether the episode just ended (triggers yield bonus).
        accumulated_stress: Per-zone list of mean stress over the episode so far.

    Returns:
        (reward, info) where info contains each reward component.
    """
    n = task_config.n_zones
    moisture = np.asarray(next_state["soil_moisture"])
    stress_now = np.asarray(next_state["stress_index"])
    stress_prev = np.asarray(state["stress_index"])
    rain_forecast = float(state.get("rain_forecast_mm", 0.0))
    water_this_step = float(next_state.get("water_used_step", 0.0))
    water_budget = task_config.water_budget_liters
    # water_used_total = float(next_state.get("water_used_liters", 0.0))

    components: dict[str, float] = {}

    # ------------------------------------------------------------------
    # +0.40  all zones in moisture target range [0.30, 0.60]
    # ------------------------------------------------------------------
    zones_in_range = np.sum((moisture >= 0.30) & (moisture <= 0.60))
    moisture_score = (zones_in_range / n) * 0.40
    components["moisture_target"] = float(moisture_score)

    # ------------------------------------------------------------------
    # +0.35  water efficiency ratio
    # ------------------------------------------------------------------
    baseline_water = BASELINE_WATER_PER_STEP_PER_ZONE * n
    if baseline_water > 0:
        water_efficiency_ratio = max(0.0, 1.0 - (water_this_step / baseline_water))
    else:
        water_efficiency_ratio = 1.0
    efficiency_reward = water_efficiency_ratio * 0.35
    components["water_efficiency"] = float(efficiency_reward)

    # ------------------------------------------------------------------
    # −0.30  per zone where stress_index increased this step
    # ------------------------------------------------------------------
    stress_increased = np.sum(stress_now > stress_prev)
    stress_penalty = -0.30 * float(stress_increased)
    components["stress_penalty"] = float(stress_penalty)

    # ------------------------------------------------------------------
    # −0.20  per zone where soil_moisture > 0.9 (waterlogging)
    # ------------------------------------------------------------------
    waterlogged = np.sum(moisture > 0.9)
    waterlog_penalty = -0.20 * float(waterlogged)
    components["waterlog_penalty"] = float(waterlog_penalty)

    # ------------------------------------------------------------------
    # −0.15  if irrigation applied when rain_forecast_mm > 10
    # ------------------------------------------------------------------
    zone_actions = action.get("zone_actions", [0] * n)
    any_irrigated = any(a > 0 for a in zone_actions)
    rain_waste_penalty = 0.0
    if rain_forecast > 10.0 and any_irrigated:
        rain_waste_penalty = -0.15
    components["rain_waste_penalty"] = float(rain_waste_penalty)

    # Medium task: reward learning to wait when significant rain is forecast.
    rain_wait_reward = 0.0
    if task_config.name == "medium" and rain_forecast > 10.0 and not any_irrigated:
        rain_wait_reward = 0.10
    components["rain_wait_reward"] = float(rain_wait_reward)

    # ------------------------------------------------------------------
    # Nutrients: reward balanced soil, penalize deficiency/excess, and add a small fertilizer cost
    # ------------------------------------------------------------------
    nutrients = next_state.get("nutrients", {})
    fert = next_state.get("fertilizer_kg_ha_last", {})
    try:
        a = float(nutrients.get("a", 55.0))
        b = float(nutrients.get("b", 55.0))
        c = float(nutrients.get("c", 55.0))
        avg_n = (a + b + c) / 3.0
    except Exception:
        avg_n = 55.0

    nutrient_reward = 0.0
    if 50.0 <= avg_n <= 80.0:
        nutrient_reward = 0.15
    elif avg_n < 35.0:
        nutrient_reward = -0.15
    elif avg_n > 92.0:
        nutrient_reward = -0.10
    components["nutrient_balance"] = float(nutrient_reward)

    try:
        fert_total = float(fert.get("a", 0.0)) + float(fert.get("b", 0.0)) + float(fert.get("c", 0.0))
    except Exception:
        fert_total = 0.0
    fert_cost = -0.01 * (fert_total / 10.0)  # small cost so "more fertilizer" isn't always better
    components["fertilizer_cost"] = float(fert_cost)

    # Hard task: prioritise critical growth stages and ration under scarcity.
    critical_stage_penalty = 0.0
    scarcity_penalty = 0.0
    if task_config.name == "hard":
        stages = np.asarray(next_state.get("crop_growth_stage", np.zeros(n)))
        critical_mask = stages >= 2  # flowering + maturity windows
        critical_stress = np.sum(critical_mask & (stress_now > 0.35))
        critical_stage_penalty = -0.18 * float(critical_stress)

        budget_remaining = float(next_state.get("water_budget_remaining", 1.0))
        if budget_remaining < 0.35 and baseline_water > 0:
            scarcity_penalty = -0.12 * float(water_this_step / baseline_water)
    components["critical_stage_penalty"] = float(critical_stage_penalty)
    components["scarcity_penalty"] = float(scarcity_penalty)

    # ------------------------------------------------------------------
    # +1.00  episode-end bonus if average_yield_score > 0.75
    # ------------------------------------------------------------------
    yield_bonus = 0.0
    if episode_done and accumulated_stress is not None:
        yield_score = 1.0 - float(np.mean(accumulated_stress))
        if yield_score > 0.75:
            yield_bonus = 1.0
        components["yield_score"] = float(yield_score)
    components["yield_bonus"] = float(yield_bonus)

    # Total
    total = sum(components.values())
    components["total"] = float(total)

    info = {
        "reward_components": components,
        "water_efficiency_ratio": float(water_efficiency_ratio),
        "zones_in_moisture_range": int(zones_in_range),
        "stress_increased_zones": int(stress_increased),
        "waterlogged_zones": int(waterlogged),
    }
    return float(total), info
