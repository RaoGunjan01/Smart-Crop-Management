"""Rule-based (Baseline) agent for the Irrigation environment."""

from __future__ import annotations

import os
from typing import Any, Optional

import numpy as np

from irrigation_env.env import IrrigationEnv


class RuleBasedAgent:
    """An agent that follows simple, human-like irrigation rules."""

    SKIP = 0
    IRRIGATE_LOW = 1
    IRRIGATE_MED = 2
    IRRIGATE_HIGH = 3
    IRRIGATE_FLOOD = 4

    def act(self, obs: np.ndarray | None, state: dict[str, Any]) -> np.ndarray:
        """Select actions based on current state.

        Args:
            obs: Flat observation array (unused directly; state used instead).
            state: Full state dict from ``env.state()``.

        Returns:
            Action array of shape (n_zones + 1,).
        """
        # This agent is intentionally rule-based. It derives decisions from `state`,
        # but we keep it defensive so `/auto-step` doesn't crash if a key is missing.
        soil_moisture_raw = state.get("soil_moisture", None)
        if soil_moisture_raw is None:
            raise KeyError("soil_moisture missing from state")

        soil_moisture = np.asarray(soil_moisture_raw, dtype=float)
        stress = np.asarray(state.get("stress_index", np.zeros_like(soil_moisture)), dtype=float)
        dsi = np.asarray(state.get("days_since_irrigation", np.zeros_like(soil_moisture)), dtype=float)
        n = int(state.get("n_zones", soil_moisture.shape[0]))
        rain_forecast = float(state.get("rain_forecast_mm", 0.0))

        zone_actions = []

        for i in range(n):
            m = float(soil_moisture[i]) if i < soil_moisture.shape[0] else 0.0
            s = float(stress[i]) if i < stress.shape[0] else 0.0
            since = float(dsi[i]) if i < dsi.shape[0] else 0.0

            if rain_forecast > 10.0:
                zone_actions.append(self.SKIP)
            elif m < 0.18 or s > 0.60:
                zone_actions.append(self.IRRIGATE_HIGH)
            elif m < 0.26 or (m < 0.30 and s > 0.35):
                zone_actions.append(self.IRRIGATE_MED)
            elif m < 0.33 and since >= 2:
                zone_actions.append(self.IRRIGATE_LOW)
            elif m > 0.70:
                zone_actions.append(self.SKIP)
            else:
                zone_actions.append(self.SKIP)

        # Global action: always 0 (no special global command)
        global_action = 0
        return np.array(zone_actions + [global_action], dtype=np.int64)


def run_episode(
    env: IrrigationEnv,
    seed: Optional[int] = None,
    verbose: bool = True,
) -> list[dict[str, Any]]:
    """Run one full episode using the rule-based agent.

    Args:
        env: The environment instance.
        seed: Random seed for reset.
        verbose: Whether to print render info.

    Returns:
        The total episode log.
    """
    agent = RuleBasedAgent()
    obs, info = env.reset(seed=seed)
    done = False

    while not done:
        state = env.state()
        action = agent.act(obs, state)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        if verbose:
            env.render()

    return env.episode_log


if __name__ == "__main__":
    # Quick demo on easy task
    env = IrrigationEnv(task="easy")
    log = run_episode(env, seed=42)
    print(f"\nEpisode finished with {len(log)} steps.")
