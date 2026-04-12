"""Observation-driven rule-based policy for the irrigation environment.

This module provides a strong heuristic agent that reads the flat observation
vector (not full ``state()``), demonstrating that the task is solvable without
learned parameters. It is not a trained RL policy.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from irrigation_env.env import IrrigationEnv


class RuleBasedAgent:
    """Heuristic policy using soil moisture, stress, and rain from ``obs``."""

    SKIP = 0
    IRRIGATE_LOW = 1
    IRRIGATE_MED = 2

    def act(self, obs: np.ndarray | None, state: dict[str, Any]) -> np.ndarray:
        """Build per-zone + global actions from the observation vector.

        Args:
            obs: Flat observation from ``env.step`` / ``env.reset`` (required).
            state: Ignored; kept for API compatibility with ``baseline_agent``.

        Returns:
            Action array of shape ``(n_zones + 1,)``.
        """
        if obs is None:
            raise ValueError("obs is required for RuleBasedAgent.act()")
        obs_arr = np.asarray(obs, dtype=np.float64)
        dim = int(obs_arr.shape[0])
        n = (dim - 8) // 4
        if n < 1 or dim != n * 4 + 8:
            raise ValueError(f"Unexpected observation shape {obs_arr.shape}")

        soil = obs_arr[0:n]
        stress = obs_arr[2 * n : 3 * n]
        if stress.shape != (n,) or not np.isfinite(stress).all():
            raise ValueError("obs stress_index slice invalid")
        rain_norm = float(obs_arr[4 * n + 2])
        rain_forecast_mm = rain_norm * 50.0

        zone_actions: list[int] = []
        for i in range(n):
            moisture = float(soil[i])

            if rain_forecast_mm > 10.0:
                zone_actions.append(self.SKIP)
            elif moisture < 0.35:
                zone_actions.append(self.IRRIGATE_MED)
            elif 0.35 <= moisture <= 0.45:
                zone_actions.append(self.IRRIGATE_LOW)
            else:
                zone_actions.append(self.SKIP)

        global_action = 0
        return np.array(zone_actions + [global_action], dtype=np.int64)


def run_episode(
    env: IrrigationEnv,
    seed: Optional[int] = None,
    verbose: bool = True,
) -> list[dict[str, Any]]:
    """Run one full episode using the observation-based rule policy.

    Same signature as ``agents.baseline_agent.run_episode`` for drop-in use.
    """
    agent = RuleBasedAgent()
    obs, _info = env.reset(seed=seed)
    done = False

    while not done:
        state = env.state()
        action = agent.act(obs, state)
        obs, _reward, terminated, truncated, _info = env.step(action)
        done = terminated or truncated

        if verbose:
            env.render()

    return env.episode_log


if __name__ == "__main__":
    env = IrrigationEnv(task="easy")
    log = run_episode(env, seed=42, verbose=False)
    print(f"Episode finished with {len(log)} steps.")
