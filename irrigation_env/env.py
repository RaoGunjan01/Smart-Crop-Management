from __future__ import annotations

from typing import Any, Optional

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from irrigation_env.simulator import SoilMoistureSimulator
from irrigation_env.tasks import TaskConfig, get_task
from irrigation_env.reward import compute_reward


PER_ZONE_ACTIONS = 5
GLOBAL_ACTION_OFFSET = PER_ZONE_ACTIONS


class IrrigationEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        task: str | TaskConfig = "easy",
        seed: Optional[int] = None,
        render_mode: Optional[str] = None,
    ) -> None:
        """Initialise the environment.

        Args:
            task: Task name ('easy', 'medium', 'hard') or a :class:`TaskConfig`.
            seed: Master random seed.
            render_mode: Render mode (currently 'human' only).
        """
        super().__init__()
        if isinstance(task, str):
            self.task_config: TaskConfig = get_task(task)
        else:
            self.task_config = task

        self.render_mode = render_mode
        self._master_seed = seed

        n = self.task_config.n_zones

        # ------------------------------------------------------------------
        # Observation space (flat)
        # ------------------------------------------------------------------
        obs_dim = n * 4 + 8  # 4 per-zone features + 8 global
        self.observation_space = spaces.Box(
            low=np.zeros(obs_dim, dtype=np.float32),
            high=np.ones(obs_dim, dtype=np.float32) * 100.0,
            dtype=np.float32,
        )

        # ------------------------------------------------------------------
        # Action space: per-zone (0-4) + global slot (0-6)
        # ------------------------------------------------------------------
        self.action_space = spaces.MultiDiscrete(
            [PER_ZONE_ACTIONS] * n + [7],  # last slot: 0-6 (0-4 unused, 5 or 6 used)
            dtype=np.int64,
        )

        # Simulator (created on reset)
        self._sim: Optional[SoilMoistureSimulator] = None

        self._episode_log: list[dict[str, Any]] = []
        self._accumulated_stress: list[float] = []
        self._step_count: int = 0
        self._cost_per_liter: float = 0.05

        self._nutrients: dict[str, float] = {"a": 60.0, "b": 50.0, "c": 55.0}
        self._pending_fertilizer_kg_ha: dict[str, float] = {"a": 0.0, "b": 0.0, "c": 0.0}
        self._last_fertilizer_kg_ha: dict[str, float] = {"a": 0.0, "b": 0.0, "c": 0.0}
        self._crop: str = "auto"
        self._season: str = "summer"
        self._land_ha: float = 1.0

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        effective_seed = seed if seed is not None else self._master_seed
        options = options or {}
        crop = str(options.get("crop") or "auto")
        season = str(options.get("season") or "summer")
        land_ha = float(options.get("land_ha") or 1.0)
        self._crop = crop
        self._season = season
        self._land_ha = max(0.1, land_ha)

        crop_days = {
            "wheat": 120,
            "corn": 110,
            "rice": 130,
            "tomato": 90,
            "soybean": 100,
        }
        if crop in crop_days:
            self.task_config.n_days = int(crop_days[crop])

        self._sim = SoilMoistureSimulator(self.task_config, seed=effective_seed)
        self._sim.season = season
        self._sim.land_ha = self._land_ha
        self._episode_log = []
        self._accumulated_stress = []
        self._step_count = 0

        if isinstance(options.get("nutrients"), dict):
            self.set_nutrients(options["nutrients"])
        self._pending_fertilizer_kg_ha = {"a": 0.0, "b": 0.0, "c": 0.0}
        self._last_fertilizer_kg_ha = {"a": 0.0, "b": 0.0, "c": 0.0}

        obs = self._build_observation()
        info = {"task": self.task_config.name, "n_zones": self.task_config.n_zones}
        return obs, info

    def set_nutrients(self, nutrients: dict[str, Any]) -> None:
        for k in ("a", "b", "c"):
            if k in nutrients:
                try:
                    v = float(nutrients[k])
                except Exception:
                    continue
                self._nutrients[k] = float(np.clip(v, 0.0, 100.0))

    def set_pending_fertilizer(self, fertilizer_kg_ha: dict[str, Any]) -> None:
        for k in ("a", "b", "c"):
            if k in fertilizer_kg_ha:
                try:
                    v = float(fertilizer_kg_ha[k])
                except Exception:
                    continue
                self._pending_fertilizer_kg_ha[k] = float(np.clip(v, 0.0, 50.0))

    def _advance_nutrients(self) -> None:
        for k in ("a", "b", "c"):
            self._nutrients[k] = float(np.clip(self._nutrients[k] - 0.25, 0.0, 100.0))

        applied = dict(self._pending_fertilizer_kg_ha)
        for k in ("a", "b", "c"):
            self._nutrients[k] = float(np.clip(self._nutrients[k] + applied[k] * 1.2, 0.0, 100.0))

        self._last_fertilizer_kg_ha = applied
        self._pending_fertilizer_kg_ha = {"a": 0.0, "b": 0.0, "c": 0.0}

    # ------------------------------------------------------------------
    # step
    # ------------------------------------------------------------------

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Take one environment step.

        Args:
            action: Array of shape (n_zones + 1,). Last element is global action.

        Returns:
            (observation, reward, terminated, truncated, info)
        """
        assert self._sim is not None, "Call reset() before step()."
        n = self.task_config.n_zones
        zone_actions = [int(action[i]) for i in range(n)]
        global_action = int(action[n])

        # Apply nutrient dynamics for this step (decay + any scheduled fertilizer)
        self._advance_nutrients()
        # Provide nutrient context to simulator for growth/stress dynamics
        avg_n = float(np.mean([self._nutrients["a"], self._nutrients["b"], self._nutrients["c"]])) / 100.0
        self._sim.nutrient_factor = float(np.clip(avg_n, 0.0, 1.0))

        # Capture state before step
        pre_state = self._get_state_dict()
        pre_state["rain_forecast_mm"] = self._sim.rain_forecast_mm

        # Advance simulation
        sim_result = self._sim.step(zone_actions, global_action)
        water_used = self._sim.water_used_liters

        # Build next_state dict for reward
        next_state_dict = self._get_state_dict()
        next_state_dict["water_used_step"] = sim_result["water_this_step"]
        next_state_dict["water_used_liters"] = water_used

        # Termination conditions
        self._step_count += 1
        terminated = self._sim.current_step >= self._sim.total_steps
        truncated = False

        # Accumulate stress for yield calculation
        self._accumulated_stress.append(float(np.mean(next_state_dict["stress_index"])))

        # Compute reward
        action_dict = {"zone_actions": zone_actions, "global_action": global_action}
        reward, reward_info = compute_reward(
            state=pre_state,
            action=action_dict,
            next_state=next_state_dict,
            task_config=self.task_config,
            episode_done=terminated,
            accumulated_stress=self._accumulated_stress,
        )

        # Log step for grader
        log_entry: dict[str, Any] = {
            "step": self._step_count,
            "stress_index": sim_result["stress_index"].copy(),
            "soil_moisture": sim_result["soil_moisture"].copy(),
            "water_used_step": sim_result["water_this_step"],
            "stress_increased": bool(
                np.any(sim_result["stress_index"] > sim_result["prev_stress"])
            ),
            "terminated": terminated,
            "reward": reward,
        }
        self._episode_log.append(log_entry)

        obs = self._build_observation()
        info = {
            **reward_info,
            "day": self._sim.day,
            "time_of_day": self._sim.time_of_day,
            "water_used_liters": water_used,
            "water_used_step": sim_result["water_this_step"],
            "fertilizer_kg_ha_step": dict(self._last_fertilizer_kg_ha),
            "last_action": action_dict,
            "water_budget_remaining": 1.0,
            "budget_exhausted": False,
            "episode_log": self._episode_log,
        }

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # state
    # ------------------------------------------------------------------

    def state(self) -> dict[str, Any]:
        """Return the full current environment state as a Python dict.

        Returns:
            Dict with keys matching the OpenEnv state schema.
        """
        assert self._sim is not None, "Call reset() before state()."
        return self._get_state_dict()

    # ------------------------------------------------------------------
    # render
    # ------------------------------------------------------------------

    def render(self) -> None:
        """Print a human-readable state summary to stdout."""
        assert self._sim is not None, "Call reset() before render()."
        sim = self._sim
        n = self.task_config.n_zones
        print("\n───────────────────────────────────────────")
        print(f"  Day {sim.day:3d}/{self.task_config.n_days}  "
              f"Time: {['Morning','Midday','Afternoon','Night'][sim.time_of_day]}  "
              f"Step: {self._step_count}")
        print(f"  Temp: {sim.temp_c:.1f}°C  Humidity: {sim.humidity:.2f}  "
              f"Rain Forecast: {sim.rain_forecast_mm:.1f}mm")
        print(f"  ET: {sim.evapotranspiration:.2f}mm  "
              f"Water Used: {sim.water_used_liters:.0f}L / "
              f"{self.task_config.water_budget_liters:.0f}L")
        for i in range(n):
            stage_names = ["Seed", "Vegetative", "Flowering", "Maturation"]
            stage = int(sim.crop_growth_stage[i])
            print(
                f"  Zone {i}: moisture={sim.soil_moisture[i]:.3f}  "
                f"stress={sim.stress_index[i]:.3f}  "
                f"stage={stage_names[stage]}  "
                f"DSI={sim.days_since_irrigation[i]}"
            )
        print("───────────────────────────────────────────\n")

    def close(self) -> None:
        self._sim = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_state_dict(self) -> dict[str, Any]:
        sim = self._sim
        budget = self.task_config.water_budget_liters
        return {
            "soil_moisture": sim.soil_moisture.copy(),
            "crop_growth_stage": sim.crop_growth_stage.copy(),
            "stress_index": sim.stress_index.copy(),
            "days_since_irrigation": sim.days_since_irrigation.copy(),
            "temp_c": float(sim.temp_c),
            "humidity": float(sim.humidity),
            "rain_forecast_mm": float(sim.rain_forecast_mm),
            "evapotranspiration": float(sim.evapotranspiration),
            "water_budget_remaining": float(
                max(0.0, budget - sim.water_used_liters) / max(budget, 1.0)
            ),
            "cost_per_liter": float(self._cost_per_liter),
            "day_of_season": int(sim.day),
            "time_of_day": int(sim.time_of_day),
            "n_zones": self.task_config.n_zones,
            "traditional_water_liters": getattr(sim, "traditional_water_liters", 0.0),
            "water_used_liters": float(sim.water_used_liters),
            "n_days": int(self.task_config.n_days),
            "nutrients": dict(self._nutrients),
            "fertilizer_kg_ha_last": dict(self._last_fertilizer_kg_ha),
            "crop": self._crop,
            "season": self._season,
            "land_ha": float(self._land_ha),
            "max_steps_per_day": int(getattr(self.task_config, "max_steps_per_day", 4)),
        }

    def _build_observation(self) -> np.ndarray:
        sim = self._sim
        n = self.task_config.n_zones
        budget = self.task_config.water_budget_liters

        per_zone = np.concatenate([
            sim.soil_moisture.astype(np.float32),
            sim.crop_growth_stage.astype(np.float32),
            sim.stress_index.astype(np.float32),
            np.clip(sim.days_since_irrigation / 30.0, 0.0, 1.0).astype(np.float32),
        ])

        global_obs = np.array([
            (sim.temp_c - 5.0) / 40.0,           # normalised temp
            sim.humidity,
            sim.rain_forecast_mm / 50.0,          # normalised rain
            sim.evapotranspiration / 10.0,         # normalised ET
            max(0.0, budget - sim.water_used_liters) / max(budget, 1.0),
            self._cost_per_liter,
            sim.day / max(self.task_config.n_days, 1),
            sim.time_of_day / 3.0,
        ], dtype=np.float32)

        return np.concatenate([per_zone, global_obs]).astype(np.float32)

    @property
    def episode_log(self) -> list[dict[str, Any]]:
        return self._episode_log
