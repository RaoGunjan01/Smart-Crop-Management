from __future__ import annotations

from typing import Optional

import numpy as np


# Irrigation volume in litres per zone action level
IRRIGATION_LITRES: dict[int, float] = {
    0: 0.0,    # SKIP
    1: 50.0,   # IRRIGATE_LOW
    2: 150.0,  # IRRIGATE_MED
    3: 300.0,  # IRRIGATE_HIGH
    4: 600.0,  # IRRIGATE_FLOOD
}

# Effective soil moisture fraction added per litre (per zone, per unit area)
LITRE_TO_MOISTURE_RATIO = 0.001


class SoilMoistureSimulator:
    def __init__(self, task_config: Any, seed: Optional[int] = None) -> None:
        self.n_zones = task_config.n_zones
        self.task_name = task_config.name
        self.total_steps = task_config.n_days * 4
        self.water_budget = task_config.water_budget_liters
        self.rng = np.random.default_rng(seed)
        self.rain_probability = float(getattr(task_config, "rain_probability", 0.1))

        # Environment State
        self.current_step = 0
        self.soil_moisture = np.full(self.n_zones, 0.45, dtype=np.float32)
        self.crop_growth_stage = np.zeros(self.n_zones, dtype=np.int32)
        self.stress_index = np.zeros(self.n_zones, dtype=np.float32)
        self.days_since_irrigation = np.zeros(self.n_zones, dtype=np.int32)
        self.water_used_liters = 0.0
        self._growth_progress = np.zeros(self.n_zones, dtype=np.float32)

        # Set by the environment each step (0-1). Higher nutrients => faster growth and better resilience.
        self.nutrient_factor: float = 0.65
        self.season: str = "summer"
        self.land_ha: float = 1.0

        # Global weather placeholders
        self.base_temp_c = 25.0
        self.base_humidity = 0.55
        self.temp_c = 25.0
        self.humidity = 0.55
        self.rain_forecast_mm = 0.0
        
        # 🧪 Soil Physics (Hackathon Winner Logic)
        # Easy = High Retention (Clay), Hard = Fast Drain (Sand)
        if self.task_name == "hard":
            self.soil_conductivity = 0.12  # Drains fast
            self.max_infiltration_rate = 15.0 # mm per step
        else:
            self.soil_conductivity = 0.05  # Retains water
            self.max_infiltration_rate = 8.0

        self.traditional_water_liters = 0.0

    def step(
        self,
        zone_actions: list[int],
        global_action: int,
    ) -> dict[str, np.ndarray | float]:
        """Advance simulation by 6 hours."""
        prev_stress = self.stress_index.copy()

        # Baseline "traditional" irrigation benchmark per step (liters).
        self.traditional_water_liters += 25.0 * self.n_zones * float(self.land_ha)

        # 1. Global action override (Same as before)
        if global_action == 6:  # PAUSE_ALL
            zone_actions = [0] * self.n_zones
        elif global_action == 5:  # EMERGENCY_IRRIGATE
            zone_actions = [max(a, 2) for a in zone_actions]

        # 2. Apply Irrigation (Simplified infiltration)
        water_this_step = 0.0
        for i, action in enumerate(zone_actions):
            liters = IRRIGATION_LITRES.get(action, 0.0) * float(self.land_ha)
            water_this_step += liters
            
            # Realism: Some irrigation is lost if intensity is too high (Flood)
            efficiency = 1.0 if action < 4 else 0.85 
            self.soil_moisture[i] += (liters * efficiency) * LITRE_TO_MOISTURE_RATIO
            
            if liters > 0:
                self.days_since_irrigation[i] = 0
            else:
                self.days_since_irrigation[i] += 1

        self.water_used_liters += water_this_step

        # 3. Simulate Weather & ET
        self._update_weather()
        effective_et = self._calculate_effective_et()

        # 4. Update Soil Moisture & Stress
        for i in range(self.n_zones):
            # A. Natural drying (ET + Soil Drift)
            self.soil_moisture[i] -= (effective_et / 12.0) + (self.soil_conductivity / 100.0)
            
            # B. Rain Physics (Infiltration vs Runoff)
            rain_step = self.rain_forecast_mm 
            effective_rain = min(rain_step, self.max_infiltration_rate)
            self.soil_moisture[i] += effective_rain / 100.0
            
            self.soil_moisture[i] = np.clip(self.soil_moisture[i], 0.0, 1.0)

            # C. Dynamic Stress Index
            if self.soil_moisture[i] < 0.3:
                # Severity increases with heat
                heat_multiplier = 1.0 + max(0.0, (self.temp_c - 30.0) / 10.0)
                # Low nutrients make drought stress worse
                nut_mult = 1.0 + max(0.0, (0.6 - float(self.nutrient_factor)) * 1.5)
                self.stress_index[i] += 0.05 * (0.3 - self.soil_moisture[i]) * heat_multiplier * nut_mult
            elif self.soil_moisture[i] > 0.85:
                # Waterlogging stress
                self.stress_index[i] += 0.02 * (self.soil_moisture[i] - 0.85)
            else:
                # Better nutrients => faster recovery when conditions are OK
                recovery = 0.88 - min(0.18, float(self.nutrient_factor) * 0.18)
                self.stress_index[i] *= recovery

            self.stress_index[i] = np.clip(self.stress_index[i], 0.0, 1.0)

            # Growth simulation: nutrients + good moisture accelerate growth
            nf = float(np.clip(self.nutrient_factor, 0.0, 1.0))
            moist_bonus = 0.006 if 0.30 <= float(self.soil_moisture[i]) <= 0.60 else 0.0
            self._growth_progress[i] += (0.004 + 0.010 * nf) + moist_bonus
            self.crop_growth_stage[i] = min(3, int(self._growth_progress[i]))

        self.current_step += 1

        return {
            "soil_moisture": self.soil_moisture.copy(),
            "stress_index": self.stress_index.copy(),
            "prev_stress": prev_stress,
            "water_this_step": water_this_step,
        }

    def _update_weather(self) -> None:
        # Step: 0=Morning, 1=Midday, 2=Afternoon, 3=Night
        tod = self.time_of_day

        self.base_temp_c += self.rng.uniform(-0.25, 0.25)
        season_temp = {
            "spring": (14.0, 30.0),
            "summer": (22.0, 40.0),
            "monsoon": (20.0, 34.0),
            "winter": (6.0, 22.0),
        }
        tmin, tmax = season_temp.get(self.season, (18.0, 32.0))
        self.base_temp_c = float(np.clip(self.base_temp_c, tmin, tmax))

        self.base_humidity += self.rng.uniform(-0.01, 0.01)
        season_h = {
            "spring": (0.25, 0.75),
            "summer": (0.20, 0.70),
            "monsoon": (0.55, 0.95),
            "winter": (0.20, 0.80),
        }
        hmin, hmax = season_h.get(self.season, (0.3, 0.8))
        self.base_humidity = float(np.clip(self.base_humidity, hmin, hmax))

        diurnal_temp = [-2.0, +3.0, +1.0, -3.5][tod]
        diurnal_h = [+0.05, -0.08, -0.02, +0.10][tod]
        self.temp_c = float(np.clip(self.base_temp_c + diurnal_temp, 5.0, 55.0))
        self.humidity = float(np.clip(self.base_humidity + diurnal_h, 0.05, 0.95))

        # Rain pattern
        season_rain = {"spring": 0.10, "summer": 0.06, "monsoon": 0.18, "winter": 0.03}
        rain_prob = float(season_rain.get(self.season, 0.08)) * max(0.0, self.rain_probability)
        if self.rng.random() < rain_prob:
            self.rain_forecast_mm = self.rng.uniform(5.0, 30.0)
        else:
            self.rain_forecast_mm = max(0.0, self.rain_forecast_mm - 8.0)

    def _calculate_effective_et(self) -> float:
        # 🌡️ Temperature/Humidity base
        base_et = (self.temp_c / 18.0) * (1.2 - self.humidity)
        
        # 🌱 Crop Growth Impact (Older plants use more water)
        # Stage 0: 1.0x, Stage 1: 1.25x, Stage 2: 1.6x, Stage 3: 2.0x
        avg_stage = np.mean(self.crop_growth_stage)
        crop_k = 1.0 + (avg_stage * 0.33)
        
        # 🌬️ Simulating Wind/Sun intensity based on time of day
        solar_proxy = [0.7, 1.5, 1.2, 0.2][self.time_of_day]
        
        final_et = base_et * crop_k * solar_proxy
            
        return float(max(0.05, final_et))

    @property
    def evapotranspiration(self) -> float:
        """Compatibility property for external access."""
        return self._calculate_effective_et()

    @property
    def day(self) -> int:
        return self.current_step // 4

    @property
    def time_of_day(self) -> int:
        return self.current_step % 4
