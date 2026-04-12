---
title: Smart Crop Management System
emoji: 🌾
colorFrom: green
colorTo: blue
sdk: docker
app_port: 8000
pinned: false
---

# Smart Crop Management System

A complete **OpenEnv-compatible** Gymnasium environment for smart crop irrigation and nutrient-aware growth simulation. Designed for demo use, it simulates multi-zone soil moisture dynamics, weather, and crop growth.

---

## Features

- 🌱 Reference ET formulation aligned with FAO-56 Penman–Monteith methodology (simplified)
- 🌦️ Realistic weather simulation (temperature, rain, humidity)
- 🎯 Multi-zone irrigation with 5 action levels per zone
- 📊 Three difficulty levels (Easy / Medium / Hard)
- 🚀 FastAPI REST interface for agent integration
- 🤖 Rule-based baseline agent
- 🧪 Full pytest suite

---

## Real-World Applicability

This environment models **drip and sprinkler irrigation decisions** that farmers make day-to-day in **semi-arid regions** (for example India and sub-Saharan Africa), where water and pumping costs dominate margins.

- **`cost_per_liter` (0.008 INR/L)** is set in the same ballpark as **real Indian groundwater pumping and distribution costs**, so the economic pressure in the reward signal is grounded in practice.
- The **five crop types** (wheat, corn, rice, tomato, soybean) with **growth horizons of 120, 110, 130, 90, and 100 days** match **order-of-magnitude agronomic season lengths** used in planning and extension advice.
- **Evapotranspiration (ET)** in the simulator mirrors **FAO-56 reference evapotranspiration** thinking—temperature, humidity, crop stage, and **time-of-day (solar) effects**—so the water balance behaves like published crop water-use patterns, not arbitrary noise.
- A policy trained here could **inform or benchmark IoT-connected drip controllers**: the same discrete “how much per zone per time slot” structure maps naturally onto valve schedules and soil-moisture–driven automation.

---

## What Makes This Interesting

- **Multi-zone coordination:** Many irrigation benchmarks collapse to a single zone with binary on/off control. Here, actions are **MultiDiscrete with five levels per zone**, across **up to eight zones at once**—a **coordination problem** (like multi-agent scheduling) inside one environment.
- **Nutrient–water coupling:** **Nutrient status** scales growth and stress recovery, so water is not the only lever—agents face a **second, less obvious axis** that is uncommon in pure irrigation RL setups.
- **Diurnal structure:** Four **time-of-day slots** (morning, midday, afternoon, night) encode **real ET variation** (e.g. midday losses), so *when* you irrigate matters, not just *how much*.

---

## Quickstart

```bash
# Install dependencies
pip install -r requirements.txt

# Run baseline agent (easy task)
python -m agents.baseline_agent easy

# Run tests
pytest tests/ -v

# Start FastAPI server
uvicorn api.main:app --host 0.0.0.0 --port 8000

```

---

## API Endpoints

| Method | Endpoint | Body / Params | Description |
|--------|----------|---------------|-------------|
| `GET`  | `/health` | — | Health check |
| `POST` | `/reset` | `{"task": "easy", "seed": 42}` | Reset environment |
| `POST` | `/step`  | `{"action": [0, 0, ..., 0]}` | Take one step |
| `GET`  | `/state` | — | Full current state |

### Example: Reset and step

```bash
curl -X POST http://localhost:8000/reset \
  -H "Content-Type: application/json" \
  -d '{"task": "medium", "seed": 42}'

curl -X POST http://localhost:8000/step \
  -H "Content-Type: application/json" \
  -d '{"action": [2, 0, 0, 2, 0]}'
```

---

## Difficulty Levels

| Level  | Zones | Days | Noise | Budget (L) | Rain Prob | Sensor Failures |
|--------|-------|------|-------|------------|-----------|-----------------|
| Easy   | 1     | 30   | 0.00  | 10,000     | 30 %      | None            |
| Medium | 4     | 60   | 0.05  | 8,000      | 20 %      | None            |
| Hard   | 8     | 90   | 0.10  | 4,000      | 10 %      | 5 %/step        |

---

## State Space

The observation is a **flat `float32` numpy array**:

| Feature | Shape | Range | Description |
|---------|-------|-------|-------------|
| `soil_moisture[i]` | (n_zones,) | [0, 1] | Current soil moisture per zone |
| `crop_growth_stage[i]` | (n_zones,) | {0,1,2,3} | Seed/Vegetative/Flowering/Maturation |
| `stress_index[i]` | (n_zones,) | [0, 1] | Cumulative stress per zone |
| `days_since_irrigation[i]` | (n_zones,) | [0, 30] | Days since last irrigation |
| `temp_c` | scalar | [5, 45] | Ambient temperature (°C) |
| `humidity` | scalar | [0, 1] | Relative humidity |
| `rain_forecast_mm` | scalar | [0, 50] | Forecasted rainfall (mm) |
| `evapotranspiration` | scalar | [0, 10] | ET rate (mm/day) |
| `water_budget_remaining` | scalar | [0, 1] | Fraction of budget left |
| `cost_per_liter` | scalar | — | Water cost ($/L) |
| `day_of_season` | scalar | [0, 120] | Current day |
| `time_of_day` | scalar | {0,1,2,3} | Morning/Midday/Afternoon/Night |

---

## Action Space

`MultiDiscrete`: one action per zone (0–4) **plus** one global action slot.

| Code | Per-Zone Action | Description |
|------|-----------------|-------------|
| 0 | SKIP | No irrigation |
| 1 | IRRIGATE_LOW | 50 L |
| 2 | IRRIGATE_MED | 150 L |
| 3 | IRRIGATE_HIGH | 300 L |
| 4 | IRRIGATE_FLOOD | 600 L |
| 5 | EMERGENCY_IRRIGATE | Force HIGH on all zones |
| 6 | PAUSE_ALL | Skip all zones (override) |

---

## Reward Function

| Component | Weight | Condition |
|-----------|--------|-----------|
| Moisture target | +0.40 | All zones in [0.30, 0.60] |
| Water efficiency | +0.35 | `1 - (water_used / baseline)` |
| Stress penalty | −0.30 | Per zone where stress increased |
| Waterlogging | −0.20 | Per zone where moisture > 0.90 |
| Rain waste | −0.15 | Irrigating when forecast > 10 mm |
| Yield bonus | +1.00 | Episode end, avg yield score > 0.75 |

---

## Grader

```python
from irrigation_env.grader import grade_episode
score = grade_episode(episode_log)  # → float [0.0, 1.0]
```

| Component | Weight | Formula |
|-----------|--------|---------|
| Yield score | 0.4 | `1 - mean(final stress)` |
| Water efficiency | 0.3 | `1 - (water_used / budget)` |
| Stress events | 0.2 | `1 - (stress_steps / total_steps)` |
| Completion | 0.1 | `1.0` if finished within budget |

---

## Running Agents

### Baseline agent
```bash
python -m agents.baseline_agent easy
python -m agents.baseline_agent medium
python -m agents.baseline_agent hard
```

### Build your own agent
You can write your own agent that calls the API endpoints (`/reset`, `/step`, `/state`, `/auto-step`)
or directly interacts with `IrrigationEnv` in Python.

---

## Docker

```bash
docker build -t irrigation-rl .
docker run -p 8000:8000 irrigation-rl
```

---

## Project Structure

```
irrigation_env/
  __init__.py       ← package exports
  env.py            ← Gymnasium environment
  simulator.py      ← FAO-56 soil/weather simulator
  tasks.py          ← Easy/Medium/Hard task configs
  reward.py         ← isolated reward function
  grader.py         ← episode scoring
api/
  main.py           ← FastAPI wrapper
agents/
  baseline_agent.py ← rule-based agent
tests/
  test_env.py       ← pytest suite
models/             ← (unused)
Dockerfile
requirements.txt
README.md
```

---

## License

MIT
