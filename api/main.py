from __future__ import annotations

import os
import math
import threading
import webbrowser
from typing import Any, Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict

from irrigation_env.env import IrrigationEnv

app = FastAPI(title="Smart Crop Management Sytem")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_env: Optional[IrrigationEnv] = None

ui_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "ui")
app.mount("/ui", StaticFiles(directory=ui_path), name="ui")

@app.on_event("startup")
async def _startup() -> None:
    if os.getenv("HF_SPACE_ID") or os.getenv("SPACE_ID"):
        return
    def _open() -> None:
        try:
            port = os.getenv("PORT", "8000")
            webbrowser.open(f"http://localhost:{port}/ui/dashboard.html", new=2)
        except Exception:
            pass
    threading.Timer(0.5, _open).start()

@app.get("/")
async def read_root():
    return FileResponse(os.path.join(ui_path, "dashboard.html"))

class ResetRequest(BaseModel):
    task: str = "easy"
    seed: Optional[int] = 42
    crop: Optional[str] = None
    season: Optional[str] = None
    land_ha: Optional[float] = None
    nutrients: Optional[dict[str, float]] = None

class StepRequest(BaseModel):
    action: list[int]

class HealthResponse(BaseModel):
    status: str = "ok"
    rl_active: bool = False

class ResetResponse(BaseModel):
    observation: list[float]
    info: dict[str, Any]
    task: str
    n_zones: int

class StepResponse(BaseModel):
    observation: list[float]
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, Any]
    reasoning: Optional[str] = None

class StateResponse(BaseModel):
    model_config = ConfigDict(extra="allow")
    soil_moisture: list[float]
    stress_index: list[float]
    water_used_liters: float
    traditional_water_liters: float
    cost_per_liter: float

@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(status="ok", rl_active=False)


@app.post("/reset", response_model=ResetResponse)
async def reset(request: ResetRequest) -> ResetResponse:
    global _env
    try:
        _env = IrrigationEnv(task=request.task)
        obs, info = _env.reset(
            seed=request.seed,
            options={
                "crop": request.crop,
                "season": request.season,
                "land_ha": request.land_ha,
                "nutrients": request.nutrients or {},
            },
        )

        return ResetResponse(
            observation=_obs_to_list(obs),
            info=info,
            task=request.task,
            n_zones=_env.task_config.n_zones,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reset failed: {str(e)}")


@app.post("/step", response_model=StepResponse)
async def step(request: StepRequest) -> StepResponse:
    if _env is None:
        raise HTTPException(status_code=400, detail="Call /reset first.")

    try:
        action_arr = np.array(request.action, dtype=np.int64)
        obs, reward, terminated, truncated, info = _env.step(action_arr)

        info_serialisable = {
            k: (v.tolist() if isinstance(v, np.ndarray) else v)
            for k, v in info.items()
            if k != "episode_log"
        }

        return StepResponse(
            observation=_obs_to_list(obs),
            reward=float(reward),
            terminated=terminated,
            truncated=truncated,
            info=info_serialisable,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Step failed: {str(e)}")


@app.post("/auto-step", response_model=StepResponse)
async def auto_step() -> StepResponse:
    if _env is None:
        raise HTTPException(status_code=400, detail="Call /reset first.")

    try:
        st = _env.state()
        agent_obs = _env._build_observation()
        avg_moist = float(np.mean(st["soil_moisture"]))
        temp = st["temp_c"]
        rain = st["rain_forecast_mm"]
        
        from agents.baseline_agent import RuleBasedAgent
        action_arr = RuleBasedAgent().act(agent_obs, st)
        action_list = (
            action_arr.tolist()
            if hasattr(action_arr, "tolist")
            else list(action_arr)
        )
        reason_prefix = "Rule System: "

        fert = {}
        try:
            nst = st.get("nutrients", {})
            a = float(nst.get("a", 55.0))
            b = float(nst.get("b", 55.0))
            c = float(nst.get("c", 55.0))
            avg_n = (a + b + c) / 3.0
            deficit = max(0.0, 65.0 - avg_n)
            per = float(np.clip(deficit * 0.25, 0.0, 10.0))
            fert = {"a": per, "b": per, "c": per}
        except Exception:
            fert = {"a": 0.0, "b": 0.0, "c": 0.0}
        _env.set_pending_fertilizer(fert)

        res = await step(StepRequest(action=action_list))
        
        any_irr = any(a > 0 for a in action_list[:-1])
        if any_irr:
            reason = f"Irrigating because moisture ({avg_moist:.2f}) is low and heat ({temp:.1f}°C) is high."
        elif rain > 5.0:
            reason = f"Skipping irrigation to utilize {rain:.1f}mm forecasted rain."
        else:
            reason = "Moisture levels optimal. No action required."
            
        res.reasoning = reason_prefix + reason
        return res
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Auto-step failed: {str(e)}")


@app.get("/state", response_model=StateResponse)
async def get_state() -> StateResponse:
    if _env is None:
        raise HTTPException(status_code=400, detail="Call /reset first.")
    
    res = _state_to_serialisable(_env.state())
    
    water_saved = max(0, res["traditional_water_liters"] - res["water_used_liters"])
    raw_money_saved = water_saved * res["cost_per_liter"]
    land_ha = float(res.get("land_ha", 1.0) or 1.0)
    money_cap = 500.0 * math.sqrt(max(0.1, land_ha))
    money_saved = min(raw_money_saved, money_cap)
    health_score = 100 * (1.0 - np.mean(res["stress_index"]))
    water_eff = water_saved / max(res["traditional_water_liters"], 1)
    eco_score = (health_score * 0.6 + water_eff * 100 * 0.4)
    
    res["roi"] = {
        "water_saved": float(water_saved),
        "money_saved": float(money_saved),
        "eco_score": float(eco_score),
        "health_score": float(health_score)
    }
    
    projections = []
    curr_m = np.mean(res["soil_moisture"])
    et = _env._sim._calculate_effective_et() / (10.0 + 5.0)
    for i in range(1, 11):
        p = max(0.0, curr_m - (et * i))
        projections.append(float(p))
    res["projections"] = projections
    
    return StateResponse(**res)

def _obs_to_list(obs: np.ndarray) -> list[float]:
    return [float(x) for x in obs]


def _state_to_serialisable(state: dict[str, Any]) -> dict[str, Any]:
    return {
        k: (v.tolist() if isinstance(v, np.ndarray) else v)
        for k, v in state.items()
    }
