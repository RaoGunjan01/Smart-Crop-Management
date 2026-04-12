from __future__ import annotations

import asyncio
import contextlib
import io
import os
import threading
import webbrowser
from contextlib import asynccontextmanager
from typing import Any, Optional

import numpy as np
from fastapi import Body, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from irrigation_env.env import IrrigationEnv
from irrigation_env.grader import grade_easy, grade_hard, grade_medium

from .llm_proxy import ensure_llm_proxy_traffic, irrigation_advice_line

# Universal behaviour tasks (labels for UI / openenv; simulator geometry from TaskConfig).
IRRIGATION_TASK_META: dict[str, dict[str, str]] = {
    "easy": {"title": "Don't Kill the Crop", "focus": "Keep mean stress under control."},
    "medium": {"title": "Respect the Rain", "focus": "Restrain irrigation when rain is forecast."},
    "hard": {"title": "Do More With Less", "focus": "Balance stress vs water use."},
}


@asynccontextmanager
async def _lifespan(app: FastAPI):
    # LiteLLM proxy ping when env vars are available (may retry from /health).
    await asyncio.to_thread(ensure_llm_proxy_traffic)
    if os.getenv("ENABLE_LOCAL_BROWSER") and not (
        os.getenv("HF_SPACE_ID") or os.getenv("SPACE_ID")
    ):

        def _open() -> None:
            try:
                webbrowser.open("http://localhost:8000/ui/dashboard.html", new=2)
            except Exception:
                pass

        threading.Timer(0.5, _open).start()
    yield


app = FastAPI(
    title="Smart Crop Management Sytem",
    version="1.0.0",
    lifespan=_lifespan,
)

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
    status: str = "healthy"
    rl_active: bool = False
    llm_proxy_configured: bool = False


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    proxy_ok = "API_BASE_URL" in os.environ and "API_KEY" in os.environ
    await asyncio.to_thread(ensure_llm_proxy_traffic)
    return HealthResponse(
        status="healthy", rl_active=False, llm_proxy_configured=proxy_ok
    )


class EpisodeGradeBody(BaseModel):
    episode_log: list[dict[str, Any]]


class GenericGradeBody(BaseModel):
    task: str = "easy"
    episode_log: list[dict[str, Any]]


def _grader_import_path(task: str) -> str:
    return f"irrigation_env.grader:grade_{task}"


def _task_descriptor(task_id: str) -> dict[str, Any]:
    configs = {
        "easy": {"difficulty": "easy", "max_steps": 120},
        "medium": {"difficulty": "medium", "max_steps": 240},
        "hard": {"difficulty": "hard", "max_steps": 360},
    }
    meta = IRRIGATION_TASK_META[task_id]
    extra = configs[task_id]
    return {
        "id": task_id,
        "name": meta["title"],
        "description": meta["focus"],
        "difficulty": extra["difficulty"],
        "max_steps": extra["max_steps"],
        "grader": _grader_import_path(task_id),
        "grader_endpoint": f"/grade/{task_id}",
        "score_range": [0.0, 1.0],
    }


def _run_task_grader(task: str, episode_log: list[dict[str, Any]]) -> float:
    graders: dict[str, Any] = {
        "easy": grade_easy,
        "medium": grade_medium,
        "hard": grade_hard,
    }
    task_key = str(task).lower()
    if task_key not in graders:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown task '{task}'. Choose from: {list(graders)}",
        )
    return float(graders[task_key](episode_log))


@app.get("/tasks")
async def list_tasks() -> dict[str, Any]:
    tasks = [_task_descriptor(task_id) for task_id in ("easy", "medium", "hard")]
    return {"tasks": tasks, "count": len(tasks)}


@app.post("/grader")
async def grade_task_http(body: GenericGradeBody) -> dict[str, Any]:
    score = _run_task_grader(body.task, body.episode_log)
    return {
        "task": body.task,
        "score": score,
        "score_range": [0.0, 1.0],
    }


@app.get("/grade/easy")
async def grade_easy_metadata() -> dict[str, Any]:
    return {
        "task": "easy",
        "import_path": "irrigation_env.grader:grade_easy",
        "score_range": [0.0, 1.0],
        "invoke": "POST /grade/easy with JSON body {\"episode_log\": [...]}",
    }


@app.post("/grade/easy")
async def grade_easy_http(body: EpisodeGradeBody) -> dict[str, Any]:
    from irrigation_env.grader import grade_easy

    return {"score": float(grade_easy(body.episode_log)), "task": "easy"}


@app.get("/grade/medium")
async def grade_medium_metadata() -> dict[str, Any]:
    return {
        "task": "medium",
        "import_path": "irrigation_env.grader:grade_medium",
        "score_range": [0.0, 1.0],
        "invoke": "POST /grade/medium with JSON body {\"episode_log\": [...]}",
    }


@app.post("/grade/medium")
async def grade_medium_http(body: EpisodeGradeBody) -> dict[str, Any]:
    from irrigation_env.grader import grade_medium

    return {"score": float(grade_medium(body.episode_log)), "task": "medium"}


@app.get("/grade/hard")
async def grade_hard_metadata() -> dict[str, Any]:
    return {
        "task": "hard",
        "import_path": "irrigation_env.grader:grade_hard",
        "score_range": [0.0, 1.0],
        "invoke": "POST /grade/hard with JSON body {\"episode_log\": [...]}",
    }


@app.post("/grade/hard")
async def grade_hard_http(body: EpisodeGradeBody) -> dict[str, Any]:
    from irrigation_env.grader import grade_hard

    return {"score": float(grade_hard(body.episode_log)), "task": "hard"}


@app.post("/reset")
async def reset(body: dict[str, Any] | None = Body(default=None)) -> dict[str, Any]:
    global _env
    try:
        payload = body or {}
        if isinstance(payload.get("input"), dict):
            payload = payload["input"]

        req = ResetRequest(
            task=payload.get("task", payload.get("msg", "easy")),
            seed=payload.get("seed", 42),
            crop=payload.get("crop"),
            season=payload.get("season"),
            land_ha=payload.get("land_ha"),
            nutrients=payload.get("nutrients") or {},
        )

        _env = IrrigationEnv(task=req.task)
        obs, info = _env.reset(
            seed=req.seed,
            options={
                "crop": req.crop,
                "season": req.season,
                "land_ha": req.land_ha,
                "nutrients": req.nutrients or {},
            },
        )

        return {
            "observation": _obs_to_list(obs),
            "info": info,
            "task": req.task,
            "n_zones": _env.task_config.n_zones,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reset failed: {str(e)}")


@app.post("/step")
async def step(request: StepRequest) -> dict[str, Any]:
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

        return {
            "observation": _obs_to_list(obs),
            "reward": float(reward),
            "terminated": terminated,
            "truncated": truncated,
            "info": info_serialisable,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Step failed: {str(e)}")


@app.post("/auto-step")
async def auto_step() -> dict[str, Any]:
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

        base_reason = reason_prefix + reason
        llm_note = await asyncio.to_thread(irrigation_advice_line, st, base_reason)
        res["reasoning"] = f"{base_reason} | {llm_note}" if llm_note else base_reason
        return res

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Auto-step failed: {str(e)}")


@app.get("/state")
async def get_state() -> dict[str, Any]:
    if _env is None:
        raise HTTPException(status_code=400, detail="Call /reset first.")

    res = _state_to_serialisable(_env.state())

    water_saved = max(0, res["traditional_water_liters"] - res["water_used_liters"])
    money_saved = water_saved * res["cost_per_liter"]
    health_score = 100 * (1.0 - np.mean(res["stress_index"]))
    water_eff = water_saved / max(res["traditional_water_liters"], 1)
    eco_score = health_score * 0.6 + water_eff * 100 * 0.4

    res["roi"] = {
        "water_saved": float(water_saved),
        "money_saved": float(money_saved),
        "eco_score": float(eco_score),
        "health_score": float(health_score),
    }

    projections = []
    curr_m = np.mean(res["soil_moisture"])
    et = _env._sim._calculate_effective_et() / (10.0 + 5.0)
    for i in range(1, 11):
        p = max(0.0, curr_m - (et * i))
        projections.append(float(p))
    res["projections"] = projections

    tc = _env.task_config
    tid = str(tc.name)
    meta = IRRIGATION_TASK_META.get(tid, {"title": tid, "focus": ""})
    res["irrigation_task_id"] = tid
    res["irrigation_task_title"] = meta["title"]
    res["irrigation_task_focus"] = meta["focus"]
    res["water_budget_liters"] = float(tc.water_budget_liters)
    res["episode_step_count"] = len(_env.episode_log)

    log = _env.episode_log
    if log:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            res["universal_grades"] = {
                "dont_kill_crop": float(grade_easy(log)),
                "respect_the_rain": float(grade_medium(log)),
                "do_more_with_less": float(grade_hard(log)),
            }
    else:
        res["universal_grades"] = None

    return res


def _obs_to_list(obs: np.ndarray) -> list[float]:
    return [float(x) for x in obs]


def _state_to_serialisable(state: dict[str, Any]) -> dict[str, Any]:
    return {
        k: (v.tolist() if isinstance(v, np.ndarray) else v)
        for k, v in state.items()
    }
