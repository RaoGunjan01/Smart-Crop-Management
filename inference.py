import json
import os
import sys
from typing import Any, Optional

import numpy as np
from openai import OpenAI

from agents.baseline_agent import RuleBasedAgent
from irrigation_env.env import IrrigationEnv
from irrigation_env.grader import grade_episode


API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME") or os.getenv("IMAGE_NAME")

TASK_NAME = os.getenv("TASK_NAME", os.getenv("IRRIGATION_TASK", "easy"))
BENCHMARK = os.getenv("BENCHMARK", os.getenv("IRRIGATION_BENCHMARK", "precision-irrigation-scheduling"))
MAX_STEPS = int(os.getenv("MAX_STEPS", "2000"))
SUCCESS_SCORE_THRESHOLD = float(os.getenv("SUCCESS_SCORE_THRESHOLD", "0.1"))


def _require_api_key() -> str:
    if not API_KEY:
        raise RuntimeError("Missing required env var: HF_TOKEN")
    return API_KEY


def log_start(task: str, env: str, model: str) -> None:
    sys.stdout.write(f"[START] task={task} env={env} model={model}\n")
    sys.stdout.flush()


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    err = error if error else "null"
    done_val = str(bool(done)).lower()
    sys.stdout.write(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={err}\n"
    )
    sys.stdout.flush()


def log_end(success: bool, steps: int, score: float, rewards: list[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    success_val = str(bool(success)).lower()
    sys.stdout.write(f"[END] success={success_val} steps={steps} score={score:.2f} rewards={rewards_str}\n")
    sys.stdout.flush()


def _llm_action(
    client: OpenAI,
    model: str,
    state: dict[str, Any],
) -> list[int] | None:
    n = int(state.get("n_zones", 1))
    soil = [float(x) for x in np.asarray(state.get("soil_moisture", [0.45] * n)).tolist()]
    stress = [float(x) for x in np.asarray(state.get("stress_index", [0.0] * n)).tolist()]
    rain = float(state.get("rain_forecast_mm", 0.0))
    nutrients = state.get("nutrients", {})

    prompt = {
        "task": "Choose irrigation action per zone (0-4) plus global slot (0).",
        "n_zones": n,
        "soil_moisture": soil,
        "stress_index": stress,
        "rain_forecast_mm": rain,
        "nutrients": nutrients,
        "output_schema": {"action": [0] * (n + 1)},
        "rules": [
            "Return JSON only.",
            "action must be length n_zones+1.",
            "Each zone action in [0,4].",
            "Last slot is global action, always 0.",
        ],
    }

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are an irrigation control policy. Output JSON only."},
                {"role": "user", "content": json.dumps(prompt)},
            ],
            temperature=0.2,
        )
        text = resp.choices[0].message.content or ""
        data = json.loads(text)
        action = data.get("action")
        if not isinstance(action, list) or len(action) != n + 1:
            return None
        out: list[int] = []
        for i, a in enumerate(action):
            if not isinstance(a, int):
                return None
            if i < n and not (0 <= a <= 4):
                return None
            if i == n:
                out.append(0)
            else:
                out.append(a)
        return out
    except Exception:
        return None


def _action_to_str(action: list[int]) -> str:
    return "irrigate(" + ",".join(str(int(x)) for x in action) + ")"


def run_task(client: OpenAI, model: str, task: str, seed: int) -> tuple[float, int, list[float]]:
    env = IrrigationEnv(task=task)
    obs, info = env.reset(seed=seed)
    agent = RuleBasedAgent()

    done = False
    steps = 0
    total_reward = 0.0
    used_llm_once = False
    rewards: list[float] = []

    while not done:
        st = env.state()
        if not used_llm_once:
            llm_act = _llm_action(client, model, st)
            used_llm_once = True
        else:
            llm_act = None

        act_arr = agent.act(obs, st).tolist() if llm_act is None else llm_act

        env.set_pending_fertilizer({"a": 0.0, "b": 0.0, "c": 0.0})
        obs, reward, terminated, truncated, step_info = env.step(np.array(act_arr, dtype=np.int64))
        done = bool(terminated or truncated)
        steps += 1
        total_reward += float(reward)
        rewards.append(float(reward))

        log_step(
            step=steps,
            action=_action_to_str(act_arr),
            reward=float(reward),
            done=done,
            error=None,
        )

        if steps >= min(MAX_STEPS, env.task_config.n_days * 4):
            break

    score = float(grade_episode(env.episode_log, verbose=False))
    try:
        env.close()
    except Exception:
        pass
    return score, steps, rewards


def main() -> None:
    _ = LOCAL_IMAGE_NAME
    client = OpenAI(base_url=API_BASE_URL, api_key=_require_api_key())

    success = False
    steps_taken = 0
    score = 0.0
    rewards: list[float] = []

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    try:
        score, steps_taken, rewards = run_task(client, MODEL_NAME, TASK_NAME, seed=42)
        score = float(np.clip(score, 0.0, 1.0))
        success = score >= SUCCESS_SCORE_THRESHOLD
    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


if __name__ == "__main__":
    main()

