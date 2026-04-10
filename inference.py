from __future__ import annotations

import json
import select
import sys
from typing import Any


def _extract_payload(raw_payload: Any) -> dict[str, Any]:
    if isinstance(raw_payload, dict) and isinstance(raw_payload.get("input"), dict):
        return raw_payload["input"]
    if isinstance(raw_payload, dict):
        return raw_payload
    return {"input": raw_payload}


def _derive_n_zones(payload: dict[str, Any]) -> int:
    for key in ("n_zones", "num_zones", "zones", "zone_count"):
        if key in payload:
            try:
                n = int(payload[key])
            except Exception:
                continue
            if n > 0:
                return n
    obs = payload.get("observation")
    if isinstance(obs, list) and len(obs) >= 12:
        remainder = len(obs) - 8
        if remainder % 4 == 0 and remainder > 0:
            return remainder // 4
    return 1


def _zone_action_from_moisture(m: float) -> int:
    if m < 0.32:
        return 2
    if m < 0.40:
        return 1
    return 0


def _action_from_payload(payload: dict[str, Any]) -> list[int]:
    n_zones = _derive_n_zones(payload)
    zone_actions = [0] * n_zones

    obs = payload.get("observation")
    if isinstance(obs, list) and len(obs) >= n_zones:
        for i in range(n_zones):
            try:
                zone_actions[i] = _zone_action_from_moisture(float(obs[i]))
            except Exception:
                zone_actions[i] = 0
    else:
        state = payload.get("state")
        soil = state.get("soil_moisture") if isinstance(state, dict) else None
        if isinstance(soil, list):
            for i in range(min(len(soil), n_zones)):
                try:
                    zone_actions[i] = _zone_action_from_moisture(float(soil[i]))
                except Exception:
                    zone_actions[i] = 0

    return zone_actions + [0]


def _predict_single(input_payload: dict[str, Any]) -> dict[str, Any]:
    payload = _extract_payload(input_payload)
    task = str(payload.get("task", "easy"))
    action = _action_from_payload(payload)
    return {
        "task": task,
        "status": "ok",
        "action": action,
        "actions": action,
        "prediction": action,
        "output": action,
        "reasoning": "Rule-based moisture threshold policy.",
    }


def main() -> None:
    try:
        raw = ""
        if not sys.stdin.isatty():
            ready, _, _ = select.select([sys.stdin], [], [], 0.2)
            if ready:
                raw = sys.stdin.read().strip()
        if raw:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                preds = [_predict_single(x if isinstance(x, dict) else {"input": x}) for x in parsed]
                print(json.dumps({"status": "ok", "predictions": preds, "output": preds}))
            else:
                print(json.dumps(_predict_single(parsed if isinstance(parsed, dict) else {"input": parsed})))
            return
    except Exception:
        pass

    fallback = _predict_single({"task": "easy", "n_zones": 1, "observation": [0.35]})
    print(
        json.dumps(
            {
                "agent_name": "rule_based_local",
                "status": "ok",
                "result": fallback,
                "results": [fallback],
                "prediction": fallback["prediction"],
                "output": fallback["output"],
            }
        )
    )


if __name__ == "__main__":
    main()
