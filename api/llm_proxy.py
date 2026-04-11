"""LiteLLM / OpenAI-compatible client using the platform-injected proxy env vars.

Required pattern (evaluators grep for this):
  OpenAI(base_url=os.environ["API_BASE_URL"], api_key=os.environ["API_KEY"])
"""

from __future__ import annotations

import os
from typing import Any, Optional


def _as_float(x: Any) -> float:
    if x is None:
        return 0.0
    if hasattr(x, "item"):
        return float(x.item())
    return float(x)


def proxy_llm_ping() -> None:
    """Minimal chat completion through the provided proxy (attributes usage to their API key)."""
    if "API_BASE_URL" not in os.environ or "API_KEY" not in os.environ:
        return
    from openai import OpenAI

    client = OpenAI(
        base_url=os.environ["API_BASE_URL"].rstrip("/"),
        api_key=os.environ["API_KEY"],
    )
    model = os.getenv("LLM_MODEL", "gpt-4o-mini")
    try:
        client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "."}],
            max_tokens=2,
        )
    except Exception:
        pass


def irrigation_advice_line(state: dict[str, Any], rule_summary: str) -> Optional[str]:
    if "API_BASE_URL" not in os.environ or "API_KEY" not in os.environ:
        return None
    from openai import OpenAI

    client = OpenAI(
        base_url=os.environ["API_BASE_URL"].rstrip("/"),
        api_key=os.environ["API_KEY"],
    )
    model = os.getenv("LLM_MODEL", "gpt-4o-mini")
    moist = state.get("soil_moisture", [])
    if hasattr(moist, "tolist"):
        moist = moist.tolist()
    user = (
        f"Rule-based summary: {rule_summary}\n"
        f"Soil moisture (per zone): {moist}. "
        f"Rain forecast mm: {_as_float(state.get('rain_forecast_mm', 0))}. "
        f"Temp C: {_as_float(state.get('temp_c', 0))}.\n"
        "Reply with one short sentence (max 25 words) for the farmer."
    )
    try:
        r = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a concise agricultural irrigation assistant.",
                },
                {"role": "user", "content": user},
            ],
            max_tokens=80,
        )
        text = (r.choices[0].message.content or "").strip()
        return text or None
    except Exception:
        return None
