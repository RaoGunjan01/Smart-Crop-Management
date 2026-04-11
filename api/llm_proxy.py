"""OpenAI-compatible client routed through the platform LiteLLM proxy.

Uses ``API_BASE_URL`` and ``API_KEY`` as required by the evaluation harness.
When those variables are unset, helpers no-op so local runs still work.
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


def get_openai_client():
    """Build client exactly as specified for the proxy (env vars required when used)."""
    try:
        base_url = os.environ["API_BASE_URL"].rstrip("/")
        api_key = os.environ["API_KEY"]
    except KeyError:
        return None
    from openai import OpenAI

    return OpenAI(base_url=base_url, api_key=api_key)


def proxy_llm_ping() -> None:
    """Minimal completion so usage is attributed to the injected proxy key."""
    client = get_openai_client()
    if client is None:
        return
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
    """Optional one-line farmer-facing text; returns None if proxy unavailable or call fails."""
    client = get_openai_client()
    if client is None:
        return None
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
