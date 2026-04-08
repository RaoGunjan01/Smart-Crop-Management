"""Task configuration dataclasses for the Irrigation RL environment."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TaskConfig:
    """Configuration for an irrigation scheduling task."""

    name: str
    n_zones: int
    n_days: int
    sensor_noise_std: float
    water_budget_liters: float
    rain_probability: float
    crop_type: str
    sensor_failure_probability: float = 0.0
    max_steps_per_day: int = 4  # time slots: 0=morning,1=midday,2=afternoon,3=night
    kc_base: float = 1.0        # crop coefficient base value


# ---------------------------------------------------------------------------
# Predefined difficulty presets
# ---------------------------------------------------------------------------

EASY = TaskConfig(
    name="easy",
    n_zones=1,
    n_days=30,
    sensor_noise_std=0.0,
    water_budget_liters=10_000,
    rain_probability=0.3,
    crop_type="wheat",
)

MEDIUM = TaskConfig(
    name="medium",
    n_zones=4,
    n_days=60,
    sensor_noise_std=0.05,
    water_budget_liters=8_000,
    rain_probability=0.2,
    crop_type="mixed",
)

HARD = TaskConfig(
    name="hard",
    n_zones=8,
    n_days=90,
    sensor_noise_std=0.1,
    water_budget_liters=4_000,
    rain_probability=0.1,
    sensor_failure_probability=0.05,
    crop_type="mixed",
)

TASK_REGISTRY: dict[str, TaskConfig] = {
    "easy": EASY,
    "medium": MEDIUM,
    "hard": HARD,
}


def get_task(name: str) -> TaskConfig:
    """Return a task config by name (case-insensitive).

    Args:
        name: Task name — 'easy', 'medium', or 'hard'.

    Returns:
        Corresponding :class:`TaskConfig` instance.

    Raises:
        ValueError: If the name is not recognised.
    """
    key = name.lower()
    if key not in TASK_REGISTRY:
        raise ValueError(f"Unknown task '{name}'. Choose from: {list(TASK_REGISTRY)}")
    return TASK_REGISTRY[key]
