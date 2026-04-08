"""Pytest unit tests for the Irrigation RL environment."""

from __future__ import annotations

import numpy as np
import pytest

from irrigation_env.env import IrrigationEnv
from irrigation_env.grader import grade_episode
from irrigation_env.reward import compute_reward
from irrigation_env.tasks import EASY, MEDIUM, HARD, TaskConfig


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def easy_env() -> IrrigationEnv:
    return IrrigationEnv(task="easy")


@pytest.fixture
def medium_env() -> IrrigationEnv:
    return IrrigationEnv(task="medium")


@pytest.fixture
def hard_env() -> IrrigationEnv:
    return IrrigationEnv(task="hard")


# ---------------------------------------------------------------------------
# 1. reset() returns valid observation shape for all 3 tasks
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("task,n_zones", [("easy", 1), ("medium", 4), ("hard", 8)])
def test_reset_observation_shape(task: str, n_zones: int) -> None:
    """reset() must return a flat obs whose length matches env obs space."""
    env = IrrigationEnv(task=task)
    obs, info = env.reset(seed=0)
    expected_dim = n_zones * 4 + 8
    assert obs.shape == (expected_dim,), (
        f"Expected obs shape ({expected_dim},) got {obs.shape}"
    )
    assert obs.shape == env.observation_space.shape
    assert obs.dtype == np.float32


# ---------------------------------------------------------------------------
# 2. step() returns correct tuple structure
# ---------------------------------------------------------------------------

def test_step_tuple_structure(easy_env: IrrigationEnv) -> None:
    """step() must return (obs, reward, terminated, truncated, info)."""
    easy_env.reset(seed=1)
    action = easy_env.action_space.sample()
    result = easy_env.step(action)
    assert len(result) == 5, "step() should return 5 elements"
    obs, reward, terminated, truncated, info = result
    assert isinstance(obs, np.ndarray)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)


# ---------------------------------------------------------------------------
# 3. reward is always a float
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("task", ["easy", "medium", "hard"])
def test_reward_is_float(task: str) -> None:
    """Reward must always be a Python float."""
    env = IrrigationEnv(task=task)
    env.reset(seed=2)
    for _ in range(5):
        action = env.action_space.sample()
        _, reward, terminated, truncated, _ = env.step(action)
        assert isinstance(reward, float), f"reward type {type(reward)}"
        if terminated or truncated:
            break


# ---------------------------------------------------------------------------
# 4. terminated=True at season end
# ---------------------------------------------------------------------------

def test_terminated_at_season_end(easy_env: IrrigationEnv) -> None:
    """Episode must terminate after n_days * 4 steps."""
    easy_env.reset(seed=3)
    total_budget_steps = EASY.n_days * 4
    terminated = False
    for _ in range(total_budget_steps + 10):
        action = easy_env.action_space.sample()
        _, _, terminated, truncated, _ = easy_env.step(action)
        if terminated or truncated:
            break
    assert terminated or truncated, "Episode should have ended"


# ---------------------------------------------------------------------------
# 5. state() dict keys match schema
# ---------------------------------------------------------------------------

REQUIRED_STATE_KEYS = {
    "soil_moisture",
    "crop_growth_stage",
    "stress_index",
    "days_since_irrigation",
    "temp_c",
    "humidity",
    "rain_forecast_mm",
    "evapotranspiration",
    "water_budget_remaining",
    "cost_per_liter",
    "day_of_season",
    "time_of_day",
    "n_zones",
}


@pytest.mark.parametrize("task", ["easy", "medium", "hard"])
def test_state_dict_keys(task: str) -> None:
    """state() must contain all required schema keys."""
    env = IrrigationEnv(task=task)
    env.reset(seed=4)
    s = env.state()
    missing = REQUIRED_STATE_KEYS - set(s.keys())
    assert not missing, f"Missing state keys: {missing}"


# ---------------------------------------------------------------------------
# 6. grader returns float in [0.0, 1.0]
# ---------------------------------------------------------------------------

def test_grader_returns_valid_float(easy_env: IrrigationEnv) -> None:
    """grade_episode() must return a float in [0.0, 1.0]."""
    easy_env.reset(seed=5)
    for _ in range(20):
        action = easy_env.action_space.sample()
        _, _, terminated, truncated, info = easy_env.step(action)
        if terminated or truncated:
            break

    # Build a minimal episode log compatible with grader
    log = easy_env.episode_log
    assert len(log) > 0, "Episode log should not be empty"
    grade = grade_episode(log)
    assert isinstance(grade, float), f"grade type {type(grade)}"
    assert 0.0 <= grade <= 1.0, f"grade {grade} out of [0, 1]"


# ---------------------------------------------------------------------------
# 7. baseline agent episode log has correct length
# ---------------------------------------------------------------------------

def test_baseline_agent_episode_log_length(easy_env: IrrigationEnv) -> None:
    """Baseline agent must produce an episode log of the expected length."""
    from agents.baseline_agent import run_episode

    log = run_episode(easy_env, seed=6, verbose=False)
    expected_max = EASY.n_days * 4
    assert len(log) > 0, "Episode log is empty"
    assert len(log) <= expected_max, (
        f"Log length {len(log)} exceeds max {expected_max}"
    )


# ---------------------------------------------------------------------------
# 8. compute_reward returns a float
# ---------------------------------------------------------------------------

def test_compute_reward_isolated(easy_env: IrrigationEnv) -> None:
    """compute_reward() must be callable standalone and return a float."""
    easy_env.reset(seed=7)
    state = easy_env.state()
    action = {"zone_actions": [0], "global_action": 0}
    easy_env.step(np.array([0, 0], dtype=np.int64))
    next_state = easy_env.state()
    next_state["water_used_step"] = 0.0
    next_state["water_used_liters"] = 0.0

    reward, info = compute_reward(state, action, next_state, EASY)
    assert isinstance(reward, float)
    assert "reward_components" in info


# ---------------------------------------------------------------------------
# 9. action space sample is valid
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("task", ["easy", "medium", "hard"])
def test_action_space_sample_valid(task: str) -> None:
    """Sampled actions must lie within the action space."""
    env = IrrigationEnv(task=task)
    env.reset(seed=8)
    for _ in range(10):
        action = env.action_space.sample()
        assert env.action_space.contains(action), f"Invalid sampled action: {action}"
