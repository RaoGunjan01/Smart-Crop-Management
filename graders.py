from __future__ import annotations

from typing import Any

from irrigation_env.grader import grade_easy as _grade_easy
from irrigation_env.grader import grade_hard as _grade_hard
from irrigation_env.grader import grade_medium as _grade_medium


def grade_easy(episode_log: list[dict[str, Any]]) -> float:
    return float(_grade_easy(episode_log))


def grade_medium(episode_log: list[dict[str, Any]]) -> float:
    return float(_grade_medium(episode_log))


def grade_hard(episode_log: list[dict[str, Any]]) -> float:
    return float(_grade_hard(episode_log))


def grade_task(task: str, episode_log: list[dict[str, Any]]) -> float:
    task_key = str(task).lower()
    if task_key == "easy":
        return grade_easy(episode_log)
    if task_key == "medium":
        return grade_medium(episode_log)
    if task_key == "hard":
        return grade_hard(episode_log)
    raise ValueError(f"Unknown task '{task}'. Expected easy, medium, or hard.")
