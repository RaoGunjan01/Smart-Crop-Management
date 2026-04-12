from __future__ import annotations

from typing import Any

from irrigation_env.grader import grade_easy, grade_hard, grade_medium


def grade_easy_task(episode_log: list[dict[str, Any]]) -> float:
    return float(grade_easy(episode_log))


def grade_medium_task(episode_log: list[dict[str, Any]]) -> float:
    return float(grade_medium(episode_log))


def grade_hard_task(episode_log: list[dict[str, Any]]) -> float:
    return float(grade_hard(episode_log))


def grade_task(task: str, episode_log: list[dict[str, Any]]) -> float:
    task_key = str(task).lower()
    if task_key == "easy":
        return grade_easy_task(episode_log)
    if task_key == "medium":
        return grade_medium_task(episode_log)
    if task_key == "hard":
        return grade_hard_task(episode_log)
    raise ValueError(f"Unknown task '{task}'. Expected easy, medium, or hard.")
