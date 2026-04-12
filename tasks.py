from __future__ import annotations

TASKS = [
    {
        "name": "easy",
        "title": "Don't Kill the Crop",
        "description": "Keep crop stress under control from start to finish.",
        "max_steps": 120,
        "grader": {"module": "graders", "function": "grade_easy"},
    },
    {
        "name": "medium",
        "title": "Respect the Rain",
        "description": "Read the rain forecast and hold irrigation when rain is coming.",
        "max_steps": 240,
        "grader": {"module": "graders", "function": "grade_medium"},
    },
    {
        "name": "hard",
        "title": "Do More With Less",
        "description": "Maximize yield while minimizing water across the episode.",
        "max_steps": 360,
        "grader": {"module": "graders", "function": "grade_hard"},
    },
]


def list_tasks() -> list[dict[str, object]]:
    return TASKS
