# Contributing

## Install dependencies

From the repository root, install the package in editable mode so imports resolve:

```bash
pip install -e .
```

You can also use `requirements.txt` if you prefer a flat dependency list.

## Run tests

```bash
pytest tests/
```

## Run the API server

```bash
uvicorn api.main:app --reload
```

## Run the baseline agent

The baseline policy uses full `state()` from the environment:

```bash
python -m agents.baseline_agent
```

For an observation-only rule policy (drop-in `run_episode` signature), use:

```bash
python -m agents.rl_agent
```
