FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY irrigation_env/ ./irrigation_env/
COPY api/ ./api/
COPY agents/ ./agents/
COPY ui/ ./ui/
COPY tasks/ ./tasks/
COPY graders.py ./graders.py
COPY tasks.py ./tasks.py
COPY inference.py ./inference.py
COPY openenv.yaml ./openenv.yaml

EXPOSE 8000

CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
