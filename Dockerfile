FROM python:3.11-slim

WORKDIR /app

# Copy requirements first for layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY irrigation_env/ ./irrigation_env/
COPY api/ ./api/
COPY agents/ ./agents/
COPY ui/ ./ui/
COPY openenv.yaml ./openenv.yaml
COPY inference.py ./inference.py

EXPOSE 7860

CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-7860}"]
