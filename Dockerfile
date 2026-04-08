FROM python:3.11-slim

WORKDIR /app

# Copy requirements first for layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY irrigation_env/ ./irrigation_env/
COPY api/ ./api/
COPY agents/ ./agents/
<<<<<<< HEAD
COPY ui/ ./ui/

EXPOSE 7860

CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-7860}"]
=======

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
>>>>>>> ab1add790e655feb4ca89628ef19aac92674de9b
