# serving/Dockerfile  (Path B: HF Endpoint proxy, small image)
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# System deps (curl only for health/debug; keep image slim)
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./requirements.txt
RUN pip install -r requirements.txt

# Copy only the backend folder
COPY . .

# Default port for many PaaS providers (Render honors $PORT)
ENV PORT=8080
EXPOSE 8080

# Start the chat router (NOT the ONNX app)
CMD ["uvicorn", "app_chat:app", "--host", "0.0.0.0", "--port", "8080"]
