FROM python:3.11-slim

WORKDIR /app

# system deps (onnxruntime may need these)
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# copy app and models (you can mount models via volume in prod)
COPY serving /app/serving
COPY models /app/models

ENV ONNX_DIR=/app/models/onnx
EXPOSE 8000
CMD ["uvicorn", "serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
