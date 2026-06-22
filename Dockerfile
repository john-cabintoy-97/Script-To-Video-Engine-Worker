FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    WORK_DIR=/app/workspace

WORKDIR /app

# System dependencies for FFmpeg compilation and native wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY main.py .

RUN mkdir -p /app/workspace

EXPOSE 7860

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
