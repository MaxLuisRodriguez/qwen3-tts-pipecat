FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY pipecat_demo/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Preserve repo layout so the app's repo-root env lookup remains valid.
COPY pipecat_demo /app/pipecat_demo

WORKDIR /app/pipecat_demo

# Set environment variables (can be overridden)
ENV LLM_SERVICE_URL=http://llm-service:8000
ENV TTS_SERVICE_URL=http://tts-service:8001

# Run demo
CMD ["python", "app.py"]
