FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY services/tts_qwen3/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Preserve repo layout so relative paths in the service continue to resolve.
COPY services/tts_qwen3 /app/services/tts_qwen3
COPY kernel /app/kernel

WORKDIR /app/services/tts_qwen3

# Expose port
EXPOSE 8001

# Health check
HEALTHCHECK --interval=10s --timeout=5s --retries=3 \
    CMD curl -f http://localhost:8001/health || exit 1

# Run server
CMD ["python", "server.py"]
