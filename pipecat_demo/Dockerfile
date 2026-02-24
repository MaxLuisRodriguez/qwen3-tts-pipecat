FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy demo code
COPY . .

# Set environment variables (can be overridden)
ENV LLM_SERVICE_URL=http://llm-service:8000
ENV TTS_SERVICE_URL=http://tts-service:8001

# Run demo
CMD ["python", "app.py"]
