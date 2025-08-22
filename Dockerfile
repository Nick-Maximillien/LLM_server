# Base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies for Hugging Face + GPTQ
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && pip install --no-cache-dir "numpy<2"

# Copy source code
COPY . .

# Environment vars for Hugging Face + logging
ENV TRANSFORMERS_CACHE=/tmp/hf_cache
ENV HF_HOME=/tmp/hf_cache
ENV PYTHONUNBUFFERED=1

# Expose port 10000
EXPOSE 10000

# Run FastAPI with uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "10000"]
