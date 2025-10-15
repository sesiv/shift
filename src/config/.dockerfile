FROM python:3.12-slim

WORKDIR /app

# Install system dependencies including build tools
# llama-cpp need cmake, compilers, and git
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    cmake \
    make \
    git \
    curl \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .

RUN --mount=type=cache,target=/root/.cache/pip \
       pip install -r requirements.txt

# Copy application files
COPY . .

# Expose all required ports
EXPOSE 8501 8000 5002 5003 5004 5005
