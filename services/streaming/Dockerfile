FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create output directory
RUN mkdir -p /var/www/streams

# Expose port
EXPOSE 8083

# Run the application
CMD ["python", "-m", "uvicorn", "streaming_server:app", "--host", "0.0.0.0", "--port", "8083"]