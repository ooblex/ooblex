FROM node:18-slim

# Install dependencies for wrtc
RUN apt-get update && apt-get install -y \
    python3 \
    make \
    g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy package files
COPY package*.json ./

# Install dependencies
RUN npm install

# Copy application code
COPY . .

# Expose port (if needed)
EXPOSE 3000

# Start service
CMD ["node", "server.js"]
