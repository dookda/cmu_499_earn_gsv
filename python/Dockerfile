FROM giswqs/segment-geospatial:latest

# Elevate permissions for apt-get operations
USER root

# Install necessary libraries
RUN apt-get update && apt-get install -y \
    libegl1 libsm6 libxext6 libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Switch back to the default user (if needed)
USER 1000

# Set working directory
WORKDIR /app

# Copy application code
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt
