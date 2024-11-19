FROM osgeo/gdal:latest

# Install Python and necessary system packages
USER root
RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-venv \
    libegl1 libsm6 libxext6 libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Create and activate a virtual environment
RUN python3 -m venv /venv
ENV PATH="/venv/bin:$PATH"

# Copy requirements.txt first to cache dependencies
COPY requirements.txt .

# Install Python dependencies in the virtual environment
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Set default command to run the application
CMD ["python", "app.py"]
