FROM giswqs/segment-geospatial:latest

# Install necessary system libraries
USER root
RUN apt-get update && apt-get install -y \
    libegl1 libsm6 libxext6 libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Create and activate a virtual environment
WORKDIR /app
RUN python3 -m venv /venv

# Set environment variables for the virtual environment
ENV PATH="/venv/bin:$PATH"

# Copy application code
COPY . .

# Install Python dependencies in the virtual environment
RUN pip install --no-cache-dir -r requirements.txt

# Set the default command to run the application
CMD ["python", "app.py"]
