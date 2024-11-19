FROM giswqs/segment-geospatial:latest

# Install necessary system libraries
USER root
RUN apt-get update && apt-get install -y \
    libegl1 libsm6 libxext6 libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Create and set up the virtual environment
WORKDIR /app
RUN python3 -m venv /venv

# Activate the virtual environment by setting the PATH
ENV PATH="/venv/bin:$PATH"

# Copy application code
COPY . .

# Install Python dependencies inside the virtual environment
RUN pip install --no-cache-dir -r requirements.txt

# Set the default command
CMD ["python", "app.py"]
