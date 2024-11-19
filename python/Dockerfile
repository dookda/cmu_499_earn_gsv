FROM osgeo/gdal:ubuntu-small-3.6.3

# Install Python and pip
USER root
RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-venv \
    libegl1 libsm6 libxext6 libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0
# Set working directory
WORKDIR /app

# Create and activate a virtual environment
RUN python3 -m venv /venv
ENV PATH="/venv/bin:$PATH"

# Copy requirements.txt and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# No need to copy the rest of the code since it will be mounted
CMD ["python", "app2.py"]