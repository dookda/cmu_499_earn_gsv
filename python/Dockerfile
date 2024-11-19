FROM osgeo/gdal:ubuntu-small-3.6.3

# Install Python and pip
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
COPY requirements2.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

WORKDIR /app
COPY ./app /app

CMD ["python3", "app.py"]