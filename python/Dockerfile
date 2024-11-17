# Start from the base image
FROM giswqs/segment-geospatial:latest

# Set working directory
WORKDIR /app

# Switch to root user
USER root

# We don't need to COPY since we're using volumes
# Removing the COPY instruction since the files will be mounted

# Install dependencies and run the application
CMD /bin/bash -c "pip install -r requirements.txt && python app.py"

# Expose port 5000 (container port from the compose file)
EXPOSE 5000