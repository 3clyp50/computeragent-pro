#!/bin/bash

# Stop and remove existing container
docker stop computeragent-pro || true
docker rm computeragent-pro || true

# Build new image
docker build -t computeragent-pro:latest .

# Run container
docker run -d \
    --name computeragent-pro \
    --gpus all \
    -p 8000:8000 \
    --env-file .env \
    computeragent-pro:latest

# Check container status
docker ps | grep computeragent-pro