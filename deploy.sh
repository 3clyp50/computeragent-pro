#!/bin/bash
# deploy.sh

# Pull latest changes
git pull

# Build docker image
docker build -t computeragent-pro .

# Stop existing container
docker stop computeragent-pro || true
docker rm computeragent-pro || true

# Run new container
docker run -d \
    --name computeragent-pro \
    --gpus all \
    -p 8000:8000 \
    --restart unless-stopped \
    computeragent-pro