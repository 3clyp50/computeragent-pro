#!/bin/bash

# Stop existing containers
docker-compose down

# Build and start the FastAPI service
docker-compose up -d --build

# Check logs
docker-compose logs -f
