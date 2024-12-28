#!/bin/bash

# Connect to the existing Caddy network
docker network create caddy_network || true

# Stop existing containers
docker-compose down

# Build and start the FastAPI service
docker-compose up -d --build fastapi

# Check logs
docker-compose logs -f fastapi