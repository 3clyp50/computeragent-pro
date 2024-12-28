#!/bin/bash

# Create the network if it doesn't exist
docker network create caddy_network || true

# Stop and remove existing containers
docker-compose down

# Pull latest images
docker-compose pull

# Build and start services
docker-compose up -d

# Check logs
docker-compose logs -f