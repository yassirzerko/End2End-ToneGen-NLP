#!/bin/bash

# Check if Docker is installed
if ! command -v docker &> /dev/null
then
    echo "Error: Docker is not installed. Please install Docker to continue."
    exit 1
fi

if [ $# -ne 1 ]; then
    echo "Missing argument(s): <client_port>"
    exit 1
fi

CLIENT_PORT=$1

if [ -z "$(find trained_models -name '*.pkl' -print -quit)" ]; then
    echo "Starting the training of the models"
    docker build -t train-model -f /src/Dockerfile
    docker run -v "$(pwd)/trained_models:/app/trained_models" train-model
else
    echo "Skipping training as .pkl files already exist in trained_models directory or its subdirectories."
fi

docker-compose up -e CLIENT_PORT="$CLIENT_PORT"

# Run Docker containers based on the images
docker run -d -p 5000:5000 --name flask-container my-flask-app
docker run -d -p 3000:3000 --name node-container my-node-app
