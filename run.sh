#!/bin/bash

echo "🚧 Building Docker image..."
docker build -t collateral-agent .

echo "🚀 Running container on port 8000..."
docker run -p 8000:8000 -v "$(pwd)/datasets:/app/datasets" collateral-agent