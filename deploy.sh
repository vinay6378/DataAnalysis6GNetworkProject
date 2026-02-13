#!/bin/bash

# Manufacturing Efficiency Deployment Script
# This script deploys the application to production

set -e

echo "🚀 Starting Manufacturing Efficiency Dashboard Deployment..."

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

# Create necessary directories
mkdir -p data logs

# Copy environment file if it doesn't exist
if [ ! -f .env ]; then
    cp .env.example .env
    echo "⚠️  Please edit .env file with your production settings"
fi

# Build and start the application
echo "📦 Building Docker image..."
docker-compose build

echo "🔄 Starting application..."
docker-compose up -d

# Wait for the application to be healthy
echo "⏳ Waiting for application to be healthy..."
sleep 30

# Check if the application is running
if curl -f http://localhost:8501/_stcore/health > /dev/null 2>&1; then
    echo "✅ Application deployed successfully!"
    echo "🌐 Dashboard is available at: http://localhost:8501"
else
    echo "❌ Application failed to start. Check logs with: docker-compose logs"
    exit 1
fi

echo "📊 Deployment completed successfully!"
echo "📋 Next steps:"
echo "   1. Access the dashboard at http://localhost:8501"
echo "   2. Configure your domain and SSL for production"
echo "   3. Set up monitoring and backups"
