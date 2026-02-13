#!/bin/bash

# AWS EC2 Deployment Script
# Run this script on your EC2 instance

# Update system
sudo yum update -y

# Install Docker
sudo yum install -y docker
sudo service docker start
sudo usermod -a -G docker ec2-user

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Create app directory
mkdir -p /home/ec2-user/manufacturing-efficiency
cd /home/ec2-user/manufacturing-efficiency

# Clone your repository (replace with your repo URL)
git clone https://github.com/vinay6378/DataAnalysis6GNetworkProject.git .

# Make deploy script executable
chmod +x deploy.sh

# Run deployment
./deploy.sh

# Configure security group to allow port 8501
echo "⚠️  Remember to configure EC2 security group to allow inbound traffic on port 8501"

echo "🌐 Application will be available at: http://YOUR_EC2_PUBLIC_IP:8501"
