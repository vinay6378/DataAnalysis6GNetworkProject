# Google Cloud Platform Deployment Guide

## Prerequisites
- Google Cloud SDK installed
- GCP account and project

## Deployment Steps

### 1. Set up GCP Project
```bash
gcloud config set project YOUR_PROJECT_ID
```

### 2. Enable Cloud Run API
```bash
gcloud services enable run.googleapis.com
```

### 3. Build and Deploy to Cloud Run
```bash
# Build the image
gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/manufacturing-efficiency

# Deploy to Cloud Run
gcloud run deploy manufacturing-efficiency \
  --image gcr.io/YOUR_PROJECT_ID/manufacturing-efficiency \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --port 8501
```

### 4. Get the URL
```bash
gcloud run services describe manufacturing-efficiency \
  --platform managed \
  --region us-central1 \
  --format "value(status.url)"
```

## Alternative: Cloud Run with Dockerfile
```bash
# Deploy using existing Dockerfile
gcloud run deploy manufacturing-efficiency \
  --source . \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

## Notes
- Cloud Run automatically scales based on traffic
- Pay-per-use pricing model
- SSL/TLS certificates automatically provided
- No server management required
