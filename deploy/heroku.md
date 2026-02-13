# Heroku Deployment Guide

## Prerequisites
- Heroku CLI installed
- Heroku account

## Deployment Steps

### 1. Login to Heroku
```bash
heroku login
```

### 2. Create Heroku App
```bash
heroku create your-app-name
```

### 3. Set Build Pack
```bash
heroku buildpacks:set heroku/python
```

### 4. Add Environment Variables
```bash
heroku config:set STREAMLIT_SERVER_PORT=8501
heroku config:set STREAMLIT_SERVER_ADDRESS=0.0.0.0
heroku config:set STREAMLIT_SERVER_HEADLESS=true
```

### 5. Deploy
```bash
git add .
git commit -m "Deploy to Heroku"
git push heroku main
```

### 6. Open App
```bash
heroku open
```

## Notes
- Heroku automatically detects the Python app from requirements.txt
- The app will be available at https://your-app-name.herokuapp.com
- Free tier has limitations, consider paid plans for production
