# Streamlit Community Cloud Deployment Guide

## 🚀 Deploy to Streamlit Community Cloud

### Prerequisites
1. GitHub repository with your code
2. Streamlit account (free)
3. All files committed to GitHub

### Step 1: Prepare Your Repository
Your repository should have:
- `streamlit_dashboard.py` (main app file)
- `requirements.txt` (dependencies)
- `model.joblib` (trained model)
- `Thales_Group_Manufacturing.csv` (dataset)
- `.streamlit/config.toml` (configuration)

### Step 2: Deploy to Streamlit Cloud

1. **Go to Streamlit Cloud**
   - Visit https://share.streamlit.io
   - Click "Sign in" and connect your GitHub account

2. **Create New App**
   - Click "New app"
   - Select your GitHub repository
   - Select branch: `main`
   - Main file: `streamlit_dashboard.py`
   - Click "Deploy"

3. **Wait for Deployment**
   - Streamlit will automatically:
     - Install dependencies from `requirements.txt`
     - Build your app
     - Deploy to a public URL

### Step 3: Access Your App
- Your app will be available at: `https://your-username-your-repo-streamlit-app.streamlit.app`
- Share this URL with stakeholders

## 📋 Streamlit Cloud Features

### Free Tier Benefits
- **Public apps** with unlimited viewers
- **Private apps** (up to 3 collaborators)
- **30 hours** of compute time per month
- **1 GB** storage per app

### Limitations
- Apps go to sleep after 10 minutes of inactivity
- Cold start takes 30-60 seconds
- No custom domains on free tier

## 🔧 Configuration

### Environment Variables (Optional)
In your Streamlit app settings:
1. Go to your app dashboard
2. Click "Advanced settings"
3. Add environment variables if needed

### Secrets Management
For sensitive data:
```python
import streamlit as st

# Access secrets
secret_key = st.secrets["SECRET_KEY"]
```

## 📊 Performance Tips

### Reduce Cold Start Time
1. **Optimize imports** - only import what you need
2. **Cache model loading** - use `@st.cache_resource`
3. **Lazy load data** - load data when needed

### Memory Optimization
```python
# Cache expensive operations
@st.cache_data
def load_data():
    return pd.read_csv('Thales_Group_Manufacturing.csv')

@st.cache_resource
def load_model():
    return joblib.load('model.joblib')
```

## 🔄 Updates and Maintenance

### Updating Your App
1. Push changes to GitHub
2. Streamlit automatically redeploys
3. Or manually trigger redeploy in dashboard

### Monitoring
- Check app logs in Streamlit dashboard
- Monitor usage statistics
- Set up alerts for errors

## 🔒 Security Considerations

### Public Apps
- Don't expose sensitive data
- Use environment variables for secrets
- Validate user inputs

### Private Apps
- Limit collaborators
- Use GitHub authentication
- Regular security updates

## 📈 Scaling Options

### Pro Tier ($9/month)
- **Private apps** with unlimited collaborators
- **Custom domains**
- **Priority support**
- **More compute hours**

### Enterprise Tier
- **SSO/SAML**
- **Advanced security**
- **Dedicated resources**
- **Custom integrations**

## 🐛 Troubleshooting

### Common Issues

#### 1. Import Errors
```bash
# Check requirements.txt
pip install -r requirements.txt
```

#### 2. Model Loading Issues
```python
# Use absolute paths
import os
model_path = os.path.join(os.path.dirname(__file__), 'model.joblib')
```

#### 3. Memory Issues
```python
# Add memory optimization
import gc
# Clear memory after use
gc.collect()
```

#### 4. Slow Loading
```python
# Add loading spinner
with st.spinner('Loading model...'):
    model = load_model()
```

### Debug Mode
Add this to your app for debugging:
```python
if st.secrets.get("DEBUG_MODE", False):
    st.write("Debug info:", debug_data)
```

## 📞 Support Resources

### Documentation
- [Streamlit Docs](https://docs.streamlit.io/)
- [Deployment Guide](https://docs.streamlit.io/knowledge-base/tutorials/deploy)

### Community
- [Streamlit Community](https://discuss.streamlit.io/)
- [GitHub Issues](https://github.com/streamlit/streamlit/issues)

### Help
- Streamlit support team
- Community forums
- Stack Overflow

---

**Quick Start Summary:**
1. Push code to GitHub
2. Go to https://share.streamlit.io
3. Connect GitHub and select repo
4. Choose `streamlit_dashboard.py`
5. Click "Deploy"
6. Share your URL! 🎉
