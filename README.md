# Lung Cancer Prediction App 🚬🫁

[![Streamlit app](https://static.streamlit.io/badges/10.svg)](https://lung-cancer-app-w4mfgfp9oevkgct94qm4f9.streamlit.app/)
[![GitHub Repo](https://img.shields.io/github/stars/DineshM2006/lung-cancer-app)](https://github.com/DineshM2006/lung-cancer-app)

**Live Demo**: https://lung-cancer-app-w4mfgfp9oevkgct94qm4f9.streamlit.app/

## Overview
Interactive web app to predict lung cancer risk using ML model.

- **Input**: 16 features (e.g., AGE, SMOKING (1=no/2=yes), GENDER).
- **Output**: Probability (metric), Risk: **YES** (>90%) or **NO**.
- Model: Pickled sklearn classifier (model.pkl).

## Usage
### Deployed
Click the link above, use sidebar sliders/selects, **Predict**.

### Local
```
git clone https://github.com/DineshM2006/lung-cancer-app.git
cd lung-cancer-app
pip install -r requirements.txt
streamlit run app.py
```

Open http://localhost:8501.

## Features List
- AGE: 20-90 slider
- SMOKING, YELLOW_FINGERS, etc.: 1=low/no, 2=high/yes
- GENDER: 1=F, 2=M (encoded)

## Model Training
See train_lung_model.py (local).

## Dependencies
See requirements.txt.

**Built with Streamlit & scikit-learn.**
