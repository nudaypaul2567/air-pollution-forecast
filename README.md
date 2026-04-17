# Air Pollution Forecasting: Ensemble ML Approach for O₃ and NO₂

## Overview
An ensemble machine learning framework combining XGBoost and LightGBM for short-term forecasting of ground-level ozone (O₃) and nitrogen dioxide (NO₂) using Copernicus Atmosphere Monitoring Service (CAMS) reanalysis data.

## Key Results
- **O₃ Prediction:** R² = 0.936, RMSE = 5.58 µg/m³
- **NO₂ Prediction:** R² = 0.874, RMSE = 8.37 µg/m³
- **Dataset:** 5 years of hourly data from Delhi, India (July 2019 - June 2024)
- **Deployment:** FastAPI backend + React interactive dashboard

## Project Structure
\\\
air-pollution-forecast/
├── docs/                      # Documentation and paper
├── data/
│   ├── raw/                  # Original data
│   └── processed/            # Cleaned data
├── notebooks/                # Jupyter notebooks for EDA
├── src/
│   ├── models/              # Model training scripts
│   ├── features/            # Feature engineering
│   └── deployment/          # FastAPI backend
├── models/                  # Trained model files
├── tests/                   # Unit tests
├── config/                  # Configuration files
├── requirements.txt
└── README.md
\\\

## Installation

\\\ash
# Create virtual environment
python -m venv venv
venv\Scripts\activate  # On Windows

# Install dependencies
pip install -r requirements.txt
\\\

## Authors
- Nakka Uday Paul (nakkauday@karunya.edu)
- T Kavita (kavithat@karunya.edu)

## License
MIT License
