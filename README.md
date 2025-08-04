# Car Rental Dynamic Book Prediction

A machine learning system that helps customers decide whether to **book now** or **wait** for better car rental prices within a 7-day window.

## Overview

This project implements four different ML approaches to solve the "Book or Wait" decision problem:

- **XGBoost** (ROC-AUC: 0.9121) - Production ready
- **Logistic Regression** (ROC-AUC: 0.9032) - Baseline
- **Prophet** (88.9% accuracy) - Time series forecasting
- **LSTM** (ROC-AUC: 0.5499) - Sequential learning (data limited)

## Project Structure

```text
├── data/                    # Training datasets (MLOps standard)
│   ├── processed/          # Ready-to-use training data
│   └── README.md
├── models/                 # Trained models and artifacts
├── notebooks/              # Jupyter notebooks and experimental code
│   ├── data/              # Experimental and analysis datasets
│   └── scripts/           # Data generation and utility scripts
├── src/
│   └── car_rental_prediction/
│       ├── models/        # Production model implementations
│       └── *.md          # Strategy and documentation
└── README.md
```

## Quick Start

1. **Install dependencies:**

   ```bash
   uv sync
   ```

2. **Run model training:**

   ```bash
   python src/car_rental_prediction/models/xgboost_model.py
   ```

3. **Explore notebooks:**

   ```bash
   jupyter lab notebooks/
   ```

## Key Results

- **Customer Savings**: 5-15% through optimal booking timing
- **Best Model**: XGBoost with 0.9121 ROC-AUC
- **Production Ready**: Complete feature pipeline and model artifacts
- **Business Impact**: Validated cost-benefit analysis showing positive ROI

## Documentation

- **Model Strategy**: `src/car_rental_prediction/BOOK_OR_WAIT_MODEL_STRATEGY.md`
- **Data Generation**: `src/car_rental_prediction/DATA_GENERATION_STRATEGY.md`
- **Training Data**: `data/README.md`

## Models

All models are trained on synthetic data representing realistic car rental scenarios with proper temporal patterns and business logic validation.