# CSE 158 Assignment 2 - Restaurant Rating Prediction

Personalized restaurant recommendation system using the Google Restaurants dataset.

## Task

Predict a personalized rating for a user-restaurant pair. Evaluated using RMSE and MAE.

## Dataset

Google Restaurants dataset from Google Local (Google Maps) with user reviews, ratings, and historical review data.

## Results

| Model | Test RMSE | Test MAE |
|-------|-----------|----------|
| Ridge Regression | 1.2473 | 0.8320 |
| SVD | 0.9123 | 0.6630 |
| SBERT + XGBoost | 0.8887 | 0.6695 |

Best model: **SBERT + XGBoost** (Test RMSE: 0.8887)

## Setup

```bash
uv venv
uv pip install -e .
```

## Usage

```bash
jupyter notebook improved_baseline.ipynb
```
