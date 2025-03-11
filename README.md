# Half-Time Draw Prediction

This project aims to predict whether a football match will be a draw at half-time.

## Project Structure

- `main.py`: Main script to run the entire pipeline
- `data_loading.py`: Functions for loading and exploring data
- `data_processing.py`: Functions for data cleaning, preprocessing, and feature engineering
- `model_training.py`: Functions for model training and evaluation
- `util.py`: Utility functions

## Setup

1. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

2. Run the main script:
   ```
   python main.py
   ```

## Data

The project uses football match data from the `matches.json` file, which contains various features about football matches.

## Models

The project trains and evaluates several machine learning models:
- Logistic Regression
- Random Forest
- Gradient Boosting
- XGBoost

## Results

The models are evaluated using various metrics including accuracy, precision, recall, F1 score, ROC AUC, and PR AUC.
