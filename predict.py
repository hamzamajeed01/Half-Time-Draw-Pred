#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Half-Time Draw Prediction - Real-world Prediction Script

This script loads real-world match data and makes predictions using the trained model.
"""

import os
import pandas as pd
import numpy as np
import json
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# Import custom modules
from data_loading import load_data
from data_processing import handle_missing_values
from util import load_configuration, save_results, print_section_header

def load_test_data(file_path='test.json'):
    """
    Load test data for prediction
    
    Parameters:
    -----------
    file_path : str
        Path to the test data file
        
    Returns:
    --------
    pd.DataFrame
        Test data
    """
    print_section_header("LOADING TEST DATA")
    
    # Load data
    df = load_data(file_path)
    
    if df is None:
        print(f"Error loading test data from {file_path}")
        return None
    
    print(f"Loaded {len(df)} matches from {file_path}")
    
    # Ensure all columns are numeric
    for col in df.columns:
        if df[col].dtype == 'object':
            try:
                df[col] = pd.to_numeric(df[col])
                print(f"Converted column '{col}' to numeric")
            except:
                print(f"Warning: Dropping non-numeric column '{col}'")
                df = df.drop(columns=[col])
    
    return df

def preprocess_test_data(df, config):
    """
    Preprocess test data for real-world predictions
    
    Parameters:
    -----------
    df : pd.DataFrame
        Test data
    config : dict
        Configuration parameters
        
    Returns:
    --------
    pd.DataFrame
        Preprocessed test data
    """
    print_section_header("PREPROCESSING TEST DATA")
    
    # Handle missing values
    df = handle_missing_values(df, strategy=config['missing_strategy'])
    
    # Remove target variable if it exists (we're predicting it)
    target_col = config['target_variable']
    if target_col in df.columns:
        print(f"Found target variable '{target_col}' in test data. Removing it as we're predicting it.")
        df = df.drop(columns=[target_col])
    
    # Ensure we have all the required features from training
    if 'feature_names' in config and config['feature_names']:
        required_features = config['feature_names']
        
        # Check for missing features
        missing_features = [f for f in required_features if f not in df.columns]
        if missing_features:
            print(f"Warning: Missing {len(missing_features)} features that were used during training.")
            print(f"Adding missing features with default value 0: {missing_features}")
            for feature in missing_features:
                df[feature] = 0
        
        # Check for extra features
        extra_features = [f for f in df.columns if f not in required_features]
        if extra_features:
            print(f"Warning: Found {len(extra_features)} extra features not used during training.")
            print(f"Dropping extra features: {extra_features}")
            df = df.drop(columns=extra_features)
        
        # Ensure columns are in the same order as during training
        df = df[required_features]
        print(f"Reordered columns to match training data. Final shape: {df.shape}")
    
    return df

def make_predictions(df, config_path='config/config.json'):
    """
    Make predictions on test data
    
    Parameters:
    -----------
    df : pd.DataFrame
        Test data
    config_path : str
        Path to the configuration file
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with predictions
    """
    print_section_header("MAKING PREDICTIONS")
    
    # Load configuration
    config = load_configuration(config_path)
    
    # Load preprocessor
    scaler_path = 'models/scaler.joblib'
    selector_path = 'models/selector.joblib'
    
    # Determine model path
    if 'best_model' in config:
        best_model = config['best_model']
        model_path = f"models/{best_model.lower().replace(' ', '_')}.joblib"
        print(f"Using best model: {best_model}")
    else:
        # Default to XGBoost if no best model is specified
        model_path = 'models/xgboost.joblib'
        print(f"No best model specified in config. Using XGBoost.")
    
    # Load model, scaler, and selector
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    selector = joblib.load(selector_path)
    
    print(f"Data shape before feature selection: {df.shape}")
    
    # Apply feature selection based on the method used during training
    feature_selection_method = config.get('feature_selection_method', 'mutual_info')
    print(f"Using feature selection method: {feature_selection_method}")
    
    try:
        X_selected = selector.transform(df)
        print(f"Applied feature selection successfully. Shape after selection: {X_selected.shape[0]} x {X_selected.shape[1]}")
    except Exception as e:
        print(f"Error during feature selection: {str(e)}")
        print("This might be due to data format issues. Please check your test data.")
        return None
    
    # Apply scaling
    try:
        X_scaled = scaler.transform(X_selected)
        print("Applied feature scaling successfully")
    except Exception as e:
        print(f"Error during scaling: {str(e)}")
        return None
    
    # Make predictions
    try:
        y_pred = model.predict(X_scaled)
        y_pred_proba = model.predict_proba(X_scaled)[:, 1]
        print(f"Successfully made predictions for {len(y_pred)} samples")
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return None
    
    # Create results DataFrame
    results = pd.DataFrame({
        'match_id': df.index if 'match_id' not in df.columns else df['match_id'],
        'prediction': y_pred,
        'probability': y_pred_proba
    })
    
    # Add human-readable prediction
    results['result'] = results['prediction'].apply(lambda x: 'DRAW' if x == 1 else 'NO DRAW')
    
    # Add confidence level
    def get_confidence(prob):
        if prob < 0.5:
            prob = 1 - prob  # Convert to confidence for NO DRAW
        
        if prob >= 0.9:
            return 'Very High'
        elif prob >= 0.75:
            return 'High'
        elif prob >= 0.6:
            return 'Medium'
        else:
            return 'Low'
    
    results['confidence'] = results['probability'].apply(get_confidence)
    
    # Display predictions in a nice format
    print("\nPrediction Results:")
    display_results = results[['match_id', 'result', 'probability', 'confidence']]
    display_results = display_results.rename(columns={
        'match_id': 'Match ID',
        'result': 'Prediction',
        'probability': 'Probability',
        'confidence': 'Confidence'
    })
    
    # Format probability as percentage
    display_results['Probability'] = display_results['Probability'].apply(lambda x: f"{x*100:.1f}%")
    
    try:
        print(tabulate(display_results, headers='keys', tablefmt='pretty', showindex=False))
    except ImportError:
        print(display_results)
    
    # Summary statistics
    draw_count = results[results['prediction'] == 1].shape[0]
    total_count = results.shape[0]
    draw_percentage = (draw_count / total_count) * 100
    
    print(f"\nSummary:")
    print(f"Total matches: {total_count}")
    print(f"Predicted draws: {draw_count} ({draw_percentage:.1f}%)")
    print(f"Predicted no draws: {total_count - draw_count} ({100 - draw_percentage:.1f}%)")
    
    return results

def main():
    """Main function to run the prediction pipeline"""
    print_section_header("HALF-TIME DRAW PREDICTION - REAL-WORLD PREDICTION")
    
    # Load test data
    print("\nSelect test data file:")
    print("1. Use default (test.json)")
    print("2. Enter custom file path")
    file_choice = input("Enter your choice (1-2, default: 1): ") or "1"
    
    if file_choice == "1":
        test_file = "test.json"
        print(f"Using default file: {test_file}")
    else:
        test_file = input("Enter the path to the test data file: ")
    
    df = load_test_data(test_file)
    
    if df is None:
        print("Error loading test data. Exiting.")
        return
    
    # Load configuration
    config_path = 'config/config.json'
    try:
        config = load_configuration(config_path)
    except FileNotFoundError:
        print(f"Configuration file not found at {config_path}. Please run the training pipeline first.")
        return
    
    # Preprocess test data
    df = preprocess_test_data(df, config)
    
    # Make predictions
    results = make_predictions(df, config_path)
    
    if results is None:
        print("Failed to make predictions. Please check the errors above.")
        return
    
    # Save results
    print("\nSelect where to save results:")
    print("1. Use default (results/predictions.csv)")
    print("2. Enter custom file path")
    save_choice = input("Enter your choice (1-2, default: 1): ") or "1"
    
    if save_choice == "1":
        output_path = "results/predictions.csv"
        print(f"Using default path: {output_path}")
    else:
        output_path = input("Enter the path to save the results: ")
    
    save_results(results, output_path)
    
    print_section_header("PREDICTION COMPLETED")
    print(f"Predictions saved to {output_path}")
    print("\nYou can use these predictions to make informed decisions about half-time draw bets.")
    print("Higher confidence predictions are more reliable.")

if __name__ == "__main__":
    main() 