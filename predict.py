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
    
    # First, try to load the raw data to extract halftime_result
    halftime_result = None
    try:
        if file_path.endswith('.json'):
            raw_df = pd.read_json(file_path)
            if 'halftime_result' in raw_df.columns:
                halftime_result = raw_df['halftime_result'].copy()
                print(f"Successfully preserved halftime_result column from raw data")
        elif file_path.endswith('.csv'):
            raw_df = pd.read_csv(file_path)
            if 'halftime_result' in raw_df.columns:
                halftime_result = raw_df['halftime_result'].copy()
                print(f"Successfully preserved halftime_result column from raw data")
    except Exception as e:
        print(f"Warning: Could not load raw data to preserve halftime_result: {str(e)}")
    
    # Load data through the standard pipeline
    df = load_data(file_path)
    
    if df is None:
        print(f"Error loading test data from {file_path}")
        return None
    
    print(f"Loaded {len(df)} matches from {file_path}")
    
    # Save halftime_result for display purposes before it gets dropped
    if halftime_result is None and 'halftime_result' in df.columns:
        halftime_result = df['halftime_result'].copy()
        print(f"Preserved halftime_result column from processed data")
    
    # Ensure all columns are numeric
    for col in df.columns:
        if df[col].dtype == 'object':
            try:
                df[col] = pd.to_numeric(df[col])
                print(f"Converted column '{col}' to numeric")
            except:
                print(f"Warning: Dropping non-numeric column '{col}'")
                df = df.drop(columns=[col])
    
    # Add back halftime_result for display purposes
    if halftime_result is not None:
        df['halftime_result_display'] = halftime_result
        print(f"Added halftime_result_display column to dataset (length: {len(halftime_result)})")
        # Print a sample to verify
        print(f"Sample halftime_result values: {halftime_result.head(3).tolist()}")
    else:
        print(f"Warning: Could not preserve halftime_result column")
    
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
    
    # Save halftime_result_display if it exists
    halftime_result_display = None
    if 'halftime_result_display' in df.columns:
        halftime_result_display = df['halftime_result_display'].copy()
        df = df.drop(columns=['halftime_result_display'])
    
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
        extra_features = [f for f in df.columns if f not in required_features and f != 'halftime_result_display']
        if extra_features:
            print(f"Warning: Found {len(extra_features)} extra features not used during training.")
            print(f"Dropping extra features: {extra_features}")
            df = df.drop(columns=extra_features)
        
        # Ensure columns are in the same order as during training
        df = df[required_features]
        print(f"Reordered columns to match training data. Final shape: {df.shape}")
    
    # Add back halftime_result_display
    if halftime_result_display is not None:
        df['halftime_result_display'] = halftime_result_display
    
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
    
    # Debug: Check if halftime_result_display is in the dataframe
    print(f"Columns in dataframe before processing: {df.columns.tolist()}")
    if 'halftime_result_display' in df.columns:
        print(f"halftime_result_display column found with {df['halftime_result_display'].count()} non-null values")
        print(f"Sample values: {df['halftime_result_display'].head(3).tolist()}")
    else:
        print("WARNING: halftime_result_display column not found in the dataframe")
    
    # Save halftime_result for display purposes if it exists
    halftime_result_display = None
    if 'halftime_result_display' in df.columns:
        halftime_result_display = df['halftime_result_display'].copy()
        print(f"Saved halftime_result_display with {len(halftime_result_display)} values")
        df = df.drop(columns=['halftime_result_display'])
    
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
    
    # Add actual result if available
    if halftime_result_display is not None:
        print(f"Processing halftime_result_display with {len(halftime_result_display)} values")
        
        # Create a function to determine if the halftime result was a draw
        def is_draw(halftime_result):
            try:
                # Extract home and away goals from halftime_result
                import re
                match = re.search(r'\((\d+)\s*-\s*(\d+)\)', str(halftime_result))
                if match:
                    ht_home = int(match.group(1))
                    ht_away = int(match.group(2))
                    result = 'DRAW' if ht_home == ht_away else 'NO DRAW'
                   
                    return result
                
                # Pattern 2: 0-1 format (without parentheses)
                match = re.search(r'(\d+)\s*-\s*(\d+)', str(halftime_result))
                if match:
                    ht_home = int(match.group(1))
                    ht_away = int(match.group(2))
                    result = 'DRAW' if ht_home == ht_away else 'NO DRAW'
                    print(f"Extracted scores from pattern 2: {ht_home}-{ht_away}, Result: {result}")
                    return result
                
                print(f"No match found in: {halftime_result}")
                return 'UNKNOWN'
            except Exception as e:
                print(f"Error processing halftime result: {halftime_result}, Error: {str(e)}")
                return 'UNKNOWN'
        
        # Process a few examples to debug
        print("Debug: Processing a few examples:")
        for i, val in enumerate(halftime_result_display.head(3)):
            print(f"Example {i+1}: {val} -> {is_draw(val)}")
        
        # Add actual result to results DataFrame
        results['actual_result'] = halftime_result_display.apply(is_draw)
        print(f"Added actual results based on halftime_result_display column")
        print(f"Actual results distribution: {results['actual_result'].value_counts().to_dict()}")
    else:
        print("Warning: halftime_result_display is None. Cannot add actual results.")
    
    # Display predictions in a nice format
    print("\nPrediction Results:")
    
    # Debug: Check if actual_result is in the results DataFrame
    print(f"Columns in results DataFrame: {results.columns.tolist()}")
    
    # Include actual result in display if available
    if 'actual_result' in results.columns:
        print(f"actual_result column found with {results['actual_result'].count()} non-null values")
        display_results = results[['match_id', 'result', 'actual_result', 'probability', 'confidence']]
        display_results = display_results.rename(columns={
            'match_id': 'Match ID',
            'result': 'Prediction',
            'actual_result': 'Actual Result',
            'probability': 'Probability',
            'confidence': 'Confidence'
        })
    else:
        print("WARNING: actual_result column not found in the results DataFrame")
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
    
    # Add accuracy statistics if actual results are available
    if 'actual_result' in results.columns:
        correct_predictions = results[results['result'] == results['actual_result']].shape[0]
        accuracy = (correct_predictions / total_count) * 100
        print(f"\nAccuracy:")
        print(f"Correct predictions: {correct_predictions} out of {total_count} ({accuracy:.1f}%)")
    else:
        print("\nNo actual results available for accuracy calculation.")
    
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
    
    # Debug: Check if halftime_result_display is in the dataframe after loading
    print(f"\nColumns in dataframe after loading: {df.columns.tolist()}")
    if 'halftime_result_display' in df.columns:
        print(f"halftime_result_display column found with {df['halftime_result_display'].count()} non-null values")
        print(f"Sample values: {df['halftime_result_display'].head(3).tolist()}")
    else:
        print("WARNING: halftime_result_display column not found after loading")
    
    # Load configuration
    config_path = 'config/config.json'
    try:
        config = load_configuration(config_path)
    except FileNotFoundError:
        print(f"Configuration file not found at {config_path}. Please run the training pipeline first.")
        return
    
    # Add halftime_result_display to feature_names if it exists
    if 'feature_names' in config and 'halftime_result_display' in df.columns:
        # Make a copy of the original feature names
        original_feature_names = config['feature_names'].copy()
        # We'll restore this later
        config['original_feature_names'] = original_feature_names
    
    # Preprocess test data
    df_processed = preprocess_test_data(df, config)
    
    # Debug: Check if halftime_result_display is in the dataframe after preprocessing
    print(f"\nColumns in dataframe after preprocessing: {df_processed.columns.tolist()}")
    if 'halftime_result_display' in df_processed.columns:
        print(f"halftime_result_display column found with {df_processed['halftime_result_display'].count()} non-null values")
        print(f"Sample values: {df_processed['halftime_result_display'].head(3).tolist()}")
    else:
        print("WARNING: halftime_result_display column not found after preprocessing")
        
        # If we lost the column during preprocessing, add it back
        if 'halftime_result_display' in df.columns:
            print("Restoring halftime_result_display column from original dataframe")
            df_processed['halftime_result_display'] = df['halftime_result_display']
    
    # Restore original feature_names if we modified it
    if 'original_feature_names' in config:
        config['feature_names'] = config['original_feature_names']
        del config['original_feature_names']
    
    # Make predictions
    results = make_predictions(df_processed, config_path)
    
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