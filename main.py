#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Half-Time Draw Prediction

This script orchestrates the entire machine learning pipeline for predicting
half-time draws in football matches.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

# Import custom modules
from data_loading import load_data, explore_data, save_data
from data_processing import (
    handle_missing_values, detect_outliers, handle_outliers, 
    create_target_variable, visualize_data, select_features, 
    scale_features, save_preprocessor
)
from model_training import (
    split_data, train_logistic_regression, train_random_forest,
    train_gradient_boosting, train_xgboost, evaluate_model,
    compare_models, save_model
)
from util import (
    setup_environment, predict_new_match, save_results,
    plot_learning_curve, create_requirements_file, create_readme_file,
    print_section_header, save_visualization, save_configuration
)

def get_user_input():
    """Get user input for various parameters"""
    print("\n===== USER INPUT =====")
    
    # Data file
    data_file = input("Enter the path to the data file (default: matches.json): ") or "matches.json"
    
    # Missing values strategy
    print("\nHow would you like to handle missing values?")
    print("1. Ask during execution")
    print("2. Drop rows with missing values")
    print("3. Replace with median/mode")
    missing_choice = input("Enter your choice (1-3, default: 3): ") or "3"
    missing_strategy = {
        "1": "ask",
        "2": "drop",
        "3": "median"
    }.get(missing_choice, "median")
    
    # Outliers strategy
    print("\nHow would you like to handle outliers?")
    print("1. Cap outliers to boundaries")
    print("2. Remove rows with outliers")
    print("3. Do not handle outliers")
    outliers_choice = input("Enter your choice (1-3, default: 1): ") or "1"
    outliers_strategy = {
        "1": "cap",
        "2": "remove",
        "3": "none"
    }.get(outliers_choice, "cap")
    
    # Feature selection method
    print("\nWhich feature selection method would you like to use?")
    print("1. Mutual Information")
    print("2. ANOVA F-value")
    print("3. Principal Component Analysis (PCA)")
    feature_selection_choice = input("Enter your choice (1-3, default: 1): ") or "1"
    feature_selection_method = {
        "1": "mutual_info",
        "2": "anova",
        "3": "pca"
    }.get(feature_selection_choice, "mutual_info")
    
    # Scaling method
    print("\nWhich scaling method would you like to use?")
    print("1. Standard Scaling (mean=0, std=1)")
    print("2. MinMax Scaling (range [0, 1])")
    scaling_choice = input("Enter your choice (1-2, default: 1): ") or "1"
    scaling_method = {
        "1": "standard",
        "2": "minmax"
    }.get(scaling_choice, "standard")
    
    return {
        "data_file": data_file,
        "missing_strategy": missing_strategy,
        "outliers_strategy": outliers_strategy,
        "feature_selection_method": feature_selection_method,
        "scaling_method": scaling_method
    }

def main():
    """Main function to run the pipeline"""
    # Setup environment
    setup_environment()
    
    # Create requirements and README files
    create_requirements_file()
    create_readme_file()
    
    # Get user input
    params = get_user_input()
    
    print_section_header("HALF-TIME DRAW PREDICTION")
    print("Starting the machine learning pipeline...")
    
    # Step 1: Load data
    print_section_header("STEP 1: DATA LOADING")
    df = load_data(params["data_file"])
    if df is None:
        print("Error loading data. Exiting.")
        return
    
    # Step 2: Explore data
    print_section_header("STEP 2: DATA EXPLORATION")
    missing_percentage = explore_data(df)
    
    # Now that we have loaded the data, ask for the number of features
    print("\nAfter exploring the data, we can now determine the number of features to use.")
    print(f"Total number of features available: {df.shape[1]}")
    n_features = input(f"How many features would you like to select? (default: 15, max: {df.shape[1]}): ") or "15"
    n_features = min(int(n_features), df.shape[1])
    params["n_features"] = n_features
    
    # Step 3: Data cleaning and preprocessing
    print_section_header("STEP 3: DATA CLEANING AND PREPROCESSING")
    
    # Handle missing values
    df = handle_missing_values(df, strategy=params["missing_strategy"])
    
    # Create target variable
    df = create_target_variable(df)
    
    # Detect outliers
    outliers_count = detect_outliers(df, method='iqr')
    
    # Handle outliers
    df = handle_outliers(df, method='iqr', strategy=params["outliers_strategy"])
    
    # Step 4: Data visualization
    print_section_header("STEP 4: DATA VISUALIZATION")
    visualize_data(df, save_dir='visualizations')
    
    # Step 5: Feature selection and scaling
    print_section_header("STEP 5: FEATURE SELECTION AND SCALING")
    
    # Split data into train and test sets (80/20)
    X_train, X_test, y_train, y_test = split_data(df)
    
    # Select features
    X_train_selected, selector = select_features(
        X_train, y_train, method=params["feature_selection_method"], k=params["n_features"]
    )
    
    # Apply feature selection to test set
    if params["feature_selection_method"] == 'pca':
        # For PCA, we need to apply the same transformation
        X_test_selected = selector.transform(X_test)
    else:
        # For other methods, we can use the selector directly
        X_test_selected = selector.transform(X_test)
    
    # Scale features
    X_train_scaled, X_test_scaled, scaler = scale_features(
        X_train_selected, X_test_selected, method=params["scaling_method"]
    )
    
    # Create configuration dictionary with all parameters
    config = {
        "data_file": params["data_file"],
        "missing_strategy": params["missing_strategy"],
        "outliers_strategy": params["outliers_strategy"],
        "feature_selection_method": params["feature_selection_method"],
        "n_features": params["n_features"],
        "scaling_method": params["scaling_method"],
        "feature_names": list(X_train.columns) if hasattr(X_train, 'columns') else None,
        "selected_features": list(X_train.columns[selector.get_support()]) if hasattr(selector, 'get_support') and hasattr(X_train, 'columns') else None,
        "target_variable": "half_time_draw",
        "data_shape": df.shape,
        "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Calculate and store average values for specified columns
    columns_to_store_avg = ['mge', 'diff', 'ic', 'rc', 'ro']
    config["column_averages"] = {}
    
    for col in columns_to_store_avg:
        if col in df.columns:
            avg_value = df[col].mean()
            config["column_averages"][col] = float(avg_value)  # Convert numpy.float64 to Python float for JSON serialization
            print(f"Stored average value for '{col}': {avg_value:.4f}")
        else:
            print(f"Warning: Column '{col}' not found in the dataset. Cannot store average value.")
    
    # Save preprocessor with configuration
    save_preprocessor(scaler, selector, output_dir='models', config=config)
    
    # Step 6: Model training
    print_section_header("STEP 6: MODEL TRAINING")
    
    # Train models
    logreg_model = train_logistic_regression(X_train_scaled, y_train)
    rf_model = train_random_forest(X_train_scaled, y_train)
    gb_model = train_gradient_boosting(X_train_scaled, y_train)
    xgb_model = train_xgboost(X_train_scaled, y_train)
    
    # Step 7: Model evaluation
    print_section_header("STEP 7: MODEL EVALUATION")
    
    # Create a dictionary of models
    models = {
        'Logistic Regression': logreg_model,
        'Random Forest': rf_model,
        'Gradient Boosting': gb_model,
        'XGBoost': xgb_model
    }
    
    # Compare models
    best_model_name = compare_models(models, X_test_scaled, y_test, save_dir='results')
    
    # Update configuration with best model
    config["best_model"] = best_model_name
    save_configuration(config, output_dir='config')
    
    # Step 8: Save models
    print_section_header("STEP 8: SAVING MODELS")
    
    # Save all models
    for name, model in models.items():
        save_model(model, name.lower().replace(' ', '_'), output_dir='models')
    
    # Step 9: Learning curves
    print_section_header("STEP 9: LEARNING CURVES")
    
    # Plot learning curves for all models
    for name, model in models.items():
        print(f"\nPlotting learning curve for {name}...")
        plot_learning_curve(model, X_train_scaled, y_train, model_name=name, save_dir='results')
    
    print_section_header("PIPELINE COMPLETED")
    print(f"Best model: {best_model_name}")
    print(f"Models saved to models/")
    print(f"Configuration saved to config/config.json")
    print(f"Visualizations saved to visualizations/")
    print(f"Results saved to results/")
    print("Thank you for using the Half-Time Draw Prediction pipeline!")

if __name__ == "__main__":
    main()
