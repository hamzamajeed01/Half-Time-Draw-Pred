import pandas as pd
import numpy as np
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(file_path):
    """
    Load data from JSON file and return a pandas DataFrame
    
    Parameters:
    -----------
    file_path : str
        Path to the JSON file
        
    Returns:
    --------
    pd.DataFrame
        Loaded data
    """
    print("Loading data from:", file_path)
    try:
        if file_path.endswith('.json'):
            df = pd.read_json(file_path)
            print(f"Successfully loaded data from {file_path}")
            print(f"Dataset shape: {df.shape}")
            
            # Drop specified columns
            columns_to_drop = ['id', 'date', 'hour', 'match', 'result', 'home_goals', 'away_goals', 'home_goalsht', 'away_goalsht']
            existing_columns = [col for col in columns_to_drop if col in df.columns]
            
            if existing_columns:
                df = df.drop(columns=existing_columns)
                print(f"\nDropped the following columns: {existing_columns}")
                print(f"Dataset shape after dropping columns: {df.shape}")
            
            return df
        elif file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
            print(f"Successfully loaded data from {file_path}")
            print(f"Dataset shape: {df.shape}")
            
            # Drop specified columns
            columns_to_drop = ['id', 'date', 'hour', 'match', 'result', 'home_goals', 'away_goals', 'home_goalsht', 'away_goalsht']
            existing_columns = [col for col in columns_to_drop if col in df.columns]
            
            if existing_columns:
                df = df.drop(columns=existing_columns)
                print(f"\nDropped the following columns: {existing_columns}")
                print(f"Dataset shape after dropping columns: {df.shape}")
            
            return df
        else:
            raise ValueError("Unsupported file format. Please provide a JSON or CSV file.")
    except FileNotFoundError:
        print(f"Error: '{file_path}' not found.")
        return None
    except ValueError as e:
        print(f"Error reading file: {e}")
        return None

def explore_data(df):
    """
    Perform initial data exploration
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame to explore
        
    Returns:
    --------
    None
    """
    print("\n===== DATA EXPLORATION =====")
    print("\nFirst 5 rows:")
    print(df.head())
    
    print("\nShape of the DataFrame:", df.shape)
    print("\nData types of each column:\n", df.dtypes)
    print("\nDescriptive statistics:\n", df.describe())
    
    # Check for missing values
    missing_values = df.isnull().sum()
    missing_percentage = (missing_values / len(df)) * 100
    
    print("\nMissing values (NaNs) per column:")
    for col, count in missing_values.items():
        if count > 0:
            print(f"{col}: {count} ({missing_percentage[col]:.2f}%)")
    
    # Check for unique values in categorical columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    print("\nUnique values in categorical columns:")
    for col in categorical_cols:
        print(f"{col}: {df[col].nunique()} unique values")
        if df[col].nunique() < 10:  # Only show if there are few unique values
            print(f"Values: {df[col].unique()}")
    
    return missing_percentage

def save_data(df, file_path):
    """
    Save DataFrame to a CSV file
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame to save
    file_path : str
        Path to save the CSV file
        
    Returns:
    --------
    None
    """
    try:
        df.to_csv(file_path, index=False)
        print(f"Data successfully saved to {file_path}")
    except Exception as e:
        print(f"Error saving data: {e}")
