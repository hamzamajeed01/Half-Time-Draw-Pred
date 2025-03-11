import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest

def setup_environment():
    """
    Set up the environment for the project
    
    Parameters:
    -----------
    None
        
    Returns:
    --------
    None
    """
    # Set up matplotlib style
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Set up seaborn style
    sns.set(style="whitegrid")
    
    # Create directories if they don't exist
    os.makedirs('models', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    os.makedirs('visualizations', exist_ok=True)
    
    print("Environment setup complete.")

def predict_new_match(match_data, model_path=None, scaler_path=None, selector_path=None, config_path=None):
    """
    Predict the outcome of a new match
    
    Parameters:
    -----------
    match_data : pd.DataFrame or dict
        Data for the new match
    model_path : str or None
        Path to the saved model. If None, will use the best model from config
    scaler_path : str or None
        Path to the saved scaler. If None, will use default path
    selector_path : str or None
        Path to the saved feature selector. If None, will use default path
    config_path : str or None
        Path to the configuration file. If None, will use default path
        
    Returns:
    --------
    tuple
        (prediction, probability) - Prediction and probability of a draw
    """
    # Load configuration if provided
    if config_path is not None:
        config = load_configuration(config_path)
    else:
        try:
            config = load_configuration()
        except FileNotFoundError:
            config = None
            print("No configuration file found. Using provided paths.")
    
    # Determine paths based on configuration
    if model_path is None and config is not None and "best_model" in config:
        best_model = config["best_model"]
        model_path = f"models/{best_model.lower().replace(' ', '_')}.joblib"
        print(f"Using best model from configuration: {best_model}")
    
    if scaler_path is None:
        scaler_path = "models/scaler.joblib"
    
    if selector_path is None:
        selector_path = "models/selector.joblib"
    
    # Load the model
    model = joblib.load(model_path)
    
    # Convert dict to DataFrame if necessary
    if isinstance(match_data, dict):
        match_data = pd.DataFrame([match_data])
    
    # Preprocess the data
    X = match_data.copy()
    
    # Apply feature selection if provided
    if selector_path is not None:
        selector = joblib.load(selector_path)
        if hasattr(selector, 'transform'):
            X = selector.transform(X)
    
    # Apply scaler if provided
    if scaler_path is not None:
        scaler = joblib.load(scaler_path)
        X = scaler.transform(X)
    
    # Make prediction
    prediction = model.predict(X)[0]
    probability = model.predict_proba(X)[0, 1]
    
    return prediction, probability

def save_results(results, output_path='results/results.csv'):
    """
    Save results to a CSV file
    
    Parameters:
    -----------
    results : pd.DataFrame
        Results to save
    output_path : str
        Path to save the results
        
    Returns:
    --------
    None
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save results
    results.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")

def save_visualization(fig, filename, save_dir='visualizations'):
    """
    Save a matplotlib figure to a file
    
    Parameters:
    -----------
    fig : matplotlib.figure.Figure
        Figure to save
    filename : str
        Name of the file to save
    save_dir : str
        Directory to save the file
        
    Returns:
    --------
    None
    """
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Save figure
    filepath = os.path.join(save_dir, filename)
    fig.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to {filepath}")

def plot_learning_curve(model, X, y, cv=5, train_sizes=np.linspace(0.1, 1.0, 5), model_name=None, save_dir=None):
    """
    Plot the learning curve for a model
    
    Parameters:
    -----------
    model : object
        Model to evaluate
    X : array-like
        Features
    y : array-like
        Target
    cv : int
        Number of cross-validation folds
    train_sizes : array-like
        Training set sizes to evaluate
    model_name : str
        Name of the model for the plot title
    save_dir : str or None
        Directory to save the plot, if None, the plot is not saved
        
    Returns:
    --------
    None
    """
    from sklearn.model_selection import learning_curve
    
    train_sizes, train_scores, test_scores = learning_curve(
        model, X, y, cv=cv, train_sizes=train_sizes, scoring='f1', n_jobs=-1
    )
    
    # Calculate mean and standard deviation
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    
    # Plot learning curve
    fig = plt.figure(figsize=(10, 6))
    plt.grid()
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_mean, 'o-', color="g", label="Cross-validation score")
    
    if model_name:
        plt.title(f"Learning Curve - {model_name}")
    else:
        plt.title("Learning Curve")
        
    plt.xlabel("Training examples")
    plt.ylabel("F1 Score")
    plt.legend(loc="best")
    plt.tight_layout()
    
    # Save the figure if save_dir is provided
    if save_dir:
        filename = f"learning_curve_{model_name.lower().replace(' ', '_')}.png" if model_name else "learning_curve.png"
        save_visualization(fig, filename, save_dir)
    
    plt.show()

def create_requirements_file(output_path='requirements.txt'):
    """
    Create a requirements.txt file for the project
    
    Parameters:
    -----------
    output_path : str
        Path to save the requirements file
        
    Returns:
    --------
    None
    """
    requirements = [
        "pandas==1.5.3",
        "numpy==1.24.3",
        "matplotlib==3.7.1",
        "seaborn==0.12.2",
        "scikit-learn==1.2.2",
        "xgboost==1.7.5",
        "joblib==1.2.0"
    ]
    
    with open(output_path, 'w') as f:
        for req in requirements:
            f.write(f"{req}\n")
    
    print(f"Requirements file created at {output_path}")

def create_readme_file(output_path='README.md'):
    """
    Create a README.md file for the project
    
    Parameters:
    -----------
    output_path : str
        Path to save the README file
        
    Returns:
    --------
    None
    """
    readme_content = """# Half-Time Draw Prediction

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
"""
    
    with open(output_path, 'w') as f:
        f.write(readme_content)
    
    print(f"README file created at {output_path}")

def print_section_header(title):
    """
    Print a section header
    
    Parameters:
    -----------
    title : str
        Title of the section
        
    Returns:
    --------
    None
    """
    print(f"\n{'='*10} {title} {'='*10}")

def save_configuration(config, output_dir='config'):
    """
    Save configuration parameters to a JSON file
    
    Parameters:
    -----------
    config : dict
        Configuration parameters
    output_dir : str
        Directory to save the configuration file
        
    Returns:
    --------
    str
        Path to the saved configuration file
    """
    import json
    
    # Create directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save configuration
    config_path = os.path.join(output_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    
    print(f"Configuration saved to {config_path}")
    return config_path

def load_configuration(config_path='config/config.json'):
    """
    Load configuration parameters from a JSON file
    
    Parameters:
    -----------
    config_path : str
        Path to the configuration file
        
    Returns:
    --------
    dict
        Configuration parameters
    """
    import json
    
    # Load configuration
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    print(f"Configuration loaded from {config_path}")
    return config
