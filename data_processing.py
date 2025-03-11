import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.decomposition import PCA
import joblib
import os

def handle_missing_values(df, strategy='ask'):
    """
    Handle missing values in the DataFrame
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with missing values
    strategy : str
        Strategy to handle missing values ('ask', 'drop', or 'median')
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with handled missing values
    """
    print("\n===== HANDLING MISSING VALUES =====")
    
    # Check if there are any missing values
    missing_values = df.isnull().sum()
    if missing_values.sum() == 0:
        print("No missing values found in the dataset.")
        return df
    
    # Calculate missing percentage
    missing_percentage = (missing_values / len(df)) * 100
    
    # Columns with high percentage of missing values (>50%)
    high_missing_cols = missing_percentage[missing_percentage > 50].index.tolist()
    if high_missing_cols:
        print(f"\nDropping columns with >50% missing values: {high_missing_cols}")
        df = df.drop(columns=high_missing_cols)
    
    # Handle remaining missing values
    if strategy == 'ask':
        user_input = input("\nHow would you like to handle remaining missing values?\n1. Drop rows with missing values\n2. Replace with median/mode\nEnter 1 or 2: ")
        if user_input == '1':
            strategy = 'drop'
        else:
            strategy = 'median'
    
    if strategy == 'drop':
        rows_before = len(df)
        df = df.dropna()
        rows_after = len(df)
        print(f"Dropped {rows_before - rows_after} rows with missing values.")
    else:
        # Fill numerical columns with median
        numerical_cols = df.select_dtypes(include=['number']).columns
        for col in numerical_cols:
            if df[col].isnull().any():
                median_value = df[col].median()
                df[col] = df[col].fillna(median_value)
                print(f"Filled missing values in '{col}' with median: {median_value:.4f}")
        
        # Fill categorical columns with mode
        categorical_cols = df.select_dtypes(exclude=['number']).columns
        for col in categorical_cols:
            if df[col].isnull().any():
                mode_value = df[col].mode()[0]
                df[col] = df[col].fillna(mode_value)
                print(f"Filled missing values in '{col}' with mode: {mode_value}")
    
    # Remove duplicates
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        df = df.drop_duplicates()
        print(f"Removed {duplicates} duplicate rows.")
    
    print(f"Final dataset shape after handling missing values: {df.shape}")
    return df

def detect_outliers(df, columns=None, method='iqr'):
    """
    Detect outliers in numerical columns
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame to detect outliers in
    columns : list or None
        List of columns to check for outliers. If None, all numerical columns are checked.
    method : str
        Method to detect outliers ('iqr' or 'zscore')
        
    Returns:
    --------
    dict
        Dictionary with column names as keys and number of outliers as values
    """
    print("\n===== DETECTING OUTLIERS =====")
    
    if columns is None:
        columns = df.select_dtypes(include=['number']).columns
    
    outliers_count = {}
    
    for col in columns:
        if method == 'iqr':
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
        else:  # zscore
            from scipy import stats
            z_scores = np.abs(stats.zscore(df[col].dropna()))
            outliers = df[col][z_scores > 3]
        
        outliers_count[col] = len(outliers)
        print(f"Column '{col}': {len(outliers)} outliers detected ({len(outliers)/len(df)*100:.2f}%)")
        
        # Plot histogram with outlier boundaries if IQR method
        if method == 'iqr' and len(outliers) > 0:
            plt.figure(figsize=(10, 4))
            sns.histplot(df[col], kde=True)
            plt.axvline(x=lower_bound, color='r', linestyle='--', label=f'Lower bound: {lower_bound:.2f}')
            plt.axvline(x=upper_bound, color='r', linestyle='--', label=f'Upper bound: {upper_bound:.2f}')
            plt.title(f'Distribution of {col} with Outlier Boundaries')
            plt.legend()
            plt.show()
    
    return outliers_count

def handle_outliers(df, columns=None, method='iqr', strategy='cap'):
    """
    Handle outliers in numerical columns
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame to handle outliers in
    columns : list or None
        List of columns to handle outliers in. If None, all numerical columns are used.
    method : str
        Method to detect outliers ('iqr' or 'zscore')
    strategy : str
        Strategy to handle outliers ('cap', 'remove', or 'none')
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with handled outliers
    """
    print("\n===== HANDLING OUTLIERS =====")
    
    if strategy == 'none':
        print("No outlier handling requested.")
        return df
    
    if columns is None:
        columns = df.select_dtypes(include=['number']).columns
    
    df_processed = df.copy()
    
    for col in columns:
        if method == 'iqr':
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            if strategy == 'cap':
                # Cap the outliers
                df_processed[col] = df_processed[col].clip(lower=lower_bound, upper=upper_bound)
                print(f"Capped outliers in '{col}' to range [{lower_bound:.4f}, {upper_bound:.4f}]")
            elif strategy == 'remove':
                # Create a mask for outliers
                mask = (df_processed[col] >= lower_bound) & (df_processed[col] <= upper_bound)
                # Count outliers before removal
                outliers_count = (~mask).sum()
                if outliers_count > 0:
                    # Remove rows with outliers
                    df_processed = df_processed[mask]
                    print(f"Removed {outliers_count} rows with outliers in '{col}'")
        
        elif method == 'zscore':
            from scipy import stats
            z_scores = np.abs(stats.zscore(df[col].dropna()))
            mask = z_scores <= 3
            
            if strategy == 'cap':
                # Get the values at z-score = 3
                upper_bound = df[col][z_scores <= 3].max()
                lower_bound = df[col][z_scores <= 3].min()
                # Cap the outliers
                df_processed[col] = df_processed[col].clip(lower=lower_bound, upper=upper_bound)
                print(f"Capped outliers in '{col}' to range [{lower_bound:.4f}, {upper_bound:.4f}]")
            elif strategy == 'remove':
                # Count outliers before removal
                outliers_count = (~mask).sum()
                if outliers_count > 0:
                    # Remove rows with outliers
                    df_processed = df_processed[mask]
                    print(f"Removed {outliers_count} rows with outliers in '{col}'")
    
    print(f"Dataset shape after handling outliers: {df_processed.shape}")
    return df_processed

def create_target_variable(df):
    """
    Create the target variable for half-time draw prediction
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame to process
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with target variable
    """
    print("\n===== CREATING TARGET VARIABLE =====")
    
    # Drop specified columns if they still exist
    columns_to_drop = ['result', 'home_goals', 'away_goals', 'home_goalsht', 'away_goalsht']
    existing_columns = [col for col in columns_to_drop if col in df.columns]
    if existing_columns:
        df = df.drop(columns=existing_columns)
        print(f"Dropped columns: {existing_columns}")
    
    # Extract home and away goals from halftime_result
    if 'halftime_result' in df.columns:
        try:
            # Extract home and away goals from halftime_result
            df[['ht_home', 'ht_away']] = df['halftime_result'].str.extract(r'(\d+)\s*-\s*(\d+)').astype(int)
            
            # Create target variable
            df['half_time_draw'] = df.apply(lambda x: 1 if x['ht_home'] == x['ht_away'] else 0, axis=1)
            
            # Drop temporary columns
            df = df.drop(columns=['ht_home', 'ht_away'])
            
            # Drop halftime_result column
            df = df.drop(columns=['halftime_result'])
            print("Created target variable 'half_time_draw'")
            print("Dropped column: 'halftime_result'")
            
            # Print class distribution
            print("\nClass distribution:")
            print(df['half_time_draw'].value_counts())
            print(f"Percentage of draws: {df['half_time_draw'].mean() * 100:.2f}%")
        except Exception as e:
            print(f"Error creating target variable: {e}")
    else:
        print("Warning: 'halftime_result' column not found. Cannot create target variable.")
    
    return df

def visualize_data(df, save_dir=None):
    """
    Visualize the data
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame to visualize
    save_dir : str or None
        Directory to save visualizations, if None, visualizations are not saved
        
    Returns:
    --------
    None
    """
    print("\n===== DATA VISUALIZATION =====")
    
    # Import save_visualization function from util
    from util import save_visualization
    
    # Class distribution
    if 'half_time_draw' in df.columns:
        fig = plt.figure(figsize=(8, 5))
        sns.countplot(x='half_time_draw', data=df)
        plt.title('Half-Time Draw Distribution')
        plt.xlabel('Half-Time Draw (1=Yes, 0=No)')
        plt.ylabel('Count')
        if save_dir:
            save_visualization(fig, 'class_distribution.png', save_dir)
        plt.show()
    
    # Correlation heatmap
    fig = plt.figure(figsize=(12, 10))
    numerical_df = df.select_dtypes(include=['number'])
    correlation_matrix = numerical_df.corr()
    mask = np.triu(correlation_matrix)
    sns.heatmap(correlation_matrix, annot=False, mask=mask, cmap='coolwarm', 
                linewidths=0.5, vmin=-1, vmax=1)
    plt.title('Correlation Heatmap')
    plt.tight_layout()
    if save_dir:
        save_visualization(fig, 'correlation_heatmap.png', save_dir)
    plt.show()
    
    # Top correlated features with target
    if 'half_time_draw' in numerical_df.columns:
        n = min(20, len(numerical_df.columns) - 1)  # Top 20 or less if fewer columns
        target_correlations = correlation_matrix['half_time_draw'].drop('half_time_draw')
        top_correlated_features = target_correlations.abs().sort_values(ascending=False).head(n)
        
        fig = plt.figure(figsize=(10, 6))
        sns.barplot(x=top_correlated_features.values, y=top_correlated_features.index)
        plt.title(f'Top {n} Features Correlated with Half-Time Draw')
        plt.xlabel('Absolute Correlation')
        plt.tight_layout()
        if save_dir:
            save_visualization(fig, 'top_correlated_features.png', save_dir)
        plt.show()
        
        # Print top correlated features
        print(f"\nTop {n} features correlated with 'half_time_draw':")
        for feature, corr in target_correlations.abs().sort_values(ascending=False).head(n).items():
            print(f"{feature}: {corr:.4f} ({target_correlations[feature]:.4f})")
    
    # Distribution of numerical features
    numerical_cols = df.select_dtypes(include=['number']).columns
    num_cols = len(numerical_cols)
    if num_cols > 0:
        # Calculate number of rows and columns for subplots
        n_rows = (num_cols + 2) // 3  # Ceiling division by 3
        n_cols = min(3, num_cols)
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 4))
        axes = axes.flatten() if n_rows > 1 or n_cols > 1 else [axes]
        
        for i, col in enumerate(numerical_cols):
            if i < len(axes):
                sns.histplot(df[col], kde=True, ax=axes[i])
                axes[i].set_title(f'Distribution of {col}')
        
        # Hide unused subplots
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)
        
        plt.tight_layout()
        if save_dir:
            save_visualization(fig, 'numerical_distributions.png', save_dir)
        plt.show()
    
    # Scatter plots of top correlated features with target
    if 'half_time_draw' in numerical_df.columns:
        top_features = target_correlations.abs().sort_values(ascending=False).head(5).index
        
        if len(top_features) > 0:
            fig, axes = plt.subplots(len(top_features), 1, figsize=(10, 4 * len(top_features)))
            axes = [axes] if len(top_features) == 1 else axes
            
            for i, feature in enumerate(top_features):
                sns.boxplot(x='half_time_draw', y=feature, data=df, ax=axes[i])
                axes[i].set_title(f'{feature} by Half-Time Draw')
                axes[i].set_xlabel('Half-Time Draw (1=Yes, 0=No)')
            
            plt.tight_layout()
            if save_dir:
                save_visualization(fig, 'feature_boxplots.png', save_dir)
            plt.show()

def select_features(X, y, method='mutual_info', k=10):
    """
    Select the best features for the model
    
    Parameters:
    -----------
    X : pd.DataFrame
        Feature matrix
    y : pd.Series
        Target variable
    method : str
        Feature selection method ('mutual_info', 'anova', or 'pca')
    k : int
        Number of features to select
        
    Returns:
    --------
    tuple
        (X_selected, selector) - Selected features and the selector object
    """
    print(f"\n===== FEATURE SELECTION ({method.upper()}) =====")
    
    k = min(k, X.shape[1])  # Ensure k is not larger than the number of features
    
    if method == 'mutual_info':
        selector = SelectKBest(mutual_info_classif, k=k)
        X_selected = selector.fit_transform(X, y)
        
        # Get selected feature names
        selected_features = X.columns[selector.get_support()]
        print(f"Selected {k} features using Mutual Information:")
        for i, feature in enumerate(selected_features):
            print(f"{i+1}. {feature} (Score: {selector.scores_[selector.get_support()][i]:.4f})")
    
    elif method == 'anova':
        selector = SelectKBest(f_classif, k=k)
        X_selected = selector.fit_transform(X, y)
        
        # Get selected feature names
        selected_features = X.columns[selector.get_support()]
        print(f"Selected {k} features using ANOVA F-value:")
        for i, feature in enumerate(selected_features):
            print(f"{i+1}. {feature} (Score: {selector.scores_[selector.get_support()][i]:.4f})")
    
    elif method == 'pca':
        # Standardize the data before PCA
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Apply PCA
        selector = PCA(n_components=k)
        X_selected = selector.fit_transform(X_scaled)
        
        # Print explained variance
        explained_variance = selector.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance)
        print(f"Selected {k} principal components:")
        for i, var in enumerate(explained_variance):
            print(f"PC{i+1}: {var:.4f} ({cumulative_variance[i]:.4f} cumulative)")
        
        # Plot explained variance
        plt.figure(figsize=(10, 5))
        plt.bar(range(1, k+1), explained_variance, alpha=0.5, label='Individual')
        plt.step(range(1, k+1), cumulative_variance, where='mid', label='Cumulative')
        plt.axhline(y=0.95, color='r', linestyle='--', label='95% Explained Variance')
        plt.xlabel('Principal Components')
        plt.ylabel('Explained Variance Ratio')
        plt.title('PCA Explained Variance')
        plt.legend()
        plt.tight_layout()
        plt.show()
    
    else:
        raise ValueError(f"Unknown feature selection method: {method}")
    
    return X_selected, selector

def scale_features(X_train, X_test, method='standard'):
    """
    Scale the features
    
    Parameters:
    -----------
    X_train : pd.DataFrame or np.ndarray
        Training feature matrix
    X_test : pd.DataFrame or np.ndarray
        Testing feature matrix
    method : str
        Scaling method ('standard' or 'minmax')
        
    Returns:
    --------
    tuple
        (X_train_scaled, X_test_scaled, scaler) - Scaled features and the scaler object
    """
    print(f"\n===== FEATURE SCALING ({method.upper()}) =====")
    
    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'minmax':
        scaler = MinMaxScaler()
    else:
        raise ValueError(f"Unknown scaling method: {method}")
    
    # Fit on training data and transform both training and test data
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"Features scaled using {method} scaling")
    
    return X_train_scaled, X_test_scaled, scaler

def save_preprocessor(scaler, selector, output_dir='models'):
    """
    Save the preprocessor objects
    
    Parameters:
    -----------
    scaler : object
        Scaler object
    selector : object
        Feature selector object
    output_dir : str
        Directory to save the preprocessor objects
        
    Returns:
    --------
    None
    """
    # Create directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save scaler
    scaler_path = os.path.join(output_dir, 'scaler.joblib')
    joblib.dump(scaler, scaler_path)
    print(f"Scaler saved to {scaler_path}")
    
    # Save selector
    selector_path = os.path.join(output_dir, 'selector.joblib')
    joblib.dump(selector, selector_path)
    print(f"Feature selector saved to {selector_path}")

def load_preprocessor(output_dir='models'):
    """
    Load the preprocessor objects
    
    Parameters:
    -----------
    output_dir : str
        Directory where the preprocessor objects are saved
        
    Returns:
    --------
    tuple
        (scaler, selector) - Loaded preprocessor objects
    """
    # Load scaler
    scaler_path = os.path.join(output_dir, 'scaler.joblib')
    scaler = joblib.load(scaler_path)
    print(f"Scaler loaded from {scaler_path}")
    
    # Load selector
    selector_path = os.path.join(output_dir, 'selector.joblib')
    selector = joblib.load(selector_path)
    print(f"Feature selector loaded from {selector_path}")
    
    return scaler, selector
