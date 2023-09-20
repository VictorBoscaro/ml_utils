Copy code
import pandas as pd
import numpy as np

def remove_highly_correlated_features(df, threshold=0.9):
    """
    Remove features that are highly correlated with others.
    
    Parameters:
        df (pd.DataFrame): The input dataframe with features.
        threshold (float): The correlation coefficient threshold for feature removal.
        
    Returns:
        pd.DataFrame: A dataframe with highly correlated features removed.
    """
    # Compute the correlation matrix
    corr_matrix = df.corr().abs()
    
    # Create a boolean mask to identify which features to keep
    mask = np.ones(corr_matrix.shape, dtype=bool)
    np.fill_diagonal(mask, 0)
    
    # Identify pairs of features that are highly correlated
    highly_correlated_pairs = []
    for i in range(corr_matrix.shape[0]):
        for j in range(i+1, corr_matrix.shape[1]):
            if corr_matrix.iloc[i, j] >= threshold:
                highly_correlated_pairs.append((corr_matrix.index[i], corr_matrix.columns[j]))
                
    # Identify features to remove
    features_to_remove = set()
    for feature1, feature2 in highly_correlated_pairs:
        if feature1 not in features_to_remove and feature2 not in features_to_remove:
            features_to_remove.add(feature2)  # Remove one of the correlated features
    
    # Remove the features
    df_filtered = df.drop(columns=list(features_to_remove))
    
    return df_filtered
