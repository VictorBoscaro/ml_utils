from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Function to scale the data
def scale_data(df):
    scaler = StandardScaler()
    return scaler.fit_transform(df)

# Function to perform PCA
def perform_pca(scaled_data, n_components=2):
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(scaled_data)
    return pca, pca_result

# Function to create a DataFrame from PCA results
def create_pca_df(pca_result, n_components=2):
    columns = [f"PC{i+1}" for i in range(n_components)]
    return pd.DataFrame(data=pca_result, columns=columns)

# Function to get explained variance
def get_explained_variance(pca):
    return pca.explained_variance_ratio_

# Function to create a biplot
def create_biplot(score, coeff, labels):
    plt.figure(figsize=(12, 8))
    plt.scatter(score[:, 0], score[:, 1])
    for i in range(len(coeff[:, 0])):
        plt.arrow(0, 0, coeff[i, 0], coeff[i, 1], color='r', alpha=0.5)
        plt.text(coeff[i, 0]*1.15, coeff[i, 1]*1.15, labels[i], color='g', ha='center', va='center')

# Function to run the entire PCA process and visualization
def run_pca_pipeline(df, n_components=2):
    # Step 2: Preprocessing
    scaled_data = scale_data(df)
    
    # Step 3: Perform PCA
    pca, pca_result = perform_pca(scaled_data, n_components)
    
    # Step 4: Create DataFrame
    pca_df = create_pca_df(pca_result, n_components)
    
    # Step 5: Explained Variance
    explained_variance = get_explained_variance(pca)
    print(f"Explained Variance per component: {explained_variance}")
    
    # Step 6: Create Biplot
    create_biplot(pca_df.values, pca.components_, df.columns)
    plt.xlabel(f'PC1 - {explained_variance[0]*100:.2f}%')
    plt.ylabel(f'PC2 - {explained_variance[1]*100:.2f}%')
    plt.show()
