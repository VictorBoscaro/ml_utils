import pandas as pd
import numpy as np

def interpret_pca_components(pca, feature_names, top_n=5):
    # Create a DataFrame for the PCA components
    components_df = pd.DataFrame(pca.components_, columns=feature_names)
    
    for i, component in components_df.iterrows():
        print(f"Principal Component {i+1}")
        
        # Sort the features by the absolute value of their component loading
        sorted_features = component.abs().sort_values(ascending=False)
        
        # Print the top features for this component
        top_features = sorted_features[:top_n].index.tolist()
        print("Top Features:")
        
        for feature in top_features:
            print(f"{feature} (loading: {component[feature]:.4f})")
        
        print("\n")
