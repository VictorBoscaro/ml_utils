import plotly.graph_objects as go
import numpy as np

# Your data and functions for calculating CIs
data = [25, 30, 35, 40, 45]

# Function to calculate CI using Z-score
def calculate_confidence_interval(data, confidence=0.95):
    sample_mean = np.mean(data)
    sample_size = len(data)
    standard_error = np.std(data, ddof=1) / np.sqrt(sample_size)
    z_score = 1.96  # for 95% confidence
    margin_of_error = z_score * standard_error
    lower_bound = sample_mean - margin_of_error
    upper_bound = sample_mean + margin_of_error
    return lower_bound, upper_bound

# Function to calculate CI using bootstrapping
def bootstrap_confidence_interval(data, num_bootstraps=1000, confidence=0.95):
    bootstrap_means = []
    for _ in range(num_bootstraps):
        bootstrap_sample = np.random.choice(data, size=len(data), replace=True)
        bootstrap_mean = np.mean(bootstrap_sample)
        bootstrap_means.append(bootstrap_mean)
    lower_bound = np.percentile(bootstrap_means, 2.5)
    upper_bound = np.percentile(bootstrap_means, 97.5)
    return lower_bound, upper_bound

# Calculate CIs
z_lower, z_upper = calculate_confidence_interval(data)
bootstrap_lower, bootstrap_upper = bootstrap_confidence_interval(data)

# Create the Plotly figure
fig = go.Figure()

# Add Z-score-based CI as horizontal bar
fig.add_trace(go.Scatter(
    x=[z_lower, z_upper],
    y=['Z-Score Method', 'Z-Score Method'],
    mode="lines+markers",
    name="95% CI (Z-Score)",
    line=dict(width=4)
))

# Add Bootstrap-based CI as horizontal bar
fig.add_trace(go.Scatter(
    x=[bootstrap_lower, bootstrap_upper],
    y=['Bootstrap Method', 'Bootstrap Method'],
    mode="lines+markers",
    name="95% CI (Bootstrap)",
    line=dict(width=4)
))

# Add sample mean for reference
sample_mean = np.mean(data)
fig.add_trace(go.Scatter(
    x=[sample_mean],
    y=['Sample Mean'],
    mode="markers",
    marker=dict(size=8),
    name="Sample Mean"
))

# Configure layout
fig.update_layout(
    title="Confidence Intervals Comparison",
    xaxis_title="Value",
    yaxis_title="Method",
    yaxis=dict(autorange="reversed")
)

# Show plot
fig.show()

def permutation_test(group1, group2, n_permutations=10000):
    """
    Perform a permutation test to compare means between two groups.
    
    Parameters:
    - group1: np.array, first group of data
    - group2: np.array, second group of data
    - n_permutations: int, number of permutations
    
    Returns:
    - p_value: float, p-value of the test
    """
    
    # Observed difference in means
    obs_diff = np.mean(group2) - np.mean(group1)
    
    # Combine the two datasets
    combined = np.concatenate([group1, group2])
    
    # Perform permutation test
    extreme_count = 0
    for _ in range(n_permutations):
        np.random.shuffle(combined)
        new_group1 = combined[:len(group1)]
        new_group2 = combined[len(group1):]
        new_diff = np.mean(new_group2) - np.mean(new_group1)
        
        if new_diff >= obs_diff:
            extreme_count += 1
    
    # Calculate p-value
    p_value = extreme_count / n_permutations
    return p_value
