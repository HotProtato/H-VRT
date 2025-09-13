
import numpy as np
import pandas as pd

def calculate_feature_hhi_metric(X, labels):
    """
    Calculates the mean HHI of variance concentration per feature.

    For each feature, this metric measures how concentrated its internal variance is
    across the different clusters/partitions. A lower score is better, indicating
    that for an average feature, its variance is more evenly distributed amongst
    the partitions.

    Args:
        X (pd.DataFrame or np.ndarray): The input data used for partitioning.
        labels (np.ndarray): The partition labels for each sample in X.

    Returns:
        float: The mean HHI score across all features.
    """
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)

    unique_labels, counts = np.unique(labels, return_counts=True)
    n_clusters = len(unique_labels)
    if n_clusters <= 1:
        return 1.0  # Max concentration if only one cluster

    feature_hhi_scores = []
    # For each feature...
    for col in X.columns:
        feature_data = X[col]
        
        # Calculate variance of the feature within each cluster
        cluster_variances = []
        for i, k in enumerate(unique_labels):
            # Variance is only defined for clusters with more than 1 member
            if counts[i] > 1:
                cluster_variances.append(feature_data[labels == k].var(ddof=0))
            else:
                cluster_variances.append(0)
        
        cluster_variances = np.nan_to_num(cluster_variances)
        total_variance_sum = np.sum(cluster_variances)

        if total_variance_sum == 0:
            # If all cluster variances are 0, variance is perfectly spread.
            # HHI for perfect equality is 1/n.
            hhi = 1.0 / n_clusters if n_clusters > 0 else 1.0
        else:
            # Calculate proportions
            proportions = cluster_variances / total_variance_sum
            # Calculate HHI for the feature
            hhi = np.sum(proportions**2)
        
        feature_hhi_scores.append(hhi)

    # Return the average HHI across all features
    return np.mean(feature_hhi_scores)
