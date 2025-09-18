from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.utils import deprecated

from .feature_data import FeatureReport, SpanReport, VarianceReport


@deprecated
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
    for col in X.select_dtypes(include=np.number).columns:
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
            hhi = np.sum(proportions ** 2)

        feature_hhi_scores.append(hhi)

    # Return the average HHI across all features
    return np.mean(feature_hhi_scores)


def _get_partition_stats(X, labels, col, unique_labels, counts) -> Tuple[np.ndarray, np.ndarray]:
    """Calculates the variance and span of a feature within each partition in a single pass."""
    feature_data = X[col]
    variance_list = []
    span_list = []

    for i, k in enumerate(unique_labels):
        if counts[i] > 1:
            partition_data = feature_data[labels == k]
            variance_list.append(partition_data.var(ddof=0))
            span_list.append(partition_data.max() - partition_data.min())
        else:
            variance_list.append(0)
            span_list.append(0)

    return np.nan_to_num(np.array(variance_list)), np.nan_to_num(np.array(span_list))


def _check_params(X, labels):
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)
    unique_labels, counts = np.unique(labels, return_counts=True)
    n_clusters = len(unique_labels)
    if n_clusters <= 1:
        raise ValueError("At least 2 leaf nodes expected.")
    return X, unique_labels, counts, n_clusters


def full_report(
        X: pd.DataFrame or np.ndarray,
        labels: list[int],
        variance_proportions: bool = True,
        span_proportions: bool = True,
) -> Dict[str, FeatureReport]:
    """
    Generates a comprehensive, performant report on the quality of the partitions and local performance of each feature.
    :param X: Data, indexed in line with the labels.
    :param labels: The labels from partitioner, indexed in line with the data.
    :param variance_proportions: True if the variance data should be proportional, false if raw.
    :param span_proportions: True if the span data should be proportional, false if raw.
    :return: A dictionary of feature reports, indexed by feature name.
    """
    X, unique_labels, counts, n_clusters = _check_params(X, labels)
    # Select only continuous (numeric) features for reporting
    continuous_features = X.select_dtypes(include=np.number)

    feature_report_data: Dict[str, FeatureReport] = {}

    for col in continuous_features.columns:
        report = FeatureReport(name=col)

        # Efficiently get both stats in one pass
        raw_variances, raw_spans = _get_partition_stats(X, labels, col, unique_labels, counts)

        # --- Variance Report ---
        total_variance = raw_variances.sum()
        if total_variance == 0:
            var_hhi = 0.0
            var_dist = np.zeros(n_clusters)
        else:
            proportional_variances = raw_variances / total_variance
            var_hhi = np.sum(proportional_variances ** 2)
            var_dist = proportional_variances if variance_proportions else raw_variances

        report.variance_report = VarianceReport(
            is_proportional=variance_proportions,
            min_variance=np.min(var_dist),
            max_variance=np.max(var_dist),
            mean_variance=np.mean(var_dist),
            median_variance=np.median(var_dist),
            hhi=var_hhi,
            distribution=var_dist,
            histogram=np.histogram(var_dist) if var_dist.sum() > 0 else (np.array([]), np.array([]))
        )

        # --- Span Report ---
        total_span = raw_spans.sum()
        if total_span == 0:
            span_hhi = 0.0
            span_dist = np.zeros(n_clusters)
        else:
            proportional_spans = raw_spans / total_span
            span_hhi = np.sum(proportional_spans ** 2)
            span_dist = proportional_spans if span_proportions else raw_spans

        report.span_report = SpanReport(
            is_proportional=span_proportions,
            min_span=np.min(span_dist),
            max_span=np.max(span_dist),
            mean_span=np.mean(span_dist),
            median_span=np.median(span_dist),
            hhi=span_hhi,
            distribution=span_dist,
            histogram=np.histogram(span_dist) if span_dist.sum() > 0 else (np.array([]), np.array([]))
        )

        feature_report_data[col] = report

    return feature_report_data