
# H-VRT: Hybrid Variance-Reduction Tree Partitioner

[![PyPI version](https://badge.fury.io/py/hvrt-partitioner.svg)](https://badge.fury.io/py/hvrt-partitioner) <!--- Placeholder badge -->

A fast, scalable algorithm for creating fine-grained data partitions, optimized for speed on large datasets. This tool is ideal for pre-processing before fitting local models (e.g., linear regression) on distinct segments of your data, a technique often used for piece-wise approximations of complex, non-linear relationships.

The `HVRT_Partitioner` uses a novel heuristic to generate partitions: it trains a `DecisionTreeRegressor` on a synthetic target variable created by summing the z-scores of the input features. This approach avoids the expensive iterative process of traditional clustering algorithms like KMeans, leading to a significant speed advantage at high partition counts.

## Key Features

- **Extremely Fast:** Orders of magnitude faster than KMeans for creating a large number of partitions on datasets with millions of samples.
- **Scalable:** Training time scales efficiently as the desired number of partitions increases.
- **Simple:** Implements a straightforward, non-iterative partitioning logic.
- **Enabler for Local Models:** Perfectly suited for the first stage of a "divide and conquer" modeling strategy, such as a Mixture of Experts or partitioned regression framework.

## Installation

```bash
pip install hvrt-partitioner
```

## Quick Start

Here is a simple example of how to use the `HVRTPartitioner` to partition a dataset into 200 segments.

```python
import numpy as np
from hvrt import HVRTPartitioner

# 1. Generate sample data
X_sample = np.random.rand(10000, 10)

# 2. Initialize and fit the partitioner
# We want to create a maximum of 200 partitions
partitioner = HVRT_Partitioner(max_leaf_nodes=200)
partitioner.fit(X_sample)

# 3. Get the partition labels for each sample
# The output is an array of integers, where each integer is a partition ID.
partition_labels = partitioner.get_partitions(X_sample)

print(f"Successfully assigned {len(X_sample)} samples to {len(np.unique(partition_labels))} partitions.")
print("First 10 partition labels:", partition_labels[:10])
```

## Evaluating Partitions: The Feature Variance HHI

This package also includes a custom evaluation metric, `calculate_feature_hhi_metric`, which was designed to measure the quality of partitions created by `H-VRT`.

The metric calculates a Herfindahl-Hirschman Index (HHI) for each feature, measuring how concentrated its variance is across the different partitions. It then returns the mean HHI score across all features.

**A lower score is better.** A low score indicates that, on average, each feature has its internal variance spread evenly across the partitions, which is the primary goal of this algorithm.

### Usage

```python
from hvrt import calculate_feature_hhi_metric

# ... after fitting the partitioner and getting labels

# 4. Evaluate the quality of the partitions
hhi_score = calculate_feature_hhi_metric(X_sample, partition_labels)
# NOTE: This method is now deprecated in preference of "full_report".

print(f"\nFeature Variance HHI Score (lower is better): {hhi_score:.6f}")
```

## How It Works

The core heuristic is simple yet effective:

1.  **Standardize:** For each feature, the data is standardized using a Z-score transformation (`(value - mean) / std_dev`).
2.  **Synthesize Target:** A new, single target vector (`y`) is created by summing the z-scores across all features for each sample. This vector represents a measure of each sample's combined deviation from the mean.
3.  **Category Encoding:** Target encoding is used for all categorical values against the synthetic y to assist in forming the tree.
4.  **Fit Tree:** A standard `DecisionTreeRegressor` is trained to predict this synthetic `y` using the original features. The `max_leaf_nodes` parameter is used to control the granularity of the tree.
5.  **Extract Partitions:** The terminal leaves of the fitted tree serve as the final partitions. The `.get_partitions()` method simply returns the ID of the leaf node that each sample falls into.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
