
# H-VRT: Hybrid Variance-Reduction Tree Partitioner

[![PyPI version](https://badge.fury.io/py/hvrt-partitioner.svg)](https://badge.fury.io/py/hvrt-partitioner) <!--- Placeholder badge -->

A fast, scalable algorithm for creating fine-grained data partitions, optimized for speed on large datasets. This tool is ideal for pre-processing before fitting local models (e.g., linear regression) on distinct segments of your data, a technique often used for piece-wise approximations of complex, non-linear relationships.

The `HVRTPartitioner` uses a novel heuristic to generate partitions: it trains a `DecisionTreeRegressor` on a synthetic target variable created by summing the z-scores of the input features. This approach avoids the expensive iterative process of traditional clustering algorithms like KMeans, leading to a significant speed advantage at high partition counts.

## Key Features

- **Extremely Fast:** Orders of magnitude faster than KMeans for creating a large number of partitions on datasets with millions of samples.
- **Scalable:** Training time scales efficiently as the desired number of partitions increases.
- **Simple:** Implements a straightforward, non-iterative partitioning logic.
- **Analysis Tools:** Includes a powerful `PartitionProfiler` to analyze, visualize, and save reports on partition quality.

## Installation

For the core functionality:
```bash
pip install .
```

To include plotting and visualization capabilities, install with the `[viz]` extra:
```bash
pip install .[viz]
```

## Quick Start

Here is a simple example of how to use the `HVRTPartitioner` to partition a dataset into 200 segments.

```python
import numpy as np
import pandas as pd
from hvrt import HVRTPartitioner

# 1. Generate sample data
X_sample = pd.DataFrame(np.random.rand(10000, 10), columns=[f'feat_{i}' for i in range(10)])

# 2. Initialize and fit the partitioner
# We want to create a maximum of 200 partitions
partitioner = HVRTPartitioner(max_leaf_nodes=200)
partitioner.fit(X_sample)

# 3. Get the partition labels for each sample
# The output is an array of integers, where each integer is a partition ID.
partition_labels = partitioner.get_partitions(X_sample)

print(f"Successfully assigned {len(X_sample)} samples to {len(np.unique(partition_labels))} partitions.")
print("First 10 partition labels:", partition_labels[:10])
```

## Analyzing Partitions

The library includes powerful tools for understanding the quality of your partitions.

### High-Level Analysis: `PartitionProfiler`

The `PartitionProfiler` is the easiest way to get a comprehensive overview of your partitions. It can generate summary tables, create insightful visualizations, and save all artifacts to disk.

```python
from hvrt import PartitionProfiler

# ... after getting partition_labels from the partitioner

# Initialize the profiler
profiler = PartitionProfiler(
    data=X_sample,
    partition_labels=pd.Series(partition_labels, index=X_sample.index),
    output_path="my_profiler_output" # Optional: saves all plots and data
)

# Run the full analysis
profiler.run_profiling()
```

This will print a summary table and generate several plots:
- A distribution of sample counts across partitions.
- For the top features (by HHI), it will generate distribution plots.
- If the number of partitions is large, it intelligently switches to a **Binned Violin Plot** to show the trend of a feature's distribution across the sorted partitions.

### Low-Level Analysis: `full_report`

For programmatic access to detailed metrics, you can use the `full_report` function. It returns a dictionary of dataclass objects containing rich statistical information about the variance and value-span for each feature.

```python
from hvrt import full_report

# ... after getting partition_labels

report = full_report(X_sample, partition_labels)

# Access detailed metrics for a specific feature
feature_a_variance_hhi = report['feat_0'].variance_report.hhi
print(f"\nVariance HHI for feat_0: {feature_a_variance_hhi:.4f}")
```

---

*Note: A previous metric `calculate_feature_hhi_metric` is now deprecated in favor of the more comprehensive `full_report` function and `PartitionProfiler` class.*

## How It Works

The core heuristic is simple yet effective:

1.  **Scale:** For each feature, the data is scaled. By default, this is a Z-score transformation, but any scikit-learn compatible scaler can be provided.
2.  **Synthesize Target:** A new, single target vector (`y`) is created by summing the scaled features for each sample. This vector represents a measure of each sample's combined deviation from the mean.
3.  **Fit Tree:** A standard `DecisionTreeRegressor` is trained to predict this synthetic `y` using the original features. The `max_leaf_nodes` parameter is used to control the granularity of the tree.
4.  **Extract Partitions:** The terminal leaves of the fitted tree serve as the final partitions. The `.get_partitions()` method simply returns the ID of the leaf node that each sample falls into.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
