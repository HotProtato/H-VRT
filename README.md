

# H-VRT: Hybrid Variance-Reduction Tree Partitioner

[![PyPI version](https://badge.fury.io/py/hvrt-partitioner.svg)](https://badge.fury.io/py/hvrt-partitioner)

A fast, scalable algorithm for creating fine-grained data partitions, optimized for speed on large datasets. This tool is ideal for pre-processing before fitting local models (e.g., linear regression) on distinct segments of your data, a technique often used for piece-wise approximations of complex, non-linear relationships.

## Key Features

- **Extremely Fast:** Orders of magnitude faster than KMeans for creating a large number of partitions.
- **Scalable:** Training time scales efficiently as the desired number of partitions increases.
- **Configurable:** Allows for custom scaling methods to be used in the partitioning process.
- **Analysis Tools:** Includes a powerful `PartitionProfiler` and `DistributionAnalyzer` to analyze, visualize, and save reports on partition quality and effects.

## Installation

From PyPI:
```bash
pip install hvrt-partitioner
```

To include plotting and visualization capabilities, install with the `[viz]` extra:
```bash
pip install hvrt-partitioner[viz]
```

## Quick Start

```python
import numpy as np
import pandas as pd
from hvrt import HVRTPartitioner

# 1. Generate sample data
X_sample = pd.DataFrame(np.random.rand(10000, 10), columns=[f'feat_{i}' for i in range(10)])

# 2. Initialize and fit the partitioner
partitioner = HVRTPartitioner(max_leaf_nodes=200)
partitioner.fit(X_sample)

# 3. Get the partition labels for each sample
partition_labels = partitioner.get_partitions(X_sample)

print(f"Successfully assigned {len(X_sample)} samples to {len(np.unique(partition_labels))} partitions.")
```

## Analyzing Partitions

The library includes powerful tools for understanding the quality and effects of your partitions.

### High-Level Summary: `PartitionProfiler`

The `PartitionProfiler` provides a comprehensive overview of your partitions. It generates summary tables, creates insightful visualizations (like the Binned Violin Plot for large partition counts), and saves all artifacts to disk.

```python
from hvrt import PartitionProfiler

profiler = PartitionProfiler(
    data=X_sample,
    partition_labels=pd.Series(partition_labels, index=X_sample.index),
    output_path="my_profiler_output" # Optional
)
profiler.run_profiling()
```

### Distributional Analysis: `DistributionAnalyzer`

This tool answers the question: **"How much does partitioning distort the original shape of my data?"** It generates an overlaid plot comparing the original distribution of a feature to the reconstructed distribution from the partitions.

![Distribution Analyzer Plot](sample/distribution_analyzer_example.png)

*In the plot above, the reconstructed distribution (red dashed line) closely tracks the original (blue line), indicating that the partitioning process has successfully preserved the feature's overall structure.*

```python
from hvrt import DistributionAnalyzer

analyzer = DistributionAnalyzer(data=X_sample, partition_labels=pd.Series(partition_labels, index=X_sample.index))
analyzer.fit()
analyzer.plot_comparison(
    feature_name='feat_0',
    save_path='analyzer_output/feat_0_comparison.png' # Optional
)
```

### Low-Level Metrics: `full_report`

For programmatic access to detailed metrics, `full_report` returns a dictionary of dataclass objects containing rich statistical information about the variance and value-span for each feature.

*Note: A previous metric `calculate_feature_hhi_metric` is now deprecated in favor of the more comprehensive `full_report` function and the higher-level analysis tools.*

## How It Works

The core heuristic is simple yet effective:

1.  **Scale:** Data is scaled for each feature. By default, this is a Z-score transformation, but any scikit-learn compatible scaler can be provided.
2.  **Synthesize Target:** A new target vector (`y`) is created by summing the scaled features for each sample.
3.  **Fit Tree:** A `DecisionTreeRegressor` is trained to predict this synthetic `y`. The `max_leaf_nodes` parameter controls the tree's granularity.
4.  **Extract Partitions:** The terminal leaves of the fitted tree serve as the final partitions.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
