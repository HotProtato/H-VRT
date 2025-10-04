# H-VRT: Hybrid Variance-Reduction Tree Partitioner

[![PyPI version](https://badge.fury.io/py/hvrt-partitioner.svg)](https://badge.fury.io/py/hvrt-partitioner)

A fast, scalable algorithm for creating fine-grained data partitions, optimized for speed on large datasets. This tool is ideal for pre-processing before fitting local models (e.g., linear regression) on distinct segments of your data, a technique often used for piece-wise approximations of complex, non-linear relationships.

## Key Features

- **Extremely Fast:** Orders of magnitudes faster than KMeans for creating a large number of partitions.
- **Scalable:** Training time scales efficiently as the desired number of partitions increases.
- **Configurable:** Allows for custom scaling methods to be used in the partitioning process.
- **Analysis Tools:** Includes a powerful `PartitionProfiler` to analyze, visualize, and save reports on partition quality and effects.

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

## API and Parameters

The `HVRTPartitioner` is configured upon initialization. Here are the key parameters:

-   `max_leaf_nodes` (int, optional): The maximum number of partitions to create. This directly controls the granularity of the partitioning.
-   `weights` (Dict[str, float], optional): A dictionary to increase or reduce the impact of specific features on the partitioning process. Keys are feature names, and values are the corresponding weights.
-   `impute` (bool, default=`True`): If `True`, missing values are imputed using the mean for numeric features and the most frequent value for categorical features. If `False`, NaNs are preserved, allowing for custom imputation strategies (e.g., localized imputation within partitions) after the data is labeled.
-   `category_encoding` (str, default=`'ohe'`): The encoding strategy for categorical features in your input data `X`.
    -   `'ohe'`: Uses `OneHotEncoder`. This is generally better for accuracy but can lead to larger, more complex trees.
    -   `'target'`: Uses `TargetEncoder`. This results in smaller, less complex trees but may offer less predictive accuracy.
-   `target_categories` (bool, default=`False`): Set this to `True` if your target variable `y` is categorical. The partitioner will one-hot encode `y` and use it to guide the partitioning, optimizing for class purity.
-   `scaler` (TransformerMixin, optional): A scikit-learn compatible scaler for the target generation process. Defaults to `StandardScaler`.
-   `min_variance_reduction` (float, default=`0.01`): The minimum percentage of average variance that a split must reduce to be considered.

**Note on Compatibility:** Using `category_encoding='target'` is incompatible with `target_categories=True`. The partitioner will raise a `ValueError` if these settings are used together.

## Analyzing Partitions

The library includes powerful tools for understanding the quality and effects of your partitions.

### High-Level Summary: `PartitionProfiler`

The `PartitionProfiler` provides a comprehensive overview of your partitions. It generates summary tables, creates insightful visualizations (like the Binned Violin Plot for large partition counts), and saves all artifacts to disk.

```python
from src.metrics.partition_profiler import PartitionProfiler
import pandas as pd

profiler = PartitionProfiler(
    data=X_sample,
    partition_labels=pd.Series(partition_labels, index=X_sample.index),
    output_path="my_profiler_output" # Optional
)
profiler.run_profiling()
```

### Low-Level Metrics: `full_report`

For programmatic access to detailed metrics, `full_report` returns a dictionary of dataclass objects containing rich statistical information about the variance and value-span for each feature.

```python
from src.metrics.metrics import full_report

report = full_report(X_sample, partition_labels)
# Access reports for a specific feature, e.g., 'feat_0'
feature_0_report = report['feat_0']
print(f"Feature 0 Variance HHI: {feature_0_report.variance_report.hhi}")
```

**Note:** A superior approach to visually reviewing partitions is currently being developed and will be integrated in a future release.

## How It Works

The core heuristic is simple yet effective:

1.  **Synthesize Target:** A multi-output synthetic target is created. By default, this consists of the z-score scaled values of the input features `X`. If a target variable `y` is provided, it is also scaled and appended to this synthetic target.
2.  **Encode Features:**
    -   **Continuous Features:** Passed through directly or imputed if `impute=True`.
    -   **Categorical Features:** Encoded based on the `category_encoding` parameter (`'ohe'` or `'target'`).
3.  **Fit Tree:** A `DecisionTreeRegressor` is trained to predict the multi-output synthetic target using the encoded features. The `max_leaf_nodes` parameter controls the tree's granularity.
4.  **Extract Partitions:** The terminal leaves of the fitted tree serve as the final partitions.

## A point of clarity

I have decided to allow for a list of y-values, where the logic is if you're wanting to have a feature contribute to the formation of the tree, but you cannot guarentee it will be present during inference or expect a lot of Na values, use it as y.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Future

1. I am looking to provide support for Fsspec and potentially SQLAlclhemy for storing and populating metrics.
2. Another framework that specifically uses the sum of z-scores as a synthetic y, deliberately capturing overfitting will be in the works. The goal if this framework is instead, to objectively capture feature complexity for a given system.
3. A visual reconstruction of features given every other feature, where stronger deviations indicate greater complexity or greater influence from other features will be made. This will require PyTorch.
4. **Advanced Inference-Time Imputation:** An advanced imputation strategy is being explored for handling missing values in new data. When a sample is missing a value required for a split, instead of relying on a simple default, the algorithm could look ahead. By analyzing the sample's other known features, it could calculate the probability of the sample belonging to each possible downstream partition. If the probabilistic analysis is inconclusive, it would fall back to a standard global mean imputation. This would enable more accurate and robust partitioning for incomplete data.