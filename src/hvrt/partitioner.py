from typing import Dict

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor


class HVRTPartitioner:
    """
    A fast, scalable algorithm for creating data partitions by training a decision tree
    on a synthetic target variable derived from the z-scores of the input features.

    This method is designed for creating a large number of fine-grained partitions
    ("micro-approximations") and is optimized for speed at scale.
    """
    def __init__(self, max_leaf_nodes=None, weights: Dict[str, float]=None, **tree_kwargs):
        """
       Initializes the HVRTPartitioner with the specified parameters.

        :param max_leaf_nodes: The number of partitions to create.
        :param weights: Increase or reduce the impact of each feature on the partitioning through weights.
        :param tree_kwargs: Additional arguments to be passed to the scikit-learn Decision Tree Regressor.
        """
        self.max_leaf_nodes = max_leaf_nodes
        self.weights = weights
        self.tree_kwargs = tree_kwargs
        self.tree_kwargs.setdefault("random_state", 42)
        self.tree_ = None
        self.scaler_ = None

    def fit(self, X):
        """
        Fits the partitioner to the data X.

        Args:
            X (pd.DataFrame or np.ndarray): The input data with continuous features.

        Returns:
            self: The fitted partitioner instance.
        """
        if not isinstance(X, (pd.DataFrame, np.ndarray)):
            raise ValueError("Input data X must be a pandas DataFrame or a numpy array.")

        # 1. Z-score normalization
        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X)

        # 2. Update values based on weights if applicable
        if self.weights:
            for col in self.weights:
                X_scaled[:, col] *= self.weights[col]
        
        # 3. Create the synthetic target 'y' by summing the z-scores
        y_synthetic = X_scaled.sum(axis=1)

        # 4. Train the Decision Tree Regressor to create partitions
        self.tree_ = DecisionTreeRegressor(
            max_leaf_nodes=self.max_leaf_nodes,
            **self.tree_kwargs
        )
        self.tree_.fit(X, y_synthetic)
        return self

    def get_partitions(self, X):
        """
        Assigns each sample in X to a partition (leaf node).

        Args:
            X (pd.DataFrame or np.ndarray): The input data.

        Returns:
            np.ndarray: An array of integers where each integer represents the
                        ID of the leaf node (partition) each sample belongs to.
        """
        if self.tree_ is None:
            raise RuntimeError("The partitioner has not been fitted yet. Call fit() first.")
        return self.tree_.apply(X)
