
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor


class HVRT_Partitioner:
    """
    A fast, scalable algorithm for creating data partitions by training a decision tree
    on a synthetic target variable derived from the z-scores of the input features.

    This method is designed for creating a large number of fine-grained partitions
    ("micro-approximations") and is optimized for speed at scale.
    """
    def __init__(self, max_leaf_nodes=None):
        """
        Initializes the HVRT_Partitioner.

        Args:
            max_leaf_nodes (int, optional): The maximum number of leaf nodes (partitions)
                                          for the decision tree. Defaults to None (unlimited).
        """
        self.max_leaf_nodes = max_leaf_nodes
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
        
        # 2. Create the synthetic target 'y' by summing the z-scores
        y_synthetic = X_scaled.sum(axis=1)

        # 3. Train the Decision Tree Regressor to create partitions
        self.tree_ = DecisionTreeRegressor(
            max_leaf_nodes=self.max_leaf_nodes,
            random_state=42
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
