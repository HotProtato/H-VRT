from typing import Dict

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, TargetEncoder
from sklearn.tree import DecisionTreeRegressor
from sklearn.compose import ColumnTransformer


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
        self.encoder_ = None

    def fit(self, X):
        """
        Fits the partitioner to the data X.

        Args:
            X (pd.DataFrame or np.ndarray): The input data.

        Returns:
            self: The fitted partitioner instance.
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        continuous_features = X.select_dtypes(include=np.number)
        categorical_features = X.select_dtypes(exclude=np.number)

        # 1. Z-score normalization for continuous features
        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(continuous_features)

        # 2. Apply weights to continuous features
        if self.weights:
            for i, col in enumerate(continuous_features.columns):
                if col in self.weights:
                    X_scaled[:, i] *= self.weights[col]

        # 3. Create synthetic target 'y'
        y_synthetic = pd.Series(X_scaled.sum(axis=1), index=X.index)

        # 4. Target encode categorical features if they exist
        X_for_tree = X.copy()
        if not categorical_features.empty:
            categorical_feature_names = list(categorical_features.columns)
            self.encoder_ = ColumnTransformer(
                [ ('target_encoder', TargetEncoder(target_type='continuous'), categorical_feature_names)],
                remainder='passthrough'
            )
            X_for_tree = self.encoder_.fit_transform(X, y_synthetic)

        # 5. Train the Decision Tree Regressor
        self.tree_ = DecisionTreeRegressor(
            max_leaf_nodes=self.max_leaf_nodes,
            **self.tree_kwargs
        )
        self.tree_.fit(X_for_tree, y_synthetic)
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

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        X_for_apply = X.copy()
        if self.encoder_:
            X_for_apply = self.encoder_.transform(X_for_apply)

        return self.tree_.apply(X_for_apply)
