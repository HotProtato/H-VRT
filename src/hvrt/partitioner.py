from typing import Dict

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, TargetEncoder
from sklearn.tree import DecisionTreeRegressor
from sklearn.compose import ColumnTransformer
from sklearn.base import TransformerMixin, clone


class HVRTPartitioner:
    """
    A fast, scalable algorithm for creating data partitions.

    It works by training a decision tree on a synthetic multi-output target.
    By default, this target is derived from the scaled values of the input features,
    creating an unsupervised partitioning.

    If a target variable `y` is provided, it is added to the synthetic target,
    making the partitions sensitive to the target variable (semi-supervised partitioning).
    This is useful for creating segments where a simpler model (e.g., linear regression)
    can be fitted locally.
    """
    def __init__(self, max_leaf_nodes=None, weights: Dict[str, float]=None, scaler: TransformerMixin=StandardScaler(), min_variance_reduction: float=0.01, **tree_kwargs):
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
        self.scaler_ = clone(scaler)
        self.encoder_ = None
        self.min_var_reduction = min_variance_reduction
        # min_impurity_decrease should only be used via min_var_reduction which makes the value relative as a % of the sum of y.
        self.tree_kwargs = {param: value for param, value in tree_kwargs.items() if param != "min_impurity_decrease"}

    def fit(self, X, y=None):
        """
        Fits the partitioner to the data X, optionally using a target variable y
        to influence the partitioning.

        Args:
            X (pd.DataFrame or np.ndarray): The input feature data.
            y (pd.Series or np.ndarray, optional): The target variable. If provided,
                it will be used to create target-aware partitions. Defaults to None.

        Returns:
            self: The fitted partitioner instance.
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        continuous_features = X.select_dtypes(include=np.number).copy()
        categorical_features = X.select_dtypes(exclude=np.number)

        if y is not None:
            if not np.issubdtype(y.dtype, np.number):
                raise ValueError("The target y must be numeric.")
            y_name = y.name if hasattr(y, 'name') and y.name else "target"
            continuous_features[y_name] = y

        # 1. Normalization for continuous features
        X_scaled = self.scaler_.fit_transform(continuous_features)

        # 2. Apply weights to continuous features
        if self.weights:
            for i, col in enumerate(continuous_features.columns):
                if col in self.weights:
                    X_scaled[:, i] *= self.weights[col]

        # 3. The scaled continuous features become the multi-output target 'y'
        y_multi_output = X_scaled

        # 4. Target encode categorical features if they exist
        X_for_tree = X.copy()
        if not categorical_features.empty:
            # TargetEncoder requires a 1D target, so we use the mean of our multi-output y
            y_for_encoder = pd.Series(y_multi_output.mean(axis=1), index=X.index)

            categorical_feature_names = list(categorical_features.columns)
            self.encoder_ = ColumnTransformer(
                [('target_encoder', TargetEncoder(target_type='continuous'), categorical_feature_names)],
                remainder='passthrough'
            )
            X_for_tree = self.encoder_.fit_transform(X, y_for_encoder)

        # 5. Calculate minimum impurity reduction from %.
        # Base the calculation on the sum of squares of the multi-output target.
        min_impurity_reduction = np.sum(y_multi_output**2) * self.min_var_reduction

        # 6. Train the Decision Tree Regressor with the multi-output target
        self.tree_ = DecisionTreeRegressor(
            max_leaf_nodes=self.max_leaf_nodes,
            min_impurity_decrease=min_impurity_reduction,
            **self.tree_kwargs
        )
        self.tree_.fit(X_for_tree, y_multi_output)
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

    def fit_predict(self, X):
        if self.tree_:
            return self.get_partitions(X)
        self.fit(X)
        return self.get_partitions(X)