from typing import Dict, Union, List

import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin, clone, BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, TargetEncoder
from sklearn.tree import DecisionTreeRegressor


class NanTolerantScaler(BaseEstimator, TransformerMixin):
    """
    A scaler that computes z-scores while ignoring NaNs.
    NaNs in the input are preserved in the output.
    """
    def fit(self, X, y=None):
        self.mean_ = np.nanmean(X, axis=0)
        self.scale_ = np.nanstd(X, axis=0)
        # Handle columns with zero std dev (all constant values)
        self.scale_[self.scale_ == 0] = 1
        return self

    def transform(self, X):
        # Ensure X is a numpy array for broadcasting
        X_np = np.asarray(X)
        return (X_np - self.mean_) / self.scale_

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)


class HVRTPartitioner:
    """
    A fast, scalable algorithm for creating data partitions.

    It works by training a decision tree on a synthetic multi-output target.
    By default, this target is derived from the scaled values of the input features,
    creating an unsupervised partitioning.

    If a target variable `y` (Series or DataFrame) is provided, it is added to the
    synthetic target, making the partitions sensitive to the target variable(s)
    (semi-supervised partitioning).
    """
    def __init__(self, max_leaf_nodes=None, weights: Dict[str, float]=None, scaler: TransformerMixin=StandardScaler(), min_impurity_reduction: float=0.01, impute: bool = True, category_encoding: str = 'ohe', target_categories: bool = False, categorical_features: List[str] = None, **tree_kwargs):
        """
        Initializes the HVRTPartitioner with the specified parameters.

        :param max_leaf_nodes: The number of partitions to create.
        :param weights: Increase or reduce the impact of each feature on the partitioning through weights.
        :param scaler: A scikit-learn compatible scaler for the target generation. Note: ignored if `impute=False`.
        :param min_impurity_reduction: The minimum percentage of average impurity that a split must reduce.
        :param impute: If True (default), missing values are imputed with the mean. If False, NaNs are preserved, allowing for custom imputation strategies post-partitioning.
        :param category_encoding: The encoding method for categorical features in X. Can be 'ohe' (OneHotEncoder) or 'target' (TargetEncoder). Defaults to 'ohe'.
        :param target_categories: If True, the target variable y is treated as categorical and one-hot encoded.
        :param tree_kwargs: Additional arguments to be passed to the scikit-learn Decision Tree Regressor.
        """
        self.max_leaf_nodes = max_leaf_nodes
        self.weights = weights
        self.tree_kwargs = tree_kwargs
        self.tree_kwargs.setdefault("random_state", 42)
        self.tree_ = None
        self.scaler_ = clone(scaler)
        self.min_impurity_reduction = min_impurity_reduction
        self.impute = impute
        self.category_encoding = category_encoding
        self.target_categories = target_categories
        self.categorical_features = categorical_features
        self.tree_kwargs = {param: value for param, value in tree_kwargs.items() if param != "min_impurity_decrease"}

        if self.category_encoding not in ['ohe', 'target']:
            raise ValueError("category_encoding must be either 'ohe' or 'target'.")
        if self.category_encoding == 'target' and self.target_categories:
            raise ValueError("category_encoding='target' is incompatible with target_categories=True.")

        self.y_target_preprocessor_ = None
        self.tree_preprocessor_ = None

    def fit(self, X, y: Union[pd.Series, pd.DataFrame, np.ndarray] = None):
        """
        Fits the partitioner to the data X, optionally using a target variable y
        to influence the partitioning.

        Args:
            X (pd.DataFrame or np.ndarray): The input feature data.
            y (pd.Series, pd.DataFrame, or np.ndarray, optional): The target variable(s).
              If provided, it will be used to create target-aware partitions. Defaults to None.

        Returns:
            self: The fitted partitioner instance.
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])

        if self.categorical_features:
            # Use explicitly defined categorical features
            X_categorical = X[self.categorical_features]
            X_continuous = X.drop(columns=self.categorical_features, errors='ignore')
            # Further filter continuous to ensure they are numeric
            X_continuous = X_continuous.select_dtypes(include=np.number)
        else:
            # Infer categorical and continuous features from dtypes
            X_continuous = X.select_dtypes(include=np.number)
            X_categorical = X.select_dtypes(exclude=np.number)

        y_target_features = X_continuous.copy()
        if y is not None:
            if isinstance(y, pd.Series):
                y_df = y.to_frame(name=y.name or 'target')
            elif isinstance(y, np.ndarray):
                if y.ndim == 1:
                    y_df = pd.DataFrame(y, columns=['target'])
                else:
                    y_df = pd.DataFrame(y, columns=[f'target_{i}' for i in range(y.shape[1])])
            elif isinstance(y, pd.DataFrame):
                y_df = y
            else:
                raise TypeError("y must be a pandas Series, DataFrame, or a numpy array.")

            if self.target_categories:
                y_df = pd.get_dummies(y_df, columns=y_df.columns)
            elif not all(np.issubdtype(dtype, np.number) for dtype in y_df.dtypes):
                raise ValueError("All columns in the target y must be numeric when target_categories is False.")

            duplicate_cols = y_target_features.columns.intersection(y_df.columns)
            if not duplicate_cols.empty:
                raise ValueError(f"Duplicate column names found between X and y: {duplicate_cols.tolist()}")

            y_target_features = pd.concat([y_target_features, y_df], axis=1)

        # Define pipelines based on flags
        if self.impute:
            self.y_target_preprocessor_ = Pipeline([
                ('imputer', SimpleImputer(strategy='mean')),
                ('scaler', self.scaler_)
            ])
            continuous_pipeline = SimpleImputer(strategy='mean')
            cat_imputer = SimpleImputer(strategy='most_frequent')
        else:
            self.y_target_preprocessor_ = NanTolerantScaler()
            continuous_pipeline = 'passthrough'
            cat_imputer = None

        y_multi_output = self.y_target_preprocessor_.fit_transform(y_target_features)

        if self.category_encoding == 'ohe':
            cat_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        else:  # 'target'
            cat_encoder = TargetEncoder(target_type='continuous')

        if cat_imputer:
            categorical_pipeline = Pipeline([
                ('imputer', cat_imputer),
                ('encoder', cat_encoder)
            ])
        else:
            categorical_pipeline = cat_encoder

        self.tree_preprocessor_ = ColumnTransformer(
            transformers=[
                ('continuous', continuous_pipeline, X_continuous.columns.tolist()),
                ('categorical', categorical_pipeline, X_categorical.columns.tolist())
            ],
            remainder='passthrough'
        )

        if self.weights:
            # Find column indices for weights and apply them
            weighted_cols_indices = [i for i, col in enumerate(y_target_features.columns) if col in self.weights]
            for i in weighted_cols_indices:
                y_multi_output[:, i] *= self.weights[y_target_features.columns[i]]

        y_for_encoder = np.nanmean(y_multi_output, axis=1)
        X_for_tree = self.tree_preprocessor_.fit_transform(X, y_for_encoder)



        min_impurity_reduction = np.mean(np.nan_to_num(y_multi_output)**2) * self.min_impurity_reduction
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
            if hasattr(self.tree_preprocessor_, 'feature_names_in_') and X.shape[1] == len(self.tree_preprocessor_.feature_names_in_):
                X = pd.DataFrame(X, columns=self.tree_preprocessor_.feature_names_in_)
            else:
                X = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])

        X_transformed = self.tree_preprocessor_.transform(X)
        return self.tree_.apply(X_transformed)



    def fit_predict(self, X, y=None):
        self.fit(X, y)
        return self.get_partitions(X)

    def get_tree(self):
        if self.tree_ is None:
            raise RuntimeError("The partitioner has not been fitted yet.")
        return self.tree_