
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from sklearn.neighbors import KernelDensity

# Optional visualization libraries
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    _VISUALIZATION_ENABLED = True
except ImportError:
    _VISUALIZATION_ENABLED = False

class DistributionAnalyzer:
    """
    Analyzes and compares the global data distribution against the reconstructed
    distribution from partitioned data, using Kernel Density Estimation.
    """
    def __init__(self, data: pd.DataFrame, partition_labels: pd.Series):
        """
        Initializes the analyzer.

        Args:
            data: The original DataFrame.
            partition_labels: A Series mapping each sample to a partition ID.
        """
        self.data = data
        self.partitions = partition_labels
        self.fitted_kdes_: Dict[str, Dict[str, Any]] = {}

    def fit(self, kde_params: Optional[Dict] = None):
        """
        Fits KernelDensity models to the global data and to each partition for all features.

        Args:
            kde_params (dict, optional): Parameters to pass to the KernelDensity model,
                                       e.g., {'kernel': 'gaussian', 'bandwidth': 0.5}.
        """
        if kde_params is None:
            kde_params = {'kernel': 'gaussian', 'bandwidth': 'scott'}

        print("Fitting Kernel Density models...")
        for col in self.data.select_dtypes(include=np.number).columns:
            self.fitted_kdes_[col] = {'partitions': {}}
            
            # Fit global KDE
            global_kde = KernelDensity(**kde_params)
            global_kde.fit(self.data[[col]])
            self.fitted_kdes_[col]['global'] = global_kde

            # Fit partition KDEs
            for pid in self.partitions.unique():
                partition_data = self.data.loc[self.partitions == pid, [col]]
                if not partition_data.empty:
                    part_kde = KernelDensity(**kde_params)
                    part_kde.fit(partition_data)
                    self.fitted_kdes_[col]['partitions'][pid] = part_kde
        return self

    def plot_comparison(self, feature_name: str, save_path: Optional[str] = None):
        """
        Generates and saves an overlaid KDE plot comparing the original distribution
        to the reconstructed distribution from the partitions.

        Args:
            feature_name (str): The name of the feature to plot.
            save_path (str, optional): If provided, saves the plot to this path.
        """
        if not _VISUALIZATION_ENABLED:
            print("Plotting libraries not installed. Skipping visualization.")
            return

        if feature_name not in self.fitted_kdes_:
            raise ValueError(f"KDEs not fitted for feature '{feature_name}'. Please run .fit() first.")

        print(f"Generating comparison plot for '{feature_name}'...")
        
        # 1. Generate plot points for the global KDE
        global_kde = self.fitted_kdes_[feature_name]['global']
        x_min, x_max = self.data[feature_name].min(), self.data[feature_name].max()
        plot_points = np.linspace(x_min, x_max, 1000)[:, np.newaxis]
        global_log_dens = global_kde.score_samples(plot_points)

        # 2. Generate plot points for the reconstructed KDE
        # We create the reconstructed density by taking a weighted average of the partition densities.
        reconstructed_log_dens = np.zeros_like(global_log_dens)
        total_samples = len(self.data)
        for pid, part_kde in self.fitted_kdes_[feature_name]['partitions'].items():
            partition_size = (self.partitions == pid).sum()
            weight = partition_size / total_samples
            reconstructed_log_dens += np.exp(part_kde.score_samples(plot_points)) * weight
        
        # Convert summed densities back to log scale for stability, handling zeros
        reconstructed_log_dens = np.log(reconstructed_log_dens + 1e-9)

        # 3. Plot the two distributions
        plt.figure(figsize=(12, 7))
        plt.plot(plot_points[:, 0], np.exp(global_log_dens), color='blue', lw=2, label='Original Distribution')
        plt.plot(plot_points[:, 0], np.exp(reconstructed_log_dens), color='red', lw=2, linestyle='--', label='Partitioned Reconstruction')
        plt.title(f"Original vs. Reconstructed Distribution for '{feature_name}'")
        plt.xlabel(feature_name)
        plt.ylabel("Density")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)

        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
