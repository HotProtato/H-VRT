from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class VarianceReport:
    """Contains statistics about the distribution of variance."""
    is_proportional: bool
    min_variance: float
    max_variance: float
    mean_variance: np.floating
    median_variance: np.floating
    hhi: float
    distribution: np.ndarray
    histogram: tuple


@dataclass
class SpanReport:
    """Contains statistics about the distribution of feature spans."""
    is_proportional: bool
    min_span: float
    max_span: float
    mean_span: np.floating
    median_span: np.floating
    hhi: float
    distribution: np.ndarray
    histogram: tuple


@dataclass
class FeatureReport:
    """A comprehensive report for a single feature, containing optional sub-reports."""
    name: str
    variance_report: Optional[VarianceReport] = None
    span_report: Optional[SpanReport] = None
