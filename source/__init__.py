"""
Diabetes Data Visualization Package

A toolkit for analyzing diabetes patient data and creating visualizations.
"""

from .data_loader import load_diabetes_data, get_dataset_summary
from .visualizer import plot_correlation_heatmap, plot_correlation_piechart
from .analyzer import calculate_correlations, identify_risk_factors

__version__ = "1.0.0"
__all__ = [
    "load_diabetes_data",
    "get_dataset_summary",
    "plot_correlation_heatmap",
    "plot_correlation_piechart",
    "calculate_correlations",
    "identify_risk_factors",
]
