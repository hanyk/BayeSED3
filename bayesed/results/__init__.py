"""
BayeSED Results Analysis Package - Simplified Implementation

This package provides simplified, efficient access to BayeSED analysis results
with dramatically reduced complexity while maintaining full backward compatibility.
"""

from .bayesed_results import BayeSEDResults
from .utils import standardize_parameter_names, plot_posterior_comparison

__all__ = ['BayeSEDResults', 'standardize_parameter_names', 'plot_posterior_comparison']
