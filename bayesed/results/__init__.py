"""
BayeSED Results Analysis Package

This package provides efficient access to BayeSED analysis results
with dramatically reduced complexity while maintaining full backward compatibility.
"""

from .bayesed_results import BayeSEDResults
from .utils import standardize_parameter_names, plot_posterior_comparison

__all__ = ['BayeSEDResults', 'standardize_parameter_names', 'plot_posterior_comparison']
