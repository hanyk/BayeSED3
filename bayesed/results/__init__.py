"""
BayeSED Results Analysis Package

Public entry: BayeSEDResults (facade). Internals: ResultsProducts +
ResultsPresentation (ADR-0003).
"""

from .bayesed_results import BayeSEDResults
from .utils import standardize_parameter_names, plot_posterior_comparison

__all__ = ['BayeSEDResults', 'standardize_parameter_names', 'plot_posterior_comparison']
