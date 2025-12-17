"""
BayeSEDResults redesigned module.

This module contains the redesigned BayeSEDResults implementation with improved
reliability, efficiency, and user-friendliness.
"""

from .exceptions import (
    BayeSEDResultsError,
    FileDiscoveryError,
    ConfigurationError,
    DataLoadingError,
    ValidationError
)

from .models import (
    AccessScope,
    LoadedDataInfo,
    FileStructure,
    ConfigurationInfo
)

from .base import (
    BaseComponent,
    BaseFileHandler,
    BaseValidator
)

from .logger import (
    BayeSEDResultsLogger,
    get_logger,
    set_global_verbosity
)

from .file_discovery import FileDiscovery
from .configuration_manager import ConfigurationManager
from .validation_engine import ValidationEngine
from .data_loader import DataLoader
from .parameter_manager import ParameterManager
from .bayesed_results import BayeSEDResults
from .utils import (
    list_catalog_names,
    list_model_configs,
    plot_posterior_comparison,
    standardize_parameter_names
)

__all__ = [
    # Exceptions
    'BayeSEDResultsError',
    'FileDiscoveryError', 
    'ConfigurationError',
    'DataLoadingError',
    'ValidationError',
    
    # Data models
    'AccessScope',
    'LoadedDataInfo',
    'FileStructure',
    'ConfigurationInfo',
    
    # Base classes
    'BaseComponent',
    'BaseFileHandler',
    'BaseValidator',
    
    # Logging
    'BayeSEDResultsLogger',
    'get_logger',
    'set_global_verbosity',
    
    # Components
    'FileDiscovery',
    'ConfigurationManager',
    'ValidationEngine',
    'DataLoader',
    'ParameterManager',
    
    # Main class (enhanced functionality)
    'BayeSEDResults',
    
    # Utility functions
    'list_catalog_names',
    'list_model_configs', 
    'plot_posterior_comparison',
    'standardize_parameter_names'
]