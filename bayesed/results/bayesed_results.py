"""
Enhanced BayeSEDResults main class with integrated components.

This module contains the redesigned BayeSEDResults class that integrates
FileDiscovery, ConfigurationManager, ValidationEngine, DataLoader, and
ParameterManager components to provide a more reliable, efficient, and
user-friendly interface for accessing BayeSED analysis results.
"""

import os
from pathlib import Path
from typing import List, Dict, Optional, Union, Any
import logging
import numpy as np
from datetime import datetime

from .base import BaseComponent
from .models import AccessScope, LoadedDataInfo, FileStructure, ConfigurationInfo
from .exceptions import (
    BayeSEDResultsError, FileDiscoveryError, ConfigurationError,
    DataLoadingError, ValidationError
)
from .logger import get_logger
from .file_discovery import FileDiscovery
from .configuration_manager import ConfigurationManager
from .validation_engine import ValidationEngine
from .data_loader import DataLoader
from .parameter_manager import ParameterManager


class BayeSEDResults(BaseComponent):
    """
    Enhanced BayeSEDResults class with integrated component architecture.
    
    This class provides a comprehensive interface for accessing BayeSED analysis
    results with improved reliability, efficiency, and user-friendliness. It
    integrates all redesigned components while maintaining compatibility with
    existing usage patterns.
    
    Key Features
    ------------
    - **Explicit Configuration Selection**: Clear handling of multiple model configurations
    - **Scope-Aware Access**: Distinction between sample-level and object-level operations
    - **Robust Validation**: Comprehensive file and data consistency checking
    - **Efficient Caching**: Intelligent caching strategies for different access patterns
    - **Enhanced Error Handling**: Detailed error messages with actionable suggestions
    - **Comprehensive Logging**: Full visibility into data loading and processing
    - **Backward Compatibility**: Maintains existing method signatures where possible
    
    Access Patterns
    ---------------
    **Sample-Level Access**: Work with entire catalogs containing multiple objects
    - Optimized for bulk operations and statistical analysis
    - Loads HDF5 data for all objects in the catalog
    - Methods return arrays/tables with data for all objects
    
    **Object-Level Access**: Work with specific individual objects
    - Optimized for detailed single-object analysis
    - Filters data to specific object(s) during loading
    - Methods return single values or object-specific data
    
    Parameters
    ----------
    output_dir : str or Path
        Directory containing BayeSED output files
    catalog_name : str, optional
        Catalog name to load results for. If None, will be auto-detected
        if only one catalog exists, otherwise requires explicit specification
    model_config : str or int, optional
        Model configuration to load. Required when multiple configurations exist.
        Can be full name, partial match, or integer index
    object_id : str, optional
        Object ID for object-level access. If None, uses sample-level access
    verbose : bool, default False
        Enable verbose logging for debugging and introspection
    validate_on_init : bool, default True
        Perform comprehensive validation during initialization
        
    Examples
    --------
    >>> # Sample-level access (entire catalog)
    >>> results = BayeSEDResults('output', catalog_name='galaxy_sample')
    >>> all_ages = results.get_parameter_values('log(age/yr)')
    >>> all_objects = results.list_objects()
    
    >>> # Object-level access (specific object)
    >>> results = BayeSEDResults('output', catalog_name='galaxy_sample', 
    ...                         object_id='obj_42')
    >>> obj_age = results.get_parameter_values('log(age/yr)')
    >>> posterior_samples = results.get_posterior_samples()
    
    >>> # Explicit configuration selection
    >>> results = BayeSEDResults('output', catalog_name='galaxy_sample',
    ...                         model_config='stellar_nebular_dust')
    
    >>> # Scope transition
    >>> sample_results = BayeSEDResults('output', catalog_name='galaxy_sample')
    >>> object_view = sample_results.get_object_view('obj_42')
    
    >>> # Introspection and debugging
    >>> results = BayeSEDResults('output', catalog_name='galaxy_sample', verbose=True)
    >>> scope_info = results.get_access_scope()
    >>> status = results.get_status_report()
    """
    
    def __init__(self, output_dir: Union[str, Path], catalog_name: Optional[str] = None,
                 model_config: Optional[Union[str, int]] = None, object_id: Optional[str] = None,
                 verbose: bool = False, validate_on_init: bool = True):
        """
        Initialize BayeSEDResults with integrated component architecture.
        
        Parameters
        ----------
        output_dir : str or Path
            Directory containing BayeSED output files
        catalog_name : str, optional
            Catalog name to load results for
        model_config : str or int, optional
            Model configuration to load
        object_id : str, optional
            Object ID for object-level access
        verbose : bool, default False
            Enable verbose logging
        validate_on_init : bool, default True
            Perform validation during initialization
        """
        # Set up logging first
        verbosity = 2 if verbose else 0  # Default to quiet (0), verbose (2) if requested
        bayesed_logger = get_logger(self.__class__.__name__, verbosity=verbosity)
        self.logger = bayesed_logger.get_logger()
        super().__init__(self.logger)
        
        # Store initialization parameters
        self.output_dir = Path(output_dir).resolve()
        self.catalog_name = catalog_name
        self.model_config = model_config
        self.object_id = object_id
        self.verbose = verbose
        self.validate_on_init = validate_on_init
        
        # Initialize components
        self._file_discovery = FileDiscovery(self.logger)
        self._config_manager = ConfigurationManager(self.logger)
        self._validation_engine = ValidationEngine(self.logger)
        self._data_loader = DataLoader(self.logger)
        self._parameter_manager = ParameterManager(self.logger)
        
        # State variables
        self._file_structure: Optional[FileStructure] = None
        self._configuration_info: Optional[ConfigurationInfo] = None
        self._access_scope: Optional[AccessScope] = None
        self._loaded_data_info: Optional[LoadedDataInfo] = None
        
        # Initialize the system
        self.initialize()
    
    def initialize(self, **kwargs) -> None:
        """
        Initialize the BayeSEDResults system with all components.
        
        This method orchestrates the initialization of all components in the
        correct order, handling file discovery, configuration selection,
        validation, and data loading.
        """
        try:
            self.logger.debug(f"Initializing BayeSEDResults for directory: {self.output_dir}")
            
            # Step 1: File Discovery
            self._discover_files()
            
            # Step 2: Configuration Management
            self._select_configuration()
            
            # Step 3: Determine Access Scope
            self._determine_access_scope()
            
            # Step 4: Validation (if enabled)
            if self.validate_on_init:
                self._validate_system()
            
            # Step 5: Initialize Data Loading
            self._initialize_data_loading()
            
            # Step 6: Initialize Parameter Management
            self._initialize_parameter_management()
            
            # Mark as initialized
            self._initialized = True
            
            self.logger.info("BayeSEDResults initialization completed successfully")
            
            # Log status summary
            if self.verbose:
                self._log_initialization_summary()
                
        except Exception as e:
            self.logger.error(f"Failed to initialize BayeSEDResults: {e}")
            raise BayeSEDResultsError(
                f"Initialization failed: {e}",
                suggestions=[
                    "Check that the output directory exists and contains BayeSED results",
                    "Verify catalog_name and model_config parameters",
                    "Use verbose=True for detailed error information"
                ]
            )
    
    def _discover_files(self) -> None:
        """Discover and organize all output files."""
        self.logger.debug("Starting file discovery...")
        
        if not self.output_dir.exists():
            raise FileDiscoveryError(
                f"Output directory does not exist: {self.output_dir}",
                suggestions=["Check the path and ensure the directory exists"]
            )
        
        # Initialize and run file discovery
        self._file_discovery.initialize(base_path=self.output_dir)
        self._file_structure = self._file_discovery.discover_files(self.output_dir)
        
        # If catalog_name was specified, ensure we use the correct catalog's file structure
        if self.catalog_name is not None:
            catalog_structure = self._file_discovery.get_catalog_structure(self.catalog_name)
            if catalog_structure is None:
                available_catalogs = self._file_discovery.list_catalogs()
                raise FileDiscoveryError(
                    f"Catalog '{self.catalog_name}' not found in output directory",
                    suggestions=[
                        f"Available catalogs: {available_catalogs}",
                        "Check that the catalog name matches the directory structure"
                    ]
                )
            self._file_structure = catalog_structure
            self.logger.debug(f"Using file structure for catalog '{self.catalog_name}'")
        
        self.logger.debug(f"Discovered {len(self._file_structure.hdf5_files)} HDF5 files")
        self.logger.debug(f"Output mode: {self._file_structure.output_mode}")
    
    def _select_configuration(self) -> None:
        """Select and validate model configuration."""
        self.logger.debug("Starting configuration selection...")
        
        # Auto-detect catalog if not specified
        if self.catalog_name is None:
            available_catalogs = self._file_discovery.list_catalogs()
            if len(available_catalogs) == 1:
                self.catalog_name = available_catalogs[0]
                self.logger.debug(f"Auto-selected catalog: {self.catalog_name}")
                # Update file structure to use the correct catalog
                catalog_structure = self._file_discovery.get_catalog_structure(self.catalog_name)
                if catalog_structure is not None:
                    self._file_structure = catalog_structure
                    self.logger.debug(f"Updated file structure for catalog '{self.catalog_name}'")
            else:
                raise ConfigurationError(
                    "Multiple catalogs found, explicit catalog_name required",
                    suggestions=[f"Available catalogs: {available_catalogs}"]
                )
        
        # Initialize configuration manager with the correct file structure
        self._config_manager.initialize(
            file_structure=self._file_structure,
            catalog_name=self.catalog_name
        )
        
        # Select configuration
        selected_config_name = self._config_manager.select_configuration(self.model_config)
        self._configuration_info = self._config_manager.get_configuration_info(selected_config_name)
        
        self.logger.debug(f"Selected configuration: {self._configuration_info.name}")
    
    def _determine_access_scope(self) -> None:
        """Determine the access scope (sample vs object level)."""
        self.logger.debug("Determining access scope...")
        
        # Count total objects from HDF5 file
        total_objects = self._data_loader.get_object_count(self._configuration_info.file_path)
        
        # Determine scope type
        if self.object_id is not None:
            scope_type = 'object'
            filtered_objects = 1
        else:
            scope_type = 'sample'
            filtered_objects = total_objects
        
        self._access_scope = AccessScope(
            scope_type=scope_type,
            catalog_name=self.catalog_name,
            object_filter=self.object_id,
            total_objects=total_objects,
            filtered_objects=filtered_objects
        )
        
        self.logger.debug(f"Access scope: {self._access_scope.get_scope_description()}")
    
    def _validate_system(self) -> None:
        """Perform comprehensive system validation."""
        self.logger.debug("Starting system validation...")
        
        # Initialize validation engine
        self._validation_engine.initialize(
            file_structure=self._file_structure,
            configuration_info=self._configuration_info,
            access_scope=self._access_scope
        )
        
        # Run validation
        validation_errors = self._validation_engine.validate(None)
        
        if validation_errors:
            error_summary = f"Validation failed with {len(validation_errors)} errors"
            self.logger.error(error_summary)
            for error in validation_errors:
                self.logger.error(f"  - {error}")
            
            raise ValidationError(
                error_summary,
                validation_errors=validation_errors
            )
        
        self.logger.debug("System validation completed successfully")
    
    def _initialize_data_loading(self) -> None:
        """Initialize the data loading component."""
        self.logger.debug("Initializing data loading...")
        
        self._data_loader.initialize(
            file_structure=self._file_structure,
            configuration_info=self._configuration_info,
            access_scope=self._access_scope
        )
        
        # Create loaded data info
        self._loaded_data_info = LoadedDataInfo(
            configuration=self._configuration_info,
            hdf5_file=self._configuration_info.file_path,
            scope=self._access_scope,
            load_time=datetime.now()
        )
        
        self.logger.debug("Data loading initialization completed")
    
    def _initialize_parameter_management(self) -> None:
        """Initialize the parameter management component."""
        self.logger.debug("Initializing parameter management...")
        
        # Get paramnames files for parameter management
        paramnames_files = self._data_loader.get_paramnames_files()
        
        if paramnames_files:
            self._parameter_manager.initialize(paramnames_files, self._access_scope)
            self.logger.debug(f"Parameter management initialized with {len(paramnames_files)} paramnames files")
        else:
            self.logger.warning("No paramnames files found - parameter management limited to HDF5 data")
    
    def _log_initialization_summary(self) -> None:
        """Log a summary of the initialization results."""
        self.logger.debug("=== BayeSEDResults Initialization Summary ===")
        self.logger.debug(f"Output Directory: {self.output_dir}")
        self.logger.debug(f"Catalog: {self.catalog_name}")
        self.logger.debug(f"Configuration: {self._configuration_info.name}")
        self.logger.debug(f"Access Scope: {self._access_scope.get_scope_description()}")
        self.logger.debug(f"Output Mode: {self._file_structure.output_mode}")
        self.logger.debug(f"HDF5 File: {Path(self._configuration_info.file_path).name}")
        self.logger.debug(f"File Size: {self._configuration_info.get_size_mb():.1f} MB")
        self.logger.debug("=" * 50)
    
    # ========================================================================
    # Public API Methods - Enhanced versions of existing methods
    # ========================================================================
    
    def get_parameter_names(self, include_derived: bool = True) -> List[str]:
        """
        Get list of parameter names (enhanced version of existing method).
        
        This method provides efficient parameter name access with scope-aware
        caching and improved error handling.
        
        Parameters
        ----------
        include_derived : bool, default True
            Whether to include derived parameters
            
        Returns
        -------
        List[str]
            List of parameter names
        """
        self._ensure_initialized()
        
        if self._parameter_manager.is_initialized():
            return self._parameter_manager.get_parameter_names(include_derived)
        else:
            # Fallback to HDF5 parameter names
            return self._data_loader.get_parameter_names_from_hdf5(include_derived)
    
    def get_free_parameters(self) -> List[str]:
        """
        Get list of free (fitted) parameters (enhanced version of existing method).
        
        Returns
        -------
        List[str]
            List of free parameter names
        """
        self._ensure_initialized()
        
        if self._parameter_manager.is_initialized():
            return self._parameter_manager.get_free_parameters()
        else:
            return self._data_loader.get_free_parameters_from_hdf5()
    
    def get_derived_parameters(self) -> List[str]:
        """
        Get list of derived parameters (enhanced version of existing method).
        
        Returns
        -------
        List[str]
            List of derived parameter names
        """
        self._ensure_initialized()
        
        if self._parameter_manager.is_initialized():
            return self._parameter_manager.get_derived_parameters()
        else:
            return self._data_loader.get_derived_parameters_from_hdf5()
    
    def get_parameter_values(self, parameter_name: str, 
                           object_ids: Optional[List[str]] = None) -> Union[np.ndarray, float]:
        """
        Get parameter values with scope-aware filtering (enhanced method).
        
        Parameters
        ----------
        parameter_name : str
            Name of the parameter to retrieve
        object_ids : List[str], optional
            Specific object IDs to filter (for sample-level access)
            
        Returns
        -------
        np.ndarray or float
            Parameter values (array for sample-level, scalar for object-level)
        """
        self._ensure_initialized()
        
        return self._data_loader.get_parameter_values(
            parameter_name=parameter_name,
            object_ids=object_ids,
            access_scope=self._access_scope
        )
    
    def list_objects(self) -> List[str]:
        """
        List available objects with scope awareness (enhanced method).
        
        Returns
        -------
        List[str]
            List of object IDs in current scope
        """
        self._ensure_initialized()
        
        return self._data_loader.list_objects(self._access_scope)
    
    def rename_parameters(self, parameter_mapping: Dict[str, str]) -> None:
        """
        Rename parameters with enhanced validation (enhanced method).
        
        Parameters
        ----------
        parameter_mapping : Dict[str, str]
            Dictionary mapping old parameter names to new names
        """
        self._ensure_initialized()
        
        if self._parameter_manager.is_initialized():
            self._parameter_manager.rename_parameters(parameter_mapping)
            # Clear GetDist samples cache so renamed samples are reloaded
            self._data_loader._samples_cache.clear()
        else:
            raise DataLoadingError(
                "Parameter renaming requires paramnames files",
                suggestions=["Ensure save_sample_par was enabled during BayeSED run"]
            )
    
    def set_parameter_labels(self, custom_labels: Dict[str, str]) -> None:
        """
        Set custom parameter labels (enhanced method).
        
        Parameters
        ----------
        custom_labels : Dict[str, str]
            Dictionary mapping parameter names to LaTeX labels
        """
        self._ensure_initialized()
        
        if self._parameter_manager.is_initialized():
            self._parameter_manager.set_parameter_labels(custom_labels)
        else:
            self.logger.warning("Parameter labeling requires paramnames files")
    
    # ========================================================================
    # New Methods - Scope Management and Introspection
    # ========================================================================
    
    def get_object_view(self, object_id: str) -> 'BayeSEDResults':
        """
        Create an object-level view from sample-level access.
        
        Parameters
        ----------
        object_id : str
            Object ID to create view for
            
        Returns
        -------
        BayeSEDResults
            New BayeSEDResults instance with object-level scope
        """
        self._ensure_initialized()
        
        if self._access_scope.is_object_level():
            raise BayeSEDResultsError(
                "Cannot create object view from object-level access",
                suggestions=["Use sample-level access to create object views"]
            )
        
        # Validate object exists
        available_objects = self.list_objects()
        if object_id not in available_objects:
            raise DataLoadingError(
                f"Object '{object_id}' not found in catalog",
                suggestions=[f"Available objects: {available_objects[:10]}..."]
            )
        
        # Create new instance with object-level scope
        return BayeSEDResults(
            output_dir=self.output_dir,
            catalog_name=self.catalog_name,
            model_config=self._configuration_info.name,
            object_id=object_id,
            verbose=self.verbose,
            validate_on_init=False  # Skip validation for derived instance
        )
    
    def get_access_scope(self) -> AccessScope:
        """
        Get current access scope information.
        
        Returns
        -------
        AccessScope
            Current access scope details
        """
        self._ensure_initialized()
        return self._access_scope
    
    def get_status_report(self) -> Dict[str, Any]:
        """
        Get comprehensive status report for debugging.
        
        Returns
        -------
        Dict[str, Any]
            Detailed status information
        """
        self._ensure_initialized()
        
        return {
            'initialization': {
                'output_dir': str(self.output_dir),
                'catalog_name': self.catalog_name,
                'model_config': self.model_config,
                'object_id': self.object_id,
                'initialized': self._initialized
            },
            'file_structure': {
                'output_mode': self._file_structure.output_mode,
                'hdf5_files': len(self._file_structure.hdf5_files),
                'has_object_files': self._file_structure.has_object_files()
            },
            'configuration': {
                'name': self._configuration_info.name,
                'file_path': self._configuration_info.file_path,
                'file_size_mb': self._configuration_info.get_size_mb(),
                'object_count': self._configuration_info.object_count,
                'parameter_count': self._configuration_info.parameter_count
            },
            'access_scope': {
                'scope_type': self._access_scope.scope_type,
                'catalog_name': self._access_scope.catalog_name,
                'object_filter': self._access_scope.object_filter,
                'total_objects': self._access_scope.total_objects,
                'filtered_objects': self._access_scope.filtered_objects
            },
            'components': {
                'file_discovery': self._file_discovery.is_initialized(),
                'config_manager': self._config_manager.is_initialized(),
                'validation_engine': self._validation_engine.is_initialized(),
                'data_loader': self._data_loader.is_initialized(),
                'parameter_manager': self._parameter_manager.is_initialized()
            },
            'cache_status': self._data_loader.get_cache_status() if self._data_loader.is_initialized() else {}
        }
    
    def clear_cache(self, cache_type: Optional[str] = None) -> None:
        """
        Clear caches with scope awareness.
        
        Parameters
        ----------
        cache_type : str, optional
            Specific cache type to clear ('data', 'parameters', 'all')
        """
        self._ensure_initialized()
        
        if cache_type in [None, 'all', 'data']:
            self._data_loader.clear_cache()
            
        if cache_type in [None, 'all', 'parameters']:
            if self._parameter_manager.is_initialized():
                self._parameter_manager.clear_cache()
        
        self.logger.info(f"Cleared caches: {cache_type or 'all'}")
    
    # ========================================================================
    # Enhanced API Methods - Superior functionality with better performance
    # ========================================================================
    
    def get_evidence(self, object_ids: Optional[List[str]] = None) -> Union[Dict[str, float], float]:
        """
        Get Bayesian evidence values with enhanced scope awareness.
        
        This enhanced method provides superior functionality compared to the original
        by supporting both sample-level and object-level access patterns with
        better error handling and performance.
        
        Parameters
        ----------
        object_ids : List[str], optional
            Specific object IDs to get evidence for. If None, uses current scope.
            
        Returns
        -------
        Dict[str, float] or float
            Evidence values. Returns dict for sample-level, float for object-level.
        """
        self._ensure_initialized()
        
        return self._data_loader.get_evidence(
            object_ids=object_ids,
            access_scope=self._access_scope
        )
    
    def get_posterior_samples(self, object_id: Optional[str] = None) -> 'astropy.table.Table':
        """
        Get posterior samples with enhanced scope awareness and caching.
        
        This enhanced method provides superior functionality with better performance
        through intelligent caching and scope-aware loading.
        
        Parameters
        ----------
        object_id : str, optional
            Object ID to get samples for. If None, uses current scope.
            
        Returns
        -------
        astropy.table.Table
            Posterior samples table
        """
        self._ensure_initialized()
        
        return self._data_loader.get_posterior_samples(
            object_id=object_id,
            access_scope=self._access_scope
        )
    
    def get_bestfit_spectrum(self, object_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get best-fit spectrum with enhanced validation and error handling.
        
        This enhanced method provides superior functionality with better error
        handling and validation compared to the original implementation.
        
        Parameters
        ----------
        object_id : str, optional
            Object ID to get spectrum for. If None, uses current scope.
            
        Returns
        -------
        Dict[str, Any]
            Best-fit spectrum data
        """
        self._ensure_initialized()
        
        return self._data_loader.get_bestfit_spectrum(
            object_id=object_id,
            access_scope=self._access_scope
        )
    
    def load_hdf5_results(self, filter_snr: bool = True, min_snr: float = 0.0) -> 'astropy.table.Table':
        """
        Load HDF5 results with enhanced scope filtering and validation.
        
        This enhanced method provides superior functionality with scope-aware
        filtering and better performance compared to the original implementation.
        
        Parameters
        ----------
        filter_snr : bool, default True
            Whether to filter by SNR
        min_snr : float, default 0.0
            Minimum SNR threshold
            
        Returns
        -------
        astropy.table.Table
            HDF5 results table with scope filtering applied
        """
        self._ensure_initialized()
        
        return self._data_loader.load_hdf5_results(
            filter_snr=filter_snr,
            min_snr=min_snr,
            access_scope=self._access_scope
        )
    
    def list_model_configurations(self, detailed: bool = False) -> Union[List[str], Dict[str, Dict[str, Any]]]:
        """
        List model configurations with enhanced metadata and validation.
        
        This enhanced method provides superior functionality with better metadata
        and validation compared to the original implementation.
        
        Parameters
        ----------
        detailed : bool, default False
            Whether to return detailed metadata
            
        Returns
        -------
        List[str] or Dict[str, Dict[str, Any]]
            Configuration names or detailed metadata
        """
        self._ensure_initialized()
        
        return self._config_manager.list_configurations(detailed=detailed)
    
    def validate_model_config(self, model_config: Union[str, int]) -> str:
        """
        Validate model configuration with enhanced error messages.
        
        This enhanced method provides superior functionality with better error
        messages and suggestions compared to the original implementation.
        
        Parameters
        ----------
        model_config : str or int
            Model configuration to validate
            
        Returns
        -------
        str
            Validated configuration name
        """
        self._ensure_initialized()
        
        return self._config_manager.validate_model_config(model_config)
    
    def get_getdist_samples(self, object_id: Optional[str] = None) -> 'getdist.MCSamples':
        """
        Get GetDist samples with enhanced caching and parameter management.
        
        This enhanced method provides superior functionality with better caching
        and parameter management compared to the original implementation.
        
        Parameters
        ----------
        object_id : str, optional
            Object ID to get samples for. If None, uses current scope.
            
        Returns
        -------
        getdist.MCSamples
            GetDist samples object
        """
        self._ensure_initialized()
        
        return self._data_loader.get_getdist_samples(
            object_id=object_id,
            access_scope=self._access_scope,
            parameter_manager=self._parameter_manager
        )
    
    def plot_posterior(self, params: Optional[List[str]] = None, 
                      object_id: Optional[str] = None,
                      method: str = 'getdist', filled: bool = True,
                      show: bool = True, output_file: Optional[str] = None,
                      **kwargs) -> Any:
        """
        Plot posterior distributions with enhanced functionality.
        
        This enhanced method provides superior functionality with better parameter
        handling, scope awareness, and plotting options compared to the original.
        
        Parameters
        ----------
        params : List[str], optional
            Parameters to plot. If None, uses free parameters.
        object_id : str, optional
            Object ID to plot for. If None, uses current scope.
        method : str, default 'getdist'
            Plotting method to use
        filled : bool, default True
            Whether to use filled contours
        show : bool, default True
            Whether to display the plot
        output_file : str, optional
            Output file path for saving
        **kwargs
            Additional plotting arguments
            
        Returns
        -------
        Any
            Plot object (depends on method)
        """
        self._ensure_initialized()
        
        # Handle object_id selection
        if object_id is None:
            if self._access_scope.is_object_level() and self._access_scope.object_filter:
                object_id = self._access_scope.object_filter
            else:
                # For sample-level access or when no object in scope, use first available object with warning
                objects = self.list_objects()
                if objects:
                    object_id = objects[0]
                    self.logger.warning(f"No object_id provided for plot_posterior. Using first available object: {object_id}")
                else:
                    raise DataLoadingError("No objects available for plotting")
        
        # Get samples using enhanced method
        samples = self.get_getdist_samples(object_id=object_id)
        
        # Use default parameters if none specified
        if params is None:
            params = self.get_free_parameters()
        
        # Import GetDist plotting
        try:
            from getdist import plots
        except ImportError:
            raise ImportError("GetDist is required for plotting. Install with: pip install getdist")
        
        # Create plotter
        g = plots.get_subplot_plotter()
        
        # Create plot based on number of parameters
        if len(params) == 1:
            g.plot_1d(samples, params[0], **kwargs)
        else:
            g.triangle_plot([samples], params, filled=filled, **kwargs)
        
        # Save if requested
        if output_file:
            g.export(output_file)
        
        # Show if requested
        if show:
            try:
                import matplotlib.pyplot as plt
                plt.show()
            except Exception as e:
                self.logger.warning(f"Could not display plot: {e}")
        
        return g
    
    def plot_bestfit(self, object_id: Optional[str] = None, 
                    output_file: Optional[str] = None, show: bool = True,
                    **kwargs) -> Any:
        """
        Plot best-fit SED with enhanced functionality.
        
        This enhanced method provides superior functionality with better error
        handling and plotting options compared to the original implementation.
        
        Parameters
        ----------
        object_id : str, optional
            Object ID to plot for. If None, uses current scope.
        output_file : str, optional
            Output file path for saving
        show : bool, default True
            Whether to display the plot
        **kwargs
            Additional plotting arguments
            
        Returns
        -------
        Any
            Plot object
        """
        self._ensure_initialized()
        
        # Determine object to plot
        if object_id is None:
            if self._access_scope.is_object_level() and self._access_scope.object_filter:
                object_id = self._access_scope.object_filter
            else:
                # For sample-level access, use first available object with warning
                objects = self.list_objects()
                if objects:
                    object_id = objects[0]
                    self.logger.warning(f"No object_id provided for plot_bestfit. Using first available object: {object_id}")
                else:
                    raise DataLoadingError("No objects available for plotting best-fit spectrum")
        
        # Get bestfit file path instead of loaded data
        bestfit_files = self._data_loader._find_bestfit_files(object_id)
        
        if not bestfit_files:
            raise DataLoadingError(f"No best-fit files found for object '{object_id}'")
        
        # Use the first bestfit file (there should typically be only one)
        fits_file = bestfit_files[0]
        
        # Import plotting function
        from ..plotting import plot_bestfit
        
        # Create plot with file path
        fig = plot_bestfit(fits_file, show=show, output_file=output_file, **kwargs)
        
        return fig
    
    def compute_parameter_correlations(self, params: Optional[List[str]] = None,
                                     object_ids: Optional[List[str]] = None) -> 'numpy.ndarray':
        """
        Compute parameter correlations with enhanced scope awareness.
        
        This is a new enhanced method that provides superior functionality for
        computing parameter correlations across different scopes.
        
        Parameters
        ----------
        params : List[str], optional
            Parameters to compute correlations for. If None, uses free parameters.
        object_ids : List[str], optional
            Object IDs to include. If None, uses current scope.
            
        Returns
        -------
        numpy.ndarray
            Correlation matrix
        """
        self._ensure_initialized()
        
        if not self._access_scope.is_sample_level():
            raise BayeSEDResultsError(
                "Parameter correlations require sample-level access",
                suggestions=["Use sample-level initialization without object_id parameter"]
            )
        
        return self._data_loader.compute_parameter_correlations(
            params=params or self.get_free_parameters(),
            object_ids=object_ids,
            access_scope=self._access_scope
        )
    
    def get_parameter_statistics(self, params: Optional[List[str]] = None,
                               object_ids: Optional[List[str]] = None) -> Dict[str, Dict[str, float]]:
        """
        Get parameter statistics with enhanced scope awareness.
        
        This is a new enhanced method that provides superior functionality for
        computing parameter statistics across different scopes.
        
        Parameters
        ----------
        params : List[str], optional
            Parameters to compute statistics for. If None, uses free parameters.
        object_ids : List[str], optional
            Object IDs to include. If None, uses current scope.
            
        Returns
        -------
        Dict[str, Dict[str, float]]
            Parameter statistics (mean, std, median, etc.)
        """
        self._ensure_initialized()
        
        return self._data_loader.get_parameter_statistics(
            params=params or self.get_free_parameters(),
            object_ids=object_ids,
            access_scope=self._access_scope
        )
    
    def plot_posterior_free(self, output_file: Optional[str] = None, 
                           show: bool = True, object_id: Optional[str] = None,
                           **kwargs) -> Any:
        """
        Plot posterior distributions (corner plot) for free parameters.
        
        Enhanced plotting method for free parameter posteriors with scope awareness.
        Note: This method requires GetDist sample files to be available.
        
        Parameters
        ----------
        output_file : str, optional
            Output file path for saving plot
        show : bool, default True
            Whether to display the plot
        object_id : str, optional
            Object ID to plot. If None and in sample-level mode, uses first object.
        **kwargs
            Additional plotting parameters
            
        Returns
        -------
        Any
            Plot object
        """
        self._ensure_initialized()
        
        try:
            # Handle object_id selection
            if object_id is None:
                if self._access_scope.is_object_level() and self._access_scope.object_filter:
                    object_id = self._access_scope.object_filter
                else:
                    # For sample-level access or when no object in scope, use first available object with warning
                    objects = self.list_objects()
                    if objects:
                        object_id = objects[0]
                        self.logger.warning(f"No object_id provided for plot_posterior_free. Using first available object: {object_id}")
                    else:
                        raise DataLoadingError("No objects available for plotting")
            
            free_params = self.get_free_parameters()
            
            # Filter parameters to only include those available and varying in GetDist samples
            # Use the object_id we have (either passed in or determined above)
            current_object_id = object_id
            if current_object_id is None and self._access_scope.is_object_level():
                # For object-level access, use the object from the scope
                current_object_id = self._access_scope.object_filter
            
            if current_object_id:
                try:
                    samples = self.get_getdist_samples(object_id=current_object_id)
                    getdist_names = [p.name for p in samples.paramNames.names]
                    
                    # Filter out missing and fixed parameters
                    varying_params = []
                    for param_name in free_params:
                        if param_name not in getdist_names:
                            self.logger.debug(f"Excluding missing parameter: {param_name}")
                            continue
                            
                        # Check if parameter actually varies
                        param_index = getdist_names.index(param_name)
                        param_samples = samples.samples[:, param_index]
                        
                        # Use numpy to check variance more robustly
                        import numpy as np
                        if np.var(param_samples) > 1e-10:  # Small threshold for numerical precision
                            varying_params.append(param_name)
                        else:
                            self.logger.debug(f"Excluding fixed parameter: {param_name} (variance: {np.var(param_samples)})")
                    
                    free_params = varying_params
                    self.logger.debug(f"Filtered to {len(free_params)} varying free parameters available in GetDist samples")
                except Exception as e:
                    self.logger.warning(f"Could not filter parameters: {e}")
            
            return self.plot_posterior(params=free_params, object_id=object_id,
                                     output_file=output_file, show=show, **kwargs)
        except DataLoadingError as e:
            if "GetDist samples" in str(e) or "No chains found" in str(e):
                self.logger.warning(f"GetDist plotting not available: {e}")
                self.logger.info("Consider using individual parameter plotting or ensure GetDist-compatible sample files exist")
                # Return a placeholder or simple message
                print(f"Free parameters ({len(self.get_free_parameters())}): {self.get_free_parameters()}")
                return None
            else:
                raise
    
    def plot_posterior_derived(self, max_params: int = 10, 
                              output_file: Optional[str] = None,
                              show: bool = True, object_id: Optional[str] = None,
                              **kwargs) -> Any:
        """
        Plot posterior distributions (corner plot) for derived parameters.
        
        Enhanced plotting method for derived parameter posteriors with scope awareness.
        Note: This method requires GetDist sample files to be available.
        
        Parameters
        ----------
        max_params : int, default 10
            Maximum number of parameters to plot
        output_file : str, optional
            Output file path for saving plot
        show : bool, default True
            Whether to display the plot
        object_id : str, optional
            Object ID to plot. If None and in sample-level mode, uses first object.
        **kwargs
            Additional plotting parameters
            
        Returns
        -------
        Any
            Plot object
        """
        self._ensure_initialized()
        
        try:
            # Handle object_id selection
            if object_id is None:
                if self._access_scope.is_object_level() and self._access_scope.object_filter:
                    object_id = self._access_scope.object_filter
                else:
                    # For sample-level access or when no object in scope, use first available object with warning
                    objects = self.list_objects()
                    if objects:
                        object_id = objects[0]
                        self.logger.warning(f"No object_id provided for plot_posterior_derived. Using first available object: {object_id}")
                    else:
                        raise DataLoadingError("No objects available for plotting")
            
            derived_params = self.get_derived_parameters()
            
            # Filter parameters to only include those available in GetDist samples
            if object_id:
                try:
                    samples = self.get_getdist_samples(object_id=object_id)
                    getdist_names = [p.name for p in samples.paramNames.names]
                    derived_params = [p for p in derived_params if p in getdist_names]
                    self.logger.debug(f"Filtered to {len(derived_params)} derived parameters available in GetDist samples")
                except Exception as e:
                    self.logger.warning(f"Could not filter parameters: {e}")
            
            # Limit to max_params if specified
            if max_params and len(derived_params) > max_params:
                derived_params = derived_params[:max_params]
                
            return self.plot_posterior(params=derived_params, object_id=object_id,
                                     output_file=output_file, show=show, **kwargs)
        except DataLoadingError as e:
            if "GetDist samples" in str(e) or "No chains found" in str(e):
                self.logger.warning(f"GetDist plotting not available: {e}")
                self.logger.info("Consider using individual parameter plotting or ensure GetDist-compatible sample files exist")
                # Return a placeholder or simple message
                limited_params = self.get_derived_parameters()[:max_params] if max_params else self.get_derived_parameters()
                print(f"Derived parameters ({len(limited_params)}): {limited_params}")
                return None
            else:
                raise

    # ========================================================================
    # Utility Methods
    # ========================================================================
    
    def print_summary(self) -> None:
        """Print a comprehensive summary of the loaded results."""
        self._ensure_initialized()
        
        print("=" * 60)
        print("BayeSEDResults Summary")
        print("=" * 60)
        print(f"Output Directory: {self.output_dir}")
        print(f"Catalog: {self.catalog_name}")
        print(f"Configuration: {self._configuration_info.name}")
        print(f"Access Scope: {self._access_scope.get_scope_description()}")
        print(f"Output Mode: {self._file_structure.output_mode}")
        print()
        
        # Parameter summary
        try:
            all_params = self.get_parameter_names()
            free_params = self.get_free_parameters()
            derived_params = self.get_derived_parameters()
            
            print("Parameter Summary:")
            print(f"  Total Parameters: {len(all_params)}")
            print(f"  Free Parameters: {len(free_params)}")
            print(f"  Derived Parameters: {len(derived_params)}")
            print()
        except Exception as e:
            print(f"Parameter summary unavailable: {e}")
        
        # Object summary
        try:
            objects = self.list_objects()
            print(f"Objects: {len(objects)} available")
            if len(objects) <= 10:
                print(f"  Object IDs: {objects}")
            else:
                print(f"  First 5: {objects[:5]}")
                print(f"  Last 5: {objects[-5:]}")
        except Exception as e:
            print(f"Object summary unavailable: {e}")
        
        print("=" * 60)
    
    def __repr__(self) -> str:
        """String representation of BayeSEDResults."""
        if not self._initialized:
            return f"BayeSEDResults(uninitialized, output_dir='{self.output_dir}')"
        
        scope_desc = self._access_scope.get_scope_description()
        return f"BayeSEDResults(catalog='{self.catalog_name}', {scope_desc})"
    
    def __str__(self) -> str:
        """Human-readable string representation."""
        return self.__repr__()
    
    def _ensure_initialized(self) -> None:
        """Ensure the system is properly initialized."""
        if not hasattr(self, '_initialized') or not self._initialized:
            raise BayeSEDResultsError(
                "BayeSEDResults not properly initialized",
                suggestions=["Check initialization parameters and try again"]
            )
    
    def set_verbosity(self, level: int) -> None:
        """
        Set logging verbosity level.
        
        Parameters
        ----------
        level : int
            Verbosity level: 0 (quiet), 1 (normal), 2 (verbose)
        """
        from .logger import set_global_verbosity
        set_global_verbosity(level)
    
    def enable_verbose_logging(self) -> None:
        """Enable verbose logging for debugging."""
        self.set_verbosity(2)
    
    def enable_quiet_logging(self) -> None:
        """Enable quiet logging (default) - only warnings and errors."""
        self.set_verbosity(0)