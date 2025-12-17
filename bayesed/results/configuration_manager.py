"""
ConfigurationManager component for BayeSEDResults redesign.

This module implements model configuration discovery, selection, and validation
functionality, providing explicit configuration selection with helpful error
messages and support for partial matching and index-based selection.
"""

import re
from pathlib import Path
from typing import List, Dict, Optional, Union, Tuple, Any
import logging
from difflib import get_close_matches

from .base import BaseComponent
from .models import FileStructure, ConfigurationInfo
from .exceptions import ConfigurationError
from .logger import get_logger


class ConfigurationManager(BaseComponent):
    """
    Configuration management component for BayeSED model configurations.
    
    This component handles model configuration discovery, selection, and validation,
    building on existing patterns while adding enhanced error handling and
    user-friendly selection methods.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize ConfigurationManager component."""
        if logger is None:
            bayesed_logger = get_logger(__name__)
            logger = bayesed_logger.get_logger()
        super().__init__(logger)
        self._file_structure: Optional[FileStructure] = None
        self._configurations: Dict[str, ConfigurationInfo] = {}
    
    def initialize(self, file_structure: FileStructure, **kwargs) -> None:
        """
        Initialize the ConfigurationManager with file structure information.
        
        Parameters
        ----------
        file_structure : FileStructure
            File structure information from FileDiscovery
        **kwargs
            Additional initialization parameters
        """
        self._file_structure = file_structure
        self._configurations = {}
        
        # Build configuration information
        self._build_configuration_info()
        
        self._initialized = True
        self.logger.debug(f"ConfigurationManager initialized for catalog '{file_structure.catalog_name}' "
                         f"with {len(self._configurations)} configurations")
    
    def _build_configuration_info(self) -> None:
        """Build detailed configuration information from file structure."""
        if not self._file_structure:
            return
        
        for config_name, hdf5_file in self._file_structure.hdf5_files.items():
            file_path = Path(hdf5_file)
            
            # Get file statistics
            try:
                stat = file_path.stat()
                file_size = stat.st_size
                creation_time = stat.st_mtime
            except OSError:
                file_size = 0
                creation_time = None
            
            # Create configuration info (object/parameter counts will be filled later if needed)
            config_info = ConfigurationInfo(
                name=config_name,
                file_path=hdf5_file,
                file_size=file_size,
                object_count=0,  # Will be populated when HDF5 is loaded
                parameter_count=0,  # Will be populated when HDF5 is loaded
                creation_time=creation_time,
                catalog_name=self._file_structure.catalog_name
            )
            
            self._configurations[config_name] = config_info
            self.logger.debug(f"Built configuration info for: {config_name}")
    
    def list_configurations(self, catalog_name: Optional[str] = None) -> List[str]:
        """
        List available model configurations for the catalog.
        
        Parameters
        ----------
        catalog_name : str, optional
            Name of the catalog (for validation, should match initialized catalog)
            
        Returns
        -------
        List[str]
            Sorted list of available configuration names
            
        Raises
        ------
        ConfigurationError
            If catalog name doesn't match or no configurations available
        """
        self._ensure_initialized()
        
        # Validate catalog name if provided
        if catalog_name is not None and catalog_name != self._file_structure.catalog_name:
            raise ConfigurationError(
                f"Catalog name mismatch: expected '{self._file_structure.catalog_name}', "
                f"got '{catalog_name}'",
                suggestions=[f"Use catalog name: '{self._file_structure.catalog_name}'"]
            )
        
        if not self._configurations:
            raise ConfigurationError(
                f"No model configurations found for catalog '{self._file_structure.catalog_name}'",
                suggestions=[
                    "Check that BayeSED analysis has been run",
                    "Verify HDF5 output files are present",
                    "Ensure file permissions allow reading"
                ]
            )
        
        config_names = sorted(self._configurations.keys())
        self.logger.debug(f"Listed {len(config_names)} configurations: {config_names}")
        
        return config_names
    
    def select_configuration(self, model_config: Union[str, int, None]) -> str:
        """
        Select and validate a model configuration.
        
        This method implements the core selection logic with support for:
        - Explicit configuration names
        - Partial matching
        - Index-based selection
        - Auto-selection for single configurations
        
        Parameters
        ----------
        model_config : str, int, or None
            Configuration specification:
            - str: Exact or partial configuration name
            - int: Index into sorted configuration list (0-based)
            - None: Auto-select if only one configuration exists
            
        Returns
        -------
        str
            Selected configuration name
            
        Raises
        ------
        ConfigurationError
            If configuration selection fails or is ambiguous
        """
        self._ensure_initialized()
        
        available_configs = self.list_configurations()
        
        # Case 1: Auto-selection for single configuration (Requirement 1.3)
        if model_config is None:
            if len(available_configs) == 1:
                selected = available_configs[0]
                self.logger.debug(f"Auto-selected single configuration: {selected}")
                return selected
            else:
                raise ConfigurationError(
                    f"Multiple model configurations available for catalog "
                    f"'{self._file_structure.catalog_name}'. Explicit selection required.",
                    available_configs=available_configs,
                    suggestions=[
                        "Specify a configuration name or index",
                        f"Available configurations: {', '.join(available_configs)}"
                    ]
                )
        
        # Case 2: Index-based selection
        if isinstance(model_config, int):
            return self._select_by_index(model_config, available_configs)
        
        # Case 3: String-based selection (exact or partial matching)
        if isinstance(model_config, str):
            return self._select_by_name(model_config, available_configs)
        
        # Invalid type
        raise ConfigurationError(
            f"Invalid model_config type: {type(model_config)}. "
            f"Expected str, int, or None.",
            available_configs=available_configs,
            suggestions=[
                "Use a configuration name (str)",
                "Use a configuration index (int)",
                "Use None for auto-selection"
            ]
        )
    
    def _select_by_index(self, index: int, available_configs: List[str]) -> str:
        """
        Select configuration by index.
        
        Parameters
        ----------
        index : int
            0-based index into sorted configuration list
        available_configs : List[str]
            List of available configuration names
            
        Returns
        -------
        str
            Selected configuration name
            
        Raises
        ------
        ConfigurationError
            If index is out of range
        """
        if index < 0 or index >= len(available_configs):
            # Generate helpful index suggestions
            index_suggestions = []
            for i, config in enumerate(available_configs):
                index_suggestions.append(f"  {i}: {config}")
            
            raise ConfigurationError(
                f"Configuration index {index} out of range. "
                f"Valid indices: 0-{len(available_configs)-1}",
                available_configs=available_configs,
                invalid_config=str(index),
                suggestions=[
                    "Valid indices and configurations:",
                    *index_suggestions
                ]
            )
        
        selected = available_configs[index]
        self.logger.debug(f"Selected configuration by index {index}: {selected}")
        return selected
    
    def _select_by_name(self, name: str, available_configs: List[str]) -> str:
        """
        Select configuration by name with partial matching support.
        
        Parameters
        ----------
        name : str
            Configuration name (exact or partial)
        available_configs : List[str]
            List of available configuration names
            
        Returns
        -------
        str
            Selected configuration name
            
        Raises
        ------
        ConfigurationError
            If name doesn't match any configuration or matches multiple
        """
        name = name.strip()
        
        # Case 1: Exact match
        if name in available_configs:
            self.logger.debug(f"Selected configuration by exact match: {name}")
            return name
        
        # Case 2: Partial matching
        partial_matches = self._find_partial_matches(name, available_configs)
        
        if len(partial_matches) == 1:
            selected = partial_matches[0]
            self.logger.debug(f"Selected configuration by partial match '{name}' -> '{selected}'")
            return selected
        elif len(partial_matches) > 1:
            raise ConfigurationError(
                f"Ambiguous configuration name '{name}'. Multiple matches found.",
                available_configs=available_configs,
                invalid_config=name,
                suggestions=[
                    f"Matching configurations: {', '.join(partial_matches)}",
                    "Use a more specific name to avoid ambiguity",
                    "Use exact configuration name or index"
                ]
            )
        
        # Case 3: No matches - provide helpful suggestions
        close_matches = get_close_matches(name, available_configs, n=3, cutoff=0.3)
        
        suggestions = [f"Available configurations: {', '.join(available_configs)}"]
        if close_matches:
            suggestions.insert(0, f"Did you mean one of: {', '.join(close_matches)}?")
        
        raise ConfigurationError(
            f"Configuration '{name}' not found for catalog '{self._file_structure.catalog_name}'",
            available_configs=available_configs,
            invalid_config=name,
            suggestions=suggestions
        )
    
    def _find_partial_matches(self, pattern: str, available_configs: List[str]) -> List[str]:
        """
        Find configurations that partially match the given pattern.
        
        Parameters
        ----------
        pattern : str
            Pattern to match against
        available_configs : List[str]
            List of available configuration names
            
        Returns
        -------
        List[str]
            List of configurations that match the pattern
        """
        pattern_lower = pattern.lower()
        matches = []
        
        for config in available_configs:
            config_lower = config.lower()
            
            # Check for substring match
            if pattern_lower in config_lower:
                matches.append(config)
                continue
            
            # Check for word boundary matches (more flexible)
            # Split on common separators and check if pattern matches any part
            config_parts = re.split(r'[_\-\.]', config_lower)
            if any(part.startswith(pattern_lower) for part in config_parts):
                matches.append(config)
        
        return matches
    
    def resolve_configuration(self, model_config: Union[str, int, None]) -> Tuple[str, ConfigurationInfo]:
        """
        Resolve configuration specification to name and detailed information.
        
        This is a convenience method that combines selection and information retrieval.
        
        Parameters
        ----------
        model_config : str, int, or None
            Configuration specification
            
        Returns
        -------
        Tuple[str, ConfigurationInfo]
            Selected configuration name and detailed information
            
        Raises
        ------
        ConfigurationError
            If configuration selection fails
        """
        self._ensure_initialized()
        
        # Select configuration
        config_name = self.select_configuration(model_config)
        
        # Get detailed information
        config_info = self.get_configuration_info(config_name)
        
        if config_info is None:
            # This shouldn't happen if selection worked, but be defensive
            raise ConfigurationError(
                f"Configuration information not available for '{config_name}'",
                suggestions=["Try re-initializing the ConfigurationManager"]
            )
        
        return config_name, config_info
    
    def get_configuration_info(self, config_name: str) -> Optional[ConfigurationInfo]:
        """
        Get detailed information about a specific configuration.
        
        Parameters
        ----------
        config_name : str
            Name of the configuration
            
        Returns
        -------
        ConfigurationInfo or None
            Configuration information if found, None otherwise
        """
        self._ensure_initialized()
        return self._configurations.get(config_name)
    
    def validate_configuration(self, config_name: str) -> bool:
        """
        Validate that a configuration exists and is accessible.
        
        Parameters
        ----------
        config_name : str
            Name of the configuration to validate
            
        Returns
        -------
        bool
            True if configuration is valid and accessible, False otherwise
        """
        self._ensure_initialized()
        
        if config_name not in self._configurations:
            return False
        
        config_info = self._configurations[config_name]
        
        # Check that the HDF5 file exists and is readable
        try:
            file_path = Path(config_info.file_path)
            return file_path.exists() and file_path.is_file()
        except (OSError, PermissionError):
            return False
    
    def get_configuration_summary(self) -> Dict[str, Any]:
        """
        Get summary information about all configurations.
        
        Returns
        -------
        Dict[str, Any]
            Summary information including counts, sizes, and metadata
        """
        self._ensure_initialized()
        
        if not self._configurations:
            return {
                'catalog_name': self._file_structure.catalog_name,
                'configuration_count': 0,
                'configurations': [],
                'total_size_mb': 0.0
            }
        
        total_size = sum(config.file_size for config in self._configurations.values())
        
        config_summaries = []
        for name, config in sorted(self._configurations.items()):
            config_summaries.append({
                'name': name,
                'file_size_mb': config.get_size_mb(),
                'file_path': config.file_path,
                'accessible': self.validate_configuration(name)
            })
        
        return {
            'catalog_name': self._file_structure.catalog_name,
            'configuration_count': len(self._configurations),
            'configurations': config_summaries,
            'total_size_mb': total_size / (1024 * 1024) if total_size > 0 else 0.0,
            'output_mode': self._file_structure.output_mode
        }
    
    def update_configuration_metadata(self, config_name: str, 
                                    object_count: Optional[int] = None,
                                    parameter_count: Optional[int] = None) -> None:
        """
        Update metadata for a configuration (typically called after HDF5 loading).
        
        Parameters
        ----------
        config_name : str
            Name of the configuration
        object_count : int, optional
            Number of objects in the configuration
        parameter_count : int, optional
            Number of parameters in the configuration
        """
        self._ensure_initialized()
        
        if config_name in self._configurations:
            config = self._configurations[config_name]
            if object_count is not None:
                config.object_count = object_count
            if parameter_count is not None:
                config.parameter_count = parameter_count
            
            self.logger.debug(f"Updated metadata for {config_name}: "
                            f"{object_count} objects, {parameter_count} parameters")
    
    def suggest_similar_configurations(self, invalid_name: str, max_suggestions: int = 3) -> List[str]:
        """
        Suggest similar configuration names for an invalid input.
        
        Parameters
        ----------
        invalid_name : str
            The invalid configuration name
        max_suggestions : int, optional
            Maximum number of suggestions to return
            
        Returns
        -------
        List[str]
            List of similar configuration names
        """
        self._ensure_initialized()
        
        available_configs = list(self._configurations.keys())
        return get_close_matches(invalid_name, available_configs, 
                               n=max_suggestions, cutoff=0.3)
    
    def get_catalog_name(self) -> str:
        """
        Get the catalog name for this configuration manager.
        
        Returns
        -------
        str
            Catalog name
        """
        self._ensure_initialized()
        return self._file_structure.catalog_name
    
    def requires_explicit_selection(self) -> bool:
        """
        Check if explicit configuration selection is required.
        
        Returns
        -------
        bool
            True if multiple configurations exist and explicit selection is required
        """
        self._ensure_initialized()
        return len(self._configurations) > 1
    
    def get_auto_selection(self) -> Optional[str]:
        """
        Get auto-selected configuration if only one exists.
        
        Returns
        -------
        str or None
            Configuration name if auto-selection is possible, None otherwise
        """
        self._ensure_initialized()
        
        if len(self._configurations) == 1:
            return list(self._configurations.keys())[0]
        else:
            return None