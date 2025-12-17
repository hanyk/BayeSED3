"""
Parameter management component for BayeSEDResults redesign.

This module provides efficient parameter name access, renaming, and standardization
functionality, building on existing parameter management patterns while adding
scope-aware optimization and improved consistency handling.
"""

import os
from pathlib import Path
from typing import List, Dict, Optional, Union, Any
import logging

from .base import BaseComponent
from .models import AccessScope
from .exceptions import DataLoadingError, ValidationError


class ParameterManager(BaseComponent):
    """
    Manages parameter access, renaming, and labeling with scope-aware optimization.
    
    This component builds on existing parameter management patterns from the current
    BayeSEDResults implementation while adding:
    - Scope-aware caching strategies
    - Improved parameter standardization utilities
    - Enhanced consistency validation
    - Better integration with the component architecture
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize ParameterManager.
        
        Parameters
        ----------
        logger : logging.Logger, optional
            Logger instance for this component
        """
        super().__init__(logger)
        self._parameter_names_cache: Dict[str, List[str]] = {}
        self._renamed_parameters: Dict[str, str] = {}
        self._renamed_parameter_names: Optional[List[str]] = None  # Ordered list like old version
        self._custom_labels: Dict[str, str] = {}
        self._standardization_mapping: Dict[str, str] = {}
        self._paramnames_files: Dict[str, str] = {}
        
    def initialize(self, paramnames_files: Dict[str, str], access_scope: AccessScope) -> None:
        """
        Initialize the ParameterManager with paramnames files and access scope.
        
        Parameters
        ----------
        paramnames_files : Dict[str, str]
            Dictionary mapping object_base to paramnames file paths
        access_scope : AccessScope
            Current access scope for optimization
        """
        self._paramnames_files = paramnames_files.copy()
        self._access_scope = access_scope
        self._initialized = True
        
        self.logger.info(f"ParameterManager initialized with {len(paramnames_files)} paramnames files")
        self.logger.debug(f"Access scope: {access_scope.get_scope_description()}")
    
    def get_parameter_names(self, include_derived: bool = True, object_base: Optional[str] = None) -> List[str]:
        """
        Get list of parameter names, enhanced version of existing get_parameter_names().
        
        This method builds on the existing efficient parameter name access from paramnames
        files while adding scope-aware caching and better error handling.
        
        Parameters
        ----------
        include_derived : bool, default True
            Whether to include derived parameters (marked with *)
        object_base : str, optional
            Specific object to get parameters for. If None, uses first available
            
        Returns
        -------
        List[str]
            List of parameter names (derived parameters have * suffix removed)
            
        Examples
        --------
        >>> param_mgr = ParameterManager()
        >>> param_mgr.initialize(paramnames_files, scope)
        >>> all_params = param_mgr.get_parameter_names()
        >>> free_only = param_mgr.get_parameter_names(include_derived=False)
        """
        self._ensure_initialized()
        
        # Determine which object to use
        if object_base is None:
            if not self._paramnames_files:
                raise DataLoadingError("No paramnames files available")
            object_base = list(self._paramnames_files.keys())[0]
        
        # Check cache first
        cache_key = f"{object_base}_{include_derived}"
        if cache_key in self._parameter_names_cache:
            self.logger.debug(f"Using cached parameter names for {object_base}")
            return self._parameter_names_cache[cache_key]
        
        # Load parameter names from file
        param_names = self._get_parameter_names_from_files(object_base)
        
        # Filter based on include_derived BEFORE renaming (so we can distinguish free vs derived)
        if include_derived:
            # Keep all parameters (both free and derived)
            filtered_names = param_names
        else:
            # Only free parameters (no * suffix)
            filtered_names = [p for p in param_names if not p.endswith('*')]
        
        # Apply renaming if configured
        if self._renamed_parameters:
            filtered_names = self._apply_renaming(filtered_names)
        
        # Remove * suffix from derived parameters for consistent interface (after renaming)
        if include_derived:
            result = [p.rstrip('*') for p in filtered_names]
        else:
            # Already filtered to free parameters, but remove * if any remain
            result = [p.rstrip('*') for p in filtered_names]
        
        # Cache the result
        self._parameter_names_cache[cache_key] = result
        
        self.logger.debug(f"Loaded {len(result)} parameter names for {object_base}")
        return result
    
    def get_free_parameters(self, object_base: Optional[str] = None) -> List[str]:
        """
        Get list of free (fitted) parameters, enhanced version of existing method.
        
        Parameters
        ----------
        object_base : str, optional
            Specific object to get parameters for
            
        Returns
        -------
        List[str]
            List of free parameter names
        """
        return self.get_parameter_names(include_derived=False, object_base=object_base)
    
    def get_derived_parameters(self, object_base: Optional[str] = None) -> List[str]:
        """
        Get list of derived parameters, enhanced version of existing method.
        
        Parameters
        ----------
        object_base : str, optional
            Specific object to get parameters for
            
        Returns
        -------
        List[str]
            List of derived parameter names
        """
        self._ensure_initialized()
        
        # Determine which object to use
        if object_base is None:
            if not self._paramnames_files:
                raise DataLoadingError("No paramnames files available")
            object_base = list(self._paramnames_files.keys())[0]
        
        # Load all parameter names
        param_names = self._get_parameter_names_from_files(object_base)
        
        # Apply renaming if configured
        if self._renamed_parameters:
            param_names = self._apply_renaming(param_names)
        
        # Filter to derived parameters only and remove * suffix
        derived = [p.rstrip('*') for p in param_names if p.endswith('*')]
        
        self.logger.debug(f"Found {len(derived)} derived parameters for {object_base}")
        return derived
    
    def rename_parameters(self, parameter_mapping: Dict[str, str]) -> None:
        """
        Rename parameters with enhanced consistency validation.
        
        This method builds on the existing rename_parameters() functionality while
        adding better validation and scope-aware cache management.
        
        Parameters
        ----------
        parameter_mapping : Dict[str, str]
            Dictionary mapping old parameter names to new parameter names
            
        Raises
        ------
        ValidationError
            If parameter names in mapping don't exist
        """
        self._ensure_initialized()
        
        # Validate that all old parameter names exist
        # Need to check both with and without * suffix since mapping can have either
        all_params = set()
        all_params_with_star = set()
        for object_base in self._paramnames_files:
            params = self._get_parameter_names_from_files(object_base)
            # Include both regular and derived parameter names (with and without *)
            all_params.update([p.rstrip('*') for p in params])
            all_params_with_star.update(params)  # Keep original with *
        
        invalid_params = []
        for old_name in parameter_mapping.keys():
            # Check if name exists as-is (with *) or without *
            if old_name not in all_params_with_star and old_name.rstrip('*') not in all_params:
                invalid_params.append(old_name)
        
        if invalid_params:
            available = sorted(list(all_params))[:10]  # Show first 10 for brevity
            raise ValidationError(
                f"Invalid parameter names in renaming mapping: {invalid_params}",
                suggestions=[
                    f"Available parameters include: {available}",
                    "Use get_parameter_names() to see all available parameters"
                ]
            )
        
        # Update the renaming mapping dict
        self._renamed_parameters.update(parameter_mapping)
        
        # Build ordered list of renamed parameter names (like old version)
        # Start with original parameter names from first paramnames file
        if self._paramnames_files:
            first_object_base = list(self._paramnames_files.keys())[0]
            original_names = self._get_parameter_names_from_files(first_object_base)
            
            # Create renamed list by applying mapping
            if self._renamed_parameter_names is None:
                self._renamed_parameter_names = original_names.copy()
            
            # Apply the renaming to the cached parameter names list
            for i, param_name in enumerate(self._renamed_parameter_names):
                # Handle both regular parameters and derived parameters (with * suffix)
                if param_name in parameter_mapping:
                    # Direct mapping exists (e.g., 'log(Mstar)[0,0]*' -> 'log(Mstar)')
                    self._renamed_parameter_names[i] = parameter_mapping[param_name]
                elif param_name.endswith('*') and param_name.rstrip('*') in parameter_mapping:
                    # For derived parameters, check if mapping already handles the * suffix
                    # If mapping target doesn't have *, use it as-is (GetDist strips *)
                    new_name = parameter_mapping[param_name.rstrip('*')]
                    # Only add * back if the original paramnames file had it AND
                    # the mapping target doesn't explicitly handle derived params
                    # But since GetDist strips *, we should keep the name without *
                    self._renamed_parameter_names[i] = new_name
                elif not param_name.endswith('*') and param_name in parameter_mapping:
                    # Regular parameter (no *)
                    self._renamed_parameter_names[i] = parameter_mapping[param_name]
        
        # Clear caches to force regeneration with new names
        self._parameter_names_cache.clear()
        
        self.logger.info(f"Renamed {len(parameter_mapping)} parameters")
        self.logger.debug(f"Parameter renaming mapping: {parameter_mapping}")
    
    def set_parameter_labels(self, custom_labels: Dict[str, str]) -> None:
        """
        Set custom LaTeX labels for parameters, enhanced version of existing method.
        
        Parameters
        ----------
        custom_labels : Dict[str, str]
            Dictionary mapping parameter names to LaTeX labels
        """
        self._ensure_initialized()
        
        # Validate that parameter names exist
        all_params = set()
        for object_base in self._paramnames_files:
            params = self._get_parameter_names_from_files(object_base)
            all_params.update([p.rstrip('*') for p in params])
        
        # Apply any existing renaming to the parameter names
        if self._renamed_parameters:
            renamed_params = set()
            for param in all_params:
                renamed_params.add(self._renamed_parameters.get(param, param))
            all_params = renamed_params
        
        invalid_params = []
        for param_name in custom_labels.keys():
            if param_name not in all_params:
                invalid_params.append(param_name)
        
        if invalid_params:
            self.logger.warning(f"Custom labels provided for unknown parameters: {invalid_params}")
        
        # Update custom labels
        self._custom_labels.update(custom_labels)
        
        self.logger.info(f"Set custom labels for {len(custom_labels)} parameters")
    
    def standardize_names(self, standard_mapping: Dict[str, str]) -> None:
        """
        Apply parameter standardization for cross-result comparisons.
        
        This method provides utilities for standardizing parameter names across
        different BayeSEDResults objects to enable consistent comparisons.
        
        Parameters
        ----------
        standard_mapping : Dict[str, str]
            Dictionary mapping current parameter names to standardized names
        """
        self._ensure_initialized()
        
        # Store standardization mapping separately from renaming
        self._standardization_mapping.update(standard_mapping)
        
        # Apply standardization as renaming
        self.rename_parameters(standard_mapping)
        
        self.logger.info(f"Applied standardization for {len(standard_mapping)} parameters")
    
    def get_custom_labels(self) -> Dict[str, str]:
        """
        Get current custom parameter labels.
        
        Returns
        -------
        Dict[str, str]
            Dictionary of parameter names to custom labels
        """
        return self._custom_labels.copy()
    
    def get_renamed_parameters(self) -> Dict[str, str]:
        """
        Get current parameter renaming mapping.
        
        Returns
        -------
        Dict[str, str]
            Dictionary of old names to new names
        """
        return self._renamed_parameters.copy()
    
    def get_renamed_parameter_names(self) -> Optional[List[str]]:
        """
        Get ordered list of renamed parameter names (like old version).
        
        Returns
        -------
        Optional[List[str]]
            Ordered list of renamed parameter names, or None if no renaming
        """
        return self._renamed_parameter_names.copy() if self._renamed_parameter_names else None
    
    def clear_cache(self, cache_type: Optional[str] = None) -> None:
        """
        Clear parameter caches.
        
        Parameters
        ----------
        cache_type : str, optional
            Specific cache to clear ('names', 'labels', 'renaming').
            If None, clears all caches.
        """
        if cache_type is None or cache_type == 'names':
            self._parameter_names_cache.clear()
            self.logger.debug("Cleared parameter names cache")
        
        if cache_type is None or cache_type == 'labels':
            self._custom_labels.clear()
            self.logger.debug("Cleared custom labels")
        
        if cache_type is None or cache_type == 'renaming':
            self._renamed_parameters.clear()
            self._standardization_mapping.clear()
            self.logger.debug("Cleared parameter renaming")
    
    def _get_parameter_names_from_files(self, object_base: str) -> List[str]:
        """
        Get parameter names directly from paramnames file, enhanced from existing method.
        
        This method builds on the existing efficient file reading approach while
        adding better error handling and validation.
        
        Parameters
        ----------
        object_base : str
            Object base name to get parameters for
            
        Returns
        -------
        List[str]
            List of parameter names (including * suffix for derived parameters)
        """
        if object_base not in self._paramnames_files:
            raise DataLoadingError(
                f"No paramnames file available for object '{object_base}'",
                suggestions=[f"Available objects: {list(self._paramnames_files.keys())}"]
            )
        
        paramnames_file = self._paramnames_files[object_base]
        
        if not os.path.exists(paramnames_file):
            raise DataLoadingError(
                f"Paramnames file not found: {paramnames_file}",
                suggestions=["Check that the file path is correct and accessible"]
            )
        
        try:
            with open(paramnames_file, 'r') as f:
                param_names = []
                for line in f:
                    line = line.strip()
                    if line:
                        # Extract parameter name (first column)
                        param_name = line.split()[0]
                        param_names.append(param_name)
            
            self.logger.debug(f"Read {len(param_names)} parameter names from {paramnames_file}")
            return param_names
            
        except Exception as e:
            raise DataLoadingError(
                f"Failed to read paramnames file {paramnames_file}: {e}",
                suggestions=["Check file format and permissions"]
            )
    
    def _apply_renaming(self, param_names: List[str]) -> List[str]:
        """
        Apply parameter renaming to a list of parameter names.
        
        Parameters
        ----------
        param_names : List[str]
            Original parameter names (may include * suffix for derived parameters)
            
        Returns
        -------
        List[str]
            Parameter names with renaming applied (preserves * suffix if original had it)
        """
        renamed = []
        for param_name in param_names:
            is_derived = param_name.endswith('*')
            base_name = param_name.rstrip('*') if is_derived else param_name
            
            # Check both with and without * in mapping
            if param_name in self._renamed_parameters:
                # Direct mapping exists (e.g., 'log(Mstar)[0,0]*' -> 'log(Mstar)')
                new_name = self._renamed_parameters[param_name]
                # Preserve * suffix if original had it (for filtering purposes)
                renamed.append(new_name + '*' if is_derived else new_name)
            elif base_name in self._renamed_parameters:
                # Mapping exists for base name
                new_name = self._renamed_parameters[base_name]
                # Preserve * suffix if original had it (for filtering purposes)
                renamed.append(new_name + '*' if is_derived else new_name)
            else:
                renamed.append(param_name)
        
        return renamed
    
    def validate_parameter_consistency(self, other_manager: 'ParameterManager') -> List[str]:
        """
        Validate parameter consistency between two ParameterManager instances.
        
        This method helps identify parameter naming differences that might
        prevent proper comparison between different BayeSEDResults objects.
        
        Parameters
        ----------
        other_manager : ParameterManager
            Another ParameterManager to compare against
            
        Returns
        -------
        List[str]
            List of inconsistency messages (empty if consistent)
        """
        self._ensure_initialized()
        other_manager._ensure_initialized()
        
        inconsistencies = []
        
        # Get parameter sets from both managers
        try:
            self_params = set(self.get_parameter_names())
            other_params = set(other_manager.get_parameter_names())
        except Exception as e:
            inconsistencies.append(f"Failed to compare parameter names: {e}")
            return inconsistencies
        
        # Check for missing parameters
        missing_in_other = self_params - other_params
        missing_in_self = other_params - self_params
        
        if missing_in_other:
            inconsistencies.append(f"Parameters in first but not second: {sorted(missing_in_other)}")
        
        if missing_in_self:
            inconsistencies.append(f"Parameters in second but not first: {sorted(missing_in_self)}")
        
        # Check parameter counts
        if len(self_params) != len(other_params):
            inconsistencies.append(
                f"Different parameter counts: {len(self_params)} vs {len(other_params)}"
            )
        
        return inconsistencies