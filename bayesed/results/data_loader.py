"""
DataLoader component for BayeSEDResults redesign.

This module implements efficient data loading and caching functionality with
scope-aware optimization for both sample-level and object-level access patterns.
"""

import os
import h5py
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Union, Tuple, Any, Set
import logging
from collections import defaultdict
import weakref
import gc

from .base import BaseComponent
from .models import FileStructure, AccessScope, LoadedDataInfo, ConfigurationInfo
from .exceptions import DataLoadingError, ValidationError
from .logger import get_logger


class DataLoader(BaseComponent):
    """
    Enhanced data loading component with scope-aware caching and lazy loading.
    
    This component builds on existing BayeSEDResults loading patterns while adding
    intelligent caching strategies, scope-aware optimization, and memory management.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize DataLoader component."""
        if logger is None:
            bayesed_logger = get_logger(__name__)
            logger = bayesed_logger.get_logger()
        super().__init__(logger)
        
        # Core data structures
        self._file_structure: Optional[FileStructure] = None
        self._configuration_info: Optional[Any] = None
        self._access_scope: Optional[AccessScope] = None
        
        # Cache management
        self._hdf5_cache: Dict[str, Any] = {}
        self._samples_cache: Dict[str, Any] = {}
        self._spectra_cache: Dict[str, Any] = {}
        self._parameter_cache: Dict[str, Any] = {}
        
        # Cache metadata
        self._cache_stats: Dict[str, Dict[str, int]] = {
            'hdf5': {'hits': 0, 'misses': 0},
            'samples': {'hits': 0, 'misses': 0},
            'spectra': {'hits': 0, 'misses': 0},
            'parameters': {'hits': 0, 'misses': 0}
        }
        
        # Track which files have been logged to avoid duplicate messages
        self._logged_files: Set[str] = set()
        
        # Memory management
        self._memory_limit_mb: Optional[float] = None
        self._auto_cleanup: bool = True
    
    def initialize(self, file_structure: FileStructure, 
                   configuration_info: Optional[Any] = None,
                   access_scope: Optional[AccessScope] = None, **kwargs) -> None:
        """
        Initialize the DataLoader with file structure, configuration info, and access scope.
        
        Parameters
        ----------
        file_structure : FileStructure
            File structure information from FileDiscovery
        configuration_info : ConfigurationInfo, optional
            Configuration information including file paths
        access_scope : AccessScope, optional
            Access scope for scope-aware optimization
        **kwargs
            Additional initialization parameters including:
            - memory_limit_mb: Memory limit for caching (default: None)
            - auto_cleanup: Enable automatic cache cleanup (default: True)
        """
        self._file_structure = file_structure
        self._configuration_info = configuration_info
        self._access_scope = access_scope or AccessScope(
            scope_type="sample",
            catalog_name=file_structure.catalog_name,
            total_objects=0
        )
        
        # Configuration options
        self._memory_limit_mb = kwargs.get('memory_limit_mb')
        self._auto_cleanup = kwargs.get('auto_cleanup', True)
        
        # Clear existing caches
        self.clear_all_caches()
        
        self._initialized = True
        self.logger.debug(f"DataLoader initialized for catalog '{file_structure.catalog_name}' "
                         f"with scope '{self._access_scope.scope_type}'")
    
    def load_hdf5_data(self, file_path: str, access_scope: Optional[AccessScope] = None,
                       object_filter: Optional[Union[str, List[str]]] = None) -> Dict[str, Any]:
        """
        Load HDF5 data with scope-aware caching and filtering.
        
        Enhanced version of existing load_hdf5_results() with scope awareness.
        
        Parameters
        ----------
        file_path : str
            Path to HDF5 file
        access_scope : AccessScope, optional
            Specific access scope for this operation
        object_filter : str or List[str], optional
            Object IDs to filter (for object-level access)
            
        Returns
        -------
        Dict[str, Any]
            Loaded HDF5 data, potentially filtered by scope
            
        Raises
        ------
        DataLoadingError
            If HDF5 file cannot be loaded
        """
        self._ensure_initialized()
        
        # Use provided scope or default
        scope = access_scope or self._access_scope
        
        # Create cache key based on file path and scope
        cache_key = self._create_cache_key('hdf5', file_path, scope, object_filter)
        
        # Check cache first
        if cache_key in self._hdf5_cache:
            self._cache_stats['hdf5']['hits'] += 1
            self.logger.debug(f"HDF5 cache hit for: {Path(file_path).name}")
            return self._hdf5_cache[cache_key]
        
        self._cache_stats['hdf5']['misses'] += 1
        
        try:
            # Load HDF5 data
            self.logger.info(f"Loading HDF5 data from: {Path(file_path).name}")
            
            with h5py.File(file_path, 'r') as f:
                data = {}
                
                # Load all datasets
                for key in f.keys():
                    try:
                        dataset = f[key]
                        if isinstance(dataset, h5py.Dataset):
                            data[key] = dataset[:]
                        elif isinstance(dataset, h5py.Group):
                            # Handle groups recursively
                            data[key] = self._load_hdf5_group(dataset)
                    except Exception as e:
                        self.logger.warning(f"Failed to load dataset '{key}': {e}")
                
                # Apply object filtering if specified
                if object_filter is not None:
                    data = self._filter_hdf5_data(data, object_filter)
                elif scope.is_object_level() and scope.object_filter:
                    data = self._filter_hdf5_data(data, scope.object_filter)
            
            # Cache the result
            self._hdf5_cache[cache_key] = data
            
            # Check memory usage and cleanup if needed
            if self._auto_cleanup:
                self._check_memory_usage()
            
            self.logger.debug(f"Loaded HDF5 data: {len(data)} datasets, "
                            f"cache size: {len(self._hdf5_cache)}")
            
            return data
            
        except Exception as e:
            raise DataLoadingError(
                f"Failed to load HDF5 file: {file_path}",
                file_path=file_path,
                data_type="HDF5",
                suggestions=[
                    "Check that the file exists and is readable",
                    "Verify the HDF5 file is not corrupted",
                    "Ensure sufficient memory is available"
                ]
            ) from e
    
    def load_posterior_samples(self, object_base: str, access_scope: Optional[AccessScope] = None,
                              config_name: Optional[str] = None) -> Optional[Any]:
        """
        Load posterior samples with scope-aware caching.
        
        Enhanced version of existing get_posterior_samples() with caching and scope awareness.
        
        Parameters
        ----------
        object_base : str
            Base name for sample files (without extensions)
        access_scope : AccessScope, optional
            Specific access scope for this operation
        config_name : str, optional
            Configuration name for cache organization
            
        Returns
        -------
        Any or None
            Loaded posterior samples (GetDist samples object) or None if not found
        """
        self._ensure_initialized()
        
        scope = access_scope or self._access_scope
        cache_key = self._create_cache_key('samples', object_base, scope, config_name)
        
        # Check cache first
        if cache_key in self._samples_cache:
            self._cache_stats['samples']['hits'] += 1
            self.logger.debug(f"Samples cache hit for: {Path(object_base).name}")
            return self._samples_cache[cache_key]
        
        self._cache_stats['samples']['misses'] += 1
        
        try:
            # Look for sample files
            paramnames_file = f"{object_base}_sample_par.paramnames"
            samples_file = f"{object_base}_sample_par.txt"
            
            if not (Path(paramnames_file).exists() and Path(samples_file).exists()):
                self.logger.debug(f"Sample files not found for: {object_base}")
                return None
            
            self.logger.info(f"Loading posterior samples from: {Path(object_base).name}")
            
            # Try to import GetDist for sample loading
            try:
                import getdist
                samples = getdist.loadMCSamples(object_base + "_sample_par")
                
                # Cache the result
                self._samples_cache[cache_key] = samples
                
                self.logger.debug(f"Loaded GetDist samples: {samples.numrows} samples, "
                                f"{samples.n} parameters")
                
                return samples
                
            except ImportError:
                self.logger.warning("GetDist not available, loading raw sample data")
                
                # Fallback: load raw data
                samples_data = {
                    'samples': np.loadtxt(samples_file),
                    'paramnames': self._load_paramnames(paramnames_file)
                }
                
                self._samples_cache[cache_key] = samples_data
                return samples_data
                
        except Exception as e:
            self.logger.warning(f"Failed to load posterior samples for {object_base}: {e}")
            return None
    
    def load_bestfit_spectrum(self, object_id: str, config_name: Optional[str] = None,
                             access_scope: Optional[AccessScope] = None) -> Optional[Dict[str, Any]]:
        """
        Load best-fit spectrum data with caching.
        
        Enhanced version of existing get_bestfit_spectrum() with caching.
        
        Parameters
        ----------
        object_id : str
            Object identifier
        config_name : str, optional
            Configuration name
        access_scope : AccessScope, optional
            Access scope for this operation
            
        Returns
        -------
        Dict[str, Any] or None
            Loaded spectrum data or None if not found
        """
        self._ensure_initialized()
        
        scope = access_scope or self._access_scope
        cache_key = self._create_cache_key('spectra', object_id, scope, config_name)
        
        # Check cache first
        if cache_key in self._spectra_cache:
            self._cache_stats['spectra']['hits'] += 1
            self.logger.debug(f"Spectrum cache hit for: {object_id}")
            return self._spectra_cache[cache_key]
        
        self._cache_stats['spectra']['misses'] += 1
        
        try:
            # Find bestfit files for this object
            bestfit_files = self._find_bestfit_files(object_id, config_name)
            
            if not bestfit_files:
                self.logger.debug(f"No bestfit files found for object: {object_id}")
                return None
            
            self.logger.info(f"Loading bestfit spectrum for object: {object_id}")
            
            # Load FITS data
            spectrum_data = {}
            
            try:
                from astropy.io import fits
                
                for fits_file in bestfit_files:
                    with fits.open(fits_file) as hdul:
                        for i, hdu in enumerate(hdul):
                            if hdu.data is not None:
                                spectrum_data[f'hdu_{i}'] = {
                                    'data': hdu.data,
                                    'header': dict(hdu.header) if hdu.header else {}
                                }
                
            except ImportError:
                self.logger.warning("Astropy not available, cannot load FITS files")
                return None
            
            # Cache the result
            self._spectra_cache[cache_key] = spectrum_data
            
            self.logger.debug(f"Loaded spectrum data: {len(spectrum_data)} HDUs")
            
            return spectrum_data
            
        except Exception as e:
            self.logger.warning(f"Failed to load bestfit spectrum for {object_id}: {e}")
            return None
    
    def get_parameter_values(self, param_name: str, access_scope: Optional[AccessScope] = None,
                            config_name: Optional[str] = None) -> Optional[np.ndarray]:
        """
        Get parameter values with scope-aware access and caching.
        
        Enhanced parameter access with scope awareness for efficient data retrieval.
        
        Parameters
        ----------
        param_name : str
            Name of the parameter
        access_scope : AccessScope, optional
            Access scope for filtering
        config_name : str, optional
            Configuration name
            
        Returns
        -------
        np.ndarray or None
            Parameter values array or None if not found
        """
        self._ensure_initialized()
        
        scope = access_scope or self._access_scope
        cache_key = self._create_cache_key('parameters', param_name, scope, config_name)
        
        # Check cache first
        if cache_key in self._parameter_cache:
            self._cache_stats['parameters']['hits'] += 1
            return self._parameter_cache[cache_key]
        
        self._cache_stats['parameters']['misses'] += 1
        
        try:
            # Load from HDF5 data
            if config_name and config_name in self._file_structure.hdf5_files:
                hdf5_file = self._file_structure.hdf5_files[config_name]
                hdf5_data = self.load_hdf5_data(hdf5_file, scope)
                
                if param_name in hdf5_data:
                    param_values = hdf5_data[param_name]
                    
                    # Cache the result
                    self._parameter_cache[cache_key] = param_values
                    
                    self.logger.debug(f"Retrieved parameter '{param_name}': {len(param_values)} values")
                    return param_values
            
            self.logger.debug(f"Parameter '{param_name}' not found")
            return None
            
        except Exception as e:
            self.logger.warning(f"Failed to get parameter values for '{param_name}': {e}")
            return None
    
    def get_object_view(self, object_id: str) -> 'DataLoader':
        """
        Create an object-level view of the data loader.
        
        New method building on existing object filtering patterns.
        
        Parameters
        ----------
        object_id : str
            Object ID to filter to
            
        Returns
        -------
        DataLoader
            New DataLoader instance configured for object-level access
        """
        self._ensure_initialized()
        
        # Create object-level access scope
        object_scope = AccessScope(
            scope_type="object",
            catalog_name=self._access_scope.catalog_name,
            object_filter=object_id,
            total_objects=self._access_scope.total_objects,
            filtered_objects=1
        )
        
        # Create new DataLoader instance with object scope
        object_loader = DataLoader(self.logger)
        object_loader.initialize(self._file_structure, object_scope)
        
        self.logger.debug(f"Created object view for: {object_id}")
        
        return object_loader
    
    def clear_cache(self, cache_type: Optional[str] = None, 
                   scope: Optional[AccessScope] = None) -> None:
        """
        Clear caches with optional type and scope filtering.
        
        Enhanced version of existing clear_samples_cache() with scope awareness.
        
        Parameters
        ----------
        cache_type : str, optional
            Type of cache to clear ('hdf5', 'samples', 'spectra', 'parameters')
            If None, clears all caches
        scope : AccessScope, optional
            Clear only caches matching this scope
        """
        self._ensure_initialized()
        
        if cache_type is None:
            # Clear all caches
            cache_types = ['hdf5', 'samples', 'spectra', 'parameters']
        else:
            cache_types = [cache_type]
        
        cleared_count = 0
        
        for ctype in cache_types:
            cache_dict = getattr(self, f'_{ctype}_cache', {})
            
            if scope is None:
                # Clear entire cache
                cleared_count += len(cache_dict)
                cache_dict.clear()
            else:
                # Clear only entries matching scope
                keys_to_remove = []
                for key in cache_dict.keys():
                    if self._cache_key_matches_scope(key, scope):
                        keys_to_remove.append(key)
                
                for key in keys_to_remove:
                    del cache_dict[key]
                    cleared_count += 1
        
        # Force garbage collection
        gc.collect()
        
        self.logger.info(f"Cleared {cleared_count} cache entries" +
                        (f" for cache type: {cache_type}" if cache_type else "") +
                        (f" matching scope: {scope.scope_type}" if scope else ""))
    
    def clear_all_caches(self) -> None:
        """Clear all caches and reset cache statistics."""
        self._hdf5_cache.clear()
        self._samples_cache.clear()
        self._spectra_cache.clear()
        self._parameter_cache.clear()
        self._logged_files.clear()
        
        # Reset statistics
        for cache_type in self._cache_stats:
            self._cache_stats[cache_type] = {'hits': 0, 'misses': 0}
        
        gc.collect()
        self.logger.debug("Cleared all caches and reset statistics")
    
    # Cache management and memory optimization methods
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive cache statistics.
        
        Returns
        -------
        Dict[str, Any]
            Cache statistics including hit rates, sizes, and memory usage
        """
        stats = {
            'cache_sizes': {
                'hdf5': len(self._hdf5_cache),
                'samples': len(self._samples_cache),
                'spectra': len(self._spectra_cache),
                'parameters': len(self._parameter_cache)
            },
            'hit_rates': {},
            'total_operations': {},
            'memory_usage_mb': self._estimate_cache_memory_usage()
        }
        
        # Calculate hit rates
        for cache_type, cache_stats in self._cache_stats.items():
            total_ops = cache_stats['hits'] + cache_stats['misses']
            hit_rate = cache_stats['hits'] / total_ops if total_ops > 0 else 0.0
            
            stats['hit_rates'][cache_type] = hit_rate
            stats['total_operations'][cache_type] = total_ops
        
        return stats
    
    def set_memory_limit(self, limit_mb: Optional[float]) -> None:
        """
        Set memory limit for caching.
        
        Parameters
        ----------
        limit_mb : float or None
            Memory limit in megabytes, None for no limit
        """
        self._memory_limit_mb = limit_mb
        self.logger.debug(f"Set memory limit: {limit_mb} MB" if limit_mb else "Removed memory limit")
        
        # Check current usage against new limit
        if limit_mb and self._auto_cleanup:
            self._check_memory_usage()
    
    def optimize_cache_for_scope(self, target_scope: AccessScope) -> None:
        """
        Optimize cache contents for a specific access scope.
        
        Parameters
        ----------
        target_scope : AccessScope
            Target scope to optimize for
        """
        self._ensure_initialized()
        
        if target_scope.is_object_level():
            # For object-level access, clear sample-level caches
            self._clear_sample_level_caches()
            self.logger.debug("Optimized cache for object-level access")
        else:
            # For sample-level access, clear object-specific caches
            self._clear_object_level_caches()
            self.logger.debug("Optimized cache for sample-level access")
    
    # Private helper methods
    
    def _create_cache_key(self, cache_type: str, identifier: str, 
                         scope: AccessScope, extra: Optional[str] = None) -> str:
        """Create a cache key incorporating scope information."""
        key_parts = [
            cache_type,
            identifier,
            scope.scope_type,
            scope.object_filter or "all",
            extra or "default"
        ]
        return "|".join(str(part) for part in key_parts)
    
    def _cache_key_matches_scope(self, cache_key: str, scope: AccessScope) -> bool:
        """Check if a cache key matches a given scope."""
        key_parts = cache_key.split("|")
        if len(key_parts) >= 4:
            key_scope_type = key_parts[2]
            key_object_filter = key_parts[3]
            
            return (key_scope_type == scope.scope_type and 
                   key_object_filter == (scope.object_filter or "all"))
        return False
    
    def _load_hdf5_group(self, group: h5py.Group) -> Dict[str, Any]:
        """Recursively load HDF5 group contents."""
        group_data = {}
        
        for key in group.keys():
            item = group[key]
            if isinstance(item, h5py.Dataset):
                try:
                    group_data[key] = item[:]
                except Exception as e:
                    self.logger.warning(f"Failed to load dataset '{key}' in group: {e}")
            elif isinstance(item, h5py.Group):
                group_data[key] = self._load_hdf5_group(item)
        
        return group_data
    
    def _filter_hdf5_data(self, data: Dict[str, Any], 
                         object_filter: Union[str, List[str]]) -> Dict[str, Any]:
        """Filter HDF5 data by object IDs."""
        if isinstance(object_filter, str):
            object_filter = [object_filter]
        
        # Find object_id dataset
        if 'object_id' not in data:
            self.logger.warning("Cannot filter HDF5 data: no 'object_id' dataset found")
            return data
        
        object_ids = data['object_id']
        
        # Convert to strings for comparison
        if hasattr(object_ids, 'decode'):
            object_ids = [oid.decode() if hasattr(oid, 'decode') else str(oid) 
                         for oid in object_ids]
        else:
            object_ids = [str(oid) for oid in object_ids]
        
        # Find indices of matching objects
        indices = []
        for i, oid in enumerate(object_ids):
            if oid in object_filter:
                indices.append(i)
        
        if not indices:
            self.logger.warning(f"No objects found matching filter: {object_filter}")
            return {}
        
        # Filter all datasets
        filtered_data = {}
        for key, dataset in data.items():
            if isinstance(dataset, np.ndarray) and len(dataset) == len(object_ids):
                filtered_data[key] = dataset[indices]
            else:
                filtered_data[key] = dataset
        
        self.logger.debug(f"Filtered HDF5 data: {len(indices)} objects selected")
        
        return filtered_data
    
    def _load_paramnames(self, paramnames_file: str) -> List[str]:
        """Load parameter names from paramnames file."""
        try:
            with open(paramnames_file, 'r') as f:
                paramnames = []
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        # Extract parameter name (first part before any whitespace)
                        param_name = line.split()[0] if line.split() else line
                        paramnames.append(param_name)
                return paramnames
        except Exception as e:
            self.logger.warning(f"Failed to load paramnames from {paramnames_file}: {e}")
            return []
    
    def _find_bestfit_files(self, object_id: str, config_name: Optional[str] = None) -> List[str]:
        """Find bestfit FITS files for an object."""
        bestfit_files = []
        
        if not self._file_structure.bestfit_files:
            return bestfit_files
        
        if object_id in self._file_structure.bestfit_files:
            obj_files = self._file_structure.bestfit_files[object_id]
            
            # Handle the actual data structure from FileDiscovery
            # obj_files is a list of file paths, not a dict by config
            if isinstance(obj_files, list):
                # If config_name is specified, filter files by config name
                if config_name:
                    bestfit_files = [f for f in obj_files if config_name in f]
                else:
                    bestfit_files = obj_files[:]
            elif isinstance(obj_files, dict):
                # Handle dict structure if it exists
                if config_name and config_name in obj_files:
                    bestfit_files.extend(obj_files[config_name])
                else:
                    # Return files from all configurations
                    for config_files in obj_files.values():
                        bestfit_files.extend(config_files)
        
        return bestfit_files
    
    def _estimate_cache_memory_usage(self) -> float:
        """Estimate total cache memory usage in MB."""
        total_size = 0
        
        # Estimate HDF5 cache size
        for data in self._hdf5_cache.values():
            total_size += self._estimate_dict_size(data)
        
        # Estimate samples cache size
        for samples in self._samples_cache.values():
            if hasattr(samples, 'samples'):  # GetDist samples
                total_size += samples.samples.nbytes if hasattr(samples.samples, 'nbytes') else 0
            elif isinstance(samples, dict) and 'samples' in samples:
                total_size += samples['samples'].nbytes if hasattr(samples['samples'], 'nbytes') else 0
        
        # Estimate spectra cache size
        for spectrum_data in self._spectra_cache.values():
            total_size += self._estimate_dict_size(spectrum_data)
        
        # Estimate parameter cache size
        for param_data in self._parameter_cache.values():
            if hasattr(param_data, 'nbytes'):
                total_size += param_data.nbytes
        
        return total_size / (1024 * 1024)  # Convert to MB
    
    def _estimate_dict_size(self, data_dict: Dict[str, Any]) -> int:
        """Estimate memory size of a dictionary containing numpy arrays."""
        total_size = 0
        
        for value in data_dict.values():
            if isinstance(value, np.ndarray):
                total_size += value.nbytes
            elif isinstance(value, dict):
                total_size += self._estimate_dict_size(value)
            elif isinstance(value, (list, tuple)):
                # Rough estimate for lists/tuples
                total_size += len(value) * 8  # Assume 8 bytes per item
        
        return total_size
    
    def _check_memory_usage(self) -> None:
        """Check memory usage and cleanup if necessary."""
        if not self._memory_limit_mb:
            return
        
        current_usage = self._estimate_cache_memory_usage()
        
        if current_usage > self._memory_limit_mb:
            self.logger.warning(f"Cache memory usage ({current_usage:.1f} MB) "
                              f"exceeds limit ({self._memory_limit_mb} MB)")
            
            # Cleanup strategy: remove oldest entries from largest cache
            self._cleanup_largest_cache()
    
    def _cleanup_largest_cache(self) -> None:
        """Clean up the largest cache to free memory."""
        cache_sizes = {
            'hdf5': len(self._hdf5_cache),
            'samples': len(self._samples_cache),
            'spectra': len(self._spectra_cache),
            'parameters': len(self._parameter_cache)
        }
        
        # Find largest cache
        largest_cache = max(cache_sizes.items(), key=lambda x: x[1])
        cache_type = largest_cache[0]
        
        if largest_cache[1] > 0:
            # Clear half of the largest cache
            cache_dict = getattr(self, f'_{cache_type}_cache')
            keys_to_remove = list(cache_dict.keys())[:len(cache_dict) // 2]
            
            for key in keys_to_remove:
                del cache_dict[key]
            
            self.logger.info(f"Cleaned up {len(keys_to_remove)} entries from {cache_type} cache")
            
            # Force garbage collection
            gc.collect()
    
    def _clear_sample_level_caches(self) -> None:
        """Clear caches that are specific to sample-level access."""
        keys_to_remove = []
        
        for cache_dict in [self._hdf5_cache, self._parameter_cache]:
            for key in cache_dict.keys():
                if "|sample|" in key:
                    keys_to_remove.append((cache_dict, key))
        
        for cache_dict, key in keys_to_remove:
            del cache_dict[key]
    
    def _clear_object_level_caches(self) -> None:
        """Clear caches that are specific to object-level access."""
        keys_to_remove = []
        
        for cache_dict in [self._hdf5_cache, self._samples_cache, 
                          self._spectra_cache, self._parameter_cache]:
            for key in cache_dict.keys():
                if "|object|" in key:
                    keys_to_remove.append((cache_dict, key))
        
        for cache_dict, key in keys_to_remove:
            del cache_dict[key]
    
    # Public utility methods
    
    def is_data_loaded(self, data_type: str, identifier: str) -> bool:
        """
        Check if specific data is already loaded in cache.
        
        Parameters
        ----------
        data_type : str
            Type of data ('hdf5', 'samples', 'spectra', 'parameters')
        identifier : str
            Data identifier
            
        Returns
        -------
        bool
            True if data is cached
        """
        cache_key = self._create_cache_key(data_type, identifier, self._access_scope)
        cache_dict = getattr(self, f'_{data_type}_cache', {})
        return cache_key in cache_dict
    
    def get_loaded_data_info(self) -> LoadedDataInfo:
        """
        Get information about currently loaded data.
        
        Returns
        -------
        LoadedDataInfo
            Information about loaded data and cache status
        """
        self._ensure_initialized()
        
        # Create a dummy configuration info
        config_info = ConfigurationInfo(
            name="multiple" if len(self._file_structure.hdf5_files) > 1 else 
                 list(self._file_structure.hdf5_files.keys())[0],
            file_path="",
            catalog_name=self._file_structure.catalog_name
        )
        
        cache_status = {
            'hdf5': len(self._hdf5_cache) > 0,
            'samples': len(self._samples_cache) > 0,
            'spectra': len(self._spectra_cache) > 0,
            'parameters': len(self._parameter_cache) > 0
        }
        
        return LoadedDataInfo(
            configuration=config_info,
            hdf5_file=list(self._file_structure.hdf5_files.values())[0] if self._file_structure.hdf5_files else "",
            scope=self._access_scope,
            cache_status=cache_status
        )
    
    def preload_data(self, config_name: str, data_types: Optional[List[str]] = None) -> None:
        """
        Preload data for a configuration to improve subsequent access speed.
        
        Parameters
        ----------
        config_name : str
            Configuration name to preload
        data_types : List[str], optional
            Types of data to preload ('hdf5', 'samples', 'spectra')
            If None, preloads HDF5 data only
        """
        self._ensure_initialized()
        
        if data_types is None:
            data_types = ['hdf5']
        
        if config_name not in self._file_structure.hdf5_files:
            self.logger.warning(f"Configuration '{config_name}' not found for preloading")
            return
        
        self.logger.info(f"Preloading data for configuration: {config_name}")
        
        # Preload HDF5 data
        if 'hdf5' in data_types:
            hdf5_file = self._file_structure.hdf5_files[config_name]
            self.load_hdf5_data(hdf5_file)
        
        # Preload samples if requested and available
        if 'samples' in data_types and self._file_structure.posterior_files:
            for obj_id, config_files in self._file_structure.posterior_files.items():
                if config_name in config_files:
                    # Extract base name from paramnames file
                    paramnames_file = config_files[config_name].get('paramnames', '')
                    if paramnames_file:
                        object_base = paramnames_file.replace('_sample_par.paramnames', '')
                        self.load_posterior_samples(object_base, config_name=config_name)
        
        self.logger.debug(f"Preloading complete for {config_name}")
    
    def get_access_scope(self) -> AccessScope:
        """Get the current access scope."""
        self._ensure_initialized()
        return self._access_scope
    
    def set_access_scope(self, access_scope: AccessScope) -> None:
        """
        Set a new access scope and optimize caches accordingly.
        
        Parameters
        ----------
        access_scope : AccessScope
            New access scope
        """
        self._ensure_initialized()
        
        old_scope = self._access_scope
        self._access_scope = access_scope
        
        # Optimize cache for new scope if it changed significantly
        if (old_scope.scope_type != access_scope.scope_type or 
            old_scope.object_filter != access_scope.object_filter):
            self.optimize_cache_for_scope(access_scope)
        
        self.logger.debug(f"Access scope changed from {old_scope.scope_type} to {access_scope.scope_type}")
    def get_object_count(self, hdf5_file_path: str) -> int:
        """
        Get the number of objects in an HDF5 file.
        
        Parameters
        ----------
        hdf5_file_path : str
            Path to the HDF5 file
            
        Returns
        -------
        int
            Number of objects in the file
        """
        try:
            with h5py.File(hdf5_file_path, 'r') as f:
                if 'ID' in f:
                    return len(f['ID'])
                elif 'object_id' in f:
                    return len(f['object_id'])
                else:
                    # Try to infer from parameters shape
                    if 'parameters' in f:
                        return f['parameters'].shape[0]
                    else:
                        return 0
        except Exception as e:
            raise DataLoadingError(f"Failed to read object count from {hdf5_file_path}: {e}")
    
    def get_parameter_names_from_hdf5(self, include_derived: bool = True) -> List[str]:
        """
        Get parameter names directly from HDF5 file.
        
        This method inspects the HDF5 file structure to find the correct parameter names.
        It tries multiple possible locations and formats.
        
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
        
        # Get the HDF5 file path
        if not self._configuration_info or not self._configuration_info.file_path:
            raise DataLoadingError("No HDF5 file path available")
        
        hdf5_file = self._configuration_info.file_path
        cache_key = f"param_names_{hdf5_file}_{include_derived}"
        
        if cache_key in self._parameter_cache:
            self._cache_stats['parameters']['hits'] += 1
            return self._parameter_cache[cache_key]
        
        try:
            with h5py.File(hdf5_file, 'r') as f:
                param_names = []
                
                # Debug: Print all available datasets
                self.logger.debug(f"Available datasets in HDF5 file: {list(f.keys())}")
                
                # Try different possible parameter name locations
                if 'parameters_name' in f:
                    param_names = [name.decode() if isinstance(name, bytes) else str(name) 
                                 for name in f['parameters_name'][:]]
                    self.logger.debug(f"Found parameters_name dataset with {len(param_names)} parameters")
                elif 'parameter_names' in f:
                    param_names = [name.decode() if isinstance(name, bytes) else str(name) 
                                 for name in f['parameter_names'][:]]
                    self.logger.debug(f"Found parameter_names dataset with {len(param_names)} parameters")
                elif 'param_names' in f:
                    param_names = [name.decode() if isinstance(name, bytes) else str(name) 
                                 for name in f['param_names'][:]]
                    self.logger.debug(f"Found param_names dataset with {len(param_names)} parameters")
                else:
                    # If no parameter names dataset found, try to infer from table structure
                    if 'parameters' in f:
                        # Check if we can get column names from the table structure
                        params_dataset = f['parameters']
                        self.logger.debug(f"Parameters dataset shape: {params_dataset.shape}")
                        
                        # For now, create generic parameter names
                        num_params = params_dataset.shape[1] if len(params_dataset.shape) > 1 else 0
                        param_names = [f"param_{i}" for i in range(num_params)]
                        self.logger.warning(f"No parameter names found, created {len(param_names)} generic names")
                    else:
                        raise DataLoadingError("No parameter names or parameters dataset found in HDF5 file")
                
                # The parameters_name dataset contains the base parameter names directly
                # Filter derived parameters if requested
                if not include_derived:
                    original_count = len(param_names)
                    param_names = [p for p in param_names if not p.endswith('*')]
                    self.logger.debug(f"Filtered derived parameters: {original_count} -> {len(param_names)}")
                else:
                    # Remove * suffix for consistency
                    param_names = [p.rstrip('*') for p in param_names]
                
                self._parameter_cache[cache_key] = param_names
                self._cache_stats['parameters']['misses'] += 1
                
                self.logger.debug(f"Final parameter names (first 10): {param_names[:10]}")
                return param_names
                
        except Exception as e:
            raise DataLoadingError(f"Failed to read parameter names from HDF5: {e}")
    

    
    def get_free_parameters_from_hdf5(self) -> List[str]:
        """Get free parameters from HDF5 file."""
        return self.get_parameter_names_from_hdf5(include_derived=False)
    
    def get_derived_parameters_from_hdf5(self) -> List[str]:
        """Get derived parameters from HDF5 file."""
        all_params = self.get_parameter_names_from_hdf5(include_derived=True)
        free_params = self.get_parameter_names_from_hdf5(include_derived=False)
        return [p for p in all_params if p not in free_params]
    
    def get_paramnames_files(self) -> Dict[str, str]:
        """
        Get available paramnames files for parameter management.
        
        Returns paramnames files filtered by the current access scope (object_id and configuration).
        
        Returns
        -------
        Dict[str, str]
            Dictionary mapping object_base to paramnames file paths
        """
        self._ensure_initialized()
        
        paramnames_files = {}
        
        if not self._file_structure.has_object_files():
            return paramnames_files
        
        # Get the current object_id and configuration from access scope
        object_id = None
        if self._access_scope.is_object_level() and self._access_scope.object_filter:
            object_id = self._access_scope.object_filter
        
        config_name = None
        if self._configuration_info and self._configuration_info.name:
            config_name = self._configuration_info.name
        
        # Look for paramnames files in object directories, filtered by current scope
        if self._file_structure.posterior_files:
            for obj_id, base_names_dict in self._file_structure.posterior_files.items():
                # Filter by object_id if specified
                if object_id and obj_id != object_id:
                    continue
                
                # base_names_dict is: base_name -> file_type -> path
                for base_name, file_types in base_names_dict.items():
                    # Filter by configuration name if specified
                    if config_name and config_name not in base_name:
                        continue
                    
                    if 'paramnames' in file_types:
                        # Use just the object_id as the key for cleaner logging
                        # The base_name contains the full file path which makes logging messy
                        object_base = obj_id
                        paramnames_files[object_base] = file_types['paramnames']
                        # If we found a match for the specific object and config, we're done
                        if object_id and config_name:
                            break
                
                # If we found a match for the specific object and config, we're done
                if object_id and config_name and paramnames_files:
                    break
        
        return paramnames_files
    
    def get_parameter_values(self, parameter_name: str, 
                           object_ids: Optional[List[str]] = None,
                           access_scope: Optional[AccessScope] = None) -> 'astropy.table.Table':
        """
        Get all parameter columns that start with the given parameter name.
        
        This method loads the HDF5 results table and extracts all columns
        that start with the specified parameter name, returning a sub-table
        with all statistical estimates (mean, median, percentiles, etc.).
        
        Parameters
        ----------
        parameter_name : str
            Base name of the parameter to retrieve (e.g., 'log(age/yr)[0,1]')
        object_ids : List[str], optional
            Specific object IDs to filter
        access_scope : AccessScope, optional
            Access scope for filtering
            
        Returns
        -------
        astropy.table.Table
            Sub-table containing ID column and all parameter columns that 
            start with the given parameter name
        """
        self._ensure_initialized()
        
        if access_scope is None:
            access_scope = self._access_scope
        
        # Load the HDF5 results table
        hdf5_table = self.load_hdf5_results(filter_snr=False, min_snr=0.0, access_scope=access_scope)
        
        # Find all columns that start with the parameter name
        available_columns = hdf5_table.colnames
        matching_columns = []
        
        # Look for columns that start with the parameter name
        for col in available_columns:
            if col.startswith(parameter_name):
                matching_columns.append(col)
        
        if not matching_columns:
            raise DataLoadingError(
                f"No columns found starting with '{parameter_name}' in HDF5 table",
                suggestions=[f"Available parameter prefixes: {list(set([col.split('_')[0] for col in available_columns if '_' in col and col != 'ID']))[:10]}..."]
            )
        
        # Create sub-table with ID and all matching parameter columns
        columns_to_include = ['ID'] + sorted(matching_columns)
        sub_table = hdf5_table[columns_to_include]
        
        # Apply object filtering if specified
        if object_ids is not None:
            # Filter table by object IDs
            object_mask = [obj_id in object_ids for obj_id in sub_table['ID']]
            if any(object_mask):
                filtered_table = sub_table[object_mask]
                return filtered_table
            else:
                raise DataLoadingError(f"None of the specified objects found in data")
        
        # Apply scope filtering for object-level access
        if access_scope.is_object_level() and access_scope.object_filter:
            # Find the specific object
            object_mask = sub_table['ID'] == access_scope.object_filter
            if any(object_mask):
                return sub_table[object_mask]
            else:
                raise DataLoadingError(f"Object '{access_scope.object_filter}' not found in data")
        
        self.logger.debug(f"Found {len(matching_columns)} columns for parameter '{parameter_name}': {matching_columns}")
        
        # Return sub-table for sample-level access
        return sub_table
    
    def list_objects(self, access_scope: Optional[AccessScope] = None) -> List[str]:
        """
        List available objects by filtering the loaded HDF5 table.
        
        Parameters
        ----------
        access_scope : AccessScope, optional
            Access scope for filtering
            
        Returns
        -------
        List[str]
            List of object IDs
        """
        self._ensure_initialized()
        
        if access_scope is None:
            access_scope = self._access_scope
        
        # Load the HDF5 results table
        hdf5_table = self.load_hdf5_results(filter_snr=False, min_snr=0.0, access_scope=access_scope)
        
        # Extract object IDs from the table
        object_ids = [str(obj_id) for obj_id in hdf5_table['ID']]
        
        # Apply scope filtering
        if access_scope.is_object_level() and access_scope.object_filter:
            if access_scope.object_filter in object_ids:
                return [access_scope.object_filter]
            else:
                return []
        
        return object_ids
    
    def get_cache_status(self) -> Dict[str, Any]:
        """
        Get cache status information.
        
        Returns
        -------
        Dict[str, Any]
            Cache status and statistics
        """
        return {
            'cache_sizes': {
                'hdf5': len(self._hdf5_cache),
                'samples': len(self._samples_cache),
                'spectra': len(self._spectra_cache),
                'parameters': len(self._parameter_cache)
            },
            'cache_stats': self._cache_stats.copy()
        }
    
    # ========================================================================
    # Enhanced API Methods - Superior functionality for enhanced BayeSEDResults
    # ========================================================================
    
    def get_evidence(self, object_ids: Optional[List[str]] = None,
                    access_scope: Optional[AccessScope] = None) -> Union[Dict[str, float], float]:
        """
        Get Bayesian evidence values by filtering the loaded HDF5 table.
        
        Parameters
        ----------
        object_ids : List[str], optional
            Specific object IDs to get evidence for
        access_scope : AccessScope, optional
            Access scope for filtering
            
        Returns
        -------
        Union[Dict[str, float], float]
            Evidence values
        """
        self._ensure_initialized()
        
        scope = access_scope or self._access_scope
        
        # Load the HDF5 results table
        hdf5_table = self.load_hdf5_results(filter_snr=False, min_snr=0.0, access_scope=scope)
        
        # Extract evidence columns
        evidence_columns = ['logZ', 'INSlogZ', 'logZerr', 'INSlogZerr']
        available_evidence_cols = [col for col in evidence_columns if col in hdf5_table.colnames]
        
        if not available_evidence_cols:
            raise DataLoadingError(f"No evidence columns found in HDF5 table. Available columns: {hdf5_table.colnames[:10]}...")
        
        # Apply object filtering if specified
        if object_ids is not None:
            # Filter table by object IDs
            object_mask = [obj_id in object_ids for obj_id in hdf5_table['ID']]
            if any(object_mask):
                filtered_table = hdf5_table[object_mask]
            else:
                raise DataLoadingError(f"None of the specified objects found in data")
        else:
            filtered_table = hdf5_table
        
        # Extract evidence data
        evidence_data = {}
        for col in available_evidence_cols:
            evidence_data[col] = filtered_table[col]
        
        # Return based on scope and number of objects
        if scope.is_object_level() or len(filtered_table) == 1:
            # Single object - return scalar values
            return {col: float(values[0]) for col, values in evidence_data.items()}
        else:
            # Multiple objects - return arrays
            return evidence_data
    
    def get_posterior_samples(self, object_id: Optional[str] = None,
                            access_scope: Optional[AccessScope] = None) -> Any:
        """
        Get posterior samples with enhanced scope awareness and caching.
        
        Parameters
        ----------
        object_id : str, optional
            Object ID to get samples for
        access_scope : AccessScope, optional
            Access scope for filtering
            
        Returns
        -------
        astropy.table.Table
            Posterior samples table
        """
        self._ensure_initialized()
        
        scope = access_scope or self._access_scope
        
        # Determine object to load
        if object_id is None:
            if scope.is_object_level() and scope.object_filter:
                object_id = scope.object_filter
            else:
                # For sample-level access, use first available object with warning
                objects = self.list_objects(scope)
                if objects:
                    object_id = objects[0]
                    self.logger.warning(f"No object_id provided for get_posterior_samples. Using first available object: {object_id}")
                else:
                    raise DataLoadingError("No objects available for posterior samples")
        
        # Load samples using existing method
        return self.load_posterior_samples(object_id, access_scope=scope)
    
    def get_bestfit_spectrum(self, object_id: Optional[str] = None,
                           access_scope: Optional[AccessScope] = None) -> Dict[str, Any]:
        """
        Get best-fit spectrum with enhanced validation and error handling.
        
        Parameters
        ----------
        object_id : str, optional
            Object ID to get spectrum for
        access_scope : AccessScope, optional
            Access scope for filtering
            
        Returns
        -------
        Dict[str, Any]
            Best-fit spectrum data
        """
        self._ensure_initialized()
        
        scope = access_scope or self._access_scope
        
        # Determine object to load
        if object_id is None:
            if scope.is_object_level() and scope.object_filter:
                object_id = scope.object_filter
            else:
                # For sample-level access, use first available object with warning
                objects = self.list_objects(scope)
                if objects:
                    object_id = objects[0]
                    self.logger.warning(f"No object_id provided for get_bestfit_spectrum. Using first available object: {object_id}")
                else:
                    raise DataLoadingError("No objects available for best-fit spectrum")
        
        # Load spectrum using existing method
        spectrum_data = self.load_bestfit_spectrum(object_id, access_scope=scope)
        
        if spectrum_data is None:
            raise DataLoadingError(f"No best-fit spectrum found for object '{object_id}'")
        
        return spectrum_data
    
    def load_hdf5_results(self, filter_snr: bool = True, min_snr: float = 0.0,
                         access_scope: Optional[AccessScope] = None) -> Any:
        """
        Load HDF5 results as astropy Table using simple, direct approach.
        
        This method uses a straightforward approach to load all parameters from
        the HDF5 file into an astropy Table, similar to the manual h5py approach
        but with enhanced error handling and caching.
        
        Parameters
        ----------
        filter_snr : bool, default True
            Whether to filter by SNR
        min_snr : float, default 0.0
            Minimum SNR threshold
        access_scope : AccessScope, optional
            Access scope for filtering
            
        Returns
        -------
        astropy.table.Table
            HDF5 results table with all parameters and object IDs
        """
        self._ensure_initialized()
        
        scope = access_scope or self._access_scope
        
        # Get the HDF5 file path from file structure
        if not self._file_structure or not self._file_structure.hdf5_files:
            raise DataLoadingError("No HDF5 files available")
        
        # Use the first (and typically only) HDF5 file for the current configuration
        file_path = list(self._file_structure.hdf5_files.values())[0]
        
        # Check cache first
        cache_key = f"hdf5_table_{Path(file_path).name}_{filter_snr}_{min_snr}"
        if cache_key in self._hdf5_cache:
            self._cache_stats['hdf5']['hits'] += 1
            self.logger.debug(f"HDF5 table cache hit for: {Path(file_path).name}")
            return self._hdf5_cache[cache_key]
        
        self._cache_stats['hdf5']['misses'] += 1
        
        try:
            from astropy.table import Table, hstack
            import h5py
            
            # Show relative path from current working directory, but only once per file
            try:
                relative_path = Path(file_path).relative_to(Path.cwd())
            except ValueError:
                # If file is not under current directory, show absolute path
                relative_path = Path(file_path)
            
            # Only log the loading message once per file
            if file_path not in self._logged_files:
                self.logger.info(f"Loading HDF5 results table from: {relative_path}")
                self._logged_files.add(file_path)
            
            with h5py.File(file_path, 'r') as h:
                # Debug: Print all available datasets
                self.logger.debug(f"Available datasets in HDF5 file: {list(h.keys())}")
                
                # Get parameter names and decode from bytes to strings
                colnames = []
                if 'parameters_name' in h:
                    colnames = [x.decode('utf-8') if isinstance(x, bytes) else str(x) 
                               for x in h['parameters_name'][:]]
                    self.logger.debug(f"Found parameters_name with {len(colnames)} names")
                elif 'parameter_names' in h:
                    colnames = [x.decode('utf-8') if isinstance(x, bytes) else str(x) 
                               for x in h['parameter_names'][:]]
                    self.logger.debug(f"Found parameter_names with {len(colnames)} names")
                elif 'param_names' in h:
                    colnames = [x.decode('utf-8') if isinstance(x, bytes) else str(x) 
                               for x in h['param_names'][:]]
                    self.logger.debug(f"Found param_names with {len(colnames)} names")
                else:
                    raise DataLoadingError("No parameter names dataset found in HDF5 file. Available datasets: " + str(list(h.keys())))
                
                # Create ID table
                if 'ID' in h:
                    id_table = Table([h['ID'][:]], names=['ID'])
                else:
                    raise DataLoadingError("No 'ID' dataset found in HDF5 file")
                
                # Create parameters table
                if 'parameters' in h:
                    parameters_table = Table(h['parameters'][:], names=colnames, copy=False)
                else:
                    raise DataLoadingError("No 'parameters' dataset found in HDF5 file")
                
                # Combine ID and parameters using hstack
                full_table = hstack([id_table, parameters_table])
                
                # Apply SNR filtering if requested
                if filter_snr and 'SNR' in full_table.colnames:
                    mask = full_table['SNR'] > min_snr
                    filtered_table = full_table[mask]
                    self.logger.info(f"Filtered {len(full_table)} objects to {len(filtered_table)} with SNR > {min_snr}")
                else:
                    filtered_table = full_table
                    self.logger.info(f"Loaded {len(filtered_table)} objects (no SNR filtering)")
                
                # Cache the result
                self._hdf5_cache[cache_key] = filtered_table
                
                return filtered_table
                
        except ImportError as e:
            raise DataLoadingError("astropy and h5py are required for HDF5 table loading") from e
        except Exception as e:
            raise DataLoadingError(
                f"Failed to load HDF5 results table: {file_path}",
                file_path=file_path,
                data_type="HDF5 Table",
                suggestions=[
                    "Check that the HDF5 file contains 'ID', 'parameters', and 'parameters_name' datasets",
                    "Verify the HDF5 file is not corrupted",
                    "Ensure astropy and h5py are installed"
                ]
            ) from e
    
    def get_getdist_samples(self, object_id: Optional[str] = None,
                          access_scope: Optional[AccessScope] = None,
                          parameter_manager: Optional[Any] = None) -> Any:
        """
        Get GetDist samples with enhanced caching and parameter management.
        
        Parameters
        ----------
        object_id : str, optional
            Object ID to get samples for
        access_scope : AccessScope, optional
            Access scope for filtering
        parameter_manager : ParameterManager, optional
            Parameter manager for renaming/labeling
            
        Returns
        -------
        getdist.MCSamples
            GetDist samples object
        """
        self._ensure_initialized()
        
        scope = access_scope or self._access_scope
        
        # Determine object to load
        if object_id is None:
            if scope.is_object_level() and scope.object_filter:
                object_id = scope.object_filter
            else:
                # For sample-level access, use first available object with warning
                objects = self.list_objects(scope)
                if objects:
                    object_id = objects[0]
                    self.logger.warning(f"No object_id provided for get_getdist_samples. Using first available object: {object_id}")
                else:
                    raise DataLoadingError("No objects available for GetDist samples")
        
        # Check cache first
        cache_key = f"getdist_{object_id}"
        if cache_key in self._samples_cache:
            self._cache_stats['samples']['hits'] += 1
            return self._samples_cache[cache_key]
        
        try:
            from getdist import loadMCSamples
            
            # Find sample files for this object
            sample_files = None
            if self._file_structure.posterior_files:
                for obj_id, obj_files in self._file_structure.posterior_files.items():
                    if obj_id == object_id:
                        sample_files = obj_files
                        break
            
            if not sample_files:
                # Provide helpful diagnostics
                suggestions = []
                
                # Check if object exists in HDF5 file
                try:
                    available_objects = self.list_objects(access_scope=scope)
                    if object_id not in available_objects:
                        suggestions.append(f"Object '{object_id}' not found in catalog '{scope.catalog_name}'")
                        suggestions.append(f"Available objects in catalog: {len(available_objects)} total")
                        if available_objects:
                            suggestions.append(f"First 10 objects: {available_objects[:10]}")
                    else:
                        suggestions.append(f"Object '{object_id}' exists in catalog but has no posterior sample files")
                except Exception:
                    pass  # If we can't check, just continue with file-based suggestions
                
                # List objects that DO have posterior files
                if self._file_structure.posterior_files:
                    objects_with_files = list(self._file_structure.posterior_files.keys())
                    suggestions.append(f"Objects with posterior files: {len(objects_with_files)} total")
                    if objects_with_files:
                        suggestions.append(f"First 10 objects with files: {objects_with_files[:10]}")
                else:
                    suggestions.append("No posterior sample files found in this catalog")
                    suggestions.append("Posterior samples may not have been generated for this catalog")
                
                raise DataLoadingError(
                    f"No posterior sample files found for object '{object_id}' in catalog '{scope.catalog_name}'",
                    suggestions=suggestions
                )
            
            # Filter by selected configuration if available
            # sample_files structure: base_name -> file_type -> path
            # base_name is a file path that contains the config name
            base_name_to_use = None
            if self._configuration_info and self._configuration_info.name:
                # Try to find matching config name in base_name paths
                selected_config = self._configuration_info.name
                for base_name in sample_files.keys():
                    # Check if the selected config name appears in the base_name path
                    # The base_name is a full path ending with config_name_sample_par
                    if selected_config in base_name:
                        # Extract config name from filename for more precise matching
                        base_name_config = os.path.basename(base_name).replace('_sample_par', '')
                        # Match if config name exactly matches or is contained in the filename
                        if selected_config == base_name_config or base_name_config.endswith(selected_config):
                            base_name_to_use = base_name
                            break
                
                if base_name_to_use:
                    sample_files = {base_name_to_use: sample_files[base_name_to_use]}
                    self.logger.debug(f"Selected configuration '{selected_config}' matches base_name '{base_name_to_use}'")
                else:
                    available_configs = [os.path.basename(bn).replace('_sample_par', '') for bn in sample_files.keys()]
                    self.logger.warning(
                        f"Configuration '{selected_config}' not found in posterior files for object '{object_id}'. "
                        f"Available configurations in files: {available_configs[:3]}... "
                        f"Using first available: {os.path.basename(list(sample_files.keys())[0])}"
                    )
            
            # Load samples - use the base name (without extension) for GetDist
            # sample_files is now: base_name -> file_type -> path
            base_name = base_name_to_use if base_name_to_use else list(sample_files.keys())[0]
            files = sample_files[base_name]  # Get file paths dict (file_type -> path)
            samples = loadMCSamples(base_name)
            
            # Apply parameter renaming/labeling if parameter manager is available (like old version)
            if parameter_manager and parameter_manager.is_initialized():
                renamed_param_names_list = parameter_manager.get_renamed_parameter_names()
                custom_labels = parameter_manager.get_custom_labels()
                
                # Check if we need to apply parameter renaming (like old version)
                if renamed_param_names_list is not None:
                    from getdist import MCSamples
                    
                    # Read original parameter names from paramnames file (like old version)
                    paramnames_file = files.get('paramnames')
                    if not paramnames_file:
                        # Fallback: try to construct path
                        paramnames_file = base_name + '_sample_par.paramnames'
                    
                    original_names = []
                    original_labels = []
                    if os.path.exists(paramnames_file):
                        with open(paramnames_file, 'r') as f:
                            for line in f:
                                if line.strip():
                                    parts = line.strip().split(None, 1)  # Split on first whitespace only
                                    original_names.append(parts[0])
                                    # Use label if provided, otherwise use parameter name
                                    original_labels.append(parts[1] if len(parts) > 1 else parts[0])
                    
                    # Create mapping from original to renamed (like old version)
                    rename_mapping = {}
                    renamed_labels = []
                    for i, (orig_name, new_name) in enumerate(zip(original_names, renamed_param_names_list)):
                        if orig_name != new_name:
                            rename_mapping[orig_name] = new_name
                        # Update labels to match renamed parameters
                        if i < len(original_labels):
                            label = original_labels[i]
                            # If the label was the same as the parameter name, update it
                            if label == orig_name:
                                renamed_labels.append(new_name)
                            else:
                                renamed_labels.append(label)
                        else:
                            renamed_labels.append(new_name)
                    
                    if rename_mapping:
                        # Get the actual parameter names from the loaded samples (after GetDist processing)
                        # This accounts for any parameters that GetDist may have removed (fixed parameters, etc.)
                        # Note: GetDist strips * from derived parameter names, so actual_param_names won't have *
                        actual_param_names = [param.name for param in samples.paramNames.names]
                        actual_param_labels = [param.label for param in samples.paramNames.names]
                        
                        # Create a mapping that handles GetDist's name format (without *)
                        # Map both with and without * to handle derived parameters
                        getdist_rename_mapping = {}
                        for orig_name, new_name in rename_mapping.items():
                            # Map the original name (might have *)
                            getdist_rename_mapping[orig_name] = new_name
                            # Also map without * (since GetDist strips *)
                            if orig_name.endswith('*'):
                                getdist_rename_mapping[orig_name.rstrip('*')] = new_name
                        
                        # Apply renaming to the parameters that actually exist in the samples
                        renamed_actual_names = []
                        renamed_actual_labels = []
                        
                        for name, label in zip(actual_param_names, actual_param_labels):
                            # Apply renaming if this parameter should be renamed
                            # Check both the name as-is and with * (for derived parameters)
                            new_name = None
                            if name in getdist_rename_mapping:
                                new_name = getdist_rename_mapping[name]
                            elif name + '*' in getdist_rename_mapping:
                                new_name = getdist_rename_mapping[name + '*']
                            
                            if new_name:
                                renamed_actual_names.append(new_name)
                                # Update label if it was the same as the parameter name
                                if label == name or label == name + '*':
                                    renamed_actual_labels.append(new_name)
                                else:
                                    renamed_actual_labels.append(label)
                            else:
                                renamed_actual_names.append(name)
                                renamed_actual_labels.append(label)
                        
                        # Apply custom labels if they exist
                        if custom_labels:
                            for i, name in enumerate(renamed_actual_names):
                                if name in custom_labels:
                                    renamed_actual_labels[i] = custom_labels[name]
                        
                        # Create new MCSamples with the correctly sized parameter lists
                        new_samples = MCSamples(
                            samples=samples.samples,
                            names=renamed_actual_names,
                            labels=renamed_actual_labels,
                            weights=getattr(samples, 'weights', None),
                            loglikes=getattr(samples, 'loglikes', None),
                            name_tag=getattr(samples, 'name_tag', None),
                            label=getattr(samples, 'label', None)
                        )
                        
                        # Copy other important attributes if they exist
                        if hasattr(samples, 'ranges'):
                            new_samples.ranges = samples.ranges
                        if hasattr(samples, 'sampler'):
                            new_samples.sampler = samples.sampler
                        
                        samples = new_samples
                
                # Apply custom labels even if no parameter renaming is needed (like old version)
                elif custom_labels:
                    from getdist import MCSamples
                    
                    # Get the parameter names and labels
                    param_names = [param.name for param in samples.paramNames.names]
                    param_labels = [param.label for param in samples.paramNames.names]
                    
                    # Apply custom labels
                    updated_labels = []
                    for i, name in enumerate(param_names):
                        if name in custom_labels:
                            updated_labels.append(custom_labels[name])
                        else:
                            updated_labels.append(param_labels[i])
                    
                    # Create new MCSamples with custom labels
                    new_samples = MCSamples(
                        samples=samples.samples,
                        names=param_names,
                        labels=updated_labels,
                        weights=getattr(samples, 'weights', None),
                        loglikes=getattr(samples, 'loglikes', None),
                        name_tag=getattr(samples, 'name_tag', None),
                        label=getattr(samples, 'label', None)
                    )
                    
                    # Copy other important attributes if they exist
                    if hasattr(samples, 'ranges'):
                        new_samples.ranges = samples.ranges
                    if hasattr(samples, 'sampler'):
                        new_samples.sampler = samples.sampler
                    
                    samples = new_samples
            
            # Cache the result
            self._samples_cache[cache_key] = samples
            self._cache_stats['samples']['misses'] += 1
            
            return samples
            
        except ImportError:
            raise DataLoadingError("GetDist is required for sample loading")
        except Exception as e:
            raise DataLoadingError(f"Failed to load GetDist samples for object '{object_id}': {e}")
    
    def compute_parameter_correlations(self, params: Optional[List[str]] = None,
                                     object_ids: Optional[List[str]] = None,
                                     access_scope: Optional[AccessScope] = None) -> np.ndarray:
        """
        Compute parameter correlations with enhanced scope awareness.
        
        Parameters
        ----------
        params : List[str], optional
            Parameters to compute correlations for
        object_ids : List[str], optional
            Object IDs to include
        access_scope : AccessScope, optional
            Access scope for filtering
            
        Returns
        -------
        numpy.ndarray
            Correlation matrix
        """
        self._ensure_initialized()
        
        scope = access_scope or self._access_scope
        
        if not scope.is_sample_level():
            raise DataLoadingError("Parameter correlations require sample-level access")
        
        # Load HDF5 data
        hdf5_data = self.load_hdf5_data(
            file_path=self._file_structure.hdf5_files[list(self._file_structure.hdf5_files.keys())[0]],
            access_scope=scope,
            object_filter=object_ids
        )
        
        # Get parameter values
        if params is None:
            params = self.get_free_parameters_from_hdf5()
        
        # Extract parameter data
        param_data = []
        for param in params:
            if param in hdf5_data:
                param_data.append(hdf5_data[param])
            else:
                self.logger.warning(f"Parameter '{param}' not found in HDF5 data")
        
        if not param_data:
            raise DataLoadingError("No valid parameters found for correlation computation")
        
        # Compute correlation matrix
        param_matrix = np.array(param_data)
        correlation_matrix = np.corrcoef(param_matrix)
        
        return correlation_matrix
    
    def get_parameter_statistics(self, params: Optional[List[str]] = None,
                               object_ids: Optional[List[str]] = None,
                               access_scope: Optional[AccessScope] = None) -> Dict[str, Dict[str, float]]:
        """
        Get parameter statistics with enhanced scope awareness.
        
        Parameters
        ----------
        params : List[str], optional
            Parameters to compute statistics for
        object_ids : List[str], optional
            Object IDs to include
        access_scope : AccessScope, optional
            Access scope for filtering
            
        Returns
        -------
        Dict[str, Dict[str, float]]
            Parameter statistics (mean, std, median, etc.)
        """
        self._ensure_initialized()
        
        scope = access_scope or self._access_scope
        
        # Load HDF5 data
        hdf5_data = self.load_hdf5_data(
            file_path=self._file_structure.hdf5_files[list(self._file_structure.hdf5_files.keys())[0]],
            access_scope=scope,
            object_filter=object_ids
        )
        
        # Get parameter values
        if params is None:
            params = self.get_free_parameters_from_hdf5()
        
        # Compute statistics for each parameter
        statistics = {}
        for param in params:
            if param in hdf5_data:
                values = np.array(hdf5_data[param])
                
                # Remove any invalid values
                valid_mask = np.isfinite(values)
                valid_values = values[valid_mask]
                
                if len(valid_values) > 0:
                    statistics[param] = {
                        'mean': float(np.mean(valid_values)),
                        'std': float(np.std(valid_values)),
                        'median': float(np.median(valid_values)),
                        'min': float(np.min(valid_values)),
                        'max': float(np.max(valid_values)),
                        'count': len(valid_values),
                        'valid_fraction': len(valid_values) / len(values)
                    }
                    
                    # Add percentiles
                    percentiles = [5, 16, 25, 75, 84, 95]
                    for p in percentiles:
                        statistics[param][f'p{p}'] = float(np.percentile(valid_values, p))
                else:
                    self.logger.warning(f"No valid values found for parameter '{param}'")
            else:
                self.logger.warning(f"Parameter '{param}' not found in HDF5 data")
        
        return statistics