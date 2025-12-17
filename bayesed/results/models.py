"""
Data models for BayeSEDResults redesign.

This module defines the data structures used throughout the redesigned
BayeSEDResults implementation for managing access patterns, file information,
and configuration metadata.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from datetime import datetime
from pathlib import Path


@dataclass
class AccessScope:
    """
    Model for managing sample vs object access patterns.
    
    This class encapsulates the distinction between sample-level access
    (working with entire catalogs) and object-level access (working with
    specific individual objects).
    """
    scope_type: str  # 'sample' or 'object'
    catalog_name: str
    object_filter: Optional[str] = None
    total_objects: int = 0
    filtered_objects: int = 0
    
    def is_sample_level(self) -> bool:
        """Check if this is sample-level access (no object filtering)."""
        return self.scope_type == 'sample' and self.object_filter is None
    
    def is_object_level(self) -> bool:
        """Check if this is object-level access (specific objects)."""
        return self.scope_type == 'object' or self.object_filter is not None
    
    def get_scope_description(self) -> str:
        """Get human-readable description of the current scope."""
        if self.is_sample_level():
            return f"Sample-level access to '{self.catalog_name}' ({self.total_objects} objects)"
        else:
            return f"Object-level access to '{self.object_filter}' in '{self.catalog_name}'"


@dataclass
class ConfigurationInfo:
    """
    Information about a model configuration.
    
    This class stores metadata about a specific model configuration,
    including file paths, object counts, and creation information.
    """
    name: str
    file_path: str
    file_size: int = 0
    object_count: int = 0
    parameter_count: int = 0
    creation_time: Optional[datetime] = None
    catalog_name: str = ""
    
    def get_size_mb(self) -> float:
        """Get file size in megabytes."""
        return self.file_size / (1024 * 1024) if self.file_size > 0 else 0.0
    
    def __str__(self) -> str:
        """String representation of configuration info."""
        size_str = f"{self.get_size_mb():.1f} MB" if self.file_size > 0 else "unknown size"
        return (f"{self.name} ({self.object_count} objects, "
                f"{self.parameter_count} parameters, {size_str})")


@dataclass  
class FileStructure:
    """
    Information about the discovered file structure.
    
    This class organizes information about all discovered files
    in a BayeSED output directory, including HDF5 files and
    object-specific files.
    """
    output_dir: str
    catalog_name: str
    output_mode: str  # 'hdf5_only', 'full_structure', 'mixed'
    hdf5_files: Dict[str, str] = field(default_factory=dict)  # config_name -> file_path
    bestfit_files: Optional[Dict[str, Dict[str, List[str]]]] = None  # object_id -> config_name -> files
    posterior_files: Optional[Dict[str, Dict[str, Dict[str, str]]]] = None  # object_id -> config_name -> file_type -> path
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_configuration_names(self) -> List[str]:
        """Get list of available configuration names."""
        return sorted(self.hdf5_files.keys())
    
    def has_object_files(self) -> bool:
        """Check if object-specific files are available."""
        return self.output_mode in ['full_structure', 'mixed']
    
    def is_hdf5_only(self) -> bool:
        """Check if only HDF5 files are available."""
        return self.output_mode == 'hdf5_only'


@dataclass
class LoadedDataInfo:
    """
    Information about currently loaded data.
    
    This class tracks what data has been loaded, including file paths,
    scope information, and cache status.
    """
    configuration: ConfigurationInfo
    hdf5_file: str
    scope: AccessScope
    objects_loaded: List[str] = field(default_factory=list)
    parameters_loaded: List[str] = field(default_factory=list)
    cache_status: Dict[str, bool] = field(default_factory=dict)
    load_time: Optional[datetime] = None
    
    def get_load_summary(self) -> str:
        """Get summary of loaded data."""
        obj_count = len(self.objects_loaded)
        param_count = len(self.parameters_loaded)
        cache_info = f"{sum(self.cache_status.values())}/{len(self.cache_status)} cached" if self.cache_status else "no cache"
        
        return (f"Loaded {obj_count} objects, {param_count} parameters "
                f"from {Path(self.hdf5_file).name} ({cache_info})")
    
    def is_cached(self, cache_type: str) -> bool:
        """Check if a specific cache type is active."""
        return self.cache_status.get(cache_type, False)