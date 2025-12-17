"""
ValidationEngine component for BayeSEDResults redesign.

This module implements comprehensive validation functionality for file consistency,
data integrity, and object ID validation across different data sources.
"""

import os
import h5py
from pathlib import Path
from typing import List, Dict, Optional, Union, Tuple, Any, Set
import logging
from collections import defaultdict

from .base import BaseValidator
from .models import FileStructure, ConfigurationInfo
from .exceptions import ValidationError, AggregatedValidationError, DataLoadingError
from .logger import get_logger


class ValidationEngine(BaseValidator):
    """
    Comprehensive validation engine for BayeSED data consistency and integrity.
    
    This component validates file accessibility, data consistency between different
    file types, object ID consistency, and provides detailed diagnostic reporting.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize ValidationEngine component."""
        if logger is None:
            bayesed_logger = get_logger(__name__)
            logger = bayesed_logger.get_logger()
        super().__init__(logger)
        self._file_structure: Optional[FileStructure] = None
        self._validation_cache: Dict[str, Any] = {}
    
    def initialize(self, file_structure: FileStructure, **kwargs) -> None:
        """
        Initialize the ValidationEngine with file structure information.
        
        Parameters
        ----------
        file_structure : FileStructure
            File structure information from FileDiscovery
        **kwargs
            Additional initialization parameters
        """
        self._file_structure = file_structure
        self._validation_cache = {}
        
        self._initialized = True
        self.logger.debug(f"ValidationEngine initialized for catalog '{file_structure.catalog_name}'")
    
    def validate(self, data: Any, **kwargs) -> List[str]:
        """
        Main validation method that performs comprehensive validation.
        
        Parameters
        ----------
        data : Any
            Data to validate (typically FileStructure or file paths)
        **kwargs
            Additional validation parameters
            
        Returns
        -------
        List[str]
            List of validation error messages (empty if all valid)
        """
        self._ensure_initialized()
        
        validation_errors = []
        
        # Validate file structure if provided
        if isinstance(data, FileStructure):
            validation_errors.extend(self._validate_file_structure(data))
        
        # Validate specific files if file paths provided
        file_paths = kwargs.get('file_paths', [])
        if file_paths:
            validation_errors.extend(self._validate_file_paths(file_paths))
        
        # Validate data consistency if requested
        if kwargs.get('check_consistency', True):
            validation_errors.extend(self._validate_data_consistency())
        
        return validation_errors
    
    def validate_file_consistency(self, files: Optional[Dict[str, Any]] = None) -> List[str]:
        """
        Check consistency across different file types.
        
        This method validates that HDF5 files, FITS files, and sample files
        are consistent with each other and accessible.
        
        Parameters
        ----------
        files : Dict[str, Any], optional
            Dictionary of files to validate. If None, uses internal file structure.
            
        Returns
        -------
        List[str]
            List of validation error messages
        """
        self._ensure_initialized()
        
        if files is None:
            files = {
                'hdf5_files': self._file_structure.hdf5_files,
                'bestfit_files': self._file_structure.bestfit_files,
                'posterior_files': self._file_structure.posterior_files
            }
        
        validation_errors = []
        
        # Validate HDF5 files
        hdf5_files = files.get('hdf5_files', {})
        for config_name, hdf5_path in hdf5_files.items():
            errors = self._validate_hdf5_file(hdf5_path, config_name)
            validation_errors.extend(errors)
        
        # Validate bestfit files if available
        bestfit_files = files.get('bestfit_files', {})
        if bestfit_files:
            errors = self._validate_bestfit_files(bestfit_files)
            validation_errors.extend(errors)
        
        # Validate posterior files if available
        posterior_files = files.get('posterior_files', {})
        if posterior_files:
            errors = self._validate_posterior_files(posterior_files)
            validation_errors.extend(errors)
        
        # Cross-validate file consistency
        if hdf5_files and (bestfit_files or posterior_files):
            errors = self._cross_validate_files(hdf5_files, bestfit_files, posterior_files)
            validation_errors.extend(errors)
        
        self.logger.debug(f"File consistency validation found {len(validation_errors)} errors")
        return validation_errors
    
    def validate_object_ids(self, hdf5_data: Optional[Dict[str, Any]] = None, 
                           object_files: Optional[Dict[str, Any]] = None) -> List[str]:
        """
        Verify object ID consistency across data sources.
        
        This method ensures that object IDs are consistent between HDF5 files
        and individual object files (FITS, samples).
        
        Parameters
        ----------
        hdf5_data : Dict[str, Any], optional
            HDF5 data containing object IDs
        object_files : Dict[str, Any], optional
            Object-specific files organized by object ID
            
        Returns
        -------
        List[str]
            List of validation error messages
        """
        self._ensure_initialized()
        
        validation_errors = []
        
        # Extract object IDs from HDF5 files
        hdf5_object_ids = set()
        if hdf5_data:
            hdf5_object_ids = self._extract_hdf5_object_ids(hdf5_data)
        else:
            # Load object IDs from file structure
            for config_name, hdf5_path in self._file_structure.hdf5_files.items():
                try:
                    ids = self._get_object_ids_from_hdf5(hdf5_path)
                    hdf5_object_ids.update(ids)
                except Exception as e:
                    validation_errors.append(f"Failed to read object IDs from {hdf5_path}: {e}")
        
        # Extract object IDs from object files
        file_object_ids = set()
        if object_files:
            file_object_ids = set(object_files.keys())
        else:
            # Use file structure
            if self._file_structure.bestfit_files:
                file_object_ids.update(self._file_structure.bestfit_files.keys())
            if self._file_structure.posterior_files:
                file_object_ids.update(self._file_structure.posterior_files.keys())
        
        # Compare object ID sets
        if hdf5_object_ids and file_object_ids:
            missing_in_files = hdf5_object_ids - file_object_ids
            missing_in_hdf5 = file_object_ids - hdf5_object_ids
            
            if missing_in_files:
                validation_errors.append(
                    f"Objects in HDF5 but missing individual files: {sorted(list(missing_in_files))[:10]}"
                    + ("..." if len(missing_in_files) > 10 else "")
                )
            
            if missing_in_hdf5:
                validation_errors.append(
                    f"Individual files exist but objects missing from HDF5: {sorted(list(missing_in_hdf5))[:10]}"
                    + ("..." if len(missing_in_hdf5) > 10 else "")
                )
        
        self.logger.debug(f"Object ID validation: {len(hdf5_object_ids)} HDF5 objects, "
                         f"{len(file_object_ids)} file objects, {len(validation_errors)} errors")
        
        return validation_errors
    
    def generate_diagnostic_report(self) -> Dict[str, Any]:
        """
        Create detailed diagnostic report about the data structure and validation status.
        
        Returns
        -------
        Dict[str, Any]
            Comprehensive diagnostic information
        """
        self._ensure_initialized()
        
        report = {
            'catalog_name': self._file_structure.catalog_name,
            'output_mode': self._file_structure.output_mode,
            'validation_timestamp': self._get_timestamp(),
            'file_summary': {},
            'validation_results': {},
            'recommendations': []
        }
        
        # File summary
        report['file_summary'] = {
            'hdf5_files': len(self._file_structure.hdf5_files),
            'configurations': list(self._file_structure.hdf5_files.keys()),
            'has_bestfit_files': self._file_structure.bestfit_files is not None,
            'has_posterior_files': self._file_structure.posterior_files is not None,
            'total_objects': self._count_total_objects()
        }
        
        # Run comprehensive validation
        validation_errors = self.validate_file_consistency()
        object_id_errors = self.validate_object_ids()
        
        report['validation_results'] = {
            'file_consistency_errors': validation_errors,
            'object_id_errors': object_id_errors,
            'total_errors': len(validation_errors) + len(object_id_errors),
            'is_valid': len(validation_errors) == 0 and len(object_id_errors) == 0
        }
        
        # Generate recommendations
        report['recommendations'] = self._generate_recommendations(validation_errors, object_id_errors)
        
        self.logger.info(f"Generated diagnostic report: {report['validation_results']['total_errors']} total errors")
        
        return report
    
    def check_data_integrity(self, data: Any, data_type: str = "unknown") -> List[str]:
        """
        Validate data format and content integrity.
        
        Parameters
        ----------
        data : Any
            Data to validate (file path, loaded data, etc.)
        data_type : str, optional
            Type of data being validated
            
        Returns
        -------
        List[str]
            List of validation error messages
        """
        validation_errors = []
        
        if isinstance(data, (str, Path)):
            # Validate file path
            file_path = Path(data)
            if not file_path.exists():
                validation_errors.append(f"{data_type} file does not exist: {file_path}")
            elif not file_path.is_file():
                validation_errors.append(f"{data_type} path is not a file: {file_path}")
            elif not os.access(file_path, os.R_OK):
                validation_errors.append(f"{data_type} file is not readable: {file_path}")
            else:
                # Check file format based on extension
                if file_path.suffix.lower() == '.hdf5':
                    validation_errors.extend(self._validate_hdf5_format(file_path))
                elif file_path.suffix.lower() == '.fits':
                    validation_errors.extend(self._validate_fits_format(file_path))
                elif file_path.suffix.lower() in ['.txt', '.paramnames']:
                    validation_errors.extend(self._validate_text_format(file_path))
        
        return validation_errors
    
    def validate_or_raise_aggregated(self, data: Any, **kwargs) -> None:
        """
        Validate data and raise AggregatedValidationError if validation fails.
        
        This method collects all validation errors and presents them together
        rather than stopping at the first error.
        
        Parameters
        ----------
        data : Any
            Data to validate
        **kwargs
            Additional validation parameters
            
        Raises
        ------
        AggregatedValidationError
            If validation errors are found
        """
        validation_errors = self.validate(data, **kwargs)
        
        if validation_errors:
            raise AggregatedValidationError(
                validation_errors,
                suggestions=[
                    "Check file permissions and accessibility",
                    "Verify BayeSED analysis completed successfully",
                    "Ensure all required files are present"
                ]
            )
    
    # Private helper methods
    
    def _validate_file_structure(self, file_structure: FileStructure) -> List[str]:
        """Validate the overall file structure."""
        validation_errors = []
        
        # Check that we have at least some files
        if not file_structure.hdf5_files:
            validation_errors.append("No HDF5 files found in file structure")
        
        # Validate output directory
        output_dir = Path(file_structure.output_dir)
        if not output_dir.exists():
            validation_errors.append(f"Output directory does not exist: {output_dir}")
        elif not output_dir.is_dir():
            validation_errors.append(f"Output path is not a directory: {output_dir}")
        
        return validation_errors
    
    def _validate_file_paths(self, file_paths: List[str]) -> List[str]:
        """Validate a list of file paths."""
        validation_errors = []
        
        for file_path in file_paths:
            errors = self.check_data_integrity(file_path, "specified")
            validation_errors.extend(errors)
        
        return validation_errors
    
    def _validate_data_consistency(self) -> List[str]:
        """Validate consistency between different data sources."""
        validation_errors = []
        
        # This is a placeholder for more sophisticated consistency checks
        # that would compare parameter names, object counts, etc. between files
        
        return validation_errors
    
    def _validate_hdf5_file(self, hdf5_path: str, config_name: str) -> List[str]:
        """Validate a single HDF5 file."""
        validation_errors = []
        
        # Basic file validation
        errors = self.check_data_integrity(hdf5_path, f"HDF5 ({config_name})")
        validation_errors.extend(errors)
        
        # If file exists, validate HDF5 structure
        if not errors:  # Only if basic validation passed
            try:
                with h5py.File(hdf5_path, 'r') as f:
                    # Check for required datasets/groups
                    if 'object_id' not in f:
                        validation_errors.append(f"HDF5 file missing 'object_id' dataset: {hdf5_path}")
                    
                    # Check that file is not empty
                    if len(f.keys()) == 0:
                        validation_errors.append(f"HDF5 file is empty: {hdf5_path}")
                        
            except Exception as e:
                validation_errors.append(f"Failed to read HDF5 file {hdf5_path}: {e}")
        
        return validation_errors
    
    def _validate_bestfit_files(self, bestfit_files: Dict[str, Dict[str, List[str]]]) -> List[str]:
        """Validate bestfit FITS files."""
        validation_errors = []
        
        for obj_id, config_files in bestfit_files.items():
            for config_name, file_list in config_files.items():
                for fits_file in file_list:
                    errors = self.check_data_integrity(fits_file, f"FITS ({obj_id}, {config_name})")
                    validation_errors.extend(errors)
        
        return validation_errors
    
    def _validate_posterior_files(self, posterior_files: Dict[str, Dict[str, Dict[str, str]]]) -> List[str]:
        """Validate posterior sample files."""
        validation_errors = []
        
        for obj_id, config_files in posterior_files.items():
            for config_name, file_dict in config_files.items():
                for file_type, file_path in file_dict.items():
                    errors = self.check_data_integrity(file_path, f"{file_type} ({obj_id}, {config_name})")
                    validation_errors.extend(errors)
        
        return validation_errors
    
    def _cross_validate_files(self, hdf5_files: Dict[str, str], 
                             bestfit_files: Dict[str, Dict[str, List[str]]], 
                             posterior_files: Dict[str, Dict[str, Dict[str, str]]]) -> List[str]:
        """Cross-validate consistency between different file types."""
        validation_errors = []
        
        # Check that configurations match between HDF5 and object files
        hdf5_configs = set(hdf5_files.keys())
        
        if bestfit_files:
            bestfit_configs = set()
            for obj_files in bestfit_files.values():
                bestfit_configs.update(obj_files.keys())
            
            missing_configs = hdf5_configs - bestfit_configs
            if missing_configs:
                validation_errors.append(
                    f"Configurations in HDF5 but missing bestfit files: {sorted(missing_configs)}"
                )
        
        if posterior_files:
            posterior_configs = set()
            for obj_files in posterior_files.values():
                posterior_configs.update(obj_files.keys())
            
            missing_configs = hdf5_configs - posterior_configs
            if missing_configs:
                validation_errors.append(
                    f"Configurations in HDF5 but missing posterior files: {sorted(missing_configs)}"
                )
        
        return validation_errors
    
    def _extract_hdf5_object_ids(self, hdf5_data: Dict[str, Any]) -> Set[str]:
        """Extract object IDs from loaded HDF5 data."""
        object_ids = set()
        
        # This would extract object IDs from actual HDF5 data
        # For now, return empty set as placeholder
        
        return object_ids
    
    def _get_object_ids_from_hdf5(self, hdf5_path: str) -> Set[str]:
        """Get object IDs directly from HDF5 file."""
        object_ids = set()
        
        try:
            with h5py.File(hdf5_path, 'r') as f:
                if 'object_id' in f:
                    # Read object IDs and convert to strings
                    ids = f['object_id'][:]
                    if hasattr(ids, 'decode'):  # Handle bytes
                        object_ids = {id_val.decode() if hasattr(id_val, 'decode') else str(id_val) 
                                     for id_val in ids}
                    else:
                        object_ids = {str(id_val) for id_val in ids}
                        
        except Exception as e:
            self.logger.warning(f"Failed to read object IDs from {hdf5_path}: {e}")
        
        return object_ids
    
    def _validate_hdf5_format(self, file_path: Path) -> List[str]:
        """Validate HDF5 file format."""
        validation_errors = []
        
        try:
            with h5py.File(file_path, 'r') as f:
                # Basic format validation
                if len(f.keys()) == 0:
                    validation_errors.append(f"HDF5 file is empty: {file_path}")
        except Exception as e:
            validation_errors.append(f"Invalid HDF5 format in {file_path}: {e}")
        
        return validation_errors
    
    def _validate_fits_format(self, file_path: Path) -> List[str]:
        """Validate FITS file format."""
        validation_errors = []
        
        try:
            # Try to import astropy for FITS validation
            from astropy.io import fits
            with fits.open(file_path) as hdul:
                if len(hdul) == 0:
                    validation_errors.append(f"FITS file has no HDUs: {file_path}")
        except ImportError:
            # If astropy not available, just check file size
            if file_path.stat().st_size == 0:
                validation_errors.append(f"FITS file is empty: {file_path}")
        except Exception as e:
            validation_errors.append(f"Invalid FITS format in {file_path}: {e}")
        
        return validation_errors
    
    def _validate_text_format(self, file_path: Path) -> List[str]:
        """Validate text file format."""
        validation_errors = []
        
        try:
            # Check if file is readable as text
            with open(file_path, 'r') as f:
                # Try to read first line
                first_line = f.readline()
                if not first_line and file_path.stat().st_size > 0:
                    validation_errors.append(f"Text file appears corrupted: {file_path}")
        except UnicodeDecodeError:
            validation_errors.append(f"Text file contains invalid characters: {file_path}")
        except Exception as e:
            validation_errors.append(f"Cannot read text file {file_path}: {e}")
        
        return validation_errors
    
    def _count_total_objects(self) -> int:
        """Count total number of objects across all data sources."""
        total_objects = 0
        
        # Count from HDF5 files
        for hdf5_path in self._file_structure.hdf5_files.values():
            try:
                object_ids = self._get_object_ids_from_hdf5(hdf5_path)
                total_objects = max(total_objects, len(object_ids))
            except Exception:
                pass
        
        # Count from object files if available
        if self._file_structure.bestfit_files:
            total_objects = max(total_objects, len(self._file_structure.bestfit_files))
        
        if self._file_structure.posterior_files:
            total_objects = max(total_objects, len(self._file_structure.posterior_files))
        
        return total_objects
    
    def _generate_recommendations(self, file_errors: List[str], object_errors: List[str]) -> List[str]:
        """Generate recommendations based on validation errors."""
        recommendations = []
        
        if file_errors:
            recommendations.append("Fix file accessibility and format issues")
            if any("permission" in error.lower() for error in file_errors):
                recommendations.append("Check file permissions (chmod +r)")
            if any("corrupt" in error.lower() or "invalid" in error.lower() for error in file_errors):
                recommendations.append("Re-run BayeSED analysis to regenerate corrupted files")
        
        if object_errors:
            recommendations.append("Resolve object ID inconsistencies")
            if any("missing individual files" in error for error in object_errors):
                recommendations.append("Consider using HDF5-only mode for missing object files")
            if any("missing from HDF5" in error for error in object_errors):
                recommendations.append("Remove orphaned object files or regenerate HDF5 data")
        
        if not file_errors and not object_errors:
            recommendations.append("All validation checks passed - data appears consistent")
        
        return recommendations
    
    def _get_timestamp(self) -> str:
        """Get current timestamp for reporting."""
        from datetime import datetime
        return datetime.now().isoformat()
    
    # Public utility methods
    
    def is_file_structure_valid(self) -> bool:
        """
        Check if the current file structure is valid.
        
        Returns
        -------
        bool
            True if file structure passes all validation checks
        """
        self._ensure_initialized()
        
        validation_errors = self.validate_file_consistency()
        object_id_errors = self.validate_object_ids()
        
        return len(validation_errors) == 0 and len(object_id_errors) == 0
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """
        Get a summary of validation results.
        
        Returns
        -------
        Dict[str, Any]
            Summary of validation status
        """
        self._ensure_initialized()
        
        file_errors = self.validate_file_consistency()
        object_errors = self.validate_object_ids()
        
        return {
            'is_valid': len(file_errors) == 0 and len(object_errors) == 0,
            'file_errors': len(file_errors),
            'object_id_errors': len(object_errors),
            'total_errors': len(file_errors) + len(object_errors),
            'catalog_name': self._file_structure.catalog_name,
            'output_mode': self._file_structure.output_mode
        }
    
    def validate_configuration(self, config_name: str) -> List[str]:
        """
        Validate a specific configuration.
        
        Parameters
        ----------
        config_name : str
            Name of the configuration to validate
            
        Returns
        -------
        List[str]
            List of validation errors for this configuration
        """
        self._ensure_initialized()
        
        validation_errors = []
        
        # Check if configuration exists
        if config_name not in self._file_structure.hdf5_files:
            validation_errors.append(f"Configuration '{config_name}' not found")
            return validation_errors
        
        # Validate the HDF5 file for this configuration
        hdf5_path = self._file_structure.hdf5_files[config_name]
        validation_errors.extend(self._validate_hdf5_file(hdf5_path, config_name))
        
        # Validate object files for this configuration if available
        if self._file_structure.bestfit_files:
            for obj_id, config_files in self._file_structure.bestfit_files.items():
                if config_name in config_files:
                    for fits_file in config_files[config_name]:
                        errors = self.check_data_integrity(fits_file, f"FITS ({obj_id}, {config_name})")
                        validation_errors.extend(errors)
        
        if self._file_structure.posterior_files:
            for obj_id, config_files in self._file_structure.posterior_files.items():
                if config_name in config_files:
                    for file_type, file_path in config_files[config_name].items():
                        errors = self.check_data_integrity(file_path, f"{file_type} ({obj_id}, {config_name})")
                        validation_errors.extend(errors)
        
        return validation_errors