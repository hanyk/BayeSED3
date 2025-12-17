"""
Exception classes for BayeSEDResults redesign.

This module defines the error hierarchy for the redesigned BayeSEDResults
implementation, providing specific exception types for different failure modes.
"""

from typing import List, Optional, Dict, Any


class BayeSEDResultsError(Exception):
    """Base exception for BayeSEDResults errors."""
    
    def __init__(self, message: str, suggestions: Optional[List[str]] = None, 
                 error_context: Optional[Dict[str, Any]] = None):
        """
        Initialize BayeSEDResults error.
        
        Parameters
        ----------
        message : str
            Main error message
        suggestions : List[str], optional
            List of suggested solutions or alternatives
        error_context : Dict[str, Any], optional
            Additional context information about the error
        """
        self.suggestions = suggestions or []
        self.error_context = error_context or {}
        
        # Build comprehensive error message
        full_message = message
        if self.suggestions:
            full_message += "\n\nSuggestions:"
            for suggestion in self.suggestions:
                full_message += f"\n  - {suggestion}"
                
        super().__init__(full_message)


class FileDiscoveryError(BayeSEDResultsError):
    """Errors during file discovery and validation."""
    
    def __init__(self, message: str, missing_files: Optional[List[str]] = None,
                 corrupted_files: Optional[List[str]] = None, **kwargs):
        """
        Initialize file discovery error.
        
        Parameters
        ----------
        message : str
            Main error message
        missing_files : List[str], optional
            List of missing file paths
        corrupted_files : List[str], optional
            List of corrupted file paths
        **kwargs
            Additional arguments passed to BayeSEDResultsError
        """
        self.missing_files = missing_files or []
        self.corrupted_files = corrupted_files or []
        
        # Add file information to error context
        error_context = kwargs.get('error_context', {})
        error_context.update({
            'missing_files': self.missing_files,
            'corrupted_files': self.corrupted_files
        })
        kwargs['error_context'] = error_context
        
        super().__init__(message, **kwargs)


class ConfigurationError(BayeSEDResultsError):
    """Errors in model configuration selection."""
    
    def __init__(self, message: str, available_configs: Optional[List[str]] = None,
                 invalid_config: Optional[str] = None, **kwargs):
        """
        Initialize configuration error.
        
        Parameters
        ----------
        message : str
            Main error message
        available_configs : List[str], optional
            List of available configuration names
        invalid_config : str, optional
            The invalid configuration that was requested
        **kwargs
            Additional arguments passed to BayeSEDResultsError
        """
        self.available_configs = available_configs or []
        self.invalid_config = invalid_config
        
        # Add configuration information to error context
        error_context = kwargs.get('error_context', {})
        error_context.update({
            'available_configs': self.available_configs,
            'invalid_config': self.invalid_config
        })
        kwargs['error_context'] = error_context
        
        # Auto-generate suggestions if available configs provided
        if self.available_configs and not kwargs.get('suggestions'):
            suggestions = [f"Available configurations: {', '.join(self.available_configs)}"]
            if self.invalid_config:
                # Try to find similar configuration names
                similar_configs = [
                    config for config in self.available_configs 
                    if self.invalid_config.lower() in config.lower()
                ]
                if similar_configs:
                    suggestions.append(f"Did you mean one of: {', '.join(similar_configs)}?")
            kwargs['suggestions'] = suggestions
        
        super().__init__(message, **kwargs)


class DataLoadingError(BayeSEDResultsError):
    """Errors during data loading operations."""
    
    def __init__(self, message: str, file_path: Optional[str] = None,
                 data_type: Optional[str] = None, **kwargs):
        """
        Initialize data loading error.
        
        Parameters
        ----------
        message : str
            Main error message
        file_path : str, optional
            Path to the file that failed to load
        data_type : str, optional
            Type of data that failed to load (e.g., 'HDF5', 'FITS', 'samples')
        **kwargs
            Additional arguments passed to BayeSEDResultsError
        """
        self.file_path = file_path
        self.data_type = data_type
        
        # Add loading information to error context
        error_context = kwargs.get('error_context', {})
        error_context.update({
            'file_path': self.file_path,
            'data_type': self.data_type
        })
        kwargs['error_context'] = error_context
        
        super().__init__(message, **kwargs)


class ValidationError(BayeSEDResultsError):
    """Errors in data validation and consistency checks."""
    
    def __init__(self, message: str, validation_errors: Optional[List[str]] = None,
                 **kwargs):
        """
        Initialize validation error.
        
        Parameters
        ----------
        message : str
            Main error message
        validation_errors : List[str], optional
            List of specific validation error messages
        **kwargs
            Additional arguments passed to BayeSEDResultsError
        """
        self.validation_errors = validation_errors or []
        
        # Add validation errors to error context
        error_context = kwargs.get('error_context', {})
        error_context.update({
            'validation_errors': self.validation_errors
        })
        kwargs['error_context'] = error_context
        
        # Build comprehensive message with all validation errors
        if self.validation_errors:
            full_message = message + "\n\nValidation errors:"
            for i, error in enumerate(self.validation_errors, 1):
                full_message += f"\n  {i}. {error}"
            message = full_message
        
        super().__init__(message, **kwargs)


class AggregatedValidationError(ValidationError):
    """
    Special validation error that aggregates multiple validation failures.
    
    This exception is used when multiple validation errors are encountered
    and should be presented together rather than stopping at the first error.
    """
    
    def __init__(self, validation_errors: List[str], **kwargs):
        """
        Initialize aggregated validation error.
        
        Parameters
        ----------
        validation_errors : List[str]
            List of validation error messages to aggregate
        **kwargs
            Additional arguments passed to ValidationError
        """
        message = f"Multiple validation errors encountered ({len(validation_errors)} total)"
        super().__init__(message, validation_errors=validation_errors, **kwargs)