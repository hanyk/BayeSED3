"""
Abstract base classes for BayeSEDResults redesign.

This module defines the abstract base classes that establish the interface
contracts for core components in the redesigned BayeSEDResults system.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import logging

from .models import AccessScope, FileStructure, ConfigurationInfo
from .exceptions import BayeSEDResultsError


class BaseComponent(ABC):
    """
    Abstract base class for all BayeSEDResults components.
    
    This class provides common functionality and interface contracts
    that all components must implement.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize base component.
        
        Parameters
        ----------
        logger : logging.Logger, optional
            Logger instance for this component. If None, creates a default logger.
        """
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self._initialized = False
    
    @abstractmethod
    def initialize(self, **kwargs) -> None:
        """
        Initialize the component with required parameters.
        
        This method must be implemented by all components to handle
        their specific initialization requirements.
        """
        pass
    
    def is_initialized(self) -> bool:
        """Check if the component has been initialized."""
        return self._initialized
    
    def _ensure_initialized(self) -> None:
        """Ensure the component is initialized before use."""
        if not self._initialized:
            raise BayeSEDResultsError(
                f"{self.__class__.__name__} must be initialized before use",
                suggestions=["Call initialize() method first"]
            )


class BaseFileHandler(BaseComponent):
    """
    Abstract base class for components that handle file operations.
    
    This class provides common file handling functionality and
    interface contracts for file-based operations.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize file handler."""
        super().__init__(logger)
        self._base_path: Optional[Path] = None
    
    @abstractmethod
    def discover_files(self, base_path: Union[str, Path]) -> FileStructure:
        """
        Discover and organize files in the given directory.
        
        Parameters
        ----------
        base_path : str or Path
            Base directory to search for files
            
        Returns
        -------
        FileStructure
            Organized information about discovered files
        """
        pass
    
    @abstractmethod
    def validate_file_access(self, file_path: Union[str, Path]) -> bool:
        """
        Validate that a file exists and is readable.
        
        Parameters
        ----------
        file_path : str or Path
            Path to file to validate
            
        Returns
        -------
        bool
            True if file is accessible, False otherwise
        """
        pass
    
    def set_base_path(self, base_path: Union[str, Path]) -> None:
        """Set the base path for file operations."""
        self._base_path = Path(base_path)
        self.logger.debug(f"Set base path to: {self._base_path}")
    
    def get_base_path(self) -> Optional[Path]:
        """Get the current base path."""
        return self._base_path


class BaseValidator(BaseComponent):
    """
    Abstract base class for validation components.
    
    This class provides common validation functionality and
    interface contracts for data validation operations.
    """
    
    @abstractmethod
    def validate(self, data: Any, **kwargs) -> List[str]:
        """
        Validate the given data.
        
        Parameters
        ----------
        data : Any
            Data to validate
        **kwargs
            Additional validation parameters
            
        Returns
        -------
        List[str]
            List of validation error messages. Empty list if validation passes.
        """
        pass
    
    def is_valid(self, data: Any, **kwargs) -> bool:
        """
        Check if data is valid (convenience method).
        
        Parameters
        ----------
        data : Any
            Data to validate
        **kwargs
            Additional validation parameters
            
        Returns
        -------
        bool
            True if data is valid, False otherwise
        """
        return len(self.validate(data, **kwargs)) == 0
    
    def validate_or_raise(self, data: Any, **kwargs) -> None:
        """
        Validate data and raise exception if invalid.
        
        Parameters
        ----------
        data : Any
            Data to validate
        **kwargs
            Additional validation parameters
            
        Raises
        ------
        ValidationError
            If validation fails
        """
        from .exceptions import ValidationError
        
        errors = self.validate(data, **kwargs)
        if errors:
            raise ValidationError(
                f"Validation failed for {type(data).__name__}",
                validation_errors=errors
            )