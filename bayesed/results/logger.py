"""
Logging infrastructure for BayeSEDResults redesign.

This module provides configurable logging functionality with different
verbosity levels and structured output for debugging and user feedback.
"""

import logging
import sys
from typing import Optional, Dict, Any, List
from pathlib import Path
from datetime import datetime


class BayeSEDResultsLogger:
    """
    Configurable logger for BayeSEDResults operations.
    
    This class provides structured logging with different verbosity levels
    and specialized formatting for BayeSEDResults operations.
    """
    
    # Verbosity level mappings
    VERBOSITY_LEVELS = {
        0: logging.WARNING,   # Quiet - only warnings and errors
        1: logging.INFO,      # Normal - info, warnings, and errors  
        2: logging.DEBUG      # Verbose - all messages including debug
    }
    
    def __init__(self, name: str = "BayeSEDResults", verbosity: int = 0,
                 log_file: Optional[str] = None):
        """
        Initialize BayeSEDResults logger.
        
        Parameters
        ----------
        name : str
            Logger name (default: "BayeSEDResults")
        verbosity : int
            Verbosity level: 0 (quiet), 1 (normal), 2 (verbose) (default: 0)
        log_file : str, optional
            Path to log file. If None, logs only to console.
        """
        self.name = name
        self.verbosity = verbosity
        self.log_file = log_file
        
        # Create logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)  # Set to lowest level, handlers control output
        
        # Clear any existing handlers to ensure clean state
        self.logger.handlers.clear()
        
        # Prevent propagation to parent loggers to avoid interference
        self.logger.propagate = False
        
        # Set up handlers
        self._setup_console_handler()
        if log_file:
            self._setup_file_handler(log_file)
    
    def _setup_console_handler(self) -> None:
        """Set up console logging handler."""
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(self.VERBOSITY_LEVELS.get(self.verbosity, logging.INFO))
        
        # Create formatter
        if self.verbosity >= 2:
            # Verbose format with timestamp and level
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%H:%M:%S'
            )
        else:
            # Simple format for normal/quiet modes
            formatter = logging.Formatter('%(levelname)s: %(message)s')
        
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
    
    def _setup_file_handler(self, log_file: str) -> None:
        """Set up file logging handler."""
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)  # Always log everything to file
        
        # Detailed format for file logging
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        )
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
    
    def set_verbosity(self, verbosity: int) -> None:
        """
        Change the verbosity level.
        
        Parameters
        ----------
        verbosity : int
            New verbosity level: 0 (quiet), 1 (normal), 2 (verbose)
        """
        self.verbosity = verbosity
        
        # Update console handler level
        for handler in self.logger.handlers:
            if isinstance(handler, logging.StreamHandler) and handler.stream == sys.stdout:
                handler.setLevel(self.VERBOSITY_LEVELS.get(verbosity, logging.INFO))
                break
    
    def log_file_discovery(self, output_dir: str, files_found: Dict[str, Any]) -> None:
        """
        Log file discovery results.
        
        Parameters
        ----------
        output_dir : str
            Directory that was searched
        files_found : Dict[str, Any]
            Dictionary of discovered files by type
        """
        self.logger.info(f"Scanning directory: {output_dir}")
        
        for file_type, files in files_found.items():
            if isinstance(files, list):
                count = len(files)
                self.logger.debug(f"Found {count} {file_type} files")
                if self.verbosity >= 2:
                    for file_path in files:
                        self.logger.debug(f"  - {Path(file_path).name}")
            elif isinstance(files, dict):
                count = len(files)
                self.logger.debug(f"Found {count} {file_type} configurations")
    
    def log_configuration_selection(self, catalog_name: str, config_name: str,
                                  available_configs: List[str]) -> None:
        """
        Log configuration selection.
        
        Parameters
        ----------
        catalog_name : str
            Name of the catalog
        config_name : str
            Selected configuration name
        available_configs : List[str]
            List of all available configurations
        """
        self.logger.info(f"Selected configuration '{config_name}' for catalog '{catalog_name}'")
        
        if len(available_configs) > 1:
            self.logger.debug(f"Available configurations: {', '.join(available_configs)}")
    
    def log_data_loading(self, file_path: str, data_type: str, 
                        object_count: int = 0, parameter_count: int = 0) -> None:
        """
        Log data loading operation.
        
        Parameters
        ----------
        file_path : str
            Path to file being loaded
        data_type : str
            Type of data being loaded
        object_count : int
            Number of objects loaded
        parameter_count : int
            Number of parameters loaded
        """
        file_name = Path(file_path).name
        self.logger.info(f"Loading {data_type} from {file_name}")
        
        if object_count > 0 or parameter_count > 0:
            self.logger.debug(f"  - {object_count} objects, {parameter_count} parameters")
    
    def log_cache_operation(self, operation: str, cache_type: str, 
                           hit: bool = False, size_mb: float = 0.0) -> None:
        """
        Log cache operations.
        
        Parameters
        ----------
        operation : str
            Cache operation ('hit', 'miss', 'store', 'clear')
        cache_type : str
            Type of cache ('hdf5', 'samples', 'parameters')
        hit : bool
            Whether this was a cache hit (for 'access' operations)
        size_mb : float
            Size of cached data in MB
        """
        if operation == 'hit':
            self.logger.debug(f"Cache hit for {cache_type}")
        elif operation == 'miss':
            self.logger.debug(f"Cache miss for {cache_type}")
        elif operation == 'store':
            size_str = f" ({size_mb:.1f} MB)" if size_mb > 0 else ""
            self.logger.debug(f"Cached {cache_type}{size_str}")
        elif operation == 'clear':
            self.logger.debug(f"Cleared {cache_type} cache")
    
    def log_validation_results(self, validation_type: str, errors: List[str]) -> None:
        """
        Log validation results.
        
        Parameters
        ----------
        validation_type : str
            Type of validation performed
        errors : List[str]
            List of validation errors (empty if validation passed)
        """
        if not errors:
            self.logger.debug(f"{validation_type} validation passed")
        else:
            self.logger.warning(f"{validation_type} validation failed with {len(errors)} errors")
            for error in errors:
                self.logger.warning(f"  - {error}")
    
    def get_logger(self) -> logging.Logger:
        """Get the underlying logger instance."""
        return self.logger


# Global logger instance
_global_logger: Optional[BayeSEDResultsLogger] = None


def get_logger(name: str = "BayeSEDResults", verbosity: int = 0,
               log_file: Optional[str] = None) -> BayeSEDResultsLogger:
    """
    Get or create a BayeSEDResults logger.
    
    Parameters
    ----------
    name : str
        Logger name (default: "BayeSEDResults")
    verbosity : int
        Verbosity level: 0 (quiet), 1 (normal), 2 (verbose) (default: 0)
    log_file : str, optional
        Path to log file. If None, logs only to console.
        
    Returns
    -------
    BayeSEDResultsLogger
        Configured logger instance
    """
    global _global_logger
    
    # Always create a new logger instance to ensure clean state
    # This prevents issues with logger state being affected by external libraries
    if _global_logger is None or _global_logger.name != name or _global_logger.verbosity != verbosity:
        _global_logger = BayeSEDResultsLogger(name, verbosity, log_file)
    
    return _global_logger


def set_global_verbosity(verbosity: int) -> None:
    """
    Set verbosity for the global logger.
    
    Parameters
    ----------
    verbosity : int
        Verbosity level: 0 (quiet), 1 (normal), 2 (verbose)
    """
    global _global_logger
    
    if _global_logger is not None:
        _global_logger.set_verbosity(verbosity)


def enable_verbose_logging() -> None:
    """Enable verbose logging (level 2) for debugging."""
    set_global_verbosity(2)


def enable_normal_logging() -> None:
    """Enable normal logging (level 1) with INFO messages."""
    set_global_verbosity(1)


def enable_quiet_logging() -> None:
    """Enable quiet logging (level 0) - only warnings and errors (default)."""
    set_global_verbosity(0)