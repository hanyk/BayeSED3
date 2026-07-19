"""
Results products: HDF5 discovery/load, tables, evidence, sample/bestfit data.

Held by BayeSEDResults (ADR-0003). Presentation lives in results_presentation.py.
"""

import os
import h5py
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Union, Any
import logging

logger = logging.getLogger(__name__)


class ResultsProducts:
    """Deep products module for BayeSED Results (load + scientific accessors)."""

    def _to_str(x) -> str:
        """Convert bytes to string, handling HDF5 byte columns."""
        return x.decode('utf-8') if isinstance(x, bytes) else str(x)
    def _match_strings(candidates: List[str], pattern: str, match_mode: str,
                       case_sensitive: bool) -> List[str]:
        """
        Match pattern against candidate strings using specified mode.

        Parameters
        ----------
        candidates : list of str
            Strings to search through
        pattern : str
            Pattern to match
        match_mode : str
            'contains' for substring, 'regex' for regex matching
        case_sensitive : bool
            Whether matching is case sensitive

        Returns
        -------
        list of str
            Matching candidates
        """
        import re

        if match_mode == 'contains':
            if case_sensitive:
                return [c for c in candidates if pattern in c]
            else:
                pattern_lower = pattern.lower()
                return [c for c in candidates if pattern_lower in c.lower()]

        elif match_mode == 'regex':
            try:
                if case_sensitive:
                    compiled = re.compile(pattern)
                else:
                    compiled = re.compile(pattern, re.IGNORECASE)
                return [c for c in candidates if compiled.search(c)]
            except re.error as e:
                raise ValueError(f"Invalid regular expression '{pattern}': {e}")

        else:
            raise ValueError(f"Invalid match_mode '{match_mode}'. Must be 'contains' or 'regex'")
    def __init__(self, output_dir: Union[str, Path], catalog_name: Optional[str] = None,
                 model_config: Optional[Union[str, int]] = None, object_id: Optional[str] = None):
        """Initialize ResultsProducts."""

        # Store initialization parameters
        self.output_dir = Path(output_dir).resolve()
        self.catalog_name = catalog_name
        self.model_config = model_config
        self.object_id = object_id

        # Main data storage - loaded once
        self._hdf5_table = None
        self._hdf5_file = None
        self.parameters = None  # Public access to HDF5 table

        # Parameter labeling
        self._custom_labels = {}

        # Initialize
        self._find_hdf5_file()
        self._load_hdf5_table()

        logger.info(f"BayeSEDResults initialized for catalog '{self.catalog_name}'")
        if self.object_id:
            logger.info(f"Object-level access for object: {self.object_id}")
    def _extract_catalog_name_from_filename(self, filename: str) -> str:
        """
        Extract catalog name from HDF5 filename.
        
        Pattern: {catalog_name}_{config_starting_with_alphanumeric}...
        The catalog name is everything before the first part that starts with a combination 
        of numbers and letters (alphanumeric with at least one digit and one letter).
        
        Examples:
        - gal_0csp_sfh200_... → gal
        - qso_0csp_sfh200_... → qso  
        - test_inoise1_0csp_... → test_inoise1
        - W0533_ALMA_0csp_... → W0533_ALMA
        - seedcat2_STARFORMING_0csp_... → seedcat2_STARFORMING (STARFORMING doesn't start with alphanumeric)
        """
        import re
        
        parts = filename.split('_')
        
        # Find the first part that starts with a combination of numbers and letters
        catalog_parts = []
        for part in parts:
            if part and self._is_model_config_part(part):
                # Found the config part - everything before this is catalog name
                break
            catalog_parts.append(part)
        
        if not catalog_parts:
            # Fallback to first part if no valid config part found
            return parts[0] if parts else filename
            
        return '_'.join(catalog_parts)
    def _is_model_config_part(self, part: str) -> bool:
        """
        Check if a filename part represents a model configuration.
        
        A model config part should start with numbers followed by letters (in that order).
        This is more restrictive to avoid false positives with catalog names that contain numbers.
        
        Parameters
        ----------
        part : str
            Filename part to check
            
        Returns
        -------
        bool
            True if this part looks like a model config, False otherwise
        """
        if not part:
            return False
            
        import re
        
        # Model config must start with digit(s) followed by letter(s)
        # Examples: 0csp, 1a2b, 2dal8, 5abc, 123def
        # This pattern ensures numbers come before letters
        if re.match(r'^\d+[a-zA-Z]', part):
            return True
        
        return False
    def _find_hdf5_file(self) -> None:
        """Find the HDF5 file to load based on catalog_name and model_config."""
        if not self.output_dir.exists():
            raise FileNotFoundError(f"Output directory does not exist: {self.output_dir}")

        # Look for HDF5 files in the output directory
        hdf5_files = list(self.output_dir.glob("*.hdf5"))

        if not hdf5_files:
            raise FileNotFoundError(f"No HDF5 files found in output directory: {self.output_dir}")

        # Auto-detect catalog_name if not provided
        if not self.catalog_name:
            # Extract unique catalog names from all HDF5 files
            # Pattern: {catalog_name}_{config_starting_with_digit}...
            catalog_names = set()
            for hdf5_file in hdf5_files:
                filename = Path(hdf5_file).stem
                catalog_name = self._extract_catalog_name_from_filename(filename)
                catalog_names.add(catalog_name)

            if len(catalog_names) == 1:
                # Only one catalog - auto-select it
                self.catalog_name = catalog_names.pop()
                logger.info(f"Auto-detected catalog name: '{self.catalog_name}'")
            else:
                # Multiple catalogs - require explicit selection
                sorted_catalogs = sorted(catalog_names)
                raise ValueError(
                    f"Multiple catalogs found. Please specify catalog_name parameter. "
                    f"Available catalogs: {sorted_catalogs}"
                )

        # Find files matching the catalog with strict filtering
        catalog_pattern = f"{self.catalog_name}_*.hdf5"
        all_catalog_files = list(self.output_dir.glob(catalog_pattern))
        
        # Filter files to only include those where the part after catalog_name starts with digits+letters
        catalog_files = []
        for f in all_catalog_files:
            filename = f.stem
            if filename.startswith(self.catalog_name + '_'):
                # Extract the part after catalog_name and underscore
                remaining_part = filename[len(self.catalog_name) + 1:]
                # Split by underscore and check if first part starts with digits+letters
                if remaining_part:
                    first_part = remaining_part.split('_')[0]
                    if self._is_model_config_part(first_part):
                        catalog_files.append(f)

        if not catalog_files:
            raise FileNotFoundError(f"No HDF5 files found for catalog '{self.catalog_name}' with valid model config pattern in {self.output_dir}")

        # Handle model_config selection
        if self.model_config is not None:
            # User specified a model_config - find exact match
            target_file = None

            if isinstance(self.model_config, int):
                # Integer index - get available configs and select by index
                available_configs = []
                for f in catalog_files:
                    filename = f.stem
                    if filename.startswith(self.catalog_name + '_'):
                        config_name = filename[len(self.catalog_name) + 1:]
                        available_configs.append((config_name, f))

                available_configs.sort()  # Sort for consistent ordering

                if 0 <= self.model_config < len(available_configs):
                    config_name, target_file = available_configs[self.model_config]
                    self.model_config = config_name  # Update to string name
                else:
                    raise ValueError(f"Model config index {self.model_config} out of range. "
                                   f"Available configs (0-{len(available_configs)-1}): "
                                   f"{[c[0] for c in available_configs]}")

            else:
                # String name - find matching config
                model_config_str = str(self.model_config)
                
                # First pass: Look for exact matches
                exact_matches = []
                for f in catalog_files:
                    filename = f.stem
                    if filename.startswith(self.catalog_name + '_'):
                        config_name = filename[len(self.catalog_name) + 1:]
                        if config_name == model_config_str:
                            exact_matches.append((config_name, f))
                
                if len(exact_matches) == 1:
                    # Found exactly one exact match
                    config_name, target_file = exact_matches[0]
                    self.model_config = config_name
                elif len(exact_matches) > 1:
                    # Multiple exact matches - this shouldn't happen but handle it
                    config_list = [match[0] for match in exact_matches[:3]]
                    if len(exact_matches) > 3:
                        config_list.append(f"... and {len(exact_matches) - 3} more")
                    raise ValueError(
                        f"Multiple exact matches found for model config '{model_config_str}' in catalog '{self.catalog_name}'. "
                        f"Found {len(exact_matches)} matches: {', '.join(config_list)}. "
                        f"This indicates duplicate files - please remove duplicates."
                    )
                else:
                    # No exact matches - try partial matching or regex matching
                    partial_matches = []
                    
                    # Check if model_config_str looks like a regex pattern
                    is_regex_pattern = ('(?=' in model_config_str and '.*' in model_config_str)
                    
                    for f in catalog_files:
                        filename = f.stem
                        if filename.startswith(self.catalog_name + '_'):
                            config_name = filename[len(self.catalog_name) + 1:]
                            
                            if is_regex_pattern:
                                # Use regex matching
                                import re
                                try:
                                    if re.search(model_config_str, config_name):
                                        partial_matches.append((config_name, f))
                                except re.error:
                                    # If regex fails, fall back to simple string matching
                                    if model_config_str in config_name:
                                        partial_matches.append((config_name, f))
                            else:
                                # Use simple string matching
                                if model_config_str in config_name:
                                    partial_matches.append((config_name, f))
                    
                    if len(partial_matches) == 1:
                        # Found exactly one partial match
                        config_name, target_file = partial_matches[0]
                        self.model_config = config_name
                        if is_regex_pattern:
                            logger.info(f"Using regex match for '{model_config_str}': '{config_name}'")
                        else:
                            logger.info(f"Using partial match for '{model_config_str}': '{config_name}'")
                    elif len(partial_matches) > 1:
                        # Multiple partial matches - require more specific selection
                        config_list = [match[0] for match in partial_matches[:3]]
                        if len(partial_matches) > 3:
                            config_list.append(f"... and {len(partial_matches) - 3} more")
                        raise ValueError(
                            f"Multiple partial matches found for model config '{model_config_str}' in catalog '{self.catalog_name}'. "
                            f"Found {len(partial_matches)} matches: {', '.join(config_list)}. "
                            f"Please specify a more specific model_config to uniquely identify the desired configuration."
                        )
                    else:
                        # No matches at all
                        available_configs = []
                        for f in catalog_files:
                            filename = f.stem
                            if filename.startswith(self.catalog_name + '_'):
                                config_name = filename[len(self.catalog_name) + 1:]
                                available_configs.append(config_name)

                        raise ValueError(f"Model config '{model_config_str}' not found for catalog '{self.catalog_name}'. "
                                       f"Available configs: {available_configs}")

            self._hdf5_file = str(target_file)
            # Log the selected HDF5 file relative to current working directory
            try:
                relative_path = target_file.resolve().relative_to(Path.cwd().resolve())
                logger.info(f"Selected HDF5 file for catalog '{self.catalog_name}', config '{self.model_config}': {relative_path}")
            except ValueError:
                # If file is not relative to cwd, show absolute path
                logger.info(f"Selected HDF5 file for catalog '{self.catalog_name}', config '{self.model_config}': {target_file.resolve()}")

        else:
            # No model_config specified - use first available or require selection if multiple
            if len(catalog_files) == 1:
                # Only one config available - use it
                self._hdf5_file = str(catalog_files[0])
                filename = Path(self._hdf5_file).stem
                self.model_config = filename[len(self.catalog_name) + 1:]
                # Log the auto-selected HDF5 file relative to current working directory
                try:
                    relative_path = catalog_files[0].resolve().relative_to(Path.cwd().resolve())
                    logger.info(f"Auto-selected single config '{self.model_config}' for catalog '{self.catalog_name}': {relative_path}")
                except ValueError:
                    # If file is not relative to cwd, show absolute path
                    logger.info(f"Auto-selected single config '{self.model_config}' for catalog '{self.catalog_name}': {catalog_files[0].resolve()}")

            else:
                # Multiple configs available - require explicit selection
                available_configs = []
                for f in catalog_files:
                    filename = f.stem
                    if filename.startswith(self.catalog_name + '_'):
                        config_name = filename[len(self.catalog_name) + 1:]
                        available_configs.append(config_name)

                raise ValueError(f"Multiple model configurations found for catalog '{self.catalog_name}'. "
                               f"Please specify model_config parameter. Available configs: {available_configs}")
    def _load_hdf5_table(self) -> None:
        """Load HDF5 data using astropy Table + h5py approach."""
        try:
            from astropy.table import Table, hstack
        except ImportError:
            raise ImportError("astropy is required for HDF5 table loading. Install with: pip install astropy")

        try:
            with h5py.File(self._hdf5_file, 'r') as h:
                logger.debug(f"Available datasets in HDF5 file: {list(h.keys())}")

                # Get parameter names and decode from bytes to strings
                colnames = []
                if 'parameters_name' in h:
                    colnames = [x.decode('utf-8') if isinstance(x, bytes) else str(x)
                               for x in h['parameters_name'][:]]
                    logger.debug(f"Found parameters_name with {len(colnames)} names")
                elif 'parameter_names' in h:
                    colnames = [x.decode('utf-8') if isinstance(x, bytes) else str(x)
                               for x in h['parameter_names'][:]]
                    logger.debug(f"Found parameter_names with {len(colnames)} names")
                else:
                    raise ValueError(f"No parameter names dataset found in HDF5 file. Available datasets: {list(h.keys())}")

                # Create ID table
                if 'ID' in h:
                    id_table = Table([h['ID'][:]], names=['ID'])
                else:
                    raise ValueError("No 'ID' dataset found in HDF5 file")

                # Create parameters table
                if 'parameters' in h:
                    parameters_table = Table(h['parameters'][:], names=colnames, copy=False)
                else:
                    raise ValueError("No 'parameters' dataset found in HDF5 file")

                # Combine ID and parameters using hstack
                self._hdf5_table = hstack([id_table, parameters_table])
                
                # Make the table available as public parameters attribute
                self.parameters = self._hdf5_table

                logger.info(f"Loaded HDF5 table: {len(self._hdf5_table)} objects, {len(colnames)} parameters")

        except Exception as e:
            raise RuntimeError(f"Failed to load HDF5 file {self._hdf5_file}: {e}")
    def _find_config_files(self, object_id: str, file_pattern: str) -> List[Path]:
        """
        Find files for the current model configuration in an object directory.

        Parameters
        ----------
        object_id : str
            Object ID to look for files in
        file_pattern : str
            File pattern to match (e.g., "*_sample_par.txt", "*_bestfit.fits")

        Returns
        -------
        List[Path]
            List of matching files for the current configuration
        """
        object_dir = self.output_dir / self.catalog_name / object_id
        if not object_dir.exists():
            return []

        # Find all files matching the pattern
        all_files = list(object_dir.glob(file_pattern))

        if not self.model_config:
            # No specific config - return all files
            return all_files

        # Filter files that match the current model configuration
        # Use exact matching first, then partial matching
        exact_matches = []
        partial_matches = []
        
        for file_path in all_files:
            filename = file_path.stem
            
            # Extract the config part from filename
            # Files are typically named like: {config_name}_sample_par.txt or {config_name}_bestfit.fits
            # Remove common suffixes to get the config part
            config_part = filename
            for suffix in ['_sample_par', '_bestfit', '_paramnames']:
                if config_part.endswith(suffix):
                    config_part = config_part[:-len(suffix)]
                    break
            
            # Check for exact match first
            if config_part == self.model_config:
                exact_matches.append(file_path)
            # Then check for partial match
            elif self.model_config in config_part:
                partial_matches.append(file_path)

        # Return exact matches if found, otherwise partial matches
        if exact_matches:
            return exact_matches
        else:
            return partial_matches
    def _select_unique_file(self, files: List[Path], file_type: str, object_id: str, log_selection: bool = False) -> Path:
        """
        Select a unique file from a list, raising an error if multiple files are found.

        Parameters
        ----------
        files : List[Path]
            List of matching files
        file_type : str
            Description of file type for error messages (e.g., "bestfit FITS", "sample_par.txt")
        object_id : str
            Object ID for error messages
        log_selection : bool, default False
            Whether to log the selected file. Set to True only at the main entry points
            to avoid duplicate logging.

        Returns
        -------
        Path
            The unique file path

        Raises
        ------
        FileNotFoundError
            If no files are found
        ValueError
            If multiple files are found
        """
        if not files:
            object_dir = self.output_dir / self.catalog_name / object_id
            raise FileNotFoundError(f"No {file_type} files found for object {object_id} and config '{self.model_config}' in {object_dir}")

        if len(files) == 1:
            selected_file = files[0]
            # Log the selected file only if requested (to avoid duplicate logging)
            if log_selection:
                try:
                    relative_path = selected_file.resolve().relative_to(Path.cwd().resolve())
                    logger.info(f"Selected {file_type} file: {relative_path}")
                except ValueError:
                    # If file is not relative to cwd, show absolute path
                    logger.info(f"Selected {file_type} file: {selected_file.resolve()}")
            return selected_file

        # Multiple files found - show first 3 and raise error
        file_list = [f.name for f in files[:3]]
        if len(files) > 3:
            file_list.append(f"... and {len(files) - 3} more")
        
        object_dir = self.output_dir / self.catalog_name / object_id
        raise ValueError(
            f"Multiple {file_type} files found for object {object_id} and config '{self.model_config}' in {object_dir}. "
            f"Found {len(files)} files: {', '.join(file_list)}. "
            f"Please ensure only one {file_type} file exists for this configuration, or specify a more specific model_config."
        )
    def get_parameter_names(self, pattern: Optional[str] = None,
                           param_match_mode: str = 'contains',
                           case_sensitive: bool = True) -> List[str]:
        """
        Get list of parameter names with optional filtering.

        Parameters
        ----------
        pattern : str, optional
            Pattern to filter parameter names. If None, returns all parameter names.
            Supports:
            - 'contains': Parameters that contain this substring anywhere (default)
            - 'regex': Regular expression pattern matching
        param_match_mode : str, default 'contains'
            Matching mode for parameter names:
            - 'contains': Substring matching anywhere in parameter name
            - 'regex': Regular expression pattern
        case_sensitive : bool, default True
            Whether matching should be case sensitive

        Returns
        -------
        List[str]
            List of parameter names (filtered if pattern is provided)

        Examples
        --------
        >>> # Get all parameter names (backward compatible)
        >>> all_params = results.get_parameter_names()
        
        >>> # Find parameters containing "mass"
        >>> mass_params = results.get_parameter_names("mass")
        
        >>> # Find percentile parameters using regex
        >>> percentiles = results.get_parameter_names(r".*_(16|50|84)$", param_match_mode='regex')
        
        >>> # Case-insensitive search
        >>> sfr_params = results.get_parameter_names("SFR", case_sensitive=False)
        """
        if self._hdf5_table is None:
            raise RuntimeError("HDF5 table not loaded")

        # Get all parameter names (excluding 'ID')
        all_param_names = [col for col in self._hdf5_table.colnames if col != 'ID']

        # If no pattern specified, return all parameters (backward compatible)
        if pattern is None:
            return all_param_names

        # Apply filtering using the same logic as get_parameter_values
        matching_params = self._match_strings(all_param_names, pattern, param_match_mode, case_sensitive)

        if not matching_params and pattern is not None:
            # Provide helpful error message
            error_msg = f"No parameters found matching '{pattern}' with mode '{param_match_mode}'"
            if not case_sensitive:
                error_msg += " (case-insensitive)"
            error_msg += f".\nUse get_parameter_names() to see all available parameters."
            raise ValueError(error_msg)

        return matching_params
    def get_free_parameters(self) -> List[str]:
        """
        Get list of free (fitted) parameters by analyzing HDF5 table structure.
        
        In BayeSED3 HDF5 files, parameters are ordered with free parameters first,
        followed by derived parameters starting from the first parameter that 
        begins with "log(scale)" or "log(norm)".

        Returns
        -------
        List[str]
            List of free parameter names in original order
        """
        # First try to read paramnames file (original behavior)
        try:
            # Find a paramnames file to read parameter structure
            if self.object_id:
                # Object-level access - use specific object
                paramnames_files = self._find_config_files(self.object_id, "*_sample_par.paramnames")
            else:
                # Sample-level access - use first available object
                objects = self.list_objects()
                if objects:
                    paramnames_files = self._find_config_files(objects[0], "*_sample_par.paramnames")
                else:
                    paramnames_files = []

            if paramnames_files:
                # Ensure we have exactly one paramnames file (no logging for intermediate calls)
                paramnames_file = self._select_unique_file(paramnames_files, "paramnames", 
                                                         self.object_id if self.object_id else objects[0], log_selection=False)
                free_params = []

                with open(paramnames_file, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#'):
                            # Split by whitespace - first column is parameter name
                            parts = line.split()
                            if parts:
                                param_name = parts[0]
                                # Free parameters don't have * suffix
                                if not param_name.endswith('*'):
                                    free_params.append(param_name)

                logger.debug(f"Found {len(free_params)} free parameters from paramnames file")
                return free_params

        except Exception as e:
            logger.debug(f"Could not read paramnames file: {e}. Using HDF5 structure analysis.")

        # Fallback: Analyze HDF5 table structure
        if self._hdf5_table is None:
            raise RuntimeError("HDF5 table not loaded")

        # Get all parameter names (excluding 'ID') in original order
        all_param_names = [col for col in self._hdf5_table.colnames if col != 'ID']
        
        # Filter to only parameters containing "{mean}" to get the main parameter list
        mean_params = [param for param in all_param_names if '{mean}' in param]
        
        if not mean_params:
            logger.warning("No parameters with '{mean}' found. Falling back to all parameters.")
            return all_param_names
        
        # Find the first parameter that starts with "log(scale)" or "log(norm)"
        free_params = []
        for param in mean_params:
            if param.startswith('log(scale)') or param.startswith('log(norm)'):
                # Found the boundary - stop here
                break
            # Strip the _{mean} suffix to get clean parameter name
            clean_param = param.replace('_{mean}', '')
            free_params.append(clean_param)
        
        logger.info(f"Found {len(free_params)} free parameters using HDF5 structure analysis")
        logger.debug(f"Free parameters: {free_params}")
        
        return free_params
    def get_derived_parameters(self) -> List[str]:
        """
        Get list of derived parameters by analyzing HDF5 table structure.
        
        In BayeSED3 HDF5 files, derived parameters start from the first parameter 
        that begins with "log(scale)" or "log(norm)" and continue to the end.

        Returns
        -------
        List[str]
            List of derived parameter names in original order
        """
        # First try to read paramnames file (original behavior)
        try:
            # Find a paramnames file to read parameter structure
            if self.object_id:
                # Object-level access - use specific object
                paramnames_files = self._find_config_files(self.object_id, "*_sample_par.paramnames")
            else:
                # Sample-level access - use first available object
                objects = self.list_objects()
                if objects:
                    paramnames_files = self._find_config_files(objects[0], "*_sample_par.paramnames")
                else:
                    paramnames_files = []

            if paramnames_files:
                # Ensure we have exactly one paramnames file (no logging for intermediate calls)
                paramnames_file = self._select_unique_file(paramnames_files, "paramnames", 
                                                         self.object_id if self.object_id else objects[0], log_selection=False)
                derived_params = []

                with open(paramnames_file, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#'):
                            # Split by whitespace - first column is parameter name
                            parts = line.split()
                            if parts:
                                param_name = parts[0]
                                # Derived parameters have * suffix
                                if param_name.endswith('*'):
                                    # Remove the * suffix for the returned name
                                    derived_params.append(param_name.rstrip('*'))

                logger.debug(f"Found {len(derived_params)} derived parameters from paramnames file")
                return derived_params

        except Exception as e:
            logger.debug(f"Could not read paramnames file: {e}. Using HDF5 structure analysis.")

        # Fallback: Analyze HDF5 table structure
        if self._hdf5_table is None:
            raise RuntimeError("HDF5 table not loaded")

        # Get all parameter names (excluding 'ID') in original order
        all_param_names = [col for col in self._hdf5_table.colnames if col != 'ID']
        
        # Filter to only parameters containing "{mean}" to get the main parameter list
        mean_params = [param for param in all_param_names if '{mean}' in param]
        
        if not mean_params:
            logger.warning("No parameters with '{mean}' found. Returning empty derived parameters list.")
            return []
        
        # Find the first parameter that starts with "log(scale)" or "log(norm)"
        derived_params = []
        found_boundary = False
        
        for param in mean_params:
            if param.startswith('log(scale)') or param.startswith('log(norm)'):
                # Found the boundary - start collecting derived parameters
                found_boundary = True
            
            if found_boundary:
                # Strip the _{mean} suffix to get clean parameter name
                clean_param = param.replace('_{mean}', '')
                derived_params.append(clean_param)
        
        logger.info(f"Found {len(derived_params)} derived parameters using HDF5 structure analysis")
        logger.debug(f"Derived parameters: {derived_params}")

        return derived_params
    def _transpose_parameter_table(self, table: 'astropy.table.Table') -> 'astropy.table.Table':
        """
        Transpose a parameter table so objects become columns and parameters become rows.

        Parameters
        ----------
        table : astropy.table.Table
            Input table with ID column and parameter columns

        Returns
        -------
        astropy.table.Table
            Transposed table with parameter names as rows and object values as columns
        """
        import pandas as pd
        from astropy.table import Table

        if len(table) == 0:
            return table

        # Get object IDs and parameter names
        object_ids = [str(obj_id) for obj_id in table['ID']]
        param_names = [col for col in table.colnames if col != 'ID']

        # Efficient: convert to pandas DataFrame, transpose, then back to astropy Table
        # pandas handles mixed types efficiently with its transpose
        df = table[param_names].to_pandas()
        df.index = object_ids
        df_transposed = df.T

        # Convert back to astropy Table
        new_table = Table.from_pandas(df_transposed)

        # Add parameter column as first column
        new_table['parameter'] = param_names
        # Move 'parameter' column to first position
        col_order = ['parameter'] + [c for c in new_table.colnames if c != 'parameter']
        new_table = new_table[col_order]

        logger.debug(f"Transposed table: {len(param_names)} parameters x {len(object_ids)} objects")
        return new_table
    def get_parameter_values(self, parameter_name: str,
                           object_ids: Optional[Union[str, List[str]]] = None,
                           param_match_mode: str = 'contains',
                           object_match_mode: str = 'contains',
                           case_sensitive: bool = True,
                           transpose: bool = False) -> 'astropy.table.Table':
        """
        Get parameter values by filtering the loaded HDF5 table with flexible matching.

        Parameters
        ----------
        parameter_name : str
            Parameter name or pattern to search for. Supports:
            - 'contains': Columns that contain this substring anywhere (default)
            - 'regex': Regular expression pattern matching
        object_ids : str or List[str], optional
            Specific object ID(s) to filter. Can be a single string or list of strings.
            Supports object_match_mode for flexible matching:
            - 'contains': Object IDs containing this substring
            - 'regex': Regular expression pattern matching
            - 'exact': Exact match
        param_match_mode : str, default 'contains'
            Matching mode for parameter names:
            - 'contains': Substring matching anywhere in column name
            - 'regex': Regular expression pattern matching
        object_match_mode : str, default 'contains'
            Matching mode for object IDs:
            - 'contains': Substring matching in object IDs
            - 'regex': Regular expression pattern matching
            - 'exact': Exact match
        case_sensitive : bool, default True
            Whether matching should be case sensitive
        transpose : bool, default False
            If True, transpose the table so objects become columns and
            parameters become rows. Useful for comparing parameters across objects.

        Returns
        -------
        astropy.table.Table
            Sub-table containing ID column and all parameter columns that
            match the given parameter name according to the specified mode.
            If transpose=True, returns transposed table with objects as columns.

        Examples
        --------
        >>> # Find all columns containing "mass" anywhere (default)
        >>> mass_table = results.get_parameter_values("mass")

        >>> # Case-insensitive search for columns containing "SFR"
        >>> sfr_table = results.get_parameter_values("sfr", case_sensitive=False)

        >>> # Get transposed table for comparing a parameter across objects
        >>> mass_table = results.get_parameter_values("mass", transpose=True)

        >>> # Regex pattern to find percentile columns
        >>> percentiles = results.get_parameter_values(r".*_(16|50|84)$", param_match_mode='regex')

        >>> # Find columns ending with "_err" or "_error" using regex
        >>> errors = results.get_parameter_values(r"_(err|error)$", param_match_mode='regex')

        >>> # Filter objects by regex pattern
        >>> results.get_parameter_values("z", object_ids=r".*_150135", object_match_mode='regex')
        """
        if self._hdf5_table is None:
            raise RuntimeError("HDF5 table not loaded")

        import re

        available_columns = self._hdf5_table.colnames
        matching_columns = self._match_strings(available_columns, parameter_name, param_match_mode, case_sensitive)

        if not matching_columns:
            error_msg = f"No columns found matching '{parameter_name}' with mode '{param_match_mode}'"
            if not case_sensitive:
                error_msg += " (case-insensitive)"
            error_msg += f" in HDF5 table.\nUse get_parameter_names() to see all available parameters."
            
            raise ValueError(error_msg)

        # Create sub-table with ID and all matching parameter columns (preserve original order)
        columns_to_include = ['ID'] + matching_columns
        sub_table = self._hdf5_table[columns_to_include]

        # Pre-convert object IDs to strings (handles bytes) - single pass
        all_object_ids_str = np.array([self._to_str(obj_id) for obj_id in sub_table['ID']])

        # Apply object filtering if specified
        if object_ids is not None:
            # Convert object_ids to list and decode bytes if needed
            if isinstance(object_ids, (str, bytes)):
                object_ids_list = [self._to_str(object_ids)]
            else:
                object_ids_list = [self._to_str(x) for x in object_ids]

            # Create combined mask for all patterns using vectorized operations
            combined_mask = np.zeros(len(all_object_ids_str), dtype=bool)

            # Pre-compute lowercase object IDs if case-insensitive (single pass)
            if not case_sensitive:
                all_object_ids_lower = np.array([obj_id.lower() for obj_id in all_object_ids_str])

            for pattern in object_ids_list:
                if object_match_mode == 'contains':
                    if case_sensitive:
                        mask = np.array([pattern in obj_id for obj_id in all_object_ids_str])
                    else:
                        pattern_lower = pattern.lower()
                        mask = np.array([pattern_lower in obj_id for obj_id in all_object_ids_lower])
                elif object_match_mode == 'regex':
                    try:
                        if case_sensitive:
                            compiled_pattern = re.compile(pattern)
                        else:
                            compiled_pattern = re.compile(pattern, re.IGNORECASE)
                        mask = np.array([bool(compiled_pattern.search(obj_id)) for obj_id in all_object_ids_str])
                    except re.error as e:
                        raise ValueError(f"Invalid regex pattern '{pattern}': {e}")
                elif object_match_mode == 'exact':
                    if case_sensitive:
                        mask = np.array([pattern == obj_id for obj_id in all_object_ids_str])
                    else:
                        pattern_lower = pattern.lower()
                        mask = np.array([pattern_lower == obj_id for obj_id in all_object_ids_lower])
                else:
                    raise ValueError(f"Invalid object_match_mode '{object_match_mode}'. "
                                   f"Must be 'contains', 'regex', or 'exact'")
                combined_mask = combined_mask | mask

            if np.any(combined_mask):
                sub_table = sub_table[combined_mask]
                # Update object ID array for second filter stage
                all_object_ids_str = all_object_ids_str[combined_mask]
            else:
                raise ValueError(f"None of the specified objects {object_ids} found in data "
                               f"with object_match_mode='{object_match_mode}'. "
                               f"Available objects: {list(all_object_ids_str)}")

        # Apply scope filtering for object-level access
        if self.object_id is not None:
            self_object_id_str = self._to_str(self.object_id)
            mask = all_object_ids_str == self_object_id_str
            if np.any(mask):
                sub_table = sub_table[mask]
            else:
                raise ValueError(f"Object '{self.object_id}' not found in data")

        logger.debug(f"Found {len(matching_columns)} columns for parameter '{parameter_name}': {matching_columns}")

        # Transpose table if requested (always last)
        if transpose:
            sub_table = self._transpose_parameter_table(sub_table)

        # Return sub-table for sample-level access
        return sub_table
    def list_objects(self) -> List[str]:
        """
        List available objects.

        Returns
        -------
        List[str]
            List of object IDs
        """
        if self._hdf5_table is None:
            raise RuntimeError("HDF5 table not loaded")

        # Extract object IDs from the table
        object_ids = [str(obj_id) for obj_id in self._hdf5_table['ID']]

        # Apply scope filtering for object-level access
        if self.object_id is not None:
            if self.object_id in object_ids:
                return [self.object_id]
            else:
                return []

        return object_ids
    def load_hdf5_results(self, filter_snr: bool = True, min_snr: float = 0.0) -> 'astropy.table.Table':
        """
        Load HDF5 results table with optional SNR filtering.

        Parameters
        ----------
        filter_snr : bool, default True
            Whether to filter by SNR
        min_snr : float, default 0.0
            Minimum SNR threshold

        Returns
        -------
        astropy.table.Table
            HDF5 results table
        """
        if self._hdf5_table is None:
            raise RuntimeError("HDF5 table not loaded")

        # Apply SNR filtering if requested
        if filter_snr and 'SNR' in self._hdf5_table.colnames:
            mask = self._hdf5_table['SNR'] > min_snr
            filtered_table = self._hdf5_table[mask]
            logger.info(f"Filtered {len(self._hdf5_table)} objects to {len(filtered_table)} with SNR > {min_snr}")
        else:
            filtered_table = self._hdf5_table
            logger.info(f"Loaded {len(filtered_table)} objects (no SNR filtering)")

        # Apply object-level filtering if needed
        if self.object_id is not None:
            object_mask = [str(obj_id) == str(self.object_id) for obj_id in filtered_table['ID']]
            if any(object_mask):
                return filtered_table[object_mask]
            else:
                raise ValueError(f"Object '{self.object_id}' not found in data")

        return filtered_table
    def get_evidence(self, object_ids: Optional[List[str]] = None,
                    return_format: str = 'auto') -> Union[Dict[str, float], 'astropy.table.Table', float]:
        """
        Get Bayesian evidence values in a user-friendly format.

        Parameters
        ----------
        object_ids : str or List[str], optional
            Specific object ID(s) to get evidence for. Can be a single string or list of strings.
            If None, uses current scope.
        return_format : str, default 'auto'
            Format for returned data:
            - 'auto': Smart format based on scope and number of objects
            - 'table': Always return astropy Table with object IDs
            - 'dict': Return dictionary format
            - 'value': Return single evidence value (for single object)

        Returns
        -------
        Union[Dict[str, float], astropy.table.Table, float]
            Evidence values in requested format:
            - Single object: Dict with evidence statistics or single value
            - Multiple objects: astropy Table with object IDs and evidence

        Examples
        --------
        >>> # Single object (object-level access)
        >>> evidence = results.get_evidence()  # Returns dict with logZ, logZerr, etc.
        >>> print(f"Evidence: {evidence['INSlogZ']:.2f} ± {evidence['INSlogZerr']:.2f}")

        >>> # Multiple objects as table
        >>> evidence_table = results.get_evidence(return_format='table')
        >>> print(evidence_table['ID', 'INSlogZ', 'INSlogZerr'])

        >>> # Just the evidence value for single object
        >>> logZ = results.get_evidence(return_format='value')  # Returns float
        """
        # Load the HDF5 results table
        hdf5_table = self.load_hdf5_results(filter_snr=False, min_snr=0.0)

        # Define evidence columns in order of preference
        evidence_columns = ['INSlogZ', 'logZ', 'INSlogZerr', 'logZerr']
        available_evidence_cols = [col for col in evidence_columns if col in hdf5_table.colnames]

        if not available_evidence_cols:
            raise ValueError(f"No evidence columns found in HDF5 table. "
                           f"Expected columns: {evidence_columns}")

        # Apply object filtering if specified
        if object_ids is not None:
            # Convert single string to list for consistent handling
            if isinstance(object_ids, str):
                object_ids = [object_ids]

            # Filter table by object IDs
            object_mask = [str(obj_id) in [str(x) for x in object_ids] for obj_id in hdf5_table['ID']]
            if any(object_mask):
                filtered_table = hdf5_table[object_mask]
            else:
                available_objects = [str(obj_id) for obj_id in hdf5_table['ID']]
                raise ValueError(f"None of the specified objects {object_ids} found in data. "
                               f"Available objects: {available_objects}")
        else:
            filtered_table = hdf5_table

        # Determine return format
        is_single_object = (self.object_id is not None or len(filtered_table) == 1)

        if return_format == 'auto':
            if is_single_object:
                return_format = 'dict'
            else:
                return_format = 'table'

        # Extract relevant columns (ID + evidence columns)
        columns_to_extract = ['ID'] + available_evidence_cols
        evidence_table = filtered_table[columns_to_extract]

        # Return in requested format
        if return_format == 'table':
            logger.info(f"Returning evidence table for {len(evidence_table)} objects")
            return evidence_table

        elif return_format == 'dict':
            if is_single_object:
                # Single object - return clean dictionary
                row = evidence_table[0]
                result = {
                    'object_id': str(row['ID']),
                    'log_evidence': float(row.get('INSlogZ', row.get('logZ', float('nan')))),
                    'log_evidence_error': float(row.get('INSlogZerr', row.get('logZerr', float('nan'))))
                }

                # Add all available evidence columns
                for col in available_evidence_cols:
                    result[col] = float(row[col])

                logger.info(f"Returning evidence dict for object {result['object_id']}: "
                          f"logZ = {result['log_evidence']:.2f} ± {result['log_evidence_error']:.2f}")
                return result
            else:
                # Multiple objects - return dict of arrays
                result = {}
                for col in available_evidence_cols:
                    result[col] = evidence_table[col]
                result['object_ids'] = [str(obj_id) for obj_id in evidence_table['ID']]

                logger.info(f"Returning evidence dict for {len(evidence_table)} objects")
                return result

        elif return_format == 'value':
            if not is_single_object:
                raise ValueError("return_format='value' only supported for single object access")

            row = evidence_table[0]
            # Return the best available evidence value
            evidence_value = float(row.get('INSlogZ', row.get('logZ', float('nan'))))

            logger.info(f"Returning evidence value for object {row['ID']}: {evidence_value:.2f}")
            return evidence_value

        else:
            raise ValueError(f"Invalid return_format '{return_format}'. "
                           f"Must be one of: 'auto', 'table', 'dict', 'value'")
    def get_posterior_samples(self, object_id: Optional[str] = None, 
                             settings: Optional[Dict[str, Any]] = None) -> Any:
        """
        Get posterior samples by loading GetDist samples on-demand.

        Parameters
        ----------
        object_id : str, optional
            Object ID to get samples for
        settings : dict, optional
            Settings dictionary to pass to getdist.loadMCSamples.
            Default: {'range_confidence': 0.001, 'min_weight_ratio': 1e-99}

        Returns
        -------
        Any
            GetDist samples object or astropy Table if GetDist not available
        """
        # Determine object to load
        if object_id is None:
            if self.object_id is not None:
                object_id = self.object_id
            else:
                # For sample-level access, use first available object with warning
                objects = self.list_objects()
                if objects:
                    object_id = objects[0]
                    logger.warning(f"No object_id provided for get_posterior_samples. Using first available object: {object_id}. More objects can be obtained with results.list_objects().")
                else:
                    raise ValueError("No objects available for posterior samples")

        # Find sample files for this object matching the current configuration
        sample_files = self._find_config_files(object_id, "*_sample_par.txt")
        paramnames_files = self._find_config_files(object_id, "*_sample_par.paramnames")

        # Ensure we have exactly one of each file type (no logging for intermediate calls)
        sample_file = self._select_unique_file(sample_files, "sample_par.txt", object_id, log_selection=False)
        paramnames_file = self._select_unique_file(paramnames_files, "paramnames", object_id, log_selection=False)

        # Extract base name for GetDist (remove .txt suffix, keep _sample_par)
        base_name = str(sample_file).replace(".txt", "")

        try:
            # Try to use GetDist - pass the full base name including _sample_par
            import getdist
            
            # Set default settings if none provided
            if settings is None:
                settings = {'range_confidence': 0.001, 'min_weight_ratio': 1e-99}
            
            samples = getdist.loadMCSamples(base_name, settings=settings)
            logger.info(f"Loaded GetDist samples for {object_id}: {samples.numrows} samples, {samples.n} parameters")
            return samples

        except ImportError:
            logger.warning("GetDist not available, loading raw sample data")

            # Fallback: load as astropy Table
            from astropy.table import Table

            # Load parameter names
            param_names = []
            with open(paramnames_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        param_names.append(line.split()[0])

            # Load samples
            samples_data = np.loadtxt(sample_file)

            # Create astropy Table
            if len(param_names) == samples_data.shape[1]:
                samples_table = Table(samples_data, names=param_names)
            else:
                # Generic column names if mismatch
                col_names = [f'param_{i}' for i in range(samples_data.shape[1])]
                samples_table = Table(samples_data, names=col_names)

            logger.info(f"Loaded raw samples for {object_id}: {len(samples_table)} samples, {len(samples_table.colnames)} parameters")
            return samples_table
    def get_bestfit_spectrum(self, object_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get best-fit spectrum by loading FITS files on-demand.

        Parameters
        ----------
        object_id : str, optional
            Object ID to get spectrum for

        Returns
        -------
        Dict[str, Any]
            Best-fit spectrum data loaded from FITS files
        """
        # Determine object to load
        if object_id is None:
            if self.object_id is not None:
                object_id = self.object_id
            else:
                # For sample-level access, use first available object with warning
                objects = self.list_objects()
                if objects:
                    object_id = objects[0]
                    logger.warning(f"No object_id provided for get_bestfit_spectrum. Using first available object: {object_id}. More objects can be obtained with results.list_objects().")
                else:
                    raise ValueError("No objects available for best-fit spectrum")

        # Find FITS files for this object matching the current configuration
        fits_files = self._find_config_files(object_id, "*_bestfit.fits")

        if not fits_files:
            object_dir = self.output_dir / self.catalog_name / object_id
            raise FileNotFoundError(f"No bestfit FITS files found for object {object_id} and config '{self.model_config}' in {object_dir}")

        try:
            from astropy.io import fits

            spectrum_data = {}

            # Load all FITS files for this object
            for i, fits_file in enumerate(fits_files):
                config_name = fits_file.stem.replace("_bestfit", "")

                with fits.open(fits_file) as hdul:
                    file_data = {}
                    for j, hdu in enumerate(hdul):
                        if hdu.data is not None:
                            file_data[f'hdu_{j}'] = {
                                'data': hdu.data,
                                'header': dict(hdu.header) if hdu.header else {}
                            }

                    spectrum_data[config_name] = file_data

                logger.debug(f"Loaded FITS file: {fits_file.name}")

            logger.info(f"Loaded bestfit spectrum for {object_id}: {len(fits_files)} FITS files")
            return spectrum_data

        except ImportError:
            raise ImportError("astropy is required for FITS file loading. Install with: pip install astropy")
    def list_model_components(self, object_id: Optional[str] = None) -> List[str]:
        """
        List available model components in the best-fit FITS file.

        Parameters
        ----------
        object_id : str, optional
            Object ID to get components for. If None, uses first available object.

        Returns
        -------
        List[str]
            List of model component names (HDU names like 'model:stellar', 'model:tor')

        Examples
        --------
        >>> # Get available model components
        >>> components = results.list_model_components()
        >>> print(components)  # ['model:stellar', 'model:tor', 'model:blr']
        >>>
        >>> # Use with plot_bestfit to show/hide specific components
        >>> results.plot_bestfit(show_components={'model:tor': False})
        """
        if object_id is None:
            if self.object_id is not None:
                object_id = self.object_id
            else:
                objects = self.list_objects()
                if objects:
                    object_id = objects[0]
                else:
                    raise ValueError("No objects available")

        fits_files = self._find_config_files(object_id, "*_bestfit.fits")
        if not fits_files:
            object_dir = self.output_dir / self.catalog_name / object_id
            raise FileNotFoundError(f"No bestfit FITS files found for object {object_id} in {object_dir}")

        fits_file = self._select_unique_file(fits_files, "bestfit FITS", object_id, log_selection=False)

        from astropy.io import fits
        with fits.open(fits_file) as hdul:
            components = [h.name for h in hdul if h.name.startswith('model:') and h.name != 'model:total']

        logger.info(f"Found {len(components)} model components: {components}")
        return components
    def print_summary(self) -> None:
        """Print a summary of the loaded results."""
        print("=" * 60)
        print("BayeSEDResults Summary")
        print("=" * 60)
        print(f"Output Directory: {self.output_dir}")
        print(f"Catalog: {self.catalog_name}")
        print(f"HDF5 File: {Path(self._hdf5_file).name}")
        if self.object_id:
            print(f"Object ID: {self.object_id}")
        print()

        # Parameter summary
        try:
            all_params = self.get_parameter_names()
            free_params = self.get_free_parameters()
            derived_params = self.get_derived_parameters()

            print("Parameter Summary:")
            print(f"  HDF5 Columns: {len(all_params)}")
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
        if self.object_id:
            return f"BayeSEDResults(catalog='{self.catalog_name}', object_id='{self.object_id}')"
        else:
            return f"BayeSEDResults(catalog='{self.catalog_name}', sample-level)"
    def __str__(self) -> str:
        """Human-readable string representation."""
        return self.__repr__()
    def rename_parameters(self, parameter_mapping: Dict[str, str]) -> None:
        """
        Rename parameters for consistency (simplified implementation).

        Note: In the simplified implementation, this updates the custom labels
        used for plotting rather than modifying the underlying data.

        Parameters
        ----------
        parameter_mapping : Dict[str, str]
            Dictionary mapping old parameter names to new names
        """
        # In simplified implementation, we update the custom labels
        if not hasattr(self, '_custom_labels'):
            self._custom_labels = {}

        # Apply the parameter mapping to custom labels
        for old_name, new_name in parameter_mapping.items():
            self._custom_labels[old_name] = new_name

        logger.info(f"Applied parameter renaming for {len(parameter_mapping)} parameters")
        logger.debug(f"Parameter mapping: {parameter_mapping}")
    def list_model_configurations(self) -> List[str]:
        """
        List available model configurations for the current catalog.

        Returns
        -------
        List[str]
            List of configuration names that belong to the current catalog
        """
        if not self.catalog_name:
            raise ValueError("Cannot list configurations without a catalog name")

        # Look for HDF5 files that match the catalog pattern: {catalog_name}_{config}.hdf5
        pattern = f"{self.catalog_name}_*.hdf5"
        hdf5_files = list(self.output_dir.glob(pattern))

        configurations = []
        for hdf5_file in hdf5_files:
            # Extract configuration name from filename
            filename = hdf5_file.stem
            # Remove catalog name prefix: "gal_config" -> "config"
            if filename.startswith(self.catalog_name + '_'):
                config_name = filename[len(self.catalog_name) + 1:]
                configurations.append(config_name)

        # Sort configurations for consistent ordering
        configurations.sort()

        logger.info(f"Found {len(configurations)} model configurations for catalog '{self.catalog_name}'")
        return configurations
    def get_configuration_summary(self) -> Dict[str, Any]:
        """
        Get summary of model configurations for the current catalog.

        Returns
        -------
        Dict[str, Any]
            Summary of available configurations for the current catalog
        """
        configurations = self.list_model_configurations()

        # Extract current configuration name from the loaded HDF5 file
        current_config = None
        if self._hdf5_file:
            filename = Path(self._hdf5_file).stem
            if filename.startswith(self.catalog_name + '_'):
                current_config = filename[len(self.catalog_name) + 1:]

        summary = {
            'catalog_name': self.catalog_name,
            'total_configurations': len(configurations),
            'configurations': configurations,
            'current_configuration': current_config,
            'output_directory': str(self.output_dir),
            'hdf5_file': Path(self._hdf5_file).name if self._hdf5_file else None
        }

        return summary
    def validate_model_config(self, model_config: Union[str, int]) -> str:
        """
        Validate model configuration and return the full configuration name.

        Parameters
        ----------
        model_config : str or int
            Model configuration to validate (name or index)

        Returns
        -------
        str
            Full configuration name if valid

        Raises
        ------
        ValueError
            If the model configuration is not found
        """
        if not self.catalog_name:
            raise ValueError("Cannot validate model config without a catalog name")

        # Get available configurations
        available_configs = self.list_model_configurations()

        if not available_configs:
            raise ValueError(f"No model configurations found for catalog '{self.catalog_name}'")

        if isinstance(model_config, int):
            # Integer index
            if 0 <= model_config < len(available_configs):
                return available_configs[model_config]
            else:
                raise ValueError(f"Model config index {model_config} out of range. "
                               f"Available configs (0-{len(available_configs)-1}): {available_configs}")

        else:
            # String name
            model_config_str = str(model_config)

            # Check for exact match
            if model_config_str in available_configs:
                return model_config_str

            # Check for partial match
            matches = [config for config in available_configs if model_config_str in config]

            if len(matches) == 1:
                return matches[0]
            elif len(matches) > 1:
                raise ValueError(f"Ambiguous model config '{model_config_str}'. "
                               f"Multiple matches found: {matches}")
            else:
                raise ValueError(f"Model config '{model_config_str}' not found for catalog '{self.catalog_name}'. "
                               f"Available configs: {available_configs}")
