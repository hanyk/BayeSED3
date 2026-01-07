import os
import h5py
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Union, Any
import logging

# Set up simple logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BayeSEDResults:
    """
    BayeSEDResults class with integrated functionality.

    This class provides the same API as the original BayeSEDResults but with
    a much simpler internal implementation:
    - Loads HDF5 data once during initialization using astropy Table + h5py
    - No complex component architecture or caching systems
    - All data access through filtering the loaded HDF5 table
    - On-demand loading for GetDist samples and FITS spectra

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

    Attributes
    ----------
    parameters : astropy.table.Table
        The loaded HDF5 parameter table containing all objects and their
        fitted parameter values. Available immediately after initialization.
        Contains 'ID' column plus all parameter columns from the HDF5 file.

    """

    def __init__(self, output_dir: Union[str, Path], catalog_name: Optional[str] = None,
                 model_config: Optional[Union[str, int]] = None, object_id: Optional[str] = None):
        """Initialize BayeSEDResults."""

        # Store initialization parameters
        self.output_dir = Path(output_dir).resolve()
        self.catalog_name = catalog_name
        self.model_config = model_config
        self.object_id = object_id

        # Main data storage - loaded once
        self._hdf5_table = None
        self._hdf5_file = None
        self.parameters = None  # Public access to HDF5 table
        
        # Cache for GetDist samples to avoid duplicate loading
        self._getdist_samples_cache = {}

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
                    # No exact matches - try partial matching
                    partial_matches = []
                    for f in catalog_files:
                        filename = f.stem
                        if filename.startswith(self.catalog_name + '_'):
                            config_name = filename[len(self.catalog_name) + 1:]
                            if model_config_str in config_name:
                                partial_matches.append((config_name, f))
                    
                    if len(partial_matches) == 1:
                        # Found exactly one partial match
                        config_name, target_file = partial_matches[0]
                        self.model_config = config_name
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

    # ========================================================================
    # Public API Methods - Maintaining backward compatibility
    # ========================================================================

    def get_parameter_names(self, pattern: Optional[str] = None,
                           match_mode: str = 'contains',
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
        match_mode : str, default 'contains'
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
        >>> percentiles = results.get_parameter_names(r".*_(16|50|84)$", match_mode='regex')
        
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
        import re
        matching_params = []

        if match_mode == 'contains':
            # Substring matching anywhere in parameter name
            if case_sensitive:
                for param in all_param_names:
                    if pattern in param:
                        matching_params.append(param)
            else:
                search_pattern = pattern.lower()
                for param in all_param_names:
                    if search_pattern in param.lower():
                        matching_params.append(param)

        elif match_mode == 'regex':
            # Regular expression matching
            try:
                if case_sensitive:
                    regex_pattern = re.compile(pattern)
                else:
                    regex_pattern = re.compile(pattern, re.IGNORECASE)
                
                for param in all_param_names:
                    if regex_pattern.search(param):
                        matching_params.append(param)
            except re.error as e:
                raise ValueError(f"Invalid regular expression '{pattern}': {e}")

        else:
            raise ValueError(f"Invalid match_mode '{match_mode}'. Must be 'contains' or 'regex'")

        if not matching_params and pattern is not None:
            # Provide helpful error message
            error_msg = f"No parameters found matching '{pattern}' with mode '{match_mode}'"
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

    def get_parameter_values(self, parameter_name: str,
                           object_ids: Optional[Union[str, List[str]]] = None,
                           match_mode: str = 'contains',
                           case_sensitive: bool = True) -> 'astropy.table.Table':
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
        match_mode : str, default 'contains'
            Matching mode for parameter names:
            - 'contains': Substring matching anywhere in column name
            - 'regex': Regular expression pattern
        case_sensitive : bool, default True
            Whether matching should be case sensitive

        Returns
        -------
        astropy.table.Table
            Sub-table containing ID column and all parameter columns that
            match the given parameter name according to the specified mode

        Examples
        --------
        >>> # Find all columns containing "mass" anywhere (default)
        >>> mass_table = results.get_parameter_values("mass")
        
        >>> # Case-insensitive search for columns containing "SFR"
        >>> sfr_table = results.get_parameter_values("sfr", case_sensitive=False)
        
        >>> # Regex pattern to find percentile columns
        >>> percentiles = results.get_parameter_values(r".*_(16|50|84)$", match_mode='regex')
        
        >>> # Find columns ending with "_err" or "_error" using regex
        >>> errors = results.get_parameter_values(r"_(err|error)$", match_mode='regex')
        """
        if self._hdf5_table is None:
            raise RuntimeError("HDF5 table not loaded")

        import re

        available_columns = self._hdf5_table.colnames
        matching_columns = []

        if match_mode == 'contains':
            # Substring matching anywhere in column name
            if case_sensitive:
                for col in available_columns:
                    if parameter_name in col:
                        matching_columns.append(col)
            else:
                search_pattern = parameter_name.lower()
                for col in available_columns:
                    if search_pattern in col.lower():
                        matching_columns.append(col)

        elif match_mode == 'regex':
            # Regular expression matching
            try:
                if case_sensitive:
                    pattern = re.compile(parameter_name)
                else:
                    pattern = re.compile(parameter_name, re.IGNORECASE)
                
                for col in available_columns:
                    if pattern.search(col):
                        matching_columns.append(col)
            except re.error as e:
                raise ValueError(f"Invalid regular expression '{parameter_name}': {e}")

        else:
            raise ValueError(f"Invalid match_mode '{match_mode}'. Must be 'contains' or 'regex'")

        if not matching_columns:
            error_msg = f"No columns found matching '{parameter_name}' with mode '{match_mode}'"
            if not case_sensitive:
                error_msg += " (case-insensitive)"
            error_msg += f" in HDF5 table.\nUse get_parameter_names() to see all available parameters."
            
            raise ValueError(error_msg)

        # Create sub-table with ID and all matching parameter columns (preserve original order)
        columns_to_include = ['ID'] + matching_columns
        sub_table = self._hdf5_table[columns_to_include]

        # Apply object filtering if specified
        if object_ids is not None:
            # Convert single string to list for consistent handling
            if isinstance(object_ids, str):
                object_ids = [object_ids]

            # Filter table by object IDs
            object_mask = [str(obj_id) in [str(x) for x in object_ids] for obj_id in sub_table['ID']]
            if any(object_mask):
                filtered_table = sub_table[object_mask]
                return filtered_table
            else:
                available_objects = [str(obj_id) for obj_id in sub_table['ID']]
                raise ValueError(f"None of the specified objects {object_ids} found in data. "
                               f"Available objects: {available_objects}")

        # Apply scope filtering for object-level access
        if self.object_id is not None:
            # Find the specific object
            object_mask = [str(obj_id) == str(self.object_id) for obj_id in sub_table['ID']]
            if any(object_mask):
                return sub_table[object_mask]
            else:
                raise ValueError(f"Object '{self.object_id}' not found in data")

        logger.debug(f"Found {len(matching_columns)} columns for parameter '{parameter_name}': {matching_columns}")

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

    def get_getdist_samples(self, object_id: Optional[str] = None, 
                           settings: Optional[Dict[str, Any]] = None) -> Any:
        """
        Get GetDist samples with parameter management.

        Parameters
        ----------
        object_id : str, optional
            Object ID to get samples for
        settings : dict, optional
            Settings dictionary to pass to getdist.loadMCSamples.
            Default: {'range_confidence': 0.001, 'min_weight_ratio': 1e-99}

        Returns
        -------
        getdist.MCSamples
            GetDist samples object
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
                    logger.warning(f"No object_id provided for get_getdist_samples. Using first available object: {object_id}. More objects can be obtained with results.list_objects().")
                else:
                    raise ValueError("No objects available for GetDist samples")

        # Check cache first to avoid duplicate loading
        # Include settings in cache key to handle different settings
        settings_key = str(sorted(settings.items())) if settings else "default"
        cache_key = f"{object_id}_{self.model_config}_{settings_key}"
        if cache_key in self._getdist_samples_cache:
            return self._getdist_samples_cache[cache_key]

        # Find sample files for this object matching the current configuration
        sample_files = self._find_config_files(object_id, "*_sample_par.txt")
        paramnames_files = self._find_config_files(object_id, "*_sample_par.paramnames")

        # Ensure we have exactly one of each file type
        sample_file = self._select_unique_file(sample_files, "sample_par.txt", object_id, log_selection=False)
        paramnames_file = self._select_unique_file(paramnames_files, "paramnames", object_id, log_selection=False)

        # Log the selected files once here (main entry point for GetDist operations)
        try:
            sample_rel_path = sample_file.resolve().relative_to(Path.cwd().resolve())
            logger.info(f"Loading GetDist samples for {object_id}: {sample_rel_path}")
        except ValueError:
            # If files are not relative to cwd, show absolute paths
            logger.info(f"Loading GetDist samples for {object_id}: {sample_file.resolve()}")

        # Extract base name for GetDist (remove .txt suffix, keep _sample_par)
        base_name = str(sample_file).replace(".txt", "")

        try:
            # Try to use GetDist - pass the full base name including _sample_par
            import getdist
            from getdist import MCSamples

            # Set default settings if none provided
            if settings is None:
                settings = {'range_confidence': 0.001, 'min_weight_ratio': 1e-99}

            samples = getdist.loadMCSamples(base_name, settings=settings)

            # Apply parameter labeling if custom labels are set
            if hasattr(self, '_custom_labels') and self._custom_labels:
                # Get the parameter names and labels
                param_names = [param.name for param in samples.paramNames.names]
                param_labels = [param.label for param in samples.paramNames.names]

                # Apply custom labels
                updated_labels = []
                for i, name in enumerate(param_names):
                    if name in self._custom_labels:
                        updated_labels.append(self._custom_labels[name])
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

            logger.info(f"Loaded GetDist samples for {object_id}: {samples.numrows} samples, {samples.n} parameters")
            
            # Cache the samples to avoid duplicate loading
            self._getdist_samples_cache[cache_key] = samples
            return samples

        except ImportError:
            raise ImportError("GetDist is required for GetDist samples. Install with: pip install getdist")

    def set_parameter_labels(self, custom_labels: Dict[str, str]) -> None:
        """
        Set custom parameter labels for plotting.

        Parameters
        ----------
        custom_labels : Dict[str, str]
            Dictionary mapping parameter names to LaTeX labels
        """
        # Store custom labels for use in GetDist plotting
        self._custom_labels = custom_labels.copy()
        
        # Clear GetDist samples cache since labels have changed
        self._getdist_samples_cache.clear()
        
        logger.info(f"Set custom labels for {len(custom_labels)} parameters")

    def plot_posterior(self, object_id: Optional[str] = None,
                      params: Optional[List[str]] = None,
                      method: str = 'getdist', filled: bool = True,
                      show: bool = True, output_file: Optional[str] = None,
                      show_median: bool = True, show_confidence_intervals: bool = True,
                      confidence_level: float = 0.68, **kwargs) -> Any:
        """
        Plot posterior distributions using GetDist.

        Parameters
        ----------
        object_id : str, optional
            Object ID to plot for. If None, uses current scope.
        params : List[str], optional
            Parameters to plot. If None, uses free parameters.
        method : str, default 'getdist'
            Plotting method to use
        filled : bool, default True
            Whether to use filled contours
        show : bool, default True
            Whether to display the plot
        output_file : str, optional
            Output file path for saving
        show_median : bool, default True
            Whether to show median markers on plots
        show_confidence_intervals : bool, default True
            Whether to show confidence intervals on 1D marginal plots
        confidence_level : float, default 0.68
            Confidence level for intervals (0.68 for 1-sigma, 0.95 for 2-sigma)
        **kwargs
            Additional plotting arguments

        Returns
        -------
        Any
            Plot object (depends on method)
        """
        # Handle object_id selection
        if object_id is None:
            if self.object_id is not None:
                object_id = self.object_id
            else:
                # For sample-level access, use first available object with warning
                objects = self.list_objects()
                if objects:
                    object_id = objects[0]
                    logger.warning(f"No object_id provided for plot_posterior. Using first available object: {object_id}. More objects can be obtained with results.list_objects().")
                else:
                    raise ValueError("No objects available for plotting")

        # Get samples using GetDist
        samples = self.get_getdist_samples(object_id=object_id)

        # Use default parameters if none specified
        if params is None:
            params = self.get_free_parameters()

        # Import GetDist plotting
        try:
            from getdist import plots
            import matplotlib.pyplot as plt
            import numpy as np
        except ImportError:
            raise ImportError("GetDist is required for plotting. Install with: pip install getdist")

        # Helper functions for confidence intervals
        def _compute_statistics(samples_gd, param_names):
            """Compute medians and confidence intervals using GetDist built-in methods."""
            markers = {}
            confidence_intervals = {}

            if not (show_median or show_confidence_intervals):
                return markers, confidence_intervals

            for param in param_names:
                try:
                    # Use GetDist's built-in 1D density analysis
                    density = samples_gd.get1DDensity(param)
                    
                    if show_median:
                        # Use GetDist's built-in median calculation (properly weighted)
                        # The most reliable way is to use the 1D density's percentile method
                        try:
                            # Get the 50th percentile (median) from the 1D density
                            # This properly handles weighted samples
                            markers[param] = density.P50()  # 50th percentile = median
                        except AttributeError:
                            try:
                                # Alternative: use the confidence interval method
                                # Get the central value (median) by using a very small confidence interval
                                central_val = samples_gd.confidence(param, 0.001)  # Very narrow interval around median
                                if central_val is not None:
                                    if hasattr(central_val, '__len__') and len(central_val) > 0:
                                        markers[param] = central_val[0] if len(central_val) == 1 else (central_val[0] + central_val[1]) / 2
                                    else:
                                        markers[param] = central_val
                                else:
                                    markers[param] = density.mean
                            except:
                                # Final fallback: use mean
                                markers[param] = density.mean
                    
                    if show_confidence_intervals:
                        # Use GetDist's built-in confidence interval calculation
                        # This properly handles weighted samples and edge cases
                        lower = density.getLower(confidence_level)
                        upper = density.getUpper(confidence_level)
                        confidence_intervals[param] = (lower, upper)
                        
                except Exception as e:
                    logger.debug(f"GetDist built-in statistics failed for parameter {param}: {e}")
                    # Fallback to manual calculation for this parameter
                    try:
                        gd_param_names = samples_gd.getParamNames().list()
                        if param not in gd_param_names:
                            continue
                            
                        idx = gd_param_names.index(param)
                        samples_array = samples_gd.samples[:, idx]
                        weights = getattr(samples_gd, 'weights', None)

                        if show_median:
                            if weights is not None:
                                markers[param] = np.average(samples_array, weights=weights)
                            else:
                                markers[param] = np.median(samples_array)

                        if show_confidence_intervals:
                            lower_percentile = (1 - confidence_level) / 2
                            upper_percentile = 1 - lower_percentile
                            
                            if weights is not None:
                                # Weighted quantiles
                                sorted_idx = np.argsort(samples_array)
                                sorted_samples = samples_array[sorted_idx]
                                sorted_weights = weights[sorted_idx]
                                cumsum_weights = np.cumsum(sorted_weights)
                                cumsum_weights = cumsum_weights / cumsum_weights[-1]
                                lower_idx = np.searchsorted(cumsum_weights, lower_percentile)
                                upper_idx = np.searchsorted(cumsum_weights, upper_percentile)
                                confidence_intervals[param] = (sorted_samples[lower_idx], sorted_samples[upper_idx])
                            else:
                                quantiles = np.quantile(samples_array, [lower_percentile, upper_percentile])
                                confidence_intervals[param] = (quantiles[0], quantiles[1])
                    except Exception as fallback_e:
                        logger.debug(f"Fallback statistics computation also failed for {param}: {fallback_e}")

            return markers, confidence_intervals

        def _add_ci_to_1d_plot(ax, lower, upper, confidence_level):
            """Add confidence interval to 1D plot."""
            ax.axvspan(lower, upper, color='blue', alpha=0.15)
            # No legend - would be cluttered in corner plots

        def _add_ci_to_triangle_plot(fig, params, confidence_intervals, confidence_level):
            """Add confidence intervals to diagonal (1D marginal) plots in triangle plot."""
            if not fig or not hasattr(fig, 'axes'):
                return

            # Find diagonal axes: 1D marginals have lines but no collections
            diagonal_axes = [
                ax for ax in fig.axes
                if (len(ax.lines) > 0 and len(ax.collections) == 0 and
                    (not ax.get_ylabel() or ax.get_ylabel() == ''))
            ]

            if len(diagonal_axes) != len(params):
                # Fallback: match by parameter name in xlabel
                import re
                for param in params:
                    if param not in confidence_intervals:
                        continue
                    lower, upper = confidence_intervals[param]
                    param_base = param.split('[')[0].strip()

                    for ax in diagonal_axes:
                        xlabel = ax.get_xlabel() or ''
                        # Try matching parameter name
                        if (param in xlabel or param_base in xlabel or
                            param_base in re.sub(r'\\[a-zA-Z]+\{([^}]+)\}', r'\1', xlabel)):
                            _add_ci_to_1d_plot(ax, lower, upper, confidence_level)
                            break
            else:
                # Match by position (most reliable)
                for idx, param in enumerate(params):
                    if param in confidence_intervals and idx < len(diagonal_axes):
                        lower, upper = confidence_intervals[param]
                        _add_ci_to_1d_plot(diagonal_axes[idx], lower, upper, confidence_level)

        # Compute statistics if needed
        markers = {}
        confidence_intervals = {}
        if show_median or show_confidence_intervals:
            try:
                markers, confidence_intervals = _compute_statistics(samples, params)
            except Exception as e:
                import warnings
                warnings.warn(
                    f"Could not compute statistics: {e}. Statistics will not be displayed.",
                    UserWarning
                )
                show_median = False
                show_confidence_intervals = False

        # Create plotter
        g = plots.get_subplot_plotter()
        
        # Configure publication-ready font sizes for GetDist plots
        g.settings.figure_legend_frame = True
        g.settings.legend_fontsize = 14
        g.settings.axes_fontsize = 16
        g.settings.lab_fontsize = 16
        g.settings.axes_labelsize = 16

        # Create plot based on number of parameters
        is_1d = len(params) == 1
        
        if is_1d:
            # 1D plot
            g.plot_1d(samples, params[0], **kwargs)
            
            # Add confidence interval for 1D plot
            if show_confidence_intervals and params[0] in confidence_intervals:
                ax = plt.gca()
                lower, upper = confidence_intervals[params[0]]
                _add_ci_to_1d_plot(ax, lower, upper, confidence_level)
        else:
            # Triangle plot with markers and confidence intervals
            g.triangle_plot([samples], params, filled=filled,
                           markers=markers if markers else None,
                           marker_args={'color': 'red', 'linestyle': '--',
                                      'linewidth': 1.5, 'alpha': 0.7},
                           **kwargs)

            # Add confidence intervals to diagonal plots
            if show_confidence_intervals:
                try:
                    fig = (getattr(g, 'fig', None) or
                          getattr(g, 'figure', None) or
                          plt.gcf())
                    _add_ci_to_triangle_plot(fig, params, confidence_intervals, confidence_level)
                except Exception as e:
                    import warnings
                    warnings.warn(
                        f"Could not add confidence intervals: {e}",
                        UserWarning
                    )

        # Save if requested
        if output_file:
            g.export(output_file)

        # Show if requested
        if show:
            try:
                import matplotlib.pyplot as plt
                plt.show()
            except Exception as e:
                logger.warning(f"Could not display plot: {e}")

        return g

    def plot_bestfit(self, object_id: Optional[str] = None,
                    output_file: Optional[str] = None, show: bool = True,
                    filter_file: Optional[str] = None, filter_selection_file: Optional[str] = None,
                    use_rest_frame: bool = True, flux_unit: str = 'fnu', use_log_scale: Optional[bool] = None,
                    model_names: Optional[List[str]] = None, show_emission_lines: bool = True,
                    emission_line_fontsize: int = 12, title_fontsize: int = 16, 
                    label_fontsize: int = 16, legend_fontsize: int = 14,
                    figsize: tuple = (12, 8), dpi: int = 300, 
                    focus_on_data_range: bool = True, **kwargs) -> Any:
        """
        Plot best-fit SED using the bayesed.plotting module.

        Parameters
        ----------
        object_id : str, optional
            Object ID to plot for. If None, uses current scope.
        output_file : str, optional
            Output file path for saving. If None, saves as {fits_file}.png
        show : bool, default True
            Whether to display the plot
        filter_file : str, optional
            Path to filter response file for overlay
        filter_selection_file : str, optional
            Path to filter selection file (filters_selected format)
        use_rest_frame : bool, default True
            Use rest-frame wavelengths. If False, uses observed-frame
        flux_unit : str, default 'fnu'
            Flux unit: 'fnu' (μJy), 'nufnu' (νFν in μJy*Hz), or 'flambda'
        use_log_scale : bool, optional
            Use logarithmic scale for axes. If None (default), auto-detects based on data range.
            Auto-detection uses log scale when either axis spans more than 1 order of magnitude
            (range ratio > 10). If negative values are present, defaults to linear scale.
            Set to True to force log scale, or False to force linear scale.
        model_names : list of str, optional
            Custom names for model components. If None, auto-generates from HDU names
        show_emission_lines : bool, default True
            Show emission line markers for spectroscopy
        emission_line_fontsize : int, default 12
            Font size for emission line labels. Larger values make labels more readable.
        title_fontsize : int, default 16
            Font size for the plot title
        label_fontsize : int, default 16
            Font size for axis labels (x and y axis)
        legend_fontsize : int, default 14
            Font size for legend text
        figsize : tuple, default (12, 8)
            Figure size (width, height) in inches
        dpi : int, default 300
            Resolution for saved figure
        focus_on_data_range : bool, default True
            If True, set x-axis limits to focus on the wavelength range where data exists
            (photometry and spectroscopy), ignoring the full model range. If False, use
            the full wavelength range from both models and data
        **kwargs
            Additional plotting arguments passed to matplotlib plotting functions

        Returns
        -------
        matplotlib.figure.Figure
            The matplotlib figure object

        Examples
        --------
        >>> # Basic usage
        >>> results.plot_bestfit()
        >>> 
        >>> # Customize plot
        >>> results.plot_bestfit(
        ...     use_rest_frame=True,
        ...     flux_unit='nufnu',
        ...     use_log_scale=True,
        ...     figsize=(14, 10)
        ... )
        >>> 
        >>> # With filter overlay
        >>> results.plot_bestfit(
        ...     filter_file='filters.txt',
        ...     filter_selection_file='filters_selected.txt'
        ... )
        """
        # Determine object to plot
        if object_id is None:
            if self.object_id is not None:
                object_id = self.object_id
            else:
                # For sample-level access, use first available object with warning
                objects = self.list_objects()
                if objects:
                    object_id = objects[0]
                    logger.warning(f"No object_id provided for plot_bestfit. Using first available object: {object_id}. More objects can be obtained with results.list_objects().")
                else:
                    raise ValueError("No objects available for plotting best-fit spectrum")

        # Find bestfit FITS files for this object matching the current configuration
        fits_files = self._find_config_files(object_id, "*_bestfit.fits")

        # Ensure we have exactly one bestfit file (log selection for main user operation)
        fits_file = self._select_unique_file(fits_files, "bestfit FITS", object_id, log_selection=True)

        # Import plotting function
        from ..plotting import plot_bestfit

        # Create plot with file path, passing all the plotting options
        fig = plot_bestfit(
            fits_file, 
            output_file=output_file, 
            show=show,
            filter_file=filter_file,
            filter_selection_file=filter_selection_file,
            use_rest_frame=use_rest_frame,
            flux_unit=flux_unit,
            use_log_scale=use_log_scale,
            model_names=model_names,
            show_emission_lines=show_emission_lines,
            emission_line_fontsize=emission_line_fontsize,
            title_fontsize=title_fontsize,
            label_fontsize=label_fontsize,
            legend_fontsize=legend_fontsize,
            figsize=figsize,
            dpi=dpi,
            focus_on_data_range=focus_on_data_range,
            **kwargs
        )

        return fig

    def plot_posterior_free(self, object_id: Optional[str] = None,
                           output_file: Optional[str] = None,
                           show: bool = True,
                           show_median: bool = True, show_confidence_intervals: bool = True,
                           confidence_level: float = 0.68, **kwargs) -> Any:
        """
        Plot posterior distributions (corner plot) for free parameters.

        Parameters
        ----------
        object_id : str, optional
            Object ID to plot. If None and in sample-level mode, uses first object.
        output_file : str, optional
            Output file path for saving plot
        show : bool, default True
            Whether to display the plot
        show_median : bool, default True
            Whether to show median markers on plots
        show_confidence_intervals : bool, default True
            Whether to show confidence intervals on 1D marginal plots
        confidence_level : float, default 0.68
            Confidence level for intervals (0.68 for 1-sigma, 0.95 for 2-sigma)
        **kwargs
            Additional plotting parameters

        Returns
        -------
        Any
            Plot object
        """
        try:
            # Handle object_id selection
            if object_id is None:
                if self.object_id is not None:
                    object_id = self.object_id
                else:
                    # For sample-level access, use first available object with warning
                    objects = self.list_objects()
                    if objects:
                        object_id = objects[0]
                        logger.warning(f"No object_id provided for plot_posterior_free. Using first available object: {object_id}. More objects can be obtained with results.list_objects().")
                    else:
                        raise ValueError("No objects available for plotting")

            free_params = self.get_free_parameters()

            # Filter parameters to only include those available and varying in GetDist samples
            if object_id:
                try:
                    samples = self.get_getdist_samples(object_id=object_id)
                    getdist_names = [p.name for p in samples.paramNames.names]

                    # Filter out missing and fixed parameters
                    varying_params = []
                    for param_name in free_params:
                        if param_name not in getdist_names:
                            logger.debug(f"Excluding missing parameter: {param_name}")
                            continue

                        # Check if parameter actually varies
                        param_index = getdist_names.index(param_name)
                        param_samples = samples.samples[:, param_index]

                        # Use numpy to check variance more robustly
                        import numpy as np
                        if np.var(param_samples) > 1e-10:  # Small threshold for numerical precision
                            varying_params.append(param_name)
                        else:
                            logger.debug(f"Excluding fixed parameter: {param_name} (variance: {np.var(param_samples)})")

                    free_params = varying_params
                    logger.debug(f"Filtered to {len(free_params)} varying free parameters available in GetDist samples")
                except Exception as e:
                    logger.warning(f"Could not filter parameters: {e}")

            return self.plot_posterior(params=free_params, object_id=object_id,
                                     output_file=output_file, show=show,
                                     show_median=show_median, 
                                     show_confidence_intervals=show_confidence_intervals,
                                     confidence_level=confidence_level, **kwargs)
        except Exception as e:
            if "GetDist" in str(e) or "No chains found" in str(e):
                logger.warning(f"GetDist plotting not available: {e}")
                logger.info("Consider using individual parameter plotting or ensure GetDist-compatible sample files exist")
                # Return a placeholder or simple message
                print(f"Free parameters ({len(self.get_free_parameters())}): {self.get_free_parameters()}")
                return None
            else:
                raise

    def plot_posterior_derived(self, object_id: Optional[str] = None,
                              max_params: int = 10,
                              output_file: Optional[str] = None,
                              show: bool = True,
                              show_median: bool = True, show_confidence_intervals: bool = True,
                              confidence_level: float = 0.68, **kwargs) -> Any:
        """
        Plot posterior distributions (corner plot) for derived parameters.

        Parameters
        ----------
        object_id : str, optional
            Object ID to plot. If None and in sample-level mode, uses first object.
        max_params : int, default 10
            Maximum number of parameters to plot
        output_file : str, optional
            Output file path for saving plot
        show : bool, default True
            Whether to display the plot
        show_median : bool, default True
            Whether to show median markers on plots
        show_confidence_intervals : bool, default True
            Whether to show confidence intervals on 1D marginal plots
        confidence_level : float, default 0.68
            Confidence level for intervals (0.68 for 1-sigma, 0.95 for 2-sigma)
        **kwargs
            Additional plotting parameters

        Returns
        -------
        Any
            Plot object
        """
        try:
            # Handle object_id selection
            if object_id is None:
                if self.object_id is not None:
                    object_id = self.object_id
                else:
                    # For sample-level access, use first available object with warning
                    objects = self.list_objects()
                    if objects:
                        object_id = objects[0]
                        logger.warning(f"No object_id provided for plot_posterior_derived. Using first available object: {object_id}. More objects can be obtained with results.list_objects().")
                    else:
                        raise ValueError("No objects available for plotting")

            derived_params = self.get_derived_parameters()

            # Filter parameters to only include those available in GetDist samples
            if object_id:
                try:
                    samples = self.get_getdist_samples(object_id=object_id)
                    getdist_names = [p.name for p in samples.paramNames.names]
                    derived_params = [p for p in derived_params if p in getdist_names]
                    logger.debug(f"Filtered to {len(derived_params)} derived parameters available in GetDist samples")
                except Exception as e:
                    logger.warning(f"Could not filter parameters: {e}")

            # Limit to max_params if specified
            if max_params and len(derived_params) > max_params:
                derived_params = derived_params[:max_params]

            return self.plot_posterior(params=derived_params, object_id=object_id,
                                     output_file=output_file, show=show,
                                     show_median=show_median,
                                     show_confidence_intervals=show_confidence_intervals,
                                     confidence_level=confidence_level, **kwargs)
        except Exception as e:
            if "GetDist" in str(e) or "No chains found" in str(e):
                logger.warning(f"GetDist plotting not available: {e}")
                logger.info("Consider using individual parameter plotting or ensure GetDist-compatible sample files exist")
                # Return a placeholder or simple message
                limited_params = self.get_derived_parameters()[:max_params] if max_params else self.get_derived_parameters()
                print(f"Derived parameters ({len(limited_params)}): {limited_params}")
                return None
            else:
                raise

    def plot_parameter_scatter(self, x_parameter: Union[str, np.ndarray, List[float]], 
                              y_parameter: Union[str, List[str]],
                              x_err: Optional[Union[str, np.ndarray, List[float]]] = None,
                              y_err: Optional[Union[str, List[str]]] = None,
                              color_parameter: Optional[str] = None,
                              object_ids: Optional[Union[str, List[str]]] = None,
                              show_diagonal: bool = True,
                              show_colorbar: bool = True,
                              show_stats: bool = True,
                              figsize: tuple = (8, 6),
                              xlim: Optional[tuple] = None,
                              ylim: Optional[tuple] = None,
                              output_file: Optional[str] = None,
                              show: bool = True,
                              xlabel: Optional[str] = None,
                              ylabel: Optional[str] = None,
                              title: Optional[str] = None,
                              legend_labels: Optional[List[str]] = None,
                              legend_loc: str = 'best',
                              **kwargs) -> Any:
        """
        Create a scatter plot comparing one x-parameter with one or multiple y-parameters with optional error bars.
        
        This method creates scatter plots similar to:
        plt.scatter(results.parameters['log(Mstar)[0,1]_{True}'], 
                   results.parameters['log(Mstar)[0,0]_{median}'])
        
        When y_parameter is a list, all y-parameters will be plotted against the same x-parameter
        in the same figure with different colors and point styles for easy comparison.
        
        Parameters
        ----------
        x_parameter : str or array-like
            X-axis values. Can be either:
            - str: Full column name from HDF5 file (e.g., 'log(Mstar)[0,1]_{True}')
            - array-like: External numpy array or list of values
        y_parameter : str or List[str]
            Full column name(s) for y-axis values (e.g., 'log(Mstar)[0,0]_{median}' or 
            ['log(Mstar)[0,0]_{median}', 'log(age)[0,0]_{median}']). When a list is provided,
            all y-parameters will be plotted against the x-parameter with different colors and markers.
        x_err : str or array-like, optional
            X-axis error bars. Can be either:
            - str: Column name or percentile specification from HDF5 file:
              * 'sigma' for sigma column
              * 'percentile' or 'percentile_68' for 68% confidence interval
              * 'percentile_95' for 95% confidence interval  
              * 'percentile_90' for 90% confidence interval
              * Direct column name (e.g., 'log(Mstar)[0,0]_{sigma}')
            - array-like: External error values. Can be:
              * 1D array: Symmetric errors (same length as x_parameter)
              * 2D array: Asymmetric errors, shape (2, N) where first row is lower errors
                and second row is upper errors (for confidence intervals)
            Note: If x_parameter is external array, x_err must also be external array or None
        y_err : str or List[str], optional
            Column name(s) for y-axis error bars. Same options as x_err. If y_parameter is a list,
            y_err can be a single string (applied to all y-parameters) or a list of strings
            (one for each y-parameter, or None for no error bars on specific parameters).
        color_parameter : str, optional
            Full column name for color coding points. If None and y_parameter is a list, 
            uses different colors for each y-parameter. If specified with multiple y-parameters,
            all points will be colored by this parameter.
        object_ids : str or List[str], optional
            Specific object ID(s) to include. If None, uses all available objects.
        show_diagonal : bool, default True
            Whether to show the diagonal line (y=x)
        show_colorbar : bool, default True
            Whether to show colorbar (only used if color_parameter is specified)
        show_stats : bool, default True
            Whether to show correlation statistics on the plot. Includes Pearson r,
            Spearman ρ, and error-weighted correlation (if errors are available).
            For multiple y-parameters, shows stats for each y vs x relationship.
        figsize : tuple, default (8, 6)
            Figure size (width, height) in inches
        xlim : tuple, optional
            X-axis limits as (xmin, xmax). If None, uses automatic scaling.
        ylim : tuple, optional
            Y-axis limits as (ymin, ymax). If None, uses automatic scaling.
        output_file : str, optional
            Output file path for saving plot
        show : bool, default True
            Whether to display the plot
        xlabel : str, optional
            Custom x-axis label. If None, uses x_parameter name.
        ylabel : str, optional
            Custom y-axis label. If None, uses y_parameter name(s).
        title : str, optional
            Custom plot title. If None, auto-generates from parameter names.
        legend_labels : List[str], optional
            Custom labels for the legend. If provided, must have the same length as y_parameter list.
            If None, uses automatically extracted labels from parameter names.
        legend_loc : str, default 'best'
            Location for the legend. Options: 'best', 'upper right', 'upper left', 'lower left',
            'lower right', 'right', 'center left', 'center right', 'lower center', 'upper center', 'center'.
        **kwargs
            Additional arguments passed to plt.errorbar() or plt.scatter()
            
        Returns
        -------
        matplotlib.figure.Figure
            The matplotlib figure object
            
        Examples
        --------
        >>> # Basic scatter plot (single y-parameter)
        >>> results.plot_parameter_scatter('log(Mstar)[0,1]_{True}', 
        ...                               'log(Mstar)[0,0]_{median}')
        >>> 
        >>> # Multiple y-parameters comparison
        >>> results.plot_parameter_scatter('log(Mstar)[0,1]_{True}', 
        ...                               ['log(Mstar)[0,0]_{median}', 'log(age)[0,0]_{median}'])
        >>> 
        >>> # With error bars using sigma (single y-parameter)
        >>> results.plot_parameter_scatter('log(Mstar)[0,1]_{True}', 
        ...                               'log(Mstar)[0,0]_{median}',
        ...                               y_err='log(Mstar)[0,0]_{sigma}')
        >>> 
        >>> # Multiple y-parameters with different error bars
        >>> results.plot_parameter_scatter('log(Mstar)[0,0]_{median}', 
        ...                               ['log(age)[0,0]_{median}', 'SFR_{median}'],
        ...                               y_err=['log(age)[0,0]_{sigma}', 'SFR_{sigma}'])
        >>> 
        >>> # Multiple y-parameters with same error type
        >>> results.plot_parameter_scatter('log(Mstar)[0,0]_{median}', 
        ...                               ['log(age)[0,0]_{median}', 'SFR_{median}'],
        ...                               y_err='percentile')
        >>> 
        >>> # With error bars using percentiles (68% confidence interval)
        >>> results.plot_parameter_scatter('log(Mstar)[0,0]_{median}', 
        ...                               'log(age)[0,0]_{median}',
        ...                               x_err='percentile', y_err='percentile')
        >>> 
        >>> # With 95% confidence interval error bars
        >>> results.plot_parameter_scatter('log(Mstar)[0,0]_{median}', 
        ...                               'log(age)[0,0]_{median}',
        ...                               x_err='percentile_95', y_err='percentile_95')
        >>> 
        >>> # Mixed error types: 90% percentiles for x, sigma for y
        >>> results.plot_parameter_scatter('log(Mstar)[0,0]_{median}', 
        ...                               'SFR_{median}',
        ...                               x_err='percentile_90', y_err='sigma')
        >>> 
        >>> # Custom axis ranges with multiple y-parameters
        >>> results.plot_parameter_scatter('log(Mstar)[0,1]_{True}', 
        ...                               ['log(Mstar)[0,0]_{median}', 'log(age)[0,0]_{median}'],
        ...                               xlim=(9, 12), ylim=(8, 12))
        >>> 
        >>> # Turn off statistics display
        >>> results.plot_parameter_scatter('log(Mstar)[0,1]_{True}', 
        ...                               ['log(Mstar)[0,0]_{median}', 'log(age)[0,0]_{median}'],
        ...                               show_stats=False)
        >>> 
        >>> # With color coding (overrides automatic coloring for multiple y-parameters)
        >>> results.plot_parameter_scatter('log(Mstar)[0,1]_{True}', 
        ...                               ['log(Mstar)[0,0]_{median}', 'log(age)[0,0]_{median}'],
        ...                               color_parameter='log(age)[0,0]_{median}')
        >>> 
        >>> # With custom legend labels
        >>> results.plot_parameter_scatter('log(Mstar)[0,1]_{True}', 
        ...                               ['log(Mstar)[0,0]_{median}', 'log(age)[0,0]_{median}'],
        ...                               legend_labels=['Stellar Mass', 'Age'])
        >>> 
        >>> # Custom legend position
        >>> results.plot_parameter_scatter('log(Mstar)[0,1]_{True}', 
        ...                               ['log(Mstar)[0,0]_{median}', 'log(age)[0,0]_{median}'],
        ...                               legend_loc='upper left')
        """
        try:
            import matplotlib.pyplot as plt
            import numpy as np
        except ImportError:
            raise ImportError("matplotlib and numpy are required for plotting. Install with: pip install matplotlib numpy")
        
        # Convert y_parameter to list for consistent handling
        if isinstance(y_parameter, str):
            y_parameters = [y_parameter]
            single_y = True
        else:
            y_parameters = y_parameter
            single_y = False
        
        # Handle y_err parameter - convert to list matching y_parameters
        if y_err is None:
            y_err_list = [None] * len(y_parameters)
        elif isinstance(y_err, str):
            # Single error specification - apply to all y-parameters
            y_err_list = [y_err] * len(y_parameters)
        elif isinstance(y_err, list):
            if len(y_err) != len(y_parameters):
                raise ValueError(f"Length of y_err list ({len(y_err)}) must match length of y_parameter list ({len(y_parameters)})")
            y_err_list = y_err
        else:
            raise ValueError("y_err must be None, str, or list of str")
        
        # Handle legend_labels parameter
        if legend_labels is not None:
            if not isinstance(legend_labels, list):
                raise ValueError("legend_labels must be a list of strings")
            if len(legend_labels) != len(y_parameters):
                raise ValueError(f"Length of legend_labels ({len(legend_labels)}) must match length of y_parameter list ({len(y_parameters)})")
            custom_legend_labels = legend_labels
        else:
            custom_legend_labels = None
        
        # Get the HDF5 table with appropriate filtering
        hdf5_table = self.load_hdf5_results(filter_snr=False, min_snr=0.0)
        
        # Apply object filtering if specified
        if object_ids is not None:
            if isinstance(object_ids, str):
                object_ids = [object_ids]
            object_mask = [str(obj_id) in [str(x) for x in object_ids] for obj_id in hdf5_table['ID']]
            if any(object_mask):
                hdf5_table = hdf5_table[object_mask]
            else:
                available_objects = [str(obj_id) for obj_id in hdf5_table['ID']]
                raise ValueError(f"None of the specified objects {object_ids} found in data. "
                               f"Available objects: {available_objects}")
        
        # Check if columns exist
        available_cols = hdf5_table.colnames
        
        # Handle x_parameter - can be string (column name) or array-like (external data)
        if isinstance(x_parameter, str):
            # Current behavior - validate column exists in HDF5
            if x_parameter not in available_cols:
                # Try to find similar columns
                similar_cols = [col for col in available_cols if x_parameter.split('_')[0] in col or x_parameter.split('[')[0] in col]
                if similar_cols:
                    raise ValueError(f"Column '{x_parameter}' not found. Similar columns: {similar_cols[:10]}")
                else:
                    raise ValueError(f"Column '{x_parameter}' not found. Available columns: {available_cols[:10]}...")
            
            # Extract x data from HDF5
            x_values = np.array(hdf5_table[x_parameter])
            x_label_default = x_parameter
        else:
            # New behavior - use external array
            x_values = np.asarray(x_parameter)
            if len(x_values) != len(hdf5_table):
                raise ValueError(f"External x_parameter array length ({len(x_values)}) "
                               f"doesn't match dataset length ({len(hdf5_table)})")
            x_label_default = "External X Parameter"
        
        # Check all y_parameters exist
        for y_param in y_parameters:
            if y_param not in available_cols:
                # Try to find similar columns
                similar_cols = [col for col in available_cols if y_param.split('_')[0] in col or y_param.split('[')[0] in col]
                if similar_cols:
                    raise ValueError(f"Column '{y_param}' not found. Similar columns: {similar_cols[:10]}")
                else:
                    raise ValueError(f"Column '{y_param}' not found. Available columns: {available_cols[:10]}...")
        
        # Handle error bars
        def get_error_values(param, err_spec, param_name):
            """Get error values for a parameter."""
            if err_spec is None:
                return None
            
            if err_spec.startswith('percentile'):
                # Parse percentile specification
                if err_spec == 'percentile':
                    # Default to 68% confidence interval (16th/84th percentiles)
                    lower_p, upper_p = 0.16, 0.84
                elif '_' in err_spec:
                    # Custom percentile like 'percentile_95' or 'percentile_90'
                    try:
                        confidence_level = float(err_spec.split('_')[1])
                        if confidence_level > 1:
                            confidence_level = confidence_level / 100.0  # Convert percentage to fraction
                        
                        # Calculate symmetric percentiles around median
                        lower_p = (1 - confidence_level) / 2
                        upper_p = 1 - lower_p
                        
                        logger.info(f"Using {confidence_level*100:.1f}% confidence interval: {lower_p:.3f}/{upper_p:.3f} percentiles")
                    except (ValueError, IndexError):
                        logger.warning(f"Invalid percentile specification '{err_spec}'. Using default 68% interval.")
                        lower_p, upper_p = 0.16, 0.84
                else:
                    logger.warning(f"Invalid percentile specification '{err_spec}'. Using default 68% interval.")
                    lower_p, upper_p = 0.16, 0.84
                
                # Auto-calculate from specified percentiles
                base_param = param.replace('_{median}', '').replace('_{mean}', '').replace('_{MAP}', '').replace('_{MAL}', '')
                
                # Format percentiles as strings (handle both decimal and percentage formats)
                if lower_p < 1:
                    lower_str = f'{lower_p:.3f}'.rstrip('0').rstrip('.')
                    upper_str = f'{upper_p:.3f}'.rstrip('0').rstrip('.')
                else:
                    lower_str = f'{lower_p:.0f}'
                    upper_str = f'{upper_p:.0f}'
                
                # Try different column name formats
                possible_formats = [
                    (f'_{{{lower_str}}}', f'_{{{upper_str}}}'),  # _{0.16}, _{0.84}
                    (f'_{lower_str}', f'_{upper_str}'),          # _0.16, _0.84
                    (f'_p{int(lower_p*100)}', f'_p{int(upper_p*100)}'),  # _p16, _p84
                ]
                
                lower_col = upper_col = None
                for lower_fmt, upper_fmt in possible_formats:
                    lower_candidate = base_param + lower_fmt
                    upper_candidate = base_param + upper_fmt
                    
                    if lower_candidate in available_cols and upper_candidate in available_cols:
                        lower_col = lower_candidate
                        upper_col = upper_candidate
                        break
                
                if lower_col and upper_col:
                    lower_values = np.array(hdf5_table[lower_col])
                    upper_values = np.array(hdf5_table[upper_col])
                    param_values = np.array(hdf5_table[param])
                    
                    # Calculate asymmetric errors
                    lower_err = param_values - lower_values
                    upper_err = upper_values - param_values
                    
                    logger.info(f"Using percentile columns: {lower_col}, {upper_col}")
                    # Return as [lower_errors, upper_errors] for asymmetric error bars
                    return np.array([lower_err, upper_err])
                else:
                    # Show what columns were tried
                    tried_cols = []
                    for lower_fmt, upper_fmt in possible_formats:
                        tried_cols.extend([base_param + lower_fmt, base_param + upper_fmt])
                    
                    logger.warning(f"Percentile columns not found for {param_name}. Tried: {tried_cols}")
                    
                    # Show available similar columns
                    similar_cols = [col for col in available_cols if base_param in col and any(p in col for p in ['0.', '_p', '_{0.', '_{1.'])]
                    if similar_cols:
                        logger.info(f"Available percentile-like columns: {similar_cols}")
                    
                    return None
            
            elif err_spec == 'sigma':
                # Auto-find sigma column
                base_param = param.replace('_{median}', '').replace('_{mean}', '').replace('_{MAP}', '').replace('_{MAL}', '')
                sigma_col = base_param + '_{sigma}'
                
                if sigma_col in available_cols:
                    return np.array(hdf5_table[sigma_col])
                else:
                    logger.warning(f"Sigma column not found: {sigma_col}")
                    return None
            
            else:
                # Direct column name
                if err_spec in available_cols:
                    return np.array(hdf5_table[err_spec])
                else:
                    logger.warning(f"Error column '{err_spec}' not found for {param_name}")
                    return None
        
        # Handle x_err - can be string (column name) or array-like (external data)
        x_err_values = None
        if x_err is not None:
            if isinstance(x_err, str):
                # String x_err - must have string x_parameter for HDF5 column lookup
                if isinstance(x_parameter, str):
                    x_err_values = get_error_values(x_parameter, x_err, 'x_parameter')
                else:
                    raise ValueError("Cannot use string x_err with external x_parameter array. "
                                   "Provide x_err as an array or set to None.")
            else:
                # External error array
                x_err_array = np.asarray(x_err)
                
                # Handle both symmetric (1D) and asymmetric (2D) error arrays
                if x_err_array.ndim == 1:
                    # Symmetric errors
                    if len(x_err_array) != len(x_values):
                        raise ValueError(f"External x_err array length ({len(x_err_array)}) "
                                       f"doesn't match x_parameter length ({len(x_values)})")
                    x_err_values = x_err_array
                elif x_err_array.ndim == 2:
                    # Asymmetric errors (confidence intervals)
                    if x_err_array.shape[0] != 2:
                        raise ValueError(f"2D x_err array must have shape (2, N) for asymmetric errors, "
                                       f"got shape {x_err_array.shape}")
                    if x_err_array.shape[1] != len(x_values):
                        raise ValueError(f"2D x_err array second dimension ({x_err_array.shape[1]}) "
                                       f"doesn't match x_parameter length ({len(x_values)})")
                    x_err_values = x_err_array
                else:
                    raise ValueError(f"External x_err array must be 1D (symmetric) or 2D (asymmetric), "
                                   f"got {x_err_array.ndim}D array")
        
        # Handle color parameter if specified
        color_values = None
        if color_parameter is not None:
            if color_parameter not in available_cols:
                logger.warning(f"Color parameter '{color_parameter}' not found. Available columns: "
                             f"{[col for col in available_cols if color_parameter.split('_')[0] in col or color_parameter.split('[')[0] in col][:5]}")
                logger.warning("Proceeding without color coding.")
            else:
                color_values = np.array(hdf5_table[color_parameter])
        
        # Create the plot
        fig, ax = plt.subplots(figsize=figsize)
        
        # Define colors and markers for multiple y-parameters
        colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
        markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
        
        # Extract common and different parts for ylabel and legend
        def extract_common_and_different_parts(param_names):
            """
            Extract common and different parts from parameter names for ylabel and legend.
            
            This function analyzes parameter names to find meaningful common parts and 
            extract the differentiating parts for cleaner labels.
            
            Examples:
            - ['log(age/yr)[0,0]_{mean}', 'log(age/yr)[0,0]_{median}', 'log(age/yr)[0,0]_{MAP}'] 
              -> common: 'log(age/yr)[0,0]', different: ['mean', 'median', 'MAP']
            - ['log(Mstar)[0,0]_{median}', 'log(age)[0,0]_{median}']
              -> common: '[0,0]_{median}', different: ['log(Mstar)', 'log(age)']
            - ['SFR_{median}', 'log(Z)[0,0]_{median}']
              -> common: '_{median}', different: ['SFR', 'log(Z)[0,0]']
            """
            if len(param_names) == 1:
                return param_names[0], [param_names[0]]
            
            # Strategy: Look for patterns in parameter names
            # Common patterns: parameter_{statistic}, parameter[index]_{statistic}
            
            # First, try to identify if all parameters have the same base with different statistics
            # Look for patterns like _{mean}, _{median}, _{MAP}, etc.
            
            # Check if all parameters follow the pattern: base_{statistic}
            base_parts = []
            stat_parts = []
            all_have_underscore_stat = True
            
            for name in param_names:
                # Look for the last underscore followed by a statistic name
                if '_{' in name:
                    # Handle both _{stat} and _{stat} patterns
                    last_underscore_brace = name.rfind('_{')
                    if last_underscore_brace != -1:
                        base_part = name[:last_underscore_brace]
                        stat_part = name[last_underscore_brace+2:]  # Skip _{
                        if stat_part.endswith('}'):
                            stat_part = stat_part[:-1]  # Remove trailing }
                        base_parts.append(base_part)
                        stat_parts.append(stat_part)
                    else:
                        all_have_underscore_stat = False
                        break
                elif '_' in name:
                    # Simple pattern: parameter_statistic
                    parts = name.rsplit('_', 1)
                    if len(parts) == 2:
                        base_part = parts[0]
                        stat_part = parts[1]
                        base_parts.append(base_part)
                        stat_parts.append(stat_part)
                    else:
                        all_have_underscore_stat = False
                        break
                else:
                    all_have_underscore_stat = False
                    break
            
            # Case 1: All parameters have the pattern base_{statistic}
            if all_have_underscore_stat and len(base_parts) == len(param_names):
                unique_bases = set(base_parts)
                
                if len(unique_bases) == 1:
                    # All base parts are identical - different statistics of same parameter
                    # This is the case we want: log(age/yr)[0,1]_{mean} vs log(age/yr)[0,1]_{median}
                    common_part = base_parts[0]
                    different_parts = stat_parts
                    return common_part, different_parts
                else:
                    # Different base parts, but check if they have same statistic
                    unique_stats = set(stat_parts)
                    if len(unique_stats) == 1:
                        # Same statistic, different parameters
                        common_part = '_{' + stat_parts[0] + '}'
                        different_parts = base_parts
                        return common_part, different_parts
            
            # Case 2: Fallback to character-by-character comparison
            # Find common prefix
            common_prefix = ""
            min_len = min(len(name) for name in param_names)
            for i in range(min_len):
                if all(name[i] == param_names[0][i] for name in param_names):
                    common_prefix += param_names[0][i]
                else:
                    break
            
            # Find common suffix
            common_suffix = ""
            for i in range(1, min_len + 1):
                if all(name[-i] == param_names[0][-i] for name in param_names):
                    common_suffix = param_names[0][-i] + common_suffix
                else:
                    break
            
            # Extract different parts (middle parts between common prefix and suffix)
            different_parts = []
            for name in param_names:
                start_idx = len(common_prefix)
                end_idx = len(name) - len(common_suffix) if common_suffix else len(name)
                different_part = name[start_idx:end_idx]
                different_parts.append(different_part)
            
            # Construct common part
            common_part = common_prefix + common_suffix
            
            # Clean up and validate results
            if len(common_part.strip()) < 3:  # Too short to be meaningful
                # Fallback: use simplified parameter names
                different_parts = []
                for name in param_names:
                    # Extract base parameter name (before first underscore or bracket)
                    base_name = name.split('_')[0].split('[')[0]
                    different_parts.append(base_name)
                common_part = "Multiple Parameters"
            else:
                # Clean up different parts - remove common separators
                cleaned_different = []
                for part in different_parts:
                    # Remove leading/trailing underscores, brackets, etc.
                    cleaned = part.strip('_{}[]')
                    if not cleaned:
                        # If nothing left after cleaning, use a fallback
                        idx = different_parts.index(part)
                        original_name = param_names[idx]
                        cleaned = original_name.split('_')[0].split('[')[0]
                    cleaned_different.append(cleaned)
                different_parts = cleaned_different
            
            return common_part, different_parts
        
        # Get common and different parts for ylabel and legend (only if custom labels not provided)
        if not single_y and custom_legend_labels is None:
            common_ylabel, auto_legend_labels = extract_common_and_different_parts(y_parameters)
        else:
            common_ylabel = y_parameters[0] if single_y else "Multiple Parameters"
            auto_legend_labels = custom_legend_labels if custom_legend_labels else [y_parameters[0]]
        
        # Use custom legend labels if provided, otherwise use auto-extracted labels
        final_legend_labels = custom_legend_labels if custom_legend_labels else auto_legend_labels
        
        # Store data for statistics calculation
        all_stats_data = []
        
        # Plot each y-parameter
        for i, (y_param, y_err_spec) in enumerate(zip(y_parameters, y_err_list)):
            # Extract y data and errors
            y_values = np.array(hdf5_table[y_param])
            y_err_values = get_error_values(y_param, y_err_spec, f'y_parameter[{i}]')
            
            # Prepare plotting arguments
            plot_kwargs = {'alpha': 0.7}
            
            # Set color and marker
            if color_parameter is None and not single_y:
                # Use different colors/markers for each y-parameter
                plot_kwargs['color'] = colors[i % len(colors)]
                plot_kwargs['marker'] = markers[i % len(markers)]
                plot_kwargs['label'] = final_legend_labels[i]  # Use final legend labels
            elif color_parameter is not None:
                # Use color parameter for coloring
                plot_kwargs['c'] = color_values
                plot_kwargs['marker'] = markers[i % len(markers)] if not single_y else 'o'
                if not single_y:
                    plot_kwargs['label'] = final_legend_labels[i]  # Use final legend labels
            else:
                # Single y-parameter, default styling
                plot_kwargs['marker'] = 'o'
            
            if x_err_values is not None or y_err_values is not None:
                # Use errorbar plot
                plot_kwargs.update({'markersize': 5, 'capsize': 3, 'linestyle': 'none'})
            else:
                # Use scatter plot
                plot_kwargs.update({'s': 50})
            
            # Override with user kwargs (but preserve color/marker for multi-y)
            user_kwargs = kwargs.copy()
            if not single_y and color_parameter is None:
                # Preserve automatic color/marker for multi-y plots
                user_kwargs.pop('color', None)
                user_kwargs.pop('c', None)
                user_kwargs.pop('marker', None)
            plot_kwargs.update(user_kwargs)
            
            # Create the plot
            if x_err_values is not None or y_err_values is not None:
                # Use errorbar
                if color_parameter is not None:
                    # For colored error bars, plot scatter first then error bars
                    scatter = ax.scatter(x_values, y_values, c=color_values, 
                                       s=plot_kwargs.get('s', 50), 
                                       marker=plot_kwargs.get('marker', 'o'),
                                       alpha=plot_kwargs.get('alpha', 0.7),
                                       label=plot_kwargs.get('label'))
                    
                    # Add error bars without markers (use fixed color for error bars)
                    ax.errorbar(x_values, y_values, xerr=x_err_values, yerr=y_err_values, 
                               fmt='none', alpha=0.5, capsize=plot_kwargs.get('capsize', 3),
                               color='gray')
                else:
                    # No color parameter - use regular errorbar
                    # Remove 'c' from plot_kwargs to avoid conflicts
                    errorbar_kwargs = {k: v for k, v in plot_kwargs.items() 
                                     if k not in ['marker', 's', 'c']}
                    ax.errorbar(x_values, y_values, xerr=x_err_values, yerr=y_err_values, 
                               fmt=plot_kwargs.get('marker', 'o'), **errorbar_kwargs)
            else:
                # Use scatter plot
                if color_parameter is not None:
                    scatter = ax.scatter(x_values, y_values, **plot_kwargs)
                else:
                    ax.scatter(x_values, y_values, **plot_kwargs)
            
            # Store data for statistics
            all_stats_data.append({
                'x_values': x_values,
                'y_values': y_values,
                'x_err_values': x_err_values,
                'y_err_values': y_err_values,
                'y_param': y_param
            })
        
        # Add colorbar for any plot with color parameter (single or multi-y)
        if color_parameter is not None and show_colorbar:
            # Use the scatter plot for colorbar (created in either error bar or scatter plot sections)
            if 'scatter' in locals():
                cbar = plt.colorbar(scatter, ax=ax)
                cbar.set_label(color_parameter)
        
        # Add legend for multiple y-parameters (inside the figure)
        if not single_y and color_parameter is None:
            ax.legend(loc=legend_loc, frameon=True, fancybox=True, shadow=True, framealpha=0.9)
        
        # Set axis limits if specified
        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)
        
        # Calculate and display correlation statistics if requested
        if show_stats:
            try:
                from scipy import stats
                
                stats_text_lines = []
                
                for i, stats_data in enumerate(all_stats_data):
                    x_vals = stats_data['x_values']
                    y_vals = stats_data['y_values']
                    x_err_vals = stats_data['x_err_values']
                    y_err_vals = stats_data['y_err_values']
                    y_param = stats_data['y_param']
                    
                    # Remove NaN values for correlation calculations
                    valid_mask = ~(np.isnan(x_vals) | np.isnan(y_vals))
                    x_clean = x_vals[valid_mask]
                    y_clean = y_vals[valid_mask]
                    
                    if len(x_clean) > 2:  # Need at least 3 points for meaningful correlation
                        # Pearson correlation
                        pearson_r, pearson_p = stats.pearsonr(x_clean, y_clean)
                        
                        # Spearman correlation (rank-based, more robust)
                        spearman_rho, spearman_p = stats.spearmanr(x_clean, y_clean)
                        
                        # Prepare statistics text for this y-parameter
                        if not single_y:
                            # Use the final legend label for cleaner display
                            stats_text_lines.append(f'{final_legend_labels[i]}:')
                        
                        stats_text_lines.append(f'N = {len(x_clean)}')
                        stats_text_lines.append(f'r = {pearson_r:.3f}')
                        if pearson_p < 0.001:
                            stats_text_lines[-1] += ' (p<0.001)'
                        elif pearson_p < 0.01:
                            stats_text_lines[-1] += ' (p<0.01)'
                        elif pearson_p < 0.05:
                            stats_text_lines[-1] += ' (p<0.05)'
                        
                        stats_text_lines.append(f'ρ = {spearman_rho:.3f}')
                        if spearman_p < 0.001:
                            stats_text_lines[-1] += ' (p<0.001)'
                        elif spearman_p < 0.01:
                            stats_text_lines[-1] += ' (p<0.01)'
                        elif spearman_p < 0.05:
                            stats_text_lines[-1] += ' (p<0.05)'
                        
                        # Error-weighted correlation if errors are available
                        if x_err_vals is not None and y_err_vals is not None:
                            try:
                                # Calculate error-weighted correlation
                                x_err_clean = x_err_vals[valid_mask] if x_err_vals.ndim == 1 else np.mean(x_err_vals[:, valid_mask], axis=0)
                                y_err_clean = y_err_vals[valid_mask] if y_err_vals.ndim == 1 else np.mean(y_err_vals[:, valid_mask], axis=0)
                                
                                # Avoid division by zero
                                x_err_clean = np.maximum(x_err_clean, 1e-10)
                                y_err_clean = np.maximum(y_err_clean, 1e-10)
                                
                                # Weights inversely proportional to combined error
                                weights = 1.0 / (x_err_clean**2 + y_err_clean**2)
                                weights = weights / np.sum(weights)  # Normalize weights
                                
                                # Weighted means
                                x_weighted_mean = np.sum(weights * x_clean)
                                y_weighted_mean = np.sum(weights * y_clean)
                                
                                # Weighted correlation coefficient
                                numerator = np.sum(weights * (x_clean - x_weighted_mean) * (y_clean - y_weighted_mean))
                                x_var = np.sum(weights * (x_clean - x_weighted_mean)**2)
                                y_var = np.sum(weights * (y_clean - y_weighted_mean)**2)
                                
                                if x_var > 0 and y_var > 0:
                                    weighted_r = numerator / np.sqrt(x_var * y_var)
                                    stats_text_lines.append(f'r_w = {weighted_r:.3f}')
                                
                            except Exception as e:
                                logger.debug(f"Could not calculate weighted correlation for {y_param}: {e}")
                        
                        # Add separator for multiple y-parameters
                        if not single_y and i < len(all_stats_data) - 1:
                            stats_text_lines.append('')
                    
                    else:
                        logger.warning(f"Not enough valid data points for correlation statistics for {y_param}")
                
                # Display statistics if we have any
                if stats_text_lines:
                    stats_text = '\n'.join(stats_text_lines)
                    
                    # Position the text box in the upper left corner
                    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, 
                           verticalalignment='top', horizontalalignment='left',
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                           fontsize=11 if not single_y else 12, family='monospace')
                    
            except ImportError:
                logger.warning("scipy not available for correlation statistics")
            except Exception as e:
                logger.warning(f"Could not calculate correlation statistics: {e}")
        
        # Add diagonal line if requested
        if show_diagonal:
            # Get the current axis limits after the scatter plot
            xlim_current = ax.get_xlim()
            ylim_current = ax.get_ylim()
            
            # Calculate the range for the diagonal line based on the overlapping region
            line_min = max(xlim_current[0], ylim_current[0])
            line_max = min(xlim_current[1], ylim_current[1])
            
            # Only draw diagonal if there's an overlapping range
            if line_min < line_max:
                ax.plot([line_min, line_max], [line_min, line_max], 'k--', alpha=0.8, linewidth=1, label='y=x')
                # Only add legend if we don't already have one from multiple y-parameters
                if single_y or color_parameter is not None:
                    ax.legend()
        
        # Set labels
        ax.set_xlabel(xlabel if xlabel is not None else x_label_default, fontsize=16)
        
        if ylabel is not None:
            ax.set_ylabel(ylabel, fontsize=16)
        elif single_y:
            ax.set_ylabel(y_parameters[0], fontsize=16)
        else:
            # For multiple y-parameters, use the extracted common part (unless custom labels override)
            if custom_legend_labels is None:
                ax.set_ylabel(common_ylabel, fontsize=16)
            else:
                # With custom legend labels, use a generic ylabel
                ax.set_ylabel('Multiple Parameters', fontsize=16)
        
        # Set title
        if title is not None:
            ax.set_title(title, fontsize=16)
        
        # Make axes equal if values are in similar range (only if show_diagonal is True and single y-parameter)
        if show_diagonal and single_y:
            ax.set_aspect('equal', adjustable='box')
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Tight layout (legend is now inside, so no need for extra margin adjustment)
        plt.tight_layout()
        
        # Save if requested
        if output_file:
            fig.savefig(output_file, dpi=300, bbox_inches='tight')
            logger.info(f"Saved scatter plot to: {output_file}")
        
        # Show if requested
        if show:
            plt.show()
        
        total_points = len(x_values) * len(y_parameters)
        if single_y:
            logger.info(f"Created scatter plot: {len(x_values)} points")
        else:
            logger.info(f"Created multi-parameter scatter plot: {len(x_values)} points × {len(y_parameters)} parameters = {total_points} total points")
        
        return fig

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

    # ========================================================================
    # Additional Methods - Missing from original implementation
    # ========================================================================

    def compute_parameter_correlations(self, params: Optional[List[str]] = None,
                                     object_ids: Optional[Union[str, List[str]]] = None) -> 'numpy.ndarray':
        """
        Compute parameter correlation matrix.

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
        import numpy as np

        # Use free parameters if none specified
        if params is None:
            params = self.get_free_parameters()

        if not params:
            raise ValueError("No parameters available for correlation computation")

        # Get HDF5 table with appropriate filtering
        hdf5_table = self.load_hdf5_results(filter_snr=False, min_snr=0.0)

        # Apply object filtering if specified
        if object_ids is not None:
            object_mask = [str(obj_id) in [str(x) for x in object_ids] for obj_id in hdf5_table['ID']]
            if any(object_mask):
                hdf5_table = hdf5_table[object_mask]
            else:
                raise ValueError("None of the specified objects found in data")

        # Extract parameter data
        param_data = []
        available_params = []

        for param in params:
            # Find columns that match this parameter (handle statistical estimates)
            matching_cols = [col for col in hdf5_table.colnames if col.startswith(param)]

            if matching_cols:
                # Use the first matching column (could be enhanced to use specific estimates)
                col_name = matching_cols[0]
                param_data.append(hdf5_table[col_name])
                available_params.append(param)
            else:
                logger.warning(f"Parameter '{param}' not found in HDF5 data, skipping")

        if len(available_params) < 2:
            raise ValueError(f"Need at least 2 parameters for correlation computation, found {len(available_params)}")

        # Convert to numpy array and compute correlation matrix
        data_matrix = np.column_stack(param_data)
        correlation_matrix = np.corrcoef(data_matrix.T)

        logger.info(f"Computed correlation matrix for {len(available_params)} parameters across {len(hdf5_table)} objects")

        return correlation_matrix

    def get_parameter_statistics(self, params: Optional[List[str]] = None,
                               object_ids: Optional[List[str]] = None) -> Dict[str, Dict[str, float]]:
        """
        Get parameter statistics (mean, std, percentiles, etc.).

        Parameters
        ----------
        params : List[str], optional
            Parameters to compute statistics for. If None, uses free parameters.
        object_ids : List[str], optional
            Object IDs to include. If None, uses current scope.

        Returns
        -------
        Dict[str, Dict[str, float]]
            Dictionary with parameter names as keys and statistics as values
        """
        import numpy as np

        # Use free parameters if none specified
        if params is None:
            params = self.get_free_parameters()

        if not params:
            raise ValueError("No parameters available for statistics computation")

        # Get HDF5 table with appropriate filtering
        hdf5_table = self.load_hdf5_results(filter_snr=False, min_snr=0.0)

        # Apply object filtering if specified
        if object_ids is not None:
            object_mask = [str(obj_id) in [str(x) for x in object_ids] for obj_id in hdf5_table['ID']]
            if any(object_mask):
                hdf5_table = hdf5_table[object_mask]
            else:
                raise ValueError("None of the specified objects found in data")

        # Compute statistics for each parameter
        statistics = {}

        for param in params:
            # Find columns that match this parameter (handle statistical estimates)
            matching_cols = [col for col in hdf5_table.colnames if col.startswith(param)]

            if matching_cols:
                # Use the first matching column (could be enhanced to use specific estimates)
                col_name = matching_cols[0]
                param_values = np.array(hdf5_table[col_name])

                # Remove any NaN or infinite values
                valid_mask = np.isfinite(param_values)
                if np.any(valid_mask):
                    valid_values = param_values[valid_mask]

                    param_stats = {
                        'count': len(valid_values),
                        'mean': float(np.mean(valid_values)),
                        'std': float(np.std(valid_values)),
                        'min': float(np.min(valid_values)),
                        'max': float(np.max(valid_values)),
                        'median': float(np.median(valid_values)),
                        'p16': float(np.percentile(valid_values, 16)),
                        'p84': float(np.percentile(valid_values, 84)),
                        'p25': float(np.percentile(valid_values, 25)),
                        'p75': float(np.percentile(valid_values, 75))
                    }

                    statistics[param] = param_stats
                else:
                    logger.warning(f"No valid values found for parameter '{param}'")
            else:
                logger.warning(f"Parameter '{param}' not found in HDF5 data, skipping")

        logger.info(f"Computed statistics for {len(statistics)} parameters across {len(hdf5_table)} objects")

        return statistics

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
