"""
Simplified BayeSEDResults implementation.

This module contains the simplified BayeSEDResults class that eliminates
complex component architecture while maintaining full backward compatibility
with the existing API.
"""

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
    Simplified BayeSEDResults class with integrated functionality.

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
        Model configuration to load (not implemented in simplified version)
    object_id : str, optional
        Object ID for object-level access


    """

    def __init__(self, output_dir: Union[str, Path], catalog_name: Optional[str] = None,
                 model_config: Optional[Union[str, int]] = None, object_id: Optional[str] = None):
        """Initialize simplified BayeSEDResults."""

        # Store initialization parameters
        self.output_dir = Path(output_dir).resolve()
        self.catalog_name = catalog_name
        self.model_config = model_config
        self.object_id = object_id

        # Main data storage - loaded once
        self._hdf5_table = None
        self._hdf5_file = None

        # Parameter labeling
        self._custom_labels = {}

        # Initialize
        self._find_hdf5_file()
        self._load_hdf5_table()

        logger.info(f"Simplified BayeSEDResults initialized for catalog '{self.catalog_name}'")
        if self.object_id:
            logger.info(f"Object-level access for object: {self.object_id}")

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
            # Extract catalog name from first HDF5 file (before first underscore)
            filename = Path(hdf5_files[0]).stem
            self.catalog_name = filename.split('_')[0]
            logger.info(f"Auto-detected catalog name: '{self.catalog_name}'")

        # Find files matching the catalog
        catalog_pattern = f"{self.catalog_name}_*.hdf5"
        catalog_files = list(self.output_dir.glob(catalog_pattern))

        if not catalog_files:
            raise FileNotFoundError(f"No HDF5 files found for catalog '{self.catalog_name}' in {self.output_dir}")

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

                for f in catalog_files:
                    filename = f.stem
                    if filename.startswith(self.catalog_name + '_'):
                        config_name = filename[len(self.catalog_name) + 1:]

                        # Exact match or partial match
                        if config_name == model_config_str or model_config_str in config_name:
                            target_file = f
                            self.model_config = config_name  # Update to full name
                            break

                if target_file is None:
                    available_configs = []
                    for f in catalog_files:
                        filename = f.stem
                        if filename.startswith(self.catalog_name + '_'):
                            config_name = filename[len(self.catalog_name) + 1:]
                            available_configs.append(config_name)

                    raise ValueError(f"Model config '{model_config_str}' not found for catalog '{self.catalog_name}'. "
                                   f"Available configs: {available_configs}")

            self._hdf5_file = str(target_file)
            logger.info(f"Selected HDF5 file for catalog '{self.catalog_name}', config '{self.model_config}': {target_file.name}")

        else:
            # No model_config specified - use first available or require selection if multiple
            if len(catalog_files) == 1:
                # Only one config available - use it
                self._hdf5_file = str(catalog_files[0])
                filename = Path(self._hdf5_file).stem
                self.model_config = filename[len(self.catalog_name) + 1:]
                logger.info(f"Auto-selected single config '{self.model_config}' for catalog '{self.catalog_name}'")

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
        config_files = []
        for file_path in all_files:
            filename = file_path.stem
            # Check if the filename contains the model config
            # Files are typically named like: {config_name}_sample_par.txt or {config_name}_bestfit.fits
            if self.model_config in filename:
                config_files.append(file_path)

        return config_files

    # ========================================================================
    # Public API Methods - Maintaining backward compatibility
    # ========================================================================

    def get_parameter_names(self) -> List[str]:
        """
        Get list of parameter names.

        Returns
        -------
        List[str]
            List of parameter names
        """
        if self._hdf5_table is None:
            raise RuntimeError("HDF5 table not loaded")

        # Get all parameter names (excluding 'ID')
        param_names = [col for col in self._hdf5_table.colnames if col != 'ID']

        return param_names

    def get_free_parameters(self) -> List[str]:
        """
        Get list of free (fitted) parameters by reading paramnames files.

        Returns
        -------
        List[str]
            List of free parameter names
        """
        # Read paramnames file to distinguish free vs derived parameters
        try:
            # Find a paramnames file to read parameter structure
            if self.object_id:
                # Object-level access - use specific object
                paramnames_files = self._find_config_files(self.object_id, "*_sample_par.paramnames")
            else:
                # Sample-level access - use first available object
                objects = self.list_objects()
                if not objects:
                    return []
                paramnames_files = self._find_config_files(objects[0], "*_sample_par.paramnames")

            if not paramnames_files:
                logger.warning("No paramnames files found, assuming all parameters are free")
                return self.get_parameter_names()

            # Read the first paramnames file
            paramnames_file = paramnames_files[0]
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
            logger.warning(f"Failed to read paramnames file: {e}. Falling back to all parameters.")
            return self.get_parameter_names()

    def get_derived_parameters(self) -> List[str]:
        """
        Get list of derived parameters by reading paramnames files.

        Returns
        -------
        List[str]
            List of derived parameter names
        """
        # Read paramnames file to distinguish free vs derived parameters
        try:
            # Find a paramnames file to read parameter structure
            if self.object_id:
                # Object-level access - use specific object
                paramnames_files = self._find_config_files(self.object_id, "*_sample_par.paramnames")
            else:
                # Sample-level access - use first available object
                objects = self.list_objects()
                if not objects:
                    return []
                paramnames_files = self._find_config_files(objects[0], "*_sample_par.paramnames")

            if not paramnames_files:
                logger.warning("No paramnames files found, returning empty derived parameters list")
                return []

            # Read the first paramnames file
            paramnames_file = paramnames_files[0]
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
            logger.warning(f"Failed to read paramnames file: {e}. Returning empty derived parameters list.")
            return []

    def get_parameter_values(self, parameter_name: str,
                           object_ids: Optional[Union[str, List[str]]] = None) -> 'astropy.table.Table':
        """
        Get parameter values by filtering the loaded HDF5 table.

        Parameters
        ----------
        parameter_name : str
            Name of the parameter to retrieve
        object_ids : str or List[str], optional
            Specific object ID(s) to filter. Can be a single string or list of strings.

        Returns
        -------
        astropy.table.Table
            Sub-table containing ID column and all parameter columns that
            start with the given parameter name
        """
        if self._hdf5_table is None:
            raise RuntimeError("HDF5 table not loaded")

        # Find all columns that start with the parameter name
        available_columns = self._hdf5_table.colnames
        matching_columns = [col for col in available_columns if col.startswith(parameter_name)]

        if not matching_columns:
            raise ValueError(
                f"No columns found starting with '{parameter_name}' in HDF5 table. "
                f"Available parameter prefixes: {list(set([col.split('_')[0] for col in available_columns if '_' in col and col != 'ID']))[:10]}..."
            )

        # Create sub-table with ID and all matching parameter columns
        columns_to_include = ['ID'] + sorted(matching_columns)
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



    def get_posterior_samples(self, object_id: Optional[str] = None) -> Any:
        """
        Get posterior samples by loading GetDist samples on-demand.

        Parameters
        ----------
        object_id : str, optional
            Object ID to get samples for

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
                    logger.warning(f"No object_id provided for get_posterior_samples. Using first available object: {object_id}")
                else:
                    raise ValueError("No objects available for posterior samples")

        # Find sample files for this object matching the current configuration
        sample_files = self._find_config_files(object_id, "*_sample_par.txt")
        paramnames_files = self._find_config_files(object_id, "*_sample_par.paramnames")

        if not sample_files or not paramnames_files:
            object_dir = self.output_dir / self.catalog_name / object_id
            raise FileNotFoundError(f"No sample files found for object {object_id} and config '{self.model_config}' in {object_dir}")

        # Use the first matching file (should be unique for a given config)
        sample_file = sample_files[0]
        paramnames_file = paramnames_files[0]

        # Extract base name for GetDist (remove .txt suffix, keep _sample_par)
        base_name = str(sample_file).replace(".txt", "")

        try:
            # Try to use GetDist - pass the full base name including _sample_par
            import getdist
            samples = getdist.loadMCSamples(base_name)
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
                    logger.warning(f"No object_id provided for get_bestfit_spectrum. Using first available object: {object_id}")
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

    def get_getdist_samples(self, object_id: Optional[str] = None) -> Any:
        """
        Get GetDist samples with parameter management.

        Parameters
        ----------
        object_id : str, optional
            Object ID to get samples for

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
                    logger.warning(f"No object_id provided for get_getdist_samples. Using first available object: {object_id}")
                else:
                    raise ValueError("No objects available for GetDist samples")

        # Find sample files for this object matching the current configuration
        sample_files = self._find_config_files(object_id, "*_sample_par.txt")
        paramnames_files = self._find_config_files(object_id, "*_sample_par.paramnames")

        if not sample_files or not paramnames_files:
            object_dir = self.output_dir / self.catalog_name / object_id
            raise FileNotFoundError(f"No sample files found for object {object_id} and config '{self.model_config}' in {object_dir}")

        # Use the first matching file (should be unique for a given config)
        sample_file = sample_files[0]
        paramnames_file = paramnames_files[0]

        # Extract base name for GetDist (remove .txt suffix, keep _sample_par)
        base_name = str(sample_file).replace(".txt", "")

        try:
            # Try to use GetDist - pass the full base name including _sample_par
            import getdist
            from getdist import MCSamples

            samples = getdist.loadMCSamples(base_name)

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
        logger.info(f"Set custom labels for {len(custom_labels)} parameters")

    def plot_posterior(self, params: Optional[List[str]] = None,
                      object_id: Optional[str] = None,
                      method: str = 'getdist', filled: bool = True,
                      show: bool = True, output_file: Optional[str] = None,
                      **kwargs) -> Any:
        """
        Plot posterior distributions using GetDist.

        Parameters
        ----------
        params : List[str], optional
            Parameters to plot. If None, uses free parameters.
        object_id : str, optional
            Object ID to plot for. If None, uses current scope.
        method : str, default 'getdist'
            Plotting method to use
        filled : bool, default True
            Whether to use filled contours
        show : bool, default True
            Whether to display the plot
        output_file : str, optional
            Output file path for saving
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
                    logger.warning(f"No object_id provided for plot_posterior. Using first available object: {object_id}")
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
        except ImportError:
            raise ImportError("GetDist is required for plotting. Install with: pip install getdist")

        # Create plotter
        g = plots.get_subplot_plotter()

        # Create plot based on number of parameters
        if len(params) == 1:
            g.plot_1d(samples, params[0], **kwargs)
        else:
            g.triangle_plot([samples], params, filled=filled, **kwargs)

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
                    **kwargs) -> Any:
        """
        Plot best-fit SED using the bayesed.plotting module.

        Parameters
        ----------
        object_id : str, optional
            Object ID to plot for. If None, uses current scope.
        output_file : str, optional
            Output file path for saving
        show : bool, default True
            Whether to display the plot
        **kwargs
            Additional plotting arguments

        Returns
        -------
        Any
            Plot object
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
                    logger.warning(f"No object_id provided for plot_bestfit. Using first available object: {object_id}")
                else:
                    raise ValueError("No objects available for plotting best-fit spectrum")

        # Find bestfit FITS files for this object
        object_dir = self.output_dir / self.catalog_name / object_id
        if not object_dir.exists():
            raise FileNotFoundError(f"Object directory not found: {object_dir}")

        # Look for bestfit FITS files
        fits_files = list(object_dir.glob("*_bestfit.fits"))

        if not fits_files:
            raise FileNotFoundError(f"No bestfit FITS files found for object {object_id} in {object_dir}")

        # Use the first bestfit file (there should typically be only one)
        fits_file = fits_files[0]

        # Import plotting function
        from ..plotting import plot_bestfit

        # Create plot with file path
        fig = plot_bestfit(fits_file, show=show, output_file=output_file, **kwargs)

        return fig

    def plot_posterior_free(self, output_file: Optional[str] = None,
                           show: bool = True, object_id: Optional[str] = None,
                           **kwargs) -> Any:
        """
        Plot posterior distributions (corner plot) for free parameters.

        Parameters
        ----------
        output_file : str, optional
            Output file path for saving plot
        show : bool, default True
            Whether to display the plot
        object_id : str, optional
            Object ID to plot. If None and in sample-level mode, uses first object.
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
                        logger.warning(f"No object_id provided for plot_posterior_free. Using first available object: {object_id}")
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
                                     output_file=output_file, show=show, **kwargs)
        except Exception as e:
            if "GetDist" in str(e) or "No chains found" in str(e):
                logger.warning(f"GetDist plotting not available: {e}")
                logger.info("Consider using individual parameter plotting or ensure GetDist-compatible sample files exist")
                # Return a placeholder or simple message
                print(f"Free parameters ({len(self.get_free_parameters())}): {self.get_free_parameters()}")
                return None
            else:
                raise

    def plot_posterior_derived(self, max_params: int = 10,
                              output_file: Optional[str] = None,
                              show: bool = True, object_id: Optional[str] = None,
                              **kwargs) -> Any:
        """
        Plot posterior distributions (corner plot) for derived parameters.

        Parameters
        ----------
        max_params : int, default 10
            Maximum number of parameters to plot
        output_file : str, optional
            Output file path for saving plot
        show : bool, default True
            Whether to display the plot
        object_id : str, optional
            Object ID to plot. If None and in sample-level mode, uses first object.
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
                        logger.warning(f"No object_id provided for plot_posterior_derived. Using first available object: {object_id}")
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
                                     output_file=output_file, show=show, **kwargs)
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

    def print_summary(self) -> None:
        """Print a summary of the loaded results."""
        print("=" * 60)
        print("BayeSEDResults Summary (Simplified)")
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
            print(f"  Total Parameters: {len(all_params)}")
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
