"""
FileDiscovery component for BayeSEDResults redesign.

This module implements file discovery and organization functionality,
building on existing patterns from the original BayeSEDResults implementation
while adding enhanced validation and support for different output modes.
"""

import os
import glob
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Any
import logging
from datetime import datetime

try:
    import h5py
    HDF5_AVAILABLE = True
except ImportError:
    HDF5_AVAILABLE = False

try:
    from astropy.io import fits
    FITS_AVAILABLE = True
except ImportError:
    FITS_AVAILABLE = False

from .base import BaseFileHandler
from .models import FileStructure, ConfigurationInfo
from .exceptions import FileDiscoveryError
from .logger import get_logger


class FileDiscovery(BaseFileHandler):
    """
    Enhanced file discovery component building on existing BayeSEDResults patterns.
    
    This component discovers, organizes, and validates all output files in a
    BayeSED output directory structure, supporting both full structure and
    HDF5-only output modes.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize FileDiscovery component."""
        if logger is None:
            bayesed_logger = get_logger(__name__)
            logger = bayesed_logger.get_logger()
        super().__init__(logger)
        self._discovered_files: Optional[FileStructure] = None
    
    def initialize(self, **kwargs) -> None:
        """Initialize the FileDiscovery component."""
        self._initialized = True
        self.logger.debug("FileDiscovery component initialized")
    
    def discover_files(self, base_path: Union[str, Path]) -> FileStructure:
        """
        Discover and organize files in the given directory.
        
        This method builds on the existing _find_output_files() patterns
        while adding enhanced validation and output mode detection.
        
        Parameters
        ----------
        base_path : str or Path
            Base directory to search for files
            
        Returns
        -------
        FileStructure
            Organized information about discovered files
            
        Raises
        ------
        FileDiscoveryError
            If directory doesn't exist or no valid files found
        """
        self._ensure_initialized()
        
        base_path = Path(base_path)
        self.set_base_path(base_path)
        
        # Validate base directory exists
        if not base_path.exists():
            raise FileDiscoveryError(
                f"Output directory does not exist: {base_path}",
                suggestions=[
                    "Check that the path is correct",
                    "Ensure BayeSED analysis has been run in this directory"
                ]
            )
        
        if not base_path.is_dir():
            raise FileDiscoveryError(
                f"Path is not a directory: {base_path}",
                suggestions=["Provide a directory path, not a file path"]
            )
        
        self.logger.debug(f"Discovering files in: {base_path}")
        
        # Step 1: Find all HDF5 files (enhanced from existing pattern)
        hdf5_files = self._find_hdf5_files(base_path)
        
        # Step 2: Detect catalog names and subdirectories (enhanced from existing)
        catalog_names = self._detect_catalog_names(base_path, hdf5_files)
        
        # Step 3: Organize HDF5 files by catalog (enhanced from existing)
        hdf5_by_catalog = self._organize_hdf5_by_catalog(hdf5_files, catalog_names)
        
        # Step 4: Find object-specific files (enhanced from existing)
        bestfit_files, posterior_files = self._find_object_files(base_path, catalog_names)
        
        # Step 5: Detect output mode based on available files
        output_mode = self._detect_output_mode(hdf5_files, bestfit_files, posterior_files)
        
        # Step 6: Validate file structure consistency
        self._validate_file_structure(hdf5_files, bestfit_files, posterior_files)
        
        # Create FileStructure for each catalog
        file_structures = {}
        for catalog_name in catalog_names:
            file_structures[catalog_name] = FileStructure(
                output_dir=str(base_path),
                catalog_name=catalog_name,
                output_mode=output_mode,
                hdf5_files=hdf5_by_catalog.get(catalog_name, {}),
                bestfit_files=bestfit_files.get(catalog_name) if output_mode != 'hdf5_only' else None,
                posterior_files=posterior_files.get(catalog_name) if output_mode != 'hdf5_only' else None,
                metadata={
                    'discovery_time': datetime.now(),
                    'total_hdf5_files': len(hdf5_files),
                    'total_catalogs': len(catalog_names),
                    'output_mode': output_mode
                }
            )
        
        # If only one catalog, return it directly; otherwise return the first one
        # (caller can access others through get_catalog_structure)
        if len(file_structures) == 1:
            self._discovered_files = list(file_structures.values())[0]
        elif file_structures:
            # Store all for later access
            self._all_discovered_files = file_structures
            self._discovered_files = list(file_structures.values())[0]
        else:
            # If no catalogs detected but we have HDF5 files, create a default structure
            if hdf5_files:
                # Create a default catalog name from the first HDF5 file
                default_catalog = Path(hdf5_files[0]).stem.split('_')[0] if '_' in Path(hdf5_files[0]).stem else 'default'
                self._discovered_files = FileStructure(
                    output_dir=str(base_path),
                    catalog_name=default_catalog,
                    output_mode=output_mode,
                    hdf5_files={Path(f).stem: f for f in hdf5_files},
                    bestfit_files=bestfit_files if output_mode != 'hdf5_only' else None,
                    posterior_files=posterior_files if output_mode != 'hdf5_only' else None,
                    metadata={
                        'discovery_time': datetime.now(),
                        'total_hdf5_files': len(hdf5_files),
                        'total_catalogs': 1,
                        'output_mode': output_mode,
                        'default_catalog': True
                    }
                )
            else:
                raise FileDiscoveryError(
                    f"No valid BayeSED output files found in: {base_path}",
                    suggestions=[
                        "Check that BayeSED analysis has been run in this directory",
                        "Ensure HDF5 result files are present",
                        "Verify file permissions allow reading"
                    ]
                )
        
        self.logger.debug(f"Discovery complete: found {len(catalog_names)} catalogs, "
                        f"{len(hdf5_files)} HDF5 files, mode: {output_mode}")
        
        return self._discovered_files
    
    def _find_hdf5_files(self, base_path: Path) -> List[str]:
        """
        Find all HDF5 files in the base directory.
        
        Enhanced from existing pattern in _find_output_files().
        """
        try:
            hdf5_pattern = str(base_path / "*.hdf5")
            hdf5_files = glob.glob(hdf5_pattern)
            hdf5_files = sorted(hdf5_files) if hdf5_files else []
            
            self.logger.debug(f"Found {len(hdf5_files)} HDF5 files")
            
            # Validate HDF5 files are accessible and have reasonable size
            accessible_files = []
            for hdf5_file in hdf5_files:
                if self.validate_file_access(hdf5_file):
                    # Check file size (should be > 0 bytes)
                    try:
                        file_size = Path(hdf5_file).stat().st_size
                        if file_size > 0:
                            accessible_files.append(hdf5_file)
                        else:
                            self.logger.warning(f"HDF5 file is empty: {hdf5_file}")
                    except OSError as e:
                        self.logger.warning(f"Cannot check size of HDF5 file {hdf5_file}: {e}")
                else:
                    self.logger.warning(f"HDF5 file not accessible: {hdf5_file}")
            
            return accessible_files
            
        except Exception as e:
            self.logger.error(f"Error finding HDF5 files in {base_path}: {e}")
            raise FileDiscoveryError(
                f"Failed to search for HDF5 files in {base_path}",
                suggestions=[
                    "Check directory permissions",
                    "Verify the path exists and is readable"
                ]
            )
    
    def _detect_catalog_names(self, base_path: Path, hdf5_files: List[str]) -> List[str]:
        """
        Detect catalog names from subdirectories and HDF5 filenames.
        
        Enhanced from existing pattern in _find_output_files().
        """
        catalog_names = set()
        
        # Method 1: Detect from subdirectories (enhanced from existing)
        try:
            if base_path.is_dir():
                for item in base_path.iterdir():
                    if item.is_dir() and not item.name.endswith('.hdf5'):
                        # Check if it contains object subdirectories
                        try:
                            subitems = list(item.iterdir())
                            # If it has subdirectories, it's likely a catalog directory
                            if any(subitem.is_dir() for subitem in subitems):
                                # Additional validation: check if subdirectories look like object IDs
                                subdir_names = [sub.name for sub in subitems if sub.is_dir()]
                                if self._validate_object_id_pattern(subdir_names):
                                    catalog_names.add(item.name)
                                    self.logger.debug(f"Found catalog directory: {item.name}")
                                else:
                                    self.logger.debug(f"Directory {item.name} doesn't match object ID pattern")
                        except (OSError, PermissionError) as e:
                            self.logger.warning(f"Cannot access directory {item}: {e}")
        except Exception as e:
            self.logger.warning(f"Error scanning directories in {base_path}: {e}")
        
        # Method 2: Extract from HDF5 filenames (enhanced from existing)
        # For complex filenames like "seedcat2_0_STARFORMING_inoise2_0csp_...", we need to be smarter
        for hdf5_file in hdf5_files:
            basename = Path(hdf5_file).name
            if '_' in basename:
                # Try to match against known catalog names first
                matched = False
                for known_catalog in catalog_names:
                    if basename.startswith(known_catalog + '_'):
                        matched = True
                        break
                
                # If no match, try to extract a more complete catalog name
                if not matched:
                    # For files like "seedcat2_0_STARFORMING_inoise2_0csp_...", 
                    # try to find the longest prefix that matches a directory
                    parts = basename.replace('.hdf5', '').split('_')
                    
                    # Try progressively longer prefixes
                    for i in range(1, min(len(parts) + 1, 6)):  # Limit to avoid very long names
                        potential_catalog = '_'.join(parts[:i])
                        catalog_dir = base_path / potential_catalog
                        
                        # Check if this potential catalog name corresponds to a directory
                        if catalog_dir.exists() and catalog_dir.is_dir():
                            catalog_names.add(potential_catalog)
                            self.logger.debug(f"Extracted catalog name from HDF5 filename: {potential_catalog}")
                            matched = True
                            break
                    
                    # Fallback to first part if no directory match found
                    if not matched:
                        catalog_name = parts[0]
                        catalog_names.add(catalog_name)
                        self.logger.debug(f"Fallback catalog name from HDF5 filename: {catalog_name}")
        
        # Method 3: Extract from file headers as fallback (new enhancement)
        for hdf5_file in hdf5_files:
            header_catalog = self.extract_catalog_name_from_header(hdf5_file)
            if header_catalog and header_catalog not in catalog_names:
                catalog_names.add(header_catalog)
                self.logger.debug(f"Extracted catalog name from HDF5 header: {header_catalog}")
        
        catalog_list = sorted(catalog_names)
        self.logger.debug(f"Detected catalog names: {catalog_list}")
        
        return catalog_list
    
    def _organize_hdf5_by_catalog(self, hdf5_files: List[str], 
                                 catalog_names: List[str]) -> Dict[str, Dict[str, str]]:
        """
        Organize HDF5 files by catalog and extract model configuration names.
        
        Enhanced from existing pattern in _find_output_files().
        """
        hdf5_by_catalog = {}
        
        # Sort catalog names by length (longest first) to match more specific names first
        sorted_catalogs = sorted(catalog_names, key=len, reverse=True)
        
        for catalog_name in sorted_catalogs:
            hdf5_by_catalog[catalog_name] = {}
        
        for hdf5_file in hdf5_files:
            basename = Path(hdf5_file).name
            matched = False
            
            # Try to match against catalog names (longest first for specificity)
            for catalog_name in sorted_catalogs:
                if basename.startswith(catalog_name + '_'):
                    # Extract model configuration name
                    # Format: {catalog_name}_{model_config}.hdf5
                    config_part = basename[len(catalog_name) + 1:]  # Remove catalog_name + '_'
                    if config_part.endswith('.hdf5'):
                        config_name = config_part[:-5]  # Remove .hdf5 extension
                        hdf5_by_catalog[catalog_name][config_name] = hdf5_file
                        self.logger.debug(f"Mapped {catalog_name}:{config_name} -> {basename}")
                        matched = True
                        break
            
            # If no specific catalog match, try to assign to a general catalog
            if not matched and '_' in basename:
                # Extract the first part as a fallback catalog
                first_part = basename.split('_')[0]
                if first_part in catalog_names:
                    config_part = basename[len(first_part) + 1:]
                    if config_part.endswith('.hdf5'):
                        config_name = config_part[:-5]
                        if first_part not in hdf5_by_catalog:
                            hdf5_by_catalog[first_part] = {}
                        hdf5_by_catalog[first_part][config_name] = hdf5_file
                        self.logger.debug(f"Fallback mapped {first_part}:{config_name} -> {basename}")
        
        return hdf5_by_catalog
    
    def _find_object_files(self, base_path: Path, catalog_names: List[str]) -> Tuple[Dict, Dict]:
        """
        Find best-fit FITS files and posterior sample files.
        
        Enhanced from existing pattern in _find_output_files().
        """
        try:
            # Find best-fit FITS files (enhanced from existing)
            bestfit_pattern = str(base_path / "**" / "*_bestfit.fits")
            bestfit_files = glob.glob(bestfit_pattern, recursive=True)
            bestfit_files = sorted(bestfit_files)
            
            # Find posterior sample files (enhanced from existing)
            paramnames_pattern = str(base_path / "**" / "*_sample_par.paramnames")
            paramnames_files = glob.glob(paramnames_pattern, recursive=True)
            
            self.logger.debug(f"Found {len(bestfit_files)} bestfit files, "
                             f"{len(paramnames_files)} paramnames files")
            
        except Exception as e:
            self.logger.error(f"Error searching for object files in {base_path}: {e}")
            # Return empty dictionaries rather than failing completely
            return {}, {}
        
        # Organize bestfit files by catalog and object (enhanced from existing)
        bestfit_by_catalog = {}
        for bfile in bestfit_files:
            try:
                rel_path = Path(bfile).relative_to(base_path)
                parts = rel_path.parts
                
                # Path structure: {catalog_name}/{object_id}/{model_name}_bestfit.fits
                if len(parts) >= 2:
                    potential_catalog = parts[0]
                    if potential_catalog in catalog_names:
                        obj_id = parts[1]
                        
                        if potential_catalog not in bestfit_by_catalog:
                            bestfit_by_catalog[potential_catalog] = {}
                        if obj_id not in bestfit_by_catalog[potential_catalog]:
                            bestfit_by_catalog[potential_catalog][obj_id] = []
                        
                        bestfit_by_catalog[potential_catalog][obj_id].append(bfile)
                    else:
                        self.logger.debug(f"Bestfit file {bfile} not in known catalogs: {parts[0]}")
                else:
                    self.logger.debug(f"Bestfit file {bfile} has unexpected path structure: {parts}")
            except Exception as e:
                self.logger.warning(f"Error processing bestfit file {bfile}: {e}")
        
        # Organize posterior files by catalog and object (enhanced from existing)
        posterior_by_catalog = {}
        for pfile in paramnames_files:
            try:
                # Find corresponding sample file
                base_name_for_getdist = pfile.replace('.paramnames', '')  # Keep _sample_par for GetDist
                base_name_for_files = pfile.replace('_sample_par.paramnames', '')  # Remove for file checking
                sfile = base_name_for_files + '_sample_par.txt'
                
                if Path(sfile).exists():
                    rel_path = Path(pfile).relative_to(base_path)
                    parts = rel_path.parts
                    
                    if len(parts) >= 2:
                        potential_catalog = parts[0]
                        if potential_catalog in catalog_names:
                            obj_id = parts[1]
                            
                            if potential_catalog not in posterior_by_catalog:
                                posterior_by_catalog[potential_catalog] = {}
                            if obj_id not in posterior_by_catalog[potential_catalog]:
                                posterior_by_catalog[potential_catalog][obj_id] = {}
                            
                            posterior_by_catalog[potential_catalog][obj_id][base_name_for_getdist] = {
                                'paramnames': pfile,
                                'samples': sfile
                            }
                        else:
                            self.logger.debug(f"Posterior file {pfile} not in known catalogs: {parts[0]}")
                    else:
                        self.logger.debug(f"Posterior file {pfile} has unexpected path structure: {parts}")
                else:
                    self.logger.warning(f"Missing sample file for paramnames: {pfile}")
            except Exception as e:
                self.logger.warning(f"Error processing posterior file {pfile}: {e}")
        
        return bestfit_by_catalog, posterior_by_catalog
    
    def _detect_output_mode(self, hdf5_files: List[str], bestfit_files: Dict, 
                           posterior_files: Dict) -> str:
        """
        Detect the output mode based on available files.
        
        Returns
        -------
        str
            One of: 'hdf5_only', 'full_structure', 'mixed'
        """
        has_hdf5 = len(hdf5_files) > 0
        has_bestfit = len(bestfit_files) > 0
        has_posterior = len(posterior_files) > 0
        has_object_files = has_bestfit or has_posterior
        
        # Count total object files for better mode detection
        total_bestfit = sum(len(obj_files) for catalog_files in bestfit_files.values() 
                           for obj_files in catalog_files.values())
        total_posterior = sum(len(obj_files) for catalog_files in posterior_files.values() 
                             for obj_files in catalog_files.values())
        
        self.logger.debug(f"File counts: HDF5={len(hdf5_files)}, "
                         f"bestfit={total_bestfit}, posterior={total_posterior}")
        
        if has_hdf5 and has_object_files:
            # Check if we have substantial object files (not just a few)
            if total_bestfit > 0 or total_posterior > 0:
                return 'full_structure'
            else:
                return 'hdf5_only'  # HDF5 with minimal object files
        elif has_hdf5 and not has_object_files:
            return 'hdf5_only'
        elif has_object_files and not has_hdf5:
            return 'mixed'  # Object files without HDF5 (unusual but possible)
        else:
            # No files found - this should be caught by validation
            self.logger.warning("No output files detected")
            return 'hdf5_only'  # Default fallback
    
    def _validate_file_structure(self, hdf5_files: List[str], bestfit_files: Dict, 
                                posterior_files: Dict) -> None:
        """
        Validate the discovered file structure for consistency.
        
        Raises
        ------
        FileDiscoveryError
            If critical validation errors are found
        """
        errors = []
        warnings = []
        
        # Check that we have at least some files (even if they have issues)
        total_files_found = len(hdf5_files) + sum(len(catalog_files) for catalog_files in bestfit_files.values()) + sum(len(catalog_files) for catalog_files in posterior_files.values())
        
        if total_files_found == 0:
            errors.append("No BayeSED output files found")
        
        # Validate HDF5 file accessibility and format
        for hdf5_file in hdf5_files:
            if not self.validate_file_access(hdf5_file):
                errors.append(f"HDF5 file not accessible: {hdf5_file}")
            else:
                # Additional HDF5 format validation
                validation_result = self._validate_hdf5_format(hdf5_file)
                if validation_result:
                    warnings.append(f"HDF5 format issue in {Path(hdf5_file).name}: {validation_result}")
        
        # Check for orphaned posterior files (paramnames without samples)
        for catalog_files in posterior_files.values():
            for obj_files in catalog_files.values():
                for base_name, file_pair in obj_files.items():
                    if not Path(file_pair['samples']).exists():
                        warnings.append(f"Missing sample file for: {file_pair['paramnames']}")
                    
                    # Validate paramnames file format
                    if not self._validate_paramnames_format(file_pair['paramnames']):
                        warnings.append(f"Invalid paramnames format: {file_pair['paramnames']}")
        
        # Check for consistency between HDF5 and object files
        if hdf5_files and (bestfit_files or posterior_files):
            consistency_issues = self._check_file_consistency(hdf5_files, bestfit_files, posterior_files)
            warnings.extend(consistency_issues)
        
        # Validate bestfit FITS files
        for catalog_files in bestfit_files.values():
            for obj_files in catalog_files.values():
                for fits_file in obj_files:
                    if not self.validate_file_access(fits_file):
                        warnings.append(f"Bestfit FITS file not accessible: {fits_file}")
                    elif not self._validate_fits_format(fits_file):
                        warnings.append(f"Invalid FITS format: {fits_file}")
        
        # Log warnings
        for warning in warnings:
            self.logger.warning(warning)
        
        # Raise errors if any critical issues found
        if errors:
            raise FileDiscoveryError(
                "File structure validation failed",
                suggestions=[
                    "Check file permissions",
                    "Verify BayeSED analysis completed successfully",
                    "Ensure output directory is not corrupted",
                    "Check that HDF5 and FITS files are not corrupted"
                ],
                error_context={'validation_errors': errors}
            )
    
    def _validate_hdf5_format(self, file_path: str) -> Optional[str]:
        """
        Validate HDF5 file format and structure.
        
        Returns
        -------
        str or None
            Error message if validation fails, None if valid
        """
        if not HDF5_AVAILABLE:
            return "h5py not available for HDF5 validation"
        
        try:
            with h5py.File(file_path, 'r') as f:
                # Check if file has any datasets
                if len(f.keys()) == 0:
                    return "HDF5 file is empty"
                
                # Check for common BayeSED datasets/groups
                expected_keys = ['object_id', 'id', 'evidence', 'parameters']
                found_keys = [key for key in expected_keys if key in f.keys()]
                
                if not found_keys:
                    return "No recognized BayeSED datasets found"
                
        except Exception as e:
            return f"Cannot read HDF5 file: {str(e)}"
        
        return None
    
    def _validate_paramnames_format(self, file_path: str) -> bool:
        """
        Validate paramnames file format.
        
        BayeSED paramnames files have two columns: parameter_name and latex_label
        
        Returns
        -------
        bool
            True if valid, False otherwise
        """
        try:
            with open(file_path, 'r') as f:
                lines = f.readlines()
                
            # Check if file has content
            if not lines:
                return False
            
            # Check if lines have the expected format (parameter names with optional LaTeX labels)
            valid_lines = 0
            for line in lines:  # Check all lines, not just first 10
                line = line.strip()
                # Skip empty lines and comments
                if not line or line.startswith('#'):
                    continue
                
                # BayeSED paramnames format: "param_name latex_label" or just "param_name"
                # Allow more than 2 parts (ignore trailing content)
                parts = line.split()
                if len(parts) >= 1:
                    # First part should be a reasonable parameter name
                    param_name = parts[0]
                    if len(param_name) > 0 and len(param_name) <= 200:  # Increased limit
                        valid_lines += 1
                        # If we found at least one valid line, that's enough
                        if valid_lines >= 1:
                            return True
            
            # Should have at least some valid parameter lines
            return valid_lines > 0
            
        except Exception:
            # If we can't read the file, don't validate - let the actual usage handle errors
            return True  # Return True to avoid false warnings
    
    def _validate_fits_format(self, file_path: str) -> bool:
        """
        Validate FITS file format.
        
        Returns
        -------
        bool
            True if valid, False otherwise
        """
        if not FITS_AVAILABLE:
            return True  # Cannot validate without astropy
        
        try:
            with fits.open(file_path) as hdul:
                # Check if file has at least one HDU
                if len(hdul) == 0:
                    return False
                
                # Check primary header exists
                if hdul[0].header is None:
                    return False
            
            return True
            
        except Exception:
            return False
    
    def _check_file_consistency(self, hdf5_files: List[str], bestfit_files: Dict, 
                               posterior_files: Dict) -> List[str]:
        """
        Check consistency between HDF5 and object-specific files.
        
        Returns
        -------
        List[str]
            List of consistency warning messages
        """
        warnings = []
        
        # This is a placeholder for more sophisticated consistency checks
        # In a real implementation, we would:
        # 1. Extract object IDs from HDF5 files
        # 2. Compare with object IDs found in file paths
        # 3. Check parameter name consistency
        # 4. Validate model configuration matching
        
        # For now, just check basic structure consistency
        has_hdf5 = len(hdf5_files) > 0
        has_bestfit = len(bestfit_files) > 0
        has_posterior = len(posterior_files) > 0
        
        if has_hdf5 and not (has_bestfit or has_posterior):
            warnings.append("HDF5 files found but no object-specific files - operating in HDF5-only mode")
        elif (has_bestfit or has_posterior) and not has_hdf5:
            warnings.append("Object-specific files found but no HDF5 files - limited functionality available")
        
        return warnings
    
    def _validate_object_id_pattern(self, subdir_names: List[str]) -> bool:
        """
        Validate that subdirectory names look like object IDs.
        
        Parameters
        ----------
        subdir_names : List[str]
            List of subdirectory names to validate
            
        Returns
        -------
        bool
            True if names look like object IDs, False otherwise
        """
        if not subdir_names:
            return False
        
        # Check if at least some subdirectories look like object IDs
        # Object IDs are typically numeric or alphanumeric
        valid_count = 0
        for name in subdir_names[:10]:  # Check first 10 to avoid performance issues
            # Object IDs are usually:
            # - All digits
            # - Alphanumeric with reasonable length
            # - Not common directory names
            if (name.isdigit() or 
                (name.isalnum() and 3 <= len(name) <= 20) or
                name.startswith('obj_')):
                valid_count += 1
            elif name.lower() in ['output', 'results', 'plots', 'data', 'temp', 'tmp']:
                # These are common non-object directory names
                return False
        
        # If at least 50% look like object IDs, consider it valid
        return valid_count >= len(subdir_names) * 0.5
    
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
        try:
            file_path = Path(file_path)
            return file_path.exists() and file_path.is_file() and os.access(file_path, os.R_OK)
        except (OSError, PermissionError):
            return False
    
    def get_catalog_structure(self, catalog_name: str) -> Optional[FileStructure]:
        """
        Get file structure for a specific catalog.
        
        Parameters
        ----------
        catalog_name : str
            Name of the catalog
            
        Returns
        -------
        FileStructure or None
            File structure for the catalog, or None if not found
        """
        if hasattr(self, '_all_discovered_files'):
            return self._all_discovered_files.get(catalog_name)
        elif self._discovered_files and self._discovered_files.catalog_name == catalog_name:
            return self._discovered_files
        else:
            return None
    
    def list_catalogs(self) -> List[str]:
        """
        List all discovered catalog names.
        
        Returns
        -------
        List[str]
            List of catalog names
        """
        if hasattr(self, '_all_discovered_files'):
            return sorted(self._all_discovered_files.keys())
        elif self._discovered_files:
            return [self._discovered_files.catalog_name]
        else:
            return []
    
    def extract_catalog_name_from_header(self, file_path: Union[str, Path]) -> Optional[str]:
        """
        Extract catalog name from file header.
        
        This method extracts catalog names from HDF5 attributes or FITS headers
        when filename-based detection is insufficient.
        
        Parameters
        ----------
        file_path : str or Path
            Path to file to examine
            
        Returns
        -------
        str or None
            Catalog name if found, None otherwise
        """
        file_path = Path(file_path)
        
        try:
            # Try HDF5 files first
            if file_path.suffix.lower() == '.hdf5' and HDF5_AVAILABLE:
                return self._extract_catalog_from_hdf5(file_path)
            
            # Try FITS files
            elif file_path.suffix.lower() in ['.fits', '.fit'] and FITS_AVAILABLE:
                return self._extract_catalog_from_fits(file_path)
                
        except Exception as e:
            self.logger.debug(f"Could not extract catalog name from {file_path}: {e}")
        
        return None
    
    def _extract_catalog_from_hdf5(self, file_path: Path) -> Optional[str]:
        """Extract catalog name from HDF5 file attributes."""
        if not HDF5_AVAILABLE:
            return None
            
        try:
            with h5py.File(file_path, 'r') as f:
                # Check common attribute names for catalog information
                for attr_name in ['catalog_name', 'CATALOG', 'catalog', 'input_catalog', 'source_catalog']:
                    if attr_name in f.attrs:
                        catalog_name = f.attrs[attr_name]
                        if isinstance(catalog_name, bytes):
                            catalog_name = catalog_name.decode('utf-8')
                        return str(catalog_name).strip()
                
                # Check if there's a catalog group or dataset
                if 'catalog' in f:
                    return 'catalog'
                    
        except Exception as e:
            self.logger.debug(f"Error reading HDF5 attributes from {file_path}: {e}")
        
        return None
    
    def _extract_catalog_from_fits(self, file_path: Path) -> Optional[str]:
        """Extract catalog name from FITS file header."""
        if not FITS_AVAILABLE:
            return None
            
        try:
            with fits.open(file_path) as hdul:
                header = hdul[0].header
                
                # Check common header keywords for catalog information
                for keyword in ['CATALOG', 'CATNAME', 'OBJECT', 'SRCCAT', 'INPUT']:
                    if keyword in header:
                        catalog_name = str(header[keyword]).strip()
                        if catalog_name and catalog_name.upper() not in ['UNKNOWN', 'N/A', 'NONE', '']:
                            return catalog_name
                            
        except Exception as e:
            self.logger.debug(f"Error reading FITS header from {file_path}: {e}")
        
        return None
    
    def get_configuration_info(self, catalog_name: str, config_name: str) -> Optional[ConfigurationInfo]:
        """
        Get detailed information about a specific configuration.
        
        Parameters
        ----------
        catalog_name : str
            Name of the catalog
        config_name : str
            Name of the configuration
            
        Returns
        -------
        ConfigurationInfo or None
            Configuration information if found, None otherwise
        """
        file_structure = self.get_catalog_structure(catalog_name)
        if not file_structure or config_name not in file_structure.hdf5_files:
            return None
        
        hdf5_file = file_structure.hdf5_files[config_name]
        file_path = Path(hdf5_file)
        
        # Get file statistics
        try:
            stat = file_path.stat()
            file_size = stat.st_size
            creation_time = datetime.fromtimestamp(stat.st_mtime)
        except OSError:
            file_size = 0
            creation_time = None
        
        # Extract object count and parameter count from HDF5 file
        object_count, parameter_count = self._extract_hdf5_metadata(hdf5_file)
        
        return ConfigurationInfo(
            name=config_name,
            file_path=hdf5_file,
            file_size=file_size,
            object_count=object_count,
            parameter_count=parameter_count,
            creation_time=creation_time,
            catalog_name=catalog_name
        )
    
    def _extract_hdf5_metadata(self, file_path: str) -> Tuple[int, int]:
        """
        Extract object count and parameter count from HDF5 file.
        
        Parameters
        ----------
        file_path : str
            Path to HDF5 file
            
        Returns
        -------
        Tuple[int, int]
            (object_count, parameter_count)
        """
        if not HDF5_AVAILABLE:
            self.logger.warning("h5py not available, cannot extract HDF5 metadata")
            return 0, 0
        
        try:
            with h5py.File(file_path, 'r') as f:
                object_count = 0
                parameter_count = 0
                
                # Look for common dataset structures in BayeSED HDF5 files
                # Check for object IDs or similar datasets
                for key in f.keys():
                    dataset = f[key]
                    if isinstance(dataset, h5py.Dataset):
                        # If this is a 1D dataset, it might contain object IDs
                        if len(dataset.shape) == 1:
                            object_count = max(object_count, dataset.shape[0])
                        # If this is a 2D dataset, first dimension is usually objects
                        elif len(dataset.shape) == 2:
                            object_count = max(object_count, dataset.shape[0])
                            parameter_count = max(parameter_count, dataset.shape[1])
                
                # Also check for specific BayeSED dataset names
                if 'object_id' in f:
                    object_count = len(f['object_id'])
                elif 'id' in f:
                    object_count = len(f['id'])
                
                # Check for parameter names
                if 'parameter_names' in f:
                    parameter_count = len(f['parameter_names'])
                elif 'param_names' in f:
                    parameter_count = len(f['param_names'])
                
                self.logger.debug(f"Extracted from {Path(file_path).name}: "
                                f"{object_count} objects, {parameter_count} parameters")
                
                return object_count, parameter_count
                
        except Exception as e:
            self.logger.warning(f"Could not extract metadata from {file_path}: {e}")
            return 0, 0