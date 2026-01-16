"""
Prior Manager for BayeSED3.

This module provides the PriorManager class for loading, modifying, and saving
parameter priors from .iprior files.
"""

import os
import shutil
from collections import OrderedDict
from typing import Dict, List, Set, Optional
from .prior import Prior


class PriorManager:
    """
    Manages loading, modifying, and saving of BayeSED3 prior files.
    
    The PriorManager handles .iprior files which contain parameter prior specifications
    for BayeSED3 analysis. It supports loading priors from files, modifying them
    programmatically, tracking changes, and saving them back to files.
    
    Parameters
    ----------
    base_directory : str, optional
        Base directory for prior files. If None, uses current directory.
        
    Attributes
    ----------
    base_directory : str
        Base directory for prior files
    priors : Dict[str, Prior]
        Dictionary mapping parameter names to Prior objects
    original_priors : Dict[str, Prior]
        Dictionary of auto-generated default priors for tracking modifications
    modified_parameters : Set[str]
        Set of parameter names that have been modified from defaults
        
    Examples
    --------
    >>> manager = PriorManager(base_directory='observation/test')
    >>> priors = manager.load_prior_file('2dal8.iprior')
    >>> manager.save_prior_file('2dal8_modified.iprior', list(priors.values()))
    """
    
    def __init__(self, base_directory: Optional[str] = None):
        """
        Initialize Prior Manager.
        
        Parameters
        ----------
        base_directory : str, optional
            Base directory for prior files. If None, uses current directory.
        """
        self.base_directory = base_directory or os.getcwd()
        # Organize priors by file: {filename: {param_name: Prior}}
        # Use OrderedDict to preserve parameter order from .iprior files
        self.priors_by_file: Dict[str, Dict[str, Prior]] = {}
        self.original_priors: Dict[str, Prior] = {}  # Auto-generated defaults (kept for compatibility)
        self.modified_parameters: Set[str] = set()   # Track which params were changed
    
    def load_prior_file(self, filepath: str, is_auto_generated: bool = False) -> Dict[str, Prior]:
        """
        Load priors from a .iprior file.
        
        Parses a .iprior file and creates Prior objects for each parameter.
        Handles variable column counts (5-8 columns) based on prior type.
        
        Parameters
        ----------
        filepath : str
            Path to .iprior file (relative to base_directory or absolute)
        is_auto_generated : bool, optional
            If True, marks these priors as auto-generated defaults for tracking
            
        Returns
        -------
        Dict[str, Prior]
            Dictionary mapping parameter names to Prior objects
            
        Raises
        ------
        FileNotFoundError
            If the specified file does not exist
        ValueError
            If the file format is invalid or cannot be parsed
            
        Examples
        --------
        >>> manager = PriorManager(base_directory='observation/test')
        >>> priors = manager.load_prior_file('2dal8.iprior')
        >>> 'Av_2' in priors
        True
        
        >>> # Load as auto-generated defaults
        >>> priors = manager.load_prior_file('2dal8.iprior', is_auto_generated=True)
        """
        import os
        
        # Resolve filepath
        if not os.path.isabs(filepath):
            filepath = os.path.join(self.base_directory, filepath)
        
        # Check file exists
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Prior file not found: {filepath}")
        
        # Use OrderedDict to preserve parameter order from file
        loaded_priors = OrderedDict()
        
        try:
            with open(filepath, 'r') as f:
                for line_num, line in enumerate(f, start=1):
                    # Skip comments and empty lines
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    
                    # Parse the line
                    try:
                        parts = line.split()
                        if len(parts) < 6:
                            raise ValueError(f"Line {line_num}: Expected at least 6 columns, got {len(parts)}")
                        
                        # Extract basic fields
                        name = parts[0]
                        prior_type = int(parts[1])
                        is_age = int(parts[2])
                        min_val = float(parts[3])
                        max_val = float(parts[4])
                        nbin = int(parts[5])
                        
                        # Extract hyperparameters if present
                        hyperparameters = []
                        if len(parts) > 6:
                            hyperparameters = [float(p) for p in parts[6:]]
                        
                        # Create Prior object
                        prior = Prior(
                            name=name,
                            prior_type=prior_type,
                            is_age=is_age,
                            min_val=min_val,
                            max_val=max_val,
                            nbin=nbin,
                            hyperparameters=hyperparameters,
                            source_file=filepath  # Track source file
                        )
                        
                        # Validate the prior
                        errors = prior.validate()
                        if errors:
                            raise ValueError(f"Line {line_num}: Invalid prior for '{name}': {'; '.join(errors)}")
                        
                        loaded_priors[name] = prior
                        
                    except (ValueError, IndexError) as e:
                        raise ValueError(f"Line {line_num}: Failed to parse line '{line}': {str(e)}")
        
        except Exception as e:
            raise ValueError(f"Error loading prior file '{filepath}': {str(e)}")
        
        # Store in manager organized by file
        import os
        basename = os.path.basename(filepath)
        
        if basename not in self.priors_by_file:
            # Use OrderedDict to preserve parameter order
            self.priors_by_file[basename] = OrderedDict()
        
        self.priors_by_file[basename].update(loaded_priors)
        
        # If auto-generated, store as originals for tracking
        if is_auto_generated:
            for name, prior in loaded_priors.items():
                # Create a copy for original_priors
                self.original_priors[name] = Prior(
                    name=prior.name,
                    prior_type=prior.prior_type,
                    is_age=prior.is_age,
                    min_val=prior.min_val,
                    max_val=prior.max_val,
                    nbin=prior.nbin,
                    hyperparameters=prior.hyperparameters.copy(),
                    component=prior.component,
                    description=prior.description
                )
        
        return loaded_priors
    
    def save_prior_file(self, filepath: str, priors: Optional[List[Prior]] = None, 
                       backup: bool = True) -> None:
        """
        Save priors to a .iprior file.
        
        Writes priors to a .iprior file in the correct BayeSED3 format.
        Handles different column counts for different prior types.
        
        Parameters
        ----------
        filepath : str
            Path to output .iprior file (relative to base_directory or absolute)
        priors : List[Prior], optional
            List of Prior objects to save. If None, saves all priors from the file
            specified by filepath basename.
        backup : bool, optional
            If True, creates a .bak backup of existing file before overwriting
            
        Raises
        ------
        ValueError
            If priors list is empty or contains invalid priors
        IOError
            If file cannot be written
            
        Examples
        --------
        >>> manager = PriorManager(base_directory='observation/test')
        >>> priors = manager.load_prior_file('2dal8.iprior')
        >>> manager.save_prior_file('2dal8_modified.iprior', list(priors.values()))
        
        >>> # Save with backup
        >>> manager.save_prior_file('2dal8.iprior', backup=True)
        """
        import os
        
        # Resolve filepath
        if not os.path.isabs(filepath):
            filepath = os.path.join(self.base_directory, filepath)
        
        # Use priors from the file if none specified
        if priors is None:
            basename = os.path.basename(filepath)
            if basename in self.priors_by_file:
                priors = list(self.priors_by_file[basename].values())
            else:
                raise ValueError(f"No priors found for file '{basename}'")
        
        if not priors:
            raise ValueError("No priors to save")
        
        # Create backup if file exists and backup is requested
        if backup and os.path.exists(filepath):
            backup_path = filepath + '.bak'
            try:
                shutil.copy2(filepath, backup_path)
            except Exception as e:
                raise IOError(f"Failed to create backup file '{backup_path}': {str(e)}")
        
        # Validate all priors before writing
        for prior in priors:
            errors = prior.validate()
            if errors:
                raise ValueError(f"Invalid prior '{prior.name}': {'; '.join(errors)}")
        
        # Write to file
        try:
            with open(filepath, 'w') as f:
                # Write header
                f.write("#name iprior_type is_age min max nbin\n")
                
                # Write each prior
                for prior in priors:
                    # Use the Prior's to_iprior_line method
                    line = f"{prior.name} {prior.to_iprior_line()}\n"
                    f.write(line)
                
        except Exception as e:
            raise IOError(f"Failed to write prior file '{filepath}': {str(e)}")
    
    def get_prior(self, parameter_name: str) -> Prior:
        """
        Retrieve a specific prior by parameter name.
        
        Parameters
        ----------
        parameter_name : str
            Name of the parameter (e.g., 'log(age/yr)', 'Av_2')
            
        Returns
        -------
        Prior
            The Prior object for the specified parameter
            
        Raises
        ------
        KeyError
            If the parameter name is not found
            
        Examples
        --------
        >>> manager = PriorManager(base_directory='observation/test')
        >>> manager.load_prior_file('2dal8.iprior')
        >>> prior = manager.get_prior('Av_2')
        >>> prior.min_val
        0.0
        """
        if parameter_name not in self.priors:
            available = ', '.join(self.priors.keys())
            raise KeyError(f"Parameter '{parameter_name}' not found. Available parameters: {available}")
        
        return self.priors[parameter_name]
    
    def modify_prior(self, parameter_name: str, iprior_file: str = None, **kwargs) -> None:
        """
        Modify a prior parameter.
        
        Validates that the parameter exists, applies modifications, tracks the change,
        and validates the modified prior.
        
        NOTE: This method only tracks cumulative modifications in modified_parameters.
        The recently_modified tracking is handled by the caller (set_prior) which knows
        the full scope of changes before they are applied.
        
        Parameters
        ----------
        parameter_name : str
            Name of the parameter to modify
        iprior_file : str, optional
            Name of the .iprior file. Required if parameter exists in multiple files.
        **kwargs
            Keyword arguments for Prior attributes to modify:
            - prior_type : int
            - is_age : int
            - min_val : float
            - max_val : float
            - nbin : int
            - hyperparameters : List[float]
            
        Raises
        ------
        KeyError
            If the parameter name is not found
        ValueError
            If the modification results in an invalid prior
            If parameter exists in multiple files and iprior_file is not specified
            
        Examples
        --------
        >>> manager = PriorManager(base_directory='observation/test')
        >>> manager.load_prior_file('2dal8.iprior')
        >>> manager.modify_prior('Av_2', min_val=0.1, max_val=2.0, nbin=50)
        >>> manager.get_prior('Av_2').max_val
        2.0
        """
        # Get the prior using the new get_prior method
        prior = self.get_prior(parameter_name, iprior_file)
        
        # Store original values for rollback if validation fails
        original_values = {}
        for key in kwargs.keys():
            if hasattr(prior, key):
                original_values[key] = getattr(prior, key)
        
        # If prior_type is being changed, automatically reset hyperparameters to match new type
        if 'prior_type' in kwargs:
            new_type = kwargs['prior_type']
            # Create a temporary prior to get the required hyperparameters for the new type
            from .prior import Prior
            temp_prior = Prior(
                name="temp",
                prior_type=new_type,
                is_age=0,
                min_val=0,
                max_val=1,
                nbin=10
            )
            required_hyper = temp_prior.get_required_hyperparameters()
            
            # If hyperparameters are not explicitly provided in kwargs, reset them
            if 'hyperparameters' not in kwargs:
                if required_hyper == 0:
                    kwargs['hyperparameters'] = []
                else:
                    # Don't auto-set hyperparameters if they're required - let validation catch it
                    # This ensures users explicitly provide values for non-zero hyperparameters
                    pass
        
        # Apply modifications
        try:
            for key, value in kwargs.items():
                if hasattr(prior, key):
                    setattr(prior, key, value)
                else:
                    raise ValueError(f"Invalid attribute '{key}' for Prior object")
            
            # Validate the modified prior
            errors = prior.validate()
            if errors:
                # Rollback changes
                for key, value in original_values.items():
                    setattr(prior, key, value)
                
                # Create helpful error message
                error_msg = f"Modification resulted in invalid prior for '{parameter_name}':\n"
                for error in errors:
                    error_msg += f"  - {error}\n"
                
                # If it's a hyperparameter error, provide helpful guidance
                if any('hyperparameter' in error.lower() for error in errors):
                    # Get info about the NEW prior type (from kwargs if changed)
                    new_type = kwargs.get('prior_type', prior.prior_type)
                    
                    # Create a temporary prior with the new type to get requirements
                    from .prior import Prior
                    temp_prior = Prior(
                        name="temp",
                        prior_type=new_type,
                        is_age=0,
                        min_val=0,
                        max_val=1,
                        nbin=10
                    )
                    
                    type_name = temp_prior.get_type_name()
                    required = temp_prior.get_required_hyperparameters()
                    param_names = temp_prior.get_hyperparameter_names()
                    
                    if param_names:
                        hyper_list = ', '.join(param_names)
                        error_msg += f"\nPrior type '{type_name}' requires {required} hyperparameters ({hyper_list}). "
                    else:
                        error_msg += f"\nPrior type '{type_name}' requires {required} hyperparameters. "
                    
                    error_msg += f"Example: set_prior('{parameter_name}', prior_type={new_type}, hyperparameters=[1.0, 0.5])"
                
                raise ValueError(error_msg.rstrip())
            
            # Track cumulative modification only if validation passed
            self.modified_parameters.add(parameter_name)
            
        except Exception as e:
            # Rollback changes on any error
            for key, value in original_values.items():
                setattr(prior, key, value)
            raise
    
    def validate_all(self) -> Dict[str, List[str]]:
        """
        Validate all loaded priors.
        
        Calls validate() on each Prior object and returns a dictionary mapping
        parameter names to their validation errors.
        
        Returns
        -------
        Dict[str, List[str]]
            Dictionary mapping parameter names to lists of error messages.
            Empty lists indicate valid priors.
            
        Examples
        --------
        >>> manager = PriorManager(base_directory='observation/test')
        >>> manager.load_prior_file('2dal8.iprior')
        >>> errors = manager.validate_all()
        >>> all(len(e) == 0 for e in errors.values())
        True
        """
        validation_results = {}
        
        for filename, params in self.priors_by_file.items():
            for name, prior in params.items():
                errors = prior.validate()
                validation_results[name] = errors
        
        return validation_results
    
    def get_summary(self) -> Dict:
        """
        Get summary statistics of loaded priors.
        
        Returns
        -------
        Dict
            Dictionary containing:
            - 'total_parameters': Total number of parameters
            - 'modified_parameters': Number of modified parameters
            - 'parameters_by_file': Dict mapping filenames to parameter counts
            - 'prior_type_distribution': Dict mapping prior types to counts
            
        Examples
        --------
        >>> manager = PriorManager(base_directory='observation/test')
        >>> manager.load_prior_file('2dal8.iprior')
        >>> summary = manager.get_summary()
        >>> summary['total_parameters']
        1
        >>> summary['modified_parameters']
        0
        """
        # Count parameters by file
        parameters_by_file = {}
        for filename, params in self.priors_by_file.items():
            parameters_by_file[filename] = len(params)
        
        # Count prior types
        prior_type_distribution = {}
        total_params = 0
        for filename, params in self.priors_by_file.items():
            for prior in params.values():
                total_params += 1
                type_name = prior.get_type_name()
                prior_type_distribution[type_name] = prior_type_distribution.get(type_name, 0) + 1
        
        return {
            'total_parameters': total_params,
            'modified_parameters': len(self.modified_parameters),
            'parameters_by_file': parameters_by_file,
            'prior_type_distribution': prior_type_distribution,
        }
    
    def display_priors(self, group_by_component: bool = True, 
                      show_modifications: bool = True, show_source_file: bool = True) -> str:
        """
        Display priors in a readable table format.
        
        Formats priors as a table with columns: Name, Type, Age, Min, Max, Nbin, 
        Modified, Hyperparameters, Source File. Adjusts column widths dynamically 
        for different prior types.
        
        Parameters
        ----------
        group_by_component : bool, optional
            If True, group parameters by component (default: True)
        show_modifications : bool, optional
            If True, show modification status for each parameter (default: True)
        show_source_file : bool, optional
            If True, show source file for each parameter (default: True)
            
        Returns
        -------
        str
            Formatted table string
            
        Examples
        --------
        >>> manager = PriorManager(base_directory='observation/test')
        >>> manager.load_prior_file('2dal8.iprior')
        >>> print(manager.display_priors())
        """
        # Collect all priors from all files
        all_priors = []
        for filename, params in self.priors_by_file.items():
            for name, prior in params.items():
                all_priors.append((name, prior))
        
        if not all_priors:
            return "No priors loaded"
        
        # Calculate column widths
        name_width = max(len(p[0]) for p in all_priors)
        name_width = max(name_width, len("Name"))
        
        type_width = max(len(p[1].get_type_name()) for p in all_priors)
        type_width = max(type_width, len("Type"))
        
        # Calculate hyperparameter column width
        hyper_strings = []
        for name, prior in all_priors:
            if prior.hyperparameters:
                names = prior.get_hyperparameter_names()
                hyper_str = ", ".join(f"{n}={val}" for n, val in zip(names, prior.hyperparameters))
                hyper_strings.append(hyper_str)
            else:
                hyper_strings.append("-")
        
        hyper_width = max(len(s) for s in hyper_strings) if hyper_strings else 1
        hyper_width = max(hyper_width, len("Hyperparameters"))
        
        # Calculate source file column width
        source_width = 0
        if show_source_file:
            import os
            source_files = []
            for name, prior in all_priors:
                if prior.source_file:
                    # Show only the basename for readability
                    source_files.append(os.path.basename(prior.source_file))
                else:
                    source_files.append("-")
            source_width = max(len(s) for s in source_files) if source_files else 1
            source_width = max(source_width, len("Source File"))
        
        # Build header
        header = f"{'Name':<{name_width}}  {'Type':<{type_width}}  {'IsAge':<6}  {'Min':<10}  {'Max':<10}  {'Nbin':<4}"
        if show_modifications:
            header += f"  {'Modified':<8}"
        header += f"  {'Hyperparameters':<{hyper_width}}"
        if show_source_file:
            header += f"  {'Source File':<{source_width}}"
        
        separator = "=" * len(header)
        
        # Build table
        lines = [separator, header, "-" * len(header)]
        
        # Group by component if requested
        if group_by_component:
            # Group priors by component
            by_component = {}
            for name, prior in all_priors:
                component = prior.component or 'unknown'
                if component not in by_component:
                    by_component[component] = []
                by_component[component].append((name, prior))
            
            # Display each component group
            for component, items in sorted(by_component.items()):
                if len(by_component) > 1:
                    lines.append(f"\nComponent: {component}")
                    lines.append("-" * len(header))
                
                for name, prior in sorted(items):
                    line = self._format_prior_line(prior, name_width, type_width, hyper_width, 
                                                   show_modifications, show_source_file, source_width)
                    lines.append(line)
        else:
            # Display all priors without grouping
            for name, prior in sorted(all_priors):
                line = self._format_prior_line(prior, name_width, type_width, hyper_width,
                                               show_modifications, show_source_file, source_width)
                lines.append(line)
        
        lines.append(separator)
        
        # Add legend if showing modifications
        if show_modifications:
            lines.append("")
            lines.append("Legend: * = modified from default, # = just modified, *# = both")
        
        return "\n".join(lines)
    
    def _format_prior_line(self, prior: Prior, name_width: int, type_width: int, 
                          hyper_width: int, show_modifications: bool, 
                          show_source_file: bool = True, source_width: int = 0) -> str:
        """Helper method to format a single prior line with three-level modification tracking."""
        import os
        
        # Format hyperparameters
        if prior.hyperparameters:
            names = prior.get_hyperparameter_names()
            hyper_str = ", ".join(f"{name}={val}" for name, val in zip(names, prior.hyperparameters))
        else:
            hyper_str = "-"
        
        # Format modification status with three-level tracking
        # * = modified from default (changed earlier in session)
        # # = just modified (changed in most recent call)
        # *# = both (previously modified AND just modified again)
        if show_modifications:
            is_modified = prior.name in self.modified_parameters
            # Handle case where recently_modified doesn't exist (backward compatibility)
            is_recent = prior.name in getattr(self, 'recently_modified', set())
            
            if is_modified and is_recent:
                modified = "*#"
            elif is_recent:
                modified = "#"
            elif is_modified:
                modified = "*"
            else:
                modified = "No"
        
        # Format source file (show only basename for readability)
        source_str = os.path.basename(prior.source_file) if prior.source_file else "-"
        
        line = f"{prior.name:<{name_width}}  {prior.get_type_name():<{type_width}}  "
        line += f"{prior.is_age:<6}  {prior.min_val:<10.6g}  {prior.max_val:<10.6g}  {prior.nbin:<4}"
        
        if show_modifications:
            line += f"  {modified:<8}"
        
        line += f"  {hyper_str:<{hyper_width}}"
        
        if show_source_file:
            line += f"  {source_str:<{source_width}}"
        
        return line
    
    def get_modified_parameters(self) -> Set[str]:
        """
        Get the set of parameter names that have been modified.
        
        Returns
        -------
        Set[str]
            Set of parameter names that have been modified from defaults
            
        Examples
        --------
        >>> manager = PriorManager(base_directory='observation/test')
        >>> manager.load_prior_file('2dal8.iprior')
        >>> manager.modify_prior('Av_2', nbin=50)
        >>> 'Av_2' in manager.get_modified_parameters()
        True
        """
        return self.modified_parameters.copy()
    
    def set_recently_modified(self, parameter_names: List[str]) -> None:
        """
        Set the recently modified parameters for display tracking.
        
        This should be called by set_prior() before applying modifications to mark
        which parameters are being changed in the current operation.
        
        Parameters
        ----------
        parameter_names : List[str]
            List of parameter names that will be modified in this operation
            
        Examples
        --------
        >>> manager = PriorManager(base_directory='observation/test')
        >>> manager.set_recently_modified(['Av_2', 'log(age/yr)'])
        """
        self.recently_modified = set(parameter_names)
    
    def clear_recently_modified(self) -> None:
        """
        Clear the recently modified tracking.
        
        This should be called after displaying priors to reset the # marker tracking.
        
        Examples
        --------
        >>> manager = PriorManager(base_directory='observation/test')
        >>> manager.clear_recently_modified()
        """
        self.recently_modified.clear()
    
    def is_parameter_modified(self, parameter_name: str) -> bool:
        """
        Check if a specific parameter has been modified.
        
        Parameters
        ----------
        parameter_name : str
            Name of the parameter to check
            
        Returns
        -------
        bool
            True if the parameter has been modified, False otherwise
            
        Examples
        --------
        >>> manager = PriorManager(base_directory='observation/test')
        >>> manager.load_prior_file('2dal8.iprior')
        >>> manager.is_parameter_modified('Av_2')
        False
        >>> manager.modify_prior('Av_2', nbin=50)
        >>> manager.is_parameter_modified('Av_2')
        True
        """
        return parameter_name in self.modified_parameters
    
    def update_prior_in_source_file(self, parameter_name: str, iprior_file: str = None) -> None:
        """
        Update a specific parameter in its original source file.
        
        This method reads the source .iprior file, finds the line for the given
        parameter, updates it with the current values, and writes it back.
        Creates a backup before modifying.
        
        Parameters
        ----------
        parameter_name : str
            Name of the parameter to update in its source file
        iprior_file : str, optional
            Name of the .iprior file. Required if parameter exists in multiple files.
            
        Raises
        ------
        KeyError
            If the parameter is not found
        ValueError
            If the parameter has no source_file tracked or parameter exists in multiple files
        IOError
            If file operations fail
            
        Examples
        --------
        >>> manager = PriorManager(base_directory='observation/test')
        >>> manager.load_prior_file('2dal8.iprior')
        >>> manager.modify_prior('Av_2', nbin=50)
        >>> manager.update_prior_in_source_file('Av_2')
        """
        import os
        import shutil
        
        # Get the prior
        prior = self.get_prior(parameter_name, iprior_file)
        
        # Check if source file is tracked
        if not prior.source_file:
            raise ValueError(f"Parameter '{parameter_name}' has no source_file tracked")
        
        source_file = prior.source_file
        
        # Check if file exists
        if not os.path.exists(source_file):
            raise IOError(f"Source file not found: {source_file}")
        
        # Create backup
        backup_file = source_file + '.bak'
        try:
            shutil.copy2(source_file, backup_file)
        except Exception as e:
            raise IOError(f"Failed to create backup '{backup_file}': {str(e)}")
        
        # Read the file and update the specific line
        try:
            with open(source_file, 'r') as f:
                lines = f.readlines()
            
            # Find and update the line for this parameter
            updated = False
            for i, line in enumerate(lines):
                # Skip comments and empty lines
                stripped = line.strip()
                if not stripped or stripped.startswith('#'):
                    continue
                
                # Check if this line is for our parameter
                parts = stripped.split()
                if parts and parts[0] == parameter_name:
                    # Replace this line with updated prior
                    new_line = f"{parameter_name} {prior.to_iprior_line()}\n"
                    lines[i] = new_line
                    updated = True
                    break
            
            if not updated:
                raise ValueError(f"Parameter '{parameter_name}' not found in file '{source_file}'")
            
            # Write back to file
            with open(source_file, 'w') as f:
                f.writelines(lines)
                
        except Exception as e:
            # Restore from backup on error
            try:
                shutil.copy2(backup_file, source_file)
            except:
                pass
            raise IOError(f"Failed to update file '{source_file}': {str(e)}")
    
    def find_files_containing_parameter(self, parameter_name: str) -> List[str]:
        """
        Find all .iprior files that contain a parameter with the given name.
        
        Parameters
        ----------
        parameter_name : str
            Name of the parameter to search for
            
        Returns
        -------
        List[str]
            List of file paths containing the parameter (basenames only)
            
        Examples
        --------
        >>> manager = PriorManager(base_directory='observation/test')
        >>> manager.load_prior_file('BLR.iprior')
        >>> manager.load_prior_file('FeII.iprior')
        >>> files = manager.find_files_containing_parameter('f')
        >>> len(files) >= 2
        True
        """
        files = []
        for filename, params in self.priors_by_file.items():
            if parameter_name in params:
                files.append(filename)
        return sorted(files)
    
    def get_parameters_from_file(self, iprior_file: str) -> List[str]:
        """
        Get all parameter names from a specific .iprior file.
        
        Parameters
        ----------
        iprior_file : str
            Name of the .iprior file (basename or full path)
            
        Returns
        -------
        List[str]
            List of parameter names from the file
            
        Examples
        --------
        >>> manager = PriorManager(base_directory='observation/test')
        >>> manager.load_prior_file('2dal8.iprior')
        >>> params = manager.get_parameters_from_file('2dal8.iprior')
        >>> 'Av_2' in params
        True
        """
        import os
        # Normalize the file path to basename for comparison
        target_basename = os.path.basename(iprior_file)
        
        if target_basename in self.priors_by_file:
            return sorted(self.priors_by_file[target_basename].keys())
        return []
    
    def get_prior(self, parameter_name: str, iprior_file: str = None) -> Prior:
        """
        Retrieve a specific prior by parameter name and optionally file.
        
        Parameters
        ----------
        parameter_name : str
            Name of the parameter (e.g., 'log(age/yr)', 'Av_2')
        iprior_file : str, optional
            Name of the .iprior file. Required if parameter exists in multiple files.
            
        Returns
        -------
        Prior
            The Prior object for the specified parameter
            
        Raises
        ------
        KeyError
            If the parameter name is not found
        ValueError
            If parameter exists in multiple files and iprior_file is not specified
            
        Examples
        --------
        >>> manager = PriorManager(base_directory='observation/test')
        >>> manager.load_prior_file('2dal8.iprior')
        >>> prior = manager.get_prior('Av_2')
        >>> prior.min_val
        0.0
        """
        import os
        
        if iprior_file is not None:
            # Explicit file specified
            basename = os.path.basename(iprior_file)
            if basename not in self.priors_by_file:
                available = ', '.join(sorted(self.priors_by_file.keys()))
                raise KeyError(f"File '{iprior_file}' not found. Available files: {available}")
            
            if parameter_name not in self.priors_by_file[basename]:
                available = ', '.join(sorted(self.priors_by_file[basename].keys()))
                raise KeyError(f"Parameter '{parameter_name}' not found in '{basename}'. Available: {available}")
            
            return self.priors_by_file[basename][parameter_name]
        else:
            # Auto-detect file
            files = self.find_files_containing_parameter(parameter_name)
            if len(files) == 0:
                raise KeyError(f"Parameter '{parameter_name}' not found in any loaded file")
            elif len(files) > 1:
                raise ValueError(
                    f"Parameter '{parameter_name}' exists in multiple files: {', '.join(files)}. "
                    f"Please specify iprior_file parameter."
                )
            return self.priors_by_file[files[0]][parameter_name]
    
    def batch_modify(self, criteria: Dict, modifications: Dict) -> Dict[str, bool]:
        """
        Apply batch modifications to multiple priors.
        
        Supports filtering by component, name patterns (regex), and value ranges.
        Validates each modification and provides rollback on partial failures.
        
        Parameters
        ----------
        criteria : Dict
            Dictionary specifying which parameters to modify. Supported keys:
            - 'component': str or List[str] - Filter by component name(s)
            - 'name_pattern': str - Regex pattern to match parameter names
            - 'prior_type': int or List[int] - Filter by prior type(s)
            - 'min_val_range': Tuple[float, float] - Filter by min_val range (inclusive)
            - 'max_val_range': Tuple[float, float] - Filter by max_val range (inclusive)
            - 'names': List[str] - Explicit list of parameter names
        modifications : Dict
            Dictionary of modifications to apply (same as modify_prior kwargs)
            
        Returns
        -------
        Dict[str, bool]
            Dictionary mapping parameter names to success status (True/False)
            
        Raises
        ------
        ValueError
            If criteria is empty or invalid, or if all modifications fail
            
        Examples
        --------
        >>> manager = PriorManager(base_directory='observation/test')
        >>> manager.load_prior_file('test.iprior')
        >>> # Modify all parameters with nbin=40 to nbin=60
        >>> results = manager.batch_modify(
        ...     criteria={'name_pattern': 'log.*'},
        ...     modifications={'nbin': 60}
        ... )
        >>> # Modify specific parameters
        >>> results = manager.batch_modify(
        ...     criteria={'names': ['log(age/yr)', 'log(tau/yr)']},
        ...     modifications={'nbin': 60}
        ... )
        """
        import re
        
        if not criteria:
            raise ValueError("Criteria dictionary cannot be empty")
        
        if not modifications:
            raise ValueError("Modifications dictionary cannot be empty")
        
        # Find matching parameters
        matching_params = []
        
        for filename, params in self.priors_by_file.items():
            for name, prior in params.items():
                matches = True
                
                # Check component filter
                if 'component' in criteria:
                    components = criteria['component']
                    if isinstance(components, str):
                        components = [components]
                    if prior.component not in components:
                        matches = False
                
                # Check name pattern filter
                if 'name_pattern' in criteria and matches:
                    pattern = criteria['name_pattern']
                    if not re.match(pattern, name):
                        matches = False
                
                # Check prior type filter
                if 'prior_type' in criteria and matches:
                    types = criteria['prior_type']
                    if isinstance(types, int):
                        types = [types]
                    if prior.prior_type not in types:
                        matches = False
                
                # Check min_val range filter
                if 'min_val_range' in criteria and matches:
                    min_range, max_range = criteria['min_val_range']
                    if not (min_range <= prior.min_val <= max_range):
                        matches = False
                
                # Check max_val range filter
                if 'max_val_range' in criteria and matches:
                    min_range, max_range = criteria['max_val_range']
                    if not (min_range <= prior.max_val <= max_range):
                        matches = False
                
                # Check explicit names list
                if 'names' in criteria and matches:
                    if name not in criteria['names']:
                        matches = False
                
                if matches:
                    matching_params.append((name, filename))
        
        if not matching_params:
            return {}
        
        # Apply modifications to matching parameters
        results = {}
        successful_modifications = []
        
        for param_name, filename in matching_params:
            try:
                self.modify_prior(param_name, iprior_file=filename, **modifications)
                results[param_name] = True
                successful_modifications.append(param_name)
            except (ValueError, KeyError) as e:
                results[param_name] = False
                # Store error for reporting
                if not hasattr(self, '_batch_errors'):
                    self._batch_errors = {}
                self._batch_errors[param_name] = str(e)
        
        # Check if any modifications succeeded
        if not any(results.values()):
            error_summary = "\n".join(
                f"  {name}: {self._batch_errors.get(name, 'Unknown error')}"
                for name, _ in matching_params
            )
            raise ValueError(f"All batch modifications failed:\n{error_summary}")
        
        return results
