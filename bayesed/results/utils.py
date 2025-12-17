"""
Utility functions for BayeSEDResults operations.

This module provides utility functions that work with the enhanced BayeSEDResults
implementation, including functions for file discovery and posterior comparison plotting.
"""

from typing import List, Optional, Dict, Any
from pathlib import Path
import logging

from .file_discovery import FileDiscovery
from .configuration_manager import ConfigurationManager
from .logger import get_logger


def list_catalog_names(output_dir: str) -> List[str]:
    """
    List available catalog names in an output directory.
    
    This function uses the enhanced FileDiscovery component to discover
    available catalogs in a BayeSED output directory.
    
    Parameters
    ----------
    output_dir : str
        Directory containing BayeSED output files
        
    Returns
    -------
    List[str]
        List of available catalog names
        
    Examples
    --------
    >>> catalogs = list_catalog_names('output')
    >>> print(f"Available catalogs: {catalogs}")
    """
    logger = get_logger('list_catalog_names').get_logger()
    file_discovery = FileDiscovery(logger)
    
    try:
        file_discovery.initialize(base_path=output_dir)
        file_structure = file_discovery.discover_files(output_dir)
        return file_discovery.list_catalogs()
    except Exception as e:
        logger.error(f"Failed to discover catalogs in {output_dir}: {e}")
        return []


def list_model_configs(output_dir: str, catalog_name: str) -> List[str]:
    """
    List available model configurations for a catalog.
    
    This function uses the enhanced ConfigurationManager component to discover
    available model configurations for a specific catalog.
    
    Parameters
    ----------
    output_dir : str
        Directory containing BayeSED output files
    catalog_name : str
        Catalog name to list configurations for
        
    Returns
    -------
    List[str]
        List of available model configuration names
        
    Examples
    --------
    >>> configs = list_model_configs('output', 'galaxies')
    >>> print(f"Available configurations: {configs}")
    """
    logger = get_logger('list_model_configs').get_logger()
    file_discovery = FileDiscovery(logger)
    config_manager = ConfigurationManager(logger)
    
    try:
        file_discovery.initialize(base_path=output_dir)
        file_structure = file_discovery.discover_files(output_dir)
        
        config_manager.initialize(
            file_structure=file_structure,
            catalog_name=catalog_name
        )
        
        return config_manager.list_configurations()
    except Exception as e:
        logger.error(f"Failed to discover configurations for {catalog_name} in {output_dir}: {e}")
        return []


def plot_posterior_comparison(
    results_list: List['BayeSEDResults'],
    labels: Optional[List[str]] = None,
    params: Optional[List[str]] = None,
    show: bool = True,
    output_file: Optional[str] = None,
    **kwargs: Any,
) -> Any:
    """
    Plot comparison of posterior samples from multiple BayeSEDResults objects.

    This function allows easy comparison of results from different
    model configurations, objects, or analysis runs.

    Parameters
    ----------
    results_list : list of BayeSEDResults
        List of BayeSEDResults objects to compare
    labels : list of str, optional
        Labels for each result set (default: 'Result 1', 'Result 2', etc.)
    params : list of str, optional
        Parameters to plot. If None (default), uses all common free parameters
        across all results. This excludes derived parameters and focuses on
        the fitted model parameters.
    show : bool, optional
        Whether to display the plot (default: True)
    output_file : str, optional
        Output file path for saving the plot.
    **kwargs
        Additional arguments passed to GetDist triangle_plot

    Returns
    -------
    getdist.plots.GetDistPlotter
        GetDist plotter object
    """
    if not results_list:
        raise ValueError("results_list cannot be empty")

    try:
        from getdist import plots
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("GetDist is required. Install with: pip install getdist")

    # Get GetDist samples for each result and collect available parameter names
    samples_list: List[Any] = []
    sample_param_sets: List[set] = []

    for i, result in enumerate(results_list):
        samples_gd = result.get_getdist_samples()

        # Set name tag for legend
        if labels and i < len(labels):
            label = labels[i]
        else:
            label = f'Result {i+1}'

        samples_gd.name_tag = label
        samples_gd.label = label
        if hasattr(samples_gd, 'name'):
            samples_gd.name = label

        samples_list.append(samples_gd)

        # Collect parameter names actually present in this MCSamples object
        try:
            param_names = [p.name for p in samples_gd.paramNames.names]
        except Exception:
            param_names = []
        sample_param_sets.append(set(param_names))

    # Determine parameters to plot
    if params is None:
        # Use intersection of free parameters across all results (default behavior)
        # Get free parameters from each result (these are already standardized if standardize_parameter_names was called)
        free_params_list: List[List[str]] = []
        for result in results_list:
            try:
                free_params = result.get_free_parameters()
                free_params_list.append(free_params)
            except Exception:
                # Fallback: use all parameters from GetDist samples if get_free_parameters fails
                free_params_list.append([p.name for p in result.get_getdist_samples().paramNames.names])
        
        # Find intersection of free parameters across all results
        if free_params_list:
            common_free_params_set = set(free_params_list[0])
            for free_params in free_params_list[1:]:
                common_free_params_set = common_free_params_set.intersection(set(free_params))
            
            # Also ensure these parameters are actually present in GetDist samples
            common_params_set = sample_param_sets[0]
            for param_set in sample_param_sets[1:]:
                common_params_set = common_params_set.intersection(param_set)
            
            # Intersection of free params and GetDist sample params
            final_params_set = common_free_params_set.intersection(common_params_set)
            
            # Additional check: filter out any parameters that might be derived
            # by checking if they're in the derived parameters list
            for result in results_list:
                try:
                    derived_params = set(result.get_derived_parameters())
                    # Remove any parameters that are derived
                    final_params_set = final_params_set - derived_params
                except Exception:
                    # If we can't get derived parameters, continue
                    pass
            
            # Preserve order from first result's free parameters
            params = [p for p in free_params_list[0] if p in final_params_set]
        else:
            # Fallback: use intersection of parameter names actually present in all samples
            first_sample_params = [p.name for p in samples_list[0].paramNames.names]
            common_params_set = sample_param_sets[0]
            for param_set in sample_param_sets[1:]:
                common_params_set = common_params_set.intersection(param_set)
            
            # Preserve order from first sample, only include common params
            params = [p for p in first_sample_params if p in common_params_set]

        if not params:
            raise ValueError(
                "No common free parameters found across all results' GetDist samples. "
                "Check that you have saved posterior samples for the same parameters "
                "in each run, or pass an explicit 'params' list that is valid."
            )
    else:
        # User provided params explicitly: filter to those present in all samples
        params_in_all: List[str] = []
        dropped: List[str] = []
        for p in params:
            if all(p in s for s in sample_param_sets):
                params_in_all.append(p)
            else:
                dropped.append(p)

        if not params_in_all:
            available = sorted(sample_param_sets[0].intersection(*sample_param_sets[1:]))
            raise ValueError(
                f"No provided parameters are present in all results' GetDist samples. "
                f"Requested: {params}. Common available parameters: {available}"
            )

        if dropped:
            import warnings

            warnings.warn(
                f"Some parameters were not present in all results and will be skipped: {dropped}",
                UserWarning,
            )

        params = params_in_all

    # Create plotter
    g = plots.get_subplot_plotter()

    # Set plotting options for better comparison visibility
    plot_kwargs = {
        'filled': True,
        'contour_colors': ['red', 'blue', 'green', 'orange', 'purple'],
        'contour_ls': ['-', '--', '-.', ':', '-'],
        'contour_lws': [1.5, 1.5, 1.5, 1.5, 1.5],
    }
    plot_kwargs.update(kwargs)

    # Use triangle_plot with samples list and params
    g.triangle_plot(samples_list, params, **plot_kwargs)

    if output_file:
        g.export(output_file)

    if show:
        try:
            plt.show()
        except Exception as e:
            import warnings

            warnings.warn(
                f"Could not display plot: {e}. "
                f"Try using output_file parameter to save the plot instead.",
                UserWarning,
            )

    return g


def standardize_parameter_names(
    results_list: List['BayeSEDResults'],
    standard_names: Optional[Dict[str, str]] = None,
    remove_component_ids: bool = True,
    custom_labels: Optional[Dict[str, str]] = None,
) -> None:
    """
    Standardize parameter names across multiple BayeSEDResults objects.

    This function renames parameters in all results to use consistent names,
    making comparison easier. It automatically detects equivalent parameters
    and renames them to a standard format.

    Parameters
    ----------
    results_list : list of BayeSEDResults
        List of BayeSEDResults objects to standardize
    standard_names : dict, optional
        Dictionary mapping normalized parameter names to standard names.
        If None, automatically generates clean parameter names.
    remove_component_ids : bool, optional
        If True (default), removes component IDs like [0,0], [0,1] from parameter names.
        This creates cleaner names like 'log(age/yr)' instead of 'log(age/yr)[0,0]'.
        If False, uses the first result's parameter names as the standard.
    custom_labels : dict, optional
        Dictionary mapping parameter names to custom LaTeX labels for plotting.
        Example: {'log(age/yr)': r'\\log(t/\\mathrm{yr})'}
    """
    if not results_list:
        return
    import re

    def normalize_param_name(param_name: str) -> str:
        """Remove component IDs like [0,0], [0,1] to find equivalent parameters."""
        return re.sub(r'\[\d+,\d+\]', '', param_name)

    # Collect original parameter names from GetDist samples for each result
    samples_param_names: List[List[str]] = []
    for result in results_list:
        samples = result.get_getdist_samples()
        try:
            names = [p.name for p in samples.paramNames.names]
        except Exception:
            names = []
        samples_param_names.append(names)

    # If no standard names provided, create them based on remove_component_ids setting
    if standard_names is None:
        if remove_component_ids:
            # Create clean parameter names without component IDs
            all_normalized_params: set[str] = set()
            for names in samples_param_names:
                for param in names:
                    all_normalized_params.add(normalize_param_name(param))

            # Use normalized names as the standard (clean names without [0,0], [0,1], etc.)
            standard_names = {norm_name: norm_name for norm_name in all_normalized_params}
        else:
            # Use the first result's parameter names as the standard (preserves component IDs)
            reference_names = samples_param_names[0] if samples_param_names else []
            standard_names = {normalize_param_name(p): p for p in reference_names}

    # Rename parameters in all results based on their GetDist names
    for result, names in zip(results_list, samples_param_names):
        mapping: Dict[str, str] = {}
        
        # Get paramnames file names for this result to handle * suffix for derived parameters
        paramnames_names = []
        try:
            # Access the paramnames file through the result's data loader
            if hasattr(result, '_data_loader') and hasattr(result._data_loader, '_file_structure'):
                file_structure = result._data_loader._file_structure
                # Get the object_id from the result's access scope
                object_id = None
                if hasattr(result, '_access_scope') and result._access_scope.is_object_level():
                    object_id = result._access_scope.object_filter
                
                # Get the configuration name
                config_name = None
                if hasattr(result, '_configuration_info') and result._configuration_info:
                    config_name = result._configuration_info.name
                
                if file_structure.posterior_files:
                    # Structure: posterior_files[object_id][base_name][file_type] = path
                    # Find the paramnames file for the current object and configuration
                    target_obj_id = object_id
                    if target_obj_id and target_obj_id in file_structure.posterior_files:
                        # Use the specific object_id
                        base_names_dict = file_structure.posterior_files[target_obj_id]
                        # base_names_dict is: base_name -> file_type -> path
                        for base_name, file_types in base_names_dict.items():
                            # Check if this matches the selected configuration
                            if config_name and config_name not in base_name:
                                continue
                            if 'paramnames' in file_types:
                                paramnames_file = file_types['paramnames']
                                import os
                                if os.path.exists(paramnames_file):
                                    with open(paramnames_file, 'r') as f:
                                        for line in f:
                                            if line.strip():
                                                paramnames_names.append(line.strip().split()[0])
                                    break
                        if not paramnames_names and base_names_dict:
                            # Fallback: use first available config for this object
                            first_base_name = list(base_names_dict.keys())[0]
                            first_file_types = base_names_dict[first_base_name]
                            if 'paramnames' in first_file_types:
                                paramnames_file = first_file_types['paramnames']
                                import os
                                if os.path.exists(paramnames_file):
                                    with open(paramnames_file, 'r') as f:
                                        for line in f:
                                            if line.strip():
                                                paramnames_names.append(line.strip().split()[0])
                    else:
                        # Fallback: use first object if object_id not specified or not found
                        for obj_id, base_names_dict in file_structure.posterior_files.items():
                            # base_names_dict is: base_name -> file_type -> path
                            for base_name, file_types in base_names_dict.items():
                                # Check if this matches the selected configuration
                                if config_name and config_name not in base_name:
                                    continue
                                if 'paramnames' in file_types:
                                    paramnames_file = file_types['paramnames']
                                    import os
                                    if os.path.exists(paramnames_file):
                                        with open(paramnames_file, 'r') as f:
                                            for line in f:
                                                if line.strip():
                                                    paramnames_names.append(line.strip().split()[0])
                                        break
                            if paramnames_names:
                                break
        except Exception as e:
            # If we can't read paramnames file, continue with just GetDist names
            import warnings
            warnings.warn(f"Could not read paramnames file for standardization: {e}")
            pass

        # Create mapping from GetDist names to normalized names
        for param in names:
            normalized = normalize_param_name(param)
            if normalized in standard_names:
                standard_param = standard_names[normalized]
                if param != standard_param:
                    mapping[param] = standard_param
                    
                    # Also add mapping for paramnames file version (with * if derived)
                    # Match paramnames names to GetDist names
                    for pname in paramnames_names:
                        pname_normalized = normalize_param_name(pname.rstrip('*'))
                        if pname_normalized == normalized:
                            # Add mapping for paramnames version
                            # Note: standard_param is already normalized (no indices, no *)
                            # For derived parameters, we map to standard name (without *)
                            # The * suffix is handled by rename_parameters() logic
                            mapping[pname] = standard_param
                            # Also add mapping without * for rename_parameters() lookup logic
                            # (rename_parameters checks param_name.rstrip('*') in mapping)
                            if pname.endswith('*'):
                                pname_base = pname.rstrip('*')
                                if pname_base not in mapping:
                                    mapping[pname_base] = standard_param
                            break  # Found match, move to next GetDist param

        if mapping:
            result.rename_parameters(mapping)

    # Apply custom labels if provided
    if custom_labels:
        for result in results_list:
            result.set_parameter_labels(custom_labels)