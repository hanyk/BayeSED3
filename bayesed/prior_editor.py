"""
Interactive prior editing and table presentation.

Internal module (ADR-0006): set_prior / print_priors live here. Priors owns
lifecycle and thin-delegates after priors_init. Not part of the public
bayesed package export.
"""

from __future__ import annotations

import copy
import logging
import os
import re
from typing import Dict, Optional

from .prior import Prior
from .prior_manager import PriorManager


class PriorEditor:
    """Interactive edit/query/print for loaded priors (via PriorManager)."""

    def __init__(self, prior_manager: PriorManager):
        self._prior_manager = prior_manager

    def _mod_marker(self, param_name: str, modified_mode: str) -> str:
        if modified_mode == "preview":
            return "Preview"
        is_modified = param_name in self._prior_manager.modified_parameters
        is_recent = param_name in getattr(self._prior_manager, "recently_modified", set())
        if is_modified and is_recent:
            return "*#"
        if is_modified:
            return "*"
        if is_recent:
            return "#"
        return "No"

    def _hyper_str(self, prior: Prior) -> str:
        if prior.hyperparameters:
            names = prior.get_hyperparameter_names()
            return ", ".join(f"{n}={v}" for n, v in zip(names, prior.hyperparameters))
        return "-"

    def _print_prior_table(
        self,
        by_file: Dict[str, Dict[str, Prior]],
        *,
        modified_mode: str = "status",
        sort_names: bool = True,
        show_legend: Optional[bool] = None,
        print_file_headers: bool = True,
    ) -> None:
        """Print Name/Type/IsAge/Min/Max/Nbin/Modified/Hyperparameters tables.

        Canonical layout matches print_priors (always includes Hyperparameters).
        modified_mode: "status" (* / # / *# / No) or "preview" (Preview).
        """
        if modified_mode not in ("status", "preview"):
            raise ValueError(f"Unknown modified_mode: {modified_mode!r}")

        has_modifications = False
        for filename in sorted(by_file.keys()):
            params_dict = by_file[filename]
            if not params_dict:
                continue
            names = sorted(params_dict.keys()) if sort_names else list(params_dict.keys())
            priors = [params_dict[n] for n in names]

            if print_file_headers:
                print(f"\nFile: {filename}")
                print("-" * 80)

            name_width = max(max(len(n) for n in names), len("Name"))
            type_width = max(max(len(p.get_type_name()) for p in priors), len("Type"))
            isage_width = max(max(len(str(p.is_age)) for p in priors), len("IsAge"))
            min_width = max(max(len(f"{p.min_val:.6g}") for p in priors), len("Min"))
            max_width = max(max(len(f"{p.max_val:.6g}") for p in priors), len("Max"))
            nbin_width = max(max(len(str(p.nbin)) for p in priors), len("Nbin"))
            mod_markers = [self._mod_marker(n, modified_mode) for n in names]
            modified_width = max(max(len(m) for m in mod_markers), len("Modified"))
            hyper_strings = [self._hyper_str(p) for p in priors]
            hyper_width = max(max(len(s) for s in hyper_strings), len("Hyperparameters"))

            header = (
                f"{'Name':<{name_width}}  {'Type':<{type_width}}  "
                f"{'IsAge':<{isage_width}}  {'Min':<{min_width}}  {'Max':<{max_width}}  "
                f"{'Nbin':<{nbin_width}}  {'Modified':<{modified_width}}  "
                f"{'Hyperparameters':<{hyper_width}}"
            )
            print(header)
            print("-" * len(header))

            for name, prior, mod_marker, hyper_str in zip(names, priors, mod_markers, hyper_strings):
                if mod_marker not in ("No", "Preview"):
                    has_modifications = True
                min_str = f"{prior.min_val:.6g}"
                max_str = f"{prior.max_val:.6g}"
                # Display name: prefer dict key for preview copies that may share .name
                disp_name = name
                line = (
                    f"{disp_name:<{name_width}}  {prior.get_type_name():<{type_width}}  "
                    f"{prior.is_age:<{isage_width}}  {min_str:<{min_width}}  "
                    f"{max_str:<{max_width}}  {prior.nbin:<{nbin_width}}  "
                    f"{mod_marker:<{modified_width}}  {hyper_str:<{hyper_width}}"
                )
                print(line)

        if show_legend is None:
            show_legend = modified_mode == "status" and has_modifications
        if show_legend:
            print("\nLegend: * = modified from default, # = just modified, *# = both")

    def set_prior(self, parameter_name=None, iprior_file=None, prior_type=None, is_age=None,
                  min_val=None, max_val=None, nbin=None, hyperparameters=None, confirm=True,
                  reset_to_default=False, component_id=None) -> None:
        """
        Modify prior parameters or query information about loaded priors.
        
        This method supports three modes of operation for modifications:
        1. **Single/multiple parameters (auto-detect file)**: Specify parameter_name(s) only
        2. **Single/multiple parameters (explicit file)**: Specify parameter_name(s) and iprior_file
        3. **All parameters in a file**: Specify only iprior_file (applies to all parameters)
        
        When called without modification parameters, it provides informational queries:
        - No arguments: List all .iprior files
        - Only iprior_file: Display the file content (like print_priors)
        - Only parameter_name: Show information about the parameter
        
        **Pattern Matching for Parameter Selection:**
        
        When parameter_name is a string (not a list), the method uses intelligent pattern 
        matching with three-tier auto-detection:
        
        1. **Exact match** (highest priority): Matches parameter name exactly
           - No confirmation required
           - Example: 'log(age/yr)' matches only 'log(age/yr)'
        
        2. **Regex match** (if pattern looks like regex): Treats as regular expression
           - Detected by special characters: .*, .+, ?, [, ], ^, $, |, {, }, \\, +
           - Requires user confirmation for multiple matches
           - Example: 'log.*' matches 'log(age/yr)', 'log(tau/yr)', 'log(Z/Zsun)'
           - Example: '^Av_.*' matches 'Av_1', 'Av_2', 'Av_young', 'Av_old'
        
        3. **Partial match** (substring fallback): Matches any parameter containing the pattern
           - Requires user confirmation for multiple matches
           - Example: 'age' matches 'log(age/yr)', 'age_young', 'stellar_age'
        
        For regex or partial matches affecting multiple parameters, the method displays:
        - All matched parameters grouped by file
        - Modifications to be applied
        - Confirmation prompt: "Apply these modifications to N parameter(s)? [y/N]: "
        
        If no matches are found, a descriptive error is raised with suggestions.
        
        The iprior_file parameter is optional but required when a parameter name exists in
        multiple files. When parameter_name is not provided with modifications, changes apply 
        to all parameters in the specified file.
        
        Automatically validates that the correct number of hyperparameters is provided
        for the specified prior type. Updates parameters in their original source .iprior files,
        creating backups (.bak) before modifying.
        
        Parameters
        ----------
        parameter_name : str or list of str, optional
            Name of the parameter(s) to modify (e.g., 'log(age/yr)') or list of parameter names.
            If None, applies modifications to ALL parameters in the specified iprior_file.
        iprior_file : str, optional
            The .iprior file containing the parameter(s). Can be just the filename 
            (e.g., '2dal8.iprior') or full path. Required when:
            - parameter_name exists in multiple files (for disambiguation)
            - parameter_name is None (to specify which file to modify)
            If not provided with parameter_name, the method will auto-detect and raise 
            an error if ambiguous.
        prior_type : int or str, optional
            Prior type as integer (-9 to 9) or string name (e.g., 'Gaussian', 'Linear-Inc').
            Negative integers indicate log10 space. For string names, prefix with 'Log10_' 
            for log10 space (e.g., 'Log10_Gaussian').
            Valid string names: 'Mirror', 'Linear-Inc', 'Linear-Dec', 'Uniform', 
            'TruncGaussian', 'Gaussian', 'Gamma', 'StudentT', 'Beta', 'Weibull'
        is_age : int, optional
            Age-dependent flag (0 or 1)
        min_val : float, optional
            Minimum value
        max_val : float, optional
            Maximum value
        nbin : int, optional
            Number of bins
        hyperparameters : list of float, optional
            Hyperparameters for the prior distribution (e.g., [mu, sigma] for Gaussian).
            The number of hyperparameters is validated against the prior type requirements.
        confirm : bool, optional
            Whether to ask for user confirmation before applying modifications.
            Default is True (always ask for confirmation).
            Set to False to skip confirmation (useful for scripting/automation).
        component_id : int, optional
            Filter by model component ID (from .iprior.{N} filename suffix).
            When set, only parameters from files with matching .iprior.{N} suffix
            are considered. Useful for multi-component models where the same
            parameter name exists across multiple instances.
            Example: set_prior('Av_2', component_id=0, min_val=0.1)
        reset_to_default : bool, optional
            If True, reset the parameter(s) to their auto-generated default values.
            Default is False. When True, all other modification parameters are ignored.
            This is useful for reverting customizations back to defaults.
            Works with all three modes (single parameter, explicit file, bulk).
            
        Raises
        ------
        ValueError
            If neither parameter_name nor iprior_file is specified
            If the parameter doesn't exist in any loaded file
            If the parameter exists in multiple files and iprior_file is not specified
            If the parameter doesn't exist in the specified iprior_file
            If the number of hyperparameters doesn't match the prior type requirements
            If the prior type string is not recognized
            If reset_to_default=True but no default values are available
        FileNotFoundError
            If the specified .iprior file doesn't exist or wasn't loaded
            
        Examples
        --------
        # Informational queries (no modifications)
        bayesed.set_prior()  # List all .iprior files
        bayesed.set_prior(iprior_file='BLR.iprior')  # Display file content
        bayesed.set_prior('f')  # Show info about parameter 'f'
        
        # Mode 1: Simple case - parameter name is unique across all files
        # Shows preview and requires confirmation (unless confirm=False)
        bayesed.set_prior('log(age/yr)', min_val=8.0, max_val=10.0)
        # Output:
        # ================================================================================
        # Exact match: Modifying 1 parameter:
        # ================================================================================
        # 
        # File: csp_sfh200_bc2003_hr_stelib_chab_neb_2000r_i0100.iprior
        # --------------------------------------------------------------------------------
        # Name         Type     IsAge  Min        Max        Nbin  Modified  Hyperparameters
        # --------------------------------------------------------------------------------
        # log(age/yr)  Uniform  1      8.0        10.0       40    Preview   -
        # 
        # Note: 'Preview' shows how parameters will look after modifications are applied.
        #       No changes have been made yet - confirmation required.
        # 
        # Apply these modifications to 1 parameter(s)? [y/N]: y
        
        # Mode 2: Ambiguous parameter - must specify file
        # This will raise: "Parameter 'f' exists in multiple files: BLR.iprior, FeII.iprior, AGN.iprior. 
        #                   Please specify iprior_file parameter."
        # bayesed.set_prior('f', min_val=0.1, max_val=0.9)  # ERROR!
        
        # Correct way to handle ambiguous parameter:
        bayesed.set_prior('f', iprior_file='BLR.iprior', min_val=0.1, max_val=0.9)
        
        # Mode 3: Bulk operation - modify ALL parameters in a file
        # Set all parameters in 2dal8.iprior to use Gaussian priors
        bayesed.set_prior(iprior_file='2dal8.iprior', prior_type='Gaussian', 
                          hyperparameters=[1.0, 0.3])
        
        # Increase resolution for all parameters in a specific file
        bayesed.set_prior(iprior_file='csp_sfh200_bc2003_hr_stelib_chab_neb_2000r_i0100.iprior',
                          nbin=60)
        
        # Explicit file specification with specific parameters
        bayesed.set_prior('Av_2', iprior_file='2dal8.iprior', 
                          prior_type='Gaussian', min_val=0.0, max_val=4.0,
                          hyperparameters=[1.0, 0.3])
        
        # Set same modifications for multiple parameters in the same file
        bayesed.set_prior(['log(age/yr)', 'log(tau/yr)'], 
                          iprior_file='csp_sfh200_bc2003_hr_stelib_chab_neb_2000r_i0100.iprior',
                          nbin=60)
        
        # Pattern Matching Examples
        # --------------------------
        
        # Exact match (requires confirmation by default)
        bayesed.set_prior('log(age/yr)', min_val=8.0, max_val=10.0)
        # Shows preview table with 'Preview' in Modified column
        # Asks: "Apply these modifications to 1 parameter(s)? [y/N]:"
        
        # Regex match (requires confirmation for multiple matches)
        bayesed.set_prior('log.*', nbin=60)
        # Matches: 'log(age/yr)', 'log(tau/yr)', 'log(Z/Zsun)', etc.
        # Displays:
        # ================================================================================
        # Pattern matched 3 parameter(s) using regex matching:
        # ================================================================================
        # 
        # File: csp_sfh200_bc2003_hr_stelib_chab_neb_2000r_i0100.iprior
        # --------------------------------------------------------------------------------
        # Name         Type     IsAge  Min        Max        Nbin  Modified  Hyperparameters
        # --------------------------------------------------------------------------------
        # log(age/yr)  Uniform  1      8.0        10.1284    60    Preview   -
        # log(tau/yr)  Uniform  1      8.0        10.1284    60    Preview   -
        # log(Z/Zsun)  Uniform  0      -2.0       0.5        60    Preview   -
        # 
        # Note: 'Preview' shows how parameters will look after modifications are applied.
        #       No changes have been made yet - confirmation required.
        # 
        # Apply these modifications to 3 parameter(s)? [y/N]: y
        
        # Regex with anchors (match parameters starting with 'Av_')
        bayesed.set_prior('^Av_.*', prior_type='Gaussian', hyperparameters=[1.0, 0.3])
        # Matches: 'Av_1', 'Av_2', 'Av_young', 'Av_old', etc.
        
        # Partial match (substring, requires confirmation)
        bayesed.set_prior('age', min_val=8.0, max_val=10.0)
        # Matches any parameter containing 'age': 'log(age/yr)', 'age_young', 'stellar_age'
        # User must confirm before applying
        
        # Pattern matching with file scope (limit to specific file)
        bayesed.set_prior('Av.*', iprior_file='2dal8.iprior', min_val=0.0, max_val=4.0)
        # Only matches parameters in 2dal8.iprior that match the pattern
        
        # No matches (raises error with helpful message)
        # bayesed.set_prior('nonexistent_param', min_val=0.0)
        # Raises: ValueError: No parameters found matching 'nonexistent_param'
        #         Available parameters: log(age/yr), log(tau/yr), ...
        
        # Informational Query Pattern Matching Examples
        # ----------------------------------------------
        
        # Query by exact parameter name
        bayesed.set_prior('log(age/yr)')
        # Output:
        # Pattern 'log(age/yr)' matched 1 parameter(s) using exact matching:
        # ================================================================================
        # 
        # File: csp_sfh200_bc2003_hr_stelib_chab_neb_2000r_i0100.iprior
        # --------------------------------------------------------------------------------
        # Name         Type     IsAge  Min        Max        Nbin  Modified  Hyperparameters
        # --------------------------------------------------------------------------------
        # log(age/yr)  Uniform  1      8.0        10.1284    40    No        -
        # 
        # Legend: * = modified from default, # = just modified, *# = both
        # ================================================================================
        
        # Query by partial match - finds parameters containing 'age'
        bayesed.set_prior('age')
        # Output:
        # Pattern 'age' matched 2 parameter(s) using partial matching:
        # ================================================================================
        # 
        # File: csp_sfh200_bc2003_hr_stelib_chab_neb_2000r_i0000.iprior
        #   [ ] log(age/yr): Uniform, Range=[8.0, 10.1284], Nbin=40
        # 
        # File: csp_sfh200_bc2003_hr_stelib_chab_neb_2000r_i0100.iprior
        #   [ ] log(age/yr): Uniform, Range=[8.0, 10.1284], Nbin=40
        # 
        # Legend: [*] = modified from default
        
        # Query by regex - finds all parameters starting with 'log'
        bayesed.set_prior('log.*')
        # Matches: log(age/yr), log(tau/yr), log(Z/Zsun), log(M_BH/M_sun), etc.
        
        # Query with file scope - limit to specific file
        bayesed.set_prior('f', iprior_file='BLR.iprior')
        # Shows info only for 'f' in BLR.iprior
        
        # No match query - shows helpful error
        # bayesed.set_prior('nonexistent_xyz')
        # Raises: KeyError: "No parameters match pattern 'nonexistent_xyz'. 
        #         Available parameters: Av_2, cos_inc, f, log(age/yr), ..."
        
        # Skip Confirmation (for scripting/automation)
        # --------------------------------------------
        
        # Bulk modification without confirmation
        bayesed.set_prior(iprior_file='2dal8.iprior', nbin=60, confirm=False)
        # Applies changes immediately without asking for confirmation
        
        # Pattern matching without confirmation
        bayesed.set_prior('Av.*', min_val=0.0, max_val=4.0, confirm=False)
        # Modifies all matching parameters without confirmation
        
        # Default behavior (with confirmation)
        bayesed.set_prior(iprior_file='2dal8.iprior', nbin=60)
        # Shows preview and asks: "Apply these modifications to N parameter(s)? [y/N]:"
        
        # Reset to Default Values
        # -----------------------
        
        # Reset a single parameter to its auto-generated default
        bayesed.set_prior('log(age/yr)', reset_to_default=True)
        # Reverts 'log(age/yr)' back to the default values from priors_init()
        
        # Reset all parameters in a file to defaults
        bayesed.set_prior(iprior_file='2dal8.iprior', reset_to_default=True)
        # Reverts all parameters in 2dal8.iprior to their defaults
        
        # Reset with pattern matching
        bayesed.set_prior('Av.*', reset_to_default=True)
        # Resets all parameters matching 'Av.*' pattern to defaults
        # Requires confirmation for multiple matches
        
        # Reset without confirmation (for automation)
        bayesed.set_prior('log(age/yr)', reset_to_default=True, confirm=False)
        # Resets immediately without asking for confirmation
        
        # When reset_to_default=True, all other modification parameters are ignored
        bayesed.set_prior('log(age/yr)', reset_to_default=True, nbin=60)
        # The nbin=60 is ignored; parameter is reset to default
        
        Note: Each parameter is updated in its source .iprior file with a backup created.
        """
        
        # Helper function for pattern matching
        def find_matching_parameters(pattern: str, iprior_file: str = None) -> tuple:
            """
            Find parameters using auto-detection: exact → regex → partial.
            
            Parameters
            ----------
            pattern : str
                Pattern to match against parameter names
            iprior_file : str, optional
                Limit search to specific file
                
            Returns
            -------
            tuple
                (match_type, matches) where:
                - match_type: 'exact', 'regex', or 'partial'
                - matches: list of tuples [(param_name, filename), ...]
            """
            # Get all parameters to search
            if iprior_file:
                # Search only in specified file
                basename = os.path.basename(iprior_file)
                if basename not in self._prior_manager.priors_by_file:
                    return ('exact', [])
                all_params = [(name, basename) for name in self._prior_manager.priors_by_file[basename].keys()]
            else:
                # Search across all files
                all_params = []
                for filename, params in self._prior_manager.priors_by_file.items():
                    for name in params.keys():
                        all_params.append((name, filename))

            # Filter by component_id if specified
            if component_id is not None:
                all_params = [(n, f) for n, f in all_params
                              if self._prior_manager.priors_by_file[f][n].component_id == component_id]

            # Step 1: Try exact match first
            exact_matches = [(name, fname) for name, fname in all_params if name == pattern]
            if exact_matches:
                return ('exact', exact_matches)
            
            # Step 2: Try regex if pattern looks like regex
            # Regex indicators: .*, .+, ?, [, ], ^, $, |, {, }, \, +
            regex_chars = [r'.*', r'.+', '?', '[', ']', '^', '$', '|', '{', '}', '\\', '+']
            looks_like_regex = any(char in pattern for char in regex_chars)
            
            if looks_like_regex:
                try:
                    regex = re.compile(pattern)
                    regex_matches = [(name, fname) for name, fname in all_params if regex.search(name)]
                    if regex_matches:
                        return ('regex', regex_matches)
                except re.error:
                    # Invalid regex, continue to partial match
                    pass
            
            # Step 3: Try partial match (substring)
            partial_matches = [(name, fname) for name, fname in all_params if pattern in name]
            if partial_matches:
                return ('partial', partial_matches)
            
            # No matches found
            return ('none', [])
        
        # Helper function for user confirmation
        def confirm_multiple_parameter_modification(match_type: str, matches: list, modifications: dict) -> bool:
            """
            Ask user to confirm modification of multiple parameters.
            
            Parameters
            ----------
            match_type : str
                Type of match: 'regex', 'partial', or 'bulk'
            matches : list
                List of tuples [(param_name, filename), ...]
            modifications : dict
                Dictionary of modifications to apply
                
            Returns
            -------
            bool
                True if user confirms, False otherwise
            """
            print(f"\n{'='*80}")
            if match_type == 'bulk':
                print(f"Bulk modification: {len(matches)} parameter(s) in file:")
            elif match_type == 'exact':
                if len(matches) == 1:
                    print(f"Exact match: Modifying 1 parameter:")
                else:
                    print(f"Exact match: Modifying {len(matches)} parameter(s):")
            else:
                print(f"Pattern matched {len(matches)} parameter(s) using {match_type} matching:")
            print(f"{'='*80}")
            
            # Group by file for display
            by_file = {}
            for param_name, filename in matches:
                if filename not in by_file:
                    by_file[filename] = []
                by_file[filename].append(param_name)
            
            # Preview table (unified formatter, ADR-0006 D2)
            preview_by_file = {}
            for filename in sorted(by_file.keys()):
                file_priors = self._prior_manager.priors_by_file[filename]
                matched_params = by_file[filename]
                filtered_priors = {name: file_priors[name] for name in matched_params if name in file_priors}
                preview_priors = {}
                for param_name, prior in filtered_priors.items():
                    preview = copy.copy(prior)
                    for key, value in modifications.items():
                        if hasattr(preview, key):
                            setattr(preview, key, value)
                    preview_priors[param_name] = preview
                preview_by_file[filename] = preview_priors
            self._print_prior_table(preview_by_file, modified_mode="preview", show_legend=False)

            print(f"{'='*80}")
            
            # Check if we're changing prior_type without hyperparameters when they're required
            if 'prior_type' in modifications and 'hyperparameters' not in modifications:
                new_type = modifications['prior_type']
                temp_prior = Prior(name="temp", prior_type=new_type, is_age=0, min_val=0, max_val=1, nbin=10)
                required_hyper = temp_prior.get_required_hyperparameters()
                
                if required_hyper > 0:
                    # Check if any matched parameters will have issues
                    type_name = temp_prior.get_type_name()
                    param_names = temp_prior.get_hyperparameter_names()
                    print(f"\n⚠️  WARNING: Prior type '{type_name}' requires {required_hyper} hyperparameters ({', '.join(param_names) if param_names else ''})")
                    print(f"   You haven't provided hyperparameters. This will cause an error.")
                    return False
            
            # Ask for confirmation
            response = input(f"\nApply these modifications to {len(matches)} parameter(s)? [y/N]: ").strip().lower()
            return response in ['y', 'yes']
        
        # Handle reset_to_default mode
        if reset_to_default:
            # Validate that we have default values available
            if not self._prior_manager.original_priors:
                raise ValueError(
                    "No default prior values available. "
                    "Defaults are only available when priors were loaded with is_auto_generated=True."
                )
            
            # Determine which parameters to reset
            param_names_to_reset = []
            
            if parameter_name is None:
                # Mode 3: Reset ALL parameters in the specified file
                if iprior_file is None:
                    raise ValueError(
                        "Must specify either parameter_name or iprior_file when reset_to_default=True.\n"
                        "Examples:\n"
                        "  - set_prior('log(age/yr)', reset_to_default=True)  # Reset single parameter\n"
                        "  - set_prior(iprior_file='2dal8.iprior', reset_to_default=True)  # Reset all in file"
                    )
                
                param_names_to_reset = self._prior_manager.get_parameters_from_file(iprior_file)
                if not param_names_to_reset:
                    raise FileNotFoundError(
                        f"No parameters found in file '{iprior_file}'. "
                        f"Make sure the file was loaded with priors_init()."
                    )
            else:
                # Mode 1 or 2: Reset specific parameter(s) with pattern matching support
                param_names = [parameter_name] if isinstance(parameter_name, str) else parameter_name
                
                for param_name in param_names:
                    if isinstance(param_name, str):
                        match_type, matches = find_matching_parameters(param_name, iprior_file)
                        
                        if match_type == 'none':
                            # No matches found
                            all_params = set()
                            for filename, params in self._prior_manager.priors_by_file.items():
                                all_params.update(params.keys())
                            available = ', '.join(sorted(all_params))
                            raise KeyError(
                                f"No parameters match pattern '{param_name}'. "
                                f"Available parameters: {available}"
                            )
                        
                        elif match_type == 'exact':
                            # Exact match
                            if len(matches) == 1:
                                param_names_to_reset.append(matches[0][0])
                            else:
                                # Multiple files have this exact parameter
                                if iprior_file is None:
                                    files = [fname for _, fname in matches]
                                    raise ValueError(
                                        f"Parameter '{param_name}' exists in multiple files: {', '.join(files)}. "
                                        f"Please specify iprior_file parameter.\n"
                                        f"Example: set_prior('{param_name}', iprior_file='{files[0]}', reset_to_default=True)"
                                    )
                                else:
                                    param_names_to_reset.append(param_name)
                        
                        elif match_type in ['regex', 'partial']:
                            # Multiple matches - add all matched parameters
                            for matched_param, _ in matches:
                                param_names_to_reset.append(matched_param)
                    else:
                        param_names_to_reset.append(param_name)
            
            # Check that all parameters have defaults available
            missing_defaults = []
            for param_name in param_names_to_reset:
                if param_name not in self._prior_manager.original_priors:
                    missing_defaults.append(param_name)
            
            if missing_defaults:
                raise ValueError(
                    f"No default values available for parameter(s): {', '.join(missing_defaults)}. "
                    f"Defaults are only available for parameters loaded with is_auto_generated=True."
                )
            
            # Show preview of reset if confirm=True
            if confirm:
                # Build matches list for preview display
                matches_for_preview = []
                for param_name in param_names_to_reset:
                    files = self._prior_manager.find_files_containing_parameter(param_name)
                    if len(files) == 1:
                        matches_for_preview.append((param_name, files[0]))
                    elif iprior_file:
                        matches_for_preview.append((param_name, os.path.basename(iprior_file)))
                
                # Create a modifications dict with default values for preview
                # We'll use the confirm_multiple_parameter_modification function but with default values
                # First, we need to show the preview with default values
                print(f"\n{'='*80}")
                if parameter_name is None:
                    print(f"Reset to default: {len(param_names_to_reset)} parameter(s) in file:")
                else:
                    if len(matches_for_preview) == 1:
                        print(f"Reset to default: 1 parameter:")
                    else:
                        match_type_for_display = 'exact' if all(p == parameter_name for p in param_names_to_reset) else match_type
                        print(f"Reset to default: {len(param_names_to_reset)} parameter(s) using {match_type_for_display} matching:")
                print(f"{'='*80}")
                
                # Group by file for display
                by_file = {}
                for param_name, filename in matches_for_preview:
                    if filename not in by_file:
                        by_file[filename] = []
                    by_file[filename].append(param_name)
                
                # Preview defaults (unified formatter)
                preview_by_file = {}
                for filename in sorted(by_file.keys()):
                    matched_params = by_file[filename]
                    preview_by_file[filename] = {
                        param_name: self._prior_manager.original_priors[param_name]
                        for param_name in matched_params
                        if param_name in self._prior_manager.original_priors
                    }
                self._print_prior_table(preview_by_file, modified_mode="preview", show_legend=False)

                response = input(f"Reset {len(param_names_to_reset)} parameter(s) to defaults? [y/N]: ").strip().lower()
                if response not in ['y', 'yes']:
                    print("\nReset cancelled.")
                    return
            
            # Set recently_modified tracking
            self._prior_manager.set_recently_modified(param_names_to_reset)
            
            # Track which files were modified
            modified_files = set()
            
            # Reset each parameter to its default
            for param_name in param_names_to_reset:
                # Get the default prior
                default_prior = self._prior_manager.original_priors[param_name]
                
                # Determine which file this parameter is in
                files = self._prior_manager.find_files_containing_parameter(param_name)
                if len(files) == 1:
                    target_file = files[0]
                elif iprior_file:
                    target_file = os.path.basename(iprior_file)
                else:
                    target_file = None
                
                if target_file:
                    modified_files.add(target_file)
                
                # Apply default values
                self._prior_manager.modify_prior(
                    param_name,
                    iprior_file=target_file,
                    prior_type=default_prior.prior_type,
                    is_age=default_prior.is_age,
                    min_val=default_prior.min_val,
                    max_val=default_prior.max_val,
                    nbin=default_prior.nbin,
                    hyperparameters=default_prior.hyperparameters.copy()
                )
                
                # Update the parameter in its source file
                try:
                    self._prior_manager.update_prior_in_source_file(param_name, iprior_file=target_file)
                except Exception as e:
                    logging.getLogger(__name__).warning(
                        f"Failed to update '{param_name}' in source file: {e}. "
                        f"Reset is saved in memory but not persisted to disk."
                    )
                
                # Remove from modified_parameters since it's back to default
                self._prior_manager.modified_parameters.discard(param_name)
            
            # Display the reset parameters
            print()
            if len(modified_files) == 1:
                self.print_priors(iprior_file=list(modified_files)[0], parameter_names=param_names_to_reset)
            else:
                self.print_priors(parameter_names=param_names_to_reset)
            
            # Clear recently_modified after display
            self._prior_manager.clear_recently_modified()
            
            print(f"\n✓ Reset {len(param_names_to_reset)} parameter(s) to default values")
            return
        
        # Convert string type name to integer if needed
        if prior_type is not None and isinstance(prior_type, str):
            try:
                prior_type = Prior.type_name_to_int(prior_type)
            except ValueError as e:
                raise ValueError(f"Invalid prior type name: {e}")
        
        # Validate hyperparameters
        if prior_type is not None:
            # Validate hyperparameters for NEW prior type
            temp_prior = Prior(
                name="temp",
                prior_type=prior_type,
                is_age=0,
                min_val=0,
                max_val=1,
                nbin=10
            )
            required_hyper = temp_prior.get_required_hyperparameters()
            
            # Check if hyperparameters are required but not provided
            if required_hyper > 0 and hyperparameters is None:
                type_name = temp_prior.get_type_name()
                param_names = temp_prior.get_hyperparameter_names()
                
                if param_names:
                    hyper_list = ', '.join(param_names)
                    raise ValueError(
                        f"Prior type '{type_name}' requires {required_hyper} hyperparameters ({hyper_list}). "
                        f"Example: set_prior(..., prior_type={prior_type}, hyperparameters=[1.0, 0.5])"
                    )
                else:
                    raise ValueError(
                        f"Prior type '{type_name}' requires {required_hyper} hyperparameters. "
                        f"Example: set_prior(..., prior_type={prior_type}, hyperparameters=[...])"
                    )
            
            # Check if hyperparameters count matches requirements
            if hyperparameters is not None and len(hyperparameters) != required_hyper:
                type_name = temp_prior.get_type_name()
                param_names = temp_prior.get_hyperparameter_names()
                
                if param_names:
                    hyper_list = ', '.join(param_names)
                    raise ValueError(
                        f"Prior type '{type_name}' requires {required_hyper} hyperparameters ({hyper_list}), "
                        f"but {len(hyperparameters)} were provided. "
                        f"Example: set_prior(..., prior_type={prior_type}, hyperparameters=[1.0, 0.5])"
                    )
                else:
                    raise ValueError(
                        f"Prior type '{type_name}' requires {required_hyper} hyperparameters, "
                        f"but {len(hyperparameters)} were provided."
                    )
        elif hyperparameters is not None:
            # Validate hyperparameters for EXISTING prior type (when prior_type not being changed)
            # We need to check what the current prior type is for the parameter(s)
            # This validation will happen during modification, but we can do a preliminary check
            # if we have a single parameter name
            if isinstance(parameter_name, str):
                # Try to get the current prior to check its type
                try:
                    # Use pattern matching to find the parameter
                    match_type, matches = find_matching_parameters(parameter_name, iprior_file)
                    if match_type != 'none' and len(matches) > 0:
                        # Get the first match to check hyperparameter requirements
                        param_name, filename = matches[0]
                        current_prior = self._prior_manager.get_prior(param_name, filename)
                        required_hyper = current_prior.get_required_hyperparameters()
                        
                        if len(hyperparameters) != required_hyper:
                            type_name = current_prior.get_type_name()
                            param_names = current_prior.get_hyperparameter_names()
                            
                            if param_names:
                                hyper_list = ', '.join(param_names)
                                raise ValueError(
                                    f"Parameter '{param_name}' has prior type '{type_name}' which requires "
                                    f"{required_hyper} hyperparameters ({hyper_list}), but {len(hyperparameters)} were provided. "
                                    f"Example: set_prior('{param_name}', hyperparameters=[1.0, 0.5])"
                                )
                            else:
                                raise ValueError(
                                    f"Parameter '{param_name}' has prior type '{type_name}' which requires "
                                    f"{required_hyper} hyperparameters, but {len(hyperparameters)} were provided."
                                )
                except Exception as e:
                    # If we can't validate now, it will be caught during modification
                    pass
        
        # Build modifications dict
        modifications = {}
        if prior_type is not None:
            modifications['prior_type'] = prior_type
        if is_age is not None:
            modifications['is_age'] = is_age
        if min_val is not None:
            modifications['min_val'] = min_val
        if max_val is not None:
            modifications['max_val'] = max_val
        if nbin is not None:
            modifications['nbin'] = nbin
        if hyperparameters is not None:
            modifications['hyperparameters'] = hyperparameters
        
        # Handle informational queries (no modifications specified)
        if not modifications:
            if parameter_name is None and iprior_file is None:
                # List all .iprior files
                files = sorted(self._prior_manager.priors_by_file.keys())
                print(f"\nLoaded .iprior files ({len(files)}):")
                for f in files:
                    param_count = len(self._prior_manager.priors_by_file[f])
                    print(f"  - {f} ({param_count} parameters)")
                return
            elif parameter_name is None and iprior_file is not None:
                # Print the file content using print_priors()
                self.print_priors(iprior_file=iprior_file)
                return
            elif parameter_name is not None:
                # Show info about the parameter using pattern matching
                match_type, matches = find_matching_parameters(parameter_name, iprior_file)
                
                if match_type == 'none':
                    # No matches found - provide helpful error with available parameters
                    all_params = set()
                    for filename, params in self._prior_manager.priors_by_file.items():
                        all_params.update(params.keys())
                    available = ', '.join(sorted(list(all_params)[:10]))  # Show first 10
                    if len(all_params) > 10:
                        available += f", ... ({len(all_params)} total)"
                    raise KeyError(
                        f"No parameters match pattern '{parameter_name}'. "
                        f"Available parameters: {available}"
                    )
                
                # For all matches (single or multiple), use consistent table format
                # Display using print_priors() but filter to only matched parameters
                print(f"\nPattern '{parameter_name}' matched {len(matches)} parameter(s) using {match_type} matching:")
                print("=" * 80)
                
                # Group matches by file
                by_file = {}
                for param_name, filename in matches:
                    if filename not in by_file:
                        by_file[filename] = []
                    by_file[filename].append(param_name)
                
                # Status table for matched parameters (unified formatter)
                status_by_file = {}
                for filename in sorted(by_file.keys()):
                    file_priors = self._prior_manager.priors_by_file[filename]
                    matched_params = by_file[filename]
                    status_by_file[filename] = {
                        name: file_priors[name]
                        for name in matched_params
                        if name in file_priors
                    }
                self._print_prior_table(status_by_file, modified_mode="status")
                print("=" * 80)
                return
        
        # Validate that at least one of parameter_name or iprior_file is specified for modifications
        if parameter_name is None and iprior_file is None:
            raise ValueError(
                "Must specify at least one of parameter_name or iprior_file for modifications.\n"
                "Examples:\n"
                "  - set_prior('log(age/yr)', min_val=8.0)  # Mode 1: auto-detect file\n"
                "  - set_prior('f', iprior_file='BLR.iprior', min_val=0.1)  # Mode 2: explicit file\n"
                "  - set_prior(iprior_file='2dal8.iprior', nbin=60)  # Mode 3: all params in file"
            )
        
        # Determine which parameters to modify based on the mode
        param_names_to_modify = []
        
        if parameter_name is None:
            # Mode 3: Bulk operation - modify ALL parameters in the specified file
            if iprior_file is None:
                raise ValueError("iprior_file must be specified when parameter_name is None")
            
            param_names_to_modify = self._prior_manager.get_parameters_from_file(iprior_file)
            if not param_names_to_modify:
                raise FileNotFoundError(
                    f"No parameters found in file '{iprior_file}'. "
                    f"Make sure the file was loaded with priors_init()."
                )
            
            # Show confirmation for bulk modification (if confirm=True)
            # Create matches list in the format expected by confirm function
            matches = [(param_name, os.path.basename(iprior_file)) for param_name in param_names_to_modify]
            if confirm and not confirm_multiple_parameter_modification('bulk', matches, modifications):
                print("\nModifications cancelled.")
                return
        else:
            # Mode 1 or 2: Specific parameter(s) with pattern matching support
            # Handle single parameter or list of parameters
            param_names = [parameter_name] if isinstance(parameter_name, str) else parameter_name
            
            for param_name in param_names:
                # Use pattern matching for string parameters
                if isinstance(param_name, str):
                    match_type, matches = find_matching_parameters(param_name, iprior_file)
                    
                    if match_type == 'none':
                        # No matches found — provide helpful error
                        if component_id is not None:
                            ids = sorted(set(p.component_id for ps in self._prior_manager.priors_by_file.values() for p in ps.values()))
                            raise KeyError(
                                f"No parameters match '{param_name}' with component_id={component_id} "
                                f"(available component_ids: {ids})"
                            )
                        all_params = ', '.join(sorted(set(
                            n for ps in self._prior_manager.priors_by_file.values() for n in ps
                        )))
                        raise KeyError(
                            f"No parameters match '{param_name}'. "
                            f"Available parameters: {all_params}"
                        )
                    
                    elif match_type == 'exact':
                        # Exact match - require confirmation if confirm=True
                        if len(matches) == 1:
                            # Single exact match - show confirmation if confirm=True
                            if confirm and not confirm_multiple_parameter_modification('exact', matches, modifications):
                                print("\nModifications cancelled.")
                                return
                            param_names_to_modify.append(matches[0][0])
                        else:
                            # Multiple files have this exact parameter name
                            if iprior_file is None:
                                files = [fname for _, fname in matches]
                                raise ValueError(
                                    f"Parameter '{param_name}' exists in multiple files: {', '.join(files)}. "
                                    f"Please specify iprior_file parameter to disambiguate.\n"
                                    f"Example: set_prior('{param_name}', iprior_file='{files[0]}', ...)"
                                )
                            else:
                                # iprior_file specified - show confirmation if confirm=True
                                if confirm and not confirm_multiple_parameter_modification('exact', matches, modifications):
                                    print("\nModifications cancelled.")
                                    return
                                param_names_to_modify.append(param_name)
                    
                    elif match_type in ['regex', 'partial']:
                        # Multiple matches - require user confirmation (if confirm=True)
                        if confirm and not confirm_multiple_parameter_modification(match_type, matches, modifications):
                            print("\nModifications cancelled.")
                            return
                        
                        # User confirmed or confirm=False - add all matched parameters
                        for matched_param, _ in matches:
                            param_names_to_modify.append(matched_param)
                else:
                    # Not a string (shouldn't happen, but handle gracefully)
                    param_names_to_modify.append(param_name)
        
        # Set recently_modified tracking BEFORE applying modifications
        # This allows the display to show which parameters were just changed
        self._prior_manager.set_recently_modified(param_names_to_modify)
        
        # Track which files were actually modified
        modified_files = set()
        
        # Apply modifications to each parameter
        for param_name in param_names_to_modify:
            # Determine which file this parameter is in
            files = self._prior_manager.find_files_containing_parameter(param_name)
            if len(files) == 1:
                target_file = files[0]
            elif iprior_file:
                target_file = os.path.basename(iprior_file)
            else:
                # This shouldn't happen because we already checked, but just in case
                target_file = None
            
            if target_file:
                modified_files.add(target_file)
            
            self._prior_manager.modify_prior(param_name, iprior_file=target_file, **modifications)
            # Update the parameter in its original source file
            try:
                self._prior_manager.update_prior_in_source_file(param_name, iprior_file=target_file)
            except Exception as e:
                # Log warning but don't fail - the modification is still in memory
                logging.getLogger(__name__).warning(
                    f"Failed to update '{param_name}' in source file: {e}. "
                    f"Modification is saved in memory but not persisted to disk."
                )
        
        # Display all parameters from the modified file(s)
        print()  # Add blank line before display
        if len(modified_files) == 1:
            # Only one file was modified - show all parameters in that file
            self.print_priors(iprior_file=list(modified_files)[0])
        else:
            # Multiple files modified - show all parameters from each modified file
            for filename in sorted(modified_files):
                self.print_priors(iprior_file=filename)
        
        # Clear recently_modified after display
        self._prior_manager.clear_recently_modified()
        
        # Print success message (consistent with reset_to_default path)
        print(f"\n✓ Modified {len(param_names_to_modify)} parameter(s)")

    def print_priors(self, iprior_file: str = None, parameter_names: list = None):
        """
        Print priors organized by .iprior file.
        
        Lists all loaded .iprior files and displays their contents one by one,
        similar to the 'cat' command. This approach clearly separates parameters
        from different files and handles duplicate parameter names naturally.
        
        Parameters are displayed in the same order as they appear in the .iprior file,
        preserving the original file structure.
        
        Parameters
        ----------
        iprior_file : str, optional
            If provided, only print priors from this specific .iprior file.
            Can be just the filename (e.g., '2dal8.iprior') or full path.
            If None, prints all loaded .iprior files and returns the list of filenames.
        
        Returns
        -------
        list of str or None
            When iprior_file is None, returns list of all .iprior filenames.
            When iprior_file is specified, returns None.
        
        Raises
        ------
        FileNotFoundError
            If the specified iprior_file is not found

        Notes
        -----
        Callers reach this via `Priors` / `SEDInference`, which raise if
        `priors_init()` has not succeeded.
        
        Examples
        --------
        >>> inference.priors_init(params)
        >>> files = inference.print_priors()  # Print all files and get list
        >>> print(files)
        ['2dal7.iprior', 'BLR.iprior', 'FeII.iprior', ...]
        >>> inference.print_priors(iprior_file='2dal8.iprior')  # Print specific file
        """
        
        # Group parameters by source file
        params_by_file = self._prior_manager.priors_by_file
        
        # Store whether we're printing all files (for return value)
        print_all = (iprior_file is None and parameter_names is None)
        
        # Filter by specific file if requested
        if iprior_file is not None:
            target_basename = os.path.basename(iprior_file)
            if target_basename not in params_by_file:
                available = ', '.join(sorted(params_by_file.keys()))
                raise FileNotFoundError(
                    f"File '{iprior_file}' not found in loaded priors. "
                    f"Available files: {available}"
                )
            params_by_file = {target_basename: params_by_file[target_basename]}
        
        # Filter by specific parameter names if requested
        if parameter_names is not None:
            filtered_params_by_file = {}
            for filename, params_dict in params_by_file.items():
                filtered_params = {name: prior for name, prior in params_dict.items() 
                                 if name in parameter_names}
                if filtered_params:  # Only include files that have matching parameters
                    filtered_params_by_file[filename] = filtered_params
            params_by_file = filtered_params_by_file
        
        # Print header
        print("\n" + "=" * 80)
        print("Prior Configuration")
        print("=" * 80)
        
        # Preserve .iprior file order within each file (sort_names=False)
        self._print_prior_table(
            params_by_file,
            modified_mode="status",
            sort_names=False,
        )
        print("=" * 80)

        
        # Return list of files if printing all, None otherwise
        return sorted(self._prior_manager.priors_by_file.keys()) if print_all else None
