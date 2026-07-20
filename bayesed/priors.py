"""
Priors lifecycle for BayeSED3 model configurations.

Owns load/generate/validate of .iprior files; interactive edit/print delegates
to PriorEditor (ADR-0002, ADR-0006). SEDInference keeps thin facades.
"""

import copy
import glob
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections import defaultdict
from io import StringIO
from typing import TYPE_CHECKING

from .core import BayeSEDInterface
from .prior import Prior
from .prior_editor import PriorEditor
from .prior_manager import PriorManager

if TYPE_CHECKING:
    from .core import BayeSEDParams


class Priors:
    """
    Prior lifecycle for a model configuration.

    Typical use is via SEDInference.priors_init / set_prior facades. Direct use:

    >>> priors = Priors()
    >>> priors.priors_init(params)
    >>> priors.set_prior('log(age/yr)', min_val=8.5, max_val=9.8)
    """

    def __init__(self):
        self._params = None
        self._prior_manager = None
        self._editor = None

    def priors_init(self, params: 'BayeSEDParams', regenerate_defaults: bool = False, 
                    clean_input_dir: bool = False, force: bool = False, verbose: bool = False) -> None:
        """
        Initialize priors for the given parameters.
        
        This method loads .iprior files from the input directory and makes them
        available for inspection and modification. It intelligently handles file
        generation and loading based on what exists and what's needed.
        
        Key Features
        ------------
        1. **Automatic file generation**: If no .iprior files exist, generates them automatically
        2. **Fresh defaults**: Uses pristine defaults from priors_defaults directory for reset_to_default
        3. **Relevant file filtering**: Only loads files relevant to your model configuration
        4. **Force regenerate**: Can regenerate files even if they exist (overwrites with fresh)
        5. **Quiet by default**: Minimal output unless verbose=True
        6. **Fast defaults**: Reuses existing priors_defaults instead of regenerating
        
        Behavior
        --------
        - **No files exist**: Generates files in input directory, copies to priors_defaults
        - **Files exist, no defaults**: Generates defaults in priors_defaults (once)
        - **Files exist, defaults exist**: Reuses existing defaults (fast)
        - **regenerate_defaults=True**: Generates fresh files in priors_defaults, copies to input dir
        
        The fresh defaults are stored in `{outdir}/priors_defaults/` and reused across runs,
        making reset_to_default operations much faster. Defaults are generated using the SAME
        model configuration as your params, ensuring they match your actual model setup.
        
        Parameters
        ----------
        params : BayeSEDParams
            Model configuration parameters (stored internally for later use).
            The SAME configuration is used to generate fresh defaults.
        regenerate_defaults : bool, optional
            If True, clear and regenerate priors_defaults directory. Generates fresh
            files in priors_defaults directory and copies them to input directory,
            overwriting existing files. Only overwrites files relevant to current model config.
            
            **IMPORTANT:** This is a destructive operation! By default, you will be
            prompted to confirm before files are overwritten. Use `force=True` to skip
            confirmation (useful for automated scripts).
            
            **When to use:**
            - Model configuration changed (different SSP, components, etc.)
            - Prior files corrupted or incorrectly modified
            - After BayeSED binary update
            - Need a fresh start with pristine defaults
            
            **When NOT to use:**
            - Normal workflow (wastes time regenerating)
            - After setting custom priors (will overwrite your changes!)
            - Just to reset priors (use set_prior with reset_to_default=True instead)
            
            Default is False.
        clean_input_dir : bool, optional
            If True, removes ALL .iprior files from input directory before generating
            new ones. This ensures a completely clean slate, removing any stale files
            from previous model configurations.
            
            **IMPORTANT:** This is a destructive operation! By default, you will be
            prompted to confirm before files are deleted. Use `force=True` to skip
            confirmation (useful for automated scripts).
            
            **When to use:**
            - Model configuration changed significantly (different components)
            - Input directory has mixed files from multiple models
            - Want to ensure only current model's files exist
            - Shared directory needs cleanup
            
            **When NOT to use:**
            - Input directory is shared with other analyses
            - You have manually created .iprior files you want to preserve
            - Normal workflow (unnecessary)
            
            **Relationship with regenerate_defaults:**
            - `clean_input_dir=True` alone: Clears input dir, generates fresh files
            - `regenerate_defaults=True` alone: Clears priors_defaults, overwrites input files
            - Both True: Clears both directories, complete fresh start
            - Both False: Normal operation, reuses existing files
            
            Default is False.
        force : bool, optional
            If True, skip confirmation prompt when `clean_input_dir=True`.
            This is useful for automated scripts where user interaction is not possible.
            
            **WARNING:** Using `force=True` with `clean_input_dir=True` will delete
            files without asking! Only use this when you're certain you want to delete
            all .iprior files in the input directory.
            
            Default is False (requires confirmation).
        verbose : bool, optional
            If True, show detailed progress messages during initialization.
            If False (default), operates quietly with minimal output.
            Default is False.
        
        Raises
        ------
        ValueError
            If no .iprior files are found and auto-generation fails
        FileNotFoundError
            If the input directory doesn't exist
            
        Examples
        --------
        Basic usage (quiet mode):
        
        >>> from bayesed import SEDInference, BayeSEDParams
        >>> params = BayeSEDParams.galaxy(input_file='observation/test/gal.txt', outdir='output')
        >>> inference = SEDInference()
        >>> inference.priors_init(params)  # Quiet, automatic
        
        Verbose mode for debugging:
        
        >>> inference.priors_init(params, verbose=True)
        # Shows: file generation, defaults loading, file filtering
        
        Force regenerate (fresh start):
        
        >>> inference.priors_init(params, regenerate_defaults=True)
        # Overwrites existing files with fresh ones
        
        Clean input directory (with confirmation):
        
        >>> inference.priors_init(params, clean_input_dir=True)
        # Prompts: "Do you want to proceed? (yes/no):"
        # User must type 'yes' to confirm deletion
        
        Clean input directory (automated script, no confirmation):
        
        >>> inference.priors_init(params, clean_input_dir=True, force=True)
        # Deletes files without asking - use with caution!
        
        Shared directory with multiple models:
        
        >>> # Directory has files for both galaxy and AGN models
        >>> params_gal = BayeSEDParams.galaxy(input_file='observation/shared/gal.txt')
        >>> inference_gal = SEDInference()
        >>> inference_gal.priors_init(params_gal)
        # Loads only galaxy files, skips AGN files
        
        >>> params_agn = BayeSEDParams.agn(input_file='observation/shared/qso.txt')
        >>> inference_agn = SEDInference()
        >>> inference_agn.priors_init(params_agn)
        # Loads only AGN files, skips galaxy files
        
        Notes
        -----
        - The BayeSED binary will NOT regenerate .iprior files if they already exist
        - regenerate_defaults works by generating in priors_defaults and copying with overwrite
        - Only files relevant to your model configuration are loaded
        - Fresh defaults are stored in {outdir}/priors_defaults/ and reused for speed
        - If priors_defaults exists, no binary execution needed for defaults (much faster)
        - clean_input_dir requires user confirmation unless force=True is used
        
        See Also
        --------
        set_prior : Modify prior values
        print_priors : Display current prior configuration
        reset_to_default : Reset priors to default values (requires priors_init first)
        """
        # Clear editor at entry so an early re-init failure cannot leave
        # set_prior/print_priors bound to a stale PriorEditor (ADR-0006).
        self._editor = None

        # Store params for later use
        self._params = params
        
        # Determine input directory from params.input_file
        if not params.input_file:
            raise ValueError("params.input_file must be set to determine prior file location")
        
        input_dir = os.path.dirname(params.input_file)
        if not input_dir:
            input_dir = '.'
        
        # Check if directory exists
        if not os.path.exists(input_dir):
            raise FileNotFoundError(f"Input directory not found: {input_dir}")
        
        # Determine priors_defaults directory under outdir
        outdir = params.outdir if params.outdir else 'result'
        priors_defaults_dir = os.path.join(outdir, 'priors_defaults')
        
        # Create priors_defaults directory if it doesn't exist
        os.makedirs(priors_defaults_dir, exist_ok=True)
        
        # Replace manager before load/generate work.
        self._prior_manager = PriorManager(base_directory=input_dir)
        
        # Find all .iprior files in the directory
        iprior_pattern = os.path.join(input_dir, '*.iprior*')
        all_iprior_files = glob.glob(iprior_pattern)
        
        # Filter to only accept valid .iprior files:
        # - Files ending with .iprior exactly
        # - Files ending with .iprior.X where X is a digit (e.g., .iprior.0, .iprior.4)
        # Reject files like .iprior.bak, .iprior.test_backup, etc.
        valid_iprior_pattern = re.compile(r'.*\.iprior(\.\d+)?$')
        all_iprior_files = [f for f in all_iprior_files if valid_iprior_pattern.match(os.path.basename(f))]
        
        # Clean input directory if requested
        if clean_input_dir and all_iprior_files:
            # Show warning and ask for confirmation (unless force=True)
            if not force:
                print(f"\n{'='*60}")
                print(f"WARNING: clean_input_dir=True will DELETE all .iprior files!")
                print(f"{'='*60}")
                print(f"Directory: {input_dir}")
                print(f"Files to be deleted ({len(all_iprior_files)}):")
                for iprior_file in sorted(all_iprior_files):
                    print(f"  - {os.path.basename(iprior_file)}")
                print(f"\nThis action cannot be undone!")
                print(f"{'='*60}")
                
                # Ask for confirmation
                try:
                    response = input("Do you want to proceed? (yes/no): ").strip().lower()
                    if response not in ['yes', 'y']:
                        print("Operation cancelled by user.")
                        raise ValueError("clean_input_dir operation cancelled by user")
                except (EOFError, KeyboardInterrupt):
                    print("\nOperation cancelled by user.")
                    raise ValueError("clean_input_dir operation cancelled by user")
            
            # User confirmed (or force=True) - proceed with deletion
            if verbose or force:
                print(f"\nCleaning input directory: {input_dir}")
                print(f"  Removing {len(all_iprior_files)} .iprior file(s)...")
            
            for iprior_file in all_iprior_files:
                try:
                    os.remove(iprior_file)
                    if verbose:
                        print(f"    Removed: {os.path.basename(iprior_file)}")
                except Exception as e:
                    logging.getLogger(__name__).warning(
                        f"Failed to remove {os.path.basename(iprior_file)}: {e}"
                    )
            
            if verbose or not force:
                print("✓ Input directory cleaned successfully\n")
            
            # Re-scan after cleaning
            all_iprior_files = []
        
        # Prefer .iprior.X files over base .iprior files
        # Group files by base name (without .X suffix)
        files_by_base = defaultdict(list)
        for f in all_iprior_files:
            basename = os.path.basename(f)
            # Extract base name (e.g., "kin_con2.iprior" from "kin_con2.iprior.4")
            if '.iprior.' in basename:
                base = basename.rsplit('.', 1)[0]  # Remove the .X suffix
            else:
                base = basename
            files_by_base[base].append(f)
        
        # For each base name, prefer .iprior.X files over base .iprior files
        iprior_files = []
        for base, files in files_by_base.items():
            # Sort to get numbered variants first (e.g., .iprior.4 before .iprior)
            # Numbered variants have more dots in the filename
            files_sorted = sorted(files, key=lambda x: -x.count('.'))
            # Take all numbered variants, or the base file if no numbered variants exist
            numbered = [f for f in files_sorted if '.iprior.' in os.path.basename(f)]
            if numbered:
                iprior_files.extend(numbered)
            else:
                iprior_files.append(files_sorted[0])
        
        # Generate prior files if needed
        # IMPORTANT: The BayeSED binary will NOT regenerate .iprior files if they
        # already exist in the directory. So for regenerate_defaults, we generate in
        # priors_defaults directory and then copy with overwrite.
        
        files_existed_before = bool(iprior_files)
        
        if regenerate_defaults and iprior_files:
            # User wants to regenerate defaults, but files exist
            # Generate in priors_defaults first to know which files will be created
            
            # User confirmed (or force=True) - proceed with regeneration
            if verbose:
                print(f"\nRegenerating defaults...")
            
            # Binary won't regenerate if files exist, so generate in priors_defaults and copy
            
            # Clear priors_defaults directory to ensure clean regeneration
            # This removes stale files from previous model configurations
            if os.path.exists(priors_defaults_dir):
                if verbose:
                    print(f"  Clearing {priors_defaults_dir} for clean regeneration...")
                try:
                    shutil.rmtree(priors_defaults_dir)
                    os.makedirs(priors_defaults_dir, exist_ok=True)
                except Exception as e:
                    logging.getLogger(__name__).warning(
                        f"Failed to clear priors_defaults directory: {e}. Continuing anyway."
                    )
            
            # Generate in priors_defaults directory and load as defaults
            try:
                if verbose:
                    print(f"  Generating fresh defaults in {priors_defaults_dir}...")
                relevant_filenames = self._generate_fresh_defaults(params, priors_defaults_dir, verbose=verbose)
                
                # Find generated files in priors_defaults dir
                defaults_pattern = os.path.join(priors_defaults_dir, '*.iprior*')
                defaults_files = glob.glob(defaults_pattern)
                defaults_files = [f for f in defaults_files if valid_iprior_pattern.match(os.path.basename(f))]
                
                # Now we know which files will be copied - check which ones exist and will be overwritten
                files_to_overwrite = []
                for defaults_file in defaults_files:
                    dest_file = os.path.join(input_dir, os.path.basename(defaults_file))
                    if os.path.exists(dest_file):
                        files_to_overwrite.append(dest_file)
                
                # Ask for confirmation if files will be overwritten (unless force=True)
                if files_to_overwrite and not force:
                    print(f"\n{'='*60}")
                    print(f"WARNING: regenerate_defaults=True will OVERWRITE .iprior files!")
                    print(f"{'='*60}")
                    print(f"Directory: {input_dir}")
                    print(f"Files to be overwritten ({len(files_to_overwrite)}):")
                    for iprior_file in sorted(files_to_overwrite):
                        print(f"  - {os.path.basename(iprior_file)}")
                    print(f"\nAny manual edits to these files will be lost!")
                    print(f"{'='*60}")
                    
                    # Ask for confirmation
                    try:
                        response = input("Do you want to proceed? (yes/no): ").strip().lower()
                        if response not in ['yes', 'y']:
                            print("Operation cancelled by user.")
                            raise ValueError("regenerate_defaults operation cancelled by user")
                    except (EOFError, KeyboardInterrupt):
                        print("\nOperation cancelled by user.")
                        raise ValueError("regenerate_defaults operation cancelled by user")
                
                if verbose:
                    print(f"  Copying {len(defaults_files)} file(s) to {input_dir} (overwriting existing)...")
                
                # Copy files from priors_defaults to input dir (overwrite existing)
                # Track the copied files - these are the ONLY files we should load
                copied_files = []
                for defaults_file in defaults_files:
                    dest_file = os.path.join(input_dir, os.path.basename(defaults_file))
                    shutil.copy2(defaults_file, dest_file)
                    copied_files.append(dest_file)
                    if verbose:
                        print(f"    Copied: {os.path.basename(defaults_file)}")
                
                # Use ONLY the copied files - ignore any other files in input_dir
                # This ensures we load exactly what was generated for the current model
                iprior_files = copied_files
                
                if verbose or not force:
                    print("✓ Defaults regenerated and files updated successfully\n")
                
            except Exception as e:
                raise ValueError(f"Failed to regenerate defaults: {e}") from e
            
            # Skip the normal defaults loading logic - we already loaded defaults above
            # Skip the filtering logic - we already have the exact files to load
            files_existed_before = None  # Special flag: defaults already loaded, skip filtering
        
        if not iprior_files:
            # No files exist - generate them
            if verbose:
                print(f"No .iprior files found in {input_dir}. Generating them automatically...")
            
            try:
                self._generate_prior_files(params, input_dir, verbose=verbose)
                # Re-scan for .iprior files after generation and apply same filtering logic
                all_iprior_files = glob.glob(iprior_pattern)
                all_iprior_files = [f for f in all_iprior_files if valid_iprior_pattern.match(os.path.basename(f))]
                
                # Re-apply preference logic: prefer .iprior.X files over base .iprior files
                files_by_base = defaultdict(list)
                for f in all_iprior_files:
                    basename = os.path.basename(f)
                    if '.iprior.' in basename:
                        base = basename.rsplit('.', 1)[0]
                    else:
                        base = basename
                    files_by_base[base].append(f)
                
                iprior_files = []
                for base, files in files_by_base.items():
                    files_sorted = sorted(files, key=lambda x: -x.count('.'))
                    numbered = [f for f in files_sorted if '.iprior.' in os.path.basename(f)]
                    if numbered:
                        iprior_files.extend(numbered)
                    else:
                        iprior_files.append(files_sorted[0])
                
                # Also copy to priors_defaults for future use
                if iprior_files and verbose:
                    print(f"  Copying generated files to {priors_defaults_dir} for future use...")
                for iprior_file in iprior_files:
                    dest_file = os.path.join(priors_defaults_dir, os.path.basename(iprior_file))
                    shutil.copy2(iprior_file, dest_file)
                    
            except Exception as e:
                raise ValueError(
                    f"Failed to auto-generate .iprior files: {e}\n"
                    f"You can generate them manually by:\n"
                    f"  1. Running BayeSED once with your configuration, or\n"
                    f"  2. Copying .iprior files from example directories (e.g., observation/test/)"
                ) from e
        
        if not iprior_files:
            raise ValueError(
                f"No .iprior files found in directory: {input_dir}\n"
                f"Auto-generation failed. Please ensure .iprior files exist before calling priors_init().\n"
                f"You can generate them by:\n"
                f"  1. Running BayeSED once with your configuration, or\n"
                f"  2. Copying .iprior files from example directories (e.g., observation/test/)"
            )
        
        # Generate fresh defaults and get list of relevant files
        # This serves two purposes:
        # 1. Get pristine default values for reset_to_default functionality
        # 2. Determine which .iprior files are relevant for current model config
        #
        # Logic for when to use priors_defaults directory:
        # - If priors_defaults exists and has files: Use existing files (FAST, no binary execution)
        # - If priors_defaults doesn't exist or is empty: Generate files there (one-time cost)
        # - If files existed before (and weren't force-regenerated): Need defaults
        # - If files didn't exist (or were force-regenerated): Use generated files directly
        
        relevant_filenames = set()
        
        # Check if priors_defaults already has files
        defaults_pattern = os.path.join(priors_defaults_dir, '*.iprior*')
        existing_defaults = glob.glob(defaults_pattern)
        existing_defaults = [f for f in existing_defaults if valid_iprior_pattern.match(os.path.basename(f))]
        
        # Handle defaults loading based on whether files existed and were regenerated
        if files_existed_before is None:
            # Special case: regenerate_defaults=True already loaded defaults above
            # Skip defaults loading, just use the relevant_filenames we already have
            if verbose:
                print("✓ Defaults already loaded from regeneration")
        elif files_existed_before:
            # Files existed and weren't regenerated - might be user-modified
            # Need to load defaults for reset_to_default functionality
            
            if existing_defaults:
                # Defaults already exist - reuse them (FAST!)
                if verbose:
                    print(f"✓ Reusing existing defaults from {priors_defaults_dir} (fast)")
                relevant_filenames = self._load_defaults_from_directory(priors_defaults_dir, verbose=verbose)
            else:
                # No defaults exist - generate them (one-time cost)
                if verbose:
                    print(f"Generating fresh defaults in {priors_defaults_dir}...")
                try:
                    relevant_filenames = self._generate_fresh_defaults(params, priors_defaults_dir, verbose=verbose)
                    if verbose:
                        print("✓ Fresh defaults generated")
                        if relevant_filenames:
                            print(f"✓ Identified {len(relevant_filenames)} relevant file(s) for current model config")

                    # If generated filenames don't exist in input dir (e.g. new binary
                    # adds .iprior.{N} suffix), copy defaults to input dir and skip filter.
                    if relevant_filenames:
                        input_basenames = {os.path.basename(f) for f in iprior_files}
                        if not (input_basenames & relevant_filenames):
                            if verbose:
                                print(f"  Input dir has stale .iprior files, copying generated defaults...")
                            defaults_files = [os.path.join(priors_defaults_dir, b) for b in relevant_filenames]
                            copied = []
                            for f in defaults_files:
                                dst = os.path.join(input_dir, os.path.basename(f))
                                shutil.copy2(f, dst)
                                copied.append(dst)
                            iprior_files = copied
                            files_existed_before = None  # skip filter, use these files directly
                except Exception as e:
                    logging.getLogger(__name__).warning(
                        f"Failed to generate fresh defaults: {e}. "
                        f"reset_to_default functionality will not be available."
                    )
        else:
            # Files didn't exist - just generated fresh
            # Use the generated files directly as defaults (no separate defaults needed)
            if verbose:
                print("✓ Using newly generated files as defaults")
            # Load the freshly generated files as defaults
            for iprior_file in iprior_files:
                filename = os.path.basename(iprior_file)
                relevant_filenames.add(filename)
                try:
                    self._prior_manager.load_prior_file(filename, is_auto_generated=True)
                except Exception as e:
                    logging.getLogger(__name__).warning(f"Failed to load {filename} as default: {e}")
        
        # Filter to only load relevant .iprior files (matching current model config)
        # Exception: When regenerate_defaults=True, we already have the exact files to load
        if files_existed_before is None:
            # regenerate_defaults=True case: iprior_files already contains exactly what we need
            files_to_load = iprior_files
            if verbose:
                print(f"✓ Loading {len(files_to_load)} regenerated file(s) (no filtering needed)")
        elif relevant_filenames:
            # Normal case: filter by relevant filenames
            files_to_load = [f for f in iprior_files if os.path.basename(f) in relevant_filenames]
            if verbose and len(files_to_load) < len(iprior_files):
                skipped = len(iprior_files) - len(files_to_load)
                print(f"✓ Filtering: loading {len(files_to_load)} relevant file(s), skipping {skipped} irrelevant file(s)")
        else:
            # Couldn't determine relevant files - load all (fallback)
            files_to_load = iprior_files
            if verbose:
                print("⚠ Could not determine relevant files, loading all .iprior files")
        
        # Load relevant .iprior files from input directory (working copies)
        for iprior_file in files_to_load:
            filename = os.path.basename(iprior_file)
            try:
                # Load as working copies (NOT as defaults)
                self._prior_manager.load_prior_file(filename, is_auto_generated=False)
            except Exception as e:
                # Provide helpful error message but continue loading other files
                logging.getLogger(__name__).warning(
                    f"Failed to load prior file '{filename}': {e}. Skipping."
                )
        
        # Verify we loaded at least some priors
        if not self._prior_manager.priors_by_file:
            raise ValueError(
                f"Failed to load any valid priors from .iprior files in: {input_dir}\n"
                f"Please check that the .iprior files are properly formatted."
            )
        
        # Always list the loaded .iprior files (even when not in verbose mode)
        print(f"Loaded {len(files_to_load)} prior file(s):")
        for iprior_file in sorted(files_to_load):
            print(f"  - {os.path.basename(iprior_file)}")

        self._editor = PriorEditor(self._prior_manager)

    def _load_defaults_from_directory(self, defaults_dir: str, verbose: bool = False) -> set:
        """
        Load default priors from an existing priors_defaults directory.
        
        This method loads .iprior files from the priors_defaults directory without
        running the binary, making it much faster than regenerating defaults.
        
        Parameters
        ----------
        defaults_dir : str
            Path to priors_defaults directory containing .iprior files
        verbose : bool, optional
            If True, show detailed progress messages. Default is False.
            
        Returns
        -------
        set
            Set of relevant .iprior filenames (basenames only) that were loaded.
        """
        
        relevant_filenames = set()
        
        # Find all .iprior files in defaults directory
        iprior_pattern = os.path.join(defaults_dir, '*.iprior*')
        defaults_files = glob.glob(iprior_pattern)
        
        # Filter to only valid .iprior files
        iprior_regex = re.compile(r'.*\.iprior(\.\d+)?$')
        defaults_files = [f for f in defaults_files if iprior_regex.match(f)]
        
        if not defaults_files:
            return relevant_filenames
        
        # Load these as defaults (original_priors)
        for iprior_file in defaults_files:
            filename = os.path.basename(iprior_file)
            relevant_filenames.add(filename)
            
            # Parse the file directly
            with open(iprior_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    
                    try:
                        parts = line.split()
                        if len(parts) < 6:
                            continue
                        
                        name = parts[0]
                        prior_type = int(parts[1])
                        is_age = int(parts[2])
                        min_val = float(parts[3])
                        max_val = float(parts[4])
                        nbin = int(parts[5])
                        hyperparameters = [float(p) for p in parts[6:]] if len(parts) > 6 else []
                        
                        # Create Prior object
                        prior = Prior(
                            name=name,
                            prior_type=prior_type,
                            is_age=is_age,
                            min_val=min_val,
                            max_val=max_val,
                            nbin=nbin,
                            hyperparameters=hyperparameters
                        )
                        
                        # Store as default in this instance (only if not already present)
                        if name not in self._prior_manager.original_priors:
                            self._prior_manager.original_priors[name] = prior
                    except (ValueError, IndexError):
                        # Skip malformed lines
                        continue
        
        if verbose:
            print(f"✓ Loaded {len(self._prior_manager.original_priors)} default priors from {defaults_dir}")
        
        return relevant_filenames
    def _generate_fresh_defaults(self, params: 'BayeSEDParams', defaults_dir: str, verbose: bool = False) -> set:
        """
        Generate fresh default priors in the priors_defaults directory.
        
        This method generates .iprior files in the priors_defaults directory
        using the BayeSED binary with the SAME model configuration as params,
        and loads them as the true defaults. This ensures that defaults are not
        affected by any manual modifications to .iprior files in the user's
        input directory, and that defaults match the actual model configuration.
        
        The generated files are stored persistently in priors_defaults and can be
        reused across runs, making subsequent priors_init calls much faster.
        
        Parameters
        ----------
        params : BayeSEDParams
            Model configuration parameters. The SAME configuration (model components,
            settings, etc.) is used to generate defaults, ensuring they match your
            actual model setup.
        defaults_dir : str
            Path to priors_defaults directory where files should be generated
        verbose : bool, optional
            If True, show detailed progress messages. Default is False.
            
        Returns
        -------
        set
            Set of relevant .iprior filenames (basenames only) for the current model config.
            This can be used to filter which files to load from the input directory.
            
        Raises
        ------
        RuntimeError
            If default prior generation fails
        """
        
        if verbose:
            print(f"Generating fresh defaults in {defaults_dir}...")
        
        relevant_filenames = set()
        
        # Generate .iprior files in priors_defaults directory
        # _generate_prior_files already handles:
        #   - Creating empty input file
        #   - Deep copying params
        #   - Running binary with same model config
        try:
            self._generate_prior_files(params, defaults_dir, verbose=verbose)
        except Exception as e:
            raise RuntimeError(f"Failed to generate fresh defaults: {e}")
        
        # Find all generated .iprior files (with proper filtering)
        iprior_pattern = os.path.join(defaults_dir, '*.iprior*')
        defaults_iprior_files = glob.glob(iprior_pattern)
        
        # Filter to only valid .iprior files (ending with .iprior or .iprior.X where X is digits)
        iprior_regex = re.compile(r'.*\.iprior(\.\d+)?$')
        defaults_iprior_files = [f for f in defaults_iprior_files if iprior_regex.match(f)]
        
        if not defaults_iprior_files:
            raise RuntimeError(f"No .iprior files generated in {defaults_dir}")
        
        # Collect relevant filenames
        for iprior_file in defaults_iprior_files:
            relevant_filenames.add(os.path.basename(iprior_file))
        
        # Load these as defaults (original_priors)
        for iprior_file in defaults_iprior_files:
            # Parse the file directly
            with open(iprior_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    
                    try:
                        parts = line.split()
                        if len(parts) < 6:
                            continue
                        
                        name = parts[0]
                        prior_type = int(parts[1])
                        is_age = int(parts[2])
                        min_val = float(parts[3])
                        max_val = float(parts[4])
                        nbin = int(parts[5])
                        hyperparameters = [float(p) for p in parts[6:]] if len(parts) > 6 else []
                        
                        # Create Prior object
                        prior = Prior(
                            name=name,
                            prior_type=prior_type,
                            is_age=is_age,
                            min_val=min_val,
                            max_val=max_val,
                            nbin=nbin,
                            hyperparameters=hyperparameters
                        )
                        
                        # Store as default in this instance (only if not already present)
                        if name not in self._prior_manager.original_priors:
                            self._prior_manager.original_priors[name] = prior
                    except (ValueError, IndexError):
                        # Skip malformed lines
                        continue
        
        if verbose:
            print(f"✓ Loaded {len(self._prior_manager.original_priors)} default priors from {defaults_dir}")
        
        return relevant_filenames
    def _generate_prior_files(self, params: 'BayeSEDParams', input_dir: str, verbose: bool = False) -> None:
        """
        Generate .iprior files by running BayeSED binary with a temporary empty input file.
        
        This method creates a temporary empty input file and runs the BayeSED binary
        with the user's model configuration. The binary will generate template .iprior
        files for all model components and then exit with "No valid objects" message.
        
        Parameters
        ----------
        params : BayeSEDParams
            Model configuration parameters
        input_dir : str
            Directory where .iprior files should be generated
        verbose : bool, optional
            If True, show detailed progress messages. Default is False.
            
        Raises
        ------
        RuntimeError
            If prior file generation fails
        FileNotFoundError
            If the BayeSED binary cannot be found
        """
        
        if verbose:
            print("Generating .iprior files by running BayeSED binary...")
        
        # Create a temporary empty input file in the target directory
        # This ensures .iprior files are generated in the correct location
        temp_input_file = os.path.join(input_dir, '.temp_empty_input.txt')
        
        try:
            # Create empty input file with just a header
            with open(temp_input_file, 'w') as f:
                f.write("# gal 0 0 0\n")
                f.close()
                # No data rows - this will cause "No valid objects" error
            
            # Create a copy of params with the temporary input file
            temp_params = copy.deepcopy(params)
            temp_params.input_file = temp_input_file
            temp_params.outdir = input_dir
            
            # Set priors_only flag to speed up generation (if supported)
            temp_params.priors_only = True
            
            # Create BayeSED interface with minimal settings for prior generation
            # Suppress OpenMPI detection message if not in verbose mode
            if not verbose:
                old_stdout = sys.stdout
                sys.stdout = StringIO()
                try:
                    interface = BayeSEDInterface(mpi_mode='1', np=1)
                finally:
                    sys.stdout = old_stdout
            else:
                interface = BayeSEDInterface(mpi_mode='1', np=1)
            
            # Run the binary multiple times until no new .iprior files are created
            # The binary checks what exists and creates what's missing incrementally
            # Note: Binary may create files with suffixes like .iprior.0, .iprior.4, etc.
            max_iterations = 10  # Safety limit to prevent infinite loops
            iteration = 0
            
            while iteration < max_iterations:
                iteration += 1
                
                # Count existing .iprior files before running (including numbered variants)
                iprior_pattern = os.path.join(input_dir, '*.iprior*')
                iprior_files_before = set(glob.glob(iprior_pattern))
                # Filter to only .iprior files (not .iprior.bak)
                iprior_files_before = {f for f in iprior_files_before if not f.endswith('.bak')}
                num_before = len(iprior_files_before)
                
                if verbose:
                    if iteration == 1:
                        print(f"Iteration {iteration}: Starting with {num_before} existing .iprior file(s)")
                    else:
                        print(f"Iteration {iteration}: {num_before} .iprior file(s) exist, checking for more...")
                
                # Suppress binary output by redirecting stdout/stderr
                old_stdout = sys.stdout
                old_stderr = sys.stderr
                sys.stdout = StringIO()
                sys.stderr = StringIO()
                
                try:
                    # Run BayeSED - it will generate missing .iprior files and then fail
                    interface.run(temp_params, validate=False)
                except Exception as e:
                    # Expected behavior: binary generates files then exits with error
                    pass
                finally:
                    # Restore stdout/stderr
                    sys.stdout = old_stdout
                    sys.stderr = old_stderr
                
                # Count .iprior files after running (including numbered variants)
                iprior_files_after = set(glob.glob(iprior_pattern))
                iprior_files_after = {f for f in iprior_files_after if not f.endswith('.bak')}
                num_after = len(iprior_files_after)
                new_files = iprior_files_after - iprior_files_before
                
                if new_files:
                    if verbose:
                        print(f"  ✓ Generated {len(new_files)} new file(s):")
                        for f in sorted(new_files):
                            print(f"    - {os.path.basename(f)}")
                else:
                    # No new files created - all .iprior files have been generated
                    if verbose:
                        print(f"  ✓ No new files generated - all .iprior files complete")
                    break
            
            if verbose and iteration >= max_iterations:
                print(f"  ⚠ Warning: Reached maximum iterations ({max_iterations})")
            
            # Verify that at least some .iprior files were created
            final_iprior_files = glob.glob(iprior_pattern)
            final_iprior_files = [f for f in final_iprior_files if not f.endswith('.bak')]
            if not final_iprior_files:
                raise RuntimeError(
                    f"BayeSED binary ran but did not generate any .iprior files in {input_dir}. "
                    f"Check that your model configuration is correct."
                )
            
            if verbose:
                print(f"\n✓ Prior file generation complete: {len(final_iprior_files)} total file(s)")
                for f in sorted(final_iprior_files):
                    print(f"  - {os.path.basename(f)}")
                
        finally:
            # Clean up temporary input file
            if os.path.exists(temp_input_file):
                try:
                    os.remove(temp_input_file)
                except:
                    pass  # Ignore cleanup errors
    def set_prior(self, parameter_name=None, iprior_file=None, prior_type=None, is_age=None,
                  min_val=None, max_val=None, nbin=None, hyperparameters=None, confirm=True,
                  reset_to_default=False, component_id=None) -> None:
        """Modify or query priors. See PriorEditor.set_prior (ADR-0006)."""
        if self._editor is None:
            raise ValueError(
                "Must call priors_init() before set_prior(). "
                "Example: inference.priors_init(params)"
            )
        return self._editor.set_prior(
            parameter_name=parameter_name,
            iprior_file=iprior_file,
            prior_type=prior_type,
            is_age=is_age,
            min_val=min_val,
            max_val=max_val,
            nbin=nbin,
            hyperparameters=hyperparameters,
            confirm=confirm,
            reset_to_default=reset_to_default,
            component_id=component_id,
        )

    def print_priors(self, iprior_file: str = None, parameter_names: list = None):
        """Print priors by .iprior file. See PriorEditor.print_priors (ADR-0006)."""
        if self._editor is None:
            raise ValueError(
                "Must call priors_init() before print_priors(). "
                "Example: inference.priors_init(params)"
            )
        return self._editor.print_priors(
            iprior_file=iprior_file,
            parameter_names=parameter_names,
        )

    def validate_priors(self):
        """
        Validate all priors for the current configuration.
        
        Returns
        -------
        dict
            Dictionary mapping parameter names to lists of error messages.
            Empty lists indicate valid priors.
        
        Raises
        ------
        ValueError
            If priors_init() has not been called
        
        Examples
        --------
        >>> inference.priors_init(params)
        >>> errors = inference.validate_priors()
        >>> if any(errors.values()):
        ...     print("Validation errors found!")
        """
        if self._editor is None or self._prior_manager is None:
            raise ValueError(
                "Must call priors_init() before validate_priors(). "
                "Example: inference.priors_init(params)"
            )
        
        return self._prior_manager.validate_all()
    def list_prior_types(self) -> None:
        """
        Print a table of all available prior types with their hyperparameter requirements.
        
        Displays a formatted table showing all supported prior types, their names,
        required hyperparameters (including parameter names), and brief descriptions.
        
        Examples
        --------
        >>> inference = SEDInference()
        >>> inference.list_prior_types()
        """
        print("\n" + "=" * 100)
        print("Available Prior Types")
        print("=" * 100)
        print(f"{'Type':<6} {'Name':<18} {'Hyperparams':<30} {'Description'}")
        print("-" * 100)
        
        type_info = [
            (0, "Mirror", 3, "[model_id, par_type, par_id]", "Mirror prior (references other parameters)"),
            (1, "Uniform", 0, "[]", "Uniform prior"),
            (2, "Linear-Inc", 0, "[]", "Linear increasing prior"),
            (3, "Linear-Dec", 0, "[]", "Linear decreasing prior"),
            (4, "TruncGaussian", 2, "[mu, sigma]", "Truncated Gaussian distribution"),
            (5, "Gaussian", 2, "[mu, sigma]", "Gaussian (normal) distribution"),
            (6, "Gamma", 2, "[alpha, beta]", "Gamma distribution"),
            (7, "StudentT", 3, "[mu, sigma, nu]", "Student's t-distribution"),
            (8, "Beta", 2, "[a, b]", "Beta distribution"),
            (9, "Weibull", 2, "[alpha, beta]", "Weibull distribution"),
        ]
        
        for type_id, name, n_hyper, hyper_names, desc in type_info:
            print(f"{type_id:<6} {name:<18} {hyper_names:<30} {desc}")
        
        print("=" * 100)
        print("\nNote: Negative types (e.g., -1, -5) indicate log10 space")
        print("      For string names, use 'Log10_' prefix (e.g., 'Log10_Gaussian')")
        print("\nHyperparameter meanings:")
        print("  mu    = Mean of the distribution")
        print("  sigma = Standard deviation (must be positive)")
        print("  alpha = Shape parameter (must be positive)")
        print("  beta  = Rate/scale parameter (must be positive)")
        print("  nu    = Degrees of freedom (must be positive)")
        print("  a, b  = Shape parameters (must be positive)")
        print("\nMirror prior (type 0) - Special hyperparameters:")
        print("  model_id = Model component ID to reference (-1 for input file data)")
        print("  par_type = Parameter type to mirror:")
        print("             0: Other information in input file (model_id must be -1)")
        print("             1: Free parameters of model with given model_id")
        print("             2: Derived parameters of model with given model_id")
        print("  par_id   = Parameter ID within the selected model/type")
