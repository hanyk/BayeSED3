"""
Bayesian inference configuration and execution for BayeSED3.

This module provides the SEDInference class for configuring Bayesian inference settings
(MultiNest, GSL, NNLM, etc.) and executing fits.
"""

from dataclasses import dataclass
from typing import Optional

# Import parameter classes directly from source modules to avoid circular dependencies
# TYPE_CHECKING is used for type hints to avoid importing at module level
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Type hints only - imports happen inside methods to avoid circular dependency
    from .params import (
        MultiNestParams, GSLIntegrationQAGParams, GSLMultifitRobustParams,
        NNLMParams, NdumperParams
    )
    from .core import BayeSEDParams, BayeSEDInterface
    from .results import BayeSEDResults


class SEDInference:
    """
    Handles Bayesian inference configuration and execution for BayeSED3.
    
    This class provides a clean interface for configuring inference settings
    (MultiNest, GSL, NNLM, etc.) and executing fits using BayeSEDParams.
    
    Examples
    --------
    >>> from bayesed.inference import SEDInference
    >>> from bayesed import BayeSEDParams
    >>> 
    >>> # Configure inference settings
    >>> inference = SEDInference()
    >>> inference.multinest(nlive=500, tol=0.5)
    >>> inference.gsl(epsrel=0.1)
    >>> inference.nnlm(method=0, Niter1=10000)
    >>> 
    >>> # Execute fit
    >>> params = BayeSEDParams(...)  # Configured with data and model
    >>> results = inference.run(params)
    >>> 
    >>> # Access results
    >>> posterior = results.posterior
    >>> best_fit = results.best_fit
    >>> evidence = results.evidence
    """
    
    def __init__(self):
        """Initialize SEDInference instance."""
        self._multinest = None
        self._gsl_integration_qag = None
        self._gsl_multifit_robust = None
        self._nnlm = None
        self._ndumper = None
        self._logzero = None
        self._unweighted_samples = None
        self._nsample = None
        self._niteration = None
        # Prior management
        self._params = None
        self._prior_manager = None
    
    def multinest(self, INS=1, mmodal=0, ceff=0, nlive=100, efr=0.1, tol=0.5,
                  updInt=1000, Ztol=-1e90, seed=1, fb=0, resume=0, outfile=0):
        """
        Configure MultiNest nested sampling parameters.
        
        Parameters
        ----------
        INS : int, default 1
            Importance Nested Sampling flag
        mmodal : int, default 0
            Multimodal flag
        ceff : int, default 0
            Constant efficiency mode flag
        nlive : int, default 100
            Number of live points
        efr : float, default 0.1
            Sampling efficiency
        tol : float, default 0.5
            Tolerance
        updInt : int, default 1000
            Update interval
        Ztol : float, default -1e90
            Evidence tolerance
        seed : int, default 1
            Random seed
        fb : int, default 0
            Feedback level
        resume : int, default 0
            Resume from a previous run
        outfile : int, default 0
            Write output files
        
        Returns
        -------
        self
            Returns self for method chaining
        """
        # Import here to avoid circular dependency
        from .params import MultiNestParams
        self._multinest = MultiNestParams(
            INS=INS, mmodal=mmodal, ceff=ceff, nlive=nlive, efr=efr, tol=tol,
            updInt=updInt, Ztol=Ztol, seed=seed, fb=fb, resume=resume, outfile=outfile
        )
        return self
    
    def gsl(self, epsabs=0.0, epsrel=0.1, limit=1000, key=1,
            robust_type="ols", robust_tune=1.0):
        """
        Configure GSL integration and robust fitting parameters.
        
        Parameters
        ----------
        epsabs : float, default 0.0
            Absolute error for GSL integration
        epsrel : float, default 0.1
            Relative error for GSL integration
        limit : int, default 1000
            Limit on the number of subintervals for GSL integration
        key : int, default 1
            Key for the integration rule
        robust_type : str, default "ols"
            Type of robust fitting: "ols" or "huber"
        robust_tune : float, default 1.0
            Tuning parameter for robust fitting
        
        Returns
        -------
        self
            Returns self for method chaining
        """
        # Import here to avoid circular dependency
        from .params import GSLIntegrationQAGParams, GSLMultifitRobustParams
        self._gsl_integration_qag = GSLIntegrationQAGParams(
            epsabs=epsabs, epsrel=epsrel, limit=limit, key=key
        )
        self._gsl_multifit_robust = GSLMultifitRobustParams(
            type=robust_type, tune=robust_tune
        )
        return self
    
    def nnlm(self, method=0, Niter1=10000, tol1=0.0, Niter2=10, tol2=0.01,
             p1=0.05, p2=0.95):
        """
        Configure NNLM (Non-Negative Least Mean) parameters.
        
        Parameters
        ----------
        method : int, default 0
            Method: 0=eazy, 1=scd, 2=lee_ls, 3=scd_kl, 4=lee_kl
        Niter1 : int, default 10000
            Number of iterations for first step
        tol1 : float, default 0.0
            Tolerance for first step
        Niter2 : int, default 10
            Number of iterations for second step
        tol2 : float, default 0.01
            Tolerance for second step
        p1 : float, default 0.05
            Parameter p1
        p2 : float, default 0.95
            Parameter p2
        
        Returns
        -------
        self
            Returns self for method chaining
        """
        # Import here to avoid circular dependency
        from .params import NNLMParams
        self._nnlm = NNLMParams(
            method=method, Niter1=Niter1, tol1=tol1, Niter2=Niter2, tol2=tol2,
            p1=p1, p2=p2
        )
        return self
    
    def ndumper(self, max_number=1, iconverged_min=0, Xmin_squared_Nd=-1.0):
        """
        Configure MultiNest dumper parameters.
        
        Parameters
        ----------
        max_number : int, default 1
            Maximum number
        iconverged_min : int, default 0
            Minimum convergence flag
        Xmin_squared_Nd : float, default -1.0
            Xmin^2/Nd value
        
        Returns
        -------
        self
            Returns self for method chaining
        """
        # Import here to avoid circular dependency
        from .params import NdumperParams
        self._ndumper = NdumperParams(
            max_number=max_number,
            iconverged_min=iconverged_min,
            Xmin_squared_Nd=Xmin_squared_Nd
        )
        return self
    
    def set_logzero(self, logzero):
        """
        Set logZero (maximum allowed Nsigma for MultiNest).
        
        Parameters
        ----------
        logzero : float
            Maximum allowed Nsigma for MultiNest
        
        Returns
        -------
        self
            Returns self for method chaining
        """
        self._logzero = logzero
        return self
    
    def set_unweighted_samples(self, unweighted_samples: bool):
        """
        Set whether to use unweighted samples.
        
        Parameters
        ----------
        unweighted_samples : bool
            Whether to use unweighted samples
        
        Returns
        -------
        self
            Returns self for method chaining
        """
        self._unweighted_samples = unweighted_samples
        return self
    
    def set_nsample(self, nsample: int):
        """
        Set number of samples.
        
        Parameters
        ----------
        nsample : int
            Number of samples
        
        Returns
        -------
        self
            Returns self for method chaining
        """
        self._nsample = nsample
        return self
    
    def set_niteration(self, niteration: int):
        """
        Set number of iterations.
        
        Parameters
        ----------
        niteration : int
            Number of iterations
        
        Returns
        -------
        self
            Returns self for method chaining
        """
        self._niteration = niteration
        return self
    
    def priors_init(self, params: 'BayeSEDParams', force_regenerate: bool = False, verbose: bool = False) -> None:
        """
        Initialize priors for the given parameters.
        
        This method loads .iprior files from the input directory and makes them
        available for inspection and modification. It intelligently handles file
        generation and loading based on what exists and what's needed.
        
        Key Features
        ------------
        1. **Automatic file generation**: If no .iprior files exist, generates them automatically
        2. **Fresh defaults**: Generates pristine default values for reset_to_default functionality
        3. **Relevant file filtering**: Only loads files relevant to your model configuration
        4. **Force regenerate**: Can regenerate files even if they exist (overwrites with fresh)
        5. **Quiet by default**: Minimal output unless verbose=True
        
        Behavior
        --------
        - **No files exist**: Generates files in input directory, uses as defaults
        - **Files exist**: Keeps existing files, generates pristine defaults in temp directory
        - **force_regenerate=True**: Generates fresh files in temp dir, copies to input dir (overwrites)
        
        The fresh defaults are generated using the SAME model configuration as your
        params, ensuring defaults match your actual model setup. Only files relevant
        to your model are loaded, even if the directory contains files for other models.
        
        Parameters
        ----------
        params : BayeSEDParams
            Model configuration parameters (stored internally for later use).
            The SAME configuration is used to generate fresh defaults.
        force_regenerate : bool, optional
            If True, regenerate .iprior files even if they exist. Generates fresh
            files in a temp directory and copies them to input directory, overwriting
            existing files. Only overwrites files relevant to current model config.
            Default is False.
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
        
        >>> inference.priors_init(params, force_regenerate=True)
        # Overwrites existing files with fresh ones
        
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
        - force_regenerate works by generating in temp dir and copying with overwrite
        - Only files relevant to your model configuration are loaded
        - Fresh defaults are always generated to support reset_to_default functionality
        
        See Also
        --------
        set_prior : Modify prior values
        print_priors : Display current prior configuration
        reset_to_default : Reset priors to default values (requires priors_init first)
        """
        import os
        import glob
        import tempfile
        import shutil
        from .prior_manager import PriorManager
        
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
        
        # Find all .iprior files in the directory
        iprior_pattern = os.path.join(input_dir, '*.iprior*')
        all_iprior_files = glob.glob(iprior_pattern)
        
        # Filter to only accept valid .iprior files:
        # - Files ending with .iprior exactly
        # - Files ending with .iprior.X where X is a digit (e.g., .iprior.0, .iprior.4)
        # Reject files like .iprior.bak, .iprior.test_backup, etc.
        import re
        valid_iprior_pattern = re.compile(r'.*\.iprior(\.\d+)?$')
        all_iprior_files = [f for f in all_iprior_files if valid_iprior_pattern.match(os.path.basename(f))]
        
        # Prefer .iprior.X files over base .iprior files
        # Group files by base name (without .X suffix)
        from collections import defaultdict
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
        # already exist in the directory. So for force_regenerate, we generate in
        # a temp directory and then copy with overwrite.
        
        files_existed_before = bool(iprior_files)
        
        if force_regenerate and iprior_files:
            # User wants to force regenerate, but files exist
            # Binary won't regenerate if files exist, so generate in temp dir and copy
            if verbose:
                print(f"Force regenerating .iprior files in {input_dir}...")
            
            import tempfile
            import shutil
            
            # Generate in temp directory
            with tempfile.TemporaryDirectory() as temp_dir:
                try:
                    self._generate_prior_files(params, temp_dir, verbose=verbose)
                    
                    # Find generated files in temp dir
                    temp_pattern = os.path.join(temp_dir, '*.iprior*')
                    temp_files = glob.glob(temp_pattern)
                    temp_files = [f for f in temp_files if valid_iprior_pattern.match(os.path.basename(f))]
                    
                    if verbose:
                        print(f"  Copying {len(temp_files)} file(s) to {input_dir} (overwriting existing)...")
                    
                    # Copy files from temp to input dir (overwrite existing)
                    for temp_file in temp_files:
                        dest_file = os.path.join(input_dir, os.path.basename(temp_file))
                        shutil.copy2(temp_file, dest_file)
                        if verbose:
                            print(f"    Copied: {os.path.basename(temp_file)}")
                    
                except Exception as e:
                    raise ValueError(f"Failed to force regenerate .iprior files: {e}") from e
            
            # Re-scan for .iprior files after copying
            iprior_files = glob.glob(iprior_pattern)
            iprior_files = [f for f in iprior_files if valid_iprior_pattern.match(os.path.basename(f))]
            files_existed_before = False  # Treat as fresh (we just regenerated)
        
        if not iprior_files:
            # No files exist - generate them
            if verbose:
                print(f"No .iprior files found in {input_dir}. Generating them automatically...")
            
            try:
                self._generate_prior_files(params, input_dir, verbose=verbose)
                # Re-scan for .iprior files after generation
                iprior_files = glob.glob(iprior_pattern)
                # Apply same validation: only .iprior or .iprior.X (where X is digit)
                iprior_files = [f for f in iprior_files if valid_iprior_pattern.match(os.path.basename(f))]
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
        
        # Initialize PriorManager
        self._prior_manager = PriorManager(base_directory=input_dir)
        
        # Generate fresh defaults and get list of relevant files
        # This serves two purposes:
        # 1. Get pristine default values for reset_to_default functionality
        # 2. Determine which .iprior files are relevant for current model config
        #
        # Logic for when to use temp directory:
        # - If files existed before (and weren't force-regenerated): Use temp dir
        #   Reason: Files might be user-modified, need pristine defaults
        # - If files didn't exist (or were force-regenerated): Use generated files directly
        #   Reason: Files are fresh from binary, already pristine
        
        relevant_filenames = set()
        
        if files_existed_before:
            # Files existed and weren't regenerated - might be user-modified
            # Generate fresh defaults in temp directory to get pristine values
            try:
                relevant_filenames = self._generate_fresh_defaults(params, verbose=verbose)
                if verbose:
                    print("✓ Fresh defaults generated")
                    if relevant_filenames:
                        print(f"✓ Identified {len(relevant_filenames)} relevant file(s) for current model config")
            except Exception as e:
                import logging
                logging.getLogger(__name__).warning(
                    f"Failed to generate fresh defaults: {e}. "
                    f"reset_to_default functionality will not be available."
                )
        else:
            # Files didn't exist (or were force-regenerated) - just generated fresh
            # Use the generated files directly as defaults (no temp directory needed)
            if verbose:
                print("✓ Using newly generated files as defaults (no temp directory needed)")
            # Load the freshly generated files as defaults
            for iprior_file in iprior_files:
                filename = os.path.basename(iprior_file)
                relevant_filenames.add(filename)
                try:
                    self._prior_manager.load_prior_file(filename, is_auto_generated=True)
                except Exception as e:
                    import logging
                    logging.getLogger(__name__).warning(f"Failed to load {filename} as default: {e}")
        
        # Filter to only load relevant .iprior files (matching current model config)
        if relevant_filenames:
            # We know which files are relevant - only load those
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
                import logging
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
    
    def _generate_fresh_defaults(self, params: 'BayeSEDParams', verbose: bool = False) -> set:
        """
        Generate fresh default priors in a temporary directory.
        
        This method creates a temporary directory, generates .iprior files there
        using the BayeSED binary with the SAME model configuration as params,
        and loads them as the true defaults. This ensures that defaults are not
        affected by any manual modifications to .iprior files in the user's
        input directory, and that defaults match the actual model configuration.
        
        Parameters
        ----------
        params : BayeSEDParams
            Model configuration parameters. The SAME configuration (model components,
            settings, etc.) is used to generate defaults, ensuring they match your
            actual model setup.
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
        import os
        import tempfile
        import glob
        import re
        from .prior import Prior
        
        if verbose:
            print("Generating fresh defaults...")
        
        relevant_filenames = set()
        
        # Create temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            # Generate .iprior files in temp directory
            # _generate_prior_files already handles:
            #   - Creating empty input file
            #   - Deep copying params
            #   - Running binary with same model config
            try:
                self._generate_prior_files(params, temp_dir, verbose=verbose)
            except Exception as e:
                raise RuntimeError(f"Failed to generate fresh defaults: {e}")
            
            # Find all generated .iprior files (with proper filtering)
            iprior_pattern = os.path.join(temp_dir, '*.iprior*')
            temp_iprior_files = glob.glob(iprior_pattern)
            
            # Filter to only valid .iprior files (ending with .iprior or .iprior.X where X is digits)
            iprior_regex = re.compile(r'.*\.iprior(\.\d+)?$')
            temp_iprior_files = [f for f in temp_iprior_files if iprior_regex.match(f)]
            
            if not temp_iprior_files:
                raise RuntimeError("No .iprior files generated in temporary directory")
            
            # Collect relevant filenames
            for iprior_file in temp_iprior_files:
                relevant_filenames.add(os.path.basename(iprior_file))
            
            # Load these as defaults (original_priors)
            for iprior_file in temp_iprior_files:
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
                print(f"✓ Loaded {len(self._prior_manager.original_priors)} default priors")
        
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
        import os
        import tempfile
        import subprocess
        import shutil
        from .core import BayeSEDInterface
        
        if verbose:
            print("Generating .iprior files by running BayeSED binary...")
        
        # Create a temporary empty input file in the target directory
        # This ensures .iprior files are generated in the correct location
        temp_input_file = os.path.join(input_dir, '.temp_empty_input.txt')
        
        try:
            # Create empty input file with just a header
            with open(temp_input_file, 'w') as f:
                f.write("# Temporary empty input file for prior generation\n")
                f.write("# id z\n")
                # No data rows - this will cause "No valid objects" error
            
            # Create a copy of params with the temporary input file
            import copy
            temp_params = copy.deepcopy(params)
            temp_params.input_file = temp_input_file
            
            # Set priors_only flag to speed up generation (if supported)
            temp_params.priors_only = True
            
            # Create BayeSED interface with minimal settings for prior generation
            # Suppress OpenMPI detection message if not in verbose mode
            if not verbose:
                import sys
                from io import StringIO
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
                import glob
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
                import sys
                from io import StringIO
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
                  reset_to_default=False) -> None:
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
        import os
        import re
        from .prior import Prior
        
        # Check that priors_init() was called
        if self._prior_manager is None:
            raise ValueError(
                "Must call priors_init() before set_prior(). "
                "Example: inference.priors_init(params)"
            )
        
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
            
            # Show BEFORE and AFTER preview in table format
            for filename in sorted(by_file.keys()):
                print(f"\nFile: {filename}")
                print("-" * 80)
                
                # Get all priors for this file
                file_priors = self._prior_manager.priors_by_file[filename]
                matched_params = by_file[filename]
                
                # Filter to only matched parameters
                filtered_priors = {name: file_priors[name] for name in matched_params if name in file_priors}
                
                # Create preview of what parameters will look like after modifications
                preview_priors = {}
                for param_name, prior in filtered_priors.items():
                    # Create a copy with modifications applied
                    import copy
                    preview = copy.copy(prior)
                    for key, value in modifications.items():
                        if hasattr(preview, key):
                            setattr(preview, key, value)
                    preview_priors[param_name] = preview
                
                # Determine column widths dynamically
                max_name_len = max(len(name) for name in preview_priors.keys()) if preview_priors else 10
                max_type_len = max(len(p.get_type_name()) for p in preview_priors.values()) if preview_priors else 10
                max_name_len = max(max_name_len, len("Name"))
                max_type_len = max(max_type_len, len("Type"))
                
                # Calculate dynamic widths for numeric columns
                max_isage_len = max(len(str(p.is_age)) for p in preview_priors.values()) if preview_priors else 1
                max_isage_len = max(max_isage_len, len("IsAge"))
                
                max_min_len = max(len(f"{p.min_val:.6g}") for p in preview_priors.values()) if preview_priors else 3
                max_min_len = max(max_min_len, len("Min"))
                
                max_max_len = max(len(f"{p.max_val:.6g}") for p in preview_priors.values()) if preview_priors else 3
                max_max_len = max(max_max_len, len("Max"))
                
                max_nbin_len = max(len(str(p.nbin)) for p in preview_priors.values()) if preview_priors else 4
                max_nbin_len = max(max_nbin_len, len("Nbin"))
                
                max_modified_len = len("Preview")  # All will show "Preview"
                max_modified_len = max(max_modified_len, len("Modified"))
                
                # Always show hyperparameters column for consistency with print_priors
                has_hyperparams = True
                
                # Build header (consistent with print_priors format)
                header = f"{'Name':<{max_name_len}}  {'Type':<{max_type_len}}  {'IsAge':<{max_isage_len}}  {'Min':<{max_min_len}}  {'Max':<{max_max_len}}  {'Nbin':<{max_nbin_len}}  {'Modified':<{max_modified_len}}  Hyperparameters"
                
                print(header)
                print("-" * len(header))
                
                # Display each parameter with PREVIEW of changes
                for param_name in sorted(preview_priors.keys()):
                    preview = preview_priors[param_name]
                    
                    # Format values (consistent with print_priors)
                    min_str = f"{preview.min_val:.6g}"
                    max_str = f"{preview.max_val:.6g}"
                    
                    # Show "Preview" in Modified column to indicate this is a preview
                    mod_marker = "Preview"
                    
                    # Build row
                    row = f"{param_name:<{max_name_len}}  {preview.get_type_name():<{max_type_len}}  {preview.is_age:<{max_isage_len}}  {min_str:<{max_min_len}}  {max_str:<{max_max_len}}  {preview.nbin:<{max_nbin_len}}  {mod_marker:<{max_modified_len}}"
                    
                    # Always add hyperparameters column for consistency
                    if preview.hyperparameters:
                        names = preview.get_hyperparameter_names()
                        hyper_str = ", ".join(f"{n}={v}" for n, v in zip(names, preview.hyperparameters))
                        row += f"  {hyper_str}"
                    else:
                        row += "  -"
                    
                    print(row)
            
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
                
                # Show preview table with default values
                for filename in sorted(by_file.keys()):
                    print(f"\nFile: {filename}")
                    print("-" * 80)
                    
                    # Get all priors for this file
                    file_priors = self._prior_manager.priors_by_file[filename]
                    matched_params = by_file[filename]
                    
                    # Filter to only matched parameters
                    filtered_priors = {name: file_priors[name] for name in matched_params if name in file_priors}
                    
                    # Create preview with DEFAULT values
                    preview_priors = {}
                    for param_name in filtered_priors.keys():
                        default_prior = self._prior_manager.original_priors[param_name]
                        preview_priors[param_name] = default_prior
                    
                    # Determine column widths dynamically
                    max_name_len = max(len(name) for name in preview_priors.keys()) if preview_priors else 10
                    max_type_len = max(len(p.get_type_name()) for p in preview_priors.values()) if preview_priors else 10
                    max_name_len = max(max_name_len, len("Name"))
                    max_type_len = max(max_type_len, len("Type"))
                    
                    # Calculate dynamic widths for numeric columns
                    max_isage_len = max(len(str(p.is_age)) for p in preview_priors.values()) if preview_priors else 1
                    max_isage_len = max(max_isage_len, len("IsAge"))
                    
                    max_min_len = max(len(f"{p.min_val:.6g}") for p in preview_priors.values()) if preview_priors else 3
                    max_min_len = max(max_min_len, len("Min"))
                    
                    max_max_len = max(len(f"{p.max_val:.6g}") for p in preview_priors.values()) if preview_priors else 3
                    max_max_len = max(max_max_len, len("Max"))
                    
                    max_nbin_len = max(len(str(p.nbin)) for p in preview_priors.values()) if preview_priors else 4
                    max_nbin_len = max(max_nbin_len, len("Nbin"))
                    
                    max_modified_len = len("Preview")  # All will show "Preview"
                    max_modified_len = max(max_modified_len, len("Modified"))
                    
                    # Check if any priors have hyperparameters
                    has_hyperparams = any(p.hyperparameters for p in preview_priors.values())
                    
                    # Build header
                    if has_hyperparams:
                        header = f"{'Name':<{max_name_len}}  {'Type':<{max_type_len}}  {'IsAge':<{max_isage_len}}  {'Min':<{max_min_len}}  {'Max':<{max_max_len}}  {'Nbin':<{max_nbin_len}}  {'Modified':<{max_modified_len}}  Hyperparameters"
                    else:
                        header = f"{'Name':<{max_name_len}}  {'Type':<{max_type_len}}  {'IsAge':<{max_isage_len}}  {'Min':<{max_min_len}}  {'Max':<{max_max_len}}  {'Nbin':<{max_nbin_len}}  {'Modified':<{max_modified_len}}"
                    
                    print(header)
                    print("-" * len(header))
                    
                    # Display each parameter with DEFAULT values
                    for param_name in sorted(preview_priors.keys()):
                        preview = preview_priors[param_name]
                        
                        # Format values
                        min_str = f"{preview.min_val:.6g}"
                        max_str = f"{preview.max_val:.6g}"
                        
                        # Show "Preview" in Modified column
                        mod_marker = "Preview"
                        
                        # Build row
                        row = f"{param_name:<{max_name_len}}  {preview.get_type_name():<{max_type_len}}  {preview.is_age:<{max_isage_len}}  {min_str:<{max_min_len}}  {max_str:<{max_max_len}}  {preview.nbin:<{max_nbin_len}}  {mod_marker:<{max_modified_len}}"
                        
                        # Add hyperparameters if present
                        if has_hyperparams:
                            if preview.hyperparameters:
                                names = preview.get_hyperparameter_names()
                                hyper_str = ", ".join(f"{n}={v}" for n, v in zip(names, preview.hyperparameters))
                                row += f"  {hyper_str}"
                            else:
                                row += "  -"
                        
                        print(row)
                
                
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
                    import logging
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
                
                # Display each file's matched parameters in table format
                for filename in sorted(by_file.keys()):
                    print(f"\nFile: {filename}")
                    print("-" * 80)
                    
                    # Get all priors for this file
                    file_priors = self._prior_manager.priors_by_file[filename]
                    matched_params = by_file[filename]
                    
                    # Filter to only matched parameters
                    filtered_priors = {name: file_priors[name] for name in matched_params if name in file_priors}
                    
                    # Use PriorManager's display method for consistent formatting
                    # Create a temporary display just for these parameters
                    display_lines = []
                    
                    # Determine column widths dynamically
                    max_name_len = max(len(name) for name in filtered_priors.keys()) if filtered_priors else 10
                    max_type_len = max(len(p.get_type_name()) for p in filtered_priors.values()) if filtered_priors else 10
                    max_name_len = max(max_name_len, len("Name"))
                    max_type_len = max(max_type_len, len("Type"))
                    
                    # Calculate dynamic widths for numeric columns
                    max_isage_len = max(len(str(p.is_age)) for p in filtered_priors.values()) if filtered_priors else 1
                    max_isage_len = max(max_isage_len, len("IsAge"))
                    
                    max_min_len = max(len(f"{p.min_val:.6g}") for p in filtered_priors.values()) if filtered_priors else 3
                    max_min_len = max(max_min_len, len("Min"))
                    
                    max_max_len = max(len(f"{p.max_val:.6g}") for p in filtered_priors.values()) if filtered_priors else 3
                    max_max_len = max(max_max_len, len("Max"))
                    
                    max_nbin_len = max(len(str(p.nbin)) for p in filtered_priors.values()) if filtered_priors else 4
                    max_nbin_len = max(max_nbin_len, len("Nbin"))
                    
                    # Calculate Modified column width based on possible values
                    mod_markers = []
                    for param_name in filtered_priors.keys():
                        is_modified = param_name in self._prior_manager.modified_parameters
                        is_recent = param_name in getattr(self._prior_manager, 'recently_modified', set())
                        if is_modified and is_recent:
                            mod_markers.append("*#")
                        elif is_modified:
                            mod_markers.append("*")
                        elif is_recent:
                            mod_markers.append("#")
                        else:
                            mod_markers.append("No")
                    max_modified_len = max(len(m) for m in mod_markers) if mod_markers else 2
                    max_modified_len = max(max_modified_len, len("Modified"))
                    
                    # Check if any priors have hyperparameters
                    has_hyperparams = any(p.hyperparameters for p in filtered_priors.values())
                    
                    # Build header (consistent with print_priors format)
                    if has_hyperparams:
                        header = f"{'Name':<{max_name_len}}  {'Type':<{max_type_len}}  {'IsAge':<{max_isage_len}}  {'Min':<{max_min_len}}  {'Max':<{max_max_len}}  {'Nbin':<{max_nbin_len}}  {'Modified':<{max_modified_len}}  Hyperparameters"
                    else:
                        header = f"{'Name':<{max_name_len}}  {'Type':<{max_type_len}}  {'IsAge':<{max_isage_len}}  {'Min':<{max_min_len}}  {'Max':<{max_max_len}}  {'Nbin':<{max_nbin_len}}  {'Modified':<{max_modified_len}}"
                    
                    print(header)
                    print("-" * len(header))
                    
                    # Display each parameter
                    for param_name in sorted(filtered_priors.keys()):
                        prior = filtered_priors[param_name]
                        
                        # Determine modification status
                        is_modified = param_name in self._prior_manager.modified_parameters
                        is_recent = param_name in getattr(self._prior_manager, 'recently_modified', set())
                        
                        if is_modified and is_recent:
                            mod_marker = "*#"
                        elif is_modified:
                            mod_marker = "*"
                        elif is_recent:
                            mod_marker = "#"
                        else:
                            mod_marker = "No"
                        
                        # Format values (consistent with print_priors)
                        min_str = f"{prior.min_val:.6g}"
                        max_str = f"{prior.max_val:.6g}"
                        
                        # Build row
                        row = f"{param_name:<{max_name_len}}  {prior.get_type_name():<{max_type_len}}  {prior.is_age:<{max_isage_len}}  {min_str:<{max_min_len}}  {max_str:<{max_max_len}}  {prior.nbin:<{max_nbin_len}}  {mod_marker:<{max_modified_len}}"
                        
                        # Add hyperparameters if present
                        if has_hyperparams:
                            if prior.hyperparameters:
                                names = prior.get_hyperparameter_names()
                                hyper_str = ", ".join(f"{n}={v}" for n, v in zip(names, prior.hyperparameters))
                                row += f"  {hyper_str}"
                            else:
                                row += "  -"
                        
                        print(row)
                
                # Add legend
                print("\nLegend: * = modified from default, # = just modified, *# = both")
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
                        # No matches found - provide helpful error
                        all_params = set()
                        for filename, params in self._prior_manager.priors_by_file.items():
                            all_params.update(params.keys())
                        available = ', '.join(sorted(all_params))
                        raise KeyError(
                            f"No parameters match pattern '{param_name}'. "
                            f"Available parameters: {available}"
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
                import logging
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
        ValueError
            If priors_init() has not been called
        FileNotFoundError
            If the specified iprior_file is not found
        
        Examples
        --------
        >>> inference.priors_init(params)
        >>> files = inference.print_priors()  # Print all files and get list
        >>> print(files)
        ['2dal7.iprior', 'BLR.iprior', 'FeII.iprior', ...]
        >>> inference.print_priors(iprior_file='2dal8.iprior')  # Print specific file
        """
        import os
        
        if self._prior_manager is None:
            raise ValueError(
                "Must call priors_init() before print_priors(). "
                "Example: inference.priors_init(params)"
            )
        
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
        
        # Print each file separately
        for filename in sorted(params_by_file.keys()):
            params_dict = params_by_file[filename]
            params = [(name, prior) for name, prior in params_dict.items()]
            
            print(f"\nFile: {filename}")
            print("-" * 80)
            
            # Calculate column widths dynamically for this file
            name_width = max(len(p[0]) for p in params)
            name_width = max(name_width, len("Name"))
            
            type_width = max(len(p[1].get_type_name()) for p in params)
            type_width = max(type_width, len("Type"))
            
            # Calculate dynamic widths for numeric columns
            isage_width = max(len(str(p[1].is_age)) for p in params)
            isage_width = max(isage_width, len("IsAge"))
            
            min_width = max(len(f"{p[1].min_val:.6g}") for p in params)
            min_width = max(min_width, len("Min"))
            
            max_width = max(len(f"{p[1].max_val:.6g}") for p in params)
            max_width = max(max_width, len("Max"))
            
            nbin_width = max(len(str(p[1].nbin)) for p in params)
            nbin_width = max(nbin_width, len("Nbin"))
            
            # Calculate Modified column width based on possible values
            mod_markers = []
            for name, prior in params:
                is_modified = prior.name in self._prior_manager.modified_parameters
                is_recent = prior.name in getattr(self._prior_manager, 'recently_modified', set())
                if is_modified and is_recent:
                    mod_markers.append("*#")
                elif is_modified:
                    mod_markers.append("*")
                elif is_recent:
                    mod_markers.append("#")
                else:
                    mod_markers.append("No")
            modified_width = max(len(m) for m in mod_markers) if mod_markers else 2
            modified_width = max(modified_width, len("Modified"))
            
            # Calculate hyperparameter column width
            hyper_strings = []
            for name, prior in params:
                if prior.hyperparameters:
                    names = prior.get_hyperparameter_names()
                    hyper_str = ", ".join(f"{n}={v}" for n, v in zip(names, prior.hyperparameters))
                    hyper_strings.append(hyper_str)
                else:
                    hyper_strings.append("-")
            
            hyper_width = max(len(s) for s in hyper_strings) if hyper_strings else 1
            hyper_width = max(hyper_width, len("Hyperparameters"))
            
            # Print header for this file
            header = f"{'Name':<{name_width}}  {'Type':<{type_width}}  {'IsAge':<{isage_width}}  {'Min':<{min_width}}  {'Max':<{max_width}}  {'Nbin':<{nbin_width}}  {'Modified':<{modified_width}}  {'Hyperparameters':<{hyper_width}}"
            print(header)
            print("-" * len(header))
            
            # Print each parameter (preserve order from .iprior file)
            for name, prior in params:
                # Format hyperparameters
                if prior.hyperparameters:
                    names = prior.get_hyperparameter_names()
                    hyper_str = ", ".join(f"{n}={v}" for n, v in zip(names, prior.hyperparameters))
                else:
                    hyper_str = "-"
                
                # Format modification status with markers
                is_modified = prior.name in self._prior_manager.modified_parameters
                is_recent = prior.name in getattr(self._prior_manager, 'recently_modified', set())
                
                if is_modified and is_recent:
                    mod_marker = "*#"
                elif is_modified:
                    mod_marker = "*"
                elif is_recent:
                    mod_marker = "#"
                else:
                    mod_marker = "No"
                
                # Format numeric values
                min_str = f"{prior.min_val:.6g}"
                max_str = f"{prior.max_val:.6g}"
                
                line = f"{prior.name:<{name_width}}  {prior.get_type_name():<{type_width}}  "
                line += f"{prior.is_age:<{isage_width}}  {min_str:<{min_width}}  {max_str:<{max_width}}  {prior.nbin:<{nbin_width}}  "
                line += f"{mod_marker:<{modified_width}}  {hyper_str:<{hyper_width}}"
                
                print(line)
        
        # Add legend if any modifications are shown
        has_modifications = any(
            param_name in self._prior_manager.modified_parameters or 
            param_name in getattr(self._prior_manager, 'recently_modified', set())
            for params_dict in params_by_file.values()
            for param_name in params_dict.keys()
        )
        
        if has_modifications:
            print("\nLegend: * = modified from default, # = just modified, *# = both")
        
        print("=" * 80)
        
        # Return list of files if printing all, None otherwise
        return sorted(self._prior_manager.priors_by_file.keys()) if print_all else None
    
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
        if self._prior_manager is None:
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
            (0, "Mirror", 3, "[mu, sigma, nu]", "Mirror prior (special case)"),
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
    
    def run(self, params: 'BayeSEDParams', validate=True, auto_select_mpi_mode=None,
            mpi_mode=None, np=None, Ntest=None):
        """
        Apply inference settings to BayeSEDParams and execute fit.
        
        This method applies all configured inference settings to the BayeSEDParams object,
        then executes the fit using BayeSEDInterface.run().
        
        Parameters
        ----------
        params : BayeSEDParams
            BayeSEDParams object configured with data and model
        validate : bool, default True
            Whether to validate parameters before execution
        auto_select_mpi_mode : bool, optional
            Whether to automatically select MPI mode
        mpi_mode : str, optional
            MPI mode: '1' for bayesed_mn_1, 'n' for bayesed_mn_n
        np : int, optional
            Number of MPI processes
        Ntest : int, optional
            Number of test objects to process
        
        Returns
        -------
        BayeSEDResults
            Result object with posterior, best_fit, and evidence attributes
        """
        # Import here to avoid circular dependency
        from .core import BayeSEDInterface
        
        # Apply inference settings to params
        if self._multinest is not None:
            params.multinest = self._multinest
        if self._gsl_integration_qag is not None:
            params.gsl_integration_qag = self._gsl_integration_qag
        if self._gsl_multifit_robust is not None:
            params.gsl_multifit_robust = self._gsl_multifit_robust
        if self._nnlm is not None:
            params.NNLM = self._nnlm
        if self._ndumper is not None:
            params.Ndumper = self._ndumper
        if self._logzero is not None:
            params.logZero = self._logzero
        if self._unweighted_samples is not None:
            params.unweighted_samples = self._unweighted_samples
        if self._nsample is not None:
            params.Nsample = self._nsample
        if self._niteration is not None:
            params.niteration = self._niteration
        
        # Execute fit - create interface with MPI settings
        interface = BayeSEDInterface(mpi_mode=mpi_mode or '1', np=np, Ntest=Ntest)
        execution = interface.run(params, validate=validate, auto_select_mpi_mode=auto_select_mpi_mode)
        
        # Load and return results with extracted model configs
        return self.load_results(params.outdir, params, execution.model_configs)
    
    def load_results(self, output_dir: str, params: 'BayeSEDParams' = None, model_configs: list = None):
        """
        Load results from output directory.
        
        Parameters
        ----------
        output_dir : str
            Output directory path
        params : BayeSEDParams, optional
            BayeSEDParams object to extract catalog name from input file.
            If provided, catalog name will be extracted from params.input_file
            and passed to BayeSEDResults for more reliable initialization.
        model_configs : list of str, optional
            List of model configuration names extracted from execution output.
            For multiple configs, they will be combined with AND logic to match
            HDF5 files that contain all components.
        
        Returns
        -------
        BayeSEDResults
            Result object with posterior, best_fit, and evidence attributes
        """
        # Import here to avoid circular dependency
        from .results import BayeSEDResults
        
        catalog_name = None
        if params and params.input_file:
            try:
                # Extract catalog name from input file for more reliable results loading
                from .utils import extract_catalog_name
                catalog_name = extract_catalog_name(params.input_file)
            except Exception as e:
                # If extraction fails, let BayeSEDResults auto-detect
                # This provides a fallback while still attempting the more reliable method
                import logging
                logging.getLogger(__name__).warning(
                    f"Could not extract catalog name from {params.input_file}: {e}. "
                    f"BayeSEDResults will attempt auto-detection."
                )
        
        # Combine model configs for matching
        model_config_pattern = None
        if model_configs and len(model_configs) > 0:
            if len(model_configs) == 1:
                # Single model config - use as is
                model_config_pattern = model_configs[0]
            else:
                # Multiple model configs - combine with AND logic using regex
                # Create a pattern that requires all components to be present
                # Example: for configs ['csp_sfh201_bc2003', 'agn_disk_bbb'], 
                # create pattern that matches files containing both
                import re
                # Escape special regex characters in each config
                escaped_configs = [re.escape(config) for config in model_configs]
                # Create positive lookahead pattern for each config (AND logic)
                lookaheads = [f"(?=.*{config})" for config in escaped_configs]
                model_config_pattern = "".join(lookaheads) + ".*"
        
        return BayeSEDResults(output_dir, catalog_name=catalog_name, model_config=model_config_pattern)

