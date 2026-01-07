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

