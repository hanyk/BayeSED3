"""
Core classes for BayeSED3 Python interface.

This module contains the main classes for configuring and running
BayeSED3 SED analysis: BayeSEDParams, BayeSEDInterface,
and exception classes.
"""

import os
import platform
import subprocess
import multiprocessing
import sys
import shlex
import shutil
import tarfile
import requests
from tqdm import tqdm
from dataclasses import dataclass, field
from typing import List, Optional, Union
import numpy as np
from astropy.table import Table
from astropy.io import ascii

# Import parameter classes
from .params import (
    FANNParams, AGNParams, BlackbodyParams, BigBlueBumpParams,
    GreybodyParams, AKNNParams, LineParams, LuminosityParams,
    NPSFHParams, PolynomialParams, PowerlawParams, RBFParams,
    SFHParams, SSPParams, SEDLibParams, SysErrParams, ZParams,
    NNLMParams, NdumperParams, OutputSFHParams, MultiNestParams,
    SFROverParams, SNRmin1Params, SNRmin2Params,
    GSLIntegrationQAGParams, GSLMultifitRobustParams, KinParams,
    LineListParams, MakeCatalogParams, CloudyParams, CosmologyParams,
    DALParams, RDFParams, TemplateParams, RenameParams
)

# Import utility functions
from .utils import (
    _to_array, create_input_catalog, create_filters_from_svo,
    create_filters_selected, infer_filter_itype_icalib
)

# Import plotting function
from .plotting import plot_bestfit



class IDConstants:
    """
    Constants for ID and igroup assignment logic.

    These constants define the spacing and offsets used when automatically
    assigning IDs to galaxy and AGN instances.
    """
    # Galaxy ID spacing
    GALAXY_ID_INCREMENT = 2  # Increment by 2 to leave room for DEM (uses base_id + 1)
    GALAXY_IGROUP_INCREMENT = 1  # Increment igroup by 1 for each galaxy

    # AGN ID spacing
    AGN_COMPONENT_COUNT = 5  # Maximum components: disk, blr, feii, nlr, torus
    AGN_ID_INCREMENT_FIRST = 2  # When no existing AGN, increment id by 2
    AGN_ID_INCREMENT_SUBSEQUENT = 6  # When existing AGN, increment by 6 (conservative)
    AGN_IGROUP_INCREMENT_FIRST = 1  # When no existing AGN, increment igroup by 1
    AGN_IGROUP_INCREMENT_SUBSEQUENT = 6  # When existing AGN, increment by 6

    # AGN Component Offsets (relative to base_igroup and base_id)
    # Updated: Disk now uses base+0 (was base+1) to eliminate wasted ID space
    AGN_OFFSET_DISK = 0
    AGN_OFFSET_BLR = 1
    AGN_OFFSET_FEII = 2
    AGN_OFFSET_NLR = 3
    AGN_OFFSET_TORUS = 4


@dataclass
class BayeSEDParams:
    """
    Configuration parameters for BayeSED3 SED analysis.

    This dataclass contains all parameters needed to configure and run a BayeSED3
    analysis, including model components (SSP, SFH, AGN, etc.), algorithm settings
    (MultiNest), and output options.

    The class provides convenience builder methods for common configurations:
    - `galaxy()`: Simple galaxy SED fitting (creates new instance)
    - `agn()`: AGN (Active Galactic Nuclei) fitting (creates new instance)

    For multiple instances in a single configuration (like the GUI), use the class-based approach:

    - Use `SEDModel.GalaxyInstance` and `SEDModel.AGNInstance` classes to create instances
    - Add them with `add_galaxy()` and `add_agn()` methods
    - Better encapsulation and easier to manage multiple instances

    Parameters can be validated using the `validate()` method before running analysis.

    Examples
    --------
    >>> # Simple galaxy fitting using builder method
    >>> params = BayeSEDParams.galaxy(
    ...     input_file='observation/my_catalog.txt',
    ...     outdir='output',
    ...     ssp_model='bc2003_hr_stelib_chab_neb_2000r',
    ...     sfh_type='exponential'
    ... )
    >>>
    >>> # AGN fitting
    >>> params = BayeSEDParams.agn(
    ...     input_file='observation/qso_catalog.txt',
    ...     outdir='output',
    ...     agn_components=None  # All components (default)
    ... )
    >>>
    >>> # Manual configuration
    >>> params = BayeSEDParams(
    ...     input_type=0,
    ...     input_file='observation/my_catalog.txt',
    ...     outdir='output',
    ...     ssp=[SSPParams(igroup=0, id=0, name='bc2003_hr_stelib_chab_neb_2000r')],
    ...     sfh=[SFHParams(id=0, itype_sfh=2)],
    ...     dal=[DALParams(id=0, ilaw=8)]
    ... )
    >>>
    >>> # Multiple instances - using class-based approach (recommended)
    >>> from bayesed.model import SEDModel
    >>> params = BayeSEDParams(input_type=0, input_file='data.txt')
    >>> galaxy1 = SEDModel.create_galaxy()
    >>> galaxy2 = SEDModel.create_galaxy()
    >>> agn1 = SEDModel.create_agn(base_igroup=1, base_id=2)  # All components (default)
    >>> agn2 = SEDModel.create_agn(base_igroup=7, base_id=8, agn_components=['dsk', 'blr'])
    >>> params.add_galaxy(galaxy1).add_galaxy(galaxy2)
    >>> params.add_agn(agn1).add_agn(agn2)
    >>>
    >>> # Validate before running
    >>> params.validate()

    Attributes
    ----------
    input_type : int
        Input data type: 0 for flux in μJy, 1 for AB magnitude
    input_file : str
        Path to input catalog file containing photometry/spectroscopy data
    outdir : str
        Output directory for analysis results (default: "result")
    verbose : int
        Verbosity level: 0 (quiet), 1 (normal), 2 (verbose) (default: 2)
    ssp : List[SSPParams]
        Stellar population synthesis model parameters
    sfh : List[SFHParams]
        Star formation history parameters
    dal : List[DALParams]
        Dust attenuation law parameters
    AGN : List[AGNParams]
        AGN component parameters
    multinest : MultiNestParams, optional
        MultiNest nested sampling algorithm parameters
    filters : str, optional
        Path to filter definition file
    filters_selected : str, optional
        Path to filter selection file
    # ... (many more attributes)

    See Also
    --------
    BayeSEDInterface : Interface for running analysis with these parameters
    BayeSEDResults : Class for loading and accessing analysis results
    """
    # Basic Parameters
    input_type: int
    input_file: str
    outdir: str = "result"
    verbose: int = 2
    help: bool = False

    # Model related parameters
    fann: List[FANNParams] = field(default_factory=list)
    AGN: List[AGNParams] = field(default_factory=list)
    blackbody: List[BlackbodyParams] = field(default_factory=list)
    big_blue_bump: List[BigBlueBumpParams] = field(default_factory=list)
    greybody: List[GreybodyParams] = field(default_factory=list)
    aknn: List[AKNNParams] = field(default_factory=list)
    line: List[LineParams] = field(default_factory=list)
    lines: List[LineParams] = field(default_factory=list)
    lines1: List[LineParams] = field(default_factory=list)
    luminosity: Optional[LuminosityParams] = None
    np_sfh: Optional[NPSFHParams] = None
    polynomial: Optional[PolynomialParams] = None
    powerlaw: List[PowerlawParams] = field(default_factory=list)
    rbf: List[RBFParams] = field(default_factory=list)
    sfh: List[SFHParams] = field(default_factory=list)
    ssp: List[SSPParams] = field(default_factory=list)
    sedlib: List[SEDLibParams] = field(default_factory=list)
    sys_err_mod: Optional[SysErrParams] = None
    sys_err_obs: Optional[SysErrParams] = None
    z: Optional[ZParams] = None
    inn: List[SEDLibParams] = field(default_factory=list)
    cloudy: List[CloudyParams] = field(default_factory=list)
    cosmology: Optional[CosmologyParams] = None
    dal: List[DALParams] = field(default_factory=list)
    rdf: Optional[RDFParams] = None
    template: List[TemplateParams] = field(default_factory=list)

    # Output control parameters
    save_bestfit: int = 0
    save_sample_par: bool = False
    save_pos_sfh: Optional[str] = None
    save_pos_spec: bool = False
    save_sample_obs: bool = False
    save_sample_spec: bool = False
    save_summary: bool = False
    suffix: Optional[str] = None
    output_SFH: Optional[OutputSFHParams] = None
    output_mock_photometry: Optional[int] = None
    output_mock_spectra: bool = False
    output_model_absolute_magnitude: bool = False
    output_pos_obs: bool = False

    # Algorithm parameters
    NNLM: Optional[NNLMParams] = None
    Ndumper: Optional[NdumperParams] = None
    multinest: Optional[MultiNestParams] = None
    gsl_integration_qag: Optional[GSLIntegrationQAGParams] = None
    gsl_multifit_robust: Optional[GSLMultifitRobustParams] = None
    kin: List[KinParams] = field(default_factory=list)

    # Other parameters
    rename: List[RenameParams] = field(default_factory=list)
    rename_all: Optional[str] = None
    SFR_over: Optional[SFROverParams] = None
    SNRmin1: Optional[SNRmin1Params] = None
    SNRmin2: Optional[SNRmin2Params] = None
    unweighted_samples: bool = False
    filters: Optional[str] = None
    filters_selected: Optional[str] = None
    no_spectra_fit: bool = False
    NfilterPoints: Optional[int] = None
    Nsample: Optional[int] = None
    Ntest: Optional[int] = None
    import_files: Optional[List[str]] = None
    IGM: Optional[int] = None
    logZero: Optional[float] = None
    lw_max: Optional[float] = None
    LineList: Optional[LineListParams] = None
    make_catalog: Optional[MakeCatalogParams] = None
    niteration: Optional[int] = None
    no_photometry_fit: bool = False
    build_sedlib: Optional[int] = None
    check: bool = False
    cl: Optional[str] = None
    priors_only: bool = False

    export: Optional[str] = None

    def validate(self, check_files=True):
        """
        Validate BayeSEDParams configuration.

        Parameters
        ----------
        check_files : bool
            Whether to check file existence (default: True)

        Returns
        -------
        list
            List of validation errors (empty if valid)

        Raises
        ------
        BayeSEDValidationError
            If validation fails and errors are found
        """
        errors = []

        # Check required fields
        if not hasattr(self, 'input_file') or not self.input_file:
            errors.append("input_file is required")

        if not hasattr(self, 'input_type') or self.input_type is None:
            errors.append("input_type is required")

        # Check file existence
        if check_files and self.input_file:
            if not os.path.exists(self.input_file):
                errors.append(f"Input file does not exist: {self.input_file}")

        if check_files and self.filters and not os.path.exists(self.filters):
            errors.append(f"Filters file does not exist: {self.filters}")

        if check_files and self.filters_selected and not os.path.exists(self.filters_selected):
            errors.append(f"Filters selected file does not exist: {self.filters_selected}")

        # Check parameter ranges
        if hasattr(self, 'input_type') and self.input_type not in [0, 1]:
            errors.append(f"input_type must be 0 (flux in uJy) or 1 (AB magnitude), got {self.input_type}")

        if hasattr(self, 'verbose') and self.verbose not in [0, 1, 2]:
            errors.append(f"verbose must be 0, 1, or 2, got {self.verbose}")

        # Check that at least one model component is specified
        model_components = [
            self.fann, self.AGN, self.blackbody, self.big_blue_bump, self.greybody,
            self.aknn, self.line, self.lines, self.lines1, self.powerlaw, self.rbf,
            self.ssp, self.sedlib, self.template, self.cloudy
        ]
        if not any(model_components):
            errors.append("At least one model component must be specified (fann, AGN, ssp, etc.)")

        # Check ID consistency (SSP and SFH should have matching IDs)
        ssp_ids = {s.id for s in self.ssp}
        sfh_ids = {s.id for s in self.sfh}
        if ssp_ids and sfh_ids and not ssp_ids.intersection(sfh_ids):
            errors.append(f"SSP IDs {ssp_ids} and SFH IDs {sfh_ids} do not overlap - components must have matching IDs")

        # Check DAL IDs match SFH IDs
        dal_ids = {d.id for d in self.dal}
        if dal_ids and sfh_ids and not dal_ids.intersection(sfh_ids):
            errors.append(f"DAL IDs {dal_ids} and SFH IDs {sfh_ids} do not overlap - components must have matching IDs")

        if errors:
            raise BayeSEDValidationError(errors)

        return []

    @classmethod
    def _get_next_ids(cls, existing_params=None):
        """
        Calculate the next available igroup and id for new components.

        This helper method scans existing parameters to find the maximum
        igroup and id values, then returns the next available values.

        Parameters
        ----------
        existing_params : BayeSEDParams, optional
            Existing BayeSEDParams instance to check for used IDs.
            If None, starts from 0.

        Returns
        -------
        tuple
            (next_igroup, next_id) - Next available igroup and id values
        """
        max_igroup = -1
        max_id = -1

        if existing_params:
            # Check SSP (has both igroup and id)
            if existing_params.ssp:
                for ssp in existing_params.ssp:
                    max_igroup = max(max_igroup, ssp.igroup)
                    max_id = max(max_id, ssp.id)

            # Check AGN (has both igroup and id)
            if existing_params.AGN:
                for agn in existing_params.AGN:
                    max_igroup = max(max_igroup, agn.igroup)
                    max_id = max(max_id, agn.id)

            # Check Big Blue Bump (has both igroup and id)
            if existing_params.big_blue_bump:
                for bbb in existing_params.big_blue_bump:
                    max_igroup = max(max_igroup, bbb.igroup)
                    max_id = max(max_id, bbb.id)

            # Check Greybody, Blackbody, FANN, AKNN (have both igroup and id)
            for param_list in [existing_params.greybody, existing_params.blackbody,
                              existing_params.fann, existing_params.aknn]:
                if param_list:
                    for param in param_list:
                        max_igroup = max(max_igroup, param.igroup)
                        max_id = max(max_id, param.id)

            # Check lines1 (has both igroup and id)
            if existing_params.lines1:
                for line in existing_params.lines1:
                    max_igroup = max(max_igroup, line.igroup)
                    max_id = max(max_id, line.id)

            # Check SFH, DAL, Kin (have id only)
            for param_list in [existing_params.sfh, existing_params.dal, existing_params.kin]:
                if param_list:
                    for param in param_list:
                        max_id = max(max_id, param.id)

        # For galaxy: increment igroup by 1, id by 2 (to leave room for DEM)
        next_igroup = max_igroup + 1
        next_id = max_id + 2

        return next_igroup, next_id

    @classmethod
    def _get_max_ids_igroups(cls, existing_params=None):
        """
        Calculate the maximum igroup and id from existing parameters.

        This helper method scans existing parameters to find the maximum
        igroup and id values currently in use.

        Parameters
        ----------
        existing_params : BayeSEDParams, optional
            Existing BayeSEDParams instance to check for used IDs.
            If None, returns (-1, -1).

        Returns
        -------
        tuple
            (max_id, max_igroup) - Maximum id and igroup values currently in use
        """
        max_igroup = -1
        max_id = -1

        if existing_params:
            # Check SSP (has both igroup and id)
            if existing_params.ssp:
                for ssp in existing_params.ssp:
                    max_igroup = max(max_igroup, ssp.igroup)
                    max_id = max(max_id, ssp.id)

            # Check AGN (has both igroup and id)
            if existing_params.AGN:
                for agn in existing_params.AGN:
                    max_igroup = max(max_igroup, agn.igroup)
                    max_id = max(max_id, agn.id)

            # Check Big Blue Bump (has both igroup and id)
            if existing_params.big_blue_bump:
                for bbb in existing_params.big_blue_bump:
                    max_igroup = max(max_igroup, bbb.igroup)
                    max_id = max(max_id, bbb.id)

            # Check Greybody, Blackbody, FANN, AKNN (have both igroup and id)
            for param_list in [existing_params.greybody, existing_params.blackbody,
                              existing_params.fann, existing_params.aknn]:
                if param_list:
                    for param in param_list:
                        max_igroup = max(max_igroup, param.igroup)
                        max_id = max(max_id, param.id)

            # Check lines1 (has both igroup and id)
            if existing_params.lines1:
                for line in existing_params.lines1:
                    max_igroup = max(max_igroup, line.igroup)
                    max_id = max(max_id, line.id)

            # Check SFH, DAL, Kin (have id only)
            for param_list in [existing_params.sfh, existing_params.dal, existing_params.kin]:
                if param_list:
                    for param in param_list:
                        max_id = max(max_id, param.id)

        return max_id, max_igroup

    @classmethod
    def galaxy(cls, input_file, ssp_model='bc2003_hr_stelib_chab_neb_2000r',
               sfh_type='exponential', dal_law='calzetti', outdir='result',
               ssp_k=1, ssp_f_run=1, ssp_Nstep=1, ssp_i0=0, ssp_i1=0, ssp_i2=0, ssp_i3=0,
               ssp_iscalable=1, sfh_itype_ceh=0, sfh_itruncated=0,
               base_igroup=None, base_id=None, **kwargs):
        """
        Create BayeSEDParams for simple galaxy SED fitting.

        This is a convenience method for common galaxy-only configurations.
        Internally uses SEDModel.create_galaxy() to ensure consistency.

        Parameters
        ----------
        input_file : str
            Path to input file containing observed SED data
        ssp_model : str
            SSP model name (default: 'bc2003_hr_stelib_chab_neb_2000r')
        sfh_type : str or int
            Star formation history type. Can be a string name or integer code:

            String names:
            - 'instantaneous', 'instantaneous_burst', 'burst' (0): Instantaneous burst
            - 'constant' (1): Constant SFH
            - 'exponential', 'exponentially_declining' (2): Exponentially declining SFH
            - 'exponentially_increasing', 'increasing' (3): Exponentially increasing SFH
            - 'single_burst', 'burst_length_tau' (4): Single burst of length tau
            - 'delayed', 'delayed_exponential' (5): Delayed exponential SFH
            - 'beta' (6): Beta SFH
            - 'lognormal', 'log_normal' (7): Lognormal SFH
            - 'double_powerlaw', 'double_power_law' (8): Double power-law SFH
            - 'nonparametric', 'non_parametric' (9): Non-parametric SFH

            Or integer codes 0-9 (default: 'exponential' / 2)

        dal_law : str or int
            Dust attenuation law. Can be a string name or integer code:

            String names:
            - 'sed_model', 'sed_normalization' (0): SED model with L_dust normalization
            - 'starburst', 'starburst_calzetti', 'calzetti_fast' (1): Starburst (Calzetti+2000, FAST)
            - 'milky_way', 'milky_way_cardelli', 'cardelli' (2): Milky Way (Cardelli+1989, FAST)
            - 'star_forming', 'star_forming_salim', 'salim' (3): Star-forming (Salim+2018)
            - 'mw_allen', 'allen' (4): MW (Allen+76, hyperz)
            - 'mw_fitzpatrick', 'fitzpatrick_mw' (5): MW (Fitzpatrick+86, hyperz)
            - 'lmc', 'lmc_fitzpatrick', 'fitzpatrick_lmc' (6): LMC (Fitzpatrick+86, hyperz)
            - 'smc', 'smc_fitzpatrick', 'fitzpatrick_smc' (7): SMC (Fitzpatrick+86, hyperz)
            - 'calzetti', 'calzetti2000', 'starburst_calzetti2000' (8): SB (Calzetti2000, hyperz)
            - 'star_forming_reddy', 'reddy' (9): Star-forming (Reddy+2015)

            Or integer codes 0-9 (default: 'calzetti' for Calzetti2000)
        outdir : str
            Output directory (default: 'result')
        ssp_k, ssp_f_run, ssp_Nstep, ssp_i0, ssp_i1, ssp_i2, ssp_i3 : int
            SSP parameters for customization. These correspond to the k, f_run, Nstep, i0, i1, i2, i3
            parameters in SSPParams. Defaults: k=1, f_run=1, Nstep=1, i0=0, i1=0, i2=0, i3=0.
            Set ssp_i1=1 to enable nebular emission (common default).
        ssp_iscalable : int
            SSP scalability parameter (default: 1). Controls normalization method:
            - 1: Normalization determined with NNLM (Nonnegative Linear Models, faster, suitable for high-SNR data)
            - 0: Normalization as a free parameter of MultiNest sampling (more robust for low-SNR data)
        sfh_itype_ceh : int
            Chemical evolution history type for SFH (default: 0):
            - 0: No chemical evolution (constant metallicity)
            - 1: Chemical evolution enabled (metallicity evolves with time)
        sfh_itruncated : int
            SFH truncation flag (default: 0):
            - 0: Normal SFH
            - 1: Truncated SFH (for modeling quenched/passive galaxies)
        base_igroup : int, optional
            Base igroup for this galaxy instance. If None, automatically determined from existing
            components (default: starts from 0 if no existing components).
        base_id : int, optional
            Base ID for this galaxy instance. If None, automatically determined from existing
            components (default: starts from 0 if no existing components).
            Note: DEM will use base_id + 1, so IDs are incremented by 2 to leave room.
        **kwargs
            Additional parameters to pass to BayeSEDParams constructor

        Returns
        -------
        BayeSEDParams
            Configured parameters for galaxy fitting

        Example
        -------
        >>> # Standard galaxy fitting
        >>> params = BayeSEDParams.galaxy('observation/data.txt')
        >>> interface = BayeSEDInterface()
        >>> interface.run(params)
        >>>
        >>> # Advanced galaxy fitting with chemical evolution
        >>> params = BayeSEDParams.galaxy(
        ...     'observation/data.txt',
        ...     ssp_i1=1,              # Enable nebular emission
        ...     ssp_iscalable=1,       # NNLM normalization (default, faster)
        ...     sfh_itype_ceh=1,       # Enable chemical evolution
        ...     sfh_itruncated=0       # Normal SFH (default)
        ... )
        >>>
        >>> # Full Bayesian sampling for normalization (slower but more flexible)
        >>> params = BayeSEDParams.galaxy(
        ...     'observation/validation.txt',
        ...     ssp_iscalable=0        # MultiNest sampling for normalization
        ... )
        """
        # Import here to avoid circular dependency
        from .model import SEDModel

        # Determine base IDs/igroups from existing params if provided
        existing_params = kwargs.pop('existing_params', None)
        max_id, max_igroup = cls._get_max_ids_igroups(existing_params)

        # Set base IDs/igroups (auto-increment if not specified)
        if base_igroup is None:
            base_igroup = max_igroup + 1 if max_igroup >= 0 else 0
        if base_id is None:
            # Increment by 2 to leave room for DEM (which uses base_id + 1)
            base_id = max_id + 2 if max_id >= 0 else 0

        # Create galaxy instance using SEDModel (handles SFH/DAL string-to-int conversion)
        galaxy_instance = SEDModel.create_galaxy(
            ssp_model=ssp_model,
            sfh_type=sfh_type,
            dal_law=dal_law,
            ssp_k=ssp_k,
            ssp_f_run=ssp_f_run,
            ssp_Nstep=ssp_Nstep,
            ssp_i0=ssp_i0,
            ssp_i1=ssp_i1,
            ssp_i2=ssp_i2,
            ssp_i3=ssp_i3,
            ssp_iscalable=ssp_iscalable,
            sfh_itype_ceh=sfh_itype_ceh,
            sfh_itruncated=sfh_itruncated,
            base_igroup=base_igroup,
            base_id=base_id
        )

        # Create BayeSEDParams with defaults
        defaults = {
            'input_type': 0,  # Flux in uJy
            'input_file': input_file,
            'outdir': outdir,
            'save_bestfit': 0,
            'save_sample_par': True,
        }
        defaults.update(kwargs)

        params = cls(**defaults)

        # Add galaxy instance (with auto_assign_ids=False since we've already calculated IDs)
        params.add_galaxy(galaxy_instance, auto_assign_ids=False)

        return params

    @classmethod
    def agn(cls, input_file, ssp_model='bc2003_hr_stelib_chab_neb_2000r',
            sfh_type='exponential', dal_law='calzetti', agn_components=None,
            outdir='result', base_igroup=None, base_id=None, **kwargs):
        """
        Create BayeSEDParams for AGN (Active Galactic Nuclei) SED fitting.

        This convenience method sets up a typical AGN configuration with:
        - Galaxy components (SSP, SFH, DAL)
        - AGN accretion disk (Big Blue Bump)
        - AGN emission lines (BLR, NLR)
        - FeII template

        Internally uses SEDModel.create_galaxy() and SEDModel.create_agn() to ensure consistency.

        Parameters
        ----------
        input_file : str
            Path to input file containing observed SED data
        ssp_model : str
            SSP model name (default: 'bc2003_hr_stelib_chab_neb_2000r')
        sfh_type : str or int
            Star formation history type. Can be a string name or integer code:

            String names:
            - 'instantaneous', 'instantaneous_burst', 'burst' (0): Instantaneous burst
            - 'constant' (1): Constant SFH
            - 'exponential', 'exponentially_declining' (2): Exponentially declining SFH
            - 'exponentially_increasing', 'increasing' (3): Exponentially increasing SFH
            - 'single_burst', 'burst_length_tau' (4): Single burst of length tau
            - 'delayed', 'delayed_exponential' (5): Delayed exponential SFH
            - 'beta' (6): Beta SFH
            - 'lognormal', 'log_normal' (7): Lognormal SFH
            - 'double_powerlaw', 'double_power_law' (8): Double power-law SFH
            - 'nonparametric', 'non_parametric' (9): Non-parametric SFH

            Or integer codes 0-9 (default: 'exponential' / 2)
        dal_law : str or int
            Dust attenuation law. Can be a string name or integer code:

            String names:
            - 'sed_model', 'sed_normalization' (0): SED model with L_dust normalization
            - 'starburst', 'starburst_calzetti', 'calzetti_fast' (1): Starburst (Calzetti+2000, FAST)
            - 'milky_way', 'milky_way_cardelli', 'cardelli' (2): Milky Way (Cardelli+1989, FAST)
            - 'star_forming', 'star_forming_salim', 'salim' (3): Star-forming (Salim+2018)
            - 'mw_allen', 'allen' (4): MW (Allen+76, hyperz)
            - 'mw_fitzpatrick', 'fitzpatrick_mw' (5): MW (Fitzpatrick+86, hyperz)
            - 'lmc', 'lmc_fitzpatrick', 'fitzpatrick_lmc' (6): LMC (Fitzpatrick+86, hyperz)
            - 'smc', 'smc_fitzpatrick', 'fitzpatrick_smc' (7): SMC (Fitzpatrick+86, hyperz)
            - 'calzetti', 'calzetti2000', 'starburst_calzetti2000' (8): SB (Calzetti2000, hyperz)
            - 'star_forming_reddy', 'reddy' (9): Star-forming (Reddy+2015)

            Or integer codes 0-9 (default: 'calzetti' for Calzetti2000)
        agn_components : list of str, optional
            List of AGN component names to include. Valid components: 'dsk'/'disk'/'bbb', 'blr', 'nlr', 'feii'.
            If None (default), includes all components: ['dsk', 'blr', 'nlr', 'feii'].
            Note: 'dsk'/'disk'/'bbb' adds a BBB disk by default. Use SEDModel.AGNInstance methods to add other disk types (AGN, FANN, AKNN).
            Examples:
            - agn_components=['dsk', 'blr']  # Just disk (BBB) and BLR
            - agn_components=['dsk', 'feii']  # Disk (BBB) and FeII only
            - agn_components=['dsk']  # Just disk (BBB)
        outdir : str
            Output directory (default: 'result')
        base_igroup : int, optional
            Base igroup for AGN components. If None, automatically determined from existing
            components. Main AGN uses this, Disk (DSK) uses base_igroup+1, BLR uses base_igroup+2, etc.
        base_id : int, optional
            Base ID for AGN components. If None, automatically determined from existing
            components. Main AGN uses this, Disk (DSK) uses base_id+1, BLR uses base_id+2, etc.
        **kwargs
            Additional parameters to pass to BayeSEDParams constructor.
            Can include 'blr_lines_file' and 'nlr_lines_file' for AGN line components.

        Returns
        -------
        BayeSEDParams
            Configured parameters for AGN fitting

        Example
        -------
        >>> params = BayeSEDParams.agn('observation/qso.txt')
        >>> interface = BayeSEDInterface()
        >>> interface.run(params)

        >>> # Select specific components
        >>> params = BayeSEDParams.agn('observation/qso.txt', agn_components=['dsk', 'blr'])
        """
        # Import here to avoid circular dependency
        from .model import SEDModel

        # Determine base IDs/igroups from existing params if provided
        existing_params = kwargs.pop('existing_params', None)
        max_id, max_igroup = cls._get_max_ids_igroups(existing_params)

        # Set base IDs/igroups for galaxy (auto-increment if not specified)
        galaxy_base_igroup = max_igroup + 1 if max_igroup >= 0 else 0
        galaxy_base_id = max_id + 2 if max_id >= 0 else 0  # Leave room for DEM

        # Create galaxy instance using SEDModel
        galaxy_instance = SEDModel.create_galaxy(
            ssp_model=ssp_model,
            sfh_type=sfh_type,
            dal_law=dal_law,
            base_igroup=galaxy_base_igroup,
            base_id=galaxy_base_id
        )

        # Create BayeSEDParams with defaults
        defaults = {
            'input_type': 0,  # Flux in uJy
            'input_file': input_file,
            'outdir': outdir,
            'save_bestfit': 0,
            'save_sample_par': True,
        }
        defaults.update(kwargs)

        params = cls(**defaults)

        # Add galaxy instance first (with auto_assign_ids=False since we've already calculated IDs)
        params.add_galaxy(galaxy_instance, auto_assign_ids=False)

        # Now calculate AGN base IDs based on the updated params (after galaxy is added)
        # This ensures AGN IDs don't conflict with galaxy IDs
        if base_igroup is None or base_id is None:
            # Get max IDs from params after galaxy is added
            max_id_after_galaxy, max_igroup_after_galaxy = cls._get_max_ids_igroups(params)

            if base_igroup is None:
                # AGN components start after galaxy components
                # If there are existing AGN instances, increment by 6 to leave room for all components
                if max_igroup_after_galaxy >= 0:
                    # Check if there are existing AGN components
                    has_existing_agn = (params.AGN or params.big_blue_bump)
                    if has_existing_agn:
                        base_igroup = max_igroup_after_galaxy + IDConstants.AGN_IGROUP_INCREMENT_SUBSEQUENT
                    else:
                        base_igroup = max_igroup_after_galaxy + IDConstants.AGN_IGROUP_INCREMENT_FIRST
                else:
                    base_igroup = 1  # Start at 1 (0 is for galaxy)

            if base_id is None:
                # Similar logic for IDs
                if max_id_after_galaxy >= 0:
                    has_existing_agn = (params.AGN or params.big_blue_bump)
                    if has_existing_agn:
                        base_id = max_id_after_galaxy + IDConstants.AGN_ID_INCREMENT_SUBSEQUENT
                    else:
                        base_id = max_id_after_galaxy + IDConstants.AGN_ID_INCREMENT_FIRST
                else:
                    base_id = 0  # Start at 0

        # Create AGN instance using SEDModel
        # Extract line file paths from kwargs if provided
        blr_lines_file = kwargs.pop('blr_lines_file', 'observation/test/lines_BLR.txt')
        nlr_lines_file = kwargs.pop('nlr_lines_file', 'observation/test/lines_NLR.txt')

        agn_instance = SEDModel.create_agn(
            base_igroup=base_igroup,
            base_id=base_id,
            agn_components=agn_components,
            blr_lines_file=blr_lines_file,
            nlr_lines_file=nlr_lines_file
        )

        # Add AGN instance (with auto_assign_ids=False since we've already calculated IDs)
        params.add_agn(agn_instance, auto_assign_ids=False)

        return params

    def configure_multinest(self, nlive=400, efr=0.3, updInt=1000, fb=2,
                           tol=0.5, maxiter=0, mmodal=0, INS=1, ceff=0,
                           Ztol=-1e90, seed=1, resume=0, outfile=1, logZero=-1e90, acpt=0.01):
        """
        Configure MultiNest parameters with common settings.

        This convenience method sets up MultiNest nested sampling parameters
        with sensible defaults based on MultiNest recommendations. See
        README_multinest.txt for detailed parameter descriptions.

        Parameters
        ----------
        nlive : int
            Number of live points (default: 400). Higher values = more accurate but slower.
            Recommended: 400-1000 for most problems.
        efr : float
            Sampling efficiency (default: 0.3). Controls target sampling efficiency.
            Recommended: 0.8 for parameter estimation, 0.3 for evidence evaluation.
        updInt : int
            Update interval for output files in iterations (default: 1000).
            Output files are written after every updInt*10 iterations and at the end.
            Higher values reduce I/O overhead but provide less frequent updates.
        fb : int
            Feedback level (default: 2):
            - 0: Silent (no progress updates)
            - 1: Minimal feedback
            - 2: Normal feedback (recommended)
            - 3: Verbose feedback
        tol : float
            Evidence tolerance factor (default: 0.5). Stopping criterion for convergence.
            A value of 0.5 gives good enough accuracy for most problems.
        maxiter : int
            Maximum number of iterations (default: 0 = unlimited).
            MultiNest terminates when either maxiter is reached or convergence (tol) is satisfied.
        mmodal : int
            Mode separation flag (default: 0):
            - 0: Disabled (single mode)
            - 1: Enabled (separate multiple modes)
        INS : int
            Importance Nested Sampling flag (default: 1):
            - 0: Disabled (vanilla nested sampling)
            - 1: Enabled (INS provides more accurate evidence, ~order of magnitude better)
            Note: INS requires more memory and is not compatible with mmodal=1.
        ceff : int
            Constant efficiency mode flag (default: 0):
            - 0: Disabled (standard mode)
            - 1: Enabled (tunes enlargement factor to match target efficiency)
            Note: Evidence values may not be accurate in ceff mode unless INS is enabled.
        Ztol : float
            Null log-evidence threshold (default: -1e90, effectively disabled).
            When mmodal=1, only modes with local log-evidence > Ztol are reported.
            Set to very large negative number (e.g., -1e90) to report all modes.
        seed : int
            Random number generator seed (default: 1).
            Use negative value for seed from system clock.
        resume : int
            Resume from previous run (default: 0):
            - 0: Start fresh (or delete [root]resume.dat to force fresh start)
            - 1: Resume from checkpoint file
        outfile : int
            Write output files (default: 1):
            - 0: Disabled
            - 1: Enabled (creates posterior files, stats, etc.)
        logZero : float
            Log-likelihood threshold (default: -1e90, effectively disabled).
            Points with loglike < logZero will be ignored by MultiNest.
        acpt : float
            Acceptance rate threshold (default: 0.01). Used in constant efficiency mode.

        Returns
        -------
        self
            Returns self for method chaining

        Example
        -------
        >>> # For parameter estimation (faster)
        >>> params = BayeSEDParams.galaxy('input.txt')
        >>> params.configure_multinest(nlive=400, efr=0.8, fb=2)
        >>>
        >>> # For evidence evaluation (more accurate)
        >>> params.configure_multinest(nlive=400, efr=0.3, fb=2)
        >>>
        >>> # With mode separation
        >>> params.configure_multinest(nlive=400, mmodal=1, Ztol=-1e90)
        """
        if self.multinest is None:
            self.multinest = MultiNestParams()

        self.multinest.nlive = nlive
        self.multinest.efr = efr
        self.multinest.updInt = updInt
        self.multinest.fb = fb
        self.multinest.tol = tol
        self.multinest.maxiter = maxiter
        self.multinest.mmodal = mmodal
        self.multinest.INS = INS
        self.multinest.ceff = ceff
        self.multinest.Ztol = Ztol
        self.multinest.seed = seed
        self.multinest.resume = resume
        self.multinest.outfile = outfile
        self.multinest.logZero = logZero
        self.multinest.acpt = acpt

        return self

    def add_observation(self, observation):
        """
        Add observation data to the configuration.

        This method sets input data and filter-related parameters from an SEDObservation
        object, providing a cleaner API for data preparation.

        Parameters
        ----------
        observation : SEDObservation, PhotometryObservation, or SpectrumObservation
            Observation data object

        Returns
        -------
        self
            Returns self for method chaining

        Example
        -------
        >>> from bayesed.data import SEDObservation
        >>> obs = SEDObservation(
        ...     ids=[1, 2, 3],
        ...     z_min=[0.1, 0.2, 0.3],
        ...     z_max=[0.2, 0.3, 0.4],
        ...     phot_filters=['SLOAN/SDSS.u', 'SLOAN/SDSS.g'],
        ...     phot_fluxes=np.array([[100.0, 150.0], [110.0, 160.0], [120.0, 170.0]]),
        ...     phot_errors=np.array([[10.0, 15.0], [11.0, 16.0], [12.0, 17.0]]),
        ...     input_type=0
        ... )
        >>> obs.validate()
        >>> input_file = obs.to_bayesed_input('observation/my_galaxy')
        >>> params = BayeSEDParams()
        >>> params.add_observation(obs)  # Sets input_file, input_type, filters, etc.
        """
        # Import here to avoid circular dependency
        from .data import SEDObservation

        if not isinstance(observation, SEDObservation):
            raise TypeError(f"observation must be an SEDObservation instance, got {type(observation)}")

        # Set input file if observation has been converted to BayeSED input format
        # Note: User should call obs.to_bayesed_input() first to create the input file
        # We can't automatically create it here because we don't know the output path

        # Set input data and filter parameters from observation attributes
        self.input_type = observation.input_type
        if observation.filters is not None:
            self.filters = observation.filters
        if observation.filters_selected is not None:
            self.filters_selected = observation.filters_selected
        if observation.NfilterPoints is not None:
            self.NfilterPoints = observation.NfilterPoints

        # Set data quality control parameters
        self.no_photometry_fit = observation.no_photometry_fit
        self.no_spectra_fit = observation.no_spectra_fit
        if observation.SNRmin1 is not None:
            self.SNRmin1 = observation.SNRmin1
        if observation.SNRmin2 is not None:
            self.SNRmin2 = observation.SNRmin2
        if observation.sys_err_obs is not None:
            self.sys_err_obs = observation.sys_err_obs

        return self

    def add_model(self, model):
        """
        Add physical model settings to the configuration.

        This method applies additional physical model settings (IGM, cosmology, priors, etc.)
        from an SEDModel instance to the current configuration.

        Parameters
        ----------
        model : SEDModel
            SEDModel instance with additional physical model settings

        Returns
        -------
        self
            Returns self for method chaining

        Example
        -------
        >>> from bayesed.model import SEDModel
        >>> model = SEDModel()
        >>> model.set_igm(igm_model=1)
        >>> model.set_cosmology(H0=70.0, omigaA=0.7, omigam=0.3)
        >>> params = BayeSEDParams()
        >>> params.add_model(model)  # Sets IGM, cosmology, etc.
        """
        # Import here to avoid circular dependency
        from .model import SEDModel

        if not isinstance(model, SEDModel):
            raise TypeError(f"model must be an SEDModel instance, got {type(model)}")

        # Apply model settings if they were set
        if model._igm is not None:
            self.igm = model._igm
        if model._cosmology is not None:
            self.cosmology = model._cosmology
        if model._redshift_prior is not None:
            self.z = model._redshift_prior
        if model._sys_err_mod is not None:
            self.sys_err_mod = model._sys_err_mod
        if model._kinematics is not None:
            if not self.kin:
                self.kin = []
            self.kin.append(model._kinematics)
        if model._luminosity is not None:
            if not self.luminosity:
                self.luminosity = []
            self.luminosity.append(model._luminosity)
        if model._sfr_over is not None:
            self.sfr_over = model._sfr_over
        if model._lw_max is not None:
            self.lw_max = model._lw_max
        if model._line_list is not None:
            if not self.line_list:
                self.line_list = []
            self.line_list.append(model._line_list)

        return self

    def add_galaxy(self, galaxy_instance, auto_assign_ids=True):
        """
        Add a GalaxyInstance to the configuration.

        This method adds all components from a GalaxyInstance object to the current
        configuration, providing a cleaner API for managing multiple galaxy instances.

        Parameters
        ----------
        galaxy_instance : SEDModel.GalaxyInstance
            Galaxy instance to add
        auto_assign_ids : bool, optional
            If True (default), automatically assign new IDs and igroups to avoid conflicts.
            If False, use the IDs from the galaxy_instance as-is.

        Returns
        -------
        self
            Returns self for method chaining

        Example
        -------
        >>> from bayesed.model import SEDModel
        >>> # Auto-assign IDs (recommended for multiple instances)
        >>> params = BayeSEDParams(input_type=0, input_file='data.txt')
        >>> galaxy1 = SEDModel.create_galaxy()
        >>> params.add_galaxy(galaxy1)  # Auto-assigns IDs to avoid conflicts
        >>>
        >>> galaxy2 = SEDModel.create_galaxy()
        >>> params.add_galaxy(galaxy2)  # Auto-assigns new IDs (igroup=1, id=2)
        """
        from bayesed.model import SEDModel

        if not isinstance(galaxy_instance, SEDModel.GalaxyInstance):
            raise TypeError(
                f"galaxy_instance must be SEDModel.GalaxyInstance, got {type(galaxy_instance)}"
            )

        # Auto-assign IDs if requested and instance uses default IDs
        if auto_assign_ids:
            # Check if this instance uses default IDs (0, 0) or would conflict
            current_id = galaxy_instance.id
            current_igroup = galaxy_instance.igroup

            # Get max IDs from existing configuration
            max_id, max_igroup = self.__class__._get_max_ids_igroups(self)

            # Check if IDs would conflict or are defaults
            would_conflict = False
            if current_id <= max_id or current_igroup <= max_igroup:
                would_conflict = True

            # If using defaults (0, 0) or would conflict, assign new IDs
            if (current_id == 0 and current_igroup == 0) or would_conflict:
                next_igroup, next_id = self.get_next_galaxy_ids()

                # Save DEM and KIN from original instance before recreating
                original_dem = galaxy_instance.dem
                original_kin = galaxy_instance.kin

                # Create new instance with updated IDs
                galaxy_instance = SEDModel.create_galaxy(
                    ssp_model=galaxy_instance.ssp.name,
                    sfh_type=galaxy_instance.sfh.itype_sfh,
                    dal_law=galaxy_instance.dal.ilaw,
                    ssp_k=galaxy_instance.ssp.k,
                    ssp_f_run=galaxy_instance.ssp.f_run,
                    ssp_Nstep=galaxy_instance.ssp.Nstep,
                    ssp_i0=galaxy_instance.ssp.i0,
                    ssp_i1=galaxy_instance.ssp.i1,
                    ssp_i2=galaxy_instance.ssp.i2,
                    ssp_i3=galaxy_instance.ssp.i3,
                    ssp_iscalable=galaxy_instance.ssp.iscalable,  # Preserve iscalable
                    sfh_itype_ceh=galaxy_instance.sfh.itype_ceh,  # Preserve itype_ceh
                    sfh_itruncated=galaxy_instance.sfh.itruncated,  # Preserve itruncated
                    base_igroup=next_igroup,
                    base_id=next_id
                )

                # Copy DEM and KIN if they existed in original instance
                if original_dem:
                    galaxy_instance.add_dust_emission(
                        model_type='greybody' if isinstance(original_dem, GreybodyParams) else 'blackbody',
                        iscalable=original_dem.iscalable,
                        w_min=original_dem.w_min,
                        w_max=original_dem.w_max,
                        Nw=original_dem.Nw,
                        ithick=original_dem.ithick if isinstance(original_dem, GreybodyParams) else 0
                    )
                if original_kin:
                    galaxy_instance.kin = original_kin

        self.ssp.append(galaxy_instance.ssp)
        self.sfh.append(galaxy_instance.sfh)
        self.dal.append(galaxy_instance.dal)
        if galaxy_instance.dem:
            if isinstance(galaxy_instance.dem, GreybodyParams):
                self.greybody.append(galaxy_instance.dem)
            elif isinstance(galaxy_instance.dem, BlackbodyParams):
                self.blackbody.append(galaxy_instance.dem)
        if galaxy_instance.kin:
            self.kin.append(galaxy_instance.kin)
        return self

    def add_agn(self, agn_instance, auto_assign_ids=True):
        """
        Add an AGNInstance to the configuration.

        This method adds all components from an AGNInstance object to the current
        configuration, providing a cleaner API for managing multiple AGN instances.

        Parameters
        ----------
        agn_instance : SEDModel.AGNInstance
            AGN instance to add
        auto_assign_ids : bool, optional
            If True (default), automatically assign new IDs and igroups to avoid conflicts.
            If False, use the IDs from the agn_instance as-is.

        Returns
        -------
        self
            Returns self for method chaining

        Example
        -------
        >>> from bayesed.model import SEDModel
        >>> # Auto-assign IDs (recommended for multiple instances)
        >>> params = BayeSEDParams(input_type=0, input_file='data.txt')
        >>> agn1 = SEDModel.create_agn(base_igroup=0, base_id=0)  # All components (default)
        >>> params.add_agn(agn1)  # Auto-assigns IDs to avoid conflicts
        >>>
        >>> agn2 = SEDModel.create_agn(base_igroup=0, base_id=0, agn_components=['dsk', 'blr'])
        >>> params.add_agn(agn2)  # Auto-assigns new IDs
        >>>
        >>> # Manual ID assignment with different disk types
        >>> agn3 = SEDModel.create_agn(base_igroup=7, base_id=8, agn_components=['dsk', 'nlr', 'feii'])
        >>> agn3.add_disk_agn(name='agnsed')  # Replace default BBB with AGN disk
        >>> params.add_agn(agn3, auto_assign_ids=False)  # Uses provided IDs
        """
        from bayesed.model import SEDModel

        if not isinstance(agn_instance, SEDModel.AGNInstance):
            raise TypeError(
                f"agn_instance must be SEDModel.AGNInstance, got {type(agn_instance)}"
            )

        # Auto-assign IDs if requested
        if auto_assign_ids:
            # Get max IDs from existing configuration
            max_id, max_igroup = self.__class__._get_max_ids_igroups(self)

            # Check if IDs would conflict or are defaults
            would_conflict = False
            if agn_instance.base_id <= max_id or agn_instance.base_igroup <= max_igroup:
                would_conflict = True

            # If using defaults (0, 0) or would conflict, assign new IDs
            # Note: For AGN, we always auto-assign if IDs would conflict, even if not defaults
            if would_conflict or (agn_instance.base_id == 0 and agn_instance.base_igroup == 0):
                next_base_igroup, next_base_id = self.get_next_agn_ids()

                # Get line file paths if they exist
                blr_file = 'observation/test/lines_BLR.txt'
                nlr_file = 'observation/test/lines_NLR.txt'
                if agn_instance.blr:
                    blr_file = agn_instance.blr.file
                if agn_instance.nlr:
                    nlr_file = agn_instance.nlr.file

                # Preserve the original instance but update IDs
                # Create a new instance with updated base IDs
                new_instance = SEDModel.AGNInstance(
                    base_igroup=next_base_igroup,
                    base_id=next_base_id,
                    dsk=agn_instance.dsk,  # Preserve disk component
                    blr=agn_instance.blr,
                    nlr=agn_instance.nlr,
                    feii=agn_instance.feii,
                    tor=agn_instance.tor
                )
                # Update IDs in components if they exist
                # Updated offsets: disk uses base+0, others shifted down by 1
                if new_instance.dsk:
                    new_instance.dsk.igroup = next_base_igroup + IDConstants.AGN_OFFSET_DISK
                    new_instance.dsk.id = next_base_id + IDConstants.AGN_OFFSET_DISK
                if new_instance.blr:
                    new_instance.blr.igroup = next_base_igroup + IDConstants.AGN_OFFSET_BLR
                    new_instance.blr.id = next_base_id + IDConstants.AGN_OFFSET_BLR
                if new_instance.feii:
                    new_instance.feii.igroup = next_base_igroup + IDConstants.AGN_OFFSET_FEII
                    new_instance.feii.id = next_base_id + IDConstants.AGN_OFFSET_FEII
                if new_instance.nlr:
                    new_instance.nlr.igroup = next_base_igroup + IDConstants.AGN_OFFSET_NLR
                    new_instance.nlr.id = next_base_id + IDConstants.AGN_OFFSET_NLR
                if new_instance.tor:
                    new_instance.tor.igroup = next_base_igroup + IDConstants.AGN_OFFSET_TORUS
                    new_instance.tor.id = next_base_id + IDConstants.AGN_OFFSET_TORUS
                agn_instance = new_instance

        # Add disk component - route to appropriate parameter list based on type
        if agn_instance.dsk:
            if isinstance(agn_instance.dsk, BigBlueBumpParams):
                if not self.big_blue_bump:
                    self.big_blue_bump = []
                self.big_blue_bump.append(agn_instance.dsk)
                # Add DAL for BBB disk
                dal_params = agn_instance.get_dal_params()
                if dal_params:
                    self.dal.append(dal_params)
            elif isinstance(agn_instance.dsk, AGNParams):
                if not self.AGN:
                    self.AGN = []
                self.AGN.append(agn_instance.dsk)
                # Add DAL for AGN disk
                dal_params = agn_instance.get_dal_params()
                if dal_params:
                    self.dal.append(dal_params)
            elif isinstance(agn_instance.dsk, FANNParams):
                if not self.fann:
                    self.fann = []
                self.fann.append(agn_instance.dsk)
            elif isinstance(agn_instance.dsk, AKNNParams):
                if not self.aknn:
                    self.aknn = []
                self.aknn.append(agn_instance.dsk)
        if agn_instance.blr:
            if not self.lines1:
                self.lines1 = []
            self.lines1.append(agn_instance.blr)
        if agn_instance.nlr:
            if not self.lines1:
                self.lines1 = []
            self.lines1.append(agn_instance.nlr)
        if agn_instance.feii:
            self.aknn.append(agn_instance.feii)
            # Add kinematic for FeII
            kin_params = agn_instance.get_kin_params()
            if kin_params:
                self.kin.append(kin_params)
        if agn_instance.tor:
            # Torus can be either FANN or AKNN
            if isinstance(agn_instance.tor, FANNParams):
                if not self.fann:
                    self.fann = []
                self.fann.append(agn_instance.tor)
            elif isinstance(agn_instance.tor, AKNNParams):
                if not self.aknn:
                    self.aknn = []
                self.aknn.append(agn_instance.tor)
        return self

    def get_next_galaxy_ids(self):
        """
        Calculate the next available IDs for a new galaxy instance.

        Returns
        -------
        tuple
            (next_igroup, next_id) - Next available igroup and id for a galaxy instance
        """
        max_id, max_igroup = self.__class__._get_max_ids_igroups(self)
        next_igroup = (max_igroup + IDConstants.GALAXY_IGROUP_INCREMENT
                      if max_igroup >= 0 else 0)
        next_id = (max_id + IDConstants.GALAXY_ID_INCREMENT
                  if max_id >= 0 else 0)
        return next_igroup, next_id

    def get_next_agn_ids(self):
        """
        Calculate the next available IDs for a new AGN instance.

        Returns
        -------
        tuple
            (next_base_igroup, next_base_id) - Next available base igroup and id for an AGN instance
        """
        max_id, max_igroup = self.__class__._get_max_ids_igroups(self)

        has_existing_agn = (self.AGN or self.big_blue_bump)

        if max_igroup >= 0:
            if has_existing_agn:
                next_base_igroup = max_igroup + IDConstants.AGN_IGROUP_INCREMENT_SUBSEQUENT
            else:
                next_base_igroup = max_igroup + IDConstants.AGN_IGROUP_INCREMENT_FIRST
        else:
            next_base_igroup = 1

        if max_id >= 0:
            if has_existing_agn:
                next_base_id = max_id + IDConstants.AGN_ID_INCREMENT_SUBSEQUENT
            else:
                next_base_id = max_id + IDConstants.AGN_ID_INCREMENT_FIRST
        else:
            next_base_id = 0

        return next_base_igroup, next_base_id

    def validate_ids(self, new_id, new_igroup=None):
        """
        Validate that new IDs don't conflict with existing ones.

        Parameters
        ----------
        new_id : int
            Proposed new id value
        new_igroup : int, optional
            Proposed new igroup value (if applicable)

        Returns
        -------
        tuple
            (is_valid, conflicts) - True if valid, False with list of conflict messages
        """
        max_id, max_igroup = self.__class__._get_max_ids_igroups(self)

        conflicts = []
        if new_id <= max_id:
            conflicts.append(
                f"id {new_id} conflicts with existing id {max_id} "
                f"(must be > {max_id})"
            )
        if new_igroup is not None and new_igroup <= max_igroup:
            conflicts.append(
                f"igroup {new_igroup} conflicts with existing igroup {max_igroup} "
                f"(must be > {max_igroup})"
            )

        return (len(conflicts) == 0, conflicts)

    def get_used_ids(self):
        """
        Return all currently used IDs and igroups.

        Returns
        -------
        dict
            Dictionary with 'ids' and 'igroups' keys, each containing a list of used values
        """
        used_ids = set()
        used_igroups = set()

        # Check SSP
        if self.ssp:
            for ssp in self.ssp:
                used_ids.add(ssp.id)
                used_igroups.add(ssp.igroup)

        # Check AGN
        if self.AGN:
            for agn in self.AGN:
                used_ids.add(agn.id)
                used_igroups.add(agn.igroup)

        # Check Big Blue Bump
        if self.big_blue_bump:
            for bbb in self.big_blue_bump:
                used_ids.add(bbb.id)
                used_igroups.add(bbb.igroup)

        # Check Greybody, Blackbody, FANN, AKNN
        for param_list in [self.greybody, self.blackbody, self.fann, self.aknn]:
            if param_list:
                for param in param_list:
                    used_ids.add(param.id)
                    used_igroups.add(param.igroup)

        # Check lines1
        if self.lines1:
            for line in self.lines1:
                used_ids.add(line.id)
                used_igroups.add(line.igroup)

        # Check SFH, DAL, Kin (id only)
        for param_list in [self.sfh, self.dal, self.kin]:
            if param_list:
                for param in param_list:
                    used_ids.add(param.id)

        return {
            'ids': sorted(used_ids),
            'igroups': sorted(used_igroups)
        }


class BayeSEDValidationError(Exception):
    """Exception raised when BayeSED parameter validation fails."""

    def __init__(self, errors):
        """
        Initialize validation error.

        Parameters
        ----------
        errors : list
            List of error messages
        """
        self.errors = errors
        message = "BayeSED validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
        super().__init__(message)


class BayeSEDExecutionError(Exception):
    """Exception raised when BayeSED execution fails."""

    def __init__(self, message, returncode=None, output=None, suggestions=None, error_type='unknown'):
        """
        Initialize execution error.

        Parameters
        ----------
        message : str
            Main error message
        returncode : int, optional
            Process return code
        output : str, optional
            Complete output from BayeSED execution
        suggestions : list, optional
            List of suggestions for fixing the error
        error_type : str, optional
            Type of error (e.g., 'file_not_found', 'mpi_error', etc.)
        """
        self.returncode = returncode
        self.output = output
        self.suggestions = suggestions or []
        self.error_type = error_type

        # Build detailed error message
        full_message = message
        if returncode is not None:
            full_message += f" (return code: {returncode})"

        if self.suggestions:
            full_message += "\n\nSuggestions:\n"
            for i, suggestion in enumerate(self.suggestions, 1):
                full_message += f"  {i}. {suggestion}\n"

        if output and len(output) > 0:
            # Include last 20 lines of output for context
            output_lines = output.split('\n')
            if len(output_lines) > 20:
                full_message += "\nLast 20 lines of output:\n"
                full_message += '\n'.join(output_lines[-20:])
            else:
                full_message += "\nOutput:\n"
                full_message += output

        super().__init__(full_message)





class BayeSEDInterface:
    """
    Python interface for running BayeSED3 SED analysis.

    This class provides a high-level interface to the BayeSED3 binary executables,
    handling MPI setup, parameter validation, data preparation, and result loading.
    It simplifies common workflows while maintaining full access to all BayeSED3
    capabilities.

    Key Features
    ------------
    - **Automatic MPI Setup**: Handles OpenMPI installation and configuration
    - **Smart Mode Selection**: Automatically selects optimal MPI mode based on sample size
    - **Data Preparation**: Convenience methods for creating input catalogs and downloading filters
    - **End-to-End Workflows**: Builder methods that combine data preparation with parameter setup
    - **Result Loading**: Easy access to analysis results via `BayeSEDResults`
    - **Validation**: Built-in parameter validation before execution

    Examples
    --------
    >>> # Basic usage: run analysis with existing input file
    >>> bayesed = BayeSEDInterface(mpi_mode='1', np=4)
    >>> params = BayeSEDParams.galaxy(
    ...     input_file='observation/my_catalog.txt',
    ...     outdir='output'
    ... )
    >>> result = bayesed.run(params)
    >>>
    >>> # End-to-end workflow: prepare data and run analysis
    >>> from bayesed.data import SEDObservation
    >>> from bayesed.utils import create_filters_from_svo
    >>> import os
    >>>
    >>> # Step 1: Create observation data
    >>> obs = SEDObservation(
    ...     ids=[1, 2, 3],
    ...     z_min=[0.1, 0.2, 0.3],
    ...     z_max=[0.2, 0.3, 0.4],
    ...     phot_filters=['SLOAN/SDSS.u', 'SLOAN/SDSS.g', 'SLOAN/SDSS.r'],
    ...     phot_fluxes=fluxes,  # shape (3, 3)
    ...     phot_errors=errors,  # shape (3, 3),
    ...     input_type=0
    ... )
    >>> obs.validate()
    >>>
    >>> # Step 2: Create input catalog file
    >>> os.makedirs('observation/my_analysis', exist_ok=True)
    >>> input_file = obs.to_bayesed_input('observation/my_analysis', 'input_catalog')
    >>>
    >>> # Step 3: Download filters (optional)
    >>> filter_dir = 'observation/my_analysis/filters'
    >>> os.makedirs(filter_dir, exist_ok=True)
    >>> create_filters_from_svo(
    ...     svo_filterIDs=['SLOAN/SDSS.u', 'SLOAN/SDSS.g', 'SLOAN/SDSS.r'],
    ...     filters_file=os.path.join(filter_dir, 'filters.txt'),
    ...     filters_selected_file=os.path.join(filter_dir, 'filters_selected.txt')
    ... )
    >>> obs.filters = os.path.join(filter_dir, 'filters.txt')
    >>> obs.filters_selected = os.path.join(filter_dir, 'filters_selected.txt')
    >>>
    >>> # Step 4: Create model and add observation
    >>> params = BayeSEDParams.galaxy(
    ...     input_file=input_file,
    ...     outdir='observation/my_analysis/output'
    ... )
    >>> params.add_observation(obs)
    >>>
    >>> # Step 5: Run analysis
    >>> bayesed = BayeSEDInterface()
    >>> result = bayesed.run(params)
    >>>
    >>> # Load results after analysis
    >>> results = bayesed.load_results('observation/my_analysis/output')
    >>> bestfit = results.get_bestfit_spectrum()
    >>> posteriors = results.get_posterior_samples()

    See Also
    --------
    BayeSEDParams : Configuration parameters for analysis
    BayeSEDResults : Class for loading and accessing analysis results
    """
    def __init__(self, mpi_mode='1', openmpi_mirror=None, np=None, Ntest=None, output_queue=None, auto_select_mpi_mode=False):
        """
        Initialize BayeSED interface.

        Parameters
        ----------
        mpi_mode : str, optional
            MPI mode: '1' for bayesed_mn_1 (parallelize within objects, default),
                     'n' for bayesed_mn_n (parallelize across objects),
                     'auto' for automatic selection based on sample size and model complexity.
            Only '1', 'n', and 'auto' are allowed values.
        openmpi_mirror : str, optional
            Custom OpenMPI download mirror URL
        np : int, optional
            Number of processes (default: auto-detect from job scheduler or use all available cores).
            When running under PBS/qsub, SLURM, or SGE, automatically detects from environment variables
            (PBS_NCPUS, PBS_NODEFILE, SLURM_CPUS_PER_TASK, NSLOTS). Otherwise, uses multiprocessing.cpu_count().
        Ntest : int, optional
            Number of objects for test run
        output_queue : optional
            Output queue for GUI integration
        auto_select_mpi_mode : bool, optional
            If True, automatically select MPI mode based on sample size and model complexity
            when running (default: False). This is a convenience feature - you can also
            set mpi_mode='auto' to enable automatic selection.
        """
        # Handle 'auto' mode
        if mpi_mode == 'auto':
            mpi_mode = '1'  # Default, will be auto-selected later if auto_select_mpi_mode is True
            auto_select_mpi_mode = True

        # Validate mpi_mode
        if mpi_mode not in ['1', 'n']:
            raise ValueError(
                f"Invalid mpi_mode: '{mpi_mode}'. Allowed values are '1' (parallelize within objects), "
                f"'n' (parallelize across objects), or 'auto' (automatic selection)."
            )
        self.mpi_mode = f"mn_{mpi_mode}"
        self.openmpi_mirror = openmpi_mirror
        self.np = np
        self.Ntest = Ntest
        self.auto_select_mpi_mode = auto_select_mpi_mode
        self._get_system_info()
        self.mpi_cmd = self._setup_openmpi()
        self.num_processes = self._get_max_threads() if np is None else np
        self.executable_path = self._get_executable()

    def _select_mpi_mode(self, params, n_objects=None):
        """
        Automatically select optimal MPI mode based on sample size and model complexity.

        Parameters
        ----------
        params : BayeSEDParams
            Parameters for the analysis
        n_objects : int, optional
            Number of objects in the input file. If None, will be counted from input_file.

        Returns
        -------
        str
            Selected MPI mode: '1' or 'n'
        """
        # Count objects if not provided
        if n_objects is None:
            n_objects = self._count_objects(params.input_file)

        # Assess model complexity
        has_agn = len(params.AGN) > 0 or len(params.big_blue_bump) > 0 or len(params.aknn) > 0
        has_complex_sfh = any(sfh.itype_sfh >= 200 for sfh in params.sfh)  # Non-parametric SFH
        has_kinematics = len(params.kin) > 0
        has_multiple_components = (len(params.ssp) + len(params.AGN) + len(params.greybody) +
                                 len(params.blackbody) + len(params.powerlaw)) > 2

        is_complex_model = has_agn or has_complex_sfh or has_kinematics or has_multiple_components

        # Selection logic:
        # - Use 'n' (across objects) for large samples (>10) with simple models
        # - Use '1' (within objects) for small samples or complex models
        if n_objects > 10 and not is_complex_model:
            selected_mode = 'n'
        else:
            selected_mode = '1'

        return selected_mode

    def _count_objects(self, input_file):
        """
        Count number of objects in input file.

        Parameters
        ----------
        input_file : str
            Path to input file

        Returns
        -------
        int
            Number of objects
        """
        import os
        if not os.path.exists(input_file):
            return 0

        try:
            with open(input_file, 'r') as f:
                lines = f.readlines()

            n_objects = 0
            for line in lines:
                line = line.strip()
                if line and not line.startswith('#'):
                    # Check if it's a data line (has numeric values)
                    parts = line.split()
                    if len(parts) >= 3:  # At least ID, z_min, z_max
                        try:
                            float(parts[1])  # z_min should be numeric
                            n_objects += 1
                        except ValueError:
                            continue

            return n_objects
        except Exception:
            return 0

    def _get_system_info(self):
        self.os = platform.system().lower()
        self.arch = platform.machine().lower()

    def _get_max_threads(self):
        """
        Get the number of available CPU cores/processes.

        This method checks for job scheduler environment variables first (PBS, SLURM, etc.),
        then falls back to system CPU count.

        Priority order:
        1. PBS_NCPUS (PBS/Torque)
        2. SLURM_CPUS_PER_TASK or SLURM_NTASKS (SLURM)
        3. NSLOTS (SGE)
        4. multiprocessing.cpu_count() (fallback)

        Returns
        -------
        int
            Number of available CPU cores/processes
        """
        # Check PBS/Torque environment variables
        pbs_ncpus = os.environ.get('PBS_NCPUS')
        if pbs_ncpus:
            try:
                return int(pbs_ncpus)
            except ValueError:
                pass

        # Check PBS_NODEFILE (count lines = number of CPU slots allocated)
        # PBS_NODEFILE typically contains one hostname per CPU slot allocated
        pbs_nodefile = os.environ.get('PBS_NODEFILE')
        if pbs_nodefile and os.path.exists(pbs_nodefile):
            try:
                with open(pbs_nodefile, 'r') as f:
                    # Count all lines (each line typically represents one CPU slot)
                    lines = [line.strip() for line in f if line.strip()]
                    return len(lines)
            except Exception:
                pass

        # Check SLURM environment variables
        slurm_cpus = os.environ.get('SLURM_CPUS_PER_TASK') or os.environ.get('SLURM_NTASKS')
        if slurm_cpus:
            try:
                return int(slurm_cpus)
            except ValueError:
                pass

        # Check SGE environment variable
        sge_nslots = os.environ.get('NSLOTS')
        if sge_nslots:
            try:
                return int(sge_nslots)
            except ValueError:
                pass

        # Fallback to system CPU count
        return multiprocessing.cpu_count()

    def _setup_openmpi(self):
        openmpi_version = "4.1.6"
        openmpi_url = self.openmpi_mirror or f"https://download.open-mpi.org/release/open-mpi/v4.1/openmpi-{openmpi_version}.tar.gz"
        openmpi_dir = f"openmpi-{openmpi_version}"
        openmpi_file = f"{openmpi_dir}.tar.gz"
        install_dir = os.path.abspath("openmpi")

        # Priority 1: Check for conda-installed OpenMPI (highest priority)
        conda_prefix = os.environ.get('CONDA_PREFIX')
        if conda_prefix:
            conda_mpirun = os.path.join(conda_prefix, 'bin', 'mpirun')
            if os.path.exists(conda_mpirun):
                try:
                    result = subprocess.run([conda_mpirun, "--version"], capture_output=True, text=True)
                    installed_version = result.stdout.split()[3]
                    if installed_version == openmpi_version:
                        print(f"Using conda-installed OpenMPI {installed_version}")
                        return conda_mpirun
                    else:
                        print(f"Conda has OpenMPI {installed_version}, but we need {openmpi_version}")
                except Exception as e:
                    print(f"Error checking conda OpenMPI version: {e}")

        # Priority 2: Check if the correct version of OpenMPI is already installed on the system
        system_mpirun = shutil.which("mpirun")
        if system_mpirun:
            try:
                result = subprocess.run([system_mpirun, "--version"], capture_output=True, text=True)
                installed_version = result.stdout.split()[3]
                if installed_version == openmpi_version:
                    print(f"Using system-installed OpenMPI {installed_version}")
                    return system_mpirun
                else:
                    if not os.path.exists(install_dir):
                        print(f"System has OpenMPI {installed_version}, but we need {openmpi_version}")
            except Exception as e:
                print(f"Error checking OpenMPI version: {e}")

        # Priority 3: Check for locally compiled OpenMPI
        local_mpirun = os.path.join(install_dir, "bin", "mpirun")
        if os.path.exists(local_mpirun):
            print(f"Using locally compiled OpenMPI from {install_dir}")
            # Set up environment variables for local OpenMPI
            os.environ["PATH"] = f"{os.path.dirname(local_mpirun)}:{os.environ.get('PATH', '')}"
            os.environ["LD_LIBRARY_PATH"] = f"{os.path.join(install_dir, 'lib')}:{os.environ.get('LD_LIBRARY_PATH', '')}"
            return local_mpirun

        # Priority 4: Auto-compile OpenMPI (fallback for repository installations)
        # Ask user for permission before auto-compiling
        if not os.path.exists(install_dir):
            # Check if conda is available
            conda_available = shutil.which("conda") is not None

            print("\n" + "="*70)
            print("OpenMPI 4.1.6 not found!")
            print("="*70)

            # Check if we're in an interactive environment
            is_interactive = sys.stdin.isatty() and sys.stdout.isatty()

            # If conda is available, offer to install via conda automatically
            if conda_available and is_interactive:
                print("\n⚠️  Conda detected! BayeSED3 can automatically install OpenMPI via conda.")
                print("   This is the fastest and easiest method.")
                print("\n   Command: conda install openmpi=4.1.6")

                try:
                    response = input("\nDo you want to install OpenMPI via conda now? [Y/n]: ").strip().lower()
                    install_via_conda = response not in ['n', 'no']
                except (EOFError, KeyboardInterrupt):
                    print("\n\nCancelled by user.")
                    install_via_conda = False

                if install_via_conda:
                    print("\n⏳ Installing OpenMPI 4.1.6 via conda...")
                    try:
                        # Run conda install
                        result = subprocess.run(
                            ["conda", "install", "-y", "openmpi=4.1.6"],
                            capture_output=True,
                            text=True,
                            check=False
                        )

                        if result.returncode == 0:
                            print("✅ OpenMPI 4.1.6 successfully installed via conda!")
                            # Re-check for conda-installed OpenMPI
                            conda_prefix = os.environ.get('CONDA_PREFIX')
                            if conda_prefix:
                                conda_mpirun = os.path.join(conda_prefix, 'bin', 'mpirun')
                                if os.path.exists(conda_mpirun):
                                    try:
                                        result = subprocess.run([conda_mpirun, "--version"], capture_output=True, text=True)
                                        installed_version = result.stdout.split()[3]
                                        if installed_version == openmpi_version:
                                            print(f"✅ Using conda-installed OpenMPI {installed_version}")
                                            return conda_mpirun
                                    except Exception as e:
                                        print(f"⚠️  Error verifying OpenMPI version: {e}")

                            print("\n⚠️  OpenMPI was installed but not immediately detected.")
                            print("   Please restart your Python session and try again.")
                            raise FileNotFoundError(
                                f"\nOpenMPI {openmpi_version} was installed via conda, but needs a Python session restart.\n"
                                f"Please restart Python and try again."
                            )
                        else:
                            print(f"⚠️  Conda installation failed with return code {result.returncode}")
                            if result.stderr:
                                print(f"Error: {result.stderr}")
                            print("\nFalling back to other installation options...\n")
                    except Exception as e:
                        print(f"⚠️  Error running conda install: {e}")
                        print("\nFalling back to other installation options...\n")

            # Show manual installation options
            if conda_available:
                print("\n⚠️  Manual installation options:")
                print("   1. Conda: conda install openmpi=4.1.6")
                print("   2. System package manager (see below)")
            else:
                print("\n⚠️  RECOMMENDED: Install OpenMPI via system package manager:")
                print("   macOS:    brew install openmpi")
                print("   Ubuntu:   sudo apt-get install openmpi-bin libopenmpi-dev")
                print("   Fedora:   sudo dnf install openmpi openmpi-devel")

            print("\n" + "-"*70)
            print("Alternatively, BayeSED3 can automatically download and compile")
            print(f"OpenMPI {openmpi_version} from source (takes 10-30 minutes).")
            print("-"*70)

            if is_interactive:
                try:
                    response = input("\nDo you want to auto-compile OpenMPI now? [y/N]: ").strip().lower()
                    auto_compile = response in ['y', 'yes']
                except (EOFError, KeyboardInterrupt):
                    print("\n\nCancelled by user.")
                    auto_compile = False
            else:
                # Non-interactive environment (script, GUI, etc.)
                print("\n⚠️  Non-interactive environment detected.")
                print("   Auto-compilation requires user interaction.")
                print("   Please install OpenMPI manually using one of the methods above.")
                auto_compile = False

            if not auto_compile:
                raise FileNotFoundError(
                    f"\nOpenMPI {openmpi_version} is required but not found.\n"
                    f"Please install it using one of these methods:\n"
                    f"  1. Conda (recommended): conda install openmpi=4.1.6\n"
                    f"  2. System package manager (see above)\n"
                    f"  3. Manual compilation (see README.md)\n"
                    f"\nAfter installation, restart your Python session."
                )

            # User agreed to auto-compile - proceed with compilation
            print(f"\n⏳ Starting automatic compilation of OpenMPI {openmpi_version}...")
            print("   This may take 10-30 minutes depending on your system.\n")

            # Check if the tarball already exists and is complete
            if os.path.exists(openmpi_file):
                print(f"OpenMPI {openmpi_version} tarball already exists. Checking if it's complete...")
                try:
                    with tarfile.open(openmpi_file, 'r:gz') as tar:
                        tar.getmembers()  # This will raise an exception if the file is incomplete
                    print("Existing tarball is complete. Skipping download.")
                except Exception as e:
                    print(f"Existing tarball is incomplete or corrupted. Re-downloading: {e}")
                    os.remove(openmpi_file)

            if not os.path.exists(openmpi_file):
                print(f"Downloading OpenMPI {openmpi_version}...")
                response = requests.get(openmpi_url, stream=True)
                total_size = int(response.headers.get('content-length', 0))

                with open(openmpi_file, 'wb') as file, tqdm(
                    desc=openmpi_file,
                    total=total_size,
                    unit='iB',
                    unit_scale=True,
                    unit_divisor=1024,
                ) as progress_bar:
                    for data in response.iter_content(chunk_size=1024):
                        size = file.write(data)
                        progress_bar.update(size)

            print("Extracting OpenMPI...")
            with tarfile.open(openmpi_file, 'r:gz') as tar:
                tar.extractall()

            print("Compiling and installing OpenMPI...")
            os.chdir(openmpi_dir)
            subprocess.run(["./configure", f"--prefix={install_dir}"], check=True)
            subprocess.run(["make", "-j", str(self._get_max_threads())], check=True)
            subprocess.run(["make", "install"], check=True)
            os.chdir("..")

            print("Cleaning up temporary files...")
            shutil.rmtree(openmpi_dir)

            mpirun_path = os.path.join(install_dir, "bin", "mpirun")
            if not os.path.exists(mpirun_path):
                raise FileNotFoundError(f"mpirun not found at {mpirun_path}. OpenMPI installation may have failed.")

            # Set up environment variables for OpenMPI
            os.environ["PATH"] = f"{os.path.dirname(mpirun_path)}:{os.environ.get('PATH', '')}"
            os.environ["LD_LIBRARY_PATH"] = f"{os.path.join(install_dir, 'lib')}:{os.environ.get('LD_LIBRARY_PATH', '')}"

            print(f"\n✅ OpenMPI {openmpi_version} successfully compiled and installed!")
            return mpirun_path

        # If we get here, we should have found OpenMPI in one of the priorities above
        # This should not happen, but raise an error if it does
        raise FileNotFoundError(
            "Could not find OpenMPI 4.1.6. "
            "Checked conda installation, system installation, local compilation, and auto-compilation failed."
        )

    def _get_executable(self):
        from .utils import _get_resource_path

        executable = f"bayesed_{self.mpi_mode}"
        if self.os == "linux" or (self.os == "windows" and "microsoft" in platform.uname().release.lower()):
            platform_dir = "linux"
        elif self.os == "darwin":
            platform_dir = "mac"
        else:
            raise ValueError(f"Unsupported operating system: {self.os}")

        # Use resource path resolution for both conda and repository installations
        relative_path = os.path.join("bin", platform_dir, executable)
        executable_path = _get_resource_path(relative_path)

        return executable_path


    def _validate_cpu_constraint(self, params):
        """
        Validate CPU constraint for bayesed_mn_n: Ncpu <= Nobj + 1

        This is a hard constraint - bayesed_mn_n will refuse to run if violated.
        """
        if self.mpi_mode != 'mn_n':
            return  # Only applies to bayesed_mn_n

        # Count number of objects in input file
        try:
            input_file = params.input_file
            if not os.path.exists(input_file):
                # Can't validate if file doesn't exist yet
                return

            # Read input file to count objects
            # Format: First line is header comment with catalog name and counts
            # Then data rows, one per object
            with open(input_file, 'r') as f:
                lines = f.readlines()

            # Skip header comments and count data rows
            n_objects = 0
            for line in lines:
                line = line.strip()
                if line and not line.startswith('#'):
                    # Check if it's a data line (has numeric values)
                    parts = line.split()
                    if len(parts) >= 3:  # At least ID, z_min, z_max
                        try:
                            float(parts[1])  # z_min should be numeric
                            n_objects += 1
                        except ValueError:
                            continue

            # Validate constraint
            if self.num_processes > n_objects + 1:
                raise ValueError(
                    f"Invalid configuration for bayesed_mn_n: "
                    f"Number of CPUs ({self.num_processes}) exceeds number of objects + 1 ({n_objects + 1}). "
                    f"Constraint: Ncpu <= Nobj + 1. "
                    f"Binary will refuse to run. "
                    f"Suggestions: "
                    f"1) Switch to mpi_mode='1' (parallelize within objects), or "
                    f"2) Reduce number of CPUs to {n_objects + 1} or fewer."
                )
        except Exception as e:
            # If we can't read the file or count objects, warn but don't fail
            # The binary will fail with a clearer error message
            if isinstance(e, ValueError):
                raise  # Re-raise validation errors
            print(f"Warning: Could not validate CPU constraint: {e}")

    def run(self, params, validate=True, auto_select_mpi_mode=None):
        """
        Run BayeSED analysis.

        Parameters
        ----------
        params : BayeSEDParams or list
            Parameters for the analysis, or list of CLI arguments
        validate : bool
            Whether to validate parameters before execution (default: True)
        auto_select_mpi_mode : bool, optional
            If True, automatically select optimal MPI mode based on sample size and model complexity.
            If None, uses the instance's auto_select_mpi_mode setting (default: None).

        Returns
        -------
        subprocess.CompletedProcess
            Result of the subprocess execution
        """
        if isinstance(params, list):
            args = params
        else:
            # Auto-select MPI mode if requested
            if auto_select_mpi_mode is None:
                auto_select_mpi_mode = self.auto_select_mpi_mode

            if auto_select_mpi_mode:
                selected_mode = self._select_mpi_mode(params)
                if selected_mode != self.mpi_mode.replace('mn_', ''):
                    print(f"Auto-selected MPI mode: '{selected_mode}' "
                          f"(based on sample size and model complexity)")
                    # Temporarily change mode for this run
                    original_mode = self.mpi_mode
                    original_executable = self.executable_path
                    self.mpi_mode = f"mn_{selected_mode}"
                    # Update executable path
                    self.executable_path = self._get_executable()
                    try:
                        return self._run_with_params(params, validate)
                    finally:
                        # Restore original mode
                        self.mpi_mode = original_mode
                        self.executable_path = original_executable
                else:
                    return self._run_with_params(params, validate)
            else:
                return self._run_with_params(params, validate)

    def _run_with_params(self, params, validate=True):
        """
        Internal method to run BayeSED with parameters.

        Parameters
        ----------
        params : BayeSEDParams
            Parameters for the analysis
        validate : bool
            Whether to validate parameters before execution

        Returns
        -------
        subprocess.CompletedProcess
            Result of the subprocess execution
        """
        # Progress indicator: Validation
        if validate:
            print("=" * 70)
            print("BayeSED Inference - Starting")
            print("=" * 70)
            print("Step 1/3: Validating configuration...")
            try:
                params.validate(check_files=True)
                print("✓ Configuration validated successfully")
            except BayeSEDValidationError as e:
                print("✗ Configuration validation failed:")
                for error in e.errors:
                    print(f"  - {error}")
                raise  # Re-raise validation errors

        # Validate CPU constraint for bayesed_mn_n
        try:
            self._validate_cpu_constraint(params)
        except ValueError as e:
            print("✗ CPU constraint validation failed:")
            print(f"  {e}")
            raise  # Re-raise validation errors

        args = self._params_to_args(params)

        # Set TMPDIR environment variable
        os.environ['TMPDIR'] = '/tmp'

        # Set working directory to BayeSED3 root so binary can find data files (data/, models/, nets/)
        # Convert relative paths to absolute paths for both conda and repository installations
        from .utils import _is_conda_installation, _get_bayesed3_root, _ensure_absolute_path

        cwd = None
        try:
            # Get BayeSED3 root directory (works for both conda and repository installations)
            bayesed3_root = _get_bayesed3_root()
            # Set working directory to BayeSED3 root so binary can find data files
            cwd = bayesed3_root

            # Convert relative paths to absolute paths before changing directory
            # This ensures user's paths work correctly regardless of working directory
            if params.input_file and not os.path.isabs(params.input_file):
                params.input_file = _ensure_absolute_path(params.input_file)

            if params.outdir and not os.path.isabs(params.outdir):
                params.outdir = _ensure_absolute_path(params.outdir)

            if params.filters and not os.path.isabs(params.filters):
                params.filters = _ensure_absolute_path(params.filters)

            if params.filters_selected and not os.path.isabs(params.filters_selected):
                params.filters_selected = _ensure_absolute_path(params.filters_selected)

            # Rebuild args with updated paths
            args = self._params_to_args(params)
        except FileNotFoundError as e:
            # Gracefully handle if BayeSED3 root cannot be determined
            print(f"Warning: Could not determine BayeSED3 root directory: {e}")
            print("Binary may not be able to find data files (data/, models/, nets/).")
            cwd = None

        # Build MPI command - handle PBS/qsub environment
        cmd = [self.mpi_cmd]

        # For PBS/Torque: use PBS_NODEFILE if available (better for multi-node jobs)
        pbs_nodefile = os.environ.get('PBS_NODEFILE')
        if pbs_nodefile and os.path.exists(pbs_nodefile):
            # Use machinefile/hostfile for multi-node PBS jobs
            # This ensures mpirun uses the correct nodes allocated by PBS
            cmd.extend(['--machinefile', pbs_nodefile])
            # Still need -np to specify number of processes
            cmd.extend(['-np', str(self.num_processes)])
        else:
            # Standard mpirun for single-node or non-PBS environments
            cmd.extend(['--use-hwthread-cpus', '-np', str(self.num_processes)])

        cmd.append(self.executable_path)
        cmd.extend(args)

        # Add Ntest if specified
        if self.Ntest is not None:
            cmd.extend(['--Ntest', str(self.Ntest)])

        # Progress indicator: Execution
        print("\nStep 2/3: Running BayeSED analysis...")
        # Print full command - properly escape arguments with spaces or special characters
        cmd_str_parts = [shlex.quote(arg) for arg in cmd]
        print(f"  Full command: {' '.join(cmd_str_parts)}")
        print(f"  MPI mode: {self.mpi_mode}, Processes: {self.num_processes}")
        # Show job scheduler info if detected
        if os.environ.get('PBS_NODEFILE'):
            print(f"  Job scheduler: PBS/qsub (using PBS_NODEFILE)")
        elif os.environ.get('PBS_NCPUS'):
            print(f"  Job scheduler: PBS/qsub (using PBS_NCPUS)")
        elif os.environ.get('SLURM_CPUS_PER_TASK') or os.environ.get('SLURM_NTASKS'):
            print(f"  Job scheduler: SLURM")
        elif os.environ.get('NSLOTS'):
            print(f"  Job scheduler: SGE")
        if self.Ntest is not None:
            print(f"  Processing first {self.Ntest} object(s) only")
        print("-" * 70)

        try:
            # Set cwd for conda installations so binary can find data files
            popen_kwargs = {
                'stdout': subprocess.PIPE,
                'stderr': subprocess.STDOUT,
                'universal_newlines': True,
                'bufsize': 1
            }
            if cwd is not None:
                popen_kwargs['cwd'] = cwd

            self.process = subprocess.Popen(cmd, **popen_kwargs)

            output_lines = []
            for line in iter(self.process.stdout.readline, ''):
                print(line, end='', flush=True)
                output_lines.append(line)

            self.process.wait()
            output_text = ''.join(output_lines)

            if self.process.returncode == 0:
                print("-" * 70)
                print("\nStep 3/3: Inference completed successfully!")
                print("=" * 70)

                # Create summary information
                summary_lines = [
                    f"Results saved to: {params.outdir}",
                    f"MPI mode: {self.mpi_mode}",
                    f"Processes used: {self.num_processes}",
                ]

                if self.Ntest is not None:
                    summary_lines.append(f"Objects processed: {self.Ntest} (test run)")
                else:
                    # Try to count objects from input file
                    try:
                        n_objects = self._count_objects(params.input_file)
                        summary_lines.append(f"Objects in input: {n_objects}")
                    except:
                        pass

                if params.multinest:
                    summary_lines.append(f"MultiNest nlive: {params.multinest.nlive}")

                # Print summary
                for line in summary_lines:
                    print(f"  {line}")

                print("=" * 70)
                print("\nTo load results, use:")
                print(f"  results = bayesed.load_results('{params.outdir}')")
                print("=" * 70 + "\n")
                return self.process
            else:
                # Raise a more informative error
                error_msg = self._parse_error_message(output_text, self.process.returncode)
                raise BayeSEDExecutionError(
                    f"BayeSED execution failed with return code {self.process.returncode}",
                    returncode=self.process.returncode,
                    output=output_text,
                    suggestions=error_msg.get('suggestions', []),
                    error_type=error_msg.get('type', 'unknown')
                )

        except BayeSEDExecutionError:
            raise  # Re-raise our custom errors
        except subprocess.SubprocessError as e:
            raise BayeSEDExecutionError(
                f"Failed to execute BayeSED: {str(e)}",
                returncode=None,
                output=None,
                suggestions=[
                    "Check that the binary executable exists and is executable",
                    "Verify OpenMPI is properly installed",
                    "Check file permissions on the executable"
                ],
                error_type='subprocess_error'
            ) from e
        except Exception as e:
            raise BayeSEDExecutionError(
                f"Unexpected error during BayeSED execution: {str(e)}",
                returncode=None,
                output=None,
                suggestions=["Check the error message above for details"],
                error_type='unexpected_error'
            ) from e

    def _parse_error_message(self, output_text, returncode):
        """
        Parse BayeSED error output to extract useful information and suggestions.

        Parameters
        ----------
        output_text : str
            Complete output text from BayeSED execution
        returncode : int
            Process return code

        Returns
        -------
        dict
            Dictionary with 'type', 'suggestions', and other parsed information
        """
        suggestions = []
        error_type = 'unknown'

        output_lower = output_text.lower()

        # Check for common error patterns
        if 'file not found' in output_lower or 'no such file' in output_lower:
            error_type = 'file_not_found'
            suggestions.append("Check that input files exist and paths are correct")
            suggestions.append("Verify file paths are absolute or relative to current working directory")

        if 'iprior' in output_lower and ('missing' in output_lower or 'not found' in output_lower):
            error_type = 'missing_iprior'
            suggestions.append("Required .iprior files are missing - they should be auto-generated")
            suggestions.append("Check that model configuration is correct")
            suggestions.append("Try running with verbose mode (multinest.fb=1-3) to see which files are needed")

        if 'mpi' in output_lower and ('error' in output_lower or 'failed' in output_lower):
            error_type = 'mpi_error'
            suggestions.append("Check OpenMPI installation")
            suggestions.append(f"Verify MPI processes ({self.num_processes}) don't exceed available resources")
            if self.mpi_mode == 'mn_n':
                suggestions.append("For mpi_mode='n', ensure Ncpu <= Nobj + 1")

        if 'memory' in output_lower or 'out of memory' in output_lower:
            error_type = 'memory_error'
            suggestions.append("Reduce number of MPI processes")
            suggestions.append("Reduce MultiNest nlive parameter")
            suggestions.append("Process fewer objects at once (use Ntest parameter)")

        if 'timeout' in output_lower:
            error_type = 'timeout'
            suggestions.append("Inference took too long - consider reducing model complexity")
            suggestions.append("Try reducing MultiNest nlive parameter")
            suggestions.append("Use Ntest parameter to test with fewer objects first")

        if 'at least one model must be given' in output_lower:
            error_type = 'no_model'
            suggestions.append("Specify at least one model component (SSP, AGN, etc.)")
            suggestions.append("Check that model parameters are correctly configured")

        if 'constraint' in output_lower and 'cpu' in output_lower:
            error_type = 'cpu_constraint'
            suggestions.append("For mpi_mode='n', Ncpu must be <= Nobj + 1")
            suggestions.append("Either reduce number of CPUs or switch to mpi_mode='1'")

        # If no specific pattern matched, provide generic suggestions
        if not suggestions:
            suggestions.append("Check the output above for specific error messages")
            suggestions.append("Verify all input files exist and are correctly formatted")
            suggestions.append("Try running with verbose mode (multinest.fb=1-3) for more details")

        return {
            'type': error_type,
            'suggestions': suggestions
        }

    def check_config(self, params):
        """
        Check configuration without running analysis (dry-run validation).

        Parameters
        ----------
        params : BayeSEDParams
            Parameters to validate

        Returns
        -------
        dict
            Dictionary with validation results:
            - 'valid': bool - Whether configuration is valid
            - 'errors': list - List of error messages (empty if valid)
            - 'warnings': list - List of warning messages
        """
        errors = []
        warnings = []

        try:
            params.validate(check_files=True)
        except BayeSEDValidationError as e:
            errors.extend(e.errors)

        # Additional checks that don't raise errors
        if self.mpi_mode == 'mn_n':
            try:
                self._validate_cpu_constraint(params)
            except ValueError as e:
                errors.append(str(e))

        # Check if output directory exists
        if hasattr(params, 'outdir') and params.outdir:
            outdir_parent = os.path.dirname(params.outdir) if os.path.dirname(params.outdir) else '.'
            if not os.path.exists(outdir_parent):
                warnings.append(f"Output directory parent does not exist: {outdir_parent}")

        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings
        }

    @staticmethod
    def get_default_params(input_file, model_type='galaxy'):
        """
        Get default BayeSEDParams template for common use cases.

        This method provides a starting point for configuration, which can then
        be customized as needed.

        Parameters
        ----------
        input_file : str
            Path to input file
        model_type : str
            Type of model: 'galaxy' (default) or 'agn'

        Returns
        -------
        BayeSEDParams
            Default parameter configuration

        Example
        -------
        >>> bayesed = BayeSEDInterface()
        >>> params = bayesed.get_default_params('data.txt', model_type='galaxy')
        >>> # Customize as needed
        >>> params.multinest.nlive = 400
        >>> bayesed.run(params)
        """
        if model_type.lower() == 'agn':
            return BayeSEDParams.agn(input_file, agn_components=['dsk'])
        else:
            return BayeSEDParams.galaxy(input_file)

    def _format_fann_params(self, fann_params):
        return f"{fann_params.igroup},{fann_params.id},{fann_params.name},{fann_params.iscalable}"

    def _format_AGN_params(self, AGN_params):
        return f"{AGN_params.igroup},{AGN_params.id},{AGN_params.name},{AGN_params.iscalable},{AGN_params.imodel},{AGN_params.icloudy},{AGN_params.suffix},{AGN_params.w_min},{AGN_params.w_max},{AGN_params.Nw}"

    def _format_blackbody_params(self, blackbody_params):
        return f"{blackbody_params.igroup},{blackbody_params.id},{blackbody_params.bb},{blackbody_params.iscalable},{blackbody_params.w_min},{blackbody_params.w_max},{blackbody_params.Nw}"

    def _format_big_blue_bump_params(self, big_blue_bump_params):
        return f"{big_blue_bump_params.igroup},{big_blue_bump_params.id},{big_blue_bump_params.name},{big_blue_bump_params.iscalable},{big_blue_bump_params.w_min},{big_blue_bump_params.w_max},{big_blue_bump_params.Nw}"

    def _format_greybody_params(self, greybody_params):
        return f"{greybody_params.igroup},{greybody_params.id},{greybody_params.name},{greybody_params.iscalable},{greybody_params.ithick},{greybody_params.w_min},{greybody_params.w_max},{greybody_params.Nw}"

    def _format_aknn_params(self, aknn_params):
        return f"{aknn_params.igroup},{aknn_params.id},{aknn_params.name},{aknn_params.iscalable},{aknn_params.k},{aknn_params.f_run},{aknn_params.eps},{aknn_params.iRad},{aknn_params.iprep},{aknn_params.Nstep},{aknn_params.alpha}"

    def _format_line_params(self, line_params):
        return f"{line_params.igroup},{line_params.id},{line_params.name},{line_params.iscalable},{line_params.file},{line_params.R},{line_params.Nsample},{line_params.Nkin}"

    def _format_luminosity_params(self, luminosity_params):
        return f"{luminosity_params.id},{luminosity_params.w_min},{luminosity_params.w_max}"

    def _format_np_sfh_params(self, np_sfh_params):
        return f"{np_sfh_params.prior_type},{np_sfh_params.interpolation_method},{np_sfh_params.num_bins},{np_sfh_params.regul}"

    def _format_polynomial_params(self, polynomial_params):
        return str(polynomial_params.order)

    def _format_powerlaw_params(self, powerlaw_params):
        return f"{powerlaw_params.igroup},{powerlaw_params.id},{powerlaw_params.pw},{powerlaw_params.iscalable},{powerlaw_params.w_min},{powerlaw_params.w_max},{powerlaw_params.Nw}"

    def _format_rbf_params(self, rbf_params):
        return f"{rbf_params.igroup},{rbf_params.id},{rbf_params.name},{rbf_params.iscalable}"

    def _format_sfh_params(self, sfh_params):
        return f"{sfh_params.id},{sfh_params.itype_sfh},{sfh_params.itruncated},{sfh_params.itype_ceh}"

    def _format_ssp_params(self, ssp_params):
        return f"{ssp_params.igroup},{ssp_params.id},{ssp_params.name},{ssp_params.iscalable},{ssp_params.k},{ssp_params.f_run},{ssp_params.Nstep},{ssp_params.i0},{ssp_params.i1},{ssp_params.i2},{ssp_params.i3}"

    def _format_sedlib_params(self, sedlib_params):
        return f"{sedlib_params.igroup},{sedlib_params.id},{sedlib_params.name},{sedlib_params.iscalable},{sedlib_params.dir},{sedlib_params.itype},{sedlib_params.f_run},{sedlib_params.ikey}"

    def _format_sys_err_params(self, sys_err_params):
        return f"{sys_err_params.iprior_type},{sys_err_params.is_age},{sys_err_params.min},{sys_err_params.max},{sys_err_params.nbin}"

    def _format_z_params(self, z_params):
        return f"{z_params.iprior_type},{z_params.is_age},{z_params.min},{z_params.max},{z_params.nbin}"

    def _format_SFR_over_params(self, SFR_over_params):
        return f"{SFR_over_params.past_Myr1},{SFR_over_params.past_Myr2},{SFR_over_params.save_bestfit}"

    def _format_SNRmin1_params(self, SNRmin1_params):
        return f"{SNRmin1_params.phot},{SNRmin1_params.spec}"

    def _format_SNRmin2_params(self, SNRmin2_params):
        return f"{SNRmin2_params.phot},{SNRmin2_params.spec}"

    def _format_NNLM_params(self, NNLM_params):
        return f"{NNLM_params.method},{NNLM_params.Niter1},{NNLM_params.tol1},{NNLM_params.Niter2},{NNLM_params.tol2},{NNLM_params.p1},{NNLM_params.p2}"

    def _format_Ndumper_params(self, Ndumper_params):
        return f"{Ndumper_params.max_number},{Ndumper_params.iconverged_min},{Ndumper_params.Xmin_squared_Nd}"

    def _format_output_SFH_params(self, output_SFH_params):
        return f"{output_SFH_params.ntimes},{output_SFH_params.ilog}"

    def _format_multinest_params(self, multinest_params):
        return f"{int(multinest_params.INS)},{int(multinest_params.mmodal)},{int(multinest_params.ceff)},{multinest_params.nlive},{multinest_params.efr},{multinest_params.tol},{multinest_params.updInt},{multinest_params.Ztol},{multinest_params.seed},{multinest_params.fb},{int(multinest_params.resume)},{int(multinest_params.outfile)},{multinest_params.logZero},{multinest_params.maxiter},{multinest_params.acpt}"

    def _format_gsl_integration_qag_params(self, gsl_integration_qag_params):
        return f"{gsl_integration_qag_params.epsabs},{gsl_integration_qag_params.epsrel},{gsl_integration_qag_params.limit}"

    def _format_gsl_multifit_robust_params(self, gsl_multifit_robust_params):
        return f"{gsl_multifit_robust_params.type},{gsl_multifit_robust_params.tune}"

    def _format_kin_params(self, kin_params):
        return f"{kin_params.id},{kin_params.velscale},{kin_params.num_gauss_hermites_con},{kin_params.num_gauss_hermites_eml}"

    def _format_LineList_params(self, LineList_params):
        return f"{LineList_params.file},{LineList_params.type}"

    def _format_make_catalog_params(self, make_catalog_params):
        return f"{make_catalog_params.id1},{make_catalog_params.logscale_min1},{make_catalog_params.logscale_max1},{make_catalog_params.id2},{make_catalog_params.logscale_min2},{make_catalog_params.logscale_max2}"

    def _format_cloudy_params(self, cloudy_params):
        return f"{cloudy_params.igroup},{cloudy_params.id},{cloudy_params.cloudy},{cloudy_params.iscalable}"

    def _format_cosmology_params(self, cosmology_params):
        return f"{cosmology_params.H0},{cosmology_params.omigaA},{cosmology_params.omigam}"

    def _format_dal_params(self, dal_params):
        return f"{dal_params.id},{dal_params.con_eml_tot},{dal_params.ilaw}"

    def _format_rdf_params(self, rdf_params):
        return f"{rdf_params.id},{rdf_params.num_polynomials}"

    def _format_template_params(self, template_params):
        return f"{template_params.igroup},{template_params.id},{template_params.name},{template_params.iscalable}"

    def load_results(self, output_dir, catalog_name, model_config=None, object_id=None):
        """
        Load BayeSED analysis results from output directory.

        Parameters
        ----------
        output_dir : str
            Directory containing BayeSED output files
        catalog_name : str
            Catalog name to load results for. Use list_catalog_names_in_directory() to discover available catalogs.
        model_config : str or int, optional
            Model configuration to load. Required when multiple configurations exist for the catalog.
            Can be:
            - Full model name (exact match)
            - Partial model name (must match exactly one configuration)
            - Integer index (0-based)
        object_id : str or int, optional
            Object ID to load results for (if None, loads all objects)

        Returns
        -------
        BayeSEDResults
            Results object containing loaded data

        Raises
        ------
        ValueError
            If catalog_name is not found or multiple model configurations exist and model_config is not specified

        Examples
        --------
        >>> # Discover available catalogs first
        >>> from bayesed import list_catalog_names
        >>> catalogs = list_catalog_names('output')
        >>> print("Available catalogs:", catalogs)
        >>>
        >>> # Load results with explicit catalog
        >>> interface = BayeSEDInterface()
        >>> results = interface.load_results('output', 'my_catalog')
        >>>
        >>> # Load specific model configuration
        >>> results = interface.load_results('output', 'my_catalog', model_config='config_with_NNLM')
        """
        # Import enhanced BayeSEDResults from results module
        from .results import BayeSEDResults
        return BayeSEDResults(output_dir, catalog_name, model_config=model_config, object_id=object_id)

    def prepare_input_catalog(self, output_path, catalog_name, ids, z_min, z_max,
                              distance_mpc=None, ebv=None,
                              phot_band_names=None, phot_fluxes=None, phot_errors=None,
                              phot_type="fnu", phot_flux_limits=None, phot_mag_limits=None,
                              phot_nsigma=None, other_columns=None,
                              spec_band_names=None, spec_wavelengths=None,
                              spec_fluxes=None, spec_errors=None, spec_lsf_sigma=None,
                              spec_flux_type="fnu"):
        """
        Create input catalog file for BayeSED analysis - convenience method.

        This method wraps the standalone `create_input_catalog()` function, making it
        easy to create input catalog files from data arrays. The input catalog file
        is required for BayeSED analysis and contains photometry and/or spectroscopy
        data for one or more objects.

        Parameters
        ----------
        output_path : str
            Path where the input catalog file will be created
        catalog_name : str
            Name of the catalog (used in header)
        ids : array-like
            Object IDs (length N)
        z_min, z_max : array-like
            Redshift ranges (length N)
        distance_mpc : array-like, optional
            Distance in Mpc (length N). Defaults to zeros if not provided.
        ebv : array-like, optional
            E(B-V) extinction values (length N). Defaults to zeros if not provided.
        phot_band_names : list, optional
            List of photometric band names
        phot_fluxes : array-like, optional
            Photometric fluxes (N, Nphot). Units depend on phot_type.
        phot_errors : array-like, optional
            Photometric errors (N, Nphot). Units depend on phot_type.
        phot_type : str
            Photometry type: "fnu" (μJy) or "abmag" (AB magnitudes). Default: "fnu"
        phot_flux_limits : array-like, optional
            Flux limits for nondetections (μJy). Used when phot_type="fnu".
        phot_mag_limits : array-like, optional
            Magnitude limits for nondetections (AB mag). Used when phot_type="abmag".
        phot_nsigma : float, optional
            Number of sigma for limits (used with phot_flux_limits or phot_mag_limits)
        other_columns : dict, optional
            Dict mapping column names to array-like data (length N).
            Values can be numpy arrays, lists, pandas Series, or astropy Table Columns.
            Example: {'ra': [1.0, 2.0, 3.0], 'dec': [0.1, 0.2, 0.3]}
            Note: Uses dict type because column names and count are unknown in advance,
            unlike fixed parameters like distance_mpc and ebv which have known names.
        spec_band_names : list, optional
            List of spectroscopic band names
        spec_wavelengths : list of array-like, optional
            List of wavelength arrays, one per band in spec_band_names.
            Each array should be (Nw,) for single object or (N, Nw) for multiple objects, in Angstrom.
        spec_fluxes : list of array-like, optional
            List of flux arrays, one per band in spec_band_names.
            Each array should be (Nw,) for single object or (N, Nw) for multiple objects.
        spec_errors : list of array-like, optional
            List of error arrays, one per band in spec_band_names.
            Each array should be (Nw,) for single object or (N, Nw) for multiple objects.
        spec_lsf_sigma : list of array-like, optional
            List of LSF sigma arrays, one per band in spec_band_names.
            Each array should be (Nw,) for single object or (N, Nw) for multiple objects, in microns.
        spec_flux_type : str
            Spectral flux type: "fnu" (μJy) or "flambda" (erg s^-1 cm^-2 Å^-1). Default: "fnu"

        Returns
        -------
        list of str
            List of band names (photometry and spectral bands combined).
            Empty list if no bands are provided.

        Example
        -------
        >>> bayesed = BayeSEDInterface()
        >>> band_names = bayesed.prepare_input_catalog(
        ...     output_path='observation/my_catalog.txt',
        ...     catalog_name='my_galaxy_sample',
        ...     ids=[1, 2, 3],
        ...     z_min=[0.1, 0.2, 0.3],
        ...     z_max=[0.2, 0.3, 0.4],
        ...     phot_band_names=['u', 'g', 'r'],
        ...     phot_fluxes=fluxes,  # shape (3, 3) in μJy
        ...     phot_errors=errors   # shape (3, 3) in μJy
        ... )
        >>> params = BayeSEDParams(
        ...     input_type=0,
        ...     input_file='observation/my_catalog.txt',
        ...     ...
        ... )
        """
        import os

        # Convert inputs to numpy arrays using shared utility (handles pandas Series, astropy Columns, etc.)
        ids = _to_array(ids)
        z_min = _to_array(z_min)
        z_max = _to_array(z_max)
        distance_mpc = _to_array(distance_mpc)
        ebv = _to_array(ebv)
        phot_fluxes = _to_array(phot_fluxes)
        phot_errors = _to_array(phot_errors)
        phot_flux_limits = _to_array(phot_flux_limits)
        phot_mag_limits = _to_array(phot_mag_limits)
        # spec_* are lists of arrays
        if spec_wavelengths is not None:
            spec_wavelengths = _to_array(spec_wavelengths)
        if spec_fluxes is not None:
            spec_fluxes = _to_array(spec_fluxes)
        if spec_errors is not None:
            spec_errors = _to_array(spec_errors)
        if spec_lsf_sigma is not None:
            spec_lsf_sigma = _to_array(spec_lsf_sigma)
        # other_columns dict values
        if other_columns:
            other_columns = {k: _to_array(v) for k, v in other_columns.items()}

        # Ensure output directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        # Call the standalone function with numpy arrays (for optimal performance)
        band_names = create_input_catalog(
            output_path=output_path,
            catalog_name=catalog_name,
            ids=ids,
            z_min=z_min,
            z_max=z_max,
            distance_mpc=distance_mpc,
            ebv=ebv,
            phot_band_names=phot_band_names,
            phot_fluxes=phot_fluxes,
            phot_errors=phot_errors,
            phot_type=phot_type,
            phot_flux_limits=phot_flux_limits,
            phot_mag_limits=phot_mag_limits,
            phot_nsigma=phot_nsigma,
            other_columns=other_columns,
            spec_band_names=spec_band_names,
            spec_wavelengths=spec_wavelengths,
            spec_fluxes=spec_fluxes,
            spec_errors=spec_errors,
            spec_lsf_sigma=spec_lsf_sigma,
            spec_flux_type=spec_flux_type
        )

        return band_names

    def prepare_filters_from_svo(self, svo_filter_ids, output_dir='observation/filters',
                                 filters_file=None, filters_selected_file=None,
                                 overwrite=False, verbose=True):
        """
        Download filters from SVO and create filter selection file - convenience method.

        This method downloads filter transmission curves from SVO Filter Profile Service
        and optionally creates a filter selection file, making it easy to set up filters
        for analysis.

        Parameters
        ----------
        svo_filter_ids : list of str
            List of SVO filter IDs (e.g., ['SLOAN/SDSS.u', 'SLOAN/SDSS.g'])
        output_dir : str
            Directory where filter files will be saved
        filters_file : str, optional
            Path for filters file (default: `{output_dir}/filters.txt`)
        filters_selected_file : str, optional
            Path for filter selection file (default: `{output_dir}/filters_selected.txt`)
        overwrite : bool
            Whether to overwrite existing files
        verbose : bool
            Whether to print progress messages

        Returns
        -------
        dict
            Dictionary with 'filters_file' and 'filters_selected_file' paths

        Example
        -------
        >>> bayesed = BayeSEDInterface()
        >>> filter_files = bayesed.prepare_filters_from_svo(
        ...     ['SLOAN/SDSS.u', 'SLOAN/SDSS.g', 'SLOAN/SDSS.r'],
        ...     output_dir='observation/filters'
        ... )
        >>> params.filters = filter_files['filters_file']
        >>> params.filters_selected = filter_files['filters_selected_file']
        """
        import os

        os.makedirs(output_dir, exist_ok=True)

        if filters_file is None:
            filters_file = os.path.join(output_dir, 'filters.txt')
        if filters_selected_file is None:
            filters_selected_file = os.path.join(output_dir, 'filters_selected.txt')

        create_filters_from_svo(
            svo_filterIDs=svo_filter_ids,
            filters_file=filters_file,
            filters_selected_file=filters_selected_file,
            overwrite=overwrite,
            verbose=verbose
        )

        return {
            'filters_file': filters_file,
            'filters_selected_file': filters_selected_file
        }

    def prepare_filters_selected(self, filters_file, output_selection_file=None,
                                 selected_indices=None, filter_names=None,
                                 output_filters_file=None, **kwargs):
        """
        Create filter selection file and optionally new filter definition file from existing filters.

        This method creates a filter selection file from an existing filter definition file(s),
        and optionally creates a new combined filter definition file containing only the selected
        filters' transmission data. This is especially useful when working with multiple filter
        files or when you want to create a new filter file from existing filters in a folder.

        Parameters
        ----------
        filters_file : str or list of str
            Path to filter definition file(s). Can be a single file or a list of files.
            When multiple files are provided, filters from all files are combined.
        output_selection_file : str, optional
            Path for output filter selection file (default: `{filters_file}_selected.txt`)
        selected_indices : list of int, optional
            Indices of filters to select (0-based). If None, all filters are selected.
        filter_names : list of str, optional
            Names of filters to select. Must have same length as selected_indices.
        output_filters_file : str, optional
            Path for output combined filter definition file containing transmission curves
            for selected filters. If provided, a new filter file will be created with only
            the selected filters' transmission data. This is recommended when combining
            multiple filter files or creating a new filter file from existing filters.
            If None, no new filter file is created.
        **kwargs
            Additional arguments passed to create_filters_selected() (mag_lim, SNR_min, etc.)

        Returns
        -------
        dict
            Dictionary with keys:
            - 'selection_file': Path to the created filter selection file
            - 'filters_file': Path to the created filter definition file (if output_filters_file provided)
            - 'filters_table': astropy.table.Table with filter information

        Example
        -------
        >>> bayesed = BayeSEDInterface()
        >>>
        >>> # Create selection file from single filter file
        >>> result = bayesed.prepare_filters_selected(
        ...     'observation/filters/filters.txt',
        ...     selected_indices=[0, 1, 2]  # Select first 3 filters
        ... )
        >>> params.filters_selected = result['selection_file']
        >>>
        >>> # Create new filter file and selection from multiple filter files
        >>> result = bayesed.prepare_filters_selected(
        ...     ['filters_optical.txt', 'filters_nir.txt', 'filters_mir.txt'],
        ...     output_selection_file='filters_selected.txt',
        ...     output_filters_file='filters_combined.txt',  # Create new filter file
        ...     selected_indices=[0, 2, 4, 10, 15]
        ... )
        >>> params.filters = result['filters_file']
        >>> params.filters_selected = result['selection_file']
        >>>
        >>> # Create new filter file from existing filters in a folder
        >>> import glob
        >>> filter_files = glob.glob('observation/filters/*.txt')
        >>> result = bayesed.prepare_filters_selected(
        ...     filter_files,
        ...     output_filters_file='observation/filters/filters_combined.txt',
        ...     output_selection_file='observation/filters/filters_selected.txt',
        ...     selected_indices=[0, 1, 2, 5, 8]  # Select specific filters
        ... )
        >>> params.filters = result['filters_file']
        >>> params.filters_selected = result['selection_file']
        """
        import os

        if output_selection_file is None:
            if isinstance(filters_file, list):
                base = os.path.splitext(filters_file[0])[0]
            else:
                base = os.path.splitext(filters_file)[0]
            output_selection_file = base + '_selected.txt'

        # Call create_filters_selected with output_filters_file support
        filters_table = create_filters_selected(
            filters_file=filters_file,
            output_selection_file=output_selection_file,
            selected_indices=selected_indices,
            filter_names=filter_names,
            output_filters_file=output_filters_file,
            **kwargs
        )

        result = {
            'selection_file': output_selection_file,
            'filters_table': filters_table
        }

        if output_filters_file:
            result['filters_file'] = output_filters_file

        return result

    def _params_to_args(self, params):
        if isinstance(params, list):
            return params

        args = []

        if params.help:
            args.append('-h')
            return args

        args.extend([
            '-i', f"{params.input_type},{params.input_file}",
            '--outdir', params.outdir,
            '--save_bestfit', str(params.save_bestfit),
            '-v', str(params.verbose)
        ])

        # Add other command line arguments of bayesed
        if params.fann:
            for fann_params in params.fann:
                args.extend(['-a', self._format_fann_params(fann_params)])

        if params.AGN:
            for AGN_params in params.AGN:
                args.extend(['-AGN', self._format_AGN_params(AGN_params)])

        if params.blackbody:
            for blackbody_params in params.blackbody:
                args.extend(['-bb', self._format_blackbody_params(blackbody_params)])

        if params.big_blue_bump:
            for big_blue_bump_params in params.big_blue_bump:
                args.extend(['-bbb', self._format_big_blue_bump_params(big_blue_bump_params)])

        if params.greybody:
            for greybody_params in params.greybody:
                args.extend(['-gb', self._format_greybody_params(greybody_params)])

        if params.aknn:
            for aknn_params in params.aknn:
                args.extend(['-k', self._format_aknn_params(aknn_params)])

        if params.line:
            for line_params in params.line:
                args.extend(['-l', self._format_line_params(line_params)])

        if params.lines:
            for lines_params in params.lines:
                args.extend(['-ls', self._format_line_params(lines_params)])

        if params.lines1:
            for line_params in params.lines1:
                args.extend(['-ls1', self._format_line_params(line_params)])

        if params.luminosity:
            args.extend(['--luminosity', self._format_luminosity_params(params.luminosity)])

        if params.np_sfh:
            args.extend(['--np_sfh', self._format_np_sfh_params(params.np_sfh)])

        if params.polynomial:
            args.extend(['--polynomial', self._format_polynomial_params(params.polynomial)])

        if params.powerlaw:
            for powerlaw_params in params.powerlaw:
                args.extend(['-pw', self._format_powerlaw_params(powerlaw_params)])

        if params.rbf:
            for rbf_params in params.rbf:
                args.extend(['-rbf', self._format_rbf_params(rbf_params)])

        if params.sfh:
            for sfh_params in params.sfh:
                args.extend(['--sfh', self._format_sfh_params(sfh_params)])

        if params.ssp:
            for ssp_params in params.ssp:
                args.extend(['-ssp', self._format_ssp_params(ssp_params)])

        if params.sedlib:
            for sedlib_params in params.sedlib:
                args.extend(['-sedlib', self._format_sedlib_params(sedlib_params)])

        if params.sys_err_mod:
            args.extend(['--sys_err_mod', self._format_sys_err_params(params.sys_err_mod)])

        if params.sys_err_obs:
            args.extend(['--sys_err_obs', self._format_sys_err_params(params.sys_err_obs)])

        if params.z:
            args.extend(['--z', self._format_z_params(params.z)])

        if params.rename:
            for rename_params in params.rename:
                args.extend(['--rename', f"{rename_params.id},{rename_params.ireplace},{rename_params.name}"])

        if params.rename_all:
            args.extend(['--rename_all', params.rename_all])

        if params.save_pos_sfh:
            args.extend(['--save_pos_sfh', params.save_pos_sfh])

        if params.save_pos_spec:
            args.append('--save_pos_spec')

        if params.save_sample_obs:
            args.append('--save_sample_obs')

        if params.save_sample_spec:
            args.append('--save_sample_spec')

        if params.save_summary:
            args.append('--save_summary')

        if params.suffix:
            args.extend(['--suffix', params.suffix])

        if params.SFR_over:
            args.extend(['--SFR_over', f"{params.SFR_over.past_Myr1},{params.SFR_over.past_Myr2}"])

        if params.SNRmin1:
            args.extend(['--SNRmin1', f"{params.SNRmin1.phot},{params.SNRmin1.spec}"])

        if params.SNRmin2:
            args.extend(['--SNRmin2', f"{params.SNRmin2.phot},{params.SNRmin2.spec}"])

        if params.unweighted_samples:
            args.append('--unweighted_samples')

        if params.filters:
            args.extend(['--filters', params.filters])

        if params.filters_selected:
            args.extend(['--filters_selected', params.filters_selected])

        if params.no_spectra_fit:
            args.append('--no_spectra_fit')

        if params.NNLM:
            args.extend(['--NNLM', self._format_NNLM_params(params.NNLM)])

        if params.Ndumper:
            args.extend(['--Ndumper', self._format_Ndumper_params(params.Ndumper)])

        if params.NfilterPoints:
            args.extend(['--NfilterPoints', str(params.NfilterPoints)])

        if params.Nsample:
            args.extend(['--Nsample', str(params.Nsample)])

        if params.output_SFH:
            args.extend(['--output_SFH', self._format_output_SFH_params(params.output_SFH)])

        if params.output_mock_photometry:
            args.extend(['--output_mock_photometry', str(params.output_mock_photometry)])

        if params.multinest:
            args.extend(['--multinest', self._format_multinest_params(params.multinest)])

        if params.gsl_integration_qag:
            args.extend(['--gsl_integration_qag', f"{params.gsl_integration_qag.epsabs},{params.gsl_integration_qag.epsrel},{params.gsl_integration_qag.limit}"])

        if params.gsl_multifit_robust:
            args.extend(['--gsl_multifit_robust', f"{params.gsl_multifit_robust.type},{params.gsl_multifit_robust.tune}"])

        if params.import_files:
            for import_file in params.import_files:
                args.extend(['--import', import_file])

        if params.inn:
            for inn_params in params.inn:
                args.extend(['--inn', self._format_sedlib_params(inn_params)])

        if params.IGM is not None:
            args.extend(['--IGM', str(params.IGM)])

        if params.kin:
            for kin_param in params.kin:
                args.extend(['--kin', self._format_kin_params(kin_param)])

        if params.logZero is not None:
            args.extend(['--logZero', str(params.logZero)])

        if params.lw_max is not None:
            args.extend(['--lw_max', str(params.lw_max)])

        if params.LineList:
            args.extend(['--LineList', self._format_LineList_params(params.LineList)])

        if params.make_catalog:
            args.extend(['--make_catalog', self._format_make_catalog_params(params.make_catalog)])

        if params.niteration is not None:
            args.extend(['--niteration', str(params.niteration)])

        if params.no_photometry_fit:
            args.append('--no_photometry_fit')

        if params.cloudy:
            for cloudy_params in params.cloudy:
                args.extend(['--cloudy', self._format_cloudy_params(cloudy_params)])

        if params.cosmology:
            args.extend(['--cosmology', self._format_cosmology_params(params.cosmology)])

        if params.dal:
            for dal_params in params.dal:
                args.extend(['--dal', self._format_dal_params(dal_params)])

        if params.export:
            args.extend(['--export', params.export])

        if params.output_mock_spectra:
            args.append('--output_mock_spectra')

        if params.output_model_absolute_magnitude:
            args.append('--output_model_absolute_magnitude')

        if params.output_pos_obs:
            args.append('--output_pos_obs')

        if params.priors_only:
            args.append('--priors_only')

        if params.rdf:
            args.extend(['--rdf', self._format_rdf_params(params.rdf)])

        if params.template:
            for template_params in params.template:
                args.extend(['-t', self._format_template_params(template_params)])

        if params.build_sedlib is not None:
            args.extend(['--build_sedlib', str(params.build_sedlib)])
        if params.check:
            args.append('--check')
        if params.cl:
            args.extend(['--cl', params.cl])

        if params.save_sample_par:
            args.append('--save_sample_par')

        return args
