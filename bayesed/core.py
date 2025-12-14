"""
Core classes for BayeSED3 Python interface.

This module contains the main classes for configuring and running
BayeSED3 SED analysis: BayeSEDParams, BayeSEDInterface, BayeSEDResults,
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
        >>> params = BayeSEDParams.galaxy('observation/data.txt')
        >>> interface = BayeSEDInterface()
        >>> interface.run(params)
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
            sfh_itype_ceh=0,
            sfh_itruncated=0,
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



class BayeSEDResults:
    """
    Class for loading and accessing BayeSED analysis results.

    This class provides convenient access to output files including:
    - Best-fit parameters and spectra
    - Posterior distribution samples
    - Bayesian evidence values
    - Fit statistics
    - Easy parameter name access and categorization
    - Parameter value extraction and analysis

    Key Features
    ------------
    - **Parameter Discovery**: Easy access to free and derived parameter names
    - **Parameter Categorization**: Group parameters by type (stellar, dust, SFH, AGN, etc.)
    - **Value Extraction**: Get parameter values for specific objects or parameters
    - **Summary Methods**: Quick overview of parameter structure and counts
    - **Flexible Data Sources**: Read from HDF5 files or posterior sample files

    BayeSED Output Structure:
    -------------------------
    A single BayeSED run with specific SED model, galaxy sample, and outdir produces:

    1. **Global HDF5 File** (one per model configuration):
       - Location: Top-level of output_dir
       - Naming: `{catalog_name}_{model_name}.hdf5`
         - catalog_name: Extracted from input file header (first line: `# catalog_name ...`)
         - Model name: Encodes model components (e.g., '0Stellar+Nebular_2dal8_10_sys_err0')
       - Contains:
         * 'ID' dataset: Object IDs (strings)
         * 'parameters' dataset: Best-fit parameters for all objects (N_objects × N_params)
         * 'parameters_name' dataset: Parameter names (strings)
         * Columns include: logZ, INSlogZ, logZerr, INSlogZerr (Bayesian evidence)
       - Example: `test_inoise1_0Stellar+Nebular_2dal8_10_sys_err0.hdf5`

    2. **Catalog Name Subdirectory** (created when any save_ command is used):
       - Location: `{output_dir}/{catalog_name}/`
       - catalog_name: Extracted from input file header (first line: `# catalog_name ...`)
       - Contains: Subdirectories for each object in the sample

    3. **Object-Specific Subdirectories**:
       - Location: `{output_dir}/{catalog_name}/{object_id}/`
       - Structure: One subdirectory per object
       - Example: `output1/test_inoise1/spec-0285-51930-0184_GALAXY_STARFORMING/`

    4. **Best-fit FITS Files** (if save_bestfit > 0):
       - Location: Inside object subdirectories
       - Naming: `{model_name}_bestfit.fits`
       - Contains: Best-fit SED spectrum, observed data, residuals, component contributions
       - Example: `0Stellar+Nebular_2dal8_10_sys_err0_bestfit.fits`

    5. **Posterior Sample Files** (if save_sample_par=True):
       - Location: Inside object subdirectories
       - Files:
         * `{model_name}_sample_par.paramnames`: Parameter definitions (GetDist format)
         * `{model_name}_sample_par.txt`: Posterior samples (GetDist format)
       - Example: `0Stellar+Nebular_2dal8_10_sys_err0_sample_par.paramnames`

    Complete Example Structure:
        output1/
        ├── test_inoise1_0Stellar+Nebular_2dal8_10_sys_err0.hdf5  # Global results (catalog_name prefix)
        ├── test_inoise1_0Stellar+Nebular_2dal8_10_2bbb_2dal7_15_...hdf5  # Another model config
        └── test_inoise1/  # catalog_name from input file header
            ├── spec-0285-51930-0184_GALAXY_STARFORMING/
            │   ├── 0Stellar+Nebular_2dal8_10_sys_err0_bestfit.fits
            │   ├── 0Stellar+Nebular_2dal8_10_sys_err0_sample_par.paramnames
            │   └── 0Stellar+Nebular_2dal8_10_sys_err0_sample_par.txt
            └── spec-0508-52366-0522_GALAXY_/
                └── ...

    Notes:
    - All HDF5 files in the same output_dir typically contain all objects from the same sample
    - HDF5 filenames use catalog_name as prefix (from input file header)
    - Multiple HDF5 files with the same catalog_name prefix indicate different model configurations
    - Object IDs match between HDF5 files and subdirectory names
    - The catalog_name subdirectory is only created when save_bestfit, save_sample_par, or other save_ options are enabled

    Parameters
    ----------
    output_dir : str
        Directory containing BayeSED output files
    catalog_name : str, optional
        Catalog name (from input file header) to scope results to.
        If None, will auto-detect from available catalogs (uses first if multiple).
        All methods will operate within this catalog's scope.
    object_id : str or int, optional
        Object ID to load results for (if None, loads first available object)

    Examples
    --------
    >>> # Basic usage
    >>> results = BayeSEDResults('output')
    >>>
    >>> # Quick parameter overview
    >>> results.print_parameter_summary()
    >>>
    >>> # Get parameter names by type
    >>> free_params = results.get_free_parameter_names()
    >>> derived_params = results.get_derived_parameter_names()
    >>> stellar_params = results.get_parameters_by_type('stellar')
    >>>
    >>> # Get parameter values
    >>> ages = results.get_parameter_values('log(age/yr)[0,1]')
    >>> stellar_masses = results.get_parameter_values('log(M*/Msun)[0,1]')
    >>>
    >>> # Get multiple parameters at once
    >>> params = results.get_parameter_values(['log(age/yr)[0,1]', 'log(Z/Zsun)[0,1]'])
    >>>
    >>> # Load full results table
    >>> table = results.load_hdf5_results()
    >>> print(table.colnames)  # All available parameters
    """

    def __init__(self, output_dir, catalog_name=None, object_id=None):
        import os
        import glob

        self.output_dir = os.path.abspath(output_dir)
        self.object_id = object_id

        if not os.path.exists(self.output_dir):
            raise FileNotFoundError(f"Output directory does not exist: {self.output_dir}")

        # Find output files first (to detect available catalog_names)
        self._find_output_files()

        # Set catalog_name (auto-detect if not provided)
        if catalog_name is None:
            if self.catalog_names:
                # Use first available catalog_name
                self.catalog_name = self.catalog_names[0]
            elif self.catalog_names_from_hdf5:
                # Use first catalog_name from HDF5 filenames
                self.catalog_name = sorted(self.catalog_names_from_hdf5)[0]
            else:
                self.catalog_name = None
        else:
            # Validate catalog_name exists
            if catalog_name not in self.catalog_names and catalog_name not in self.catalog_names_from_hdf5:
                available = sorted(set(self.catalog_names) | self.catalog_names_from_hdf5)
                raise ValueError(
                    f"Catalog name '{catalog_name}' not found in output directory. "
                    f"Available catalog names: {available}"
                )
            self.catalog_name = catalog_name

        # Re-filter files to scope to this catalog_name
        if self.catalog_name:
            self._scope_to_catalog()

    def _find_output_files(self):
        """
        Find all output files in the directory and organize by model configuration.

        This method:
        1. Finds all HDF5 files (global results) in top-level directory
        2. Detects catalog_name subdirectories (from input file header)
        3. Finds all best-fit FITS files recursively
        4. Finds all posterior sample files recursively
        5. Organizes files by catalog_name, model configuration, and object ID
        6. Selects appropriate files based on object_id if specified

        Note: When save_ commands are used, BayeSED creates a subdirectory named
        after the catalog_name (from input file header) containing object subdirectories.
        Path structure: {output_dir}/{catalog_name}/{object_id}/{model_name}_bestfit.fits
        """
        import os
        import glob

        # Find HDF5 global results files (top-level only)
        hdf5_files = glob.glob(os.path.join(self.output_dir, '*.hdf5'))
        self.hdf5_files = sorted(hdf5_files) if hdf5_files else []
        self.hdf5_file = self.hdf5_files[0] if self.hdf5_files else None

        # Detect catalog_name subdirectories first (more reliable than parsing filenames)
        # These are created when save_ commands are used
        self.catalog_names = []
        if os.path.isdir(self.output_dir):
            for item in os.listdir(self.output_dir):
                item_path = os.path.join(self.output_dir, item)
                # Check if it's a directory and not an HDF5 file
                if os.path.isdir(item_path) and not item.endswith('.hdf5'):
                    # Check if it contains object subdirectories (has subdirectories)
                    try:
                        subitems = os.listdir(item_path)
                        # If it has subdirectories, it's likely a catalog_name directory
                        if any(os.path.isdir(os.path.join(item_path, subitem)) for subitem in subitems):
                            self.catalog_names.append(item)
                    except (OSError, PermissionError):
                        pass

        # Organize HDF5 files by catalog_name (which is the prefix in the filename)
        # Filename format: {catalog_name}_{model_name}.hdf5
        # Since catalog_name can contain underscores, we match HDF5 files to catalog_names
        # by checking if the filename starts with catalog_name + '_'
        self.hdf5_files_by_catalog = {}
        self.catalog_names_from_hdf5 = set()

        # First, try to match HDF5 files to known catalog_names from subdirectories
        for hdf5_file in self.hdf5_files:
            basename = os.path.basename(hdf5_file)
            matched = False
            # Try to match against known catalog_names (from subdirectories)
            for catalog_name in self.catalog_names:
                if basename.startswith(catalog_name + '_'):
                    if catalog_name not in self.hdf5_files_by_catalog:
                        self.hdf5_files_by_catalog[catalog_name] = []
                    self.hdf5_files_by_catalog[catalog_name].append(hdf5_file)
                    self.catalog_names_from_hdf5.add(catalog_name)
                    matched = True
                    break

            # If no match found, fall back to splitting on first underscore
            # (for cases where catalog_name subdirectory doesn't exist)
            if not matched and '_' in basename:
                parts = basename.split('_', 1)
                catalog_name = parts[0]
                self.catalog_names_from_hdf5.add(catalog_name)
                if catalog_name not in self.hdf5_files_by_catalog:
                    self.hdf5_files_by_catalog[catalog_name] = []
                self.hdf5_files_by_catalog[catalog_name].append(hdf5_file)

        # For backward compatibility, also maintain hdf5_files_by_prefix
        # (in case some files use old naming convention)
        self.hdf5_files_by_prefix = {}
        for hdf5_file in self.hdf5_files:
            basename = os.path.basename(hdf5_file)
            if '_' in basename:
                prefix = basename.split('_')[0]
                if prefix not in self.hdf5_files_by_prefix:
                    self.hdf5_files_by_prefix[prefix] = []
                self.hdf5_files_by_prefix[prefix].append(hdf5_file)

        # Merge with catalog_names from HDF5 filenames (for cases where subdirectory doesn't exist)
        self.catalog_names.extend(self.catalog_names_from_hdf5)
        # Remove duplicates and sort
        self.catalog_names = sorted(set(self.catalog_names))

        # If we have a single catalog_name, prefer its HDF5 file as default
        if len(self.catalog_names) == 1 and self.catalog_names[0] in self.hdf5_files_by_catalog:
            catalog_hdf5_files = self.hdf5_files_by_catalog[self.catalog_names[0]]
            if catalog_hdf5_files:
                self.hdf5_file = catalog_hdf5_files[0]

        # Find best-fit FITS files (recursively in subdirectories)
        bestfit_files = glob.glob(os.path.join(self.output_dir, '**', '*_bestfit.fits'), recursive=True)
        self.bestfit_files = sorted(bestfit_files)

        # Organize bestfit files by catalog_name and object ID
        self.bestfit_files_by_catalog = {}
        self.bestfit_files_by_object = {}
        for bfile in self.bestfit_files:
            rel_path = os.path.relpath(bfile, self.output_dir)
            parts = rel_path.split(os.sep) if os.sep in rel_path else [os.path.basename(bfile)]

            # Path structure: {catalog_name}/{object_id}/{model_name}_bestfit.fits
            # or legacy: {prefix}/{object_id}/{model_name}_bestfit.fits
            if len(parts) >= 2:
                # Check if first part is a catalog_name
                catalog_name = parts[0] if parts[0] in self.catalog_names else None
                obj_id = parts[1]  # Second part is object ID

                # Organize by catalog_name
                if catalog_name:
                    if catalog_name not in self.bestfit_files_by_catalog:
                        self.bestfit_files_by_catalog[catalog_name] = {}
                    if obj_id not in self.bestfit_files_by_catalog[catalog_name]:
                        self.bestfit_files_by_catalog[catalog_name][obj_id] = []
                    self.bestfit_files_by_catalog[catalog_name][obj_id].append(bfile)

                # Also organize by object ID (for backward compatibility)
                if obj_id not in self.bestfit_files_by_object:
                    self.bestfit_files_by_object[obj_id] = []
                self.bestfit_files_by_object[obj_id].append(bfile)

        # Find posterior sample files (recursively in subdirectories)
        paramnames_files = glob.glob(os.path.join(self.output_dir, '**', '*_sample_par.paramnames'), recursive=True)

        # Match paramnames and sample files
        self.posterior_files = {}
        self.posterior_files_by_catalog = {}
        self.posterior_files_by_object = {}
        for pfile in paramnames_files:
            base = pfile.replace('_sample_par.paramnames', '')
            sfile = base + '_sample_par.txt'
            if os.path.exists(sfile):
                self.posterior_files[base] = {
                    'paramnames': pfile,
                    'samples': sfile
                }

                # Organize by catalog_name and object ID
                rel_path = os.path.relpath(pfile, self.output_dir)
                if os.sep in rel_path:
                    parts = rel_path.split(os.sep)
                    if len(parts) >= 2:
                        catalog_name = parts[0] if parts[0] in self.catalog_names else None
                        obj_id = parts[1]

                        # Organize by catalog_name
                        if catalog_name:
                            if catalog_name not in self.posterior_files_by_catalog:
                                self.posterior_files_by_catalog[catalog_name] = {}
                            if obj_id not in self.posterior_files_by_catalog[catalog_name]:
                                self.posterior_files_by_catalog[catalog_name][obj_id] = {}
                            self.posterior_files_by_catalog[catalog_name][obj_id][base] = {
                                'paramnames': pfile,
                                'samples': sfile
                            }

                        # Also organize by object ID (for backward compatibility)
                        if obj_id not in self.posterior_files_by_object:
                            self.posterior_files_by_object[obj_id] = {}
                        self.posterior_files_by_object[obj_id][base] = {
                            'paramnames': pfile,
                            'samples': sfile
                        }

        # Select object if specified (before catalog scoping)
        if self.object_id is not None:
            obj_str = str(self.object_id)

            # Try to find bestfit file for this object
            # First check organized by object ID
            if obj_str in self.bestfit_files_by_object:
                # Use first matching file (could be multiple model configs)
                self.bestfit_file = self.bestfit_files_by_object[obj_str][0]
            else:
                # Fallback: search in full paths
                matching = [f for f in self.bestfit_files if obj_str in os.path.relpath(f, self.output_dir)]
                if not matching:
                    matching = [f for f in self.bestfit_files if obj_str in os.path.basename(f)]
                if matching:
                    self.bestfit_file = matching[0]
                else:
                    self.bestfit_file = None
        else:
            # No object specified - use first available
            self.bestfit_file = self.bestfit_files[0] if self.bestfit_files else None

    def _scope_to_catalog(self):
        """
        Filter all file lists to scope to the selected catalog_name.
        This ensures all methods operate within a single catalog's scope.
        """
        import os

        if not self.catalog_name:
            return

        # Filter HDF5 files to this catalog
        if hasattr(self, 'hdf5_files_by_catalog') and self.catalog_name in self.hdf5_files_by_catalog:
            self.hdf5_files = self.hdf5_files_by_catalog[self.catalog_name]
            if self.hdf5_files:
                self.hdf5_file = self.hdf5_files[0]
        elif hasattr(self, 'hdf5_files'):
            # Try to find HDF5 files with this catalog_name prefix
            filtered = [f for f in self.hdf5_files
                        if os.path.basename(f).startswith(self.catalog_name + '_')]
            if filtered:
                self.hdf5_files = filtered
                self.hdf5_file = filtered[0]

        # Filter bestfit files to this catalog
        if hasattr(self, 'bestfit_files_by_catalog') and self.catalog_name in self.bestfit_files_by_catalog:
            # Rebuild bestfit_files list from catalog-scoped files
            self.bestfit_files = []
            for obj_id, files in self.bestfit_files_by_catalog[self.catalog_name].items():
                self.bestfit_files.extend(files)
            self.bestfit_files = sorted(self.bestfit_files)

            # Update bestfit_files_by_object to only include this catalog's objects
            self.bestfit_files_by_object = {}
            for obj_id, files in self.bestfit_files_by_catalog[self.catalog_name].items():
                self.bestfit_files_by_object[obj_id] = files

            # Update bestfit_file selection for object_id
            if self.object_id is not None:
                obj_str = str(self.object_id)
                if obj_str in self.bestfit_files_by_object:
                    self.bestfit_file = self.bestfit_files_by_object[obj_str][0] if self.bestfit_files_by_object[obj_str] else None
                else:
                    self.bestfit_file = None
            else:
                self.bestfit_file = self.bestfit_files[0] if self.bestfit_files else None
        else:
            # No bestfit files for this catalog - ensure attributes exist
            if not hasattr(self, 'bestfit_files'):
                self.bestfit_files = []
            if not hasattr(self, 'bestfit_files_by_object'):
                self.bestfit_files_by_object = {}
            self.bestfit_file = None

        # Filter posterior files to this catalog
        if hasattr(self, 'posterior_files_by_catalog') and self.catalog_name in self.posterior_files_by_catalog:
            # Rebuild posterior_files dict from catalog-scoped files
            self.posterior_files = {}
            for obj_id, obj_files in self.posterior_files_by_catalog[self.catalog_name].items():
                self.posterior_files.update(obj_files)

            # Update posterior_files_by_object to only include this catalog's objects
            self.posterior_files_by_object = {}
            for obj_id, obj_files in self.posterior_files_by_catalog[self.catalog_name].items():
                self.posterior_files_by_object[obj_id] = obj_files
        else:
            # No posterior files for this catalog - ensure attributes exist
            if not hasattr(self, 'posterior_files'):
                self.posterior_files = {}
            if not hasattr(self, 'posterior_files_by_object'):
                self.posterior_files_by_object = {}

    def get_bestfit_spectrum(self):
        """
        Load best-fit spectrum from FITS file.

        Returns
        -------
        dict
            Dictionary containing wavelength, flux, and other data from FITS file
        """
        if self.bestfit_file is None:
            raise FileNotFoundError("No best-fit FITS file found")

        try:
            from astropy.io import fits
            with fits.open(self.bestfit_file) as hdul:
                data = {}
                for hdu in hdul:
                    if hdu.data is not None:
                        # Try to extract wavelength and flux
                        if hasattr(hdu.data, 'columns'):
                            for col in hdu.data.columns:
                                data[col.name.lower()] = hdu.data[col.name]
                        else:
                            data[hdu.name.lower()] = hdu.data

                return data
        except ImportError:
            raise ImportError("astropy is required for reading FITS files. Install with: pip install astropy")
        except Exception as e:
            raise RuntimeError(f"Error reading best-fit FITS file: {e}")

    def get_posterior_samples(self, object_base=None):
        """
        Load posterior distribution samples as an astropy Table.

        Parameters
        ----------
        object_base : str, optional
            Base name for the object (if None, uses object_id from initialization or first available)

        Returns
        -------
        astropy.table.Table
            Table containing posterior samples with:
            - Parameter columns: Sample values for each parameter
            - 'posterior_weights': Posterior weights (P_{posterior}) if available
            - 'loglike': Log-likelihood values if available

        Examples
        --------
        >>> results = BayeSEDResults('output')
        >>> samples = results.get_posterior_samples()
        >>> print(samples.colnames)  # All parameter names
        >>> ages = samples['log(age/yr)[0,1]']  # Access parameter samples
        >>> weights = samples['posterior_weights']  # Access weights
        """
        if not self.posterior_files:
            raise FileNotFoundError("No posterior sample files found")

        # If object_id was set during initialization, use it to filter files
        if object_base is None and self.object_id is not None:
            obj_str = str(self.object_id)
            if hasattr(self, 'posterior_files_by_object') and obj_str in self.posterior_files_by_object:
                # Get first available file for this object
                obj_files = self.posterior_files_by_object[obj_str]
                if obj_files:
                    object_base = list(obj_files.keys())[0]
                    files = obj_files[object_base]
                else:
                    raise FileNotFoundError(f"No posterior sample files found for object '{obj_str}'")
            else:
                # Fall back to searching all files
                if object_base is None:
                    object_base = list(self.posterior_files.keys())[0]
                files = self.posterior_files[object_base]
        else:
            # No object_id specified, use object_base or first available
            if object_base is None:
                object_base = list(self.posterior_files.keys())[0]

            if object_base not in self.posterior_files:
                raise ValueError(f"Object base '{object_base}' not found in posterior files")

            files = self.posterior_files[object_base]

        # Read parameter names
        with open(files['paramnames'], 'r') as f:
            paramnames = [line.strip().split()[0] for line in f if line.strip()]

        # Read samples
        try:
            import numpy as np
            from astropy.table import Table

            samples_all = np.loadtxt(files['samples'])

            # The samples file typically contains P_{posterior} and loglike as first two columns
            # which are not in the paramnames file. Extract these for weighted sampling.
            table_data = {}

            if samples_all.shape[1] > len(paramnames):
                # Extract P_{posterior} and loglike from first columns
                n_skip = samples_all.shape[1] - len(paramnames)
                if n_skip >= 1:
                    table_data['posterior_weights'] = samples_all[:, 0]
                if n_skip >= 2:
                    table_data['loglike'] = samples_all[:, 1]
                # Extract parameter columns (skip the first n_skip columns)
                samples = samples_all[:, n_skip:]
            else:
                samples = samples_all

            # Add parameter columns to table
            for i, param_name in enumerate(paramnames):
                table_data[param_name] = samples[:, i]

            # Create astropy table
            return Table(table_data)

        except Exception as e:
            raise RuntimeError(f"Error reading posterior samples: {e}")

    def get_evidence(self, object_id=None):
        """
        Get Bayesian evidence values and errors from parameter table.

        Returns an astropy table with evidence-related columns for all objects
        or a specific object. BayeSED provides two evidence estimates:
        - 'logZ': Standard nested sampling evidence
        - 'INSlogZ': Importance Nested Sampling (INS) evidence (more accurate)

        Corresponding error estimates:
        - 'logZerr': Error for standard evidence
        - 'INSlogZerr': Error for INS evidence

        Parameters
        ----------
        object_id : str or int, optional
            Object ID to get evidence for. If None, returns evidence for all objects.

        Returns
        -------
        astropy.table.Table
            Table containing ID and evidence-related columns:
            - 'ID': Object identifiers
            - 'logZ': Standard evidence (if available)
            - 'logZerr': Standard evidence error (if available)
            - 'INSlogZ': INS evidence (if available)
            - 'INSlogZerr': INS evidence error (if available)

        Examples
        --------
        >>> results = BayeSEDResults('output')
        >>>
        >>> # Get evidence for all objects
        >>> evidence_table = results.get_evidence()
        >>> print(evidence_table)
        >>>
        >>> # Get evidence for specific object
        >>> evidence_obj = results.get_evidence(object_id='galaxy_1')
        >>> print(f"INS Evidence: {evidence_obj['INSlogZ'][0]:.2f} ± {evidence_obj['INSlogZerr'][0]:.2f}")
        >>>
        >>> # Access specific columns
        >>> all_evidence = results.get_evidence()
        >>> best_evidence = all_evidence['INSlogZ'] if 'INSlogZ' in all_evidence.colnames else all_evidence['logZ']
        """
        try:
            params_table = self.parameters

            # Find evidence-related columns
            evidence_cols = ['ID']  # Always include ID
            for col in ['logZ', 'logZerr', 'INSlogZ', 'INSlogZerr']:
                if col in params_table.colnames:
                    evidence_cols.append(col)

            if len(evidence_cols) == 1:  # Only ID column found
                raise ValueError("No evidence columns found in parameter table")

            # Extract evidence table
            evidence_table = params_table[evidence_cols]

            # Filter by object_id if specified
            if object_id is not None:
                obj_str = str(object_id)
                matching_rows = evidence_table[evidence_table['ID'] == obj_str]
                if len(matching_rows) == 0:
                    # Try partial match
                    matching_rows = evidence_table[[obj_str in str(id) for id in evidence_table['ID']]]
                if len(matching_rows) == 0:
                    raise ValueError(f"Object ID '{object_id}' not found in results")
                return matching_rows
            else:
                # Return evidence for all objects
                return evidence_table

        except Exception as e:
            raise RuntimeError(f"Error reading evidence from parameter table: {e}")

    @property
    def parameters(self):
        """
        Get the main parameters table as an astropy Table.

        This property provides direct access to the HDF5 results table containing
        all parameters and statistics. Parameter names can be accessed via parameters.colnames.

        Returns
        -------
        astropy.table.Table
            Table containing all parameters and results

        Examples
        --------
        >>> results = BayeSEDResults('output')
        >>> params = results.parameters
        >>> print(params.colnames)  # All parameter names
        >>> ages = params['log(age/yr)[0,1]_{median}']  # Access specific parameter
        >>> free_param_names = [col for col in params.colnames if not any(x in col for x in ['SNR', 'logZ', 'scale'])]
        """
        return self.load_hdf5_results(filter_snr=False)

    def load_hdf5_results(self, hdf5_file=None, filter_snr=True, min_snr=0.0):
        """
        Load parameter estimation results from HDF5 file.

        This method loads the global results file containing best-fit parameters
        and statistics for all objects, following the standard BayeSED3 HDF5 format.

        Parameters
        ----------
        hdf5_file : str, optional
            Path to HDF5 file. If None, uses the first HDF5 file found in output_dir
        filter_snr : bool
            If True, filter results to only include objects with SNR > min_snr (default: True)
        min_snr : float
            Minimum SNR threshold for filtering (default: 0.0)

        Returns
        -------
        astropy.table.Table
            Table containing:
            - 'ID': Object identifiers
            - Parameter columns: Best-fit parameter values (one column per parameter)
            - 'SNR': Signal-to-noise ratio
            - Other statistics columns as available

        Examples
        --------
        >>> results = BayeSEDResults('observation/agn_host_decomp/output')
        >>> params_table = results.load_hdf5_results()
        >>> print(params_table.colnames)  # Show available columns
        >>>
        >>> # Filter by SNR
        >>> high_snr = params_table[params_table['SNR'] > 10]
        >>>
        >>> # Access specific parameter
        >>> ages = params_table['log(age/yr)[0,1]_{median}']
        """
        if hdf5_file is None:
            if self.hdf5_file is None:
                raise FileNotFoundError("No HDF5 file found. Specify hdf5_file parameter or ensure output directory contains *.hdf5 files")
            hdf5_file = self.hdf5_file

        try:
            import h5py
            from astropy.table import Table, hstack
        except ImportError as e:
            if 'h5py' in str(e):
                raise ImportError("h5py is required for reading HDF5 files. Install with: pip install h5py")
            else:
                raise ImportError("astropy is required for reading HDF5 results. Install with: pip install astropy")

        try:
            with h5py.File(hdf5_file, 'r') as h:
                # Get parameter names (decode from bytes to utf-8)
                if 'parameters_name' not in h:
                    raise ValueError(f"HDF5 file '{hdf5_file}' does not contain 'parameters_name' dataset")

                colnames = [x.decode('utf-8') for x in h['parameters_name'][:]]

                # Get object IDs
                if 'ID' not in h:
                    raise ValueError(f"HDF5 file '{hdf5_file}' does not contain 'ID' dataset")

                ids = h['ID'][:]
                # Convert bytes to strings if needed
                if len(ids) > 0 and isinstance(ids[0], bytes):
                    ids = [id.decode('utf-8') for id in ids]
                else:
                    ids = [str(id) for id in ids]

                # Get parameter values
                if 'parameters' not in h:
                    raise ValueError(f"HDF5 file '{hdf5_file}' does not contain 'parameters' dataset")

                parameters = h['parameters'][:]

                # Create ID table
                id_table = Table([ids], names=['ID'])

                # Create parameters table
                parameters_table = Table(parameters, names=colnames, copy=False)

                # Combine ID and parameters
                result_table = hstack([id_table, parameters_table])

                # Filter by SNR if requested
                if filter_snr and 'SNR' in result_table.colnames:
                    mask = result_table['SNR'] > min_snr
                    result_table = result_table[mask]

                return result_table

        except Exception as e:
            raise RuntimeError(f"Error reading HDF5 file '{hdf5_file}': {e}")

    def list_model_configurations(self, catalog_name=None):
        """
        List all model configurations found in the output directory.

        Parameters
        ----------
        catalog_name : str, optional
            Filter by catalog_name (prefix in HDF5 filename). If None, returns all configurations.

        Returns
        -------
        dict
            Dictionary mapping model configuration names to HDF5 file paths.
            Keys are full model names (from HDF5 filenames), values are file paths.
            If catalog_name is specified, only returns configurations for that catalog.
        """
        import os
        configs = {}
        hdf5_files_to_use = self.hdf5_files
        if catalog_name is not None and hasattr(self, 'hdf5_files_by_catalog'):
            if catalog_name in self.hdf5_files_by_catalog:
                hdf5_files_to_use = self.hdf5_files_by_catalog[catalog_name]

        for hdf5_file in hdf5_files_to_use:
            basename = os.path.basename(hdf5_file)
            # Model name is the full filename without .hdf5 extension
            # Format: {catalog_name}_{model_name}.hdf5
            if basename.endswith('.hdf5'):
                model_name = basename.replace('.hdf5', '')
                configs[model_name] = hdf5_file
        return configs

    def list_catalog_names(self):
        """
        List all catalog names found in the output directory.

        Catalog names are extracted from input file headers and used as subdirectory
        names when save_ commands are enabled. Each catalog_name subdirectory contains
        object-specific result files.

        Returns
        -------
        list
            List of catalog names (strings) found in the output directory.
            Empty list if no catalog_name subdirectories are found.
        """
        return self.catalog_names.copy() if hasattr(self, 'catalog_names') else []

    def list_objects(self, catalog_name=None, hdf5_prefix=None):
        """
        List all objects with results in the output directory.

        Object IDs are read from the HDF5 file(s) in the output directory.
        Since all HDF5 files in the same output_dir typically contain all objects
        from the same sample, this method reads from the first available HDF5 file
        unless a specific catalog_name is requested.

        When catalog_name is provided, this method uses the organized file structure
        for faster lookups. The catalog_name is the prefix used in HDF5 filenames.

        Parameters
        ----------
        catalog_name : str, optional
            Filter objects by catalog_name (prefix in HDF5 filename).
            If provided, uses the organized file structure for faster lookups.
            If None, searches all catalogs.
        hdf5_prefix : str, optional
            DEPRECATED: Use catalog_name instead. Filter objects by HDF5 file prefix.
            Kept for backward compatibility.

        Returns
        -------
        list
            List of object identifiers (strings)
        """
        import os

        # If catalog_name is specified, use it (it's the prefix in HDF5 filenames)
        if catalog_name is None and hdf5_prefix is not None:
            # Backward compatibility: treat hdf5_prefix as catalog_name
            catalog_name = hdf5_prefix

        # If catalog_name is specified and we have organized files, use them
        if catalog_name is not None and hasattr(self, 'bestfit_files_by_catalog'):
            if catalog_name in self.bestfit_files_by_catalog:
                objects = list(self.bestfit_files_by_catalog[catalog_name].keys())
                return sorted(objects)

        # Try to read object IDs from HDF5 files first
        # Since all HDF5 files in the same output_dir typically contain all objects
        # from the same sample, we can use any HDF5 file unless catalog_name is specified
        if self.hdf5_files:
            try:
                import h5py
                all_ids = []

                # Select HDF5 files to read from
                hdf5_files_to_read = []
                if catalog_name is not None:
                    # Filter by catalog_name (which is the prefix in HDF5 filename)
                    if catalog_name in self.hdf5_files_by_catalog:
                        hdf5_files_to_read = self.hdf5_files_by_catalog[catalog_name]
                    else:
                        # Try direct matching
                        for hdf5_file in self.hdf5_files:
                            hdf5_basename = os.path.basename(hdf5_file)
                            if hdf5_basename.startswith(catalog_name + '_'):
                                hdf5_files_to_read.append(hdf5_file)
                else:
                    # No catalog_name specified - use all files (but typically all contain same objects)
                    hdf5_files_to_read = self.hdf5_files

                # Read object IDs from selected HDF5 files
                for hdf5_file in hdf5_files_to_read:
                    with h5py.File(hdf5_file, 'r') as f:
                        if 'ID' in f:
                            ids = f['ID'][:]
                            # Convert bytes to strings if needed
                            if len(ids) > 0 and isinstance(ids[0], bytes):
                                ids = [id.decode('utf-8') for id in ids]
                            else:
                                ids = [str(id) for id in ids]
                            all_ids.extend(ids)

                # Remove duplicates while preserving order
                seen = set()
                unique_ids = []
                for obj_id in all_ids:
                    if obj_id not in seen:
                        seen.add(obj_id)
                        unique_ids.append(obj_id)

                if unique_ids:
                    return unique_ids
            except ImportError:
                pass  # h5py not available, fall back to file-based method
            except Exception as e:
                # If HDF5 reading fails, fall back to file-based method
                pass

        # Fallback: Extract object IDs from organized bestfit files
        if hasattr(self, 'bestfit_files_by_object') and self.bestfit_files_by_object:
            objects = list(self.bestfit_files_by_object.keys())
            # Filter by prefix if specified
            if hdf5_prefix is not None:
                # Check if object's files are in the prefix directory
                filtered_objects = []
                for obj_id in objects:
                    files = self.bestfit_files_by_object[obj_id]
                    # Check first file's path
                    if files:
                        rel_path = os.path.relpath(files[0], self.output_dir)
                        if os.sep in rel_path:
                            path_parts = rel_path.split(os.sep)
                            if len(path_parts) > 0 and path_parts[0] == hdf5_prefix:
                                filtered_objects.append(obj_id)
                return filtered_objects
            return objects

        # Final fallback: Extract from file paths directly
        objects = []
        for f in self.bestfit_files:
            rel_path = os.path.relpath(f, self.output_dir)

            # Filter by prefix if specified
            if hdf5_prefix is not None:
                if os.sep in rel_path:
                    path_parts = rel_path.split(os.sep)
                    if len(path_parts) > 0 and path_parts[0] != hdf5_prefix:
                        continue
                else:
                    if not os.path.basename(f).startswith(hdf5_prefix + '_'):
                        continue

            # Filter by catalog_name if specified
            if catalog_name is not None:
                if os.sep in rel_path:
                    path_parts = rel_path.split(os.sep)
                    if len(path_parts) > 0 and path_parts[0] != catalog_name:
                        continue

            if os.sep in rel_path:
                parts = rel_path.split(os.sep)
                # Path structure: {catalog_name}/{object_id}/{model_name}_bestfit.fits
                # or legacy: {prefix}/{object_id}/{model_name}_bestfit.fits
                if len(parts) >= 2:
                    # Second part is object ID (first is catalog_name or prefix)
                    obj_id = parts[1]
                    objects.append(obj_id)
            else:
                base = os.path.basename(f).replace('_bestfit.fits', '')
                objects.append(base)

        # Remove duplicates while preserving order
        seen = set()
        unique_objects = []
        for obj in objects:
            if obj not in seen:
                seen.add(obj)
                unique_objects.append(obj)
        return unique_objects

    def plot_posterior(self, params=None, object_base=None,
                       method='getdist', filled=True, show=True,
                       output_file=None, figsize=None, show_median=True,
                       show_confidence_intervals=True, confidence_level=0.68, **kwargs):
        """
        Plot posterior probability distribution functions (PDFs).

        Creates corner plots (triangle plots) showing 1D and 2D marginal
        posterior distributions for selected parameters. Uses GetDist library
        which correctly handles weighted nested sampling samples from MultiNest.

        Parameters
        ----------
        params : list of str, str, optional
            Parameter specification. Options:
            - List of parameter names: Plot specific parameters
            - Single parameter name (str): Create 1D PDF plot
            - None: Plot all available parameters (may be slow for many parameters)

            To see available parameters, use:
            >>> print(results.get_posterior_samples().colnames)  # All available parameter names
        object_base : str, optional
            Base name for the object (if None, uses first available)
        method : str
            Plotting method: 'getdist' (default and only supported method)
            - 'getdist': Uses GetDist library (handles weights when reading MultiNest output files)
        filled : bool
            If True, use filled contours for 2D PDFs (default: True)
        show : bool
            Whether to display the plot (default: True)
        output_file : str, optional
            Output file path for saving the plot
        figsize : tuple, optional
            Figure size (width, height) in inches. If None, auto-sized
        show_median : bool
            If True, add vertical lines indicating median values on 1D PDF plots (default: True)
        show_confidence_intervals : bool
            If True, add shaded regions indicating confidence intervals on 1D PDF plots (default: True)
        confidence_level : float
            Confidence level for intervals (default: 0.68 for 1-sigma).
            For 0.68, shows 16th-84th percentiles. For 0.95, shows 2.5th-97.5th percentiles.
        **kwargs
            Additional keyword arguments passed to plotting functions.
            For 1D plots, you can control font sizes with:
            - axes_fontsize: Font size for tick labels (default: 10 for 1D plots)
            - axes_labelsize: Font size for axis labels (default: 12 for 1D plots)
            - subplot_size_inch: Size of subplot in inches (default: 3.0 for 1D plots)

        Returns
        -------
        matplotlib.figure.Figure or getdist.plots.GetDistPlotter
            The figure object or GetDist plotter

        Examples
        --------
        >>> results = BayeSEDResults('observation/agn_host_decomp/output')
        >>>
        >>> # Plot single parameter (1D PDF)
        >>> results.plot_posterior(params='log(age/yr)[0,1]')
        >>>
        >>> # Plot specific parameters (corner plot)
        >>> results.plot_posterior(
        ...     params=['log(age/yr)[0,1]', 'log(Z/Zsun)[0,1]', 'Av_2[0,1]']
        ... )
        >>>
        >>> # Plot all available parameters (may be slow)
        >>> results.plot_posterior()  # params=None uses all available
        >>>
        >>> # Check what's available first
        >>> print(results.get_posterior_samples().colnames)  # All available parameter names
        """
        # Get available parameters from GetDist samples (includes renamed parameters)
        try:
            samples_gd = self.get_getdist_samples(object_base=object_base)
            available_params = [param.name for param in samples_gd.paramNames.names]
        except Exception as e:
            raise RuntimeError(f"Could not load parameter names: {e}")

        # Handle parameter specifications
        if params is None:
            params = available_params
        elif isinstance(params, str):
            # Single parameter name - convert to list
            params = [params]

        # Validate that all requested parameters are available
        if isinstance(params, list):
            invalid_params = [p for p in params if p not in available_params]
            if invalid_params:
                import warnings
                warnings.warn(
                    f"Some parameters are not available for plotting: {invalid_params}. "
                    f"Available parameters: {available_params}",
                    UserWarning
                )
                # Filter to only valid parameters
                params = [p for p in params if p in available_params]

                if not params:
                    raise ValueError(
                        f"No valid parameters for plotting. Available parameters: {available_params}"
                    )

        if method == 'getdist':
            return self._plot_posterior_getdist(
                params=params, object_base=object_base, filled=filled,
                show=show, output_file=output_file,
                show_median=show_median, show_confidence_intervals=show_confidence_intervals,
                confidence_level=confidence_level, **kwargs
            )
        else:
            raise ValueError(f"Unknown plotting method: {method}. Use 'getdist'")











    def _plot_posterior_getdist(self, params=None, object_base=None, filled=True,
                                show=True, output_file=None, show_median=True,
                                show_confidence_intervals=True, confidence_level=0.68, **kwargs):
        """Plot posterior PDFs using GetDist library.

        Simplified and more robust implementation that leverages GetDist's built-in
        capabilities for parameter validation, statistics computation, and plotting.
        """
        import os
        import warnings
        import numpy as np

        # Import GetDist and matplotlib
        try:
            import matplotlib
            import matplotlib.pyplot as plt
            from getdist import plots, loadMCSamples
        except ImportError as e:
            raise ImportError(
                f"Required libraries not available: {e}. "
                "Install with: pip install getdist matplotlib"
            )

        # Helper: Switch to interactive backend if needed
        def _ensure_interactive_backend():
            if not show:
                return
            current_backend = matplotlib.get_backend().lower()
            if current_backend in ['agg', 'pdf', 'svg', 'ps']:
                for backend in ['TkAgg', 'Qt5Agg', 'QtAgg', 'MacOSX']:
                    try:
                        matplotlib.use(backend, force=True)
                        return
                    except (ImportError, ValueError):
                        continue
                warnings.warn(
                    "Could not switch to interactive backend. Plot may not display.",
                    UserWarning
                )

        # Helper: Get parameter names from file
        def _get_paramnames_from_file(paramnames_file):
            with open(paramnames_file, 'r') as f:
                return [line.strip().split()[0] for line in f if line.strip()]

        # Helper: Compute statistics from GetDist samples
        def _compute_statistics(samples_gd, param_names):
            """Compute medians and confidence intervals using GetDist."""
            markers = {}
            confidence_intervals = {}

            if not (show_median or show_confidence_intervals):
                return markers, confidence_intervals

            lower_percentile = (1 - confidence_level) / 2
            upper_percentile = 1 - lower_percentile

            # Get parameter indices
            gd_param_names = samples_gd.getParamNames().list()

            for param in param_names:
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

            return markers, confidence_intervals

        # Helper: Add confidence intervals to 1D plot
        def _add_ci_to_1d_plot(ax, lower, upper, confidence_level):
            ax.axvspan(lower, upper, color='blue', alpha=0.15,
                       label=f'{int(confidence_level*100)}% CI')
            ax.legend()

        # Helper: Add confidence intervals to diagonal plots
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

        # Main function logic
        _ensure_interactive_backend()

        # Get object_base and files
        if object_base is None:
            if not self.posterior_files:
                raise FileNotFoundError("No posterior sample files found")
            object_base = list(self.posterior_files.keys())[0]

        files = self.posterior_files[object_base]
        chain_dir = os.path.dirname(files['paramnames'])
        base_name = os.path.basename(files['paramnames']).replace('.paramnames', '')
        root_path = os.path.join(chain_dir, base_name)

        # Load samples using GetDist (handles validation and variation checking)
        # Use get_getdist_samples() to ensure parameter renaming is applied
        try:
            samples_gd = self.get_getdist_samples(object_base=object_base)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load GetDist samples: {e}. "
                "Check that the posterior sample files are valid."
            ) from e

        # Get available parameter names from GetDist (only parameters in samples)
        gd_param_names = samples_gd.getParamNames().list()

        # Get parameter names to plot
        if params is None:
            # When params=None, use all parameters available in the samples
            params = gd_param_names
        else:
            # When params is provided, validate and filter
            params_valid = [p for p in params if p in gd_param_names]
            params_not_found = [p for p in params if p not in gd_param_names]

            # Only warn if user explicitly provided params that don't exist
            if params_not_found:
                # Show first few missing params to avoid overwhelming output
                if len(params_not_found) > 5:
                    missing_str = f"{params_not_found[:5]} ... (and {len(params_not_found) - 5} more)"
                else:
                    missing_str = str(params_not_found)
                warnings.warn(
                    f"Parameters not found in samples (skipped): {missing_str}. "
                    f"Use results.get_posterior_samples().colnames to see available parameters.",
                    UserWarning
                )

            if not params_valid:
                # Provide helpful error message with suggestion
                available_str = ', '.join(gd_param_names[:10])
                if len(gd_param_names) > 10:
                    available_str += f", ... (and {len(gd_param_names) - 10} more)"
                raise ValueError(
                    f"No valid parameters found. Requested: {params}. "
                    f"Available parameters: {available_str}. "
                    f"Use results.get_posterior_samples().colnames to see all available parameters."
                )

            params = params_valid  # Use only valid parameters

        if not params:
            raise ValueError(
                "No parameters available for plotting. "
                "The posterior samples appear to be empty or invalid."
            )

        # Extract font settings from kwargs
        axes_fontsize = kwargs.pop('axes_fontsize', None)
        axes_labelsize = kwargs.pop('axes_labelsize', None)
        subplot_size_inch = kwargs.pop('subplot_size_inch', None)

        # Get the (possibly renamed) samples and use them directly
        samples_gd = self.get_getdist_samples(object_base=object_base)

        # Create GetDist plotter
        g = plots.get_subplot_plotter()
        roots = [samples_gd]  # Use MCSamples object directly instead of file paths

        # Compute statistics if needed
        markers = {}
        confidence_intervals = {}
        if show_median or show_confidence_intervals:
            try:
                markers, confidence_intervals = _compute_statistics(samples_gd, params)
            except Exception as e:
                warnings.warn(
                    f"Could not compute statistics: {e}. Statistics will not be displayed.",
                    UserWarning
                )
                show_median = False
                show_confidence_intervals = False

        # Configure plot settings
        is_1d = len(params) == 1
        if is_1d:
            g.settings.axes_fontsize = axes_fontsize or 10
            g.settings.axes_labelsize = axes_labelsize or 12
            g.settings.subplot_size_inch = subplot_size_inch or 3.0
        else:
            if axes_fontsize is not None:
                g.settings.axes_fontsize = axes_fontsize
            if axes_labelsize is not None:
                g.settings.axes_labelsize = axes_labelsize
            if subplot_size_inch is not None:
                g.settings.subplot_size_inch = subplot_size_inch

        # Create plot
        try:
            if is_1d:
                # 1D plot
                try:
                    g.plot_1d(roots, params[0], **kwargs)
                except AttributeError:
                    g.triangle_plot(roots, params, filled=filled,
                                   markers=markers if markers else None, **kwargs)

                # Add confidence interval
                if show_confidence_intervals and params[0] in confidence_intervals:
                    ax = plt.gca()
                    lower, upper = confidence_intervals[params[0]]
                    _add_ci_to_1d_plot(ax, lower, upper, confidence_level)
            else:
                # Triangle plot
                g.triangle_plot(roots, params, filled=filled,
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
                        warnings.warn(
                            f"Could not add confidence intervals: {e}",
                            UserWarning
                        )
        except (ValueError, np.linalg.LinAlgError) as e:
            error_str = str(e).lower()
            if "singular" in error_str or "matrix" in error_str:
                raise ValueError(
                    f"GetDist plotting failed: singular matrix error. "
                    f"This indicates insufficient parameter variation for KDE. "
                    f"Try selecting different parameters with more variation."
                ) from e
            raise

        # Export if requested
        if output_file:
            g.export(output_file)

        # Show plot if requested
        if show:
            try:
                import matplotlib.pyplot as plt
                # Use the most reliable display method
                plt.show()
            except Exception as e:
                import warnings
                warnings.warn(
                    f"Could not display plot: {e}. "
                    f"Try using output_file parameter to save the plot instead.",
                    UserWarning
                )

        return g

    def get_free_parameters(self):
        """
        Get list of free parameter names.

        Free parameters are the fitted parameters (those without '*' marker).

        Returns
        -------
        list of str
            List of free parameter names

        Examples
        --------
        >>> results = BayeSEDResults('output')
        >>> free_params = results.get_free_parameters()
        >>> print(f"Free parameters: {free_params}")
        """
        param_names = self._get_parameter_names_from_files()
        return [p for p in param_names if not p.endswith('*')]

    def get_derived_parameters(self):
        """
        Get list of derived parameter names.

        Derived parameters are computed from free parameters (identified by '*' marker
        in the paramnames file, but the '*' is stripped from the returned names).

        Returns
        -------
        list of str
            List of derived parameter names (without '*' markers)

        Examples
        --------
        >>> results = BayeSEDResults('output')
        >>> derived_params = results.get_derived_parameters()
        >>> print(f"Derived parameters: {derived_params}")
        """
        param_names = self._get_parameter_names_from_files()
        return [p.rstrip('*') for p in param_names if p.endswith('*')]

    def get_parameter_names(self):
        """
        Get list of all parameter names (free + derived).

        This is more efficient than using get_posterior_samples().colnames
        since it reads only the small paramnames file instead of loading
        the entire samples data.

        Returns
        -------
        list of str
            List of all parameter names (without '*' markers)

        Examples
        --------
        >>> results = BayeSEDResults('output')
        >>> all_params = results.get_parameter_names()
        >>> print(f"All parameters: {all_params}")
        >>>
        >>> # More efficient than:
        >>> # samples = results.get_posterior_samples()
        >>> # all_params = [col for col in samples.colnames
        >>> #               if col not in ['posterior_weights', 'loglike']]
        """
        param_names = self._get_parameter_names_from_files()
        return [p.rstrip('*') for p in param_names]

    def _get_parameter_names_from_files(self):
        """
        Get parameter names directly from paramnames file without loading samples.

        This is much more efficient than loading the entire samples file when we
        only need the parameter names.

        Returns
        -------
        list of str
            List of all parameter names (excluding posterior_weights and loglike)
        """
        # If we have renamed parameter names, use those instead
        if hasattr(self, '_renamed_parameter_names'):
            return self._renamed_parameter_names

        if not self.posterior_files:
            raise FileNotFoundError("No posterior sample files found")

        # Get first available paramnames file
        object_base = list(self.posterior_files.keys())[0]
        files = self.posterior_files[object_base]

        # Read parameter names from paramnames file
        with open(files['paramnames'], 'r') as f:
            param_names = [line.strip().split()[0] for line in f if line.strip()]

        return param_names

    def get_getdist_samples(self, object_base=None):
        """
        Get GetDist MCSamples object for direct GetDist usage.

        This method provides access to the GetDist MCSamples object, which can be
        used for advanced plotting and analysis with the GetDist library.

        Parameters
        ----------
        object_base : str, optional
            Base name for the object (if None, uses first available)

        Returns
        -------
        getdist.MCSamples
            GetDist samples object that can be used with GetDist plotting functions

        Examples
        --------
        >>> results = BayeSEDResults('output')
        >>> samples_gd = results.get_getdist_samples()
        >>>
        >>> # Use with GetDist directly
        >>> from getdist import plots
        >>> g = plots.get_subplot_plotter()
        >>> g.triangle_plot([samples_gd], ['log(age/yr)[0,1]', 'log(Z/Zsun)[0,1]'])
        >>>
        >>> # Get parameter statistics
        >>> print(samples_gd.getMargeStats())
        """
        import os

        try:
            from getdist import loadMCSamples, MCSamples
        except ImportError:
            raise ImportError("GetDist is required. Install with: pip install getdist")

        if not self.posterior_files:
            raise FileNotFoundError("No posterior sample files found")

        # Get object_base and files
        if object_base is None:
            object_base = list(self.posterior_files.keys())[0]

        if object_base not in self.posterior_files:
            raise ValueError(f"Object base '{object_base}' not found in posterior files")

        files = self.posterior_files[object_base]
        chain_dir = os.path.dirname(files['paramnames'])
        base_name = os.path.basename(files['paramnames']).replace('.paramnames', '')
        root_path = os.path.join(chain_dir, base_name)

        # Load samples using GetDist first
        try:
            samples_gd = loadMCSamples(root_path)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load GetDist samples from {root_path}: {e}. "
                "Check that the posterior sample files are valid."
            ) from e

        # Check if we need to apply parameter renaming
        if hasattr(self, '_renamed_parameter_names'):
            # Read original parameter names from file
            original_names = []
            original_labels = []
            with open(files['paramnames'], 'r') as f:
                for line in f:
                    if line.strip():
                        parts = line.strip().split(None, 1)  # Split on first whitespace only
                        original_names.append(parts[0])
                        # Use label if provided, otherwise use parameter name
                        original_labels.append(parts[1] if len(parts) > 1 else parts[0])

            # Create mapping from original to renamed
            rename_mapping = {}
            renamed_labels = []
            for i, (orig_name, new_name) in enumerate(zip(original_names, self._renamed_parameter_names)):
                if orig_name != new_name:
                    rename_mapping[orig_name] = new_name
                # Update labels to match renamed parameters
                if i < len(original_labels):
                    label = original_labels[i]
                    # If the label was the same as the parameter name, update it
                    if label == orig_name:
                        renamed_labels.append(new_name)
                    else:
                        renamed_labels.append(label)
                else:
                    renamed_labels.append(new_name)

            if rename_mapping:
                # Create new MCSamples object with renamed parameters
                # Get the sample data, weights, and likelihoods
                samples_data = samples_gd.samples
                weights = getattr(samples_gd, 'weights', None)
                loglikes = getattr(samples_gd, 'loglikes', None)

                # Get the actual parameter names from the loaded samples (after GetDist processing)
                # This accounts for any parameters that GetDist may have removed (fixed parameters, etc.)
                actual_param_names = [param.name for param in samples_gd.paramNames.names]
                actual_param_labels = [param.label for param in samples_gd.paramNames.names]

                # Apply renaming to the parameters that actually exist in the samples
                renamed_actual_names = []
                renamed_actual_labels = []

                for name, label in zip(actual_param_names, actual_param_labels):
                    # Apply renaming if this parameter should be renamed
                    if name in rename_mapping:
                        new_name = rename_mapping[name]
                        renamed_actual_names.append(new_name)
                        # Update label if it was the same as the parameter name
                        if label == name:
                            renamed_actual_labels.append(new_name)
                        else:
                            renamed_actual_labels.append(label)
                    else:
                        renamed_actual_names.append(name)
                        renamed_actual_labels.append(label)

                # Apply custom labels if they exist
                if hasattr(self, '_custom_labels') and self._custom_labels:
                    for i, name in enumerate(renamed_actual_names):
                        if name in self._custom_labels:
                            renamed_actual_labels[i] = self._custom_labels[name]

                # Create new MCSamples with the correctly sized parameter lists
                new_samples = MCSamples(
                    samples=samples_data,
                    names=renamed_actual_names,
                    labels=renamed_actual_labels,
                    weights=weights,
                    loglikes=loglikes,
                    name_tag=getattr(samples_gd, 'name_tag', None),
                    label=getattr(samples_gd, 'label', None)
                )

                # Copy other important attributes if they exist
                if hasattr(samples_gd, 'ranges'):
                    new_samples.ranges = samples_gd.ranges
                if hasattr(samples_gd, 'sampler'):
                    new_samples.sampler = samples_gd.sampler

                return new_samples

        # Apply custom labels even if no parameter renaming is needed
        if hasattr(self, '_custom_labels') and self._custom_labels:
            # Get the parameter names and labels
            param_names = [param.name for param in samples_gd.paramNames.names]
            param_labels = [param.label for param in samples_gd.paramNames.names]

            # Apply custom labels
            updated_labels = []
            for i, name in enumerate(param_names):
                if name in self._custom_labels:
                    updated_labels.append(self._custom_labels[name])
                else:
                    updated_labels.append(param_labels[i])

            # Create new MCSamples with custom labels
            samples_data = samples_gd.samples
            weights = getattr(samples_gd, 'weights', None)
            loglikes = getattr(samples_gd, 'loglikes', None)

            new_samples = MCSamples(
                samples=samples_data,
                names=param_names,
                labels=updated_labels,
                weights=weights,
                loglikes=loglikes,
                name_tag=getattr(samples_gd, 'name_tag', None),
                label=getattr(samples_gd, 'label', None)
            )

            # Copy other important attributes if they exist
            if hasattr(samples_gd, 'ranges'):
                new_samples.ranges = samples_gd.ranges
            if hasattr(samples_gd, 'sampler'):
                new_samples.sampler = samples_gd.sampler

            return new_samples

        return samples_gd

    def rename_parameters(self, parameter_mapping):
        """
        Rename parameters in this BayeSEDResults object.

        This method permanently renames parameters in the loaded samples,
        making it easier to compare results with different parameter naming schemes.

        Parameters
        ----------
        parameter_mapping : dict
            Dictionary mapping old parameter names to new parameter names.
            Format: {old_name: new_name}

        Examples
        --------
        >>> results = BayeSEDResults('output')
        >>> results.rename_parameters({
        ...     'log(age/yr)[0,1]': 'log(age/yr)[0,0]',
        ...     'log(Z/Zsun)[0,1]': 'log(Z/Zsun)[0,0]'
        ... })
        >>> # Now parameters have consistent names for comparison
        """
        if not hasattr(self, '_samples_cache'):
            self._samples_cache = {}

        # Apply renaming to all cached samples
        for object_base, samples_gd in self._samples_cache.items():
            for old_name, new_name in parameter_mapping.items():
                # Find and rename the parameter in ParamInfo objects
                for param_info in samples_gd.paramNames.names:
                    if param_info.name == old_name:
                        param_info.name = new_name
                        break

                # Also update the paramNames.list if it exists
                if hasattr(samples_gd.paramNames, 'list'):
                    for param_info in samples_gd.paramNames.list:
                        if param_info.name == old_name:
                            param_info.name = new_name
                            break

        # Create a renamed parameter names cache to override file-based reading
        if not hasattr(self, '_renamed_parameter_names'):
            # Start with original parameter names
            self._renamed_parameter_names = self._get_parameter_names_from_files().copy()

        # Apply the renaming to the cached parameter names
        for i, param_name in enumerate(self._renamed_parameter_names):
            if param_name in parameter_mapping:
                self._renamed_parameter_names[i] = parameter_mapping[param_name]

        # Clear any cached parameter lists so they get regenerated with new names
        if hasattr(self, '_free_parameters_cache'):
            delattr(self, '_free_parameters_cache')
        if hasattr(self, '_derived_parameters_cache'):
            delattr(self, '_derived_parameters_cache')
        if hasattr(self, '_parameter_names_cache'):
            delattr(self, '_parameter_names_cache')



    def set_parameter_labels(self, custom_labels):
        """
        Set custom LaTeX labels for parameters in GetDist plots.

        This method allows you to customize how parameter names appear in plots
        by providing LaTeX-formatted labels.

        Parameters
        ----------
        custom_labels : dict
            Dictionary mapping parameter names to LaTeX labels.
            Example: {'log(age/yr)': r'\\log(t/\\mathrm{yr})'}

        Examples
        --------
        >>> results = BayeSEDResults('output')
        >>> custom_labels = {
        ...     'log(age/yr)': r'\\log(t/\\mathrm{yr})',
        ...     'log(Z/Zsun)': r'\\log(Z/Z_\\odot)',
        ...     'Av_2': r'A_V'$'
        ... }
        >>> results.set_parameter_labels(custom_labels)
        >>> results.plot_posterior(params=['log(age/yr)', 'log(Z/Zsun)'])
        """
        if not hasattr(self, '_custom_labels'):
            self._custom_labels = {}

        self._custom_labels.update(custom_labels)

        # Clear any cached samples so they get regenerated with new labels
        if hasattr(self, '_samples_cache'):
            delattr(self, '_samples_cache')

    def plot_free_parameters(self, object_base=None, method='getdist', filled=True,
                           show=True, output_file=None, figsize=None, **kwargs):
        """
        Plot all free parameters in a corner plot.

        This is a convenience method that plots all free parameters (fitted model parameters)
        in a triangle/corner plot. Custom labels set with set_parameter_labels() are automatically used.

        Parameters
        ----------
        object_base : str, optional
            Base name for the object (if None, uses first available)
        method : str
            Plotting method: 'getdist' (default and only supported method)
        filled : bool
            If True, use filled contours for 2D PDFs (default: True)
        show : bool
            Whether to display the plot (default: True)
        output_file : str, optional
            Output file path for saving the plot
        figsize : tuple, optional
            Figure size (width, height) in inches. If None, auto-sized
        **kwargs
            Additional keyword arguments passed to plotting functions

        Returns
        -------
        matplotlib.figure.Figure or getdist.plots.GetDistPlotter
            The figure object or GetDist plotter

        Examples
        --------
        >>> results = BayeSEDResults('output')
        >>> # Plot all free parameters
        >>> results.plot_free_parameters()
        >>>
        >>> # With custom labels
        >>> results.set_parameter_labels({'log(age/yr)': r'\\log(t/\\mathrm{yr})'})
        >>> results.plot_free_parameters()
        """
        free_params = self.get_free_parameters()
        return self.plot_posterior(
            params=free_params,
            object_base=object_base,
            method=method,
            filled=filled,
            show=show,
            output_file=output_file,
            figsize=figsize,
            **kwargs
        )

    def plot_derived_parameters(self, max_params=10, object_base=None, method='getdist',
                              filled=True, show=True, output_file=None, figsize=None, **kwargs):
        """
        Plot derived parameters in a corner plot.

        This is a convenience method that plots derived parameters (computed from fitted parameters)
        in a triangle/corner plot. Custom labels set with set_parameter_labels() are automatically used.

        Parameters
        ----------
        max_params : int, optional
            Maximum number of derived parameters to plot (default: 10)
        object_base : str, optional
            Base name for the object (if None, uses first available)
        method : str
            Plotting method: 'getdist' (default and only supported method)
        filled : bool
            If True, use filled contours for 2D PDFs (default: True)
        show : bool
            Whether to display the plot (default: True)
        output_file : str, optional
            Output file path for saving the plot
        figsize : tuple, optional
            Figure size (width, height) in inches. If None, auto-sized
        **kwargs
            Additional keyword arguments passed to plotting functions

        Returns
        -------
        matplotlib.figure.Figure or getdist.plots.GetDistPlotter
            The figure object or GetDist plotter

        Examples
        --------
        >>> results = BayeSEDResults('output')
        >>> # Plot first 10 derived parameters
        >>> results.plot_derived_parameters()
        >>>
        >>> # Plot more derived parameters
        >>> results.plot_derived_parameters(max_params=20)
        >>>
        >>> # With custom labels
        >>> results.set_parameter_labels({'log(Mstar)[0,1]': r'\\log(M_\\star/M_\\odot)'})
        >>> results.plot_derived_parameters()
        """
        derived_params = self.get_derived_parameters()

        # Limit the number of parameters to avoid overcrowded plots
        if len(derived_params) > max_params:
            derived_params = derived_params[:max_params]
            import warnings
            warnings.warn(
                f"Plotting only first {max_params} derived parameters out of {len(self.get_derived_parameters())}. "
                f"Use max_params to change this limit.",
                UserWarning
            )

        if not derived_params:
            raise ValueError("No derived parameters found to plot.")

        return self.plot_posterior(
            params=derived_params,
            object_base=object_base,
            method=method,
            filled=filled,
            show=show,
            output_file=output_file,
            figsize=figsize,
            **kwargs
        )

    def plot_bestfit(self, fits_file=None, output_file=None, show=True,
                     filter_file=None, filter_selection_file=None,
                     use_rest_frame=True, flux_unit='fnu', use_log_scale=None,
                     model_names=None, show_emission_lines=True,
                     figsize=(12, 8), dpi=300, focus_on_data_range=True, **kwargs):
        """
        Plot best-fit SED from FITS file.

        This is a general-purpose plotting function that handles various data types
        (photometry, spectroscopy) and supports customization options.

        Parameters
        ----------
        fits_file : str, optional
            Path to FITS file. If None, uses self.bestfit_file
        output_file : str, optional
            Output file path for saving the plot. If None, saves as {fits_file}.png
        show : bool
            Whether to display the plot (default: True)
        filter_file : str, optional
            Path to filter response file for overlay
        filter_selection_file : str, optional
            Path to filter selection file (filters_selected format)
        use_rest_frame : bool
            Use rest-frame wavelengths (default: True). If False, uses observed-frame
        flux_unit : str
            Flux unit: 'fnu' (μJy), 'nufnu' (νFν in μJy*Hz), or 'flambda' (default: 'fnu')
        use_log_scale : bool, optional
            Use logarithmic scale for axes. If None (default), auto-detects based on data range.
        model_names : list of str, optional
            Custom names for model components. If None, auto-generates from HDU names
        show_emission_lines : bool
            Show emission line markers for spectroscopy (default: True)
        figsize : tuple
            Figure size (width, height) in inches (default: (12, 8))
        dpi : int
            Resolution for saved figure (default: 300)
        focus_on_data_range : bool
            If True, set x-axis limits to focus on the wavelength range where data exists
        **kwargs
            Additional keyword arguments passed to matplotlib plotting functions

        Returns
        -------
        matplotlib.figure.Figure
            The matplotlib figure object
        """
        # Determine which fits_file to use
        fits_file_to_use = fits_file or self.bestfit_file
        if fits_file_to_use is None:
            raise FileNotFoundError(
                "No best-fit FITS file found. "
                "Ensure save_bestfit > 0 was used when running BayeSED, "
                "or provide a fits_file parameter."
            )

        from .plotting import plot_bestfit as plot_bestfit_func
        return plot_bestfit_func(
            fits_file=fits_file_to_use,
            output_file=output_file,
            show=show,
            filter_file=filter_file,
            filter_selection_file=filter_selection_file,
            use_rest_frame=use_rest_frame,
            flux_unit=flux_unit,
            use_log_scale=use_log_scale,
            model_names=model_names,
            show_emission_lines=show_emission_lines,
            figsize=figsize,
            dpi=dpi,
            focus_on_data_range=focus_on_data_range,
            **kwargs
        )

    def summary(self):
        """
        Print a summary of available parameters.

        Shows counts and examples of free and derived parameters.
        """
        try:
            free_params = self.get_free_parameters()
            derived_params = self.get_derived_parameters()

            print("=" * 60)
            print("BayeSED Results Summary")
            print("=" * 60)

            print(f"Free parameters ({len(free_params)}):")
            for param in free_params:
                print(f"  - {param}")

            print(f"\nDerived parameters ({len(derived_params)}):")
            # Show first 10 derived parameters as examples
            for param in derived_params[:10]:
                print(f"  - {param}")
            if len(derived_params) > 10:
                print(f"  - ... and {len(derived_params) - 10} more")

            print("\n" + "=" * 60)
            print("Quick plotting commands:")
            print("  results.plot_free_parameters()      # Plot all free parameters")
            print("  results.plot_derived_parameters()   # Plot all derived parameters")
            print("  results.plot_posterior(params=['param1', 'param2'])  # Plot specific parameters")
            print("  results.plot_bestfit()              # Plot best-fit SED")
            print("\nComparison plotting:")
            print("  plot_posterior_comparison([results1, results2])  # Compare multiple results")
            print("\nTo see all parameter names:")
            print("  results.get_parameter_names()       # List all parameters (efficient)")
            print("  results.get_free_parameters()       # List free parameters")
            print("  results.get_derived_parameters()    # List derived parameters")
            print("=" * 60)

        except Exception as e:
            print(f"Error generating summary: {e}")
def plot_posterior_comparison(results_list, labels=None, params=None, show=True, output_file=None, **kwargs):
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
    **kwargs
        Additional arguments passed to GetDist triangle_plot

    Returns
    -------
    getdist.plots.GetDistPlotter
        GetDist plotter object

    Examples
    --------
    >>> from bayesed import plot_posterior_comparison
    >>> from bayesed import standardize_parameter_names
    >>> results1 = BayeSEDResults('output_model1')
    >>> results2 = BayeSEDResults('output_model2')
    >>> # Standardize parameter names for easy comparison
    >>> standardize_parameter_names([results1, results2])
    >>> # Then compare
    >>> plot_posterior_comparison(
    ...     [results1, results2],
    ...     labels=['Model 1', 'Model 2'],
    ...     params=['log(age/yr)[0,1]', 'log(Z/Zsun)[0,1]']
    ... )
    """
    try:
        from getdist import plots
    except ImportError:
        raise ImportError("GetDist is required. Install with: pip install getdist")

    if not results_list:
        raise ValueError("results_list cannot be empty")

    # Get GetDist samples for each result
    samples_list = []

    for i, result in enumerate(results_list):
        samples_gd = result.get_getdist_samples()

        # Set name tag for legend
        if labels and i < len(labels):
            label = labels[i]
        else:
            label = f'Result {i+1}'

        # Set multiple label attributes for GetDist compatibility
        samples_gd.name_tag = label
        samples_gd.label = label
        if hasattr(samples_gd, 'name'):
            samples_gd.name = label

        samples_list.append(samples_gd)

    # Determine parameters to plot
    if params is None:
        # Use common free parameters across all results (after any renaming)
        all_free_params = []
        for result in results_list:
            # Get actual free parameters (not all parameters)
            free_params = result.get_free_parameters()
            all_free_params.append(set(free_params))

        # Find intersection of all parameter lists
        common_params = all_free_params[0]
        for param_set in all_free_params[1:]:
            common_params = common_params.intersection(param_set)

        params = list(common_params)

        if not params:
            raise ValueError("No common free parameters found across all results. "
                           "Use standardize_parameter_names() to align parameter names first.")

    # Create plotter
    g = plots.get_subplot_plotter()

    # Set plotting options for better comparison visibility
    plot_kwargs = {
        'filled': True,
        'contour_colors': ['red', 'blue', 'green', 'orange', 'purple'],
        'contour_ls': ['-', '--', '-.', ':', '-'],  # Different line styles
        'contour_lws': [1.5, 1.5, 1.5, 1.5, 1.5],  # Line widths
    }
    plot_kwargs.update(kwargs)

    # Use triangle_plot with samples list and params
    g.triangle_plot(samples_list, params, **plot_kwargs)

    # Export if requested
    if output_file:
        g.export(output_file)

    # Show plot if requested
    if show:
        try:
            import matplotlib.pyplot as plt
            # Use the most reliable display method
            plt.show()
        except Exception as e:
            import warnings
            warnings.warn(
                f"Could not display plot: {e}. "
                f"Try using output_file parameter to save the plot instead.",
                UserWarning
            )

    return g


# Helper function to standardize parameter names
def standardize_parameter_names(results_list, standard_names=None, remove_component_ids=True, custom_labels=None):
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
        Example: {'log(age/yr)': r'\\log(t/\\mathrm{yr})', 'log(Z/Zsun)': r'\\log(Z/Z_\\odot)'}

    Examples
    --------
    >>> from bayesed import standardize_parameter_names, plot_posterior_comparison
    >>> results1 = BayeSEDResults('output_model1')  # has log(age/yr)[0,0]
    >>> results2 = BayeSEDResults('output_model2')  # has log(age/yr)[0,1]
    >>>
    >>> # Basic standardization
    >>> standardize_parameter_names([results1, results2])
    >>>
    >>> # With custom LaTeX labels
    >>> custom_labels = {
    ...     'log(age/yr)': r'\\log(t/\\mathrm{yr})',
    ...     'log(Z/Zsun)': r'\\log(Z/Z_\\odot)',
    ...     'Av_2': r'A_V'$'
    ... }
    >>> standardize_parameter_names([results1, results2], custom_labels=custom_labels)
    >>> plot_posterior_comparison([results1, results2], labels=['Model 1', 'Model 2'])
    """
    import re

    def normalize_param_name(param_name):
        """Remove component IDs like [0,0], [0,1] to find equivalent parameters."""
        # Remove patterns like [0,0], [0,1], [1,0], etc.
        normalized = re.sub(r'\[\d+,\d+\]', '', param_name)
        return normalized

    # If no standard names provided, create them based on remove_component_ids setting
    if standard_names is None:
        if remove_component_ids:
            # Create clean parameter names without component IDs
            # Collect all unique normalized parameter names across all results
            all_normalized_params = set()
            for result in results_list:
                result_params = result.get_free_parameters()
                for param in result_params:
                    normalized = normalize_param_name(param)
                    all_normalized_params.add(normalized)

            # Use normalized names as the standard (clean names without [0,0], [0,1], etc.)
            standard_names = {norm_name: norm_name for norm_name in all_normalized_params}
        else:
            # Use the first result's parameter names as the standard (preserves component IDs)
            reference_params = results_list[0].get_free_parameters()
            standard_names = {normalize_param_name(p): p for p in reference_params}

    # Rename parameters in all results
    for result in results_list:
        result_params = result.get_free_parameters()
        mapping = {}

        for param in result_params:
            normalized = normalize_param_name(param)
            if normalized in standard_names:
                standard_param = standard_names[normalized]
                if param != standard_param:
                    mapping[param] = standard_param

        if mapping:
            result.rename_parameters(mapping)

    # Apply custom labels if provided
    if custom_labels:
        for result in results_list:
            result.set_parameter_labels(custom_labels)


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

    def load_results(self, output_dir, object_id=None):
        """
        Load BayeSED analysis results from output directory.

        Parameters
        ----------
        output_dir : str
            Directory containing BayeSED output files
        object_id : str or int, optional
            Object ID to load results for (if None, loads first available object)

        Returns
        -------
        BayeSEDResults
            Results object containing loaded data
        """
        return BayeSEDResults(output_dir, object_id=object_id)

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
