"""
Physical model configuration for BayeSED3.

This module provides the SEDModel class for configuring physical models (SSP, SFH, AGN components,
dust, kinematics, IGM, cosmology, priors, etc.) and nested classes for managing multiple instances.
"""

import os
from dataclasses import dataclass
from typing import Optional, Union

# Import parameter classes directly from params module to avoid circular dependencies
# TYPE_CHECKING is used for type hints to avoid importing at module level
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Type hints only - imports happen inside methods to avoid circular dependency
    from .params import (
        SSPParams, SFHParams, DALParams, GreybodyParams, BlackbodyParams,
        KinParams, BigBlueBumpParams, AGNParams, FANNParams, AKNNParams,
        LineParams, CosmologyParams, ZParams, SysErrParams, SFROverParams,
        LuminosityParams, LineListParams
    )


class SEDModel:
    """
    Physical model configuration for BayeSED3 SED analysis.
    
    This class helps create BayeSEDParams objects with physical model configuration
    (SSP, SFH, AGN components, dust, kinematics, IGM, cosmology, priors, etc.).
    
    The class provides nested classes GalaxyInstance and AGNInstance for managing
    multiple galaxy and AGN components in complex SED models.
    
    Examples
    --------
    >>> from bayesed.model import SEDModel
    >>> 
    >>> # Create galaxy instance
    >>> galaxy = SEDModel.create_galaxy(
    ...     ssp_model='bc2003_hr_stelib_chab_neb_2000r',
    ...     sfh_type='exponential',
    ...     dal_law='calzetti'
    ... )
    >>> galaxy.add_dust_emission()
    >>> 
    >>> # Create AGN instance
    >>> agn = SEDModel.create_agn(
    ...     agn_components=['dsk', 'blr', 'nlr', 'feii']
    ... )
    >>> 
    >>> # Configure additional model settings
    >>> model = SEDModel()
    >>> model.set_igm(igm_model=1)
    >>> model.set_cosmology(H0=70.0, omigaA=0.7, omigam=0.3)
    """
    
    # Store model settings as instance attributes
    def __init__(self):
        """Initialize SEDModel instance for additional physical model settings."""
        self._igm = None
        self._cosmology = None
        self._redshift_prior = None
        self._sys_err_mod = None
        self._kinematics = None
        self._luminosity = None
        self._sfr_over = None
        self._lw_max = None
        self._line_list = None
    
    @classmethod
    def create_galaxy(cls, ssp_model='bc2003_hr_stelib_chab_neb_2000r',
                     sfh_type='exponential', dal_law='calzetti',
                     ssp_k=1, ssp_f_run=1, ssp_Nstep=1, ssp_i0=0, ssp_i1=0, ssp_i2=0, ssp_i3=0,
                     sfh_itype_ceh=0, sfh_itruncated=0,
                     base_igroup=0, base_id=0):
        """
        Create a GalaxyInstance with default parameters.
        
        Parameters
        ----------
        ssp_model : str
            SSP model name (default: 'bc2003_hr_stelib_chab_neb_2000r')
        sfh_type : str or int
            Star formation history type (default: 'exponential')
        dal_law : str or int
            Dust attenuation law (default: 'calzetti')
        ssp_k, ssp_f_run, ssp_Nstep, ssp_i0, ssp_i1, ssp_i2, ssp_i3 : int
            SSP parameters
        sfh_itype_ceh : int
            Chemical evolution history type for SFH (default: 0)
        sfh_itruncated : int
            Truncation flag for SFH (default: 0)
        base_igroup : int
            Base igroup for this galaxy instance (default: 0)
        base_id : int
            Base ID for this galaxy instance (default: 0)
        
        Returns
        -------
        SEDModel.GalaxyInstance
            New galaxy instance
        """
        return cls.GalaxyInstance.create(
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
            sfh_itype_ceh=sfh_itype_ceh,
            sfh_itruncated=sfh_itruncated,
            base_igroup=base_igroup,
            base_id=base_id
        )
    
    @classmethod
    def create_agn(cls, base_igroup=0, base_id=0, agn_components=None,
                   blr_lines_file='observation/test/lines_BLR.txt',
                   nlr_lines_file='observation/test/lines_NLR.txt'):
        """
        Create an AGNInstance with specified components.
        
        Parameters
        ----------
        base_igroup : int, optional
            Base igroup for this AGN instance (default: 0, will be auto-assigned when added to params)
        base_id : int, optional
            Base ID for this AGN instance (default: 0, will be auto-assigned when added to params)
        agn_components : list of str, optional
            List of component names to include. Valid components: 'dsk', 'disk', 'bbb', 'blr', 'nlr', 'feii', 'tor', 'torus'.
            If None (default), includes all components: ['dsk', 'blr', 'nlr', 'feii'].
            Note: 'dsk'/'disk'/'bbb' adds a BBB disk by default. Use instance methods to add other disk types (AGN, FANN, AKNN).
            Note: 'tor' or 'torus' adds a FANN torus by default. Use instance methods to add AKNN torus.
        blr_lines_file : str
            Path to BLR line list file
        nlr_lines_file : str
            Path to NLR line list file
        
        Returns
        -------
        SEDModel.AGNInstance
            New AGN instance with specified components
        """
        return cls.AGNInstance.create(
            base_igroup=base_igroup,
            base_id=base_id,
            agn_components=agn_components,
            blr_lines_file=blr_lines_file,
            nlr_lines_file=nlr_lines_file
        )
    
    def set_igm(self, igm_model=1):
        """
        Set IGM (Intergalactic Medium) absorption model.
        
        Parameters
        ----------
        igm_model : int
            IGM model: 0=None, 1=Madau 1995 (default), 2=Meiksin 2006, 3=hyperz, 4=FSPS, 5=Inoue+2014
        """
        self._igm = igm_model
        return self
    
    def set_cosmology(self, H0=70.0, omigaA=0.7, omigam=0.3):
        """
        Set cosmology parameters.
        
        Parameters
        ----------
        H0 : float
            Hubble constant in km/s/Mpc (default: 70.0)
        omigaA : float
            Omega Lambda (default: 0.7)
        omigam : float
            Omega Matter (default: 0.3)
        """
        # Import here to avoid circular dependency
        from .params import CosmologyParams
        self._cosmology = CosmologyParams(H0=H0, omigaA=omigaA, omigam=omigam)
        return self
    
    def set_redshift_prior(self, iprior_type=1, min=0.0, max=4.0, nbin=40):
        """
        Set redshift prior parameters.
        
        Parameters
        ----------
        iprior_type : int
            Prior type (default: 1)
        min : float
            Minimum redshift (default: 0.0)
        max : float
            Maximum redshift (default: 4.0)
        nbin : int
            Number of bins (default: 40)
        """
        # Import here to avoid circular dependency
        from .params import ZParams
        self._redshift_prior = ZParams(iprior_type=iprior_type, min=min, max=max, nbin=nbin)
        return self
    
    def set_sys_err_mod(self, iprior_type=1, min=0.0, max=0.1, nbin=40):
        """
        Set systematic error in model.
        
        Parameters
        ----------
        iprior_type : int
            Prior type (default: 1)
        min : float
            Minimum value (default: 0.0)
        max : float
            Maximum value (default: 0.1)
        nbin : int
            Number of bins (default: 40)
        """
        # Import here to avoid circular dependency
        from .params import SysErrParams
        self._sys_err_mod = SysErrParams(iprior_type=iprior_type, min=min, max=max, nbin=nbin)
        return self
    
    def set_kinematics(self, id, velscale=10, num_gauss_hermites_con=0, num_gauss_hermites_eml=0):
        """
        Set kinematics parameters.
        
        Parameters
        ----------
        id : int
            Model ID
        velscale : int
            Velocity scale (default: 10)
        num_gauss_hermites_con : int
            Number of Gauss-Hermite terms for continuum (default: 0)
        num_gauss_hermites_eml : int
            Number of Gauss-Hermite terms for emission lines (default: 0)
        """
        # Import here to avoid circular dependency
        from .params import KinParams
        self._kinematics = KinParams(
            id=id,
            velscale=velscale,
            num_gauss_hermites_con=num_gauss_hermites_con,
            num_gauss_hermites_eml=num_gauss_hermites_eml
        )
        return self
    
    def set_luminosity(self, id=-1, w_min=None, w_max=None):
        """
        Set luminosity constraints.
        
        Parameters
        ----------
        id : int
            Model ID (-1 for all models, default: -1)
        w_min : float, optional
            Minimum wavelength
        w_max : float, optional
            Maximum wavelength
        """
        # Import here to avoid circular dependency
        from .params import LuminosityParams
        if w_min is None or w_max is None:
            raise ValueError("w_min and w_max must be provided")
        self._luminosity = LuminosityParams(id=id, w_min=w_min, w_max=w_max)
        return self
    
    def set_sfr_over(self, past_Myr1=10.0, past_Myr2=100.0):
        """
        Set star formation rate averaging.
        
        Parameters
        ----------
        past_Myr1 : float
            First past time in Myr (default: 10.0)
        past_Myr2 : float
            Second past time in Myr (default: 100.0)
        """
        # Import here to avoid circular dependency
        from .params import SFROverParams
        self._sfr_over = SFROverParams(past_Myr1=past_Myr1, past_Myr2=past_Myr2)
        return self
    
    def set_lw_max(self, lw_max):
        """
        Set maximum wavelength for model.
        
        Parameters
        ----------
        lw_max : float
            Maximum wavelength
        """
        self._lw_max = lw_max
        return self
    
    def set_line_list(self, file, type=0):
        """
        Set emission line list.
        
        Parameters
        ----------
        file : str
            File containing the line list
        type : int
            Type of line list: 0=intrinsic, 1=emergent, 2=intrinsic cumulative, 3=emergent cumulative (default: 0)
        """
        # Import here to avoid circular dependency
        from .params import LineListParams
        self._line_list = LineListParams(file=file, type=type)
        return self
    
    @dataclass
    class GalaxyInstance:
        """
        Represents a galaxy instance with SSP, SFH, DAL, and optionally DEM components.
        
        This nested class encapsulates all components that belong to a single galaxy instance,
        making it easier to manage multiple galaxy components in complex SED models.
        
        Attributes
        ----------
        ssp : SSPParams
            Stellar population synthesis model
        sfh : SFHParams
            Star formation history
        dal : DALParams
            Dust attenuation law
        dem : Optional[Union[GreybodyParams, BlackbodyParams]]
            Dust emission model (optional)
        kin : Optional[KinParams]
            Kinematic parameters (optional)
        """
        ssp: 'SSPParams'
        sfh: 'SFHParams'
        dal: 'DALParams'
        dem: Optional[Union['GreybodyParams', 'BlackbodyParams']] = None
        kin: Optional['KinParams'] = None
        
        @property
        def id(self):
            """Get the ID shared by SSP, SFH, and DAL."""
            return self.ssp.id
        
        @property
        def igroup(self):
            """Get the igroup from SSP."""
            return self.ssp.igroup
        
        @classmethod
        def create(cls, ssp_model='bc2003_hr_stelib_chab_neb_2000r',
                   sfh_type='exponential', dal_law='calzetti',
                   ssp_k=1, ssp_f_run=1, ssp_Nstep=1, ssp_i0=0, ssp_i1=0, ssp_i2=0, ssp_i3=0,
                   sfh_itype_ceh=0, sfh_itruncated=0,
                   base_igroup=0, base_id=0):
            """
            Create a GalaxyInstance with default parameters.
            
            Parameters
            ----------
            ssp_model : str
                SSP model name (default: 'bc2003_hr_stelib_chab_neb_2000r')
            sfh_type : str or int
                Star formation history type (default: 'exponential')
            dal_law : str or int
                Dust attenuation law (default: 8)
            ssp_k, ssp_f_run, ssp_Nstep, ssp_i0, ssp_i1, ssp_i2, ssp_i3 : int
                SSP parameters
            sfh_itype_ceh : int
                Chemical evolution history type for SFH (default: 0)
            sfh_itruncated : int
                Truncation flag for SFH (default: 0)
            base_igroup : int
                Base igroup for this galaxy instance (default: 0)
            base_id : int
                Base ID for this galaxy instance (default: 0)
            
            Returns
            -------
            SEDModel.GalaxyInstance
                New galaxy instance
            """
            # Import here to avoid circular dependency
            from .params import SSPParams, SFHParams, DALParams
            
            # Map SFH type strings to integers
            sfh_type_map = {
                'instantaneous': 0, 'instantaneous_burst': 0, 'burst': 0,
                'constant': 1,
                'exponential': 2, 'exponentially_declining': 2,
                'exponentially_increasing': 3, 'increasing': 3,
                'single_burst': 4, 'burst_length_tau': 4,
                'delayed': 5, 'delayed_exponential': 5,
                'beta': 6,
                'lognormal': 7, 'log_normal': 7,
                'double_powerlaw': 8, 'double_power_law': 8,
                'nonparametric': 9, 'non_parametric': 9,
            }
            if isinstance(sfh_type, str):
                sfh_type_lower = sfh_type.lower().replace(' ', '_').replace('-', '_')
                sfh_type = sfh_type_map.get(sfh_type_lower, sfh_type)
                if isinstance(sfh_type, str):
                    try:
                        sfh_type = int(sfh_type)
                    except ValueError:
                        raise ValueError(f"Unknown sfh_type '{sfh_type}'")
            
            sfh_type_int = int(sfh_type)
            if not (0 <= sfh_type_int <= 9):
                raise ValueError(f"sfh_type must be between 0 and 9, got {sfh_type_int}")
            
            # Map DAL law strings to integers
            dal_law_map = {
                'sed_model': 0, 'sed_normalization': 0,
                'starburst': 1, 'starburst_calzetti': 1, 'calzetti_fast': 1,
                'milky_way': 2, 'milky_way_cardelli': 2, 'cardelli': 2,
                'star_forming': 3, 'star_forming_salim': 3, 'salim': 3,
                'mw_allen': 4, 'allen': 4,
                'mw_fitzpatrick': 5, 'fitzpatrick_mw': 5,
                'lmc': 6, 'lmc_fitzpatrick': 6, 'fitzpatrick_lmc': 6,
                'smc': 7, 'smc_fitzpatrick': 7, 'fitzpatrick_smc': 7,
                'calzetti': 8, 'calzetti2000': 8, 'starburst_calzetti2000': 8,
                'star_forming_reddy': 9, 'reddy': 9,
            }
            if isinstance(dal_law, str):
                dal_law_lower = dal_law.lower().replace(' ', '_').replace('-', '_')
                dal_law = dal_law_map.get(dal_law_lower, dal_law)
                if isinstance(dal_law, str):
                    try:
                        dal_law = int(dal_law)
                    except ValueError:
                        raise ValueError(f"Unknown dal_law '{dal_law}'")
            
            dal_law_int = int(dal_law)
            if not (0 <= dal_law_int <= 9):
                raise ValueError(f"dal_law must be between 0 and 9, got {dal_law_int}")
            
            ssp = SSPParams(
                igroup=base_igroup,
                id=base_id,
                name=ssp_model,
                iscalable=1,
                k=ssp_k,
                f_run=ssp_f_run,
                Nstep=ssp_Nstep,
                i0=ssp_i0,
                i1=ssp_i1,
                i2=ssp_i2,
                i3=ssp_i3
            )
            
            sfh = SFHParams(
                id=base_id,
                itype_sfh=sfh_type_int,
                itruncated=sfh_itruncated,
                itype_ceh=sfh_itype_ceh
            )
            
            dal = DALParams(
                id=base_id,
                con_eml_tot=2,
                ilaw=dal_law_int
            )
            
            return cls(ssp=ssp, sfh=sfh, dal=dal)
        
        def add_dust_emission(self, model_type='greybody', iscalable=-2,
                             w_min=1.0, w_max=1000.0, Nw=200, ithick=0):
            """
            Add dust emission model to this galaxy instance.
            
            Following the GUI pattern, DEM uses the same igroup as SSP and id = SSP id + 1.
            
            Parameters
            ----------
            model_type : str
                Type of dust emission model: 'greybody' (default) or 'blackbody'
            iscalable : int
                Scalability parameter. Default: -2 (scaled by dust mass)
            w_min, w_max : float
                Wavelength range in microns (default: 1.0 to 1000.0)
            Nw : int
                Number of wavelength points (default: 200)
            ithick : int
                Thickness parameter for greybody (default: 0, only used for greybody)
            
            Returns
            -------
            self
                Returns self for method chaining
            """
            # Import here to avoid circular dependency
            from .params import GreybodyParams, BlackbodyParams
            
            dem_id = self.ssp.id + 1
            dem_igroup = self.ssp.igroup
            
            if model_type.lower() == 'greybody':
                self.dem = GreybodyParams(
                    igroup=dem_igroup,
                    id=dem_id,
                    name='gb',
                    iscalable=iscalable,
                    ithick=ithick,
                    w_min=w_min,
                    w_max=w_max,
                    Nw=Nw
                )
            elif model_type.lower() == 'blackbody':
                self.dem = BlackbodyParams(
                    igroup=dem_igroup,
                    id=dem_id,
                    bb='bb',
                    iscalable=iscalable,
                    w_min=w_min,
                    w_max=w_max,
                    Nw=Nw
                )
            else:
                raise ValueError(f"Unknown model_type: {model_type}. Use 'greybody' or 'blackbody'")
            
            return self
    
    @dataclass
    class AGNInstance:
        """
        Represents an AGN instance with disk (DSK), BLR, NLR, FeII, and optionally TOR components.
        
        This nested class encapsulates all components that belong to a single AGN instance,
        making it easier to manage multiple AGN components in complex SED models.
        
        Attributes
        ----------
        base_igroup : int
            Base igroup for this AGN instance
        base_id : int
            Base ID for this AGN instance
        dsk : Optional[Union[BigBlueBumpParams, AGNParams, FANNParams, AKNNParams]]
            Disk (accretion disk) component - can be BBB, AGN, FANN, or AKNN type
        blr : Optional[LineParams]
            Broad Line Region component
        nlr : Optional[LineParams]
            Narrow Line Region component
        feii : Optional[AKNNParams]
            FeII template component
        tor : Optional[Union[FANNParams, AKNNParams]]
            Torus component (optional, can be FANN or AKNN)
        """
        base_igroup: int
        base_id: int
        dsk: Optional[Union['BigBlueBumpParams', 'AGNParams', 'FANNParams', 'AKNNParams']] = None
        blr: Optional['LineParams'] = None
        nlr: Optional['LineParams'] = None
        feii: Optional['AKNNParams'] = None
        tor: Optional[Union['FANNParams', 'AKNNParams']] = None
        
        def add_disk_bbb(self, name='bbb', iscalable=1,
                        w_min=0.1, w_max=10.0, Nw=1000):
            """Add Big Blue Bump (BBB) disk component (uses base + 1)."""
            from .params import BigBlueBumpParams
            self.dsk = BigBlueBumpParams(
                igroup=self.base_igroup + 1,
                id=self.base_id + 1,
                name=name,
                iscalable=iscalable,
                w_min=w_min,
                w_max=w_max,
                Nw=Nw
            )
            return self
        
        def add_disk_agn(self, name='agnsed', iscalable=1, imodel=0, icloudy=0,
                        suffix='', w_min=0.1, w_max=10.0, Nw=1000):
            """Add AGN disk component (uses base + 1)."""
            from .params import AGNParams
            self.dsk = AGNParams(
                igroup=self.base_igroup + 1,
                id=self.base_id + 1,
                name=name,
                iscalable=iscalable,
                imodel=imodel,
                icloudy=icloudy,
                suffix=suffix,
                w_min=w_min,
                w_max=w_max,
                Nw=Nw
            )
            return self
        
        def add_disk_fann(self, name='disk_fann', iscalable=1):
            """Add FANN disk component (uses base + 1)."""
            from .params import FANNParams
            self.dsk = FANNParams(
                igroup=self.base_igroup + 1,
                id=self.base_id + 1,
                name=name,
                iscalable=iscalable
            )
            return self
        
        def add_disk_aknn(self, name='disk_aknn', iscalable=1,
                         k=1, f_run=1, eps=0, iRad=0, iprep=0, Nstep=1, alpha=0):
            """Add AKNN disk component (uses base + 1)."""
            from .params import AKNNParams
            self.dsk = AKNNParams(
                igroup=self.base_igroup + 1,
                id=self.base_id + 1,
                name=name,
                iscalable=iscalable,
                k=k,
                f_run=f_run,
                eps=eps,
                iRad=iRad,
                iprep=iprep,
                Nstep=Nstep,
                alpha=alpha
            )
            return self
        
        def add_broad_line_region(self, file='observation/test/lines_BLR.txt',
                                 name='BLR', iscalable=1, R=300,
                                 Nsample=2, Nkin=3):
            """Add Broad Line Region component (uses base + 2)."""
            from .params import LineParams
            if not os.path.exists(file):
                raise FileNotFoundError(f"BLR line list file not found: {file}")
            self.blr = LineParams(
                igroup=self.base_igroup + 2,
                id=self.base_id + 2,
                name=name,
                iscalable=iscalable,
                file=file,
                R=R,
                Nsample=Nsample,
                Nkin=Nkin
            )
            return self
        
        def add_narrow_line_region(self, file='observation/test/lines_NLR.txt',
                                  name='NLR', iscalable=1, R=2000,
                                  Nsample=2, Nkin=2):
            """Add Narrow Line Region component (uses base + 4)."""
            from .params import LineParams
            if not os.path.exists(file):
                raise FileNotFoundError(f"NLR line list file not found: {file}")
            self.nlr = LineParams(
                igroup=self.base_igroup + 4,
                id=self.base_id + 4,
                name=name,
                iscalable=iscalable,
                file=file,
                R=R,
                Nsample=Nsample,
                Nkin=Nkin
            )
            return self
        
        def add_feii_template(self, name='FeII', iscalable=1,
                             k=5, f_run=1, eps=0.01, iRad=0, iprep=0, Nstep=100,
                             alpha=1.0):
            """Add FeII template component (uses base + 3)."""
            from .params import AKNNParams
            self.feii = AKNNParams(
                igroup=self.base_igroup + 3,
                id=self.base_id + 3,
                name=name,
                iscalable=iscalable,
                k=k,
                f_run=f_run,
                eps=eps,
                iRad=iRad,
                iprep=iprep,
                Nstep=Nstep,
                alpha=alpha
            )
            return self
        
        def add_torus_fann(self, name='clumpy201410tor', iscalable=1):
            """Add FANN (Fast Artificial Neural Network) torus component (uses base + 5)."""
            from .params import FANNParams
            self.tor = FANNParams(
                igroup=self.base_igroup + 5,
                id=self.base_id + 5,
                name=name,
                iscalable=iscalable
            )
            return self
        
        def add_torus_aknn(self, name='torus_aknn', iscalable=1,
                          k=1, f_run=1, eps=0, iRad=0, iprep=0, Nstep=1, alpha=0):
            """Add AKNN torus component (uses base + 5)."""
            from .params import AKNNParams
            self.tor = AKNNParams(
                igroup=self.base_igroup + 5,
                id=self.base_id + 5,
                name=name,
                iscalable=iscalable,
                k=k,
                f_run=f_run,
                eps=eps,
                iRad=iRad,
                iprep=iprep,
                Nstep=Nstep,
                alpha=alpha
            )
            return self
        
        def get_dal_params(self):
            """Get DAL parameters for disk component (same id as disk)."""
            from .params import DALParams, BigBlueBumpParams, AGNParams
            if self.dsk:
                # DAL is only for BBB and AGN types
                if isinstance(self.dsk, (BigBlueBumpParams, AGNParams)):
                    return DALParams(
                        id=self.base_id + 1,
                        con_eml_tot=2,
                        ilaw=7  # AGN-specific dust law
                    )
            return None
        
        def get_kin_params(self):
            """Get kinematic parameters for FeII component (same id as FeII)."""
            from .params import KinParams
            if self.feii:
                return KinParams(
                    id=self.base_id + 3,
                    velscale=10,
                    num_gauss_hermites_con=2,
                    num_gauss_hermites_eml=0
                )
            return None
        
        @classmethod
        def create(cls, base_igroup=0, base_id=0, agn_components=None,
                   blr_lines_file='observation/test/lines_BLR.txt',
                   nlr_lines_file='observation/test/lines_NLR.txt'):
            """
            Create an AGNInstance with specified components.
            
            Parameters
            ----------
            base_igroup : int, optional
                Base igroup for this AGN instance (default: 0, will be auto-assigned when added to params)
            base_id : int, optional
                Base ID for this AGN instance (default: 0, will be auto-assigned when added to params)
            agn_components : list of str, optional
                List of component names to include. Valid components: 'dsk', 'disk', 'bbb', 'blr', 'nlr', 'feii', 'tor', 'torus'.
                If None (default), includes all components: ['dsk', 'blr', 'nlr', 'feii'].
                Note: 'dsk'/'disk'/'bbb' adds a BBB disk by default. Use instance methods to add other disk types (AGN, FANN, AKNN).
                Note: 'tor' or 'torus' adds a FANN torus by default. Use instance methods to add AKNN torus.
            blr_lines_file : str
                Path to BLR line list file
            nlr_lines_file : str
                Path to NLR line list file
            
            Returns
            -------
            SEDModel.AGNInstance
                New AGN instance with specified components
            """
            instance = cls(base_igroup=base_igroup, base_id=base_id)
            
            # Default to all components if not specified
            if agn_components is None:
                agn_components = ['dsk', 'blr', 'nlr', 'feii']
            
            # Ensure agn_components is a list
            if not isinstance(agn_components, (list, tuple)):
                raise ValueError(f"agn_components must be a list of component names, got {type(agn_components)}")
            
            agn_components_list = [comp.lower() for comp in agn_components]
            
            # Component name mapping
            component_map = {
                'dsk': 'dsk', 'disk': 'dsk',
                'bbb': 'dsk', 'big_blue_bump': 'dsk', 'bigbluebump': 'dsk',
                'blr': 'blr', 'broad_line_region': 'blr', 'broadlineregion': 'blr',
                'nlr': 'nlr', 'narrow_line_region': 'nlr', 'narrowlineregion': 'nlr',
                'feii': 'feii', 'fe_ii': 'feii', 'fe_2': 'feii',
                'tor': 'tor', 'torus': 'tor',
            }
            
            normalized_components = []
            for comp in agn_components_list:
                normalized = component_map.get(comp, comp)
                if normalized not in ['dsk', 'blr', 'nlr', 'feii', 'tor']:
                    raise ValueError(
                        f"Unknown AGN component '{comp}'. Valid components: "
                        f"'dsk'/'disk'/'bbb', 'blr', 'nlr', 'feii', 'tor'/'torus'"
                    )
                normalized_components.append(normalized)
            
            # Remove duplicates while preserving order
            seen = set()
            unique_components = []
            for comp in normalized_components:
                if comp not in seen:
                    seen.add(comp)
                    unique_components.append(comp)
            
            # Add components
            if 'dsk' in unique_components:
                instance.add_disk_bbb()
            if 'blr' in unique_components:
                instance.add_broad_line_region(file=blr_lines_file)
            if 'feii' in unique_components:
                instance.add_feii_template()
            if 'nlr' in unique_components:
                instance.add_narrow_line_region(file=nlr_lines_file)
            if 'tor' in unique_components:
                instance.add_torus_fann()
            
            return instance

