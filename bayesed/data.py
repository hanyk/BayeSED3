"""
Data preparation classes for BayeSED3.

This module provides classes for handling observation data (photometry, spectroscopy, or both)
and converting them to BayeSED3's file-based input format.
"""

import os
from dataclasses import dataclass
from typing import Optional, List, Union
import numpy as np

# Import utility functions directly from utils module
from .utils import _to_array, create_input_catalog

# Import parameter classes directly from params module to avoid circular dependencies
# TYPE_CHECKING is used for type hints to avoid importing at module level
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Type hints only - avoids circular dependency with __init__.py
    from .params import SNRmin1Params, SNRmin2Params, SysErrParams


@dataclass
class SEDObservation:
    """
    Unified class for handling photometry, spectroscopy, or both together.
    
    This is the primary class for observation data since BayeSED input files can contain
    both photometry and spectroscopy together in a single file.
    
    Attributes
    ----------
    ids : array-like
        Object IDs (length N)
    z_min : array-like
        Minimum redshift for each object (length N)
    z_max : array-like
        Maximum redshift for each object (length N)
    distance_mpc : array-like, optional
        Distance in Mpc (length N, defaults to zeros)
    ebv : array-like, optional
        E(B-V) reddening (length N, defaults to zeros)
    
    Photometry (optional):
    phot_filters : list of str, optional
        List of photometric filter names (e.g., ['SLOAN/SDSS.u', 'SLOAN/SDSS.g'])
    phot_fluxes : array-like, optional
        Photometric fluxes, shape (N, Nphot). Units depend on input_type.
    phot_errors : array-like, optional
        Photometric errors, shape (N, Nphot). Units depend on input_type.
    phot_flux_limits : array-like, optional
        Flux limits for nondetections, shape (N, Nphot) or (Nphot,)
    phot_mag_limits : array-like, optional
        Magnitude limits for nondetections, shape (N, Nphot) or (Nphot,)
    phot_nsigma : float, optional
        Number of sigma for nondetection threshold
    
    Spectroscopy (optional):
    spec_band_names : list of str, optional
        List of spectral band names (e.g., ['B', 'R'])
    spec_wavelengths : list of arrays, optional
        List of wavelength arrays, one per band. Each array is (Nw,) for single object
        or (N, Nw) for multiple objects, in Angstrom.
    spec_fluxes : list of arrays, optional
        List of flux arrays, one per band. Each array is (Nw,) for single object
        or (N, Nw) for multiple objects. Units depend on spec_flux_type.
    spec_errors : list of arrays, optional
        List of error arrays, one per band. Each array is (Nw,) for single object
        or (N, Nw) for multiple objects. Units depend on spec_flux_type.
    spec_lsf_sigma : list of arrays, optional
        List of LSF sigma arrays, one per band. Each array is (Nw,) for single object
        or (N, Nw) for multiple objects, in microns.
    spec_flux_type : str, optional
        Type of spectral flux: 'fnu' (microJy, default) or 'flambda' (erg s^-1 cm^-2 Å^-1)
    
    Input data and filter parameters:
    input_type : int, default 0
        0 for flux in μJy, 1 for AB magnitude (only applies to photometry data)
    filters : str, optional
        Path to filter definition file
    filters_selected : str, optional
        Path to filter selection file
    NfilterPoints : int, optional
        Number of filter points
    
    Data quality control parameters:
    no_photometry_fit : bool, default False
        Whether to skip photometry fitting
    no_spectra_fit : bool, default False
        Whether to skip spectroscopy fitting
    SNRmin1 : SNRmin1Params, optional
        Minimum SNR thresholds for photometry/spectroscopy
    SNRmin2 : SNRmin2Params, optional
        Additional SNR thresholds
    sys_err_obs : SysErrParams, optional
        Systematic error in observations
    
    Other columns:
    other_columns : dict, optional
        Dictionary mapping column names to array-like data (length N)
    """
    # Base columns
    ids: Union[List, np.ndarray]
    z_min: Union[List, np.ndarray]
    z_max: Union[List, np.ndarray]
    distance_mpc: Optional[Union[List, np.ndarray]] = None
    ebv: Optional[Union[List, np.ndarray]] = None
    
    # Photometry
    phot_filters: Optional[List[str]] = None
    phot_fluxes: Optional[np.ndarray] = None
    phot_errors: Optional[np.ndarray] = None
    phot_flux_limits: Optional[np.ndarray] = None
    phot_mag_limits: Optional[np.ndarray] = None
    phot_nsigma: Optional[float] = None
    
    # Spectroscopy
    spec_band_names: Optional[List[str]] = None
    spec_wavelengths: Optional[List[np.ndarray]] = None
    spec_fluxes: Optional[List[np.ndarray]] = None
    spec_errors: Optional[List[np.ndarray]] = None
    spec_lsf_sigma: Optional[List[np.ndarray]] = None
    spec_flux_type: str = "fnu"
    
    # Input data and filter parameters
    input_type: int = 0  # 0 for flux in μJy, 1 for AB magnitude (only applies to photometry)
    filters: Optional[str] = None
    filters_selected: Optional[str] = None
    NfilterPoints: Optional[int] = None
    
    # Data quality control parameters
    no_photometry_fit: bool = False
    no_spectra_fit: bool = False
    SNRmin1: Optional['SNRmin1Params'] = None
    SNRmin2: Optional['SNRmin2Params'] = None
    sys_err_obs: Optional['SysErrParams'] = None
    
    # Other columns
    other_columns: Optional[dict] = None
    
    def validate(self):
        """
        Validate observation data.
        
        Raises
        ------
        ValueError
            If data is invalid or inconsistent
        """
        ids = _to_array(self.ids)
        z_min = _to_array(self.z_min)
        z_max = _to_array(self.z_max)
        
        N = len(ids)
        if not (len(z_min) == len(z_max) == N):
            raise ValueError("ids, z_min, and z_max must have equal length")
        
        if N == 0:
            raise ValueError("Must have at least one object")
        
        # Validate photometry
        if self.phot_filters is not None:
            if self.phot_fluxes is None or self.phot_errors is None:
                raise ValueError("phot_fluxes and phot_errors are required when phot_filters is provided")
            
            phot_fluxes = _to_array(self.phot_fluxes)
            phot_errors = _to_array(self.phot_errors)
            
            if phot_fluxes.shape[0] != N:
                raise ValueError(f"phot_fluxes must have length N={N} in first dimension")
            if phot_errors.shape != phot_fluxes.shape:
                raise ValueError("phot_fluxes and phot_errors must have the same shape")
            if phot_fluxes.shape[1] != len(self.phot_filters):
                raise ValueError(f"phot_fluxes.shape[1]={phot_fluxes.shape[1]} must equal len(phot_filters)={len(self.phot_filters)}")
        
        # Validate spectroscopy
        if self.spec_band_names is not None:
            if self.spec_wavelengths is None or self.spec_fluxes is None or self.spec_errors is None:
                raise ValueError("spec_wavelengths, spec_fluxes, and spec_errors are required when spec_band_names is provided")
            if self.spec_lsf_sigma is None:
                raise ValueError("spec_lsf_sigma is required when spec_band_names is provided")
            
            Nspec = len(self.spec_band_names)
            if len(self.spec_wavelengths) != Nspec:
                raise ValueError(f"spec_wavelengths must have length {Nspec} (one per band)")
            if len(self.spec_fluxes) != Nspec:
                raise ValueError(f"spec_fluxes must have length {Nspec} (one per band)")
            if len(self.spec_errors) != Nspec:
                raise ValueError(f"spec_errors must have length {Nspec} (one per band)")
            if len(self.spec_lsf_sigma) != Nspec:
                raise ValueError(f"spec_lsf_sigma must have length {Nspec} (one per band)")
            
            for i, band in enumerate(self.spec_band_names):
                wl = _to_array(self.spec_wavelengths[i])
                fl = _to_array(self.spec_fluxes[i])
                err = _to_array(self.spec_errors[i])
                lsf = _to_array(self.spec_lsf_sigma[i])
                
                # Handle 1D case for single object
                if wl.ndim == 1:
                    if N == 1:
                        wl = wl.reshape(1, -1)
                    else:
                        raise ValueError(f"spec_wavelengths for band '{band}' is 1D but N={N} > 1")
                
                if not (wl.shape == fl.shape == err.shape == lsf.shape):
                    raise ValueError(f"All spectral arrays for band '{band}' must have identical shape")
                if wl.shape[0] != N:
                    raise ValueError(f"Spectral arrays for band '{band}' must have length N={N} in first dimension")
        
        # Validate input_type
        if self.input_type not in (0, 1):
            raise ValueError("input_type must be 0 (flux in μJy) or 1 (AB magnitude)")
        
        # Validate spec_flux_type
        if self.spec_flux_type not in ("fnu", "flambda"):
            raise ValueError("spec_flux_type must be 'fnu' (microJy) or 'flambda' (erg s^-1 cm^-2 Å^-1)")
        
        # Validate filter files if provided
        if self.filters is not None and not os.path.exists(self.filters):
            raise FileNotFoundError(f"Filter definition file not found: {self.filters}")
        if self.filters_selected is not None and not os.path.exists(self.filters_selected):
            raise FileNotFoundError(f"Filter selection file not found: {self.filters_selected}")
    
    def to_bayesed_input(self, output_path: str, catalog_name: str = "input_catalog") -> str:
        """
        Convert observation data to BayeSED input file format.
        
        Parameters
        ----------
        output_path : str
            Directory path where the input catalog file will be created
        catalog_name : str, default "input_catalog"
            Name of the catalog (used in header)
        
        Returns
        -------
        str
            Path to the created input catalog file
        """
        # Construct full file path
        catalog_file = os.path.join(output_path, f"{catalog_name}.txt")
        
        # Determine phot_type from input_type
        phot_type = "fnu" if self.input_type == 0 else "abmag"
        
        # Call create_input_catalog with file path
        create_input_catalog(
            output_path=catalog_file,
            catalog_name=catalog_name,
            ids=self.ids,
            z_min=self.z_min,
            z_max=self.z_max,
            distance_mpc=self.distance_mpc,
            ebv=self.ebv,
            phot_band_names=self.phot_filters,
            phot_fluxes=self.phot_fluxes,
            phot_errors=self.phot_errors,
            phot_type=phot_type,
            phot_flux_limits=self.phot_flux_limits,
            phot_mag_limits=self.phot_mag_limits,
            phot_nsigma=self.phot_nsigma,
            other_columns=self.other_columns,
            spec_band_names=self.spec_band_names,
            spec_wavelengths=self.spec_wavelengths,
            spec_fluxes=self.spec_fluxes,
            spec_errors=self.spec_errors,
            spec_lsf_sigma=self.spec_lsf_sigma,
            spec_flux_type=self.spec_flux_type
        )
        
        # Return path to created file
        return catalog_file


@dataclass
class PhotometryObservation(SEDObservation):
    """
    Convenience class for photometry-only data.
    
    This is a wrapper around SEDObservation that simplifies creating photometry-only observations.
    """
    
    def __post_init__(self):
        """Ensure no spectroscopy data is provided."""
        if self.spec_band_names is not None:
            raise ValueError("PhotometryObservation cannot have spectroscopy data. Use SEDObservation for combined data.")


@dataclass
class SpectrumObservation(SEDObservation):
    """
    Convenience class for spectroscopy-only data.
    
    This is a wrapper around SEDObservation that simplifies creating spectroscopy-only observations.
    """
    
    def __post_init__(self):
        """Ensure no photometry data is provided."""
        if self.phot_filters is not None:
            raise ValueError("SpectrumObservation cannot have photometry data. Use SEDObservation for combined data.")

