"""
Parameter dataclasses for BayeSED3.

This module contains all parameter dataclasses used to configure BayeSED3 analysis,
including model parameters (SSP, SFH, AGN, etc.), algorithm parameters (MultiNest, GSL, etc.),
and other configuration parameters.
"""

from dataclasses import dataclass
from typing import Optional, Union

@dataclass
class FANNParams:
    """
    Parameters for FANN (Fast Artificial Neural Network) model.
    """
    igroup: int  # Group ID
    id: int  # Model ID
    name: str  # Name of the FANN model
    iscalable: int  # Whether the model is scalable (0: No, 1: Yes)

@dataclass
class AGNParams:
    """
    Parameters for AGN (Active Galactic Nuclei) model.
    """
    igroup: int  # Group ID
    id: int  # Model ID
    name: str  # AGN model type (qsosed|agnsed|fagnsed|relagn|relqso|agnslim)
    iscalable: int  # Whether the model is scalable (0: No, 1: Yes)
    imodel: int  # Model subtype
    icloudy: int  # Cloudy model flag
    suffix: str  # Suffix for the model name
    w_min: float  # Minimum wavelength
    w_max: float  # Maximum wavelength
    Nw: int  # Number of wavelength points

@dataclass
class BlackbodyParams:
    """
    Parameters for blackbody model.
    """
    igroup: int  # Group ID
    id: int  # Model ID
    bb: str  # Blackbody model name
    iscalable: int  # Whether the model is scalable (0: No, 1: Yes)
    w_min: float  # Minimum wavelength
    w_max: float  # Maximum wavelength
    Nw: int  # Number of wavelength points

@dataclass
class BigBlueBumpParams:
    """
    Parameters for big blue bump model.
    """
    igroup: int  # Group ID
    id: int  # Model ID
    name: str  # Name of the big blue bump model
    iscalable: int  # Whether the model is scalable (0: No, 1: Yes)
    w_min: float  # Minimum wavelength
    w_max: float  # Maximum wavelength
    Nw: int  # Number of wavelength points

@dataclass
class GreybodyParams:
    """
    Parameters for greybody model.
    """
    igroup: int  # Group ID
    id: int  # Model ID
    name: str  # Name of the greybody model
    iscalable: int  # Whether the model is scalable (0: No, 1: Yes)
    ithick: int = 0  # Thickness parameter
    w_min: float = 1  # Minimum wavelength
    w_max: float = 1000  # Maximum wavelength
    Nw: int = 200  # Number of wavelength points

@dataclass
class AKNNParams:
    """
    Parameters for AKNN model.
    """
    igroup: int  # Group ID
    id: int  # Model ID
    name: str  # Name of the AKNN model
    iscalable: int  # Whether the model is scalable (0: No, 1: Yes)
    k: int = 1  # Number of nearest neighbors
    f_run: int = 1  # Running flag
    eps: float = 0  # Epsilon value
    iRad: int = 0  # Radius flag
    iprep: int = 0  # Preprocessing flag
    Nstep: int = 1  # Number of steps
    alpha: float = 0  # Alpha parameter

@dataclass
class LineParams:
    """
    Set a series of emission line with name, wavelength/A and ratio in the given file as one SED model.
    """
    igroup: int  # Group ID
    id: int  # Model ID
    name: str  # Name of the emission line model
    file: str  # File containing emission line data
    R: float  # Ratio parameter
    Nkin: int  # Number of kinematic components
    iscalable: int  # Whether the model is scalable (0: No, 1: Yes)
    Nsample: int = 2  # Number of samples

@dataclass
class LuminosityParams:
    """
    Compute luminosity between w_min and w_max in rest-frame for model with given id(-1 for all).
    """
    id: int  # Model ID (-1 for all models)
    w_min: float  # Minimum wavelength
    w_max: float  # Maximum wavelength

@dataclass
class NPSFHParams:
    """
    Set the prior type(0-7), interpolation method (0-3), number of bins and regul for the nonparametric SFH.
    """
    prior_type: int = 5  # Prior type (0-7)
    interpolation_method: int = 0  # Interpolation method (0-3)
    num_bins: int = 10  # Number of bins
    regul: float = 100.0  # Regularization parameter

@dataclass
class PolynomialParams:
    """
    Multiplicative polynomial of order n.
    """
    order: int  # Order of the polynomial

@dataclass
class PowerlawParams:
    """
    Power law spectrum.
    """
    igroup: int  # Group ID
    id: int  # Model ID
    pw: str  # Power law model name
    iscalable: int  # Whether the model is scalable (0: No, 1: Yes)
    w_min: float  # Minimum wavelength
    w_max: float  # Maximum wavelength
    Nw: int  # Number of wavelength points

@dataclass
class RBFParams:
    """
    Select rbf model by name.
    """
    igroup: int  # Group ID
    id: int  # Model ID
    name: str  # Name of the RBF model
    iscalable: int  # Whether the model is scalable (0: No, 1: Yes)

@dataclass
class SFHParams:
    """
    Parameters for Star Formation History.
    """
    itype_sfh: int  # SFH type (0-9)
    id: int  # Model ID
    itruncated: int = 0  # Truncation flag (0: No, 1: Yes)
    itype_ceh: int = 0  # Chemical evolution history type

@dataclass
class SSPParams:
    """
    Parameters for SSP (Simple Stellar Population) model.
    """
    name: str  # Name of the SSP model
    igroup: int  # Group ID
    id: int  # Model ID
    iscalable: int  # Whether the model is scalable (0: No, 1: Yes)
    k: int = 1  # Number of components
    f_run: int = 1  # Running flag
    Nstep: int = 1  # Number of steps
    i0: int = 0  # Parameter i0
    i1: int = 0  # Parameter i1
    i2: int = 0  # Parameter i2
    i3: int = 0  # Parameter i3

@dataclass
class SEDLibParams:
    """
    Use SEDs from a sedlib with the given name.
    """
    igroup: int  # Group ID
    id: int  # Model ID
    name: str  # Name of the SED library
    iscalable: int  # Whether the model is scalable (0: No, 1: Yes)
    dir: str  # Directory containing the SED library
    itype: int  # Type of the SED library
    f_run: int  # Running flag
    ikey: int  # Key parameter

@dataclass
class SysErrParams:
    """
    Set the prior for the fractional systematic error of model or observation.
    """
    iprior_type: int = 1  # Prior type
    is_age: int = 0  # Age-dependent flag
    min: float = 0  # Minimum value
    max: float = 0.2  # Maximum value
    nbin: int = 40  # Number of bins

@dataclass
class ZParams:
    """
    Set the prior for the redshift z.
    """
    iprior_type: int = 1  # Prior type
    is_age: int = 0  # Age-dependent flag
    min: float = 0.0  # Minimum redshift
    max: float = 1.0  # Maximum redshift
    nbin: int = 40  # Number of bins

@dataclass
class NNLMParams:
    """
    The method, Niter1, tol1, Niter2, tol2, p1, p2 for the determination of nonnegative scale using NNLM.
    """
    method: int = 0  # Method (0=eazy, 1=scd, 2=lee_ls, 3=scd_kl, 4=lee_kl)
    Niter1: int = 10000  # Number of iterations for first step
    tol1: float = 0.0  # Tolerance for first step
    Niter2: int = 10  # Number of iterations for second step
    tol2: float = 0.01  # Tolerance for second step
    p1: float = 0.05  # Parameter p1
    p2: float = 0.95  # Parameter p2

@dataclass
class NdumperParams:
    """
    Set the max number, iconverged_min and Xmin^2/Nd for the dumper of multinest.
    """
    max_number: int = 1  # Maximum number
    iconverged_min: int = 0  # Minimum convergence flag
    Xmin_squared_Nd: float = -1.0  # Xmin^2/Nd value

@dataclass
class OutputSFHParams:
    """
    Output the SFH over the past tage year as derived_pars for ntimes and in ilog scale.
    """
    ntimes: int  # Number of time points
    ilog: int  # Log scale flag (0: linear, 1: log)

@dataclass
class MultiNestParams:
    """
    Parameters for MultiNest.
    """
    INS: int = 1  # Importance Nested Sampling flag
    mmodal: int = 0  # Multimodal flag
    ceff: int = 0  # Constant efficiency mode flag
    nlive: int = 100  # Number of live points
    efr: float = 0.1  # Sampling efficiency
    tol: float = 0.5  # Tolerance
    updInt: int = 1000  # Update interval
    Ztol: float = -1e90  # Evidence tolerance
    seed: int = 1  # Random seed
    fb: int = 0  # Feedback level
    resume: int = 0  # Resume from a previous run
    outfile: int = 0  # Write output files
    logZero: float = -1e90  # Log of Zero
    maxiter: int = 100000  # Maximum number of iterations
    acpt: float = 0.01  # Acceptance rate

@dataclass
class SFROverParams:
    past_Myr1: float = 10.0  # First past time in Myr
    past_Myr2: float = 100.0  # Second past time in Myr

@dataclass
class SNRmin1Params:
    phot: float = 0.0  # Minimum SNR for photometry
    spec: float = 0.0  # Minimum SNR for spectroscopy

@dataclass
class SNRmin2Params:
    phot: float = 0.0  # Minimum SNR for photometry
    spec: float = 0.0  # Minimum SNR for spectroscopy

@dataclass
class GSLIntegrationQAGParams:
    epsabs: float = 0.0  # Absolute error
    epsrel: float = 0.1  # Relative error
    limit: int = 1000  # Limit on the number of subintervals
    key: int = 1  # Key for the integration rule

@dataclass
class GSLMultifitRobustParams:
    type: str = "ols"  # Type of robust fitting (ols or huber)
    tune: float = 1.0  # Tuning parameter

@dataclass
class KinParams:
    id: int  # Model ID
    velscale: int = 10  # Velocity scale
    num_gauss_hermites_con: int = 0  # Number of Gauss-Hermite terms for continuum
    num_gauss_hermites_eml: int = 0  # Number of Gauss-Hermite terms for emission lines

@dataclass
class LineListParams:
    file: str  # File containing the line list
    type: int  # Type of line list (0: intrinsic, 1: emergent, 2: intrinsic cumulative, 3: emergent cumulative)

@dataclass
class MakeCatalogParams:
    id1: int  # ID of the first model
    logscale_min1: float  # Minimum log scale for the first model
    logscale_max1: float  # Maximum log scale for the first model
    id2: int  # ID of the second model
    logscale_min2: float  # Minimum log scale for the second model
    logscale_max2: float  # Maximum log scale for the second model

@dataclass
class CloudyParams:
    igroup: int  # Group ID
    id: int  # Model ID
    cloudy: str  # Cloudy model name
    iscalable: int  # Whether the model is scalable (0: No, 1: Yes)

@dataclass
class CosmologyParams:
    H0: float  # Hubble constant
    omigaA: float  # Omega Lambda
    omigam: float  # Omega Matter

@dataclass
class DALParams:
    ilaw: int  # Dust attenuation law type
    id: int  # Model ID
    con_eml_tot: int = 2  # Continuum, emission lines, or total (0: continuum, 1: emission, 2: total)

@dataclass
class RDFParams:
    id: int  # Model ID
    num_polynomials: int  # Number of polynomials for modeling sigma_diff

@dataclass
class TemplateParams:
    igroup: int  # Group ID
    id: int  # Model ID
    name: str  # Name of the template
    iscalable: int  # Whether the model is scalable (0: No, 1: Yes)

@dataclass
class RenameParams:
    id: int  # Model ID
    ireplace: int  # Replace flag (0: add suffix, 1: replace name)
    name: str  # New name or suffix

