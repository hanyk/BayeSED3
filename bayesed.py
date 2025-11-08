import argparse
import os
import requests
from tqdm import tqdm
import tarfile
import shutil
import platform
import subprocess
import multiprocessing
import sys
from dataclasses import dataclass, field
from typing import List, Optional, Union
import numpy as np
from astropy.table import Table
from astropy.io import ascii

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

@dataclass
class BayeSEDParams:
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

class BayeSEDInterface:
    def __init__(self, mpi_mode='1', openmpi_mirror=None, np=None, Ntest=None, output_queue=None):
        self.mpi_mode = f"mn_{mpi_mode}"
        self.openmpi_mirror = openmpi_mirror
        self.np = np
        self.Ntest = Ntest
        self._get_system_info()
        self.mpi_cmd = self._setup_openmpi()
        self.num_processes = self._get_max_threads() if np is None else np
        self.executable_path = self._get_executable()

    def _get_system_info(self):
        self.os = platform.system().lower()
        self.arch = platform.machine().lower()

    def _get_max_threads(self):
        return multiprocessing.cpu_count()

    def _setup_openmpi(self):
        openmpi_version = "4.1.6"
        openmpi_url = self.openmpi_mirror or f"https://download.open-mpi.org/release/open-mpi/v4.1/openmpi-{openmpi_version}.tar.gz"
        openmpi_dir = f"openmpi-{openmpi_version}"
        openmpi_file = f"{openmpi_dir}.tar.gz"
        install_dir = os.path.abspath("openmpi")

        # Check if the correct version of OpenMPI is already installed on the system
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

        # If the correct version of OpenMPI is not found, proceed with the installation
        if not os.path.exists(install_dir):
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

        return mpirun_path

    def _get_executable(self):
        base_path = "./bin"
        executable = f"bayesed_{self.mpi_mode}"
        if self.os == "linux" or (self.os == "windows" and "microsoft" in platform.uname().release.lower()):
            platform_dir = "linux"
        elif self.os == "darwin":
            platform_dir = "mac"
        else:
            raise ValueError(f"Unsupported operating system: {self.os}")

        executable_path = os.path.join(base_path, platform_dir, executable)
        if not os.path.exists(executable_path):
            raise FileNotFoundError(f"Executable not found: {executable_path}")

        return executable_path

    def run(self, params):
        if isinstance(params, list):
            args = params
        else:
            args = self._params_to_args(params)

        # Set TMPDIR environment variable
        os.environ['TMPDIR'] = '/tmp'

        cmd = [self.mpi_cmd, '--use-hwthread-cpus', '-np', str(self.num_processes), self.executable_path] + args

        # Add Ntest if specified
        if self.Ntest is not None:
            cmd.extend(['--Ntest', str(self.Ntest)])

        print(f"Executing command: {' '.join(cmd)}")

        try:
            self.process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True, bufsize=1)

            for line in iter(self.process.stdout.readline, ''):
                print(line, end='', flush=True)

            self.process.wait()

            if self.process.returncode == 0:
                print("BayeSED execution completed\n", flush=True)
            else:
                print(f"BayeSED execution failed, return code: {self.process.returncode}\n", flush=True)

        except Exception as e:
            print(f"Error: {str(e)}\n", flush=True)

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

def create_input_catalog(
    output_path: str,
    catalog_name: str,
    ids,
    z_min,
    z_max,
    distance_mpc: Optional[np.ndarray] = None,
    ebv: Optional[np.ndarray] = None,
    phot_band_names: Optional[List[str]] = None,
    phot_fluxes: Optional[np.ndarray] = None,
    phot_errors: Optional[np.ndarray] = None,
    phot_type: str = "fnu",
    phot_flux_limits: Optional[np.ndarray] = None,
    phot_mag_limits: Optional[np.ndarray] = None,
    phot_nsigma: Optional[float] = None,
    other_columns: Optional[dict] = None,
    spec_band_names: Optional[List[str]] = None,
    spec_wavelengths: Optional[dict] = None,
    spec_fluxes: Optional[dict] = None,
    spec_errors: Optional[dict] = None,
    spec_lsf_sigma: Optional[dict] = None,
    spec_flux_type: str = "fnu",
):
    """
    Create a input catalog file for bayesed3.

    Required base columns (length N):
    - ID, z_min, z_max; d/Mpc and E(B-V) default to 0 if not provided

    Photometry (optional):
    - phot_band_names: list of band names like ["G","R","Z",...]
    - phot_type: "fnu" (μJy) or "abmag" (AB mags)
      * If phot_type == "fnu": phot_fluxes/phot_errors are in μJy
      * If phot_type == "abmag": phot_fluxes/phot_errors are AB magnitudes and mag errors
        - Converted to μJy via f_μJy = 10^((23.9 - m_AB)/2.5)
        - σ_f = f_μJy * ln(10)/2.5 * σ_m
    - phot_fluxes: shape (N, Nphot)
    - phot_errors: shape (N, Nphot)
    - Nondetections (optional, recommended):
      * If phot_type == "fnu" and S/N < 1 in a band, represent as F = Flim and s = -Flim/N
        Provide per-band limits in phot_flux_limits (μJy) and N in phot_nsigma
      * If phot_type == "abmag" and σm > 1.08574, represent as m = mlim and σm = -1.08574/N
        Provide per-band limits in phot_mag_limits (AB mag) and N in phot_nsigma

    Other columns (optional):
    - other_columns: dict {column_name: np.ndarray (length N)}

    Spectra (optional), per band (e.g. B,R,Z):
    - spec_band_names: list of band names, e.g. ["B","R","Z"]
    - spec_wavelengths: dict band -> array (N, Nw) in Angstrom
    - spec_fluxes/spec_errors: either F_lambda or f_nu depending on spec_flux_type
      * When spec_flux_type == "flambda": arrays must be in F_lambda (erg s^-1 cm^-2 Å^-1)
      * When spec_flux_type == "fnu": arrays must be in microJy (μJy)
    - spec_lsf_sigma: dict band -> array (N, Nw) of LSF sigma in microns (as in convert.py)

    File format details (must match readers):
    - Header comment: "<catalog_name> <Nphot> <Nother> <Nspec> 0"
    - Wavelength columns stored in microns
    - Spectral flux and error stored in microJy (converted from F_lambda by F_nu = F_lambda * lambda^2 / c)
    - LSF sigma columns stored in microns

    Parameters are minimally validated; the caller is responsible for consistent shapes.
    
    Returns
    -------
    list of str
        List of band names (photometry and spectral bands combined).
        Empty list if no bands are provided.
    """

    ids = np.asarray(ids)
    z_min = np.asarray(z_min)
    z_max = np.asarray(z_max)
    # Defaults: zeros if not provided
    if distance_mpc is None:
        distance_mpc = np.zeros_like(z_min, dtype=float)
    else:
        distance_mpc = np.asarray(distance_mpc)
    if ebv is None:
        ebv = np.zeros_like(z_min, dtype=float)
    else:
        ebv = np.asarray(ebv)

    N = len(ids)
    if not (len(z_min) == len(z_max) == len(distance_mpc) == len(ebv) == N):
        raise ValueError("Base columns must all have equal length N")

    names: List[str] = []
    cols: List[np.ndarray] = []

    # Base columns (ordered)
    names.extend(["ID", "z_min", "z_max", "d/Mpc", "E(B-V)"])
    cols.extend([ids, z_min, z_max, distance_mpc, ebv])

    # Photometry
    Nphot = 0
    if phot_band_names is not None and len(phot_band_names) > 0:
        phot_fluxes = np.asarray(phot_fluxes)
        phot_errors = np.asarray(phot_errors)
        if phot_fluxes.shape != phot_errors.shape:
            raise ValueError("phot_fluxes and phot_errors must have the same shape (N, Nphot)")
        if phot_fluxes.shape[0] != N:
            raise ValueError("photometry arrays must have length N in the first dimension")
        Nphot = phot_fluxes.shape[1]
        if Nphot != len(phot_band_names):
            raise ValueError("len(phot_band_names) must equal phot_fluxes.shape[1]")
        pt = phot_type.lower()
        if pt not in ("fnu", "abmag"):
            raise ValueError("phot_type must be 'fnu' or 'abmag'")
        # Apply nondetection conventions prior to unit conversion
        if pt == "fnu" and phot_flux_limits is not None and phot_nsigma is not None:
            lim = np.asarray(phot_flux_limits)
            if lim.shape != phot_fluxes.shape:
                if lim.ndim == 1 and lim.shape[0] == Nphot:
                    lim = np.broadcast_to(lim, phot_fluxes.shape)
                else:
                    raise ValueError("phot_flux_limits must be shape (N,Nphot) or (Nphot,)")
            Nsig = float(phot_nsigma)
            snr = np.divide(phot_fluxes, phot_errors, out=np.zeros_like(phot_fluxes, dtype=float), where=phot_errors!=0)
            mask_nd = snr < 1.0
            # F = Flim; s = -Flim/N
            phot_fluxes = phot_fluxes.copy()
            phot_errors = phot_errors.copy()
            phot_fluxes[mask_nd] = lim[mask_nd]
            phot_errors[mask_nd] = -lim[mask_nd] / Nsig
        elif pt == "abmag" and phot_mag_limits is not None and phot_nsigma is not None:
            limm = np.asarray(phot_mag_limits)
            if limm.shape != phot_fluxes.shape:
                if limm.ndim == 1 and limm.shape[0] == Nphot:
                    limm = np.broadcast_to(limm, phot_fluxes.shape)
                else:
                    raise ValueError("phot_mag_limits must be shape (N,Nphot) or (Nphot,)")
            Nsig = float(phot_nsigma)
            # nondetection if sigma_m > 1.08574
            mask_nd = phot_errors > 1.08574
            # m = mlim; sigma_m = -1.08574/N
            phot_fluxes = phot_fluxes.copy()
            phot_errors = phot_errors.copy()
            phot_fluxes[mask_nd] = limm[mask_nd]
            phot_errors[mask_nd] = -1.08574 / Nsig

        if pt == "abmag":
            # Convert AB mag to μJy: f_μJy = 10^((23.9 - m)/2.5)
            f_microjy = 10.0 ** ((23.9 - phot_fluxes) / 2.5)
            # Error propagation: σ_f = f * ln(10)/2.5 * σ_m
            k = np.log(10.0) / 2.5
            e_microjy = f_microjy * k * phot_errors
        else:
            f_microjy = phot_fluxes
            e_microjy = phot_errors
        for j, band in enumerate(phot_band_names):
            names.append(f"f_{band}")
            cols.append(f_microjy[:, j])
            names.append(f"e_{band}")
            cols.append(e_microjy[:, j])

    # Other columns
    Nother = 0
    if other_columns:
        for col_name, values in other_columns.items():
            values = np.asarray(values)
            if len(values) != N:
                raise ValueError(f"other column '{col_name}' must have length N")
            names.append(col_name)
            cols.append(values)
        Nother = len(other_columns)

    # Spectra
    Nspec = 0
    if spec_band_names:
        Nspec = len(spec_band_names)
        # Add Nw_<band> columns first
        for band in spec_band_names:
            wl = np.asarray(spec_wavelengths[band])  # (N, Nw) in Angstrom
            if wl.shape[0] != N:
                raise ValueError(f"spec_wavelengths['{band}'] must have first dimension N")
            names.append(f"Nw_{band}")
            cols.append(np.full(N, wl.shape[1], dtype=int))

        # Then add per-pixel columns: wavelength (micron), flux (microJy), error (microJy), lsf sigma (micron)
        c_ang_per_s = 2.9979246e+18  # Angstrom/s (must match tools/bayesed2bagpipes.py)
        use_fnu = spec_flux_type.lower() == "fnu"
        for band in spec_band_names:
            wl_ang = np.asarray(spec_wavelengths[band])           # (N, Nw)
            f_arr = np.asarray(spec_fluxes[band])                  # (N, Nw)
            e_arr = np.asarray(spec_errors[band])                  # (N, Nw)
            if spec_lsf_sigma is None or band not in spec_lsf_sigma:
                raise ValueError(f"spec_lsf_sigma['{band}'] is required and must be in microns")
            s_micron = np.asarray(spec_lsf_sigma[band])            # (N, Nw) in microns

            if not (wl_ang.shape == f_arr.shape == e_arr.shape == s_micron.shape):
                raise ValueError(f"All spectral arrays for band '{band}' must have identical shape (N, Nw)")

            wl_micron = wl_ang * 1e-4  # store wavelength in microns
            if use_fnu:
                # Already microJy
                f_microjy = f_arr
                e_microjy = e_arr
            else:
                # Convert F_lambda (erg/s/cm^2/Å) to microJy per pixel
                # F_nu (erg/s/cm^2/Hz) = F_lambda * lambda^2 / c
                # 1 microJy = 1e-29 erg/s/cm^2/Hz
                f_microjy = (f_arr * (wl_ang ** 2) / c_ang_per_s) / 1e-29
                e_microjy = (e_arr * (wl_ang ** 2) / c_ang_per_s) / 1e-29

            Nw = wl_ang.shape[1]
            for w in range(Nw):
                names.append(f"w_{band}{w}")
                cols.append(wl_micron[:, w])
                names.append(f"f_{band}{w}")
                cols.append(f_microjy[:, w])
                names.append(f"e_{band}{w}")
                cols.append(e_microjy[:, w])
                names.append(f"s_{band}{w}")
                cols.append(s_micron[:, w])

    # Construct the table in one shot to avoid repeated reallocation
    table = Table(cols, names=names)
    table.meta['comments'] = [f"{catalog_name} {Nphot} {Nother} {Nspec} 0"]

    # Write ASCII with the same options as convert.py
    ascii.write(
        table,
        output_path,
        overwrite=True,
        fast_writer='force',
        strip_whitespace=False,
        fill_values=[(ascii.masked, '-999'), ('nan', '-999'), ('inf', '-999')]
    )
    
    # Collect and return band names
    band_names = []
    if phot_band_names is not None:
        band_names.extend(phot_band_names)
    if spec_band_names is not None:
        band_names.extend(spec_band_names)
    
    return band_names

    def _format_fann_params(self, fann_params):
        return f"{fann_params.igroup},{fann_params.id},{fann_params.name},{fann_params.iscalable}"

    def _format_AGN_params(self, AGN_params):
        return f"{AGN_params.igroup},{AGN_params.id},{AGN_params.name},{AGN_params.iscalable}"

    def _format_blackbody_params(self, blackbody_params):
        return f"{blackbody_params.igroup},{blackbody_params.id},{blackbody_params.name},{blackbody_params.iscalable}"

    def _format_big_blue_bump_params(self, big_blue_bump_params):
        return f"{big_blue_bump_params.igroup},{big_blue_bump_params.id},{big_blue_bump_params.name},{big_blue_bump_params.iscalable},{big_blue_bump_params.w_min},{big_blue_bump_params.w_max},{big_blue_bump_params.Nw}"

    def _format_greybody_params(self, greybody_params):
        return f"{greybody_params.igroup},{greybody_params.id},{greybody_params.name},{greybody_params.iscalable},{greybody_params.ithick},{greybody_params.w_min},{greybody_params.w_max},{greybody_params.Nw}"

    def _format_aknn_params(self, aknn_params):
        return f"{aknn_params.igroup},{aknn_params.id},{aknn_params.name},{aknn_params.iscalable},{aknn_params.k},{aknn_params.f_run},{aknn_params.eps},{aknn_params.iRad},{aknn_params.iprep},{aknn_params.Nstep},{aknn_params.alpha}"

    def _format_line_params(self, line_params):
        return f"{line_params.igroup},{line_params.id},{line_params.name},{line_params.iscalable},{line_params.file},{line_params.R},{line_params.Nsample},{line_params.Nkin}"

    def _format_luminosity_params(self, luminosity_params):
        return f"{luminosity_params.igroup},{luminosity_params.id},{luminosity_params.name},{luminosity_params.iscalable}"

    def _format_np_sfh_params(self, np_sfh_params):
        return f"{np_sfh_params.igroup},{np_sfh_params.id},{np_sfh_params.name},{np_sfh_params.iscalable}"

    def _format_polynomial_params(self, polynomial_params):
        return f"{polynomial_params.igroup},{polynomial_params.id},{polynomial_params.name},{polynomial_params.iscalable}"

    def _format_powerlaw_params(self, powerlaw_params):
        return f"{powerlaw_params.igroup},{powerlaw_params.id},{powerlaw_params.name},{powerlaw_params.iscalable}"

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
        return f"{gsl_integration_qag_params.epsabs},{gsl_integration_qag_params.epsrel},{gsl_integration_qag_params.limit},{gsl_integration_qag_params.key}"

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


# Filter transmission type constants
FILTER_TYPE_ENERGY = 0  # Filter response in energy units
FILTER_TYPE_PHOTON = 1  # Filter response in photon-counting units (default)

# Filter calibration scheme constants (LEPHARE convention)
FILTER_CALIB_STANDARD = 0  # Standard calibration (B_ν = constant, default)
FILTER_CALIB_SPITZER_IRAC = 1  # SPITZER/IRAC and ISO calibration (ν B_ν = constant)
FILTER_CALIB_SUBMM = 2  # Sub-mm calibration (B_ν = ν)
FILTER_CALIB_BLACKBODY = 3  # Blackbody calibration (T = 10,000 K)
FILTER_CALIB_SPITZER_MIPS = 4  # SPITZER/MIPS mixed calibration
FILTER_CALIB_SCUBA = 5  # SCUBA mixed calibration

# Documentation strings for filter types and calibration schemes
FILTER_TYPE_NAMES = {
    FILTER_TYPE_ENERGY: "Energy",
    FILTER_TYPE_PHOTON: "Photon"
}

FILTER_CALIB_NAMES = {
    FILTER_CALIB_STANDARD: "Standard",
    FILTER_CALIB_SPITZER_IRAC: "SPITZER/IRAC",
    FILTER_CALIB_SUBMM: "Sub-mm",
    FILTER_CALIB_BLACKBODY: "Blackbody",
    FILTER_CALIB_SPITZER_MIPS: "SPITZER/MIPS",
    FILTER_CALIB_SCUBA: "SCUBA"
}


def infer_filter_itype_icalib(filter_metadata: dict, filter_id: str = ""):
    """
    Infer itype and icalib from SVO filter metadata.
    
    This function attempts to automatically determine the filter transmission type
    (itype) and calibration scheme (icalib) based on filter metadata from astroquery SvoFps.
    Uses Facility, Instrument, filterID, name, and DetectorType attributes.
    
    Parameters
    ----------
    filter_metadata : dict
        Dictionary containing filter metadata from SvoFps (e.g., from filter index table row)
    filter_id : str, optional
        SVO filter ID (used as fallback for metadata extraction)
    
    Returns
    -------
    tuple
        (itype, icalib) where:
        - itype: 0 (Energy) or 1 (Photon, default)
        - icalib: 0-5 (calibration scheme, default 0)
    
    Notes
    -----
    Calibration scheme inference (icalib):
    - 1: SPITZER/IRAC, PACS, SPIRE, IRAS (ν B_ν = constant)
    - 4: SPITZER/MIPS (mixed calibration)
    - 5: SCUBA (mixed calibration)
    - 0: Standard (default, B_ν = constant)
    
    Transmission type inference (itype) from DetectorType:
    DetectorType values and their corresponding itype:
    - "CCD" (Charge-Coupled Device): Photon-counting → itype=1
    - "Photomultiplier": Photon-counting → itype=1
    - "Array": Usually detector arrays (CCD/CMOS) → itype=1
    - "Bolometer": Energy-based (measures power/energy) → itype=0
    - "Calorimeter": Energy-based → itype=0
    - "Energy": Explicitly energy-based → itype=0
    - "Photon": Explicitly photon-counting → itype=1
    - Default: 1 (Photon) for most optical/NIR detectors
    """
    # Default values
    itype = FILTER_TYPE_PHOTON  # Default to photon-counting
    icalib = FILTER_CALIB_STANDARD  # Default to standard calibration
    
    # Collect metadata strings for pattern matching
    metadata_strings = []
    detector_type = None
    
    # Get Facility
    if 'Facility' in filter_metadata and filter_metadata['Facility']:
        facility = str(filter_metadata['Facility']).upper()
        metadata_strings.append(facility)
    
    # Get Instrument
    if 'Instrument' in filter_metadata and filter_metadata['Instrument']:
        instrument = str(filter_metadata['Instrument']).upper()
        metadata_strings.append(instrument)
    
    # Get filterID (contains facility/instrument info)
    if 'filterID' in filter_metadata and filter_metadata['filterID']:
        filter_id = str(filter_metadata['filterID']).upper()
        metadata_strings.append(filter_id)
    
    # Get name (might be in PhotSystem or filterID)
    if 'PhotSystem' in filter_metadata and filter_metadata['PhotSystem']:
        name = str(filter_metadata['PhotSystem']).upper()
        metadata_strings.append(name)
    
    # Get DetectorType (if available)
    if 'DetectorType' in filter_metadata and filter_metadata['DetectorType']:
        detector_type = str(filter_metadata['DetectorType']).upper()
        metadata_strings.append(detector_type)
    
    # Also check original filter ID
    if filter_id:
        metadata_strings.append(filter_id.upper())
    
    # Combine all metadata for pattern matching
    combined_metadata = " ".join(metadata_strings)
    
    # Infer icalib from facility/instrument patterns
    # Based on filters/cigale/convert script logic
    if any(keyword in combined_metadata for keyword in ['IRAC', 'PACS', 'SPIRE', 'IRAS']):
        icalib = FILTER_CALIB_SPITZER_IRAC  # 1
    elif 'MIPS' in combined_metadata:
        icalib = FILTER_CALIB_SPITZER_MIPS  # 4
    elif 'SCUBA' in combined_metadata:
        icalib = FILTER_CALIB_SCUBA  # 5
    # Note: Sub-mm (2) and Blackbody (3) are harder to infer automatically
    # They would need additional metadata or manual specification
    
    # Infer itype from detector type and metadata
    # DetectorType values and their meanings:
    # - "CCD" (Charge-Coupled Device): Photon-counting detector → itype=1
    # - "Photomultiplier": Photon-counting detector → itype=1
    # - "Array": Usually refers to detector arrays (CCD/CMOS) → itype=1
    # - "Bolometer": Energy-based detector (measures power/energy) → itype=0
    # - "Calorimeter": Energy-based detector → itype=0
    # - "Energy": Explicitly energy-based → itype=0
    # - "Photon": Explicitly photon-counting → itype=1
    # Most optical/NIR detectors are photon-counting (default itype=1)
    
    if detector_type:
        # Check for explicit energy-based detector indicators
        energy_keywords = ['BOLOMETER', 'CALORIMETER', 'ENERGY']
        if any(keyword in detector_type for keyword in energy_keywords):
            itype = FILTER_TYPE_ENERGY  # 0
        
        # Check for explicit photon-counting indicators (though default is already photon)
        photon_keywords = ['PHOTON', 'CCD', 'PHOTOMULTIPLIER', 'ARRAY', 'CMOS']
        # Note: Most detectors are photon-counting, so this is mainly for explicit cases
        # Default itype=1 (Photon) already covers CCD, Photomultiplier, Array, etc.
    
    # Additional pattern matching for itype (if DetectorType not available or ambiguous)
    # Some instruments/facilities might have known energy-based responses
    # This is rare, so we keep photon as default
    
    return itype, icalib


def create_filters_from_svo(
    svo_filterIDs: List[str],
    filters_file: str,
    filters_selected_file: Optional[str] = None,
    itype: Optional[int] = None,
    icalib: Optional[int] = None,
    itype_per_filter: Optional[List[int]] = None,
    icalib_per_filter: Optional[List[int]] = None,
    auto_infer: bool = True,
    wave_units: str = "micron",
    selected_indices: Optional[List[int]] = None,
    filter_names: Optional[List[str]] = None,
    **filters_selected_kwargs
):
    """
    Create filter files and filters_selected file from SVO filter IDs using astroquery SvoFps.
    
    This function:
    1. Loads filters from the SVO Filter Profile Service using astroquery SvoFps
    2. Automatically infers itype and icalib from filter metadata (if auto_infer=True)
    3. Creates a filters file with transmission curves
    4. Creates a corresponding filters_selected file
    
    Parameters
    ----------
    svo_filterIDs : list of str
        List of SVO filter IDs in format 'Facility/FilterName' (e.g., 
        ['SLOAN/SDSS.u', 'SLOAN/SDSS.g', 'SLOAN/SDSS.r', '2MASS/2MASS.H']).
        These must match the filterID format required by SVO Filter Profile Service.
        Use SvoFps.get_filter_index() to list available filter IDs.
    filters_file : str
        Path to output filters file. This will contain the filter transmission curves.
    filters_selected_file : str, optional
        Path to output filters_selected file. If None, will be generated from filters_file
        by replacing '.txt' with '_selected.txt' or appending '_selected.txt'.
    itype : int, optional
        Filter transmission type for all filters. If None and auto_infer=True, will be
        inferred from filter metadata. If None and auto_infer=False, defaults to 1 (Photon).
        - 0 = Energy (filter response in energy units)
        - 1 = Photon (filter response in photon-counting units, default)
    icalib : int, optional
        Filter calibration scheme for all filters. If None and auto_infer=True, will be
        inferred from filter metadata. If None and auto_infer=False, defaults to 0 (Standard).
        - 0 = Standard calibration (B_ν = constant, default)
        - 1 = SPITZER/IRAC and ISO calibration (ν B_ν = constant)
        - 2 = Sub-mm calibration (B_ν = ν)
        - 3 = Blackbody calibration (T = 10,000 K)
        - 4 = SPITZER/MIPS mixed calibration
        - 5 = SCUBA mixed calibration
    itype_per_filter : list of int, optional
        Per-filter itype values. If provided, must have same length as svo_filterIDs.
        Overrides itype for individual filters. Use None in list to auto-infer that filter.
    icalib_per_filter : list of int, optional
        Per-filter icalib values. If provided, must have same length as svo_filterIDs.
        Overrides icalib for individual filters. Use None in list to auto-infer that filter.
    auto_infer : bool, optional
        If True (default), automatically infer itype and icalib from filter metadata
        when not explicitly provided. Uses infer_filter_itype_icalib() function.
    wave_units : str, optional
        Wavelength units for output file. Options: "micron", "angstrom" (default: "micron").
        BayeSED typically uses microns.
    selected_indices : list of int, optional
        If provided, only filters at these indices (0-based) will be included in
        filters_selected file. If None (default), all filters are selected.
    filter_names : list of str, optional
        List of custom filter names corresponding to selected_indices.
        Must have the same length as selected_indices if provided.
        If None, filter names will be extracted from SVO filter IDs.
    **filters_selected_kwargs
        Additional keyword arguments passed to create_filters_selected().
        See create_filters_selected() documentation for available options.
    
    Returns
    -------
    tuple
        (filters_file, filters_selected_file) paths
    
    Examples
    --------
    >>> # Create filters with automatic inference of itype and icalib
    >>> create_filters_from_svo(
    ...     ['SLOAN/SDSS.u', 'SLOAN/SDSS.g', 'SLOAN/SDSS.r', 'SLOAN/SDSS.i', 'SLOAN/SDSS.z'],
    ...     'observation/test1/filters_SDSS.txt',
    ...     'observation/test1/filters_SDSS_selected.txt',
    ...     auto_infer=True  # Automatically infer from metadata
    ... )
    
    >>> # Create filters with manual itype/icalib
    >>> create_filters_from_svo(
    ...     ['SLOAN/SDSS.u', 'SLOAN/SDSS.g', 'SLOAN/SDSS.r'],
    ...     'filters.txt',
    ...     itype=1,  # All filters use photon-counting
    ...     icalib=0  # All filters use standard calibration
    ... )
    
    >>> # Create filters with per-filter values
    >>> create_filters_from_svo(
    ...     ['SLOAN/SDSS.u', 'Spitzer/IRAC.I1', 'Spitzer/MIPS.24mu'],
    ...     'filters.txt',
    ...     itype_per_filter=[1, None, None],  # Auto-infer IRAC and MIPS
    ...     icalib_per_filter=[0, None, None]  # Auto-infer IRAC and MIPS
    ... )
    
    >>> # List available filters first (using astroquery)
    >>> from astroquery.svo_fps import SvoFps
    >>> import astropy.units as u
    >>> filter_index = SvoFps.get_filter_index(0*u.AA, 100000*u.AA)
    >>> print(filter_index['filterID'][:10])  # Show first 10 filter IDs
    
    Notes
    -----
    - Requires astroquery package: pip install astroquery
    - Filter transmission curves are saved in two-column format: wavelength transmission
    - Wavelengths are converted to microns by default (BayeSED convention)
    - The filters file format matches BayeSED's expected format:
      * Lines starting with '#' define filters: "# itype icalib description"
      * Following lines contain wavelength and transmission pairs
    - Automatic inference uses filter Facility, Instrument, and filterID metadata
    - IRAC, PACS, SPIRE, IRAS filters → icalib=1
    - MIPS filters → icalib=4
    - SCUBA filters → icalib=5
    - Other filters → icalib=0 (Standard)
    """
    try:
        from astroquery.svo_fps import SvoFps
        import astropy.units as u
        import numpy as np
    except ImportError:
        raise ImportError(
            "astroquery package is required. Install with: pip install astroquery"
        )
    
    if not svo_filterIDs:
        raise ValueError("svo_filterIDs must contain at least one filter ID")
    
    # Validate per-filter lists if provided
    if itype_per_filter is not None:
        if len(itype_per_filter) != len(svo_filterIDs):
            raise ValueError(
                f"itype_per_filter (length {len(itype_per_filter)}) must have "
                f"the same length as svo_filterIDs (length {len(svo_filterIDs)})"
            )
        # Validate values
        for i, val in enumerate(itype_per_filter):
            if val is not None and val not in (FILTER_TYPE_ENERGY, FILTER_TYPE_PHOTON):
                raise ValueError(f"itype_per_filter[{i}] must be 0 (Energy) or 1 (Photon), got {val}")
    
    if icalib_per_filter is not None:
        if len(icalib_per_filter) != len(svo_filterIDs):
            raise ValueError(
                f"icalib_per_filter (length {len(icalib_per_filter)}) must have "
                f"the same length as svo_filterIDs (length {len(svo_filterIDs)})"
            )
        # Validate values
        for i, val in enumerate(icalib_per_filter):
            if val is not None and val not in range(6):
                raise ValueError(f"icalib_per_filter[{i}] must be 0-5, got {val}")
    
    # Validate global itype and icalib if provided
    if itype is not None and itype not in (FILTER_TYPE_ENERGY, FILTER_TYPE_PHOTON):
        raise ValueError(f"itype must be 0 (Energy) or 1 (Photon), got {itype}")
    if icalib is not None and icalib not in range(6):
        raise ValueError(f"icalib must be 0-5, got {icalib}")
    
    # Validate filter_names if provided
    if filter_names is not None:
        if selected_indices is None:
            raise ValueError("filter_names can only be used when selected_indices is provided")
        if len(filter_names) != len(selected_indices):
            raise ValueError(
                f"filter_names (length {len(filter_names)}) must have the same length "
                f"as selected_indices (length {len(selected_indices)})"
            )
    
    # Load filters from SVO using astroquery
    print(f"Loading {len(svo_filterIDs)} filter(s) from SVO Filter Profile Service...")
    
    filter_transmission_data = []  # Store transmission data tables
    filter_metadata_list = []  # Store metadata dictionaries
    filter_descriptions = []
    filter_itypes = []
    filter_icalibs = []
    filter_names_from_svo = []  # Collect filter names from SVO
    
    for i, filter_id in enumerate(svo_filterIDs):
        try:
            print(f"  Loading filter {i+1}/{len(svo_filterIDs)}: {filter_id}")
            
            # Get transmission data
            transmission_table = SvoFps.get_transmission_data(filter_id)
            
            # Get metadata from filter index
            filter_metadata = {
                'filterID': filter_id,
                'Facility': None,
                'Instrument': None,
                'Band': None,
                'WavelengthEff': None,
                'WavelengthMean': None,
                'WavelengthMin': None,
                'WavelengthMax': None,
                'WavelengthCen': None,
                'WavelengthPivot': None,
                'WavelengthPeak': None,
                'FWHM': None,
                'WidthEff': None,
                'DetectorType': None,
                'PhotSystem': None,
                'MagSys': None,
                'ZeroPoint': None,
                'ZeroPointUnit': None,
                'ZeroPointType': None,
                'ProfileReference': None,
                'CalibrationReference': None,
                'Description': None,
                'Comments': None
            }
            
            # Try to get metadata from filter index
            # Query index with a reasonable wavelength range based on transmission data
            try:
                # First get a rough wavelength range from transmission data
                wave_values = transmission_table['Wavelength'].value  # In Angstroms
                if len(wave_values) > 0:
                    wave_min_aa = float(np.min(wave_values))
                    wave_max_aa = float(np.max(wave_values))
                    # Query index with a slightly wider range to ensure we get the filter
                    wave_range_min = max(0, wave_min_aa - 1000) * u.AA
                    wave_range_max = (wave_max_aa + 1000) * u.AA
                    
                    # Query filter index for this wavelength range
                    filter_index_subset = SvoFps.get_filter_index(wave_range_min, wave_range_max)
                    
                    # Find matching filter in index
                    matching_filters = filter_index_subset[filter_index_subset['filterID'] == filter_id]
                    if len(matching_filters) > 0:
                        row = matching_filters[0]
                        # Extract all available metadata
                        for key in filter_metadata.keys():
                            if key in row.colnames:
                                value = row[key]
                                # Handle masked values
                                if hasattr(value, 'mask') and value.mask:
                                    filter_metadata[key] = None
                                else:
                                    filter_metadata[key] = value
            except Exception as e:
                # If index query fails, continue without metadata (will use fallback)
                pass
            
            # Fallback: Try to extract metadata from filter_id if not found in index
            if filter_metadata['Facility'] is None and '/' in filter_id:
                parts = filter_id.split('/')
                if len(parts) >= 2:
                    filter_metadata['Facility'] = parts[0]
                    if len(parts) >= 3:
                        filter_metadata['Instrument'] = parts[1]
            
            # Build comprehensive description from filter metadata
            # Includes: name, facility/instrument, wavelength, FWHM, detector type, photometric system
            desc_parts = []
            
            # Add filter name/ID (most important identifier)
            filter_name = filter_id.split('/')[-1] if '/' in filter_id else filter_id
            desc_parts.append(filter_name)
            
            # Add Facility and Instrument if available
            facility_instrument = []
            if filter_metadata['Facility'] and str(filter_metadata['Facility']).strip() not in ['-', '']:
                facility_instrument.append(str(filter_metadata['Facility']))
            if filter_metadata['Instrument'] and str(filter_metadata['Instrument']).strip() not in ['-', '']:
                facility_instrument.append(str(filter_metadata['Instrument']))
            if facility_instrument:
                desc_parts.append(f"({', '.join(facility_instrument)})")
            
            # Add wavelength information (prefer metadata from index, fallback to transmission data)
            wave_eff_um = None
            try:
                # First try metadata from index
                if filter_metadata['WavelengthEff'] is not None:
                    wave_eff_val = filter_metadata['WavelengthEff']
                    if hasattr(wave_eff_val, 'value'):
                        wave_eff_um = wave_eff_val.to(u.um).value if hasattr(wave_eff_val, 'to') else wave_eff_val.value * 1e-4
                    else:
                        wave_eff_um = float(wave_eff_val) * 1e-4  # Convert Angstroms to microns
                elif filter_metadata['WavelengthMean'] is not None:
                    wave_mean_val = filter_metadata['WavelengthMean']
                    if hasattr(wave_mean_val, 'value'):
                        wave_eff_um = wave_mean_val.to(u.um).value if hasattr(wave_mean_val, 'to') else wave_mean_val.value * 1e-4
                    else:
                        wave_eff_um = float(wave_mean_val) * 1e-4
                
                # Fallback to transmission data if metadata not available
                if wave_eff_um is None:
                    wave_values = transmission_table['Wavelength'].value  # In Angstroms
                    if len(wave_values) > 0:
                        wave_eff_aa = float(np.average(wave_values, weights=transmission_table['Transmission'].value))
                        wave_eff_um = wave_eff_aa * 1e-4
                
                if wave_eff_um is not None:
                    desc_parts.append(f"λ_eff={wave_eff_um:.3f}μm")
            except (AttributeError, ValueError, TypeError, KeyError):
                pass  # Skip if wavelength info not available
            
            # Add FWHM if available (prefer metadata from index)
            try:
                fwhm_um = None
                if filter_metadata['FWHM'] is not None:
                    fwhm_val = filter_metadata['FWHM']
                    if hasattr(fwhm_val, 'value'):
                        fwhm_um = fwhm_val.to(u.um).value if hasattr(fwhm_val, 'to') else fwhm_val.value * 1e-4
                    else:
                        fwhm_um = float(fwhm_val) * 1e-4  # Convert Angstroms to microns
                
                if fwhm_um is not None:
                    desc_parts.append(f"FWHM={fwhm_um:.3f}μm")
            except (ValueError, TypeError):
                pass
            
            # Add WidthEff if available and different from FWHM
            try:
                if filter_metadata['WidthEff'] is not None:
                    width_eff_val = filter_metadata['WidthEff']
                    if hasattr(width_eff_val, 'value'):
                        width_eff_um = width_eff_val.to(u.um).value if hasattr(width_eff_val, 'to') else width_eff_val.value * 1e-4
                    else:
                        width_eff_um = float(width_eff_val) * 1e-4
                    desc_parts.append(f"WidthEff={width_eff_um:.3f}μm")
            except (ValueError, TypeError):
                pass
            
            # Add DetectorType if available
            if filter_metadata['DetectorType'] and str(filter_metadata['DetectorType']).strip() not in ['-', '']:
                desc_parts.append(f"Det={filter_metadata['DetectorType']}")
            
            # Add PhotSystem or MagSys if available
            phot_system = None
            if filter_metadata['PhotSystem'] and str(filter_metadata['PhotSystem']).strip() not in ['-', '']:
                phot_system = str(filter_metadata['PhotSystem'])
            elif filter_metadata['MagSys'] and str(filter_metadata['MagSys']).strip() not in ['-', '']:
                phot_system = str(filter_metadata['MagSys'])
            if phot_system:
                desc_parts.append(f"Sys={phot_system}")
            
            # Add Band if available (more specific than just filter name)
            if filter_metadata['Band'] and str(filter_metadata['Band']).strip() not in ['-', '']:
                band = str(filter_metadata['Band']).strip()
                if band != filter_name:  # Only add if different from filter name
                    desc_parts.append(f"Band={band}")
            
            # Add ZeroPoint if available
            try:
                if filter_metadata['ZeroPoint'] is not None:
                    zp_val = filter_metadata['ZeroPoint']
                    zp_unit = filter_metadata.get('ZeroPointUnit', 'Jy')
                    if hasattr(zp_val, 'value'):
                        zp = zp_val.value
                    else:
                        zp = float(zp_val)
                    desc_parts.append(f"ZP={zp:.2f}{zp_unit}")
            except (ValueError, TypeError):
                pass
            
            # Add ZeroPointType if available
            if filter_metadata['ZeroPointType'] and str(filter_metadata['ZeroPointType']).strip() not in ['-', '']:
                zpt_type = str(filter_metadata['ZeroPointType'])
                if zpt_type.lower() not in ['pogson', '']:
                    desc_parts.append(f"ZPT={zpt_type}")
            
            # Add ProfileReference if available
            if filter_metadata['ProfileReference'] and str(filter_metadata['ProfileReference']).strip() not in ['-', '']:
                profile_ref = str(filter_metadata['ProfileReference']).strip()
                desc_parts.append(f"ProfileReference={profile_ref}")
            
            # Add CalibrationReference if available
            if filter_metadata['CalibrationReference'] and str(filter_metadata['CalibrationReference']).strip() not in ['-', '']:
                calib_ref = str(filter_metadata['CalibrationReference']).strip()
                desc_parts.append(f"CalibRef={calib_ref}")
            
            # Combine all parts into description
            desc = " ".join(desc_parts)
            
            # Collect filter name for filters_selected file
            filter_names_from_svo.append(filter_name)
            
            # Determine itype and icalib for this filter
            # Priority: per-filter list > global value > auto-inference > default
            if itype_per_filter is not None and itype_per_filter[i] is not None:
                filt_itype = itype_per_filter[i]
            elif itype is not None:
                filt_itype = itype
            elif auto_infer:
                filt_itype, _ = infer_filter_itype_icalib(filter_metadata, filter_id)
            else:
                filt_itype = FILTER_TYPE_PHOTON  # Default
            
            if icalib_per_filter is not None and icalib_per_filter[i] is not None:
                filt_icalib = icalib_per_filter[i]
            elif icalib is not None:
                filt_icalib = icalib
            elif auto_infer:
                _, filt_icalib = infer_filter_itype_icalib(filter_metadata, filter_id)
            else:
                filt_icalib = FILTER_CALIB_STANDARD  # Default
            
            filter_transmission_data.append(transmission_table)
            filter_metadata_list.append(filter_metadata)
            filter_descriptions.append(desc)
            filter_itypes.append(filt_itype)
            filter_icalibs.append(filt_icalib)
            
            # Print inferred values
            if auto_infer and (itype is None or icalib is None or 
                              (itype_per_filter is not None and itype_per_filter[i] is None) or
                              (icalib_per_filter is not None and icalib_per_filter[i] is None)):
                detector_info = ""
                if filter_metadata['DetectorType']:
                    detector_info = f", DetectorType={filter_metadata['DetectorType']}"
                print(f"    → Inferred: itype={filt_itype} ({FILTER_TYPE_NAMES[filt_itype]}), "
                      f"icalib={filt_icalib} ({FILTER_CALIB_NAMES[filt_icalib]}){detector_info}")
            
        except Exception as e:
            print(f"  Warning: Failed to load filter '{filter_id}': {e}")
            print(f"  Skipping this filter. Check filter ID format (e.g., 'SLOAN/SDSS.u' or 'Spitzer/IRAC.I1').")
            import traceback
            traceback.print_exc()
            raise
    
    # Create filters file
    print(f"\nWriting filter transmission curves to: {filters_file}")
    with open(filters_file, 'w') as f:
        # Write header line: "# itype icalib description"
        f.write("# itype icalib description\n")
        
        # Write each filter
        for i, (trans_table, desc, filt_itype, filt_icalib) in enumerate(
            zip(filter_transmission_data, filter_descriptions, filter_itypes, filter_icalibs)
        ):
            # Write filter definition line: "# itype icalib description"
            f.write(f"# {filt_itype} {filt_icalib} {desc}\n")
            
            # Get wavelength and transmission data from table
            wave_values = trans_table['Wavelength'].value  # In Angstroms
            trans_values = trans_table['Transmission'].value
            
            # Convert to numpy arrays
            wave_values = np.asarray(wave_values)
            trans_values = np.asarray(trans_values)
            
            # Ensure wave_values is 1D (handle multi-bin filters)
            if wave_values.ndim > 1:
                wave_values = wave_values.flatten()
            if trans_values.ndim > 1:
                trans_values = trans_values.flatten()
            
            # Convert wavelength to microns if needed
            if wave_units == "micron":
                # SVO transmission data is in Angstroms, convert to microns
                wave_values = wave_values * 1e-4  # Angstrom to micron
            
            # Write wavelength and transmission pairs
            for w, t in zip(wave_values, trans_values):
                f.write(f"{w:.6e} {t:.6e}\n")
            
            # Add blank line between filters (optional, for readability)
            f.write("\n")
    
    print(f"Successfully created filters file: {filters_file}")
    
    # Create filters_selected file
    if filters_selected_file is None:
        # Generate default name
        if filters_file.endswith('.txt'):
            filters_selected_file = filters_file.replace('.txt', '_selected.txt')
        else:
            filters_selected_file = filters_file + '_selected.txt'
    
    print(f"\nCreating filters_selected file: {filters_selected_file}")
    
    # Determine filter names to use: user-provided filter_names take precedence, otherwise use filt.name from svo_filters
    filter_names_to_use = None
    if filter_names is not None:
        # User provided custom names, use them
        filter_names_to_use = filter_names
    elif filter_names_from_svo:
        # Use filter names from SVO
        if selected_indices is None:
            # All filters selected, use all names
            filter_names_to_use = filter_names_from_svo
        else:
            # Only selected filters, extract names for selected indices
            filter_names_to_use = [filter_names_from_svo[i] for i in selected_indices if 0 <= i < len(filter_names_from_svo)]
    
    # Call create_filters_selected with the generated filters file
    create_filters_selected(
        filters_file=filters_file,
        output_selection_file=filters_selected_file,
        selected_indices=selected_indices,
        filter_names=filter_names_to_use,
        validate_itype_icalib=False,  # We already validated
        **filters_selected_kwargs
    )
    
    return filters_file, filters_selected_file


def create_filters_selected(
    filters_file: Union[str, List[str]],
    output_selection_file: Optional[str] = None,
    mag_lim: float = 99.0,
    mag_lim_err: float = 0.1,
    Nsigma: float = 5.0,
    mag_err_min: float = 0.02,
    SNR_min: float = 1.0,
    inoise: int = 0,
    Bsky: float = 0.0,
    D: float = 0.0,
    t: float = 0.0,
    Bdet: float = 0.0,
    Nread: float = 0.0,
    Rn: float = 0.0,
    Npx: float = 0.0,
    Npx_sig: float = 0.0,
    selected_indices: Optional[List[int]] = None,
    filter_names: Optional[List[str]] = None,
    validate_itype_icalib: bool = True,
    output_filters_file: Optional[str] = None,
):
    """
    Create a filters_selected file from one or more filters description files.
    
    This function replicates the functionality of the select_all bash script.
    It reads one or more filters files (with lines starting with '#') and creates a 
    filters_selected file containing only the selected filters. Selected filters
    will always have iused=1 and iselected=1.
    
    Filter Definition File Format:
    ------------------------------
    The input filters file(s) should contain lines starting with '#' in the format:
        # itype icalib description
    
    Where:
    - itype: Filter transmission type
        * 0 = Energy (filter response in energy units)
        * 1 = Photon (filter response in photon-counting units, default)
    
    - icalib: Filter calibration scheme (LEPHARE convention)
        * 0 = Standard calibration (B_ν = constant, default)
        * 1 = SPITZER/IRAC and ISO calibration (ν B_ν = constant)
        * 2 = Sub-mm calibration (B_ν = ν)
        * 3 = Blackbody calibration (T = 10,000 K)
        * 4 = SPITZER/MIPS mixed calibration
        * 5 = SCUBA mixed calibration
    
    Parameters
    ----------
    filters_file : str or list of str
        Path(s) to the input filters description file(s) (e.g., filters.txt).
        Can be a single file path (str) or a list of file paths (List[str]).
        Each file should contain filter definitions starting with '#'.
        Format: "# itype icalib description"
        When multiple files are provided, filters from all files are combined
        and can be selected from the combined list.
    output_selection_file : str, optional
        Path to the output filters_selected file. Only selected filters will
        be written to this file. If None (default), no filters_selected file is created.
    mag_lim : float, optional
        Magnitude limit (default: 99.0).
    mag_lim_err : float, optional
        Magnitude limit error (default: 0.1).
    Nsigma : float, optional
        Number of sigma for detection (default: 5.0).
    mag_err_min : float, optional
        Minimum magnitude error (default: 0.02).
    SNR_min : float, optional
        Minimum signal-to-noise ratio (default: 1.0).
    inoise : int, optional
        Noise flag (default: 0).
    Bsky, D, t, Bdet, Nread, Rn, Npx, Npx_sig : float, optional
        Additional filter parameters (default: 0.0).
    selected_indices : list of int, optional
        If provided, only filters at these indices (0-based) will be selected
        and written to the output file. Selected filters will have iused=1
        and iselected=1.
        If None (default), all filters are selected and written to the file.
        The function will print all available filters with their indices
        to help you choose which ones to select.
    filter_names : list of str, optional
        List of custom filter names corresponding to selected_indices.
        Must have the same length as selected_indices. Names are applied in order.
        If provided, custom names will be used instead of automatically extracted names.
        Example: ['u_band', 'r_band', 'i_band'] for selected_indices=[0, 2, 4]
    validate_itype_icalib : bool, optional
        If True (default), validate that itype is 0 or 1 and icalib is 0-5.
        Invalid values will trigger warnings and use defaults (itype=1, icalib=0).
    output_filters_file : str, optional
        Path to output combined filter file containing transmission curves for selected filters.
        If provided and multiple filter files are used, a new filter file will be created
        with the transmission data (wavelength and transmission pairs) for all selected filters.
        If None (default), no combined filter file is created.
        When multiple files are provided, this is recommended to create a single filter file
        containing only the selected filters' transmission data.
    
    Examples
    --------
    >>> create_filters_selected(
    ...     'observation/test2/filters.txt',
    ...     output_selection_file='observation/test2/filters_selected.txt'
    ... )
    
    >>> # Select only specific filters (indices 0, 2, 4)
    >>> create_filters_selected(
    ...     'filters.txt',
    ...     output_selection_file='filters_selected.txt',
    ...     selected_indices=[0, 2, 4]
    ... )
    
    >>> # Select filters from multiple filter files and save combined filter file
    >>> create_filters_selected(
    ...     ['filters_optical.txt', 'filters_nir.txt', 'filters_mir.txt'],
    ...     output_selection_file='filters_selected.txt',
    ...     selected_indices=[0, 2, 4, 10, 15],
    ...     output_filters_file='filters_combined.txt'
    ... )
    
    >>> # Create only combined filter file without selection file
    >>> create_filters_selected(
    ...     ['filters_optical.txt', 'filters_nir.txt'],
    ...     selected_indices=[0, 2, 4],
    ...     output_filters_file='filters_combined.txt'
    ... )
    
    >>> # Select filters with custom names
    >>> create_filters_selected(
    ...     'filters.txt',
    ...     output_selection_file='filters_selected.txt',
    ...     selected_indices=[0, 2, 4],
    ...     filter_names=['u_band', 'r_band', 'i_band']
    ... )
    
    >>> # Get filter list programmatically to help select filters
    >>> filters = create_filters_selected('filters.txt')
    >>> # Find filters by description
    >>> optical_mask = ['optical' in desc.lower() for desc in filters['Description']]
    >>> optical_filters = filters[optical_mask]
    >>> # Get selected filter IDs
    >>> selected = filters['ID'][filters['is_selected']].tolist()
    >>> # Create files with selected filters
    >>> create_filters_selected(
    ...     'filters.txt',
    ...     output_selection_file='filters_selected.txt',
    ...     output_filters_file='filters_combined.txt',
    ...     selected_indices=selected
    ... )
    >>> # Display the table
    >>> print(filters)
    >>> # Access columns
    >>> print(filters['ID'])
    >>> print(filters['Name'])
    
    Returns
    -------
    astropy.table.Table
        Table containing filter information with the following columns:
        - 'ID': int - Filter ID (position in full filter list, 0-based)
        - 'Name': str - Filter name
        - 'Type': str - Human-readable filter transmission type ('Energy' or 'Photon')
        - 'Calib': str - Human-readable filter calibration scheme ('Standard', 'SPITZER/IRAC', etc.)
        - 'Source': str - Source filter file (with parent directory if multiple files)
        - 'Description': str - Filter description (without itype/icalib prefix)
        - 'is_selected': bool - Whether this filter is selected
        - 'itype': str - Filter transmission type ('0' or '1')
        - 'icalib': str - Filter calibration scheme ('0'-'5')
        - 'source_file': str - Full path to source filter file
        - 'display_description': str - Description without itype/icalib prefix (same as Description)
    
    Notes
    -----
    The itype and icalib values are read from the filter definition file.
    These values control how filters are convolved with SED models:
    
    - itype determines whether the filter response function is interpreted
      as energy-based (0) or photon-counting (1).
    
    - icalib specifies the calibration scheme used, which applies a correction
      factor (fac_corr) to flux estimates, especially important at long
      wavelengths. The calibration schemes follow the LEPHARE convention.
    """
    # Normalize filters_file to a list
    if isinstance(filters_file, str):
        filters_files = [filters_file]
    else:
        filters_files = filters_file
    
    if not filters_files:
        raise ValueError("At least one filters file must be provided")
    
    # Read filters from all files (including transmission data)
    all_filter_lines = []
    filter_file_map = []  # Track which file each filter came from
    all_filter_data = []  # Store full filter data including transmission curves
    
    for file_path in filters_files:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Filters file not found: '{file_path}'")
        
        file_filter_lines = []
        file_filter_data = []  # Store filter data for this file
        
        with open(file_path, 'r') as f:
            current_filter_header = None
            current_filter_data = []
            
            for line in f:
                line_stripped = line.strip()
                if line_stripped.startswith('#'):
                    # Save previous filter if exists
                    if current_filter_header is not None:
                        file_filter_data.append({
                            'header': current_filter_header,
                            'data': current_filter_data
                        })
                    
                    # Start new filter
                    current_filter_header = line_stripped
                    current_filter_data = []
                    file_filter_lines.append(line_stripped)
                elif line_stripped and current_filter_header is not None:
                    # This is transmission data (wavelength transmission pair)
                    current_filter_data.append(line_stripped)
            
            # Save last filter
            if current_filter_header is not None:
                file_filter_data.append({
                    'header': current_filter_header,
                    'data': current_filter_data
                })
        
        if len(file_filter_lines) < 2:
            raise ValueError(f"Filters file '{file_path}' must contain at least the header line and one filter definition (lines starting with '#')")
        
        # Skip the first line (header: "# itype icalib description")
        file_filter_definitions = file_filter_lines[1:]
        file_filter_data_definitions = file_filter_data[1:]  # Skip header filter data
        
        # Add filters from this file to the combined list
        for i, filter_line in enumerate(file_filter_definitions):
            all_filter_lines.append(filter_line)
            filter_file_map.append(file_path)
            # Store corresponding filter data
            if i < len(file_filter_data_definitions):
                all_filter_data.append(file_filter_data_definitions[i])
            else:
                all_filter_data.append({'header': filter_line, 'data': []})
    
    if len(all_filter_lines) == 0:
        raise ValueError("No filter definitions found in any of the provided files")
    
    # Use all_filter_lines as filter_definitions
    filter_definitions = all_filter_lines
    
    # Collect filter information for listing
    filter_info = []
    
    # First pass: collect all filter information
    for i, filter_line in enumerate(filter_definitions):
        parts = filter_line.split(None, 3)
        if len(parts) < 4:
            parts.extend([''] * (4 - len(parts)))
        
        filter_type_str = parts[1] if len(parts) > 1 else '1'
        filter_calib_str = parts[2] if len(parts) > 2 else '0'
        description = parts[3] if len(parts) > 3 else ''
        
        # Use default filter name (simplified - no complex extraction)
        filter_short_name = f"F{i}"
        
        description = filter_line[1:].strip() if filter_line.startswith('#') else filter_line.strip()
        # Store full description for output file, but create a display description without itype/icalib
        # Remove itype and icalib from the beginning of description if they exist
        display_description = description
        desc_parts = description.split(None, 2)
        if len(desc_parts) >= 2:
            # Check if first two parts match itype and icalib
            if desc_parts[0] == filter_type_str and desc_parts[1] == filter_calib_str:
                # Remove itype and icalib, keep the rest
                display_description = desc_parts[2] if len(desc_parts) > 2 else ""
        
        # Get the source file for this filter
        source_file = filter_file_map[i] if i < len(filter_file_map) else filters_files[0]
        # For display, show parent directory if multiple files have same basename
        source_basename = os.path.basename(source_file)
        source_dirname = os.path.basename(os.path.dirname(source_file))
        if len(filters_files) > 1 and source_dirname and source_dirname != '.':
            source_file_name = f"{source_dirname}/{source_basename}"
        else:
            source_file_name = source_basename
        
        filter_info.append({
            'id': i,  # ID equals position in full filter list (0-based)
            'name': filter_short_name,
            'itype': filter_type_str,
            'icalib': filter_calib_str,
            'description': description,  # Full description for output file
            'display_description': display_description,  # Description without itype/icalib for display
            'source_file': source_file,  # Full path to source file
            'source_file_name': source_file_name  # Just the filename for display
        })
    
    # Validate filter_names if provided
    if filter_names is not None:
        if selected_indices is None:
            # When all filters are selected, filter_names should match all filters
            if len(filter_names) != len(filter_info):
                raise ValueError(f"filter_names (length {len(filter_names)}) must have the same length as total filters (length {len(filter_info)}) when selected_indices=None")
        else:
            # When specific filters are selected, filter_names should match selected_indices
            if len(filter_names) != len(selected_indices):
                raise ValueError(f"filter_names (length {len(filter_names)}) must have the same length as selected_indices (length {len(selected_indices)})")
    
    # Determine which filters to select
    if selected_indices is None:
        # Select all filters
        selected_filter_indices = list(range(len(filter_info)))
    else:
        # Select only specified filters
        selected_filter_indices = [i for i in selected_indices if 0 <= i < len(filter_info)]
        if len(selected_filter_indices) != len(selected_indices):
            invalid = [i for i in selected_indices if i < 0 or i >= len(filter_info)]
            print(f"Warning: Invalid filter indices {invalid} will be ignored.")
    
    # Apply custom names to filter_info if provided
    if filter_names is not None:
        for idx, filter_idx in enumerate(selected_filter_indices):
            if 0 <= filter_idx < len(filter_info):
                filter_info[filter_idx]['name'] = str(filter_names[idx])
    
    # List all available filters
    if len(filters_files) == 1:
        print(f"\nFound {len(filter_info)} available filters in '{filters_files[0]}':")
    else:
        # Show more distinguishing information for multiple files
        # If files have same basename, show parent directory or relative path
        def format_file_display(file_path):
            basename = os.path.basename(file_path)
            dirname = os.path.basename(os.path.dirname(file_path))
            # If directory name exists and is meaningful, show dirname/basename
            if dirname and dirname != '.' and dirname != os.path.basename(os.getcwd()):
                return f"{dirname}/{basename}"
            # Otherwise, try relative path from current directory
            try:
                rel_path = os.path.relpath(file_path)
                if rel_path != basename:
                    return rel_path
            except ValueError:
                pass
            # Fallback to basename
            return basename
        
        file_list_str = ', '.join([format_file_display(f) for f in filters_files])
        print(f"\nFound {len(filter_info)} available filters from {len(filters_files)} file(s) ({file_list_str}):")
    if filter_names:
        print(f"Note: Custom names provided for {len(filter_names)} selected filter(s).")
    
    # Automatically determine max description length based on terminal width
    try:
        terminal_width = shutil.get_terminal_size().columns
    except (AttributeError, OSError):
        # Fallback if terminal size cannot be determined
        terminal_width = 100
    
    # Calculate space used by fixed columns
    # ID(6) + Name(12) + Type(8) + Calib(8) + Source(20) + spacing(~4) = ~58
    fixed_columns_width = 6 + 12 + 8 + 8 + 20 + 4
    # Reserve some margin and calculate available space for description
    max_desc_len = max(20, terminal_width - fixed_columns_width - 5)  # -5 for margin
    
    # Show source file column only if multiple files were provided
    show_source_column = len(filters_files) > 1
    
    print("-" * min(120, terminal_width))
    if show_source_column:
        print(f"{'ID':<6} {'Name':<12} {'Type':<8} {'Calib':<8} {'Source':<20} {'Description'}")
    else:
        print(f"{'ID':<6} {'Name':<12} {'Type':<8} {'Calib':<8} {'Description'}")
    print("-" * min(120, terminal_width))
    # Add is_selected and display names to filter_info for return value
    for info in filter_info:
        is_selected = info['id'] in selected_filter_indices
        info['is_selected'] = is_selected
        
        # Format itype and icalib with descriptive names if possible
        try:
            itype_val = int(info['itype'])
            itype_display = FILTER_TYPE_NAMES.get(itype_val, info['itype'])
        except (ValueError, TypeError):
            itype_display = info['itype']
        info['itype_display'] = itype_display
        
        try:
            icalib_val = int(info['icalib'])
            icalib_display = FILTER_CALIB_NAMES.get(icalib_val, info['icalib'])
        except (ValueError, TypeError):
            icalib_display = info['icalib']
        info['icalib_display'] = icalib_display
    
    # Display the filter table
    for info in filter_info:
        # Use display_description (without itype/icalib) for the table display
        display_desc = info.get('display_description', info['description'])
        # Truncate description to fit on one line
        if len(display_desc) > max_desc_len:
            display_desc = display_desc[:max_desc_len-3] + "..."
        is_selected = info['is_selected']
        name_marker = "*" if (filter_names and is_selected) else " "
        
        if show_source_column:
            source_display = info.get('source_file_name', '')[:18]  # Truncate if too long
            print(f"{info['id']:<6} {info['name']:<11}{name_marker} {info['itype_display']:<8} {info['icalib_display']:<8} {source_display:<20} {display_desc}")
        else:
            print(f"{info['id']:<6} {info['name']:<11}{name_marker} {info['itype_display']:<8} {info['icalib_display']:<8} {display_desc}")
    print("-" * min(120, terminal_width))
    if filter_names:
        print("* = Custom name provided")
    print(f"\nFilter Type (itype): 0=Energy, 1=Photon")
    print(f"Filter Calib (icalib): 0=Standard, 1=SPITZER/IRAC, 2=Sub-mm, 3=Blackbody, 4=SPITZER/MIPS, 5=SCUBA")
    print(f"\nUse selected_indices parameter to select specific filters (e.g., selected_indices=[0, 2, 4])")
    print(f"Use filter_names parameter to provide custom names (e.g., filter_names=['u_band', 'r_band', 'i_band'])")
    print(f"If selected_indices=None (default), all filters will be selected.\n")
    
    # Write the output selection file (only selected filters) if requested
    selected_count = 0
    if output_selection_file is not None:
        with open(output_selection_file, 'w') as f:
            # Write header
            header = "iused iselected id name mid mag_lim mag_lim_err Nsigma mag_err_min SNR_min inoise Bsky D t Bdet Nread Rn Npx Npx_sig # itype icalib description"
            f.write(header + "\n")
            
            # Process only selected filters
            for i in selected_filter_indices:
                info = filter_info[i]
                
                # Convert itype and icalib to integers with validation
                try:
                    filter_itype = int(info['itype'])
                    if validate_itype_icalib and filter_itype not in (FILTER_TYPE_ENERGY, FILTER_TYPE_PHOTON):
                        print(f"Warning: Filter {i} has invalid itype={filter_itype}. "
                              f"Valid values are 0 (Energy) or 1 (Photon). Using default 1.")
                        filter_itype = FILTER_TYPE_PHOTON
                except (ValueError, TypeError):
                    if validate_itype_icalib:
                        print(f"Warning: Filter {i} has non-numeric itype='{info['itype']}'. Using default 1.")
                    filter_itype = FILTER_TYPE_PHOTON
                
                try:
                    filter_icalib = int(info['icalib'])
                    if validate_itype_icalib and filter_icalib not in range(6):
                        print(f"Warning: Filter {i} has invalid icalib={filter_icalib}. "
                              f"Valid values are 0-5. Using default 0.")
                        filter_icalib = FILTER_CALIB_STANDARD
                except (ValueError, TypeError):
                    if validate_itype_icalib:
                        print(f"Warning: Filter {i} has non-numeric icalib='{info['icalib']}'. Using default 0.")
                    filter_icalib = FILTER_CALIB_STANDARD
                
                # Selected filters always have iused=1 and iselected=1
                iused = 1
                iselected = 1
                
                # Filter ID equals position in the full filter list
                filter_id = info['id']
                
                # Format the output line
                # Format: iused iselected id name mid mag_lim mag_lim_err Nsigma mag_err_min SNR_min inoise Bsky D t Bdet Nread Rn Npx Npx_sig # itype icalib description
                # Use display_description to avoid duplicating itype/icalib in the description field
                output_description = info.get('display_description', info['description'])
                output_line = (
                    f"{iused} {iselected} {filter_id} {info['name']} "
                    f"-1 {mag_lim} {mag_lim_err} {Nsigma} {mag_err_min} {SNR_min} "
                    f"{inoise} {Bsky} {D} {t} {Bdet} {Nread} {Rn} {Npx} {Npx_sig} "
                    f"# {filter_itype} {filter_icalib} {output_description}"
                )
                f.write(output_line + "\n")
                selected_count += 1
        
        print(f"Created filters_selected file: {output_selection_file} with {selected_count} selected filter(s) (iused=1, iselected=1)")
    else:
        # Count selected filters even if not writing file
        selected_count = len(selected_filter_indices)
    
    # Create combined filter file with transmission data if requested
    if output_filters_file is not None:
        print(f"\nCreating filter file with transmission data: {output_filters_file}")
        with open(output_filters_file, 'w') as f:
            # Write header
            f.write("# itype icalib description\n")
            
            # Write selected filters with their transmission data
            for i in selected_filter_indices:
                info = filter_info[i]
                filter_data = all_filter_data[i] if i < len(all_filter_data) else None
                
                # Get itype and icalib values
                try:
                    filter_itype = int(info['itype'])
                except (ValueError, TypeError):
                    filter_itype = FILTER_TYPE_PHOTON
                
                try:
                    filter_icalib = int(info['icalib'])
                except (ValueError, TypeError):
                    filter_icalib = FILTER_CALIB_STANDARD
                
                # Reconstruct the filter header line in the format "# itype icalib description"
                # Use the original filter line from filter_definitions, which already has the correct format
                if i < len(filter_definitions):
                    filter_header = filter_definitions[i]
                else:
                    # Fallback: reconstruct from info
                    description = info.get('display_description', info['description'])
                    filter_header = f"# {filter_itype} {filter_icalib} {description}"
                
                if filter_data and filter_data['data']:
                    # Write filter header
                    f.write(f"{filter_header}\n")
                    
                    # Write transmission data (wavelength transmission pairs)
                    for data_line in filter_data['data']:
                        f.write(f"{data_line}\n")
                    
                    # Add blank line between filters
                    f.write("\n")
                else:
                    # Fallback: write header only if no data available
                    print(f"Warning: No transmission data found for filter {i}, writing header only")
                    f.write(f"{filter_header}\n\n")
        
        if len(filters_files) > 1:
            print(f"Successfully created combined filter file: {output_filters_file} with {len(selected_filter_indices)} filter(s)")
        else:
            print(f"Successfully created filter file: {output_filters_file} with {len(selected_filter_indices)} selected filter(s)")
    
    # Convert filter_info list to astropy Table for easier programmatic use
    # Create table with columns matching the displayed format
    table_data = {
        'ID': [info['id'] for info in filter_info],
        'Name': [info['name'] for info in filter_info],
        'Type': [info['itype_display'] for info in filter_info],
        'Calib': [info['icalib_display'] for info in filter_info],
        'Source': [info['source_file_name'] for info in filter_info],
        'Description': [info['display_description'] for info in filter_info],
        'is_selected': [info['is_selected'] for info in filter_info],
        'itype': [info['itype'] for info in filter_info],
        'icalib': [info['icalib'] for info in filter_info],
        'source_file': [info['source_file'] for info in filter_info],
        'display_description': [info['display_description'] for info in filter_info],
    }
    
    filter_table = Table(table_data)
    return filter_table

def main():
    # Add parameter parsing
    parser = argparse.ArgumentParser(description='BayeSED interface', add_help=False)
    parser.add_argument('-h', '--help', action='store_true', help='Show this help message and exit')
    parser.add_argument('--mpi', type=str, default='1', help='MPI mode: 1 for bayesed_mn_1, n for bayesed_mn_n')
    parser.add_argument('--np', type=int, default=None, help='Number of processes (default: use all available cores)')
    parser.add_argument('-i', '--input', type=str, help='Input type (int) and file, separated by comma')
    parser.add_argument('--outdir', type=str, default='result', help='Output directory')
    parser.add_argument('--save_bestfit', type=int, default=0, help='Save best fit')
    parser.add_argument('-v', '--verbose', type=int, default=2, help='Verbosity level')
    parser.add_argument('--save_sample_par', action='store_true',
                       help='Save the posterior sample of parameters')
    # Add other parameters of BayeSEDParams...

    args, unknown = parser.parse_known_args()

    mpi_mode = args.mpi
    if mpi_mode not in ['1', 'n']:
        print("Invalid MPI mode. Using default '1'.")
        mpi_mode = '1'

    bayesed = BayeSEDInterface(mpi_mode=mpi_mode)

    # Use all available cores if --np is not specified
    if args.np is None:
        bayesed.num_processes = bayesed._get_max_threads()
    else:
        bayesed.num_processes = args.np

    if args.help:
        # If user requests help, we will call BayeSED's help
        bayesed.run(BayeSEDParams(input_type=0, input_file="", outdir="", help=True))
        return

    if not args.input:
        parser.error("the following arguments are required: -i/--input")

    input_type, input_file = args.input.split(',')

    # Only pass the parameters defined in BayeSEDParams
    params_dict = {k: v for k, v in vars(args).items() if k in BayeSEDParams.__dataclass_fields__}
    params_dict['input_type'] = int(input_type)
    params_dict['input_file'] = input_file
    params = BayeSEDParams(**params_dict)

    bayesed.run(params)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python bayesed.py [--np <num_processes>] [--mpi <mpi_mode>] [-h] [BayeSED arguments...]")
        sys.exit(1)

    mpi_mode = '1'  # Default to bayesed_mn_1
    bayesed_args = []
    i = 1
    num_processes = None  # Default to None, which will use all available cores
    while i < len(sys.argv):
        if sys.argv[i] == "--np":
            num_processes = int(sys.argv[i+1])
            i += 2
        elif sys.argv[i] == "--mpi":
            mpi_mode = sys.argv[i+1]
            if mpi_mode not in ['1', 'n']:
                print("Invalid MPI mode. Using default '1'.")
                mpi_mode = '1'
            i += 2
        else:
            bayesed_args.append(sys.argv[i])
            i += 1

    bayesed = BayeSEDInterface(mpi_mode=mpi_mode)

    if '-h' in bayesed_args or '--help' in bayesed_args:
        num_processes = 1
    elif num_processes is None:
        num_processes = bayesed._get_max_threads()

    bayesed.num_processes = num_processes

    bayesed.run(bayesed_args)
