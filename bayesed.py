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
from typing import List, Optional

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
            args.extend(['-z', self._format_z_params(params.z)])
        
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
