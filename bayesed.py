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
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class FANNParams:
    """
    Parameters for FANN (Fast Artificial Neural Network) model.
    """
    igroup: int
    id: int
    name: str
    iscalable: int

@dataclass
class AGNParams:
    """
    Parameters for AGN (Active Galactic Nuclei) model.
    """
    igroup: int
    id: int
    AGN: str
    iscalable: int
    imodel: int
    icloudy: int
    suffix: str
    w_min: float
    w_max: float
    Nw: int

@dataclass
class BlackbodyParams:
    """
    Parameters for blackbody model.
    """
    igroup: int
    id: int
    bb: str
    iscalable: int
    w_min: float
    w_max: float
    Nw: int

@dataclass
class BigBlueBumpParams:
    """
    Parameters for big blue bump model.
    """
    igroup: int
    id: int
    bbb: str
    iscalable: int
    w_min: float
    w_max: float
    Nw: int

@dataclass
class GreybodyParams:
    """
    Parameters for greybody model.
    """
    igroup: int
    id: int
    gb: str
    iscalable: int
    w_min: float
    w_max: float
    Nw: int

@dataclass
class AKNNParams:
    """
    Parameters for AKNN model.
    """
    igroup: int
    id: int
    name: str
    iscalable: int
    param5: int
    param6: int
    param7: int
    param8: int

@dataclass
class LineParams:
    """
    Set a series of emission line with name, wavelength/A and ratio in the given file as one SED model.
    """
    igroup: int
    id: int
    name: str
    iscalable: int
    file: str
    R: float
    Nsample: int
    Nkin: int

@dataclass
class LuminosityParams:
    """
    Compute luminosity between w_min and w_max in rest-frame for model with given id(-1 for all).
    """
    id: int
    w_min: float
    w_max: float

@dataclass
class NPSFHParams:
    """
    Set the prior type(0-7), interpolation method (0-3), number of bins and regul for the nonparametric SFH.
    """
    prior_type: int
    interpolation_method: int
    num_bins: int
    regul: float

@dataclass
class PolynomialParams:
    """
    Multiplicative polynomial of order n.
    """
    order: int

@dataclass
class PowerlawParams:
    """
    Power law spectrum.
    """
    igroup: int
    id: int
    pw: str
    iscalable: int
    w_min: float
    w_max: float
    Nw: int

@dataclass
class RBFParams:
    """
    Select rbf model by name.
    """
    igroup: int
    id: int
    name: str
    iscalable: int

@dataclass
class SFHParams:
    """
    Parameters for Star Formation History.
    """
    id: int
    itype_sfh: int
    itruncated: int
    itype_ceh: int

@dataclass
class SSPParams:
    """
    Parameters for SSP (Simple Stellar Population) model.
    """
    igroup: int
    id: int
    name: str
    iscalable: int
    k: int
    f_run: int
    Nstep: int
    i0: int
    i1: int
    i2: int
    i3: int

@dataclass
class SEDLibParams:
    """
    Use SEDs from a sedlib with the given name.
    """
    igroup: int
    id: int
    name: str
    iscalable: int
    dir: str
    itype: int
    f_run: int
    ikey: int

@dataclass
class SysErrParams:
    """
    Set the prior for the fractional systematic error of model or observation.
    """
    iprior_type: int
    is_age: int
    min: float
    max: float
    nbin: int

@dataclass
class ZParams:
    """
    Set the prior for the redshift z.
    """
    iprior_type: int
    is_age: int
    min: float
    max: float
    nbin: int

@dataclass
class NNLMParams:
    """
    The method, Niter1, tol1, Niter2, tol2, p1, p2 for the determination of nonnegative scale using NNLM.
    """
    method: int  # 0=eazy, 1=scd, 2=lee_ls, 3=scd_kl, 4=lee_kl
    Niter1: int
    tol1: float
    Niter2: int
    tol2: float
    p1: float
    p2: float

@dataclass
class NdumperParams:
    """
    Set the max number, iconverged_min and Xmin^2/Nd for the dumper of multinest.
    """
    max_number: int
    iconverged_min: int
    Xmin_squared_Nd: float

@dataclass
class OutputSFHParams:
    """
    Output the SFH over the past tage year as derived_pars for ntimes and in ilog scale.
    """
    ntimes: int
    ilog: int

@dataclass
class MultiNestParams:
    """
    Parameters for MultiNest.
    """
    is_: bool
    mmodal: bool
    ceff: bool
    nlive: int
    efr: float
    tol: float
    updInt: int
    Ztol: float
    seed: int
    fb: int
    resume: bool
    outfile: bool
    logZero: float
    maxiter: int
    acpt: float

@dataclass
class SFROverParams:
    past_Myr1: float
    past_Myr2: float

@dataclass
class SNRmin1Params:
    phot: float
    spec: float

@dataclass
class SNRmin2Params:
    phot: float
    spec: float

@dataclass
class GSLIntegrationQAGParams:
    epsabs: float
    epsrel: float
    limit: int
    key: int

@dataclass
class GSLMultifitRobustParams:
    type: str
    tune: float

@dataclass
class KinParams:
    velscale: int
    num_gauss_hermites_continuum: int
    num_gauss_hermites_emission: int

@dataclass
class LineListParams:
    file: str
    type: int

@dataclass
class MakeCatalogParams:
    id1: int
    logscale_min1: float
    logscale_max1: float
    id2: int
    logscale_min2: float
    logscale_max2: float

@dataclass
class CloudyParams:
    igroup: int
    id: int
    cloudy: str
    iscalable: int

@dataclass
class CosmologyParams:
    H0: float
    omigaA: float
    omigam: float

@dataclass
class DALParams:
    id: int
    con_eml_tot: int
    ilaw: int

@dataclass
class RDFParams:
    id: int
    num_polynomials: int

@dataclass
class TemplateParams:
    igroup: int
    id: int
    name: str
    iscalable: int

@dataclass
class BayeSEDParams:
    # Basic Parameters
    input_file: str
    outdir: str = "result"
    verbose: int = 2
    help: bool = False

    # Model related parameters
    fann: Optional[FANNParams] = None
    AGN: Optional[AGNParams] = None
    blackbody: Optional[BlackbodyParams] = None
    big_blue_bump: Optional[BigBlueBumpParams] = None
    greybody: Optional[GreybodyParams] = None
    aknn: Optional[AKNNParams] = None
    line: Optional[LineParams] = None
    lines: Optional[LineParams] = None
    lines1: Optional[LineParams] = None
    luminosity: Optional[LuminosityParams] = None
    np_sfh: Optional[NPSFHParams] = None
    polynomial: Optional[PolynomialParams] = None
    powerlaw: Optional[PowerlawParams] = None
    rbf: Optional[RBFParams] = None
    sfh: Optional[SFHParams] = None
    ssp: Optional[SSPParams] = None
    sedlib: Optional[SEDLibParams] = None
    sys_err_mod: Optional[SysErrParams] = None
    sys_err_obs: Optional[SysErrParams] = None
    z: Optional[ZParams] = None
    inn: Optional[SEDLibParams] = None
    cloudy: Optional[CloudyParams] = None
    cosmology: Optional[CosmologyParams] = None
    dal: Optional[DALParams] = None
    rdf: Optional[RDFParams] = None
    template: Optional[TemplateParams] = None

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
    kin: Optional[KinParams] = None

    # Other parameters
    rename: Optional[str] = None
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
    def __init__(self, executable_type='mn_1', openmpi_mirror=None):
        self.executable_type = executable_type
        self.openmpi_mirror = openmpi_mirror
        self._get_system_info()
        self.mpi_cmd = self._setup_openmpi()
        self.num_processes = self._get_max_threads()
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
        
        if not os.path.exists(install_dir):
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
            os.remove(openmpi_file)
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
        executable = f"bayesed_{self.executable_type}"
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
        
        # Check if there is a -h or --help parameter
        if '-h' in args or '--help' in args:
            # For help command, execute the executable file directly without using mpirun
            cmd = [self.executable_path] + args
        else:
            # When executing normally, use mpirun
            cmd = [self.mpi_cmd, '-np', str(self.num_processes), self.executable_path] + args
        
        print(f"Executing command: {' '.join(cmd)}")
        
        try:
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, bufsize=1)
            
            # Read and print output in real-time
            if process.stdout:
                for outline in process.stdout:
                    print(outline, end='')  # Print each line of output directly
            else:
                print("Warning: Unable to capture stdout")
            
            # Capture and print any errors
            if process.stderr:
                for error_line in process.stderr:
                    print(f"Error: {error_line}", end='', file=sys.stderr)
            
            # Wait for the process to finish
            return_code = process.wait()
            
            # Check the return code
            if return_code != 0:
                print(f"BayeSED execution failed, return code: {return_code}")
                return False
            else:
                print("BayeSED execution completed successfully")
                return True
        except Exception as e:
            print(f"Error occurred while executing BayeSED: {str(e)}")
            return False

    def _params_to_args(self, params):
        if isinstance(params, list):
            return params
        
        args = []
        
        if params.help:
            args.append('-h')
            return args
        
        args.extend([
            '-i', f"{params.input_file}",
            '--outdir', params.outdir,
            '--save_bestfit', str(params.save_bestfit),
            '-v', str(params.verbose)
        ])
        
        # Add other command line arguments of bayesed
        if params.fann:
            args.extend(['-a', self._format_fann_params(params.fann)])
        
        if params.AGN:
            args.extend(['-AGN', self._format_AGN_params(params.AGN)])
        
        if params.blackbody:
            args.extend(['-bb', self._format_blackbody_params(params.blackbody)])
        
        if params.big_blue_bump:
            args.extend(['-bbb', self._format_big_blue_bump_params(params.big_blue_bump)])
        
        if params.greybody:
            args.extend(['-gb', self._format_greybody_params(params.greybody)])
        
        if params.aknn:
            args.extend(['-k', self._format_aknn_params(params.aknn)])
        
        if params.line:
            args.extend(['-l', self._format_line_params(params.line)])
        
        if params.lines:
            args.extend(['-L', self._format_line_params(params.lines)])
        
        if params.lines1:
            args.extend(['-L1', self._format_line_params(params.lines1)])
        
        if params.luminosity:
            args.extend(['--luminosity', self._format_luminosity_params(params.luminosity)])
        
        if params.np_sfh:
            args.extend(['--np_sfh', self._format_np_sfh_params(params.np_sfh)])
        
        if params.polynomial:
            args.extend(['--polynomial', self._format_polynomial_params(params.polynomial)])
        
        if params.powerlaw:
            args.extend(['-pw', self._format_powerlaw_params(params.powerlaw)])
        
        if params.rbf:
            args.extend(['-rbf', self._format_rbf_params(params.rbf)])
        
        if params.sfh:
            args.extend(['--sfh', self._format_sfh_params(params.sfh)])
        
        if params.ssp:
            args.extend(['-ssp', self._format_ssp_params(params.ssp)])
        
        if params.sedlib:
            args.extend(['-sedlib', self._format_sedlib_params(params.sedlib)])
        
        if params.sys_err_mod:
            args.extend(['--sys_err_mod', self._format_sys_err_params(params.sys_err_mod)])
       
        if params.sys_err_obs:
            args.extend(['--sys_err_obs', self._format_sys_err_params(params.sys_err_obs)])
        
        if params.z:
            args.extend(['-z', self._format_z_params(params.z)])
        
        if params.rename:
            args.extend(['--rename', params.rename])
        
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
            args.extend(['--inn', self._format_sedlib_params(params.inn)])
        
        if params.IGM:
            args.extend(['--IGM', str(params.IGM)])
        
        if params.kin:
            args.extend(['--kin', self._format_kin_params(params.kin)])
        
        if params.logZero:
            args.extend(['--logZero', str(params.logZero)])
        
        if params.lw_max:
            args.extend(['--lw_max', str(params.lw_max)])
        
        if params.LineList:
            args.extend(['--LineList', self._format_LineList_params(params.LineList)])
        
        if params.make_catalog:
            args.extend(['--make_catalog', self._format_make_catalog_params(params.make_catalog)])
        
        if params.niteration:
            args.extend(['--niteration', str(params.niteration)])
        
        if params.no_photometry_fit:
            args.append('--no_photometry_fit')
        
        if params.cloudy:
            args.extend(['--cloudy', self._format_cloudy_params(params.cloudy)])
        
        if params.cosmology:
            args.extend(['--cosmology', self._format_cosmology_params(params.cosmology)])
        
        if params.dal:
            args.extend(['--dal', self._format_dal_params(params.dal)])
        
        if params.export:
            args.extend(['--export', params.export])
        
        if params.import_files:
            for import_file in params.import_files:
                args.extend(['--import', import_file])
        
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
            args.extend(['-t', self._format_template_params(params.template)])
        
        if params.build_sedlib is not None:
            args.extend(['--build_sedlib', str(params.build_sedlib)])
        if params.check:
            args.append('--check')
        if params.cl:
            args.extend(['--cl', params.cl])
        if params.filters:
            args.extend(['--filters', params.filters])
        if params.filters_selected:
            args.extend(['--filters_selected', params.filters_selected])
        if params.gsl_integration_qag:
            args.extend(['--gsl_integration_qag', self._format_gsl_integration_qag_params(params.gsl_integration_qag)])
        if params.gsl_multifit_robust:
            args.extend(['--gsl_multifit_robust', self._format_gsl_multifit_robust_params(params.gsl_multifit_robust)])
        if params.import_files:
            for import_file in params.import_files:
                args.extend(['--import', import_file])
        if params.inn:
            args.extend(['--inn', self._format_sedlib_params(params.inn)])
        if params.IGM is not None:
            args.extend(['--IGM', str(params.IGM)])
        if params.kin:
            args.extend(['--kin', self._format_kin_params(params.kin)])
        if params.LineList:
            args.extend(['--LineList', self._format_LineList_params(params.LineList)])
        if params.logZero is not None:
            args.extend(['--logZero', str(params.logZero)])
        if params.lw_max is not None:
            args.extend(['--lw_max', str(params.lw_max)])
        if params.niteration is not None:
            args.extend(['--niteration', str(params.niteration)])
        if params.no_photometry_fit:
            args.append('--no_photometry_fit')
        
        return args

    def _format_fann_params(self, fann_params):
        return f"{fann_params.igroup},{fann_params.id},{fann_params.name},{fann_params.iscalable}"

    def _format_AGN_params(self, AGN_params):
        return f"{AGN_params.igroup},{AGN_params.id},{AGN_params.name},{AGN_params.iscalable}"

    def _format_blackbody_params(self, blackbody_params):
        return f"{blackbody_params.igroup},{blackbody_params.id},{blackbody_params.name},{blackbody_params.iscalable}"

    def _format_big_blue_bump_params(self, big_blue_bump_params):
        return f"{big_blue_bump_params.igroup},{big_blue_bump_params.id},{big_blue_bump_params.name},{big_blue_bump_params.iscalable}"

    def _format_greybody_params(self, greybody_params):
        return f"{greybody_params.igroup},{greybody_params.id},{greybody_params.name},{greybody_params.iscalable}"

    def _format_aknn_params(self, aknn_params):
        return f"{aknn_params.igroup},{aknn_params.id},{aknn_params.name},{aknn_params.iscalable}"

    def _format_line_params(self, line_params):
        return f"{line_params.igroup},{line_params.id},{line_params.name},{line_params.iscalable}"

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
        return f"{int(multinest_params.is_)},{int(multinest_params.mmodal)},{int(multinest_params.ceff)},{multinest_params.nlive},{multinest_params.efr},{multinest_params.tol},{multinest_params.updInt},{multinest_params.Ztol},{multinest_params.seed},{multinest_params.fb},{int(multinest_params.resume)},{int(multinest_params.outfile)},{multinest_params.logZero},{multinest_params.maxiter},{multinest_params.acpt}"

    def _format_gsl_integration_qag_params(self, gsl_integration_qag_params):
        return f"{gsl_integration_qag_params.epsabs},{gsl_integration_qag_params.epsrel},{gsl_integration_qag_params.limit},{gsl_integration_qag_params.key}"

    def _format_gsl_multifit_robust_params(self, gsl_multifit_robust_params):
        return f"{gsl_multifit_robust_params.type},{gsl_multifit_robust_params.tune}"

    def _format_kin_params(self, kin_params):
        return f"{kin_params.velscale},{kin_params.num_gauss_hermites_continuum},{kin_params.num_gauss_hermites_emission},{int(kin_params.save_sample_obs)}"

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
    parser.add_argument('--exe', type=str, default='mn_1', help='Executable type')
    parser.add_argument('--np', type=int, default=1, help='Number of processes')
    parser.add_argument('-i', '--input_file', type=str, help='Input file')
    parser.add_argument('--outdir', type=str, default='result', help='Output directory')
    parser.add_argument('--save_bestfit', type=int, default=0, help='Save best fit')
    parser.add_argument('-v', '--verbose', type=int, default=2, help='Verbosity level')
    # Add other parameters of BayeSEDParams...

    args, unknown = parser.parse_known_args()

    bayesed = BayeSEDInterface(executable_type=args.exe)
    bayesed.num_processes = args.np

    if args.help:
        # If user requests help, we will call BayeSED's help
        bayesed.run(BayeSEDParams(input_file="", outdir="", help=True))
        return

    if not args.input_file:
        parser.error("the following arguments are required: -i/--input_file")

    # Only pass the parameters defined in BayeSEDParams
    params_dict = {k: v for k, v in vars(args).items() if k in BayeSEDParams.__dataclass_fields__}
    params = BayeSEDParams(**params_dict)
    
    bayesed.run(params)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python bayesed.py [--np <num_processes>] [--exe <executable_type>] [-h] [BayeSED arguments...]")
        sys.exit(1)

    executable_type = 'mn_1'  # Default to bayesed_mn_1
    bayesed_args = []
    i = 1
    num_processes = 1
    while i < len(sys.argv):
        if sys.argv[i] == "--np":
            num_processes = int(sys.argv[i+1])
            i += 2
        elif sys.argv[i] == "--exe":
            executable_type = sys.argv[i+1]
            i += 2
        else:
            bayesed_args.append(sys.argv[i])
            i += 1

    bayesed = BayeSEDInterface(executable_type=executable_type)
    
    if '--help' in bayesed_args:
        num_processes = 1
    
    bayesed.num_processes = num_processes

    bayesed.run(bayesed_args)
