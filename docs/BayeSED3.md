# BayeSED3: A code for Bayesian SED synthesis and analysis of galaxies and AGNs 

<p align="center">
  <img src="../BayeSED3.jpg" alt="BayeSED3 Logo" width="200"/>
  <br>
  <em>"With four parameters I can fit an elephant, and with five I can make him wiggle his trunk."</em>
  <br>
  <small>- Attributed to John von Neumann</small>
</p>

## Overview

BayeSED3 is a general and sophisticated tool for the full Bayesian interpretation of spectral energy distributions (SEDs) of galaxies and AGNs. It performs:

- Bayesian parameter estimation using posteriori probability distributions (PDFs)
- Bayesian SED model comparison using Bayesian evidence
- Support for various built-in SED models and machine learning model emulation
- Multi-component SED synthesis and analysis

### Key Features

- Multi-component SED synthesis and analysis of galaxies and AGNs
- Flexible stellar population synthesis modeling
- Flexible dust attenuation and emission modeling
- Flexible stellar and gas kinematics modeling
- Non-parametric and parametric star formation history options
- Comprehensive AGN component modeling (Accretion disk, BLR, NLR, Torus)
- Intergalactic medium (IGM) absorption modeling
- Handling of both photometric and spectroscopic data
- Bayesian parameter estimation and model comparison
- Machine learning techniques for SED model emulation
- Parallel processing support for improved performance
- User-friendly CLI, Python script and GUI interfaces

## Installation

1. Clone the repository:
```bash
git clone https://github.com/hanyk/BayeSED3.git
```

2. Install OpenMPI (automatically installed or manual installation):
```bash
cd BayeSED3
wget https://download.open-mpi.org/release/open-mpi/v4.1/openmpi-4.1.6.tar.gz
tar xzvf openmpi-4.1.6.tar.gz
cd openmpi-4.1.6
./configure --prefix=$PWD/../openmpi
make
make install
```

3. Install Python dependencies:
```bash
pip install -r requirements.txt
```

4. Optional dependencies:
- HDF5 utilities:
  - Ubuntu/Debian: `sudo apt-get install h5utils`
  - Fedora: `sudo dnf install hdf5-tools`
  - macOS: `brew install h5utils`
- GUI dependencies (tkinter):
  - Ubuntu/Debian: `sudo apt-get install python3-tk`
  - Fedora: `sudo dnf install python3-tkinter`
  - macOS: `brew install python-tk`

## System Compatibility

- Linux: x86_64 architecture
- macOS: x86_64 architecture (ARM supported via Rosetta 2)
- Windows: Supported through Windows Subsystem for Linux (WSL)

## Usage Examples

1. SDSS spectroscopic SED analysis:
```bash
python run_test.py gal plot
python run_test.py qso plot
```

2. Photometric SED analysis:
```bash
python run_test.py test1 plot
python run_test.py test2 plot
```

3. Mock CSST photometric and/or spectroscopic SED analysis:
```bash
python run_test.py test3 phot plot
python run_test.py test3 spec plot
python run_test.py test3 both plot
```

4. AGN Host Galaxy Decomposition:
For a demonstration of AGN host galaxy decomposition using image and SED analysis, see:
```bash
jupyter-notebook observation/agn_host_decomp/demo.ipynb
```

### Command Line Interface
```bash
./bayesed [OPTIONS] -i inputfile
```

### Graphical User Interface (GUI)
Launch the GUI:
```bash
python bayesed_gui.py
```
The GUI provides an intuitive way to set up complex SED analysis scenarios with meaningful defaults.

## Input Data Format

### File Structure Overview
The input data file consists of a header line followed by data lines. Each data line contains four distinct parts:

1. Basic Information
2. Photometric Data
3. Additional Information
4. Spectroscopic Data (if present)

### Header Format
```
# dataset_name num_photometric_bands num_additional_columns num_spectroscopic_bands
```
Example(observation/test1/test_inoise1.txt):
```
# test_inoise1 54 490 3
```
Where:
- First number (54): Number of photometric band pairs in Part 2
- Second number (490): Number of additional columns in Part 3
- Third number (3): Number of spectroscopic bands in Part 4

### Data Column Organization

#### Part 1: Basic Information
```
ID              # Object identifier
z_min           # Minimum redshift
z_max           # Maximum redshift
d               # Distance parameter
E(B-V)          # Extinction parameter
```

#### Part 2: Photometric Data
Pairs of magnitude and error values in AB system:
```
mAB(u)          # Magnitude in u band
mAB_err(u)      # Error in u band magnitude
mAB(B)          # Magnitude in B band
mAB_err(B)      # Error in B band magnitude
...             # Additional band pairs
```

#### Part 3: Additional Information
```
# Various additional columns including:
f[0,1]_{True}                # Model parameters
sys_err0_{True}              # System error parameters
sys_err1_{True}              # Additional system errors
m_cut[C0411]_{True}          # Cut parameters
m_cut[C1411]_{True}          # Additional cut parameters
logphi1_y1[C3902]_{True}     # Model-specific parameters
...                          # Other additional columns
```

#### Part 4: Spectroscopic Data
First, number of points per band:
```
Nw_CSST_GU      # Number of points for GU band
Nw_CSST_GV      # Number of points for GV band
Nw_CSST_GI      # Number of points for GI band
```

Then, for each spectral point:
```
w_CSST_GU0      # Wavelength (microns)
f_CSST_GU0      # Flux
e_CSST_GU0      # Error in flux
s_CSST_GU0      # Systematic error
```


## Detailed Input/Output Specifications

### Input Requirements

1. Input Data:
   
   a. Multi-band Photometric Data:
   - Format options:
     * Flux measurements in μJy (input_type=0)
     * Magnitudes in AB system (input_type=1)
   - Required columns:
     * Flux/Magnitude
     * Error measurements
   
   b. Multi-band Spectroscopic Data:
   - Required columns:
     * Wavelength (in μm)
     * Wavelength dispersion (in μm)
     * Flux (in μJy)
     * Flux error (in μJy)

2. Filter Definition File:
   - example: `observation/test1/filters_COSMOS_CSST_Euclid_LSST_WFIRST.txt`
   - Contains:
     * Wavelength coverage
     * Response functions

3. Filter Selection File:
   - example: `observation/test1/filters_CSST_seleted.txt`
   - Purpose: Specify which filters to use in the analysis
   - Format: List of filter IDs and usage flags

4. Model Component Specifications:
   - Required components:
     * Stellar population synthesis (SSP)
     * Star formation history (SFH)
     * Dust attenuation law (DAL)
   - Optional components:
     * AGN components
     * Dust emission
     * Emission lines

5. Parameter Prior Distributions:
   - File format: `.iprior` files
   - Auto-generated templates available
   - Specifies:
     * Parameter ranges
     * Prior types
     * Sampling configurations

### Output Files

1. Global Results File (*.hdf5):
   - Contains for all sources:
     * Model parameter estimates
     * Error estimates
     * Bayesian evidence values
     * Fit statistics
   - Format: HDF5 hierarchical data format
   - Advantages:
     * Efficient storage
     * Fast access
     * Hierarchical organization

2. Individual Source Best-fit Results (*_bestfit.fits):
   - Contains:
     * Best-fit model spectrum
     * Input data points
     * Residuals
     * Component contributions
   - Format: FITS binary table
   - Includes:
     * Wavelength grid
     * Flux values
     * Error estimates

3. Posterior Distribution Samples:
   - Files:
     * *_sample_par.paramnames: Parameter definitions
     * *_sample_par.txt: Sample values
   - Format: GetDist compatible


## Command Line Options

### Basic Usage
```bash
./bayesed [OPTIONS] -i inputfile
```

### Essential Options

- `-i, --input ARG1[,ARGn]`: Input file containing observed photometric SEDs with given unit
  - `0`: Flux in μJy
  - `1`: AB magnitude
  Example: `-i 0,observation/ULTRAVISTA/ULTRAVISTA0.txt`

- `--filters ARG`: Set the file containing filter definitions
  Example: `--filters filter/filters.txt`

- `--filters_selected ARG`: Set all used filters in the observation
  Example: `--filters_selected filter/filters_selected.txt`

- `--outdir ARG`: Output directory for all results
  Example: `--outdir result/`

### Model Selection Options

#### Stellar Population Models
- `-ssp, --ssp ARG1[,ARGn]`: Select a SSP model for the CSP model
  Example: `-ssp igroup,id,ynII,iscalable,k,f_run,Nstep,i0,i1,i2,i3`

- `-sfh, --sfh ARG1[,ARGn]`: Select a SFH for the CSP model
  Types:
  - 0: Instantaneous burst
  - 1: Constant
  - 2: Exponentially declining
  - 3: Exponentially increasing
  - 4: Single burst of length tau
  - 5: Delayed
  - 6: Beta
  - 7: Lognormal
  - 8: Double power-law
  - 9: Nonparametric

#### Dust Models
- `--dal ARG1[,ARGn]`: Set the dust attenuation law
  Laws:
  - 0: SED model with L_dust normalization
  - 1: Starburst (Calzetti+2000)
  - 2: Milky Way (Cardelli+1989)
  - 3: Star-forming (Salim+2018)
  - 4: MW (Allen+76)
  - 5: MW (Fitzpatrick+86)
  - 6: LMC (Fitzpatrick+86)
  - 7: SMC (Fitzpatrick+86)
  - 8: SB (Calzetti2000)
  - 9: Star-forming (Reddy+2015)

- `-gb, --greybody ARG1[,ARGn]`: Grey body spectrum
  Example: `--greybody igroup,id,gb,iscalable,ithick,w_min,w_max,Nw`

- `-bb, --blackbody ARG1[,ARGn]`: Black body spectrum
  Example: `--blackbody igroup,id,bb,iscalable,w_min,w_max,Nw`

### AGN Component Options

- `-bbb, --big_blue_bump ARG1[,ARGn]`: The big blue bump continuum spectrum of AGN
  Example: `-bbb igroup,id,bbb,iscalable,w_min,w_max,Nw`

- `-AGN, --AGN ARG1[,ARGn]`: The qsosed|agnsed|fagnsed|relagn|relqso|agnslim model of AGN
  Example: `-AGN igroup,id,AGN,iscalable,imodel,icloudy,suffix,w_min,w_max,Nw`

### Emission Line Component Options
- `-ls, --lines ARG1[,ARGn]`: Set emission lines from a file
  Example: `-ls igroup_start,id_start,file,iscalable,0.0,R,Nsample,Nkin`

- `-ls1, --lines1 ARG1[,ARGn]`: Set emission lines as one SED model
  Example: `-ls1 igroup,id,name,iscalable,file,R,Nsample,Nkin`

### Machine Learning Options

- `-a, --fann ARG1[,ARGn]`: Select FANN model by name
  Example: `-a igroup,id,name,iscalable`

- `-k, --aknn ARG1[,ARGn]`: Select AKNN model by name
  Example: `-k igroup,id,name,iscalable,k,f_run,eps,iRad,iprep,Nstep,alpha`

- `--rbf ARG1[,ARGn]`: Select RBF model by name
  Example: `-k igroup,id,name,iscalable`

- `--inn ARG1[,ARGn]`: Select INN model by name
  Example: `--inn igroup,id,name,iscalable,f_run,ikey`

### Output Control Options

- `--save_bestfit ARG`: Save best fitting result (0:fits 1:hdf5 2:both)
- `--save_pos_spec`: Save posterior distribution of model spectra
- `--save_sample_par`: Save posterior sample of parameters
- `--save_summary`: Save the summary file
- `--output_mock_photometry ARG`: Output mock photometry with best fit
- `--output_mock_spectra`: Output mock spectra with best fit
- `--save_pos_sfh ARG1[,ARGn]`: Save the posterior distribution of SFH
  Example: `--save_pos_sfh 100,1`
- `--save_sample_obs`: Save posteriori sample of observables
- `--save_sample_spec`: Save the posterior sample of model spectra

### Analysis Options

- `--build_sedlib ARG`: Build a SED library using employed models (0:rest,1:observed)
- `--make_catalog ARG1[,ARGn]`: Make catalog using model SEDs
  Example: `--make_catalog id1,logscale_min1,logscale_max1,id2,logscale_min2,logscale_max2`
- `--output_SFH ARG1[,ARGn]`: Output SFH over past tage year
  Example: `--output_SFH 10,0`
- `--SFR_over ARG1[,ARGn]`: Compute average SFR over past given Myrs
  Example: `--SFR_over 10,100`
- `--output_model_absolute_magnitude`: Output model absolute magnitude of best fit

### System Settings

- `--cosmology ARG1[,ARGn]`: Set cosmological parameters
  Example: `--cosmology 70,0.7,0.3` (default)
- `--IGM ARG1[,ARGn]`: Select IGM attenuation model (0-5)
- `--kin ARG1[,ARGn]`: Set kinematics parameters
  Example: `--kin -1,10,0,0` (default)
- `--gsl_integration_qag ARG1[,ARGn]`: Set GSL integration parameters
  Example: `--gsl_integration_qag 0,0.1,1000` (default)
- `--gsl_multifit_robust ARG1[,ARGn]`: Set robust fitting parameters
  Example: `--gsl_multifit_robust ols,1` (default)

### MultiNest Settings

- `--multinest ARG1[,ARGn]`: Configure MultiNest parameters
  Format: `IS,mmodal,ceff,nlive,efr,tol,updInt,Ztol,seed,fb,resume,outfile,logZero,maxiter,acpt_min`
  Default: `1,0,0,100,0.1,0.5,1000,-1e90,1,0,0,0,-1e90,100000,0.01`
- `--logZero ARG`: Max allowed Nsigma for multinest logZero
  Example: `--logZero 100000` (default)
- `--Ndumper ARG1[,ARGn]`: Set dumper parameters
  Example: `--Ndumper 1,0,-1` (default)

### Additional Options

- `-p, --polynomial ARG`: Multiplicative polynomial of order n
- `-t, --template ARG1[,ARGn]`: Use template SED with given name
- `--cloudy ARG1[,ARGn]`: Use SED model from CLOUDY code (v17.02)
- `--sedlib ARG1[,ARGn]`: Use SEDs from a sedlib
- `--LineList ARG1[,ARGn]`: LineList file and type for cloudy model
- `--import ARG1[,ARGn]`: Import command line options from file
- `--export ARG`: Export all options including defaults
- `--check`: Print all inputs and their category
- `--verbose ARG`: Set verbose level

For a complete list of options, run:
```bash
./bayesed --help
```

## Running BayeSED3

### Command Line Interface
```bash
./bayesed [OPTIONS] -i inputfile
```

Key options:
- `-i, --input`: Input file path and type
- `--outdir`: Output directory
- `--filters`: Filter definition file
- `--filters_selected`: Selected filters file
- `--multinest`: MultiNest sampling parameters
- `--save_bestfit`: Save best-fit results (0:fits, 1:hdf5, 2:both)
- `--save_sample_par`: Save parameter posterior samples
- `--save_pos_spec`: Save spectra posterior distribution

### Python Interface

The Python interface provides a programmatic way to configure and run BayeSED3 analyses.

```python
from bayesed import BayeSEDInterface, BayeSEDParams

# Initialize interface
bayesed = BayeSEDInterface(mpi_mode='1')

# Configure parameters
params = BayeSEDParams(
    input_type=0,
    input_file='data.txt',
    outdir='output',
    # Add model components as shown below...
)

# Run analysis
bayesed.run(params)
```

#### Available Model Components

##### 1. Stellar Population Models

###### Simple Stellar Population (SSP)
```python
params = BayeSEDParams(
    ssp=[SSPParams(
        igroup=0,
        id=0,
        name='bc2003_hr_stelib_chab_neb_300r',  # Model name
        iscalable=1,  # Scalable parameter
        i1=1  # Additional parameters
    )]
)
```

Available SSP models:
- BC03 (Bruzual & Charlot 2003)
  - High resolution (hr)
  - Low resolution (lr)
  - Different stellar libraries: STELIB, BaSeL
  - IMF options: Chabrier, Salpeter

###### Star Formation History (SFH)
```python
params = BayeSEDParams(
    sfh=[SFHParams(
        id=0,
        itype_sfh=2,  # SFH type
        itype_ceh=1   # Chemical evolution history type
    )]
)
```

SFH Types:
- 0: Instantaneous burst
- 1: Constant
- 2: Exponentially declining
- 3: Exponentially increasing
- 4: Single burst of length tau
- 5: Delayed
- 6: Beta
- 7: Lognormal
- 8: Double power-law
- 9: Nonparametric

##### 2. Dust Models

###### Dust Attenuation Law (DAL)
```python
params = BayeSEDParams(
    dal=[DALParams(
        id=0,
        ilaw=8  # Attenuation law type
    )]
)
```

Available laws:
- 0: SED model with L_dust normalization
- 1: Starburst (Calzetti+2000, FAST)
- 2: Milky Way (Cardelli+1989, FAST)
- 3: Star-forming (Salim+2018)
- 4: MW (Allen+76, hyperz)
- 5: MW (Fitzpatrick+86, hyperz)
- 6: LMC (Fitzpatrick+86, hyperz)
- 7: SMC (Fitzpatrick+86, hyperz)
- 8: SB (Calzetti2000, hyperz)
- 9: Star-forming (Reddy+2015)

###### Dust Emission
```python
params = BayeSEDParams(
    greybody=[GreybodyParams(
        igroup=0,
        id=1,
        name='gb',
        iscalable=-2
    )]
)
```

##### 3. AGN Components

###### Accretion Disk (Big Blue Bump)
```python
params = BayeSEDParams(
    big_blue_bump=[BigBlueBumpParams(
        igroup=1,
        id=1,
        name='bbb',
        iscalable=1,
        w_min=0.1,
        w_max=10,
        Nw=1000
    )]
)
```

###### Broad Line Region (BLR)
```python
params = BayeSEDParams(
    lines1=[LineParams(
        igroup=2,
        id=2,
        name='BLR',
        iscalable=1,
        file='lines_BLR.txt',
        R=300,
        Nkin=3
    )]
)
```

###### Narrow Line Region (NLR)
```python
params = BayeSEDParams(
    lines1=[LineParams(
        igroup=4,
        id=4,
        name='NLR',
        iscalable=1,
        file='lines_NLR.txt',
        R=2000,
        Nkin=2
    )]
)
```

###### FeII Emission
```python
params = BayeSEDParams(
    aknn=[AKNNParams(
        igroup=3,
        id=3,
        name='FeII',
        iscalable=1
    )]
)
```

##### 4. Additional Components

###### IGM Absorption
```python
params = BayeSEDParams(
    IGM=1  # 0:None, 1:Madau (1995, default), 2:Meiksin (2006), 3:hyperz, 4:FSPS, 5:Inoue+2014
)
```

#### Complete Configuration Examples

1. Basic Galaxy Configuration:
```python
params = BayeSEDParams(
    input_type=0,
    input_file='galaxy.txt',
    ssp=[SSPParams(
        igroup=0, id=0,
        name='bc2003_hr_stelib_chab_neb_2000r',
        iscalable=1, i1=1
    )],
    sfh=[SFHParams(id=0, itype_sfh=2)],  # Exponentially declining
    dal=[DALParams(id=0, ilaw=8)]  # Calzetti2000
)
```

2. AGN Configuration:
```python
params = BayeSEDParams(
    input_type=0,
    input_file='agn.txt',
    ssp=[SSPParams(...)],
    big_blue_bump=[BigBlueBumpParams(
        igroup=1, id=1, name='bbb',
        iscalable=1, w_min=0.1, w_max=10, Nw=1000
    )],
    lines1=[
        LineParams(  # Broad Line Region
            igroup=2, id=2, name='BLR',
            iscalable=1, file='lines_BLR.txt',
            R=300, Nkin=3
        ),
        LineParams(  # Narrow Line Region
            igroup=4, id=4, name='NLR',
            iscalable=1, file='lines_NLR.txt',
            R=2000, Nkin=2
        )
    ],
    aknn=[AKNNParams(  # FeII emission
        igroup=3, id=3, name='FeII',
        iscalable=1
    )]
)
```

3. Complex Multi-component Configuration:
```python
params = BayeSEDParams(
    input_type=0,
    input_file='combined.txt',
    # Stellar population
    ssp=[SSPParams(
        igroup=0, id=0,
        name='bc2003_lr_BaSeL_chab',
        iscalable=1
    )],
    sfh=[SFHParams(id=0, itype_sfh=2)],
    dal=[DALParams(id=0, ilaw=7)],  # SMC extinction
    # Dust emission
    greybody=[GreybodyParams(
        igroup=0, id=1,
        name='gb', iscalable=-2
    )],
    # AGN torus model
    fann=[FANNParams(
        igroup=1, id=2,
        name='clumpy201410tor',
        iscalable=1
    )],
    # MultiNest settings
    multinest=MultiNestParams(
        nlive=400,
        efr=0.1,
        updInt=1000,
        fb=2
    )
)
```

### GUI Interface
```bash
python bayesed_gui.py
```

## Output Files

1. Best-fit results:
   - `.fits` or `.hdf5` format
   - Contains best-fit parameters and model spectra

2. Parameter samples:
   - Posterior distributions
   - Corner plots
   - Chain statistics

3. Model spectra:
   - Best-fit spectrum
   - Uncertainty ranges
   - Component contributions

4. Diagnostic plots:
   - SED fits
   - Residuals
   - Parameter correlations

## Examples

### 1. SDSS Spectroscopic SED Analysis
```bash
python run_test.py gal plot
python run_test.py qso plot
```

Example output for galaxy fitting:
![Best-fit gal](../output/gal/spec-0285-51930-0184_GALAXY_STARFORMING/0Stellar+Nebular_2dal8_10_sys_err0_bestfit.fits.png)

Example output for QSO fitting:
![Best-fit qso](../output/qso/spec-2091-53447-0584_QSO_BROADLINE/0Stellar+Nebular_2dal8_10_1bbb_2dal7_15_2BLR_kin_eml3_13_3FeII_kin_con2_6_4NLR_kin_eml2_13_sys_err0_bestfit.fits.png)

### 2. Photometric SED Analysis
```bash
python run_test.py test1 plot
python run_test.py test2 plot
```

Example output for mock photometric data:
![Best-fit mock_phot](../test1/test_inoise1/0/0csp_sfh200_bc2003_lr_BaSeL_chab_i0000_2dal8_10_z_CSST_bestfit.fits.png)

Example output for W0533 ALMA data:
![Best-fit W0533](../test2/W0533_ALMA/W0533/0csp_sfh200_bc2003_lr_BaSeL_chab_i0000_2dal7_10_1gb_8_2clumpy201410tor_1_bestfit.fits.png)

### 3. Mock CSST Analysis
```bash
python run_test.py test3 phot plot  # Photometric analysis
python run_test.py test3 spec plot  # Spectroscopic analysis
python run_test.py test3 both plot  # Combined analysis
```

Example outputs for CSST mock data:

Photometric analysis:
![Best-fit csst_mock_phot](../test3/seedcat2_0_STARFORMING_inoise2/8144596/0csp_sfh201_bc2003_hr_stelib_chab_neb_300r_i0100_rdf0_2dal8_10_z_phot_bestfit.fits.png)

Spectroscopic analysis:
![Best-fit csst_mock_spec](../test3/seedcat2_0_STARFORMING_inoise2/8144596/0csp_sfh201_bc2003_hr_stelib_chab_neb_300r_i0100_rdf0_2dal8_10_z_spec_bestfit.fits.png)

Combined analysis:
![Best-fit csst_mock_both](../test3/seedcat2_0_STARFORMING_inoise2/8144596/0csp_sfh201_bc2003_hr_stelib_chab_neb_300r_i0100_rdf0_2dal8_10_z_both_bestfit.fits.png)

Parameter tree diagram:
![pdftree csst_mock_all](../test3/seedcat2_0_STARFORMING_inoise2/8144596/pdftree.png)

### 4. AGN Host Galaxy Decomposition
For a demonstration of AGN host galaxy decomposition using image and SED analysis, see:
```bash
jupyter-notebook observation/agn_host_decomp/demo.ipynb
```

This example shows a new approach to constraining properties of AGN host galaxies by combining image and SED decomposition, as described in [our recent paper](https://ui.adsabs.harvard.edu/abs/2024arXiv241005857Y/abstract).

### GUI Interface

The graphical user interface provides an intuitive way to set up complex SED analysis scenarios:

![BayeSED3 GUI](../BayeSED3_GUI.png)

### Key Configuration Notes

1. MPI Settings:
   - Use `--use-hwthread-cpus` for better performance
   - Platform-specific binaries (linux/mac, x86_64 only)

2. MultiNest Parameters:
   - `nlive`: More points for better accuracy (40-400 typical)
   - `efr`: Sampling efficiency (0.05-0.1 typical)
   - `updInt`: Update interval for output files

3. Model Component IDs:
   - Consistent ID numbering across components
   - Group IDs for component organization
   - Scalable parameters for flexibility

4. File Formats:
   - Input type 0: Flux in μJy
   - Input type 1: AB magnitude
   - FITS/HDF5 output options

## Best Practices

1. Data Preparation:
   - Ensure consistent units
   - Include measurement uncertainties
   - Remove problematic data points

2. Model Selection:
   - Start with simple models
   - Add components progressively
   - Use Bayesian evidence for model comparison

3. Parameter Settings:
   - Set appropriate prior ranges
   - Adjust MultiNest parameters for convergence
   - Consider systematic errors

4. Performance Optimization:
   - Use MPI for parallel processing
   - Adjust number of live points
   - Balance accuracy and computation time

## Data Analysis Options

### Data Quality Control
- `--SNRmin1 ARG1[,ARGn]`: Minimal SNR of data (phot,spec) for determining scaling
  Example: `--SNRmin1 0,0` (default)
- `--SNRmin2 ARG1[,ARGn]`: Minimal SNR of data (phot,spec) for likelihood evaluation
  Example: `--SNRmin2 0,0` (default)
- `--sys_err_mod ARG1[,ARGn]`: Prior for fractional systematic error of model
  Example: `--sys_err_mod iprior_type,is_age,min,max,nbin` (default:1,0,0,0,40)
- `--sys_err_obs ARG1[,ARGn]`: Prior for fractional systematic error of observations
  Example: `--sys_err_obs iprior_type,is_age,min,max,nbin` (default:1,0,0,0,40)

### Analysis Control
- `--no_photometry_fit`: Skip fitting photometric data even if present
- `--no_spectra_fit`: Skip fitting spectra data even if present
- `--priors_only`: Test priors by setting loglike for observational data to zero
- `--unweighted_samples`: Use unweighted posterior samples
- `--Ntest ARG`: Number of objects for test run
- `--niteration ARG`: Number of iterations (default: 0)
- `--NfilterPoints ARG`: Number of points per filter (default: 30)
- `--Nsample ARG`: Number of samples for catalog generation and SED library building

### Prior Settings
- `--z ARG1[,ARGn]`: Set prior for redshift z
  Example: `--z iprior_type,is_age,min,max,nbin` (default:1,0,z_min,z_max,100)
- `--np_sfh ARG1[,ARGn]`: Set prior type and parameters for nonparametric SFH
  Example: `--np_sfh 5,0,10,100` (default)
  - Prior types (0-7)
  - Interpolation method (0-3)
  - Number of bins
  - Regularization parameter

## Error Handling and Troubleshooting

### Common Issues

1. Input Data Format Errors:
   - Incorrect number of columns in data files
   - Mismatched filter definitions
   - Invalid unit specifications
   - Solution: Verify data format against examples in `observation/test/`

2. Model Parameter Conflicts:
   - Incompatible component combinations
   - Invalid parameter ranges
   - Solution: Check component dependencies and parameter constraints

3. Convergence Problems:
   - Poor initial parameter estimates
   - Insufficient sampling points
   - Complex model topology
   - Solutions:
     * Increase number of live points (`nlive`)
     * Adjust sampling efficiency (`efr`)
     * Simplify model if possible
     * Check parameter priors

4. Memory Issues:
   - Large spectral datasets
   - Many posterior samples
   - Solutions:
     * Reduce number of spectral points
     * Use selective output options
     * Avoid saving full posterior spectra unless necessary

5. Performance Issues:
   - Slow convergence
   - High computation time
   - Solutions:
     * Use MPI parallelization
     * Optimize MultiNest parameters
     * Consider using model emulators (FANN/AKNN)

### Diagnostic Tools

1. Model Validation:
   - Use `--check` to verify input configuration
   - Enable verbose output for detailed logging
   - Examine MultiNest convergence statistics

2. Output Inspection:
   - Check best-fit results
   - Examine posterior distributions
   - Analyze parameter correlations
   - Review residual patterns

3. Quality Assessment:
   - Monitor systematic errors
   - Check SNR thresholds
   - Verify parameter constraints
   - Evaluate Bayesian evidence

## References

Please cite these papers when using BayeSED:
- Han, Y., & Han, Z. 2012, ApJ, 749, 123
- Han, Y., & Han, Z. 2014, ApJS, 215, 2
- Han, Y., & Han, Z. 2019, ApJS, 240, 3
- Han, Y., Fan, L., Zheng, X. Z., Bai, J.-M., & Han, Z. 2023, ApJS, 269, 39
- Han, Y., et al. 2024a, in prep.

## File Descriptions

- `bayesed.py`: Main interface class for BayeSED3
- `bayesed_gui.py`: Graphical User Interface for BayeSED3
- `run_test.py`: Script to run BayeSED3 examples
- `requirements.txt`: List of Python dependencies
- `observation/test/`: Contains test data and configuration files
- `bin/`: Contains BayeSED3 executables for different platforms
- `nets/`: Contains Fast Artificial Neural Network (FANN) and Approximate K-Nearest Neighbors (AKNN) models for SED emulation
- `data/`: Other data files used by BayeSED3

## Citation

If BayeSED has been beneficial to your research, please consider citing our papers:

- Han, Y., & Han, Z. 2012, ApJ, 749, 123
- Han, Y., & Han, Z. 2014, ApJS, 215, 2
- Han, Y., & Han, Z. 2019, ApJS, 240, 3
- Han, Y., Fan, L., Zheng, X. Z., Bai, J.-M., & Han, Z. 2023, ApJS, 269, 39
- Han, Y., et al. 2024a, in prep.

## More Information

For more information about MultiNest, please refer to the `README_multinest.txt` file.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Contributions

Issues and pull requests are welcome. Please make sure to update tests before submitting a pull request.
