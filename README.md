# BayeSED3: A code for Bayesian SED synthesis and analysis of galaxies and AGNs

<p align="center">
  <img src="BayeSED3.jpg" alt="BayeSED3 Logo" width="200"/>
  <br>
  <em>"With four parameters I can fit an elephant, and with five I can make him wiggle his trunk."</em>
  <br>
  <small>- Attributed to John von Neumann</small>
</p>

BayeSED3 is a general and sophisticated tool for the full Bayesian interpretation of spectral energy distributions (SEDs) of galaxies and AGNs. It performs Bayesian parameter estimation using posteriori probability distributions (PDFs) and Bayesian SED model comparison using Bayesian evidence. BayeSED3 supports various built-in SED models and can emulate other SED models using machine learning techniques.

## Key Features

- **Explore the [BayeSED3-AI Assistant 🚀](https://udify.app/chat/Gmni8M7sHDKWG67F) or [![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/hanyk/BayeSED3) for interactive help and guidance!**
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

## Installation Instructions

**Platform support:**
- Linux x86_64
- macOS (x86_64 and ARM64 via Rosetta 2)
- Windows (via WSL - uses Linux binaries)

### Quick Start

**BayeSED3 can be used directly from the repository root** without system-level installation:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/hanyk/BayeSED3.git
   cd BayeSED3
   ```

2. **Install OpenMPI** (required):
   ```bash
   conda install openmpi=4.1.6
   # Or via system: brew install openmpi (macOS) or apt-get install openmpi-bin (Linux)
   ```

3. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Use BayeSED3 directly:**
   ```bash
   python run_test.py gal plot
   # Or use Python interface
   python -c "from bayesed import BayeSEDInterface; ..."
   ```

**Note:** When using from repository root, you must run commands from the BayeSED3 directory or use absolute paths.

### Pip Installation (System-Level) ⭐ Recommended for Convenience

**System-level installation using pip** - allows you to use BayeSED3 from any directory. Simpler for uninstalling (only removes bayesed3, not its dependencies).

**Prerequisites:** Clone the repository first (see Quick Start above).

**Installation:**
```bash
# Install OpenMPI first (required)
conda install openmpi=4.1.6
# Or via system: brew install openmpi (macOS) or apt-get install openmpi-bin (Linux)

# Install BayeSED3 (run from repository root)
pip install .          # Regular install (for production use)
pip install -e .      # Editable install (for development - changes immediately visible)
```

**Uninstall:**
```bash
pip uninstall bayesed3
# Dependencies (OpenMPI, matplotlib, etc.) stay installed
```

**Benefits:**
- ✅ Cleaner uninstall (only removes bayesed3)
- ✅ Faster development cycle
- ✅ Works with virtual environments
- ✅ Works from any directory (after installation)
- ⚠️ Requires OpenMPI to be installed separately

**Note:** 
- **Regular install** (`pip install .`): Copies files to site-packages. Use for production.
- **Editable install** (`pip install -e .`): Links to source directory. Changes are immediately visible. Use for development.
- BayeSED3 automatically detects OpenMPI from conda, system, or local installation. If none is found, it will auto-compile OpenMPI 4.1.6.

### Conda Installation (System-Level, Automatic Dependencies)

**System-level installation using conda** - automatically handles all dependencies including OpenMPI. Note: BayeSED3 is not yet available on conda-forge. You must build it locally.

**Prerequisites:** Clone the repository first (see Quick Start above).

**Installation:**
```bash
# Build conda package from source (run from repository root)
conda build conda/

# Install the locally built package
conda install --use-local bayesed3
```

**Uninstall:**
```bash
# Remove BayeSED3 (also removes dependencies like OpenMPI, matplotlib, etc.)
conda remove bayesed3

# Optional: Clean package cache
conda clean --packages -y

# Optional: Clean build cache (if you built it locally)
conda build purge
```

**Benefits:**
- ✅ Automatic dependency management (OpenMPI, HDF5, Python packages)
- ✅ Works from any directory (after installation)
- ⚠️ Requires local build first (not available on conda-forge yet)
- ⚠️ Uninstall removes dependencies: `conda remove bayesed3` also removes OpenMPI, matplotlib, etc.

**Note:** BayeSED3 will be available on conda-forge in the future.

### Manual Installation (Advanced)

For advanced manual setup (same as Quick Start, but with more detail):

1. Clone the repository:
   ```
   git clone https://github.com/hanyk/BayeSED3.git
   cd BayeSED3
   ```

2. **Install OpenMPI 4.1.6** (REQUIRED):
   
   **Via conda (recommended):**
   ```bash
   conda install -c conda-forge openmpi=4.1.6
   ```
   
   **Via system package manager:**
   - Ubuntu/Debian: `sudo apt-get install openmpi-bin openmpi-common libopenmpi-dev`
   - Fedora: `sudo dnf install openmpi openmpi-devel`
   - macOS (Homebrew): `brew install openmpi`
   
   **Or compile from source:**
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
   ```
   pip install -r requirements.txt
   ```

4. Install HDF5 utilities (optional):
   - Ubuntu/Debian: `sudo apt-get install h5utils`
   - Fedora: `sudo dnf install hdf5-tools`
   - macOS (with Homebrew): `brew install h5utils`

5. Install tkinter (for GUI):
   - Ubuntu/Debian: `sudo apt-get install python3-tk`
   - Fedora: `sudo dnf install python3-tkinter`
   - macOS (with Homebrew): `brew install python-tk`

## Usage examples

1. SDSS spectroscopic SED analysis
   ```
   python run_test.py gal plot
   python run_test.py qso plot
   ```
![Best-fit gal](output/gal/spec-0285-51930-0184_GALAXY_STARFORMING/0Stellar+Nebular_2dal8_10_sys_err0_bestfit.fits.png)
![Best-fit qso](output/qso/spec-2091-53447-0584_QSO_BROADLINE/0Stellar+Nebular_2dal8_10_1bbb_2dal7_15_2BLR_kin_eml3_13_3FeII_kin_con2_6_4NLR_kin_eml2_13_sys_err0_bestfit.fits.png)

2. photometric SED analysis
   ```
   python run_test.py test1 plot
   python run_test.py test2 plot
   ```
![Best-fit mock_phot](test1/test_inoise1/0/0csp_sfh200_bc2003_lr_BaSeL_chab_i0000_2dal8_10_z_CSST_bestfit.fits.png)
![Best-fit W0533](test2/W0533_ALMA/W0533/0csp_sfh200_bc2003_lr_BaSeL_chab_i0000_2dal7_10_1gb_8_2clumpy201410tor_1_bestfit.fits.png)

3. mock CSST photometric and/or spectroscopic SED analysis
   ```
   python run_test.py test3 phot plot
   python run_test.py test3 spec plot
   python run_test.py test3 both plot
   ```
![Best-fit csst_mock_phot](test3/seedcat2_0_STARFORMING_inoise2/8144596/0csp_sfh201_bc2003_hr_stelib_chab_neb_300r_i0100_rdf0_2dal8_10_z_phot_bestfit.fits.png)
![Best-fit csst_mock_spec](test3/seedcat2_0_STARFORMING_inoise2/8144596/0csp_sfh201_bc2003_hr_stelib_chab_neb_300r_i0100_rdf0_2dal8_10_z_spec_bestfit.fits.png)
![Best-fit csst_mock_both](test3/seedcat2_0_STARFORMING_inoise2/8144596/0csp_sfh201_bc2003_hr_stelib_chab_neb_300r_i0100_rdf0_2dal8_10_z_both_bestfit.fits.png)
![pdftree csst_mock_all](test3/seedcat2_0_STARFORMING_inoise2/8144596/pdftree.png)

4. [ A new approach to constraining properties of AGN host galaxies by combining image and SED decomposition](https://ui.adsabs.harvard.edu/abs/2024arXiv241005857Y/abstract)

jupyter-notebook [observation/agn_host_decomp/demo.ipynb](observation/agn_host_decomp/demo.ipynb)

### Python Interface

BayeSED3 provides a high-level Python interface for programmatic SED analysis. The interface simplifies configuration, data preparation, and result access while maintaining full access to all BayeSED3 capabilities.

**Quick Start Examples:**

```python
from bayesed import BayeSEDInterface, BayeSEDParams, BayeSEDResults

# Initialize interface
bayesed = BayeSEDInterface(mpi_mode='auto')

# Simple galaxy fitting
params = BayeSEDParams.galaxy(
    input_file='observation/test/gal.txt',
    outdir='output',
    ssp_model='bc2003_hr_stelib_chab_neb_2000r',
    sfh_type='exponential',
    dal_law='calzetti'
)

# Run analysis
result = bayesed.run(params)

# Load and analyze results
results = BayeSEDResults('output', catalog_name='gal')
results.print_summary()

# Access parameters and objects
free_params = results.get_free_parameters()
available_objects = results.list_objects()

# Load all parameters as astropy Table from HDF5 file
hdf5_table = results.load_hdf5_results()
# Built-in SNR filtering
high_snr_table = results.load_hdf5_results(filter_snr=True, min_snr=5.0)


# Access all statistical estimates for specific parameters
age_table = results.get_parameter_values('log(age/yr)[0,1]')
mass_table = results.get_parameter_values('log(Mstar)[0,1]')

custom_labels = {
    # Free parameters
    'log(age/yr)[0,1]': r'\log(age/\mathrm{yr})',
    'log(tau/yr)[0,1]': r'\log(\tau/\mathrm{yr})',
    'log(Z/Zsun)[0,1]': r'\log(Z/Z_\odot)',
    'Av_2[0,1]': r'A_V',
    # Derived parameters
    'log(Mstar)[0,1]': r'\log(M_\star/M_\odot)',
    'log(SFR_{100Myr}/[M_{sun}/yr])[0,1]': r'\log(\mathrm{SFR}/M_\odot\,\mathrm{yr}^{-1})'
}

results.set_parameter_labels(custom_labels)
results.plot_bestfit()
results.plot_posterior_free()
results.plot_posterior_derived(max_params=5)
results.plot_posterior(['log(age/yr)[0,1]', 'log(Z/Zsun)[0,1]', 'log(Mstar)[0,1]', 'log(SFR_{100Myr}/[M_{sun}/yr])[0,1]'])  # Mixed free+derived parameters


# Object-level analysis
object_results = BayeSEDResults('output', catalog_name='gal',
                               object_id='spec-0285-51930-0184_GALAXY_STARFORMING')
object_results.plot_bestfit()

object_results.set_parameter_labels(custom_labels)
object_results.plot_posterior_free()
object_results.plot_posterior_derived(max_params=5)
object_results.plot_posterior(['log(age/yr)[0,1]', 'log(Z/Zsun)[0,1]', 'log(Mstar)[0,1]', 'log(SFR_{100Myr}/[M_{sun}/yr])[0,1]'])  # Mixed free+derived parameters
```

**AGN Fitting:**

```python
# AGN with all components (includes galaxy host)
params = BayeSEDParams.agn(
    input_file='observation/test/qso.txt',
    outdir='output',
    ssp_model='bc2003_hr_stelib_chab_neb_2000r',
    sfh_type='exponential',
    dal_law='calzetti',
    agn_components=['dsk', 'blr', 'nlr', 'feii']  # Disk, BLR, NLR, FeII
)

bayesed.run(params)
```

## Advanced Features

### Enhanced Results Analysis

```python
# Enhanced BayeSEDResults with automatic configuration detection
results = BayeSEDResults('output', catalog_name='gal')

# Comprehensive summary and status reporting
results.print_summary()
status = results.get_status_report()

# Efficient parameter access with caching
free_params = results.get_free_parameters()
derived_params = results.get_derived_parameters()
all_params = results.get_parameter_names()

# Object-level analysis for detailed single-object work
object_results = BayeSEDResults('output', catalog_name='gal', object_id='spec-0285-51930-0184')

# Enhanced introspection and debugging
available_objects = results.list_objects()
available_configs = results.list_model_configurations()

# Scope management for sample vs object-level analysis
scope_info = results.get_access_scope()
print(f"Scope: {scope_info.scope_type}")
print(f"Total objects: {scope_info.total_objects}")

# Logging control (quiet by default)
results.enable_verbose_logging()  # Enable detailed INFO/DEBUG messages for debugging
results.enable_quiet_logging()    # Back to quiet mode (default)
```

### Enhanced Plotting Capabilities

```python
# Enhanced plotting with automatic parameter filtering
results = BayeSEDResults('output', catalog_name='gal')

# Plot posterior distributions for free parameters (corner plot)
results.plot_posterior_free(output_file='free_params.png', show=False)

# Plot posterior distributions for derived parameters with limit
results.plot_posterior_derived(max_params=10, output_file='derived_params.png', show=False)

# Custom parameter selection
mixed_params = ['log(age/yr)[0,1]', 'log(Z/Zsun)[0,1]', 'log(Mstar)[0,1]']
results.plot_posterior(mixed_params, object_id=results.list_objects()[0], 
                      output_file='custom_params.png', show=False)

# Object-level best-fit SED plotting
object_results = BayeSEDResults('output', catalog_name='gal', 
                               object_id='spec-0285-51930-0184_GALAXY_STARFORMING')
object_results.plot_bestfit()

# Custom LaTeX labels for publication-quality plots
custom_labels = {
    'log(age/yr)[0,1]': r'\log(age/\mathrm{yr})',
    'log(tau/yr)[0,1]': r'\log(\tau/\mathrm{yr})',
    'log(Z/Zsun)[0,1]': r'\log(Z/Z_\odot)',
    'Av_2[0,1]': r'A_V',
    'log(Mstar)[0,1]': r'\log(M_\star/M_\odot)'
}
results.set_parameter_labels(custom_labels)
results.plot_posterior_free(output_file='labeled_params.png', show=False)
```

### Advanced Analytics and Model Comparison

```python
# Compare multiple models with enhanced BayeSEDResults
results1 = BayeSEDResults('output_model1', catalog_name='galaxies')
results2 = BayeSEDResults('output_model2', catalog_name='galaxies')

# Advanced analytics for sample-level analysis
correlations = results1.compute_parameter_correlations(['log(age/yr)', 'log(M*/Msun)'])
stats = results1.get_parameter_statistics(['log(age/yr)', 'log(M*/Msun)'])

# Enhanced GetDist integration with intelligent caching
samples1 = results1.get_getdist_samples()
samples2 = results2.get_getdist_samples()
samples1.label = 'Model 1'
samples2.label = 'Model 2'

# Create comparison plots
from getdist import plots
import matplotlib.pyplot as plt

g = plots.get_subplot_plotter()
g.triangle_plot([samples1, samples2], ['log(age/yr)', 'log(M*/Msun)'], filled=True)
plt.show()
```

### Working with Data Arrays

```python
import numpy as np
from bayesed import BayeSEDInterface, BayeSEDParams
from bayesed.data import SEDObservation

# Create observation from arrays
obs = SEDObservation(
    ids=[1, 2, 3],
    z_min=[0.1, 0.2, 0.3],
    z_max=[0.2, 0.3, 0.4],
    phot_filters=['SLOAN/SDSS.u', 'SLOAN/SDSS.g', 'SLOAN/SDSS.r'],
    phot_fluxes=np.array([[10.0, 20.0, 30.0], [15.0, 25.0, 35.0], [20.0, 30.0, 40.0]]),
    phot_errors=np.array([[1.0, 2.0, 3.0], [1.5, 2.5, 3.5], [2.0, 3.0, 4.0]]),
    input_type=0  # Flux in μJy
)

# Convert to BayeSED input format
input_file = obs.to_bayesed_input('observation/my_analysis', 'input_catalog')

# Download filters from SVO
bayesed = BayeSEDInterface()
filter_files = bayesed.prepare_filters_from_svo(
    svo_filter_ids=['SLOAN/SDSS.u', 'SLOAN/SDSS.g', 'SLOAN/SDSS.r'],
    output_dir='observation/my_analysis/filters'
)

# Create and run analysis
params = BayeSEDParams.galaxy(
    input_file=input_file,
    outdir='observation/my_analysis/output',
    filters=filter_files['filters_file'],
    filters_selected=filter_files['filters_selected_file']
)
bayesed.run(params)
```

### Custom Model Configuration

```python
from bayesed.model import SEDModel

# Create galaxy instance and customize
galaxy = SEDModel.create_galaxy(
    ssp_model='bc2003_hr_stelib_chab_neb_2000r',
    sfh_type='exponential',
    dal_law='calzetti'
)
galaxy.add_dust_emission()  # Add dust emission

# Create AGN instance
agn = SEDModel.create_agn(agn_components=['dsk', 'blr', 'nlr', 'feii'])
agn.add_torus_fann(name='clumpy201410tor')  # Add torus

# Assemble configuration
params = BayeSEDParams(input_type=0, input_file='observation/test/qso.txt', outdir='output')
params.add_galaxy(galaxy)
params.add_agn(agn)
bayesed.run(params)
```

For more detailed documentation and advanced usage, see [docs/BayeSED3.md](docs/BayeSED3.md).

**Comprehensive Examples:**

See [run_test2.py](run_test2.py) for comprehensive examples demonstrating the high-level Python interface, including:
- Galaxy and AGN fitting with various model configurations
- Advanced parameter settings and inference configuration
- Result loading and visualization
- Complete test cases recreating the original [`run_test.py`](run_test.py) examples

### Graphical User Interface (GUI)

Launch the GUI:
```
python bayesed_gui.py
```
The GUI provides an intuitive way to set up complex SED analysis scenarios with meaningful defaults.

![BayeSED3 GUI](BayeSED3_GUI.png)

## File Descriptions

- [`bayesed/`](bayesed/): Python package providing high-level interface to BayeSED3
  - [`core.py`](bayesed/core.py): Main interface classes (`BayeSEDInterface`, `BayeSEDParams`)
  - [`results/`](bayesed/results/): **Enhanced BayeSEDResults** with intelligent scope management, 2-5x performance improvements, advanced plotting (`plot_posterior_free`, `plot_posterior_derived`, `plot_bestfit`), analytics (`compute_parameter_correlations`, `get_parameter_statistics`), and comprehensive error handling
  - [`model.py`](bayesed/model.py): Model configuration classes (`SEDModel`)
  - [`data.py`](bayesed/data.py): Data handling classes (`SEDObservation`, `PhotometryObservation`, `SpectrumObservation`)
  - [`params.py`](bayesed/params.py): Parameter configuration classes
  - [`utils.py`](bayesed/utils.py): Utility functions for data preparation and filter management
  - [`plotting.py`](bayesed/plotting.py): Plotting functions for visualization
- [`bayesed_gui.py`](bayesed_gui.py): Graphical User Interface for BayeSED3
- [`run_test.py`](run_test.py): Script to run BayeSED3 examples using low-level Python interface (direct parameter construction)
- [`run_test2.py`](run_test2.py): Comprehensive examples demonstrating the high-level Python interface (using `BayeSEDInterface`, `BayeSEDParams`, `SEDModel`, etc.)
- [`requirements.txt`](requirements.txt): List of Python dependencies
- [`observation/test/`](observation/test/): Contains test data and configuration files
- [`bin/`](bin/): Contains BayeSED3 executables for different platforms
- [`nets/`](nets/): Contains [Fast Artificial Neural Network (FANN)](https://github.com/libfann/fann) and [Approximate K-Nearest Neighbors (AKNN)](http://www.cs.umd.edu/~mount/ANN/) models for SED emulation
- [`data/`](data/): other data files used by BayeSED3

## System Compatibility

- Linux: x86_64 architecture
- macOS: x86_64 architecture (ARM supported via Rosetta 2)
- Windows: Supported through Windows Subsystem for Linux (WSL)

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contributions

Issues and pull requests are welcome. Please make sure to update tests before submitting a pull request.

## Citation

The further development of BayeSED needs your support. If BayeSED has been of benefit to you, either directly or indirectly, please consider citing our papers:
- [Han, Y., & Han, Z. 2012, ApJ, 749, 123](https://ui.adsabs.harvard.edu/abs/2012ApJ...749..123H/abstract)
- [Han, Y., & Han, Z. 2014, ApJS, 215, 2](https://ui.adsabs.harvard.edu/abs/2014ApJS..215....2H/abstract)
- [Han, Y., & Han, Z. 2019, ApJS, 240, 3](https://ui.adsabs.harvard.edu/abs/2019ApJS..240....3H/abstract)
- [Han, Y., Fan, L., Zheng, X. Z., Bai, J.-M., & Han, Z. 2023, ApJS, 269, 39](https://ui.adsabs.harvard.edu/abs/2023ApJS..269...39H/abstract)
- Han, Y., et al. 2024a, in prep.

## More Information

For more information about [MultiNest](https://github.com/farhanferoz/MultiNest), please refer to the [README_multinest.txt](README_multinest.txt) file.

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=hanyk/BayeSED3&type=Date)](https://star-history.com/#hanyk/BayeSED3&Date)
