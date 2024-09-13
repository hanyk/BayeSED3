# BayeSED

BayeSED is a general tool for the full Bayesian interpretation of the spectral energy distributions (SEDs) of galaxies. Given the multi-band photometries and spectrum of galaxies, it can be used for the Bayesian parameter estimation by posteriori probability distributions (PDFs) and the Bayesian SED model comparison by Bayesian evidence. Except for the build-in SED models (stellar population synthesis models, blackbody, greybody and powerlaw), other SED models can be emulated with machine learning techniques. The linear combination of all selected SED model components will then be used for the full Bayesian interpretation of the observational SEDs of galaxies.


## Installation Instructions

1. Install OpenMPI (if not already installed)

   The BayeSED interface will automatically download and install OpenMPI 4.1.6. If you want to install it manually, follow these steps:

   ```
   cd BayeSED3
   wget https://download.open-mpi.org/release/open-mpi/v4.1/openmpi-4.1.6.tar.gz
   tar xzvf openmpi-4.1.6.tar.gz
   cd openmpi-4.1.6
   ./configure --prefix=../openmpi
   make
   make install
   ```

2. Install Python dependencies

   ```
   pip install -r requirements.txt
   ```

3. Install HDF5 (optional)

   - macOS: `brew install hdf5`
   - Linux: `apt install h5utils`

## Usage

1. Run examples

   ```
   python run_test.py gal
   python run_test.py qso
   ```

   This will run examples for galaxies (gal) and quasars (qso) respectively.

2. Custom runs

   You can modify the `run_test.py` file to customize BayeSED's running parameters.

## File Descriptions

- `bayesed.py`: Main interface class for BayeSED
- `run_test.py`: Script to run BayeSED examples
- `requirements.txt`: List of Python dependencies
- `observation/test/`: Contains test data and configuration files
- `bin/`: Contains BayeSED executables for different platforms

## Notes

- Only x86_64 version of BayeSED3 for Linux and MacOS is available, ARM architecture is supported for MacOS with the help of rosetta2, Windows is supported with the help of WSL.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contributions

Issues and pull requests are welcome. Please make sure to update tests before submitting a pull request.

## More Information

For more information about MultiNest, please refer to the [README_multinest.txt](README_multinest.txt) file.
