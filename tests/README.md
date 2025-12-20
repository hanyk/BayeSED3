# BayeSED3 Python Interface Examples

This directory contains Python examples extracted from the main README.md file, organized as individual test files for easy execution and learning.

## Available Examples

### 1. `quick_start.py` - Basic Usage
**What it demonstrates:**
- Basic BayeSED3 Python interface setup
- Simple galaxy SED fitting
- Loading and analyzing results
- Parameter access and plotting
- Object-level analysis

**Key features:**
- `BayeSEDInterface` initialization
- `BayeSEDParams.galaxy()` for quick galaxy setup
- `BayeSEDResults` for result analysis
- Custom parameter labels and plotting

### 2. `test_agn_fitting.py` - AGN Analysis
**What it demonstrates:**
- AGN SED fitting with all components
- Galaxy host + AGN component modeling

**Key features:**
- `BayeSEDParams.agn()` with multiple components
- Accretion disk, BLR, NLR, and FeII modeling

### 3. `test_data_arrays.py` - Working with Arrays
**What it demonstrates:**
- Creating observations from numpy arrays
- Synthetic data preparation
- Filter management from SVO database

**Key features:**
- `SEDObservation` class for data handling
- SVO filter download and preparation
- Array-to-BayeSED format conversion

### 4. `test_custom_model.py` - Advanced Modeling
**What it demonstrates:**
- Custom model configuration
- Dust emission modeling
- Complex AGN component setup

**Key features:**
- `SEDModel.create_galaxy()` and `SEDModel.create_agn()`
- Dust emission components
- Custom parameter assembly

### 5. `test_multi_model_comparison.py` - Model Comparison
**What it demonstrates:**
- Comparing different SED models
- Bayesian evidence analysis
- Parameter standardization across models

**Key features:**
- Multiple model runs with same data
- `standardize_parameter_names()` function
- `plot_posterior_comparison()` for visualization
- Manual Bayes factor calculation

### 6. `test_advanced_analytics.py` - Advanced Analysis
**What it demonstrates:**
- Parameter correlation analysis
- Statistical computations
- GetDist integration for advanced plotting

**Key features:**
- `compute_parameter_correlations()`
- `get_parameter_statistics()`
- GetDist samples and triangle plots
- Evidence analysis with different return formats

## Running the Examples

### Run Individual Examples
```bash
# Run a specific example
python tests/quick_start.py
python tests/test_agn_fitting.py
python tests/test_advanced_analytics.py
```

### Run All Examples
```bash
# Run all examples with summary
python tests/run_all_examples.py
```

## Prerequisites

Before running these examples, ensure you have:

1. **Completed BayeSED3 installation** (see main README.md)
2. **Required test data** in `observation/test/` directory
3. **Python dependencies** installed:
   ```bash
   pip install -r requirements.txt
   ```

## Expected Output

Each example will:
- Print progress messages
- Generate plots (saved as PNG files)
- Create output directories with results
- Display analysis summaries

## Notes

- Examples use real test data from `observation/test/`
- Some examples require significant computation time
- Generated plots and output files will be created in the working directory
- Examples demonstrate both basic and advanced BayeSED3 features

## Troubleshooting

If examples fail:
1. Check that BayeSED3 is properly installed
2. Verify test data exists in `observation/test/`
3. Ensure all Python dependencies are installed
4. Check that OpenMPI is properly configured

For more detailed documentation, see the main [README.md](../README.md) file.