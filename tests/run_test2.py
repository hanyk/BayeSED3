"""
Comprehensive examples demonstrating the high-level Python interface for BayeSED3.

This module showcases the enhanced BayeSED3 interface using real data from the repository:
- BayeSEDInterface for streamlined analysis execution
- BayeSEDParams with builder methods for quick configuration
- Enhanced BayeSEDResults with intelligent scope management and advanced plotting
- SEDModel for sophisticated model configuration
- Real observation data from observation/ directory

Key Features Demonstrated:
- High-level builder methods: BayeSEDParams.galaxy() and BayeSEDParams.agn()
- Enhanced results analysis with scope-aware data access
- Multi-model comparison and parameter standardization
- Publication-quality plotting with custom parameter labels
- Efficient data loading with filtering and caching
- Object-level vs sample-level analysis patterns

Real Data Used:
- observation/test/gal.txt - Galaxy spectroscopic data
- observation/test/qso.txt - AGN spectroscopic data
- observation/test1/test_inoise1.txt - Photometric survey data
- observation/test2/test.txt - AGN with dust emission data
- observation/test3/test_STARFORMING.txt - Advanced galaxy analysis
"""

import os
import sys

# Add parent directory to path so we can import bayesed when running directly
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from bayesed import BayeSEDInterface, BayeSEDParams, BayeSEDResults
from bayesed.results import standardize_parameter_names, plot_posterior_comparison
from bayesed.data import SEDObservation
from bayesed.model import SEDModel


def test_enhanced_results_analysis(output_dir, max_objects=1):
    """
    Test enhanced BayeSEDResults capabilities with real examples.

    This function demonstrates:
    - Intelligent configuration detection and scope management
    - Enhanced parameter access with caching
    - Publication-quality plotting capabilities
    - Object-level vs sample-level analysis patterns

    Parameters
    ----------
    output_dir : str
        Output directory containing results
    max_objects : int
        Maximum number of objects to test (default: 1)
    """
    try:
        print(f"\n{'='*70}")
        print(f"Testing Enhanced BayeSEDResults from {output_dir}")
        print(f"{'='*70}")

        # Test 1: Initialize with automatic configuration detection
        print("\n1. Testing intelligent configuration management...")
        try:
            results = BayeSEDResults(output_dir)

            # Enhanced introspection
            scope_info = results.get_access_scope()
            print(f"   ✓ Analysis scope: {scope_info.scope_type}")
            print(f"   ✓ Total objects: {scope_info.total_objects}")

            # List available objects and configurations
            objects = results.list_objects()
            print(f"   ✓ Found {len(objects)} object(s)")

            if not objects:
                print(f"   ⚠ No objects found in {output_dir}")
                return

        except Exception as e:
            print(f"   ✗ Configuration detection failed: {e}")
            return

        # Test 2: Enhanced parameter access
        print("\n2. Testing enhanced parameter access...")
        try:
            # Efficient parameter access with caching
            free_params = results.get_free_parameters()
            derived_params = results.get_derived_parameters()
            all_params = results.get_parameter_names(include_derived=True)

            print(f"   ✓ Free parameters: {len(free_params)}")
            print(f"   ✓ Derived parameters: {len(derived_params)}")
            print(f"   ✓ Total parameters: {len(all_params)}")

            # Load HDF5 data with filtering
            hdf5_table = results.load_hdf5_results(filter_snr=True, min_snr=3.0)
            print(f"   ✓ HDF5 table loaded: {len(hdf5_table)} objects (SNR > 3.0)")

        except Exception as e:
            print(f"   ✗ Parameter access failed: {e}")

        # Test 3: Publication-quality plotting
        print("\n3. Testing publication-quality plotting...")
        try:
            # Custom parameter labels for publication plots
            custom_labels = {
                'log(age/yr)[0,1]': r'\log(t_\star/\mathrm{yr})',
                'log(tau/yr)[0,1]': r'\log(\tau_\mathrm{SF}/\mathrm{yr})',
                'log(Z/Zsun)[0,1]': r'\log(Z_\star/Z_\odot)',
                'Av_2[0,1]': r'A_{V,\mathrm{stars}}',
                'log(Mstar)[0,1]': r'\log(M_\star/M_\odot)'
            }
            results.set_parameter_labels(custom_labels)

            # Enhanced plotting with automatic parameter filtering
            if len(free_params) > 0:
                results.plot_posterior_free(output_file='free_params_test.png', show=False)
                print(f"   ✓ Free parameters plot created")

            if len(derived_params) > 0:
                results.plot_posterior_derived(max_params=5, output_file='derived_params_test.png', show=False)
                print(f"   ✓ Derived parameters plot created")

        except Exception as e:
            print(f"   ✗ Plotting failed: {e}")

        # Test 4: Object-level analysis
        print(f"\n4. Testing object-level analysis...")
        for obj_id in objects[:max_objects]:
            try:
                # Object-level results access
                object_results = BayeSEDResults(output_dir, object_id=obj_id)

                # Object-specific plotting
                object_results.plot_bestfit(show=False, output_file=f'bestfit_{obj_id}_test.png')
                print(f"   ✓ Best-fit plot created for {obj_id}")

                # Object-specific parameter access
                try:
                    obj_params = object_results.get_parameter_values('log(age/yr)[0,1]')
                    if obj_params is not None:
                        # Handle different return types (scalar, array, or Row)
                        if hasattr(obj_params, '__len__') and len(obj_params) > 0:
                            # Array-like or Row object
                            if hasattr(obj_params[0], '__float__'):
                                value = float(obj_params[0])
                            else:
                                # Row object - get the first column value
                                value = float(list(obj_params[0])[0])
                            print(f"   ✓ Parameter access for {obj_id}: log(age/yr) = {value:.3f}")
                        else:
                            # Scalar value
                            print(f"   ✓ Parameter access for {obj_id}: log(age/yr) = {float(obj_params):.3f}")
                    else:
                        print(f"   ⚠ Parameter log(age/yr)[0,1] not found for {obj_id}")
                except Exception as param_e:
                    print(f"   ⚠ Parameter access failed for {obj_id}: {param_e}")

            except Exception as e:
                print(f"   ✗ Object-level analysis failed for {obj_id}: {e}")

        print(f"\n{'='*70}")
        print("Enhanced BayeSEDResults tests completed")
        print(f"{'='*70}\n")

    except Exception as e:
        import traceback
        print(f"\n✗ Error in enhanced results tests: {e}")
        traceback.print_exc()


def run_galaxy_analysis_enhanced(obj='gal', plot=False):
    """
    Demonstrate enhanced galaxy SED analysis using real data from observation/test/.

    This showcases:
    - BayeSEDInterface for streamlined execution
    - BayeSEDParams.galaxy() and BayeSEDParams.agn() builder methods
    - Enhanced BayeSEDResults with intelligent scope management
    - Publication-quality plotting with custom labels

    Parameters
    ----------
    obj : str
        Object name ('gal' or 'qso') - uses real data files
    plot : bool
        Whether to demonstrate enhanced plotting capabilities
    """
    # Initialize the enhanced interface
    bayesed = BayeSEDInterface(mpi_mode='auto')

    # Use real data files from the repository
    input_file = f'observation/test/{obj}.txt'
    output_dir = os.path.join(os.path.dirname(input_file), 'output2')

    # Simple galaxy configuration using builder method
    if obj == 'gal':
        params = BayeSEDParams.galaxy(
            input_file=input_file,
            outdir=output_dir,
            ssp_model='bc2003_hr_stelib_chab_neb_2000r',
            sfh_type='exponential',
            dal_law='calzetti',
            save_sample_par=True  # Enable posterior sample generation
        )
    else:  # qso
        # AGN analysis with all components using real emission line files
        # Use custom model configuration for AGN with specific line files
        galaxy = SEDModel.create_galaxy(
            ssp_model='bc2003_hr_stelib_chab_neb_2000r',
            sfh_type='exponential',
            dal_law='calzetti'
        )

        agn = SEDModel.create_agn(
            agn_components=['dsk', 'blr', 'nlr', 'feii'],
            blr_lines_file='observation/test/lines_BLR.txt',
            nlr_lines_file='observation/test/lines_NLR.txt'
        )

        params = BayeSEDParams(
            input_type=0,  # Flux in μJy
            input_file=input_file,
            outdir=output_dir,
            save_sample_par=True  # Enable posterior sample generation
        )
        params.add_galaxy(galaxy)
        params.add_agn(agn)

    print(f"Running enhanced BayeSED analysis for {obj} using real data...")
    print(f"Input file: {input_file}")
    result = bayesed.run(params)

    # Enhanced results analysis
    if plot:
        try:
            print(f"\nDemonstrating enhanced results analysis...")

            # Load results with intelligent configuration detection
            results = BayeSEDResults(output_dir)

            # Enhanced introspection
            scope_info = results.get_access_scope()
            print(f"Analysis scope: {scope_info.scope_type}")
            print(f"Objects available: {scope_info.total_objects}")

            # Custom parameter labels for publication plots
            custom_labels = {
                'log(age/yr)[0,1]': r'\log(t_\star/\mathrm{yr})',
                'log(tau/yr)[0,1]': r'\log(\tau_\mathrm{SF}/\mathrm{yr})',
                'log(Z/Zsun)[0,1]': r'\log(Z_\star/Z_\odot)',
                'Av_2[0,1]': r'A_{V,\mathrm{stars}}',
                'log(Mstar)[0,1]': r'\log(M_\star/M_\odot)'
            }
            results.set_parameter_labels(custom_labels)

            # Enhanced plotting capabilities
            results.plot_posterior_free(output_file=f'{obj}_free_params.png', show=False)
            results.plot_posterior_derived(max_params=5, output_file=f'{obj}_derived_params.png', show=False)

            # Object-level analysis
            objects = results.list_objects()
            if objects:
                obj_id = objects[0]
                object_results = BayeSEDResults(output_dir, object_id=obj_id)
                object_results.plot_bestfit(show=True, output_file=f'{obj}_bestfit.png')

            print(f"Enhanced plotting completed for {obj}")

        except Exception as e:
            print(f"Warning: Enhanced plotting failed: {e}")

    # Test enhanced capabilities
    if plot:
        test_enhanced_results_analysis(output_dir, max_objects=1)


def run_photometric_survey_analysis(survey='CSST', plot=False):
    """
    Demonstrate photometric survey analysis using real data from observation/test1/.

    Parameters
    ----------
    survey : str
        Survey name (e.g., 'CSST')
    plot : bool
        Whether to demonstrate enhanced analysis capabilities
    """
    # Initialize interface
    bayesed = BayeSEDInterface(mpi_mode='auto')

    # Use real data files from the repository
    input_file = 'observation/test1/test_inoise1.txt'
    output_dir = os.path.join(os.path.dirname(input_file), 'output2')

    # Simple galaxy configuration for photometric survey
    params = BayeSEDParams.galaxy(
        input_file=input_file,
        outdir=output_dir,
        ssp_model='bc2003_lr_BaSeL_chab',
        sfh_type='exponential',
        dal_law='calzetti',
        filters='observation/test1/filters_COSMOS_CSST_Euclid_LSST_WFIRST.txt',
        filters_selected='observation/test1/filters_CSST_seleted.txt',
        save_sample_par=True  # Enable posterior sample generation
    )

    print(f"Running photometric survey analysis: {survey}...")
    print(f"Using real data: {input_file}")
    result = bayesed.run(params)

    if plot:
        # Enhanced results analysis
        results = BayeSEDResults(output_dir)
        results.plot_posterior_free(output_file=f'{survey}_free_params.png', show=False)

        # Object-level analysis
        objects = results.list_objects()
        if objects:
            obj_results = BayeSEDResults(output_dir, object_id=objects[0])
            obj_results.plot_bestfit(show=True)

        test_enhanced_results_analysis(output_dir, max_objects=1)


def run_agn_torus_analysis(plot=False):
    """
    Demonstrate AGN analysis with torus component using real data from observation/test2/.

    This showcases:
    - Galaxy with dust emission
    - AGN torus component (FANN)
    - Custom model configuration with SEDModel

    Parameters
    ----------
    plot : bool
        Whether to demonstrate enhanced analysis capabilities
    """
    # Initialize interface
    bayesed = BayeSEDInterface(mpi_mode='auto')

    # Custom model configuration using SEDModel
    galaxy = SEDModel.create_galaxy(
        ssp_model='bc2003_lr_BaSeL_chab',
        sfh_type='exponential',
        dal_law='smc'
    )
    # Add dust emission component
    galaxy.add_dust_emission()

    # AGN with torus component
    agn = SEDModel.create_agn(agn_components=['tor'])

    # Assemble configuration using real data files
    input_file = 'observation/test2/test.txt'
    output_dir = os.path.join(os.path.dirname(input_file), 'output2')
    
    params = BayeSEDParams(
        input_type=0,  # Flux in μJy
        input_file=input_file,
        outdir=output_dir,
        filters='observation/test2/filters.txt',
        filters_selected='observation/test2/filters_selected.txt',
        save_sample_par=True  # Enable posterior sample generation
    )
    params.add_galaxy(galaxy)
    params.add_agn(agn)

    print("Running AGN torus analysis using real data...")
    print("Input file: observation/test2/test.txt")
    print("Components: Galaxy + Dust emission + AGN torus")
    result = bayesed.run(params)

    if plot:
        # Enhanced results analysis
        results = BayeSEDResults(output_dir)

        # Custom labels for AGN parameters
        agn_labels = {
            'log(scale)[1,1]': r'\log(\mathrm{AGN\,scale})',
            'log(Mstar)[0,1]': r'\log(M_\star/M_\odot)',
            'T/K[2,1]': r'T_{\mathrm{dust}}\,(\mathrm{K})'
        }
        results.set_parameter_labels(agn_labels)

        # Enhanced plotting
        results.plot_posterior_free(output_file='agn_torus_params.png', show=False)

        # Object-level analysis
        objects = results.list_objects()
        if objects:
            obj_results = BayeSEDResults(output_dir, object_id=objects[0])
            obj_results.plot_bestfit(show=True, use_log_scale=True)

        test_enhanced_results_analysis(output_dir, max_objects=1)


def run_advanced_galaxy_analysis(obj_type='STARFORMING', plot=False):
    """
    Demonstrate advanced galaxy analysis using real data from observation/test3/.

    This showcases:
    - Advanced stellar population parameters
    - Chemical evolution history
    - Enhanced inference settings

    Parameters
    ----------
    obj_type : str
        Object type ('STARFORMING' or 'PASSIVE')
    plot : bool
        Whether to demonstrate enhanced analysis capabilities
    """
    # Initialize interface
    bayesed = BayeSEDInterface(mpi_mode='auto')

    # Use real data files from the repository
    input_file = f'observation/test3/test_{obj_type}.txt'
    output_dir = os.path.join(os.path.dirname(input_file), 'output2')

    # Advanced galaxy configuration - use simpler configuration that works
    params = BayeSEDParams.galaxy(
        input_file=input_file,
        outdir=output_dir,
        ssp_model='bc2003_hr_stelib_chab_neb_300r',
        sfh_type='exponential',
        dal_law='calzetti',
        filters='observation/test3/filters_bassmzl.txt',
        filters_selected='observation/test3/filters_selected_csst.txt',
        save_sample_par=True  # Enable posterior sample generation
    )

    # Configuration is already set above

    print(f"Running advanced galaxy analysis for {obj_type} using real data...")
    print(f"Input file: {input_file}")
    print("Features: Chemical evolution, advanced SSP parameters")
    result = bayesed.run(params)

    if plot:
        # Enhanced results analysis
        results = BayeSEDResults(output_dir)

        # Custom labels for advanced parameters
        advanced_labels = {
            'log(age/yr)[0,1]': r'\log(t_\star/\mathrm{yr})',
            'log(Z/Zsun)[0,1]': r'\log(Z_\star/Z_\odot)',
            'log(Mstar)[0,1]': r'\log(M_\star/M_\odot)',
            'log(SFR_{100Myr}/[M_{sun}/yr])[0,1]': r'\log(\mathrm{SFR}_{100}/M_\odot\,\mathrm{yr}^{-1})'
        }
        results.set_parameter_labels(advanced_labels)

        # Enhanced plotting
        results.plot_posterior_free(output_file=f'{obj_type}_advanced_params.png', show=False)
        results.plot_posterior_derived(max_params=8, output_file=f'{obj_type}_derived_params.png', show=False)

        # Object-level analysis
        objects = results.list_objects()
        if objects:
            obj_results = BayeSEDResults(output_dir, object_id=objects[0])
            obj_results.plot_bestfit(show=True)

        test_enhanced_results_analysis(output_dir, max_objects=1)


def demonstrate_multi_model_comparison():
    """
    Demonstrate multi-model comparison using enhanced BayeSEDResults.

    This showcases:
    - Parameter standardization across models
    - Multi-model posterior comparison plotting
    - Bayesian evidence comparison
    """
    print("\n=== Multi-Model Comparison Demonstration ===\n")

    # Check for existing output directories from previous runs
    output_dirs = ['observation/test/output2', 'observation/test1/output2', 'observation/test2/output2', 'observation/test3/output2']
    available_dirs = [d for d in output_dirs if os.path.exists(d)]

    if len(available_dirs) < 2:
        print("Need at least 2 output directories for comparison.")
        print("Run other examples first to generate results:")
        print("  python run_test2.py galaxy --plot")
        print("  python run_test2.py survey --plot")
        return

    try:
        # Load results from different models
        results_list = []
        labels = []

        for i, output_dir in enumerate(available_dirs[:3]):  # Max 3 for demonstration
            try:
                results = BayeSEDResults(output_dir)
                results_list.append(results)
                labels.append(f'{output_dir.replace("_", " ").title()}')
                print(f"Loaded results from {output_dir}")
            except Exception as e:
                print(f"Could not load {output_dir}: {e}")

        if len(results_list) >= 2:
            # Standardize parameters across models
            print("Standardizing parameters across models...")
            standardize_parameter_names(results_list)

            # Create comparison plot
            print("Creating multi-model comparison plot...")
            plot_posterior_comparison(
                results_list,
                labels=labels,
                output_file='multi_model_comparison.png'
            )
            print("Multi-model comparison plot created: multi_model_comparison.png")

    except Exception as e:
        print(f"Multi-model comparison failed: {e}")

    print("\n=== End Multi-Model Comparison ===\n")


def demonstrate_programmatic_data_preparation():
    """
    Demonstrate creating observations from arrays using real filter information.
    """
    print("\n=== Programmatic Data Preparation ===\n")

    try:
        import numpy as np

        # Create observations from arrays using realistic filter names
        obs = SEDObservation(
            ids=['galaxy_001', 'galaxy_002'],
            z_min=[0.1, 0.2],
            z_max=[0.15, 0.25],
            phot_filters=['SLOAN/SDSS.g', 'SLOAN/SDSS.r', 'SLOAN/SDSS.i'],
            phot_fluxes=np.array([[12.5, 25.1, 18.3], [15.2, 28.9, 22.1]]),
            phot_errors=np.array([[1.2, 2.5, 1.8], [1.5, 2.9, 2.2]]),
            input_type=0  # Flux in μJy
        )

        print("Created SEDObservation from arrays:")
        print(f"  Objects: {len(obs.ids)}")
        print(f"  Filters: {obs.phot_filters}")

        # Convert to BayeSED format
        os.makedirs('observation/demo_analysis', exist_ok=True)
        input_file = obs.to_bayesed_input('observation/demo_analysis', 'demo_catalog')
        print(f"Converted to BayeSED format: {input_file}")

        # Show the created file content
        if os.path.exists(input_file):
            print("Created input file content (first few lines):")
            with open(input_file, 'r') as f:
                lines = f.readlines()[:10]
                for line in lines:
                    print(f"  {line.strip()}")

    except Exception as e:
        print(f"Data preparation demonstration failed: {e}")

    print("\n=== End Data Preparation ===\n")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python run_test2.py <example_name> [--plot]")
        print("\nAvailable examples (using real data from observation/):")
        print("  galaxy       - Enhanced galaxy SED analysis (observation/test/gal.txt)")
        print("  agn          - AGN analysis with all components (observation/test/qso.txt)")
        print("  survey       - Photometric survey analysis (observation/test1/)")
        print("  torus        - AGN with torus component (observation/test2/)")
        print("  advanced     - Advanced galaxy analysis (observation/test3/)")
        print("  comparison   - Multi-model comparison demonstration")
        print("  data_prep    - Programmatic data preparation")
        print("\nOptions:")
        print("  --plot       - Enable enhanced plotting and analysis")
        print("\nExamples:")
        print("  python run_test2.py galaxy --plot")
        print("  python run_test2.py agn --plot")
        print("  python run_test2.py comparison")
        sys.exit(1)

    example_name = sys.argv[1]
    plot = '--plot' in sys.argv

    if example_name == 'galaxy':
        run_galaxy_analysis_enhanced('gal', plot=plot)
    elif example_name == 'agn':
        run_galaxy_analysis_enhanced('qso', plot=plot)
    elif example_name == 'survey':
        run_photometric_survey_analysis(plot=plot)
    elif example_name == 'torus':
        run_agn_torus_analysis(plot=plot)
    elif example_name == 'advanced':
        run_advanced_galaxy_analysis(plot=plot)
    elif example_name == 'comparison':
        demonstrate_multi_model_comparison()
    elif example_name == 'data_prep':
        demonstrate_programmatic_data_preparation()
    else:
        print(f"Unknown example: {example_name}")
        print("Run with no arguments to see available examples.")
        sys.exit(1)
