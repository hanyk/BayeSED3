"""
Example: Recreating run_test.py tests using new design with separation of concerns.

This module demonstrates the new design with:
- Separation of data, model, and inference concerns
- GalaxyInstance and AGNInstance for detailed model component control
- SEDInference for Bayesian inference configuration and execution

Comparison:
- run_test.py uses direct parameter construction (low-level interface)
- This module uses the new design: GalaxyInstance/AGNInstance + SEDInference

Key Design Points:
- SEDModel.create_galaxy() is used to create galaxy instances (with detailed control for ssp_i1, ssp_k, sfh_itype_ceh, etc.)
- SEDModel.create_agn() is used to create AGN instances
- SEDModel instance methods (set_igm(), set_cosmology(), etc.) for additional model settings
- SEDInference handles all inference configuration (MultiNest, GSL, NNLM, etc.)
- Clear separation: Data → Model → Inference

Architecture Note:
- BayeSEDParams.galaxy() and agn() internally use SEDModel.create_galaxy() and create_agn()
- This ensures consistency: both builder methods and direct SEDModel usage produce identical results
- Use BayeSEDParams.galaxy()/agn() for quick setup, or SEDModel.create_galaxy()/create_agn() for complex configurations

Key Features Demonstrated:
- SEDModel.create_galaxy() with all SSP, SFH, and DAL parameters
- SEDModel.create_agn() with flexible component selection
- AGN components: Disk (BBB/AGN/FANN/AKNN), BLR, NLR, FeII, and Torus (FANN or AKNN)
- SEDInference.multinest() and SEDInference.nnlm() for inference configuration
- Method chaining for gradual extension
- Automatic ID and igroup management
- Automatic DAL and KIN parameter handling
"""

import os
import sys

# Add parent directory to path so we can import bayesed when running directly
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from bayesed import (BayeSEDParams, BayeSEDResults,
                     SysErrParams, ZParams, RenameParams,
                     GreybodyParams, FANNParams, NNLMParams, SNRmin1Params, RDFParams,
                     plot_bestfit)
from bayesed.data import SEDObservation
from bayesed.model import SEDModel
from bayesed.inference import SEDInference


def _test_results_class(output_dir, max_objects=1):
    """
    Test BayeSEDResults class methods with real examples.
    
    This function tests:
    - list_objects(): List all objects in output directory
    - get_evidence(): Get Bayesian evidence from parameter table
    - get_bestfit_spectrum(): Load best-fit spectrum from FITS file
    - get_posterior_samples(): Load posterior distribution samples
    - load_hdf5_results(): Load parameter table from HDF5
    - plot_bestfit(): Plot best-fit SED
    
    Parameters
    ----------
    output_dir : str
        Output directory containing results
    max_objects : int
        Maximum number of objects to test (default: 1)
    """
    try:
        print(f"\n{'='*70}")
        print(f"Testing BayeSEDResults methods from {output_dir}")
        print(f"{'='*70}")
        
        # Test 1: Initialize results and list objects
        print("\n1. Testing list_objects()...")
        try:
            results = BayeSEDResults(output_dir)
            objects = results.list_objects()
            print(f"   ✓ Found {len(objects)} object(s)")
            if objects:
                print(f"   ✓ Object IDs: {objects[:5]}{'...' if len(objects) > 5 else ''}")
            else:
                print(f"   ⚠ No objects found in {output_dir}")
                return
        except Exception as e:
            print(f"   ✗ list_objects() failed: {e}")
            return
        
        # Test 2: Load HDF5 results (parameter table)
        print("\n2. Testing load_hdf5_results()...")
        params_table = None
        try:
            params_table = results.load_hdf5_results(filter_snr=False)
            print(f"   ✓ Loaded parameter table with {len(params_table)} objects")
            print(f"   ✓ Columns: {len(params_table.colnames)} ({', '.join(params_table.colnames[:5])}{'...' if len(params_table.colnames) > 5 else ''})")
            
            # Check for evidence columns (logZ and INSlogZ are always in the table)
            evidence_cols = []
            if 'logZ' in params_table.colnames:
                evidence_cols.append('logZ')
            if 'INSlogZ' in params_table.colnames:
                evidence_cols.append('INSlogZ')
            if evidence_cols:
                print(f"   ✓ Found evidence column(s): {evidence_cols}")
            
            # Check for error columns (logZerr and INSlogZerr are always in the table)
            error_cols = []
            if 'logZerr' in params_table.colnames:
                error_cols.append('logZerr')
            if 'INSlogZerr' in params_table.colnames:
                error_cols.append('INSlogZerr')
            if error_cols:
                print(f"   ✓ Found evidence error column(s): {error_cols}")
        except Exception as e:
            print(f"   ✗ load_hdf5_results() failed: {e}")
            import traceback
            traceback.print_exc()
        
        # Test 3: Get evidence
        print("\n3. Testing get_evidence()...")
        try:
            # Show all available evidence columns if we have the table
            if params_table is not None:
                evidence_cols = []
                if 'logZ' in params_table.colnames:
                    evidence_cols.append('logZ')
                if 'INSlogZ' in params_table.colnames:
                    evidence_cols.append('INSlogZ')
                if evidence_cols:
                    print(f"   ✓ Available evidence columns: {evidence_cols}")
                    # Show evidence errors if available
                    error_cols = []
                    if 'logZerr' in params_table.colnames:
                        error_cols.append('logZerr')
                    if 'INSlogZerr' in params_table.colnames:
                        error_cols.append('INSlogZerr')
                    if error_cols:
                        print(f"   ✓ Available error columns: {error_cols}")
            
            # Test without object_id (first object) - default uses INS
            evidence = results.get_evidence()
            if evidence is not None:
                print(f"   ✓ Evidence (first object, INS): {evidence:.4f}")
            else:
                print(f"   ⚠ Evidence not found (may not be in parameter table)")
            
            # Test with use_ins=False to get standard evidence
            evidence_standard = results.get_evidence(use_ins=False)
            if evidence_standard is not None:
                if evidence is None or evidence_standard != evidence:
                    print(f"   ✓ Evidence (first object, standard): {evidence_standard:.4f}")
            
            # Test with object_id if we have objects
            if objects:
                obj_id = objects[0]
                evidence_obj = results.get_evidence(object_id=obj_id)
                if evidence_obj is not None:
                    print(f"   ✓ Evidence (object {obj_id}, INS): {evidence_obj:.4f}")
        except Exception as e:
            print(f"   ✗ get_evidence() failed: {e}")
            import traceback
            traceback.print_exc()
        
        # Test with specific objects
        for obj_id in objects[:max_objects]:
            print(f"\n{'='*70}")
            print(f"Testing methods for object: {obj_id}")
            print(f"{'='*70}")
            
            obj_results = BayeSEDResults(output_dir, object_id=obj_id)
            
            # Test 4: Get best-fit spectrum
            print(f"\n4. Testing get_bestfit_spectrum() for {obj_id}...")
            try:
                bestfit_data = obj_results.get_bestfit_spectrum()
                if bestfit_data:
                    print(f"   ✓ Loaded best-fit spectrum")
                    print(f"   ✓ Data keys: {list(bestfit_data.keys())[:10]}{'...' if len(bestfit_data) > 10 else ''}")
                    # Check for common keys
                    common_keys = ['wavelength', 'flux', 'wavelength_rest', 'wavelength_obs']
                    found_keys = [k for k in bestfit_data.keys() if any(ck in k.lower() for ck in common_keys)]
                    if found_keys:
                        print(f"   ✓ Found wavelength/flux data: {found_keys[:3]}")
                else:
                    print(f"   ⚠ No best-fit spectrum data found")
            except FileNotFoundError:
                print(f"   ⚠ Best-fit FITS file not found for {obj_id}")
            except Exception as e:
                print(f"   ✗ get_bestfit_spectrum() failed: {e}")
            
            # Test 5: Get posterior samples
            print(f"\n5. Testing get_posterior_samples() for {obj_id}...")
            try:
                posterior_data = obj_results.get_posterior_samples()
                paramnames = posterior_data['paramnames']
                samples = posterior_data['samples']
                print(f"   ✓ Loaded posterior samples")
                print(f"   ✓ Parameters: {len(paramnames)}, Samples: {len(samples)}")
                if paramnames:
                    print(f"   ✓ Parameter names: {paramnames[:5]}{'...' if len(paramnames) > 5 else ''}")
                    # Show some statistics
                    if len(samples) > 0 and len(paramnames) > 0:
                        print(f"   ✓ Sample shape: {samples.shape}")
                        print(f"   ✓ First parameter range: [{samples[:, 0].min():.4f}, {samples[:, 0].max():.4f}]")
            except FileNotFoundError:
                print(f"   ⚠ Posterior sample files not found (save_sample_par may not be enabled)")
            except Exception as e:
                print(f"   ✗ get_posterior_samples() failed: {e}")
                import traceback
                traceback.print_exc()
            
            # Test 6: Test evidence for specific object
            print(f"\n6. Testing get_evidence() for {obj_id}...")
            try:
                evidence_ins = obj_results.get_evidence(object_id=obj_id, use_ins=True)
                evidence_standard = obj_results.get_evidence(object_id=obj_id, use_ins=False)
                if evidence_ins is not None:
                    print(f"   ✓ Evidence (INS): {evidence_ins:.4f}")
                if evidence_standard is not None and evidence_standard != evidence_ins:
                    print(f"   ✓ Evidence (standard): {evidence_standard:.4f}")
                if evidence_ins is None and evidence_standard is None:
                    print(f"   ⚠ Evidence not found in parameter table")
            except Exception as e:
                print(f"   ✗ get_evidence() failed: {e}")
        
        print(f"\n{'='*70}")
        print("BayeSEDResults class tests completed")
        print(f"{'='*70}\n")
        
    except Exception as e:
        import traceback
        print(f"\n✗ Error in results class tests: {e}")
        traceback.print_exc()


def _test_pdf_plotting(output_dir, max_objects=1):
    """
    Test PDF plotting functionality for results in output_dir.
    
    This function tests:
    - Loading posterior samples
    - GetDist-based PDF plotting (handles weighted nested sampling)
    - 1D PDF plotting for individual parameters with GetDist method
      (using plot_posterior_pdf with single param)
    
    Parameters
    ----------
    output_dir : str
        Output directory containing results
    max_objects : int
        Maximum number of objects to test (default: 1)
    """
    try:
        print(f"\nTesting PDF plotting from {output_dir}...")
        results = BayeSEDResults(output_dir)
        objects = results.list_objects()
        
        if not objects:
            print(f"  No objects found in {output_dir}")
            return
        
        # Test with first object(s)
        for obj_id in objects[:max_objects]:
            print(f"\n  Testing PDF plotting for object: {obj_id}")
            obj_results = BayeSEDResults(output_dir, object_id=obj_id)
            
            # Try to get posterior samples
            try:
                posterior_data = obj_results.get_posterior_samples()
                paramnames = posterior_data['paramnames']
                samples = posterior_data['samples']
                print(f"    Found {len(paramnames)} parameters, {len(samples)} samples")
                
                if len(paramnames) == 0:
                    print("    ⚠ No parameters found in posterior samples")
                    continue
                
                # Select a few key parameters if available
                # Filter out derived parameters (those ending with *)
                non_derived_params = [pn for pn in paramnames if not pn.endswith('*')]
                
                test_params = []
                priority_params = [
                    'log(age/yr)', 'log(Z/Zsun)', 'Av_2', 'log(Mstar)', 
                    'log(SFR', 'T/K', 'log(scale)'
                ]
                
                for p in priority_params:
                    matching = [pn for pn in non_derived_params if p in pn]
                    if matching:
                        test_params.append(matching[0])
                        if len(test_params) >= 4:
                            break
                
                # If no priority params found, use first few non-derived
                if not test_params:
                    test_params = non_derived_params[:min(4, len(non_derived_params))]
                
                print(f"    Testing with {len(test_params)} parameters: {test_params[:3]}...")

                # Test GetDist PDF plotting
                print("    Testing GetDist PDF plotting...")
                try:
                    obj_results.plot_posterior_pdf(
                        params=test_params,
                        method='getdist',
                        show=True,  # Show the plot
                        output_file=None
                    )
                    print("    ✓ GetDist PDF plotting test passed")
                except ImportError:
                    print("    ⚠ GetDist not available, skipping GetDist test")
                except Exception as e:
                    print(f"    ✗ GetDist PDF plotting test failed: {e}")
                
                # Test 1D PDF plotting for a single parameter (using plot_posterior_pdf with single param)
                if len(test_params) > 0:
                    print(f"    Testing 1D PDF plotting for: {test_params[0]}")
                    
                    # Test 1D PDF plotting with GetDist
                    print("    Testing 1D PDF plotting with GetDist...")
                    try:
                        obj_results.plot_posterior_pdf(
                            params=test_params[0],  # Single parameter for 1D plot
                            method='getdist',
                            show=True,  # Show the plot
                            output_file=None
                        )
                        print("    ✓ 1D PDF plotting with GetDist test passed")
                    except ImportError:
                        print("    ⚠ GetDist not available, skipping 1D GetDist test")
                    except Exception as e:
                        print(f"    ✗ 1D PDF plotting with GetDist test failed: {e}")
                
            except FileNotFoundError:
                print(f"    ⚠ No posterior sample files found for {obj_id}")
                print("    (save_sample_par may not be enabled)")
            except Exception as e:
                print(f"    ✗ Error loading posterior samples: {e}")
                
    except Exception as e:
        import traceback
        print(f"  Warning: PDF plotting test failed: {e}")
        traceback.print_exc()


def run_bayesed_example_class_based(obj, input_dir='observation/test', output_dir='output1', np=None, Ntest=None, plot=False):
    """
    Recreate run_bayesed_example() using new design with separation of concerns.
    
    This demonstrates:
    - SEDObservation for data handling
    - SEDModel for physical model configuration
    - SEDInference for Bayesian inference configuration and execution
    - GalaxyInstance and AGNInstance for model components
    - Fluent builder pattern with add_observation() and add_model()
    
    Parameters
    ----------
    obj : str
        Object name ('gal' or 'qso')
    input_dir : str
        Input directory containing observation files
    output_dir : str
        Output directory for results
    np : int, optional
        Number of MPI processes
    Ntest : int, optional
        Number of test objects to process
    plot : bool
        Whether to plot best-fit results after completion (default: False)
    """
    # Data: Create observation (for this example, we're using existing input files)
    # In a real scenario, you would create SEDObservation from arrays and call to_bayesed_input()
    # Here we're using existing files, so we'll set parameters directly
    params = BayeSEDParams(
        input_type=0,  # 0: flux in uJy
        input_file=f'{input_dir}/{obj}.txt',
        outdir=output_dir,
        save_bestfit=0,
        save_sample_par=True,
    )
    
    # Note: In practice, you would create SEDObservation and call params.add_observation(obs)
    
    # Model: Create galaxy instance with detailed SSP parameters
    # When you need detailed control (like ssp_i1, ssp_k, etc.), use SEDModel.create_galaxy()
    # Note: BayeSEDParams.galaxy() internally uses SEDModel.create_galaxy() for consistency
    galaxy = SEDModel.create_galaxy(
        ssp_model='bc2003_hr_stelib_chab_neb_2000r',
        sfh_type='exponential',  # itype_sfh=2
        dal_law='calzetti',  # Starburst (Calzetti2000)
        ssp_k=1,
        ssp_f_run=1,
        ssp_Nstep=1,
        ssp_i0=0,
        ssp_i1=1,  # Important parameter from run_test.py
        ssp_i2=0,
        ssp_i3=0
    )
    params.add_galaxy(galaxy)
    
    # Add rename parameter
    params.rename = [RenameParams(id=0, ireplace=1, name='Stellar+Nebular')]
    
    # Set systematic error in observations (data-related)
    params.sys_err_obs = SysErrParams(
        min=0.0,
        max=0.2
    )
    
    # For additional model settings (IGM, cosmology, etc.), create a separate SEDModel instance:
    # model = SEDModel()
    # model.set_igm(igm_model=1)
    # model.set_cosmology(H0=70.0, omigaA=0.7, omigam=0.3)
    # params.add_model(model)

    if obj == 'qso':
        # Create AGN instance with all components from run_test.py
        # Note: BayeSEDParams.agn() internally uses SEDModel.create_agn() for consistency
        agn = SEDModel.create_agn(
            agn_components=['dsk', 'blr', 'nlr', 'feii'],  # All AGN components (dsk=disk/BBB)
            blr_lines_file='observation/test/lines_BLR.txt',
            nlr_lines_file='observation/test/lines_NLR.txt'
        )
        params.add_agn(agn)  # Auto-assigns IDs

    # Inference: Configure Bayesian inference and execute
    inference = SEDInference()
    inference.multinest(nlive=40, efr=0.05, updInt=100, fb=2)
    
    print(f"Running BayeSED for {obj} object (using new design with separation of concerns)...")
    result = inference.run(params, mpi_mode='1', np=np, Ntest=Ntest)
    
    # Plot results if requested
    if plot:
        try:
            print(f"\nPlotting best-fit results from {output_dir}...")
            results = BayeSEDResults(output_dir)
            # List objects for this catalog (catalog_name is auto-detected)
            objects = results.list_objects()
            if objects:
                for obj_id in objects[:5]:  # Plot first 5 objects
                    print(f"  Plotting object: {obj_id}")
                    # Create results object for this specific object
                    obj_results = BayeSEDResults(output_dir, object_id=obj_id)
                    # output_file=None means it will be saved in the same folder as the FITS file
                    obj_results.plot_bestfit(show=True, output_file=None)
            else:
                print(f"  No objects found in output directory")
        except Exception as e:
            import traceback
            print(f"  Warning: Could not plot results: {e}")
            traceback.print_exc()
    
    # Test results class methods if requested
    if plot:
        _test_results_class(output_dir, max_objects=1)
        _test_pdf_plotting(output_dir, max_objects=1)


def run_bayesed_test1_class_based(survey, obs_file, np=None, Ntest=None, plot=False):
    """
    Recreate run_bayesed_test1() using new design with separation of concerns.
    
    Parameters
    ----------
    survey : str
        Survey name (e.g., 'CSST')
    obs_file : str
        Path to observation file
    np : int, optional
        Number of MPI processes
    Ntest : int, optional
        Number of test objects to process
    plot : bool
        Whether to plot best-fit results after completion (default: False)
    """
    # Data: Create observation with filter files and data quality settings
    # Note: In practice, you would create SEDObservation from arrays
    # Here we're using existing files, so we'll set parameters directly
    params = BayeSEDParams(
        input_type=1,  # 1: Input file contains observed photometric SEDs with AB magnitude
        input_file=obs_file,
        outdir='test1',
        save_bestfit=2,  # 2: Save in both fits and hdf5 formats
        save_sample_par=True,
        suffix=f'_{survey}',
    )
    params.filters = 'observation/test1/filters_COSMOS_CSST_Euclid_LSST_WFIRST.txt'
    params.filters_selected = f'observation/test1/filters_{survey}_seleted.txt'
    params.no_spectra_fit = True
    
    # Model: Create galaxy instance
    # For simple cases, SEDModel.create_galaxy() is sufficient
    galaxy = SEDModel.create_galaxy(
        ssp_model='bc2003_lr_BaSeL_chab',
        sfh_type='exponential',  # itype_sfh=2
        dal_law='calzetti'
    )
    params.add_galaxy(galaxy)
    
    # Set redshift prior (model-related)
    params.z = ZParams(iprior_type=1, min=0.0, max=4.0, nbin=40)
    
    # Alternative: For additional model settings (IGM, cosmology, priors, etc.):
    # model = SEDModel()
    # model.set_redshift_prior(iprior_type=1, min=0.0, max=4.0, nbin=40)
    # params.add_model(model)

    # Inference: Configure Bayesian inference and execute
    inference = SEDInference()
    inference.multinest(nlive=50, efr=0.1, updInt=1000)

    print(f"Running BayeSED for survey: {survey}, observation file: {obs_file} (using new design)...")
    result = inference.run(params, mpi_mode='1', np=np, Ntest=Ntest)
    
    # Plot results if requested
    if plot:
        try:
            print(f"\nPlotting best-fit results from test1...")
            results = BayeSEDResults('test1')
            objects = results.list_objects()
            if objects:
                for obj_id in objects[:3]:  # Plot first 3 objects
                    print(f"  Plotting object: {obj_id}")
                    # Create results object for this specific object
                    obj_results = BayeSEDResults('test1', object_id=obj_id)
                    # output_file=None means it will be saved in the same folder as the FITS file
                    obj_results.plot_bestfit(show=True, 
                                            output_file=None,
                                            filter_file=params.filters,
                                            filter_selection_file=params.filters_selected)
            else:
                print("  No objects found in output directory")
        except Exception as e:
            print(f"  Warning: Could not plot results: {e}")
    
    # Test results class methods if requested
    if plot:
        _test_results_class('test1', max_objects=1)
        _test_pdf_plotting('test1', max_objects=1)


def run_bayesed_test2_class_based(np=None, Ntest=None, plot=False):
    """
    Recreate run_bayesed_test2() using new design with separation of concerns.
    
    This demonstrates:
    - GalaxyInstance with dust emission (greybody)
    - AGNInstance with torus component (FANN)
    - SEDModel and SEDInference usage
    
    Parameters
    ----------
    np : int, optional
        Number of MPI processes
    Ntest : int, optional
        Number of test objects to process
    plot : bool
        Whether to plot best-fit results after completion (default: False)
    """
    # Data: Create observation with filter files
    params = BayeSEDParams(
        input_type=0,
        input_file='observation/test2/test.txt',
        outdir='test2',
        save_bestfit=0,
        save_sample_par=True,
    )
    params.filters = 'observation/test2/filters.txt'
    params.filters_selected = 'observation/test2/filters_selected.txt'
    
    # Model: Create galaxy instance with dust emission
    # When you need to add components incrementally, use SEDModel.create_galaxy()
    galaxy = SEDModel.create_galaxy(
        ssp_model='bc2003_lr_BaSeL_chab',
        sfh_type='exponential',
        dal_law='smc'  # Different from test1
    )
    
    # Add dust emission (greybody) - incremental component addition
    galaxy.add_dust_emission(
        model_type='greybody',
        iscalable=-2,
        w_min=1.0,
        w_max=1000.0,
        Nw=200
    )
    params.add_galaxy(galaxy)
    
    # Add FANN AGN torus component
    agn = SEDModel.create_agn(
        agn_components=['tor']  # Creates FANN torus with default name 'clumpy201410tor'
    )
    params.add_agn(agn)  # Auto-assigns IDs

    # Inference: Configure Bayesian inference and execute
    inference = SEDInference()
    inference.multinest(nlive=400, efr=0.1, updInt=1000, fb=2)

    print("Running BayeSED for test2 (using new design)...")
    print("  Components: Galaxy (SSP+SFH+DAL) + Greybody dust emission + FANN AGN torus")
    result = inference.run(params, mpi_mode='1', np=np, Ntest=Ntest)
    
    # Plot results if requested
    if plot:
        try:
            print(f"\nPlotting best-fit results from test2...")
            results = BayeSEDResults('test2')
            objects = results.list_objects()
            if objects:
                for obj_id in objects[:3]:  # Plot first 3 objects
                    print(f"  Plotting object: {obj_id}")
                    # Create results object for this specific object
                    obj_results = BayeSEDResults('test2', object_id=obj_id)
                    # output_file=None means it will be saved in the same folder as the FITS file
                    obj_results.plot_bestfit(show=True, 
                                            output_file=None,
                                            filter_file=params.filters,
                                            filter_selection_file=params.filters_selected,use_log_scale=True)
            else:
                print("  No objects found in output directory")
        except Exception as e:
            print(f"  Warning: Could not plot results: {e}")
    
    # Test results class methods if requested
    if plot:
        _test_results_class('test2', max_objects=1)
        _test_pdf_plotting('test2', max_objects=1)


def example_agn_components_demonstration():
    """
    Demonstration of different ways to use AGN components with the class-based interface.
    
    This function shows various patterns for creating AGN instances with different
    component combinations, demonstrating the flexibility of the interface.
    """
    print("\n=== AGN Components Demonstration ===\n")
    
    # Example 1: All components at once (simplest)
    print("Example 1: All AGN components (Disk, BLR, NLR, FeII)")
    agn1 = SEDModel.create_agn(
        agn_components=['dsk', 'blr', 'nlr', 'feii']  # All components
    )
    print(f"  Created AGN instance with: Disk={agn1.dsk is not None}, "
          f"BLR={agn1.blr is not None}, NLR={agn1.nlr is not None}, "
          f"FeII={agn1.feii is not None}, Torus={agn1.tor is not None}")
    
    # Example 2: Specific components only
    print("\nExample 2: Only Disk and BLR")
    agn2 = SEDModel.create_agn(
        agn_components=['dsk', 'blr']  # Only these components
    )
    print(f"  Created AGN instance with: Disk={agn2.dsk is not None}, "
          f"BLR={agn2.blr is not None}")
    
    # Example 3: Add components incrementally (method chaining)
    print("\nExample 3: Incremental component addition (method chaining)")
    agn3 = SEDModel.create_agn(agn_components=[])
    agn3.add_disk_bbb(name='bbb', w_min=0.1, w_max=10.0, Nw=1000)
    agn3.add_broad_line_region(file='observation/test/lines_BLR.txt', R=300, Nkin=3)
    agn3.add_feii_template(name='FeII', k=5, f_run=1)
    print(f"  Created AGN instance incrementally with: Disk={agn3.dsk is not None}, "
          f"BLR={agn3.blr is not None}, FeII={agn3.feii is not None}")
    
    # Example 4: Torus components (FANN and AKNN)
    print("\nExample 4: Torus components")
    
    # FANN torus
    agn4_fann = SEDModel.create_agn(agn_components=['tor'])
    print(f"  FANN torus (default): {agn4_fann.tor is not None}, "
          f"type={type(agn4_fann.tor).__name__ if agn4_fann.tor else None}")
    
    # AKNN torus
    agn4_aknn = SEDModel.create_agn(agn_components=[])
    agn4_aknn.add_torus_aknn(name='torus_aknn', k=1, f_run=1, eps=0.01)
    print(f"  AKNN torus: {agn4_aknn.tor is not None}, "
          f"type={type(agn4_aknn.tor).__name__ if agn4_aknn.tor else None}")
    
    # Example 5: All components including torus
    print("\nExample 5: All components including FANN torus")
    agn5 = SEDModel.create_agn(
        agn_components=['dsk', 'blr', 'nlr', 'feii', 'tor']  # All including torus
    )
    print(f"  Created AGN instance with all components: "
          f"Disk={agn5.dsk is not None}, BLR={agn5.blr is not None}, "
          f"NLR={agn5.nlr is not None}, FeII={agn5.feii is not None}, "
          f"Torus={agn5.tor is not None}")
    
    print("\n=== End AGN Components Demonstration ===\n")


def run_bayesed_test3_class_based(obj_type, itype, np=None, Ntest=None, plot=False):
    """
    Recreate run_bayesed_test3() using new design with separation of concerns.
    
    This demonstrates support for advanced parameters like itype_ceh and inference settings.
    
    Parameters
    ----------
    obj_type : str
        Object type ('STARFORMING' or 'PASSIVE')
    itype : str
        Input type ('phot', 'spec', or 'both')
    np : int, optional
        Number of MPI processes
    Ntest : int, optional
        Number of test objects to process
    plot : bool
        Whether to plot best-fit results after completion (default: False)
    """
    if obj_type == 'STARFORMING':
        input_file = 'observation/test3/test_STARFORMING.txt'
    elif obj_type == 'PASSIVE':
        input_file = 'observation/test3/test_PASSIVE.txt'
    else:
        raise ValueError(f"Unknown obj_type: {obj_type}")

    # Data: Create observation with filter files and data quality settings
    params = BayeSEDParams(
        input_type=1,
        input_file=input_file,
        outdir='test3',
        save_bestfit=0,
        save_sample_par=True,
        suffix=f'_{itype}',
    )
    params.filters = 'observation/test3/filters_bassmzl.txt'
    params.filters_selected = 'observation/test3/filters_selected_csst.txt'
    
    # Set data quality control parameters
    if itype == 'phot':
        params.no_spectra_fit = True
    elif itype == 'spec':
        params.no_photometry_fit = True
    
    # Set SNR thresholds (data quality control)
    params.SNRmin1 = SNRmin1Params(phot=0, spec=3)
    
    # Model: Create galaxy instance with advanced parameters
    # When you need detailed control (ssp_i1, sfh_itype_ceh, etc.), use SEDModel.create_galaxy()
    galaxy = SEDModel.create_galaxy(
        ssp_model='bc2003_hr_stelib_chab_neb_300r',
        sfh_type='exponential',  # itype_sfh=2
        dal_law='calzetti',
        ssp_k=1,
        ssp_f_run=1,
        ssp_Nstep=1,
        ssp_i0=0,
        ssp_i1=1,  # Important parameter
        ssp_i2=0,
        ssp_i3=0,
        ssp_iscalable=0,  # Match run_test.py (iscalable=0)
        sfh_itype_ceh=1,  # Chemical evolution history - important for test3!
        sfh_itruncated=0
    )
    params.add_galaxy(galaxy)
    
    # Set redshift prior (model-related)
    params.z = ZParams(iprior_type=1, min=0.0, max=1.0, nbin=40)
    
    # Add advanced model parameters
    params.rdf = RDFParams(-1, 0)
    
    # Alternative: For additional model settings (IGM, cosmology, priors, etc.):
    # model = SEDModel()
    # model.set_redshift_prior(iprior_type=1, min=0.0, max=1.0, nbin=40)
    # params.add_model(model)

    # Inference: Configure Bayesian inference with advanced settings
    inference = SEDInference()
    inference.multinest(nlive=40, efr=0.1, updInt=1000, fb=2)
    inference.nnlm(method=1, Niter1=1000, tol1=0.0, Niter2=10, tol2=0.01, p1=0.025, p2=0.975)

    print(f"Running BayeSED for test3, obj_type={obj_type}, itype={itype} (using new design)...")
    result = inference.run(params, mpi_mode='1', np=np, Ntest=Ntest)
    
    # Plot results if requested
    if plot:
        try:
            print(f"\nPlotting best-fit results from test3...")
            results = BayeSEDResults('test3')
            objects = results.list_objects()
            if objects:
                for obj_id in objects[:3]:  # Plot first 3 objects
                    print(f"  Plotting object: {obj_id}")
                    # Create results object for this specific object
                    obj_results = BayeSEDResults('test3', object_id=obj_id)
                    # output_file=None means it will be saved in the same folder as the FITS file
                    obj_results.plot_bestfit(show=True, 
                                            output_file=None,
                                            filter_file=params.filters,
                                            filter_selection_file=params.filters_selected)
            else:
                print("  No objects found in output directory")
        except Exception as e:
            print(f"  Warning: Could not plot results: {e}")
    
    # Test results class methods if requested
    if plot:
        _test_results_class('test3', max_objects=1)
        _test_pdf_plotting('test3', max_objects=1)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python recreate_run_test_with_class_based.py <test_name> [args...] [--plot]")
        print("\nTest names:")
        print("  example_gal  - Recreate run_bayesed_example('gal')")
        print("  example_qso  - Recreate run_bayesed_example('qso') - demonstrates full AGN components")
        print("  test1        - Recreate run_bayesed_test1() - photometric survey")
        print("  test2        - Recreate run_bayesed_test2() - demonstrates torus component")
        print("  test3        - Recreate run_bayesed_test3() - advanced parameters")
        print("  demo_agn     - Demonstrate different AGN component patterns (no actual run)")
        print("\nOptions:")
        print("  --plot       - Plot best-fit results and test PDF plotting (GetDist) after completion")
        print("\nExamples:")
        print("  python recreate_run_test_with_class_based.py example_qso")
        print("  python recreate_run_test_with_class_based.py test2 --plot")
        print("  python recreate_run_test_with_class_based.py test1 CSST observation/test1/test_inoise1.txt --plot")
        print("  python recreate_run_test_with_class_based.py demo_agn")
        sys.exit(1)
    
    test_name = sys.argv[1]
    plot = '--plot' in sys.argv
    
    if test_name == 'example_gal':
        run_bayesed_example_class_based('gal', plot=plot)
    elif test_name == 'example_qso':
        run_bayesed_example_class_based('qso', plot=plot)
    elif test_name == 'test1':
        survey = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] != '--plot' else 'CSST'
        obs_file = sys.argv[3] if len(sys.argv) > 3 and sys.argv[3] != '--plot' else 'observation/test1/test_inoise1.txt'
        run_bayesed_test1_class_based(survey, obs_file, plot=plot)
    elif test_name == 'test2':
        run_bayesed_test2_class_based(plot=plot)
    elif test_name == 'test3':
        obj_type = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] != '--plot' else 'STARFORMING'
        itype = sys.argv[3] if len(sys.argv) > 3 and sys.argv[3] != '--plot' else 'both'
        run_bayesed_test3_class_based(obj_type, itype, plot=plot)
    elif test_name == 'demo_agn':
        # Demonstration only - no actual run
        example_agn_components_demonstration()
    else:
        print(f"Unknown test name: {test_name}")
        print("Run with no arguments to see usage information.")
        sys.exit(1)

