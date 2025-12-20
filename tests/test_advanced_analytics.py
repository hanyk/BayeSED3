#!/usr/bin/env python3
"""
Advanced Analytics Example

This example demonstrates advanced posterior analysis capabilities
including parameter correlations, statistics, and GetDist integration.
"""

from bayesed import BayeSEDResults

def test_advanced_analytics():
    """Test advanced analytics and GetDist integration."""
    
    # Load results with intelligent configuration detection
    results = BayeSEDResults('output', catalog_name='gal', 
                           model_config='0csp_sfh200_bc2003_hr_stelib_chab_neb_2000r_i0000_2dal8_10')
    
    # Enhanced introspection
    print("=== BayeSED Results Analysis ===")
    
    # Get available parameter names (with component IDs like [0,1])
    free_params = results.get_free_parameters()
    derived_params = results.get_derived_parameters()
    print(f"Free parameters: {free_params}")
    print(f"Derived parameters: {derived_params}")
    
    # Load HDF5 data with SNR filtering
    hdf5_table = results.load_hdf5_results(filter_snr=True, min_snr=3.0)
    print(f"Loaded {len(hdf5_table)} objects after SNR filtering")
    
    # Compute parameter correlations (use actual parameter names with component IDs)
    correlation_params = ['log(age/yr)[0,1]', 'log(Z/Zsun)[0,1]', 'Av_2[0,1]']
    correlations = results.compute_parameter_correlations(correlation_params)
    print(f"Parameter correlation matrix shape: {correlations.shape}")
    
    # Get parameter statistics
    stats = results.get_parameter_statistics(correlation_params)
    print("Parameter statistics:")
    for param, param_stats in stats.items():
        print(f"  {param}: mean={param_stats['mean']:.3f}, std={param_stats['std']:.3f}")
    
    # Object-level analysis
    objects = results.list_objects()
    object_id = objects[0]  # e.g., 'spec-0285-51930-0184_GALAXY_STARFORMING'
    print(f"Analyzing object: {object_id}")
    
    # GetDist integration with intelligent caching for custom posterior analysis
    samples = results.get_getdist_samples(object_id=object_id)
    samples.label = 'Galaxy Model'
    print(f"GetDist samples loaded: {samples.numrows} samples, {len(samples.paramNames.names)} parameters")
    
    # Use GetDist for advanced visualization and analysis
    try:
        from getdist import plots
        import matplotlib.pyplot as plt
        
        g = plots.get_subplot_plotter()
        g.triangle_plot([samples], correlation_params, filled=True)
        plt.savefig('advanced_analytics_triangle.png')
        print("Triangle plot saved as 'advanced_analytics_triangle.png'")
        
    except ImportError:
        print("GetDist not available for plotting, but samples were loaded successfully")
    
    return results

def test_evidence_analysis():
    """Test evidence analysis with different return formats."""
    
    results = BayeSEDResults('output', catalog_name='gal',
                           model_config='0csp_sfh200_bc2003_hr_stelib_chab_neb_2000r_i0000_2dal8_10')
    
    print("=== Evidence Analysis ===")
    
    # Test different evidence return formats
    evidence_dict = results.get_evidence(return_format='dict')
    print(f"Evidence (dict format): {evidence_dict}")
    
    evidence_table = results.get_evidence(return_format='table')
    print(f"Evidence table shape: ({len(evidence_table)}, {len(evidence_table.colnames)})")
    
    # Single object evidence
    objects = results.list_objects()
    if objects:
        single_evidence = results.get_evidence(object_ids=objects[0])
        print(f"Single object evidence: {single_evidence}")
    
    return results

if __name__ == "__main__":
    print("Testing advanced analytics...")
    test_advanced_analytics()
    
    print("\nTesting evidence analysis...")
    test_evidence_analysis()