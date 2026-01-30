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
    # Note: If multiple catalogs exist, specify catalog_name explicitly
    results = BayeSEDResults('output',catalog_name='gal')

    # Get available parameter names (with component IDs like [0,1])
    free_params = results.get_free_parameters()
    derived_params = results.get_derived_parameters()
    # Example: ['z', 'log(age/yr)[0,1]', 'log(tau/yr)[0,1]', 'log(Z/Zsun)[0,1]', 'Av_2[0,1]', ...]

    # Load HDF5 data with SNR filtering
    hdf5_table = results.load_hdf5_results(filter_snr=True, min_snr=3.0)

    # Compute parameter correlations (use actual parameter names with component IDs)
    correlations = results.compute_parameter_correlations(['log(age/yr)[0,1]', 'log(Z/Zsun)[0,1]', 'Av_2[0,1]'])

    # Get parameter statistics
    stats = results.get_parameter_statistics(['log(age/yr)[0,1]', 'log(Z/Zsun)[0,1]', 'Av_2[0,1]'])

    # Object-level analysis
    objects = results.list_objects()
    object_id = objects[0]  # e.g., 'spec-0285-51930-0184_GALAXY_STARFORMING'

    # GetDist integration with intelligent caching for custom posterior analysis
    samples = results.get_getdist_samples(object_id=object_id)
    samples.label = 'Galaxy Model'

    # Use GetDist for advanced visualization and analysis
    from getdist import plots
    import matplotlib.pyplot as plt

    g = plots.get_subplot_plotter()
    g.triangle_plot([samples], ['log(age/yr)[0,1]', 'log(Z/Zsun)[0,1]', 'Av_2[0,1]'], filled=True)
    plt.show()

    return results

def test_evidence_analysis():
    """Test evidence analysis with different return formats."""

    results = BayeSEDResults('output',catalog_name='gal')

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
