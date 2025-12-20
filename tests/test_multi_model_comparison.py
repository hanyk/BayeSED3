#!/usr/bin/env python3
"""
Multi-Model Comparison Example

This example demonstrates how to compare different models using
the same input data and analyze the results.
"""

import numpy as np
from bayesed import BayeSEDInterface, BayeSEDParams, BayeSEDResults
from bayesed.results import standardize_parameter_names, plot_posterior_comparison
from bayesed.model import SEDModel

def test_multi_model_comparison():
    """Test comparison between different SED models."""
    
    bayesed = BayeSEDInterface(mpi_mode='auto')
    
    # Same input data, different models
    input_file = 'observation/test/gal.txt'
    
    # Model 1: Exponential SFH with Calzetti dust law
    params1 = BayeSEDParams.galaxy(
        input_file=input_file,
        outdir='output_model1_exp_calzetti',
        ssp_model='bc2003_hr_stelib_chab_neb_2000r',
        sfh_type='exponential',
        dal_law='calzetti',
        save_sample_par=True
    )
    print("Running Model 1: Exponential SFH with Calzetti dust law...")
    bayesed.run(params1)
    
    # Model 2: Delayed SFH with SMC dust law
    params2 = BayeSEDParams.galaxy(
        input_file=input_file,
        outdir='output_model2_delayed_smc',
        ssp_model='bc2003_hr_stelib_chab_neb_2000r',
        sfh_type='delayed',
        dal_law='smc',
        save_sample_par=True
    )
    print("Running Model 2: Delayed SFH with SMC dust law...")
    bayesed.run(params2)
    
    # Compare results for the same object with different models
    results1 = BayeSEDResults('output_model1_exp_calzetti')
    results2 = BayeSEDResults('output_model2_delayed_smc')
    
    # Standardize parameter names across models for comparison
    results_list = [results1, results2]
    standardize_parameter_names(results_list)
    
    # Create comparison plots showing how model choice affects parameter inference
    plot_posterior_comparison(
        results_list,
        labels=['Exp+Calzetti', 'Delayed+SMC'],
        output_file='model_comparison.png'
    )
    
    # Compare Bayesian evidence to determine which model is preferred
    evidence1 = results1.get_evidence(return_format='table')
    evidence2 = results2.get_evidence(return_format='table')
    
    # Get mean evidence values across all objects
    mean_logZ1 = np.mean(evidence1['INSlogZ'])
    mean_logZ2 = np.mean(evidence2['INSlogZ'])
    mean_logZerr1 = np.mean(evidence1['INSlogZerr'])
    mean_logZerr2 = np.mean(evidence2['INSlogZerr'])
    
    print(f"Model 1 log-evidence: {mean_logZ1:.2f} +/- {mean_logZerr1:.3f}")
    print(f"Model 2 log-evidence: {mean_logZ2:.2f} +/- {mean_logZerr2:.3f}")
    
    # Calculate Bayes factor manually (since we removed compare_evidence method)
    delta_logZ = mean_logZ1 - mean_logZ2
    bayes_factor = np.exp(delta_logZ)
    
    print(f"Bayes factor (Model 1 vs Model 2): {bayes_factor:.2e}")
    if delta_logZ > 0:
        print("Model 1 (Exp+Calzetti) is preferred")
    else:
        print("Model 2 (Delayed+SMC) is preferred")
    
    return results_list

if __name__ == "__main__":
    test_multi_model_comparison()