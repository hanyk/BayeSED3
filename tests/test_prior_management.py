#!/usr/bin/env python3
"""
Prior Management Example

This example demonstrates how to manage parameter priors programmatically
without manually editing .iprior files.
"""

from bayesed import SEDInference, BayeSEDParams

def test_prior_management():
    """Test prior management API."""
    
    # Initialize and load priors
    params = BayeSEDParams.galaxy(input_file='observation/test/gal.txt', outdir='tests/output_prior_management')
    inference = SEDInference()
    inference.priors_init(params)
    
    print("=== Initial Priors ===")
    inference.print_priors()
    
    # View and modify priors
    print("\n=== Modifying log(age/yr) prior ===")
    inference.set_prior('log(age/yr)', min_val=8.5, max_val=9.8, nbin=60)
    
    # List all available prior types
    print("\n=== Available Prior Types ===")
    inference.list_prior_types()  # Shows: Uniform, Gaussian, Gamma, Beta, Student's t, Weibull, etc.
    
    # Use different prior types (Uniform, Gaussian, Gamma, Beta, etc.)
    print("\n=== Setting Gaussian prior for log(age/yr) ===")
    inference.set_prior('log(age/yr)', prior_type='Gaussian', 
                       min_val=8.0, max_val=12.0, hyperparameters=[10.0, 1.0])
    
    # Regex patterns (with confirmation)
    print("\n=== Setting Gaussian prior for all Av parameters ===")
    inference.set_prior('^Av_.*', prior_type='Gaussian', hyperparameters=[1.0, 0.3])
    
    # Partial matching
    print("\n=== Partial matching for 'age' ===")
    inference.set_prior('age', min_val=8.0, max_val=10.0)  # Matches 'log(age/yr)', etc.
    
    # Query without modifying
    print("\n=== Query parameters containing 'age' ===")
    inference.set_prior('age')  # Shows all parameters containing 'age'
    
    # Reset a single parameter to its default prior
    print("\n=== Resetting log(age/yr) to default ===")
    inference.set_prior('log(age/yr)', reset_to_default=True)
    
    # Reset multiple parameters using patterns
    print("\n=== Resetting all Av parameters to default ===")
    inference.set_prior('Av_.*', reset_to_default=True)  # Reset all Av parameters
    
    print("\n=== Final Priors ===")
    inference.print_priors()
    
    print("\nPrior management example completed successfully!")
    return inference

if __name__ == "__main__":
    test_prior_management()
