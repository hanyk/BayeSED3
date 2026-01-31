#!/usr/bin/env python3
"""
AGN Fitting Example

This example demonstrates how to fit AGN SEDs with all components
including galaxy host, accretion disk, BLR, NLR, and FeII.
"""

from bayesed import BayeSEDInterface, BayeSEDParams

def test_agn_fitting():
    """Test AGN fitting with all components."""
    
    # Initialize interface with Ntest for quick testing (optional)
    bayesed = BayeSEDInterface(mpi_mode='auto', Ntest=2)  # Process only first 2 objects
    
    # AGN with all components (includes galaxy host)
    params = BayeSEDParams.agn(
        input_file='observation/test/qso.txt',
        outdir='tests/output_agn_fitting',
        ssp_model='bc2003_hr_stelib_chab_neb_2000r',
        sfh_type='exponential',
        dal_law='calzetti',
        agn_components=['dsk', 'blr', 'nlr', 'feii']  # Disk, BLR, NLR, FeII
    )
    
    # Run analysis
    result = bayesed.run(params)
    print("AGN fitting completed successfully!")
    
    return result

if __name__ == "__main__":
    test_agn_fitting()