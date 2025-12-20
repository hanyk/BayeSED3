#!/usr/bin/env python3
"""
Custom Model Configuration Example

This example demonstrates how to create custom model configurations
with dust emission and AGN components.
"""

from bayesed import BayeSEDInterface, BayeSEDParams
from bayesed.model import SEDModel

def test_custom_model_dust_emission():
    """Test custom galaxy model with dust emission."""
    
    bayesed = BayeSEDInterface(mpi_mode='auto')
    
    # Create galaxy instance with dust emission (using real data from observation/test2/)
    galaxy = SEDModel.create_galaxy(
        ssp_model='bc2003_lr_BaSeL_chab',
        sfh_type='exponential',
        dal_law='smc'
    )
    galaxy.add_dust_emission()  # Add dust emission component
    
    # Create AGN instance with torus
    agn = SEDModel.create_agn(agn_components=['tor'])
    
    # Assemble configuration using real data files
    params = BayeSEDParams(
        input_type=0,  # Flux in μJy
        input_file='observation/test2/test.txt',
        outdir='test2_output',
        filters='observation/test2/filters.txt',
        filters_selected='observation/test2/filters_selected.txt',
        save_sample_par=True  # Enable posterior sample generation
    )
    params.add_galaxy(galaxy)
    params.add_agn(agn)
    
    result = bayesed.run(params)
    print("Custom model with dust emission completed successfully!")
    
    return result

def test_custom_model_full_agn():
    """Test custom model with full AGN components."""
    
    bayesed = BayeSEDInterface(mpi_mode='auto')
    
    # For AGN with all components (disk, BLR, NLR, FeII) using real emission line files
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
        input_type=0,
        input_file='observation/test/qso.txt',
        outdir='output_qso',
        save_sample_par=True
    )
    params.add_galaxy(galaxy)
    params.add_agn(agn)
    
    result = bayesed.run(params)
    print("Custom model with full AGN components completed successfully!")
    
    return result

if __name__ == "__main__":
    print("Testing custom model with dust emission...")
    test_custom_model_dust_emission()
    
    print("\nTesting custom model with full AGN components...")
    test_custom_model_full_agn()