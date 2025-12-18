from bayesed import BayeSEDInterface, BayeSEDParams, BayeSEDResults

# Initialize interface
bayesed = BayeSEDInterface(mpi_mode='auto')

# Simple galaxy fitting
params = BayeSEDParams.galaxy(
    input_file='observation/test/gal.txt',
    outdir='output',
    ssp_model='bc2003_hr_stelib_chab_neb_2000r',
    sfh_type='exponential',
    dal_law='calzetti'
)

# Run analysis
result = bayesed.run(params)

# Load and analyze results
results = BayeSEDResults('output', catalog_name='gal')
results.print_summary()

# Access parameters and objects
free_params = results.get_free_parameters()
available_objects = results.list_objects()

# Load all parameters as astropy Table from HDF5 file
hdf5_table = results.load_hdf5_results()
# Built-in SNR filtering
high_snr_table = results.load_hdf5_results(filter_snr=True, min_snr=5.0)


# Access all statistical estimates for specific parameters
age_table = results.get_parameter_values('log(age/yr)[0,1]')
mass_table = results.get_parameter_values('log(Mstar)[0,1]')

custom_labels = {
    # Free parameters
    'log(age/yr)[0,1]': r'\log(age/\mathrm{yr})',
    'log(tau/yr)[0,1]': r'\log(\tau/\mathrm{yr})',
    'log(Z/Zsun)[0,1]': r'\log(Z/Z_\odot)',
    'Av_2[0,1]': r'A_V',
    # Derived parameters
    'log(Mstar)[0,1]': r'\log(M_\star/M_\odot)',
    'log(SFR_{100Myr}/[M_{sun}/yr])[0,1]': r'\log(\mathrm{SFR}/M_\odot\,\mathrm{yr}^{-1})'
}

results.set_parameter_labels(custom_labels)
results.plot_bestfit()
results.plot_posterior_free()
results.plot_posterior_derived(max_params=5)
results.plot_posterior(['log(age/yr)[0,1]', 'log(Z/Zsun)[0,1]', 'log(Mstar)[0,1]', 'log(SFR_{100Myr}/[M_{sun}/yr])[0,1]'])  # Mixed free+derived parameters


# Object-level analysis
object_results = BayeSEDResults('output', catalog_name='gal',
                               object_id='spec-0285-51930-0184_GALAXY_STARFORMING')
object_results.plot_bestfit()

object_results.set_parameter_labels(custom_labels)
object_results.plot_posterior_free()
object_results.plot_posterior_derived(max_params=5)
object_results.plot_posterior(['log(age/yr)[0,1]', 'log(Z/Zsun)[0,1]', 'log(Mstar)[0,1]', 'log(SFR_{100Myr}/[M_{sun}/yr])[0,1]'])  # Mixed free+derived parameters
