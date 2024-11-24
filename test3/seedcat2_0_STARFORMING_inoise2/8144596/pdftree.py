from getdist import plots

g=plots.get_subplot_plotter(chain_dir=r'./')
roots = []
roots.append('0csp_sfh201_bc2003_hr_stelib_chab_neb_300r_i0100_rdf0_2dal8_10_z_phot_sample_par')
roots.append('0csp_sfh201_bc2003_hr_stelib_chab_neb_300r_i0100_rdf0_2dal8_10_z_spec_sample_par')
roots.append('0csp_sfh201_bc2003_hr_stelib_chab_neb_300r_i0100_rdf0_2dal8_10_z_both_sample_par')
params = ['z', 'log(age/yr)[0,0]', 'log(tau/yr)[0,0]', 'log(Z/Zsun)[0,0]', 'Av_2[0,0]', 'log(Mstar)[0,0]']
g.triangle_plot(roots, params, filled=True)
g.export('pdftree.png')
