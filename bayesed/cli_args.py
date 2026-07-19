"""
CLI argv serialization for BayeSEDParams.

Internal adapter: Model configuration -> binary argv list (ADR-0004).
Not part of the public bayesed package export; BayeSEDInterface.run calls
to_cli_args().
"""

def _format_fann_params(fann_params):
    return f"{fann_params.igroup},{fann_params.id},{fann_params.name},{fann_params.iscalable}"

def _format_AGN_params(AGN_params):
    return f"{AGN_params.igroup},{AGN_params.id},{AGN_params.name},{AGN_params.iscalable},{AGN_params.imodel},{AGN_params.icloudy},{AGN_params.suffix},{AGN_params.w_min},{AGN_params.w_max},{AGN_params.Nw}"

def _format_blackbody_params(blackbody_params):
    return f"{blackbody_params.igroup},{blackbody_params.id},{blackbody_params.bb},{blackbody_params.iscalable},{blackbody_params.w_min},{blackbody_params.w_max},{blackbody_params.Nw}"

def _format_big_blue_bump_params(big_blue_bump_params):
    return f"{big_blue_bump_params.igroup},{big_blue_bump_params.id},{big_blue_bump_params.name},{big_blue_bump_params.iscalable},{big_blue_bump_params.w_min},{big_blue_bump_params.w_max},{big_blue_bump_params.Nw}"

def _format_greybody_params(greybody_params):
    return f"{greybody_params.igroup},{greybody_params.id},{greybody_params.name},{greybody_params.iscalable},{greybody_params.ithick},{greybody_params.w_min},{greybody_params.w_max},{greybody_params.Nw}"

def _format_aknn_params(aknn_params):
    return f"{aknn_params.igroup},{aknn_params.id},{aknn_params.name},{aknn_params.iscalable},{aknn_params.k},{aknn_params.f_run},{aknn_params.eps},{aknn_params.iRad},{aknn_params.iprep},{aknn_params.Nstep},{aknn_params.alpha}"

def _format_line_params(line_params):
    return f"{line_params.igroup},{line_params.id},{line_params.name},{line_params.iscalable},{line_params.file},{line_params.R},{line_params.Nsample},{line_params.Nkin}"

def _format_luminosity_params(luminosity_params):
    return f"{luminosity_params.id},{luminosity_params.w_min},{luminosity_params.w_max}"

def _format_np_sfh_params(np_sfh_params):
    return f"{np_sfh_params.prior_type},{np_sfh_params.interpolation_method},{np_sfh_params.num_bins},{np_sfh_params.regul}"

def _format_polynomial_params(polynomial_params):
    return str(polynomial_params.order)

def _format_powerlaw_params(powerlaw_params):
    return f"{powerlaw_params.igroup},{powerlaw_params.id},{powerlaw_params.pw},{powerlaw_params.iscalable},{powerlaw_params.w_min},{powerlaw_params.w_max},{powerlaw_params.Nw}"

def _format_rbf_params(rbf_params):
    return f"{rbf_params.igroup},{rbf_params.id},{rbf_params.name},{rbf_params.iscalable}"

def _format_sfh_params(sfh_params):
    return f"{sfh_params.id},{sfh_params.itype_sfh},{sfh_params.itruncated},{sfh_params.itype_ceh}"

def _format_ssp_params(ssp_params):
    return f"{ssp_params.igroup},{ssp_params.id},{ssp_params.name},{ssp_params.iscalable},{ssp_params.k},{ssp_params.f_run},{ssp_params.Nstep},{ssp_params.i0},{ssp_params.i1},{ssp_params.i2},{ssp_params.i3}"

def _format_sedlib_params(sedlib_params):
    return f"{sedlib_params.igroup},{sedlib_params.id},{sedlib_params.name},{sedlib_params.iscalable},{sedlib_params.dir},{sedlib_params.itype},{sedlib_params.f_run},{sedlib_params.ikey}"

def _format_sys_err_params(sys_err_params):
    return f"{sys_err_params.iprior_type},{sys_err_params.is_age},{sys_err_params.min},{sys_err_params.max},{sys_err_params.nbin}"

def _format_z_params(z_params):
    return f"{z_params.iprior_type},{z_params.is_age},{z_params.min},{z_params.max},{z_params.nbin}"

def _format_SFR_over_params(SFR_over_params):
    # help.txt: --SFR_over ARG1[,ARGn] (past Myr windows; default e.g. 10,100)
    return ','.join(str(x) for x in SFR_over_params.past_Myrs)

def _format_SNRmin1_params(SNRmin1_params):
    return f"{SNRmin1_params.phot},{SNRmin1_params.spec}"

def _format_SNRmin2_params(SNRmin2_params):
    return f"{SNRmin2_params.phot},{SNRmin2_params.spec}"

def _format_NNLM_params(NNLM_params):
    return f"{NNLM_params.method},{NNLM_params.Niter1},{NNLM_params.tol1},{NNLM_params.Niter2},{NNLM_params.tol2},{NNLM_params.p1},{NNLM_params.p2}"

def _format_Ndumper_params(Ndumper_params):
    return f"{Ndumper_params.max_number},{Ndumper_params.iconverged_min},{Ndumper_params.Xmin_squared_Nd}"

def _format_output_SFH_params(output_SFH_params):
    return f"{output_SFH_params.ntimes},{output_SFH_params.ilog}"

def _format_multinest_params(multinest_params):
    return f"{int(multinest_params.INS)},{int(multinest_params.mmodal)},{int(multinest_params.ceff)},{multinest_params.nlive},{multinest_params.efr},{multinest_params.tol},{multinest_params.updInt},{multinest_params.Ztol},{multinest_params.seed},{multinest_params.fb},{int(multinest_params.resume)},{int(multinest_params.outfile)},{multinest_params.logZero},{multinest_params.maxiter},{multinest_params.acpt}"

def _format_gsl_integration_qag_params(gsl_integration_qag_params):
    return f"{gsl_integration_qag_params.epsabs},{gsl_integration_qag_params.epsrel},{gsl_integration_qag_params.limit}"

def _format_gsl_multifit_robust_params(gsl_multifit_robust_params):
    return f"{gsl_multifit_robust_params.type},{gsl_multifit_robust_params.tune}"

def _format_kin_params(kin_params):
    return f"{kin_params.id},{kin_params.velscale},{kin_params.num_gauss_hermites_con},{kin_params.num_gauss_hermites_eml}"

def _format_LineList_params(LineList_params):
    return f"{LineList_params.file},{LineList_params.type}"

def _format_make_catalog_params(make_catalog_params):
    return f"{make_catalog_params.id1},{make_catalog_params.logscale_min1},{make_catalog_params.logscale_max1},{make_catalog_params.id2},{make_catalog_params.logscale_min2},{make_catalog_params.logscale_max2}"

def _format_cloudy_params(cloudy_params):
    return f"{cloudy_params.igroup},{cloudy_params.id},{cloudy_params.cloudy},{cloudy_params.iscalable}"

def _format_cosmology_params(cosmology_params):
    return f"{cosmology_params.H0},{cosmology_params.omigaA},{cosmology_params.omigam}"

def _format_dal_params(dal_params):
    return f"{dal_params.id},{dal_params.con_eml_tot},{dal_params.ilaw}"

def _format_rdf_params(rdf_params):
    return f"{rdf_params.id},{rdf_params.num_polynomials}"

def _format_template_params(template_params):
    return f"{template_params.igroup},{template_params.id},{template_params.name},{template_params.iscalable}"

def to_cli_args(params):
    if isinstance(params, list):
        return params

    args = []

    if params.help:
        args.append('-h')
        return args

    args.extend([
        '-i', f"{params.input_type},{params.input_file}",
        '--outdir', params.outdir,
        '--save_bestfit', str(params.save_bestfit),
        '-v', str(params.verbose)
    ])

    # Add other command line arguments of bayesed
    if params.fann:
        for fann_params in params.fann:
            args.extend(['-a', _format_fann_params(fann_params)])

    if params.AGN:
        for AGN_params in params.AGN:
            args.extend(['-AGN', _format_AGN_params(AGN_params)])

    if params.blackbody:
        for blackbody_params in params.blackbody:
            args.extend(['-bb', _format_blackbody_params(blackbody_params)])

    if params.big_blue_bump:
        for big_blue_bump_params in params.big_blue_bump:
            args.extend(['-bbb', _format_big_blue_bump_params(big_blue_bump_params)])

    if params.greybody:
        for greybody_params in params.greybody:
            args.extend(['-gb', _format_greybody_params(greybody_params)])

    if params.aknn:
        for aknn_params in params.aknn:
            args.extend(['-k', _format_aknn_params(aknn_params)])

    if params.line:
        for line_params in params.line:
            args.extend(['-l', _format_line_params(line_params)])

    if params.lines:
        for lines_params in params.lines:
            args.extend(['-ls', _format_line_params(lines_params)])

    if params.lines1:
        for line_params in params.lines1:
            args.extend(['-ls1', _format_line_params(line_params)])

    if params.luminosity:
        args.extend(['--luminosity', _format_luminosity_params(params.luminosity)])

    if params.np_sfh:
        args.extend(['--np_sfh', _format_np_sfh_params(params.np_sfh)])

    if params.polynomial:
        args.extend(['--polynomial', _format_polynomial_params(params.polynomial)])

    if params.powerlaw:
        for powerlaw_params in params.powerlaw:
            args.extend(['-pw', _format_powerlaw_params(powerlaw_params)])

    if params.rbf:
        for rbf_params in params.rbf:
            args.extend(['-rbf', _format_rbf_params(rbf_params)])

    if params.sfh:
        for sfh_params in params.sfh:
            args.extend(['--sfh', _format_sfh_params(sfh_params)])

    if params.ssp:
        for ssp_params in params.ssp:
            args.extend(['-ssp', _format_ssp_params(ssp_params)])

    if params.sedlib:
        for sedlib_params in params.sedlib:
            args.extend(['-sedlib', _format_sedlib_params(sedlib_params)])

    if params.sys_err_mod:
        args.extend(['--sys_err_mod', _format_sys_err_params(params.sys_err_mod)])

    if params.sys_err_obs:
        args.extend(['--sys_err_obs', _format_sys_err_params(params.sys_err_obs)])

    if params.z:
        args.extend(['--z', _format_z_params(params.z)])

    if params.rename:
        for rename_params in params.rename:
            args.extend(['--rename', f"{rename_params.id},{rename_params.ireplace},{rename_params.name}"])

    if params.rename_all:
        args.extend(['--rename_all', params.rename_all])

    if params.save_pos_sfh:
        args.extend(['--save_pos_sfh', params.save_pos_sfh])

    if params.save_pos_spec:
        args.append('--save_pos_spec')

    if params.save_sample_obs:
        args.append('--save_sample_obs')

    if params.save_sample_spec:
        args.append('--save_sample_spec')

    if params.save_summary:
        args.append('--save_summary')

    if params.suffix:
        args.extend(['--suffix', params.suffix])

    if params.SFR_over:
        args.extend(['--SFR_over', _format_SFR_over_params(params.SFR_over)])

    if params.SNRmin1:
        args.extend(['--SNRmin1', f"{params.SNRmin1.phot},{params.SNRmin1.spec}"])

    if params.SNRmin2:
        args.extend(['--SNRmin2', f"{params.SNRmin2.phot},{params.SNRmin2.spec}"])

    if params.unweighted_samples:
        args.append('--unweighted_samples')

    if params.filters:
        args.extend(['--filters', params.filters])

    if params.filters_selected:
        args.extend(['--filters_selected', params.filters_selected])

    if params.no_spectra_fit:
        args.append('--no_spectra_fit')

    if params.NNLM:
        args.extend(['--NNLM', _format_NNLM_params(params.NNLM)])

    if params.Ndumper:
        args.extend(['--Ndumper', _format_Ndumper_params(params.Ndumper)])

    if params.NfilterPoints:
        args.extend(['--NfilterPoints', str(params.NfilterPoints)])

    if params.Nsample:
        args.extend(['--Nsample', str(params.Nsample)])

    if params.output_SFH:
        args.extend(['--output_SFH', _format_output_SFH_params(params.output_SFH)])

    if params.output_mock_photometry:
        args.extend(['--output_mock_photometry', str(params.output_mock_photometry)])

    if params.multinest:
        args.extend(['--multinest', _format_multinest_params(params.multinest)])

    if params.gsl_integration_qag:
        args.extend(['--gsl_integration_qag', f"{params.gsl_integration_qag.epsabs},{params.gsl_integration_qag.epsrel},{params.gsl_integration_qag.limit}"])

    if params.gsl_multifit_robust:
        args.extend(['--gsl_multifit_robust', f"{params.gsl_multifit_robust.type},{params.gsl_multifit_robust.tune}"])

    if params.import_files:
        for import_file in params.import_files:
            args.extend(['--import', import_file])

    if params.inn:
        for inn_params in params.inn:
            args.extend(['--inn', _format_sedlib_params(inn_params)])

    if params.IGM is not None:
        args.extend(['--IGM', str(params.IGM)])

    if params.kin:
        for kin_param in params.kin:
            args.extend(['--kin', _format_kin_params(kin_param)])

    if params.logZero is not None:
        args.extend(['--logZero', str(params.logZero)])

    if params.lw_max is not None:
        args.extend(['--lw_max', str(params.lw_max)])

    if params.LineList:
        args.extend(['--LineList', _format_LineList_params(params.LineList)])

    if params.make_catalog:
        args.extend(['--make_catalog', _format_make_catalog_params(params.make_catalog)])

    if params.niteration is not None:
        args.extend(['--niteration', str(params.niteration)])

    if params.no_photometry_fit:
        args.append('--no_photometry_fit')

    if params.cloudy:
        for cloudy_params in params.cloudy:
            args.extend(['--cloudy', _format_cloudy_params(cloudy_params)])

    if params.cosmology:
        args.extend(['--cosmology', _format_cosmology_params(params.cosmology)])

    if params.dal:
        for dal_params in params.dal:
            args.extend(['--dal', _format_dal_params(dal_params)])

    if params.export:
        args.extend(['--export', params.export])

    if params.output_mock_spectra:
        args.append('--output_mock_spectra')

    if params.output_model_absolute_magnitude:
        args.append('--output_model_absolute_magnitude')

    if params.output_pos_obs:
        args.append('--output_pos_obs')

    if params.priors_only:
        args.append('--priors_only')

    if params.rdf:
        args.extend(['--rdf', _format_rdf_params(params.rdf)])

    if params.template:
        for template_params in params.template:
            args.extend(['-t', _format_template_params(template_params)])

    if params.build_sedlib is not None:
        args.extend(['--build_sedlib', str(params.build_sedlib)])
    if params.check:
        args.append('--check')
    if params.cl:
        args.extend(['--cl', params.cl])

    if params.save_sample_par:
        args.append('--save_sample_par')

    return args

