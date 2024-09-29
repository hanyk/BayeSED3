from bayesed import BayeSEDInterface, BayeSEDParams, SSPParams, SFHParams, DALParams, MultiNestParams, SysErrParams, BigBlueBumpParams, AKNNParams, LineParams, ZParams, GreybodyParams, FANNParams, KinParams, RenameParams
import os
import sys
import subprocess

def run_bayesed_example(obj, input_dir='observation/test', output_dir='output'):
    bayesed = BayeSEDInterface(mpi_mode='1')

    params = BayeSEDParams(
        input_type=0,  # 0: Input file contains observed photometric SEDs with flux in uJy
        input_file=f'{input_dir}/{obj}.txt',
        outdir=output_dir,
        save_bestfit=0,  # 0: Save the best fitting result in fits format
        save_sample_par=True,  # Save the posterior sample of parameters
        # Galaxy model: simple stellar population (SSP), star formation history (SFH), and dust attenuation law (DAL)
        ssp=[SSPParams(
            igroup=0,
            id=0,
            name='bc2003_hr_stelib_chab_neb_2000r',
            iscalable=1,
            i1=1
        )],
        sfh=[SFHParams(
            id=0,
            itype_sfh=2  # 2: Exponentially declining SFH
        )],
        dal=[DALParams(
            id=0,
            ilaw=8  # 8: Starburst (Calzetti2000, hyperz)
        )],
        rename=[RenameParams(id=0, ireplace=1, name='Stellar+Nebular')],
        multinest=MultiNestParams(
            nlive=40,  # Number of live points
            efr=0.05,  # Sampling efficiency
            updInt=100,  # Update interval for output files
            fb=2  # Feedback level
        ),
        sys_err_obs=SysErrParams(
            min=0.0,
            max=0.2,  # Maximum fractional systematic error of observations
        )
    )

    if obj == 'qso':
        # Phenomenological AGN accretion disk model
        params.big_blue_bump = [BigBlueBumpParams(
            igroup=1,
            id=1,
            name='bbb',
            iscalable=1,
            w_min=0.1,
            w_max=10,
            Nw=1000
        )]
        params.dal.append(DALParams(
            id=1,
            ilaw=7
        ))
        # AGN BLR
        params.lines1 = [
            LineParams(
                igroup=2,
                id=2,
                name='BLR',
                iscalable=1,
                file='observation/test/lines_BLR.txt',
                R=300,
                Nkin=3
            ),
            # AGN NLR
            LineParams(
                igroup=4,
                id=4,
                name='NLR',
                iscalable=1,
                file='observation/test/lines_NLR.txt',
                R=2000,
                Nkin=2
            )
        ]
        # AGN FeII
        params.aknn = [AKNNParams(
            igroup=3,
            id=3,
            name='FeII',
            iscalable=1
        )]
        params.kin = KinParams(
            id=3,
            velscale=10,
            num_gauss_hermites_continuum=2,
            num_gauss_hermites_emission=0
        )

    print(f"Running BayeSED for {obj} object...")
    bayesed.run(params)

def run_bayesed_test1(survey, obs_file):
    bayesed = BayeSEDInterface(mpi_mode='1')

    params = BayeSEDParams(
        input_type=1,  # 1: Input file contains observed photometric SEDs with AB magnitude
        input_file=obs_file,
        outdir='test1',
        save_bestfit=2,  # 2: Save the best fitting result in both fits and hdf5 formats
        save_sample_par=True,
        multinest=MultiNestParams(
            nlive=50,
            efr=0.1,
            updInt=1000
        ),
        filters='observation/test1/filters_COSMOS_CSST_Euclid_LSST_WFIRST.txt',
        filters_selected=f'observation/test1/filters_{survey}_seleted.txt',
        ssp=[SSPParams(
            igroup=0, 
            id=0, 
            name='bc2003_lr_BaSeL_chab',
            iscalable=1, 
        )],
        sfh=[SFHParams(
            id=0,
            itype_sfh=2
        )],
        dal=[DALParams(
            id=0,
            ilaw=8
        )],
        z=ZParams(
            min=0,
            max=4
        ),
        suffix=f'_{survey}',
        no_spectra_fit=True
    )

    print(f"Running BayeSED for survey: {survey}, observation file: {obs_file}")
    bayesed.run(params)

def run_bayesed_test2():
    bayesed = BayeSEDInterface(mpi_mode='1')

    params = BayeSEDParams(
        input_type=0,
        input_file='observation/test2/test.txt',
        outdir='test2',
        save_bestfit=0,  # 0: Save the best fitting result in fits format
        save_sample_par=True,
        multinest=MultiNestParams(
            nlive=400,
            efr=0.1,
            updInt=1000,
            fb=2
        ),
        filters='observation/test2/filters.txt',
        filters_selected='observation/test2/filters_selected.txt',
        ssp=[SSPParams(
            igroup=0,
            id=0,
            name='bc2003_lr_BaSeL_chab',
            iscalable=1
        )],
        sfh=[SFHParams(
            id=0,
            itype_sfh=2
        )],
        dal=[DALParams(
            id=0,
            ilaw=7
        )],
        greybody=[GreybodyParams(
            igroup=0,
            id=1,
            name='gb',
            iscalable=-2
        )],
        fann=[FANNParams(
            igroup=1,
            id=2,
            name='clumpy201410tor',
            iscalable=1
        )]
    )

    print("Running BayeSED for test2...")
    bayesed.run(params)

def plot_results(obj, output_dir):
    if obj in ['gal', 'qso']:
        plot_script = 'observation/test/plot_bestfit.py'
        search_dir = os.path.join(output_dir, obj)
    elif obj == 'test1':
        plot_script = 'observation/test1/plot_bestfit.py'
        search_dir = 'test1'
    elif obj == 'test2':
        plot_script = 'observation/test2/plot_bestfit.py'
        search_dir = 'test2'
    else:
        print(f"Cannot find appropriate plotting script for object {obj}")
        return

    for root, dirs, files in os.walk(search_dir):
        for file in files:
            if file.endswith('.fits'):
                fits_file = os.path.join(root, file)
                cmd = ['python', plot_script, fits_file]
                subprocess.run(cmd)

if __name__ == "__main__":
    # Ensure output directory exists
    os.makedirs('output', exist_ok=True)

    # Get obj from command line arguments
    if len(sys.argv) < 2:
        print("Usage: python run_test.py <obj>")
        print("Where <obj> can be 'gal', 'qso', 'test1', or 'test2'")
        sys.exit(1)

    obj = sys.argv[1]
    plot = False
    if len(sys.argv) > 2 and sys.argv[2] == 'plot':
        plot = True

    if obj == 'test1':
        surveys = ['CSST']
        obs_files = ['observation/test1/test_inoise1.txt']
        for survey in surveys:
            for obs_file in obs_files:
                run_bayesed_test1(survey, obs_file)
    elif obj == 'test2':
        run_bayesed_test2()
    else:
        # Run BayeSED example
        run_bayesed_example(obj)

    if plot:
        plot_results(obj, 'output')