from bayesed import BayeSEDInterface, BayeSEDParams, SSPParams, SFHParams, DALParams, MultiNestParams, SysErrParams, BigBlueBumpParams, AKNNParams, LineParams, ZParams, GreybodyParams, FANNParams, KinParams
import os
import sys

def run_bayesed_example(obj, input_dir='observation/test', output_dir='output'):
    bayesed = BayeSEDInterface(executable_type='mn_1')

    params = BayeSEDParams(
        input_file=f'0,{input_dir}/{obj}.txt',
        outdir=output_dir,
        save_bestfit=0,
        save_sample_par=True,
        ssp=[SSPParams(
            igroup=0, 
            id=0, 
            name='bc2003_hr_stelib_chab_neb_2000r', 
            iscalable=1, 
            i1=1
        )],
        sfh=[SFHParams(
            id=0,
            itype_sfh=2
        )],
        dal=[DALParams(
            id=0,
            ilaw=8
        )],
        rename='0,1,Stellar+Nebular',
        multinest=MultiNestParams(
            nlive=40,
            efr=0.05,
            updInt=100,
            fb=2
        ),
        sys_err_obs=SysErrParams(
            min=0.0,
            max=0.2,
        )
    )

    if obj == 'qso':
        params.big_blue_bump = [BigBlueBumpParams(
            igroup=1,
            id=1,
            name='bbb',
            iscalable=1,
            w_min=0.1,
            w_max=10,
            Nw=1000
        )]
        params.dal = [DALParams(
            id=1,
            ilaw=7
        )]
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
    print(f"BayeSED run completed for {obj} object.")

def run_bayesed_test1(survey, obs_file):
    bayesed = BayeSEDInterface(executable_type='mn_1')

    params = BayeSEDParams(
        input_file=f'1,{obs_file}',
        outdir='test1',
        save_bestfit=2,
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
    print(f"BayeSED run completed for survey: {survey}, observation file: {obs_file}")

def run_bayesed_test2():
    bayesed = BayeSEDInterface(executable_type='mn_1')

    params = BayeSEDParams(
        input_file='0,observation/test2/test.txt',
        outdir='test2',
        save_bestfit=0,
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
    print("BayeSED run completed for test2.")

if __name__ == "__main__":
    # Ensure output directory exists
    os.makedirs('output', exist_ok=True)

    # Get obj from command line arguments
    if len(sys.argv) < 2:
        print("Usage: python run_test.py <obj>")
        print("Where <obj> can be 'gal', 'qso', 'test1', or 'test2'")
        sys.exit(1)

    obj = sys.argv[1]
    if obj not in ['gal', 'qso', 'test1', 'test2']:
        print("Error: obj must be 'gal', 'qso', 'test1', or 'test2'")
        sys.exit(1)

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

    print("Example run completed.")
