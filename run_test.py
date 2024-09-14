from bayesed import BayeSEDInterface
import os
import sys

def run_bayesed_example(obj, input_dir='observation/test', output_dir='output'):
    bayesed = BayeSEDInterface(executable_type='mn_1')

    args = [
        '-i', f'0,{input_dir}/{obj}.txt',
        '--outdir', output_dir,
        '--save_bestfit', '0',
        '--save_sample_par',
        '--ssp', '0,0,bc2003_hr_stelib_chab_neb_2000r,1,1,1,1,0,1,0,0',
        '--sfh', '0,2,0,0',
        '--dal', '0,2,8',
        '--rename', '0,1,Stellar+Nebular',
        '--multinest', '1,0,1,40,0.05,0.5,100,-1e90,1,2,0,0,-1e90,100000,0.01',
        '--sys_err_obs', '1,0,0.0,0.2,40',
        '-v', '2'
    ]

    if obj == 'qso':
        args.extend([
            '-bbb', '1,1,bbb,1,0.1,10,1000',
            '--dal', '1,2,7',
            '-ls1', '2,2,BLR,1,observation/test/lines_BLR.txt,300,2,3',
            '-k', '3,3,FeII,1,1,1,0,0,1,1,1',
            '--kin', '3,10,2,0',
            '-ls1', '4,4,NLR,1,observation/test/lines_NLR.txt,2000,2,2'
        ])

    print(f"Running BayeSED for {obj} object...")
    bayesed.run(args)
    print(f"BayeSED run completed for {obj} object.")

def run_bayesed_test1(survey, obs_file):
    bayesed = BayeSEDInterface(executable_type='mn_1')

    args = [
        '--multinest', '1,0,0,50,0.1,0.5,1000,-1e90,1,0,0,0,-1e90,100000,0.01',
        '-i', f'1,{obs_file}',
        '--filters', 'observation/test1/filters_COSMOS_CSST_Euclid_LSST_WFIRST.txt',
        '--filters_selected', f'observation/test1/filters_{survey}_seleted.txt',
        '--ssp', '0,0,bc2003_lr_BaSeL_chab,1,1,1,1,0,0,0,0',
        '--sfh', '0,2,0,0',
        '--dal', '0,2,8',
        '--z', '1,0,0,4,40',
        '--outdir', 'test1',
        '--suffix', f'_{survey}',
        '--no_spectra_fit',
        '--save_sample_par',
        '--save_bestfit', '2'
    ]

    print(f"Running BayeSED for survey: {survey}, observation file: {obs_file}")
    bayesed.run(args)
    print(f"BayeSED run completed for survey: {survey}, observation file: {obs_file}")

def run_bayesed_test2():
    bayesed = BayeSEDInterface(executable_type='mn_1')

    args = [
        '--multinest', '1,0,0,400,0.1,0.5,1000,-1e90,1,2,0,0,-1e90,100000,0.01',
        '-i', '0,observation/test2/test.txt',
        '--filters', 'observation/test2/filters.txt',
        '--filters_selected', 'observation/test2/filters_selected.txt',
        '--ssp', '0,0,bc2003_lr_BaSeL_chab,1,3,1,0,0,0,0,0',
        '--sfh', '0,2,0,0',
        '--dal', '0,2,7',
        '-gb', '0,1,gb,-2,1,1,1000,200',
        '-a', '1,2,clumpy201410tor,1',
        '--outdir', 'test2',
        '--save_bestfit', '0',
        '--save_sample_par'
    ]

    print("Running BayeSED for test2...")
    bayesed.run(args)
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
