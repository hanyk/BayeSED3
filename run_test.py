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

if __name__ == "__main__":
    # Ensure output directory exists
    os.makedirs('output', exist_ok=True)

    # Get obj from command line arguments
    if len(sys.argv) < 2:
        print("Usage: python run.py <obj>")
        print("Where <obj> can be 'gal' or 'qso'")
        sys.exit(1)

    obj = sys.argv[1]
    if obj not in ['gal', 'qso']:
        print("Error: obj must be 'gal' or 'qso'")
        sys.exit(1)

    # Run BayeSED example
    run_bayesed_example(obj)

    print("Example run completed.")
