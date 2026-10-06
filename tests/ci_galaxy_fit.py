#!/usr/bin/env python3
"""One-object galaxy fit for the GitHub fit job.

Uses the quick-test sample size: first source in ``observation/test/gal.txt``,
``nlive=40``, single-process ``mpi_mode='1'``.
"""

from pathlib import Path

from bayesed import BayeSEDInterface, BayeSEDParams, MultiNestParams


def main():
    outdir = Path('tests/output_ci_fit')
    bayesed = BayeSEDInterface(mpi_mode='1', Ntest=1)
    params = BayeSEDParams.galaxy(
        input_file='observation/test/gal.txt',
        outdir=str(outdir),
        ssp_model='bc2003_hr_stelib_chab_neb_2000r',
        sfh_type='exponential',
        dal_law='calzetti',
    )
    params.multinest = MultiNestParams(nlive=40)
    execution = bayesed.run(params)
    if execution.process.returncode != 0:
        raise SystemExit(execution.process.returncode)
    fits = sorted(outdir.rglob('*_bestfit.fits'))
    if not fits:
        raise SystemExit('fit finished but no *_bestfit.fits was written')
    print(fits[0])


if __name__ == '__main__':
    main()
