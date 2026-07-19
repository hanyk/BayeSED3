#!/usr/bin/env python3
"""Unit tests for internal CLI argv serialization (no binary run)."""

from bayesed import BayeSEDParams
from bayesed.cli_args import to_cli_args


def test_to_cli_args_galaxy_includes_core_flags():
    params = BayeSEDParams.galaxy(
        input_file='observation/test/gal.txt',
        outdir='tests/output_cli_args_unit',
        ssp_model='bc2003_hr_stelib_chab_neb_2000r',
        sfh_type='exponential',
        dal_law='calzetti',
    )
    args = to_cli_args(params)

    assert isinstance(args, list)
    assert '-i' in args
    assert 'observation/test/gal.txt' in args[args.index('-i') + 1]
    assert '--outdir' in args
    assert args[args.index('--outdir') + 1] == 'tests/output_cli_args_unit'
    assert '-ssp' in args
    assert '--sfh' in args
    assert '--dal' in args


def test_to_cli_args_passthrough_list():
    raw = ['-i', 'foo.txt', '--outdir', 'out']
    assert to_cli_args(raw) == raw


if __name__ == '__main__':
    test_to_cli_args_galaxy_includes_core_flags()
    test_to_cli_args_passthrough_list()
    print('test_cli_args: OK')
