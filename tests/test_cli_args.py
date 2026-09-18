#!/usr/bin/env python3
"""Unit tests for internal CLI argv serialization (no binary run)."""

import pytest

from bayesed import BayeSEDParams
from bayesed.cli_args import _format_SFR_over_params, to_cli_args
from bayesed.model import SEDModel
from bayesed.params import SFROverParams


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


def test_sfr_over_argv_variable_length():
    assert _format_SFR_over_params(SFROverParams()) == '10.0,100.0'
    assert (
        _format_SFR_over_params(SFROverParams(past_Myrs=[10.0, 100.0, 1000.0]))
        == '10.0,100.0,1000.0'
    )
    assert _format_SFR_over_params(SFROverParams(past_Myrs=[100.0])) == '100.0'

    params = BayeSEDParams.galaxy(
        input_file='observation/test/gal.txt',
        outdir='tests/output_cli_args_unit',
        ssp_model='bc2003_hr_stelib_chab_neb_2000r',
        sfh_type='exponential',
        dal_law='calzetti',
    )
    params.SFR_over = SFROverParams(past_Myrs=[5.0, 50.0, 500.0])
    args = to_cli_args(params)
    assert '--SFR_over' in args
    assert args[args.index('--SFR_over') + 1] == '5.0,50.0,500.0'


def test_add_model_igm_and_sfr_over_reach_argv():
    galaxy = SEDModel.create_galaxy()
    model = SEDModel()
    model.set_igm(igm_model=1)
    model.set_sfr_over(past_Myrs=(10.0, 100.0))

    params = BayeSEDParams(
        input_type=0,
        input_file='observation/test/gal.txt',
        outdir='tests/output_cli_args_unit',
    )
    params.add_galaxy(galaxy)
    params.add_model(model)

    assert params.IGM == 1
    assert params.SFR_over is not None
    args = to_cli_args(params)
    assert '--IGM' in args
    assert args[args.index('--IGM') + 1] == '1'
    assert '--SFR_over' in args
    assert args[args.index('--SFR_over') + 1] == '10.0,100.0'


def test_to_cli_args_includes_ntest():
    params = BayeSEDParams.galaxy(
        input_file='observation/test/gal.txt',
        outdir='tests/output_cli_args_unit',
        ssp_model='bc2003_hr_stelib_chab_neb_2000r',
        sfh_type='exponential',
        dal_law='calzetti',
    )
    params.Ntest = 2
    args = to_cli_args(params)
    assert '--Ntest' in args
    assert args[args.index('--Ntest') + 1] == '2'


def test_bayesed_params_rejects_unknown_field_with_suggestion():
    params = BayeSEDParams(
        input_type=0,
        input_file='observation/test/gal.txt',
    )
    with pytest.raises(AttributeError, match=r"no field 'igm'.*Did you mean 'IGM'"):
        params.igm = 1


if __name__ == '__main__':
    test_to_cli_args_galaxy_includes_core_flags()
    test_sfr_over_argv_variable_length()
    test_add_model_igm_and_sfr_over_reach_argv()
    test_to_cli_args_includes_ntest()
    test_bayesed_params_rejects_unknown_field_with_suggestion()
    print('test_cli_args: OK')
