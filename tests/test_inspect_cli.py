#!/usr/bin/env python3
"""Unit tests for the Launcher inspect path (ADR-0009; no binary, no HDF5)."""

from unittest.mock import MagicMock, patch

import json
from pathlib import Path

import pandas as pd
from astropy.table import Table

from bayesed.inspect_cli import (
    PLOT_VERBS,
    RESULTS_VERBS,
    parse_catalogs,
    parse_model_configs,
    parse_par_values,
    run_inspect as _run_inspect_impl,
)
from bayesed.launcher import run_launcher


def run_inspect(argv, *, check_outdir=False):
    """Test driver for ``inspect_cli.run_inspect``.

    By default patches ``_require_outdir`` so fake paths like ``out`` work.
    Pass ``check_outdir=True`` to exercise the real directory gate.
    """
    if check_outdir:
        return _run_inspect_impl(argv)
    with patch('bayesed.inspect_cli._require_outdir'):
        return _run_inspect_impl(argv)


def test_results_without_verbs_lists_commands_and_skips_binary():
    with patch('bayesed.launcher.subprocess.Popen') as mock_popen, \
         patch('sys.stderr', new_callable=MagicMock) as err:
        code = run_launcher(['results'])
    assert code == 1
    mock_popen.assert_not_called()
    written = ''.join(c.args[0] for c in err.write.call_args_list)
    for verb in RESULTS_VERBS:
        assert verb in written


def test_plot_without_verbs_lists_commands():
    with patch('sys.stderr', new_callable=MagicMock) as err:
        code = run_inspect(['plot', 'some/outdir'])
    assert code == 1
    written = ''.join(c.args[0] for c in err.write.call_args_list)
    assert 'Error: missing command' in written
    for verb in PLOT_VERBS:
        assert verb in written


def test_plot_without_outdir_prints_usage_only():
    with patch('sys.stderr', new_callable=MagicMock) as err:
        code = run_inspect(['plot'])
    assert code == 1
    written = ''.join(c.args[0] for c in err.write.call_args_list)
    assert 'Error: missing command' not in written
    assert written.startswith('Usage:')
    for verb in PLOT_VERBS:
        assert verb in written


def test_plot_leading_flag_prints_usage_only():
    with patch('sys.stderr', new_callable=MagicMock) as err:
        code = run_inspect(['plot', '--save'])
    assert code == 1
    written = ''.join(c.args[0] for c in err.write.call_args_list)
    assert 'Error: missing command' not in written
    assert written.startswith('Usage:')


def test_plot_posterior_without_outdir_names_the_missing_directory():
    with patch('bayesed.inspect_cli.BayeSEDResults') as ctor, \
         patch('sys.stderr', new_callable=MagicMock) as err:
        code = run_inspect(['plot', 'posterior'])
    assert code == 1
    ctor.assert_not_called()
    written = ''.join(c.args[0] for c in err.write.call_args_list)
    assert written.startswith('Error: missing outdir\nUsage:')
    assert 'posterior free' in written
    assert 'posterior derived' in written
    assert 'posterior --par' in written
    assert 'bestfit' not in written


def test_plot_bestfit_without_outdir_names_the_missing_directory():
    with patch('bayesed.inspect_cli.BayeSEDResults') as ctor, \
         patch('sys.stderr', new_callable=MagicMock) as err:
        code = run_inspect(['plot', 'bestfit', '--save'])
    assert code == 1
    ctor.assert_not_called()
    written = ''.join(c.args[0] for c in err.write.call_args_list)
    assert written.startswith('Error: missing outdir\nUsage:')
    assert 'bestfit' in written
    assert 'posterior' not in written


def test_plot_posterior_free_without_outdir_does_not_ask_for_a_mode():
    with patch('sys.stderr', new_callable=MagicMock) as err:
        code = run_inspect(['plot', 'posterior', 'free'])
    assert code == 1
    written = ''.join(c.args[0] for c in err.write.call_args_list)
    assert 'Error: missing outdir' in written
    assert 'specify free' not in written
    assert 'bestfit' not in written


def test_plot_verbs_without_outdir_cannot_be_combined():
    with patch('bayesed.inspect_cli.BayeSEDResults') as ctor, \
         patch('sys.stderr', new_callable=MagicMock) as err:
        code = run_inspect(['plot', 'posterior', 'bestfit'])
    assert code == 1
    ctor.assert_not_called()
    written = ''.join(c.args[0] for c in err.write.call_args_list)
    assert 'Error: posterior cannot be combined with other commands' in written
    assert 'Error: missing outdir' not in written
    assert 'bestfit [--cat' in written
    assert 'posterior free' in written


def test_plot_bestfit_flag_error_omits_posterior_usage():
    with patch('sys.stderr', new_callable=MagicMock) as err:
        code = run_inspect(['plot', 'out', 'bestfit', '--flux', 'jansky'])
    assert code == 1
    written = ''.join(c.args[0] for c in err.write.call_args_list)
    assert 'Error: --flux requires fnu, nufnu, or flambda' in written
    assert 'posterior' not in written


def test_plot_junk_before_verbs_is_missing_command():
    with patch('sys.stderr', new_callable=MagicMock) as err:
        code = run_inspect(['plot', 'out', 'junk', 'bestfit', 'posterior'])
    assert code == 1
    written = ''.join(c.args[0] for c in err.write.call_args_list)
    assert 'Error: missing command' in written
    assert 'cannot be combined' not in written
    assert 'bestfit [--cat' in written
    assert 'posterior free' in written


def test_plot_posterior_without_mode_omits_bestfit_usage():
    with patch('bayesed.inspect_cli.BayeSEDResults') as ctor, \
         patch('sys.stderr', new_callable=MagicMock) as err:
        code = run_inspect(['plot', 'out', 'posterior', '--save'])
    assert code == 1
    ctor.assert_not_called()
    written = ''.join(c.args[0] for c in err.write.call_args_list)
    assert 'Error: specify free, derived, or --par' in written
    assert 'bestfit' not in written
    assert 'posterior free' in written


def test_top_level_help_is_overview_not_binary():
    with patch('bayesed.launcher.subprocess.Popen') as mock_popen, \
         patch('bayesed.launcher.sys.stdout', new_callable=MagicMock) as out:
        code = run_launcher(['--help'])
    assert code == 0
    mock_popen.assert_not_called()
    written = ''.join(c.args[0] for c in out.write.call_args_list)
    assert 'bayesed3 run' in written


def test_results_outdir_only_prints_usage():
    with patch('bayesed.inspect_cli.list_catalog_names') as list_cat, \
         patch('bayesed.inspect_cli.BayeSEDResults') as ctor, \
         patch('sys.stderr', new_callable=MagicMock) as err:
        code = run_inspect(['results', 'out'])
    assert code == 1
    list_cat.assert_not_called()
    ctor.assert_not_called()
    written = ''.join(c.args[0] for c in err.write.call_args_list)
    assert 'Error: missing command' in written
    assert 'Usage:' in written
    assert 'Commands:' not in written
    assert 'results commands:' not in written
    assert 'summary [--cat PATTERN] [--cfg PATTERN] [--json]' in written
    assert 'par_names [--cat PATTERN] [--cfg PATTERN] [--json]' in written
    # summary / par_names lines omit --obj; evidence / objects include it
    # loose-match flags use PATTERN (not catalog/object/config/NAME)
    summary_line = next(
        line for line in written.splitlines() if ' summary ' in line
    )
    par_names_line = next(
        line for line in written.splitlines() if ' par_names ' in line
    )
    evidence_line = next(
        line for line in written.splitlines() if ' evidence ' in line
    )
    objects_line = next(
        line for line in written.splitlines() if ' objects ' in line
    )
    assert '--obj' not in summary_line
    assert '--obj' not in par_names_line
    assert '--obj' in evidence_line
    assert '--obj' in objects_line
    assert '[--obj PATTERN]' in evidence_line
    assert '[--cat PATTERN]' in summary_line
    assert 'catalog]' not in written
    assert 'object]' not in written
    assert 'config]' not in written
    assert '--par NAME' not in written
    assert 'par_values --par PATTERN' in written
    assert '[--wide] [--transpose]' in written


def test_results_outdir_json_without_command_prints_usage():
    with patch('bayesed.inspect_cli.list_catalog_names') as list_cat, \
         patch('sys.stderr', new_callable=MagicMock) as err:
        code = run_inspect(['results', 'out', '--json'])
    assert code == 1
    list_cat.assert_not_called()
    written = ''.join(c.args[0] for c in err.write.call_args_list)
    assert 'Error: missing command' in written
    assert 'Usage:' in written


def test_results_objects_json():
    fake = MagicMock()
    fake.list_objects.return_value = ['obj-1', 'obj-2']
    with patch('bayesed.inspect_cli.list_catalog_names', return_value=['gal']), \
         patch('bayesed.inspect_cli.BayeSEDResults', return_value=fake), \
         patch('sys.stdout', new_callable=MagicMock) as out:
        code = run_inspect(['results', 'out', 'objects', '--json'])
    assert code == 0
    written = ''.join(c.args[0] for c in out.write.call_args_list)
    assert 'obj-1' in written
    assert written.strip().startswith('[')


def test_results_evidence_json():
    table = Table({
        'ID': ['obj-1', 'obj-2'],
        'INSlogZ': [-15782.2, -24553.2],
        'logZ': [-15780.0, -24551.8],
        'INSlogZerr': [0.0389714, 0.16613],
        'logZerr': [0.808556, 0.769606],
    })
    fake = MagicMock()
    fake.get_evidence.return_value = table
    with patch('bayesed.inspect_cli.list_catalog_names', return_value=['gal']), \
         patch('bayesed.inspect_cli.BayeSEDResults', return_value=fake), \
         patch('sys.stdout', new_callable=MagicMock) as out:
        code = run_inspect(['results', 'out', 'evidence', '--json'])
    assert code == 0
    fake.get_evidence.assert_called_once_with(return_format='table')
    written = ''.join(c.args[0] for c in out.write.call_args_list)
    payload = json.loads(written)
    assert isinstance(payload, list)
    assert len(payload) == 2
    assert payload[0]['ID'] == 'obj-1'
    assert payload[0]['INSlogZ'] == -15782.2
    assert payload[0]['logZ'] == -15780.0
    assert 'log_evidence' not in payload[0]
    assert payload[1]['ID'] == 'obj-2'


def test_results_evidence_json_decodes_bytes_id():
    table = Table({
        'ID': [b'obj-bytes'],
        'INSlogZ': [-1.5],
        'logZ': [-1.4],
        'INSlogZerr': [0.1],
        'logZerr': [0.2],
    })
    fake = MagicMock()
    fake.get_evidence.return_value = table
    with patch('bayesed.inspect_cli.list_catalog_names', return_value=['gal']), \
         patch('bayesed.inspect_cli.BayeSEDResults', return_value=fake), \
         patch('sys.stdout', new_callable=MagicMock) as out:
        code = run_inspect(['results', 'out', 'evidence', '--json'])
    assert code == 0
    payload = json.loads(''.join(c.args[0] for c in out.write.call_args_list))
    assert payload[0]['ID'] == 'obj-bytes'
    assert isinstance(payload[0]['ID'], str)


def test_results_evidence_json_single_object_is_still_array():
    table = Table({
        'ID': ['obj-1'],
        'INSlogZ': [-1.5],
        'logZ': [-1.4],
        'INSlogZerr': [0.1],
        'logZerr': [0.2],
    })
    fake = MagicMock()
    fake.list_objects.return_value = ['obj-1', 'obj-2']
    fake.get_evidence.return_value = table
    with patch('bayesed.inspect_cli.list_catalog_names', return_value=['gal']), \
         patch('bayesed.inspect_cli.BayeSEDResults', return_value=fake), \
         patch('sys.stdout', new_callable=MagicMock) as out:
        code = run_inspect([
            'results', 'out', 'evidence', '--obj', 'obj-1', '--json',
        ])
    assert code == 0
    fake.get_evidence.assert_called_once_with(
        object_ids=['obj-1'], return_format='table',
    )
    payload = json.loads(''.join(c.args[0] for c in out.write.call_args_list))
    assert payload == [{
        'ID': 'obj-1',
        'INSlogZ': -1.5,
        'logZ': -1.4,
        'INSlogZerr': 0.1,
        'logZerr': 0.2,
    }]


def test_results_summary_json():
    fake = MagicMock()
    fake.output_dir = Path('/tmp/outdir')
    fake.catalog_name = 'gal'
    fake.object_id = None
    fake.hdf5_file = 'gal.h5'
    fake.get_parameter_names.return_value = ['a', 'b']
    fake.get_free_parameters.return_value = ['a']
    fake.get_derived_parameters.return_value = ['b']
    fake.list_objects.return_value = [b'obj-1']
    with patch('bayesed.inspect_cli.list_catalog_names', return_value=['gal']), \
         patch('bayesed.inspect_cli.BayeSEDResults', return_value=fake), \
         patch('sys.stdout', new_callable=MagicMock) as out:
        code = run_inspect(['results', 'out', 'summary', '--json'])
    assert code == 0
    fake.print_summary.assert_not_called()
    written = ''.join(c.args[0] for c in out.write.call_args_list)
    payload = json.loads(written)
    assert payload['output_dir'] == '/tmp/outdir'
    assert isinstance(payload['output_dir'], str)
    assert payload['catalog'] == 'gal'
    assert payload['objects'] == ['obj-1']
    assert payload['n_hdf5_columns'] == 2
    assert payload['n_free'] == 1
    assert payload['n_derived'] == 1
    assert payload['n_objects'] == 1
    assert 'free' not in payload or 'n_free' in payload
    assert '"free": [' not in written
    assert '"derived": [' not in written


def test_results_objects_json_decodes_bytes_ids():
    fake = MagicMock()
    fake.list_objects.return_value = [b'obj-a', b'obj-b']
    with patch('bayesed.inspect_cli.list_catalog_names', return_value=['gal']), \
         patch('bayesed.inspect_cli.BayeSEDResults', return_value=fake), \
         patch('sys.stdout', new_callable=MagicMock) as out:
        code = run_inspect(['results', 'out', 'objects', '--json'])
    assert code == 0
    payload = json.loads(''.join(c.args[0] for c in out.write.call_args_list))
    assert payload == ['obj-a', 'obj-b']


def test_results_par_values_json():
    fake = MagicMock()
    table = pd.DataFrame({'z': [1.0]})
    fake.get_parameter_names.return_value = ['z']
    fake.get_parameter_values.return_value = table
    with patch('bayesed.inspect_cli.list_catalog_names', return_value=['gal']), \
         patch('bayesed.inspect_cli.BayeSEDResults', return_value=fake), \
         patch('sys.stdout', new_callable=MagicMock) as out:
        code = run_inspect([
            'results', 'out', 'par_values', '--par', 'z', '--json',
        ])
    assert code == 0
    written = ''.join(c.args[0] for c in out.write.call_args_list)
    assert written.strip() == '[{"z": 1.0}]'


def test_results_par_values_json_from_astropy_table():
    """astropy Table has to_pandas, not to_dict — must still emit records."""
    fake = MagicMock()
    fake.get_parameter_names.return_value = ['log(age/yr)[0,1]_{mean}']
    fake.get_parameter_values.return_value = Table(
        {'ID': ['obj-1'], 'log(age/yr)[0,1]_{mean}': [10.08]},
    )
    with patch('bayesed.inspect_cli.list_catalog_names', return_value=['gal']), \
         patch('bayesed.inspect_cli.BayeSEDResults', return_value=fake), \
         patch('sys.stdout', new_callable=MagicMock) as out:
        code = run_inspect([
            'results', 'out', 'par_values', '--par', 'age', '--json',
        ])
    assert code == 0
    written = ''.join(c.args[0] for c in out.write.call_args_list).strip()
    assert written.startswith('[')
    assert 'obj-1' in written
    assert '10.08' in written
    assert '\\n' not in written


def test_results_catalogs_json():
    with patch('bayesed.inspect_cli.list_catalog_names', return_value=['gal']), \
         patch('sys.stdout', new_callable=MagicMock) as out:
        code = run_inspect(['results', 'out', 'catalogs', '--json'])
    assert code == 0
    written = ''.join(c.args[0] for c in out.write.call_args_list)
    assert '"gal"' in written


def test_results_model_configs_json():
    with patch('bayesed.inspect_cli.list_catalog_names', return_value=['gal']), \
         patch(
             'bayesed.inspect_cli.list_model_configs', return_value=['cfg0'],
         ), \
         patch('sys.stdout', new_callable=MagicMock) as out:
        code = run_inspect([
            'results', 'out', 'model_configs', '--cat', 'gal', '--json',
        ])
    assert code == 0
    written = ''.join(c.args[0] for c in out.write.call_args_list)
    assert '"cfg0"' in written


def test_results_fit_quality_json_is_array_per_object():
    fake = MagicMock()
    fake.list_objects.return_value = ['obj-1', 'obj-2']
    fake.get_fit_quality.return_value = {
        'object_id': 'obj-1', 'datasets': [],
    }
    with patch('bayesed.inspect_cli.list_catalog_names', return_value=['gal']), \
         patch('bayesed.inspect_cli.BayeSEDResults', return_value=fake), \
         patch('sys.stdout', new_callable=MagicMock) as out:
        code = run_inspect(['results', 'out', 'fit_quality', '--json'])
    assert code == 0
    assert fake.get_fit_quality.call_count == 2
    assert fake.get_fit_quality.call_args.kwargs == {
        'band_edges': None, 'top_n': 20, 'sigma_threshold': 3.0,
    }
    written = ''.join(c.args[0] for c in out.write.call_args_list)
    payload = json.loads(written)
    assert isinstance(payload, list)
    assert len(payload) == 2


def test_results_fit_quality_obj_filter_limits_objects():
    fake = MagicMock()
    fake.list_objects.return_value = ['obj-1', 'obj-2']
    fake.get_fit_quality.return_value = {
        'object_id': 'obj-1', 'datasets': [],
    }
    with patch('bayesed.inspect_cli.list_catalog_names', return_value=['gal']), \
         patch('bayesed.inspect_cli.BayeSEDResults', return_value=fake), \
         patch('sys.stdout', new_callable=MagicMock) as out:
        code = run_inspect([
            'results', 'out', 'fit_quality', '--obj', 'obj-1', '--json',
        ])
    assert code == 0
    assert fake.get_fit_quality.call_count == 1
    assert fake.get_fit_quality.call_args.args == ('obj-1',)
    payload = json.loads(''.join(c.args[0] for c in out.write.call_args_list))
    assert [rec['object_id'] for rec in payload] == ['obj-1']


def test_results_fit_quality_human_prints_table():
    fake = MagicMock()
    fake.list_objects.return_value = ['obj-1']
    fake.get_fit_quality.return_value = {
        'object_id': 'obj-1',
        'chi2': 2.5, 'chi2_red': 2.5, 'chi2_source': 'header',
        'datasets': [{
            'kind': 'phot', 'label': 'u',
            'wl_range_rest': [0.35, 0.45],
            'n': 3, 'chi2': 1.25, 'chi2_red': 0.416,
            'resid_median': 0.0, 'resid_rms': 0.645,
            'sigma_threshold': 3.0, 'frac_outside': 0.0,
            'flags': [], 'outliers': [],
        }],
    }
    with patch('bayesed.inspect_cli.list_catalog_names', return_value=['gal']), \
         patch('bayesed.inspect_cli.BayeSEDResults', return_value=fake), \
         patch('sys.stdout', new_callable=MagicMock) as out:
        code = run_inspect(['results', 'out', 'fit_quality'])
    assert code == 0
    written = ''.join(c.args[0] for c in out.write.call_args_list)
    assert 'obj-1' in written
    assert 'u' in written


def test_results_fit_quality_band_top_sigma_passed_through():
    fake = MagicMock()
    fake.list_objects.return_value = ['obj-1']
    fake.get_fit_quality.return_value = {
        'object_id': 'obj-1', 'datasets': [],
    }
    with patch('bayesed.inspect_cli.list_catalog_names', return_value=['gal']), \
         patch('bayesed.inspect_cli.BayeSEDResults', return_value=fake), \
         patch('sys.stdout', new_callable=MagicMock):
        code = run_inspect([
            'results', 'out', 'fit_quality',
            '--band', '0.35,0.55,0.9', '--top', '5', '--sigma', '2.5',
            '--json',
        ])
    assert code == 0
    assert fake.get_fit_quality.call_args.kwargs == {
        'band_edges': [0.35, 0.55, 0.9],
        'top_n': 5,
        'sigma_threshold': 2.5,
    }


def test_results_fit_quality_band_requires_increasing_edges():
    with patch('sys.stderr', new_callable=MagicMock) as err:
        code = run_inspect([
            'results', 'out', 'fit_quality', '--band', '0.9,0.35',
        ])
    assert code == 1
    written = ''.join(c.args[0] for c in err.write.call_args_list)
    assert 'Error: --band requires strictly increasing edges' in written


def test_results_fit_quality_band_requires_numbers():
    with patch('sys.stderr', new_callable=MagicMock) as err:
        code = run_inspect([
            'results', 'out', 'fit_quality', '--band', 'a,b',
        ])
    assert code == 1
    written = ''.join(c.args[0] for c in err.write.call_args_list)
    assert '--band requires comma-separated numbers' in written


def test_results_fit_quality_top_requires_positive_integer():
    for value in ('0', '-1', 'x'):
        with patch('sys.stderr', new_callable=MagicMock) as err:
            code = run_inspect([
                'results', 'out', 'fit_quality', '--top', value,
            ])
        assert code == 1
        written = ''.join(c.args[0] for c in err.write.call_args_list)
        assert 'Error: --top requires a positive integer' in written


def test_results_fit_quality_sigma_requires_positive_number():
    for value in ('0', '-1', 'abc'):
        with patch('sys.stderr', new_callable=MagicMock) as err:
            code = run_inspect([
                'results', 'out', 'fit_quality', '--sigma', value,
            ])
        assert code == 1
        written = ''.join(c.args[0] for c in err.write.call_args_list)
        assert 'Error: --sigma requires a positive number' in written


def test_results_fit_quality_without_outdir_names_the_missing_directory():
    with patch('bayesed.inspect_cli.BayeSEDResults') as ctor, \
         patch('sys.stderr', new_callable=MagicMock) as err:
        code = run_inspect(['results', 'fit_quality', '--json'])
    assert code == 1
    ctor.assert_not_called()
    written = ''.join(c.args[0] for c in err.write.call_args_list)
    assert written.startswith('Error: missing outdir\nUsage:')
    assert 'fit_quality [--cat PATTERN]' in written


def test_missing_cfg_is_one_line_with_cfg_flag():
    with patch('bayesed.inspect_cli.list_catalog_names', return_value=['gal']), \
         patch(
             'bayesed.inspect_cli.BayeSEDResults',
             side_effect=ValueError(
                 "Multiple model configurations found for catalog 'gal'. "
                 "Please specify model_config parameter. "
                 "Available configs: ['cfg0', 'cfg1']"
             ),
         ), \
         patch('sys.stderr', new_callable=MagicMock) as err:
        code = run_inspect(['results', 'out', 'summary', '--cat', 'gal'])
    assert code == 1
    written = ''.join(c.args[0] for c in err.write.call_args_list)
    assert written == (
        "Error: Multiple model configurations found. "
        "Please specify config with --cfg. "
        "Available configs: ['cfg0', 'cfg1']\n"
    )


def test_plot_missing_cfg_is_one_line_with_cfg_flag():
    with patch('bayesed.inspect_cli.list_catalog_names', return_value=['gal']), \
         patch(
             'bayesed.inspect_cli.BayeSEDResults',
             side_effect=ValueError(
                 "Multiple model configurations found for catalog 'gal'. "
                 "Please specify model_config parameter. "
                 "Available configs: ['cfg0', 'cfg1']"
             ),
         ), \
         patch('sys.stderr', new_callable=MagicMock) as err:
        code = run_inspect(['plot', 'out', 'bestfit', '--cat', 'gal'])
    assert code == 1
    written = ''.join(c.args[0] for c in err.write.call_args_list)
    assert written == (
        "Error: Multiple model configurations found. "
        "Please specify config with --cfg. "
        "Available configs: ['cfg0', 'cfg1']\n"
    )


def test_unknown_catalog_lists_available_in_one_line():
    with patch('bayesed.inspect_cli.list_catalog_names', return_value=['gal']), \
         patch('bayesed.inspect_cli.BayeSEDResults') as ctor, \
         patch('sys.stderr', new_callable=MagicMock) as err:
        code = run_inspect([
            'results', 'out', 'summary', '--cat', 'qso',
        ])
    assert code == 1
    ctor.assert_not_called()
    written = ''.join(c.args[0] for c in err.write.call_args_list)
    assert written == (
        "Error: No catalog matching 'qso'. Available catalogs: ['gal']\n"
    )


def test_missing_catalog_is_one_line_with_cat_flag():
    with patch('bayesed.inspect_cli.list_catalog_names', return_value=['gal', 'qso']), \
         patch(
             'bayesed.inspect_cli.BayeSEDResults',
             side_effect=ValueError(
                 "Multiple catalogs found. Please specify catalog_name parameter. "
                 "Available catalogs: ['gal', 'qso']"
             ),
         ), \
         patch('sys.stderr', new_callable=MagicMock) as err:
        code = run_inspect(['results', 'out', 'summary'])
    assert code == 1
    written = ''.join(c.args[0] for c in err.write.call_args_list)
    assert written == (
        "Error: Multiple catalogs found. Please specify catalog "
        "with --cat. Available catalogs: ['gal', 'qso']\n"
    )


def test_library_many_configs_truncates_available_list():
    configs = [f'c{i}' for i in range(12)]
    with patch('bayesed.inspect_cli.list_catalog_names', return_value=['gal']), \
         patch(
             'bayesed.inspect_cli.BayeSEDResults',
             side_effect=ValueError(
                 "Multiple model configurations found for catalog 'gal'. "
                 "Please specify model_config parameter. "
                 f"Available configs: {configs}"
             ),
         ), \
         patch('sys.stderr', new_callable=MagicMock) as err:
        code = run_inspect(['results', 'out', 'summary', '--cat', 'gal'])
    assert code == 1
    written = ''.join(c.args[0] for c in err.write.call_args_list)
    assert written == (
        "Error: Multiple model configurations found. "
        "Please specify config with --cfg. "
        "Available configs: "
        "['c0', 'c1', 'c2', 'c3', 'c4', '...', "
        "'c7', 'c8', 'c9', 'c10', 'c11']\n"
    )
    assert 'c5' not in written
    assert 'c6' not in written


def test_many_catalogs_truncates_available_list():
    catalogs = [f'c{i:02d}' for i in range(12)]
    with patch('bayesed.inspect_cli.list_catalog_names', return_value=catalogs), \
         patch('bayesed.inspect_cli.list_model_configs') as listed, \
         patch('sys.stderr', new_callable=MagicMock) as err:
        code = run_inspect(['results', 'out', 'model_configs'])
    assert code == 1
    listed.assert_not_called()
    written = ''.join(c.args[0] for c in err.write.call_args_list)
    assert written == (
        "Error: Multiple catalogs found. Please specify catalog "
        "with --cat. Available catalogs: "
        "['c00', 'c01', 'c02', 'c03', 'c04', '...', "
        "'c07', 'c08', 'c09', 'c10', 'c11']\n"
    )
    assert 'c05' not in written
    assert 'c06' not in written


def test_results_summary_uses_bayesed_results():
    fake = MagicMock()
    fake.output_dir = 'out'
    fake.catalog_name = 'gal'
    fake.hdf5_file = 'gal.h5'
    fake.get_parameter_names.return_value = ['a']
    fake.get_free_parameters.return_value = ['a']
    fake.get_derived_parameters.return_value = []
    fake.list_objects.return_value = ['obj-1']
    with patch('bayesed.inspect_cli.list_catalog_names', return_value=['gal']), \
         patch('bayesed.inspect_cli.BayeSEDResults', return_value=fake) as ctor:
        code = run_inspect(['results', 'out', 'summary'])
    assert code == 0
    ctor.assert_called_once_with('out')
    fake.print_summary.assert_called_once_with()


def test_results_summary_ignores_obj():
    fake = MagicMock()
    fake.list_objects.return_value = ['obj-1', 'obj-2']
    with patch('bayesed.inspect_cli.list_catalog_names', return_value=['gal']), \
         patch('bayesed.inspect_cli.BayeSEDResults', return_value=fake):
        code = run_inspect([
            'results', 'out', 'summary', '--obj', 'obj-1',
        ])
    assert code == 0
    fake.print_summary.assert_called_once_with()
    fake.get_evidence.assert_not_called()


def test_results_cat_flag_loads_catalog():
    fake = MagicMock()
    fake.list_objects.return_value = ['obj-1']
    with patch('bayesed.inspect_cli.list_catalog_names', return_value=['gal']), \
         patch('bayesed.inspect_cli.BayeSEDResults', return_value=fake) as ctor:
        code = run_inspect(['results', 'out', 'objects', '--cat', 'gal'])
    assert code == 0
    ctor.assert_called_once_with('out', catalog_name='gal')
    assert fake.list_objects.called


def test_results_rejects_leftover_positionals():
    with patch('bayesed.inspect_cli.BayeSEDResults') as ctor, \
         patch('sys.stderr', new_callable=MagicMock) as err:
        code = run_inspect(['results', 'out', 'summary', 'gal'])
    assert code == 1
    ctor.assert_not_called()
    written = ''.join(c.args[0] for c in err.write.call_args_list)
    assert "Error: unexpected argument 'gal'" in written


def test_results_obj_flag_filters_without_ctor_object_id():
    fake = MagicMock()
    fake.list_objects.return_value = ['obj-1', 'obj-2']
    with patch('bayesed.inspect_cli.list_catalog_names', return_value=['gal']), \
         patch('bayesed.inspect_cli.BayeSEDResults', return_value=fake) as ctor, \
         patch('sys.stdout', new_callable=MagicMock) as out:
        code = run_inspect(['results', 'out', 'objects', '--obj', 'obj-1'])
    assert code == 0
    ctor.assert_called_once_with('out')
    written = ''.join(c.args[0] for c in out.write.call_args_list)
    assert written.splitlines() == ['obj-1']


def test_results_cfg_passed_to_ctor():
    fake = MagicMock()
    fake.list_objects.return_value = []
    with patch('bayesed.inspect_cli.list_catalog_names', return_value=['gal']), \
         patch(
             'bayesed.inspect_cli.list_model_configs',
             return_value=['cfg0', 'cfg1'],
         ), \
         patch('bayesed.inspect_cli.BayeSEDResults', return_value=fake) as ctor:
        code = run_inspect([
            'results', 'out', 'summary', '--cat', 'gal', '--cfg', 'cfg0',
        ])
    assert code == 0
    ctor.assert_called_once_with(
        'out', catalog_name='gal', model_config='cfg0',
    )


def test_results_flags_must_follow_verb():
    with patch('bayesed.inspect_cli.BayeSEDResults') as ctor, \
         patch('sys.stderr', new_callable=MagicMock) as err:
        code = run_inspect([
            'results', 'out', 'summary', '--cat', 'gal', 'evidence',
        ])
    assert code == 1
    ctor.assert_not_called()
    written = ''.join(c.args[0] for c in err.write.call_args_list)
    assert "Error: unexpected argument 'evidence'" in written


def test_results_rejects_stacked_query_verbs():
    with patch('bayesed.inspect_cli.BayeSEDResults') as ctor, \
         patch('sys.stderr', new_callable=MagicMock) as err:
        code = run_inspect(['results', 'out', 'evidence', 'par_names'])
    assert code == 1
    ctor.assert_not_called()
    written = ''.join(c.args[0] for c in err.write.call_args_list)
    assert 'cannot be combined' in written


def test_results_non_verb_before_query_is_missing_command():
    with patch('bayesed.inspect_cli.BayeSEDResults') as ctor, \
         patch('sys.stderr', new_callable=MagicMock) as err:
        code = run_inspect(['results', 'out', 'junk', 'summary'])
    assert code == 1
    ctor.assert_not_called()
    written = ''.join(c.args[0] for c in err.write.call_args_list)
    assert 'Error: missing command' in written
    assert 'cannot be combined' not in written
    assert 'Usage:' in written


def test_results_non_verb_before_catalogs_is_missing_command():
    with patch('bayesed.inspect_cli.BayeSEDResults') as ctor, \
         patch('sys.stderr', new_callable=MagicMock) as err:
        code = run_inspect(['results', 'out', 'junk', 'catalogs'])
    assert code == 1
    ctor.assert_not_called()
    written = ''.join(c.args[0] for c in err.write.call_args_list)
    assert 'Error: missing command' in written
    assert 'cannot be combined' not in written
    assert 'Usage:' in written


def test_parse_single_command_non_verb_prefix_is_missing_command():
    for parse, tokens in (
        (parse_catalogs, ['out', 'junk', 'catalogs']),
        (parse_model_configs, ['out', 'junk', 'model_configs']),
        (parse_par_values, ['out', 'junk', 'par_values', '--par', 'z']),
    ):
        try:
            parse(tokens)
            raise AssertionError(f'expected ValueError from {parse.__name__}')
        except ValueError as e:
            assert str(e) == 'missing command'


def test_results_evidence_calls_existing_method():
    fake = MagicMock()
    fake.get_evidence.return_value = {
        'log_evidence': 1.0,
        'log_evidence_error': 0.1,
    }
    with patch('bayesed.inspect_cli.list_catalog_names', return_value=['gal']), \
         patch('bayesed.inspect_cli.BayeSEDResults', return_value=fake):
        code = run_inspect(['results', 'out', 'evidence'])
    assert code == 0
    fake.get_evidence.assert_called_once()


def test_results_par_names_calls_existing_methods():
    fake = MagicMock()
    fake.get_free_parameters.return_value = ['z']
    fake.get_derived_parameters.return_value = ['log(Mstar)']
    with patch('bayesed.inspect_cli.list_catalog_names', return_value=['gal']), \
         patch('bayesed.inspect_cli.BayeSEDResults', return_value=fake):
        code = run_inspect(['results', 'out', 'par_names'])
    assert code == 0
    fake.get_free_parameters.assert_called_once()
    fake.get_derived_parameters.assert_called_once()


def test_plot_bestfit_calls_plot_bestfit():
    fake = MagicMock()
    fake.object_id = None
    fake.list_objects.return_value = ['only']
    with patch('bayesed.inspect_cli.list_catalog_names', return_value=['gal']), \
         patch('bayesed.inspect_cli.BayeSEDResults', return_value=fake):
        code = run_inspect(['plot', 'out', 'bestfit'])
    assert code == 0
    fake.plot_bestfit.assert_called_once()
    fake.plot_posterior_free.assert_not_called()
    assert fake.plot_bestfit.call_args.kwargs.get('show') is True


def test_plot_bestfit_without_save_does_not_write_png():
    fake = MagicMock()
    fake.list_objects.return_value = ['only']
    with patch('bayesed.inspect_cli.list_catalog_names', return_value=['gal']), \
         patch('bayesed.inspect_cli.BayeSEDResults', return_value=fake):
        code = run_inspect(['plot', 'out', 'bestfit'])
    assert code == 0
    kwargs = fake.plot_bestfit.call_args.kwargs
    assert kwargs.get('save') is False
    assert 'output_file' not in kwargs


def _plot_bestfit_kwargs(argv):
    fake = MagicMock()
    fake.list_objects.return_value = ['only']
    with patch('bayesed.inspect_cli.list_catalog_names', return_value=['gal']), \
         patch('bayesed.inspect_cli.BayeSEDResults', return_value=fake):
        code = run_inspect(argv)
    return code, fake.plot_bestfit.call_args.kwargs if code == 0 else None


def test_plot_bestfit_flux_passes_flux_unit():
    code, kwargs = _plot_bestfit_kwargs(['plot', 'out', 'bestfit', '--flux', 'nufnu'])
    assert code == 0
    assert kwargs.get('flux_unit') == 'nufnu'


def test_plot_bestfit_flux_rejects_unknown_unit():
    with patch('sys.stderr', new_callable=MagicMock) as err:
        code = run_inspect(['plot', 'out', 'bestfit', '--flux', 'jansky'])
    assert code == 1
    written = ''.join(c.args[0] for c in err.write.call_args_list)
    assert 'Error: --flux requires fnu, nufnu, or flambda' in written


def test_plot_bestfit_log_axis_passes_use_log_scale():
    for axis in ('x', 'y', 'both'):
        code, kwargs = _plot_bestfit_kwargs(
            ['plot', 'out', 'bestfit', '--log', axis],
        )
        assert code == 0
        assert kwargs.get('use_log_scale') == axis


def test_plot_bestfit_linear_passes_use_log_scale_false():
    code, kwargs = _plot_bestfit_kwargs(['plot', 'out', 'bestfit', '--linear'])
    assert code == 0
    assert kwargs.get('use_log_scale') is False


def test_plot_bestfit_log_and_linear_rejected():
    with patch('sys.stderr', new_callable=MagicMock) as err:
        code = run_inspect(['plot', 'out', 'bestfit', '--log', 'x', '--linear'])
    assert code == 1
    written = ''.join(c.args[0] for c in err.write.call_args_list)
    assert 'Error: --log and --linear cannot be combined' in written


def test_plot_bestfit_log_rejects_unknown_axis():
    with patch('sys.stderr', new_callable=MagicMock) as err:
        code = run_inspect(['plot', 'out', 'bestfit', '--log', 'xy'])
    assert code == 1
    written = ''.join(c.args[0] for c in err.write.call_args_list)
    assert 'Error: --log requires x, y, or both' in written


def test_plot_bestfit_switch_flags_pass_false():
    cases = (
        ('--no-lines', 'show_emission_lines'),
        ('--no-total', 'show_total'),
        ('--no-components', 'show_components'),
        ('--full-range', 'focus_on_data_range'),
    )
    for flag, key in cases:
        code, kwargs = _plot_bestfit_kwargs(['plot', 'out', 'bestfit', flag])
        assert code == 0
        assert kwargs.get(key) is False


def test_plot_bestfit_show_residuals_passes_true():
    code, kwargs = _plot_bestfit_kwargs(
        ['plot', 'out', 'bestfit', '--show-residuals'],
    )
    assert code == 0
    assert kwargs.get('show_residuals') is True


def test_plot_bestfit_omits_appearance_kwargs():
    code, kwargs = _plot_bestfit_kwargs(['plot', 'out', 'bestfit'])
    assert code == 0
    for key in (
        'use_rest_frame', 'flux_unit', 'use_log_scale',
        'show_emission_lines', 'show_total', 'show_components',
        'focus_on_data_range', 'show_residuals',
    ):
        assert key not in kwargs


def test_plot_posterior_rejects_bestfit_appearance_flag():
    with patch('bayesed.inspect_cli.BayeSEDResults') as ctor, \
         patch('sys.stderr', new_callable=MagicMock) as err:
        code = run_inspect(['plot', 'out', 'posterior', 'free', '--log', 'x'])
    assert code == 1
    ctor.assert_not_called()
    written = ''.join(c.args[0] for c in err.write.call_args_list)
    assert 'Error: unknown option --log' in written


def test_plot_bestfit_observed_frame_passes_use_rest_frame_false():
    fake = MagicMock()
    fake.list_objects.return_value = ['only']
    with patch('bayesed.inspect_cli.list_catalog_names', return_value=['gal']), \
         patch('bayesed.inspect_cli.BayeSEDResults', return_value=fake):
        code = run_inspect(['plot', 'out', 'bestfit', '--observed-frame'])
    assert code == 0
    assert fake.plot_bestfit.call_args.kwargs.get('use_rest_frame') is False


def test_plot_bestfit_save_writes_sidecar():
    fake = MagicMock()
    fake.list_objects.return_value = ['only']
    with patch('bayesed.inspect_cli.list_catalog_names', return_value=['gal']), \
         patch('bayesed.inspect_cli.BayeSEDResults', return_value=fake):
        code = run_inspect(['plot', 'out', 'bestfit', '--save'])
    assert code == 0
    kwargs = fake.plot_bestfit.call_args.kwargs
    assert kwargs.get('save') is True
    assert 'output_file' not in kwargs


def test_plot_posterior_cannot_mix_with_bestfit():
    with patch('bayesed.inspect_cli.BayeSEDResults') as ctor, \
         patch('sys.stderr', new_callable=MagicMock) as err:
        code = run_inspect(['plot', 'out', 'bestfit', 'posterior'])
    assert code == 1
    ctor.assert_not_called()
    written = ''.join(c.args[0] for c in err.write.call_args_list)
    assert 'Error: posterior cannot be combined with other commands' in written


def test_plot_posterior_then_bestfit_cannot_mix():
    with patch('bayesed.inspect_cli.BayeSEDResults') as ctor, \
         patch('sys.stderr', new_callable=MagicMock) as err:
        code = run_inspect(['plot', 'out', 'posterior', 'bestfit'])
    assert code == 1
    ctor.assert_not_called()
    written = ''.join(c.args[0] for c in err.write.call_args_list)
    assert 'Error: posterior cannot be combined with other commands' in written
    assert 'bestfit [--cat' in written
    assert 'posterior free' in written


def test_plot_multiple_objects_without_obj_errors():
    fake = MagicMock()
    fake.object_id = None
    fake.list_objects.return_value = ['obj-a', 'obj-b']
    with patch('bayesed.inspect_cli.list_catalog_names', return_value=['gal']), \
         patch('bayesed.inspect_cli.BayeSEDResults', return_value=fake), \
         patch('sys.stderr', new_callable=MagicMock) as err:
        code = run_inspect(['plot', 'out', 'bestfit', '--cat', 'gal'])
    assert code == 1
    written = ''.join(c.args[0] for c in err.write.call_args_list)
    assert written == (
        "Error: Multiple objects found. Please specify object "
        "with --obj. Available objects: ['obj-a', 'obj-b']\n"
    )
    fake.plot_bestfit.assert_not_called()


def test_plot_many_objects_without_obj_lists_head_and_tail():
    ids = [f'o{i:02d}' for i in range(12)]
    fake = MagicMock()
    fake.object_id = None
    fake.list_objects.return_value = ids
    with patch('bayesed.inspect_cli.list_catalog_names', return_value=['gal']), \
         patch('bayesed.inspect_cli.BayeSEDResults', return_value=fake), \
         patch('sys.stderr', new_callable=MagicMock) as err:
        code = run_inspect(['plot', 'out', 'bestfit', '--cat', 'gal'])
    assert code == 1
    written = ''.join(c.args[0] for c in err.write.call_args_list)
    assert written == (
        "Error: Multiple objects found. Please specify object "
        "with --obj. Available objects: "
        "['o00', 'o01', 'o02', 'o03', 'o04', '...', "
        "'o07', 'o08', 'o09', 'o10', 'o11']\n"
    )
    assert 'o05' not in written
    assert 'o06' not in written
    fake.plot_bestfit.assert_not_called()


def test_plot_single_object_without_obj_uses_it_silently():
    fake = MagicMock()
    fake.object_id = None
    fake.list_objects.return_value = ['only']
    with patch('bayesed.inspect_cli.list_catalog_names', return_value=['gal']), \
         patch('bayesed.inspect_cli.BayeSEDResults', return_value=fake), \
         patch('sys.stdout', new_callable=MagicMock) as out, \
         patch('sys.stderr', new_callable=MagicMock) as err:
        code = run_inspect(['plot', 'out', 'bestfit', '--cat', 'gal'])
    assert code == 0
    assert out.write.call_args_list == []
    assert err.write.call_args_list == []
    assert fake.plot_bestfit.call_args.kwargs.get('object_id') == 'only'


def test_plot_with_object_does_not_list():
    fake = MagicMock()
    fake.object_id = 'obj-b'
    fake.list_objects.return_value = ['obj-a', 'obj-b']
    with patch('bayesed.inspect_cli.list_catalog_names', return_value=['gal']), \
         patch('bayesed.inspect_cli.BayeSEDResults', return_value=fake), \
         patch('sys.stdout', new_callable=MagicMock) as out:
        code = run_inspect([
            'plot', 'out', 'bestfit', '--cat', 'gal', '--obj', 'obj-b',
        ])
    assert code == 0
    written = ''.join(c.args[0] for c in out.write.call_args_list)
    assert 'using first' not in written
    fake.plot_bestfit.assert_called_once()


def test_plot_bestfit_passes_catalog_and_object():
    fake = MagicMock()
    fake.object_id = None
    fake.list_objects.return_value = ['obj-1']
    with patch('bayesed.inspect_cli.list_catalog_names', return_value=['gal']), \
         patch('bayesed.inspect_cli.BayeSEDResults', return_value=fake) as ctor:
        code = run_inspect([
            'plot', 'out', 'bestfit', '--cat', 'gal', '--obj', 'obj-1',
        ])
    assert code == 0
    ctor.assert_called_once_with('out', catalog_name='gal')
    fake.plot_bestfit.assert_called_once()
    assert fake.plot_bestfit.call_args.kwargs.get('show') is True
    assert fake.plot_bestfit.call_args.kwargs.get('object_id') == 'obj-1'


def test_plot_bestfit_cfg_passed_to_results():
    fake = MagicMock()
    fake.object_id = None
    fake.list_objects.return_value = ['only']
    cfg = '0Stellar+Nebular_2dal8_10_sys_err0'
    with patch('bayesed.inspect_cli.list_catalog_names', return_value=['gal']), \
         patch('bayesed.inspect_cli.list_model_configs', return_value=[cfg]), \
         patch('bayesed.inspect_cli.BayeSEDResults', return_value=fake) as ctor:
        code = run_inspect([
            'plot', 'out', 'bestfit', '--cat', 'gal',
            '--cfg', cfg,
        ])
    assert code == 0
    ctor.assert_called_once_with(
        'out',
        catalog_name='gal',
        model_config=cfg,
    )


def test_plot_bestfit_cfg_requires_value():
    with patch('bayesed.inspect_cli.BayeSEDResults') as ctor, \
         patch('sys.stderr', new_callable=MagicMock) as err:
        code = run_inspect(['plot', 'out', 'bestfit', '--cfg'])
    assert code == 1
    ctor.assert_not_called()
    written = ''.join(c.args[0] for c in err.write.call_args_list)
    assert 'Error: --cfg requires a value' in written


def test_plot_par_missing_value_before_another_flag():
    with patch('bayesed.inspect_cli.BayeSEDResults') as ctor, \
         patch('sys.stderr', new_callable=MagicMock) as err:
        code = run_inspect([
            'plot', 'out', 'posterior', '--par', '--cat', 'gal',
            '--obj', 'obj-1',
        ])
    assert code == 1
    ctor.assert_not_called()
    written = ''.join(c.args[0] for c in err.write.call_args_list)
    assert written.startswith('Error: --par requires a value\n')
    assert "unexpected argument 'gal'" not in written


def test_plot_cat_missing_value_before_another_flag():
    with patch('bayesed.inspect_cli.BayeSEDResults') as ctor, \
         patch('sys.stderr', new_callable=MagicMock) as err:
        code = run_inspect([
            'plot', 'out', 'bestfit', '--cat', '--obj', 'obj-1',
        ])
    assert code == 1
    ctor.assert_not_called()
    written = ''.join(c.args[0] for c in err.write.call_args_list)
    assert 'Error: --cat requires a value' in written


def test_plot_lab_missing_value_before_another_flag():
    with patch('bayesed.inspect_cli.BayeSEDResults') as ctor, \
         patch('sys.stderr', new_callable=MagicMock) as err:
        code = run_inspect([
            'plot', 'out', 'posterior', '--par', 'z', '--lab', '--cat', 'gal',
        ])
    assert code == 1
    ctor.assert_not_called()
    written = ''.join(c.args[0] for c in err.write.call_args_list)
    assert 'Error: --lab requires a value' in written


def test_results_obj_missing_value_before_another_flag():
    with patch('bayesed.inspect_cli.BayeSEDResults') as ctor, \
         patch('sys.stderr', new_callable=MagicMock) as err:
        code = run_inspect([
            'results', 'out', 'summary', '--obj', '--cat', 'gal',
        ])
    assert code == 1
    ctor.assert_not_called()
    written = ''.join(c.args[0] for c in err.write.call_args_list)
    assert 'Error: --obj requires a value' in written


def test_plot_bestfit_rejects_leftover_positionals():
    with patch('bayesed.inspect_cli.BayeSEDResults') as ctor, \
         patch('sys.stderr', new_callable=MagicMock) as err:
        code = run_inspect(['plot', 'out', 'bestfit', 'gal'])
    assert code == 1
    ctor.assert_not_called()
    written = ''.join(c.args[0] for c in err.write.call_args_list)
    assert "Error: unexpected argument 'gal'" in written


def test_priors_list_without_dir_lists_types():
    with patch('bayesed.inspect_cli.SEDInference') as inf_cls:
        inf = inf_cls.return_value
        code = run_inspect(['priors', 'list'])
    assert code == 0
    inf.list_prior_types.assert_called_once()


def test_priors_list_dir_uses_existing_loader_and_printer():
    manager = MagicMock()
    editor = MagicMock()
    with patch('bayesed.inspect_cli.os.path.isdir', return_value=True), \
         patch('bayesed.inspect_cli.glob.glob', return_value=['/d/2dal8.iprior']), \
         patch('bayesed.inspect_cli.PriorManager', return_value=manager) as mgr_cls, \
         patch('bayesed.inspect_cli.PriorEditor', return_value=editor) as ed_cls:
        code = run_inspect(['priors', 'list', '/d'])
    assert code == 0
    mgr_cls.assert_called_once_with(base_directory='/d')
    manager.load_prior_file.assert_called_once_with('2dal8.iprior')
    ed_cls.assert_called_once_with(manager)
    editor.print_priors.assert_called_once()


def test_results_help_does_not_forward_to_binary():
    # --help is a usage request now: exit 0, usage on stdout, and the
    # binary is never invoked.
    with patch('bayesed.launcher.subprocess.Popen') as mock_popen, \
         patch('sys.stdout', new_callable=MagicMock) as out:
        code = run_launcher(['results', '--help'])
    assert code == 0
    mock_popen.assert_not_called()
    written = ''.join(c.args[0] for c in out.write.call_args_list)
    assert 'Usage: bayesed3' in written


def test_priors_init_is_rejected():
    with patch('bayesed.inspect_cli.SEDInference') as inf_cls, \
         patch('sys.stderr', new_callable=MagicMock) as err:
        code = run_inspect(['priors', 'init'])
    assert code == 1
    inf_cls.return_value.priors_init.assert_not_called()
    written = ''.join(c.args[0] for c in err.write.call_args_list)
    assert 'list' in written


def test_priors_list_dir_load_error_exits_one():
    manager = MagicMock()
    manager.load_prior_file.side_effect = ValueError('bad iprior')
    with patch('bayesed.inspect_cli.os.path.isdir', return_value=True), \
         patch('bayesed.inspect_cli.glob.glob', return_value=['/d/2dal8.iprior']), \
         patch('bayesed.inspect_cli.PriorManager', return_value=manager), \
         patch('sys.stderr', new_callable=MagicMock) as err:
        code = run_inspect(['priors', 'list', '/d'])
    assert code == 1
    written = ''.join(c.args[0] for c in err.write.call_args_list)
    assert 'bad iprior' in written


def test_par_values_requires_par_flag():
    with patch('sys.stderr', new_callable=MagicMock) as err:
        code = run_inspect(['results', 'out', 'par_values'])
    assert code == 1
    written = ''.join(c.args[0] for c in err.write.call_args_list)
    assert 'Error: --par is required' in written
    assert 'par_values --par' in written
    assert 'summary' not in written


def test_results_summary_without_outdir_names_the_missing_directory():
    with patch('bayesed.inspect_cli.BayeSEDResults') as ctor, \
         patch('sys.stderr', new_callable=MagicMock) as err:
        code = run_inspect(['results', 'summary', '--json'])
    assert code == 1
    ctor.assert_not_called()
    written = ''.join(c.args[0] for c in err.write.call_args_list)
    assert written.startswith('Error: missing outdir\nUsage:')
    assert 'summary' in written
    assert 'evidence' not in written
    assert 'par_values' not in written


def test_results_two_verbs_without_outdir_keep_the_combine_error():
    with patch('bayesed.inspect_cli.BayeSEDResults') as ctor, \
         patch('sys.stderr', new_callable=MagicMock) as err:
        code = run_inspect(['results', 'summary', 'evidence'])
    assert code == 1
    ctor.assert_not_called()
    written = ''.join(c.args[0] for c in err.write.call_args_list)
    assert "Error: 'evidence' cannot be combined with 'summary'" in written
    assert 'Error: missing outdir' not in written
    assert 'catalogs' in written
    assert 'par_values' in written


def test_par_values_prints_parameter_table():
    fake = MagicMock()
    fake.get_parameter_names.return_value = ['age_median', 'z']
    fake.get_parameter_values.return_value = 'ID age_median\nobj-1 9.5'
    with patch('bayesed.inspect_cli.list_catalog_names', return_value=['gal']), \
         patch('bayesed.inspect_cli.BayeSEDResults', return_value=fake) as ctor, \
         patch('sys.stdout', new_callable=MagicMock) as out:
        code = run_inspect(['results', 'out', 'par_values', '--par', 'age'])
    assert code == 0
    ctor.assert_called_once_with('out')
    fake.get_parameter_values.assert_called_once_with(
        '^(?:age_median)$', param_match_mode='regex'
    )
    written = ''.join(c.args[0] for c in out.write.call_args_list)
    assert 'ID age_median' in written
    assert 'obj-1 9.5' in written


def test_par_values_wide_uses_pprint():
    fake = MagicMock()
    table = Table({'ID': ['obj-1'], 'age_median': [9.5]})
    fake.get_parameter_names.return_value = ['age_median']
    fake.get_parameter_values.return_value = table
    with patch('bayesed.inspect_cli.list_catalog_names', return_value=['gal']), \
         patch('bayesed.inspect_cli.BayeSEDResults', return_value=fake), \
         patch.object(Table, 'pprint') as pprint_mock, \
         patch('sys.stdout', new_callable=MagicMock):
        code = run_inspect([
            'results', 'out', 'par_values', '--par', 'age', '--wide',
        ])
    assert code == 0
    pprint_mock.assert_called_once_with(max_width=-1, max_lines=-1)
    fake.get_parameter_values.assert_called_once_with(
        '^(?:age_median)$', param_match_mode='regex',
    )


def test_par_values_transpose_passed_to_get_parameter_values():
    fake = MagicMock()
    fake.get_parameter_names.return_value = ['age_median']
    fake.get_parameter_values.return_value = 'table'
    with patch('bayesed.inspect_cli.list_catalog_names', return_value=['gal']), \
         patch('bayesed.inspect_cli.BayeSEDResults', return_value=fake):
        code = run_inspect([
            'results', 'out', 'par_values', '--par', 'age', '--transpose',
        ])
    assert code == 0
    fake.get_parameter_values.assert_called_once_with(
        '^(?:age_median)$',
        param_match_mode='regex',
        transpose=True,
    )


def test_par_values_json_ignores_wide_and_transpose():
    fake = MagicMock()
    table = pd.DataFrame({'z': [1.0]})
    fake.get_parameter_names.return_value = ['z']
    fake.get_parameter_values.return_value = table
    with patch('bayesed.inspect_cli.list_catalog_names', return_value=['gal']), \
         patch('bayesed.inspect_cli.BayeSEDResults', return_value=fake), \
         patch('sys.stdout', new_callable=MagicMock) as out:
        code = run_inspect([
            'results', 'out', 'par_values', '--par', 'z',
            '--wide', '--transpose', '--json',
        ])
    assert code == 0
    fake.get_parameter_values.assert_called_once_with(
        '^(?:z)$', param_match_mode='regex',
    )
    assert 'transpose' not in fake.get_parameter_values.call_args.kwargs
    written = ''.join(c.args[0] for c in out.write.call_args_list)
    assert written.strip() == '[{"z": 1.0}]'


def test_par_values_wide_and_transpose_combine():
    fake = MagicMock()
    table = Table({'parameter': ['z_mean'], 'obj-1': [1.0]})
    fake.get_parameter_names.return_value = ['z']
    fake.get_parameter_values.return_value = table
    with patch('bayesed.inspect_cli.list_catalog_names', return_value=['gal']), \
         patch('bayesed.inspect_cli.BayeSEDResults', return_value=fake), \
         patch.object(Table, 'pprint') as pprint_mock, \
         patch('sys.stdout', new_callable=MagicMock):
        code = run_inspect([
            'results', 'out', 'par_values', '--par', 'z',
            '--wide', '--transpose',
        ])
    assert code == 0
    fake.get_parameter_values.assert_called_once_with(
        '^(?:z)$', param_match_mode='regex', transpose=True,
    )
    pprint_mock.assert_called_once_with(max_width=-1, max_lines=-1)


def test_parse_par_values_bool_flags():
    outdir, cat, obj, par, cfg, as_json, wide, transpose = parse_par_values([
        'out', 'par_values', '--par', 'z', '--wide', '--transpose', '--json',
    ])
    assert outdir == 'out'
    assert par == 'z'
    assert cat is None and obj is None and cfg is None
    assert as_json and wide and transpose


def test_par_values_cat_loads_catalog_not_object():
    fake = MagicMock()
    fake.get_parameter_names.return_value = ['age_median']
    fake.get_parameter_values.return_value = 'table'
    with patch('bayesed.inspect_cli.list_catalog_names', return_value=['gal']), \
         patch('bayesed.inspect_cli.BayeSEDResults', return_value=fake) as ctor:
        code = run_inspect([
            'results', 'out', 'par_values', '--cat', 'gal', '--par', 'age',
        ])
    assert code == 0
    ctor.assert_called_once_with('out', catalog_name='gal')
    assert 'object_id' not in ctor.call_args.kwargs


def test_par_values_cfg_passed_to_ctor():
    fake = MagicMock()
    fake.get_parameter_names.return_value = ['age_median']
    fake.get_parameter_values.return_value = 'table'
    with patch('bayesed.inspect_cli.list_catalog_names', return_value=['gal']), \
         patch('bayesed.inspect_cli.list_model_configs', return_value=['cfg0']), \
         patch('bayesed.inspect_cli.BayeSEDResults', return_value=fake) as ctor:
        code = run_inspect([
            'results', 'out', 'par_values', '--cat', 'gal',
            '--cfg', 'cfg0', '--par', 'age',
        ])
    assert code == 0
    ctor.assert_called_once_with(
        'out', catalog_name='gal', model_config='cfg0',
    )
    assert 'object_id' not in ctor.call_args.kwargs
    fake.get_parameter_values.assert_called_once_with(
        '^(?:age_median)$', param_match_mode='regex',
    )


def test_par_values_obj_filters_via_get_parameter_values():
    fake = MagicMock()
    fake.get_parameter_names.return_value = ['age_median']
    fake.list_objects.return_value = [
        'spec-0285-51930-0184_GALAXY_STARFORMING',
        'other',
    ]
    fake.get_parameter_values.return_value = 'table'
    with patch('bayesed.inspect_cli.list_catalog_names', return_value=['gal']), \
         patch('bayesed.inspect_cli.BayeSEDResults', return_value=fake) as ctor:
        code = run_inspect([
            'results', 'out', 'par_values', '--par', 'age',
            '--obj', 'STARFORMING',
        ])
    assert code == 0
    ctor.assert_called_once_with('out')
    assert 'object_id' not in ctor.call_args.kwargs
    fake.get_parameter_values.assert_called_once_with(
        '^(?:age_median)$',
        param_match_mode='regex',
        object_ids=['spec-0285-51930-0184_GALAXY_STARFORMING'],
        object_match_mode='exact',
    )


def test_par_values_exact_beats_contains():
    fake = MagicMock()
    fake.get_parameter_names.return_value = ['z', 'z_err']
    fake.get_parameter_values.return_value = 'table'
    with patch('bayesed.inspect_cli.list_catalog_names', return_value=['gal']), \
         patch('bayesed.inspect_cli.BayeSEDResults', return_value=fake):
        code = run_inspect(['results', 'out', 'par_values', '--par', 'z'])
    assert code == 0
    fake.get_parameter_values.assert_called_once_with(
        '^(?:z)$', param_match_mode='regex',
    )


def test_par_values_uncompilable_pattern_matches_by_contains():
    fake = MagicMock()
    fake.get_parameter_names.return_value = ['log(age/yr)', 'z']
    fake.list_objects.return_value = ['spec-(foo', 'other']
    fake.get_parameter_values.return_value = 'table'
    with patch('bayesed.inspect_cli.list_catalog_names', return_value=['gal']), \
         patch('bayesed.inspect_cli.BayeSEDResults', return_value=fake):
        code = run_inspect([
            'results', 'out', 'par_values', '--par', 'log(age',
            '--obj', 'spec-(',
        ])
    assert code == 0
    fake.get_parameter_values.assert_called_once_with(
        r'^(?:log\(age/yr\))$',
        param_match_mode='regex',
        object_ids=['spec-(foo'],
        object_match_mode='exact',
    )


def test_par_values_rejects_leftover_positionals():
    with patch('sys.stderr', new_callable=MagicMock) as err:
        code = run_inspect([
            'results', 'out', 'par_values', 'gal', '--par', 'age',
        ])
    assert code == 1
    written = ''.join(c.args[0] for c in err.write.call_args_list)
    assert 'unexpected' in written.lower() or 'argument' in written.lower()


def test_par_values_rejects_mix_with_other_commands():
    with patch('sys.stderr', new_callable=MagicMock) as err:
        code = run_inspect([
            'results', 'out', 'summary', 'par_values', '--par', 'age',
        ])
    assert code == 1
    written = ''.join(c.args[0] for c in err.write.call_args_list)
    assert "Error: unexpected argument 'par_values'" in written


def test_par_values_empty_match_exits_one():
    fake = MagicMock()
    fake.get_parameter_names.return_value = ['age_median', 'z']
    with patch('bayesed.inspect_cli.list_catalog_names', return_value=['gal']), \
         patch('bayesed.inspect_cli.BayeSEDResults', return_value=fake), \
         patch('sys.stderr', new_callable=MagicMock) as err:
        code = run_inspect(['results', 'out', 'par_values', '--par', 'nope'])
    assert code == 1
    fake.get_parameter_values.assert_not_called()
    written = ''.join(c.args[0] for c in err.write.call_args_list)
    assert "No parameters found matching 'nope'" in written


def test_par_values_missing_outdir_is_not_unknown_catalog():
    with patch('bayesed.inspect_cli.list_catalog_names') as list_cat, \
         patch('bayesed.inspect_cli.BayeSEDResults') as ctor, \
         patch('sys.stderr', new_callable=MagicMock) as err:
        code = run_inspect([
            'results', 'missing', 'par_values', '--cat', 'gal', '--par', 'age',
        ], check_outdir=True)
    assert code == 1
    list_cat.assert_not_called()
    ctor.assert_not_called()
    written = ''.join(c.args[0] for c in err.write.call_args_list)
    assert 'not a directory: missing' in written
    assert 'No catalog matching' not in written
    assert 'Unknown catalog' not in written


def test_par_values_unknown_catalog_exits_one():
    with patch('bayesed.inspect_cli.list_catalog_names', return_value=['gal']), \
         patch('bayesed.inspect_cli.BayeSEDResults') as ctor, \
         patch('sys.stderr', new_callable=MagicMock) as err:
        code = run_inspect([
            'results', 'out', 'par_values', '--cat', 'qso', '--par', 'age',
        ])
    assert code == 1
    ctor.assert_not_called()
    written = ''.join(c.args[0] for c in err.write.call_args_list)
    assert 'qso' in written
    assert 'gal' in written


def test_params_verb_is_removed():
    with patch('sys.stderr', new_callable=MagicMock) as err:
        code = run_inspect(['results', 'out', 'params'])
    assert code == 1
    written = ''.join(c.args[0] for c in err.write.call_args_list)
    assert 'par_names' in written or 'missing command' in written
    assert 'params' not in RESULTS_VERBS


def test_results_catalogs_lists_names_without_loading_results():
    with patch('bayesed.inspect_cli.list_catalog_names', return_value=['gal', 'qso']) as listed, \
         patch('bayesed.inspect_cli.BayeSEDResults') as ctor, \
         patch('sys.stdout', new_callable=MagicMock) as out:
        code = run_inspect(['results', 'out', 'catalogs'])
    assert code == 0
    listed.assert_called_once_with('out')
    ctor.assert_not_called()
    written = ''.join(c.args[0] for c in out.write.call_args_list)
    assert written.splitlines() == ['gal', 'qso']


def test_results_catalogs_missing_outdir_is_not_empty_list():
    with patch('sys.stderr', new_callable=MagicMock) as err:
        code = run_inspect(['results', 'missing', 'catalogs'], check_outdir=True)
    assert code == 1
    written = ''.join(c.args[0] for c in err.write.call_args_list)
    assert 'not a directory: missing' in written
    assert 'No catalog matching' not in written


def test_results_catalogs_rejects_trailing_name():
    with patch('bayesed.inspect_cli.BayeSEDResults') as ctor, \
         patch('sys.stderr', new_callable=MagicMock) as err:
        code = run_inspect(['results', 'out', 'catalogs', 'gal'])
    assert code == 1
    ctor.assert_not_called()
    written = ''.join(c.args[0] for c in err.write.call_args_list)
    assert 'unexpected' in written.lower()


def test_results_catalogs_rejects_mix_with_summary():
    with patch('bayesed.inspect_cli.list_catalog_names', return_value=['gal', 'qso']), \
         patch('bayesed.inspect_cli.BayeSEDResults') as ctor, \
         patch('sys.stderr', new_callable=MagicMock) as err:
        code = run_inspect(['results', 'out', 'catalogs', 'summary'])
    assert code == 1
    ctor.assert_not_called()
    written = ''.join(c.args[0] for c in err.write.call_args_list)
    assert "Error: unexpected argument 'summary'" in written


def test_results_model_configs_prints_configs_for_catalog():
    with patch('bayesed.inspect_cli.list_catalog_names', return_value=['gal', 'qso']), \
         patch(
             'bayesed.inspect_cli.list_model_configs',
             return_value=['0Stellar+Nebular_2dal8_10_sys_err0'],
         ) as listed, \
         patch('bayesed.inspect_cli.BayeSEDResults') as ctor, \
         patch('sys.stdout', new_callable=MagicMock) as out:
        code = run_inspect(['results', 'out', 'model_configs', '--cat', 'gal'])
    assert code == 0
    ctor.assert_not_called()
    listed.assert_called_once_with('out', 'gal')
    written = ''.join(c.args[0] for c in out.write.call_args_list)
    assert written.splitlines() == ['0Stellar+Nebular_2dal8_10_sys_err0']


def test_results_model_configs_needs_catalog_when_several():
    with patch('bayesed.inspect_cli.list_catalog_names', return_value=['gal', 'qso']), \
         patch('bayesed.inspect_cli.list_model_configs') as listed, \
         patch('sys.stderr', new_callable=MagicMock) as err:
        code = run_inspect(['results', 'out', 'model_configs'])
    assert code == 1
    listed.assert_not_called()
    written = ''.join(c.args[0] for c in err.write.call_args_list)
    assert written == (
        "Error: Multiple catalogs found. Please specify catalog "
        "with --cat. Available catalogs: ['gal', 'qso']\n"
    )


def test_results_model_configs_uses_sole_catalog():
    with patch('bayesed.inspect_cli.list_catalog_names', return_value=['gal']), \
         patch(
             'bayesed.inspect_cli.list_model_configs',
             return_value=['cfg0'],
         ) as listed, \
         patch('bayesed.inspect_cli.BayeSEDResults') as ctor, \
         patch('sys.stdout', new_callable=MagicMock) as out:
        code = run_inspect(['results', 'out', 'model_configs'])
    assert code == 0
    ctor.assert_not_called()
    listed.assert_called_once_with('out', 'gal')
    written = ''.join(c.args[0] for c in out.write.call_args_list)
    assert written.splitlines() == ['cfg0']


def test_results_model_configs_errors_when_no_catalogs():
    with patch('bayesed.inspect_cli.list_catalog_names', return_value=[]), \
         patch('bayesed.inspect_cli.list_model_configs') as listed, \
         patch('sys.stderr', new_callable=MagicMock) as err:
        code = run_inspect(['results', 'out', 'model_configs'])
    assert code == 1
    listed.assert_not_called()
    written = ''.join(c.args[0] for c in err.write.call_args_list)
    assert 'No catalogs' in written


def test_results_model_configs_unknown_catalog_exits_one():
    with patch('bayesed.inspect_cli.list_catalog_names', return_value=['gal']), \
         patch('bayesed.inspect_cli.list_model_configs') as listed, \
         patch('sys.stderr', new_callable=MagicMock) as err:
        code = run_inspect([
            'results', 'out', 'model_configs', '--cat', 'qso',
        ])
    assert code == 1
    listed.assert_not_called()
    written = ''.join(c.args[0] for c in err.write.call_args_list)
    assert 'qso' in written
    assert 'No catalog matching' in written


def test_results_model_configs_rejects_positional_catalog():
    with patch('bayesed.inspect_cli.BayeSEDResults') as ctor, \
         patch('sys.stderr', new_callable=MagicMock) as err:
        code = run_inspect(['results', 'out', 'model_configs', 'gal'])
    assert code == 1
    ctor.assert_not_called()
    written = ''.join(c.args[0] for c in err.write.call_args_list)
    assert "Error: unexpected argument 'gal'" in written


def test_priors_without_list_prints_usage():
    with patch('sys.stderr', new_callable=MagicMock) as err:
        code = run_inspect(['priors'])
    assert code == 1
    written = ''.join(c.args[0] for c in err.write.call_args_list)
    assert 'list' in written


def _fake_plot_results(object_id=None, objects=None, free=None, derived=None):
    fake = MagicMock()
    fake.object_id = object_id
    fake.list_objects.return_value = objects if objects is not None else ['only']
    fake.get_free_parameters.return_value = (
        free if free is not None else ['log(Mstar)[0,1]', 'z']
    )
    fake.get_derived_parameters.return_value = (
        derived if derived is not None else []
    )
    return fake


def test_plot_bare_posterior_exits_one():
    with patch('bayesed.inspect_cli.BayeSEDResults') as ctor, \
         patch('sys.stderr', new_callable=MagicMock) as err:
        code = run_inspect(['plot', 'out', 'posterior'])
    assert code == 1
    ctor.assert_not_called()
    written = ''.join(c.args[0] for c in err.write.call_args_list)
    assert 'Error: specify free, derived, or --par' in written


def test_plot_posterior_free_does_not_treat_free_as_catalog():
    fake = _fake_plot_results()
    with patch('bayesed.inspect_cli.list_catalog_names', return_value=['gal']), \
         patch('bayesed.inspect_cli.BayeSEDResults', return_value=fake) as ctor:
        code = run_inspect(['plot', 'out', 'posterior', 'free'])
    assert code == 0
    ctor.assert_called_once_with('out')
    fake.plot_posterior_free.assert_called_once()
    assert fake.plot_posterior_free.call_args.kwargs.get('show') is True
    fake.plot_posterior.assert_not_called()
    fake.plot_posterior_derived.assert_not_called()


def test_plot_posterior_without_save_does_not_write_png():
    fake = _fake_plot_results()
    with patch('bayesed.inspect_cli.list_catalog_names', return_value=['gal']), \
         patch('bayesed.inspect_cli.BayeSEDResults', return_value=fake):
        code = run_inspect(['plot', 'out', 'posterior', 'free'])
    assert code == 0
    kwargs = fake.plot_posterior_free.call_args.kwargs
    assert kwargs.get('save') is False
    assert 'output_file' not in kwargs


def test_plot_posterior_save_writes_sidecar():
    fake = _fake_plot_results()
    with patch('bayesed.inspect_cli.list_catalog_names', return_value=['gal']), \
         patch('bayesed.inspect_cli.BayeSEDResults', return_value=fake):
        code = run_inspect(['plot', 'out', 'posterior', 'free', '--save'])
    assert code == 0
    kwargs = fake.plot_posterior_free.call_args.kwargs
    assert kwargs.get('show') is True
    assert kwargs.get('save') is True
    assert 'output_file' not in kwargs


def test_plot_posterior_derived_uses_max_params_ten():
    fake = _fake_plot_results()
    with patch('bayesed.inspect_cli.list_catalog_names', return_value=['gal']), \
         patch('bayesed.inspect_cli.BayeSEDResults', return_value=fake):
        code = run_inspect(['plot', 'out', 'posterior', 'derived'])
    assert code == 0
    fake.plot_posterior_derived.assert_called_once()
    assert fake.plot_posterior_derived.call_args.kwargs.get('show') is True
    assert fake.plot_posterior_derived.call_args.kwargs.get('save') is False
    assert 'output_file' not in fake.plot_posterior_derived.call_args.kwargs
    assert fake.plot_posterior_derived.call_args.kwargs.get('max_params') == 10
    fake.plot_posterior_free.assert_not_called()
    fake.plot_posterior.assert_not_called()


def test_plot_posterior_derived_max_flag():
    fake = _fake_plot_results()
    with patch('bayesed.inspect_cli.list_catalog_names', return_value=['gal']), \
         patch('bayesed.inspect_cli.BayeSEDResults', return_value=fake):
        code = run_inspect(['plot', 'out', 'posterior', 'derived', '--max', '3'])
    assert code == 0
    assert fake.plot_posterior_derived.call_args.kwargs.get('max_params') == 3


def test_plot_posterior_par_calls_plot_posterior():
    fake = _fake_plot_results()
    with patch('bayesed.inspect_cli.list_catalog_names', return_value=['gal']), \
         patch('bayesed.inspect_cli.BayeSEDResults', return_value=fake):
        code = run_inspect([
            'plot', 'out', 'posterior',
            '--par', 'log(Mstar)[0,1]', '--par', 'z',
        ])
    assert code == 0
    fake.set_parameter_labels.assert_not_called()
    fake.plot_posterior.assert_called_once()
    assert fake.plot_posterior.call_args.kwargs.get('params') == [
        'log(Mstar)[0,1]', 'z',
    ]
    assert fake.plot_posterior.call_args.kwargs.get('save') is False
    assert 'output_file' not in fake.plot_posterior.call_args.kwargs
    fake.plot_posterior_free.assert_not_called()
    fake.plot_posterior_derived.assert_not_called()


def test_plot_posterior_par_lab_pairs_set_labels():
    fake = _fake_plot_results()
    with patch('bayesed.inspect_cli.list_catalog_names', return_value=['gal']), \
         patch('bayesed.inspect_cli.BayeSEDResults', return_value=fake):
        code = run_inspect([
            'plot', 'out', 'posterior',
            '--par', 'log(Mstar)[0,1]', '--lab', 'a, b',
            '--par', 'z', '--lab', r'$\log z$',
        ])
    assert code == 0
    fake.set_parameter_labels.assert_called_once_with({
        'log(Mstar)[0,1]': 'a, b',
        'z': r'$\log z$',
    })
    assert fake.plot_posterior.call_args.kwargs.get('params') == [
        'log(Mstar)[0,1]', 'z',
    ]
    names = [name for name, _, _ in fake.method_calls]
    assert names.index('set_parameter_labels') < names.index('plot_posterior')


def test_plot_posterior_par_expands_contains_hits():
    fake = _fake_plot_results(
        free=['z'],
        derived=['log(age/yr)', 'log(ageMW/yr)'],
    )
    with patch('bayesed.inspect_cli.list_catalog_names', return_value=['gal']), \
         patch('bayesed.inspect_cli.BayeSEDResults', return_value=fake):
        code = run_inspect(['plot', 'out', 'posterior', '--par', 'age'])
    assert code == 0
    fake.plot_posterior.assert_called_once()
    assert fake.plot_posterior.call_args.kwargs.get('params') == [
        'log(age/yr)', 'log(ageMW/yr)',
    ]
    fake.set_parameter_labels.assert_not_called()


def test_plot_posterior_par_lab_rejected_when_not_unique():
    fake = _fake_plot_results(
        free=['z'],
        derived=['log(age/yr)', 'log(ageMW/yr)'],
    )
    with patch('bayesed.inspect_cli.list_catalog_names', return_value=['gal']), \
         patch('bayesed.inspect_cli.BayeSEDResults', return_value=fake), \
         patch('sys.stderr', new_callable=MagicMock) as err:
        code = run_inspect([
            'plot', 'out', 'posterior', '--par', 'age', '--lab', 'age',
        ])
    assert code == 1
    fake.plot_posterior.assert_not_called()
    written = ''.join(c.args[0] for c in err.write.call_args_list)
    assert 'Error: --lab requires a unique --par match' in written


def test_plot_posterior_par_no_match_exits_one():
    fake = _fake_plot_results()
    with patch('bayesed.inspect_cli.list_catalog_names', return_value=['gal']), \
         patch('bayesed.inspect_cli.BayeSEDResults', return_value=fake), \
         patch('sys.stderr', new_callable=MagicMock) as err:
        code = run_inspect(['plot', 'out', 'posterior', '--par', 'nope'])
    assert code == 1
    fake.plot_posterior.assert_not_called()
    written = ''.join(c.args[0] for c in err.write.call_args_list)
    assert "No parameters found matching 'nope'" in written


def test_plot_posterior_par_lab_optional_per_par():
    fake = _fake_plot_results(
        object_id=None,
        objects=['obj-1'],
        free=['Av_2[0,1]', 'z'],
        derived=['log(age/yr)', 'log(Z/Zsun)'],
    )
    with patch('bayesed.inspect_cli.list_catalog_names', return_value=['gal']), \
         patch('bayesed.inspect_cli.BayeSEDResults', return_value=fake):
        code = run_inspect([
            'plot', 'out', 'posterior',
            '--par', 'Av', '--lab', 'Av',
            '--par', 'age/yr',
            '--par', 'Z/Zsun',
            '--cat', 'gal', '--obj', 'obj-1',
        ])
    assert code == 0
    fake.set_parameter_labels.assert_called_once_with({'Av_2[0,1]': 'Av'})
    assert fake.plot_posterior.call_args.kwargs.get('params') == [
        'Av_2[0,1]', 'log(age/yr)', 'log(Z/Zsun)',
    ]


def test_plot_posterior_par_lab_with_unlabeled_multi_expand():
    fake = _fake_plot_results(
        free=['Av_2[0,1]', 'z'],
        derived=['log(age/yr)', 'log(ageMW/yr)'],
    )
    with patch('bayesed.inspect_cli.list_catalog_names', return_value=['gal']), \
         patch('bayesed.inspect_cli.BayeSEDResults', return_value=fake):
        code = run_inspect([
            'plot', 'out', 'posterior',
            '--par', 'Av', '--lab', 'Av',
            '--par', 'age',
        ])
    assert code == 0
    fake.set_parameter_labels.assert_called_once_with({'Av_2[0,1]': 'Av'})
    assert fake.plot_posterior.call_args.kwargs.get('params') == [
        'Av_2[0,1]', 'log(age/yr)', 'log(ageMW/yr)',
    ]


def test_plot_posterior_grouped_par_then_lab_rejected():
    with patch('bayesed.inspect_cli.BayeSEDResults') as ctor, \
         patch('sys.stderr', new_callable=MagicMock) as err:
        code = run_inspect([
            'plot', 'out', 'posterior',
            '--par', 'a', '--par', 'b', '--lab', 'x', '--lab', 'y',
        ])
    assert code == 1
    ctor.assert_not_called()
    written = ''.join(c.args[0] for c in err.write.call_args_list)
    assert 'Error: --lab must follow --par' in written


def test_plot_posterior_lab_without_par_rejected():
    with patch('bayesed.inspect_cli.BayeSEDResults') as ctor, \
         patch('sys.stderr', new_callable=MagicMock) as err:
        code = run_inspect(['plot', 'out', 'posterior', '--lab', 'x'])
    assert code == 1
    ctor.assert_not_called()
    written = ''.join(c.args[0] for c in err.write.call_args_list)
    assert 'Error: --lab must follow --par' in written


def test_plot_posterior_free_with_par_rejected():
    with patch('bayesed.inspect_cli.BayeSEDResults') as ctor, \
         patch('sys.stderr', new_callable=MagicMock) as err:
        code = run_inspect([
            'plot', 'out', 'posterior', 'free', '--par', 'z',
        ])
    assert code == 1
    ctor.assert_not_called()
    written = ''.join(c.args[0] for c in err.write.call_args_list)
    assert 'Error: free cannot be combined with --par' in written


def test_plot_posterior_free_and_derived_rejected():
    with patch('bayesed.inspect_cli.BayeSEDResults') as ctor, \
         patch('sys.stderr', new_callable=MagicMock) as err:
        code = run_inspect(['plot', 'out', 'posterior', 'free', 'derived'])
    assert code == 1
    ctor.assert_not_called()
    written = ''.join(c.args[0] for c in err.write.call_args_list)
    assert 'Error: free and derived cannot be combined' in written


def test_plot_posterior_max_only_with_derived():
    with patch('bayesed.inspect_cli.BayeSEDResults') as ctor, \
         patch('sys.stderr', new_callable=MagicMock) as err:
        code = run_inspect(['plot', 'out', 'posterior', 'free', '--max', '3'])
    assert code == 1
    ctor.assert_not_called()
    written = ''.join(c.args[0] for c in err.write.call_args_list)
    assert 'Error: --max is only valid with derived' in written


def test_plot_posterior_rejects_leftover_positionals():
    with patch('bayesed.inspect_cli.BayeSEDResults') as ctor, \
         patch('sys.stderr', new_callable=MagicMock) as err:
        code = run_inspect(['plot', 'out', 'posterior', 'free', 'gal'])
    assert code == 1
    ctor.assert_not_called()
    written = ''.join(c.args[0] for c in err.write.call_args_list)
    assert "Error: unexpected argument 'gal'" in written


def test_plot_posterior_cat_and_obj_flags():
    fake = _fake_plot_results(objects=['STARFORMING'])
    with patch('bayesed.inspect_cli.list_catalog_names', return_value=['gal']), \
         patch('bayesed.inspect_cli.BayeSEDResults', return_value=fake) as ctor:
        code = run_inspect([
            'plot', 'out', 'posterior', 'free',
            '--cat', 'gal', '--obj', 'STARFORMING',
        ])
    assert code == 0
    ctor.assert_called_once_with('out', catalog_name='gal')
    assert fake.plot_posterior_free.call_args.kwargs.get('object_id') == (
        'STARFORMING'
    )


def test_plot_posterior_cfg_passed_to_results():
    fake = _fake_plot_results(objects=['STARFORMING'])
    with patch('bayesed.inspect_cli.list_catalog_names', return_value=['gal']), \
         patch('bayesed.inspect_cli.list_model_configs', return_value=['cfg0']), \
         patch('bayesed.inspect_cli.BayeSEDResults', return_value=fake) as ctor:
        code = run_inspect([
            'plot', 'out', 'posterior', 'free',
            '--cat', 'gal', '--cfg', 'cfg0', '--obj', 'STARFORMING',
        ])
    assert code == 0
    ctor.assert_called_once_with(
        'out',
        catalog_name='gal',
        model_config='cfg0',
    )


def test_plot_multiple_configs_without_cfg_exits_one():
    with patch('bayesed.inspect_cli.list_catalog_names', return_value=['gal']), \
         patch(
             'bayesed.inspect_cli.BayeSEDResults',
             side_effect=ValueError(
                 "Multiple model configurations found for catalog 'gal'. "
                 "Please specify model_config parameter. "
                 "Available configs: ['cfg0', 'cfg1']"
             ),
         ), \
         patch('sys.stderr', new_callable=MagicMock) as err:
        code = run_inspect(['plot', 'out', 'bestfit', '--cat', 'gal'])
    assert code == 1
    written = ''.join(c.args[0] for c in err.write.call_args_list)
    assert written == (
        "Error: Multiple model configurations found. "
        "Please specify config with --cfg. "
        "Available configs: ['cfg0', 'cfg1']\n"
    )


def test_plot_posterior_multiple_objects_without_obj_errors():
    fake = _fake_plot_results(objects=['obj-a', 'obj-b'])
    with patch('bayesed.inspect_cli.list_catalog_names', return_value=['gal']), \
         patch('bayesed.inspect_cli.BayeSEDResults', return_value=fake), \
         patch('sys.stderr', new_callable=MagicMock) as err:
        code = run_inspect(['plot', 'out', 'posterior', 'free'])
    assert code == 1
    written = ''.join(c.args[0] for c in err.write.call_args_list)
    assert written == (
        "Error: Multiple objects found. Please specify object "
        "with --obj. Available objects: ['obj-a', 'obj-b']\n"
    )
    fake.plot_posterior_free.assert_not_called()


def test_plot_posterior_unknown_catalog_exits_one():
    with patch('bayesed.inspect_cli.list_catalog_names', return_value=['gal']), \
         patch('bayesed.inspect_cli.BayeSEDResults') as ctor, \
         patch('sys.stderr', new_callable=MagicMock) as err:
        code = run_inspect([
            'plot', 'out', 'posterior', 'free', '--cat', 'qso',
        ])
    assert code == 1
    ctor.assert_not_called()
    written = ''.join(c.args[0] for c in err.write.call_args_list)
    assert 'qso' in written
    assert 'No catalog matching' in written


def test_cat_loose_match_unique_contains():
    fake = MagicMock()
    fake.list_objects.return_value = ['obj-1']
    with patch(
             'bayesed.inspect_cli.list_catalog_names',
             return_value=['gal', 'qso'],
         ), \
         patch('bayesed.inspect_cli.BayeSEDResults', return_value=fake) as ctor:
        code = run_inspect(['results', 'out', 'objects', '--cat', 'ga'])
    assert code == 0
    ctor.assert_called_once_with('out', catalog_name='gal')


def test_cat_loose_match_ambiguous_errors():
    with patch(
             'bayesed.inspect_cli.list_catalog_names',
             return_value=['gal', 'mock_gal'],
         ), \
         patch('bayesed.inspect_cli.BayeSEDResults') as ctor, \
         patch('sys.stderr', new_callable=MagicMock) as err:
        code = run_inspect(['results', 'out', 'objects', '--cat', 'g'])
    assert code == 1
    ctor.assert_not_called()
    written = ''.join(c.args[0] for c in err.write.call_args_list)
    assert 'Multiple catalogs found' in written
    assert '--cat' in written


def test_plot_obj_multi_match_plots_each():
    fake = MagicMock()
    fake.list_objects.return_value = ['spec-0000', 'spec-0001', 'other']
    with patch('bayesed.inspect_cli.list_catalog_names', return_value=['gal']), \
         patch('bayesed.inspect_cli.BayeSEDResults', return_value=fake):
        code = run_inspect([
            'plot', 'out', 'bestfit', '--cat', 'gal', '--obj', 'spec-000',
        ])
    assert code == 0
    assert fake.plot_bestfit.call_count == 2
    ids = [
        c.kwargs.get('object_id') for c in fake.plot_bestfit.call_args_list
    ]
    assert ids == ['spec-0000', 'spec-0001']


def test_plot_out_is_rejected():
    fake = MagicMock()
    fake.list_objects.return_value = ['spec-a']
    with patch('bayesed.inspect_cli.list_catalog_names', return_value=['gal']), \
         patch('bayesed.inspect_cli.BayeSEDResults', return_value=fake), \
         patch('sys.stderr', new_callable=MagicMock) as err:
        code = run_inspect([
            'plot', 'out', 'bestfit', '--cat', 'gal', '--out', 'out.png',
        ])
    assert code == 1
    written = ''.join(c.args[0] for c in err.write.call_args_list)
    assert 'unknown option --out' in written
    fake.plot_bestfit.assert_not_called()


def test_results_obj_multi_filter_objects():
    fake = MagicMock()
    fake.list_objects.return_value = ['spec-a', 'spec-b', 'other']
    with patch('bayesed.inspect_cli.list_catalog_names', return_value=['gal']), \
         patch('bayesed.inspect_cli.BayeSEDResults', return_value=fake), \
         patch('sys.stdout', new_callable=MagicMock) as out:
        code = run_inspect([
            'results', 'out', 'objects', '--obj', 'spec-',
        ])
    assert code == 0
    written = ''.join(c.args[0] for c in out.write.call_args_list)
    assert written.splitlines() == ['spec-a', 'spec-b']


def test_results_run_info_prints_only_requested_attribute():
    """Each provenance verb shows its own attribute and nothing else."""
    fake = MagicMock()
    fake.run_provenance.return_value = {
        'run_arguments': 'bayesed_mn_1 -i galaxy.txt',
        'version': 'BayeSED3 (test)',
    }
    for verb, shown, hidden in (
        ('run_arguments', 'bayesed_mn_1 -i galaxy.txt', 'BayeSED3 (test)'),
        ('run_version', 'BayeSED3 (test)', 'bayesed_mn_1 -i galaxy.txt'),
    ):
        fake.run_provenance.reset_mock()
        with patch('bayesed.inspect_cli.list_catalog_names',
                   return_value=['gal']), \
             patch('bayesed.inspect_cli.BayeSEDResults',
                   return_value=fake), \
             patch('sys.stdout', new_callable=MagicMock) as out:
            code = run_inspect(['results', 'out', verb])
        assert code == 0
        written = ''.join(c.args[0] for c in out.write.call_args_list)
        assert shown in written
        assert hidden not in written


def test_results_run_info_rejects_unknown_flag():
    with patch('bayesed.inspect_cli.list_catalog_names') as list_cat, \
         patch('sys.stderr', new_callable=MagicMock) as err:
        code = run_inspect(['results', 'out', 'run_arguments', '--json'])
    assert code == 1
    list_cat.assert_not_called()
    written = ''.join(c.args[0] for c in err.write.call_args_list)
    assert 'unknown option --json' in written
    assert 'run_arguments [--cat PATTERN] [--cfg PATTERN]' in written


def test_results_run_info_reports_missing_attribute():
    fake = MagicMock()
    fake.run_provenance.side_effect = KeyError(
        "gal.hdf5: 'parameters' dataset has no 'run_arguments' attribute")
    with patch('bayesed.inspect_cli.list_catalog_names', return_value=['gal']), \
         patch('bayesed.inspect_cli.BayeSEDResults', return_value=fake), \
         patch('sys.stderr', new_callable=MagicMock) as err:
        code = run_inspect(['results', 'out', 'run_arguments'])
    assert code == 1
    written = ''.join(c.args[0] for c in err.write.call_args_list)
    assert "no 'run_arguments' attribute" in written


def test_results_run_info_reports_missing_hdf5():
    """No HDF5 in the outdir: clean load error, exit 1."""
    fake = MagicMock()
    fake.run_provenance.side_effect = FileNotFoundError(
        'no HDF5 file found in out')
    with patch('bayesed.inspect_cli.list_catalog_names',
               return_value=['gal']), \
         patch('bayesed.inspect_cli.BayeSEDResults', return_value=fake), \
         patch('sys.stderr', new_callable=MagicMock) as err:
        code = run_inspect(['results', 'out', 'run_arguments'])
    assert code == 1
    written = ''.join(c.args[0] for c in err.write.call_args_list)
    assert 'no HDF5 file found' in written


def test_results_run_info_reports_missing_parameters_dataset():
    """An HDF5 without a 'parameters' dataset: clean load error, exit 1."""
    fake = MagicMock()
    fake.run_provenance.side_effect = KeyError(
        "gal_test.hdf5: no 'parameters' dataset")
    with patch('bayesed.inspect_cli.list_catalog_names',
               return_value=['gal']), \
         patch('bayesed.inspect_cli.BayeSEDResults', return_value=fake), \
         patch('sys.stderr', new_callable=MagicMock) as err:
        code = run_inspect(['results', 'out', 'run_version'])
    assert code == 1
    written = ''.join(c.args[0] for c in err.write.call_args_list)
    assert "no 'parameters' dataset" in written


def test_verb_level_help_prints_narrow_usage():
    """`plot <outdir> bestfit --help` shows the bestfit usage, exit 0."""
    with patch('sys.stdout', new_callable=MagicMock) as out:
        code = run_inspect(['plot', 'out', 'bestfit', '--help'])
    assert code == 0
    written = ''.join(c.args[0] for c in out.write.call_args_list)
    assert 'Usage: bayesed3 plot <outdir> bestfit' in written
    assert '--show-residuals' in written
    assert 'posterior' not in written.splitlines()[0]


def test_results_verb_level_help_prints_results_usage():
    with patch('sys.stdout', new_callable=MagicMock) as out:
        code = run_inspect(['results', 'out', 'run_version', '--help'])
    assert code == 0
    written = ''.join(c.args[0] for c in out.write.call_args_list)
    assert 'run_version [--cat PATTERN] [--cfg PATTERN]' in written


def test_kind_level_help_prints_full_usage():
    with patch('sys.stdout', new_callable=MagicMock) as out:
        code = run_inspect(['results', '--help'])
    assert code == 0
    written = ''.join(c.args[0] for c in out.write.call_args_list)
    assert 'summary [--cat PATTERN]' in written
    assert 'fit_quality' in written


def test_priors_help_prints_priors_line():
    with patch('sys.stdout', new_callable=MagicMock) as out:
        code = run_inspect(['priors', '--help'])
    assert code == 0
    written = ''.join(c.args[0] for c in out.write.call_args_list)
    assert 'priors list' in written




def test_help_with_other_kind_verb_falls_back_to_full_usage():
    """A verb from the other kind does not narrow (and does not crash):
    `results out bestfit --help` shows the full results usage."""
    with patch('sys.stdout', new_callable=MagicMock) as out:
        code = run_inspect(['results', 'out', 'bestfit', '--help'])
    assert code == 0
    written = ''.join(c.args[0] for c in out.write.call_args_list)
    assert 'run_arguments' in written  # full results usage, not a crash

    with patch('sys.stdout', new_callable=MagicMock) as out:
        code = run_inspect(['plot', 'out', 'summary', '--help'])
    assert code == 0
    written = ''.join(c.args[0] for c in out.write.call_args_list)
    assert 'posterior free' in written  # full plot usage, not posterior-narrow


if __name__ == '__main__':
    test_results_without_verbs_lists_commands_and_skips_binary()
    test_results_run_info_prints_only_requested_attribute()
    test_verb_level_help_prints_narrow_usage()
    test_help_with_other_kind_verb_falls_back_to_full_usage()
    test_results_verb_level_help_prints_results_usage()
    test_kind_level_help_prints_full_usage()
    test_priors_help_prints_priors_line()
    test_results_run_info_rejects_unknown_flag()
    test_results_run_info_reports_missing_attribute()
    test_results_run_info_reports_missing_hdf5()
    test_results_run_info_reports_missing_parameters_dataset()
    test_plot_without_verbs_lists_commands()
    test_plot_without_outdir_prints_usage_only()
    test_plot_leading_flag_prints_usage_only()
    test_plot_posterior_without_outdir_names_the_missing_directory()
    test_plot_bestfit_without_outdir_names_the_missing_directory()
    test_plot_posterior_free_without_outdir_does_not_ask_for_a_mode()
    test_plot_verbs_without_outdir_cannot_be_combined()
    test_plot_bestfit_flag_error_omits_posterior_usage()
    test_plot_junk_before_verbs_is_missing_command()
    test_plot_posterior_without_mode_omits_bestfit_usage()
    test_top_level_help_is_overview_not_binary()
    test_results_outdir_only_prints_usage()
    test_results_outdir_json_without_command_prints_usage()
    test_results_objects_json()
    test_results_evidence_json()
    test_results_evidence_json_decodes_bytes_id()
    test_results_evidence_json_single_object_is_still_array()
    test_results_summary_json()
    test_results_objects_json_decodes_bytes_ids()
    test_results_par_values_json()
    test_results_par_values_json_from_astropy_table()
    test_results_catalogs_json()
    test_results_model_configs_json()
    test_missing_cfg_is_one_line_with_cfg_flag()
    test_plot_missing_cfg_is_one_line_with_cfg_flag()
    test_unknown_catalog_lists_available_in_one_line()
    test_missing_catalog_is_one_line_with_cat_flag()
    test_library_many_configs_truncates_available_list()
    test_many_catalogs_truncates_available_list()
    test_results_summary_uses_bayesed_results()
    test_results_summary_ignores_obj()
    test_results_cat_flag_loads_catalog()
    test_results_obj_flag_filters_without_ctor_object_id()
    test_results_rejects_leftover_positionals()
    test_results_cfg_passed_to_ctor()
    test_results_flags_must_follow_verb()
    test_results_rejects_stacked_query_verbs()
    test_results_non_verb_before_query_is_missing_command()
    test_results_non_verb_before_catalogs_is_missing_command()
    test_parse_single_command_non_verb_prefix_is_missing_command()
    test_results_evidence_calls_existing_method()
    test_results_par_names_calls_existing_methods()
    test_plot_bestfit_calls_plot_bestfit()
    test_plot_bestfit_without_save_does_not_write_png()
    test_plot_bestfit_flux_passes_flux_unit()
    test_plot_bestfit_flux_rejects_unknown_unit()
    test_plot_bestfit_log_axis_passes_use_log_scale()
    test_plot_bestfit_linear_passes_use_log_scale_false()
    test_plot_bestfit_log_and_linear_rejected()
    test_plot_bestfit_log_rejects_unknown_axis()
    test_plot_bestfit_switch_flags_pass_false()
    test_plot_bestfit_omits_appearance_kwargs()
    test_plot_posterior_rejects_bestfit_appearance_flag()
    test_plot_bestfit_observed_frame_passes_use_rest_frame_false()
    test_plot_bestfit_save_writes_sidecar()
    test_plot_posterior_cannot_mix_with_bestfit()
    test_plot_posterior_then_bestfit_cannot_mix()
    test_plot_multiple_objects_without_obj_errors()
    test_plot_many_objects_without_obj_lists_head_and_tail()
    test_plot_single_object_without_obj_uses_it_silently()
    test_plot_with_object_does_not_list()
    test_plot_bestfit_passes_catalog_and_object()
    test_plot_bestfit_cfg_passed_to_results()
    test_plot_bestfit_cfg_requires_value()
    test_plot_par_missing_value_before_another_flag()
    test_plot_cat_missing_value_before_another_flag()
    test_plot_lab_missing_value_before_another_flag()
    test_results_obj_missing_value_before_another_flag()
    test_plot_bestfit_rejects_leftover_positionals()
    test_priors_list_without_dir_lists_types()
    test_priors_list_dir_uses_existing_loader_and_printer()
    test_results_help_does_not_forward_to_binary()
    test_priors_init_is_rejected()
    test_priors_list_dir_load_error_exits_one()
    test_par_values_requires_par_flag()
    test_results_summary_without_outdir_names_the_missing_directory()
    test_results_two_verbs_without_outdir_keep_the_combine_error()
    test_par_values_prints_parameter_table()
    test_par_values_wide_uses_pprint()
    test_par_values_transpose_passed_to_get_parameter_values()
    test_par_values_json_ignores_wide_and_transpose()
    test_par_values_wide_and_transpose_combine()
    test_parse_par_values_bool_flags()
    test_par_values_cat_loads_catalog_not_object()
    test_par_values_cfg_passed_to_ctor()
    test_par_values_obj_filters_via_get_parameter_values()
    test_par_values_exact_beats_contains()
    test_par_values_uncompilable_pattern_matches_by_contains()
    test_par_values_rejects_leftover_positionals()
    test_par_values_rejects_mix_with_other_commands()
    test_par_values_empty_match_exits_one()
    test_par_values_missing_outdir_is_not_unknown_catalog()
    test_par_values_unknown_catalog_exits_one()
    test_params_verb_is_removed()
    test_results_catalogs_lists_names_without_loading_results()
    test_results_catalogs_missing_outdir_is_not_empty_list()
    test_results_catalogs_rejects_trailing_name()
    test_results_catalogs_rejects_mix_with_summary()
    test_results_model_configs_prints_configs_for_catalog()
    test_results_model_configs_needs_catalog_when_several()
    test_results_model_configs_uses_sole_catalog()
    test_results_model_configs_errors_when_no_catalogs()
    test_results_model_configs_unknown_catalog_exits_one()
    test_results_model_configs_rejects_positional_catalog()
    test_priors_without_list_prints_usage()
    test_plot_bare_posterior_exits_one()
    test_plot_posterior_free_does_not_treat_free_as_catalog()
    test_plot_posterior_without_save_does_not_write_png()
    test_plot_posterior_save_writes_sidecar()
    test_plot_posterior_derived_uses_max_params_ten()
    test_plot_posterior_derived_max_flag()
    test_plot_posterior_par_calls_plot_posterior()
    test_plot_posterior_par_lab_pairs_set_labels()
    test_plot_posterior_par_expands_contains_hits()
    test_plot_posterior_par_lab_rejected_when_not_unique()
    test_plot_posterior_par_no_match_exits_one()
    test_plot_posterior_par_lab_optional_per_par()
    test_plot_posterior_par_lab_with_unlabeled_multi_expand()
    test_plot_posterior_grouped_par_then_lab_rejected()
    test_plot_posterior_lab_without_par_rejected()
    test_plot_posterior_free_with_par_rejected()
    test_plot_posterior_free_and_derived_rejected()
    test_plot_posterior_max_only_with_derived()
    test_plot_posterior_rejects_leftover_positionals()
    test_plot_posterior_cat_and_obj_flags()
    test_plot_posterior_cfg_passed_to_results()
    test_plot_multiple_configs_without_cfg_exits_one()
    test_plot_posterior_multiple_objects_without_obj_errors()
    test_plot_posterior_unknown_catalog_exits_one()
    test_plot_out_is_rejected()
    print('test_inspect_cli: OK')
