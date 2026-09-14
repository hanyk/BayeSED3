#!/usr/bin/env python3
"""Unit tests for the Launcher inspect path (ADR-0009; no binary, no HDF5)."""

from unittest.mock import MagicMock, patch

from bayesed.inspect_cli import (
    PLOT_VERBS,
    RESULTS_VERBS,
    format_object_listing,
    run_inspect,
)
from bayesed.launcher import run_launcher


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
    for verb in PLOT_VERBS:
        assert verb in written


def test_help_still_forwards_to_binary():
    fake_proc = MagicMock()
    fake_proc.wait.return_value = 0
    with patch('bayesed.launcher.resolve_system_info', return_value=('darwin', 'arm64')), \
         patch('bayesed.launcher.resolve_mpirun', return_value='/mpi/mpirun'), \
         patch('bayesed.launcher.resolve_num_processes', return_value=4), \
         patch('bayesed.launcher.resolve_executable', return_value='/bin/bayesed_mn_1'), \
         patch('bayesed.launcher.resolve_binary_cwd', return_value='/data/root'), \
         patch('bayesed.launcher.absolutize_cli_path_args', side_effect=lambda a, base_dir=None: a), \
         patch('bayesed.launcher.build_mpi_command', return_value=['cmd']), \
         patch('bayesed.launcher.subprocess.Popen', return_value=fake_proc) as mock_popen:
        code = run_launcher(['--help'])
    assert code == 0
    mock_popen.assert_called_once()


def test_results_summary_uses_bayesed_results():
    fake = MagicMock()
    with patch('bayesed.inspect_cli.list_catalog_names', return_value=['gal']), \
         patch('bayesed.inspect_cli.BayeSEDResults', return_value=fake) as ctor:
        code = run_inspect(['results', 'out', 'summary'])
    assert code == 0
    ctor.assert_called_once_with('out')
    fake.print_summary.assert_called_once()


def test_results_cat_flag_loads_catalog():
    fake = MagicMock()
    with patch('bayesed.inspect_cli.list_catalog_names', return_value=['gal']), \
         patch('bayesed.inspect_cli.BayeSEDResults', return_value=fake) as ctor:
        code = run_inspect(['results', 'out', 'objects', '--cat', 'gal'])
    assert code == 0
    ctor.assert_called_once_with('out', catalog_name='gal')
    fake.list_objects.assert_called_once()


def test_results_obj_flag_loads_object():
    fake = MagicMock()
    fake.list_objects.return_value = ['obj-1']
    with patch('bayesed.inspect_cli.list_catalog_names', return_value=['gal']), \
         patch('bayesed.inspect_cli.BayeSEDResults', return_value=fake) as ctor:
        code = run_inspect(['results', 'out', 'objects', '--obj', 'obj-1'])
    assert code == 0
    ctor.assert_called_once_with('out', object_id='obj-1')


def test_results_rejects_leftover_positionals():
    with patch('bayesed.inspect_cli.BayeSEDResults') as ctor, \
         patch('sys.stderr', new_callable=MagicMock) as err:
        code = run_inspect(['results', 'out', 'summary', 'gal'])
    assert code == 1
    ctor.assert_not_called()
    written = ''.join(c.args[0] for c in err.write.call_args_list)
    assert "Error: unexpected argument 'gal'" in written


def test_results_cfg_passed_to_ctor():
    fake = MagicMock()
    with patch('bayesed.inspect_cli.list_catalog_names', return_value=['gal']), \
         patch('bayesed.inspect_cli.BayeSEDResults', return_value=fake) as ctor:
        code = run_inspect([
            'results', 'out', 'summary', '--cat', 'gal', '--cfg', 'cfg0',
        ])
    assert code == 0
    ctor.assert_called_once_with(
        'out', catalog_name='gal', model_config='cfg0',
    )


def test_results_flags_must_follow_all_verbs():
    with patch('bayesed.inspect_cli.BayeSEDResults') as ctor, \
         patch('sys.stderr', new_callable=MagicMock) as err:
        code = run_inspect([
            'results', 'out', 'summary', '--cat', 'gal', 'evidence',
        ])
    assert code == 1
    ctor.assert_not_called()
    written = ''.join(c.args[0] for c in err.write.call_args_list)
    assert "Error: unexpected argument 'evidence'" in written


def test_results_evidence_and_par_names_call_existing_methods():
    fake = MagicMock()
    fake.get_evidence.return_value = {
        'log_evidence': 1.0,
        'log_evidence_error': 0.1,
    }
    fake.get_free_parameters.return_value = ['z']
    fake.get_derived_parameters.return_value = ['log(Mstar)']
    with patch('bayesed.inspect_cli.list_catalog_names', return_value=['gal']), \
         patch('bayesed.inspect_cli.BayeSEDResults', return_value=fake):
        code = run_inspect(['results', 'out', 'evidence', 'par_names'])
    assert code == 0
    fake.get_evidence.assert_called_once()
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
    assert "Error: unexpected argument 'bestfit'" in written


def test_format_object_listing_omits_middle_when_more_than_ten():
    ids = [f'o{i:02d}' for i in range(12)]
    lines = format_object_listing(ids)
    assert lines[0] == 'Objects (12):'
    assert lines[1:6] == [f'  o{i:02d}' for i in range(5)]
    assert lines[6] == '  ...'
    assert lines[7:] == [f'  o{i:02d}' for i in range(7, 12)]
    assert 'o05' not in '\n'.join(lines)
    assert 'o06' not in '\n'.join(lines)


def test_format_object_listing_silent_for_one():
    assert format_object_listing(['only']) == []


def test_plot_without_object_lists_then_uses_first():
    fake = MagicMock()
    fake.object_id = None
    fake.list_objects.return_value = ['obj-a', 'obj-b']
    with patch('bayesed.inspect_cli.list_catalog_names', return_value=['gal']), \
         patch('bayesed.inspect_cli.BayeSEDResults', return_value=fake), \
         patch('sys.stdout', new_callable=MagicMock) as out:
        code = run_inspect(['plot', 'out', 'bestfit', '--cat', 'gal'])
    assert code == 0
    written = ''.join(c.args[0] for c in out.write.call_args_list)
    assert 'obj-a' in written
    assert 'obj-b' in written
    assert 'using first: obj-a' in written
    assert fake.plot_bestfit.call_args.kwargs.get('object_id') == 'obj-a'


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
    fake.object_id = 'obj-1'
    with patch('bayesed.inspect_cli.list_catalog_names', return_value=['gal']), \
         patch('bayesed.inspect_cli.BayeSEDResults', return_value=fake) as ctor:
        code = run_inspect([
            'plot', 'out', 'bestfit', '--cat', 'gal', '--obj', 'obj-1',
        ])
    assert code == 0
    ctor.assert_called_once_with('out', catalog_name='gal', object_id='obj-1')
    fake.plot_bestfit.assert_called_once()
    assert fake.plot_bestfit.call_args.kwargs.get('show') is True
    assert fake.plot_bestfit.call_args.kwargs.get('object_id') == 'obj-1'


def test_plot_bestfit_cfg_passed_to_results():
    fake = MagicMock()
    fake.object_id = None
    fake.list_objects.return_value = ['only']
    with patch('bayesed.inspect_cli.list_catalog_names', return_value=['gal']), \
         patch('bayesed.inspect_cli.BayeSEDResults', return_value=fake) as ctor:
        code = run_inspect([
            'plot', 'out', 'bestfit', '--cat', 'gal',
            '--cfg', '0Stellar+Nebular_2dal8_10_sys_err0',
        ])
    assert code == 0
    ctor.assert_called_once_with(
        'out',
        catalog_name='gal',
        model_config='0Stellar+Nebular_2dal8_10_sys_err0',
    )


def test_plot_bestfit_cfg_requires_value():
    with patch('bayesed.inspect_cli.BayeSEDResults') as ctor, \
         patch('sys.stderr', new_callable=MagicMock) as err:
        code = run_inspect(['plot', 'out', 'bestfit', '--cfg'])
    assert code == 1
    ctor.assert_not_called()
    written = ''.join(c.args[0] for c in err.write.call_args_list)
    assert 'Error: --cfg requires a value' in written


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
    with patch('bayesed.launcher.subprocess.Popen') as mock_popen, \
         patch('sys.stderr', new_callable=MagicMock):
        code = run_launcher(['results', '--help'])
    assert code == 1
    mock_popen.assert_not_called()


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
    assert '--par' in written


def test_par_values_prints_parameter_table():
    fake = MagicMock()
    fake.get_parameter_values.return_value = 'ID age_median\nobj-1 9.5'
    with patch('bayesed.inspect_cli.list_catalog_names', return_value=['gal']), \
         patch('bayesed.inspect_cli.BayeSEDResults', return_value=fake) as ctor, \
         patch('sys.stdout', new_callable=MagicMock) as out:
        code = run_inspect(['results', 'out', 'par_values', '--par', 'age'])
    assert code == 0
    ctor.assert_called_once_with('out')
    fake.get_parameter_values.assert_called_once_with(
        'age', param_match_mode='regex'
    )
    written = ''.join(c.args[0] for c in out.write.call_args_list)
    assert 'ID age_median' in written
    assert 'obj-1 9.5' in written


def test_par_values_cat_loads_catalog_not_object():
    fake = MagicMock()
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
    fake.get_parameter_values.return_value = 'table'
    with patch('bayesed.inspect_cli.list_catalog_names', return_value=['gal']), \
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
        'age', param_match_mode='regex',
    )


def test_par_values_obj_filters_via_get_parameter_values():
    fake = MagicMock()
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
        'age',
        param_match_mode='regex',
        object_ids='STARFORMING',
        object_match_mode='regex',
    )


def test_par_values_invalid_regex_uses_contains():
    fake = MagicMock()
    fake.get_parameter_values.return_value = 'table'
    with patch('bayesed.inspect_cli.list_catalog_names', return_value=['gal']), \
         patch('bayesed.inspect_cli.BayeSEDResults', return_value=fake):
        code = run_inspect([
            'results', 'out', 'par_values', '--par', 'log(age',
            '--obj', 'spec-(',
        ])
    assert code == 0
    fake.get_parameter_values.assert_called_once_with(
        'log(age',
        param_match_mode='contains',
        object_ids='spec-(',
        object_match_mode='contains',
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
    fake.get_parameter_values.side_effect = ValueError(
        "No columns found matching 'nope' with mode 'regex'"
    )
    with patch('bayesed.inspect_cli.list_catalog_names', return_value=['gal']), \
         patch('bayesed.inspect_cli.BayeSEDResults', return_value=fake), \
         patch('sys.stderr', new_callable=MagicMock) as err:
        code = run_inspect(['results', 'out', 'par_values', '--par', 'nope'])
    assert code == 1
    written = ''.join(c.args[0] for c in err.write.call_args_list)
    assert 'No columns found matching' in written


def test_par_values_missing_outdir_is_not_unknown_catalog():
    with patch('bayesed.inspect_cli.list_catalog_names', return_value=[]), \
         patch(
             'bayesed.inspect_cli.BayeSEDResults',
             side_effect=FileNotFoundError("No such directory: missing"),
         ), \
         patch('sys.stderr', new_callable=MagicMock) as err:
        code = run_inspect([
            'results', 'missing', 'par_values', '--cat', 'gal', '--par', 'age',
        ])
    assert code == 1
    written = ''.join(c.args[0] for c in err.write.call_args_list)
    assert 'No such directory' in written
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
    assert 'gal' in written
    assert 'qso' in written
    assert 'catalog_name' not in written
    assert '--cat' in written
    assert 'model_configs --cat gal' in written


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
    assert 'Unknown catalog' in written


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


def _fake_plot_results(object_id=None, objects=None):
    fake = MagicMock()
    fake.object_id = object_id
    fake.list_objects.return_value = objects if objects is not None else ['only']
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


def test_plot_posterior_derived_uses_max_params_ten():
    fake = _fake_plot_results()
    with patch('bayesed.inspect_cli.list_catalog_names', return_value=['gal']), \
         patch('bayesed.inspect_cli.BayeSEDResults', return_value=fake):
        code = run_inspect(['plot', 'out', 'posterior', 'derived'])
    assert code == 0
    fake.plot_posterior_derived.assert_called_once()
    assert fake.plot_posterior_derived.call_args.kwargs.get('show') is True
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
    fake = _fake_plot_results(object_id='STARFORMING')
    with patch('bayesed.inspect_cli.list_catalog_names', return_value=['gal']), \
         patch('bayesed.inspect_cli.BayeSEDResults', return_value=fake) as ctor:
        code = run_inspect([
            'plot', 'out', 'posterior', 'free',
            '--cat', 'gal', '--obj', 'STARFORMING',
        ])
    assert code == 0
    ctor.assert_called_once_with(
        'out', catalog_name='gal', object_id='STARFORMING',
    )
    assert fake.plot_posterior_free.call_args.kwargs.get('object_id') == (
        'STARFORMING'
    )


def test_plot_posterior_cfg_passed_to_results():
    fake = _fake_plot_results(object_id='STARFORMING')
    with patch('bayesed.inspect_cli.list_catalog_names', return_value=['gal']), \
         patch('bayesed.inspect_cli.BayeSEDResults', return_value=fake) as ctor:
        code = run_inspect([
            'plot', 'out', 'posterior', 'free',
            '--cat', 'gal', '--cfg', 'cfg0', '--obj', 'STARFORMING',
        ])
    assert code == 0
    ctor.assert_called_once_with(
        'out',
        catalog_name='gal',
        object_id='STARFORMING',
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
    assert 'Multiple model configurations' in written
    assert 'cfg0' in written


def test_plot_posterior_without_obj_lists_then_uses_first():
    fake = _fake_plot_results(objects=['obj-a', 'obj-b'])
    with patch('bayesed.inspect_cli.list_catalog_names', return_value=['gal']), \
         patch('bayesed.inspect_cli.BayeSEDResults', return_value=fake), \
         patch('sys.stdout', new_callable=MagicMock) as out:
        code = run_inspect(['plot', 'out', 'posterior', 'free'])
    assert code == 0
    written = ''.join(c.args[0] for c in out.write.call_args_list)
    assert 'obj-a' in written
    assert 'obj-b' in written
    assert 'using first: obj-a' in written
    assert fake.plot_posterior_free.call_args.kwargs.get('object_id') == 'obj-a'


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
    assert 'Unknown catalog' in written


if __name__ == '__main__':
    test_results_without_verbs_lists_commands_and_skips_binary()
    test_plot_without_verbs_lists_commands()
    test_help_still_forwards_to_binary()
    test_results_summary_uses_bayesed_results()
    test_results_cat_flag_loads_catalog()
    test_results_obj_flag_loads_object()
    test_results_rejects_leftover_positionals()
    test_results_cfg_passed_to_ctor()
    test_results_flags_must_follow_all_verbs()
    test_results_evidence_and_par_names_call_existing_methods()
    test_plot_bestfit_calls_plot_bestfit()
    test_plot_posterior_cannot_mix_with_bestfit()
    test_plot_posterior_then_bestfit_cannot_mix()
    test_format_object_listing_omits_middle_when_more_than_ten()
    test_format_object_listing_silent_for_one()
    test_plot_without_object_lists_then_uses_first()
    test_plot_with_object_does_not_list()
    test_plot_bestfit_passes_catalog_and_object()
    test_plot_bestfit_cfg_passed_to_results()
    test_plot_bestfit_cfg_requires_value()
    test_plot_bestfit_rejects_leftover_positionals()
    test_priors_list_without_dir_lists_types()
    test_priors_list_dir_uses_existing_loader_and_printer()
    test_results_help_does_not_forward_to_binary()
    test_priors_init_is_rejected()
    test_priors_list_dir_load_error_exits_one()
    test_par_values_requires_par_flag()
    test_par_values_prints_parameter_table()
    test_par_values_cat_loads_catalog_not_object()
    test_par_values_cfg_passed_to_ctor()
    test_par_values_obj_filters_via_get_parameter_values()
    test_par_values_invalid_regex_uses_contains()
    test_par_values_rejects_leftover_positionals()
    test_par_values_rejects_mix_with_other_commands()
    test_par_values_empty_match_exits_one()
    test_par_values_missing_outdir_is_not_unknown_catalog()
    test_par_values_unknown_catalog_exits_one()
    test_params_verb_is_removed()
    test_results_catalogs_lists_names_without_loading_results()
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
    test_plot_posterior_derived_uses_max_params_ten()
    test_plot_posterior_derived_max_flag()
    test_plot_posterior_par_calls_plot_posterior()
    test_plot_posterior_par_lab_pairs_set_labels()
    test_plot_posterior_grouped_par_then_lab_rejected()
    test_plot_posterior_lab_without_par_rejected()
    test_plot_posterior_free_with_par_rejected()
    test_plot_posterior_free_and_derived_rejected()
    test_plot_posterior_max_only_with_derived()
    test_plot_posterior_rejects_leftover_positionals()
    test_plot_posterior_cat_and_obj_flags()
    test_plot_posterior_cfg_passed_to_results()
    test_plot_multiple_configs_without_cfg_exits_one()
    test_plot_posterior_without_obj_lists_then_uses_first()
    test_plot_posterior_unknown_catalog_exits_one()
    print('test_inspect_cli: OK')
