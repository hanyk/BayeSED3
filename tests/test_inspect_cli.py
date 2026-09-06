#!/usr/bin/env python3
"""Unit tests for the Launcher inspect path (ADR-0009; no binary, no HDF5)."""

from unittest.mock import MagicMock, patch

from bayesed.inspect_cli import (
    PLOT_VERBS,
    RESULTS_VERBS,
    format_object_listing,
    parse_outdir_verbs_name,
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


def test_parse_outdir_then_verbs_then_optional_name():
    outdir, cmds, names = parse_outdir_verbs_name(
        ['out', 'summary', 'evidence', 'gal'], RESULTS_VERBS
    )
    assert outdir == 'out'
    assert cmds == ['summary', 'evidence']
    assert names == ['gal']


def test_parse_catalog_and_object_after_verbs():
    outdir, cmds, names = parse_outdir_verbs_name(
        ['out', 'bestfit', 'gal', 'obj-1'], PLOT_VERBS
    )
    assert outdir == 'out'
    assert cmds == ['bestfit']
    assert names == ['gal', 'obj-1']


def test_parse_rejects_name_before_verbs():
    try:
        parse_outdir_verbs_name(['out', 'gal', 'summary'], RESULTS_VERBS)
    except ValueError as e:
        assert 'summary' in str(e) or 'verb' in str(e).lower() or 'name' in str(e).lower()
    else:
        raise AssertionError('expected ValueError')


def test_results_summary_uses_bayesed_results():
    fake = MagicMock()
    with patch('bayesed.inspect_cli.list_catalog_names', return_value=['gal']), \
         patch('bayesed.inspect_cli.BayeSEDResults', return_value=fake) as ctor:
        code = run_inspect(['results', 'out', 'summary'])
    assert code == 0
    ctor.assert_called_once_with('out')
    fake.print_summary.assert_called_once()


def test_results_name_is_catalog_when_listed():
    fake = MagicMock()
    with patch('bayesed.inspect_cli.list_catalog_names', return_value=['gal']), \
         patch('bayesed.inspect_cli.BayeSEDResults', return_value=fake) as ctor:
        code = run_inspect(['results', 'out', 'objects', 'gal'])
    assert code == 0
    ctor.assert_called_once_with('out', catalog_name='gal')
    fake.list_objects.assert_called_once()


def test_results_name_is_object_when_not_catalog():
    fake = MagicMock()
    fake.list_objects.return_value = ['obj-1']
    with patch('bayesed.inspect_cli.list_catalog_names', return_value=['gal']), \
         patch('bayesed.inspect_cli.BayeSEDResults', return_value=fake) as ctor:
        code = run_inspect(['results', 'out', 'objects', 'obj-1'])
    assert code == 0
    ctor.assert_called_once_with('out', object_id='obj-1')


def test_results_evidence_and_params_call_existing_methods():
    fake = MagicMock()
    fake.get_evidence.return_value = {
        'log_evidence': 1.0,
        'log_evidence_error': 0.1,
    }
    fake.get_free_parameters.return_value = ['z']
    fake.get_derived_parameters.return_value = ['log(Mstar)']
    with patch('bayesed.inspect_cli.list_catalog_names', return_value=['gal']), \
         patch('bayesed.inspect_cli.BayeSEDResults', return_value=fake):
        code = run_inspect(['results', 'out', 'evidence', 'params'])
    assert code == 0
    fake.get_evidence.assert_called_once()
    fake.get_free_parameters.assert_called_once()
    fake.get_derived_parameters.assert_called_once()


def test_plot_verbs_call_existing_plot_methods():
    fake = MagicMock()
    fake.object_id = None
    fake.list_objects.return_value = ['only']
    with patch('bayesed.inspect_cli.list_catalog_names', return_value=['gal']), \
         patch('bayesed.inspect_cli.BayeSEDResults', return_value=fake):
        code = run_inspect(['plot', 'out', 'bestfit', 'posterior'])
    assert code == 0
    fake.plot_bestfit.assert_called_once()
    fake.plot_posterior_free.assert_called_once()
    assert fake.plot_bestfit.call_args.kwargs.get('show') is True
    assert fake.plot_posterior_free.call_args.kwargs.get('show') is True


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
        code = run_inspect(['plot', 'out', 'bestfit', 'gal'])
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
        code = run_inspect(['plot', 'out', 'bestfit', 'gal', 'obj-b'])
    assert code == 0
    written = ''.join(c.args[0] for c in out.write.call_args_list)
    assert 'using first' not in written
    fake.plot_bestfit.assert_called_once()


def test_plot_bestfit_passes_catalog_and_object():
    fake = MagicMock()
    fake.object_id = 'obj-1'
    with patch('bayesed.inspect_cli.list_catalog_names', return_value=['gal']), \
         patch('bayesed.inspect_cli.BayeSEDResults', return_value=fake) as ctor:
        code = run_inspect(['plot', 'out', 'bestfit', 'gal', 'obj-1'])
    assert code == 0
    ctor.assert_called_once_with('out', catalog_name='gal', object_id='obj-1')
    fake.plot_bestfit.assert_called_once()
    assert fake.plot_bestfit.call_args.kwargs.get('show') is True
    assert fake.plot_bestfit.call_args.kwargs.get('object_id') == 'obj-1'


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


def test_priors_without_list_prints_usage():
    with patch('sys.stderr', new_callable=MagicMock) as err:
        code = run_inspect(['priors'])
    assert code == 1
    written = ''.join(c.args[0] for c in err.write.call_args_list)
    assert 'list' in written


if __name__ == '__main__':
    test_results_without_verbs_lists_commands_and_skips_binary()
    test_plot_without_verbs_lists_commands()
    test_help_still_forwards_to_binary()
    test_parse_outdir_then_verbs_then_optional_name()
    test_parse_catalog_and_object_after_verbs()
    test_parse_rejects_name_before_verbs()
    test_results_summary_uses_bayesed_results()
    test_results_name_is_catalog_when_listed()
    test_results_name_is_object_when_not_catalog()
    test_results_evidence_and_params_call_existing_methods()
    test_plot_verbs_call_existing_plot_methods()
    test_format_object_listing_omits_middle_when_more_than_ten()
    test_format_object_listing_silent_for_one()
    test_plot_without_object_lists_then_uses_first()
    test_plot_with_object_does_not_list()
    test_plot_bestfit_passes_catalog_and_object()
    test_priors_list_without_dir_lists_types()
    test_priors_list_dir_uses_existing_loader_and_printer()
    test_results_help_does_not_forward_to_binary()
    test_priors_init_is_rejected()
    test_priors_list_dir_load_error_exits_one()
    test_priors_without_list_prints_usage()
    print('test_inspect_cli: OK')
