#!/usr/bin/env python3
"""Unit tests for the Launcher (ADR-0007, ADR-0009; no real binary required)."""

from unittest.mock import MagicMock, patch

from importlib.metadata import PackageNotFoundError

from bayesed.core import BayeSEDInterface
from bayesed.launcher import (
    parse_launcher_argv,
    parse_leading_log_level,
    run_launcher,
    usage_text,
)


def test_parse_launcher_strips_owned_flags():
    mode, np, args = parse_launcher_argv(
        ['--mpi', 'n', '--np', '8', '-i', '0,cat.txt', '--outdir', 'out']
    )
    assert mode == 'mn_n'
    assert np == 8
    assert args == ['-i', '0,cat.txt', '--outdir', 'out']


def test_parse_launcher_default_mpi():
    mode, np, args = parse_launcher_argv(['--help'])
    assert mode == 'mn_1'
    assert np is None
    assert args == ['--help']


def test_parse_launcher_rejects_bad_mpi():
    try:
        parse_launcher_argv(['--mpi', 'x', '-h'])
    except ValueError as e:
        assert 'Invalid MPI mode' in str(e)
    else:
        raise AssertionError('expected ValueError')


def test_usage_is_short_overview_without_inspect_flags():
    text = usage_text('bayesed3')
    assert text.startswith(
        'bayesed3 - A Bayesian SED synthesis and analysis tool for galaxies '
        'and AGNs\n'
    )
    inspect_idx = text.index('Inspect:\n')
    log_level_line = (
        '  bayesed3 [--log-level DEBUG|INFO|WARNING|ERROR|CRITICAL] '
        'results|plot|priors\n'
    )
    assert log_level_line in text
    assert text.index(log_level_line) > inspect_idx
    assert text.index(log_level_line) < text.index(
        '  bayesed3 results <outdir>'
    )
    assert (
        'bayesed3 results <outdir> '
        '[summary|evidence|par_names|objects|par_values|catalogs|model_configs]\n'
    ) in text
    assert 'bayesed3 plot <outdir> [bestfit|posterior]\n' in text
    assert 'bayesed3 priors list [dir]\n' in text
    assert 'bayesed3 run' in text
    assert '--mpi' in text
    assert 'version' in text
    assert 'More:' not in text
    assert '--cat' not in text
    assert '--cfg' not in text
    assert '--obj' not in text
    assert 'posterior free' not in text
    assert 'derived' not in text


def test_parse_leading_log_level():
    level, rest = parse_leading_log_level(
        ['--log-level', 'info', 'results', 'out', 'summary'],
    )
    assert level == 'INFO'
    assert rest == ['results', 'out', 'summary']
    level, rest = parse_leading_log_level(['results', 'out'])
    assert level is None
    assert rest == ['results', 'out']


def test_parse_leading_log_level_rejects_invalid():
    try:
        parse_leading_log_level(['--log-level', 'noisy', 'results'])
    except ValueError as e:
        assert 'invalid --log-level' in str(e).lower()
        assert 'DEBUG|INFO|WARNING|ERROR|CRITICAL' in str(e)
    else:
        raise AssertionError('expected ValueError')


def test_log_level_before_run_is_rejected():
    with patch('bayesed.launcher.subprocess.Popen') as mock_popen, \
         patch('bayesed.launcher.sys.stderr', new_callable=MagicMock) as err:
        code = run_launcher(['--log-level', 'DEBUG', 'run', '--help'])
    assert code == 1
    mock_popen.assert_not_called()
    written = ''.join(c.args[0] for c in err.write.call_args_list)
    assert '--log-level applies only to results|plot|priors' in written


def test_log_level_before_version_is_rejected():
    with patch('bayesed.launcher.sys.stderr', new_callable=MagicMock) as err, \
         patch('bayesed.launcher.sys.stdout', new_callable=MagicMock) as out:
        code = run_launcher(['--log-level', 'INFO', 'version'])
    assert code == 1
    out.write.assert_not_called()
    written = ''.join(c.args[0] for c in err.write.call_args_list)
    assert '--log-level applies only to results|plot|priors' in written


def test_inspect_default_sets_results_logger_warning():
    import logging
    logging.getLogger('bayesed.results').setLevel(logging.INFO)
    with patch('bayesed.launcher.run_inspect', return_value=0) as mock_inspect:
        code = run_launcher(['results', 'out', 'summary'])
    assert code == 0
    mock_inspect.assert_called_once_with(['results', 'out', 'summary'], prog='bayesed3')
    assert logging.getLogger('bayesed.results').level == logging.WARNING


def test_inspect_log_level_info_emits_results_info(capsys):
    import logging

    # Isolate from any prior basicConfig in this process.
    root = logging.getLogger()
    saved_handlers = list(root.handlers)
    saved_level = root.level
    try:
        for handler in saved_handlers:
            root.removeHandler(handler)
        root.setLevel(logging.WARNING)
        with patch('bayesed.launcher.run_inspect', return_value=0):
            code = run_launcher([
                '--log-level', 'INFO', 'results', 'out', 'summary',
            ])
        assert code == 0
        logging.getLogger('bayesed.results.results_products').info(
            'probe-auto-selected',
        )
        err = capsys.readouterr().err
        assert 'probe-auto-selected' in err
    finally:
        for handler in list(root.handlers):
            root.removeHandler(handler)
        for handler in saved_handlers:
            root.addHandler(handler)
        root.setLevel(saved_level)


def test_run_launcher_version_prints_python_and_binary():
    banner = (
        'BayeSED3  (abc, develop, 2026-08-22 14:54:40, Darwin 24.5.0), '
        'Copyright (C) 2014-2026 Yunkun Han'
    )
    help_out = f'noise\n{banner}\nUSAGE: ...\n'
    completed = MagicMock(stdout=help_out, stderr='')
    with patch('bayesed.launcher.package_version', return_value='2026.9.14.0.dev11'), \
         patch('bayesed.launcher.resolve_system_info', return_value=('mac', 'arm')), \
         patch('bayesed.launcher.resolve_executable', return_value='/bin/bayesed_mn_1'), \
         patch('bayesed.launcher.resolve_binary_cwd', return_value=None), \
         patch('bayesed.launcher.subprocess.run', return_value=completed) as mock_run, \
         patch('bayesed.launcher.sys.stdout', new_callable=MagicMock) as out, \
         patch('bayesed.launcher.subprocess.Popen') as mock_popen:
        code = run_launcher(['version'])
    assert code == 0
    mock_popen.assert_not_called()
    mock_run.assert_called_once()
    assert mock_run.call_args.args[0] == ['/bin/bayesed_mn_1', '--help']
    written = ''.join(c.args[0] for c in out.write.call_args_list)
    assert written == (
        'python  2026.9.14.0.dev11\n'
        f'binary  {banner}\n'
    )


def test_run_launcher_dash_dash_version_prints_python_and_binary():
    banner = (
        'BayeSED3  (abc, develop, 2026-08-22 14:54:40, Darwin 24.5.0), '
        'Copyright (C) 2014-2026'
    )
    completed = MagicMock(stdout=f'{banner}\n', stderr='')
    with patch('bayesed.launcher.package_version', return_value='2026.9.14.0.dev11'), \
         patch('bayesed.launcher.resolve_system_info', return_value=('mac', 'arm')), \
         patch('bayesed.launcher.resolve_executable', return_value='/bin/bayesed_mn_1'), \
         patch('bayesed.launcher.resolve_binary_cwd', return_value=None), \
         patch('bayesed.launcher.subprocess.run', return_value=completed), \
         patch('bayesed.launcher.sys.stdout', new_callable=MagicMock) as out:
        code = run_launcher(['--version'])
    assert code == 0
    written = ''.join(c.args[0] for c in out.write.call_args_list)
    assert written.startswith('python  2026.9.14.0.dev11\n')
    assert written.endswith(f'binary  {banner}\n')


def test_run_launcher_version_reports_binary_unavailable():
    with patch('bayesed.launcher.package_version', return_value='2026.9.14.0.dev11'), \
         patch('bayesed.launcher.resolve_system_info', return_value=('mac', 'arm')), \
         patch('bayesed.launcher.resolve_executable', return_value='/bin/bayesed_mn_1'), \
         patch('bayesed.launcher.resolve_binary_cwd', return_value=None), \
         patch(
             'bayesed.launcher.subprocess.run',
             side_effect=FileNotFoundError('missing binary'),
         ), \
         patch('bayesed.launcher.sys.stdout', new_callable=MagicMock) as out:
        code = run_launcher(['version'])
    assert code == 0
    written = ''.join(c.args[0] for c in out.write.call_args_list)
    assert written.startswith('python  2026.9.14.0.dev11\n')
    assert 'binary  (unavailable:' in written
    assert 'missing binary' in written


def test_run_launcher_version_reports_package_unavailable():
    completed = MagicMock(
        stdout='BayeSED3  (hash), Copyright (C) 2014-2026\n',
        stderr='',
    )
    with patch(
             'bayesed.launcher.package_version',
             side_effect=PackageNotFoundError('bayesed3'),
         ), \
         patch('bayesed.launcher.resolve_system_info', return_value=('mac', 'arm')), \
         patch('bayesed.launcher.resolve_executable', return_value='/bin/bayesed_mn_1'), \
         patch('bayesed.launcher.resolve_binary_cwd', return_value=None), \
         patch('bayesed.launcher.subprocess.run', return_value=completed), \
         patch('bayesed.launcher.sys.stdout', new_callable=MagicMock) as out:
        code = run_launcher(['version'])
    assert code == 0
    written = ''.join(c.args[0] for c in out.write.call_args_list)
    assert written.startswith('python  (unavailable:')
    assert 'binary  BayeSED3  (hash), Copyright (C) 2014-2026\n' in written


def test_run_launcher_version_parses_first_bayesed_line_from_help():
    help_out = (
        'noise\n'
        'BayeSED3  (hash, develop, 2026-08-22 14:54:40, Darwin), '
        'Copyright (C) 2014-2026 Yunkun Han\n'
        'USAGE: ...\n'
    )
    completed = MagicMock(stdout=help_out, stderr='')
    with patch('bayesed.launcher.package_version', return_value='1.0'), \
         patch('bayesed.launcher.resolve_system_info', return_value=('mac', 'arm')), \
         patch('bayesed.launcher.resolve_executable', return_value='/bin/bayesed_mn_1'), \
         patch('bayesed.launcher.resolve_binary_cwd', return_value=None), \
         patch('bayesed.launcher.subprocess.run', return_value=completed) as mock_run, \
         patch('bayesed.launcher.sys.stdout', new_callable=MagicMock) as out:
        code = run_launcher(['version'])
    assert code == 0
    mock_run.assert_called_once()
    assert mock_run.call_args.args[0] == ['/bin/bayesed_mn_1', '--help']
    written = ''.join(c.args[0] for c in out.write.call_args_list)
    assert 'BayeSED3  (hash, develop' in written
    assert 'Copyright' in written
    assert 'USAGE' not in written.split('binary  ', 1)[1]


def test_run_launcher_empty_argv_prints_overview_exit_zero():
    with patch('bayesed.launcher.sys.stderr', new_callable=MagicMock) as err:
        code = run_launcher([])
    assert code == 0
    written = ''.join(c.args[0] for c in err.write.call_args_list)
    assert 'bayesed3 run' in written
    assert '--cat' not in written


def test_run_launcher_help_prints_overview_not_binary():
    with patch('bayesed.launcher.subprocess.Popen') as mock_popen, \
         patch('bayesed.launcher.sys.stderr', new_callable=MagicMock) as err:
        code = run_launcher(['--help'])
    assert code == 0
    mock_popen.assert_not_called()
    written = ''.join(c.args[0] for c in err.write.call_args_list)
    assert 'bayesed3 run' in written


def test_run_launcher_dash_h_prints_overview():
    with patch('bayesed.launcher.subprocess.Popen') as mock_popen, \
         patch('bayesed.launcher.sys.stderr', new_callable=MagicMock):
        code = run_launcher(['-h'])
    assert code == 0
    mock_popen.assert_not_called()


def test_run_bare_forwards_help_to_binary():
    fake_proc = MagicMock()
    fake_proc.wait.return_value = 0
    with patch('bayesed.launcher.resolve_system_info', return_value=('darwin', 'arm64')), \
         patch('bayesed.launcher.resolve_mpirun', return_value='/mpi/mpirun'), \
         patch('bayesed.launcher.resolve_num_processes', return_value=16), \
         patch('bayesed.launcher.resolve_executable', return_value='/bin/bayesed_mn_1'), \
         patch('bayesed.launcher.resolve_binary_cwd', return_value='/data/root'), \
         patch('bayesed.launcher.absolutize_cli_path_args', side_effect=lambda a, base_dir=None: a), \
         patch('bayesed.launcher.build_mpi_command', return_value=['cmd']) as mock_build, \
         patch('bayesed.launcher.subprocess.Popen', return_value=fake_proc) as mock_popen:
        code = run_launcher(['run'])
    assert code == 0
    mock_build.assert_called_once()
    assert mock_build.call_args[0][1] == 1
    assert mock_build.call_args[0][3] == ['--help']
    mock_popen.assert_called_once()


def test_run_help_equivalent_to_bare_run():
    fake_proc = MagicMock()
    fake_proc.wait.return_value = 0
    with patch('bayesed.launcher.resolve_system_info', return_value=('darwin', 'arm64')), \
         patch('bayesed.launcher.resolve_mpirun', return_value='/mpi/mpirun'), \
         patch('bayesed.launcher.resolve_num_processes', return_value=16), \
         patch('bayesed.launcher.resolve_executable', return_value='/bin/bayesed_mn_1'), \
         patch('bayesed.launcher.resolve_binary_cwd', return_value='/data/root'), \
         patch('bayesed.launcher.absolutize_cli_path_args', side_effect=lambda a, base_dir=None: a), \
         patch('bayesed.launcher.build_mpi_command', return_value=['cmd']) as mock_build, \
         patch('bayesed.launcher.subprocess.Popen', return_value=fake_proc):
        code = run_launcher(['run', '--help'])
    assert code == 0
    assert mock_build.call_args[0][3] == ['--help']
    assert mock_build.call_args[0][1] == 1


def test_run_forwards_mpi_after_run():
    fake_proc = MagicMock()
    fake_proc.wait.return_value = 0
    with patch('bayesed.launcher.resolve_system_info', return_value=('darwin', 'arm64')), \
         patch('bayesed.launcher.resolve_mpirun', return_value='/mpi/mpirun'), \
         patch('bayesed.launcher.resolve_num_processes', return_value=8), \
         patch('bayesed.launcher.resolve_executable', return_value='/bin/bayesed_mn_n'), \
         patch('bayesed.launcher.resolve_binary_cwd', return_value='/data/root'), \
         patch('bayesed.launcher.absolutize_cli_path_args', side_effect=lambda a, base_dir=None: a), \
         patch('bayesed.launcher.build_mpi_command', return_value=['cmd']) as mock_build, \
         patch('bayesed.launcher.subprocess.Popen', return_value=fake_proc):
        code = run_launcher([
            'run', '--mpi', 'n', '--np', '8', '-i', '0,cat.txt', '--outdir', 'out',
        ])
    assert code == 0
    assert mock_build.call_args[0][3] == ['-i', '0,cat.txt', '--outdir', 'out']
    mock_build.assert_called_once()
    # mpi mode digit reaches resolve_executable via mn_n
    # resolve_executable was patched; check parse path via build's binary args only


def test_bare_binary_argv_suggests_run_rewrite():
    with patch('bayesed.launcher.subprocess.Popen') as mock_popen, \
         patch('bayesed.launcher.sys.stderr', new_callable=MagicMock) as err:
        code = run_launcher(['-i', '0,cat.txt', '--outdir', 'out'])
    assert code == 1
    mock_popen.assert_not_called()
    written = ''.join(c.args[0] for c in err.write.call_args_list)
    assert "unknown command '-i'" in written
    assert 'bayesed3 run -i 0,cat.txt --outdir out' in written
    assert 'bayesed3 --help' in written


def test_interface_run_rejects_list_argv():
    iface = BayeSEDInterface.__new__(BayeSEDInterface)
    try:
        iface.run(['--help'])
    except TypeError as e:
        assert 'BayeSEDParams' in str(e)
    else:
        raise AssertionError('expected TypeError')


if __name__ == '__main__':
    test_parse_launcher_strips_owned_flags()
    test_parse_launcher_default_mpi()
    test_parse_launcher_rejects_bad_mpi()
    test_usage_is_short_overview_without_inspect_flags()
    test_parse_leading_log_level()
    test_parse_leading_log_level_rejects_invalid()
    test_log_level_before_run_is_rejected()
    test_log_level_before_version_is_rejected()
    test_inspect_default_sets_results_logger_warning()
    # test_inspect_log_level_info_emits_results_info needs pytest capsys
    test_run_launcher_version_prints_python_and_binary()
    test_run_launcher_dash_dash_version_prints_python_and_binary()
    test_run_launcher_version_reports_binary_unavailable()
    test_run_launcher_version_reports_package_unavailable()
    test_run_launcher_version_parses_first_bayesed_line_from_help()
    test_run_launcher_empty_argv_prints_overview_exit_zero()
    test_run_launcher_help_prints_overview_not_binary()
    test_run_launcher_dash_h_prints_overview()
    test_run_bare_forwards_help_to_binary()
    test_run_help_equivalent_to_bare_run()
    test_run_forwards_mpi_after_run()
    test_bare_binary_argv_suggests_run_rewrite()
    test_interface_run_rejects_list_argv()
    print('test_launcher: OK')
