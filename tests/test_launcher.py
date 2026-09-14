#!/usr/bin/env python3
"""Unit tests for the Launcher (ADR-0007, ADR-0009; no real binary required)."""

from unittest.mock import MagicMock, patch

from bayesed.core import BayeSEDInterface
from bayesed.launcher import parse_launcher_argv, run_launcher, usage_text


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
    assert 'results' in text
    assert 'plot' in text
    assert 'priors' in text
    assert 'bayesed3 run' in text
    assert '--mpi' in text
    assert '--cat' not in text
    assert '--cfg' not in text
    assert '--obj' not in text


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
    test_run_launcher_empty_argv_prints_overview_exit_zero()
    test_run_launcher_help_prints_overview_not_binary()
    test_run_launcher_dash_h_prints_overview()
    test_run_bare_forwards_help_to_binary()
    test_run_help_equivalent_to_bare_run()
    test_run_forwards_mpi_after_run()
    test_bare_binary_argv_suggests_run_rewrite()
    test_interface_run_rejects_list_argv()
    print('test_launcher: OK')
