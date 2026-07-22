#!/usr/bin/env python3
"""Unit tests for the Launcher (ADR-0007; no real binary required)."""

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


def test_usage_mentions_help_and_owned_flags():
    text = usage_text('bayesed3')
    assert '--mpi' in text
    assert '--np' in text
    assert 'bayesed3 --help' in text


def test_run_launcher_empty_argv_prints_usage():
    with patch('bayesed.launcher.sys.stderr', new_callable=MagicMock) as err:
        code = run_launcher([])
    assert code == 1
    written = ''.join(c.args[0] for c in err.write.call_args_list)
    assert '--mpi' in written


def test_run_launcher_help_forces_one_process_and_forwards():
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
        code = run_launcher(['--help'])
    assert code == 0
    mock_build.assert_called_once()
    assert mock_build.call_args[0][1] == 1
    mock_popen.assert_called_once()
    assert mock_popen.call_args.kwargs.get('cwd') == '/data/root'


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
    test_usage_mentions_help_and_owned_flags()
    test_run_launcher_empty_argv_prints_usage()
    test_run_launcher_help_forces_one_process_and_forwards()
    test_interface_run_rejects_list_argv()
    print('test_launcher: OK')
