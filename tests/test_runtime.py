#!/usr/bin/env python3
"""Unit tests for internal runtime bootstrap helpers (no binary / OpenMPI install)."""

import os
from unittest.mock import patch

from bayesed.runtime import (
    resolve_executable,
    resolve_num_processes,
    resolve_system_info,
)

_SCHEDULER_KEYS = (
    'PBS_NCPUS',
    'PBS_NODEFILE',
    'SLURM_CPUS_PER_TASK',
    'SLURM_NTASKS',
    'NSLOTS',
)


def _without_scheduler_env(**overrides):
    """Build an environ patch that clears scheduler vars then applies overrides."""
    env = {k: v for k, v in os.environ.items() if k not in _SCHEDULER_KEYS}
    env.update(overrides)
    return env


def test_resolve_num_processes_explicit_override():
    assert resolve_num_processes(4) == 4


def test_resolve_num_processes_pbs_ncpus():
    with patch.dict(os.environ, _without_scheduler_env(PBS_NCPUS='8'), clear=True):
        assert resolve_num_processes() == 8


def test_resolve_num_processes_slurm():
    with patch.dict(
        os.environ, _without_scheduler_env(SLURM_CPUS_PER_TASK='3'), clear=True
    ):
        assert resolve_num_processes() == 3


def test_resolve_executable_linux_x86():
    with patch('bayesed.runtime._get_resource_path', side_effect=lambda p: p) as mock_path:
        path = resolve_executable('mn_1', 'linux', 'x86_64')
    assert path == os.path.join('bin', 'linux', 'x86', 'bayesed_mn_1')
    mock_path.assert_called_once_with(path)


def test_resolve_executable_darwin_arm():
    with patch('bayesed.runtime._get_resource_path', side_effect=lambda p: p):
        path = resolve_executable('mn_n', 'darwin', 'arm64')
    assert path == os.path.join('bin', 'mac', 'arm', 'bayesed_mn_n')


def test_resolve_executable_unsupported_os():
    try:
        resolve_executable('mn_1', 'freebsd', 'x86_64')
    except ValueError as e:
        assert 'Unsupported operating system' in str(e)
    else:
        raise AssertionError('expected ValueError')


def test_resolve_system_info_rejects_bad_arch():
    with patch('bayesed.runtime.platform.system', return_value='Linux'), \
         patch('bayesed.runtime.platform.machine', return_value='i386'):
        try:
            resolve_system_info()
        except ValueError as e:
            assert 'Unsupported architecture' in str(e)
        else:
            raise AssertionError('expected ValueError')


if __name__ == '__main__':
    test_resolve_num_processes_explicit_override()
    test_resolve_num_processes_pbs_ncpus()
    test_resolve_num_processes_slurm()
    test_resolve_executable_linux_x86()
    test_resolve_executable_darwin_arm()
    test_resolve_executable_unsupported_os()
    test_resolve_system_info_rejects_bad_arch()
    print('test_runtime: OK')
