#!/usr/bin/env python3
"""data_files: git ls-files, or a full-tree walk with the same skip rules."""

import os
import platform
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import setup as setup_mod  # noqa: E402


def _touch(root, relative):
    path = os.path.join(root, relative)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as handle:
        handle.write('x')


def _installed_paths(data_files):
    paths = []
    for _, group in data_files:
        paths.extend(p.replace('\\', '/') for p in group)
    return set(paths)


def test_zip_walk_keeps_docs_skips_packaging():
    with tempfile.TemporaryDirectory() as tmp:
        _touch(tmp, os.path.join('bin', 'mac', 'x86', 'bayesed_mn_1'))
        _touch(tmp, os.path.join('bin', 'linux', 'x86', 'bayesed_mn_1'))
        _touch(tmp, os.path.join('models', 'ssp.dat'))
        _touch(tmp, os.path.join('tests', 'junk.txt'))
        _touch(tmp, os.path.join('docs', 'note.md'))
        _touch(tmp, os.path.join('bayesed', 'core.py'))
        _touch(tmp, os.path.join('conda', 'meta.yaml'))
        _touch(tmp, 'setup.py')
        _touch(tmp, 'bayesed_gui.py')

        posix = _installed_paths(setup_mod.get_data_files(tmp))
        this_os = 'linux' if platform.system() == 'Linux' else 'mac'
        other_os = 'mac' if this_os == 'linux' else 'linux'

        assert f'bin/{this_os}/x86/bayesed_mn_1' in posix
        assert f'bin/{other_os}/x86/bayesed_mn_1' not in posix
        assert 'models/ssp.dat' in posix
        assert 'tests/junk.txt' in posix
        assert 'docs/note.md' in posix
        assert 'bayesed/core.py' not in posix
        assert 'conda/meta.yaml' not in posix
        assert 'setup.py' not in posix
        assert 'bayesed_gui.py' not in posix


def test_git_ls_files_skips_untracked():
    try:
        subprocess.run(['git', '--version'], check=True, capture_output=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        return

    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(['git', 'init'], cwd=tmp, check=True, capture_output=True)
        _touch(tmp, os.path.join('bin', 'mac', 'x86', 'bayesed_mn_1'))
        _touch(tmp, os.path.join('bin', 'linux', 'x86', 'bayesed_mn_1'))
        _touch(tmp, os.path.join('models', 'ssp.dat'))
        _touch(tmp, os.path.join('tests', 'tracked.txt'))
        _touch(tmp, os.path.join('bayesed', 'core.py'))
        subprocess.run(
            ['git', 'add', 'bin', 'models', 'tests', 'bayesed'],
            cwd=tmp,
            check=True,
        )
        _touch(tmp, os.path.join('models', 'untracked.dat'))

        posix = _installed_paths(setup_mod.get_data_files(tmp))
        assert 'models/ssp.dat' in posix
        assert 'tests/tracked.txt' in posix
        assert 'models/untracked.dat' not in posix
        assert 'bayesed/core.py' not in posix


if __name__ == '__main__':
    test_zip_walk_keeps_docs_skips_packaging()
    test_git_ls_files_skips_untracked()
    print('ok')
