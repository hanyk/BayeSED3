"""
Minimal setup.py for data_files support.

This file only exists because pyproject.toml doesn't support data_files.
All other metadata (dependencies, version, etc.) is in pyproject.toml.

For modern pip (>=21.3), pyproject.toml is used for everything except data_files.
"""

from setuptools import setup
import os
import subprocess


def get_data_files(root=None):
    """
    Get data files to install to $PREFIX/share/bayesed3/.

    Uses git ls-files to include only git-tracked files.
    If git is unavailable (GitHub zip / sdist), walks the tree — a clean
    snapshot of the same files.
    Auto-detects platform and filters platform-specific binaries accordingly.
    """
    data_files = []
    if root is None:
        root = os.path.dirname(os.path.abspath(__file__))

    # Auto-detect platform
    import platform as plat
    system = plat.system()
    if system == 'Linux':
        platform_filter = 'linux'
    elif system == 'Darwin':
        platform_filter = 'macos'
    else:
        # Unsupported platform - raise clear error
        raise RuntimeError(
            f"\n{'='*70}\n"
            f"ERROR: BayeSED3 is not supported on {system}\n"
            f"{'='*70}\n"
            f"Supported platforms:\n"
            f"  - Linux (x86_64)\n"
            f"  - macOS (x86_64 and ARM64)\n"
            f"\n"
            f"Windows users: Please install via Windows Subsystem for Linux (WSL):\n"
            f"  1. Install WSL: https://docs.microsoft.com/en-us/windows/wsl/install\n"
            f"  2. Inside WSL, run: pip install bayesed3\n"
            f"{'='*70}\n"
        )

    try:
        # Get all git-tracked files
        result = subprocess.run(
            ['git', 'ls-files'],
            cwd=root,
            capture_output=True,
            text=True,
            check=True
        )
        tracked_files = result.stdout.strip().split('\n')
    except (subprocess.CalledProcessError, FileNotFoundError):
        # GitHub zip / sdist: no .git. Walk the snapshot (same skip rules below).
        tracked_files = []
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [name for name in dirnames if name not in ('.git', '__pycache__')]
            for name in filenames:
                abs_path = os.path.join(dirpath, name)
                if os.path.isfile(abs_path):
                    rel = os.path.relpath(abs_path, root).replace('\\', '/')
                    tracked_files.append(rel)

    # Filter files to include
    included_files = []
    skipped_binaries = []
    for file_path in tracked_files:
        file_path = file_path.replace('\\', '/')
        if not file_path or not os.path.isfile(os.path.join(root, file_path)):
            continue

        # Skip Python package files (installed via packages)
        if file_path.startswith('bayesed/') and file_path.endswith('.py'):
            continue
        if file_path == 'bayesed_gui.py':
            continue

        # Skip packaging files
        if file_path in ['setup.py', 'pyproject.toml', 'MANIFEST.in',
                       '.gitignore', '.cursorignore']:
            continue
        if file_path.startswith('conda/') or file_path.startswith('.git/'):
            continue

        # Platform-specific binary filtering
        # Structure: bin/{linux,mac}/{x86,arm}/
        if file_path.startswith('bin/'):
            if platform_filter == 'linux' and file_path.startswith('bin/mac/'):
                skipped_binaries.append(file_path)
                continue  # Skip macOS binaries on Linux
            elif platform_filter == 'macos' and file_path.startswith('bin/linux/'):
                skipped_binaries.append(file_path)
                continue  # Skip Linux binaries on macOS

        included_files.append(file_path)

    # Group files by directory for data_files format
    if included_files:
        file_map = {}
        for file_path in included_files:
            rel_dir = os.path.dirname(file_path) if os.path.dirname(file_path) else '.'
            if rel_dir not in file_map:
                file_map[rel_dir] = []
            file_map[rel_dir].append(file_path)

        # Convert to data_files format: (install_dir, [source_files])
        for rel_dir, files in file_map.items():
            if rel_dir == '.':
                install_dir = 'share/bayesed3'
            else:
                install_dir = os.path.join('share', 'bayesed3', rel_dir)
            data_files.append((install_dir, files))

    return data_files


# pip/setuptools exec setup.py as __main__. Tests import this module without
# installing, so they can call get_data_files() on a temp tree.
if __name__ == '__main__':
    setup(
        data_files=get_data_files(),
    )
