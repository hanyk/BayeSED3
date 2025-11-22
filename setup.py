"""
Setup script for BayeSED3.

This setup.py is maintained for backward compatibility and conda packaging.
For modern Python packaging, see pyproject.toml.
For development installation, use: pip install -e .
"""

from setuptools import setup, find_packages
import os
import subprocess
import sys

# Read version from bayesed/__init__.py
def get_version():
    """Read version from bayesed/__init__.py"""
    version_file = os.path.join(os.path.dirname(__file__), 'bayesed', '__init__.py')
    with open(version_file, 'r') as f:
        for line in f:
            if line.startswith('__version__'):
                return line.split('=')[1].strip().strip('"').strip("'")
    return "3.0.0"

# Read README for long description
def get_long_description():
    """Read long description from README.md"""
    readme_file = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_file):
        with open(readme_file, 'r', encoding='utf-8') as f:
            return f.read()
    return ""

def get_data_files():
    """
    Get data files to include in package installation.
    Uses git ls-files to include all git-tracked files except pip/conda-related files.
    Installs to share/bayesed3/ to match conda installation structure.
    
    Note: For conda builds, this returns empty list because conda build script
    handles file copying separately. This is only used for pip installs.
    """
    import re
    
    # Skip data_files for conda builds - conda build script handles this separately
    # Check if we're in a conda build environment
    # CONDA_BUILD is set by conda-build, PREFIX is also set during conda-build
    # (but not in regular conda environments - they use CONDA_PREFIX instead)
    if os.environ.get('CONDA_BUILD') == '1':
        # In conda build, return empty - conda build script will copy files
        return []
    
    data_files = []
    repo_root = os.path.dirname(os.path.abspath(__file__))
    
    # Only skip pip/conda packaging files and Python package files (installed separately)
    SKIP_PATTERNS = [
        r'^setup\.py$',  # Pip packaging file
        r'^pyproject\.toml$',  # Pip packaging file
        r'^MANIFEST\.in$',  # Pip packaging file
        r'^conda/',  # Conda packaging files
        r'^bayesed/.*\.py$',  # Python package files installed via packages
        r'^bayesed_gui\.py$',  # GUI module installed via py_modules
    ]
    
    # Check if we're in a git repository
    try:
        # Get all git-tracked files
        result = subprocess.run(
            ['git', 'ls-files'],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True
        )
        tracked_files = result.stdout.strip().split('\n')
        tracked_files = [f for f in tracked_files if f]  # Remove empty strings
        
        # Filter files - only skip pip/conda related files
        included_files = []
        for file_path in tracked_files:
            # Skip if file doesn't exist
            if not os.path.exists(os.path.join(repo_root, file_path)):
                continue
            
            # Skip directories (git ls-files lists files, but be safe)
            if os.path.isdir(os.path.join(repo_root, file_path)):
                continue
            
            # Skip pip/conda related files
            skip = False
            for pattern in SKIP_PATTERNS:
                if re.search(pattern, file_path):
                    skip = True
                    break
            if skip:
                continue
            
            included_files.append(file_path)
        
        # Group files by directory for data_files format
        # Format: [('share/bayesed3/bin/linux', ['bin/linux/file1', ...]), ...]
        if included_files:
            file_map = {}
            for file_path in included_files:
                # Get relative directory from repo root
                rel_dir = os.path.dirname(file_path) if os.path.dirname(file_path) else '.'
                if rel_dir not in file_map:
                    file_map[rel_dir] = []
                file_map[rel_dir].append(file_path)
            
            # Convert to data_files format: (install_dir, [source_files])
            # Source files must be relative paths (relative to setup.py directory)
            for rel_dir, files in file_map.items():
                if rel_dir == '.':
                    install_dir = 'share/bayesed3'
                else:
                    install_dir = os.path.join('share', 'bayesed3', rel_dir)
                # Keep source file paths as relative (they're already relative to repo root)
                data_files.append((install_dir, files))
                
    except (subprocess.CalledProcessError, FileNotFoundError):
        # If git is not available or not in a git repo, skip data files
        # This allows setup.py to work even without git
        print("Warning: git not available or not in git repository. Data files will not be included.")
        print("Install from a git repository or use conda to get all data files.")
    
    return data_files

setup(
    name="bayesed3",
    version=get_version(),
    description="Bayesian SED synthesis and analysis of galaxies and AGNs",
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    author="Yunkun Han",
    author_email="hanyk@ynao.ac.cn",
    url="https://github.com/hanyk/BayeSED3",
    license="MIT",
    packages=find_packages(),
    python_requires=">=3.7",
    include_package_data=False,  # Don't automatically include files - use data_files instead
    data_files=get_data_files(),
    install_requires=[
        "numpy>=1.20.0",
        "h5py>=3.1.0",
        "astropy>=4.2",
        "matplotlib>=3.3.0",
        "GetDist>=1.3.0",
        "requests>=2.25.0",
        "tqdm>=4.60.0",
        "Pillow>=8.0.0",
        "pyperclip>=1.8.0",
        "psutil>=5.8.0",
    ],
    # Note: OpenMPI 4.1.6 is required but not a Python package.
    # It must be installed separately:
    # - Via conda: conda install openmpi=4.1.6
    # - Via system package manager (apt, brew, etc.)
    # - Or compiled from source (see README.md)
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Astronomy",
    ],
    entry_points={
        'console_scripts': [
            'BayeSED3_GUI=bayesed_gui:main',
        ],
    },
    py_modules=['bayesed_gui'],
)

