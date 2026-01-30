"""
Minimal setup.py for data_files support.

This file only exists because pyproject.toml doesn't support data_files.
All other metadata (dependencies, version, etc.) is in pyproject.toml.

For modern pip (>=21.3), pyproject.toml is used for everything except data_files.
"""

from setuptools import setup
import os
import subprocess


def get_data_files():
    """
    Get data files to install to $PREFIX/share/bayesed3/.
    
    Uses git ls-files to include only git-tracked files.
    """
    data_files = []
    
    # Check if we're in a git repository
    try:
        # Get all git-tracked files
        result = subprocess.run(
            ['git', 'ls-files'],
            capture_output=True,
            text=True,
            check=True
        )
        tracked_files = result.stdout.strip().split('\n')
        
        # Filter files to include
        included_files = []
        for file_path in tracked_files:
            if not file_path or not os.path.isfile(file_path):
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
                
    except (subprocess.CalledProcessError, FileNotFoundError):
        # If git is not available, skip data files
        # Users should use conda or work from git repository
        print("Warning: git not available. Data files will not be installed.")
        print("For full installation, use: conda install bayesed3")
        print("Or install from git repository: pip install -e .")
    
    return data_files


# Minimal setup - all metadata comes from pyproject.toml
setup(
    data_files=get_data_files(),
)

