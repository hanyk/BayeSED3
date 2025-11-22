"""
Setup script for BayeSED3.

This setup.py is maintained for backward compatibility and conda packaging.
For modern Python packaging, see pyproject.toml.
For development installation, use: pip install -e .
"""

from setuptools import setup, find_packages
import os

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

