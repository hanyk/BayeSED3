"""
Runtime bootstrap for BayeSEDInterface.

Internal adapter: OpenMPI / process count / binary path (ADR-0005).
Not part of the public bayesed package export; BayeSEDInterface calls
resolve_* helpers at init and when MPI mode changes.
"""

import multiprocessing
import os
import platform
import re
import shutil
import subprocess
import sys
import tarfile

import requests
from tqdm import tqdm

from .utils import _get_resource_path

OPENMPI_VERSION = "4.1.6"
_64BIT_ARCHS = frozenset({"x86_64", "amd64", "arm64", "aarch64"})
_ARCH_DIR_MAP = {"x86_64": "x86", "arm64": "arm", "aarch64": "arm"}


def resolve_system_info():
    """
    Detect OS and machine architecture; require 64-bit.

    Returns
    -------
    tuple[str, str]
        ``(os_name, arch)`` — lowercased ``platform.system`` / ``platform.machine``.
    """
    os_name = platform.system().lower()
    arch = platform.machine().lower()

    if arch not in _64BIT_ARCHS:
        raise ValueError(
            f"Unsupported architecture: {arch}. "
            f"BayeSED3 requires a 64-bit system (x86_64 or arm64/aarch64). "
            f"Detected: {arch}"
        )
    return os_name, arch


def resolve_num_processes(np=None):
    """
    Resolve process count from an explicit override or the job scheduler / host.

    Priority when ``np`` is None:
    1. PBS_NCPUS
    2. PBS_NODEFILE (line count)
    3. SLURM_CPUS_PER_TASK or SLURM_NTASKS
    4. NSLOTS (SGE)
    5. multiprocessing.cpu_count()
    """
    if np is not None:
        return np

    pbs_ncpus = os.environ.get('PBS_NCPUS')
    if pbs_ncpus:
        try:
            return int(pbs_ncpus)
        except ValueError:
            pass

    pbs_nodefile = os.environ.get('PBS_NODEFILE')
    if pbs_nodefile and os.path.exists(pbs_nodefile):
        try:
            with open(pbs_nodefile, 'r') as f:
                lines = [line.strip() for line in f if line.strip()]
                return len(lines)
        except Exception:
            pass

    slurm_cpus = os.environ.get('SLURM_CPUS_PER_TASK') or os.environ.get('SLURM_NTASKS')
    if slurm_cpus:
        try:
            return int(slurm_cpus)
        except ValueError:
            pass

    sge_nslots = os.environ.get('NSLOTS')
    if sge_nslots:
        try:
            return int(sge_nslots)
        except ValueError:
            pass

    return multiprocessing.cpu_count()


def _parse_openmpi_version(version_stdout):
    """Extract x.y.z from ``mpirun --version`` stdout; empty string if missing."""
    try:
        installed_version = version_stdout.split()[3]
    except IndexError:
        return '', ''
    m = re.match(r'\d+\.\d+\.\d+', installed_version)
    major_minor = m.group(0) if m else ''
    return installed_version, major_minor


def resolve_mpirun(openmpi_mirror=None):
    """
    Locate or install OpenMPI ``mpirun`` (conda → system → local → compile).

    Parameters
    ----------
    openmpi_mirror : str, optional
        Custom tarball URL when falling back to source compile.
    """
    openmpi_version = OPENMPI_VERSION
    openmpi_url = openmpi_mirror or (
        f"https://download.open-mpi.org/release/open-mpi/v4.1/"
        f"openmpi-{openmpi_version}.tar.gz"
    )
    openmpi_dir = f"openmpi-{openmpi_version}"
    openmpi_file = f"{openmpi_dir}.tar.gz"
    install_dir = os.path.abspath("openmpi")

    conda_prefix = os.environ.get('CONDA_PREFIX')
    if conda_prefix:
        conda_mpirun = os.path.join(conda_prefix, 'bin', 'mpirun')
        if os.path.exists(conda_mpirun):
            try:
                result = subprocess.run(
                    [conda_mpirun, "--version"], capture_output=True, text=True
                )
                installed_version, installed_major_minor = _parse_openmpi_version(
                    result.stdout
                )
                if installed_major_minor == openmpi_version:
                    print(f"Using conda-installed OpenMPI {installed_version}")
                    return conda_mpirun
                print(
                    f"Conda has OpenMPI {installed_version}, "
                    f"but we need {openmpi_version}"
                )
            except Exception as e:
                print(f"Error checking conda OpenMPI version: {e}")

    system_mpirun = shutil.which("mpirun")
    if system_mpirun:
        try:
            result = subprocess.run(
                [system_mpirun, "--version"], capture_output=True, text=True
            )
            installed_version, installed_major_minor = _parse_openmpi_version(
                result.stdout
            )
            if installed_major_minor == openmpi_version:
                print(f"Using system-installed OpenMPI {installed_version}")
                return system_mpirun
            if not os.path.exists(install_dir):
                print(
                    f"System has OpenMPI {installed_version}, "
                    f"but we need {openmpi_version}"
                )
        except Exception as e:
            print(f"Error checking OpenMPI version: {e}")

    local_mpirun = os.path.join(install_dir, "bin", "mpirun")
    if os.path.exists(local_mpirun):
        print(f"Using locally compiled OpenMPI from {install_dir}")
        os.environ["PATH"] = (
            f"{os.path.dirname(local_mpirun)}:{os.environ.get('PATH', '')}"
        )
        os.environ["LD_LIBRARY_PATH"] = (
            f"{os.path.join(install_dir, 'lib')}:"
            f"{os.environ.get('LD_LIBRARY_PATH', '')}"
        )
        return local_mpirun

    if not os.path.exists(install_dir):
        conda_available = shutil.which("conda") is not None

        print("\n" + "=" * 70)
        print("OpenMPI 4.1.6 not found!")
        print("=" * 70)

        is_interactive = sys.stdin.isatty() and sys.stdout.isatty()

        if conda_available and is_interactive:
            print(
                "\n⚠️  Conda detected! BayeSED3 can automatically install "
                "OpenMPI via conda."
            )
            print("   This is the fastest and easiest method.")
            print("\n   Command: conda install openmpi=4.1.6")

            try:
                response = input(
                    "\nDo you want to install OpenMPI via conda now? [Y/n]: "
                ).strip().lower()
                install_via_conda = response not in ['n', 'no']
            except (EOFError, KeyboardInterrupt):
                print("\n\nCancelled by user.")
                install_via_conda = False

            if install_via_conda:
                print("\n⏳ Installing OpenMPI 4.1.6 via conda...")
                try:
                    result = subprocess.run(
                        ["conda", "install", "-y", "openmpi=4.1.6"],
                        capture_output=True,
                        text=True,
                        check=False,
                    )

                    if result.returncode == 0:
                        print("✅ OpenMPI 4.1.6 successfully installed via conda!")
                        conda_prefix = os.environ.get('CONDA_PREFIX')
                        if conda_prefix:
                            conda_mpirun = os.path.join(
                                conda_prefix, 'bin', 'mpirun'
                            )
                            if os.path.exists(conda_mpirun):
                                try:
                                    result = subprocess.run(
                                        [conda_mpirun, "--version"],
                                        capture_output=True,
                                        text=True,
                                    )
                                    installed_version = result.stdout.split()[3]
                                    if installed_version == openmpi_version:
                                        print(
                                            f"✅ Using conda-installed OpenMPI "
                                            f"{installed_version}"
                                        )
                                        return conda_mpirun
                                except Exception as e:
                                    print(
                                        f"⚠️  Error verifying OpenMPI version: {e}"
                                    )

                        print(
                            "\n⚠️  OpenMPI was installed but not immediately "
                            "detected."
                        )
                        print(
                            "   Please restart your Python session and try again."
                        )
                        raise FileNotFoundError(
                            f"\nOpenMPI {openmpi_version} was installed via "
                            f"conda, but needs a Python session restart.\n"
                            f"Please restart Python and try again."
                        )
                    print(
                        f"⚠️  Conda installation failed with return code "
                        f"{result.returncode}"
                    )
                    if result.stderr:
                        print(f"Error: {result.stderr}")
                    print("\nFalling back to other installation options...\n")
                except FileNotFoundError:
                    raise
                except Exception as e:
                    print(f"⚠️  Error running conda install: {e}")
                    print("\nFalling back to other installation options...\n")

        if conda_available:
            print("\n⚠️  Manual installation options:")
            print("   1. Conda: conda install openmpi=4.1.6")
            print("   2. System package manager (see below)")
        else:
            print("\n⚠️  RECOMMENDED: Install OpenMPI via system package manager:")
            print("   macOS:    brew install openmpi")
            print("   Ubuntu:   sudo apt-get install openmpi-bin libopenmpi-dev")
            print("   Fedora:   sudo dnf install openmpi openmpi-devel")

        print("\n" + "-" * 70)
        print("Alternatively, BayeSED3 can automatically download and compile")
        print(f"OpenMPI {openmpi_version} from source (takes 10-30 minutes).")
        print("-" * 70)

        if is_interactive:
            try:
                response = input(
                    "\nDo you want to auto-compile OpenMPI now? [y/N]: "
                ).strip().lower()
                auto_compile = response in ['y', 'yes']
            except (EOFError, KeyboardInterrupt):
                print("\n\nCancelled by user.")
                auto_compile = False
        else:
            print("\n⚠️  Non-interactive environment detected.")
            print("   Auto-compilation requires user interaction.")
            print(
                "   Please install OpenMPI manually using one of the methods "
                "above."
            )
            auto_compile = False

        if not auto_compile:
            raise FileNotFoundError(
                f"\nOpenMPI {openmpi_version} is required but not found.\n"
                f"Please install it using one of these methods:\n"
                f"  1. Conda (recommended): conda install openmpi=4.1.6\n"
                f"  2. System package manager (see above)\n"
                f"  3. Manual compilation (see README.md)\n"
                f"\nAfter installation, restart your Python session."
            )

        print(f"\n⏳ Starting automatic compilation of OpenMPI {openmpi_version}...")
        print("   This may take 10-30 minutes depending on your system.\n")

        if os.path.exists(openmpi_file):
            print(
                f"OpenMPI {openmpi_version} tarball already exists. "
                f"Checking if it's complete..."
            )
            try:
                with tarfile.open(openmpi_file, 'r:gz') as tar:
                    tar.getmembers()
                print("Existing tarball is complete. Skipping download.")
            except Exception as e:
                print(
                    f"Existing tarball is incomplete or corrupted. "
                    f"Re-downloading: {e}"
                )
                os.remove(openmpi_file)

        if not os.path.exists(openmpi_file):
            print(f"Downloading OpenMPI {openmpi_version}...")
            response = requests.get(openmpi_url, stream=True)
            total_size = int(response.headers.get('content-length', 0))

            with open(openmpi_file, 'wb') as file, tqdm(
                desc=openmpi_file,
                total=total_size,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
            ) as progress_bar:
                for data in response.iter_content(chunk_size=1024):
                    size = file.write(data)
                    progress_bar.update(size)

        print("Extracting OpenMPI...")
        with tarfile.open(openmpi_file, 'r:gz') as tar:
            tar.extractall()

        print("Compiling and installing OpenMPI...")
        os.chdir(openmpi_dir)
        subprocess.run(["./configure", f"--prefix={install_dir}"], check=True)
        subprocess.run(
            ["make", "-j", str(resolve_num_processes())], check=True
        )
        subprocess.run(["make", "install"], check=True)
        os.chdir("..")

        print("Cleaning up temporary files...")
        shutil.rmtree(openmpi_dir)

        mpirun_path = os.path.join(install_dir, "bin", "mpirun")
        if not os.path.exists(mpirun_path):
            raise FileNotFoundError(
                f"mpirun not found at {mpirun_path}. "
                f"OpenMPI installation may have failed."
            )

        os.environ["PATH"] = (
            f"{os.path.dirname(mpirun_path)}:{os.environ.get('PATH', '')}"
        )
        os.environ["LD_LIBRARY_PATH"] = (
            f"{os.path.join(install_dir, 'lib')}:"
            f"{os.environ.get('LD_LIBRARY_PATH', '')}"
        )

        print(
            f"\n✅ OpenMPI {openmpi_version} successfully compiled and installed!"
        )
        return mpirun_path

    raise FileNotFoundError(
        "Could not find OpenMPI 4.1.6. "
        "Checked conda installation, system installation, local compilation, "
        "and auto-compilation failed."
    )


def resolve_executable(mpi_mode, os_name, arch):
    """
    Resolve path to ``bayesed_{mpi_mode}`` for the host platform/arch.

    Parameters
    ----------
    mpi_mode : str
        e.g. ``'mn_1'`` or ``'mn_n'`` (as stored on ``BayeSEDInterface``).
    os_name : str
        Lowercased OS from ``resolve_system_info``.
    arch : str
        Lowercased machine arch from ``resolve_system_info``.
    """
    executable = f"bayesed_{mpi_mode}"
    if os_name == "linux" or (
        os_name == "windows" and "microsoft" in platform.uname().release.lower()
    ):
        platform_dir = "linux"
    elif os_name == "darwin":
        platform_dir = "mac"
    else:
        raise ValueError(f"Unsupported operating system: {os_name}")

    arch_dir = _ARCH_DIR_MAP.get(arch, "x86")
    relative_path = os.path.join("bin", platform_dir, arch_dir, executable)
    return _get_resource_path(relative_path)
