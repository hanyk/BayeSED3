"""
Platform-independent Launcher for the compiled BayeSED binary (ADR-0007, ADR-0009).

Invoked as ``bayesed3`` / ``python -m bayesed``. First-token ``results`` /
``plot`` / ``priors`` present existing products. Otherwise owns only
``--mpi`` and ``--np``; remaining argv is forwarded to the binary via
``runtime``.
"""

import subprocess
import sys

from .inspect_cli import INSPECT_TOKENS, inspect_usage_lines, run_inspect
from .runtime import (
    absolutize_cli_path_args,
    build_mpi_command,
    resolve_binary_cwd,
    resolve_executable,
    resolve_mpirun,
    resolve_num_processes,
    resolve_system_info,
)


def usage_text(prog='bayesed3'):
    inspect_block = ''.join(f"       {line}\n" for line in inspect_usage_lines(prog))
    return (
        f"Usage: {prog} [--mpi MODE] [--np NPROC] [BayeSED arguments...]\n"
        + inspect_block
        + "\n"
        "Options:\n"
        "  --mpi MODE    MPI mode: '1' (within objects) or 'n' (across objects)\n"
        "  --np NPROC    Number of MPI processes (default: auto-detect)\n"
        "\n"
        "Inspect existing Results or Priors: results, plot, priors.\n"
        "All other arguments are passed directly to the BayeSED binary.\n"
        f"For binary help: {prog} --help\n"
    )


def parse_launcher_argv(argv):
    """
    Parse Launcher-owned flags from ``argv`` (no program name).

    Returns
    -------
    tuple
        ``(mpi_mode, num_processes, binary_args)`` where ``mpi_mode`` is
        ``'mn_1'`` or ``'mn_n'``, and ``num_processes`` may be ``None``.

    Raises
    ------
    ValueError
        On invalid ``--mpi`` / ``--np`` usage.
    """
    mpi_digit = '1'
    num_processes = None
    binary_args = []
    i = 0
    while i < len(argv):
        if argv[i] == '--mpi':
            if i + 1 >= len(argv):
                raise ValueError("--mpi requires a mode ('1' or 'n')")
            mode = argv[i + 1]
            if mode not in ('1', 'n'):
                raise ValueError(
                    f"Invalid MPI mode '{mode}'. Use '1' or 'n'."
                )
            mpi_digit = mode
            i += 2
        elif argv[i] == '--np':
            if i + 1 >= len(argv):
                raise ValueError('--np requires a number')
            try:
                num_processes = int(argv[i + 1])
            except ValueError as exc:
                raise ValueError(
                    f"--np requires an integer, got '{argv[i + 1]}'"
                ) from exc
            i += 2
        else:
            binary_args.append(argv[i])
            i += 1
    return f'mn_{mpi_digit}', num_processes, binary_args


def run_launcher(argv=None, prog='bayesed3'):
    """
    Run the Launcher. Returns a process exit code (0 = success).
    """
    if argv is None:
        argv = sys.argv[1:]

    if not argv:
        sys.stderr.write(usage_text(prog))
        return 1

    if argv[0] in INSPECT_TOKENS:
        return run_inspect(argv, prog=prog)

    try:
        mpi_mode, np_override, binary_args = parse_launcher_argv(argv)
    except ValueError as e:
        sys.stderr.write(f"Error: {e}\n")
        return 1

    if not binary_args:
        sys.stderr.write(usage_text(prog))
        return 1

    os_name, arch = resolve_system_info()
    mpi_cmd = resolve_mpirun()
    num_processes = resolve_num_processes(np_override)
    if '-h' in binary_args or '--help' in binary_args:
        num_processes = 1
    executable = resolve_executable(mpi_mode, os_name, arch)

    binary_args = absolutize_cli_path_args(binary_args)
    cwd = resolve_binary_cwd()
    cmd = build_mpi_command(mpi_cmd, num_processes, executable, binary_args)

    popen_kwargs = {
        'stdout': None,
        'stderr': None,
    }
    if cwd is not None:
        popen_kwargs['cwd'] = cwd

    process = subprocess.Popen(cmd, **popen_kwargs)
    return process.wait()


def main(argv=None):
    """Console-script entry: ``bayesed3`` / ``python -m bayesed``."""
    raise SystemExit(run_launcher(argv, prog='bayesed3'))
