"""
Platform-independent Launcher for the compiled BayeSED binary (ADR-0007, ADR-0009).

Invoked as ``bayesed3`` / ``python -m bayesed``. First token selects the path:
``version`` / ``--version`` report Package version and Binary banner;
``results`` / ``plot`` / ``priors`` present existing products; ``run`` forwards
remaining argv to the binary via ``runtime`` (``--mpi`` / ``--np`` only after
``run``). Empty argv and top-level ``--help`` / ``-h`` print Launcher overview.
"""

import subprocess
import sys
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version

from .inspect_cli import (
    INSPECT_TOKENS,
    PLOT_VERBS,
    RESULTS_VERBS,
    run_inspect,
)
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
    results_cmds = '|'.join(RESULTS_VERBS)
    plot_cmds = '|'.join(PLOT_VERBS)
    return (
        f"{prog} - A Bayesian SED synthesis and analysis tool for galaxies "
        f"and AGNs\n"
        "\n"
        "Version:\n"
        f"  {prog} version | --version\n"
        "\n"
        "Inspect:\n"
        f"  {prog} results <outdir> [{results_cmds}]\n"
        f"  {prog} plot <outdir> [{plot_cmds}]\n"
        f"  {prog} priors list [dir]\n"
        "\n"
        "Run:\n"
        f"  {prog} run [--help] [--mpi 1|n] [--np N] [binary options…]\n"
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


def _forward_binary(binary_args, mpi_mode, np_override):
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


def _binary_help_banner():
    """
    Return ``(banner_line, error)`` from the compiled binary's ``--help``.

    The binary has no dedicated ``--version``; its identity line is the first
    ``BayeSED3 (...)`` Binary banner on help (includes Copyright). Runs the
    ``mn_1`` executable directly (no mpirun) so banner probing stays quiet
    and cheap.
    """
    try:
        os_name, arch = resolve_system_info()
        executable = resolve_executable('mn_1', os_name, arch)
        cwd = resolve_binary_cwd()
        completed = subprocess.run(
            [executable, '--help'],
            capture_output=True,
            text=True,
            cwd=cwd,
        )
    except (OSError, RuntimeError, ValueError, FileNotFoundError) as exc:
        return None, str(exc)

    text = (completed.stdout or '') + (completed.stderr or '')
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith('BayeSED3'):
            return stripped, None
    return None, 'could not parse Binary banner'


def _version_text():
    try:
        py_ver = package_version('bayesed3')
    except PackageNotFoundError as exc:
        py_ver = f'(unavailable: {exc})'
    lines = [f"python  {py_ver}"]
    banner, err = _binary_help_banner()
    if banner is not None:
        lines.append(f"binary  {banner}")
    else:
        lines.append(f"binary  (unavailable: {err})")
    return '\n'.join(lines) + '\n'


def _rewrite_as_run(prog, argv):
    quoted = ' '.join(argv)
    return (
        f"Error: unknown command '{argv[0]}'. To fit with the compiled program, use:\n"
        f"  {prog} run {quoted}\n"
        f"Launcher overview: {prog} --help\n"
    )


def run_launcher(argv=None, prog='bayesed3'):
    """
    Run the Launcher. Returns a process exit code (0 = success).
    """
    if argv is None:
        argv = sys.argv[1:]

    if not argv or argv[0] in ('--help', '-h'):
        sys.stderr.write(usage_text(prog))
        return 0

    if argv[0] in ('version', '--version'):
        sys.stdout.write(_version_text())
        return 0

    if argv[0] in INSPECT_TOKENS:
        return run_inspect(argv, prog=prog)

    if argv[0] == 'run':
        rest = argv[1:]
        if not rest or rest == ['--help'] or rest == ['-h']:
            rest = ['--help']
        try:
            mpi_mode, np_override, binary_args = parse_launcher_argv(rest)
        except ValueError as e:
            sys.stderr.write(f"Error: {e}\n")
            return 1
        if not binary_args:
            binary_args = ['--help']
        return _forward_binary(binary_args, mpi_mode, np_override)

    sys.stderr.write(_rewrite_as_run(prog, argv))
    return 1


def main(argv=None):
    """Console-script entry: ``bayesed3`` / ``python -m bayesed``."""
    raise SystemExit(run_launcher(argv, prog='bayesed3'))
