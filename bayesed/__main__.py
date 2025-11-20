"""
BayeSED3 CLI Entry Point

This module provides command-line access to BayeSED3.
Use: python -m bayesed [--mpi MODE] [--np NPROC] [--help] [BayeSED arguments...]

For programmatic access, use:
    from bayesed import BayeSEDInterface, BayeSEDParams
"""

import sys

from .core import BayeSEDInterface


def main():
    """Main CLI entry point for BayeSED3."""
    if len(sys.argv) < 2:
        print("Usage: python -m bayesed [--mpi MODE] [--np NPROC] [--help] [BayeSED arguments...]")
        print("\nOptions:")
        print("  --mpi MODE    MPI mode: '1' (parallelize within objects) or 'n' (parallelize across objects)")
        print("  --np NPROC    Number of MPI processes (default: auto-detect)")
        print("  --help        Show BayeSED help")
        print("\nAll other arguments are passed directly to the BayeSED binary.")
        print("For help with BayeSED arguments, use: python -m bayesed --help")
        sys.exit(1)

    mpi_mode = '1'  # Default to bayesed_mn_1
    bayesed_args = []
    i = 1
    num_processes = None

    # Parse --mpi and --np arguments
    while i < len(sys.argv):
        if sys.argv[i] == "--mpi":
            if i + 1 < len(sys.argv):
                mpi_mode = sys.argv[i + 1]
                if mpi_mode not in ['1', 'n']:
                    print(f"Warning: Invalid MPI mode '{mpi_mode}'. Using default '1'.")
                    mpi_mode = '1'
                i += 2
            else:
                print("Error: --mpi requires a mode ('1' or 'n')")
                sys.exit(1)
        elif sys.argv[i] == "--np":
            if i + 1 < len(sys.argv):
                try:
                    num_processes = int(sys.argv[i + 1])
                    i += 2
                except ValueError:
                    print(f"Error: --np requires an integer, got '{sys.argv[i + 1]}'")
                    sys.exit(1)
            else:
                print("Error: --np requires a number")
                sys.exit(1)
        else:
            bayesed_args.append(sys.argv[i])
            i += 1

    # Create interface
    bayesed = BayeSEDInterface(mpi_mode=mpi_mode, np=num_processes)

    # Handle help
    if '--help' in bayesed_args or '-h' in bayesed_args:
        # For help, use 1 process
        bayesed.num_processes = 1
        bayesed.run(bayesed_args)
    else:
        # Run with all arguments passed to binary
        bayesed.run(bayesed_args)


if __name__ == "__main__":
    main()

