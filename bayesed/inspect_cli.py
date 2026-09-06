"""
Launcher inspect path (ADR-0009).

First-token ``results`` / ``plot`` / ``priors`` present existing products.
Not a public package export; ``run_launcher`` dispatches here.
"""

import glob
import os
import re
import sys

from .inference import SEDInference
from .prior_editor import PriorEditor
from .prior_manager import PriorManager
from .results import BayeSEDResults
from .results.utils import list_catalog_names

INSPECT_TOKENS = frozenset({'results', 'plot', 'priors'})
RESULTS_VERBS = ('summary', 'evidence', 'params', 'objects')
PLOT_VERBS = ('bestfit', 'posterior')
_RESULTS_VERB_SET = frozenset(RESULTS_VERBS)
_PLOT_VERB_SET = frozenset(PLOT_VERBS)
_IPRIOR_NAME = re.compile(r'.*\.iprior(\.\d+)?$')


def inspect_usage(prog='bayesed3'):
    return (
        f"Usage: {prog} results <outdir> <cmd> [<cmd>...] [name]\n"
        f"       {prog} plot <outdir> <cmd> [<cmd>...] [name]\n"
        f"       {prog} priors list [dir]\n"
        "\n"
        f"results commands: {', '.join(RESULTS_VERBS)}\n"
        f"plot commands: {', '.join(PLOT_VERBS)}\n"
        "priors commands: list\n"
    )


def results_usage(prog='bayesed3'):
    return (
        f"Usage: {prog} results <outdir> <cmd> [<cmd>...] [name]\n"
        f"Commands: {', '.join(RESULTS_VERBS)}\n"
    )


def plot_usage(prog='bayesed3'):
    return (
        f"Usage: {prog} plot <outdir> <cmd> [<cmd>...] [name]\n"
        f"Commands: {', '.join(PLOT_VERBS)}\n"
    )


def priors_usage(prog='bayesed3'):
    return (
        f"Usage: {prog} priors list [dir]\n"
        "Commands: list\n"
    )


def parse_outdir_verbs_name(tokens, verbs):
    """
    Parse ``<outdir> <cmd> [<cmd>...] [name]``.

    ``verbs`` is an iterable of allowed command names.
    """
    verb_set = frozenset(verbs)
    if not tokens:
        raise ValueError('missing outdir')
    outdir = tokens[0]
    cmds = []
    i = 1
    while i < len(tokens) and tokens[i] in verb_set:
        if tokens[i] not in cmds:
            cmds.append(tokens[i])
        i += 1
    name = None
    if i < len(tokens):
        name = tokens[i]
        i += 1
    if i < len(tokens):
        extra = tokens[i]
        if extra in verb_set:
            raise ValueError(
                f"name must come after commands, not before '{extra}'"
            )
        raise ValueError(f"unexpected argument '{extra}'")
    if not cmds:
        raise ValueError('missing command')
    return outdir, cmds, name


def _load_results(outdir, name):
    catalogs = list_catalog_names(outdir)
    if name is None:
        return BayeSEDResults(outdir)
    if name in catalogs:
        return BayeSEDResults(outdir, catalog_name=name)
    return BayeSEDResults(outdir, object_id=name)


def _print_evidence(results):
    evidence = results.get_evidence()
    if isinstance(evidence, dict) and 'log_evidence' in evidence:
        err = evidence.get('log_evidence_error')
        print(f"log_evidence: {evidence['log_evidence']} ± {err}")
        return
    print(evidence)


def _print_params(results):
    print('Free parameters:')
    for item in results.get_free_parameters():
        print(f"  {item}")
    print('Derived parameters:')
    for item in results.get_derived_parameters():
        print(f"  {item}")


def _run_results_plot(kind, rest, prog):
    verbs = _RESULTS_VERB_SET if kind == 'results' else _PLOT_VERB_SET
    usage = results_usage if kind == 'results' else plot_usage
    try:
        outdir, cmds, name = parse_outdir_verbs_name(rest, verbs)
    except ValueError as e:
        sys.stderr.write(f"Error: {e}\n")
        sys.stderr.write(usage(prog))
        return 1
    try:
        results = _load_results(outdir, name)
        if kind == 'results':
            for cmd in cmds:
                if cmd == 'summary':
                    results.print_summary()
                elif cmd == 'evidence':
                    _print_evidence(results)
                elif cmd == 'params':
                    _print_params(results)
                elif cmd == 'objects':
                    for item in results.list_objects():
                        print(item)
        else:
            for cmd in cmds:
                if cmd == 'bestfit':
                    results.plot_bestfit(show=False)
                elif cmd == 'posterior':
                    results.plot_posterior_free(show=False)
    except (ValueError, FileNotFoundError) as e:
        sys.stderr.write(f"Error: {e}\n")
        return 1
    return 0


def _iprior_files(directory):
    matches = glob.glob(os.path.join(directory, '*.iprior*'))
    return sorted(
        f for f in matches if _IPRIOR_NAME.match(os.path.basename(f))
    )


def _run_priors(rest, prog):
    if not rest or rest[0] != 'list':
        sys.stderr.write(priors_usage(prog))
        return 1
    if len(rest) == 1:
        SEDInference().list_prior_types()
        return 0
    if len(rest) > 2:
        sys.stderr.write(f"Error: unexpected argument '{rest[2]}'\n")
        sys.stderr.write(priors_usage(prog))
        return 1
    directory = rest[1]
    if not os.path.isdir(directory):
        sys.stderr.write(f"Error: not a directory: {directory}\n")
        return 1
    files = _iprior_files(directory)
    if not files:
        sys.stderr.write(f"Error: no .iprior files in {directory}\n")
        return 1
    manager = PriorManager(base_directory=directory)
    try:
        for path in files:
            manager.load_prior_file(os.path.basename(path))
        PriorEditor(manager).print_priors()
    except (ValueError, FileNotFoundError) as e:
        sys.stderr.write(f"Error: {e}\n")
        return 1
    return 0


def run_inspect(argv, prog='bayesed3'):
    """
    Run an inspect command. ``argv`` includes the first token
    (``results`` / ``plot`` / ``priors``). Returns an exit code.
    """
    if not argv or argv[0] not in INSPECT_TOKENS:
        sys.stderr.write(inspect_usage(prog))
        return 1
    token, rest = argv[0], argv[1:]
    if token == 'priors':
        return _run_priors(rest, prog)
    return _run_results_plot(token, rest, prog)
