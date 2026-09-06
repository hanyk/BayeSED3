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
        f"Usage: {prog} results <outdir> <cmd> [<cmd>...] [catalog] [object]\n"
        f"       {prog} plot <outdir> <cmd> [<cmd>...] [catalog] [object]\n"
        f"       {prog} priors list [dir]\n"
        "\n"
        f"results commands: {', '.join(RESULTS_VERBS)}\n"
        f"plot commands: {', '.join(PLOT_VERBS)}\n"
        "priors commands: list\n"
    )


def results_usage(prog='bayesed3'):
    return (
        f"Usage: {prog} results <outdir> <cmd> [<cmd>...] [catalog] [object]\n"
        f"Commands: {', '.join(RESULTS_VERBS)}\n"
    )


def plot_usage(prog='bayesed3'):
    return (
        f"Usage: {prog} plot <outdir> <cmd> [<cmd>...] [catalog] [object]\n"
        f"Commands: {', '.join(PLOT_VERBS)}\n"
    )


def priors_usage(prog='bayesed3'):
    return (
        f"Usage: {prog} priors list [dir]\n"
        "Commands: list\n"
    )


def parse_outdir_verbs_name(tokens, verbs):
    """
    Parse ``<outdir> <cmd> [<cmd>...] [catalog] [object]``.

    ``verbs`` is an iterable of allowed command names.
    Trailing names are after all commands; at most two.
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
    names = []
    while i < len(tokens):
        if tokens[i] in verb_set:
            raise ValueError(
                f"name must come after commands, not before '{tokens[i]}'"
            )
        names.append(tokens[i])
        i += 1
    if len(names) > 2:
        raise ValueError('at most catalog and object after commands')
    if not cmds:
        raise ValueError('missing command')
    return outdir, cmds, names


def _load_results(outdir, names):
    catalogs = list_catalog_names(outdir)
    if not names:
        return BayeSEDResults(outdir)
    if len(names) == 1:
        if names[0] in catalogs:
            return BayeSEDResults(outdir, catalog_name=names[0])
        return BayeSEDResults(outdir, object_id=names[0])
    catalog, object_id = names
    if catalog not in catalogs:
        raise ValueError(
            f"Unknown catalog '{catalog}'. Available catalogs: {catalogs}"
        )
    return BayeSEDResults(outdir, catalog_name=catalog, object_id=object_id)


def _print_evidence(results):
    evidence = results.get_evidence()
    if isinstance(evidence, dict) and 'log_evidence' in evidence:
        err = evidence.get('log_evidence_error')
        print(f"log_evidence: {evidence['log_evidence']} ± {err}")
        return
    print(evidence)


def format_object_listing(object_ids):
    """
    Lines listing object IDs. Empty when there is nothing to choose
    (zero or one). More than ten: first five, ``...``, last five.
    """
    n = len(object_ids)
    if n <= 1:
        return []
    lines = [f'Objects ({n}):']
    if n <= 10:
        for item in object_ids:
            lines.append(f'  {item}')
        return lines
    for item in object_ids[:5]:
        lines.append(f'  {item}')
    lines.append('  ...')
    for item in object_ids[-5:]:
        lines.append(f'  {item}')
    return lines


def _announce_plot_objects(results):
    if results.object_id:
        return results.object_id
    object_ids = results.list_objects()
    for line in format_object_listing(object_ids):
        print(line)
    if not object_ids:
        return None
    first = object_ids[0]
    if len(object_ids) > 1:
        print(f'using first: {first}')
    return first


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
        outdir, cmds, names = parse_outdir_verbs_name(rest, verbs)
    except ValueError as e:
        sys.stderr.write(f"Error: {e}\n")
        sys.stderr.write(usage(prog))
        return 1
    try:
        results = _load_results(outdir, names)
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
            plot_kw = {'show': True}
            chosen = _announce_plot_objects(results)
            if chosen:
                plot_kw['object_id'] = chosen
            for cmd in cmds:
                if cmd == 'bestfit':
                    results.plot_bestfit(**plot_kw)
                elif cmd == 'posterior':
                    results.plot_posterior_free(**plot_kw)
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
