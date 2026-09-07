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
from .results.utils import list_catalog_names, list_model_configs

INSPECT_TOKENS = frozenset({'results', 'plot', 'priors'})
RESULTS_VERBS = (
    'summary', 'evidence', 'par_names', 'objects', 'par_values',
    'catalogs', 'model_configs',
)
_DISCOVERY_VERBS = frozenset({'catalogs', 'model_configs'})
PLOT_VERBS = ('bestfit', 'posterior')
_RESULTS_VERB_SET = frozenset(RESULTS_VERBS)
_PLOT_VERB_SET = frozenset(PLOT_VERBS)
_IPRIOR_NAME = re.compile(r'.*\.iprior(\.\d+)?$')
_PAR_VALUES_FLAGS = {'--cat': 'cat', '--obj': 'obj', '--par': 'par'}


def inspect_usage(prog='bayesed3'):
    return (
        f"Usage: {prog} results <outdir> <cmd> [<cmd>...] [catalog] [object]\n"
        f"       {prog} results <outdir> catalogs\n"
        f"       {prog} results <outdir> model_configs [catalog]\n"
        f"       {prog} results <outdir> par_values --par PATTERN [--cat catalog] [--obj object]\n"
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
        f"       {prog} results <outdir> catalogs\n"
        f"       {prog} results <outdir> model_configs [catalog]\n"
        f"       {prog} results <outdir> par_values --par PATTERN [--cat catalog] [--obj object]\n"
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


def loose_match_mode(pattern):
    """``regex`` when ``pattern`` compiles; otherwise ``contains``."""
    try:
        re.compile(pattern)
    except re.error:
        return 'contains'
    return 'regex'


def parse_par_values(tokens):
    """
    Parse ``<outdir> par_values [--cat C] [--obj O] --par P``.

    ``par_values`` is the only command. No leftover positionals.
    """
    if not tokens:
        raise ValueError('missing outdir')
    outdir = tokens[0]
    rest = tokens[1:]
    if not rest or rest[0] != 'par_values':
        if 'par_values' in rest:
            raise ValueError('par_values cannot be combined with other commands')
        raise ValueError('missing command')
    flags = {'cat': None, 'obj': None, 'par': None}
    i = 1
    while i < len(rest):
        tok = rest[i]
        key = _PAR_VALUES_FLAGS.get(tok)
        if key is not None:
            if i + 1 >= len(rest):
                raise ValueError(f'{tok} requires a value')
            flags[key] = rest[i + 1]
            i += 2
            continue
        if tok.startswith('-'):
            raise ValueError(f'unknown option {tok}')
        raise ValueError(f"unexpected argument '{tok}'")
    if flags['par'] is None:
        raise ValueError('--par is required')
    return outdir, flags['cat'], flags['obj'], flags['par']


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


def _catalog_for_discovery(outdir, names):
    catalogs = list_catalog_names(outdir)
    if names:
        catalog = names[0]
        if catalog not in catalogs:
            raise ValueError(
                f"Unknown catalog '{catalog}'. Available catalogs: {catalogs}"
            )
        return catalog
    if len(catalogs) == 1:
        return catalogs[0]
    if not catalogs:
        raise ValueError(f'No catalogs in {outdir}')
    raise ValueError(
        'Multiple catalogs found. Specify a catalog after the command, '
        f"e.g. model_configs {catalogs[0]}. Available catalogs: {catalogs}"
    )


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


def _run_par_values(rest, prog):
    try:
        outdir, cat, obj, par = parse_par_values(rest)
    except ValueError as e:
        sys.stderr.write(f"Error: {e}\n")
        sys.stderr.write(results_usage(prog))
        return 1
    try:
        if cat is not None:
            catalogs = list_catalog_names(outdir)
            if catalogs and cat not in catalogs:
                raise ValueError(
                    f"Unknown catalog '{cat}'. Available catalogs: {catalogs}"
                )
            results = BayeSEDResults(outdir, catalog_name=cat)
        else:
            results = BayeSEDResults(outdir)
        kwargs = {'param_match_mode': loose_match_mode(par)}
        if obj is not None:
            kwargs['object_ids'] = obj
            kwargs['object_match_mode'] = loose_match_mode(obj)
        print(results.get_parameter_values(par, **kwargs))
    except (ValueError, FileNotFoundError) as e:
        sys.stderr.write(f"Error: {e}\n")
        return 1
    return 0


def _run_discovery(outdir, cmds, names, prog):
    try:
        if len(cmds) != 1:
            raise ValueError(
                'catalogs and model_configs cannot be combined with other commands'
            )
        cmd = cmds[0]
        if cmd == 'catalogs':
            if names:
                raise ValueError(f"unexpected argument '{names[0]}'")
            for item in list_catalog_names(outdir):
                print(item)
            return 0
        if len(names) > 1:
            raise ValueError(f"unexpected argument '{names[1]}'")
        catalog = _catalog_for_discovery(outdir, names)
        for item in list_model_configs(outdir, catalog):
            print(item)
    except ValueError as e:
        sys.stderr.write(f"Error: {e}\n")
        sys.stderr.write(results_usage(prog))
        return 1
    return 0


def _print_par_names(results):
    print('Free parameters:')
    for item in results.get_free_parameters():
        print(f"  {item}")
    print('Derived parameters:')
    for item in results.get_derived_parameters():
        print(f"  {item}")


def _results_has_par_values(rest):
    for tok in rest[1:]:
        if tok.startswith('-'):
            return False
        if tok == 'par_values':
            return True
        if tok not in _RESULTS_VERB_SET:
            return False
    return False


def _run_results_plot(kind, rest, prog):
    if kind == 'results' and _results_has_par_values(rest):
        return _run_par_values(rest, prog)
    verbs = _RESULTS_VERB_SET if kind == 'results' else _PLOT_VERB_SET
    usage = results_usage if kind == 'results' else plot_usage
    try:
        outdir, cmds, names = parse_outdir_verbs_name(rest, verbs)
    except ValueError as e:
        sys.stderr.write(f"Error: {e}\n")
        sys.stderr.write(usage(prog))
        return 1
    if kind == 'results' and any(cmd in _DISCOVERY_VERBS for cmd in cmds):
        return _run_discovery(outdir, cmds, names, prog)
    try:
        results = _load_results(outdir, names)
        if kind == 'results':
            for cmd in cmds:
                if cmd == 'summary':
                    results.print_summary()
                elif cmd == 'evidence':
                    _print_evidence(results)
                elif cmd == 'par_names':
                    _print_par_names(results)
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
