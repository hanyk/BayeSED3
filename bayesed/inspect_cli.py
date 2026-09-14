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
_QUERY_VERBS = frozenset({'summary', 'evidence', 'par_names', 'objects'})
PLOT_VERBS = ('bestfit', 'posterior')
_PLOT_VERB_SET = frozenset(PLOT_VERBS)
_IPRIOR_NAME = re.compile(r'.*\.iprior(\.\d+)?$')
_PAR_VALUES_FLAGS = {
    '--cat': 'cat', '--obj': 'obj', '--par': 'par', '--cfg': 'cfg',
}
_RESULTS_QUERY_FLAGS = {'--cat': 'cat', '--obj': 'obj', '--cfg': 'cfg'}
_MODEL_CONFIGS_FLAGS = {'--cat': 'cat'}
_PLOT_BESTFIT_FLAGS = {'--cat': 'cat', '--obj': 'obj', '--cfg': 'cfg'}
_PLOT_POSTERIOR_FLAGS = {
    '--cat': 'cat', '--obj': 'obj', '--max': 'max', '--cfg': 'cfg',
}
_DERIVED_DEFAULT_MAX = 10


def results_usage_lines(prog='bayesed3'):
    return (
        f"{prog} results <outdir> <cmd> [<cmd>...] "
        f"[--cat catalog] [--cfg config] [--obj object]",
        f"{prog} results <outdir> catalogs",
        f"{prog} results <outdir> model_configs [--cat catalog]",
        f"{prog} results <outdir> par_values --par PATTERN "
        f"[--cat catalog] [--cfg config] [--obj object]",
    )


def plot_usage_lines(prog='bayesed3'):
    return (
        f"{prog} plot <outdir> bestfit [--cat catalog] [--obj object] [--cfg config]",
        f"{prog} plot <outdir> posterior free [--cat catalog] [--obj object] [--cfg config]",
        f"{prog} plot <outdir> posterior derived [--max N] [--cat catalog] [--obj object] [--cfg config]",
        f"{prog} plot <outdir> posterior --par NAME [--lab TEXT] ... [--cat catalog] [--obj object] [--cfg config]",
    )


def priors_usage_line(prog='bayesed3'):
    return f"{prog} priors list [dir]"


def _usage_block(lines):
    text = f"Usage: {lines[0]}\n"
    for line in lines[1:]:
        text += f"       {line}\n"
    return text


def inspect_usage_lines(prog='bayesed3'):
    return (
        results_usage_lines(prog)
        + plot_usage_lines(prog)
        + (priors_usage_line(prog),)
    )


def inspect_usage(prog='bayesed3'):
    return (
        _usage_block(inspect_usage_lines(prog))
        + "\n"
        f"results commands: {', '.join(RESULTS_VERBS)}\n"
        f"plot commands: {', '.join(PLOT_VERBS)}\n"
        "priors commands: list\n"
    )


def results_usage(prog='bayesed3'):
    return (
        _usage_block(results_usage_lines(prog))
        + f"Commands: {', '.join(RESULTS_VERBS)}\n"
    )


def plot_usage(prog='bayesed3'):
    return (
        _usage_block(plot_usage_lines(prog))
        + f"Commands: {', '.join(PLOT_VERBS)}\n"
    )


def priors_usage(prog='bayesed3'):
    return (
        f"Usage: {priors_usage_line(prog)}\n"
        "Commands: list\n"
    )


def _flag_value(tokens, i, flag_map):
    """Return ``(key, value, next_i)`` if ``tokens[i]`` is a mapped flag."""
    if i >= len(tokens):
        return None
    tok = tokens[i]
    key = flag_map.get(tok)
    if key is None:
        return None
    if i + 1 >= len(tokens):
        raise ValueError(f'{tok} requires a value')
    return key, tokens[i + 1], i + 2


def _parse_flag_pairs(tokens, start, flag_map):
    """Parse ``tokens[start:]`` as ``--flag value`` pairs. Last write wins."""
    values = {name: None for name in flag_map.values()}
    i = start
    while i < len(tokens):
        got = _flag_value(tokens, i, flag_map)
        if got is not None:
            key, value, i = got
            values[key] = value
            continue
        tok = tokens[i]
        if tok.startswith('-'):
            raise ValueError(f'unknown option {tok}')
        raise ValueError(f"unexpected argument '{tok}'")
    return values


def loose_match_mode(pattern):
    """``regex`` when ``pattern`` compiles; otherwise ``contains``."""
    try:
        re.compile(pattern)
    except re.error:
        return 'contains'
    return 'regex'


def parse_par_values(tokens):
    """
    Parse ``<outdir> par_values [--cat C] [--cfg CFG] [--obj O] --par P``.

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
    flags = _parse_flag_pairs(rest, 1, _PAR_VALUES_FLAGS)
    if flags['par'] is None:
        raise ValueError('--par is required')
    return outdir, flags['cat'], flags['obj'], flags['par'], flags['cfg']


def parse_results_query(tokens):
    """
    Parse ``<outdir> <query>... [--cat C] [--cfg CFG] [--obj O]``.

    Query verbs are ``summary`` / ``evidence`` / ``par_names`` / ``objects``.
    Flags only after all verbs. No leftover positionals.
    """
    if not tokens:
        raise ValueError('missing outdir')
    outdir = tokens[0]
    rest = tokens[1:]
    cmds = []
    i = 0
    while i < len(rest) and rest[i] in _QUERY_VERBS:
        if rest[i] not in cmds:
            cmds.append(rest[i])
        i += 1
    if not cmds:
        raise ValueError('missing command')
    flags = _parse_flag_pairs(rest, i, _RESULTS_QUERY_FLAGS)
    return outdir, cmds, flags['cat'], flags['obj'], flags['cfg']


def parse_model_configs(tokens):
    """
    Parse ``<outdir> model_configs [--cat C]``.

    ``model_configs`` is the only command. No leftover positionals.
    """
    if not tokens:
        raise ValueError('missing outdir')
    outdir = tokens[0]
    rest = tokens[1:]
    if not rest or rest[0] != 'model_configs':
        if 'model_configs' in rest:
            raise ValueError(
                'model_configs cannot be combined with other commands'
            )
        raise ValueError('missing command')
    flags = _parse_flag_pairs(rest, 1, _MODEL_CONFIGS_FLAGS)
    return outdir, flags['cat']


def parse_catalogs(tokens):
    """Parse ``<outdir> catalogs``. No flags or leftover positionals."""
    if not tokens:
        raise ValueError('missing outdir')
    outdir = tokens[0]
    rest = tokens[1:]
    if not rest or rest[0] != 'catalogs':
        if 'catalogs' in rest:
            raise ValueError('catalogs cannot be combined with other commands')
        raise ValueError('missing command')
    if len(rest) > 1:
        raise ValueError(f"unexpected argument '{rest[1]}'")
    return outdir


def parse_plot_bestfit(tokens):
    """
    Parse ``<outdir> bestfit [--cat C] [--obj O]``.

    ``bestfit`` is the only command. No leftover positionals.
    """
    if not tokens:
        raise ValueError('missing outdir')
    outdir = tokens[0]
    rest = tokens[1:]
    if not rest or rest[0] != 'bestfit':
        if 'bestfit' in rest:
            raise ValueError('bestfit cannot be combined with other commands')
        raise ValueError('missing command')
    flags = _parse_flag_pairs(rest, 1, _PLOT_BESTFIT_FLAGS)
    return outdir, flags['cat'], flags['obj'], flags['cfg']


def parse_plot_posterior(tokens):
    """
    Parse ``<outdir> posterior (free | derived | --par NAME [--lab TEXT] ...)``.

    ``posterior`` is the only command. Catalog, object, and model config
    via ``--cat`` / ``--obj`` / ``--cfg``. ``--lab`` must immediately
    follow the matching ``--par``.
    """
    if not tokens:
        raise ValueError('missing outdir')
    outdir = tokens[0]
    rest = tokens[1:]
    if not rest or rest[0] != 'posterior':
        if 'posterior' in rest:
            raise ValueError('posterior cannot be combined with other commands')
        raise ValueError('missing command')
    kind = None
    cat = None
    obj = None
    cfg = None
    max_n = None
    pars = []
    labs = []
    i = 1
    while i < len(rest):
        tok = rest[i]
        if tok in ('free', 'derived'):
            if kind is not None:
                raise ValueError('free and derived cannot be combined')
            kind = tok
            i += 1
            continue
        got = _flag_value(rest, i, _PLOT_POSTERIOR_FLAGS)
        if got is not None:
            key, value, i = got
            if key == 'max':
                try:
                    max_n = int(value)
                except ValueError:
                    raise ValueError('--max requires a positive integer') from None
                if max_n < 1:
                    raise ValueError('--max requires a positive integer')
            elif key == 'cat':
                cat = value
            elif key == 'obj':
                obj = value
            else:
                cfg = value
            continue
        if tok == '--par':
            if i + 1 >= len(rest):
                raise ValueError('--par requires a value')
            pars.append(rest[i + 1])
            i += 2
            if i < len(rest) and rest[i] == '--lab':
                if i + 1 >= len(rest):
                    raise ValueError('--lab requires a value')
                labs.append(rest[i + 1])
                i += 2
            else:
                labs.append(None)
            continue
        if tok == '--lab':
            raise ValueError('--lab must follow --par')
        if tok.startswith('-'):
            raise ValueError(f'unknown option {tok}')
        raise ValueError(f"unexpected argument '{tok}'")
    has_lab = any(label is not None for label in labs)
    if has_lab and any(label is None for label in labs):
        raise ValueError('--par and --lab must alternate')
    if kind is not None and pars:
        raise ValueError(f'{kind} cannot be combined with --par')
    if kind is None and not pars:
        raise ValueError('specify free, derived, or --par')
    if max_n is not None and kind != 'derived':
        raise ValueError('--max is only valid with derived')
    return outdir, kind, cat, obj, pars, labs if has_lab else None, max_n, cfg


def _catalog_for_discovery(outdir, cat):
    catalogs = list_catalog_names(outdir)
    if cat is not None:
        if cat not in catalogs:
            raise ValueError(
                f"Unknown catalog '{cat}'. Available catalogs: {catalogs}"
            )
        return cat
    if len(catalogs) == 1:
        return catalogs[0]
    if not catalogs:
        raise ValueError(f'No catalogs in {outdir}')
    raise ValueError(
        'Multiple catalogs found. Specify a catalog with --cat, '
        f"e.g. model_configs --cat {catalogs[0]}. Available catalogs: {catalogs}"
    )


def _load_results_flags(outdir, cat, obj, cfg=None):
    kwargs = {}
    if cat is not None:
        catalogs = list_catalog_names(outdir)
        if catalogs and cat not in catalogs:
            raise ValueError(
                f"Unknown catalog '{cat}'. Available catalogs: {catalogs}"
            )
        kwargs['catalog_name'] = cat
    if cfg is not None:
        kwargs['model_config'] = cfg
    if obj is not None:
        kwargs['object_id'] = obj
    return BayeSEDResults(outdir, **kwargs)


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
        outdir, cat, obj, par, cfg = parse_par_values(rest)
    except ValueError as e:
        sys.stderr.write(f"Error: {e}\n")
        sys.stderr.write(results_usage(prog))
        return 1
    try:
        results = _load_results_flags(outdir, cat, None, cfg)
        kwargs = {'param_match_mode': loose_match_mode(par)}
        if obj is not None:
            kwargs['object_ids'] = obj
            kwargs['object_match_mode'] = loose_match_mode(obj)
        print(results.get_parameter_values(par, **kwargs))
    except (ValueError, FileNotFoundError) as e:
        sys.stderr.write(f"Error: {e}\n")
        return 1
    return 0


def _run_catalogs(rest, prog):
    try:
        outdir = parse_catalogs(rest)
    except ValueError as e:
        sys.stderr.write(f"Error: {e}\n")
        sys.stderr.write(results_usage(prog))
        return 1
    for item in list_catalog_names(outdir):
        print(item)
    return 0


def _run_model_configs(rest, prog):
    try:
        outdir, cat = parse_model_configs(rest)
        catalog = _catalog_for_discovery(outdir, cat)
        for item in list_model_configs(outdir, catalog):
            print(item)
    except ValueError as e:
        sys.stderr.write(f"Error: {e}\n")
        sys.stderr.write(results_usage(prog))
        return 1
    return 0


def _run_results_query(rest, prog):
    try:
        outdir, cmds, cat, obj, cfg = parse_results_query(rest)
    except ValueError as e:
        sys.stderr.write(f"Error: {e}\n")
        sys.stderr.write(results_usage(prog))
        return 1
    try:
        results = _load_results_flags(outdir, cat, obj, cfg)
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
    except (ValueError, FileNotFoundError) as e:
        sys.stderr.write(f"Error: {e}\n")
        return 1
    return 0


def _print_par_names(results):
    print('Free parameters:')
    for item in results.get_free_parameters():
        print(f"  {item}")
    print('Derived parameters:')
    for item in results.get_derived_parameters():
        print(f"  {item}")


def _first_results_verb(rest):
    if len(rest) < 2:
        return None
    for tok in rest[1:]:
        if tok.startswith('-'):
            return None
        return tok
    return None


def _results_has_par_values(rest):
    return _first_results_verb(rest) == 'par_values'


def _results_has_catalogs(rest):
    return _first_results_verb(rest) == 'catalogs'


def _results_has_model_configs(rest):
    return _first_results_verb(rest) == 'model_configs'


def _plot_has_posterior(rest):
    for tok in rest[1:]:
        if tok.startswith('-'):
            return False
        if tok == 'posterior':
            return True
        if tok not in _PLOT_VERB_SET:
            return False
    return False


def _plot_has_bestfit(rest):
    for tok in rest[1:]:
        if tok.startswith('-'):
            return False
        if tok == 'bestfit':
            return True
        if tok not in _PLOT_VERB_SET:
            return False
    return False


def _run_plot_bestfit(rest, prog):
    try:
        outdir, cat, obj, cfg = parse_plot_bestfit(rest)
    except ValueError as e:
        sys.stderr.write(f"Error: {e}\n")
        sys.stderr.write(plot_usage(prog))
        return 1
    try:
        results = _load_results_flags(outdir, cat, obj, cfg)
        plot_kw = {'show': True}
        chosen = _announce_plot_objects(results)
        if chosen:
            plot_kw['object_id'] = chosen
        results.plot_bestfit(**plot_kw)
    except (ValueError, FileNotFoundError) as e:
        sys.stderr.write(f"Error: {e}\n")
        return 1
    return 0


def _run_plot_posterior(rest, prog):
    try:
        outdir, kind, cat, obj, pars, labels, max_n, cfg = parse_plot_posterior(
            rest
        )
    except ValueError as e:
        sys.stderr.write(f"Error: {e}\n")
        sys.stderr.write(plot_usage(prog))
        return 1
    try:
        results = _load_results_flags(outdir, cat, obj, cfg)
        plot_kw = {'show': True}
        chosen = _announce_plot_objects(results)
        if chosen:
            plot_kw['object_id'] = chosen
        if kind == 'free':
            results.plot_posterior_free(**plot_kw)
        elif kind == 'derived':
            plot_kw['max_params'] = (
                _DERIVED_DEFAULT_MAX if max_n is None else max_n
            )
            results.plot_posterior_derived(**plot_kw)
        else:
            if labels is not None:
                results.set_parameter_labels(dict(zip(pars, labels)))
            results.plot_posterior(params=pars, **plot_kw)
    except (ValueError, FileNotFoundError) as e:
        sys.stderr.write(f"Error: {e}\n")
        return 1
    return 0


def _run_results_plot(kind, rest, prog):
    if kind == 'results' and _results_has_par_values(rest):
        return _run_par_values(rest, prog)
    if kind == 'results' and _results_has_catalogs(rest):
        return _run_catalogs(rest, prog)
    if kind == 'results' and _results_has_model_configs(rest):
        return _run_model_configs(rest, prog)
    if kind == 'results':
        return _run_results_query(rest, prog)
    if kind == 'plot' and _plot_has_posterior(rest):
        return _run_plot_posterior(rest, prog)
    if kind == 'plot' and _plot_has_bestfit(rest):
        return _run_plot_bestfit(rest, prog)
    sys.stderr.write('Error: missing command\n')
    sys.stderr.write(plot_usage(prog))
    return 1


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
