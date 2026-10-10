"""
Launcher inspect path (ADR-0009).

First-token ``results`` / ``plot`` / ``priors`` present existing products.
Not a public package export; ``run_launcher`` dispatches here.
"""

import ast
import glob
import json
import os
import re
import sys
from pathlib import Path, PurePath

from .inference import SEDInference
from .prior_editor import PriorEditor
from .prior_manager import PriorManager
from .results import BayeSEDResults
from .results.utils import list_catalog_names, list_model_configs

INSPECT_TOKENS = frozenset({'results', 'plot', 'priors'})
RESULTS_VERBS = (
    'summary', 'evidence', 'par_names', 'objects', 'par_values',
    'catalogs', 'model_configs', 'fit_quality',
    'run_arguments', 'run_version',
)
_RESULT_VERB_SET = frozenset(RESULTS_VERBS)
_RUN_INFO_KEY = {'run_arguments': 'run_arguments', 'run_version': 'version'}
_RUN_INFO_VERBS = frozenset(_RUN_INFO_KEY)
_RUN_INFO_FLAGS = {'--cat': 'cat', '--cfg': 'cfg'}
_QUERY_VERBS = frozenset({'summary', 'evidence', 'par_names', 'objects'})
PLOT_VERBS = ('bestfit', 'posterior')
_PLOT_VERB_SET = frozenset(PLOT_VERBS)
_IPRIOR_NAME = re.compile(r'.*\.iprior(\.\d+)?$')
_PAR_VALUES_FLAGS = {
    '--cat': 'cat', '--obj': 'obj', '--par': 'par', '--cfg': 'cfg',
}
_FIT_QUALITY_FLAGS = {
    '--cat': 'cat', '--cfg': 'cfg', '--obj': 'obj',
    '--band': 'band', '--top': 'top', '--sigma': 'sigma',
}
_RESULTS_QUERY_FLAGS = {'--cat': 'cat', '--obj': 'obj', '--cfg': 'cfg'}
_MODEL_CONFIGS_FLAGS = {'--cat': 'cat'}
_PLOT_SAVE_BOOL = {'--save': 'save'}
_PLOT_BESTFIT_BOOL = {
    '--save': 'save',
    '--observed-frame': 'observed_frame',
    '--linear': 'linear',
    '--no-lines': 'no_lines',
    '--no-total': 'no_total',
    '--no-components': 'no_components',
    '--full-range': 'full_range',
    '--show-residuals': 'show_residuals',
}
_PLOT_BESTFIT_FLAGS = {
    '--cat': 'cat', '--obj': 'obj', '--cfg': 'cfg',
    '--flux': 'flux', '--log': 'log',
}
_BESTFIT_FLUX_UNITS = frozenset({'fnu', 'nufnu', 'flambda'})
_BESTFIT_LOG_AXES = frozenset({'x', 'y', 'both'})
_PLOT_POSTERIOR_FLAGS = {
    '--cat': 'cat', '--obj': 'obj', '--max': 'max', '--cfg': 'cfg',
}
_JSON_BOOL = {'--json': 'json'}
_INSPECT_VALUED_FLAGS = frozenset({
    '--cat', '--obj', '--cfg', '--par', '--lab', '--max', '--flux', '--log',
    '--band', '--top', '--sigma',
})
_DERIVED_DEFAULT_MAX = 10


def results_usage_lines(prog='bayesed3'):
    return (
        f"{prog} results <outdir> summary [--cat PATTERN] [--cfg PATTERN] [--json]",
        f"{prog} results <outdir> evidence [--cat PATTERN] [--cfg PATTERN] "
        f"[--obj PATTERN] [--json]",
        f"{prog} results <outdir> par_names [--cat PATTERN] [--cfg PATTERN] [--json]",
        f"{prog} results <outdir> objects [--cat PATTERN] [--cfg PATTERN] "
        f"[--obj PATTERN] [--json]",
        f"{prog} results <outdir> catalogs [--json]",
        f"{prog} results <outdir> model_configs [--cat PATTERN] [--json]",
        f"{prog} results <outdir> par_values --par PATTERN "
        f"[--cat PATTERN] [--cfg PATTERN] [--obj PATTERN] [--json]",
        f"{prog} results <outdir> fit_quality [--cat PATTERN] "
        f"[--cfg PATTERN] [--obj PATTERN] [--band EDGES] [--top N] "
        f"[--sigma T] [--json]",
        f"{prog} results <outdir> run_arguments [--cat PATTERN] [--cfg PATTERN]",
        f"{prog} results <outdir> run_version [--cat PATTERN] [--cfg PATTERN]",
    )


def plot_usage_lines(prog='bayesed3'):
    return (
        f"{prog} plot <outdir> bestfit [--cat PATTERN] [--obj PATTERN] "
        f"[--cfg PATTERN] [--observed-frame] [--flux fnu|nufnu|flambda] "
        f"[--log x|y|both] [--linear] [--no-lines] [--no-total] "
        f"[--no-components] [--full-range] [--show-residuals] [--save]",
        f"{prog} plot <outdir> posterior free [--cat PATTERN] [--obj PATTERN] "
        f"[--cfg PATTERN] [--save]",
        f"{prog} plot <outdir> posterior derived [--max N] [--cat PATTERN] "
        f"[--obj PATTERN] [--cfg PATTERN] [--save]",
        f"{prog} plot <outdir> posterior --par PATTERN [--lab TEXT] ... "
        f"[--cat PATTERN] [--obj PATTERN] [--cfg PATTERN] [--save]",
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
    return _usage_block(inspect_usage_lines(prog))


def results_usage(prog='bayesed3'):
    return _usage_block(results_usage_lines(prog))


def plot_usage(prog='bayesed3'):
    return _usage_block(plot_usage_lines(prog))


def _one_usage(line):
    return _usage_block((line,))


def _plot_command_usage(prog, verb):
    lines = plot_usage_lines(prog)
    if verb == 'bestfit':
        return _one_usage(lines[0])
    return _usage_block(lines[1:])


def _results_command_usage(prog, verb):
    for line in results_usage_lines(prog):
        if f" {verb} " in line:
            return _one_usage(line)
    raise KeyError(verb)


def _positional_verbs(tokens, verbs):
    """Command words in ``tokens``, skipping valued-flag arguments."""
    found = []
    expect_value = False
    for tok in tokens:
        if expect_value:
            expect_value = False
            continue
        if tok in _INSPECT_VALUED_FLAGS:
            expect_value = True
            continue
        if tok.startswith('-'):
            continue
        if tok in verbs:
            found.append(tok)
    return found


def _parse_usage(kind, tokens, prog):
    """Usage for a parse error. One verb in the command slot narrows it."""
    slot = tokens[1] if len(tokens) > 1 else None
    if kind == 'plot':
        verbs = _positional_verbs(tokens, _PLOT_VERB_SET)
        if len(set(verbs)) == 1 and slot == verbs[0]:
            return _plot_command_usage(prog, verbs[0])
        return plot_usage(prog)
    verbs = _positional_verbs(tokens, _RESULT_VERB_SET)
    if len(verbs) == 1 and slot == verbs[0]:
        return _results_command_usage(prog, verbs[0])
    return results_usage(prog)


def _emit_error(message, usage):
    sys.stderr.write(f"Error: {message}\n")
    sys.stderr.write(usage)
    return 1


def _results_parse(tokens):
    verbs = _positional_verbs(tokens[1:], _RESULT_VERB_SET)
    name = verbs[0] if verbs else None
    if name == 'par_values':
        parse_par_values(tokens)
    elif name == 'catalogs':
        parse_catalogs(tokens)
    elif name == 'model_configs':
        parse_model_configs(tokens)
    else:
        parse_results_query(tokens)


def _results_conflict_message(rest):
    try:
        _results_parse(['.'] + list(rest))
    except ValueError as exc:
        return str(exc)
    return 'missing command'


def priors_usage(prog='bayesed3'):
    return f"Usage: {priors_usage_line(prog)}\n"


def _flag_value(tokens, i, flag_map):
    """Return ``(key, value, next_i)`` if ``tokens[i]`` is a mapped flag."""
    if i >= len(tokens):
        return None
    tok = tokens[i]
    key = flag_map.get(tok)
    if key is None:
        return None
    if i + 1 >= len(tokens) or tokens[i + 1].startswith('--'):
        raise ValueError(f'{tok} requires a value')
    return key, tokens[i + 1], i + 2


def _parse_flag_pairs(tokens, start, flag_map, bool_flags=None):
    """Parse ``tokens[start:]`` as flags. Last write wins for valued flags."""
    values = {name: None for name in flag_map.values()}
    if bool_flags:
        for name in bool_flags.values():
            values[name] = False
    i = start
    while i < len(tokens):
        tok = tokens[i]
        if bool_flags and tok in bool_flags:
            values[bool_flags[tok]] = True
            i += 1
            continue
        got = _flag_value(tokens, i, flag_map)
        if got is not None:
            key, value, i = got
            values[key] = value
            continue
        if tok.startswith('-'):
            raise ValueError(f'unknown option {tok}')
        raise ValueError(f"unexpected argument '{tok}'")
    return values


def match_names(candidates, pattern):
    """
    Inspect loose match: exact, then contains, then regex.

    First non-empty hit list wins. Candidate order is preserved.
    Invalid regex after empty exact/contains yields no hits.
    """
    names = list(candidates)
    exact = [name for name in names if name == pattern]
    if exact:
        return exact
    contains = [name for name in names if pattern in name]
    if contains:
        return contains
    try:
        compiled = re.compile(pattern)
    except re.error:
        return []
    return [name for name in names if compiled.search(name)]


def _no_match_message(thing, pattern, list_name, items):
    return (
        f"No {thing} matching {pattern!r}. "
        f"Available {list_name}: {list(items)}"
    )


def _require_outdir(outdir):
    path = Path(outdir)
    if not path.is_dir():
        raise ValueError(f'not a directory: {outdir}')


def _require_one(candidates, pattern, things, thing, flag, list_name):
    """Loose-match pattern to exactly one candidate name."""
    hits = match_names(candidates, pattern)
    if not hits:
        raise ValueError(_no_match_message(thing, pattern, list_name, candidates))
    if len(hits) > 1:
        raise ValueError(_multi_choice_message(
            things, thing, flag, list_name, hits,
        ))
    return hits[0]


def _resolve_cat_and_cfg(outdir, cat, cfg):
    """
    Resolve --cat / --cfg in inspect; return exact names for BayeSEDResults.

    ``cfg`` requires a Results catalog context (explicit or sole catalog).
    """
    _require_outdir(outdir)
    catalogs = list_catalog_names(outdir)
    catalog_name = None
    if cat is not None:
        catalog_name = _require_one(
            catalogs, cat, 'catalogs', 'catalog', '--cat', 'catalogs',
        )
    model_config = None
    if cfg is not None:
        if catalog_name is None:
            if not catalogs:
                raise ValueError(f'No catalogs in {outdir}')
            if len(catalogs) > 1:
                raise ValueError(_multi_choice_message(
                    'catalogs', 'catalog', '--cat', 'catalogs', catalogs,
                ))
            catalog_name = catalogs[0]
        configs = list_model_configs(outdir, catalog_name)
        model_config = _require_one(
            configs, cfg, 'configs', 'config', '--cfg', 'configs',
        )
    return catalog_name, model_config


def _resolve_plot_objects(all_objects, obj_pattern):
    """
    Object ids to plot in order.

    ``obj_pattern`` None: silent single, else Multiple.
    Pattern set: loose match; 1..N all plotted; 0 → No matching.
    """
    objects = list(all_objects)
    if obj_pattern is None:
        if not objects:
            return []
        if len(objects) == 1:
            return objects
        raise ValueError(_multi_choice_message(
            'objects', 'object', '--obj', 'objects', objects,
        ))
    hits = match_names(objects, obj_pattern)
    if not hits:
        raise ValueError(_no_match_message(
            'object', obj_pattern, 'objects', objects,
        ))
    return hits


def _resolve_results_obj_filter(all_objects, obj_pattern):
    """Optional --obj filter for results queries; several hits allowed."""
    if obj_pattern is None:
        return None
    hits = match_names(list(all_objects), obj_pattern)
    if not hits:
        raise ValueError(_no_match_message(
            'object', obj_pattern, 'objects', list(all_objects),
        ))
    return hits


def _exact_names_regex(names):
    return '^(?:' + '|'.join(re.escape(name) for name in names) + ')$'


def parse_par_values(tokens):
    """
    Parse ``<outdir> par_values [--cat C] [--cfg CFG] [--obj O] --par P [--json]``.

    ``par_values`` is the only command. No leftover positionals.
    """
    if not tokens:
        raise ValueError('missing outdir')
    outdir = tokens[0]
    rest = tokens[1:]
    if not rest or rest[0] != 'par_values':
        raise ValueError('missing command')
    flags = _parse_flag_pairs(
        rest, 1, _PAR_VALUES_FLAGS, bool_flags=_JSON_BOOL,
    )
    if flags['par'] is None:
        raise ValueError('--par is required')
    return (
        outdir, flags['cat'], flags['obj'], flags['par'], flags['cfg'],
        flags['json'],
    )


def parse_results_query(tokens):
    """
    Parse ``<outdir> <query> [--cat C] [--cfg CFG] [--obj O] [--json]``.

    Exactly one query verb: ``summary`` / ``evidence`` / ``par_names`` /
    ``objects``. Flags after the verb. No leftover positionals.
    """
    if not tokens:
        raise ValueError('missing outdir')
    outdir = tokens[0]
    rest = tokens[1:]
    if not rest or rest[0] not in _QUERY_VERBS:
        raise ValueError('missing command')
    cmd = rest[0]
    i = 1
    if i < len(rest) and rest[i] in _QUERY_VERBS:
        raise ValueError(
            f"'{rest[i]}' cannot be combined with '{cmd}'"
        )
    flags = _parse_flag_pairs(
        rest, i, _RESULTS_QUERY_FLAGS, bool_flags=_JSON_BOOL,
    )
    return outdir, cmd, flags['cat'], flags['obj'], flags['cfg'], flags['json']


def parse_run_info(tokens):
    """
    Parse ``<outdir> run_arguments|run_version [--cat C] [--cfg CFG]``.

    One provenance verb; no leftover positionals. These read whole-file
    attributes, so there is no ``--obj``.
    """
    if not tokens:
        raise ValueError('missing outdir')
    outdir = tokens[0]
    rest = tokens[1:]
    if not rest or rest[0] not in _RUN_INFO_VERBS:
        raise ValueError('missing command')
    flags = _parse_flag_pairs(rest, 1, _RUN_INFO_FLAGS)
    return outdir, rest[0], flags['cat'], flags['cfg']


def parse_model_configs(tokens):
    """
    Parse ``<outdir> model_configs [--cat C] [--json]``.

    ``model_configs`` is the only command. No leftover positionals.
    """
    if not tokens:
        raise ValueError('missing outdir')
    outdir = tokens[0]
    rest = tokens[1:]
    if not rest or rest[0] != 'model_configs':
        raise ValueError('missing command')
    flags = _parse_flag_pairs(
        rest, 1, _MODEL_CONFIGS_FLAGS, bool_flags=_JSON_BOOL,
    )
    return outdir, flags['cat'], flags['json']


def _parse_band_edges(value):
    """Parse ``--band EDGES`` into strictly increasing floats (μm, rest)."""
    try:
        edges = [float(part) for part in value.split(',')]
    except ValueError:
        raise ValueError(
            '--band requires comma-separated numbers (rest-frame μm)'
        ) from None
    if not edges or any(
            later <= earlier
            for earlier, later in zip(edges, edges[1:])):
        raise ValueError('--band requires strictly increasing edges')
    return edges


def parse_fit_quality(tokens):
    """
    Parse ``<outdir> fit_quality [--cat C] [--cfg CFG] [--obj O]
    [--band EDGES] [--top N] [--sigma T] [--json]``.

    ``fit_quality`` is the only command. No leftover positionals.
    """
    if not tokens:
        raise ValueError('missing outdir')
    outdir = tokens[0]
    rest = tokens[1:]
    if not rest or rest[0] != 'fit_quality':
        raise ValueError('missing command')
    flags = _parse_flag_pairs(
        rest, 1, _FIT_QUALITY_FLAGS, bool_flags=_JSON_BOOL,
    )
    band_edges = None
    if flags['band'] is not None:
        band_edges = _parse_band_edges(flags['band'])
    top_n = 20
    if flags['top'] is not None:
        try:
            top_n = int(flags['top'])
        except ValueError:
            raise ValueError('--top requires a positive integer') from None
        if top_n < 1:
            raise ValueError('--top requires a positive integer')
    sigma_threshold = 3.0
    if flags['sigma'] is not None:
        try:
            sigma_threshold = float(flags['sigma'])
        except ValueError:
            raise ValueError('--sigma requires a positive number') from None
        if not sigma_threshold > 0:
            raise ValueError('--sigma requires a positive number')
    return (
        outdir, flags['cat'], flags['cfg'], flags['obj'],
        band_edges, top_n, sigma_threshold, flags['json'],
    )


def parse_catalogs(tokens):
    """Parse ``<outdir> catalogs [--json]``. No leftover positionals."""
    if not tokens:
        raise ValueError('missing outdir')
    outdir = tokens[0]
    rest = tokens[1:]
    if not rest or rest[0] != 'catalogs':
        raise ValueError('missing command')
    flags = _parse_flag_pairs(rest, 1, {}, bool_flags=_JSON_BOOL)
    return outdir, flags['json']


def _bestfit_appearance(flags):
    """Map present Bestfit appearance flags onto ``plot_bestfit`` kwargs.

    Omitted flags are left out so ``plot_bestfit`` keeps its Python defaults.
    """
    appearance = {}
    if flags['observed_frame']:
        appearance['use_rest_frame'] = False
    flux = flags['flux']
    if flux is not None:
        if flux not in _BESTFIT_FLUX_UNITS:
            raise ValueError('--flux requires fnu, nufnu, or flambda')
        appearance['flux_unit'] = flux
    log_axis = flags['log']
    if log_axis is not None and flags['linear']:
        raise ValueError('--log and --linear cannot be combined')
    if log_axis is not None:
        if log_axis not in _BESTFIT_LOG_AXES:
            raise ValueError('--log requires x, y, or both')
        appearance['use_log_scale'] = log_axis
    elif flags['linear']:
        appearance['use_log_scale'] = False
    if flags['no_lines']:
        appearance['show_emission_lines'] = False
    if flags['no_total']:
        appearance['show_total'] = False
    if flags['no_components']:
        appearance['show_components'] = False
    if flags['full_range']:
        appearance['focus_on_data_range'] = False
    if flags['show_residuals']:
        appearance['show_residuals'] = True
    return appearance


def parse_plot_bestfit(tokens):
    """
    Parse ``<outdir> bestfit`` plus catalog, object, config, save, and
    Bestfit appearance flags. ``bestfit`` is the only command.
    """
    if not tokens:
        raise ValueError('missing outdir')
    outdir = tokens[0]
    rest = tokens[1:]
    if not rest or rest[0] != 'bestfit':
        if 'bestfit' in rest:
            raise ValueError('bestfit cannot be combined with other commands')
        raise ValueError('missing command')
    flags = _parse_flag_pairs(
        rest, 1, _PLOT_BESTFIT_FLAGS, bool_flags=_PLOT_BESTFIT_BOOL,
    )
    return (
        outdir, flags['cat'], flags['obj'], flags['cfg'], flags['save'],
        _bestfit_appearance(flags),
    )


def parse_plot_posterior(tokens):
    """
    Parse ``<outdir> posterior (free | derived | --par PATTERN [--lab TEXT] ...)``.

    ``posterior`` is the only command. Catalog, object, and model config
    via ``--cat`` / ``--obj`` / ``--cfg``. Each ``--lab`` is optional and
    must immediately follow its ``--par``.
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
    do_save = False
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
        if tok == '--save':
            do_save = True
            i += 1
            continue
        if tok == '--par':
            if i + 1 >= len(rest) or rest[i + 1].startswith('--'):
                raise ValueError('--par requires a value')
            pars.append(rest[i + 1])
            i += 2
            if i < len(rest) and rest[i] == '--lab':
                if i + 1 >= len(rest) or rest[i + 1].startswith('--'):
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
    if kind is not None and pars:
        raise ValueError(f'{kind} cannot be combined with --par')
    if kind is None and not pars:
        raise ValueError('specify free, derived, or --par')
    if max_n is not None and kind != 'derived':
        raise ValueError('--max is only valid with derived')
    return outdir, kind, cat, obj, pars, labs if has_lab else None, max_n, cfg, do_save


def _catalog_for_discovery(outdir, cat):
    _require_outdir(outdir)
    catalogs = list_catalog_names(outdir)
    if cat is not None:
        return _require_one(
            catalogs, cat, 'catalogs', 'catalog', '--cat', 'catalogs',
        )
    if len(catalogs) == 1:
        return catalogs[0]
    if not catalogs:
        raise ValueError(f'No catalogs in {outdir}')
    raise ValueError(_multi_choice_message(
        'catalogs', 'catalog', '--cat', 'catalogs', catalogs,
    ))


def _load_results_flags(outdir, cat, cfg=None):
    """Construct Results with exact catalog / Model configuration names only."""
    catalog_name, model_config = _resolve_cat_and_cfg(outdir, cat, cfg)
    kwargs = {}
    if catalog_name is not None:
        kwargs['catalog_name'] = catalog_name
    if model_config is not None:
        kwargs['model_config'] = model_config
    try:
        return BayeSEDResults(outdir, **kwargs)
    except ValueError as exc:
        raise ValueError(_cli_wording_for_library_error(exc)) from exc


def _print_json(data):
    print(json.dumps(_json_ready(data)))


def _json_scalar(value):
    """Coerce one leaf value to a JSON-native Python type."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, PurePath):
        return str(value)
    if isinstance(value, bytes):
        return value.decode('utf-8', errors='replace')
    if isinstance(value, bytearray):
        return bytes(value).decode('utf-8', errors='replace')
    if hasattr(value, 'item') and not isinstance(value, (str, bytes, bytearray)):
        try:
            value = value.item()
        except (ValueError, AttributeError):
            return value
        return _json_scalar(value)
    return value


def _json_ready(value):
    """Recursively convert dict/list trees to JSON-serializable values."""
    if isinstance(value, dict):
        return {str(_json_scalar(k)): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    return _json_scalar(value)


def _table_json_records(table):
    """Convert an Astropy/pandas table to a list of JSON-native dicts."""
    if hasattr(table, 'to_dict') and not hasattr(table, 'colnames'):
        records = table.to_dict(orient='records')
    elif hasattr(table, 'to_pandas'):
        records = table.to_pandas().to_dict(orient='records')
    elif hasattr(table, 'colnames'):
        records = [
            {col: _json_scalar(row[col]) for col in table.colnames}
            for row in table
        ]
    else:
        raise TypeError(
            f'cannot serialize {type(table).__name__} as JSON records'
        )
    return [_json_ready(record) for record in records]


def _format_available(items):
    """At most ten entries for ``Available …`` lists (first/last five)."""
    items = list(items)
    if len(items) <= 10:
        return items
    return items[:5] + ['...'] + items[-5:]


def _truncate_available_lists(msg):
    """Truncate trailing ``Available …: […]`` lists at the stderr boundary."""
    for label in (
        'Available catalogs: ',
        'Available configs: ',
        'Available objects: ',
    ):
        idx = msg.rfind(label)
        if idx < 0:
            continue
        start = idx + len(label)
        try:
            items = ast.literal_eval(msg[start:])
        except (ValueError, SyntaxError):
            continue
        if not isinstance(items, list):
            continue
        return msg[:start] + str(_format_available(items))
    return msg


def _multi_choice_message(things, thing, flag, available_label, items):
    return (
        f'Multiple {things} found. Please specify {thing} with {flag}. '
        f'Available {available_label}: {list(items)}'
    )


def _cli_wording_for_library_error(exc):
    """Rewrite library multi-choice ValueErrors to the Inspect one-line form."""
    msg = str(exc)
    for needle, things, thing, flag, avail in (
        (
            'Multiple catalogs found',
            'catalogs', 'catalog', '--cat', 'Available catalogs: ',
        ),
        (
            'Multiple model configurations found',
            'model configurations', 'config', '--cfg', 'Available configs: ',
        ),
        (
            'Multiple objects found',
            'objects', 'object', '--obj', 'Available objects: ',
        ),
    ):
        if needle not in msg:
            continue
        idx = msg.find(avail)
        tail = msg[idx:] if idx >= 0 else ''
        body = f'Multiple {things} found. Please specify {thing} with {flag}.'
        return f'{body} {tail}'.rstrip()
    return msg


def _emit_load_error(exc):
    sys.stderr.write(f"Error: {_truncate_available_lists(str(exc))}\n")


def _par_values_json_payload(table):
    """Serialize ``get_parameter_values`` output as JSON records."""
    return _table_json_records(table)


def _evidence_json_payload(table):
    """Serialize Evidence table rows for ``results … evidence --json``.

    Always a JSON array of objects with ``ID`` plus whatever of
    ``INSlogZ`` / ``logZ`` / ``INSlogZerr`` / ``logZerr`` are present.
    """
    return _table_json_records(table)


def _summary_payload(results):
    """Orientation fields for ``results … summary --json`` (whole Results)."""
    payload = {
        'output_dir': results.output_dir,
        'catalog': results.catalog_name,
        'hdf5': results.hdf5_file or '',
    }
    try:
        payload['n_hdf5_columns'] = len(results.get_parameter_names())
        payload['n_free'] = len(results.get_free_parameters())
        payload['n_derived'] = len(results.get_derived_parameters())
    except Exception as e:
        payload['parameter_summary_error'] = str(e)
    try:
        objects = list(results.list_objects())
        if len(objects) <= 10:
            payload['objects'] = objects
        else:
            payload['objects'] = objects[:5] + ['...'] + objects[-5:]
        payload['n_objects'] = len(objects)
    except Exception as e:
        payload['object_summary_error'] = str(e)
    return payload


def _print_evidence(results, as_json=False, object_ids=None):
    if as_json:
        kwargs = {'return_format': 'table'}
        if object_ids is not None:
            kwargs['object_ids'] = object_ids
        evidence = results.get_evidence(**kwargs)
        _print_json(_evidence_json_payload(evidence))
        return
    if object_ids is not None:
        evidence = results.get_evidence(object_ids=object_ids)
    else:
        evidence = results.get_evidence()
    if isinstance(evidence, dict) and 'log_evidence' in evidence:
        err = evidence.get('log_evidence_error')
        print(f"log_evidence: {evidence['log_evidence']} ± {err}")
        return
    print(evidence)


def _run_par_values(rest, prog):
    try:
        outdir, cat, obj, par, cfg, as_json = parse_par_values(rest)
    except ValueError as e:
        return _emit_error(str(e), _parse_usage('results', rest, prog))
    try:
        results = _load_results_flags(outdir, cat, cfg)
        matched = match_names(results.get_parameter_names(), par)
        if not matched:
            raise ValueError(f"No parameters found matching '{par}'")
        kwargs = {
            'param_match_mode': 'regex',
        }
        if obj is not None:
            matched_objs = _resolve_results_obj_filter(
                results.list_objects(), obj,
            )
            kwargs['object_ids'] = matched_objs
            kwargs['object_match_mode'] = 'exact'
        table = results.get_parameter_values(_exact_names_regex(matched), **kwargs)
        if as_json:
            _print_json(_par_values_json_payload(table))
        else:
            print(table)
    except (ValueError, FileNotFoundError) as e:
        _emit_load_error(e)
        return 1
    return 0


def _print_name_list(items, as_json):
    if as_json:
        _print_json(list(items))
    else:
        for item in items:
            print(item)


def _run_catalogs(rest, prog):
    try:
        outdir, as_json = parse_catalogs(rest)
    except ValueError as e:
        return _emit_error(str(e), _parse_usage('results', rest, prog))
    try:
        _require_outdir(outdir)
        _print_name_list(list_catalog_names(outdir), as_json)
    except ValueError as e:
        _emit_load_error(e)
        return 1
    return 0


def _run_model_configs(rest, prog):
    try:
        outdir, cat, as_json = parse_model_configs(rest)
    except ValueError as e:
        return _emit_error(str(e), _parse_usage('results', rest, prog))
    try:
        catalog = _catalog_for_discovery(outdir, cat)
        _print_name_list(list_model_configs(outdir, catalog), as_json)
    except ValueError as e:
        _emit_load_error(e)
        return 1
    return 0


def _print_fit_quality(records):
    """Human form of ``results … fit_quality``: one table per object."""
    for rec in records:
        header = f"object {rec['object_id']}"
        if 'chi2_red' in rec:
            header += (
                f"  chi2_red={rec['chi2_red']:.3f} ({rec['chi2_source']})"
            )
        print(header)
        print(f"  {'kind':<5} {'label':<16} {'n':>6} {'chi2_red':>9} "
              f"{'resid_rms':>9} {'frac_outside':>12}  flags")
        for ds in rec['datasets']:
            flags = ','.join(ds['flags']) if ds['flags'] else '-'
            print(f"  {ds['kind']:<5} {ds['label']:<16} {ds['n']:>6} "
                  f"{ds['chi2_red']:>9.3f} {ds['resid_rms']:>9.3f} "
                  f"{ds['frac_outside']:>12.3f}  {flags}")


def _run_fit_quality(rest, prog):
    try:
        (outdir, cat, cfg, obj, band_edges, top_n, sigma_threshold,
         as_json) = parse_fit_quality(rest)
    except ValueError as e:
        return _emit_error(str(e), _parse_usage('results', rest, prog))
    try:
        results = _load_results_flags(outdir, cat, cfg)
        all_objects = results.list_objects()
        object_ids = (
            _resolve_results_obj_filter(all_objects, obj)
            if obj is not None else all_objects
        )
        records = [
            results.get_fit_quality(
                oid, band_edges=band_edges, top_n=top_n,
                sigma_threshold=sigma_threshold)
            for oid in object_ids
        ]
        if as_json:
            _print_json(records)
        else:
            _print_fit_quality(records)
    except (ValueError, FileNotFoundError) as e:
        _emit_load_error(e)
        return 1
    return 0


def _run_run_info(rest, prog):
    try:
        outdir, verb, cat, cfg = parse_run_info(rest)
    except ValueError as e:
        return _emit_error(str(e), _parse_usage('results', rest, prog))
    try:
        results = _load_results_flags(outdir, cat, cfg)
        provenance = results.run_provenance()
    except (ValueError, FileNotFoundError, KeyError) as e:
        _emit_load_error(e)
        return 1
    print(provenance[_RUN_INFO_KEY[verb]])
    return 0


def _run_results_query(rest, prog):
    try:
        outdir, cmd, cat, obj, cfg, as_json = parse_results_query(rest)
    except ValueError as e:
        return _emit_error(str(e), _parse_usage('results', rest, prog))
    try:
        results = _load_results_flags(outdir, cat, cfg)
        # summary / par_names ignore --obj (whole-Results orientation).
        obj_filter = None
        if obj is not None and cmd in ('evidence', 'objects'):
            obj_filter = _resolve_results_obj_filter(
                results.list_objects(), obj,
            )
        if cmd == 'summary':
            if as_json:
                _print_json(_summary_payload(results))
            else:
                results.print_summary()
        elif cmd == 'evidence':
            _print_evidence(
                results, as_json=as_json, object_ids=obj_filter,
            )
        elif cmd == 'par_names':
            _print_par_names(results, as_json=as_json)
        elif cmd == 'objects':
            items = (
                obj_filter if obj_filter is not None
                else results.list_objects()
            )
            _print_name_list(items, as_json)
    except (ValueError, FileNotFoundError) as e:
        _emit_load_error(e)
        return 1
    return 0


def _print_par_names(results, as_json=False):
    free = list(results.get_free_parameters())
    derived = list(results.get_derived_parameters())
    if as_json:
        _print_json({'free': free, 'derived': derived})
        return
    print('Free parameters:')
    for item in free:
        print(f"  {item}")
    print('Derived parameters:')
    for item in derived:
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
        outdir, cat, obj, cfg, do_save, appearance = parse_plot_bestfit(rest)
    except ValueError as e:
        return _emit_error(str(e), _parse_usage('plot', rest, prog))
    try:
        results = _load_results_flags(outdir, cat, cfg)
        object_ids = _resolve_plot_objects(results.list_objects(), obj)
        for oid in object_ids:
            plot_kw = {'show': True, 'object_id': oid, 'save': do_save}
            plot_kw.update(appearance)
            results.plot_bestfit(**plot_kw)
    except (ValueError, FileNotFoundError) as e:
        _emit_load_error(e)
        return 1
    return 0


def _run_plot_posterior(rest, prog):
    try:
        outdir, kind, cat, obj, pars, labels, max_n, cfg, do_save = (
            parse_plot_posterior(rest)
        )
    except ValueError as e:
        return _emit_error(str(e), _parse_usage('plot', rest, prog))
    try:
        results = _load_results_flags(outdir, cat, cfg)
        object_ids = _resolve_plot_objects(results.list_objects(), obj)
        expanded = None
        label_map = {}
        if kind is None:
            universe = list(dict.fromkeys(
                list(results.get_free_parameters())
                + list(results.get_derived_parameters())
            ))
            expanded = []
            lab_list = labels if labels is not None else [None] * len(pars)
            for par, lab in zip(pars, lab_list):
                hits = match_names(universe, par)
                if not hits:
                    raise ValueError(f"No parameters found matching '{par}'")
                if lab is not None:
                    if len(hits) != 1:
                        raise ValueError(
                            '--lab requires a unique --par match'
                        )
                    label_map[hits[0]] = lab
                for name in hits:
                    if name not in expanded:
                        expanded.append(name)
            if label_map:
                results.set_parameter_labels(label_map)
        for oid in object_ids:
            plot_kw = {'show': True, 'object_id': oid, 'save': do_save}
            if kind == 'free':
                results.plot_posterior_free(**plot_kw)
            elif kind == 'derived':
                plot_kw['max_params'] = (
                    _DERIVED_DEFAULT_MAX if max_n is None else max_n
                )
                results.plot_posterior_derived(**plot_kw)
            else:
                results.plot_posterior(params=expanded, **plot_kw)
    except (ValueError, FileNotFoundError) as e:
        _emit_load_error(e)
        return 1
    return 0


def _run_results_plot(kind, rest, prog):
    if kind == 'results' and (not rest or rest[0].startswith('-')):
        sys.stderr.write(results_usage(prog))
        return 1
    if kind == 'results' and rest and rest[0] in _RESULT_VERB_SET:
        verbs = _positional_verbs(rest, _RESULT_VERB_SET)
        if len(verbs) >= 2:
            return _emit_error(
                _results_conflict_message(rest),
                results_usage(prog),
            )
        return _emit_error(
            'missing outdir',
            _results_command_usage(prog, rest[0]),
        )
    if kind == 'results' and _results_has_par_values(rest):
        return _run_par_values(rest, prog)
    if kind == 'results' and _first_results_verb(rest) == 'fit_quality':
        return _run_fit_quality(rest, prog)
    if kind == 'results' and _results_has_catalogs(rest):
        return _run_catalogs(rest, prog)
    if kind == 'results' and _results_has_model_configs(rest):
        return _run_model_configs(rest, prog)
    if kind == 'results' and _first_results_verb(rest) in _RUN_INFO_VERBS:
        return _run_run_info(rest, prog)
    if kind == 'results' and _first_results_verb(rest) is None:
        sys.stderr.write('Error: missing command\n')
        sys.stderr.write(results_usage(prog))
        return 1
    if kind == 'results':
        return _run_results_query(rest, prog)
    if kind == 'plot' and (not rest or rest[0].startswith('-')):
        sys.stderr.write(plot_usage(prog))
        return 1
    if kind == 'plot':
        verbs = _positional_verbs(rest, _PLOT_VERB_SET)
        command_is_verb = bool(rest) and (
            rest[0] in _PLOT_VERB_SET
            or (len(rest) > 1 and rest[1] in _PLOT_VERB_SET)
        )
        if (
            command_is_verb
            and 'bestfit' in verbs
            and 'posterior' in verbs
        ):
            return _emit_error(
                'posterior cannot be combined with other commands',
                plot_usage(prog),
            )
        if rest and rest[0] in _PLOT_VERB_SET:
            return _emit_error(
                'missing outdir',
                _plot_command_usage(prog, rest[0]),
            )
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
    if '--help' in rest:
        # Sub-command usage request: narrow to the named verb when exactly
        # one verb OF THIS KIND is present (e.g. `plot <outdir> bestfit
        # --help`); a verb from the other kind or a mix falls back to the
        # kind's full usage. Conventionally exits 0.
        kind_verbs = _RESULT_VERB_SET if token == 'results' else _PLOT_VERB_SET
        verbs = [v for v in _positional_verbs(rest, kind_verbs)]
        if token == 'priors':
            usage = priors_usage_line(prog)
        elif len(set(verbs)) == 1:
            verb = verbs[0]
            usage = (_results_command_usage(prog, verb)
                     if token == 'results' else _plot_command_usage(prog, verb))
        else:
            usage = (results_usage(prog) if token == 'results'
                     else plot_usage(prog))
        if not usage.startswith('Usage:'):
            usage = f'Usage: {usage}\n'
        sys.stdout.write(usage)
        return 0
    if token == 'priors':
        return _run_priors(rest, prog)
    return _run_results_plot(token, rest, prog)
