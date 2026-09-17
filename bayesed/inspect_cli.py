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
from pathlib import Path, PurePosixPath

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
_PLOT_BESTFIT_FLAGS = {
    '--cat': 'cat', '--obj': 'obj', '--cfg': 'cfg', '--out': 'out',
}
_PLOT_POSTERIOR_FLAGS = {
    '--cat': 'cat', '--obj': 'obj', '--max': 'max', '--cfg': 'cfg',
    '--out': 'out',
}
_JSON_BOOL = {'--json': 'json'}
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
    )


def plot_usage_lines(prog='bayesed3'):
    return (
        f"{prog} plot <outdir> bestfit [--cat PATTERN] [--obj PATTERN] "
        f"[--cfg PATTERN] [--out path]",
        f"{prog} plot <outdir> posterior free [--cat PATTERN] [--obj PATTERN] "
        f"[--cfg PATTERN] [--out path]",
        f"{prog} plot <outdir> posterior derived [--max N] [--cat PATTERN] "
        f"[--obj PATTERN] [--cfg PATTERN] [--out path]",
        f"{prog} plot <outdir> posterior --par PATTERN [--lab TEXT] ... "
        f"[--cat PATTERN] [--obj PATTERN] [--cfg PATTERN] [--out path]",
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


def _looks_like_file(path):
    name = PurePosixPath(path).name
    return '.' in name and not path.endswith('/') and not path.endswith('\\')


def _resolve_out(out, object_ids, kind):
    """Plan ``(object_id, output_file_or_None)`` per plot call."""
    if out is None:
        return [(oid, None) for oid in object_ids]
    if len(object_ids) > 1 and _looks_like_file(out):
        raise ValueError(
            '--out must be a directory when multiple objects are plotted '
            f'({len(object_ids)} objects)'
        )
    if len(object_ids) == 1 and _looks_like_file(out):
        return [(object_ids[0], out)]
    base = out.rstrip('/\\')
    return [(oid, f'{base}/{oid}_{kind}.png') for oid in object_ids]


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


def parse_plot_bestfit(tokens):
    """
    Parse ``<outdir> bestfit [--cat C] [--obj O] [--cfg CFG] [--out PATH]``.

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
    return outdir, flags['cat'], flags['obj'], flags['cfg'], flags['out']


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
    out = None
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
            elif key == 'out':
                out = value
            else:
                cfg = value
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
    return outdir, kind, cat, obj, pars, labs if has_lab else None, max_n, cfg, out


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
    print(json.dumps(data))


def _json_scalar(value):
    if hasattr(value, 'item'):
        return value.item()
    return value


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
    return [
        {key: _json_scalar(val) for key, val in record.items()}
        for record in records
    ]


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
        sys.stderr.write(f"Error: {e}\n")
        sys.stderr.write(results_usage(prog))
        return 1
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
        sys.stderr.write(f"Error: {e}\n")
        sys.stderr.write(results_usage(prog))
        return 1
    _print_name_list(list_catalog_names(outdir), as_json)
    return 0


def _run_model_configs(rest, prog):
    try:
        outdir, cat, as_json = parse_model_configs(rest)
    except ValueError as e:
        sys.stderr.write(f"Error: {e}\n")
        sys.stderr.write(results_usage(prog))
        return 1
    try:
        catalog = _catalog_for_discovery(outdir, cat)
        _print_name_list(list_model_configs(outdir, catalog), as_json)
    except ValueError as e:
        _emit_load_error(e)
        return 1
    return 0


def _run_results_query(rest, prog):
    try:
        outdir, cmd, cat, obj, cfg, as_json = parse_results_query(rest)
    except ValueError as e:
        sys.stderr.write(f"Error: {e}\n")
        sys.stderr.write(results_usage(prog))
        return 1
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
        outdir, cat, obj, cfg, out = parse_plot_bestfit(rest)
    except ValueError as e:
        sys.stderr.write(f"Error: {e}\n")
        sys.stderr.write(plot_usage(prog))
        return 1
    try:
        results = _load_results_flags(outdir, cat, cfg)
        object_ids = _resolve_plot_objects(results.list_objects(), obj)
        plan = _resolve_out(out, object_ids, 'bestfit')
        for oid, output_file in plan:
            plot_kw = {'show': True, 'object_id': oid}
            if output_file is not None:
                plot_kw['output_file'] = output_file
            results.plot_bestfit(**plot_kw)
    except (ValueError, FileNotFoundError) as e:
        _emit_load_error(e)
        return 1
    return 0


def _run_plot_posterior(rest, prog):
    try:
        outdir, kind, cat, obj, pars, labels, max_n, cfg, out = (
            parse_plot_posterior(rest)
        )
    except ValueError as e:
        sys.stderr.write(f"Error: {e}\n")
        sys.stderr.write(plot_usage(prog))
        return 1
    try:
        results = _load_results_flags(outdir, cat, cfg)
        object_ids = _resolve_plot_objects(results.list_objects(), obj)
        plot_kind = (
            'posterior_free' if kind == 'free'
            else 'posterior_derived' if kind == 'derived'
            else 'posterior'
        )
        plan = _resolve_out(out, object_ids, plot_kind)
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
        for oid, output_file in plan:
            plot_kw = {'show': True, 'object_id': oid}
            if output_file is not None:
                plot_kw['output_file'] = output_file
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
    if kind == 'results' and _results_has_par_values(rest):
        return _run_par_values(rest, prog)
    if kind == 'results' and _results_has_catalogs(rest):
        return _run_catalogs(rest, prog)
    if kind == 'results' and _results_has_model_configs(rest):
        return _run_model_configs(rest, prog)
    if kind == 'results' and _first_results_verb(rest) is None:
        sys.stderr.write('Error: missing command\n')
        sys.stderr.write(results_usage(prog))
        return 1
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
