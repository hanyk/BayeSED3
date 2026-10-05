"""ADR-0010: plot_posterior params vs GetDist samples."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from bayesed.results.results_presentation import (
    ResultsPresentation,
    _FIXED_VARIANCE_EPS,
)


def _fake_samples(names, columns):
    """Minimal GetDist-like samples: paramNames.names + samples array."""
    param_names = SimpleNamespace(
        names=[SimpleNamespace(name=n) for n in names]
    )
    return SimpleNamespace(
        paramNames=param_names,
        samples=np.asarray(columns, dtype=float),
    )


def test_params_in_samples_keeps_by_name_not_index():
    # Column 0 is Av; free-list order puts missing z first — must not pair by index.
    samples = _fake_samples(
        ['Av', 'age'],
        np.array([[1.0, 2.0], [1.1, 2.2], [0.9, 1.8]]),
    )
    kept = ResultsPresentation._params_in_samples(
        samples, ['z', 'Av'], require_varying=True
    )
    assert kept == ['Av']


def test_params_in_samples_drops_near_zero_variance():
    samples = _fake_samples(
        ['z', 'Av'],
        np.array([[0.05, 1.0], [0.05, 1.2], [0.05, 0.8]]),
    )
    kept = ResultsPresentation._params_in_samples(
        samples, ['z', 'Av'], require_varying=True
    )
    assert kept == ['Av']
    assert np.var(samples.samples[:, 0]) <= _FIXED_VARIANCE_EPS


def test_plot_posterior_explicit_missing_raises():
    products = MagicMock()
    products.object_id = 'obj'
    pres = ResultsPresentation(products)
    samples = _fake_samples(
        ['Av'],
        np.array([[1.0], [1.1], [0.9]]),
    )
    with patch.object(
        pres, '_load_getdist_samples',
        return_value=(samples, Path('/unused')),
    ):
        with pytest.raises(ValueError, match=r"parameter\(s\) not in posterior samples: z"):
            pres.plot_posterior(
                params=['z', 'Av'], object_id='obj', show=False, save=False,
            )


def test_plot_posterior_none_filters_then_plots():
    products = MagicMock()
    products.object_id = 'obj'
    products.get_free_parameters.return_value = ['z', 'Av']
    pres = ResultsPresentation(products)
    samples = _fake_samples(
        ['Av'],
        np.array([[1.0], [1.2], [0.8]]),
    )
    fake_g = MagicMock()
    with patch.object(
        pres, '_load_getdist_samples',
        return_value=(samples, Path('/unused')),
    ), \
         patch.object(pres, '_getdist_subplot_plotter', return_value=fake_g), \
         patch('matplotlib.pyplot'):
        pres.plot_posterior(
            params=None, object_id='obj', show=False, save=False,
        )
    fake_g.plot_1d.assert_called_once()
    assert fake_g.plot_1d.call_args.args[1] == 'Av'


def test_plot_posterior_none_empty_raises():
    products = MagicMock()
    products.object_id = 'obj'
    products.get_free_parameters.return_value = ['z']
    pres = ResultsPresentation(products)
    samples = _fake_samples(['other'], np.array([[1.0], [2.0]]))
    with patch.object(
        pres, '_load_getdist_samples',
        return_value=(samples, Path('/unused')),
    ):
        with pytest.raises(ValueError, match='no parameters available in posterior samples'):
            pres.plot_posterior(
                params=None, object_id='obj', show=False, save=False,
            )


def _sidecar_plot_env(names=('Av',)):
    products = MagicMock()
    products.object_id = 'obj'
    products.model_config = 'cfg'
    products.get_free_parameters.return_value = list(names)
    products.get_derived_parameters.return_value = list(names)
    products._custom_labels = None
    pres = ResultsPresentation(products)
    samples = _fake_samples(
        list(names),
        np.array([[1.0], [1.2], [0.8]]),
    )
    samples.numrows = 3
    samples.n = 1
    fake_g = MagicMock()
    sample_par = Path('/out/gal/obj/cfg_sample_par.txt')
    paramnames = Path('/out/gal/obj/cfg_sample_par.paramnames')
    return pres, samples, fake_g, sample_par, paramnames


def _patch_loaded_plot(pres, samples, fake_g, sample_par, paramnames):
    txt_calls = {'n': 0}

    def find_files(_object_id, pattern):
        if pattern == '*_sample_par.txt':
            txt_calls['n'] += 1
            if txt_calls['n'] == 1:
                return [sample_par]
            return [Path('/out/gal/obj/other_sample_par.txt')]
        return [paramnames]

    def select_unique(files, *_args, **_kwargs):
        return files[0]

    return (
        patch.object(pres, '_find_config_files', side_effect=find_files),
        patch.object(pres, '_select_unique_file', side_effect=select_unique),
        patch.object(pres, '_getdist_subplot_plotter', return_value=fake_g),
        patch('getdist.loadMCSamples', return_value=samples),
        patch('matplotlib.pyplot'),
    )


def test_plot_posterior_default_save_exports_sample_par_kind():
    pres, samples, fake_g, sample_par, paramnames = _sidecar_plot_env()
    patches = _patch_loaded_plot(pres, samples, fake_g, sample_par, paramnames)
    with patches[0], patches[1], patches[2], patches[3], patches[4]:
        pres.plot_posterior(params=['Av'], object_id='obj', show=False)
    assert getattr(samples, '_sample_par_path', None) is None
    fake_g.export.assert_called_once_with(
        str(Path('/out/gal/obj/cfg_posterior.png'))
    )


def test_plot_posterior_save_false_does_not_export():
    pres, samples, fake_g, sample_par, paramnames = _sidecar_plot_env()
    patches = _patch_loaded_plot(pres, samples, fake_g, sample_par, paramnames)
    with patches[0], patches[1], patches[2], patches[3], patches[4]:
        pres.plot_posterior(
            params=['Av'], object_id='obj', show=False, save=False,
        )
    fake_g.export.assert_not_called()


def test_plot_posterior_output_file_overrides_sidecar():
    pres, samples, fake_g, sample_par, paramnames = _sidecar_plot_env()
    patches = _patch_loaded_plot(pres, samples, fake_g, sample_par, paramnames)
    with patches[0], patches[1], patches[2], patches[3], patches[4]:
        pres.plot_posterior(
            params=['Av'], object_id='obj', show=False,
            output_file='/tmp/custom.png',
        )
    fake_g.export.assert_called_once_with('/tmp/custom.png')


def test_plot_posterior_free_default_save_uses_free_kind():
    pres, samples, fake_g, sample_par, paramnames = _sidecar_plot_env()
    patches = _patch_loaded_plot(pres, samples, fake_g, sample_par, paramnames)
    with patches[0], patches[1], patches[2], patches[3], patches[4]:
        pres.plot_posterior_free(object_id='obj', show=False)
    fake_g.export.assert_called_once_with(
        str(Path('/out/gal/obj/cfg_posterior_free.png'))
    )


def test_plot_posterior_derived_default_save_uses_derived_kind():
    pres, samples, fake_g, sample_par, paramnames = _sidecar_plot_env()
    patches = _patch_loaded_plot(pres, samples, fake_g, sample_par, paramnames)
    with patches[0], patches[1], patches[2], patches[3], patches[4]:
        pres.plot_posterior_derived(object_id='obj', show=False)
    fake_g.export.assert_called_once_with(
        str(Path('/out/gal/obj/cfg_posterior_derived.png'))
    )
