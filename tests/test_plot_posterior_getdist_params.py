"""ADR-0010: plot_posterior params vs GetDist samples."""

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
    with patch.object(pres, 'get_getdist_samples', return_value=samples):
        with pytest.raises(ValueError, match=r"parameter\(s\) not in posterior samples: z"):
            pres.plot_posterior(params=['z', 'Av'], object_id='obj', show=False)


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
    with patch.object(pres, 'get_getdist_samples', return_value=samples), \
         patch.object(pres, '_getdist_subplot_plotter', return_value=fake_g), \
         patch('matplotlib.pyplot'):
        pres.plot_posterior(params=None, object_id='obj', show=False)
    fake_g.plot_1d.assert_called_once()
    assert fake_g.plot_1d.call_args.args[1] == 'Av'


def test_plot_posterior_none_empty_raises():
    products = MagicMock()
    products.object_id = 'obj'
    products.get_free_parameters.return_value = ['z']
    pres = ResultsPresentation(products)
    samples = _fake_samples(['other'], np.array([[1.0], [2.0]]))
    with patch.object(pres, 'get_getdist_samples', return_value=samples):
        with pytest.raises(ValueError, match='no parameters available in posterior samples'):
            pres.plot_posterior(params=None, object_id='obj', show=False)
