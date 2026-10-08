"""Tests for plot_bestfit residual panel."""

from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from bayesed.plotting import plot_bestfit


def _model_hdu(name, wavelength, flux):
    columns = [
        fits.Column(name='wavelength_rest', format='D', array=wavelength),
        fits.Column(name='wavelength_obs', format='D', array=wavelength * 1.1),
        fits.Column(name='flux', format='D', array=flux),
    ]
    hdu = fits.BinTableHDU.from_columns(columns)
    hdu.header['EXTNAME'] = name
    return hdu


def _phot_hdu():
    n = 5
    wavelength_obs_center = np.linspace(0.5, 2.0, n)
    columns = [
        fits.Column(name='iselected', format='J', array=np.ones(n, dtype=int)),
        fits.Column(name='wavelength_obs_center', format='D', array=wavelength_obs_center),
        fits.Column(name='wavelength_rest_center', format='D', array=wavelength_obs_center / 1.1),
        fits.Column(name='flux_obs', format='D', array=np.linspace(10, 20, n)),
        fits.Column(name='flux_model', format='D', array=np.linspace(11, 19, n)),
        fits.Column(name='flux_obs_err', format='D', array=np.ones(n) * 2.0),
        fits.Column(name='flux_model_err', format='D', array=np.ones(n) * 0.5),
    ]
    hdu = fits.BinTableHDU.from_columns(columns)
    hdu.header['EXTNAME'] = 'obs:phot'
    return hdu


def _spec_hdu():
    n = 20
    wavelength_obs = np.linspace(0.4, 0.7, n)
    obs = np.sin(wavelength_obs * 10) + 10
    mod = obs + np.random.default_rng(0).normal(0, 0.5, n)
    columns = [
        fits.Column(name='iband', format='J', array=np.ones(n, dtype=int)),
        fits.Column(name='wavelength_obs', format='D', array=wavelength_obs),
        fits.Column(name='wavelength_rest', format='D', array=wavelength_obs / 1.1),
        fits.Column(name='spectra_obs', format='D', array=obs),
        fits.Column(name='spectra_mod', format='D', array=mod),
        fits.Column(name='spectra_obs_err', format='D', array=np.ones(n) * 0.5),
    ]
    hdu = fits.BinTableHDU.from_columns(columns)
    hdu.header['EXTNAME'] = 'obs:spec'
    return hdu


def _write_bestfit(path):
    hdul = fits.HDUList([
        fits.PrimaryHDU(),
        _model_hdu('model:stellar', np.linspace(0.1, 3.0, 100), np.ones(100) * 15),
        _model_hdu('model:total', np.linspace(0.1, 3.0, 100), np.ones(100) * 15),
        _phot_hdu(),
        _spec_hdu(),
    ])
    hdul.writeto(path, overwrite=True)
    hdul.close()


def test_show_residuals_false_keeps_single_axes(tmp_path):
    fits_path = Path(tmp_path) / 'bestfit.fits'
    _write_bestfit(fits_path)
    fig = plot_bestfit(
        fits_path, show=False, save=False, show_emission_lines=False,
    )
    assert len(fig.axes) == 1


def test_show_residuals_true_adds_residual_axes(tmp_path):
    fits_path = Path(tmp_path) / 'bestfit.fits'
    _write_bestfit(fits_path)
    fig = plot_bestfit(
        fits_path, show=False, save=False, show_emission_lines=False,
        show_residuals=True,
    )
    assert len(fig.axes) == 2

    ax_main, ax_resid = fig.axes
    assert 'obs - mod' in ax_resid.get_ylabel()
    assert '/ err' in ax_resid.get_ylabel()

    # Zero line and ±1 sigma reference lines should be present
    hlines = [line for line in ax_resid.lines if len(line.get_xdata()) == 2]
    y_values = {line.get_ydata()[0] for line in hlines}
    assert 0.0 in y_values
    assert 1.0 in y_values
    assert -1.0 in y_values


def test_residual_ylim_defaults_to_at_least_three_sigma(tmp_path):
    fits_path = Path(tmp_path) / 'bestfit.fits'
    _write_bestfit(fits_path)
    fig = plot_bestfit(
        fits_path, show=False, save=False, show_emission_lines=False,
        show_residuals=True,
    )
    ax_resid = fig.axes[1]
    ymin, ymax = ax_resid.get_ylim()
    assert ymin <= -3.0
    assert ymax >= 3.0


def test_residual_ylim_can_be_overridden(tmp_path):
    fits_path = Path(tmp_path) / 'bestfit.fits'
    _write_bestfit(fits_path)
    fig = plot_bestfit(
        fits_path, show=False, save=False, show_emission_lines=False,
        show_residuals=True, residual_ylim=(-1.5, 1.5),
    )
    ax_resid = fig.axes[1]
    assert ax_resid.get_ylim() == pytest.approx((-1.5, 1.5))


def test_residual_height_ratio_changes_gridspec(tmp_path):
    fits_path = Path(tmp_path) / 'bestfit.fits'
    _write_bestfit(fits_path)
    fig = plot_bestfit(
        fits_path, show=False, save=False, show_emission_lines=False,
        show_residuals=True, residual_height_ratio=3,
    )
    gs = fig.axes[0].get_gridspec()
    ratios = gs.get_height_ratios()
    assert ratios[0] / ratios[1] == pytest.approx(3.0)


def test_show_residuals_with_no_data_is_silent(tmp_path):
    """No phot/spec data: show_residuals=True should fall back to single axes."""
    fits_path = Path(tmp_path) / 'bestfit.fits'
    hdul = fits.HDUList([
        fits.PrimaryHDU(),
        _model_hdu('model:total', np.linspace(0.1, 3.0, 100), np.ones(100) * 15),
    ])
    hdul.writeto(fits_path, overwrite=True)
    hdul.close()

    fig = plot_bestfit(
        fits_path, show=False, save=False, show_emission_lines=False,
        show_residuals=True,
    )
    assert len(fig.axes) == 1


def _residual_panel_y_values(fig):
    """Collect residual data y-values from ErrorbarContainers (skip hlines)."""
    ax_resid = fig.axes[1]
    chunks = []
    for container in ax_resid.containers:
        data_line = getattr(container, 'lines', (None,))[0]
        if data_line is None:
            continue
        chunks.append(np.asarray(data_line.get_ydata(), dtype=float))
    if not chunks:
        return np.array([], dtype=float)
    return np.concatenate(chunks)


def test_residuals_invariant_under_flux_unit(tmp_path):
    """(obs-mod)/err must not change when flux_unit converts the SED panel."""
    fits_path = Path(tmp_path) / 'bestfit.fits'
    _write_bestfit(fits_path)
    common = dict(
        show=False, save=False, show_emission_lines=False, show_residuals=True,
    )
    fig_fnu = plot_bestfit(fits_path, flux_unit='fnu', **common)
    fig_nufnu = plot_bestfit(fits_path, flux_unit='nufnu', **common)
    fig_flambda = plot_bestfit(fits_path, flux_unit='flambda', **common)

    y_fnu = _residual_panel_y_values(fig_fnu)
    y_nufnu = _residual_panel_y_values(fig_nufnu)
    y_flambda = _residual_panel_y_values(fig_flambda)

    assert y_fnu.size > 0
    assert y_fnu == pytest.approx(y_nufnu, rel=1e-9, abs=1e-9)
    assert y_fnu == pytest.approx(y_flambda, rel=1e-9, abs=1e-9)


def test_residual_panels_have_no_gap(tmp_path):
    """Residual panel must sit flush against the main panel (hspace=0)."""
    fits_path = Path(tmp_path) / 'bestfit.fits'
    _write_bestfit(fits_path)
    fig = plot_bestfit(
        fits_path, show=False, save=False, show_emission_lines=False,
        show_residuals=True,
    )
    ax_main, ax_resid = fig.axes
    main_bbox = ax_main.get_position()
    resid_bbox = ax_resid.get_position()
    assert resid_bbox.y1 == pytest.approx(main_bbox.y0, abs=1e-6)


def test_xlabel_only_on_residual_panel(tmp_path):
    """With residuals: xlabel on bottom panel; main panel hides x tick labels."""
    fits_path = Path(tmp_path) / 'bestfit.fits'
    _write_bestfit(fits_path)
    fig = plot_bestfit(
        fits_path, show=False, save=False, show_emission_lines=False,
        show_residuals=True,
    )
    ax_main, ax_resid = fig.axes
    assert ax_main.get_xlabel() == ''
    assert 'rest' in ax_resid.get_xlabel() or 'obs' in ax_resid.get_xlabel()
    assert all(
        not tick.get_visible() for tick in ax_main.get_xticklabels()
    )


def test_single_panel_keeps_xlabel_on_main(tmp_path):
    """Without residuals the single panel keeps its own xlabel."""
    fits_path = Path(tmp_path) / 'bestfit.fits'
    _write_bestfit(fits_path)
    fig = plot_bestfit(
        fits_path, show=False, save=False, show_emission_lines=False,
        show_residuals=False,
    )
    ax_main = fig.axes[0]
    assert 'rest' in ax_main.get_xlabel() or 'obs' in ax_main.get_xlabel()
