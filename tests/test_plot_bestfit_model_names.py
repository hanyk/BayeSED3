"""plot_bestfit model_names must match model: extensions, excluding model:total."""

from pathlib import Path

import numpy as np
from astropy.io import fits

from bayesed.plotting import plot_bestfit


def _component_hdu(name):
    wavelength = np.array([1.0, 2.0])
    columns = [
        fits.Column(name='wavelength_rest', format='D', array=wavelength),
        fits.Column(name='wavelength_obs', format='D', array=wavelength * 1.1),
        fits.Column(name='flux', format='D', array=np.array([1.0, 2.0])),
    ]
    hdu = fits.BinTableHDU.from_columns(columns)
    hdu.header['EXTNAME'] = name
    return hdu


def _write_bestfit(path):
    hdul = fits.HDUList([
        fits.PrimaryHDU(),
        _component_hdu('model:bbb_2dal7[2,1]'),
        _component_hdu('model:BLR_kin_eml3[3,1]'),
        _component_hdu('model:total'),
    ])
    hdul.writeto(path, overwrite=True)
    hdul.close()


def _plot(path, model_names):
    return plot_bestfit(
        path, show=False, save=False, show_emission_lines=False,
        show_total=False, model_names=model_names,
    )


def test_model_names_length_must_match_components(tmp_path):
    fits_path = Path(tmp_path) / 'bestfit.fits'
    _write_bestfit(fits_path)
    for names in (['AGN-Disk'], ['AGN-Disk', 'AGN-BLR', 'extra']):
        try:
            _plot(fits_path, names)
        except ValueError as exc:
            assert '2 model components' in str(exc)
            assert f'{len(names)} names' in str(exc)
        else:
            raise AssertionError(f'expected ValueError for {names!r}')


def test_model_names_matching_length_draws_those_labels(tmp_path):
    fits_path = Path(tmp_path) / 'bestfit.fits'
    _write_bestfit(fits_path)
    fig = _plot(fits_path, ['Disk', 'BLR'])
    labels = [line.get_label() for line in fig.axes[0].lines]
    assert labels == ['Disk', 'BLR']


def test_model_names_none_keeps_auto_names(tmp_path):
    fits_path = Path(tmp_path) / 'bestfit.fits'
    _write_bestfit(fits_path)
    fig = _plot(fits_path, None)
    labels = [line.get_label() for line in fig.axes[0].lines]
    assert 'AGN-Disk' in labels
    assert 'AGN-BLR' in labels
