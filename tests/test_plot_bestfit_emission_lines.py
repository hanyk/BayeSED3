"""Tests for emission-line label placement in plot_bestfit."""

from pathlib import Path

import numpy as np
from astropy.io import fits
from matplotlib.backends.backend_agg import FigureCanvasAgg

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


def _write_bestfit(path, wl_max=3.0):
    hdul = fits.HDUList([
        fits.PrimaryHDU(),
        _model_hdu('model:stellar', np.linspace(0.1, wl_max, 100), np.ones(100) * 15),
        _model_hdu('model:total', np.linspace(0.1, wl_max, 100), np.ones(100) * 15),
    ])
    hdul.writeto(path, overwrite=True)
    hdul.close()


def _emission_line_texts(fig):
    ax1 = fig.axes[0]
    return ax1, ax1.texts


def test_labels_hang_beside_lines(tmp_path):
    """Labels are anchored on the axes-fraction transform and grow downward
    from the top, beside (not on top of) each vertical line."""
    fits_path = Path(tmp_path) / 'bestfit.fits'
    _write_bestfit(fits_path)
    fig = plot_bestfit(fits_path, show=False, save=False, show_emission_lines=True)
    ax1, texts = _emission_line_texts(fig)
    assert texts, 'expected at least one emission-line label'
    xaxis_transform = ax1.get_xaxis_transform()
    for text in texts:
        assert text.get_rotation() == 90
        assert text.get_verticalalignment() == 'top'
        assert text.get_horizontalalignment() in ('left', 'right')
        # anchor y stored via Annotation.xy in axes-fraction coords (0-1)
        _, y = text.xy
        assert 0.0 < y <= 1.0
        assert text.xycoords == xaxis_transform


def _display_frac(ax1, x):
    """Fraction of the displayed x range at x, honouring the axis scale."""
    xlim = ax1.get_xlim()
    if ax1.get_xscale() == 'log':
        return ((np.log10(x) - np.log10(xlim[0]))
                / (np.log10(xlim[1]) - np.log10(xlim[0])))
    return (x - xlim[0]) / (xlim[1] - xlim[0])


def test_labels_anchor_near_top_by_default(tmp_path):
    """Labels hang from the top of the axes: the default anchor sits at the
    very top so text grows downward from the top edge."""
    fits_path = Path(tmp_path) / 'bestfit.fits'
    _write_bestfit(fits_path)
    fig = plot_bestfit(fits_path, show=False, save=False, show_emission_lines=True)
    _, texts = _emission_line_texts(fig)
    assert texts
    for text in texts:
        _, y = text.xy
        assert y >= 0.9, f'label anchor {y} is below the top of the axes'


def test_labels_stay_inside_axes_after_draw(tmp_path):
    """After a full draw, no emission-line label extends above the axes —
    regression guard for the rotation_mode anchor bug (labels grew upward
    out of the figure)."""
    fits_path = Path(tmp_path) / 'bestfit.fits'
    _write_bestfit(fits_path)
    fig = plot_bestfit(fits_path, show=False, save=False, show_emission_lines=True)
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    ax1, texts = _emission_line_texts(fig)
    assert texts
    axes_top = ax1.get_window_extent(canvas.get_renderer()).y1
    for text in texts:
        text_top = text.get_window_extent(canvas.get_renderer()).y1
        assert text_top <= axes_top + 0.5, (
            f'label {text.get_text()!r} extends above the axes: '
            f'{text_top} > {axes_top}'
        )


def test_label_flips_side_near_right_edge(tmp_path):
    """The rightmost labelled line hangs its label on the left side of the
    vertical line so the text stays inside the axes."""
    fits_path = Path(tmp_path) / 'bestfit.fits'
    # Pa alpha at 1.875 um sits in the rightmost 10% of the display range
    # when the model data ends at 1.95 um.
    _write_bestfit(fits_path, wl_max=1.95)
    fig = plot_bestfit(fits_path, show=False, save=False, show_emission_lines=True)
    ax1, texts = _emission_line_texts(fig)
    assert texts
    rightmost = max(texts, key=lambda t: t.xy[0])
    assert _display_frac(ax1, rightmost.xy[0]) > 0.9, \
        'precondition: rightmost label is in the edge zone'
    assert rightmost.get_horizontalalignment() == 'right'
    others = [t for t in texts if t is not rightmost]
    assert others, 'expected labelled lines outside the edge zone too'
    for text in others:
        expected = 'right' if _display_frac(ax1, text.xy[0]) > 0.9 else 'left'
        assert text.get_horizontalalignment() == expected


def test_label_side_follows_axis_scale_not_flag(tmp_path):
    """``use_log_scale='y'`` keeps the x-axis linear: label side must follow
    the actual axis scale, not the truthiness of the flag (bugbot finding:
    log-fraction math on a linear axis flipped [Fe II] to the wrong side)."""
    fits_path = Path(tmp_path) / 'bestfit.fits'
    _write_bestfit(fits_path, wl_max=1.95)
    fig = plot_bestfit(fits_path, show=False, save=False,
                       show_emission_lines=True, use_log_scale='y')
    ax1, texts = _emission_line_texts(fig)
    fe2 = next(t for t in texts if t.get_text() == '[Fe II]1.644')
    # On the linear axis [Fe II] sits below the 90% edge; only log math
    # applied to a linear axis would push it past 0.9.
    assert _display_frac(ax1, fe2.xy[0]) < 0.9
    assert fe2.get_horizontalalignment() == 'left'
