"""Rest-frame emission line table shared across BayeSED3.

Single source of truth for the line list drawn by ``plot_bestfit`` and the
``line_residual_outliers`` fit-quality flag (ADR-0020).

Each entry is ``(wavelength_rest [μm], label, family)``. Colors are not
per-line: ``FAMILY_COLORS`` maps each family to one fixed color (assigned
deterministically from the tab20 palette in first-occurrence table order),
so lines of the same family always share a color.
"""

import numpy as np
from astropy import units as u
from astropy.nddata import StdDevUncertainty
from matplotlib import colormaps
from matplotlib.colors import to_hex
from specutils import Spectrum, conf
from specutils.fitting import find_lines_threshold

# (wavelength_rest [μm], label, family)
EMISSION_LINES = [
    # Hydrogen Balmer series
    (0.65646140, r'H$\alpha$ 6565', 'balmer'),
    (0.48626830, r'H$\beta$ 4863', 'balmer'),
    (0.43407410, r'H$\gamma$ 4340', 'balmer'),
    (0.41017500, r'H$\delta$ 4102', 'balmer'),
    (0.36460000, r'H$\infty$ 3646', 'balmer'),

    # Hydrogen Paschen series (near-IR)
    (1.00500000, 'Paδ 1.005', 'paschen'),
    (1.09400000, 'Paγ 1.094', 'paschen'),
    (1.28200000, 'Paβ 1.282', 'paschen'),
    (1.87500000, 'Paα 1.875', 'paschen'),

    # Brackett series (near-IR)
    (2.16600000, 'Brγ 2.166', 'brackett'),
    (4.05200000, 'Brα 4.052', 'brackett'),

    # Oxygen lines
    (0.14020000, 'O IV] 1402', 'oxygen'),
    (0.16610000, 'O III] 1661', 'oxygen'),
    (0.16660000, 'O III] 1666', 'oxygen'),
    (0.49602949, '[O III]4959', 'oxygen'),
    (0.50082397, '[O III]5007', 'oxygen'),
    (0.43632612, '[O III]4363', 'oxygen'),
    (0.37270917, '[O II]3726', 'oxygen'),
    (0.37298754, '[O II]3729', 'oxygen'),
    (0.63004000, '[O I]6300', 'oxygen'),
    (0.63640000, '[O I]6364', 'oxygen'),
    (0.73200000, '[O II]7320', 'oxygen'),
    (0.73300000, '[O II]7330', 'oxygen'),

    # Nitrogen lines
    (0.12390000, 'N V 1240', 'nitrogen'),  # 1239/1243 doublet, blended
    (0.14860000, 'N IV] 1486', 'nitrogen'),
    (0.17500000, 'N III] 1750', 'nitrogen'),
    (0.65498590, '[N II]6548', 'nitrogen'),
    (0.65852685, '[N II]6583', 'nitrogen'),
    (0.51983000, '[N I]5199', 'nitrogen'),

    # Sulfur lines
    (0.67316300, '[S II]6716', 'sulfur'),
    (0.67312100, '[S II]6731', 'sulfur'),
    (0.40686000, '[S II]4069', 'sulfur'),
    (0.40763500, '[S II]4076', 'sulfur'),
    (0.90687000, '[S III]9069', 'sulfur'),
    (0.95323000, '[S III]9532', 'sulfur'),
    (0.63120000, '[S III]6312', 'sulfur'),
    (1.02870000, '[S II]10287', 'sulfur'),
    (1.03360000, '[S II]10336', 'sulfur'),

    # Argon lines (HII regions)
    (0.47110000, '[Ar IV]4711', 'argon'),
    (0.47400000, '[Ar IV]4740', 'argon'),
    (0.71360000, '[Ar III]7136', 'argon'),
    (0.77510000, '[Ar III]7751', 'argon'),

    # Carbon lines
    (0.23250000, 'C II] 2325', 'carbon'),
    (0.15491000, 'C IV 1549', 'carbon'),
    (0.19062000, 'C III] 1909', 'carbon'),

    # Magnesium lines
    (0.27980000, 'Mg II 2798', 'magnesium'),
    (0.28035300, 'Mg II 2803', 'magnesium'),

    # Neon lines
    (0.33460000, '[Ne V]3346', 'neon'),
    (0.38691000, '[Ne III]3869', 'neon'),
    (0.39685000, '[Ne III]3967', 'neon'),
    (0.34260000, '[Ne V]3426', 'neon'),
    (0.24240000, '[Ne IV]2424', 'neon'),

    # Silicon lines
    (0.19347000, 'Si III] 1892', 'silicon'),
    (0.14003000, 'Si IV 1394', 'silicon'),
    (0.14082000, 'Si IV 1403', 'silicon'),

    # Helium lines
    (0.40260000, 'He I 4026', 'helium'),
    (0.50150000, 'He I 5015', 'helium'),
    (0.54120000, 'He II 5412', 'helium'),
    (1.08300000, 'He I 10830', 'helium'),
    (0.58756000, 'He I 5876', 'helium'),
    (0.44713000, 'He II 4471', 'helium'),
    (0.38890000, 'He I 3889', 'helium'),
    (0.66780000, 'He I 6678', 'helium'),
    (0.70650000, 'He I 7065', 'helium'),
    (0.46860000, 'He II 4686', 'helium'),
    (0.47130000, 'He I 4713', 'helium'),
    (0.49220000, 'He I 4922', 'helium'),
    (0.16400000, 'He II 1640', 'helium'),

    # Iron lines (common in AGN)
    (0.46580000, '[Fe III]4658', 'iron'),
    (0.42587000, '[Fe II]4259', 'iron'),
    (0.51270000, '[Fe II]5127', 'iron'),
    (0.16300000, '[Fe II]1630', 'iron'),
    (1.25700000, '[Fe II]1.257', 'iron'),
    (1.64400000, '[Fe II]1.644', 'iron'),

    # AGN coronal lines
    (0.57210000, '[Fe VII]5721', 'coronal'),
    (0.60870000, '[Fe VII]6087', 'coronal'),
    (0.63750000, '[Fe X]6375', 'coronal'),

    # Lyman series
    (0.12157000, r'Ly$\alpha$ 1216', 'lyman'),
    (0.10260000, r'Ly$\beta$ 1026', 'lyman'),

    # Chlorine lines
    (0.55180000, '[Cl III]5518', 'chlorine'),
    (0.55380000, '[Cl III]5538', 'chlorine'),

    # Calcium lines
    (0.39340000, 'Ca II K 3934', 'calcium'),
    (0.39690000, 'Ca II H 3969', 'calcium'),
    (0.72910000, '[Ca II]7291', 'calcium'),
    (0.73240000, '[Ca II]7324', 'calcium'),
    (0.85446000, 'Ca II 8542', 'calcium'),
    (0.85662000, 'Ca II 8662', 'calcium'),

    # Molecular hydrogen (near-IR, shocks / star formation)
    (2.12180000, 'H2 2.122', 'molecular'),
]

# Fixed family -> color mapping, assigned deterministically from the tab20
# palette in first-occurrence table order (17 families < 20 palette colors,
# so families never collide). tab20's mid-luminance colors stay readable on
# both light and dark backgrounds.
FAMILY_COLORS = {
    family: to_hex(colormaps['tab20'](index % 20))
    for index, family in enumerate(
        dict.fromkeys(family for _, _, family in EMISSION_LINES))
}

# Half-width (μm) of the window around a line for outlier flagging.
LINE_MATCH_WINDOW = 0.005

# Unique rest wavelengths, sorted, for window matching.
EMISSION_LINE_WAVELENGTHS = np.array(
    sorted({wl for wl, _, _ in EMISSION_LINES}))


def near_emission_lines(wl_rest):
    """Boolean mask: rest wavelengths within ``LINE_MATCH_WINDOW`` of a line."""
    wl = np.asarray(wl_rest, dtype=float)
    if wl.size == 0:
        return np.zeros(0, dtype=bool)
    return np.any(
        np.abs(wl[:, None] - EMISSION_LINE_WAVELENGTHS[None, :])
        <= LINE_MATCH_WINDOW,
        axis=1,
    )


C_KMS = 299792.458


def detect_lines(wl, flux, err, snr_threshold=3.0, window_kms=3000.0):
    """Return the ``EMISSION_LINES`` subset detected in a spectrum.

    Continuum-subtracts ``flux`` with a rolling 25th percentile, then
    delegates to ``specutils.fitting.find_lines_threshold`` (with the
    per-pixel ``err`` as uncertainty, ``noise_factor=snr_threshold``). A
    candidate line counts as detected when a found emission-line center
    falls within its ±``window_kms`` velocity window (3000 covers QSO
    broad lines). Spectrum and line table must share a frame.
    """
    wl = np.asarray(wl, dtype=float)
    flux = np.asarray(flux, dtype=float)
    err = np.asarray(err, dtype=float)
    if wl.size < 10:
        return []

    order = np.argsort(wl)
    wl, flux, err = wl[order], flux[order], err[order]
    valid = np.isfinite(wl) & np.isfinite(flux) & np.isfinite(err) & (err > 0)
    wl, flux, err = wl[valid], flux[valid], err[valid]
    if wl.size < 10:
        return []

    # Rolling-quantile continuum subtraction, alfa/PyEMILI style (Wesson
    # 2016, MNRAS 456, 3774; Tu et al. 2025, PyEMILI): in an emission-line-
    # dominated spectrum the median is pulled up by line flux, suppressing
    # faint lines, so the 25th percentile is used to track the true
    # inter-line continuum. The window should swallow the broadest lines
    # (2 * window_kms) while staying local; ~25x the line window in pixels
    # is a good middle ground. Pure numpy via windows.
    half_px = max(2, int(np.ceil(window_kms / _pixel_velocity_kms(wl))))
    cont_win = min(25 * (2 * half_px + 1), (wl.size // 4) * 2 + 1)
    if cont_win % 2 == 0:
        cont_win += 1
    cont_win = max(cont_win, 11)
    cont = _rolling_percentile(flux, cont_win)

    spec = Spectrum(
        spectral_axis=wl * u.um,
        flux=(flux - cont) * u.dimensionless_unscaled,
        uncertainty=StdDevUncertainty(err),
    )
    # We always pass an explicit uncertainty, so the specutils warning
    # about which continuum-fitting function was used is irrelevant here;
    # scope the suppression to this call instead of process-wide config.
    with conf.set_temp('do_continuum_function_check', False):
        found = find_lines_threshold(spec, noise_factor=snr_threshold)
    if len(found) == 0:
        return []
    center_um = found['line_center'].to_value(u.um)
    center_um = center_um[np.asarray(found['line_type']) == 'emission']
    if center_um.size == 0:
        return []

    detected = []
    for line_wl, label, family in EMISSION_LINES:
        dwl = line_wl * window_kms / C_KMS
        if np.any(np.abs(center_um - line_wl) <= dwl):
            detected.append((line_wl, label, family))
    return detected


def _pixel_velocity_kms(wl):
    """Median per-pixel velocity step of a sorted (possibly non-uniform) grid."""
    dlog = np.diff(np.log(wl))
    dlog = dlog[np.isfinite(dlog) & (dlog > 0)]
    if dlog.size == 0:
        return C_KMS / 2000.0  # fallback: assume R ~ 2000
    return float(np.median(dlog) * C_KMS)


def _rolling_percentile(a, win, q=25):
    """Rolling q-th percentile of 1-D array ``a`` (odd ``win``, edge-padded)."""
    pad = win // 2
    padded = np.pad(a, pad, mode='edge')
    windows = np.lib.stride_tricks.sliding_window_view(padded, win)
    return np.percentile(windows, q, axis=-1)
