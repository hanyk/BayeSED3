"""Rest-frame emission line table shared across BayeSED3.

Single source of truth for the line list drawn by ``plot_bestfit`` and the
``line_residual_outliers`` fit-quality flag (ADR-0020).

Each entry is ``(wavelength_rest [μm], label, family)``. Colors are not
per-line: ``FAMILY_COLORS`` maps each family to one fixed color (assigned
deterministically from the tab20 palette in first-occurrence table order),
so lines of the same family always share a color.
"""

import numpy as np
from matplotlib import colormaps
from matplotlib.colors import to_hex

# (wavelength_rest [μm], label, family)
EMISSION_LINES = [
    # Hydrogen Balmer series
    (0.65646140, r'H$\alpha$', 'balmer'),
    (0.48626830, r'H$\beta$', 'balmer'),
    (0.43407410, r'H$\gamma$', 'balmer'),
    (0.41017500, r'H$\delta$', 'balmer'),
    (0.36460000, r'H$\infty$', 'balmer'),

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
