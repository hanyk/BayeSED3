"""Tests for the family-based emission line color mapping."""

from bayesed.emission_lines import EMISSION_LINES, FAMILY_COLORS


def test_every_family_has_a_color():
    families = {family for _, _, family in EMISSION_LINES}
    assert families <= set(FAMILY_COLORS), 'every line family must have a color'


def test_family_colors_are_unique():
    assert len(FAMILY_COLORS) == len(set(FAMILY_COLORS.values())), \
        'two families sharing one color would be indistinguishable'


def test_family_color_assignment_is_deterministic():
    """Colors are assigned by family in first-occurrence table order, so the
    mapping is stable across runs and as long as table order is preserved."""
    from matplotlib import colormaps
    from matplotlib.colors import to_hex
    expected = {}
    for _, _, family in EMISSION_LINES:
        if family not in expected:
            expected[family] = to_hex(colormaps['tab20'](len(expected) % 20))
    assert FAMILY_COLORS == expected
