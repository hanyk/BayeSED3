"""
Plotting utilities for BayeSED3.
"""

import re

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

from .emission_lines import C_KMS, EMISSION_LINES, FAMILY_COLORS, detect_lines


def _draw_residual_panel(ax_resid, phot_points, spec_points, residual_ylim,
                         label_fontsize):
    """Draw the residual panel shared by ``plot_bestfit``.

    Parameters
    ----------
    ax_resid : matplotlib.axes.Axes
        The residual axes; shares its x-axis with the main SED panel.
    phot_points : list of (wavelength, residual) arrays
        Photometric residual points.
    spec_points : list of (wavelength, residual, color) tuples
        Spectroscopic residual points per band.
    residual_ylim : tuple or None
        Manual y-axis limits; None means auto-scale with at least [-3, 3].
    label_fontsize : int
        Font size for axis labels.

    Notes
    -----
    Residuals are already normalized as (obs - mod) / err, so a per-point
    error bar would be a constant ±1 everywhere and carries no information;
    the integer-sigma gridlines below serve that role instead. Photometric
    residuals are sparse, so they are drawn as markers; spectroscopic
    residuals are dense (thousands of points), so each band is drawn as a
    thin line.
    """
    all_residuals = []

    for wl, resid in phot_points:
        if len(wl) == 0:
            continue
        ax_resid.plot(
            wl, resid,
            'o', color='red', markersize=3, alpha=0.7,
        )
        all_residuals.append(resid)

    for wl, resid, color in spec_points:
        if len(wl) == 0:
            continue
        ax_resid.plot(
            wl, resid,
            '-', color=color, linewidth=0.8, alpha=0.7,
        )
        all_residuals.append(resid)

    # x-axis label is handled by the shared axis; residual panel only needs y label
    ax_resid.set_ylabel('(obs - mod) / err', fontsize=label_fontsize)

    if residual_ylim is not None:
        ax_resid.set_ylim(residual_ylim)
    elif all_residuals:
        all_resid = np.concatenate(all_residuals)
        rmin, rmax = np.nanmin(all_resid), np.nanmax(all_resid)
        ymin = min(rmin, -3.0)
        ymax = max(rmax, 3.0)
        # Add small padding (5% or 0.25 sigma, whichever is larger)
        pad = max(0.25, 0.05 * (ymax - ymin))
        ax_resid.set_ylim(ymin - pad, ymax + pad)
    else:
        ax_resid.set_ylim(-3.5, 3.5)

    # Grid at 1-sigma intervals replaces explicit horizontal reference
    # lines: ticks at integer sigma make every gridline a sigma level.
    # Only every Nth tick is labeled so the panel stays readable; empty
    # labels render as nothing. No span cap: the thin alpha=0.3 gridlines
    # stay legible even on wide ranges, and label thinning decouples
    # readability from grid density.
    ymin, ymax = ax_resid.get_ylim()
    ticks = np.arange(np.ceil(ymin), np.floor(ymax) + 0.5, 1.0)
    label_step = max(1, int(np.ceil((len(ticks) - 1) / 5)))
    ax_resid.set_yticks(ticks)
    ax_resid.set_yticklabels([
        str(int(t)) if int(t) % label_step == 0 else '' for t in ticks
    ])
    ax_resid.grid(True, alpha=0.3)


def _detect_band_lines(band_data, wl_spec, spec_obs,
                       min_line_snr, line_window_kms):
    """Run emission-line detection on one raw spectral band.

    Returns a set of ``(wavelength, label)`` pairs for detected lines.
    Raises KeyError when the band lacks an error column.
    """
    _det = detect_lines(
        wl_spec, spec_obs, band_data['spectra_obs_err'],
        snr_threshold=min_line_snr, window_kms=line_window_kms,
    )
    return {(lw, lab) for lw, lab, _fam in _det}


def _draw_emission_line_labels(ax1, visible_lines, xlim, log_x,
                               emission_line_fontsize, emission_line_ypos,
                               group_window_kms):
    """Draw guide lines and stacked/merged labels for emission lines.

    ``visible_lines`` is the sorted ``[(wl, label, color), ...]`` list of
    detected, in-range lines in rest-frame microns. A vertical dashed
    guide is drawn for every line; labels hang downward from the top of
    the panel. Lines whose velocity separation from the previous line is
    below ``group_window_kms`` belong to the same blended feature: their
    labels stack vertically in the same column and same-family members
    merge into one label ("[O III]4959,5007"). A group longer than four
    labels continues in the next column (never wrapping back).
    """

    def _label_ha(line_wl):
        # Flip the label to the left of the line when it sits in the
        # rightmost 10% of the displayed range, so the text stays inside
        # the axes on either scale.
        if log_x and xlim[0] > 0:
            frac = ((np.log10(line_wl) - np.log10(xlim[0]))
                    / (np.log10(xlim[1]) - np.log10(xlim[0])))
        else:
            frac = (line_wl - xlim[0]) / (xlim[1] - xlim[0])
        return 'right' if frac > 0.9 else 'left'

    def _split_label(label):
        """Split '[O III]4959' -> ('[O III]', '4959'); no number -> (label, None)."""
        m = re.match(r'^(\D+)(\d+)$', label)
        return (m.group(1), m.group(2)) if m else (label, None)

    # Geometry: a rotated-90° label is ~1.2 x fontsize wide in points, so
    # the horizontal slot it needs scales with the axes width in points —
    # larger figures / smaller fonts get proportionally tighter slots.
    axis_w_pt = ax1.get_window_extent().width * 72.0 / ax1.figure.dpi
    slot_frac = min(0.05, 1.2 * emission_line_fontsize / max(axis_w_pt, 1.0))
    if log_x and xlim[0] > 0:
        min_separation = (xlim[1] / xlim[0]) ** slot_frac

        def _fits(wl, last):
            return last is None or wl / last > min_separation

        def _anchor_x(wl, col):
            # Shift the anchor right by col label-widths (multiplicative).
            return wl * min_separation ** col
    else:
        # For linear scale, use absolute separation
        min_separation = (xlim[1] - xlim[0]) * slot_frac

        def _fits(wl, last):
            return last is None or wl - last > min_separation

        def _anchor_x(wl, col):
            return wl + col * min_separation

    # A rotated 90° label occupies about one line-height (~2% of the axes
    # width) of horizontal space. Column choice is pure collision
    # avoidance: each label goes into the LOWEST column whose last anchor
    # is horizontally clear, so labels hug their guide lines unless that
    # column is genuinely occupied nearby. Vertical stacking (yoff) is
    # group-scoped: lines closer than min_separation to the previous line
    # form one blended feature whose labels hang below one another in the
    # same column; same-family members merge into one label
    # ("[O III]4959,5007"). A group longer than max_stack labels continues
    # in the next column. Same-column labels less than min_separation
    # apart are always same-group stacked, so they never overlap.
    n_label_cols = 3
    max_stack = 4
    col_last_anchor = [None] * n_label_cols  # anchor x of last label per column
    group_col = 0
    stack_pts = 0.0
    count = 0
    prev_wl = None
    merged = None  # state of the label being merged, if any
    filtered_lines = []  # [wl, label, color, col, yoff_pts]
    for wl, label, color in visible_lines:
        # Rendered length: mathtext like 'H$\beta$' shows as ~1 glyph, so
        # count it as one character instead of its raw source length.
        render_len = len(re.sub(r'\$[^$]*\$', 'X', label))
        label_h = (render_len * 0.55 + 2.0) * emission_line_fontsize
        # Blended-feature grouping is a PHYSICS criterion, not a geometric
        # one: lines closer than group_window_kms in velocity space have
        # overlapping line profiles and cannot be cleanly separated.
        same_group = (prev_wl is not None
                      and (wl - prev_wl) / prev_wl * C_KMS < group_window_kms)
        name, num = _split_label(label)
        if (same_group and merged is not None
                and merged['color'] == color
                and merged['name'] == name and num):
            # Same family, same blended group: fold the new wavelength
            # into the existing label.
            merged['text'] += ',' + num
            filtered_lines[merged['idx']][1] = merged['text']
            stack_pts += (1 + len(num)) * 0.55 * emission_line_fontsize
            prev_wl = wl
            continue
        if same_group and merged is not None:
            merged = None  # different family: start a new label
        if same_group:
            if count >= max_stack:
                # Group overflow: continue it in the next column (never
                # wraps back — the occupied columns stay occupied).
                group_col = min(group_col + 1, n_label_cols - 1)
                stack_pts = 0.0
                count = 0
            col = group_col
        else:
            # New blended feature: lowest column whose last anchor is
            # horizontally clear of this label's anchor.
            col = 0
            while (col < n_label_cols - 1
                   and col_last_anchor[col] is not None
                   and not _fits(_anchor_x(wl, col), col_last_anchor[col])):
                col += 1
            group_col = col
            stack_pts = 0.0
            count = 0
        col_last_anchor[col] = _anchor_x(wl, col)
        if num:
            merged = {'idx': len(filtered_lines), 'name': name,
                      'color': color, 'text': label}
        else:
            merged = None
        filtered_lines.append([wl, label, color, col, stack_pts])
        stack_pts += label_h
        count += 1
        prev_wl = wl

    # Vertical guide line for every visible emission line
    for line_wl, _, color in visible_lines:
        ax1.axvline(x=line_wl, color=color, linestyle='--',
                    linewidth=0.8, alpha=0.7, zorder=1)

    # Add labels hanging beside the line, anchored near the top. Default
    # rotation mode applies va/ha to the rotated bbox, so va='top' makes
    # the text grow DOWNWARD from the anchor (rotation_mode='anchor' would
    # grow it upward past the top). Stacked labels are shifted down by
    # yoff points via an offset transform, which is frame-independent and
    # exact.
    for line_wl, line_label, color, col, yoff in filtered_lines:
        ax1.annotate(
            line_label,
            xy=(_anchor_x(line_wl, col), emission_line_ypos),
            xycoords=ax1.get_xaxis_transform(),
            xytext=(0, -yoff),
            textcoords='offset points',
            rotation=90,
            verticalalignment='top',
            horizontalalignment=_label_ha(line_wl),
            fontsize=emission_line_fontsize, color=color, alpha=0.9,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      alpha=0.8, edgecolor='none'),
            zorder=2)


def plot_bestfit(fits_file, output_file=None, show=True, *,
                 save=True,

                 filter_file=None, filter_selection_file=None,

                 use_rest_frame=True, flux_unit='fnu', use_log_scale=None,

                 model_names=None, show_emission_lines=True, show_components=True,
                 show_total=True, emission_line_fontsize=12, emission_line_ypos=0.98,

                 title_fontsize=16, label_fontsize=16, legend_fontsize=14,

                 figsize=(12, 8), dpi=300, focus_on_data_range=True,
                 spec_obs_labels=None, spec_mod_labels=None,

                 show_residuals=False, residual_height_ratio=4, residual_ylim=None,
                 min_line_snr=2.5, line_window_kms=3000.0,
                 **kwargs):

    """

    Plot best-fit SED from BayeSED FITS file.

    

    This is a general-purpose plotting function that handles various data types

    (photometry, spectroscopy) and supports customization options. It can be used

    standalone or via BayeSEDResults.plot_bestfit().

    

    Parameters

    ----------

    fits_file : str

        Path to FITS file containing best-fit results

    output_file : str, optional

        Output file path for saving the plot. If None, saves as {fits_file}.png

    show : bool

        Whether to display the plot (default: True)

    filter_file : str, optional

        Path to filter response file for overlay

    filter_selection_file : str, optional

        Path to filter selection file (filters_selected format)

    use_rest_frame : bool

        Use rest-frame wavelengths (default: True). If False, uses observed-frame

    flux_unit : str

        Flux unit: 'fnu' (μJy), 'nufnu' (νFν in μJy*Hz), or 'flambda' (default: 'fnu')

    use_log_scale : bool or str, optional
        Use logarithmic scale for axes. If None (default), auto-detects based on data range.
        Auto-detection uses log scale when either axis spans more than 1 order of magnitude
        (range ratio > 10). If negative values are present, defaults to linear scale.
        Valid values: True (both axes), False (neither), 'x' (log x only), 'y' (log y only), 'both' (same as True).

    model_names : list of str, optional

        Legend strings for model components, in ``model:`` extension order,
        excluding ``model:total``. ``None`` auto-generates one name per extension.
        Any other length raises ``ValueError`` before a curve is drawn.

    show_emission_lines : bool

        Show emission line markers for spectroscopy (default: True)

    show_components : bool or dict, default True

        Control visibility of individual model components.
        - True: show all components
        - False: hide all components
        - Dict: show/hide specific components by model name

    show_total : bool, default True

        Whether to plot the total model curve. Set False to hide.

    emission_line_fontsize : int, default 12

        Font size for emission line labels. Larger values make labels more readable.

    emission_line_ypos : float, default 0.98

        Vertical anchor for emission line labels (0-1 range, as fraction of
        axis height). Labels hang downward from this anchor, beside the line.

    title_fontsize : int, default 16

        Font size for the plot title

    label_fontsize : int, default 16

        Font size for axis labels (x and y axis)

    legend_fontsize : int, default 14

        Font size for legend text

    figsize : tuple, default (12, 8)

        Figure size (width, height) in inches (default: (12, 8))

    dpi : int

        Resolution for saved figure (default: 300)

    spec_obs_labels : dict, str, or None, optional

        Custom labels for observed spectra. Multiple bands supported via dict mapping

        band index to label string (e.g., {1: 'BOSS obs', 2: 'MUSE obs'}).

        Single string applies to first/only band. None uses default 'spec:obs_{band}'.

    spec_mod_labels : dict, str, or None, optional

        Custom labels for model spectra. Same format as spec_obs_labels.

        None uses default 'spec:mod' for the first band only; later bands

        share the same style and get no legend entry.

    focus_on_data_range : bool

        If True, set x-axis limits to focus on the wavelength range where data exists

        (photometry and spectroscopy), ignoring the full model range. If False, use

        the full wavelength range from both models and data (default: True)

    show_residuals : bool, default False

        If True, add a lower panel showing (observation - model) / error for

        photometry and spectroscopy data. Requires at least one ``obs:phot`` or

        ``obs:spec`` extension in the FITS file. The two panels share the

        wavelength axis and sit flush with no gap; the wavelength xlabel moves

        to the residual panel and the main panel hides its x tick labels.

        Photometric residuals are drawn as red markers and spectroscopic

        residuals as thin lines (one per band). No per-point error bars are

        drawn: residuals are already normalized by error, so the integer-sigma

        gridlines give the scale.

    residual_height_ratio : int, default 4

        Ratio of the main SED panel height to the residual panel height when

        ``show_residuals`` is True. A value of 4 gives a 4:1 split similar to

        BAGPIPES-style plots.

    residual_ylim : tuple of float or None, optional

        Manual (ymin, ymax) limits for the residual panel. If None (default),

        the limits are auto-scaled but always span at least [-3, 3] in sigma

        units so the ±1σ reference lines are visible.

    min_line_snr : float, default 2.5

        When spectroscopy is present, only emission lines detected in the

        observed spectrum with matched-filter significance >= this value

        are drawn (see ``bayesed.emission_lines.detect_lines``). Set to 0

        or None to disable detection and mark every table line.

    line_window_kms : float, default 3000.0

        Half-width of the velocity window used for line detection; wide

        enough to cover QSO broad-line components at the default.

    **kwargs

        Additional keyword arguments passed to matplotlib plotting functions

    

    Returns

    -------

    matplotlib.figure.Figure

        The matplotlib figure object

    

    Examples

    --------

    >>> # Basic usage

    >>> from bayesed import plot_bestfit

    >>> plot_bestfit('output/object1_bestfit.fits')

    >>> 

    >>> # Customize plot

    >>> plot_bestfit(

    ...     'output/object1_bestfit.fits',

    ...     use_rest_frame=True,

    ...     flux_unit='nufnu',

    ...     use_log_scale=True,

    ...     output_file='my_plot.png'

    ... )

    >>> 

    >>> # With filter overlay

    >>> plot_bestfit(

    ...     'output/object1_bestfit.fits',

    ...     filter_file='filters.txt',

    ...     filter_selection_file='filters_selected.txt'

    ... )

    """

    import numpy as np



    

    # Constants

    c = 2.9979246e+14  # um/s (speed of light)

    # Extract parameters before passing to plot() - these go to other matplotlib functions
    plot_xlim = kwargs.pop('xlim', None)
    plot_ylim = kwargs.pop('ylim', None)
    # Legend kwargs: use dict comprehension to filter None values
    legend_kwargs = {k: v for k, v in {
        'loc': kwargs.pop('legend_loc', None),
        'bbox_to_anchor': kwargs.pop('bbox_to_anchor', None),
    }.items() if v is not None}
    # Other common plot parameters that shouldn't go to ax1.plot()
    # Add here as needed: title_kwargs, xlabel_kwargs, ylabel_kwargs, etc.

    

    # Validate fits_file
    if fits_file is None:
        raise ValueError("fits_file cannot be None. Provide a path to a FITS file or ensure bestfit_file is set in BayeSEDResults.")
    
    # Open FITS file
    hdul = fits.open(fits_file)

    

    try:

        # Get header information

        header = hdul[0].header

        obj_id = header.get('ID', 'Unknown')

        snr = header.get('SNR', 'N/A')

        z_best = header.get('Z_{MAL}', header.get('z_{MAL}', 'N/A'))

        xmin2_nd = header.get('XMIN^2/ND', 'N/A')

        run_args = header.get('RUN_ARGUMENTS', '')

        

        # Determine wavelength and flux columns based on frame

        if use_rest_frame:

            wl_col_model = 'wavelength_rest'

            wl_col_phot = 'wavelength_rest_center'

            wl_col_spec = 'wavelength_rest'

        else:

            wl_col_model = 'wavelength_obs'

            wl_col_phot = 'wavelength_obs_center'

            wl_col_spec = 'wavelength_obs'

        

        # Decide whether a residual panel can be drawn

        has_phot = (

            'obs:phot' in hdul

            and '--no_photometry_fit' not in run_args

        )

        has_spec = (

            'obs:spec' in hdul

            and '--no_spectra_fit' not in run_args

        )

        draw_residuals = show_residuals and (has_phot or has_spec)



        # Create figure

        if draw_residuals:

            fig, (ax1, ax_resid) = plt.subplots(
                2, 1, figsize=figsize, sharex=True,
                gridspec_kw={
                    'height_ratios': [residual_height_ratio, 1],
                    'hspace': 0,
                },
            )

        else:

            fig, ax1 = plt.subplots(figsize=figsize)

            ax_resid = None



        # Collect wavelength ranges from models and data separately for auto-scaling x-axis

        model_wavelength_ranges = []

        data_wavelength_ranges = []

        # Emission lines detected in the observed spectra (set of
        # (wl, label, family) from EMISSION_LINES). None until a spectrum
        # is actually analyzed; photometry-only plots keep every line.
        detected_lines = None

        # Collect flux ranges for y-axis scaling when focus_on_data_range is enabled

        model_flux_values = []  # Will store (wavelength, flux) pairs for models

        data_flux_values = []    # Will store (wavelength, flux) pairs for data (obs + model)

        observed_flux_values = []  # Will store (wavelength, flux) pairs for observed data only (phot + spec)

        # Store residual data for later plotting

        residual_phot_points = []   # (wavelength, residual) for photometry

        residual_spec_points = []   # (wavelength, residual, color, label) per spec band



        # Get model components

        models = [h.name for h in hdul if h.name.startswith('model:') and h.name != 'model:total']

        

        # Auto-generate model names if not provided

        if model_names is None:

            model_names = []

            for model in models:

                model_short = model.replace('model:', '')

                # Try to infer component type from name

                if 'stellar' in model_short.lower() or 'ssp' in model_short.lower():

                    model_names.append('Stellar+Nebular')

                elif 'disk' in model_short.lower() or 'bbb' in model_short.lower() or 'agn' in model_short.lower():

                    model_names.append('AGN-Disk')

                elif 'blr' in model_short.lower():

                    model_names.append('AGN-BLR')

                elif 'nlr' in model_short.lower():

                    model_names.append('AGN-NLR')

                elif 'feii' in model_short.lower() or 'fe_ii' in model_short.lower():

                    model_names.append('AGN-FeII')

                elif 'tor' in model_short.lower():

                    model_names.append('AGN-Torus')

                else:

                    model_names.append(model_short[:20])  # Truncate long names

        if len(model_names) != len(models):
            plt.close(fig)
            raise ValueError(
                f'model_names has {len(model_names)} names for '
                f'{len(models)} model components'
            )

        # Line styles for model components — all non-solid to distinguish from total/spec.
        # Colors avoid: black (total), warm reds/oranges (spec_obs), blues/teals (spec_mod).
        _component_styles = ['--', '-.', ':', (0, (3, 1, 1, 1)), (0, (5, 2))]
        _component_colors = ['#27AE60', '#D4AC0D', '#8E44AD', '#C0392B', '#17A589']
        _comp_plotted = 0

        # Plot model components

        # Handle show_components as boolean or dict for individual control
        if isinstance(show_components, bool):
            components_to_show = {m: show_components for m in models}
        else:
            # Dict: use explicit visibility, default to True for unlisted
            components_to_show = {m: show_components.get(m, True) for m in models}

        for model, name in zip(models, model_names):
            if not components_to_show.get(model, True):
                continue

            try:

                data = hdul[model].data

                if data is None:

                    continue



                mask = data['flux'] > 0



                if np.sum(mask) == 0:

                    continue



                # Try to get wavelength column (handle both rest and obs frame)

                try:

                    wl = data[wl_col_model][mask]

                except (KeyError, IndexError):

                    # Try alternative column name

                    if use_rest_frame:

                        try:

                            wl = data['wavelength_obs'][mask]

                        except (KeyError, IndexError):

                            continue

                    else:

                        try:

                            wl = data['wavelength_rest'][mask]

                        except (KeyError, IndexError):

                            continue



                flux = data['flux'][mask]



                # Convert flux units if needed

                if flux_unit == 'nufnu':

                    # Convert Fν to νFν = c * λ^-1 * Fν

                    if use_rest_frame:

                        try:

                            flux = c * data['wavelength_obs'][mask]**-1 * flux

                        except (KeyError, IndexError):

                            flux = c * wl**-1 * flux

                    else:

                        flux = c * wl**-1 * flux

                elif flux_unit == 'flambda':

                    # Convert Fν to Fλ = c * λ^-2 * Fν

                    if use_rest_frame:

                        try:

                            flux = c * data['wavelength_obs'][mask]**-2 * flux

                        except (KeyError, IndexError):

                            flux = c * wl**-2 * flux

                    else:

                        flux = c * wl**-2 * flux



                ax1.plot(wl, flux, label=name,
                         linestyle=_component_styles[_comp_plotted % len(_component_styles)],
                         color=_component_colors[_comp_plotted % len(_component_colors)],
                         **kwargs)
                _comp_plotted += 1

                # Collect wavelength range from model

                if len(wl) > 0:

                    model_wavelength_ranges.append((np.min(wl), np.max(wl)))

                    # Collect flux values for y-axis scaling

                    model_flux_values.extend(list(zip(wl, flux)))

            except Exception as e:

                print(f"Warning: Could not plot model {model}: {e}")

                continue

        

        # Plot total model if available

        if show_total and 'model:total' in hdul:

            try:

                total = hdul['model:total'].data

                if total is not None:

                    mask = total['flux'] > 0

                    if np.sum(mask) > 0:

                        # Try to get wavelength column

                        try:

                            wl = total[wl_col_model][mask]

                        except (KeyError, IndexError):

                            if use_rest_frame:

                                try:

                                    wl = total['wavelength_obs'][mask]

                                except (KeyError, IndexError):

                                    wl = None

                            else:

                                try:

                                    wl = total['wavelength_rest'][mask]

                                except (KeyError, IndexError):

                                    wl = None

                        

                        if wl is not None:

                            flux = total['flux'][mask]

                            

                            if flux_unit == 'nufnu':

                                if use_rest_frame:

                                    try:

                                        flux = c * total['wavelength_obs'][mask]**-1 * flux

                                    except (KeyError, IndexError):

                                        flux = c * wl**-1 * flux

                                else:

                                    flux = c * wl**-1 * flux

                            elif flux_unit == 'flambda':

                                if use_rest_frame:

                                    try:

                                        flux = c * total['wavelength_obs'][mask]**-2 * flux

                                    except (KeyError, IndexError):

                                        flux = c * wl**-2 * flux

                                else:

                                    flux = c * wl**-2 * flux

                            

                            ax1.plot(wl, flux, label='Total', color='black', linewidth=2, linestyle='--')

                            # Collect wavelength range from model

                            if len(wl) > 0:

                                model_wavelength_ranges.append((np.min(wl), np.max(wl)))

                                # Collect flux values for y-axis scaling

                                model_flux_values.extend(list(zip(wl, flux)))

            except Exception as e:

                print(f"Warning: Could not plot total model: {e}")

        

        # Plot photometry data

        if 'obs:phot' in hdul and '--no_photometry_fit' not in run_args:

            try:

                obs_phot = hdul['obs:phot'].data

                if obs_phot is not None:

                    obs_phot = obs_phot[obs_phot['iselected'] == 1]

                    

                    if len(obs_phot) > 0:

                        try:

                            wl_phot = obs_phot[wl_col_phot]

                            flux_obs = obs_phot['flux_obs']

                            flux_model = obs_phot['flux_model']

                            

                            # Convert flux units if needed

                            if flux_unit == 'nufnu':

                                try:

                                    wl_obs_center = obs_phot['wavelength_obs_center']

                                    flux_obs = c * wl_obs_center**-1 * flux_obs

                                    flux_model = c * wl_obs_center**-1 * flux_model

                                except (KeyError, IndexError):

                                    pass  # Keep original flux if conversion not possible

                            elif flux_unit == 'flambda':

                                try:

                                    wl_obs_center = obs_phot['wavelength_obs_center']

                                    flux_obs = c * wl_obs_center**-2 * flux_obs

                                    flux_model = c * wl_obs_center**-2 * flux_model

                                except (KeyError, IndexError):

                                    pass  # Keep original flux if conversion not possible

                            

                            # Plot observed photometry with errors if available

                            try:

                                mask_err = obs_phot['flux_obs_err'] > 0

                                flux_err = obs_phot['flux_obs_err']

                                # Convert error units if needed

                                if flux_unit == 'nufnu':

                                    try:

                                        wl_obs_center = obs_phot['wavelength_obs_center']

                                        flux_err = c * wl_obs_center**-1 * flux_err

                                    except (KeyError, IndexError):

                                        pass

                                elif flux_unit == 'flambda':

                                    try:

                                        wl_obs_center = obs_phot['wavelength_obs_center']

                                        flux_err = c * wl_obs_center**-2 * flux_err

                                    except (KeyError, IndexError):

                                        pass

                                ax1.errorbar(wl_phot[mask_err], flux_obs[mask_err], flux_err[mask_err], 

                                           fmt='x', color='red', label='phot:obs', markersize=8, capsize=3)

                            except (KeyError, IndexError):

                                ax1.plot(wl_phot, flux_obs, 'x', color='red', label='phot:obs', markersize=8)

                            

                            # Plot model photometry with errors if available

                            try:

                                flux_model_err = obs_phot['flux_model_err']

                                ax1.errorbar(wl_phot, flux_model, flux_model_err, 

                                           fmt='+', color='blue', label='phot:mod', markersize=8, capsize=3)

                            except (KeyError, IndexError):

                                ax1.plot(wl_phot, flux_model, '+', color='blue', label='phot:mod', markersize=8)

                            # Collect wavelength range from photometry data

                            if len(wl_phot) > 0:

                                data_wavelength_ranges.append((np.min(wl_phot), np.max(wl_phot)))

                                # Collect flux values for y-axis scaling (both obs and model)

                                data_flux_values.extend(list(zip(wl_phot, flux_obs)))

                                data_flux_values.extend(list(zip(wl_phot, flux_model)))

                                # Collect observed-only flux values for lower limit determination

                                observed_flux_values.extend(list(zip(wl_phot, flux_obs)))

                                # Also include error bars in flux range

                                try:

                                    if 'flux_obs_err' in obs_phot.dtype.names:

                                        flux_err_upper = flux_obs + obs_phot['flux_obs_err']

                                        flux_err_lower = np.maximum(flux_obs - obs_phot['flux_obs_err'], 0)

                                        data_flux_values.extend(list(zip(wl_phot, flux_err_upper)))

                                        data_flux_values.extend(list(zip(wl_phot, flux_err_lower)))

                                        # Add error bars to observed-only values too

                                        observed_flux_values.extend(list(zip(wl_phot, flux_err_upper)))

                                        observed_flux_values.extend(list(zip(wl_phot, flux_err_lower)))

                                except (KeyError, IndexError):

                                    pass

                                # Collect residuals in native flux units so
                                # (obs-mod)/err is invariant under flux_unit.

                                if ax_resid is not None:

                                    try:

                                        phot_err = obs_phot['flux_obs_err']

                                        valid = phot_err > 0

                                        residual_phot_points.append((

                                            wl_phot[valid],

                                            (obs_phot['flux_obs'][valid]

                                             - obs_phot['flux_model'][valid])

                                            / phot_err[valid],

                                        ))

                                    except (KeyError, IndexError):

                                        pass

                        except (KeyError, IndexError) as e:

                            print(f"Warning: Missing required columns for photometry: {e}")

            except Exception as e:

                print(f"Warning: Could not plot photometry data: {e}")

        

        # Plot spectroscopy data

        if 'obs:spec' in hdul and '--no_spectra_fit' not in run_args:

            try:

                obs_spec = hdul['obs:spec'].data

                

                if obs_spec is not None and len(obs_spec) > 0:

                    bands = np.unique(obs_spec['iband'])
                    # Distinct colors per band so adjacent bands don't blend.
                    # Model spectra are drawn in solid gray (see below), so
                    # only the observed colors are needed here.
                    _obs_colors = ['#E74C3C', '#E67E22', '#8E44AD', '#2ECC71', '#F39C12']
                    # All model spectra share one style; only the first band
                    # gets a legend entry (dedup'd unless custom labels given).
                    _mod_labeled = False
                    # Model spectra are drawn after ALL observed spectra so
                    # the smooth gray curves sit on top of the noisy data.
                    _mod_specs = []
                    # Emission-line detection runs in the rest frame; the
                    # observed-frame path needs a valid redshift to map line
                    # wavelengths. Without one, detection is disabled (all
                    # lines shown) rather than silently mismatching.
                    _detect_enabled = bool(min_line_snr)
                    if not use_rest_frame and _detect_enabled:
                        try:
                            _z_det = float(z_best)
                        except (ValueError, TypeError):
                            _z_det = 0.0
                            _detect_enabled = False
                            print("Warning: could not parse redshift for "
                                  "emission-line detection; showing all "
                                  "candidate lines instead.")
                    else:
                        _z_det = 0.0

                    for band in bands:

                        try:

                            band_mask = obs_spec['iband'] == band

                            band_data = obs_spec[band_mask]

                            

                            if len(band_data) == 0:

                                continue

                            

                            wl_spec = band_data[wl_col_spec]

                            spec_obs = band_data['spectra_obs']

                            spec_mod = band_data['spectra_mod']

                            # Emission-line detection on the RAW observed
                            # spectrum and RAW per-pixel errors, so the
                            # threshold means the same thing in every
                            # flux_unit mode (the conversions below scale
                            # flux but not err, which would bias the
                            # significance). Runs once per band; lines are
                            # matched in the rest frame regardless of the
                            # display frame.
                            if _detect_enabled:
                                try:
                                    _det = _detect_band_lines(
                                        band_data, wl_spec, spec_obs,
                                        min_line_snr, line_window_kms)
                                    if detected_lines is None:
                                        detected_lines = set()
                                    detected_lines.update(_det)
                                except KeyError as e:
                                    print(f"Warning: emission-line detection "
                                          f"failed for band {band}: {e}")


                            

                            # Convert flux units if needed

                            if flux_unit == 'nufnu':

                                if use_rest_frame:

                                    try:

                                        spec_obs = c * band_data['wavelength_obs']**-1 * spec_obs

                                        spec_mod = c * band_data['wavelength_obs']**-1 * spec_mod

                                    except (KeyError, IndexError):

                                        spec_obs = c * wl_spec**-1 * spec_obs

                                        spec_mod = c * wl_spec**-1 * spec_mod

                                else:

                                    spec_obs = c * wl_spec**-1 * spec_obs

                                    spec_mod = c * wl_spec**-1 * spec_mod

                            elif flux_unit == 'flambda':

                                if use_rest_frame:

                                    try:

                                        spec_obs = c * band_data['wavelength_obs']**-2 * spec_obs

                                        spec_mod = c * band_data['wavelength_obs']**-2 * spec_mod

                                    except (KeyError, IndexError):

                                        spec_obs = c * wl_spec**-2 * spec_obs

                                        spec_mod = c * wl_spec**-2 * spec_mod

                                else:

                                    spec_obs = c * wl_spec**-2 * spec_obs

                                    spec_mod = c * wl_spec**-2 * spec_mod

                            

                            # Resolve spec labels — support dict per band, string, or None
                            _obs_label = f'spec:obs_{band}'
                            _mod_label = 'spec:mod'
                            if spec_obs_labels is not None:
                                if isinstance(spec_obs_labels, dict):
                                    _obs_label = spec_obs_labels.get(band, _obs_label)
                                else:
                                    _obs_label = str(spec_obs_labels)
                            if spec_mod_labels is not None:
                                if isinstance(spec_mod_labels, dict):
                                    _mod_label = spec_mod_labels.get(band, _mod_label)
                                else:
                                    _mod_label = str(spec_mod_labels)
                            # All model spectra share the same gray style, so
                            # the default legend keeps a single 'spec:mod'
                            # entry; custom labels are always honored per band.
                            if _mod_labeled and _mod_label == 'spec:mod':
                                _mod_label = None
                            else:
                                _mod_labeled = True

                            ax1.plot(wl_spec, spec_obs, label=_obs_label,
                                     color=_obs_colors[band % len(_obs_colors)], alpha=0.9)

                            # Model spectra: solid gray lines. They must not
                            # share the dashed style of the component/total
                            # model curves, and gray reads clearly against
                            # the colored observed spectra. Deferred until
                            # after all bands are drawn (see below).
                            _mod_specs.append((wl_spec, spec_mod, _mod_label))

                            # Collect wavelength range from spectroscopy data

                            if len(wl_spec) > 0:

                                data_wavelength_ranges.append((np.min(wl_spec), np.max(wl_spec)))

                                # Collect flux values for y-axis scaling

                                data_flux_values.extend(list(zip(wl_spec, spec_obs)))

                                data_flux_values.extend(list(zip(wl_spec, spec_mod)))

                                # Collect observed-only flux values for lower limit determination

                                observed_flux_values.extend(list(zip(wl_spec, spec_obs)))

                                # Collect residuals in native flux units so
                                # (obs-mod)/err is invariant under flux_unit.

                                if ax_resid is not None:

                                    try:

                                        spec_err = band_data['spectra_obs_err']

                                        valid = spec_err > 0

                                        residual_spec_points.append((

                                            wl_spec[valid],

                                            (band_data['spectra_obs'][valid]

                                             - band_data['spectra_mod'][valid])

                                            / spec_err[valid],

                                            _obs_colors[band % len(_obs_colors)],

                                        ))

                                    except (KeyError, IndexError):

                                        pass

                        except (KeyError, IndexError) as e:

                            print(f"Warning: Could not plot band {band}: {e}")

                            continue

                    # Draw model spectra after all observed spectra so they
                    # sit on top of the noisy data.
                    for wl_mod, spec_mod, _mod_label in _mod_specs:
                        ax1.plot(wl_mod, spec_mod, label=_mod_label,
                                 color='darkgray', linestyle='-',
                                 linewidth=1.2, alpha=0.9)

            except Exception as e:

                print(f"Warning: Could not plot spectroscopy data: {e}")

        

        # Draw residual panel if requested

        if ax_resid is not None:

            _draw_residual_panel(

                ax_resid, residual_phot_points, residual_spec_points,

                residual_ylim, label_fontsize,

            )

        

        # Set x-axis limits based on data ranges or all ranges

        if focus_on_data_range:

            # Use only data ranges (photometry and spectroscopy)

            wavelength_ranges = data_wavelength_ranges

            # If no data ranges found but we have model ranges, fall back to model ranges

            # (this can happen if data exists but wasn't collected properly, or if only models exist)

            if not wavelength_ranges and model_wavelength_ranges:

                wavelength_ranges = model_wavelength_ranges

        else:

            # Use all ranges (models + data)

            wavelength_ranges = model_wavelength_ranges + data_wavelength_ranges

        

        if wavelength_ranges:

            x_min = min(r[0] for r in wavelength_ranges)

            x_max = max(r[1] for r in wavelength_ranges)

            # Add small padding (5% on each side for linear scale)

            x_range = x_max - x_min

            if x_range > 0:

                padding = 0.05 * x_range

                ax1.set_xlim(x_min - padding, x_max + padding)

        

        # Plot filter responses if provided

        if filter_file and filter_selection_file:

            try:

                # Read filter selection file

                filter_info = []

                with open(filter_selection_file, 'r') as f:

                    for line in f:

                        if not line.startswith('#'):

                            parts = line.split()

                            if len(parts) > 3 and parts[0] == '1' and parts[1] == '1':

                                filter_info.append((int(parts[2]), parts[3]))

                

                if filter_info:

                    ax2 = ax1.twinx()

                    # Parse redshift

                    try:

                        z = float(z_best)

                    except (ValueError, TypeError):

                        z = 0.0

                    colors = plt.cm.tab20.colors

                    color_idx = 0

                    

                    y_max = ax1.get_ylim()[1]

                    scale_factor = 0.2 * y_max

                    

                    for filter_id, filter_name in filter_info:

                        wavelengths, responses = [], []

                        current_filter_id = -1

                        

                        with open(filter_file, 'r') as f:

                            for line in f:

                                line = line.strip()

                                if line.startswith('#'):

                                    current_filter_id += 1

                                    if current_filter_id == filter_id:

                                        continue

                                elif line and current_filter_id == filter_id:

                                    try:

                                        wl, response = map(float, line.split())

                                        wavelengths.append(wl)

                                        responses.append(response)

                                    except ValueError:

                                        continue

                        

                        if wavelengths and responses:

                            responses = np.array(responses) * scale_factor

                            wl_rest = np.array(wavelengths) / (1 + z) if z > 0 else np.array(wavelengths)

                            ax2.plot(wl_rest, responses, linestyle='--', 

                                   color=colors[color_idx % len(colors)], 

                                   label=filter_name, alpha=0.6)

                            color_idx += 1

                    

                    ax1_ylim = ax1.get_ylim()

                    ax2.set_ylim(0, ax1_ylim[1] / 2)

                    ax2.yaxis.set_ticks([])

                    ax2.yaxis.set_ticklabels([])

                    ax2.legend(loc='upper right', bbox_to_anchor=(1, 0.9), fontsize=legend_fontsize)

            except Exception as e:

                print(f"Warning: Could not plot filter responses: {e}")

        

        # Set axis labels and title

        wl_label = r'$\lambda_{\rm rest}/\rm \mu m$' if use_rest_frame else r'$\lambda_{\rm obs}/\rm \mu m$'

        # With a residual panel, the wavelength xlabel belongs on the bottom
        # panel only and the main panel hides its x tick labels (label_outer).

        if ax_resid is not None:

            ax_resid.set_xlabel(wl_label, fontsize=label_fontsize)

            ax1.label_outer()

        else:

            ax1.set_xlabel(wl_label, fontsize=label_fontsize)

        

        if flux_unit == 'nufnu':

            ax1.set_ylabel(r'$\nu F_\nu / [\rm \mu Jy \cdot Hz]$', fontsize=label_fontsize)

        elif flux_unit == 'flambda':

            ax1.set_ylabel(r'$F_\lambda / [\rm erg \cdot s^{-1} \cdot cm^{-2} \cdot \AA^{-1}]$', fontsize=label_fontsize)

        else:

            ax1.set_ylabel(r'$F_\nu / [\rm \mu Jy]$', fontsize=label_fontsize)

        

        title = (f"ID={obj_id}\nSNR={snr}, "

                f"z_best={z_best}, Xmin²/Nd={xmin2_nd}")

        ax1.set_title(title, fontsize=title_fontsize)

        

        # Set y-axis limits based on focus_on_data_range setting

        if focus_on_data_range and (data_flux_values or observed_flux_values):

            # Focus y-axis on flux values within the focused x-axis range

            if wavelength_ranges:

                x_min_focused = min(r[0] for r in wavelength_ranges)

                x_max_focused = max(r[1] for r in wavelength_ranges)

                

                # Collect flux values within the focused x-axis range

                # For lower limit: use only observed data (phot + spec), not model data

                # For upper limit: use all data (obs + model) to show full range

                focused_flux_values_all = []

                focused_flux_values_observed = []

                

                # Add all data flux values (obs + model) for upper limit

                focused_flux_values_all.extend([f for wl, f in data_flux_values if wl >= x_min_focused and wl <= x_max_focused])

                

                # Add only observed flux values for lower limit

                focused_flux_values_observed.extend([f for wl, f in observed_flux_values if wl >= x_min_focused and wl <= x_max_focused])

                

                if focused_flux_values_all or focused_flux_values_observed:

                    # Determine lower limit from observed data only

                    y_min_focused = None

                    if focused_flux_values_observed:

                        observed_flux_array = np.array(focused_flux_values_observed)

                        valid_observed_flux = observed_flux_array[observed_flux_array > 0]

                        if len(valid_observed_flux) > 0:

                            y_min_focused = np.min(valid_observed_flux)

                    

                    # Determine upper limit from all data (obs + model)

                    y_max_focused = None

                    if focused_flux_values_all:

                        all_flux_array = np.array(focused_flux_values_all)

                        valid_all_flux = all_flux_array[all_flux_array > 0]

                        if len(valid_all_flux) > 0:

                            y_max_focused = np.max(valid_all_flux)

                    

                    # Set limits if we have both

                    if y_min_focused is not None and y_max_focused is not None:

                        # Add padding (10% on each side)

                        y_range = y_max_focused - y_min_focused

                        if y_range > 0:

                            padding = 0.1 * y_range

                            ax1.set_ylim(y_min_focused - padding, y_max_focused + padding)

                    elif y_min_focused is not None:

                        # Only lower limit available, use it with some padding

                        ax1.set_ylim(bottom=y_min_focused * 0.9)

                    elif y_max_focused is not None:

                        # Only upper limit available, use it with some padding

                        ax1.set_ylim(top=y_max_focused * 1.1)

        

        # Auto-detect log scale if not specified (after plotting data)

        if use_log_scale is None:

            # Get current axis limits after plotting

            ax1.relim()  # Recalculate limits based on plotted data

            # Only autoscale y-axis if we haven't already set it manually

            if not (focus_on_data_range and (data_flux_values or observed_flux_values)):

                ax1.autoscale(axis='y')  # Auto-scale y-axis only, preserve x-axis limits

            xlim = ax1.get_xlim()

            ylim = ax1.get_ylim()

            

            # Check if ranges span multiple orders of magnitude

            if xlim[0] > 0 and ylim[0] > 0:

                x_range_ratio = xlim[1] / xlim[0] if xlim[0] > 0 else 1

                y_range_ratio = ylim[1] / ylim[0] if ylim[0] > 0 else 1

                # Use log scale if range spans more than 1 order of magnitude (10x)

                use_log_scale = (x_range_ratio > 10 or y_range_ratio > 10)

            else:

                # Default to linear if negative values present

                use_log_scale = False

        

        # Handle log scale: None, False, True, 'x', 'y', 'both'
        log_x = use_log_scale in (True, 'both', 'x')
        log_y = use_log_scale in (True, 'both', 'y')
        if log_x:
            ax1.set_xscale('log')
        if log_y:
            ax1.set_yscale('log')

            # Re-apply x-axis limits after setting log scale with multiplicative padding

            # Use the same range selection logic as before

            if focus_on_data_range:

                ranges_for_lim = data_wavelength_ranges

                # If no data ranges found but we have model ranges, fall back to model ranges

                if not ranges_for_lim and model_wavelength_ranges:

                    ranges_for_lim = model_wavelength_ranges

            else:

                ranges_for_lim = model_wavelength_ranges + data_wavelength_ranges

            

            if ranges_for_lim:

                x_min = min(r[0] for r in ranges_for_lim)

                x_max = max(r[1] for r in ranges_for_lim)

                # For log scale, use multiplicative padding

                if x_min > 0 and x_max > 0:

                    padding_factor = 1.1  # 10% padding

                    ax1.set_xlim(x_min / padding_factor, x_max * padding_factor)

            

            # Re-apply y-axis limits for log scale when focus_on_data_range is enabled

            if focus_on_data_range and (data_flux_values or observed_flux_values):

                if ranges_for_lim:

                    x_min_focused = min(r[0] for r in ranges_for_lim)

                    x_max_focused = max(r[1] for r in ranges_for_lim)

                    

                    # Collect flux values within the focused x-axis range

                    # For lower limit: use only observed data (phot + spec), not model data

                    # For upper limit: use all data (obs + model) to show full range

                    focused_flux_values_all = []

                    focused_flux_values_observed = []

                    

                    # Add all data flux values (obs + model) for upper limit

                    focused_flux_values_all.extend([f for wl, f in data_flux_values if wl >= x_min_focused and wl <= x_max_focused])

                    

                    # Add only observed flux values for lower limit

                    focused_flux_values_observed.extend([f for wl, f in observed_flux_values if wl >= x_min_focused and wl <= x_max_focused])

                    

                    if focused_flux_values_all or focused_flux_values_observed:

                        # Determine lower limit from observed data only

                        y_min_focused = None

                        if focused_flux_values_observed:

                            observed_flux_array = np.array(focused_flux_values_observed)

                            valid_observed_flux = observed_flux_array[observed_flux_array > 0]

                            if len(valid_observed_flux) > 0:

                                y_min_focused = np.min(valid_observed_flux)

                        

                        # Determine upper limit from all data (obs + model)

                        y_max_focused = None

                        if focused_flux_values_all:

                            all_flux_array = np.array(focused_flux_values_all)

                            valid_all_flux = all_flux_array[all_flux_array > 0]

                            if len(valid_all_flux) > 0:

                                y_max_focused = np.max(valid_all_flux)

                        

                        # Set limits if we have both

                        if y_min_focused is not None and y_max_focused is not None:

                            # For log scale, use multiplicative padding

                            if y_min_focused > 0 and y_max_focused > 0:

                                padding_factor = 1.1  # 10% padding

                                ax1.set_ylim(y_min_focused / padding_factor, y_max_focused * padding_factor)

                        elif y_min_focused is not None and y_min_focused > 0:

                            # Only lower limit available, use it with some padding

                            padding_factor = 1.1  # 10% padding

                            ax1.set_ylim(bottom=y_min_focused / padding_factor)

                        elif y_max_focused is not None and y_max_focused > 0:

                            # Only upper limit available, use it with some padding

                            padding_factor = 1.1  # 10% padding

                            ax1.set_ylim(top=y_max_focused * padding_factor)

        

        # Auto-adjust legend

        handles, labels = ax1.get_legend_handles_labels()

        n_items = len(handles)

        if n_items > 0:

            ncol = min(3, max(1, n_items // 5)) if n_items > 10 else 1

            # Build legend kwargs: use legend_loc if provided, else 'best'
            if 'loc' not in legend_kwargs:
                legend_kwargs['loc'] = 'best'
            legend_kwargs['ncol'] = ncol
            legend_kwargs['fontsize'] = legend_fontsize
            legend_kwargs['framealpha'] = 0.9

            ax1.legend(**legend_kwargs)

        # Set axis limits if provided (applied LAST, overriding all previous logic)
        if plot_xlim is not None:
            ax1.set_xlim(plot_xlim)
        if plot_ylim is not None:
            ax1.set_ylim(plot_ylim)



        # Add emission line markers and labels after all data is plotted and axis limits are set
        if show_emission_lines:
            # Get current axis limits
            xlim = ax1.get_xlim()
            ylim = ax1.get_ylim()
            
            # Emission lines with wavelengths and labels; color comes from the
            # fixed family -> color mapping (same family, same color).
            emission_lines = [
                (wl, label, FAMILY_COLORS[family])
                for wl, label, family in EMISSION_LINES
            ]
            
            # Remove duplicates based on wavelength (keep first occurrence)
            seen_wavelengths = set()
            unique_emission_lines = []
            for wl, label, color in emission_lines:
                if wl not in seen_wavelengths:
                    seen_wavelengths.add(wl)
                    unique_emission_lines.append((wl, label, color))
            
            # Only show lines that are within the current x-axis range
            visible_lines = [(wl, label, color) for wl, label, color in unique_emission_lines
                           if xlim[0] <= wl <= xlim[1]]

            # When spectra were analyzed, keep only lines significantly
            # detected in the data (see detect_lines); photometry-only
            # plots (detected_lines is None) keep every line.
            if detected_lines is not None:
                visible_lines = [
                    (wl, label, color) for wl, label, color in visible_lines
                    if (wl, label) in detected_lines
                ]
            
            if visible_lines:
                visible_lines.sort(key=lambda x: x[0])
                _draw_emission_line_labels(
                    ax1, visible_lines, xlim, log_x,
                    emission_line_fontsize, emission_line_ypos,
                    group_window_kms=line_window_kms)


        fig.tight_layout()

        if ax_resid is not None:

            # tight_layout re-introduces inter-panel spacing; force the
            # residual panel flush against the main panel.

            fig.subplots_adjust(hspace=0)

        

        # Save figure

        if save:
            if output_file is None:
                import os
                fits_dir = os.path.dirname(os.path.abspath(fits_file))
                fits_basename = os.path.basename(fits_file)
                png_basename = os.path.splitext(fits_basename)[0] + '.png'
                output_file = os.path.join(fits_dir, png_basename)
            plt.savefig(output_file, dpi=dpi, bbox_inches='tight')

        

        if show:

            plt.show()

        else:

            plt.close()

        

        return fig

    

    finally:

        hdul.close()





