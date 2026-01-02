"""
Plotting utilities for BayeSED3.
"""

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.table import Table

def plot_bestfit(fits_file, output_file=None, show=True, 

                 filter_file=None, filter_selection_file=None,

                 use_rest_frame=True, flux_unit='fnu', use_log_scale=None,

                 model_names=None, show_emission_lines=True, emission_line_fontsize=12,

                 title_fontsize=16, label_fontsize=16, legend_fontsize=14,

                 figsize=(12, 8), dpi=300, focus_on_data_range=True, **kwargs):

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

    use_log_scale : bool, optional

        Use logarithmic scale for axes. If None (default), auto-detects based on data range.

        Auto-detection uses log scale when either axis spans more than 1 order of magnitude

        (range ratio > 10). If negative values are present, defaults to linear scale.

        Set to True to force log scale, or False to force linear scale.

    model_names : list of str, optional

        Custom names for model components. If None, auto-generates from HDU names

    show_emission_lines : bool

        Show emission line markers for spectroscopy (default: True)

    emission_line_fontsize : int, default 12

        Font size for emission line labels. Larger values make labels more readable.

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

    focus_on_data_range : bool

        If True, set x-axis limits to focus on the wavelength range where data exists

        (photometry and spectroscopy), ignoring the full model range. If False, use

        the full wavelength range from both models and data (default: True)

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

    import matplotlib.pyplot as plt

    from astropy.io import fits

    

    # Constants

    c = 2.9979246e+14  # um/s (speed of light)

    

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

        

        # Create figure

        fig, ax1 = plt.subplots(figsize=figsize)

        

        # Collect wavelength ranges from models and data separately for auto-scaling x-axis

        model_wavelength_ranges = []

        data_wavelength_ranges = []

        # Collect flux ranges for y-axis scaling when focus_on_data_range is enabled

        model_flux_values = []  # Will store (wavelength, flux) pairs for models

        data_flux_values = []    # Will store (wavelength, flux) pairs for data (obs + model)

        observed_flux_values = []  # Will store (wavelength, flux) pairs for observed data only (phot + spec)

        

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

        

        # Plot model components

        for model, name in zip(models, model_names):

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

                

                ax1.plot(wl, flux, label=name, **kwargs)

                # Collect wavelength range from model

                if len(wl) > 0:

                    model_wavelength_ranges.append((np.min(wl), np.max(wl)))

                    # Collect flux values for y-axis scaling

                    model_flux_values.extend(list(zip(wl, flux)))

            except Exception as e:

                print(f"Warning: Could not plot model {model}: {e}")

                continue

        

        # Plot total model if available

        if 'model:total' in hdul:

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

                    for band in bands:

                        try:

                            band_mask = obs_spec['iband'] == band

                            band_data = obs_spec[band_mask]

                            

                            if len(band_data) == 0:

                                continue

                            

                            wl_spec = band_data[wl_col_spec]

                            spec_obs = band_data['spectra_obs']

                            spec_mod = band_data['spectra_mod']

                            

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

                            

                            ax1.plot(wl_spec, spec_obs, label=f'spec:obs_{band}', alpha=0.7)

                            ax1.plot(wl_spec, spec_mod, label=f'spec:mod_{band}', 

                                   linestyle='--', alpha=0.7)

                            # Collect wavelength range from spectroscopy data

                            if len(wl_spec) > 0:

                                data_wavelength_ranges.append((np.min(wl_spec), np.max(wl_spec)))

                                # Collect flux values for y-axis scaling

                                data_flux_values.extend(list(zip(wl_spec, spec_obs)))

                                data_flux_values.extend(list(zip(wl_spec, spec_mod)))

                                # Collect observed-only flux values for lower limit determination

                                observed_flux_values.extend(list(zip(wl_spec, spec_obs)))

                        except (KeyError, IndexError) as e:

                            print(f"Warning: Could not plot band {band}: {e}")

                            continue

            except Exception as e:

                print(f"Warning: Could not plot spectroscopy data: {e}")

        

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

        

        if use_log_scale:

            ax1.set_xscale('log')

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

            ax1.legend(loc='best', ncol=ncol, fontsize=legend_fontsize, framealpha=0.9)

        

        # Add emission line markers and labels after all data is plotted and axis limits are set
        if show_emission_lines:
            # Get current axis limits
            xlim = ax1.get_xlim()
            ylim = ax1.get_ylim()
            
            # Define emission lines with wavelengths, labels, and colors
            emission_lines = [
                # Hydrogen Balmer series
                (0.65646140, r'H$\alpha$', 'red'),
                (0.48626830, r'H$\beta$', 'green'),
                (0.43407410, r'H$\gamma$', 'lightgreen'),
                (0.41017500, r'H$\delta$', 'lime'),
                
                # Oxygen lines
                (0.49602949, '[O III]4959', 'blue'),
                (0.50082397, '[O III]5007', 'cyan'),
                (0.37270917, '[O II]3726', 'orange'),
                (0.37298754, '[O II]3729', 'darkorange'),
                (0.63004000, '[O I]6300', 'coral'),
                (0.63640000, '[O I]6364', 'lightcoral'),
                
                # Nitrogen lines
                (0.65498590, '[N II]6548', 'magenta'),
                (0.65852685, '[N II]6583', 'purple'),
                
                # Sulfur lines
                (0.67316300, '[S II]6716', 'pink'),
                (0.67312100, '[S II]6731', 'hotpink'),
                (0.95323000, '[S III]9532', 'plum'),
                
                # Carbon lines
                (0.15491000, 'C IV 1549', 'darkblue'),
                (0.19062000, 'C III] 1909', 'navy'),
                
                # Magnesium lines
                (0.27980000, 'Mg II 2798', 'darkgreen'),
                (0.28035300, 'Mg II 2803', 'forestgreen'),
                
                # Neon lines
                (0.38691000, '[Ne III]3869', 'teal'),
                (0.39685000, '[Ne III]3967', 'darkcyan'),
                (0.24240000, '[Ne IV]2424', 'orchid'),
                
                # Silicon lines
                (0.19347000, 'Si III] 1892', 'brown'),
                (0.14003000, 'Si IV 1394', 'maroon'),
                (0.14082000, 'Si IV 1403', 'darkred'),
                
                # Helium lines
                (0.58756000, 'He I 5876', 'gold'),
                (0.44713000, 'He II 4471', 'yellow'),
                (0.16400000, 'He II 1640', 'khaki'),
                
                # Iron lines (common in AGN)
                (0.42587000, '[Fe II]4259', 'chocolate'),
                (0.51270000, '[Fe II]5127', 'sienna'),
                (0.16300000, '[Fe II]1630', 'saddlebrown'),
                
                # Lyman series
                (0.12157000, r'Ly$\alpha$ 1216', 'indigo'),
                (0.10260000, r'Ly$\beta$ 1026', 'darkviolet'),
                
                # Calcium lines
                (0.39340000, 'Ca II K 3934', 'gray'),
                (0.39690000, 'Ca II H 3969', 'darkgray'),
                (0.85446000, 'Ca II 8542', 'lightgray'),
                (0.85662000, 'Ca II 8662', 'silver'),
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
            
            if visible_lines:
                # Sort by wavelength for better label placement
                visible_lines.sort(key=lambda x: x[0])
                
                # Calculate label position (near top of plot, but not overlapping with data)
                if use_log_scale and ylim[0] > 0:
                    # For log scale, use geometric mean for label position
                    label_y = ylim[1] * 0.85  # 85% of the way up in log space
                else:
                    # For linear scale, use arithmetic position
                    label_y = ylim[0] + 0.85 * (ylim[1] - ylim[0])
                
                # Filter out lines that are too close together to avoid label overlap
                # Calculate minimum separation based on x-axis range and log/linear scale
                if use_log_scale and xlim[0] > 0:
                    # For log scale, use relative separation
                    min_separation = (xlim[1] / xlim[0]) ** 0.02  # ~2% of log range
                    filtered_lines = []
                    for wl, label, color in visible_lines:
                        if not filtered_lines or wl / filtered_lines[-1][0] > min_separation:
                            filtered_lines.append((wl, label, color))
                else:
                    # For linear scale, use absolute separation
                    min_separation = (xlim[1] - xlim[0]) * 0.02  # 2% of range
                    filtered_lines = []
                    for wl, label, color in visible_lines:
                        if not filtered_lines or wl - filtered_lines[-1][0] > min_separation:
                            filtered_lines.append((wl, label, color))
                
                # Draw vertical lines and add labels
                for line_wl, line_label, color in filtered_lines:
                    # Draw vertical line
                    ax1.axvline(x=line_wl, color=color, linestyle='--', 
                              linewidth=0.8, alpha=0.7, zorder=1)
                    
                    # Add label at the top of the line
                    ax1.text(line_wl, label_y, line_label, 
                           rotation=90, verticalalignment='bottom', 
                           horizontalalignment='center',
                           fontsize=emission_line_fontsize, color=color, alpha=0.9,
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                                   alpha=0.8, edgecolor='none'),
                           zorder=2)

        plt.tight_layout()

        

        # Save figure

        if output_file is None:

            # Save PNG in the same directory as the FITS file

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





