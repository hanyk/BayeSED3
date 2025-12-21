from bayesed import BayeSEDInterface, BayeSEDParams, BayeSEDResults, ZParams, RDFParams, SysErrParams
from astropy.table import Table, join, hstack, vstack
import bagpipes as pipes
import numpy as np
import os
import matplotlib.pyplot as plt
import time
from scipy import stats
import corner

# IMPORTANT: SFR units and scaling consistency
# - BayeSED3: Provides log(SFR) in log10(M☉/yr) - already logarithmic
# - BAGPIPES: Provides SFR in M☉/yr - linear scale, needs log10() for comparison
# Both codes use the same physical units (M☉/yr) but different scaling

resolution1=None
input_file='observation/CESS_mock/two.txt'
# c = 2.9979246e+14  # um/s
c = 2.9979246e+18 #angstrom/s

def load_bayesed_input(ID):
    """ Load photometry and/or spectrum from the input file of BayeSED. """
    cat = Table.read(input_file, format='ascii')
    head = cat.meta['comments'][0].split()
    cat_name = head[0]
    Nphot = int(head[1])
    Nother = int(head[2])
    Nspec = int(head[3])

    # load up the relevant columns from the catalogue.

    # Find the correct row for the object we want.
    row=cat['ID']==ID

    # Extract the object we want from the catalogue.
    if Nphot>0:
        fluxes = np.lib.recfunctions.structured_to_unstructured(np.array(cat[row][cat.colnames[5:5+Nphot*2:2]]))
        fluxerrs = np.lib.recfunctions.structured_to_unstructured(np.array(cat[row][cat.colnames[6:5+Nphot*2:2]]))
        # Turn these into a 2D array.
        photometry = np.c_[fluxes.flatten(), fluxerrs.flatten()]

    if Nspec>0:
        Nskip=5+Nphot*2+Nother+Nspec
        waves = np.lib.recfunctions.structured_to_unstructured(np.array(cat[row][cat.colnames[Nskip+0::4]]))
        waves = waves.flatten()
        waves = waves*1e4
        fluxes = np.lib.recfunctions.structured_to_unstructured(np.array(cat[row][cat.colnames[Nskip+1::4]]))
        fluxes = fluxes.flatten()
        fluxes = (c*waves**-1*fluxes*1e-29)/waves # uJy*Hz = 1e-29 erg/s/cm^2
        fluxerrs = np.lib.recfunctions.structured_to_unstructured(np.array(cat[row][cat.colnames[Nskip+2::4]]))
        fluxerrs = fluxerrs.flatten()
        fluxerrs = (c*waves**-1*fluxerrs*1e-29)/waves # uJy*Hz = 1e-29 erg/s/cm^2
        wdisp = np.lib.recfunctions.structured_to_unstructured(np.array(cat[row][cat.colnames[Nskip+3::4]]))
        wdisp = wdisp.flatten()
        wdisp = wdisp*1e4
        resolution = np.c_[waves,waves/(2.35*wdisp)]
        resolution1=resolution
        # Turn these into a 2D array.
        spectrum = np.c_[waves, fluxes, fluxerrs]
        mask=fluxerrs>0
        return spectrum[mask]
    # blow up the errors associated with any missing fluxes.
    # for i in range(len(photometry)):
    # if (photometry[i, 0] == 0.) or (photometry[i, 1] <= 0):
    # photometry[i,:] = [0., 9.9*10**99.]

    # # Enforce a maximum SNR of 20, or 10 in the IRAC channels.
    # for i in range(len(photometry)):
    # if i < 10:
    # max_snr = 20.

        # else:
        # max_snr = 10.

        # if photometry[i, 0]/photometry[i, 1] > max_snr:
        # photometry[i, 1] = photometry[i, 0]/max_snr

    # if Nphot>0 and Nspec>0:
    # return spectrum,photometry
    # if Nphot>0:
    # return photometry
    # if Nspec>0:
    # return spectrum[mask]

def plot_spectrum_posterior_with_residuals(fit, ID, run_name=None, save=True, show=False, runtime_s=None):
    """Plot spectrum posterior with residuals as a tight stacked subplot.

    The top panel shows the observed spectrum and model posterior (median and 1σ).
    The bottom panel shows residuals (data − model) / error, sharing the x-axis
    with no vertical spacing between panels.
    """

    # Match font sizes with plot_bestfit.py
    plt.rcParams.update({
        'font.size': 14,
        'axes.titlesize': 16,
        'axes.labelsize': 14,
        'legend.fontsize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12
    })

    # Guard: require spectrum
    if not getattr(fit.galaxy, "spectrum_exists", False):
        return None, None

    # Ensure derived posterior arrays (e.g. "spectrum") are available
    try:
        fit.posterior.get_advanced_quantities()
    except Exception:
        pass

    spectrum = fit.galaxy.spectrum  # shape: (N, 3) = [wavelength, flux, flux_err]

    # y-scale similar to bagpipes plotting utilities
    ymax = 1.05 * np.max(spectrum[:, 1])
    y_scale = float(int(np.log10(ymax)) - 1)

    wavs = spectrum[:, 0]
    obs_flux = spectrum[:, 1] * 10 ** (-y_scale)
    obs_err = spectrum[:, 2] * 10 ** (-y_scale)

    # Build posterior spectrum percentiles on the observed grid
    # Prefer posterior spectrum on the observed grid (provided by bagpipes)
    if "spectrum" in fit.posterior.samples:
        spec_post = np.copy(fit.posterior.samples["spectrum"])  # (Nsamples, Npix)
    elif "spectrum_full" in fit.posterior.samples:
        # Fallback: use full model spectrum and interpolate to observed wavelengths
        full_wavs = fit.posterior.model_galaxy.wavelengths
        full_spec_samples = fit.posterior.samples["spectrum_full"]  # (Nsamples, Nfull)
        spec_post = np.array([
            np.interp(spectrum[:, 0], full_wavs, s) for s in full_spec_samples
        ])
    else:
        raise KeyError("Posterior samples do not contain 'spectrum' or 'spectrum_full'.")

    if "calib" in list(fit.posterior.samples):
        spec_post = spec_post / fit.posterior.samples["calib"]

    if "noise" in list(fit.posterior.samples):
        spec_post = spec_post + fit.posterior.samples["noise"]

    post = np.percentile(spec_post, (16, 50, 84), axis=0).T * 10 ** (-y_scale)
    model_med = post[:, 1]  # Define model_med here for use in title calculation

    # Figure with no vertical space between panels
    fig, (ax_main, ax_resid) = plt.subplots(
        2,
        1,
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0},
        figsize=(10, 8)
    )

    # Main panel: posterior exactly as in add_spectrum_posterior
    ax_main.plot(wavs, post[:, 1], color="sandybrown", zorder=4, lw=1.5)
    ax_main.fill_between(wavs, post[:, 0], post[:, 2], color="sandybrown", zorder=4, alpha=0.75, linewidth=0)
    # Overlay observational data as a continuous line
    ax_main.plot(wavs, obs_flux, color="dodgerblue", lw=0.8, alpha=0.9, zorder=5)
    ax_main.set_ylabel(r"$\mathrm{f_{\lambda}}\ \mathrm{/\, 10^{-17}\ erg\ s^{-1}\ cm^{-2}\ \AA^{-1}}$")

    # Add title similar to plot_bestfit.py
    # Calculate posterior median redshift.
    if "redshift" in fit.fitted_model.params:
        redshift = np.median(fit.posterior.samples["redshift"])
    else:
        redshift = fit.fitted_model.model_components["redshift"]

    # Calculate SNR (signal-to-noise ratio)
    snr = np.mean(obs_flux / obs_err) if np.any(obs_err > 0) else 0.0

    # Calculate chi^2
    residuals = (obs_flux - model_med) / obs_err
    chi2 = np.sum(residuals**2)
    n_points = len(obs_flux)
    dof = n_points - 1  # degrees of freedom (simplified)
    chi2_red = chi2 / dof if dof > 0 else np.nan

    # title = f"ID={fit.galaxy.ID}, z={redshift:.3f}, SNR={snr:.2f}, $\\chi^2_\\nu$={chi2_red:.2f}"
    # Append runtime if provided
    runtime_txt = f", runtime={runtime_s:.1f}s" if isinstance(runtime_s, (int, float)) else ""
    title = f"ID={fit.galaxy.ID}, z={redshift:.3f}, $\\chi^2_\\nu$={chi2_red:.2f}{runtime_txt}"
    ax_main.set_title(title)
    # Annotate code used at the center of the top subplot
    ax_main.text(0.5, 0.5, "BAGPIPES", transform=ax_main.transAxes,
                 ha="center", va="center", fontsize=20, color="black", alpha=0.3)

    # Residuals (data - model)/err
    with np.errstate(divide="ignore", invalid="ignore"):
        resid = (obs_flux - model_med) / obs_err

    ax_resid.axhline(0.0, color="gray", lw=1.0, zorder=1)
    ax_resid.plot(wavs, resid, color="dodgerblue", lw=0.9, alpha=0.9, zorder=2)
    ax_resid.set_ylabel("(obs-mod)/err")
    ax_resid.set_xlabel(r"$\lambda / \mathrm{\AA}$")
    ax_resid.set_xlim(wavs[0], wavs[-1])
    # ax_resid.set_ylim(-5, 5)

    # Tighten spacing further
    plt.subplots_adjust(hspace=0)

    if save:
        # Save alongside bagpipes convention
        plot_dir = os.path.join("pipes", "plots", fit.run)
        os.makedirs(plot_dir, exist_ok=True)
        out_path = os.path.join(plot_dir, f"{fit.galaxy.ID}_fit_with_residuals.png")
        plt.savefig(out_path, bbox_inches="tight", dpi=400)

    if show:
        plt.show()

def plot_posterior_comparison(bayesed_results, bagpipes_fit, object_id, save=True, show=False):
    """Compare posterior distributions between BayeSED3 and BAGPIPES.

    Parameters:
    -----------
    bayesed_results : BayeSEDResults
        BayeSED3 results object
    bagpipes_fit : bagpipes.fit
        BAGPIPES fit object
    object_id : str
        Object ID for comparison
    save : bool
        Whether to save the plot
    show : bool
        Whether to display the plot
    """

    # Common parameters to compare
    common_params = ['stellar_mass', 'sfr', 'age', 'metallicity', 'Av']

    # Parameter mapping between BayeSED3 and BAGPIPES
    # Note: SFR units are solar masses per year (M☉/yr) for both codes
    # Source: BAGPIPES documentation explicitly states SFR is in "Solar masses per year"
    param_mapping = {
        'stellar_mass': ('stellar_mass', 'stellar_mass'),
        'sfr': ('sfr', 'sfr'),  # Both in M☉/yr
        'age': ('age_mass_weighted', 'age'),
        'metallicity': ('metallicity_mass_weighted', 'metallicity'),
        'Av': ('Av', 'Av')
    }

    # Get BayeSED3 posterior samples
    bayesed_samples = {}
    try:
        obj_results = bayesed_results.get_object_results(object_id)
        for param, (bayesed_key, _) in param_mapping.items():
            if hasattr(obj_results, bayesed_key):
                bayesed_samples[param] = getattr(obj_results, bayesed_key)
    except Exception as e:
        print(f"Warning: Could not extract BayeSED3 samples: {e}")
        return None

    # Get BAGPIPES posterior samples
    bagpipes_samples = {}
    try:
        for param, (_, bagpipes_key) in param_mapping.items():
            if bagpipes_key in bagpipes_fit.posterior.samples:
                bagpipes_samples[param] = bagpipes_fit.posterior.samples[bagpipes_key]
    except Exception as e:
        print(f"Warning: Could not extract BAGPIPES samples: {e}")
        return None

    # Find common parameters with data
    available_params = set(bayesed_samples.keys()) & set(bagpipes_samples.keys())
    if not available_params:
        print("No common parameters found for comparison")
        return None

    n_params = len(available_params)
    fig, axes = plt.subplots(2, (n_params + 1) // 2, figsize=(4 * ((n_params + 1) // 2), 8))
    if n_params == 1:
        axes = [axes]
    axes = axes.flatten()

    for i, param in enumerate(sorted(available_params)):
        ax = axes[i]

        # Plot histograms
        bayesed_data = bayesed_samples[param]
        bagpipes_data = bagpipes_samples[param]

        # Handle log scale for stellar mass and SFR
        if param == 'stellar_mass':
            bayesed_data = np.log10(bayesed_data)
            bagpipes_data = np.log10(bagpipes_data)
            xlabel = f'log({param})'
        elif param == 'sfr':
            # BayeSED3 SFR is already in log, BAGPIPES SFR needs log conversion
            # bayesed_data is already log(SFR), bagpipes_data needs log10()
            bagpipes_data = np.log10(bagpipes_data)
            xlabel = 'log(SFR)'
        else:
            xlabel = param

        ax.hist(bayesed_data, bins=30, alpha=0.6, label='BayeSED3',
                color='blue', density=True)
        ax.hist(bagpipes_data, bins=30, alpha=0.6, label='BAGPIPES',
                color='orange', density=True)

        # Add median lines
        ax.axvline(np.median(bayesed_data), color='blue', linestyle='--', alpha=0.8)
        ax.axvline(np.median(bagpipes_data), color='orange', linestyle='--', alpha=0.8)

        ax.set_xlabel(xlabel)
        ax.set_ylabel('Density')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Add statistics text
        bayesed_median = np.median(bayesed_data)
        bagpipes_median = np.median(bagpipes_data)
        ax.text(0.05, 0.95, f'BayeSED3: {bayesed_median:.3f}\nBAGPIPES: {bagpipes_median:.3f}',
                transform=ax.transAxes, verticalalignment='top', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Remove empty subplots
    for i in range(n_params, len(axes)):
        fig.delaxes(axes[i])

    plt.suptitle(f'Object: {object_id}', fontsize=16)
    plt.tight_layout()

    if save:
        plot_dir = os.path.join("pipes", "plots", "comparison")
        os.makedirs(plot_dir, exist_ok=True)
        out_path = os.path.join(plot_dir, f"{object_id}_posterior_comparison.png")
        plt.savefig(out_path, bbox_inches="tight", dpi=300)
        print(f"Saved comparison plot: {out_path}")

    if show:
        plt.show()

    plt.close(fig)
    return fig


def plot_corner_comparison(bayesed_results, bagpipes_fit, object_id,
                                bayesed_params=None, bagpipes_params=None,
                                labels=None, save=True, show=False):
    """Create corner plots comparing BayeSED3 and BAGPIPES posterior samples using GetDist.

    Parameters:
    -----------
    bayesed_results : BayeSEDResults
        BayeSED3 results object
    bagpipes_fit : bagpipes.fit
        BAGPIPES fit object
    object_id : str
        Object ID for comparison
    bayesed_params : list of str, optional
        List of BayeSED3 parameter names to plot.
        If None, will print available parameters and return.
    bagpipes_params : list of str, optional
        List of BAGPIPES parameter names to plot (must match length of bayesed_params).
    labels : list of str, optional
        List of labels for the plot axes. If None, uses bayesed_params.
    save : bool
        Whether to save the plot
    show : bool
        Whether to display the plot

    Example:
    --------
    # First run to see available parameters
    plot_corner_comparison(results, fit, object_id)

    # Then specify parameters to compare
    bayesed_params = ['log(Mstar)', 'log(SFR_{10Myr}/[M_{sun}/yr])', 'Av_2']
    bagpipes_params = ['stellar_mass', 'sfr', 'Av']
    labels = ['log(M*)', 'log(SFR)', 'Av']  # Note: SFR will be log-scaled for both
    plot_corner_comparison(results, fit, object_id, bayesed_params, bagpipes_params, labels)
    """

    try:
        from getdist import plots, MCSamples
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("GetDist is required for corner plot comparison. Install with: pip install getdist")

    try:
        # Get BayeSED3 posterior samples (GetDist samples object)
        bayesed_samples = bayesed_results.get_posterior_samples(object_id=object_id)

        # Get available parameters from both codes
        if hasattr(bayesed_samples, 'samples'):
            sample_array = bayesed_samples.samples
            # For GetDist samples, parameter names are accessed differently
            if hasattr(bayesed_samples, 'getParamNames'):
                # GetDist MCSamples object
                param_objects = bayesed_samples.getParamNames()
                bayesed_param_names = [p.name for p in param_objects.names]
            elif hasattr(bayesed_samples, 'paramNames'):
                # Alternative GetDist access pattern
                if hasattr(bayesed_samples.paramNames, 'names'):
                    bayesed_param_names = bayesed_samples.paramNames.names
                else:
                    bayesed_param_names = [p.name for p in bayesed_samples.paramNames]
            else:
                # Fallback - try to get parameter names from the samples object
                bayesed_param_names = [f'param_{i}' for i in range(sample_array.shape[1])]
        else:
            print("BayeSED3 samples format not recognized")
            return None

        bagpipes_param_names = list(bagpipes_fit.posterior.samples.keys())

        # If no parameters specified, show available parameters
        if bayesed_params is None or bagpipes_params is None:
            print(f"\n=== Available Parameters ===")
            print(f"\nBayeSED3 parameters ({len(bayesed_param_names)}):")
            for i, param in enumerate(bayesed_param_names):
                print(f"  {i:2d}: {param}")

            print(f"\nBAGPIPES parameters ({len(bagpipes_param_names)}):")
            for i, param in enumerate(bagpipes_param_names):
                print(f"  {i:2d}: {param}")

            print(f"\nTo create comparison plot, specify parameter lists:")
            print(f"bayesed_params = ['param1', 'param2', ...]")
            print(f"bagpipes_params = ['param1', 'param2', ...]")
            print(f"labels = ['label1', 'label2', ...]  # optional")
            print(f"plot_corner_comparison(results, fit, '{object_id}', bayesed_params, bagpipes_params, labels)")
            return None

        # Check parameter lists have same length
        if len(bayesed_params) != len(bagpipes_params):
            print(f"Error: bayesed_params ({len(bayesed_params)}) and bagpipes_params ({len(bagpipes_params)}) must have same length")
            return None

        # Use bayesed_params as labels if not provided
        if labels is None:
            labels = bayesed_params
        elif len(labels) != len(bayesed_params):
            print(f"Error: labels ({len(labels)}) must match length of parameter lists ({len(bayesed_params)})")
            return None

        # Collect specified parameters
        bayesed_data = []
        bagpipes_data = []
        final_labels = []

        for i, (bayesed_param, bagpipes_param, label) in enumerate(zip(bayesed_params, bagpipes_params, labels)):
            # Check BayeSED3 parameter
            if bayesed_param not in bayesed_param_names:
                print(f"Warning: BayeSED3 parameter '{bayesed_param}' not found, skipping")
                continue

            # Check BAGPIPES parameter
            if bagpipes_param not in bagpipes_param_names:
                print(f"Warning: BAGPIPES parameter '{bagpipes_param}' not found, skipping")
                continue

            # Extract data
            bayesed_idx = bayesed_param_names.index(bayesed_param)
            bayesed_vals = sample_array[:, bayesed_idx]
            bagpipes_vals = bagpipes_fit.posterior.samples[bagpipes_param]
            
            # Handle SFR scaling consistency: BayeSED3 SFR is already log, BAGPIPES SFR needs log conversion
            if 'SFR' in bayesed_param and bagpipes_param == 'sfr':
                # BayeSED3 SFR is already in log scale, BAGPIPES SFR needs log conversion
                bagpipes_vals = np.log10(bagpipes_vals)
                print(f"  Applied log10 to BAGPIPES SFR for consistency with BayeSED3")

            bayesed_data.append(bayesed_vals)
            bagpipes_data.append(bagpipes_vals)
            final_labels.append(label)

            print(f"✓ Loaded: {bayesed_param} vs {bagpipes_param} -> {label}")

        if not bayesed_data or not bagpipes_data:
            print("No valid parameters found for comparison")
            return None

        print(f"\nCreating GetDist corner plot with {len(final_labels)} parameters: {final_labels}")

        # Get BayeSED3 GetDist samples directly (already a MCSamples object)
        bayesed_getdist_samples = bayesed_results.get_getdist_samples(object_id=object_id)
        
        # Create BAGPIPES MCSamples object
        bagpipes_samples_array = np.column_stack(bagpipes_data)
        print(f"BayeSED3 samples shape: {bayesed_getdist_samples.samples.shape}")
        print(f"BAGPIPES samples shape: {bagpipes_samples_array.shape}")
        
        # Create clean parameter names for GetDist (no spaces, *, or ?)
        clean_names = []
        for label in final_labels:
            clean_name = label.replace(' ', '_').replace('*', 'star').replace('?', 'q').replace('(', '').replace(')', '')
            clean_names.append(clean_name)
        
        # Create BAGPIPES MCSamples object with clean names and nice labels
        bagpipes_mcsamples = MCSamples(samples=bagpipes_samples_array, names=clean_names, 
                                     labels=final_labels, name_tag='BAGPIPES')
        
        # Set labels for BayeSED3 samples (following BayeSED's approach)
        bayesed_getdist_samples.name_tag = 'BayeSED3'
        bayesed_getdist_samples.label = 'BayeSED3'
        
        # Filter BayeSED3 samples to only include the parameters we want to compare
        # Extract the relevant parameter indices from BayeSED3 samples
        bayesed_param_names = [p.name for p in bayesed_getdist_samples.paramNames.names]
        bayesed_indices = []
        bayesed_filtered_labels = []
        
        for i, (bayesed_param, label) in enumerate(zip(bayesed_params, final_labels)):
            if bayesed_param in bayesed_param_names:
                idx = bayesed_param_names.index(bayesed_param)
                bayesed_indices.append(idx)
                bayesed_filtered_labels.append(label)
        
        # Create filtered BayeSED3 samples
        if bayesed_indices:
            bayesed_filtered_samples = bayesed_getdist_samples.samples[:, bayesed_indices]
            bayesed_filtered_mcsamples = MCSamples(samples=bayesed_filtered_samples, 
                                                 names=clean_names[:len(bayesed_indices)], 
                                                 labels=bayesed_filtered_labels, 
                                                 name_tag='BayeSED3')
        else:
            print("No matching parameters found in BayeSED3 samples")
            return None
        
        # Create GetDist triangle plot following BayeSED's approach
        g = plots.get_subplot_plotter(width_inch=12, subplot_size=3.0)
        
        # Use BayeSED's plotting style for better comparison visibility
        g.settings.figure_legend_frame = True
        g.settings.figure_legend_loc = 'upper right'
        g.settings.legend_fontsize = 12
        g.settings.axes_fontsize = 11
        g.settings.lab_fontsize = 12
        g.settings.tight_layout = True
        g.settings.axes_labelsize = 12
        
        # Use BayeSED's plotting approach with samples list
        samples_list = [bayesed_filtered_mcsamples, bagpipes_mcsamples]
        
        # Set plotting options for better comparison visibility (following BayeSED's approach)
        plot_kwargs = {
            'filled': True,
            'contour_colors': ['#2E86AB', '#F24236'],  # BayeSED3 blue, BAGPIPES red
            'contour_ls': ['-', '--'],  # Solid for BayeSED3, dashed for BAGPIPES
            'contour_lws': [1.5, 2.0],
        }
        
        # Plot using BayeSED's method
        g.triangle_plot(samples_list, clean_names[:len(bayesed_indices)], **plot_kwargs)
        
        # Clean title
        plt.suptitle(f'Object: {object_id}', 
                   fontsize=14, y=0.95, fontweight='bold')
        
        # Get the current figure
        fig = plt.gcf()

        if save:
            plot_dir = os.path.join("pipes", "plots", "comparison")
            os.makedirs(plot_dir, exist_ok=True)
            out_path = os.path.join(plot_dir, f"{object_id}_corner_comparison_getdist.png")
            plt.savefig(out_path, bbox_inches="tight", dpi=300, facecolor='white')
            print(f"Saved GetDist corner comparison: {out_path}")

        if show:
            plt.show()

        if not show:
            plt.close(fig)
        return fig

    except Exception as e:
        print(f"Error in corner plot comparison: {e}")
        import traceback
        traceback.print_exc()
        return None


def plot_parameter_scatter(bayesed_results, bagpipes_fits, object_ids, save=True, show=False):
    """Create scatter plots comparing derived parameters between BayeSED3 and BAGPIPES.

    Parameters:
    -----------
    bayesed_results : BayeSEDResults
        BayeSED3 results object
    bagpipes_fits : dict
        Dictionary of BAGPIPES fit objects keyed by object ID
    object_ids : list
        List of object IDs to compare
    save : bool
        Whether to save the plot
    show : bool
        Whether to display the plot
    """

    param_mapping = {
        'Stellar Mass': ('stellar_mass', 'stellar_mass', True),  # True for log scale
        'SFR': ('sfr', 'sfr', True),  # True for log scale (BayeSED3 already log, BAGPIPES needs log)
        'Age': ('age_mass_weighted', 'age', False),
        'Metallicity': ('metallicity_mass_weighted', 'metallicity', False),
        'Av': ('Av', 'Av', False)
    }

    n_params = len(param_mapping)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for i, (param_name, (bayesed_key, bagpipes_key, log_scale)) in enumerate(param_mapping.items()):
        if i >= len(axes):
            break

        ax = axes[i]
        bayesed_vals = []
        bagpipes_vals = []

        for obj_id in object_ids:
            try:
                # Get BayeSED3 value (median)
                obj_results = bayesed_results.get_object_results(obj_id)
                if hasattr(obj_results, bayesed_key):
                    bayesed_val = np.median(getattr(obj_results, bayesed_key))
                    # BayeSED3 SFR is already in log scale, stellar mass needs log conversion
                    if log_scale and param_name != 'SFR':
                        bayesed_val = np.log10(bayesed_val)

                    # Get BAGPIPES value (median)
                    if obj_id in bagpipes_fits and bagpipes_key in bagpipes_fits[obj_id].posterior.samples:
                        bagpipes_val = np.median(bagpipes_fits[obj_id].posterior.samples[bagpipes_key])
                        # Apply log scaling to BAGPIPES values (both stellar mass and SFR need log)
                        if log_scale:
                            bagpipes_val = np.log10(bagpipes_val)

                        bayesed_vals.append(bayesed_val)
                        bagpipes_vals.append(bagpipes_val)
            except Exception as e:
                continue

        if bayesed_vals and bagpipes_vals:
            ax.scatter(bayesed_vals, bagpipes_vals, alpha=0.7, s=50)

            # Add 1:1 line
            min_val = min(min(bayesed_vals), min(bagpipes_vals))
            max_val = max(max(bayesed_vals), max(bagpipes_vals))
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, label='1:1')

            # Calculate correlation
            correlation = np.corrcoef(bayesed_vals, bagpipes_vals)[0, 1]
            ax.text(0.05, 0.95, f'r = {correlation:.3f}', transform=ax.transAxes,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

            xlabel = f'BayeSED3 {"log(" + param_name + ")" if log_scale else param_name}'
            ylabel = f'BAGPIPES {"log(" + param_name + ")" if log_scale else param_name}'
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.3)
            ax.legend()

    # Remove empty subplots
    for i in range(n_params, len(axes)):
        fig.delaxes(axes[i])

    plt.suptitle('Parameter Comparison: BayeSED3 vs BAGPIPES', fontsize=16)
    plt.tight_layout()

    if save:
        plot_dir = os.path.join("pipes", "plots", "comparison")
        os.makedirs(plot_dir, exist_ok=True)
        out_path = os.path.join(plot_dir, "parameter_scatter_comparison.png")
        plt.savefig(out_path, bbox_inches="tight", dpi=300)
        print(f"Saved scatter comparison: {out_path}")

    if show:
        plt.show()

    plt.close(fig)
    return fig

def main():
    """Main function to run BayeSED3 vs BAGPIPES comparison."""
    # Initialize interface
    bayesed = BayeSEDInterface(mpi_mode='auto')

    # Simple galaxy fitting with no_photometry_fit=True (fit spectra only)
    params = BayeSEDParams.galaxy(
        input_file=input_file,
        filters='observation/CESS_mock/filters_bassmzl.txt',
        filters_selected='observation/CESS_mock/filters_selected_csst.txt',
        outdir='observation/CESS_mock/output',
        ssp_model='bc2003_hr_stelib_chab_neb_300r',
        sfh_type='exponential',
        dal_law='calzetti',
        ssp_iscalable=0,          # Use MultiNest sampling for normalization (more robust for low-SNR data)
        ssp_i1=1,                 #enable nebular emission
        sfh_itype_ceh=1,          #Chemical evolution enabled (metallicity evolves with time)
        no_photometry_fit=True,   # Skip photometry fitting
        no_spectra_fit=False      # Fit spectra (default)
    )

    # Set redshift prior to match BAGPIPES range (0.0 to 2.0)
    params.z = ZParams(
        iprior_type=1,
        min=0.0,
        max=2.0,
    )

    # Set RDF parameters for modeling sigma_diff between observed spectra and model
    # RDF models the difference/scatter between observations and theoretical models
    params.rdf = RDFParams(
        id=-1,                   # Model ID (-1: apply to all models, 0,1,2...: specific model)
        num_polynomials=0       # Number of polynomials (-1: default/disable, 0,1,2...: polynomial order)
    )

    # Set systematic error for model
    params.sys_err_mod = SysErrParams(
        iprior_type=3,    # Prior type (1=uniform, 3=log-uniform)
        min=0.01,         # Minimum fractional systematic error
        max=0.1,          # Maximum fractional systematic error
    )


    # Run analysis
    result = bayesed.run(params)

    # Load and analyze results
    results = BayeSEDResults('observation/CESS_mock/output', catalog_name='seedcat2')
    results.print_summary()
    results.plot_bestfit('5494348_STARFORMING')
    results.plot_bestfit('11184100_QUIESCENT')


    # goodss_filt_list = np.loadtxt("filters/goodss_filt_list.txt", dtype="str")
    # spectrum,photometry=load_bayesed_input('10524881')
    # IDs=cat['ID'].__array__().astype(str)
    IDs=results.list_objects()
    galaxy = pipes.galaxy(IDs[0], load_bayesed_input,photometry_exists=False)
    # galaxy = pipes.galaxy('25658916', load_bayesed_input,photometry_exists=False)

    exp = {}
    exp["age"] = (0.1, 15.)
    exp["tau"] = (0.3, 10.)
    exp["massformed"] = (1., 15.)
    exp["metallicity"] = (0., 2.5)

    dust = {}
    dust["type"] = "Calzetti"
    dust["Av"] = (0., 4.)

    nebular = {}
    nebular["logU"] = -2.3

    fit_instructions = {}
    fit_instructions["redshift"] = (0., 2.)
    fit_instructions["exponential"] = exp
    fit_instructions["dust"] = dust
    fit_instructions["nebular"] = nebular
    fit_instructions["veldisp"] = 300  #km/s
    # fit_instructions["veldisp"] = (1., 1000.)   #km/s
    # fit_instructions["veldisp_prior"] = "log_10"
    fit_instructions["R_curve"] = resolution1

    noise = {}
    noise["type"] = "white_scaled"
    noise["scaling"] = (1., 1.5)
    noise["scaling_prior"] = "log_10"
    fit_instructions["noise"] = noise

    # plt.plot(fit_instructions["R_curve"][:, 0], fit_instructions["R_curve"][:, 1])
    # plt.xlabel("Wavelength/ \AA")
    # plt.ylabel("Spectral resolving power")
    # plt.show()

    cat_name=results.catalog_name
    bagpipes_fits = {}  # Store fits for posterior comparison

    for ID in IDs:
        # Create individual galaxy object
        galaxy = pipes.galaxy(ID, load_bayesed_input, photometry_exists=False)

        # Fit individual galaxy with timing
        fit = pipes.fit(galaxy, fit_instructions, run=f"{cat_name}_{ID}")
        t0 = time.time()
        fit.fit(verbose=True, sampler='nautilus', n_live=400, pool=10)
        runtime_s = time.time() - t0

        # Store fit for comparison
        bagpipes_fits[ID] = fit

        # Generate spectrum plots
        plot_spectrum_posterior_with_residuals(fit, ID, cat_name, runtime_s=runtime_s)

        # Generate corner comparison plots using GetDist for better aesthetics
        print(f"\nGenerating corner comparison plot for object {ID}...")
        # Use the correct parameter names based on the available parameters
        # Note: BayeSED3 derived parameters have '*' suffix
        bayesed_params = ['z', 'log(Mstar)[0,0]', 'log(SFR_{100Myr}/[M_{sun}/yr])[0,0]', 'Av_2[0,0]']
        bagpipes_params = ['redshift', 'stellar_mass', 'sfr', 'dust:Av']
        labels = ['z', 'log(Mstar)', 'log(SFR)', 'Av']
        plot_corner_comparison(results, fit, ID, bayesed_params, bagpipes_params, labels)

    # fit_cat = pipes.fit_catalogue(IDs, fit_instructions, load_bayesed_input, photometry_exists=False, run=cat_name, make_plots=False,n_posterior=500)
    # # fit_cat.fit(verbose=True,pool=10,n_live=400)
    # fit_cat.fit(verbose=True,mpi_serial=True,n_live=400)

if __name__ == "__main__":
    main()

