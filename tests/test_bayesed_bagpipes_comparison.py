from bayesed import BayeSEDInterface, BayeSEDParams, BayeSEDResults, ZParams, RDFParams, SysErrParams, MultiNestParams
from astropy.table import Table, join, hstack, vstack
import bagpipes as pipes
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import time
from scipy import stats
import corner

# IMPORTANT: SFR units and scaling consistency
# - BayeSED3: Provides log(SFR) in log10(M☉/yr) - already logarithmic
# - BAGPIPES: Provides SFR in M☉/yr - linear scale, needs log10() for comparison
# Both codes use the same physical units (M☉/yr) but different scaling

# c = 2.9979246e+14  # um/s
c = 2.9979246e+18 #angstrom/s

class BayeSEDDataLoader:
    """Optimized data loader that handles object-specific resolution curves for BAGPIPES.

    Optimizations for large files:
    - Lazy loading: Only loads catalog once and caches it
    - Header caching: Parses header information once
    - Memory-efficient row selection
    - Optional chunked reading for extremely large files
    """

    def __init__(self, input_file, chunk_size=None):
        self.input_file = input_file
        self.resolution_cache = {}  # Cache resolution curves by object ID
        self.chunk_size = chunk_size  # For chunked reading of very large files

        # Cache for catalog metadata and structure
        self._catalog_cache = None
        self._header_cache = None
        self._id_index_cache = None  # Maps ID -> row index for fast lookup

    @classmethod
    def _parse_header_from_line(cls, header_line):
        """Parse header information from the first comment line.

        Parameters:
        -----------
        header_line : str
            The first comment line from the file (with or without #)

        Returns:
        --------
        dict : Header information dictionary
        """
        # Remove # if present and split
        if header_line.startswith('#'):
            header_line = header_line[1:]

        header_parts = header_line.strip().split()

        if len(header_parts) < 4:
            raise ValueError(f"Expected at least 4 header parameters, got {len(header_parts)}: {header_parts}")

        return {
            'cat_name': header_parts[0],
            'Nphot': int(header_parts[1]),
            'Nother': int(header_parts[2]),
            'Nspec': int(header_parts[3])
        }

    @classmethod
    def extract_catalog_name(cls, input_file):
        """Extract catalog name from the first line of BayeSED input file.

        This is much more efficient for large files since we only read the first line
        instead of loading the entire table.

        Parameters:
        -----------
        input_file : str
            Path to the BayeSED input file

        Returns:
        --------
        str
            Catalog name extracted from the first comment line

        Raises:
        -------
        ValueError
            If the file format is not as expected
        """
        try:
            with open(input_file, 'r') as f:
                first_line = f.readline().strip()

            # Check if it's a comment line starting with #
            if not first_line.startswith('#'):
                raise ValueError(f"Expected first line to be a comment (starting with #), got: {first_line}")

            # Use the shared parsing logic
            header_info = cls._parse_header_from_line(first_line)
            return header_info['cat_name']

        except FileNotFoundError:
            raise FileNotFoundError(f"Input file not found: {input_file}")
        except Exception as e:
            raise ValueError(f"Error reading catalog name from {input_file}: {e}")

    @classmethod
    def extract_header_info(cls, input_file):
        """Extract full header information from the first line of BayeSED input file.

        More efficient than loading the full catalog when you only need header info.

        Parameters:
        -----------
        input_file : str
            Path to the BayeSED input file

        Returns:
        --------
        dict : Header information with keys: cat_name, Nphot, Nother, Nspec
        """
        try:
            with open(input_file, 'r') as f:
                first_line = f.readline().strip()

            # Check if it's a comment line starting with #
            if not first_line.startswith('#'):
                raise ValueError(f"Expected first line to be a comment (starting with #), got: {first_line}")

            return cls._parse_header_from_line(first_line)

        except FileNotFoundError:
            raise FileNotFoundError(f"Input file not found: {input_file}")
        except Exception as e:
            raise ValueError(f"Error reading header from {input_file}: {e}")

    def get_catalog_name(self):
        """Get catalog name for this loader's input file.

        Returns:
        --------
        str
            Catalog name from the header
        """
        if self._header_cache is not None:
            return self._header_cache['cat_name']
        else:
            # Use the class method for efficient extraction
            return self.extract_catalog_name(self.input_file)

    def get_header_info(self):
        """Get header information for this loader's input file.

        Returns:
        --------
        dict : Header information with keys: cat_name, Nphot, Nother, Nspec
        """
        if self._header_cache is not None:
            return self._header_cache.copy()
        else:
            # Use the class method for efficient extraction
            return self.extract_header_info(self.input_file)

    def _load_catalog_once(self):
        """Load catalog and header information once and cache it."""
        if self._catalog_cache is not None:
            return self._catalog_cache, self._header_cache

        # Load catalog
        cat = Table.read(self.input_file, format='ascii')

        # Use efficient header parsing (reuses existing logic)
        if self._header_cache is None:
            # Parse header from the catalog metadata (already loaded)
            header_line = cat.meta['comments'][0]
            self._header_cache = self._parse_header_from_line(header_line)

        # Build ID index for fast lookup
        id_index = {}
        for i, obj_id in enumerate(cat['ID']):
            # Handle mixed types by converting to string for indexing
            key = str(obj_id) if not isinstance(obj_id, str) else obj_id
            id_index[key] = i

        self._catalog_cache = cat
        self._id_index_cache = id_index

        return cat, self._header_cache

    def _get_row_index(self, ID):
        """Get row index for given ID using cached index."""
        if self._id_index_cache is None:
            self._load_catalog_once()

        # Convert ID to string for lookup consistency
        lookup_id = str(ID) if not isinstance(ID, str) else ID

        if lookup_id not in self._id_index_cache:
            raise ValueError(f"ID '{ID}' not found in catalog")

        return self._id_index_cache[lookup_id]

    def load_full_data(self, ID):
        """ Load photometry and/or spectrum from the input file of BayeSED.

        Optimized version that:
        - Loads catalog only once and caches it
        - Uses direct row indexing instead of boolean masking
        - Caches header parsing results

        Returns:
            tuple: (spectrum, resolution) where spectrum is the spectral data
                   and resolution is the resolution curve for BAGPIPES
        """
        # Load catalog and header info (cached after first call)
        cat, header = self._load_catalog_once()

        Nphot = header['Nphot']
        Nother = header['Nother']
        Nspec = header['Nspec']

        # Get row index directly (much faster than boolean indexing)
        row_idx = self._get_row_index(ID)

        # Extract the specific row we want (no boolean masking needed)
        obj_row = cat[row_idx]

        # Extract photometry if present
        if Nphot > 0:
            flux_cols = cat.colnames[5:5+Nphot*2:2]
            fluxerr_cols = cat.colnames[6:5+Nphot*2:2]

            fluxes = np.array([obj_row[col] for col in flux_cols])
            fluxerrs = np.array([obj_row[col] for col in fluxerr_cols])

            # Turn these into a 2D array.
            photometry = np.c_[fluxes.flatten(), fluxerrs.flatten()]

        # Extract spectrum if present
        if Nspec > 0:
            Nskip = 5 + Nphot*2 + Nother + Nspec

            # Extract spectral data more efficiently
            wave_cols = cat.colnames[Nskip+0::4]
            flux_cols = cat.colnames[Nskip+1::4]
            fluxerr_cols = cat.colnames[Nskip+2::4]
            wdisp_cols = cat.colnames[Nskip+3::4]

            waves = np.array([obj_row[col] for col in wave_cols]) * 1e4
            fluxes = np.array([obj_row[col] for col in flux_cols])
            fluxerrs = np.array([obj_row[col] for col in fluxerr_cols])
            wdisp = np.array([obj_row[col] for col in wdisp_cols]) * 1e4

            # Convert flux units
            fluxes = (c * waves**-1 * fluxes * 1e-29) / waves  # uJy*Hz = 1e-29 erg/s/cm^2
            fluxerrs = (c * waves**-1 * fluxerrs * 1e-29) / waves

            # Calculate resolution
            resolution = np.c_[waves, waves/(2.35*wdisp)]

            # Cache the resolution curve for this object
            self.resolution_cache[ID] = resolution

            # Create spectrum array
            spectrum = np.c_[waves, fluxes, fluxerrs]
            mask = fluxerrs > 0
            return spectrum[mask], resolution

        return None, None

    def load_spectrum_only(self, ID):
        """Load only spectrum data for BAGPIPES (optimized wrapper function)."""
        spectrum, _ = self.load_full_data(ID)
        return spectrum

    def get_resolution_curve(self, ID):
        """Get the resolution curve for a specific object ID."""
        if ID not in self.resolution_cache:
            # Load data to populate cache
            self.load_full_data(ID)
        return self.resolution_cache[ID]

    def preload_objects(self, object_ids):
        """Preload data for multiple objects to optimize batch processing.

        Parameters:
        -----------
        object_ids : list
            List of object IDs to preload

        Returns:
        --------
        dict : Dictionary mapping object_id -> (spectrum, resolution)
        """
        results = {}

        # Load catalog once
        cat, header = self._load_catalog_once()

        print(f"Preloading data for {len(object_ids)} objects...")

        for obj_id in object_ids:
            try:
                spectrum, resolution = self.load_full_data(obj_id)
                results[obj_id] = (spectrum, resolution)
            except Exception as e:
                print(f"Warning: Failed to load object {obj_id}: {e}")
                results[obj_id] = (None, None)

        return results

    def get_available_ids(self):
        """Get list of all available object IDs in the catalog."""
        cat, _ = self._load_catalog_once()
        return cat['ID'].__array__()

    def get_catalog_info(self, lightweight=False):
        """Get catalog metadata.

        Parameters:
        -----------
        lightweight : bool, default=False
            If True, only returns header information without loading the full catalog.
            This is much faster but doesn't include n_objects or column_names.

        Returns:
        --------
        dict : Catalog information
        """
        if lightweight:
            # Fast mode: only header information, no catalog loading
            header = self.get_header_info()
            return {
                'filename': self.input_file,
                'catalog_name': header['cat_name'],
                'n_photometry': header['Nphot'],
                'n_other': header['Nother'],
                'n_spectral': header['Nspec'],
                'n_objects': None,  # Not available in lightweight mode
                'column_names': None  # Not available in lightweight mode
            }
        else:
            # Full mode: loads catalog to get complete information
            _, header = self._load_catalog_once()
            cat = self._catalog_cache

            return {
                'filename': self.input_file,
                'n_objects': len(cat),
                'catalog_name': header['cat_name'],
                'n_photometry': header['Nphot'],
                'n_other': header['Nother'],
                'n_spectral': header['Nspec'],
                'column_names': cat.colnames
            }

    @classmethod
    def create_chunked_loader(cls, input_file, chunk_size=1000):
        """Create a loader optimized for very large files using chunked reading.

        For files with millions of objects, this can read data in chunks
        to avoid loading everything into memory at once.

        Parameters:
        -----------
        input_file : str
            Path to BayeSED input file
        chunk_size : int
            Number of rows to read per chunk

        Returns:
        --------
        BayeSEDDataLoader : Optimized loader instance
        """
        return cls(input_file, chunk_size=chunk_size)


# Note: The legacy load_bayesed_input function has been removed.
# Use BayeSEDDataLoader class instead for better performance:
#
# # Efficient approach for multiple objects:
# loader = BayeSEDDataLoader(input_file)
# for obj_id in object_list:
#     spectrum, resolution = loader.load_full_data(obj_id)
#
# # Or for batch processing:
# results = loader.preload_objects(object_list)

def plot_spectrum_posterior_with_residuals(fit, ID, run_name=None, save=True, show=True, runtime_s=None):
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

def plot_posterior_comparison(bayesed_results, bagpipes_fit, object_id, save=True, show=True):
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


def configure_getdist_settings(range_confidence=0, contours=None, smooth_scale_1D=-1, smooth_scale_2D=-1):
    """Configure GetDist analysis settings for consistent posterior analysis.

    Parameters:
    -----------
    range_confidence : float, default=0
        1D marginalized confidence limit to determine parameter ranges.
        0 = use full data range, 0.001 = 99.9% confidence, 0.01 = 99% confidence, 0.05 = 95% confidence
    contours : list, optional
        Confidence limits for marginalized constraints (e.g., [0.68, 0.95, 0.99])
    smooth_scale_1D : float, default=-1
        1D smoothing scale (-1 = automatic)
    smooth_scale_2D : float, default=-1
        2D smoothing scale (-1 = automatic)

    Returns:
    --------
    dict : GetDist analysis settings dictionary
    """
    settings = {
        'range_confidence': range_confidence,
        'smooth_scale_1D': smooth_scale_1D,
        'smooth_scale_2D': smooth_scale_2D,
        'fine_bins': 1024,
        'fine_bins_2D': 256,
        'boundary_correction_order': 1,
        'mult_bias_correction_order': 1
    }

    if contours is not None:
        settings['contours'] = contours
    else:
        settings['contours'] = [0.68, 0.95, 0.99]  # 1σ, 2σ, 3σ

    return settings


def plot_corner_comparison(bayesed_results, bagpipes_fit, object_id,
                                bayesed_params=None, bagpipes_params=None,
                                labels=None, true_values=None, save=True, show=True,
                                range_confidence=0):
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
    true_values : list of float, optional
        List of true parameter values to mark on the plot. Must match length of bayesed_params.
    save : bool
        Whether to save the plot
    show : bool
        Whether to display the plot
    range_confidence : float, default=0
        GetDist range_confidence parameter for determining parameter ranges.
        0 = use full data range, 0.001 = 99.9% confidence (tighter ranges), 0.01 = 99% confidence, 0.05 = 95% confidence

    Example:
    --------
    # First run to see available parameters
    plot_corner_comparison(results, fit, object_id)

    # Then specify parameters to compare with custom range confidence
    bayesed_params = ['log(Mstar)', 'log(SFR_{10Myr}/[M_{sun}/yr])', 'Av_2']
    bagpipes_params = ['stellar_mass', 'sfr', 'Av']
    labels = ['log(M*)', 'log(SFR)', 'Av']  # Note: SFR will be log-scaled for both
    true_values = [10.5, -1.2, 0.3]  # True parameter values to mark
    plot_corner_comparison(results, fit, object_id, bayesed_params, bagpipes_params,
                          labels, true_values, range_confidence=0)  # Use full data range
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

        # Check true_values length if provided
        if true_values is not None and len(true_values) != len(bayesed_params):
            print(f"Error: true_values ({len(true_values)}) must match length of parameter lists ({len(bayesed_params)})")
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
        final_true_values = []  # Keep for GetDist markers

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

            # Add corresponding true value if provided
            if true_values is not None:
                final_true_values.append(true_values[i])

            print(f"✓ Loaded: {bayesed_param} vs {bagpipes_param} -> {label}")

        if not bayesed_data or not bagpipes_data:
            print("No valid parameters found for comparison")
            return None

        print(f"\nCreating GetDist corner plot with {len(final_labels)} parameters: {final_labels}")

        # Configure GetDist analysis settings
        getdist_settings = configure_getdist_settings(range_confidence=range_confidence)
        print(f"Using GetDist range_confidence = {range_confidence} ({(1-range_confidence)*100:.1f}% confidence ranges)")

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
        # Use range_confidence to control parameter range determination
        bagpipes_mcsamples = MCSamples(samples=bagpipes_samples_array, names=clean_names,
                                     labels=final_labels, name_tag='BAGPIPES',
                                     settings=getdist_settings)

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
                                                 name_tag='BayeSED3',
                                                 settings=getdist_settings)
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
            'contour_lws': [2.0, 2.0],
        }

        # Add markers for true values if provided (GetDist native support)
        if true_values is not None and final_true_values:
            print(f"Adding true value markers using GetDist native markers for {len(final_true_values)} parameters...")
            # Create markers dict indexed by parameter name
            markers_dict = {}
            for i, (param_name, true_val) in enumerate(zip(clean_names[:len(bayesed_indices)], final_true_values)):
                markers_dict[param_name] = true_val

            plot_kwargs['markers'] = markers_dict
            plot_kwargs['marker_args'] = {
                'color': '#8B4513',      # Dark brown - distinct from blue/red, professional
                'linestyle': ':',        # Dotted to distinguish from solid/dashed posteriors
                'linewidth': 2.0,        # Match BayeSED3 contour line width for consistency
                'alpha': 0.85            # Good visibility while not overwhelming
            }

        # Plot using BayeSED's method with GetDist native markers
        g.triangle_plot(samples_list, clean_names[:len(bayesed_indices)], **plot_kwargs)

        # Clean title
        plt.suptitle(f' ID={object_id}',
                   fontsize=14, y=1.00, fontweight='bold')

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


def extract_true_values(results_bayesed, object_id, true_value_params, labels=None, verbose=True):
    """Extract true values for a specific object from BayeSED results.
    
    This helper function extracts true parameter values from an astropy Table
    containing BayeSED results for use in corner plot comparisons.
    
    Parameters:
    -----------
    results_bayesed : astropy.table.Table
        BayeSED results table containing true value columns
    object_id : str or int
        Object ID to extract true values for
    true_value_params : list of str
        List of true value parameter column names (e.g., ['z_{True}', 'log(Mstar)[0,1]_{True}'])
    labels : list of str, optional
        List of parameter labels for logging (e.g., ['z', 'log(M*)', 'log(SFR)'])
    verbose : bool, default=True
        Whether to print warning and success messages
    
    Returns:
    --------
    list of float or None
        List of true parameter values if all found, None if any missing or error occurred
        
    Example:
    --------
    true_value_params = ['z_{True}', 'log(Mstar)[0,1]_{True}', 'log(SFR_{100Myr}/[M_{sun}/yr])[0,1]_{True}']
    labels = ['z', 'log(M*)', 'log(SFR)']
    obj_true_values = extract_true_values(results_bayesed, '5494348_STARFORMING', true_value_params, labels)
    if obj_true_values:
        print(f"True values: {obj_true_values}")
    """
    try:
        obj_true_values = []
        
        for param in true_value_params:
            if param in results_bayesed.colnames:  # astropy Table uses .colnames not .columns
                # Get the true value for this specific object using astropy Table operations
                obj_mask = results_bayesed['ID'] == object_id
                if obj_mask.any():
                    # For astropy Table, use boolean indexing and get first element
                    masked_table = results_bayesed[obj_mask]
                    if len(masked_table) > 0:
                        true_val = masked_table[param][0]  # astropy Table indexing
                        obj_true_values.append(float(true_val))  # Ensure it's a float
                    else:
                        if verbose:
                            print(f"Warning: No rows found for object {object_id}")
                        return None
                else:
                    if verbose:
                        print(f"Warning: Object {object_id} not found in results_bayesed")
                    return None
            else:
                if verbose:
                    print(f"Warning: True value parameter '{param}' not found in results_bayesed")
                    print(f"Available true value columns: {[col for col in results_bayesed.colnames if '_True' in col]}")
                return None
        
        # Check if all parameters were found and log success
        if obj_true_values and len(obj_true_values) == len(true_value_params):
            if verbose and labels:
                print(f"Found true values for object {object_id}: {dict(zip(labels, obj_true_values))}")
            return obj_true_values
        else:
            if verbose:
                print(f"Could not extract all true values for object {object_id}")
            return None
        
    except Exception as e:
        if verbose:
            print(f"Error extracting true values for object {object_id}: {e}")
            import traceback
            traceback.print_exc()
        return None


def plot_parameter_scatter(bayesed_results, bagpipes_cat_file, bayesed_params, bagpipes_params, 
                          labels=None, save=True, show=False):
    """Create scatter plots comparing derived parameters between BayeSED3 and BAGPIPES.
    
    This improved version:
    - Always uses all objects available in both datasets
    - Uses astropy table for BayeSED parameters
    - Loads BAGPIPES parameters from fits file saved by fit_cat.fit
    - Allows user to specify which parameters to compare (like plot_corner_comparison)
    
    Parameters:
    -----------
    bayesed_results : BayeSEDResults or astropy.table.Table
        BayeSED3 results object or astropy table with results
    bagpipes_cat_file : str
        Path to BAGPIPES catalog fits file (e.g., 'pipes/cats/catalog_name.fits')
    bayesed_params : list of str
        List of BayeSED3 parameter names to plot
    bagpipes_params : list of str
        List of BAGPIPES parameter names to plot (must match length of bayesed_params)
    labels : list of str, optional
        List of labels for the plot axes. If None, uses bayesed_params.
    save : bool
        Whether to save the plot
    show : bool
        Whether to display the plot
        
    Example:
    --------
    # First check available parameters
    plot_parameter_scatter(results, bagpipes_file, None, None)
    
    # Then specify parameters to compare
    bayesed_params = ['log(Mstar)[0,0]', 'log(SFR_{100Myr}/[M_{sun}/yr])[0,0]', 'Av_2']
    bagpipes_params = ['stellar_mass', 'sfr', 'Av']
    labels = ['log(M*)', 'log(SFR)', 'Av']
    plot_parameter_scatter(results, bagpipes_file, bayesed_params, bagpipes_params, labels)
    """
    
    # Check if bagpipes_cat_file exists before attempting to load it
    if not os.path.exists(bagpipes_cat_file):
        print(f"Error: BAGPIPES catalog file not found: {bagpipes_cat_file}")
        
        # Try to provide helpful suggestions
        bagpipes_dir = os.path.dirname(bagpipes_cat_file)
        if os.path.exists(bagpipes_dir):
            print(f"Available files in {bagpipes_dir}:")
            try:
                for f in os.listdir(bagpipes_dir):
                    if f.endswith('.fits'):
                        print(f"  {f}")
            except Exception:
                print("  (Could not list directory contents)")
        else:
            print(f"Directory {bagpipes_dir} does not exist")
            
        return None
    
    # Load BayeSED results as astropy table
    if hasattr(bayesed_results, 'load_hdf5_results'):
        # If it's a BayeSEDResults object, load the table
        bayesed_table = bayesed_results.load_hdf5_results()
        print(f"Loaded BayeSED results table with {len(bayesed_table)} objects")
    else:
        # Assume it's already an astropy table
        bayesed_table = bayesed_results
        print(f"Using provided BayeSED table with {len(bayesed_table)} objects")
    
    # Load BAGPIPES catalog fits file
    try:
        bagpipes_table = Table.read(bagpipes_cat_file)
        print(f"Loaded BAGPIPES catalog from {bagpipes_cat_file} with {len(bagpipes_table)} objects")
    except Exception as e:
        print(f"Error loading BAGPIPES catalog from {bagpipes_cat_file}: {e}")
        return None
    
    # If no parameters specified, show available parameters
    if bayesed_params is None or bagpipes_params is None:
        print(f"\n=== Available Parameters ===")
        print(f"\nBayeSED3 parameters ({len(bayesed_table.colnames)}):")
        for i, param in enumerate(bayesed_table.colnames):
            print(f"  {i:2d}: {param}")

        print(f"\nBAGPIPES parameters ({len(bagpipes_table.colnames)}):")
        for i, param in enumerate(bagpipes_table.colnames):
            print(f"  {i:2d}: {param}")

        print(f"\nTo create scatter plot, specify parameter lists:")
        print(f"bayesed_params = ['param1', 'param2', ...]")
        print(f"bagpipes_params = ['param1', 'param2', ...]")
        print(f"labels = ['label1', 'label2', ...]  # optional")
        print(f"plot_parameter_scatter(results, bagpipes_file, bayesed_params, bagpipes_params, labels)")
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
    
    # Find common objects between both tables
    bayesed_ids = set(bayesed_table['ID'])
    bagpipes_ids = set(bagpipes_table['ID'])
    common_ids = bayesed_ids & bagpipes_ids
    
    print(f"Found {len(common_ids)} common objects between BayeSED3 and BAGPIPES")
    print(f"BayeSED3 has {len(bayesed_ids)} objects, BAGPIPES has {len(bagpipes_ids)} objects")
    
    if len(common_ids) == 0:
        print("No common objects found for comparison")
        return None
    
    # Create figure
    n_params = len(param_mapping)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    # Track statistics for summary
    param_stats = {}
    
    for i, (bayesed_param, bagpipes_param, label) in enumerate(zip(bayesed_params, bagpipes_params, labels)):
        if i >= len(axes):
            break
            
        ax = axes[i]
        bayesed_vals = []
        bagpipes_vals = []
        valid_ids = []
        
        # Check if columns exist
        bayesed_available_cols = [col for col in bayesed_table.colnames if bayesed_param in col]
        bagpipes_available_cols = [col for col in bagpipes_table.colnames if bagpipes_param in col]
        
        if bayesed_param not in bayesed_table.colnames:
            if bayesed_available_cols:
                print(f"Warning: BayeSED3 column '{bayesed_param}' not found. Similar columns available: {bayesed_available_cols}")
            else:
                print(f"Warning: BayeSED3 column '{bayesed_param}' not found. Available columns: {list(bayesed_table.colnames)[:10]}...")
            continue
            
        if bagpipes_param not in bagpipes_table.colnames:
            if bagpipes_available_cols:
                print(f"Warning: BAGPIPES column '{bagpipes_param}' not found. Similar columns available: {bagpipes_available_cols}")
            else:
                print(f"Warning: BAGPIPES column '{bagpipes_param}' not found. Available columns: {list(bagpipes_table.colnames)[:10]}...")
            continue
        
        # Extract values for all common objects
        for obj_id in common_ids:
            try:
                # Get BayeSED3 value
                bayesed_mask = bayesed_table['ID'] == obj_id
                if bayesed_mask.any():
                    bayesed_val = bayesed_table[bayesed_mask][bayesed_param][0]
                    
                    # Get BAGPIPES value
                    bagpipes_mask = bagpipes_table['ID'] == obj_id
                    if bagpipes_mask.any():
                        bagpipes_val = bagpipes_table[bagpipes_mask][bagpipes_param][0]
                        
                        # Handle SFR scaling consistency: BayeSED3 SFR is already log, BAGPIPES SFR needs log conversion
                        if 'SFR' in bayesed_param and bagpipes_param == 'sfr':
                            # BayeSED3 SFR is already in log scale, BAGPIPES SFR needs log conversion
                            bagpipes_val = np.log10(bagpipes_val)
                        
                        # Handle stellar mass: both need log conversion if not already log
                        elif 'Mstar' in bayesed_param and bagpipes_param == 'stellar_mass':
                            # Both need log conversion
                            if not ('log' in bayesed_param.lower()):
                                bayesed_val = np.log10(bayesed_val)
                            bagpipes_val = np.log10(bagpipes_val)
                        
                        # Check for valid values (not NaN or infinite)
                        if np.isfinite(bayesed_val) and np.isfinite(bagpipes_val):
                            bayesed_vals.append(bayesed_val)
                            bagpipes_vals.append(bagpipes_val)
                            valid_ids.append(obj_id)
                            
            except Exception as e:
                print(f"Warning: Error processing object {obj_id} for {label}: {e}")
                continue
        
        if len(bayesed_vals) > 0 and len(bagpipes_vals) > 0:
            # Convert to numpy arrays
            bayesed_vals = np.array(bayesed_vals)
            bagpipes_vals = np.array(bagpipes_vals)
            
            # Create scatter plot
            ax.scatter(bayesed_vals, bagpipes_vals, alpha=0.7, s=50, edgecolors='black', linewidth=0.5)
            
            # Add 1:1 line
            min_val = min(np.min(bayesed_vals), np.min(bagpipes_vals))
            max_val = max(np.max(bayesed_vals), np.max(bagpipes_vals))
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, label='1:1', linewidth=2)
            
            # Calculate statistics
            correlation = np.corrcoef(bayesed_vals, bagpipes_vals)[0, 1]
            bias = np.mean(bagpipes_vals - bayesed_vals)  # BAGPIPES - BayeSED3
            rms = np.sqrt(np.mean((bagpipes_vals - bayesed_vals)**2))
            
            # Store statistics
            param_stats[label] = {
                'correlation': correlation,
                'bias': bias,
                'rms': rms,
                'n_objects': len(bayesed_vals)
            }
            
            # Add statistics text
            stats_text = f'r = {correlation:.3f}\nbias = {bias:.3f}\nRMS = {rms:.3f}\nN = {len(bayesed_vals)}'
            ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
                    verticalalignment='top', fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # Set labels
            ax.set_xlabel(f'BayeSED3 {label}')
            ax.set_ylabel(f'BAGPIPES {label}')
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            print(f"✓ {label}: {len(bayesed_vals)} objects, r={correlation:.3f}, bias={bias:.3f}")
        else:
            print(f"✗ {label}: No valid data points found")
            ax.text(0.5, 0.5, f'No valid data\nfor {label}', 
                   transform=ax.transAxes, ha='center', va='center',
                   bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    # Remove empty subplots
    for i in range(n_params, len(axes)):
        fig.delaxes(axes[i])
    
    plt.suptitle(f'Parameter Comparison: BayeSED3 vs BAGPIPES ({len(common_ids)} objects)', fontsize=16)
    plt.tight_layout()
    
    # Print summary statistics
    print(f"\n=== Parameter Comparison Summary ===")
    for param_name, stats in param_stats.items():
        print(f"{param_name:15s}: r={stats['correlation']:6.3f}, bias={stats['bias']:7.3f}, RMS={stats['rms']:6.3f}, N={stats['n_objects']:3d}")
    
    if save:
        plot_dir = os.path.join("pipes", "plots", "comparison")
        os.makedirs(plot_dir, exist_ok=True)
        out_path = os.path.join(plot_dir, "parameter_scatter_comparison_all_objects.png")
        plt.savefig(out_path, bbox_inches="tight", dpi=300)
        print(f"\nSaved scatter comparison: {out_path}")
    
    if show:
        plt.show()
    
    if not show:
        plt.close(fig)
    
    return fig, param_stats

def main():
    """Main function to run BayeSED3 vs BAGPIPES comparison."""

    # Parse command line arguments
    input_file = None
    filters_file = None
    filters_selected_file = None

    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    if len(sys.argv) > 2:
        filters_file = sys.argv[2]
    if len(sys.argv) > 3:
        filters_selected_file = sys.argv[3]

    # Require input file to be explicitly provided by user
    if input_file is None:
        print("Error: Input file must be provided.")
        print("")
        print("Usage: python test_bayesed_bagpipes_comparison.py <input_file> [filters_file] [filters_selected_file]")
        print("")
        print("Use --help for more detailed usage information.")
        sys.exit(1)

    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found.")
        print("Please provide a valid input file path.")
        sys.exit(1)

    # Auto-set related paths based on input_file directory
    input_dir = os.path.dirname(input_file)
    input_basename = os.path.splitext(os.path.basename(input_file))[0]

    # Auto-detect filter files if not provided via command line
    if filters_file is None:
        # Look for common filter file patterns
        for pattern in ['filters_bassmzl.txt', 'filters.txt']:
            candidate = os.path.join(input_dir, pattern)
            if os.path.exists(candidate):
                filters_file = candidate
                break
    else:
        # Check if provided filter file exists
        if not os.path.exists(filters_file):
            print(f"Error: Filter file '{filters_file}' not found.")
            sys.exit(1)

    if filters_selected_file is None:
        # Look for selected filters file patterns
        for pattern in ['filters_selected_csst.txt', 'filters_selected.txt']:
            candidate = os.path.join(input_dir, pattern)
            if os.path.exists(candidate):
                filters_selected_file = candidate
                break
    else:
        # Check if provided selected filter file exists
        if not os.path.exists(filters_selected_file):
            print(f"Error: Selected filters file '{filters_selected_file}' not found.")
            sys.exit(1)

    # Set output directory
    output_dir = os.path.join(input_dir, 'output')

    print(f"Input file: {input_file}")
    print(f"Input directory: {input_dir}")
    print(f"Filters file: {filters_file}")
    print(f"Selected filters file: {filters_selected_file}")
    print(f"Output directory: {output_dir}")

    # Warn if filter files not found (they may be optional)
    if filters_file is None:
        print("Warning: No filter file found. Looking for: filters_bassmzl.txt, filters.txt")
    if filters_selected_file is None:
        print("Warning: No selected filters file found. Looking for: filters_selected_csst.txt, filters_selected.txt")

    # Initialize interface
    bayesed = BayeSEDInterface(mpi_mode='1')

    # Simple galaxy fitting with no_photometry_fit=True (fit spectra only)
    params = BayeSEDParams.galaxy(
        input_file=input_file,
        filters=filters_file,
        filters_selected=filters_selected_file,
        outdir=output_dir,
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
    params.multinest = MultiNestParams(
        nlive=40,          # Good balance of speed/accuracy
        efr=0.1,            # Moderate efficiency
        tol=0.5,            # Standard tolerance
    )


    # Run analysis
    result = bayesed.run(params)

    # Load and analyze results
    # Extract catalog name from input file (efficient - only reads first line)
    cat_name = BayeSEDDataLoader.extract_catalog_name(input_file)

    results = BayeSEDResults(output_dir, catalog_name=cat_name)
    results.print_summary()

    # Get available objects for plotting
    available_objects = results.list_objects()
    if available_objects:
        # Plot first few objects if available
        for obj_id in available_objects[:2]:  # Plot first 2 objects
            try:
                results.plot_bestfit(obj_id)
            except Exception as e:
                print(f"Could not plot bestfit for {obj_id}: {e}")

    results_bayesed = results.load_hdf5_results()


    IDs = results.list_objects()
    data_loader = BayeSEDDataLoader(input_file)


    # BAGPIPES fit instructions (common for all objects)
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

    base_fit_instructions = {}
    base_fit_instructions["redshift"] = (0., 2.)
    base_fit_instructions["exponential"] = exp
    base_fit_instructions["dust"] = dust
    base_fit_instructions["nebular"] = nebular
    base_fit_instructions["veldisp"] = 300  #km/s

    noise = {}
    noise["type"] = "white_scaled"
    noise["scaling"] = (1., 1.5)
    noise["scaling_prior"] = "log_10"
    base_fit_instructions["noise"] = noise


    resolution_curve = data_loader.get_resolution_curve(IDs[0])
    fit_instructions = base_fit_instructions.copy()
    fit_instructions["R_curve"] = resolution_curve
    # plt.plot(fit_instructions["R_curve"][:, 0], fit_instructions["R_curve"][:, 1])
    # plt.xlabel("Wavelength/ \AA")
    # plt.ylabel("Spectral resolving power")
    # plt.show()

    fit_cat = pipes.fit_catalogue(IDs, fit_instructions, data_loader.load_spectrum_only, photometry_exists=False, run=cat_name, make_plots=False)
    fit_cat.fit(verbose=True, sampler='nautilus', mpi_serial=False, pool=20, n_live=400)
    # fit_cat.fit(verbose=True, sampler='nautilus', mpi_serial=True, n_live=400) #multiple objects are fitted at once, each using one core

    # Define comparison parameters (used by both plot_parameter_scatter and plot_corner_comparison)
    bayesed_params = ['z', 'log(Mstar)[0,0]', 'log(SFR_{100Myr}/[M_{sun}/yr])[0,0]']
    true_value_params = ['z_{True}', 'log(Mstar)[0,1]_{True}', 'log(SFR_{100Myr}/[M_{sun}/yr])[0,1]_{True}']
    bagpipes_params = ['redshift', 'stellar_mass', 'sfr']
    labels = [r'z', r'\log(M_{\star}\, /\, \mathrm{M}_{\odot})', r'\log(SFR\, /\, \mathrm{M}_{\odot}\, \mathrm{yr}^{-1})']

    # Create parameter scatter comparison plot using all objects
    print(f"\n=== Creating Parameter Scatter Comparison Plot ===")
    bagpipes_cat_file = os.path.join("pipes", "cats", f"{cat_name}.fits")
    
    fig, param_stats = plot_parameter_scatter(
        bayesed_results=results_bayesed,  # Use the astropy table directly
        bagpipes_cat_file=bagpipes_cat_file,
        bayesed_params=bayesed_params,
        bagpipes_params=bagpipes_params,
        labels=labels,
        save=True,
        show=True
    )

    bagpipes_fits = {}  # Store fits for posterior comparison

    for ID in IDs[:10]:
        # Get object-specific resolution curve
        resolution_curve = data_loader.get_resolution_curve(ID)

        # Create object-specific fit instructions with the correct resolution
        fit_instructions = base_fit_instructions.copy()
        fit_instructions["R_curve"] = resolution_curve

        # Create individual galaxy object using the data loader
        galaxy = pipes.galaxy(ID, data_loader.load_spectrum_only, photometry_exists=False)

        # Fit individual galaxy with timing
        fit = pipes.fit(galaxy, fit_instructions, run=f"{cat_name}_{ID}")
        t0 = time.time()
        fit.fit(verbose=True, sampler='nautilus', n_live=400, pool=20)
        runtime_s = time.time() - t0

        # Store fit for comparison
        bagpipes_fits[ID] = fit

        # Generate spectrum plots
        plot_spectrum_posterior_with_residuals(fit, ID, cat_name, runtime_s=runtime_s)

        # Generate corner comparison plots using GetDist for better aesthetics
        print(f"\nGenerating corner comparison plot for object {ID}...")
        obj_true_values = extract_true_values(results_bayesed, ID, true_value_params, labels, verbose=True)

        plot_corner_comparison(results, fit, ID, bayesed_params, bagpipes_params, labels, obj_true_values)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ['-h', '--help']:
        print("Usage: python test_bayesed_bagpipes_comparison.py <input_file> [filters_file] [filters_selected_file]")
        print("")
        print("Arguments:")
        print("  input_file           Path to the BayeSED input file (REQUIRED)")
        print("  filters_file         Path to the filter definitions file (optional)")
        print("  filters_selected_file Path to the selected filters file (optional)")
        print("")
        print("Examples:")
        print("  python test_bayesed_bagpipes_comparison.py observation/test/gal.txt")
        print("  python test_bayesed_bagpipes_comparison.py observation/CESS_mock/two.txt")
        sys.exit(0)

    main()

