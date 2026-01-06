from bayesed import BayeSEDInterface, BayeSEDParams, BayeSEDResults, ZParams, RDFParams, SysErrParams, MultiNestParams, SNRmin2Params
from astropy.table import Table, join, hstack, vstack
import bagpipes as pipes
import numpy as np
import os
import sys
import argparse
import matplotlib.pyplot as plt
import time
from scipy import stats
import corner

# c = 2.9979246e+14  # um/s
c = 2.9979246e+18 #angstrom/s

class BayeSEDDataLoader:
    """Data loader for input files of BayeSED format.

    Note: the header and data rows must have same number of columns
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


def configure_getdist_settings(range_confidence=0, min_weight_ratio=0, contours=None, smooth_scale_1D=-1, smooth_scale_2D=-1):
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
        'min_weight_ratio': min_weight_ratio,
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


def plot_posterior_corner_comparison(bayesed_results, bagpipes_fit, object_id,
                                bayesed_params=None, bagpipes_params=None,
                                labels=None, true_values=None, save=True, show=True,
                                range_confidence=0,min_weight_ratio=0, contours=None, smooth_scale_1D=-1, smooth_scale_2D=-1):
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
    plot_posterior_corner_comparison(results, fit, object_id)

    # Then specify parameters to compare with custom range confidence
    bayesed_params = ['log(Mstar)', 'log(SFR_{10Myr}/[M_{sun}/yr])', 'Av_2']
    bagpipes_params = ['stellar_mass', 'sfr', 'Av']
    labels = ['log(M*)', 'log(SFR)', 'Av']  # Note: SFR will be log-scaled for both
    true_values = [10.5, -1.2, 0.3]  # True parameter values to mark
    plot_posterior_corner_comparison(results, fit, object_id, bayesed_params, bagpipes_params,
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
            print(f"plot_posterior_corner_comparison(results, fit, '{object_id}', bayesed_params, bagpipes_params, labels)")
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
            if 'SFR' in bayesed_param and 'sfr' in bagpipes_param:
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
        analysis_settings = configure_getdist_settings(range_confidence=range_confidence,min_weight_ratio=min_weight_ratio,contours=contours)
        print(f"Using GetDist range_confidence = {range_confidence} ({(1-range_confidence)*100:.1f}% confidence ranges)")

        # Get BayeSED3 GetDist samples directly (already a MCSamples object)
        bayesed_getdist_samples = bayesed_results.get_getdist_samples(object_id=object_id,settings=analysis_settings)

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
                                     settings=analysis_settings)

        # Set labels for BayeSED3 samples (following BayeSED's approach)
        bayesed_getdist_samples.name_tag = 'BayeSED3'
        bayesed_getdist_samples.label = 'BayeSED3'

        # Filter BayeSED3 samples to only include the parameters we want to compare
        # Extract the relevant parameter indices from BayeSED3 samples
        bayesed_param_names = [p.name for p in bayesed_getdist_samples.paramNames.names]
        bayesed_indices = []
        bayesed_filtered_labels = []

        for i, (bayesed_param, label) in enumerate(zip(bayesed_params, final_labels)):
            # if bayesed_param in bayesed_param_names:
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
                                                 settings=analysis_settings)
        else:
            print("No matching parameters found in BayeSED3 samples")
            return None

        # Create GetDist triangle plot following BayeSED's approach
        g = plots.get_subplot_plotter(width_inch=16, subplot_size=3.0,subplot_size_ratio=0.8,analysis_settings=analysis_settings)

        # Use BayeSED's plotting style for better comparison visibility
        g.settings.figure_legend_frame = True
        g.settings.figure_legend_loc = 'upper right'
        g.settings.legend_fontsize = 18
        g.settings.axes_fontsize = 18
        g.settings.lab_fontsize = 18
        g.settings.tight_layout = True
        g.settings.axes_labelsize = 20

        # Use BayeSED's plotting approach with samples list
        samples_list = [bayesed_filtered_mcsamples, bagpipes_mcsamples]

        # Set plotting options for better comparison visibility (following BayeSED's approach)
        plot_kwargs = {
            'filled': True,
            'contour_colors': ['#F24236', '#2E86AB'],  # BayeSED3 red, BAGPIPES blue
            'contour_ls': ['-', '-'],  # Solid for BayeSED3, dashed for BAGPIPES
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
                'color': 'green',      # Dark brown - distinct from blue/red, professional
                'ls': '--',        # Dotted to distinguish from solid/dashed posteriors
                'lw': 2.0,        # Match BayeSED3 contour line width for consistency
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
            plt.savefig(out_path, bbox_inches="tight", dpi=400, facecolor='white')
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
                          labels=None, colorbar_param=None, colorbar_label=None,
                          x_axis_param=None, x_axis_label=None, show_unity_line=True,
                          x_axis_stats='x_axis', save=True, show=False, catalog_name=None):
    """Create scatter plots comparing derived parameters between BayeSED3 and BAGPIPES.

    This improved version:
    - Always uses all objects available in both datasets
    - Uses astropy table for BayeSED parameters
    - Loads BAGPIPES parameters from fits file saved by fit_cat.fit
    - Allows user to specify which parameters to compare (like plot_posterior_corner_comparison)
    - Dynamically adapts subplot layout based on number of parameters
    - Handles extreme outliers robustly using percentile-based axis ranges
    - Uses NMAD-based robust statistics (correlation, bias, RMS) excluding 3σ outliers
    - NMAD (Normalized Median Absolute Deviation) provides outlier-resistant statistics
    - Properly renders LaTeX labels for mathematical expressions
    - Supports colorbar based on any column in the BayeSED table

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
        Can include LaTeX formatting (e.g., r'$\\log(M_*/M_\\odot)$')
    colorbar_param : str, optional
        Name of column in bayesed_table to use for colorbar. If provided, scatter points
        will be colored based on this parameter's values.
    colorbar_label : str, optional
        Label for the colorbar. If None and colorbar_param is provided, uses colorbar_param.
        Can include LaTeX formatting (e.g., r'$\\log(\\mathrm{SFR})$')
    x_axis_param : list of str, optional
        List of column names in bayesed_table to use as x-axis for each subplot. If provided,
        creates parameter vs x_axis_param plots instead of parameter comparison plots.
        Must match length of bayesed_params. Useful for plotting parameters vs redshift, stellar mass, etc.
    x_axis_label : list of str, optional
        List of labels for the x-axis when using x_axis_param. If None, uses x_axis_param.
        Must match length of bayesed_params. Can include LaTeX formatting (e.g., r'$z$', r'$\\log(M_*/M_\\odot)$')
    show_unity_line : bool, default=True
        Whether to show unity (1:1) line. In comparison mode, shows diagonal 1:1 line (y=x).
        In x-axis mode, shows diagonal 1:1 line for comparison with true values.
    x_axis_stats : str, default='both'
        Statistics to show in x-axis mode. Options:
        - 'both': Show both BayeSED vs x-axis and BAGPIPES vs x-axis statistics
        - 'codes': Show only BayeSED vs BAGPIPES statistics (same as comparison mode)
        - 'x_axis': Show only individual code vs x-axis statistics
        Ignored in comparison mode.
    save : bool
        Whether to save the plot
    show : bool
        Whether to display the plot
    catalog_name : str, optional
        Name to include in the saved PNG filename. If provided, saves as
        'parameter_scatter_comparison_{catalog_name}.png', otherwise uses
        'parameter_scatter_comparison_all_objects.png'

    Example:
    --------
    # First check available parameters
    plot_parameter_scatter(results, bagpipes_file, None, None)

    # Then specify parameters to compare
    bayesed_params = ['log(Mstar)[0,0]', 'log(SFR_{100Myr}/[M_{sun}/yr])[0,0]', 'Av_2']
    bagpipes_params = ['stellar_mass', 'sfr', 'Av']
    labels = [r'$\\log(M_*/M_\\odot)$', r'$\\log(\\mathrm{SFR}/M_\\odot\\,\\mathrm{yr}^{-1})$', r'$A_V$']
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
        print(f"colorbar_param = 'column_name'  # optional, for coloring points")
        print(f"plot_parameter_scatter(results, bagpipes_file, bayesed_params, bagpipes_params, labels, colorbar_param)")
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

    # Validate x_axis_param and x_axis_label if provided
    use_x_axis = x_axis_param is not None
    if use_x_axis:
        if not isinstance(x_axis_param, list):
            print(f"Error: x_axis_param must be a list, got {type(x_axis_param)}")
            return None
        if len(x_axis_param) != len(bayesed_params):
            print(f"Error: x_axis_param ({len(x_axis_param)}) must match length of bayesed_params ({len(bayesed_params)})")
            return None

        # Set default x_axis_label if not provided
        if x_axis_label is None:
            x_axis_label = x_axis_param
        elif len(x_axis_label) != len(bayesed_params):
            print(f"Error: x_axis_label ({len(x_axis_label)}) must match length of bayesed_params ({len(bayesed_params)})")
            return None

        # Validate that all x_axis parameters exist in bayesed_table
        missing_x_params = [param for param in x_axis_param if param not in bayesed_table.colnames]
        if missing_x_params:
            print(f"Error: x_axis parameters not found in BayeSED table: {missing_x_params}")
            print(f"Available columns: {list(bayesed_table.colnames)[:10]}...")
            return None

        print(f"Using x-axis parameters: {x_axis_param}")
        print(f"Plot mode: Parameter vs x-axis plots")
    else:
        print(f"Plot mode: Parameter comparison plots (BayeSED vs BAGPIPES)")

    # Check if colorbar parameter exists
    colorbar_vals = None
    if colorbar_param is not None:
        if colorbar_param not in bayesed_table.colnames:
            print(f"Warning: Colorbar parameter '{colorbar_param}' not found in BayeSED table")
            print(f"Available columns: {list(bayesed_table.colnames)[:10]}...")
            colorbar_param = None
        else:
            print(f"Using colorbar parameter: {colorbar_param}")
            if colorbar_label is None:
                colorbar_label = colorbar_param

    # Find common objects between both tables
    bayesed_ids = set(bayesed_table['ID'])
    bagpipes_ids = set(bagpipes_table['#ID'])
    common_ids = bayesed_ids & bagpipes_ids

    print(f"Found {len(common_ids)} common objects between BayeSED3 and BAGPIPES")
    print(f"BayeSED3 has {len(bayesed_ids)} objects, BAGPIPES has {len(bagpipes_ids)} objects")

    if len(common_ids) == 0:
        print("No common objects found for comparison")
        return None

    # Create figure with dynamic layout based on number of parameters
    n_params = len(bayesed_params)

    # Calculate optimal subplot layout
    if n_params == 1:
        nrows, ncols = 1, 1
        figsize = (6, 5)
    elif n_params == 2:
        nrows, ncols = 1, 2
        figsize = (12, 5)
    elif n_params == 3:
        nrows, ncols = 1, 3
        figsize = (18, 5)
    elif n_params == 4:
        nrows, ncols = 2, 2
        figsize = (12, 10)
    elif n_params <= 6:
        nrows, ncols = 2, 3
        figsize = (15, 10)
    elif n_params <= 9:
        nrows, ncols = 3, 3
        figsize = (15, 15)
    elif n_params <= 12:
        nrows, ncols = 3, 4
        figsize = (20, 15)
    else:
        # For more than 12 parameters, use a 4-column layout
        nrows = (n_params + 3) // 4  # Ceiling division
        ncols = 4
        figsize = (20, 5 * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)

    # Handle single subplot case
    if n_params == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    # Track statistics for summary
    param_stats = {}

    # Prepare colorbar data if requested
    if colorbar_param is not None:
        colorbar_vals = []
        colorbar_ids = []

    for i, (bayesed_param, bagpipes_param, label) in enumerate(zip(bayesed_params, bagpipes_params, labels)):
        ax = axes[i]
        bayesed_vals = []
        bagpipes_vals = []
        x_axis_vals = []  # For x-axis parameter values
        valid_ids = []
        valid_colorbar_vals = []

        # Get x-axis parameter for this subplot if using x-axis mode
        current_x_param = x_axis_param[i] if use_x_axis else None
        current_x_label = x_axis_label[i] if use_x_axis else None

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
                    bagpipes_mask = bagpipes_table['#ID'] == obj_id
                    if bagpipes_mask.any():
                        bagpipes_val = bagpipes_table[bagpipes_mask][bagpipes_param][0]

                        # Handle SFR scaling consistency: BayeSED3 SFR is already log, BAGPIPES SFR needs log conversion
                        if 'SFR' in bayesed_param and 'sfr' in bagpipes_param:
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
                            # Get x-axis value if using x-axis mode
                            if use_x_axis:
                                x_val = bayesed_table[bayesed_mask][current_x_param][0]
                                if not np.isfinite(x_val):
                                    continue  # Skip if x-axis value is invalid
                                x_axis_vals.append(x_val)

                            bayesed_vals.append(bayesed_val)
                            bagpipes_vals.append(bagpipes_val)
                            valid_ids.append(obj_id)

                            # Add colorbar value if requested
                            if colorbar_param is not None:
                                colorbar_val = bayesed_table[bayesed_mask][colorbar_param][0]
                                if np.isfinite(colorbar_val):
                                    valid_colorbar_vals.append(colorbar_val)
                                else:
                                    # If colorbar value is invalid, remove this point
                                    bayesed_vals.pop()
                                    bagpipes_vals.pop()
                                    valid_ids.pop()
                                    if use_x_axis:
                                        x_axis_vals.pop()

            except Exception as e:
                print(f"Warning: Error processing object {obj_id} for {label}: {e}")
                continue

        if len(bayesed_vals) > 0 and len(bagpipes_vals) > 0:
            # Convert to numpy arrays
            bayesed_vals = np.array(bayesed_vals)
            bagpipes_vals = np.array(bagpipes_vals)

            # Convert x-axis and colorbar values to numpy arrays if available
            if use_x_axis:
                x_axis_vals = np.array(x_axis_vals)
            if colorbar_param is not None and valid_colorbar_vals:
                valid_colorbar_vals = np.array(valid_colorbar_vals)

            # Determine plot mode and set up data
            if use_x_axis:
                # X-axis mode: plot parameters vs x-axis parameter
                x_data_bayesed = x_axis_vals
                y_data_bayesed = bayesed_vals
                x_data_bagpipes = x_axis_vals  # Same x-axis for both
                y_data_bagpipes = bagpipes_vals

                # Calculate robust axis ranges
                x_vals_combined = x_axis_vals
                y_vals_combined = np.concatenate([bayesed_vals, bagpipes_vals])
            else:
                # Comparison mode: plot BayeSED vs BAGPIPES
                x_data_bayesed = bayesed_vals
                y_data_bayesed = bagpipes_vals
                x_data_bagpipes = None  # Not used in comparison mode
                y_data_bagpipes = None

                # Calculate robust axis ranges
                x_vals_combined = bayesed_vals
                y_vals_combined = bagpipes_vals

            # Create scatter plot with or without colorbar
            if colorbar_param is not None and len(valid_colorbar_vals) > 0:
                # Calculate robust colorbar range using percentiles to exclude extreme outliers
                # Use tighter percentile range for better color contrast
                colorbar_p_low, colorbar_p_high = 0.1, 99.0  # 0.1th to 99.0th percentile
                colorbar_vmin = np.percentile(valid_colorbar_vals, colorbar_p_low)
                colorbar_vmax = np.percentile(valid_colorbar_vals, colorbar_p_high)

                # Ensure we have a valid range (avoid vmin == vmax)
                if colorbar_vmax <= colorbar_vmin:
                    colorbar_vmin = np.min(valid_colorbar_vals)
                    colorbar_vmax = np.max(valid_colorbar_vals)
                    if colorbar_vmax <= colorbar_vmin:
                        # All values are the same, add small range
                        colorbar_vmax = colorbar_vmin + 1e-6

                # Create scatter plots based on mode
                if use_x_axis:
                    # X-axis mode: plot both BayeSED and BAGPIPES vs x-axis parameter
                    scatter_bagpipes = ax.scatter(x_data_bagpipes, y_data_bagpipes, c=valid_colorbar_vals,
                                                alpha=0.7, s=50, edgecolors='blue', linewidth=1.0,
                                                cmap='viridis', vmin=colorbar_vmin, vmax=colorbar_vmax,
                                                label='BAGPIPES', marker='s')
                    scatter_bayesed = ax.scatter(x_data_bayesed, y_data_bayesed, c=valid_colorbar_vals,
                                               alpha=0.7, s=50, edgecolors='red', linewidth=1.0,
                                               cmap='viridis', vmin=colorbar_vmin, vmax=colorbar_vmax,
                                               label='BayeSED3', marker='o')
                    # Use the first scatter for colorbar
                    scatter = scatter_bayesed
                else:
                    # Comparison mode: BayeSED vs BAGPIPES
                    scatter = ax.scatter(x_data_bayesed, y_data_bayesed, c=valid_colorbar_vals,
                                       alpha=0.7, s=50, edgecolors='black', linewidth=0.5,
                                       cmap='viridis', vmin=colorbar_vmin, vmax=colorbar_vmax)

                # Add colorbar to the subplot
                cbar = plt.colorbar(scatter, ax=ax)
                cbar.set_label(colorbar_label, fontsize=12)

                # Count and report colorbar outliers
                n_colorbar_outliers = np.sum((valid_colorbar_vals < colorbar_vmin) |
                                           (valid_colorbar_vals > colorbar_vmax))
                if n_colorbar_outliers > 0:
                    outlier_fraction = n_colorbar_outliers / len(valid_colorbar_vals)
                    print(f"  Colorbar outliers for {label}: {n_colorbar_outliers} ({100*outlier_fraction:.1f}%) outside [{colorbar_vmin:.3f}, {colorbar_vmax:.3f}]")
            else:
                # Create scatter plots without colorbar
                if use_x_axis:
                    # X-axis mode: plot both BayeSED and BAGPIPES vs x-axis parameter
                    ax.scatter(x_data_bagpipes, y_data_bagpipes, alpha=0.7, s=50,
                              edgecolors='blue', facecolors='blue', linewidth=1.0,
                              label='BAGPIPES', marker='s')
                    ax.scatter(x_data_bayesed, y_data_bayesed, alpha=0.7, s=50,
                              edgecolors='red', facecolors='red', linewidth=1.0,
                              label='BayeSED3', marker='o')
                else:
                    # Comparison mode: BayeSED vs BAGPIPES
                    ax.scatter(x_data_bayesed, y_data_bayesed, alpha=0.7, s=50,
                              edgecolors='black', linewidth=0.5)

            # Calculate robust axis ranges excluding extreme outliers
            # Use percentiles to exclude outliers (e.g., 2.5% and 97.5% for ~2-sigma range)
            if use_x_axis:
                # X-axis mode: separate ranges for x and y axes
                x_vals_for_range = x_vals_combined
                y_vals_for_range = y_vals_combined

                # For SFR parameters, use tighter percentile range due to common extreme outliers
                if 'SFR' in label or 'sfr' in label.lower():
                    p_low, p_high = 5, 95  # 5th to 95th percentile for SFR
                else:
                    p_low, p_high = 2.5, 97.5  # 2.5th to 97.5th percentile for other parameters

                # X-axis range
                x_robust_min = np.percentile(x_vals_for_range, p_low)
                x_robust_max = np.percentile(x_vals_for_range, p_high)
                x_range_padding = (x_robust_max - x_robust_min) * 0.05
                x_plot_min = x_robust_min - x_range_padding
                x_plot_max = x_robust_max + x_range_padding

                # Y-axis range
                y_robust_min = np.percentile(y_vals_for_range, p_low)
                y_robust_max = np.percentile(y_vals_for_range, p_high)
                y_range_padding = (y_robust_max - y_robust_min) * 0.05
                y_plot_min = y_robust_min - y_range_padding
                y_plot_max = y_robust_max + y_range_padding

                # Set axis limits
                ax.set_xlim(x_plot_min, x_plot_max)
                ax.set_ylim(y_plot_min, y_plot_max)

                # Add unity line in x-axis mode if requested
                if show_unity_line:
                    # Show diagonal 1:1 line (y=x) for comparison with true values
                    # This is useful when x-axis represents true values
                    unity_min = max(x_plot_min, y_plot_min)
                    unity_max = min(x_plot_max, y_plot_max)
                    if unity_min < unity_max:  # Only show if there's overlap
                        ax.plot([unity_min, unity_max], [unity_min, unity_max], 'k--',
                               alpha=0.7, linewidth=2, label='', zorder=1)

                # Set labels for x-axis mode
                ax.set_xlabel(current_x_label)
                ax.set_ylabel(f'{label}')
                ax.legend()

            else:
                # Comparison mode: square plot with 1:1 line
                all_vals = np.concatenate([x_vals_combined, y_vals_combined])

                # For SFR parameters, use tighter percentile range due to common extreme outliers
                if 'SFR' in label or 'sfr' in label.lower():
                    p_low, p_high = 5, 95  # 5th to 95th percentile for SFR
                else:
                    p_low, p_high = 2.5, 97.5  # 2.5th to 97.5th percentile for other parameters

                robust_min = np.percentile(all_vals, p_low)
                robust_max = np.percentile(all_vals, p_high)

                # Add some padding (5% of range)
                range_padding = (robust_max - robust_min) * 0.05
                plot_min = robust_min - range_padding
                plot_max = robust_max + range_padding

                # Set axis limits to robust range
                ax.set_xlim(plot_min, plot_max)
                ax.set_ylim(plot_min, plot_max)

                # Add 1:1 line using the robust range if requested
                if show_unity_line:
                    ax.plot([plot_min, plot_max], [plot_min, plot_max], 'r--', alpha=0.8, label='', linewidth=2)

                # Set labels for comparison mode
                ax.set_xlabel(f'BayeSED3 {label}')
                ax.set_ylabel(f'BAGPIPES {label}')
                ax.legend()

            # Calculate statistics
            if use_x_axis:
                # X-axis mode: calculate different statistics based on user preference
                stats_dict = {}
                outlier_details = {}  # Initialize outlier_details dictionary

                if x_axis_stats in ['both', 'codes']:
                    # BayeSED vs BAGPIPES comparison (same as comparison mode)
                    codes_residuals = bagpipes_vals - bayesed_vals
                    codes_median = np.median(codes_residuals)
                    codes_mad = np.median(np.abs(codes_residuals - codes_median))
                    codes_nmad = 1.4826 * codes_mad  # Convert MAD to equivalent standard deviation
                    codes_outlier_mask = np.abs(codes_residuals - codes_median) > 3 * codes_nmad
                    codes_outliers = np.sum(codes_outlier_mask)
                    
                    # Calculate robust statistics excluding outliers
                    inlier_mask = ~codes_outlier_mask
                    if np.sum(inlier_mask) > 2:  # Need at least 3 points for correlation
                        correlation_codes = np.corrcoef(bayesed_vals[inlier_mask], bagpipes_vals[inlier_mask])[0, 1]
                        bias_codes = np.mean(bagpipes_vals[inlier_mask] - bayesed_vals[inlier_mask])  # BAGPIPES - BayeSED3
                        rms_codes = np.sqrt(np.mean((bagpipes_vals[inlier_mask] - bayesed_vals[inlier_mask])**2))
                    else:
                        # Fallback to all data if too few inliers
                        correlation_codes = np.corrcoef(bayesed_vals, bagpipes_vals)[0, 1]
                        bias_codes = np.mean(bagpipes_vals - bayesed_vals)
                        rms_codes = np.sqrt(np.mean((bagpipes_vals - bayesed_vals)**2))
                    
                    stats_dict['codes'] = {
                        'correlation': correlation_codes,
                        'bias': bias_codes,
                        'rms': rms_codes,
                        'label': 'BayeSED vs BAGPIPES'
                    }
                    outlier_details['codes'] = codes_outliers

                if x_axis_stats in ['both', 'x_axis']:
                    # Individual codes vs x-axis parameter with robust statistics
                    
                    # BayeSED vs x-axis robust statistics
                    bayesed_residuals = bayesed_vals - x_axis_vals
                    bayesed_median = np.median(bayesed_residuals)
                    bayesed_mad = np.median(np.abs(bayesed_residuals - bayesed_median))
                    bayesed_nmad = 1.4826 * bayesed_mad  # Convert MAD to equivalent standard deviation
                    bayesed_outlier_mask = np.abs(bayesed_residuals - bayesed_median) > 3 * bayesed_nmad
                    bayesed_outliers = np.sum(bayesed_outlier_mask)
                    
                    bayesed_inlier_mask = ~bayesed_outlier_mask
                    if np.sum(bayesed_inlier_mask) > 2:  # Need at least 3 points for correlation
                        correlation_bayesed_x = np.corrcoef(x_axis_vals[bayesed_inlier_mask], bayesed_vals[bayesed_inlier_mask])[0, 1]
                        bias_bayesed_x = np.mean(bayesed_vals[bayesed_inlier_mask] - x_axis_vals[bayesed_inlier_mask])  # BayeSED - x_axis
                        rms_bayesed_x = np.sqrt(np.mean((bayesed_vals[bayesed_inlier_mask] - x_axis_vals[bayesed_inlier_mask])**2))
                    else:
                        # Fallback to all data if too few inliers
                        correlation_bayesed_x = np.corrcoef(x_axis_vals, bayesed_vals)[0, 1]
                        bias_bayesed_x = np.mean(bayesed_vals - x_axis_vals)
                        rms_bayesed_x = np.sqrt(np.mean((bayesed_vals - x_axis_vals)**2))

                    # BAGPIPES vs x-axis robust statistics
                    bagpipes_residuals = bagpipes_vals - x_axis_vals
                    bagpipes_median = np.median(bagpipes_residuals)
                    bagpipes_mad = np.median(np.abs(bagpipes_residuals - bagpipes_median))
                    bagpipes_nmad = 1.4826 * bagpipes_mad  # Convert MAD to equivalent standard deviation
                    bagpipes_outlier_mask = np.abs(bagpipes_residuals - bagpipes_median) > 3 * bagpipes_nmad
                    bagpipes_outliers = np.sum(bagpipes_outlier_mask)
                    
                    bagpipes_inlier_mask = ~bagpipes_outlier_mask
                    if np.sum(bagpipes_inlier_mask) > 2:  # Need at least 3 points for correlation
                        correlation_bagpipes_x = np.corrcoef(x_axis_vals[bagpipes_inlier_mask], bagpipes_vals[bagpipes_inlier_mask])[0, 1]
                        bias_bagpipes_x = np.mean(bagpipes_vals[bagpipes_inlier_mask] - x_axis_vals[bagpipes_inlier_mask])  # BAGPIPES - x_axis
                        rms_bagpipes_x = np.sqrt(np.mean((bagpipes_vals[bagpipes_inlier_mask] - x_axis_vals[bagpipes_inlier_mask])**2))
                    else:
                        # Fallback to all data if too few inliers
                        correlation_bagpipes_x = np.corrcoef(x_axis_vals, bagpipes_vals)[0, 1]
                        bias_bagpipes_x = np.mean(bagpipes_vals - x_axis_vals)
                        rms_bagpipes_x = np.sqrt(np.mean((bagpipes_vals - x_axis_vals)**2))

                    stats_dict['bayesed_x'] = {
                        'correlation': correlation_bayesed_x,
                        'bias': bias_bayesed_x,
                        'rms': rms_bayesed_x,
                        'label': f'BayeSED vs {current_x_label}'
                    }
                    stats_dict['bagpipes_x'] = {
                        'correlation': correlation_bagpipes_x,
                        'bias': bias_bagpipes_x,
                        'rms': rms_bagpipes_x,
                        'label': f'BAGPIPES vs {current_x_label}'
                    }
                    
                    # Store outlier counts for x-axis statistics
                    outlier_details['bayesed_x'] = bayesed_outliers
                    outlier_details['bagpipes_x'] = bagpipes_outliers

                # Count outliers (points outside the robust ranges)
                n_outliers_x = np.sum((x_axis_vals < x_plot_min) | (x_axis_vals > x_plot_max))
                n_outliers_y = np.sum((bayesed_vals < y_plot_min) | (bayesed_vals > y_plot_max) |
                                     (bagpipes_vals < y_plot_min) | (bagpipes_vals > y_plot_max))
                n_outliers = len(set(np.where((x_axis_vals < x_plot_min) | (x_axis_vals > x_plot_max) |
                                            (bayesed_vals < y_plot_min) | (bayesed_vals > y_plot_max) |
                                            (bagpipes_vals < y_plot_min) | (bagpipes_vals > y_plot_max))[0]))

                # Calculate specific outliers for different statistics if requested
                # Note: outlier_details already initialized above
                if x_axis_stats in ['both', 'x_axis']:
                    # Outliers for BayeSED vs x-axis (using NMAD-based 3-sigma criterion)
                    bayesed_residuals = bayesed_vals - x_axis_vals
                    bayesed_median = np.median(bayesed_residuals)
                    bayesed_mad = np.median(np.abs(bayesed_residuals - bayesed_median))
                    bayesed_nmad = 1.4826 * bayesed_mad  # Convert MAD to equivalent standard deviation
                    bayesed_outliers = np.sum(np.abs(bayesed_residuals - bayesed_median) > 3 * bayesed_nmad)

                    # Outliers for BAGPIPES vs x-axis (using NMAD-based 3-sigma criterion)
                    bagpipes_residuals = bagpipes_vals - x_axis_vals
                    bagpipes_median = np.median(bagpipes_residuals)
                    bagpipes_mad = np.median(np.abs(bagpipes_residuals - bagpipes_median))
                    bagpipes_nmad = 1.4826 * bagpipes_mad  # Convert MAD to equivalent standard deviation
                    bagpipes_outliers = np.sum(np.abs(bagpipes_residuals - bagpipes_median) > 3 * bagpipes_nmad)

                    outlier_details['bayesed_x'] = bayesed_outliers
                    outlier_details['bagpipes_x'] = bagpipes_outliers

                if x_axis_stats in ['both', 'codes']:
                    # Outliers for BayeSED vs BAGPIPES (using NMAD-based 3-sigma criterion)
                    codes_residuals = bagpipes_vals - bayesed_vals
                    codes_median = np.median(codes_residuals)
                    codes_mad = np.median(np.abs(codes_residuals - codes_median))
                    codes_nmad = 1.4826 * codes_mad  # Convert MAD to equivalent standard deviation
                    codes_outliers = np.sum(np.abs(codes_residuals - codes_median) > 3 * codes_nmad)
                    outlier_details['codes'] = codes_outliers

                # Use the first available statistics for the main param_stats (for backward compatibility)
                if 'codes' in stats_dict:
                    main_stats = stats_dict['codes']
                elif 'bayesed_x' in stats_dict:
                    main_stats = stats_dict['bayesed_x']
                else:
                    main_stats = {'correlation': np.nan, 'bias': np.nan, 'rms': np.nan}

            else:
                # Comparison mode: use robust statistics excluding outliers
                codes_residuals = bagpipes_vals - bayesed_vals
                codes_median = np.median(codes_residuals)
                codes_mad = np.median(np.abs(codes_residuals - codes_median))
                codes_nmad = 1.4826 * codes_mad  # Convert MAD to equivalent standard deviation
                codes_outlier_mask = np.abs(codes_residuals - codes_median) > 3 * codes_nmad
                
                # Calculate robust statistics excluding outliers
                inlier_mask = ~codes_outlier_mask
                if np.sum(inlier_mask) > 2:  # Need at least 3 points for correlation
                    correlation = np.corrcoef(bayesed_vals[inlier_mask], bagpipes_vals[inlier_mask])[0, 1]
                    bias = np.mean(bagpipes_vals[inlier_mask] - bayesed_vals[inlier_mask])  # BAGPIPES - BayeSED3
                    rms = np.sqrt(np.mean((bagpipes_vals[inlier_mask] - bayesed_vals[inlier_mask])**2))
                else:
                    # Fallback to all data if too few inliers
                    correlation = np.corrcoef(bayesed_vals, bagpipes_vals)[0, 1]
                    bias = np.mean(bagpipes_vals - bayesed_vals)
                    rms = np.sqrt(np.mean((bagpipes_vals - bayesed_vals)**2))

                # Count outliers (points outside the robust range)
                n_outliers_x = np.sum((bayesed_vals < plot_min) | (bayesed_vals > plot_max))
                n_outliers_y = np.sum((bagpipes_vals < plot_min) | (bagpipes_vals > plot_max))
                n_outliers = len(set(np.where((bayesed_vals < plot_min) | (bayesed_vals > plot_max) |
                                            (bagpipes_vals < plot_min) | (bagpipes_vals > plot_max))[0]))

                main_stats = {'correlation': correlation, 'bias': bias, 'rms': rms}

            # Store statistics for summary
            param_stats[label] = {
                'correlation': main_stats['correlation'],
                'bias': main_stats['bias'],
                'rms': main_stats['rms'],
                'n_objects': len(bayesed_vals),
                'n_outliers': n_outliers,
                'outlier_fraction': n_outliers / len(bayesed_vals) if len(bayesed_vals) > 0 else 0,
                'plot_mode': 'x_axis' if use_x_axis else 'comparison'
            }

            # Add detailed statistics and outliers for x-axis mode
            if use_x_axis and 'stats_dict' in locals():
                param_stats[label]['detailed_stats'] = stats_dict
                if 'outlier_details' in locals():
                    param_stats[label]['detailed_outliers'] = outlier_details

            # Add statistics text with outlier information
            if use_x_axis:
                # Build statistics text based on selected statistics
                stats_lines = []  # Remove object count from here

                if 'codes' in stats_dict:
                    outlier_text = f", 3σ out: {outlier_details.get('codes', 0)}" if 'outlier_details' in locals() and 'codes' in outlier_details else ""
                    stats_lines.append(f"Codes: r={stats_dict['codes']['correlation']:.3f}, bias={stats_dict['codes']['bias']:.3f}{outlier_text}")
                if 'bayesed_x' in stats_dict:
                    outlier_text = f", 3σ out: {outlier_details.get('bayesed_x', 0)}" if 'outlier_details' in locals() and 'bayesed_x' in outlier_details else ""
                    stats_lines.append(f"BayeSED: r={stats_dict['bayesed_x']['correlation']:.3f}, bias={stats_dict['bayesed_x']['bias']:.3f}{outlier_text}")
                if 'bagpipes_x' in stats_dict:
                    outlier_text = f", 3σ out: {outlier_details.get('bagpipes_x', 0)}" if 'outlier_details' in locals() and 'bagpipes_x' in outlier_details else ""
                    stats_lines.append(f"BAGPIPES: r={stats_dict['bagpipes_x']['correlation']:.3f}, bias={stats_dict['bagpipes_x']['bias']:.3f}{outlier_text}")

                stats_text = '\n'.join(stats_lines)
            else:
                stats_text = f'r = {main_stats["correlation"]:.3f}\nbias = {main_stats["bias"]:.3f}\nRMS = {main_stats["rms"]:.3f}'

            ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
                    verticalalignment='top', fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

            ax.grid(True, alpha=0.3)

            print(f"✓ {label}: {len(bayesed_vals)} objects, r={main_stats['correlation']:.3f}, bias={main_stats['bias']:.3f}")
        else:
            print(f"✗ {label}: No valid data points found")
            ax.text(0.5, 0.5, f'No valid data\nfor {label}',
                   transform=ax.transAxes, ha='center', va='center',
                   bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))

    # Remove empty subplots if any
    total_subplots = nrows * ncols
    for i in range(n_params, total_subplots):
        if i < len(axes):
            fig.delaxes(axes[i])

    # Title with catalog name and object count
    if catalog_name:
        title = f'{catalog_name} ({len(common_ids)} objects)'
    else:
        title = f'{len(common_ids)} objects'
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()

    # Print summary statistics
    print(f"\n=== Parameter Analysis Summary ===")
    print(f"Plot mode: {'Parameter vs x-axis plots' if use_x_axis else 'Parameter comparison plots'}")
    if use_x_axis:
        print(f"X-axis statistics mode: {x_axis_stats}")

    for param_name, stats in param_stats.items():
        if use_x_axis and 'detailed_stats' in stats:
            # Print detailed statistics for x-axis mode
            print(f"\n{param_name} (N={stats['n_objects']}):")
            detailed = stats['detailed_stats']
            detailed_outliers = stats.get('detailed_outliers', {})

            if 'codes' in detailed:
                s = detailed['codes']
                outlier_text = f", 3σ outliers (NMAD): {detailed_outliers.get('codes', 0)}" if 'codes' in detailed_outliers else ""
                print(f"  BayeSED vs BAGPIPES: r={s['correlation']:6.3f}, bias={s['bias']:7.3f}, RMS={s['rms']:6.3f}{outlier_text}")
            if 'bayesed_x' in detailed:
                s = detailed['bayesed_x']
                outlier_text = f", 3σ outliers (NMAD): {detailed_outliers.get('bayesed_x', 0)}" if 'bayesed_x' in detailed_outliers else ""
                print(f"  BayeSED vs x-axis:   r={s['correlation']:6.3f}, bias={s['bias']:7.3f}, RMS={s['rms']:6.3f}{outlier_text}")
            if 'bagpipes_x' in detailed:
                s = detailed['bagpipes_x']
                outlier_text = f", 3σ outliers (NMAD): {detailed_outliers.get('bagpipes_x', 0)}" if 'bagpipes_x' in detailed_outliers else ""
                print(f"  BAGPIPES vs x-axis:  r={s['correlation']:6.3f}, bias={s['bias']:7.3f}, RMS={s['rms']:6.3f}{outlier_text}")
        else:
            # Print single line summary
            print(f"{param_name:15s}: r={stats['correlation']:6.3f}, bias={stats['bias']:7.3f}, RMS={stats['rms']:6.3f}, N={stats['n_objects']:3d}")

    if save:
        plot_dir = os.path.join("pipes", "plots", "comparison")
        os.makedirs(plot_dir, exist_ok=True)

        # Generate filename with catalog name if provided
        if catalog_name:
            filename = f"{catalog_name}_parameter_scatter_comparison.png"
        else:
            filename = "parameter_scatter_comparison_all_objects.png"

        out_path = os.path.join(plot_dir, filename)
        plt.savefig(out_path, bbox_inches="tight", dpi=400)
        print(f"\nSaved scatter comparison: {out_path}")

    if show:
        plt.show()

    if not show:
        plt.close(fig)

    return fig, param_stats

def main():
    """Main function to run BayeSED3 vs BAGPIPES comparison."""

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Run BayeSED3 vs BAGPIPES comparison analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with individual object fitting
  python test_bayesed_bagpipes_comparison.py observation/CESS_mock/two.txt

  # Use catalog fitting mode of bagpipes for multiple objects
  python test_bayesed_bagpipes_comparison.py observation/CESS_mock/two.txt --bagpipes-fit-cat
        """
    )

    parser.add_argument('input_file',
                       help='Path to BayeSED input file (required)')

    parser.add_argument('--filters', '--filters-file',
                       dest='filters_file',
                       help='Path to filters file (auto-detected if not provided)')

    parser.add_argument('--filters-selected', '--filters-selected-file',
                       dest='filters_selected_file',
                       help='Path to selected filters file (auto-detected if not provided)')

    parser.add_argument('--bagpipes-fit-cat', '--fit-cat',
                       action='store_true',
                       help='Use BAGPIPES catalog fitting mode for multiple objects simultaneously. '
                            'If not specified, fits objects individually (default behavior).')

    args = parser.parse_args()

    input_file = args.input_file
    filters_file = args.filters_file
    filters_selected_file = args.filters_selected_file
    bagpipes_fit_cat = args.bagpipes_fit_cat

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
    ssp='bc2003_hr_stelib_chab_neb_300r'
    # ssp='bc2003_hr_stelib_chab_neb_2000r'
    params = BayeSEDParams.galaxy(
        input_file=input_file,
        filters=filters_file,
        filters_selected=filters_selected_file,
        outdir=output_dir,
        ssp_model=ssp,
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
        iprior_type=3,
        min=0.0,
        max=2.0,
    )

    # Set RDF parameters for modeling sigma_diff between observed spectra and model
    # RDF models the difference/scatter between observations and theoretical models
    params.rdf = RDFParams(
        id=-1,                   # Model ID (-1: apply to all models, 0,1,2...: specific model)
        num_polynomials=0       # Number of polynomials (-1: default/disable, 0: only use sigma_obs curve in input file, 1,2...: polynomial order)
    )

    # Set systematic error for model
    params.sys_err_mod = SysErrParams(
        iprior_type=3,    # Prior type (1=uniform, 3=linear-decreasing)
        min=0.01,         # Minimum fractional systematic error
        max=0.1,          # Maximum fractional systematic error
    )
    params.multinest = MultiNestParams(
        nlive=40,          # Good balance of speed/accuracy
        efr=0.1,            # Moderate efficiency
        tol=0.5,            # Standard tolerance
    )
    # params.SNRmin2 = SNRmin2Params(0.0,-1)
    # params.unweighted_samples = True


    # Run analysis
    result = bayesed.run(params)

    # Load and analyze results
    # Extract catalog name from input file (efficient - only reads first line)
    cat_name = BayeSEDDataLoader.extract_catalog_name(input_file)

    results = BayeSEDResults(output_dir, catalog_name=cat_name,model_config=ssp)
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
    exp["age"] = (0.1, 15.) # Gyr
    exp["age_prior"] = "log_10"
    exp["tau"] = (1e-3, 10.) # Gyr
    exp["tau_prior"] = "log_10"
    exp["massformed"] = (5., 12.) # log_10(mass formed)
    exp["metallicity"] = (0.005, 5.0) # Zsun
    # exp["metallicity_prior"] = "log_10"

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
    # noise["type"] = "white_scaled"
    # noise["scaling"] = (1., 1.5)
    # noise["scaling_prior"] = "log_10"
    # base_fit_instructions["noise"] = noise


    resolution_curve = data_loader.get_resolution_curve(IDs[0])
    fit_instructions = base_fit_instructions.copy()
    fit_instructions["R_curve"] = resolution_curve
    # plt.plot(fit_instructions["R_curve"][:, 0], fit_instructions["R_curve"][:, 1])
    # plt.xlabel("Wavelength/ \AA")
    # plt.ylabel("Spectral resolving power")
    # plt.show()

    bayesed_params = ['z', 'log(Mstar)[0,0]', 'log(SFR_{100Myr}/[M_{sun}/yr])[0,0]']
    bagpipes_params = ['redshift', 'stellar_mass', 'sfr']
    true_value_params = ['z_{True}', 'log(Mstar)[0,1]_{True}', 'log(SFR_{100Myr}/[M_{sun}/yr])[0,1]_{True}']
    if bagpipes_fit_cat:
        fit_cat = pipes.fit_catalogue(IDs, fit_instructions, data_loader.load_spectrum_only, photometry_exists=False, run=cat_name, make_plots=True)
        fit_cat.fit(verbose=True, sampler='nautilus', mpi_serial=False, pool=20, n_live=400)
        # fit_cat.fit(verbose=True, sampler='nautilus', mpi_serial=True, n_live=400) #multiple objects are fitted at once, each using one core

        # Define comparison parameters (used by both plot_parameter_scatter and plot_posterior_corner_comparison)
        bayesed_params1 = [ i+"_{median}" for i in bayesed_params ]
        bagpipes_params1 = [ i+"_50" for i in bagpipes_params ]
        labels1 = [r'$z$', r'$\log(M_{\star}\, /\, \mathrm{M}_{\odot})$', r'$\log(\mathrm{SFR}\, /\, \mathrm{M}_{\odot}\, \mathrm{yr}^{-1})$']

        # # Create parameter scatter comparison plot using all objects
        print(f"\n=== Creating Parameter Scatter Comparison Plot ===")
        bagpipes_cat_file = os.path.join("pipes", "cats", f"{cat_name}.fits")

        fig, param_stats = plot_parameter_scatter(
            bayesed_results=results_bayesed,  # Use the astropy table directly
            bagpipes_cat_file=bagpipes_cat_file,
            bayesed_params=bayesed_params1,
            bagpipes_params=bagpipes_params1,
            labels=labels1,
            # colorbar_param='SNR',
            x_axis_param=true_value_params,
            save=True,
            show=True,
            catalog_name=cat_name
        )
    else:
        labels = [r'z', r'\log(M_{\star}\, /\, \mathrm{M}_{\odot})', r'\log(SFR\, /\, \mathrm{M}_{\odot}\, \mathrm{yr}^{-1})']
        for ID in IDs:
        # for ID in []:
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
            #fit.fit(verbose=True, sampler='multinest', n_live=40, pool=20)
            runtime_s = time.time() - t0

            # Generate spectrum plots
            plot_spectrum_posterior_with_residuals(fit, ID, cat_name, runtime_s=runtime_s)

            # Generate corner comparison plots using GetDist
            print(f"\nGenerating corner comparison plot for object {ID}...")
            obj_true_values = extract_true_values(results_bayesed, ID, true_value_params, labels, verbose=True)

            plot_posterior_corner_comparison(results, fit, ID, bayesed_params, bagpipes_params, labels, obj_true_values,range_confidence=0.0,min_weight_ratio=0.0)



if __name__ == "__main__":
    main()

