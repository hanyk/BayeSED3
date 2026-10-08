"""
Results presentation: GetDist, plots, correlations/statistics.

Held by BayeSEDResults; constructed with a ResultsProducts instance (ADR-0003).
"""

from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union, Any
import logging

import numpy as np

logger = logging.getLogger(__name__)

# Near-zero variance: treat as fixed (same threshold as prior free-plot filter).
_FIXED_VARIANCE_EPS = 1e-10
_POSTERIOR_SIDECAR_KINDS = frozenset({
    'posterior', 'posterior_free', 'posterior_derived',
})


class ResultsPresentation:
    """Presentation adapter over ResultsProducts (plots + GetDist + analytics)."""

    def __init__(self, products):
        self._products = products
        self._getdist_samples_cache = {}

    @staticmethod
    def _getdist_param_names(samples) -> List[str]:
        return [p.name for p in samples.paramNames.names]

    @classmethod
    def _params_in_samples(
        cls,
        samples,
        names: List[str],
        *,
        require_varying: bool = False,
    ) -> List[str]:
        """Keep ``names`` that exist on ``samples`` (by name, never by index).

        When ``require_varying`` is true, also drop columns with variance
        ``<= _FIXED_VARIANCE_EPS`` (ADR-0010).
        """
        getdist_names = cls._getdist_param_names(samples)
        name_to_index = {name: i for i, name in enumerate(getdist_names)}
        kept: List[str] = []
        for name in names:
            idx = name_to_index.get(name)
            if idx is None:
                continue
            if require_varying:
                if np.var(samples.samples[:, idx]) <= _FIXED_VARIANCE_EPS:
                    continue
            kept.append(name)
        return kept

    def __getattr__(self, name):
        # Delegate data/state access to products (not methods defined here).
        return getattr(self._products, name)

    def get_getdist_samples(self, object_id: Optional[str] = None,
                           settings: Optional[Dict[str, Any]] = None) -> Any:
        """
        Get GetDist samples with parameter management.

        Parameters
        ----------
        object_id : str, optional
            Object ID to get samples for
        settings : dict, optional
            Settings dictionary to pass to getdist.loadMCSamples.
            Default: {'range_confidence': 0.001, 'min_weight_ratio': 1e-99}

        Returns
        -------
        getdist.MCSamples
            GetDist samples object
        """
        samples, _ = self._load_getdist_samples(
            object_id=object_id, settings=settings,
        )
        return samples

    def _load_getdist_samples(
        self,
        object_id: Optional[str] = None,
        settings: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Any, Path]:
        # Determine object to load
        if object_id is None:
            if self.object_id is not None:
                object_id = self.object_id
            else:
                # For sample-level access, use first available object with warning
                objects = self.list_objects()
                if objects:
                    object_id = objects[0]
                    logger.warning(f"No object_id provided for get_getdist_samples. Using first available object: {object_id}. More objects can be obtained with results.list_objects().")
                else:
                    raise ValueError("No objects available for GetDist samples")

        # Check cache first to avoid duplicate loading
        # Include settings in cache key to handle different settings
        settings_key = str(sorted(settings.items())) if settings else "default"
        cache_key = f"{object_id}_{self.model_config}_{settings_key}"
        cached = self._getdist_samples_cache.get(cache_key)
        if cached is not None:
            return cached

        # Find sample files for this object matching the current configuration
        sample_files = self._find_config_files(object_id, "*_sample_par.txt")
        paramnames_files = self._find_config_files(object_id, "*_sample_par.paramnames")

        # Ensure we have exactly one of each file type
        sample_file = self._select_unique_file(sample_files, "sample_par.txt", object_id, log_selection=False)
        paramnames_file = self._select_unique_file(paramnames_files, "paramnames", object_id, log_selection=False)

        # Log the selected files once here (main entry point for GetDist operations)
        try:
            sample_rel_path = sample_file.resolve().relative_to(Path.cwd().resolve())
            logger.info(f"Loading GetDist samples for {object_id}: {sample_rel_path}")
        except ValueError:
            # If files are not relative to cwd, show absolute paths
            logger.info(f"Loading GetDist samples for {object_id}: {sample_file.resolve()}")

        # Extract base name for GetDist (remove .txt suffix, keep _sample_par)
        base_name = str(sample_file).replace(".txt", "")

        try:
            # Try to use GetDist - pass the full base name including _sample_par
            import getdist
            from getdist import MCSamples

            # Set default settings if none provided
            if settings is None:
                settings = {'range_confidence': 0.001, 'min_weight_ratio': 1e-99}

            samples = getdist.loadMCSamples(base_name, settings=settings)

            # Apply parameter labeling if custom labels are set
            if hasattr(self._products, '_custom_labels') and self._products._custom_labels:
                # Get the parameter names and labels
                param_names = [param.name for param in samples.paramNames.names]
                param_labels = [param.label for param in samples.paramNames.names]

                # Apply custom labels
                updated_labels = []
                for i, name in enumerate(param_names):
                    if name in self._products._custom_labels:
                        updated_labels.append(self._products._custom_labels[name])
                    else:
                        updated_labels.append(param_labels[i])

                # Create new MCSamples with custom labels
                new_samples = MCSamples(
                    samples=samples.samples,
                    names=param_names,
                    labels=updated_labels,
                    weights=getattr(samples, 'weights', None),
                    loglikes=getattr(samples, 'loglikes', None),
                    name_tag=getattr(samples, 'name_tag', None),
                    label=getattr(samples, 'label', None)
                )

                # Copy other important attributes if they exist
                if hasattr(samples, 'ranges'):
                    new_samples.ranges = samples.ranges
                if hasattr(samples, 'sampler'):
                    new_samples.sampler = samples.sampler

                samples = new_samples

            sample_par = Path(sample_file)
            logger.info(f"Loaded GetDist samples for {object_id}: {samples.numrows} samples, {samples.n} parameters")
            
            # Cache the samples to avoid duplicate loading
            loaded = (samples, sample_par)
            self._getdist_samples_cache[cache_key] = loaded
            return loaded

        except ImportError:
            raise ImportError("GetDist is required for GetDist samples. Install with: pip install getdist")
    def set_parameter_labels(self, custom_labels: Dict[str, str]) -> None:
        """
        Set custom parameter labels for plotting.

        Parameters
        ----------
        custom_labels : Dict[str, str]
            Dictionary mapping parameter names to LaTeX labels
        """
        # Store custom labels for use in GetDist plotting
        self._products._custom_labels = custom_labels.copy()
        
        # Clear GetDist samples cache since labels have changed
        self._getdist_samples_cache.clear()
        
        logger.info(f"Set custom labels for {len(custom_labels)} parameters")

    @staticmethod
    def _posterior_sidecar_path(sample_par: Path, kind: str) -> str:
        if kind not in _POSTERIOR_SIDECAR_KINDS:
            raise ValueError(f'unknown posterior sidecar kind: {kind!r}')
        stem = sample_par.stem
        suffix = '_sample_par'
        if not stem.endswith(suffix):
            raise ValueError(f'expected *_sample_par path, got {sample_par}')
        stem = stem[:-len(suffix)]
        return str(sample_par.with_name(f'{stem}_{kind}.png'))

    def plot_posterior(self, object_id: Optional[str] = None,
                      params: Optional[List[str]] = None,
                      method: str = 'getdist', filled: bool = True,
                      show: bool = True, output_file: Optional[str] = None,
                      show_median: bool = True, show_confidence_intervals: bool = True,
                      confidence_level: float = 0.68, *,
                      save: bool = True,
                      _sidecar_kind: str = 'posterior',
                      **kwargs) -> Any:
        """
        Plot posterior distributions using GetDist.

        Parameters
        ----------
        object_id : str, optional
            Object ID to plot for. If None, uses current scope.
        params : List[str], optional
            Parameters to plot. If None, uses free parameters.
        method : str, default 'getdist'
            Plotting method to use
        filled : bool, default True
            Whether to use filled contours
        show : bool, default True
            Whether to display the plot
        output_file : str, optional
            Output file path for saving. If omitted and save is True, writes
            beside the loaded ``*_sample_par`` as
            ``{config stem}_posterior.png`` (``_sample_par`` stripped).
            ``plot_posterior_free`` / ``plot_posterior_derived`` use
            ``_posterior_free`` / ``_posterior_derived``.
        save : bool, default True
            Whether to write a PNG.
        show_median : bool, default True
            Whether to show median markers on plots
        show_confidence_intervals : bool, default True
            Whether to show confidence intervals on 1D marginal plots
        confidence_level : float, default 0.68
            Confidence level for intervals (0.68 for 1-sigma, 0.95 for 2-sigma)
        **kwargs
            Additional plotting arguments

        Returns
        -------
        Any
            Plot object (depends on method)
        """
        # Handle object_id selection
        if object_id is None:
            if self.object_id is not None:
                object_id = self.object_id
            else:
                # For sample-level access, use first available object with warning
                objects = self.list_objects()
                if objects:
                    object_id = objects[0]
                    logger.warning(f"No object_id provided for plot_posterior. Using first available object: {object_id}. More objects can be obtained with results.list_objects().")
                else:
                    raise ValueError("No objects available for plotting")

        samples, sample_par = self._load_getdist_samples(object_id=object_id)

        # ADR-0010: explicit params must all exist on this samples object;
        # default free list is filtered to present + varying names.
        if params is None:
            params = self._params_in_samples(
                samples,
                list(self.get_free_parameters()),
                require_varying=True,
            )
            if not params:
                raise ValueError('no parameters available in posterior samples')
        else:
            present = set(self._getdist_param_names(samples))
            missing = [p for p in params if p not in present]
            if missing:
                raise ValueError(
                    'parameter(s) not in posterior samples: '
                    + ', '.join(missing)
                )
            if not params:
                raise ValueError('no parameters available in posterior samples')

        # Import GetDist plotting
        try:
            from getdist import plots
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("GetDist is required for plotting. Install with: pip install getdist")

        # Helper functions for confidence intervals
        def _compute_statistics(samples_gd, param_names):
            """Compute medians and confidence intervals using GetDist built-in methods."""
            markers = {}
            confidence_intervals = {}

            if not (show_median or show_confidence_intervals):
                return markers, confidence_intervals

            for param in param_names:
                try:
                    # Use GetDist's built-in 1D density analysis
                    density = samples_gd.get1DDensity(param)
                    
                    if show_median:
                        # Use GetDist's built-in median calculation (properly weighted)
                        # The most reliable way is to use the 1D density's percentile method
                        try:
                            # Get the 50th percentile (median) from the 1D density
                            # This properly handles weighted samples
                            markers[param] = density.P50()  # 50th percentile = median
                        except AttributeError:
                            try:
                                # Alternative: use the confidence interval method
                                # Get the central value (median) by using a very small confidence interval
                                central_val = samples_gd.confidence(param, 0.001)  # Very narrow interval around median
                                if central_val is not None:
                                    if hasattr(central_val, '__len__') and len(central_val) > 0:
                                        markers[param] = central_val[0] if len(central_val) == 1 else (central_val[0] + central_val[1]) / 2
                                    else:
                                        markers[param] = central_val
                                else:
                                    markers[param] = density.mean
                            except:
                                # Final fallback: use mean
                                markers[param] = density.mean
                    
                    if show_confidence_intervals:
                        # Use GetDist's built-in confidence interval calculation
                        # This properly handles weighted samples and edge cases
                        lower = density.getLower(confidence_level)
                        upper = density.getUpper(confidence_level)
                        confidence_intervals[param] = (lower, upper)
                        
                except Exception as e:
                    logger.debug(f"GetDist built-in statistics failed for parameter {param}: {e}")
                    # Fallback to manual calculation for this parameter
                    try:
                        gd_param_names = samples_gd.getParamNames().list()
                        if param not in gd_param_names:
                            continue
                            
                        idx = gd_param_names.index(param)
                        samples_array = samples_gd.samples[:, idx]
                        weights = getattr(samples_gd, 'weights', None)

                        if show_median:
                            if weights is not None:
                                markers[param] = np.average(samples_array, weights=weights)
                            else:
                                markers[param] = np.median(samples_array)

                        if show_confidence_intervals:
                            lower_percentile = (1 - confidence_level) / 2
                            upper_percentile = 1 - lower_percentile
                            
                            if weights is not None:
                                # Weighted quantiles
                                sorted_idx = np.argsort(samples_array)
                                sorted_samples = samples_array[sorted_idx]
                                sorted_weights = weights[sorted_idx]
                                cumsum_weights = np.cumsum(sorted_weights)
                                cumsum_weights = cumsum_weights / cumsum_weights[-1]
                                lower_idx = np.searchsorted(cumsum_weights, lower_percentile)
                                upper_idx = np.searchsorted(cumsum_weights, upper_percentile)
                                confidence_intervals[param] = (sorted_samples[lower_idx], sorted_samples[upper_idx])
                            else:
                                quantiles = np.quantile(samples_array, [lower_percentile, upper_percentile])
                                confidence_intervals[param] = (quantiles[0], quantiles[1])
                    except Exception as fallback_e:
                        logger.debug(f"Fallback statistics computation also failed for {param}: {fallback_e}")

            return markers, confidence_intervals

        def _add_ci_to_1d_plot(ax, lower, upper, confidence_level):
            """Add confidence interval to 1D plot."""
            ax.axvspan(lower, upper, color='blue', alpha=0.15)
            # No legend - would be cluttered in corner plots

        def _add_ci_to_triangle_plot(fig, params, confidence_intervals, confidence_level):
            """Add confidence intervals to diagonal (1D marginal) plots in triangle plot."""
            if not fig or not hasattr(fig, 'axes'):
                return

            # Find diagonal axes: 1D marginals have lines but no collections
            diagonal_axes = [
                ax for ax in fig.axes
                if (len(ax.lines) > 0 and len(ax.collections) == 0 and
                    (not ax.get_ylabel() or ax.get_ylabel() == ''))
            ]

            if len(diagonal_axes) != len(params):
                # Fallback: match by parameter name in xlabel
                import re
                for param in params:
                    if param not in confidence_intervals:
                        continue
                    lower, upper = confidence_intervals[param]
                    param_base = param.split('[')[0].strip()

                    for ax in diagonal_axes:
                        xlabel = ax.get_xlabel() or ''
                        # Try matching parameter name
                        if (param in xlabel or param_base in xlabel or
                            param_base in re.sub(r'\\[a-zA-Z]+\{([^}]+)\}', r'\1', xlabel)):
                            _add_ci_to_1d_plot(ax, lower, upper, confidence_level)
                            break
            else:
                # Match by position (most reliable)
                for idx, param in enumerate(params):
                    if param in confidence_intervals and idx < len(diagonal_axes):
                        lower, upper = confidence_intervals[param]
                        _add_ci_to_1d_plot(diagonal_axes[idx], lower, upper, confidence_level)

        # Compute statistics if needed
        markers = {}
        confidence_intervals = {}
        if show_median or show_confidence_intervals:
            try:
                markers, confidence_intervals = _compute_statistics(samples, params)
            except Exception as e:
                import warnings
                warnings.warn(
                    f"Could not compute statistics: {e}. Statistics will not be displayed.",
                    UserWarning
                )
                show_median = False
                show_confidence_intervals = False

        # Create plotter
        g = self._getdist_subplot_plotter(plots, len(params))

        # Create plot based on number of parameters
        is_1d = len(params) == 1
        
        if is_1d:
            # 1D plot
            g.plot_1d(samples, params[0], **kwargs)
            
            # Add confidence interval for 1D plot
            if show_confidence_intervals and params[0] in confidence_intervals:
                ax = plt.gca()
                lower, upper = confidence_intervals[params[0]]
                _add_ci_to_1d_plot(ax, lower, upper, confidence_level)
        else:
            # Triangle plot with markers and confidence intervals
            g.triangle_plot([samples], params, filled=filled,
                           markers=markers if markers else None,
                           marker_args={'color': 'red', 'linestyle': '--',
                                      'linewidth': 1.5, 'alpha': 0.7},
                           **kwargs)

            # Add confidence intervals to diagonal plots
            if show_confidence_intervals:
                try:
                    fig = (getattr(g, 'fig', None) or
                          getattr(g, 'figure', None) or
                          plt.gcf())
                    _add_ci_to_triangle_plot(fig, params, confidence_intervals, confidence_level)
                except Exception as e:
                    import warnings
                    warnings.warn(
                        f"Could not add confidence intervals: {e}",
                        UserWarning
                    )

        # Save if requested
        if save:
            path = output_file
            if path is None:
                path = self._posterior_sidecar_path(
                    sample_par, _sidecar_kind,
                )
            g.export(path)

        # Show if requested
        if show:
            try:
                import matplotlib.pyplot as plt
                plt.show()
            except Exception as e:
                logger.warning(f"Could not display plot: {e}")

        return g

    def _getdist_subplot_plotter(self, plots_mod, n_params):
        """GetDist plotter sized so long axis labels fit the panels."""
        n = max(int(n_params), 1)
        subplot_size = 3.2 if n <= 2 else max(2.0, 7.0 / n)
        plotter = plots_mod.get_subplot_plotter(subplot_size=subplot_size)
        plotter.settings.scaling = False
        plotter.settings.figure_legend_frame = True
        plotter.settings.legend_fontsize = 12
        plotter.settings.axes_fontsize = 12
        plotter.settings.lab_fontsize = 12
        plotter.settings.axes_labelsize = 12
        return plotter

    def plot_bestfit(self, object_id: Optional[str] = None,
                    output_file: Optional[str] = None, show: bool = True,
                    *,
                    save: bool = True,
                    filter_file: Optional[str] = None, filter_selection_file: Optional[str] = None,
                    use_rest_frame: bool = True, flux_unit: str = 'fnu', use_log_scale: Optional[bool] = None,
                    model_names: Optional[List[str]] = None, show_emission_lines: bool = True,
                    show_total: bool = True, show_components: Union[bool, Dict[str, bool]] = True,
                    emission_line_fontsize: int = 12, title_fontsize: int = 16,
                    label_fontsize: int = 16, legend_fontsize: int = 14,
                    figsize: tuple = (12, 8), dpi: int = 300,
                    focus_on_data_range: bool = True,
                    show_residuals: bool = False,
                    residual_height_ratio: int = 4,
                    residual_ylim: Optional[tuple] = None,
                    **kwargs) -> Any:
        """
        Plot best-fit SED using the bayesed.plotting module.

        Parameters
        ----------
        object_id : str, optional
            Object ID to plot for. If None, uses current scope.
        output_file : str, optional
            Output file path for saving. If None, saves as {fits_file}.png
        show : bool, default True
            Whether to display the plot
        save : bool, default True
            Whether to write a PNG. When True and output_file is omitted, writes
            beside the FITS file.
        filter_file : str, optional
            Path to filter response file for overlay
        filter_selection_file : str, optional
            Path to filter selection file (filters_selected format)
        use_rest_frame : bool, default True
            Use rest-frame wavelengths. If False, uses observed-frame
        flux_unit : str, default 'fnu'
            Flux unit: 'fnu' (μJy), 'nufnu' (νFν in μJy*Hz), or 'flambda'
        use_log_scale : bool, optional
            Use logarithmic scale for axes. If None (default), auto-detects based on data range.
            Auto-detection uses log scale when either axis spans more than 1 order of magnitude
            (range ratio > 10). If negative values are present, defaults to linear scale.
            Set to True to force log scale, or False to force linear scale.
        model_names : list of str, optional
            Legend strings for model components, in ``model:`` extension order,
            excluding ``model:total``. ``None`` auto-generates one name per
            extension. Any other length raises ``ValueError`` before a curve
            is drawn.
        show_emission_lines : bool, default True
            Show emission line markers for spectroscopy
        show_total : bool, default True
            Whether to plot the total model curve. Set False to hide.
        show_components : bool or dict, default True
            Control visibility of individual model components.
            - True: show all components
            - False: hide all components
            - Dict: show/hide specific components by model name (e.g., `{'model:stellar': True, 'model:tor': False}`)
        emission_line_fontsize : int, default 12
            Font size for emission line labels. Larger values make labels more readable.
        title_fontsize : int, default 16
            Font size for the plot title
        label_fontsize : int, default 16
            Font size for axis labels (x and y axis)
        legend_fontsize : int, default 14
            Font size for legend text
        figsize : tuple, default (12, 8)
            Figure size (width, height) in inches
        dpi : int, default 300
            Resolution for saved figure
        focus_on_data_range : bool, default True
            If True, set x-axis limits to focus on the wavelength range where data exists
            (photometry and spectroscopy), ignoring the full model range. If False, use
            the full wavelength range from both models and data
        show_residuals : bool, default False
            Add a lower panel showing (observation - model) / error residuals for
            photometry and spectroscopy data.
        residual_height_ratio : int, default 4
            Main panel to residual panel height ratio when ``show_residuals`` is True.
        residual_ylim : tuple or None, optional
            Manual (ymin, ymax) for the residual panel. None auto-scales with at
            least [-3, 3] in sigma units.
        **kwargs
            Additional plotting arguments passed to matplotlib plotting functions

        Returns
        -------
        matplotlib.figure.Figure
            The matplotlib figure object

        Examples
        --------
        >>> # Basic usage
        >>> results.plot_bestfit()
        >>> 
        >>> # With residual panel
        >>> results.plot_bestfit(show_residuals=True)
        >>> 
        >>> # Customize plot
        >>> results.plot_bestfit(
        ...     use_rest_frame=True,
        ...     flux_unit='nufnu',
        ...     use_log_scale=True,
        ...     figsize=(14, 10)
        ... )
        >>> 
        >>> # Hide the total model line
        >>> results.plot_bestfit(show_total=False)
        >>>
        >>> # Hide specific components
        >>> results.plot_bestfit(show_components={'model:stellar': True, 'model:tor': False})
        >>>
        >>> # With filter overlay
        >>> results.plot_bestfit(
        ...     filter_file='filters.txt',
        ...     filter_selection_file='filters_selected.txt'
        ... )
        """
        # Determine object to plot
        if object_id is None:
            if self.object_id is not None:
                object_id = self.object_id
            else:
                # For sample-level access, use first available object with warning
                objects = self.list_objects()
                if objects:
                    object_id = objects[0]
                    logger.warning(f"No object_id provided for plot_bestfit. Using first available object: {object_id}. More objects can be obtained with results.list_objects().")
                else:
                    raise ValueError("No objects available for plotting best-fit spectrum")

        # Find bestfit FITS files for this object matching the current configuration
        fits_files = self._find_config_files(object_id, "*_bestfit.fits")

        # Ensure we have exactly one bestfit file (log selection for main user operation)
        fits_file = self._select_unique_file(fits_files, "bestfit FITS", object_id, log_selection=True)

        # Import plotting function
        from ..plotting import plot_bestfit

        # Create plot with file path, passing all the plotting options
        fig = plot_bestfit(
            fits_file, 
            output_file=output_file, 
            show=show,
            save=save,
            filter_file=filter_file,
            filter_selection_file=filter_selection_file,
            use_rest_frame=use_rest_frame,
            flux_unit=flux_unit,
            use_log_scale=use_log_scale,
            model_names=model_names,
            show_emission_lines=show_emission_lines,
            show_total=show_total,
            show_components=show_components,
            emission_line_fontsize=emission_line_fontsize,
            title_fontsize=title_fontsize,
            label_fontsize=label_fontsize,
            legend_fontsize=legend_fontsize,
            figsize=figsize,
            dpi=dpi,
            focus_on_data_range=focus_on_data_range,
            show_residuals=show_residuals,
            residual_height_ratio=residual_height_ratio,
            residual_ylim=residual_ylim,
            **kwargs
        )

        return fig
    def plot_posterior_free(self, object_id: Optional[str] = None,
                           output_file: Optional[str] = None,
                           show: bool = True,
                           show_median: bool = True, show_confidence_intervals: bool = True,
                           confidence_level: float = 0.68, *,
                           save: bool = True, **kwargs) -> Any:
        """
        Plot posterior distributions (corner plot) for free parameters.

        Parameters
        ----------
        object_id : str, optional
            Object ID to plot. If None and in sample-level mode, uses first object.
        output_file : str, optional
            Output file path for saving plot
        show : bool, default True
            Whether to display the plot
        save : bool, default True
            Whether to write a PNG. When True and output_file is omitted, writes
            beside ``*_sample_par`` as ``{config stem}_posterior_free.png``.
        show_median : bool, default True
            Whether to show median markers on plots
        show_confidence_intervals : bool, default True
            Whether to show confidence intervals on 1D marginal plots
        confidence_level : float, default 0.68
            Confidence level for intervals (0.68 for 1-sigma, 0.95 for 2-sigma)
        **kwargs
            Additional plotting parameters

        Returns
        -------
        Any
            Plot object
        """
        try:
            # Handle object_id selection
            if object_id is None:
                if self.object_id is not None:
                    object_id = self.object_id
                else:
                    # For sample-level access, use first available object with warning
                    objects = self.list_objects()
                    if objects:
                        object_id = objects[0]
                        logger.warning(f"No object_id provided for plot_posterior_free. Using first available object: {object_id}. More objects can be obtained with results.list_objects().")
                    else:
                        raise ValueError("No objects available for plotting")

            samples = self.get_getdist_samples(object_id=object_id)
            free_params = self._params_in_samples(
                samples,
                list(self.get_free_parameters()),
                require_varying=True,
            )
            if not free_params:
                raise ValueError('no parameters available in posterior samples')
            logger.debug(
                f"Filtered to {len(free_params)} varying free parameters "
                f"available in GetDist samples"
            )

            return self.plot_posterior(params=free_params, object_id=object_id,
                                     output_file=output_file, show=show,
                                     show_median=show_median, 
                                     show_confidence_intervals=show_confidence_intervals,
                                     confidence_level=confidence_level,
                                     save=save, _sidecar_kind='posterior_free', **kwargs)
        except Exception as e:
            if "GetDist" in str(e) or "No chains found" in str(e):
                logger.warning(f"GetDist plotting not available: {e}")
                logger.info("Consider using individual parameter plotting or ensure GetDist-compatible sample files exist")
                # Return a placeholder or simple message
                print(f"Free parameters ({len(self.get_free_parameters())}): {self.get_free_parameters()}")
                return None
            else:
                raise
    def plot_posterior_derived(self, object_id: Optional[str] = None,
                              max_params: int = 10,
                              output_file: Optional[str] = None,
                              show: bool = True,
                              show_median: bool = True, show_confidence_intervals: bool = True,
                              confidence_level: float = 0.68, *,
                              save: bool = True, **kwargs) -> Any:
        """
        Plot posterior distributions (corner plot) for derived parameters.

        Parameters
        ----------
        object_id : str, optional
            Object ID to plot. If None and in sample-level mode, uses first object.
        max_params : int, default 10
            Maximum number of parameters to plot
        output_file : str, optional
            Output file path for saving plot
        show : bool, default True
            Whether to display the plot
        save : bool, default True
            Whether to write a PNG. When True and output_file is omitted, writes
            beside ``*_sample_par`` as ``{config stem}_posterior_derived.png``.
        show_median : bool, default True
            Whether to show median markers on plots
        show_confidence_intervals : bool, default True
            Whether to show confidence intervals on 1D marginal plots
        confidence_level : float, default 0.68
            Confidence level for intervals (0.68 for 1-sigma, 0.95 for 2-sigma)
        **kwargs
            Additional plotting parameters

        Returns
        -------
        Any
            Plot object
        """
        try:
            # Handle object_id selection
            if object_id is None:
                if self.object_id is not None:
                    object_id = self.object_id
                else:
                    # For sample-level access, use first available object with warning
                    objects = self.list_objects()
                    if objects:
                        object_id = objects[0]
                        logger.warning(f"No object_id provided for plot_posterior_derived. Using first available object: {object_id}. More objects can be obtained with results.list_objects().")
                    else:
                        raise ValueError("No objects available for plotting")

            samples = self.get_getdist_samples(object_id=object_id)
            derived_params = self._params_in_samples(
                samples,
                list(self.get_derived_parameters()),
                require_varying=True,
            )
            logger.debug(
                f"Filtered to {len(derived_params)} varying derived parameters "
                f"available in GetDist samples"
            )

            # Limit to max_params if specified (after name+variance filter)
            if max_params and len(derived_params) > max_params:
                derived_params = derived_params[:max_params]
            if not derived_params:
                raise ValueError('no parameters available in posterior samples')

            return self.plot_posterior(params=derived_params, object_id=object_id,
                                     output_file=output_file, show=show,
                                     show_median=show_median,
                                     show_confidence_intervals=show_confidence_intervals,
                                     confidence_level=confidence_level,
                                     save=save, _sidecar_kind='posterior_derived', **kwargs)
        except Exception as e:
            if "GetDist" in str(e) or "No chains found" in str(e):
                logger.warning(f"GetDist plotting not available: {e}")
                logger.info("Consider using individual parameter plotting or ensure GetDist-compatible sample files exist")
                # Return a placeholder or simple message
                limited_params = self.get_derived_parameters()[:max_params] if max_params else self.get_derived_parameters()
                print(f"Derived parameters ({len(limited_params)}): {limited_params}")
                return None
            else:
                raise
    def plot_parameter_scatter(self, x_parameter: Union[str, np.ndarray, List[float]], 
                              y_parameter: Union[str, List[str]],
                              x_err: Optional[Union[str, np.ndarray, List[float]]] = None,
                              y_err: Optional[Union[str, List[str]]] = None,
                              color_parameter: Optional[str] = None,
                              object_ids: Optional[Union[str, List[str]]] = None,
                              show_diagonal: bool = True,
                              show_colorbar: bool = True,
                              show_stats: bool = True,
                              figsize: tuple = (8, 6),
                              xlim: Optional[tuple] = None,
                              ylim: Optional[tuple] = None,
                              output_file: Optional[str] = None,
                              show: bool = True,
                              xlabel: Optional[str] = None,
                              ylabel: Optional[str] = None,
                              title: Optional[str] = None,
                              legend_labels: Optional[List[str]] = None,
                              legend_loc: str = 'best',
                              **kwargs) -> Any:
        """
        Create a scatter plot comparing one x-parameter with one or multiple y-parameters with optional error bars.
        
        This method creates scatter plots similar to:
        plt.scatter(results.parameters['log(Mstar)[0,1]_{True}'], 
                   results.parameters['log(Mstar)[0,0]_{median}'])
        
        When y_parameter is a list, all y-parameters will be plotted against the same x-parameter
        in the same figure with different colors and point styles for easy comparison.
        
        Parameters
        ----------
        x_parameter : str or array-like
            X-axis values. Can be either:
            - str: Full column name from HDF5 file (e.g., 'log(Mstar)[0,1]_{True}')
            - array-like: External numpy array or list of values
        y_parameter : str or List[str]
            Full column name(s) for y-axis values (e.g., 'log(Mstar)[0,0]_{median}' or 
            ['log(Mstar)[0,0]_{median}', 'log(age)[0,0]_{median}']). When a list is provided,
            all y-parameters will be plotted against the x-parameter with different colors and markers.
        x_err : str or array-like, optional
            X-axis error bars. Can be either:
            - str: Column name or percentile specification from HDF5 file:
              * 'sigma' for sigma column
              * 'percentile' or 'percentile_68' for 68% confidence interval
              * 'percentile_95' for 95% confidence interval  
              * 'percentile_90' for 90% confidence interval
              * Direct column name (e.g., 'log(Mstar)[0,0]_{sigma}')
            - array-like: External error values. Can be:
              * 1D array: Symmetric errors (same length as x_parameter)
              * 2D array: Asymmetric errors, shape (2, N) where first row is lower errors
                and second row is upper errors (for confidence intervals)
            Note: If x_parameter is external array, x_err must also be external array or None
        y_err : str or List[str], optional
            Column name(s) for y-axis error bars. Same options as x_err. If y_parameter is a list,
            y_err can be a single string (applied to all y-parameters) or a list of strings
            (one for each y-parameter, or None for no error bars on specific parameters).
        color_parameter : str, optional
            Full column name for color coding points. If None and y_parameter is a list, 
            uses different colors for each y-parameter. If specified with multiple y-parameters,
            all points will be colored by this parameter.
        object_ids : str or List[str], optional
            Specific object ID(s) to include. If None, uses all available objects.
        show_diagonal : bool, default True
            Whether to show the diagonal line (y=x)
        show_colorbar : bool, default True
            Whether to show colorbar (only used if color_parameter is specified)
        show_stats : bool, default True
            Whether to show correlation statistics on the plot. Includes Pearson r,
            Spearman ρ, and error-weighted correlation (if errors are available).
            For multiple y-parameters, shows stats for each y vs x relationship.
        figsize : tuple, default (8, 6)
            Figure size (width, height) in inches
        xlim : tuple, optional
            X-axis limits as (xmin, xmax). If None, uses automatic scaling.
        ylim : tuple, optional
            Y-axis limits as (ymin, ymax). If None, uses automatic scaling.
        output_file : str, optional
            Output file path for saving plot
        show : bool, default True
            Whether to display the plot
        xlabel : str, optional
            Custom x-axis label. If None, uses x_parameter name.
        ylabel : str, optional
            Custom y-axis label. If None, uses y_parameter name(s).
        title : str, optional
            Custom plot title. If None, auto-generates from parameter names.
        legend_labels : List[str], optional
            Custom labels for the legend. If provided, must have the same length as y_parameter list.
            If None, uses automatically extracted labels from parameter names.
        legend_loc : str, default 'best'
            Location for the legend. Options: 'best', 'upper right', 'upper left', 'lower left',
            'lower right', 'right', 'center left', 'center right', 'lower center', 'upper center', 'center'.
        **kwargs
            Additional arguments passed to plt.errorbar() or plt.scatter()
            
        Returns
        -------
        matplotlib.figure.Figure
            The matplotlib figure object
            
        Examples
        --------
        >>> # Basic scatter plot (single y-parameter)
        >>> results.plot_parameter_scatter('log(Mstar)[0,1]_{True}', 
        ...                               'log(Mstar)[0,0]_{median}')
        >>> 
        >>> # Multiple y-parameters comparison
        >>> results.plot_parameter_scatter('log(Mstar)[0,1]_{True}', 
        ...                               ['log(Mstar)[0,0]_{median}', 'log(age)[0,0]_{median}'])
        >>> 
        >>> # With error bars using sigma (single y-parameter)
        >>> results.plot_parameter_scatter('log(Mstar)[0,1]_{True}', 
        ...                               'log(Mstar)[0,0]_{median}',
        ...                               y_err='log(Mstar)[0,0]_{sigma}')
        >>> 
        >>> # Multiple y-parameters with different error bars
        >>> results.plot_parameter_scatter('log(Mstar)[0,0]_{median}', 
        ...                               ['log(age)[0,0]_{median}', 'SFR_{median}'],
        ...                               y_err=['log(age)[0,0]_{sigma}', 'SFR_{sigma}'])
        >>> 
        >>> # Multiple y-parameters with same error type
        >>> results.plot_parameter_scatter('log(Mstar)[0,0]_{median}', 
        ...                               ['log(age)[0,0]_{median}', 'SFR_{median}'],
        ...                               y_err='percentile')
        >>> 
        >>> # With error bars using percentiles (68% confidence interval)
        >>> results.plot_parameter_scatter('log(Mstar)[0,0]_{median}', 
        ...                               'log(age)[0,0]_{median}',
        ...                               x_err='percentile', y_err='percentile')
        >>> 
        >>> # With 95% confidence interval error bars
        >>> results.plot_parameter_scatter('log(Mstar)[0,0]_{median}', 
        ...                               'log(age)[0,0]_{median}',
        ...                               x_err='percentile_95', y_err='percentile_95')
        >>> 
        >>> # Mixed error types: 90% percentiles for x, sigma for y
        >>> results.plot_parameter_scatter('log(Mstar)[0,0]_{median}', 
        ...                               'SFR_{median}',
        ...                               x_err='percentile_90', y_err='sigma')
        >>> 
        >>> # Custom axis ranges with multiple y-parameters
        >>> results.plot_parameter_scatter('log(Mstar)[0,1]_{True}', 
        ...                               ['log(Mstar)[0,0]_{median}', 'log(age)[0,0]_{median}'],
        ...                               xlim=(9, 12), ylim=(8, 12))
        >>> 
        >>> # Turn off statistics display
        >>> results.plot_parameter_scatter('log(Mstar)[0,1]_{True}', 
        ...                               ['log(Mstar)[0,0]_{median}', 'log(age)[0,0]_{median}'],
        ...                               show_stats=False)
        >>> 
        >>> # With color coding (overrides automatic coloring for multiple y-parameters)
        >>> results.plot_parameter_scatter('log(Mstar)[0,1]_{True}', 
        ...                               ['log(Mstar)[0,0]_{median}', 'log(age)[0,0]_{median}'],
        ...                               color_parameter='log(age)[0,0]_{median}')
        >>> 
        >>> # With custom legend labels
        >>> results.plot_parameter_scatter('log(Mstar)[0,1]_{True}', 
        ...                               ['log(Mstar)[0,0]_{median}', 'log(age)[0,0]_{median}'],
        ...                               legend_labels=['Stellar Mass', 'Age'])
        >>> 
        >>> # Custom legend position
        >>> results.plot_parameter_scatter('log(Mstar)[0,1]_{True}', 
        ...                               ['log(Mstar)[0,0]_{median}', 'log(age)[0,0]_{median}'],
        ...                               legend_loc='upper left')
        """
        try:
            import matplotlib.pyplot as plt
            import numpy as np
        except ImportError:
            raise ImportError("matplotlib and numpy are required for plotting. Install with: pip install matplotlib numpy")
        
        # Convert y_parameter to list for consistent handling
        if isinstance(y_parameter, str):
            y_parameters = [y_parameter]
            single_y = True
        else:
            y_parameters = y_parameter
            single_y = False
        
        # Handle y_err parameter - convert to list matching y_parameters
        if y_err is None:
            y_err_list = [None] * len(y_parameters)
        elif isinstance(y_err, str):
            # Single error specification - apply to all y-parameters
            y_err_list = [y_err] * len(y_parameters)
        elif isinstance(y_err, list):
            if len(y_err) != len(y_parameters):
                raise ValueError(f"Length of y_err list ({len(y_err)}) must match length of y_parameter list ({len(y_parameters)})")
            y_err_list = y_err
        else:
            raise ValueError("y_err must be None, str, or list of str")
        
        # Handle legend_labels parameter
        if legend_labels is not None:
            if not isinstance(legend_labels, list):
                raise ValueError("legend_labels must be a list of strings")
            if len(legend_labels) != len(y_parameters):
                raise ValueError(f"Length of legend_labels ({len(legend_labels)}) must match length of y_parameter list ({len(y_parameters)})")
            custom_legend_labels = legend_labels
        else:
            custom_legend_labels = None
        
        # Get the HDF5 table with appropriate filtering
        hdf5_table = self.load_hdf5_results(filter_snr=False, min_snr=0.0)
        
        # Apply object filtering if specified
        if object_ids is not None:
            if isinstance(object_ids, str):
                object_ids = [object_ids]
            object_mask = [str(obj_id) in [str(x) for x in object_ids] for obj_id in hdf5_table['ID']]
            if any(object_mask):
                hdf5_table = hdf5_table[object_mask]
            else:
                available_objects = [str(obj_id) for obj_id in hdf5_table['ID']]
                raise ValueError(f"None of the specified objects {object_ids} found in data. "
                               f"Available objects: {available_objects}")
        
        # Check if columns exist
        available_cols = hdf5_table.colnames
        
        # Handle x_parameter - can be string (column name) or array-like (external data)
        if isinstance(x_parameter, str):
            # Current behavior - validate column exists in HDF5
            if x_parameter not in available_cols:
                # Try to find similar columns
                similar_cols = [col for col in available_cols if x_parameter.split('_')[0] in col or x_parameter.split('[')[0] in col]
                if similar_cols:
                    raise ValueError(f"Column '{x_parameter}' not found. Similar columns: {similar_cols[:10]}")
                else:
                    raise ValueError(f"Column '{x_parameter}' not found. Available columns: {available_cols[:10]}...")
            
            # Extract x data from HDF5
            x_values = np.array(hdf5_table[x_parameter])
            x_label_default = x_parameter
        else:
            # New behavior - use external array
            x_values = np.asarray(x_parameter)
            if len(x_values) != len(hdf5_table):
                raise ValueError(f"External x_parameter array length ({len(x_values)}) "
                               f"doesn't match dataset length ({len(hdf5_table)})")
            x_label_default = "External X Parameter"
        
        # Check all y_parameters exist
        for y_param in y_parameters:
            if y_param not in available_cols:
                # Try to find similar columns
                similar_cols = [col for col in available_cols if y_param.split('_')[0] in col or y_param.split('[')[0] in col]
                if similar_cols:
                    raise ValueError(f"Column '{y_param}' not found. Similar columns: {similar_cols[:10]}")
                else:
                    raise ValueError(f"Column '{y_param}' not found. Available columns: {available_cols[:10]}...")
        
        # Handle error bars
        def get_error_values(param, err_spec, param_name):
            """Get error values for a parameter."""
            if err_spec is None:
                return None
            
            if err_spec.startswith('percentile'):
                # Parse percentile specification
                if err_spec == 'percentile':
                    # Default to 68% confidence interval (16th/84th percentiles)
                    lower_p, upper_p = 0.16, 0.84
                elif '_' in err_spec:
                    # Custom percentile like 'percentile_95' or 'percentile_90'
                    try:
                        confidence_level = float(err_spec.split('_')[1])
                        if confidence_level > 1:
                            confidence_level = confidence_level / 100.0  # Convert percentage to fraction
                        
                        # Calculate symmetric percentiles around median
                        lower_p = (1 - confidence_level) / 2
                        upper_p = 1 - lower_p
                        
                        logger.info(f"Using {confidence_level*100:.1f}% confidence interval: {lower_p:.3f}/{upper_p:.3f} percentiles")
                    except (ValueError, IndexError):
                        logger.warning(f"Invalid percentile specification '{err_spec}'. Using default 68% interval.")
                        lower_p, upper_p = 0.16, 0.84
                else:
                    logger.warning(f"Invalid percentile specification '{err_spec}'. Using default 68% interval.")
                    lower_p, upper_p = 0.16, 0.84
                
                # Auto-calculate from specified percentiles
                base_param = param.replace('_{median}', '').replace('_{mean}', '').replace('_{MAP}', '').replace('_{MAL}', '')
                
                # Format percentiles as strings (handle both decimal and percentage formats)
                if lower_p < 1:
                    lower_str = f'{lower_p:.3f}'.rstrip('0').rstrip('.')
                    upper_str = f'{upper_p:.3f}'.rstrip('0').rstrip('.')
                else:
                    lower_str = f'{lower_p:.0f}'
                    upper_str = f'{upper_p:.0f}'
                
                # Try different column name formats
                possible_formats = [
                    (f'_{{{lower_str}}}', f'_{{{upper_str}}}'),  # _{0.16}, _{0.84}
                    (f'_{lower_str}', f'_{upper_str}'),          # _0.16, _0.84
                    (f'_p{int(lower_p*100)}', f'_p{int(upper_p*100)}'),  # _p16, _p84
                ]
                
                lower_col = upper_col = None
                for lower_fmt, upper_fmt in possible_formats:
                    lower_candidate = base_param + lower_fmt
                    upper_candidate = base_param + upper_fmt
                    
                    if lower_candidate in available_cols and upper_candidate in available_cols:
                        lower_col = lower_candidate
                        upper_col = upper_candidate
                        break
                
                if lower_col and upper_col:
                    lower_values = np.array(hdf5_table[lower_col])
                    upper_values = np.array(hdf5_table[upper_col])
                    param_values = np.array(hdf5_table[param])
                    
                    # Calculate asymmetric errors
                    lower_err = param_values - lower_values
                    upper_err = upper_values - param_values
                    
                    logger.info(f"Using percentile columns: {lower_col}, {upper_col}")
                    # Return as [lower_errors, upper_errors] for asymmetric error bars
                    return np.array([lower_err, upper_err])
                else:
                    # Show what columns were tried
                    tried_cols = []
                    for lower_fmt, upper_fmt in possible_formats:
                        tried_cols.extend([base_param + lower_fmt, base_param + upper_fmt])
                    
                    logger.warning(f"Percentile columns not found for {param_name}. Tried: {tried_cols}")
                    
                    # Show available similar columns
                    similar_cols = [col for col in available_cols if base_param in col and any(p in col for p in ['0.', '_p', '_{0.', '_{1.'])]
                    if similar_cols:
                        logger.info(f"Available percentile-like columns: {similar_cols}")
                    
                    return None
            
            elif err_spec == 'sigma':
                # Auto-find sigma column
                base_param = param.replace('_{median}', '').replace('_{mean}', '').replace('_{MAP}', '').replace('_{MAL}', '')
                sigma_col = base_param + '_{sigma}'
                
                if sigma_col in available_cols:
                    return np.array(hdf5_table[sigma_col])
                else:
                    logger.warning(f"Sigma column not found: {sigma_col}")
                    return None
            
            else:
                # Direct column name
                if err_spec in available_cols:
                    return np.array(hdf5_table[err_spec])
                else:
                    logger.warning(f"Error column '{err_spec}' not found for {param_name}")
                    return None
        
        # Handle x_err - can be string (column name) or array-like (external data)
        x_err_values = None
        if x_err is not None:
            if isinstance(x_err, str):
                # String x_err - must have string x_parameter for HDF5 column lookup
                if isinstance(x_parameter, str):
                    x_err_values = get_error_values(x_parameter, x_err, 'x_parameter')
                else:
                    raise ValueError("Cannot use string x_err with external x_parameter array. "
                                   "Provide x_err as an array or set to None.")
            else:
                # External error array
                x_err_array = np.asarray(x_err)
                
                # Handle both symmetric (1D) and asymmetric (2D) error arrays
                if x_err_array.ndim == 1:
                    # Symmetric errors
                    if len(x_err_array) != len(x_values):
                        raise ValueError(f"External x_err array length ({len(x_err_array)}) "
                                       f"doesn't match x_parameter length ({len(x_values)})")
                    x_err_values = x_err_array
                elif x_err_array.ndim == 2:
                    # Asymmetric errors (confidence intervals)
                    if x_err_array.shape[0] != 2:
                        raise ValueError(f"2D x_err array must have shape (2, N) for asymmetric errors, "
                                       f"got shape {x_err_array.shape}")
                    if x_err_array.shape[1] != len(x_values):
                        raise ValueError(f"2D x_err array second dimension ({x_err_array.shape[1]}) "
                                       f"doesn't match x_parameter length ({len(x_values)})")
                    x_err_values = x_err_array
                else:
                    raise ValueError(f"External x_err array must be 1D (symmetric) or 2D (asymmetric), "
                                   f"got {x_err_array.ndim}D array")
        
        # Handle color parameter if specified
        color_values = None
        if color_parameter is not None:
            if color_parameter not in available_cols:
                logger.warning(f"Color parameter '{color_parameter}' not found. Available columns: "
                             f"{[col for col in available_cols if color_parameter.split('_')[0] in col or color_parameter.split('[')[0] in col][:5]}")
                logger.warning("Proceeding without color coding.")
            else:
                color_values = np.array(hdf5_table[color_parameter])
        
        # Create the plot
        fig, ax = plt.subplots(figsize=figsize)
        
        # Define colors and markers for multiple y-parameters
        colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
        markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
        
        # Extract common and different parts for ylabel and legend
        def extract_common_and_different_parts(param_names):
            """
            Extract common and different parts from parameter names for ylabel and legend.
            
            This function analyzes parameter names to find meaningful common parts and 
            extract the differentiating parts for cleaner labels.
            
            Examples:
            - ['log(age/yr)[0,0]_{mean}', 'log(age/yr)[0,0]_{median}', 'log(age/yr)[0,0]_{MAP}'] 
              -> common: 'log(age/yr)[0,0]', different: ['mean', 'median', 'MAP']
            - ['log(Mstar)[0,0]_{median}', 'log(age)[0,0]_{median}']
              -> common: '[0,0]_{median}', different: ['log(Mstar)', 'log(age)']
            - ['SFR_{median}', 'log(Z)[0,0]_{median}']
              -> common: '_{median}', different: ['SFR', 'log(Z)[0,0]']
            """
            if len(param_names) == 1:
                return param_names[0], [param_names[0]]
            
            # Strategy: Look for patterns in parameter names
            # Common patterns: parameter_{statistic}, parameter[index]_{statistic}
            
            # First, try to identify if all parameters have the same base with different statistics
            # Look for patterns like _{mean}, _{median}, _{MAP}, etc.
            
            # Check if all parameters follow the pattern: base_{statistic}
            base_parts = []
            stat_parts = []
            all_have_underscore_stat = True
            
            for name in param_names:
                # Look for the last underscore followed by a statistic name
                if '_{' in name:
                    # Handle both _{stat} and _{stat} patterns
                    last_underscore_brace = name.rfind('_{')
                    if last_underscore_brace != -1:
                        base_part = name[:last_underscore_brace]
                        stat_part = name[last_underscore_brace+2:]  # Skip _{
                        if stat_part.endswith('}'):
                            stat_part = stat_part[:-1]  # Remove trailing }
                        base_parts.append(base_part)
                        stat_parts.append(stat_part)
                    else:
                        all_have_underscore_stat = False
                        break
                elif '_' in name:
                    # Simple pattern: parameter_statistic
                    parts = name.rsplit('_', 1)
                    if len(parts) == 2:
                        base_part = parts[0]
                        stat_part = parts[1]
                        base_parts.append(base_part)
                        stat_parts.append(stat_part)
                    else:
                        all_have_underscore_stat = False
                        break
                else:
                    all_have_underscore_stat = False
                    break
            
            # Case 1: All parameters have the pattern base_{statistic}
            if all_have_underscore_stat and len(base_parts) == len(param_names):
                unique_bases = set(base_parts)
                
                if len(unique_bases) == 1:
                    # All base parts are identical - different statistics of same parameter
                    # This is the case we want: log(age/yr)[0,1]_{mean} vs log(age/yr)[0,1]_{median}
                    common_part = base_parts[0]
                    different_parts = stat_parts
                    return common_part, different_parts
                else:
                    # Different base parts, but check if they have same statistic
                    unique_stats = set(stat_parts)
                    if len(unique_stats) == 1:
                        # Same statistic, different parameters
                        common_part = '_{' + stat_parts[0] + '}'
                        different_parts = base_parts
                        return common_part, different_parts
            
            # Case 2: Fallback to character-by-character comparison
            # Find common prefix
            common_prefix = ""
            min_len = min(len(name) for name in param_names)
            for i in range(min_len):
                if all(name[i] == param_names[0][i] for name in param_names):
                    common_prefix += param_names[0][i]
                else:
                    break
            
            # Find common suffix
            common_suffix = ""
            for i in range(1, min_len + 1):
                if all(name[-i] == param_names[0][-i] for name in param_names):
                    common_suffix = param_names[0][-i] + common_suffix
                else:
                    break
            
            # Extract different parts (middle parts between common prefix and suffix)
            different_parts = []
            for name in param_names:
                start_idx = len(common_prefix)
                end_idx = len(name) - len(common_suffix) if common_suffix else len(name)
                different_part = name[start_idx:end_idx]
                different_parts.append(different_part)
            
            # Construct common part
            common_part = common_prefix + common_suffix
            
            # Clean up and validate results
            if len(common_part.strip()) < 3:  # Too short to be meaningful
                # Fallback: use simplified parameter names
                different_parts = []
                for name in param_names:
                    # Extract base parameter name (before first underscore or bracket)
                    base_name = name.split('_')[0].split('[')[0]
                    different_parts.append(base_name)
                common_part = "Multiple Parameters"
            else:
                # Clean up different parts - remove common separators
                cleaned_different = []
                for part in different_parts:
                    # Remove leading/trailing underscores, brackets, etc.
                    cleaned = part.strip('_{}[]')
                    if not cleaned:
                        # If nothing left after cleaning, use a fallback
                        idx = different_parts.index(part)
                        original_name = param_names[idx]
                        cleaned = original_name.split('_')[0].split('[')[0]
                    cleaned_different.append(cleaned)
                different_parts = cleaned_different
            
            return common_part, different_parts
        
        # Get common and different parts for ylabel and legend (only if custom labels not provided)
        if not single_y and custom_legend_labels is None:
            common_ylabel, auto_legend_labels = extract_common_and_different_parts(y_parameters)
        else:
            common_ylabel = y_parameters[0] if single_y else "Multiple Parameters"
            auto_legend_labels = custom_legend_labels if custom_legend_labels else [y_parameters[0]]
        
        # Use custom legend labels if provided, otherwise use auto-extracted labels
        final_legend_labels = custom_legend_labels if custom_legend_labels else auto_legend_labels
        
        # Store data for statistics calculation
        all_stats_data = []
        
        # Plot each y-parameter
        for i, (y_param, y_err_spec) in enumerate(zip(y_parameters, y_err_list)):
            # Extract y data and errors
            y_values = np.array(hdf5_table[y_param])
            y_err_values = get_error_values(y_param, y_err_spec, f'y_parameter[{i}]')
            
            # Prepare plotting arguments
            plot_kwargs = {'alpha': 0.7}
            
            # Set color and marker
            if color_parameter is None and not single_y:
                # Use different colors/markers for each y-parameter
                plot_kwargs['color'] = colors[i % len(colors)]
                plot_kwargs['marker'] = markers[i % len(markers)]
                plot_kwargs['label'] = final_legend_labels[i]  # Use final legend labels
            elif color_parameter is not None:
                # Use color parameter for coloring
                plot_kwargs['c'] = color_values
                plot_kwargs['marker'] = markers[i % len(markers)] if not single_y else 'o'
                if not single_y:
                    plot_kwargs['label'] = final_legend_labels[i]  # Use final legend labels
            else:
                # Single y-parameter, default styling
                plot_kwargs['marker'] = 'o'
            
            if x_err_values is not None or y_err_values is not None:
                # Use errorbar plot
                plot_kwargs.update({'markersize': 5, 'capsize': 3, 'linestyle': 'none'})
            else:
                # Use scatter plot
                plot_kwargs.update({'s': 50})
            
            # Override with user kwargs (but preserve color/marker for multi-y)
            user_kwargs = kwargs.copy()
            if not single_y and color_parameter is None:
                # Preserve automatic color/marker for multi-y plots
                user_kwargs.pop('color', None)
                user_kwargs.pop('c', None)
                user_kwargs.pop('marker', None)
            plot_kwargs.update(user_kwargs)
            
            # Create the plot
            if x_err_values is not None or y_err_values is not None:
                # Use errorbar
                if color_parameter is not None:
                    # For colored error bars, plot scatter first then error bars
                    scatter = ax.scatter(x_values, y_values, c=color_values, 
                                       s=plot_kwargs.get('s', 50), 
                                       marker=plot_kwargs.get('marker', 'o'),
                                       alpha=plot_kwargs.get('alpha', 0.7),
                                       label=plot_kwargs.get('label'))
                    
                    # Add error bars without markers (use fixed color for error bars)
                    ax.errorbar(x_values, y_values, xerr=x_err_values, yerr=y_err_values, 
                               fmt='none', alpha=0.5, capsize=plot_kwargs.get('capsize', 3),
                               color='gray')
                else:
                    # No color parameter - use regular errorbar
                    # Remove 'c' from plot_kwargs to avoid conflicts
                    errorbar_kwargs = {k: v for k, v in plot_kwargs.items() 
                                     if k not in ['marker', 's', 'c']}
                    ax.errorbar(x_values, y_values, xerr=x_err_values, yerr=y_err_values, 
                               fmt=plot_kwargs.get('marker', 'o'), **errorbar_kwargs)
            else:
                # Use scatter plot
                if color_parameter is not None:
                    scatter = ax.scatter(x_values, y_values, **plot_kwargs)
                else:
                    ax.scatter(x_values, y_values, **plot_kwargs)
            
            # Store data for statistics
            all_stats_data.append({
                'x_values': x_values,
                'y_values': y_values,
                'x_err_values': x_err_values,
                'y_err_values': y_err_values,
                'y_param': y_param
            })
        
        # Add colorbar for any plot with color parameter (single or multi-y)
        if color_parameter is not None and show_colorbar:
            # Use the scatter plot for colorbar (created in either error bar or scatter plot sections)
            if 'scatter' in locals():
                cbar = plt.colorbar(scatter, ax=ax)
                cbar.set_label(color_parameter)
        
        # Add legend for multiple y-parameters (inside the figure)
        if not single_y and color_parameter is None:
            ax.legend(loc=legend_loc, frameon=True, fancybox=True, shadow=True, framealpha=0.9)
        
        # Set axis limits if specified
        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)
        
        # Calculate and display correlation statistics if requested
        if show_stats:
            try:
                from scipy import stats
                
                stats_text_lines = []
                
                for i, stats_data in enumerate(all_stats_data):
                    x_vals = stats_data['x_values']
                    y_vals = stats_data['y_values']
                    x_err_vals = stats_data['x_err_values']
                    y_err_vals = stats_data['y_err_values']
                    y_param = stats_data['y_param']
                    
                    # Remove NaN values for correlation calculations
                    valid_mask = ~(np.isnan(x_vals) | np.isnan(y_vals))
                    x_clean = x_vals[valid_mask]
                    y_clean = y_vals[valid_mask]
                    
                    if len(x_clean) > 2:  # Need at least 3 points for meaningful correlation
                        # Pearson correlation
                        pearson_r, pearson_p = stats.pearsonr(x_clean, y_clean)
                        
                        # Spearman correlation (rank-based, more robust)
                        spearman_rho, spearman_p = stats.spearmanr(x_clean, y_clean)
                        
                        # Prepare statistics text for this y-parameter
                        if not single_y:
                            # Use the final legend label for cleaner display
                            stats_text_lines.append(f'{final_legend_labels[i]}:')
                        
                        stats_text_lines.append(f'N = {len(x_clean)}')
                        stats_text_lines.append(f'r = {pearson_r:.3f}')
                        if pearson_p < 0.001:
                            stats_text_lines[-1] += ' (p<0.001)'
                        elif pearson_p < 0.01:
                            stats_text_lines[-1] += ' (p<0.01)'
                        elif pearson_p < 0.05:
                            stats_text_lines[-1] += ' (p<0.05)'
                        
                        stats_text_lines.append(f'ρ = {spearman_rho:.3f}')
                        if spearman_p < 0.001:
                            stats_text_lines[-1] += ' (p<0.001)'
                        elif spearman_p < 0.01:
                            stats_text_lines[-1] += ' (p<0.01)'
                        elif spearman_p < 0.05:
                            stats_text_lines[-1] += ' (p<0.05)'
                        
                        # Error-weighted correlation if errors are available
                        if x_err_vals is not None and y_err_vals is not None:
                            try:
                                # Calculate error-weighted correlation
                                x_err_clean = x_err_vals[valid_mask] if x_err_vals.ndim == 1 else np.mean(x_err_vals[:, valid_mask], axis=0)
                                y_err_clean = y_err_vals[valid_mask] if y_err_vals.ndim == 1 else np.mean(y_err_vals[:, valid_mask], axis=0)
                                
                                # Avoid division by zero
                                x_err_clean = np.maximum(x_err_clean, 1e-10)
                                y_err_clean = np.maximum(y_err_clean, 1e-10)
                                
                                # Weights inversely proportional to combined error
                                weights = 1.0 / (x_err_clean**2 + y_err_clean**2)
                                weights = weights / np.sum(weights)  # Normalize weights
                                
                                # Weighted means
                                x_weighted_mean = np.sum(weights * x_clean)
                                y_weighted_mean = np.sum(weights * y_clean)
                                
                                # Weighted correlation coefficient
                                numerator = np.sum(weights * (x_clean - x_weighted_mean) * (y_clean - y_weighted_mean))
                                x_var = np.sum(weights * (x_clean - x_weighted_mean)**2)
                                y_var = np.sum(weights * (y_clean - y_weighted_mean)**2)
                                
                                if x_var > 0 and y_var > 0:
                                    weighted_r = numerator / np.sqrt(x_var * y_var)
                                    stats_text_lines.append(f'r_w = {weighted_r:.3f}')
                                
                            except Exception as e:
                                logger.debug(f"Could not calculate weighted correlation for {y_param}: {e}")
                        
                        # Add separator for multiple y-parameters
                        if not single_y and i < len(all_stats_data) - 1:
                            stats_text_lines.append('')
                    
                    else:
                        logger.warning(f"Not enough valid data points for correlation statistics for {y_param}")
                
                # Display statistics if we have any
                if stats_text_lines:
                    stats_text = '\n'.join(stats_text_lines)
                    
                    # Position the text box in the upper left corner
                    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, 
                           verticalalignment='top', horizontalalignment='left',
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                           fontsize=11 if not single_y else 12, family='monospace')
                    
            except ImportError:
                logger.warning("scipy not available for correlation statistics")
            except Exception as e:
                logger.warning(f"Could not calculate correlation statistics: {e}")
        
        # Add diagonal line if requested
        if show_diagonal:
            # Get the current axis limits after the scatter plot
            xlim_current = ax.get_xlim()
            ylim_current = ax.get_ylim()
            
            # Calculate the range for the diagonal line based on the overlapping region
            line_min = max(xlim_current[0], ylim_current[0])
            line_max = min(xlim_current[1], ylim_current[1])
            
            # Only draw diagonal if there's an overlapping range
            if line_min < line_max:
                ax.plot([line_min, line_max], [line_min, line_max], 'k--', alpha=0.8, linewidth=1, label='y=x')
                # Only add legend if we don't already have one from multiple y-parameters
                if single_y or color_parameter is not None:
                    ax.legend()
        
        # Set labels
        ax.set_xlabel(xlabel if xlabel is not None else x_label_default, fontsize=16)
        
        if ylabel is not None:
            ax.set_ylabel(ylabel, fontsize=16)
        elif single_y:
            ax.set_ylabel(y_parameters[0], fontsize=16)
        else:
            # For multiple y-parameters, use the extracted common part (unless custom labels override)
            if custom_legend_labels is None:
                ax.set_ylabel(common_ylabel, fontsize=16)
            else:
                # With custom legend labels, use a generic ylabel
                ax.set_ylabel('Multiple Parameters', fontsize=16)
        
        # Set title
        if title is not None:
            ax.set_title(title, fontsize=16)
        
        # Make axes equal if values are in similar range (only if show_diagonal is True and single y-parameter)
        if show_diagonal and single_y:
            ax.set_aspect('equal', adjustable='box')
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Tight layout (legend is now inside, so no need for extra margin adjustment)
        plt.tight_layout()
        
        # Save if requested
        if output_file:
            fig.savefig(output_file, dpi=300, bbox_inches='tight')
            logger.info(f"Saved scatter plot to: {output_file}")
        
        # Show if requested
        if show:
            plt.show()
        
        total_points = len(x_values) * len(y_parameters)
        if single_y:
            logger.info(f"Created scatter plot: {len(x_values)} points")
        else:
            logger.info(f"Created multi-parameter scatter plot: {len(x_values)} points × {len(y_parameters)} parameters = {total_points} total points")
        
        return fig
    def compute_parameter_correlations(self, params: Optional[List[str]] = None,
                                     object_ids: Optional[Union[str, List[str]]] = None) -> 'numpy.ndarray':
        """
        Compute parameter correlation matrix.

        Parameters
        ----------
        params : List[str], optional
            Parameters to compute correlations for. If None, uses free parameters.
        object_ids : List[str], optional
            Object IDs to include. If None, uses current scope.

        Returns
        -------
        numpy.ndarray
            Correlation matrix
        """
        import numpy as np

        # Use free parameters if none specified
        if params is None:
            params = self.get_free_parameters()

        if not params:
            raise ValueError("No parameters available for correlation computation")

        # Get HDF5 table with appropriate filtering
        hdf5_table = self.load_hdf5_results(filter_snr=False, min_snr=0.0)

        # Apply object filtering if specified
        if object_ids is not None:
            object_mask = [str(obj_id) in [str(x) for x in object_ids] for obj_id in hdf5_table['ID']]
            if any(object_mask):
                hdf5_table = hdf5_table[object_mask]
            else:
                raise ValueError("None of the specified objects found in data")

        # Extract parameter data
        param_data = []
        available_params = []

        for param in params:
            # Find columns that match this parameter (handle statistical estimates)
            matching_cols = [col for col in hdf5_table.colnames if col.startswith(param)]

            if matching_cols:
                # Use the first matching column (could be enhanced to use specific estimates)
                col_name = matching_cols[0]
                param_data.append(hdf5_table[col_name])
                available_params.append(param)
            else:
                logger.warning(f"Parameter '{param}' not found in HDF5 data, skipping")

        if len(available_params) < 2:
            raise ValueError(f"Need at least 2 parameters for correlation computation, found {len(available_params)}")

        # Convert to numpy array and compute correlation matrix
        data_matrix = np.column_stack(param_data)
        correlation_matrix = np.corrcoef(data_matrix.T)

        logger.info(f"Computed correlation matrix for {len(available_params)} parameters across {len(hdf5_table)} objects")

        return correlation_matrix
    def get_parameter_statistics(self, params: Optional[List[str]] = None,
                               object_ids: Optional[List[str]] = None) -> Dict[str, Dict[str, float]]:
        """
        Get parameter statistics (mean, std, percentiles, etc.).

        Parameters
        ----------
        params : List[str], optional
            Parameters to compute statistics for. If None, uses free parameters.
        object_ids : List[str], optional
            Object IDs to include. If None, uses current scope.

        Returns
        -------
        Dict[str, Dict[str, float]]
            Dictionary with parameter names as keys and statistics as values
        """
        import numpy as np

        # Use free parameters if none specified
        if params is None:
            params = self.get_free_parameters()

        if not params:
            raise ValueError("No parameters available for statistics computation")

        # Get HDF5 table with appropriate filtering
        hdf5_table = self.load_hdf5_results(filter_snr=False, min_snr=0.0)

        # Apply object filtering if specified
        if object_ids is not None:
            object_mask = [str(obj_id) in [str(x) for x in object_ids] for obj_id in hdf5_table['ID']]
            if any(object_mask):
                hdf5_table = hdf5_table[object_mask]
            else:
                raise ValueError("None of the specified objects found in data")

        # Compute statistics for each parameter
        statistics = {}

        for param in params:
            # Find columns that match this parameter (handle statistical estimates)
            matching_cols = [col for col in hdf5_table.colnames if col.startswith(param)]

            if matching_cols:
                # Use the first matching column (could be enhanced to use specific estimates)
                col_name = matching_cols[0]
                param_values = np.array(hdf5_table[col_name])

                # Remove any NaN or infinite values
                valid_mask = np.isfinite(param_values)
                if np.any(valid_mask):
                    valid_values = param_values[valid_mask]

                    param_stats = {
                        'count': len(valid_values),
                        'mean': float(np.mean(valid_values)),
                        'std': float(np.std(valid_values)),
                        'min': float(np.min(valid_values)),
                        'max': float(np.max(valid_values)),
                        'median': float(np.median(valid_values)),
                        'p16': float(np.percentile(valid_values, 16)),
                        'p84': float(np.percentile(valid_values, 84)),
                        'p25': float(np.percentile(valid_values, 25)),
                        'p75': float(np.percentile(valid_values, 75))
                    }

                    statistics[param] = param_stats
                else:
                    logger.warning(f"No valid values found for parameter '{param}'")
            else:
                logger.warning(f"Parameter '{param}' not found in HDF5 data, skipping")

        logger.info(f"Computed statistics for {len(statistics)} parameters across {len(hdf5_table)} objects")

        return statistics
