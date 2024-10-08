import sys
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

c = 2.9979246e+14  # um/s

def plot_model_data(hdul, ax, models, new_model_names=None):
    if new_model_names is None:
        new_model_names = models
    for model, new_name in zip(models, new_model_names):
        obs_phot = hdul[model].data
        mask = obs_phot['flux'] > 0
        ax.plot(obs_phot[mask]['wavelength_rest'], c*obs_phot[mask]['wavelength_obs']**-1*obs_phot[mask]['flux'], label=new_name)

def plot_photometry_data(hdul, ax):
    if 'obs:phot' in hdul and '--no_photometry_fit' not in hdul[0].header['RUN_ARGUMENTS']:
        obs_phot = hdul['obs:phot'].data
        obs_phot = obs_phot[obs_phot['iselected'] == 1]
        ax.errorbar(obs_phot['wavelength_rest_center'], obs_phot['flux_obs'], obs_phot['flux_obs_err'], fmt='x', color='red', label='phot:obs')
        ax.errorbar(obs_phot['wavelength_rest_center'], obs_phot['flux_model'], obs_phot['flux_model_err'], fmt='+', color='blue', label='phot:mod')

def plot_spectra_data(hdul, ax):
    if 'obs:spec' in hdul and '--no_spectra_fit' not in hdul[0].header['RUN_ARGUMENTS']:
        obs_spec = hdul['obs:spec'].data
        emission_lines = [
            (0.65646140, 'H_alpha', 'red'), (0.48626830, 'H_beta', 'green'),
            (0.49602949, '[O_III]4959', 'blue'), (0.50082397, '[O_III]5007', 'cyan'),
            (0.65498590, '[N_II]6548', 'magenta'), (0.65852685, '[N_II]6583', 'purple'),
            (0.37270917, '[O_II]3725', 'yellow'), (0.37298754, '[O_II]3727', 'orange'),
            (0.2798, 'MgII2798', 'brown')
        ]
        for line, label, color in emission_lines:
            ax.axvline(x=line, color=color, linestyle='--', label=label, linewidth=0.5)
        
        bands = np.unique(obs_spec['iband'])
        for band in bands:
            band_data = obs_spec[obs_spec['iband'] == band]
            ax.plot(band_data['wavelength_rest'], c*band_data['wavelength_obs']**-1*band_data['spectra_obs'], label=f'spec:obs_{band}')
            ax.plot(band_data['wavelength_rest'], c*band_data['wavelength_obs']**-1*band_data['spectra_mod'], label=f'spec:mod_{band}')

def plot_filter_response(filter_file, filter_info, hdul, ax2, scale_factor):
    try:
        z = float(hdul[0].header['Z_{MAL}'])
        colors = plt.cm.tab20.colors
        color_idx = 0

        for filter_id, filter_name in filter_info:
            wavelengths, responses = [], []
            current_filter_id = -1

            with open(filter_file, 'r') as file:
                for line in file:
                    line = line.strip()
                    if line.startswith('#'):
                        current_filter_id += 1
                        if current_filter_id == filter_id:
                            print(f"Processing filter {current_filter_id} with name {filter_name}")
                        else:
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
                ax2.plot(np.array(wavelengths)/(1+z), responses, linestyle='--', color=colors[color_idx % len(colors)], label=filter_name)
                color_idx += 1
    except Exception as e:
        print(f"Error reading filter file: {e}")

def read_filter_info(filter_selection_file):
    filter_info = []
    try:
        with open(filter_selection_file, 'r') as file:
            for line in file:
                if not line.startswith('#'):
                    parts = line.split()
                    if len(parts) > 3 and parts[0] == '1' and parts[1] == '1':
                        filter_info.append((int(parts[2]), parts[3]))
    except Exception as e:
        print(f"Error reading filter selection file: {e}")
    return filter_info

def get_data_ranges(hdul, models):
    model_x, model_y = [], []
    obs_x, obs_y = [], []
    
    # Collect model data
    for model in models:
        data = hdul[model].data
        mask = data['flux'] > 0
        model_x.extend(data[mask]['wavelength_rest'])
        model_y.extend(c * data[mask]['wavelength_obs']**-1 * data[mask]['flux'])
    
    # Collect observed data
    if 'obs:phot' in hdul:
        obs_phot = hdul['obs:phot'].data
        obs_phot = obs_phot[obs_phot['iselected'] == 1]
        obs_x.extend(obs_phot['wavelength_rest_center'])
        obs_y.extend(obs_phot['flux_obs'])
    
    if 'obs:spec' in hdul:
        obs_spec = hdul['obs:spec'].data
        obs_x.extend(obs_spec['wavelength_rest'])
        spec_flux = c * obs_spec['wavelength_obs']**-1 * obs_spec['spectra_obs']
        obs_y.extend(spec_flux)
    
    # Convert to numpy arrays and remove non-positive values
    model_x, model_y = np.array(model_x), np.array(model_y)
    obs_x, obs_y = np.array(obs_x), np.array(obs_y)
    
    mask_model = (model_x > 0) & (model_y > 0)
    mask_obs = (obs_x > 0) & (obs_y > 0)
    
    model_x, model_y = model_x[mask_model], model_y[mask_model]
    obs_x, obs_y = obs_x[mask_obs], obs_y[mask_obs]
    
    # Calculate percentile ranges
    if len(model_x) > 0:
        model_x_min, model_x_max = np.percentile(model_x, [1, 99])
        model_y_min, model_y_max = np.percentile(model_y, [1, 99])
    else:
        model_x_min, model_x_max, model_y_min, model_y_max = np.inf, -np.inf, np.inf, -np.inf
    
    if len(obs_x) > 0:
        obs_x_min, obs_x_max = np.percentile(obs_x, [1, 99])
        obs_y_min, obs_y_max = np.percentile(obs_y, [1, 99])
    else:
        obs_x_min, obs_x_max, obs_y_min, obs_y_max = np.inf, -np.inf, np.inf, -np.inf
    
    # Weighted combination of ranges (heavily favoring observed data)
    weight_obs = 0.8  # Increased weight for observed data
    weight_model = 1 - weight_obs
    
    x_min = min(model_x_min, obs_x_min)
    x_max = max(model_x_max, obs_x_max)
    y_min = min(model_y_min, obs_y_min)
    y_max = max(model_y_max, obs_y_max)
    
    x_range = x_max - x_min
    y_range = y_max - y_min
    
    x_center = weight_obs * (obs_x_min + obs_x_max) / 2 + weight_model * (model_x_min + model_x_max) / 2
    y_center = weight_obs * (obs_y_min + obs_y_max) / 2 + weight_model * (model_y_min + model_y_max) / 2
    
    # Calculate the range based more on observed data
    x_half_range = max(weight_obs * (obs_x_max - obs_x_min) / 2, weight_model * (model_x_max - model_x_min) / 2)
    y_half_range = max(weight_obs * (obs_y_max - obs_y_min) / 2, weight_model * (model_y_max - model_y_min) / 2)
    
    x_min = max(x_min, x_center - x_half_range)
    x_max = min(x_max, x_center + x_half_range)
    y_min = max(y_min, y_center - y_half_range)
    y_max = min(y_max, y_center + y_half_range)
    
    # Ensure all observed data points are included
    if len(obs_x) > 0:
        x_min = min(x_min, np.min(obs_x))
        x_max = max(x_max, np.max(obs_x))
        y_min = min(y_min, np.min(obs_y))
        y_max = max(y_max, np.max(obs_y))
    
    # Add some padding to the ranges
    x_range = x_max - x_min
    y_range = y_max - y_min
    x_min -= 0.05 * x_range
    x_max += 0.05 * x_range
    y_min = max(y_min * 0.5, y_min - 0.05 * y_range)
    y_max *= 1.1
    
    return x_min, x_max, y_min, y_max

def auto_legend_layout(ax, fig):
    # Get the current positions of the plot elements
    bbox = ax.get_position()
    plot_width = bbox.width
    plot_height = bbox.height

    # Get all legend handles and labels
    handles, labels = ax.get_legend_handles_labels()
    n_items = len(handles)

    if n_items == 0:
        return  # No legend items, so return

    # Estimate the width and height of a single legend item
    test_legend = ax.legend(handles[:1], labels[:1], loc='center')
    fig.canvas.draw()  # This is necessary to update the legend size
    legend_width = test_legend.get_window_extent().width / fig.dpi
    legend_height = test_legend.get_window_extent().height / fig.dpi
    test_legend.remove()

    # Calculate the maximum number of columns that fit in the plot width
    max_cols = max(1, int(plot_width / legend_width))

    # Calculate the number of rows needed
    n_rows = -(-n_items // max_cols)  # Ceiling division

    # Adjust columns if too many rows
    while n_rows > 5 and max_cols > 1:
        max_cols -= 1
        n_rows = -(-n_items // max_cols)

    # Create the legend with the calculated number of columns
    legend = ax.legend(loc='best', ncol=max_cols, fontsize='small')

    # Adjust legend position if it overlaps with the plot
    fig.canvas.draw()
    legend_bbox = legend.get_window_extent().transformed(fig.transFigure.inverted())
    if legend_bbox.y0 < bbox.y0 or legend_bbox.y1 > bbox.y1:
        # If legend overlaps vertically, place it below the plot
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=max_cols, fontsize='small')

    return legend

def main(fits_file, filter_file=None, filter_selection_file=None, use_log_scale=False, use_short_names=False):
    hdul = fits.open(fits_file)

    fig, ax1 = plt.subplots(figsize=(12, 8))

    models = [h.name for h in hdul if h.name.startswith('model:') and h.name != 'model:total']
    
    new_model_names = None
    if use_short_names:
        # Auto-generate shorter model names
        new_model_names = []
        for model in models:
            if 'stellar' in model.lower():
                new_model_names.append('Stellar+Neb')
            elif 'agn' in model.lower():
                if 'disk' in model.lower():
                    new_model_names.append('AGN-Disk')
                elif 'blr' in model.lower():
                    new_model_names.append('AGN-BLR')
                elif 'nlr' in model.lower():
                    new_model_names.append('AGN-NLR')
                elif 'feii' in model.lower():
                    new_model_names.append('AGN-FeII')
                else:
                    new_model_names.append('AGN-Other')
            else:
                new_model_names.append(model.split(':')[-1][:10])  # Use last 10 characters after ':' as fallback

    plot_model_data(hdul, ax1, models, new_model_names)
    plot_photometry_data(hdul, ax1)
    plot_spectra_data(hdul, ax1)

    x_min, x_max, y_min, y_max = get_data_ranges(hdul, models)
    
    if use_log_scale:
        ax1.set_xscale('log')
        ax1.set_yscale('log')
        ax1.set_xlim(x_min, x_max)
        ax1.set_ylim(y_min, y_max)
    else:
        ax1.set_xlim(x_min, x_max)
        ax1.set_ylim(y_min, y_max)

    if filter_file and filter_selection_file:
        filter_info = read_filter_info(filter_selection_file)
        ax2 = ax1.twinx()
        scale_factor = 0.2 * y_max
        plot_filter_response(filter_file, filter_info, hdul, ax2, scale_factor)

        ax2.set_ylim(0, y_max / 2)
        ax2.yaxis.set_ticks([])
        ax2.yaxis.set_ticklabels([])
        ax2.legend(loc='upper right', bbox_to_anchor=(1, 0.9))

    xmin2_nd = float(hdul[0].header['XMIN^2/ND'])
    title = (f"ID={hdul[0].header['ID']}\nSNR={hdul[0].header['SNR']}, "
             f"z_best={hdul[0].header['Z_{MAL}']}, Xmin^2/Nd={xmin2_nd:.2f}")

    ax1.set_xlabel(r'$\lambda/\rm \mu m$')
    ax1.set_ylabel(r'$\nu F\nu/[\rm \mu Jy*Hz]$')
    plt.title(title)
    
    # Use the auto_legend_layout function
    auto_legend_layout(ax1, fig)
    
    plt.tight_layout()
    plt.savefig(f"{fits_file}.png", dpi=300, bbox_inches='tight')
    plt.show()

def show_help():
    help_message = """
    Usage: python plot_bestfit_unified.py <fits_file> [filter_file] [filter_selection_file] [--log] [--short-names]

    Arguments:
    fits_file           : Path to the FITS file to be plotted (required)
    filter_file         : Path to the filter file (optional)
    filter_selection_file : Path to the filter selection file (optional)
    --log               : Use logarithmic scale for both axes (optional)
    --short-names       : Use shorter names for model components (optional)

    Example:
    python plot_bestfit_unified.py data.fits filters.txt filter_selection.txt --log --short-names

    This script plots spectral energy distributions from FITS files, including model data,
    photometry data, and spectra data. It can also overlay filter responses if the relevant
    files are provided.
    """
    print(help_message)

if __name__ == "__main__":
    if len(sys.argv) == 1 or '--help' in sys.argv or '-h' in sys.argv:
        show_help()
    else:
        fits_file = sys.argv[1]
        print(fits_file)
        filter_file = sys.argv[2] if len(sys.argv) > 2 else None
        filter_selection_file = sys.argv[3] if len(sys.argv) > 3 else None
        use_log_scale = '--log' in sys.argv
        use_short_names = '--short-names' in sys.argv
        main(fits_file, filter_file, filter_selection_file, use_log_scale, use_short_names)