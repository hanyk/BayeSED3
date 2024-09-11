import sys
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

c = 2.9979246e+14 #um/s

def plot_model_data(hdul, models, new_model_names):
    if new_model_names is None:
        new_model_names = models
    for model, new_name in zip(models, new_model_names):
        obs_phot = hdul[model].data
        mask = obs_phot['flux'] > 0
        plt.plot(obs_phot[mask]['wavelength_rest'], c*obs_phot[mask]['wavelength_obs']**-1*obs_phot[mask]['flux'], label=new_name)

def plot_photometry_data(hdul, ax):
    if 'obs:phot' in hdul and '--no_photometry_fit' not in hdul[0].header['RUN_ARGUMENTS']:
        obs_phot = hdul['obs:phot'].data
        obs_phot = obs_phot[obs_phot['iselected'] == 1]
        ax.errorbar(obs_phot['wavelength_rest_center'], obs_phot['flux_obs'], obs_phot['flux_obs_err'], fmt='x', color='red', label='phot:obs')
        ax.errorbar(obs_phot['wavelength_rest_center'], obs_phot['flux_model'], obs_phot['flux_model_err'], fmt='+', color='blue', label='phot:mod')

def plot_spectra_data(hdul, bands, ax):
    if 'obs:spec' in hdul and '--no_spectra_fit' not in hdul[0].header['RUN_ARGUMENTS']:
        obs_spec = hdul['obs:spec'].data
        # ax.axhline(y=0, color='red', linestyle='--')
        emission_lines = [
            (0.65646140, 'H_alpha', 'red'), (0.48626830, 'H_beta', 'green'),
            (0.49602949, '[O_III]4959', 'blue'), (0.50082397, '[O_III]5007', 'cyan'),
            (0.65498590, '[N_II]6548', 'magenta'), (0.65852685, '[N_II]6583', 'purple'),
            (0.37270917, '[O_II]3725', 'yellow'), (0.37298754, '[O_II]3727', 'orange'),
            (0.2798, 'MgII2798', 'brown')
        ]
        for line, label, color in emission_lines:
            ax.axvline(x=line, color=color, linestyle='--', label=label, linewidth=0.5)
        for band in bands:
            band_data = obs_spec[obs_spec['iband'] == band]
            ax.plot(band_data['wavelength_rest'], c*band_data['wavelength_obs']**-1*band_data['spectra_obs'], label=f'spec:obs_{band}')
            ax.plot(band_data['wavelength_rest'], c*band_data['wavelength_obs']**-1*band_data['spectra_mod'], label=f'spec:mod_{band}')
            ax.set_xlim(1.0*np.min(band_data['wavelength_rest']),1.0*np.max(band_data['wavelength_rest']))
            ax.set_ylim(0.1*np.min(c*band_data['wavelength_obs']**-1*band_data['spectra_obs']), 1.1*np.max(c*band_data['wavelength_obs']**-1*band_data['spectra_obs']))

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

def read_filter_info(filter_names_file):
    filter_info = []
    try:
        with open(filter_names_file, 'r') as file:
            for line in file:
                if not line.startswith('#'):
                    parts = line.split()
                    if len(parts) > 3 and parts[0] == '1' and parts[1] == '1':
                        filter_info.append((int(parts[2]), parts[3]))
    except Exception as e:
        print(f"Error reading filter names file: {e}")
    return filter_info

def main(fits_file, filter_file=None, filter_names_file=None):
    hdul = fits.open(fits_file)

    if filter_file and filter_names_file:
        filter_info = read_filter_info(filter_names_file)

    fig, ax1 = plt.subplots()

    # if 'model:total' in hdul:
        # total = hdul['model:total'].data
        # mask = total['flux'] > 0
        # ax1.plot(total[mask]['wavelength_rest'], c*total[mask]['wavelength_obs']**-1*total[mask]['flux'], label='Total',color='black')

    models = [h.name for h in hdul if h.name.startswith('model:') and h.name != 'model:total']
    new_model_names = None
    # new_model_names = ['stellar+Nebular','QSO1']  # Update this list with the new model names
    if len(models)==4:
        new_model_names = ['stellar+Nebular','AGN-Disk','AGN-BLR','AGN-NLR']  # Update this list with the new model names
    if len(models)==5:
        new_model_names = ['stellar+Nebular','AGN-Disk','AGN-BLR','AGN-FeII','AGN-NLR']  # Update this list with the new model names
    plot_model_data(hdul, models, new_model_names)
    plot_photometry_data(hdul, ax1)

    if 'obs:spec' in hdul:
        bands = np.unique(hdul['obs:spec'].data['iband'])
        plot_spectra_data(hdul, bands, ax1)

    if filter_file and filter_names_file:
        ax2 = ax1.twinx()
        scale_factor = 0.2 * ax1.get_ylim()[1]
        plot_filter_response(filter_file, filter_info, hdul, ax2, scale_factor)

        # Adjust the y-axis limits of the secondary axis to use only the lower half of the panel
        ax1_ylim = ax1.get_ylim()
        ax2.set_ylim(0, ax1_ylim[1] / 2)
        ax2.yaxis.set_ticks([])  # Hide the y-axis ticks
        ax2.yaxis.set_ticklabels([])  # Hide the y-axis labels
        ax2.legend(loc='best')

    xmin2_nd = float(hdul[0].header['XMIN^2/ND'])
    title = (f"ID={hdul[0].header['ID']}\nSNR={hdul[0].header['SNR']}, "
             f"z_best={hdul[0].header['Z_{MAL}']}, Xmin^2/Nd={xmin2_nd:.2f}")

    # ax1.set_xlim(0.2, 0.9)
    # ax1.set_ylim(1, 2e3)
    ax1.set_xlabel(r'$\lambda/\rm \mu m$')
    ax1.set_ylabel(r'$\nu F\nu/[\rm \mu Jy*Hz]$')
    plt.title(title)
    ax1.legend(loc='best', ncol=2)
    plt.savefig(f"{fits_file}.png")
    plt.show()

if __name__ == "__main__":
    fits_file = sys.argv[1]
    print(fits_file)
    if len(sys.argv) > 3:
        filter_file = sys.argv[2]
        filter_names_file = sys.argv[3]
    else:
        filter_file = None
        filter_names_file = None
    main(fits_file, filter_file, filter_names_file)
