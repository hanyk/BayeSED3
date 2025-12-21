import numpy as np
import bagpipes as pipes
import argparse
import os
import matplotlib.pyplot as plt
import time

from astropy.io import fits
from astropy.table import Table, join, hstack, vstack

# c = 2.9979246e+14  # um/s
c = 2.9979246e+18 #angstrom/s

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='A script to run BayeSED3 format input catalog file with Bagpipes')
    parser.add_argument('--cat', '-c', type=str, required=True,
                       help='Path to the catalog file')
    parser.add_argument('--ids', '-i', type=str, nargs='+',
                       help='List of galaxy IDs to fit (space-separated). If not specified, all IDs from catalog will be used.')
    parser.add_argument('--idfile', '-f', type=str,
                       help='Path to file containing galaxy IDs (one per line)')

    args = parser.parse_args()

    return args

def load_ids_from_file(filename):
    """Load galaxy IDs from a text file, one ID per line."""
    if not os.path.exists(filename):
        raise FileNotFoundError(f"ID file not found: {filename}")

    with open(filename, 'r') as f:
        ids = [line.strip() for line in f if line.strip() and not line.startswith('#')]

    return ids

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

    plt.close(fig)
    return fig, (ax_main, ax_resid)

def main():
    # Parse command line arguments
    args = parse_arguments()

    # Check if catalog file exists
    if not os.path.exists(args.cat):
        raise FileNotFoundError(f"Catalog file not found: {args.cat}")

    # Load catalog
    cat = Table.read(args.cat, format='ascii')
    head = cat.meta['comments'][0].split()
    cat_name = head[0]
    Nphot = int(head[1])
    Nother = int(head[2])
    Nspec = int(head[3])

    # Determine which IDs to use
    if args.ids:
        IDs = args.ids
    elif args.idfile:
        IDs = load_ids_from_file(args.idfile)
    else:
        IDs = cat['ID'].__array__().astype(str)

    print(f"Using catalog: {args.cat}")
    print(f"Fitting {len(IDs)} galaxies: {IDs}")

    # Initialize resolution variable at main function level
    resolution = None

    def load_bayesed_input(ID):
        """ Load photometry and/or spectrum from the input file of BayeSED. """
        nonlocal resolution  # Allow access to resolution from outer scope

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

    # goodss_filt_list = np.loadtxt("filters/goodss_filt_list.txt", dtype="str")
    # spectrum,photometry=load_bayesed_input('10524881')
    # IDs=cat['ID'].__array__().astype(str)
    # IDs=np.array(['5494348','11184100'])
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
    fit_instructions["R_curve"] = resolution

    noise = {}
    noise["type"] = "white_scaled"
    noise["scaling"] = (1., 1.5)
    noise["scaling_prior"] = "log_10"
    fit_instructions["noise"] = noise

    # plt.plot(fit_instructions["R_curve"][:, 0], fit_instructions["R_curve"][:, 1])
    # plt.xlabel("Wavelength/ \AA")
    # plt.ylabel("Spectral resolving power")
    # plt.show()

    for ID in IDs:
        # Create individual galaxy object
        galaxy = pipes.galaxy(ID, load_bayesed_input, photometry_exists=False)

        # Fit individual galaxy with timing
        fit = pipes.fit(galaxy, fit_instructions, run=f"{cat_name}_{ID}")
        t0 = time.time()
        fit.fit(verbose=True, sampler='nautilus', n_live=400, pool=10)
        runtime_s = time.time() - t0
        # fig = fit.plot_spectrum_posterior(save=True, show=True)
        # fig = fit.plot_sfh_posterior(save=True, show=True)
        # fig = fit.plot_corner(save=True, show=True)
        plot_spectrum_posterior_with_residuals(fit, ID, cat_name, runtime_s=runtime_s)

    # fit_cat = pipes.fit_catalogue(IDs, fit_instructions, load_bayesed_input, photometry_exists=False, run=cat_name, make_plots=False,n_posterior=500)
    # # fit_cat.fit(verbose=True,pool=10,n_live=400)
    # fit_cat.fit(verbose=True,mpi_serial=True,n_live=400)

if __name__ == "__main__":
    main()
