from astropy.table import Table
import matplotlib.pyplot as plt
from astropy.io import fits
import sys
import numpy as np
import argparse

c = 2.9979246e+18 #angstrom/s

# Increase global font sizes for better readability
plt.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'legend.fontsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12
})

# Parse command line arguments
parser = argparse.ArgumentParser(description='Plot best-fit spectrum and residuals.')
parser.add_argument('file', help='Input FITS file produced by the fitting pipeline')
parser.add_argument('--code-name', default='', help='Name of the code used to produce the results (displayed at plot center)')
args = parser.parse_args()

# Create figure with subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), height_ratios=[3, 1], sharex=True)

file=args.file
print(file)
hdul = fits.open(file)
if "QUIESCENT" in file: galaxy_type='QUIESCENT'
if "STARFORMING" in file: galaxy_type='STARFORMING'

# Read number of free parameters from header
np_free = int(hdul[0].header['NP'])

# Accumulators for overall chi-square across photometry and spectroscopy
chi2_total = 0.0
n_total = 0

c1 = 2.9979246e+14 #um/s
if hdul.__contains__('model:total'):
    total=hdul['model:total'].data
    mask=total['flux']>0
    # plt.loglog(total[mask]['wavelength_obs'],total[mask]['flux'],label='Total')
    xmin=min(total['wavelength_obs'])
    xmax=max(total['wavelength_obs'])
    ymin=min(total[mask]['flux'])
    ymax=max(total[mask]['flux'])

models=[hdul[i].name for i in range(0,len(hdul)) if 'model:' in hdul[i].name and hdul[i].name!='model:total']
for model in models[0:1]:
    obs_phot=hdul[model].data
    mask=obs_phot['flux']>0
    # plt.loglog(obs_phot[mask]['wavelength_obs'],obs_phot[mask]['flux'],label=model.replace('model:',''))
    # if len(models)==1:
        # xmin=min(obs_phot['wavelength_obs'])
        # xmax=max(obs_phot['wavelength_obs'])
        # ymin=min(obs_phot[mask]['flux'])
        # ymax=max(obs_phot[mask]['flux'])

if hdul.__contains__('obs:phot') and '--no_photometry_fit' not in hdul[0].header['RUN_ARGUMENTS']:
    obs_phot=hdul['obs:phot'].data
    obs_phot=obs_phot[obs_phot['iselected']==1]
    ax1.errorbar(obs_phot['wavelength_obs_center'],obs_phot['flux_obs'],obs_phot['flux_obs_err'],fmt='x',color='red',label='phot:obs')
    ax1.errorbar(obs_phot['wavelength_obs_center'],obs_phot['flux_model'],obs_phot['flux_model_err'],fmt='+',color='blue',label='phot:mod')

    # Calculate residuals for photometry
    valid_phot = (obs_phot['flux_obs_err'] > 0)
    phot_residuals = (obs_phot['flux_obs'][valid_phot] - obs_phot['flux_model'][valid_phot]) / obs_phot['flux_obs_err'][valid_phot]
    chi2_total += np.sum(phot_residuals**2)
    n_total += phot_residuals.size if hasattr(phot_residuals, 'size') else len(phot_residuals)
    ax2.errorbar(obs_phot['wavelength_obs_center'], phot_residuals, 1.0, fmt='o', color='red', alpha=0.7, markersize=3)
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax2.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
    ax2.axhline(y=-1, color='gray', linestyle='--', alpha=0.5)

    # if len(models)>1:
    xmin=min(obs_phot['wavelength_obs_center'])
    xmax=max(obs_phot['wavelength_obs_center'])
    ymin=min(obs_phot['flux_obs'][obs_phot['flux_obs']>0])
    ymax=max(obs_phot['flux_obs'][obs_phot['flux_obs']>0])

if hdul.__contains__('obs:spec') and '--no_spectra_fit' not in hdul[0].header['RUN_ARGUMENTS']:
    obs_spec=hdul['obs:spec'].data
    waves=obs_spec['wavelength_obs']*1e4 # um to angstrom
    obs_spec['spectra_obs']=1e17*(c*waves**-1*obs_spec['spectra_obs']*1e-29)/waves # uJy*Hz = 1e-29 erg/s/cm^2
    # mask=obs_spec['spectra_obs']<0
    # obs_spec['spectra_obs'][mask]=0
    obs_spec['spectra_obs_err']=1e17*(c*waves**-1*obs_spec['spectra_obs_err']*1e-29)/waves
    obs_spec['spectra_mod']=1e17*(c*waves**-1*obs_spec['spectra_mod']*1e-29)/waves
    if hdul.__contains__('obs:phot'):
        obs_phot=hdul['obs:phot'].data
        obs_phot=obs_phot[obs_phot['iselected']==1]
        bands_name=obs_phot['band']
    bands=np.unique(obs_spec['iband'])
    colors = ['tab:purple', 'tab:green', 'tab:red']
    ls = [':', '-', '--']
    chi2_per_band = {}
    for idx, i in enumerate(bands[0:4]):
        mask = (obs_spec['iband']==i)
        # mask = (obs_spec['iband']==i)*(obs_spec['spectra_obs']>0)*(obs_spec['spectra_obs_err']>0)
        # mask=(obs_spec['iband']==i)*(obs_spec['spectra_obs']>0)*(obs_spec['spectra_obs']/obs_spec['spectra_obs_err0']>0)
        ax1.plot(
            obs_spec[mask]['wavelength_obs'],
            obs_spec[mask]['spectra_obs'],
            label='obs:'+bands_name[i],
            color=colors[idx % len(colors)]
        )
        # Compute per-band chi^2
        residuals_band = (obs_spec[mask]['spectra_obs'] - obs_spec[mask]['spectra_mod']) / obs_spec[mask]['spectra_obs_err']
        chi2_band = np.sum(residuals_band**2)
        n_band = residuals_band.size if hasattr(residuals_band, 'size') else len(residuals_band)
        chi2_norm = chi2_band / n_band if n_band > 0 else np.nan
        dof_band = n_band - np_free
        chi2_red = chi2_band / dof_band if dof_band > 0 else np.nan
        chi2_per_band[bands_name[i]] = (chi2_norm, n_band, chi2_red, dof_band)
        # accumulate overall chi2 and N
        chi2_total += chi2_band
        n_total += n_band
        ax1.plot(
            obs_spec[mask]['wavelength_obs'],
            obs_spec[mask]['spectra_mod'],
            label='spec:mod_'+bands_name[i]+f' ($\\chi^2_\\nu$={chi2_red:.2f})',
            linestyle=ls[idx % len(ls)],
            color='tab:blue'
        )

        # Calculate residuals for spectroscopy
        spec_residuals = residuals_band
        ax2.plot(
            obs_spec[mask]['wavelength_obs'],
            spec_residuals,
            color=colors[idx % len(colors)]
        )

        # plt.plot(obs_spec[mask]['wavelength_obs'],obs_spec[mask]['spectra_obs']-obs_spec[mask]['spectra_mod'],label='spec:diff_'+str(i))
        # plt.plot(obs_spec[mask]['wavelength_obs'],c1*obs_spec[mask]['wavelength_obs']**-2*obs_spec[mask]['spectra_obs'],label='spec:obs_'+str(i))
        # plt.plot(obs_spec[mask]['wavelength_obs'],c1*obs_spec[mask]['wavelength_obs']**-2*obs_spec[mask]['spectra_mod'],label='spec:mod_'+str(i))
        # plt.plot(obs_spec[mask]['wavelength_obs'],c1*obs_spec[mask]['wavelength_obs']**-2*obs_spec[mask]['spectra_obs']-c1*obs_spec[mask=i]['wavelength_obs']**-2*obs_spec[mask]['spectra_mod'],label='spec:diff_'+str(i))
    # plt.axhline(y=0, color='red', linestyle='--')
    # plt.axvline(x=0.65646140, color='red', linestyle='--',label='Hα',linewidth=0.5)
    # plt.axvline(x=0.48626830, color='green', linestyle='--',label='Hβ',linewidth=0.5)
    # plt.axvline(x=0.49602949, color='blue', linestyle='--',label='[O III] 4959',linewidth=0.5)
    # plt.axvline(x=0.50082397, color='c', linestyle='--',label='[O III] 5007',linewidth=0.5)
    # plt.axvline(x=0.65498590, color='m', linestyle='--',label='[N II] 6548',linewidth=0.5)
    # plt.axvline(x=0.65852685, color='purple', linestyle='--',label='[N II] 6583',linewidth=0.5)
    xmin=min(obs_spec['wavelength_obs'])
    xmax=max(obs_spec['wavelength_obs'])
    xmin=min(xmin,min(obs_spec['wavelength_obs']))
    xmax=max(xmax,max(obs_spec['wavelength_obs']))
    ymin=min(obs_spec['spectra_obs'][obs_spec['spectra_obs']>-2])
    ymax=max(obs_spec['spectra_obs'][obs_spec['spectra_obs']>-2])
    # ymax=max(ymax,max(obs_spec['spectra_obs'][obs_spec['spectra_obs']>0]))

    # Print per-band chi^2 summary to stdout
    print('Per-band chi^2 metrics (spectroscopy):')
    for band_name, (chi2_norm_val, n_val, chi2_red_val, dof_val) in chi2_per_band.items():
        print(f'  {band_name}: chi^2/N={chi2_norm_val:.3f}, chi^2_nu={chi2_red_val:.3f} (N={n_val}, dof={dof_val})')

# Configure main plot (top subplot)
ax1.set_xlim(xmin,xmax)
ax1.set_ylim(ymin,ymax)
# ax1.set_xlim(0.65,0.66)
# ax1.set_xlim(0.1,10)
# ax1.set_ylim(1,1e4)
# ax1.set_ylabel(r'Flux ($\mu$Jy)')
ax1.set_ylabel(r"$\mathrm{f_{\lambda}}\ \mathrm{/\, 10^{-17}\ erg\ s^{-1}\ cm^{-2}\ \AA^{-1}}$")
overall_chi2_red = (chi2_total / max(n_total - np_free, 1)) if n_total > np_free else np.nan
# Extract runtime and convert from milliseconds to seconds
runtime_ms = float(hdul[0].header['spend_time/ms'])
runtime_s = runtime_ms / 1000.0
ax1.set_title(
    "ID="+hdul[0].header['ID']+","+galaxy_type+",z_true="+f"{float(hdul[0].header['z_{True}']):.3f}"+
    "\nSNR="+f"{float(hdul[0].header['SNR']):.2f}"+
    ",z_best="+f"{float(hdul[0].header['z_{MAL}']):.3f}"+
    ",$\\chi^2_\\nu$="+ (f"{overall_chi2_red:.2f}" if not np.isnan(overall_chi2_red) else "nan")+
    f",runtime={runtime_s:.1f}s"
)
# ax1.set_title("ID="+hdul[0].header['ID']+",SNR="+hdul[0].header['SNR']+",z_best="+hdul[0].header['z_{MAL}']+",z_true="+hdul[0].header['z_{True}'])
ax1.legend(ncol=2,loc='best')

# If a code name is provided, place it at the center of the main figure
if args.code_name:
    ax1.text(0.5, 0.6, args.code_name, transform=ax1.transAxes,
             ha='center', va='center', fontsize=20, fontweight='bold', color='black', alpha=0.3)

# Configure residual plot (bottom subplot)
ax2.set_xlim(xmin,xmax)
ax2.axhline(y=0, color='tab:blue', linestyle='-')
# ax2.set_xlabel(r'Rest-frame Wavelength ($\mu$m)')
ax2.set_xlabel(r"$\lambda / \mathrm{\AA}$")
ax2.set_ylabel('(obs-mod)/err')
ax2.grid(True, alpha=0.3)
# Ensure residual axis spans at least [-3, 3]
curr_ymin, curr_ymax = ax2.get_ylim()
# ax2.set_ylim(min(curr_ymin, -3.9), max(curr_ymax, 3.9))

plt.subplots_adjust(hspace=0)
plt.show()
plt.savefig(file+".png",dpi=300, bbox_inches='tight')
