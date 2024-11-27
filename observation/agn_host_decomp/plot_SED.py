from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams

rcParams['savefig.dpi'] = 150
rcParams['mathtext.fontset'] = 'stix'
rcParams['font.family'] = 'Times New Roman'
rcParams['font.size'] = 16
rcParams['xtick.direction'] = 'in'
rcParams['ytick.direction'] = 'in'

from astropy.table import Table
import sys
import re

output="observation/agn_host_decomp/output"
collection = 'sample'
xid = 'cid_71'
hdul = fits.open(rf'{output}/{collection}/{xid}/0csp_sfh801_bc2003_lr_BaSeL_chab_i0000_2dal8_10_1gb_8_2clumpy201410tor_1_3QSO1_2dal7_3_sys_err0_bestfit.fits')

fig, ax = plt.subplots(figsize=(10, 8), nrows=2, ncols=2, sharex=True, sharey=True,
                      gridspec_kw={'hspace': 0, 'wspace': 0})
c1 = 2.9979246e+14 #um/s
if hdul.__contains__('model:total'):
    total=hdul['model:total'].data
    mask=total['flux']>0
    ax[0][0].plot(total[mask]['wavelength_rest'],total[mask]['flux'],label='Total')
    xmin=min(total['wavelength_rest'])
    xmax=max(total['wavelength_rest'])
    ymin=min(total[mask]['flux'])
    ymax=max(total[mask]['flux'])

models=[hdul[i].name for i in range(0,len(hdul)) if 'model:' in hdul[i].name and hdul[i].name!='model:total']

# CSP model
model = models[0]
obs_phot=hdul[model].data
# mask=obs_phot['flux']>0
ax[0][0].loglog(obs_phot[mask]['wavelength_rest'],obs_phot[mask]['flux'],label='host galaxy', color='coral', alpha=0.6)

# clumpy model
model = models[2]
clumpy=hdul[model].data

# QSO model
model = models[3]
qso=hdul[model].data

# clumpy + QSO
ax[0][0].loglog(obs_phot[mask]['wavelength_rest'],clumpy[mask]['flux'] + qso[mask]['flux'],label='AGN', color='seagreen', alpha=0.6)

# gb model
model = models[1]
obs_phot=hdul[model].data
# mask=obs_phot['flux']>0
ax[0][0].loglog(obs_phot[mask]['wavelength_rest'],obs_phot[mask]['flux'],label='cold dust', color='grey', alpha=0.6)


if hdul.__contains__('obs:phot'):
    obs_phot=hdul['obs:phot'].data
    obs_phot=obs_phot[obs_phot['iselected']==1]

    host=[ 'host' in x for x in obs_phot['band']]
    point=[ 'point' in x for x in obs_phot['band']]
    total=[ 'total' in x for x in obs_phot['band']]

    host_mask = obs_phot[host]['flux_obs'] > 0
    point_mask = obs_phot[point]['flux_obs'] > 0
    total_mask = obs_phot[total]['flux_obs'] > 0



    host_uplims = obs_phot[host]['flux_obs_err0'][host_mask] < 0
    point_uplims = obs_phot[point]['flux_obs_err0'][point_mask] < 0
    total_uplims = obs_phot[total]['flux_obs_err0'][total_mask] < 0

    obs_phot=obs_phot[obs_phot['iselected']==1]

    total_err = obs_phot[total]['flux_obs_err'][total_mask]
    # The errors for uplims are minimized for clarity in visualization
    # total_err[total_uplims] = 4000
    total_flux=obs_phot[total]['flux_obs'][total_mask]
    total_flux[total_uplims]*=3

    ax[0][0].errorbar(obs_phot[total]['wavelength_rest_center'][total_mask],
             # obs_phot[total]['flux_obs'][total_mask],
             total_flux,
             total_err,
             uplims=total_uplims, elinewidth=1,
             alpha=0.6,
             fmt='.',color='blue',label='total observed')

    ax[0][0].errorbar(obs_phot[host]['wavelength_rest_center'][host_mask],
                 obs_phot[host]['flux_obs'][host_mask],
                 obs_phot[host]['flux_obs_err'][host_mask],
                 uplims=host_uplims, elinewidth=1,
                 fmt='x',color='red',label='host galaxy observed')

    ax[0][0].errorbar(obs_phot[point]['wavelength_rest_center'][point_mask],
                 obs_phot[point]['flux_obs'][point_mask],
                 obs_phot[point]['flux_obs_err'][point_mask],
                 uplims=point_uplims, elinewidth=1,
                 fmt='x',color='green',label='AGN observed')

    if len(models)>1:
        xmin=min(obs_phot['wavelength_rest_center']) * 0.9
        xmax=max(obs_phot['wavelength_rest_center']) * 1.2
        ymin=min(obs_phot['flux_obs'][obs_phot['flux_obs']>0])
        ymax=max(obs_phot['flux_obs'][obs_phot['flux_obs']>0]) * 2.5
ax[0][0].grid()
ax[0][0].set_xlim(xmin,xmax)
ax[0][0].set_ylim(1,ymax)
# ax[0][0].set_xlabel(r'$\lambda/\rm \mu m$')
ax[0][0].set_ylabel(r'Flux/$\rm \mu Jy$')
# ax[0][0].set_title(f"CID 71, $z={float(hdul[0].header['z_{MAL}']):.3f}$")
ax[0][0].text(10**((np.log10(xmin)+np.log10(xmax))/2), 10**4.6,
              f"CID 71, $z={float(hdul[0].header['z_{MAL}']):.3f}$",
             ha='center')


# ECDFS 379 -----------------------------------------------------------------------------------

collection = 'sample'
xid = 'ecdfs_379'
hdul = fits.open(rf'{output}/{collection}/{xid}/0csp_sfh801_bc2003_lr_BaSeL_chab_i0000_2dal8_10_1gb_8_2clumpy201410tor_1_3QSO1_2dal7_3_sys_err0_bestfit.fits')

c1 = 2.9979246e+14 #um/s
if hdul.__contains__('model:total'):
    total=hdul['model:total'].data
    mask=total['flux']>0
    ax[0][1].plot(total[mask]['wavelength_rest'],total[mask]['flux'],label='Total')
    xmin=min(total['wavelength_rest'])
    xmax=max(total['wavelength_rest'])
    ymin=min(total[mask]['flux'])
    ymax=max(total[mask]['flux'])

models=[hdul[i].name for i in range(0,len(hdul)) if 'model:' in hdul[i].name and hdul[i].name!='model:total']

# CSP model
model = models[0]
obs_phot=hdul[model].data
# mask=obs_phot['flux']>0
ax[0][1].loglog(obs_phot[mask]['wavelength_rest'],obs_phot[mask]['flux'],label='host galaxy', color='coral', alpha=0.6)

# clumpy model
model = models[2]
clumpy=hdul[model].data

# QSO model
model = models[3]
qso=hdul[model].data

# clumpy + QSO
ax[0][1].loglog(obs_phot[mask]['wavelength_rest'],clumpy[mask]['flux'] + qso[mask]['flux'],label='AGN', color='seagreen', alpha=0.6)

# gb model
model = models[1]
obs_phot=hdul[model].data
# mask=obs_phot['flux']>0
ax[0][1].loglog(obs_phot[mask]['wavelength_rest'],obs_phot[mask]['flux'],label='cold dust', color='grey', alpha=0.6)


if hdul.__contains__('obs:phot'):
    obs_phot=hdul['obs:phot'].data
    obs_phot=obs_phot[obs_phot['iselected']==1]

    host=[ 'host' in x for x in obs_phot['band']]
    point=[ 'point' in x for x in obs_phot['band']]
    total=[ 'total' in x for x in obs_phot['band']]

    host_mask = obs_phot[host]['flux_obs'] > 0
    point_mask = obs_phot[point]['flux_obs'] > 0
    total_mask = obs_phot[total]['flux_obs'] > 0



    host_uplims = obs_phot[host]['flux_obs_err0'][host_mask] < 0
    point_uplims = obs_phot[point]['flux_obs_err0'][point_mask] < 0
    total_uplims = obs_phot[total]['flux_obs_err0'][total_mask] < 0

    obs_phot=obs_phot[obs_phot['iselected']==1]

    total_err = obs_phot[total]['flux_obs_err'][total_mask]
    # total_err[total_uplims] = 2500
    total_flux=obs_phot[total]['flux_obs'][total_mask]
    total_flux[total_uplims]*=3

    ax[0][1].errorbar(obs_phot[total]['wavelength_rest_center'][total_mask],
             # obs_phot[total]['flux_obs'][total_mask],
             total_flux,
             total_err,
             # obs_phot[total]['flux_obs_err'][total_mask],
             uplims=total_uplims, elinewidth=1,
             alpha=0.6,
             fmt='.',color='blue',label='total observed')

    ax[0][1].errorbar(obs_phot[host]['wavelength_rest_center'][host_mask],
                 obs_phot[host]['flux_obs'][host_mask],
                 obs_phot[host]['flux_obs_err'][host_mask],
                 uplims=host_uplims, elinewidth=1,
                 fmt='x',color='red',label='host galaxy observed')

    ax[0][1].errorbar(obs_phot[point]['wavelength_rest_center'][point_mask],
                 obs_phot[point]['flux_obs'][point_mask],
                 obs_phot[point]['flux_obs_err'][point_mask],
                 uplims=point_uplims, elinewidth=1,
                 fmt='x',color='green',label='AGN observed')

    # for xi, yi, yerri, uplim in zip(obs_phot[total]['wavelength_rest_center'][total_mask],
    #          obs_phot[total]['flux_obs'][total_mask],
    #          obs_phot[total]['flux_obs_err'][total_mask],
    #                                total_uplims):
    #     if uplim:
    #         ax[1].annotate('', xy=(xi, 100), xytext=(xi, yi),
    #                     arrowprops=dict(arrowstyle='->', color='blue', alpha=0.6))

    # ax.errorbar(obs_phot['wavelength_rest_center'],obs_phot['flux_model'],obs_phot['flux_model_err'],fmt='+',color='blue',label='phot:mod')
    if len(models)>1:
        xmin=min(obs_phot['wavelength_rest_center']) * 0.9
        xmax=max(obs_phot['wavelength_rest_center']) * 1.2
        ymin=min(obs_phot['flux_obs'][obs_phot['flux_obs']>0])
        ymax=max(obs_phot['flux_obs'][obs_phot['flux_obs']>0]) * 2.5

ax[0][1].set_xlim(xmin,xmax)
ax[0][1].set_ylim(1,ymax)
# ax[0][1].set_xlabel(r'$\lambda/\rm \mu m$')
# ax[0][1].set_ylabel(r'Flux/$\rm \mu Jy$')
# ax[0][1].set_title(f"ECDFS 379, $z={float(hdul[0].header['z_{MAL}']):.3f}$")
ax[0][1].text(10**((np.log10(xmin)+np.log10(xmax))/2), 10**4.6,
              f"ECDFS 379, $z={float(hdul[0].header['z_{MAL}']):.3f}$",
             ha='center')

# CID 216 -----------------------------------------------------------------------------------

collection = 'sample'
xid = 'cid_216'
hdul = fits.open(rf'{output}/{collection}/{xid}/0csp_sfh801_bc2003_lr_BaSeL_chab_i0000_2dal8_10_1gb_8_2clumpy201410tor_1_3QSO1_2dal7_3_sys_err0_bestfit.fits')

c1 = 2.9979246e+14 #um/s
if hdul.__contains__('model:total'):
    total=hdul['model:total'].data
    mask=total['flux']>0
    ax[1][0].plot(total[mask]['wavelength_rest'],total[mask]['flux'],label='Total')
    xmin=min(total['wavelength_rest'])
    xmax=max(total['wavelength_rest'])
    ymin=min(total[mask]['flux'])
    ymax=max(total[mask]['flux'])

models=[hdul[i].name for i in range(0,len(hdul)) if 'model:' in hdul[i].name and hdul[i].name!='model:total']

# CSP model
model = models[0]
obs_phot=hdul[model].data
# mask=obs_phot['flux']>0
ax[1][0].loglog(obs_phot[mask]['wavelength_rest'],obs_phot[mask]['flux'],label='host galaxy', color='coral', alpha=0.6)

# clumpy model
model = models[2]
clumpy=hdul[model].data

# QSO model
model = models[3]
qso=hdul[model].data

# clumpy + QSO
ax[1][0].loglog(obs_phot[mask]['wavelength_rest'],clumpy[mask]['flux'] + qso[mask]['flux'],label='AGN', color='seagreen', alpha=0.6)

# gb model
model = models[1]
obs_phot=hdul[model].data
# mask=obs_phot['flux']>0
ax[1][0].loglog(obs_phot[mask]['wavelength_rest'],obs_phot[mask]['flux'],label='cold dust', color='grey', alpha=0.6)


if hdul.__contains__('obs:phot'):
    obs_phot=hdul['obs:phot'].data
    obs_phot=obs_phot[obs_phot['iselected']==1]

    host=[ 'host' in x for x in obs_phot['band']]
    point=[ 'point' in x for x in obs_phot['band']]
    total=[ 'total' in x for x in obs_phot['band']]

    host_mask = obs_phot[host]['flux_obs'] > 0
    point_mask = obs_phot[point]['flux_obs'] > 0
    total_mask = obs_phot[total]['flux_obs'] > 0



    host_uplims = obs_phot[host]['flux_obs_err0'][host_mask] < 0
    point_uplims = obs_phot[point]['flux_obs_err0'][point_mask] < 0
    total_uplims = obs_phot[total]['flux_obs_err0'][total_mask] < 0

    obs_phot=obs_phot[obs_phot['iselected']==1]

    total_err = obs_phot[total]['flux_obs_err'][total_mask]
    # total_err[total_uplims] = 250, 2500
    total_flux=obs_phot[total]['flux_obs'][total_mask]
    total_flux[total_uplims]*=3

    ax[1][0].errorbar(obs_phot[total]['wavelength_rest_center'][total_mask],
             # obs_phot[total]['flux_obs'][total_mask],
             total_flux,
             total_err,
             # obs_phot[total]['flux_obs_err'][total_mask],
             uplims=total_uplims, elinewidth=1,
             alpha=0.6,
             fmt='.',color='blue',label='total observed')

    ax[1][0].errorbar(obs_phot[host]['wavelength_rest_center'][host_mask],
                 obs_phot[host]['flux_obs'][host_mask],
                 obs_phot[host]['flux_obs_err'][host_mask],
                 uplims=host_uplims, elinewidth=1,
                 fmt='x',color='red',label='host galaxy observed')

    ax[1][0].errorbar(obs_phot[point]['wavelength_rest_center'][point_mask],
                 obs_phot[point]['flux_obs'][point_mask],
                 obs_phot[point]['flux_obs_err'][point_mask],
                 uplims=point_uplims, elinewidth=1,
                 fmt='x',color='green',label='AGN observed')

    # for xi, yi, yerri, uplim in zip(obs_phot[total]['wavelength_rest_center'][total_mask],
    #          obs_phot[total]['flux_obs'][total_mask],
    #          obs_phot[total]['flux_obs_err'][total_mask],
    #                                total_uplims):
    #     if uplim:
    #         ax[1].annotate('', xy=(xi, 100), xytext=(xi, yi),
    #                     arrowprops=dict(arrowstyle='->', color='blue', alpha=0.6))

    # ax.errorbar(obs_phot['wavelength_rest_center'],obs_phot['flux_model'],obs_phot['flux_model_err'],fmt='+',color='blue',label='phot:mod')
    if len(models)>1:
        xmin=min(obs_phot['wavelength_rest_center']) * 0.9
        xmax=max(obs_phot['wavelength_rest_center']) * 1.2
        ymin=min(obs_phot['flux_obs'][obs_phot['flux_obs']>0])
        ymax=max(obs_phot['flux_obs'][obs_phot['flux_obs']>0]) * 2.5

ax[1][0].set_xlim(xmin,xmax)
ax[1][0].set_ylim(.1,ymax)
ax[1][0].set_xlabel(r'$\lambda/\rm \mu m$')
ax[1][0].set_ylabel(r'Flux/$\rm \mu Jy$')
# ax[1][0].set_title(f"CID 216, $z={float(hdul[0].header['z_{MAL}']):.3f}$",)
ax[1][0].text(10**((np.log10(xmin)+np.log10(xmax))/2), 10**4.6,
              f"CID 216, $z={float(hdul[0].header['z_{MAL}']):.3f}$",
             ha='center')
ax[1][0].legend(fontsize=11, frameon=False, loc='upper left')

# CID 108 -----------------------------------------------------------------------------------

collection = 'sample'
xid = 'cid_108'
hdul = fits.open(rf'{output}/{collection}/{xid}/0csp_sfh801_bc2003_lr_BaSeL_chab_i0000_2dal8_10_1gb_8_2clumpy201410tor_1_3QSO1_2dal7_3_sys_err0_bestfit.fits')

c1 = 2.9979246e+14 #um/s
if hdul.__contains__('model:total'):
    total=hdul['model:total'].data
    mask=total['flux']>0
    ax[1][1].plot(total[mask]['wavelength_rest'],total[mask]['flux'],label='Total')
    xmin=min(total['wavelength_rest'])
    xmax=max(total['wavelength_rest'])
    ymin=min(total[mask]['flux'])
    ymax=max(total[mask]['flux'])

models=[hdul[i].name for i in range(0,len(hdul)) if 'model:' in hdul[i].name and hdul[i].name!='model:total']

# CSP model
model = models[0]
obs_phot=hdul[model].data
# mask=obs_phot['flux']>0
ax[1][1].loglog(obs_phot[mask]['wavelength_rest'],obs_phot[mask]['flux'],label='host galaxy', color='coral', alpha=0.6)

# clumpy model
model = models[2]
clumpy=hdul[model].data

# QSO model
model = models[3]
qso=hdul[model].data

# clumpy + QSO
ax[1][1].loglog(obs_phot[mask]['wavelength_rest'],clumpy[mask]['flux'] + qso[mask]['flux'],label='AGN', color='seagreen', alpha=0.6)

# gb model
model = models[1]
obs_phot=hdul[model].data
# mask=obs_phot['flux']>0
ax[1][1].loglog(obs_phot[mask]['wavelength_rest'],obs_phot[mask]['flux'],label='cold dust', color='grey', alpha=0.6)


if hdul.__contains__('obs:phot'):
    obs_phot=hdul['obs:phot'].data
    obs_phot=obs_phot[obs_phot['iselected']==1]

    host=[ 'host' in x for x in obs_phot['band']]
    point=[ 'point' in x for x in obs_phot['band']]
    total=[ 'total' in x for x in obs_phot['band']]

    host_mask = obs_phot[host]['flux_obs'] > 0
    point_mask = obs_phot[point]['flux_obs'] > 0
    total_mask = obs_phot[total]['flux_obs'] > 0



    host_uplims = obs_phot[host]['flux_obs_err0'][host_mask] < 0
    point_uplims = obs_phot[point]['flux_obs_err0'][point_mask] < 0
    total_uplims = obs_phot[total]['flux_obs_err0'][total_mask] < 0

    obs_phot=obs_phot[obs_phot['iselected']==1]

    total_err = obs_phot[total]['flux_obs_err'][total_mask]
    # total_err[total_uplims] = 250, 2500
    total_flux=obs_phot[total]['flux_obs'][total_mask]
    total_flux[total_uplims]*=3
    host_err = obs_phot[host]['flux_obs_err'][host_mask]
    # host_err[host_uplims] = 2
    host_flux= obs_phot[host]['flux_obs'][host_mask]
    host_flux[host_uplims]*=3

    ax[1][1].errorbar(obs_phot[total]['wavelength_rest_center'][total_mask],
             # obs_phot[total]['flux_obs'][total_mask],
             total_flux,
             total_err,
             # obs_phot[total]['flux_obs_err'][total_mask],
             uplims=total_uplims, elinewidth=1,
             alpha=0.6,
             fmt='.',color='blue',label='total observed')

    ax[1][1].errorbar(obs_phot[host]['wavelength_rest_center'][host_mask],
                 host_flux,
                 host_err,
                 uplims=host_uplims, elinewidth=1,
                 fmt='x',color='red',label='host galaxy observed')

    ax[1][1].errorbar(obs_phot[point]['wavelength_rest_center'][point_mask],
                 obs_phot[point]['flux_obs'][point_mask],
                 obs_phot[point]['flux_obs_err'][point_mask],
                 uplims=point_uplims, elinewidth=1,
                 fmt='x',color='green',label='AGN observed')

    # for xi, yi, yerri, uplim in zip(obs_phot[total]['wavelength_rest_center'][total_mask],
    #          obs_phot[total]['flux_obs'][total_mask],
    #          obs_phot[total]['flux_obs_err'][total_mask],
    #                                total_uplims):
    #     if uplim:
    #         ax[1].annotate('', xy=(xi, 100), xytext=(xi, yi),
    #                     arrowprops=dict(arrowstyle='->', color='blue', alpha=0.6))

    # ax.errorbar(obs_phot['wavelength_rest_center'],obs_phot['flux_model'],obs_phot['flux_model_err'],fmt='+',color='blue',label='phot:mod')
    if len(models)>1:
        xmin=min(obs_phot['wavelength_rest_center']) * 0.9
        xmax=max(obs_phot['wavelength_rest_center']) * 1.2
        ymin=min(obs_phot['flux_obs'][obs_phot['flux_obs']>0])
        ymax=max(obs_phot['flux_obs'][obs_phot['flux_obs']>0]) * 2.5

ax[1][1].set_xlim(xmin,xmax)
ax[1][1].set_ylim(0.5,ymax)
ax[1][1].set_xlabel(r'$\lambda/\rm \mu m$')
# ax[1][1].set_ylabel(r'Flux/$\rm \mu Jy$')
# ax[1][1].set_title(f"SXDS 0491, $z={float(hdul[0].header['z_{MAL}']):.3f}$")
ax[1][1].text(10**((np.log10(xmin)+np.log10(xmax))/2), 10**4.6,
              f"CID 108, $z={float(hdul[0].header['z_{MAL}']):.3f}$",
             ha='center')
ax[0][1].grid()
ax[1][0].grid()
ax[1][1].grid()
plt.tight_layout()
fig.show()
