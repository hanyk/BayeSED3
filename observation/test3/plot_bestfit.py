from astropy.table import Table
import matplotlib.pyplot as plt
from astropy.io import fits
import sys
import numpy as np

file=sys.argv[1]
print(file)
hdul = fits.open(file)

c1 = 2.9979246e+14 #um/s
if hdul.__contains__('model:total'):
    total=hdul['model:total'].data
    mask=total['flux']>0
    # plt.loglog(total[mask]['wavelength_rest'],total[mask]['flux'],label='Total')
    xmin=min(total['wavelength_rest'])
    xmax=max(total['wavelength_rest'])
    ymin=min(total[mask]['flux'])
    ymax=max(total[mask]['flux'])

models=[hdul[i].name for i in range(0,len(hdul)) if 'model:' in hdul[i].name and hdul[i].name!='model:total']
for model in models[0:1]:
    obs_phot=hdul[model].data
    mask=obs_phot['flux']>0
    # plt.loglog(obs_phot[mask]['wavelength_rest'],obs_phot[mask]['flux'],label=model.replace('model:',''))
    # if len(models)==1:
        # xmin=min(obs_phot['wavelength_rest'])
        # xmax=max(obs_phot['wavelength_rest'])
        # ymin=min(obs_phot[mask]['flux'])
        # ymax=max(obs_phot[mask]['flux'])

if hdul.__contains__('obs:phot') and '--no_photometry_fit' not in hdul[0].header['RUN_ARGUMENTS']:
    obs_phot=hdul['obs:phot'].data
    obs_phot=obs_phot[obs_phot['iselected']==1]
    plt.errorbar(obs_phot['wavelength_rest_center'],obs_phot['flux_obs'],obs_phot['flux_obs_err'],fmt='x',color='red',label='phot:obs')
    plt.errorbar(obs_phot['wavelength_rest_center'],obs_phot['flux_model'],obs_phot['flux_model_err'],fmt='+',color='blue',label='phot:mod')
    # if len(models)>1:
    xmin=min(obs_phot['wavelength_rest_center'])
    xmax=max(obs_phot['wavelength_rest_center'])
    ymin=min(obs_phot['flux_obs'][obs_phot['flux_obs']>0])
    ymax=max(obs_phot['flux_obs'][obs_phot['flux_obs']>0])

if hdul.__contains__('obs:spec') and '--no_spectra_fit' not in hdul[0].header['RUN_ARGUMENTS']:
    obs_spec=hdul['obs:spec'].data
    bands=np.unique(obs_spec['iband'])
    for i in bands[0:4]:
        mask=(obs_spec['iband']==i)
        # mask=(obs_spec['iband']==i)*(obs_spec['spectra_obs']>0)*(obs_spec['spectra_obs']/obs_spec['spectra_obs_err0']>0)
        plt.plot(obs_spec[mask]['wavelength_rest'],obs_spec[mask]['spectra_obs'],label='spec:obs_'+str(i))
        plt.loglog(obs_spec[mask]['wavelength_rest'],obs_spec[mask]['spectra_mod'],label='spec:mod_'+str(i), linestyle='--')
        # plt.plot(obs_spec[mask]['wavelength_rest'],obs_spec[mask]['spectra_obs']-obs_spec[mask]['spectra_mod'],label='spec:diff_'+str(i))
        # plt.plot(obs_spec[mask]['wavelength_rest'],c1*obs_spec[mask]['wavelength_rest']**-2*obs_spec[mask]['spectra_obs'],label='spec:obs_'+str(i))
        # plt.plot(obs_spec[mask]['wavelength_rest'],c1*obs_spec[mask]['wavelength_rest']**-2*obs_spec[mask]['spectra_mod'],label='spec:mod_'+str(i))
        # plt.plot(obs_spec[mask]['wavelength_rest'],c1*obs_spec[mask]['wavelength_rest']**-2*obs_spec[mask]['spectra_obs']-c1*obs_spec[mask=i]['wavelength_rest']**-2*obs_spec[mask]['spectra_mod'],label='spec:diff_'+str(i))
    # plt.axhline(y=0, color='red', linestyle='--')
    plt.axvline(x=0.65646140, color='red', linestyle='--',label='H_alpha',linewidth=0.5)
    plt.axvline(x=0.48626830, color='green', linestyle='--',label='H_beta',linewidth=0.5)
    plt.axvline(x=0.49602949, color='blue', linestyle='--',label='[O_III]4959',linewidth=0.5)
    plt.axvline(x=0.50082397, color='c', linestyle='--',label='[O_III]5007',linewidth=0.5)
    plt.axvline(x=0.65498590, color='m', linestyle='--',label='[N_II]6548 ',linewidth=0.5)
    plt.axvline(x=0.65852685, color='purple', linestyle='--',label='[N_II]6583 ',linewidth=0.5)
    xmin=min(obs_spec['wavelength_rest'])
    xmax=max(obs_spec['wavelength_rest'])
    xmin=min(xmin,min(obs_spec['wavelength_rest']))
    xmax=max(xmax,max(obs_spec['wavelength_rest']))
    ymin=min(obs_spec['spectra_obs'][obs_spec['spectra_obs']>0])
    ymax=max(obs_spec['spectra_obs'][obs_spec['spectra_obs']>0])
    # ymax=max(ymax,max(obs_spec['spectra_obs'][obs_spec['spectra_obs']>0]))


plt.xlim(xmin,xmax)
plt.ylim(ymin,ymax)
# plt.xlim(0.65,0.66)
# plt.xlim(0.1,10)
# plt.ylim(1,1e4)
plt.xlabel(r'$\lambda/\rm \mu m$')
plt.ylabel(r'$Flux/\rm \mu Jy$')
plt.title("ID="+hdul[0].header['ID']+",SNR="+hdul[0].header['SNR']+",z_best="+hdul[0].header['z_{MAL}']+",Xmin^2/Nd="+hdul[0].header['XMIN^2/ND'])
# plt.title("ID="+hdul[0].header['ID']+",SNR="+hdul[0].header['SNR']+",z_best="+hdul[0].header['z_{MAL}']+",z_true="+hdul[0].header['z_{True}'])
plt.legend(ncol=2,loc='upper left')
plt.show()
plt.savefig(file+".png")
