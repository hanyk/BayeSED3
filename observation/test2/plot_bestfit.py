from astropy.table import Table
import matplotlib.pyplot as plt
from astropy.io import fits
import sys
import numpy as np

file=sys.argv[1]
print(file)
hdul = fits.open(file)

if hdul.__contains__('model:total'):
    total=hdul['model:total'].data
    mask=total['flux']>0
    plt.loglog(total[mask]['wavelength_obs'],total[mask]['flux'],label='Total')
    xmin=min(total['wavelength_obs'])
    xmax=max(total['wavelength_obs'])
    ymin=min(total[mask]['flux'])
    ymax=max(total[mask]['flux'])

models=[hdul[i].name for i in range(0,len(hdul)) if 'model:' in hdul[i].name and hdul[i].name!='model:total']
for model in models:
    obs_phot=hdul[model].data
    mask=obs_phot['flux']>0
    plt.loglog(obs_phot[mask]['wavelength_obs'],obs_phot[mask]['flux'],label=model.replace('model:',''))
    if len(models)==1:
        xmin=min(obs_phot['wavelength_obs'])
        xmax=max(obs_phot['wavelength_obs'])
        ymin=min(obs_phot[mask]['flux'])
        ymax=max(obs_phot[mask]['flux'])

if hdul.__contains__('obs:phot'):
    obs_phot=hdul['obs:phot'].data
    obs_phot=obs_phot[obs_phot['iselected']==1]
    plt.errorbar(obs_phot['wavelength_obs_center'],obs_phot['flux_obs'],obs_phot['flux_obs_err'],fmt='x',color='red',label='phot:obs')
    plt.errorbar(obs_phot['wavelength_obs_center'],obs_phot['flux_model'],obs_phot['flux_model_err'],fmt='+',color='blue',label='phot:mod')
    if len(models)>1:
        xmin=min(obs_phot['wavelength_obs_center'])
        xmax=max(obs_phot['wavelength_obs_center'])
        ymin=min(obs_phot['flux_obs'][obs_phot['flux_obs']>0])
        ymax=max(obs_phot['flux_obs'][obs_phot['flux_obs']>0])

if hdul.__contains__('obs:spec') and '--no_spectra_fit' not in hdul[0].header['RUN_ARGUMENTS']:
    obs_spec=hdul['obs:spec'].data
    bands=np.unique(obs_spec['iband'])
    for i in bands:
        plt.semilogy(obs_spec[obs_spec['iband']==i]['wavelength_obs'],obs_spec[obs_spec['iband']==i]['spectra_obs'],label='spec:obs_'+str(i))
        plt.semilogy(obs_spec[obs_spec['iband']==i]['wavelength_obs'],obs_spec[obs_spec['iband']==i]['spectra_mod'],label='spec:mod_'+str(i))
    xmin=min(obs_spec['wavelength_obs'])
    xmax=max(obs_spec['wavelength_obs'])
    ymin=min(obs_spec['spectra_obs'][obs_spec['spectra_obs']>0])
    ymax=max(obs_spec['spectra_obs'][obs_spec['spectra_obs']>0])


plt.xlim(xmin,xmax)
plt.ylim(ymin,ymax)
plt.xlabel(r'$\lambda/\rm \mu m$')
plt.ylabel(r'$Flux/\rm \mu Jy$')
plt.title("ID="+hdul[0].header['ID']+",SNR="+hdul[0].header['SNR']+",z_best="+hdul[0].header['z_{MAL}']+",Xmin^2/Nd="+hdul[0].header['XMIN^2/ND'])
# plt.title("ID="+hdul[0].header['ID']+",SNR="+hdul[0].header['SNR']+",z_best="+hdul[0].header['z_{MAL}']+",z_true="+hdul[0].header['z_{True}'])
plt.legend()
plt.show()
plt.savefig(file+".png")
