from specutils import Spectrum1D
from specutils.io.registers import identify_spectrum_format
from astropy.nddata import StdDevUncertainty
from sfdmap2 import sfdmap
from astropy.table import Table
from astropy.io import fits
from astropy import constants as c
from astropy import units as u
import sys
import os.path
import numpy as np
import re
from PyAstronomy import pyasl

# Ensure at least one fits file is provided
if len(sys.argv) < 2:
    print("Usage: convert.py <sample_name> <fits_file1> <fits_file2> ...")
    sys.exit(1)

# Read the sample name
sample_name = sys.argv[1]

# Read the list of fits files
fits_files = sys.argv[2:]

# Open a text file for writing output
output_file = sample_name + ".txt"
with open(output_file, "w") as f1:
    # Initialize the SFDMap for dust extinction correction
    #https://github.com/kbarbary/sfddata.git
    m = sfdmap.SFDMap("~/sfddata")

    # Print initial information header
    print("# %s %d %d %d" % (sample_name, 0, 2, 1), file=f1)
    print("ID z_min z_max d/Mpc E(B-V) RA DEC", file=f1)

    # Process each fits file
    for file_fits in fits_files:
        ID = os.path.basename(file_fits).replace(".fits", "")

        identify_spectrum_format(file_fits)
        h = fits.open(file_fits)

        # Print initial information for each file
        print(
            "%s %g %g %g %g %g %g" %
            (ID + "_" + (str(h[2].data['CLASS'][0])).strip() + "_" +
             (str(h[2].data['SUBCLASS'][0])).strip(),
             h[2].data['z'][0],
             h[2].data['z'][0],
             0, 0, h[0].header['RA'],
             h[0].header['DEC']),
            end="", file=f1)

        spec = Spectrum1D.read(h, format="SDSS-III/IV spec")
        h1 = Table.read(file_fits, hdu=1, memmap=True)
        h2 = Table.read(file_fits, hdu=2, memmap=True)
        print(file_fits, h2['CLASS'], h2['SUBCLASS'])
        EBV = m.ebv(h[0].header['RA'], h[0].header['DEC'])

        flux0 = pyasl.unred(spec.wavelength.value, spec.flux.value, ebv=EBV)
        r = flux0 / spec.flux.value
        spec = spec * r
        spec = spec * spec.wavelength**2 / c.c
        r = spec.flux.to(u.uJy).value / spec.flux.value
        spec = spec * r

        w = spec.wavelength.to(u.um).value
        f = spec.flux.value
        fe = spec.uncertainty.represent_as(StdDevUncertainty).quantity.value
        print(fe[np.isinf(fe)])
        dlam_gal = np.diff(w)
        dlam_gal = np.append(dlam_gal, dlam_gal[-1])
        if 'WDISP' in h1.keys():
            wdisp = h1['WDISP']
        else:
            wdisp = h1['wdisp']
        wdisp = wdisp * dlam_gal

        mask = (~np.isinf(fe))
        # mask = (~spec.mask) * (~np.isinf(fe))
        w = w[mask]
        f = f[mask]
        fe = fe[mask]
        wdisp = wdisp[mask]

        print(" %d" % (len(w)), end="", file=f1)
        for j in range(0, len(w)):
            print(
                " %g %g %g %g" %
                (w[j], f[j], fe[j], wdisp[j]), end="", file=f1)
        print(file=f1)
        f1.flush()
