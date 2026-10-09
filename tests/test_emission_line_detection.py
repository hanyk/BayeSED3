"""SPZLINE-anchored tests for emission-line detection.

Uses the SDSS pipeline's own line-measurement table (SPZLINE) as an
independent, real-data reference: lines whose fitted area exceeds
``snr_min`` sigma are expected to be found by our detector. SPZLINE fits
its full candidate template regardless of whether a line is present, so
only its own significant subset (|LINEAREA/LINEAREA_ERR| >= 3) is used,
and only for positive assertions — its absences are uninformative (e.g.
He I 4922/5015 are not in its template at all).

Run directly (``python tests/test_emission_line_detection.py``) or via
pytest.
"""

import os

import numpy as np
from astropy.io import fits

from bayesed.emission_lines import C_KMS, EMISSION_LINES, detect_lines

SDSS_SPEC = os.path.join('observation', 'sdss', 'spec',
                         'spec-0285-51930-0184.fits')


def _load_sdss(path):
    """Return (rest wl um, flux, err, z) plus the significant SPZLINE lines."""
    with fits.open(path) as hdul:
        coadd = hdul['COADD'].data
        z = float(hdul['SPECOBJ'].data['Z'][0])
        spz = hdul['SPZLINE'].data

    wl_um = 10.0 ** np.asarray(coadd['loglam'], dtype=float) / 1e4 / (1.0 + z)
    flux = np.asarray(coadd['flux'], dtype=float)
    err = 1.0 / np.sqrt(np.asarray(coadd['ivar'], dtype=float))
    ok = (np.isfinite(wl_um) & np.isfinite(flux) & np.isfinite(err)
          & (err > 0))

    reference = {}
    for row in spz:
        if row['LINEZ'] == 0:  # placeholder row: never fitted
            continue
        if row['LINEAREA_ERR'] == 0:
            continue
        snr = row['LINEAREA'] / row['LINEAREA_ERR']
        if abs(snr) < 3.0:  # pipeline's own measurement is not significant
            continue
        reference[row['LINENAME']] = row['LINEWAVE'] / 1e4  # rest um
    return wl_um[ok], flux[ok], err[ok], z, reference


def _table_labels_covering(reference, window_kms):
    """Map SPZLINE reference wavelengths to our EMISSION_LINES labels."""
    matched = []
    for line_wl, label, _family in EMISSION_LINES:
        dwl = line_wl * window_kms / C_KMS
        for ref_wl in reference.values():
            if abs(ref_wl - line_wl) <= dwl:
                matched.append(label)
                break
    return set(matched)


def test_real_sdss_spectrum_recall():
    """Every SPZLINE-significant line in coverage is detected at 2.5 sigma."""
    wl, flux, err, _z, reference = _load_sdss(SDSS_SPEC)
    detected = {label for _, label, _
                in detect_lines(wl, flux, err, snr_threshold=2.5)}

    expected = _table_labels_covering(reference, window_kms=3000.0)
    missing = expected - detected
    assert not missing, (
        f'undetected lines significant in SPZLINE: {sorted(missing)}')


def test_too_few_pixels_returns_empty():
    assert detect_lines(np.array([0.5, 0.51]), np.array([1.0, 1.0]),
                        np.array([0.1, 0.1])) == []


if __name__ == '__main__':
    for name, fn in sorted(globals().items()):
        if name.startswith('test_') and callable(fn):
            fn()
            print(f'{name}: PASS')
    print('All tests passed.')
