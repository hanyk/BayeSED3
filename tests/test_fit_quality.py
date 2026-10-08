"""Tests for the fit_quality results verb (ADR-0020).

Seams under test:
- Python: ``BayeSEDResults.get_fit_quality(object_id, band_edges, top_n, sigma_threshold)``
- CLI:    ``bayesed3 results <outdir> fit_quality [--cat] [--cfg] [--obj]
          [--band EDGES] [--top N] [--sigma T] [--json]`` (in test_inspect_cli.py)

Both run against a synthetic results tree (HDF5 + bestfit FITS), no binary.
"""

from pathlib import Path

import h5py
import json

import numpy as np
import pytest
from astropy.io import fits

from bayesed.inspect_cli import run_inspect as run_inspect_impl
from bayesed.results import BayeSEDResults

CATALOG = 'gal'
CONFIG = '0csp'
OBJECT = 'obj-1'


def _write_hdf5(outdir: Path):
    with h5py.File(outdir / f'{CATALOG}_{CONFIG}.hdf5', 'w') as f:
        str_dtype = h5py.string_dtype(encoding='utf-8')
        f['ID'] = np.array([OBJECT], dtype=str_dtype)
        f['parameters_name'] = np.array(['z'], dtype=str_dtype)
        f['parameters'] = np.array([[0.1]], dtype='f8')


def _phot_hdu(bands, wl_rest, flux_obs, flux_model, flux_obs_err,
              iselected=None):
    band_col = fits.Column(
        name='band', format='16A', array=np.array(bands, dtype='U16'))
    columns = [
        band_col,
        fits.Column(name='iselected', format='J', array=(
            np.ones(len(bands), dtype=int) if iselected is None
            else np.asarray(iselected, dtype=int))),
        fits.Column(name='wavelength_obs_center', format='D',
                    array=np.asarray(wl_rest) * 1.1),
        fits.Column(name='wavelength_rest_center', format='D',
                    array=np.asarray(wl_rest)),
        fits.Column(name='flux_obs', format='D', array=np.asarray(flux_obs)),
        fits.Column(name='flux_model', format='D',
                    array=np.asarray(flux_model)),
        fits.Column(name='flux_obs_err', format='D',
                    array=np.asarray(flux_obs_err)),
        fits.Column(name='flux_model_err', format='D',
                    array=np.ones(len(bands)) * 0.5),
    ]
    hdu = fits.BinTableHDU.from_columns(columns)
    hdu.header['EXTNAME'] = 'obs:phot'
    return hdu


def _spec_hdu(ibands, wl_rest, flux_obs, flux_mod, flux_obs_err):
    columns = [
        fits.Column(name='iband', format='J',
                    array=np.asarray(ibands, dtype=int)),
        fits.Column(name='wavelength_obs', format='D',
                    array=np.asarray(wl_rest) * 1.1),
        fits.Column(name='wavelength_rest', format='D',
                    array=np.asarray(wl_rest)),
        fits.Column(name='spectra_obs', format='D', array=np.asarray(flux_obs)),
        fits.Column(name='spectra_mod', format='D', array=np.asarray(flux_mod)),
        fits.Column(name='spectra_obs_err', format='D',
                    array=np.asarray(flux_obs_err)),
    ]
    hdu = fits.BinTableHDU.from_columns(columns)
    hdu.header['EXTNAME'] = 'obs:spec'
    return hdu


def _model_hdu(name):
    hdu = fits.BinTableHDU.from_columns([
        fits.Column(name='wavelength_rest', format='D',
                    array=np.linspace(0.1, 3.0, 50)),
        fits.Column(name='flux', format='D', array=np.ones(50) * 15),
    ])
    hdu.header['EXTNAME'] = name
    return hdu


def _write_bestfit(outdir, hdus, header=None):
    fits_dir = outdir / CATALOG / OBJECT
    fits_dir.mkdir(parents=True, exist_ok=True)
    primary = fits.PrimaryHDU()
    if header:
        for key, value in header.items():
            primary.header[key] = value
    hdul = fits.HDUList([primary] + list(hdus))
    hdul.writeto(fits_dir / f'{CONFIG}_bestfit.fits', overwrite=True)
    hdul.close()


def _write_results(tmp_path):
    outdir = tmp_path / 'out'
    outdir.mkdir()
    _write_hdf5(outdir)
    return outdir


def test_get_fit_quality_without_bestfit_fits_is_one_line_error(tmp_path):
    outdir = _write_results(tmp_path)
    results = BayeSEDResults(outdir)
    with pytest.raises(
            FileNotFoundError,
            match='No bestfit FITS files found for object obj-1'):
        results.get_fit_quality('obj-1')


def test_list_objects_decodes_bytes_ids(tmp_path):
    outdir = _write_results(tmp_path)
    results = BayeSEDResults(outdir)
    assert results.list_objects() == ['obj-1']


def test_fit_quality_header_only_when_no_data_hdus(tmp_path):
    outdir = _write_results(tmp_path)
    _write_bestfit(outdir, [_model_hdu('model:total')],
                   header={'XMIN^2/ND': 2.5})
    results = BayeSEDResults(outdir)
    record = results.get_fit_quality(OBJECT)
    assert record['datasets'] == []
    assert record['object_id'] == OBJECT
    assert record['chi2_source'] == 'header'
    assert record['chi2'] == pytest.approx(2.5)
    assert record['chi2_red'] == pytest.approx(2.5)


def test_fit_quality_phot_dataset_record(tmp_path):
    outdir = _write_results(tmp_path)
    # Residuals (obs-mod)/err: -1.0, 0.5, 0.0
    _write_bestfit(outdir, [_phot_hdu(
        bands=['u', 'u', 'u'],
        wl_rest=[0.35, 0.40, 0.45],
        flux_obs=[10.0, 10.0, 10.0],
        flux_model=[11.0, 9.0, 10.0],
        flux_obs_err=[1.0, 2.0, 2.0],
    )], header={'XMIN^2/ND': 2.5})
    results = BayeSEDResults(outdir)
    record = results.get_fit_quality(OBJECT)
    assert len(record['datasets']) == 1
    ds = record['datasets'][0]
    assert ds['kind'] == 'phot'
    assert ds['label'] == 'u'
    assert ds['wl_range_rest'] == pytest.approx([0.35, 0.45])
    assert ds['n'] == 3
    assert ds['chi2'] == pytest.approx(1.25)
    assert ds['chi2_red'] == pytest.approx(1.25 / 3)
    assert ds['resid_median'] == pytest.approx(0.0)
    assert ds['resid_rms'] == pytest.approx(np.sqrt(1.25 / 3))
    assert ds['sigma_threshold'] == pytest.approx(3.0)
    assert ds['frac_outside'] == pytest.approx(0.0)
    assert ds['flags'] == []
    assert ds['outliers'] == [
        {'wl_rest': 0.35, 'flux_obs': 10.0, 'n_sigma': -1.0},
        {'wl_rest': 0.40, 'flux_obs': 10.0, 'n_sigma': 0.5},
        {'wl_rest': 0.45, 'flux_obs': 10.0, 'n_sigma': 0.0},
    ]


def test_fit_quality_spec_splits_by_iband_by_default(tmp_path):
    outdir = _write_results(tmp_path)
    # iband 0: resid [0.0, -1.0]; iband 1: resid [0.0, 0.0]
    _write_bestfit(outdir, [_spec_hdu(
        ibands=[0, 0, 1, 1],
        wl_rest=[0.60, 0.62, 0.90, 0.95],
        flux_obs=[10.0, 10.0, 8.0, 8.0],
        flux_mod=[10.0, 12.0, 8.0, 8.0],
        flux_obs_err=[1.0, 2.0, 4.0, 2.0],
    )], header={'XMIN^2/ND': 2.5})
    results = BayeSEDResults(outdir)
    record = results.get_fit_quality(OBJECT)
    assert [ds['kind'] for ds in record['datasets']] == ['spec', 'spec']
    assert [ds['label'] for ds in record['datasets']] == ['spec_0', 'spec_1']
    ds0 = record['datasets'][0]
    assert ds0['wl_range_rest'] == pytest.approx([0.60, 0.62])
    assert ds0['n'] == 2
    assert ds0['chi2'] == pytest.approx(1.0)
    assert ds0['chi2_red'] == pytest.approx(0.5)
    assert [o['n_sigma'] for o in ds0['outliers']] == [
        pytest.approx(-1.0), pytest.approx(0.0),
    ]
    assert [o['flux_obs'] for o in ds0['outliers']] == [10.0, 10.0]
    assert record['datasets'][1]['wl_range_rest'] == pytest.approx([0.90, 0.95])
    assert record['datasets'][1]['chi2'] == pytest.approx(0.0)


def test_fit_quality_band_edges_cut_spec_and_leave_phot(tmp_path):
    outdir = _write_results(tmp_path)
    # resid: phot 'u' -> [1.0]; spec ibands ignored: [-1.0, 1.0, 0.0, 0.0]
    _write_bestfit(outdir, [
        _phot_hdu(
            bands=['u'], wl_rest=[0.5], flux_obs=[11.0], flux_model=[10.0],
            flux_obs_err=[1.0]),
        _spec_hdu(
            ibands=[0, 0, 1, 1],
            wl_rest=[0.40, 0.50, 0.70, 0.80],
            flux_obs=[10.0, 10.0, 10.0, 10.0],
            flux_mod=[11.0, 9.0, 10.0, 10.0],
            flux_obs_err=[1.0, 1.0, 2.0, 2.0]),
    ], header={'XMIN^2/ND': 2.5})
    results = BayeSEDResults(outdir)
    record = results.get_fit_quality(OBJECT, band_edges=[0.55])
    assert [ds['kind'] for ds in record['datasets']] == ['phot', 'spec', 'spec']
    phot_ds = record['datasets'][0]
    assert phot_ds['label'] == 'u'
    assert phot_ds['n'] == 1
    spec0, spec1 = record['datasets'][1:]
    assert spec0['label'] == 'spec_[0.4,0.55]'
    assert spec0['wl_range_rest'] == pytest.approx([0.40, 0.50])
    assert spec0['n'] == 2
    assert spec0['chi2'] == pytest.approx(2.0)
    assert spec0['chi2_red'] == pytest.approx(1.0)
    assert spec0['resid_rms'] == pytest.approx(1.0)
    assert spec1['label'] == 'spec_[0.55,0.8]'
    assert spec1['wl_range_rest'] == pytest.approx([0.70, 0.80])
    assert spec1['n'] == 2
    assert spec1['chi2'] == pytest.approx(0.0)


def test_fit_quality_band_edge_point_belongs_to_right_segment_only(tmp_path):
    """Segments partition the range: edge point in [b1,b2) and last is closed."""
    outdir = _write_results(tmp_path)
    # A spec point sits exactly on the edge 0.55; it must not double-count.
    _write_bestfit(outdir, [_spec_hdu(
        ibands=[0, 0, 0, 0],
        wl_rest=[0.40, 0.55, 0.70, 0.80],
        flux_obs=[10.0, 10.0, 10.0, 10.0],
        flux_mod=[10.0, 10.0, 10.0, 10.0],
        flux_obs_err=[1.0, 1.0, 1.0, 1.0],
    )], header={'XMIN^2/ND': 2.5})
    results = BayeSEDResults(outdir)
    datasets = results.get_fit_quality(
        OBJECT, band_edges=[0.55])['datasets']
    assert len(datasets) == 2
    assert datasets[0]['n'] == 1          # only 0.40
    assert datasets[0]['wl_range_rest'] == pytest.approx([0.40, 0.40])
    assert datasets[1]['n'] == 3          # 0.55, 0.70, 0.80
    assert datasets[1]['wl_range_rest'] == pytest.approx([0.55, 0.80])


def test_fit_quality_sigma_gates_frac_outside_but_not_outliers(tmp_path):
    outdir = _write_results(tmp_path)
    # resid: [-1.0, 0.5, 0.0]
    _write_bestfit(outdir, [_phot_hdu(
        bands=['u', 'u', 'u'],
        wl_rest=[0.35, 0.40, 0.45],
        flux_obs=[10.0, 10.0, 10.0],
        flux_model=[11.0, 9.0, 10.0],
        flux_obs_err=[1.0, 2.0, 2.0],
    )], header={'XMIN^2/ND': 2.5})
    results = BayeSEDResults(outdir)
    ds = results.get_fit_quality(
        OBJECT, top_n=1, sigma_threshold=0.9)['datasets'][0]
    assert ds['sigma_threshold'] == pytest.approx(0.9)
    assert ds['frac_outside'] == pytest.approx(1.0 / 3)
    # top_n=1 keeps only the largest |n_sigma|; outliers ignore --sigma
    assert ds['outliers'] == [
        {'wl_rest': 0.35, 'flux_obs': 10.0, 'n_sigma': -1.0},
    ]


def test_fit_quality_computes_chi2_without_header(tmp_path):
    outdir = _write_results(tmp_path)
    # resid: [-1.0, 0.5, 0.0] -> chi2 = 1.25 over n = 3
    _write_bestfit(outdir, [_phot_hdu(
        bands=['u', 'u', 'u'],
        wl_rest=[0.35, 0.40, 0.45],
        flux_obs=[10.0, 10.0, 10.0],
        flux_model=[11.0, 9.0, 10.0],
        flux_obs_err=[1.0, 2.0, 2.0],
    )])
    results = BayeSEDResults(outdir)
    record = results.get_fit_quality(OBJECT)
    assert record['chi2_source'] == 'computed'
    assert record['chi2'] == pytest.approx(1.25)
    assert record['chi2_red'] == pytest.approx(1.25 / 3)


def test_fit_quality_flags_line_residual_outliers(tmp_path):
    outdir = _write_results(tmp_path)
    # resid = 5.0 for both spec bands and the phot band
    _write_bestfit(outdir, [
        _phot_hdu(
            bands=['u'], wl_rest=[0.6565], flux_obs=[10.0],
            flux_model=[-15.0], flux_obs_err=[5.0]),
        _spec_hdu(
            ibands=[0, 1],
            wl_rest=[0.6565, 0.69],  # 0.6565 ~ H-alpha; 0.69 far from any line
            flux_obs=[10.0, 10.0],
            flux_mod=[-15.0, -15.0],
            flux_obs_err=[5.0, 5.0]),
    ], header={'XMIN^2/ND': 2.5})
    results = BayeSEDResults(outdir)
    datasets = results.get_fit_quality(OBJECT)['datasets']
    phot_ds, spec0, spec1 = datasets
    assert spec0['flags'] == ['line_residual_outliers']
    assert spec1['flags'] == []
    assert phot_ds['flags'] == []


def test_fit_quality_cli_end_to_end_json(tmp_path, capsys):

    outdir = _write_results(tmp_path)
    # resid: phot 'u' -> [-1.0, 0.5, 0.0]
    _write_bestfit(outdir, [_phot_hdu(
        bands=['u', 'u', 'u'],
        wl_rest=[0.35, 0.40, 0.45],
        flux_obs=[10.0, 10.0, 10.0],
        flux_model=[11.0, 9.0, 10.0],
        flux_obs_err=[1.0, 2.0, 2.0],
    )], header={'XMIN^2/ND': 2.5})

    code = run_inspect_impl(
        ['results', str(outdir), 'fit_quality', '--json'],
        prog='bayesed3',
    )
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert isinstance(payload, list)
    assert payload[0]['object_id'] == OBJECT
    assert payload[0]['chi2_source'] == 'header'
    assert payload[0]['datasets'][0]['kind'] == 'phot'
    assert payload[0]['datasets'][0]['label'] == 'u'
    assert payload[0]['datasets'][0]['outliers'][0]['n_sigma'] == -1.0


def test_fit_quality_cli_end_to_end_band_edges(tmp_path, capsys):

    outdir = _write_results(tmp_path)
    _write_bestfit(outdir, [
        _phot_hdu(
            bands=['u'], wl_rest=[0.5], flux_obs=[11.0], flux_model=[10.0],
            flux_obs_err=[1.0]),
        _spec_hdu(
            ibands=[0, 0, 1, 1],
            wl_rest=[0.40, 0.50, 0.70, 0.80],
            flux_obs=[10.0, 10.0, 10.0, 10.0],
            flux_mod=[11.0, 9.0, 10.0, 10.0],
            flux_obs_err=[1.0, 1.0, 2.0, 2.0]),
    ], header={'XMIN^2/ND': 2.5})

    code = run_inspect_impl(
        ['results', str(outdir), 'fit_quality',
         '--band', '0.55', '--json'],
        prog='bayesed3',
    )
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    labels = [(ds['kind'], ds['label']) for ds in payload[0]['datasets']]
    assert labels == [
        ('phot', 'u'), ('spec', 'spec_[0.4,0.55]'),
        ('spec', 'spec_[0.55,0.8]'),
    ]


def test_fit_quality_cli_end_to_end_missing_fits_is_one_line(tmp_path):

    outdir = _write_results(tmp_path)
    code = run_inspect_impl(
        ['results', str(outdir), 'fit_quality', '--json'],
        prog='bayesed3',
    )
    assert code == 1


def test_fit_quality_header_chi2_is_total_when_datasets_exist(tmp_path):
    """Top-level ``chi2`` is the total in BOTH branches: the header card is
    reduced (XMIN^2/ND), so the total is recovered via n_total (spec-axis
    review finding: mixed units under one key)."""
    outdir = _write_results(tmp_path)
    # resid: [-1.0, 0.5, 0.0] -> dataset chi2 = 1.25 over n = 3
    _write_bestfit(outdir, [_phot_hdu(
        bands=['u', 'u', 'u'],
        wl_rest=[0.35, 0.40, 0.45],
        flux_obs=[10.0, 10.0, 10.0],
        flux_model=[11.0, 9.0, 10.0],
        flux_obs_err=[1.0, 2.0, 2.0],
    )], header={'XMIN^2/ND': 2.5})
    results = BayeSEDResults(outdir)
    record = results.get_fit_quality(OBJECT)
    assert record['chi2_source'] == 'header'
    assert record['chi2_red'] == pytest.approx(2.5)
    assert record['chi2'] == pytest.approx(7.5)


def test_fit_quality_band_edges_with_no_valid_spec_points(tmp_path):
    """All-nonpositive spec errors: phot-only datasets, no crash (bugbot
    finding: _band_segments crashed on np.min of an empty array)."""
    outdir = _write_results(tmp_path)
    _write_bestfit(outdir, [
        _phot_hdu(bands=['u'], wl_rest=[0.5], flux_obs=[11.0],
                  flux_model=[10.0], flux_obs_err=[1.0]),
        _spec_hdu(ibands=[0, 0], wl_rest=[0.40, 0.60],
                  flux_obs=[10.0, 10.0], flux_mod=[10.0, 10.0],
                  flux_obs_err=[0.0, -1.0]),
    ], header={'XMIN^2/ND': 2.5})
    results = BayeSEDResults(outdir)
    record = results.get_fit_quality(OBJECT, band_edges=[0.5])
    assert [ds['kind'] for ds in record['datasets']] == ['phot']


def test_fit_quality_object_id_defaults_to_first_object_with_warning(
        tmp_path, caplog):
    outdir = _write_results(tmp_path)
    _write_bestfit(outdir, [_model_hdu('model:total')],
                   header={'XMIN^2/ND': 2.5})
    results = BayeSEDResults(outdir)
    with caplog.at_level('WARNING'):
        record = results.get_fit_quality()
    assert record['object_id'] == OBJECT
    assert 'first available object' in caplog.text


def test_fit_quality_prefers_results_object_id_without_warning(
        tmp_path, caplog):
    outdir = _write_results(tmp_path)
    _write_bestfit(outdir, [_model_hdu('model:total')],
                   header={'XMIN^2/ND': 2.5})
    results = BayeSEDResults(outdir, object_id=OBJECT)
    with caplog.at_level('WARNING'):
        record = results.get_fit_quality()
    assert record['object_id'] == OBJECT
    assert 'first available object' not in caplog.text
