---
# Binary CLI — Complete Options Reference
---

Run `./bin/linux/bayesed_mn_1 --help` for the authoritative reference.

## Invoking the Binary

```bash
# Single process (Linux)
./bin/linux/bayesed_mn_1 -v 2 -i 0,observation/test/gal.txt \
  --outdir observation/test/output/gal \
  --ssp 0,0,bc2003_hr_stelib_chab_neb_2000r,1,1,1,0,0,1,0,0 \
  --sfh 0,2,0,0 \
  --dal 0,2,8 \
  --multinest 1,0,0,400,0.1,0.5,1000,-1e90,1,2,0,0,-1e90,100000,0.01 \
  --save_sample_par

# MPI parallel (Linux)
mpirun -np 4 ./bin/linux/bayesed_mn_n -v 2 -i 0,observation/test/gal.txt ...

# macOS: replace bin/linux/ with bin/mac/
./bin/mac/bayesed_mn_1 ...
mpirun -np 4 ./bin/mac/bayesed_mn_n ...
```

## Passing Options via File (--import) from Python

```python
# options.txt:
# --output_SFH 10,0
# --IGM 5
# --priors_only

params = BayeSEDParams(
    input_file='observation/test/gal.txt',
    outdir='tests/output',
    import_files=['options.txt']
)
```

## Input / Output

| Option | Example | Description |
|--------|---------|-------------|
| `-i, --input` | `-i 0,obs/gal.txt` | Input file; first arg: 0=flux(μJy), 1=AB mag |
| `--outdir` | `--outdir result/` | Output directory |
| `--filters` | `--filters filters.txt` | Filter definition file |
| `--filters_selected` | `--filters_selected sel.txt` | Selected filters file |
| `--save_bestfit` | `--save_bestfit 0` | 0=FITS, 1=HDF5, 2=both |
| `--suffix` | `--suffix _v2` | Add suffix to output filenames |
| `--Ntest` | `--Ntest 2` | Process only first N objects |
| `-v, --verbose` | `-v 2` | Verbosity level |
| `-h, --help` | | Display usage |
| `--check` | | Print all inputs and categories |
| `--export` | `--export out.txt` | Export all options including defaults |
| `--import` | `--import opts.txt` | Import options from file (`#` = comment) |

## SED Components

### Stellar Population
| Option | Example | Description |
|--------|---------|-------------|
| `-ssp, --ssp` | `-ssp igroup,id,name,iscalable,k,f_run,Nstep,i0,i1,i2,i3` | SSP model; i1=1 enables nebular emission |
| `-sfh, --sfh` | `-sfh id,itype_sfh,itruncated,itype_ceh` | SFH type (0-9, see below) |
| `--np_sfh` | `--np_sfh 5,0,10,100` | Nonparametric SFH: prior_type, interp, Nbins, regul |

SFH types: 0=instantaneous, 1=constant, 2=exponential decay, 3=exponential rise, 4=single burst, 5=delayed, 6=beta, 7=lognormal, 8=double power-law, 9=nonparametric

### Dust Attenuation
| Option | Example | Description |
|--------|---------|-------------|
| `--dal` | `--dal id,con_eml_tot,ilaw` | Dust law: 1=Calzetti+2000(FAST), 2=MW(Cardelli), 3=SF(Salim+2018), 4=MW(Allen), 5=MW(Fitzpatrick), 6=LMC, 7=SMC, 8=SB(Calzetti2000), 9=SF(Reddy+2015) |

### AGN Components
| Option | Example | Description |
|--------|---------|-------------|
| `-bbb, --big_blue_bump` | `-bbb igroup,id,bbb,iscalable,w_min,w_max,Nw` | Big Blue Bump accretion disk |
| `-AGN, --AGN` | `-AGN igroup,id,AGN,iscalable,imodel,...` | Physical AGN: qsosed, agnsed, fagnsed, relagn, relqso, agnslim |
| `-ls1, --lines1` | `-ls1 igroup,id,name,iscalable,file,R,Nsample,Nkin` | Emission line series as one model (BLR/NLR) |
| `-ls, --lines` | `-ls igroup_start,id_start,file,iscalable,0,R,Nsample,Nkin` | Emission line series as multiple models |
| `-l, --line` | `-l igroup,id,name,iscalable,lam0/A,R,Nsample,Nkin` | Single emission line |
| `-k, --aknn` | `-k igroup,id,name,iscalable,k,f_run,...` | AKNN ML emulator (e.g. FeII) |
| `-a, --fann` | `-a igroup,id,name,iscalable` | FANN neural network model (e.g. torus) |
| `--kin` | `--kin id,velscale,N_GH_cont,N_GH_line` | LOSVD kinematics (Gauss-Hermite) |

### Other SED Components
| Option | Example | Description |
|--------|---------|-------------|
| `-bb, --blackbody` | `-bb igroup,id,bb,iscalable,w_min,w_max,Nw` | Blackbody spectrum |
| `-gb, --greybody` | `-gb igroup,id,gb,iscalable,ithick,w_min,w_max,Nw` | Greybody (dust emission) |
| `-pw, --powerlaw` | `-pw igroup,id,pw,iscalable,w_min,w_max,Nw` | Power law spectrum |
| `-p, --polynomial` | `--polynomial 3` | Multiplicative polynomial of order N |
| `-t, --template` | `-t igroup,id,M82,iscalable` | Template SED |
| `--sedlib` | `--sedlib igroup,id,name,iscalable,dir,itype,f_run,ikey` | Pre-built SED library |
| `--cloudy` | `--cloudy igroup,id,cloudy,iscalable` | CLOUDY photoionization model |
| `-r, --rbf` | `-r igroup,id,name,iscalable` | RBF model |
| `--inn` | `--inn igroup,id,name,iscalable,f_run,ikey` | INN model |
| `--rename` | `--rename id,ireplace,newname` | Rename a component |
| `--rename_all` | `--rename_all name` | Rename combination of all models |

## Priors and Redshift

| Option | Example | Description |
|--------|---------|-------------|
| `--z` | `--z iprior_type,is_age,min,max,nbin` | Redshift prior (default: `1,0,z_min,z_max,100`) |
| `--sys_err_obs` | `--sys_err_obs iprior_type,is_age,min,max,nbin` | Systematic error prior for observations (default: `1,0,0,0,40`) |
| `--sys_err_mod` | `--sys_err_mod iprior_type,is_age,min,max,nbin` | Systematic error prior for model (default: `1,0,0,0,40`) |
| `--priors_only` | | Test priors with zero likelihood (no data fitting) |

## MultiNest Sampling

```
--multinest IS,mmodal,ceff,nlive,efr,tol,updInt,Ztol,seed,fb,resume,outfile,logZero,maxiter,acpt_min
```
Default: `1,0,0,100,0.1,0.5,1000,-1e90,1,0,0,0,-1e90,100000,0.01`

| Field | Description |
|-------|-------------|
| `IS` | Importance sampling (1=on) |
| `mmodal` | Multi-modal sampling |
| `nlive` | Number of live points (40=test, 400+=production) |
| `efr` | Sampling efficiency (0.05–0.3) |
| `tol` | Evidence tolerance |
| `updInt` | Update interval |
| `fb` | Feedback: 0=silent, 2=verbose |
| `resume` | Resume from checkpoint |

| Option | Example | Description |
|--------|---------|-------------|
| `--Ndumper` | `--Ndumper 1,0,-1` | Max dumps, iconverged_min, Xmin²/Nd |
| `--logZero` | `--logZero 100000` | Max Nsigma for logZero |
| `--cl` | `--cl 0.68,0.95` | Confidence levels for output |
| `--unweighted_samples` | | Use unweighted posterior samples |

## Spectral Fitting

| Option | Example | Description |
|--------|---------|-------------|
| `--SNRmin1` | `--SNRmin1 0,3` | Min SNR (phot,spec) for scaling determination |
| `--SNRmin2` | `--SNRmin2 0,3` | Min SNR (phot,spec) for likelihood evaluation |
| `--no_photometry_fit` | | Skip photometric data |
| `--no_spectra_fit` | | Skip spectral data |
| `--rdf` | `--rdf id,N` | Polynomial order for spectral residual modeling |
| `--lw_max` | `--lw_max 10000` | Max line coverage in km/s (default: 10000) |
| `--NNLM` | `--NNLM 0,10000,0,10,0.01,0.05,0.95` | Non-negative scaling method |
| `--niteration` | `--niteration 0` | Number of iterations |
| `--NfilterPoints` | `--NfilterPoints 30` | Filter integration points |
| `--gsl_integration_qag` | `--gsl_integration_qag 0,0.1,1000` | GSL integration settings |
| `--gsl_multifit_robust` | `--gsl_multifit_robust ols,1` | Robust fitting weight function |

## IGM Attenuation

```
--IGM [0-5]
  0: None
  1: Madau (1995) — default
  2: Meiksin (2006)
  3: hyperz
  4: FSPS
  5: Inoue+2014
```

## Cosmology

```
--cosmology H0,omegaLambda,omegaM
```
Default: `70,0.7,0.3`

## Output Options

| Option | Example | Description |
|--------|---------|-------------|
| `--output_SFH` | `--output_SFH 10,0` | Output SFH posterior as derived params (Ngrid,ilog) |
| `--output_mock_photometry` | `--output_mock_photometry 0` | Output mock photometry (0=μJy, 1=AB mag) |
| `--output_mock_spectra` | | Output mock spectra (μJy) |
| `--output_model_absolute_magnitude` | | Output model absolute magnitudes |
| `--output_pos_obs` | | Output posterior estimation of observables |
| `--save_pos_sfh` | `--save_pos_sfh 100,1` | Save posterior SFH distribution |
| `--save_pos_spec` | | Save posterior model spectra (memory intensive) |
| `--save_sample_obs` | | Save posterior sample of observables |
| `--save_sample_par` | | Save posterior sample of parameters |
| `--save_sample_spec` | | Save posterior sample of model spectra |
| `--save_summary` | | Save summary file |
| `--SFR_over` | `--SFR_over 10,100` | Compute average SFR over past N Myr |
| `-L, --luminosity` | `--luminosity id,w_min,w_max` | Compute luminosity between wavelengths |

## Catalog / Library Tools

| Option | Example | Description |
|--------|---------|-------------|
| `--make_catalog` | `--make_catalog id,logscale_min,logscale_max` | Make catalog from model SEDs |
| `--build_sedlib` | `--build_sedlib 0` | Build SED library (0=rest, 1=observed) |
| `--Nsample` | `--Nsample 1000` | Number of samples for catalog/sedlib |

## CLOUDY Integration

| Option | Example | Description |
|--------|---------|-------------|
| `--LineList` | `--LineList file,type` | Line list file for CLOUDY model (0=intrinsic, 1=emergent, 2=intrinsic cumulative, 3=emergent cumulative) |

## Complete Worked Examples

### 1. Galaxy Spectroscopic Fit (flux μJy input)

```bash
./bin/linux/bayesed_mn_1 -v 2 \
  -i 0,observation/test/gal.txt \
  --outdir observation/test/output/gal \
  --ssp 0,0,bc2003_hr_stelib_chab_neb_2000r,1,1,1,1,0,1,0,0 \
  --sfh 0,2,0,0 \
  --dal 0,2,8 \
  --rename 0,1,Stellar+Nebular \
  --sys_err_obs 1,0,0.0,0.2,40 \
  --multinest 1,0,1,40,0.05,0.5,100,-1e90,1,2,0,0,-1e90,100000,0.01 \
  --save_bestfit 0 --save_sample_par
```

### 2. QSO Multi-Component Spectroscopic Fit

```bash
./bin/linux/bayesed_mn_1 -v 2 \
  -i 0,observation/test/qso.txt \
  --outdir observation/test/output/qso \
  --ssp 0,0,bc2003_hr_stelib_chab_neb_2000r,1,1,1,1,0,1,0,0 \
  --sfh 0,2,0,0 --dal 0,2,8 \
  --rename 0,1,Stellar+Nebular \
  -bbb 1,1,bbb,1,0.1,10,1000 --dal 1,2,7 \
  -ls1 2,2,BLR,1,observation/test/lines_BLR.txt,300,2,3 \
  -k 3,3,FeII,1,1,1,0,0,1,1,1 --kin 3,10,2,0 \
  -ls1 4,4,NLR,1,observation/test/lines_NLR.txt,2000,2,2 \
  --sys_err_obs 1,0,0.0,0.2,40 \
  --multinest 1,0,1,40,0.05,0.5,100,-1e90,1,2,0,0,-1e90,100000,0.01 \
  --save_bestfit 0 --save_sample_par
```

Component ID assignment: host galaxy `igroup=0/id=0`, BBB `igroup=1/id=1`, BLR `igroup=2/id=2`, FeII `igroup=3/id=3`, NLR `igroup=4/id=4`. Each component gets its own `--dal`.

### 3. Photometric-Only Fit (AB magnitude input, no spectra)

```bash
./bin/linux/bayesed_mn_1 \
  -i 1,observation/test1/test_inoise1.txt \
  --filters observation/test1/filters_COSMOS_CSST_Euclid_LSST_WFIRST.txt \
  --filters_selected observation/test1/filters_CSST_seleted.txt \
  --ssp 0,0,bc2003_lr_BaSeL_chab,1,1,1,1,0,0,0,0 \
  --sfh 0,2,0,0 --dal 0,2,8 \
  --z 1,0,0,4,40 \
  --no_spectra_fit \
  --suffix _CSST \
  --outdir test1 \
  --multinest 1,0,0,50,0.1,0.5,1000,-1e90,1,0,0,0,-1e90,100000,0.01 \
  --save_bestfit 2 --save_sample_par
```

Key: `-i 1,...` for AB magnitude input; `--no_spectra_fit` skips spectral data; `--suffix` distinguishes survey runs.

### 4. Galaxy + Dust Emission + AGN Torus (photometric)

```bash
./bin/linux/bayesed_mn_1 \
  -i 0,observation/test2/test.txt \
  --filters observation/test2/filters.txt \
  --filters_selected observation/test2/filters_selected.txt \
  --ssp 0,0,bc2003_lr_BaSeL_chab,1,3,1,0,0,0,0,0 \
  --sfh 0,2,0,0 --dal 0,2,7 \
  -gb 0,1,gb,-2,1,1,1000,200 \
  -a 1,2,clumpy201410tor,1 \
  --outdir test2 \
  --multinest 1,0,0,400,0.1,0.5,1000,-1e90,1,2,0,0,-1e90,100000,0.01 \
  --save_bestfit 0 --save_sample_par
```

`-gb igroup,id,gb,iscalable,ithick,w_min,w_max,Nw` — greybody dust emission; `-a igroup,id,name,iscalable` — FANN torus model.

### 5. CSST Mock Fit with SFR Output and Spectral Residual Modeling

```bash
./bin/linux/bayesed_mn_1 -v 2 \
  --Ntest 1 \
  -i 1,observation/test3/test_STARFORMING.txt \
  --filters observation/test3/filters_bassmzl.txt \
  --filters_selected observation/test3/filters_selected_csst.txt \
  --ssp 0,0,bc2003_hr_stelib_chab_neb_300r,0,1,1,0,0,1,0,0 \
  --sfh 0,2,0,1 --dal 0,2,8 \
  --z 1,0,0,1,40 \
  --NNLM 1,10000,0,10,0.01,0.025,0.975 --SNRmin1 0,3 \
  --rdf -1,0 \
  --no_spectra_fit \
  --suffix _phot \
  --outdir test3 \
  --multinest 1,0,0,40,0.1,0.5,10000,-1e90,1,2,0,0,-1e90,100000,0.01 \
  --SFR_over 100 \
  --save_bestfit 0 --save_sample_par
```

`--SFR_over 100` outputs average SFR over past 100 Myr; `--rdf -1,0` enables spectral residual polynomial for all models; `--NNLM` controls non-negative scaling.

### 6. AGN Host Decomposition (galaxy + dust + torus + QSO template)

```bash
./bin/linux/bayesed_mn_1 -v 2 \
  --Ntest 4 \
  -i 0,observation/agn_host_decomp/sample.txt \
  --filters observation/agn_host_decomp/filters.txt \
  --filters_selected observation/agn_host_decomp/filters_selected_total_only.txt \
  --ssp 0,0,bc2003_lr_BaSeL_chab,1,1,1,0,0,0,0,0 \
  --sfh 0,8,0,1 --dal 0,2,8 \
  -gb 0,1,gb,-2,1,1,1000,200 \
  -a 1,2,clumpy201410tor,1 \
  -t 1,3,QSO1,-1 --dal 3,2,7 \
  --sys_err_obs 3,0,0.01,0.2,40 \
  --luminosity -1,0.25,0.25 \
  --outdir observation/agn_host_decomp/output \
  --suffix _total_only \
  --multinest 1,0,0,400,0.3,0.5,1000,-1e90,1,1,0,0,-1e90,100000,0.01 \
  --save_bestfit 0 --save_sample_par
```

`-t igroup,id,name,iscalable` — template SED (iscalable=-1 means log-scale); `--luminosity -1,0.25,0.25` computes luminosity at 0.25μm for all models (id=-1); `sfh 0,8,0,1` = double power-law SFH with chemical evolution.

### MPI Parallel Execution

Replace `bayesed_mn_1` with `bayesed_mn_n` and prepend `mpirun`:

```bash
mpirun --use-hwthread-cpus ./bin/linux/bayesed_mn_n -v 2 \
  -i 0,observation/test/gal.txt \
  --outdir observation/test/output/gal \
  ... (same options as single-process)
```
