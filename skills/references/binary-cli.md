---
# Run argv assembly
---

Open only after a Prefer `run` gate in `SKILL.md` (ADR-0014). Ordinary new fits stay `SEDInference`.

This file is **not** a flag encyclopedia and **not** the Inspect manual (`launcher-inspect.md`). Flag *shapes* (tuple order, enums, defaults) come from **live** `bayesed3 run --help` in this session. Do not paste that help into notes; query it, then assemble.

`bayesed3 run` is the package’s cross-platform Launcher: it needs the same Python environment as `SEDInference`. After `run`, Launcher owns `--mpi` / `--np`; the rest is **Run argv** forwarded to the compiled binary.

## Live `--help` caveats

- Binary USAGE omits the `run` token. Always invoke `bayesed3 run …`.
- `--outdir` example in help may show `--ourdir` — the flag is `--outdir`.
- `--mpi` / `--np` are not in binary help. Put them immediately after `run`.
- Help example SSP name `ynII` is a shape placeholder. Use a real model name (below).

## Assembly rules (`--help` does not give these)

**`igroup` / `id`:** each component needs a unique `id`. Components that share dust/kinematics use the same `igroup`. Host galaxy typically `igroup=0,id=0`; further components `1,1`, `2,2`, … Each component that is attenuated gets its own `--dal`. Runtime names may print the host as `[0,1]` when argv used `id=0` — keep the worked-line indices; do not re-number argv to match the printout.

**`iscalable`:** `1` = linear scale (free), `0` = not scaled, `-1` = log scale, `-2` = scaled by dust mass (greybody).

**`--ssp igroup,id,name,iscalable,k,f_run,Nstep,i0,i1,i2,i3`:** defaults `k=1,f_run=1,Nstep=1,i0=0,i1=0,i2=0,i3=0`. `i1=1` enables nebular emission. Common `name` values: `bc2003_hr_stelib_chab_neb_2000r` (high-res + nebular), `bc2003_hr_stelib_chab_neb_300r`, `bc2003_lr_BaSeL_chab` (low-res, no nebular).

**`--dal id,con_eml_tot,ilaw`:** `con_eml_tot=2` applies to continuum+lines (typical). `ilaw` enums are in `--help` (`8` = Calzetti2000 hyperz, matching Python `dal_law='calzetti'`).

**`--sfh id,itype_sfh,itruncated,itype_ceh`:** `itype_sfh` enums are in `--help` (`2` = exponentially declining).

**`--multinest`:** slot names and order from live `--help`: `IS,mmodal,ceff,nlive,efr,tol,updInt,Ztol,seed,fb,resume,outfile,logZero,maxiter,acpt_min` (`nlive` = 4th, `efr` = 5th). Meanings: `multinest-params.md`. When to omit or patch: `SKILL.md` FAQ / `advanced-run-modes.md`.

**`--sys_err_obs`:** spectroscopic worked lines turn on a free 0–20% observational systematic (`1,0,0.0,0.2,40`). Live `--help` default is off. Copy the worked tuple for spec; the photometric example omits it.

**`--save_sample_par`:** required when posteriors or model-configuration comparison are required. Worked lines include it; omit only if posterior samples are not needed.

## Default `.iprior` files

The compiled binary writes any missing `.iprior` files for the Model configuration, then fits with those defaults. One `bayesed3 run` with the Input catalog is enough — no dummy Input catalog, no `--priors_only`, no iteration cap. `--priors_only` in `--help` is a zero-likelihood prior test, not the default-generation path.

`--z` and `--sys_err_obs` / `--sys_err_mod` may be on Run argv. Changing a free-parameter prior type or bounds → leave this branch; use Python `set_prior`. Do not hand-edit `.iprior` files. `bayesed3 priors list <dir>` only prints.

## After the fit

Inspect: `bayesed3 results <outdir> summary` (and `evidence` when asked). Grammar: `launcher-inspect.md`. Optional Python: `BayeSEDResults(outdir)` when this session already uses Python or needs GetDist.

## Worked `bayesed3 run` lines

Galaxy spectroscopic (flux μJy).

```bash
bayesed3 run -v 2 \
  -i 0,observation/test/gal.txt \
  --outdir observation/test/output/gal \
  --ssp 0,0,bc2003_hr_stelib_chab_neb_2000r,1,1,1,1,0,1,0,0 \
  --sfh 0,2,0,0 \
  --dal 0,2,8 \
  --sys_err_obs 1,0,0.0,0.2,40 \
  --save_bestfit 0 --save_sample_par
```

QSO multi-component: host `0,0`; BBB `1,1`; BLR `2,2`; FeII `3,3`; NLR `4,4`. Each attenuated component has `--dal`. Requires the line-list files on `-ls1` (`observation/test/lines_BLR.txt`, `observation/test/lines_NLR.txt` for this fixture).

```bash
bayesed3 run -v 2 \
  -i 0,observation/test/qso.txt \
  --outdir observation/test/output/qso \
  --ssp 0,0,bc2003_hr_stelib_chab_neb_2000r,1,1,1,1,0,1,0,0 \
  --sfh 0,2,0,0 --dal 0,2,8 \
  -bbb 1,1,bbb,1,0.1,10,1000 --dal 1,2,7 \
  -ls1 2,2,BLR,1,observation/test/lines_BLR.txt,300,2,3 \
  -k 3,3,FeII,1,1,1,0,0,1,1,1 --kin 3,10,2,0 \
  -ls1 4,4,NLR,1,observation/test/lines_NLR.txt,2000,2,2 \
  --sys_err_obs 1,0,0.0,0.2,40 \
  --save_bestfit 0 --save_sample_par
```

Photometric AB mag, no spectra: `-i 1,…` and `--no_spectra_fit`. Filters from `--help` `--filters` / `--filters_selected`.

```bash
bayesed3 run \
  -i 1,observation/test1/test_inoise1.txt \
  --filters observation/test1/filters_COSMOS_CSST_Euclid_LSST_WFIRST.txt \
  --filters_selected observation/test1/filters_CSST_seleted.txt \
  --ssp 0,0,bc2003_lr_BaSeL_chab,1,1,1,1,0,0,0,0 \
  --sfh 0,2,0,0 --dal 0,2,8 \
  --z 1,0,0,4,40 \
  --no_spectra_fit \
  --outdir test1 \
  --save_bestfit 2 --save_sample_par
```

Quick test flags are not in the recipes; see `SKILL.md` FAQ.

MPI: `bayesed3 run --mpi n --np 4` then the same Run argv.

`--import file` injects extra flags (`#` comments).
