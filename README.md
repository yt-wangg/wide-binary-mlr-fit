# wide-binary-mlr-fit

Code and data to reproduce the dynamical mass–luminosity relation (MLR) of
*Dynamical calibration of a metallicity-dependent stellar mass–luminosity relation from Gaia wide binaries*
(Wang, Rix, Li & Ting, in prep.).

The relation itself, with functions to evaluate it, is distributed separately as the Python package
[`wbmlr`](https://github.com/yt-wangg/wide-binary-mlr). This repository contains only what is needed to obtain it
from the input catalogue: six scripts that run in order, a small package `wbfit` with the model, and the input data.

## Install

```bash
pip install -e .        # numpy, scipy, astropy, jax, numpyro
```

For steps 2, 4 and 6 install JAX with GPU support (see the JAX documentation). Tested with jax 0.4.35 and
numpyro 0.19.

## Pipeline

```bash
./run_all.sh            # or run the scripts one by one
```

| Step | Script | What it does | Output (`results/`) | Time |
|---|---|---|---|---|
| 1 | `1_metallicity_calibration.py` | Stage 1a: J-CaPS bias $b_G(M_G)$, extra scatter $s_z$ and Student-$t$ $\nu$ from the metallicity differences within pairs (maximum likelihood) | `stage1a.json` | 1 min, CPU |
| 2 | `2_metallicity_posterior.py` | Stage 1b: metallicity distribution of the sample and the posterior $P(Z)$ of each pair on 81 nodes | `metallicity_posterior.npz` | minutes, GPU |
| 3 | `3_good_shape.py` | $p(\tilde u)$ of bound orbits ($p(e)\propto e^{1.233}$, $dN/d\ln a\propto a^{-0.6}$), weighted to the observed separations and distances | `good_shape.json` | 15 s, CPU |
| 4 | `4_dynamics_lookup.py` | Likelihood of the good component, convolved with the measurement error, for every pair on 1024 values of $\sqrt{M_{\rm tot}}$ | `good_lookup.npz` | minutes, GPU |
| 5 | `5_hidden_companions.py` | Forward model of hidden inner companions; local outlier template $h_j(\tilde u)$ and fraction $r_j$ of every pair | `outlier_templates.npz` | 1 min, CPU |
| 6 | `6_fit_mlr.py` | Selection and outlier tables, then the NUTS fit of the monotone spline MLR (4 chains × 1000 + 7000 iterations) | `mlr_posterior.npz`, `mlr_grid.csv` | hours, GPU |

`mlr_posterior.npz` holds the posterior draws of all parameters and of the $8\times4$ spline coefficients
`theta` of $\log_{10}(M/M_\odot)$ on the knots `knots_x`, `knots_z` (cubic, clamped). `mlr_grid.csv` gives the
16, 50 and 84 per cent of the mass on a grid of 0.05 mag in $M_G$ and 0.05 dex in $Z$, together with PARSEC.
For a quick test of step 6, run `python scripts/6_fit_mlr.py --warmup 20 --samples 20 --chains 1`.

All settings are those of the paper and are fixed in the code: the sample cuts in `src/wbfit/sample.py`, the
hidden-companion model in `src/wbfit/companions.py`, the priors in `src/wbfit/mlr.py`. Steps 1, 3 and 5 are
deterministic; steps 2, 4 and 6 use fixed random seeds.

## Model (`src/wbfit`)

| Module | Content | Paper |
|---|---|---|
| `sample.py` | distances, $u$ and $\sigma_u$ (inflated), sample cuts, bound-orbit limit $U_j$ | Sect. 2, 3.5 |
| `metallicity.py` | Stage 1a and Stage 1b | Sect. 3.2 |
| `shape.py` | $p(\tilde u)$ of bound orbits | Sect. 3.3, App. A |
| `dynamics.py` | Rice-convolved likelihood tables, selection probabilities, outlier tables | Sect. 3.3–3.5 |
| `companions.py` | forward model of hidden companions and local templates | Sect. 3.4, App. D |
| `parsec.py` | PARSEC reference mass surface | Sect. 3.1 |
| `mlr.py` | monotone tensor-spline MLR, priors and the NumPyro likelihood | Sect. 3.1, 3.6 |

## Data (`data/`)

`wide_binaries.fits` contains the 15 036 pairs of the El-Badry et al. (2021) catalogue that pass the cuts of
Table 1 of the paper on separation, parallax, astrometric fidelity, the XP single-star classification, extinction
and photometry, and $u/\sigma_u>3$ with the catalogue uncertainties. The cuts on the J-CaPS flags, on $M_G$, on
$u/\sigma_u>3$ with the inflated uncertainties and on `R_chance_align` are applied by the code: Stage 1b uses 10 613
pairs and the MLR fit 10 559. The row number identifies a
pair in all outputs. Suffixes 1 and 2 (or `_1`, `_2`) denote the primary and the secondary.

| Columns | Content |
|---|---|
| `source_id`, `ra`, `dec` | Gaia DR3 identifiers and positions |
| `parallax1`, `parallax_error1`, `parallax_zpt1` | parallax of the primary and its zero point (Lindegren et al. 2021, `gaiadr3-zeropoint`) |
| `pmra`, `pmdec`, `pmra_error`, `pmdec_error` | proper motions |
| `phot_g_mean_mag`, `a_g_edhf_`, `bp_rp0_` | apparent $G$, extinction $A_G$ from Edenhofer et al. (2024), intrinsic BP–RP |
| `absg1`, `absg2` | $M_G$ of the catalogue (raw parallax of the primary), used by Stage 1a |
| `ruwe` | Gaia RUWE |
| `sep_arcsec`, `sep_AU` | angular and projected separation |
| `u`, `u_sigma` | $u=\Delta v\sqrt{s}$ and its uncertainty from the catalogue quantities [km s$^{-1}$ au$^{1/2}$] |
| `R_chance_align` | probability of a chance alignment (El-Badry et al. 2021) |
| `feh_jcaps_`, `jc_sigma_m_h_` | raw J-CaPS metallicity and its uncertainty (Li et al. 2026) |
| `jc_at_bound_bits_`, `jc_fit_converged_`, `jc_sigma_reliable_`, `jc_plx_reliable_`, `jc_ext_reliable_1` | J-CaPS quality flags |

`parsec_mass/` holds the mass of PARSEC v1.2S isochrones of age $10^{9.6}$ yr as a function of $M_G$ for
$[{\rm M/H}]=-1.0,\dots,0.6$, the reference surface of the relation.

## Citation

Wang, Rix, Li & Ting (in prep.), *Dynamical calibration of a metallicity-dependent stellar mass–luminosity relation
from Gaia wide binaries*.
