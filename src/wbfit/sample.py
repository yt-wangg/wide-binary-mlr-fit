"""Input catalogue and sample definition.

The input table ``data/wide_binaries.fits`` holds one row per pair of the El-Badry et al. (2021) catalogue after the
cuts of Table 1 of the paper (except the u/sigma_u and R_chance_align cuts, which are applied here).  The row index
is the identifier of a pair throughout the pipeline.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from astropy.table import Table

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "wide_binaries.fits"

MG_DOMAIN = (3.5, 13.5)          # absolute-magnitude domain of the relation
SNR_CUT = 3.0                    # u / sigma_u > 3 with the inflated sigma_u
MAX_R_CHANCE = 0.01              # R_chance_align of the MLR sample
ORBIT_CUT_U = 65.9525            # u at Delta mu = Delta mu_orb (El-Badry et al. 2021), km/s au^1/2
ORBIT_CUT_NSIGMA = 2.0
JCAPS_QUALITY_BOTH = ("jc_fit_converged", "jc_sigma_reliable", "jc_plx_reliable")
JCAPS_QUALITY_PRIMARY = ("jc_ext_reliable",)
# parallax-uncertainty inflation f(G) of El-Badry, Rix & Heintz (2021, Eq. 16)
ELBADRY2021_INFLATION = dict(A=0.21, G0=12.65, b=0.90, p0=1.141, p1=0.0040, p2=-0.00062)


def read_table(path=DATA) -> Table:
    return Table.read(path)


def col(table, name, rows=None):
    x = np.asarray(table[name], dtype=np.float64)
    return x if rows is None else x[rows]


def distance_pc(table):
    """One distance per pair from the parallax of the primary corrected for the Lindegren et al. (2021) zero point
    (column parallax_zpt1, computed with the gaiadr3-zeropoint package)."""
    return 1000.0 / (col(table, "parallax1") - col(table, "parallax_zpt1"))


def elbadry2021_inflation(gmag):
    c = ELBADRY2021_INFLATION
    g = np.clip(np.asarray(gmag, dtype=np.float64), 7.0, 21.0)
    return c["A"] * np.exp(-((g - c["G0"]) ** 2) / c["b"] ** 2) + c["p0"] + c["p1"] * g + c["p2"] * g**2


def sigma_u_inflation(table, rows):
    """Factor applied to sigma_u: the El-Badry et al. (2021) f(G) applied to the proper-motion uncertainties of both
    stars, projected on the direction of the proper-motion difference."""
    c = lambda name: col(table, name, rows)
    d_ra, d_dec = c("pmra2") - c("pmra1"), c("pmdec2") - c("pmdec1")
    norm = np.hypot(d_ra, d_dec)
    n_ra, n_dec = d_ra / norm, d_dec / norm
    var = []
    for k in (1, 2):
        e2 = n_ra**2 * c(f"pmra_error{k}") ** 2 + n_dec**2 * c(f"pmdec_error{k}") ** 2
        var.append((e2, elbadry2021_inflation(c(f"phot_g_mean_mag{k}"))))
    return np.sqrt((var[0][1] ** 2 * var[0][0] + var[1][1] ** 2 * var[1][0]) / (var[0][0] + var[1][0]))


def pairs(table) -> dict:
    """Quantities of all pairs with valid inputs, in row order.

    u and sigma_u are rescaled from the catalogue (raw parallax of the primary) to the zero-point-corrected distance,
    u ~ d^(3/2), and sigma_u is inflated; M_G = G + 5 - 5 log10 d - A_G for both stars.
    """
    n = len(table)
    rows = np.arange(n, dtype=np.int64)
    d = distance_pc(table)
    scale = (d * col(table, "parallax1") / 1000.0) ** 1.5
    absg = np.column_stack([col(table, f"phot_g_mean_mag{k}") + 5.0 - 5.0 * np.log10(d) - col(table, f"a_g_edhf_{k}")
                            for k in (1, 2)])
    feh = np.column_stack([col(table, "feh_jcaps_1"), col(table, "feh_jcaps_2")])
    feh_sigma = np.column_stack([col(table, "jc_sigma_m_h_1"), col(table, "jc_sigma_m_h_2")])
    u = col(table, "u") * scale
    u_sigma = col(table, "u_sigma") * scale
    valid = (np.all(np.isfinite(absg), 1) & np.all(np.isfinite(feh), 1) & np.all(np.isfinite(feh_sigma), 1)
             & np.isfinite(u) & np.isfinite(u_sigma) & np.all(feh_sigma > 0, 1) & (u > 0) & (u_sigma > 0)
             & np.all((absg >= MG_DOMAIN[0]) & (absg <= MG_DOMAIN[1]), 1))
    out = dict(rows=rows, d_pc=d, absg=absg, feh=feh, feh_sigma=feh_sigma, u=u, u_sigma=u_sigma)
    return {k: v[valid] for k, v in out.items()}


def jcaps_quality(table, rows):
    keep = np.ones(rows.size, dtype=bool)
    for name in JCAPS_QUALITY_BOTH:
        for k in (1, 2):
            keep &= np.asarray(table[f"{name}_{k}"], dtype=bool)[rows]
    for name in JCAPS_QUALITY_PRIMARY:
        keep &= np.asarray(table[f"{name}_1"], dtype=bool)[rows]
    return keep


def metallicity_sample(table) -> dict:
    """Sample of Stage 1b and of the dynamics lookup: valid pairs with reliable J-CaPS fits and u/sigma_u > 3 for
    the inflated sigma_u (10 613 pairs)."""
    p = pairs(table)
    keep = jcaps_quality(table, p["rows"])
    p = {k: v[keep] for k, v in p.items()}
    p["u_sigma"] = p["u_sigma"] * sigma_u_inflation(table, p["rows"])
    keep = p["u"] / p["u_sigma"] > SNR_CUT
    return {k: v[keep] for k, v in p.items()}


def orbit_upper(table, rows):
    """Upper limit U_j on u from the bound-orbit criterion of the catalogue, u_cat < 65.95 + 2 sigma_u,cat with the
    catalogue quantities, converted to the zero-point-corrected distance.  Pairs whose catalogue u already exceeds
    the nominal limit keep their own u as the limit."""
    u_cat, s_cat = col(table, "u", rows), col(table, "u_sigma", rows)
    upper = np.maximum(ORBIT_CUT_U + ORBIT_CUT_NSIGMA * s_cat, u_cat * (1.0 + 1e-9))
    d = distance_pc(table)[rows]
    return upper * (d * col(table, "parallax1", rows) / 1000.0) ** 1.5


def mlr_sample(table, metallicity_rows=None) -> dict:
    """Sample of the MLR fit: the metallicity sample with R_chance_align <= 0.01 (10 559 pairs), with the upper
    limit U_j of each pair and its index in the metallicity sample (``pos``)."""
    p = metallicity_sample(table)
    if metallicity_rows is not None and not np.array_equal(p["rows"], metallicity_rows):
        raise ValueError("The metallicity posterior does not belong to this sample; rerun step 2.")
    p["pos"] = np.arange(p["rows"].size)
    keep = col(table, "R_chance_align", p["rows"]) <= MAX_R_CHANCE
    p = {k: v[keep] for k, v in p.items()}
    p["u_upper"] = orbit_upper(table, p["rows"])
    p["s_au"] = col(table, "sep_arcsec", p["rows"]) * p["d_pc"]
    p["g_app"] = np.column_stack([col(table, "phot_g_mean_mag1", p["rows"]), col(table, "phot_g_mean_mag2", p["rows"])])
    return p
