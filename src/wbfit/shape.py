"""Distribution p(u~) of u~ = u / sqrt(M_tot) for bound orbits (Appendix A of the paper).

A population with log-uniform distance, semi-major axes dN/dln a ~ a^gamma, eccentricities p(e) ~ e^alpha, random
orbital phase and orientation is projected, reweighted to the observed distribution of (log s, log d), and the
weighted histogram of u~ on [0, 40] is fitted with p(u~) = A u~ exp(-B u~^2 - exp((u~ - u_c)/C)).
"""
from __future__ import annotations

import numpy as np
from scipy.integrate import quad
from scipy.optimize import curve_fit

V0 = 29.785            # km/s: circular speed at 1 au around 1 Msun
SUPPORT = 80.0         # u~ support of the good component
ALPHA_E = 1.233        # adopted eccentricity index
SMA_INDEX = -0.6       # dN/dln a ~ a^-0.6
N_ORBITS = 4_000_000
SEED = 20261009
N_S, N_D = 24, 18


def _true_anomaly(e, M):
    E = M.copy()
    for _ in range(60):
        E = E - (E - e * np.sin(E) - M) / (1 - e * np.cos(E))
    return np.arctan2(np.sqrt(1 - e**2) * np.sin(E), np.cos(E) - e)


def _project(r, a, alpha, mtot=2.0):
    e = np.minimum(r["e"] ** (1.0 / (np.asarray(alpha) + 1.0)), 0.999)
    f = _true_anomaly(e, 2 * np.pi * r["M"])
    Om, om = 2 * np.pi * r["Om"], 2 * np.pi * r["om"]
    ci = 2 * r["ci"] - 1
    p = a * (1 - e**2)
    r0 = p / (1 + e * np.cos(f))
    w = om + f
    rx = r0 * (np.cos(Om) * np.cos(w) - ci * np.sin(Om) * np.sin(w))
    ry = r0 * (np.sin(Om) * np.cos(w) + ci * np.cos(Om) * np.sin(w))
    v0 = V0 * np.sqrt(mtot / p)
    vx = -v0 * (np.cos(Om) * (np.sin(w) + e * np.sin(om)) + ci * np.sin(Om) * (np.cos(w) + e * np.cos(om)))
    vy = -v0 * (np.sin(Om) * (np.sin(w) + e * np.sin(om)) - ci * np.cos(Om) * (np.cos(w) + e * np.cos(om)))
    s = np.hypot(rx, ry)
    return np.hypot(vx, vy) * np.sqrt(s) / np.sqrt(mtot), s


def _form(x, A, B, C, uc):
    return A * x * np.exp(-B * x**2 - np.exp((x - uc) / C))


def fit_good_shape(sep_au, dist_pc, *, alpha=ALPHA_E, sma_index=SMA_INDEX, n=N_ORBITS, seed=SEED):
    """Return the fitted (A, B, C, uc) for the observed separations and distances of the catalogue."""
    sep_au, dist_pc = np.asarray(sep_au, float), np.asarray(dist_pc, float)
    se = np.linspace(np.log10(sep_au.min()), np.log10(sep_au.max()), N_S + 1)
    de = np.linspace(np.log10(dist_pc.min()), np.log10(dist_pc.max()), N_D + 1)
    H_data, _, _ = np.histogram2d(np.log10(sep_au), np.log10(dist_pc), [se, de], density=True)
    rng = np.random.default_rng(seed)
    a = 10 ** rng.uniform(1.0, 5.0, n)
    d = 10 ** rng.uniform(np.log10(5.0), np.log10(1000.0), n)
    shared = {k: rng.random(n) for k in ("e", "M", "Om", "om", "ci")}
    ut, s = _project(shared, a, np.full(n, alpha))
    m = (s >= sep_au.min()) & (s <= sep_au.max()) & (d >= dist_pc.min()) & (d <= dist_pc.max())
    w0 = a ** sma_index
    H_sim, _, _ = np.histogram2d(np.log10(s[m]), np.log10(d[m]), [se, de], weights=w0[m], density=True)
    i = np.clip(np.digitize(np.log10(s[m]), se) - 1, 0, N_S - 1)
    j = np.clip(np.digitize(np.log10(d[m]), de) - 1, 0, N_D - 1)
    w = w0[m] * H_data[i, j] / np.maximum(H_sim[i, j], 1e-12)
    h, ed = np.histogram(ut[m], bins=100, range=(0, 40), density=True, weights=w)
    c = 0.5 * (ed[1:] + ed[:-1])
    A, B, C, uc = curve_fit(_form, c, h, p0=[4.95e-3, 2.24e-3, 3.85, 36.09], maxfev=20000)[0]
    return dict(A=float(A), B=float(B), C=float(C), uc=float(uc))


def rounded(shape):
    """Constants as used in the paper: B to four significant digits, u_c and C to four decimals."""
    return dict(B=float(f"{shape['B']:.4g}"), uc=round(shape["uc"], 4), C=round(shape["C"], 4))


class GoodShape:
    """Normalised good-component density p0(u~) on 0 < u~ <= 80."""

    def __init__(self, B, uc, C):
        self.B, self.uc, self.C = float(B), float(uc), float(C)
        self.norm = quad(lambda w: w * np.exp(-(self.B * w * w + np.exp((w - self.uc) / self.C))), 0.0, SUPPORT,
                         limit=400)[0]

    def unnormalised(self, w):
        w = np.asarray(w, dtype=np.float64)
        return w * np.exp(-self.B * w**2 - np.exp(np.minimum((w - self.uc) / self.C, 50.0)))

    def log_density_jax(self, w):
        import jax.numpy as jnp
        return jnp.log(w) - self.B * w**2 - jnp.exp((w - self.uc) / self.C) - np.log(self.norm)
