"""Per-pair likelihood tables against the mass scale k = sqrt(M_tot / Msun) (Sect. 3.3-3.5 of the paper).

* ``good_lookup``: L_good,j(k) = int Rice(u_j | v, sigma_j) p0(v/k)/k dv, the good component convolved with the
  measurement error (Eq. 9).
* ``good_pass``: probability that the good component passes the selection c sigma_j < u < U_j (Eq. 13).
* ``template_tables``: the same two quantities for the outlier component with a tabulated shape h_j(u~) per pair.
The MLR fit interpolates all tables linearly in k.
"""
from __future__ import annotations

import numpy as np
from scipy.special import i0e
from scipy.stats import ncx2

from .shape import SUPPORT, GoodShape

FLOOR = 1e-30


def good_lookup(u, u_sigma, shape: GoodShape, sqrt_grid, *, nodes=128, sigma_extent=14.0, chunk=128):
    """log L_good on (n, L), float32, computed with JAX (GPU if available).  Gauss-Legendre nodes cover
    u +/- sigma_extent * sigma_u, split at the support boundary v = k * 80."""
    import jax
    import jax.numpy as jnp
    from jax.scipy.special import i0e as ji0e

    gl_x, gl_w = np.polynomial.legendre.leggauss(nodes)
    gl_x, gl_w = jnp.asarray(gl_x, jnp.float32), jnp.asarray(gl_w, jnp.float32)
    kk = jnp.asarray(sqrt_grid, jnp.float32)
    log_norm = float(np.log(shape.norm))

    @jax.jit
    def block(uc, sc):
        upper = jnp.minimum((uc + sigma_extent * sc)[:, None], kk[None, :] * SUPPORT)
        lower = jnp.minimum(jnp.maximum(0.0, uc - sigma_extent * sc)[:, None], upper)
        mid, half = 0.5 * (lower + upper), 0.5 * (upper - lower)
        v = mid[:, None, :] + half[:, None, :] * gl_x[None, :, None]
        qw = half[:, None, :] * gl_w[None, :, None]
        obs, sig2 = uc[:, None, None], sc[:, None, None] ** 2
        log_rice = jnp.log(obs) - jnp.log(sig2) - (obs - v) ** 2 / (2.0 * sig2) + jnp.log(ji0e(obs * v / sig2) + 1e-30)
        w = v / kk[None, None, :]
        wc = jnp.clip(w, 1e-12, SUPPORT)
        log_p0 = (-log_norm + jnp.log(wc) - shape.B * wc**2 - jnp.exp((wc - shape.uc) / shape.C)
                  - jnp.log(kk[None, None, :]))
        p0 = jnp.where((w > 0.0) & (w <= SUPPORT), jnp.exp(log_p0), 0.0)
        return jnp.sum(jnp.exp(log_rice) * qw * p0, axis=1)

    u, u_sigma = np.asarray(u, np.float64), np.asarray(u_sigma, np.float64)
    out = np.empty((u.size, len(sqrt_grid)), dtype=np.float32)
    for a in range(0, u.size, chunk):
        b = min(a + chunk, u.size)
        uc = np.full(chunk, u[a], np.float32); sc = np.full(chunk, u_sigma[a], np.float32)
        uc[: b - a], sc[: b - a] = u[a:b], u_sigma[a:b]
        val = np.asarray(jax.device_get(block(jnp.asarray(uc), jnp.asarray(sc))))[: b - a]
        out[a:b] = np.log(np.maximum(val, FLOOR))
    return out


def band_survival(v, u_sigma, cut, upper):
    """P(cut sigma < u_obs < upper | v) for Rice noise; rows are pairs, columns latent v."""
    v = np.asarray(v, dtype=np.float64)[None, :]
    sig = np.asarray(u_sigma, dtype=np.float64)[:, None]
    nc = np.square(v / sig)
    out = ncx2.sf(float(cut) ** 2, 2, nc)
    up = np.asarray(upper, dtype=np.float64)[:, None]
    return np.maximum(out - ncx2.sf(np.square(up / sig), 2, nc), 0.0)


def good_pass(u_sigma, sqrt_grid, shape: GoodShape, *, cut, upper, n_x=4096, n_quad=2048, n_v=4096):
    """log P(c sigma_j < u_obs < U_j | k) of the good component, (n, L).  The part above the lower limit depends on
    k / sigma only and is tabulated once; the part above U_j is integrated on a latent-velocity grid."""
    sigma = np.asarray(u_sigma, dtype=np.float64)
    grid = np.asarray(sqrt_grid, dtype=np.float64)
    w = np.linspace(0.0, SUPPORT, n_quad)
    good = w * np.exp(-shape.B * w**2 - np.exp((w - shape.uc) / shape.C))
    good = good / np.trapezoid(good, w)
    log_x = np.linspace(np.log(grid[0] / sigma.max() * 0.5), np.log(grid[-1] / sigma.min() * 2.0), n_x)
    tab = np.array([np.trapezoid(good * ncx2.sf(cut**2, 2, np.square(np.exp(lx) * w)), w) for lx in log_x])
    log_pass = np.interp(np.log(grid[None, :] / sigma[:, None]), log_x, np.log(np.maximum(tab, FLOOR)))
    vv = np.linspace(0.0, float(grid[-1]) * SUPPORT, n_v)
    trap = np.full(vv.size, vv[1] - vv[0]); trap[[0, -1]] *= 0.5
    above = ncx2.sf(np.square(np.asarray(upper)[:, None] / sigma[:, None]), 2, np.square(vv[None, :] / sigma[:, None])) * trap
    wq = vv[None, :] / grid[:, None]
    p0 = np.where(wq <= SUPPORT, shape.unnormalised(wq), 0.0)
    p0 = p0 / np.maximum(np.sum(p0 * trap[None, :], axis=1, keepdims=True), FLOOR)
    return np.log(np.maximum(np.exp(log_pass) - above @ p0.T, FLOOR))


def template_tables(u, u_sigma, w_grid, density, sqrt_grid, *, cut, upper, nodes=256, sigma_extent=14.0, n_pass=2048):
    """Outlier component with a fixed shape h_j(u~) per pair (density (n, W) on a uniform grid w_grid from 0).

    In u the outlier density is h_j(v/k)/k, truncated and renormalised to 0 <= v <= 80.  Returns
    (log L_out (n, L), log P_pass,out (n, L)), float32."""
    u, u_sigma = np.asarray(u, np.float64), np.asarray(u_sigma, np.float64)
    w_grid, density = np.asarray(w_grid, np.float64), np.asarray(density, np.float64)
    k_grid = np.asarray(sqrt_grid, np.float64)
    U, dw = SUPPORT, w_grid[1] - w_grid[0]
    cdf = np.concatenate([np.zeros((u.size, 1)),
                          np.cumsum(0.5 * (density[:, 1:] + density[:, :-1]) * np.diff(w_grid), axis=1)], axis=1)
    rows = np.arange(u.size)[:, None]

    def interp_rows(table, x, right):
        t = x / dw
        i = np.clip(np.floor(t).astype(np.int64), 0, w_grid.size - 2)
        f = t - i
        val = (1.0 - f) * table[rows, i] + f * table[rows, i + 1]
        return np.where(t < 0, 0.0, np.where(t > w_grid.size - 1, right, val))

    log_norm = np.log(np.maximum(interp_rows(cdf, (U / k_grid)[None, :], cdf[:, -1:]), FLOOR))
    x, wts = np.polynomial.legendre.leggauss(nodes)
    hi = np.minimum(u + sigma_extent * u_sigma, U)
    lo = np.minimum(np.maximum(u - sigma_extent * u_sigma, 0.0), hi)
    half = 0.5 * (hi - lo)[:, None]
    v = 0.5 * (hi + lo)[:, None] + half * x[None, :]
    obs, sig = u[:, None], u_sigma[:, None]
    kernel = np.exp(np.log(obs / sig**2) - (obs - v) ** 2 / (2.0 * sig**2) + np.log(i0e(obs * v / sig**2))) * (half * wts[None, :])
    log_bad = np.empty((u.size, k_grid.size), dtype=np.float32)
    for l, k in enumerate(k_grid):
        val = np.sum(kernel * interp_rows(density, v / k, 0.0), axis=1) / k
        log_bad[:, l] = np.log(np.maximum(val, FLOOR)) - log_norm[:, l]
    vq = np.linspace(0.0, U, n_pass)
    trap = np.full(vq.size, vq[1] - vq[0]); trap[[0, -1]] *= 0.5
    survival = band_survival(vq, u_sigma, cut, upper) * trap[None, :]
    log_pass = np.empty((u.size, k_grid.size), dtype=np.float32)
    for l, k in enumerate(k_grid):
        hv = interp_rows(density, vq[None, :] / k, 0.0) / k
        hv = hv / np.maximum(np.sum(hv * trap[None, :], axis=1, keepdims=True), FLOOR)
        log_pass[:, l] = np.log(np.maximum(np.sum(survival * hv, axis=1), FLOOR))
    return log_bad, log_pass
