"""Forward model of pairs in which a star hosts an unresolved inner companion (Appendix D of the paper).

For every pair of the MLR sample and every replicate:
  1. each star gets an inner companion with probability f(M) (linear in mass from 0.05 at 0.1 Msun to 0.1125 at
     1 Msun), log10 P/d ~ N(5.03, 2.28) on [0, 8], q uniform on [0.1, 1], e = 0 for P < 10 d and uniform on [0, 0.8]
     otherwise; companions with a_in > s/3 are not drawn;
  2. host and companion masses are solved from the PARSEC relation so that their summed G flux equals the observed
     one (companions below 0.075 Msun are dark); the analysis assumes the mass of a single star at the observed M_G;
  3. the outer orbit has |Delta v| = u~ sqrt(M_true / s) with u~ drawn from p0(u~) and a random direction;
  4. the photocentre of each host moves by (B - beta) r_rel(t) over 40 epochs in the DR3 window; a linear fit gives
     the proper-motion bias and the residual rms, from which RUWE ~ sqrt(1 + rms^2 / sigma_AL(G)^2);
  5. systems with an inner pair wider than 1 arcsec, a companion brighter than 10% of the combined G flux, or
     RUWE > 3 are removed.
The latent u~ = u / sqrt(M_assumed) of the retained systems with an inner companion gives the outlier template.  The
template h_j and fraction r_j of pair j are taken from its 200 nearest pairs in (log10 d, log10 s, M_G,1),
standardised over the sample.
"""
from __future__ import annotations

import numpy as np

G_YR = 4.0 * np.pi**2
AUYR_KMS = 4.740470463533348
SETTINGS = dict(reps=100, seed=20261102, f_inner_lo=0.05, f_inner_hi=0.1125, logp_mean=5.03, logp_sd=2.28,
                q_min=0.1, e_max=0.8, stability=3.0, resolve_arcsec=1.0, beta_detect=0.10, max_ruwe=3.0, epochs=40,
                baseline=1038.0 / 365.25, chunk=20000, template_smooth=1.5, local_neighbours=200)


class ParsecRelation:
    """log10 mass of the PARSEC surface on a (Z, M_G) grid, extended linearly beyond M_G = 13.5, and its inverse."""

    def __init__(self, surface):
        self.x = np.linspace(3.5, 13.5, 201)
        self.zg = np.linspace(-1.0, 0.6, 33)
        self.lm = np.log10(surface.mass(self.x[None, :], self.zg[:, None]))
        slope = (self.lm[:, -1] - self.lm[:, -6]) / (self.x[-1] - self.x[-6])
        xe = np.linspace(13.5, 20.0, 66)[1:]
        self.x = np.concatenate([self.x, xe])
        self.lm = np.concatenate([self.lm, self.lm[:, -1:] + slope[:, None] * (xe - 13.5)[None, :]], axis=1)

    def _zi(self, z):
        return np.clip(np.rint((np.asarray(z) - self.zg[0]) / (self.zg[1] - self.zg[0])).astype(int), 0, self.zg.size - 1)

    def mass(self, mg, z):
        mg, zi = np.asarray(mg, float), self._zi(z)
        out = np.empty_like(mg)
        for k in np.unique(zi):
            m = zi == k
            out[m] = 10 ** np.interp(mg[m], self.x, self.lm[k])
        return out

    def mg(self, mass, z):
        mass, zi = np.asarray(mass, float), self._zi(z)
        out = np.empty_like(mass)
        for k in np.unique(zi):
            m = zi == k
            out[m] = np.interp(-np.log10(mass[m]), -self.lm[k], self.x)
        return out


def sigma_al(g):
    """Approximate per-transit along-scan precision [mas]."""
    return np.sqrt(0.08**2 + (0.13 * 10 ** (0.2 * (np.asarray(g) - 15.0))) ** 2)


def kepler_E(M, e, iters=30):
    E = M + e * np.sin(M)
    for _ in range(iters):
        E = E - (E - e * np.sin(E) - M) / (1.0 - e * np.cos(E))
    return E


def inner_companions(rng, m_assumed, mg_obs, z, s_au, rel, c):
    n = m_assumed.size
    lo, hi = c["f_inner_lo"], c["f_inner_hi"]
    f = np.clip(lo + (hi - lo) * (m_assumed - 0.1) / 0.9, min(lo, hi), max(lo, hi))
    has = rng.random(n) < f
    q = rng.uniform(c["q_min"], 1.0, n)
    dg = np.linspace(0.0, 0.76, 39)
    mg_h_try = mg_obs[:, None] + dg[None, :]
    m_h_try = rel.mass(mg_h_try.ravel(), np.repeat(z, dg.size)).reshape(n, dg.size)
    m_c_try = q[:, None] * m_h_try
    flux_c = np.where(m_c_try >= 0.075, 10 ** (-0.4 * rel.mg(m_c_try.ravel(), np.repeat(z, dg.size)).reshape(n, dg.size)), 0.0)
    resid = 10 ** (-0.4 * mg_h_try) + flux_c - 10 ** (-0.4 * mg_obs)[:, None]
    k = np.clip((resid > 0).sum(axis=1) - 1, 0, dg.size - 2)
    r0, r1 = resid[np.arange(n), k], resid[np.arange(n), k + 1]
    t = np.clip(r0 / np.where(r0 - r1 != 0, r0 - r1, 1.0), 0.0, 1.0)
    mg_h = mg_obs + dg[k] + t * (dg[1] - dg[0])
    m_h = rel.mass(mg_h, z)
    m_c = q * m_h
    beta = np.where(m_c >= 0.075, 10 ** (-0.4 * rel.mg(m_c, z)), 0.0)
    beta = beta / (beta + 10 ** (-0.4 * mg_h))
    logp = rng.normal(c["logp_mean"], c["logp_sd"], n)
    while np.any(bad := (logp < 0) | (logp > 8)):
        logp[bad] = rng.normal(c["logp_mean"], c["logp_sd"], bad.sum())
    P_yr = 10 ** logp / 365.25
    a_in = (G_YR * (m_h + m_c) * P_yr**2 / (4 * np.pi**2)) ** (1 / 3)
    has &= a_in < s_au / c["stability"]
    e = np.where(P_yr * 365.25 < 10, 0.0, rng.uniform(0, c["e_max"], n))
    return dict(has=has, m_h=np.where(has, m_h, m_assumed), m_c=np.where(has, m_c, 0.0),
                beta=np.where(has, beta, 0.0), P_yr=P_yr, a_in=a_in, e=e)


def photocentre(rng, comp, c, d_pc, g_app):
    """Proper-motion bias [km/s] and RUWE of the photocentre motion over the DR3 window."""
    n = comp["has"].size
    v, ruwe = np.zeros((n, 2)), np.ones(n)
    idx = np.where(comp["has"])[0]
    if idx.size == 0:
        return v, ruwe
    P, a, e = comp["P_yr"][idx], comp["a_in"][idx], comp["e"][idx]
    fac = comp["m_c"][idx] / (comp["m_h"][idx] + comp["m_c"][idx]) - comp["beta"][idx]
    cosi = rng.uniform(-1, 1, idx.size)
    om, Om = rng.uniform(0, 2 * np.pi, (2, idx.size))
    A = np.cos(om) * np.cos(Om) - np.sin(om) * np.sin(Om) * cosi
    B = np.cos(om) * np.sin(Om) + np.sin(om) * np.cos(Om) * cosi
    F = -np.sin(om) * np.cos(Om) - np.cos(om) * np.sin(Om) * cosi
    G = -np.sin(om) * np.sin(Om) + np.cos(om) * np.cos(Om) * cosi
    t = np.linspace(0.0, c["baseline"], c["epochs"])
    M = 2 * np.pi * (t[None, :] / P[:, None] + rng.random(idx.size)[:, None])
    E = kepler_E(np.mod(M, 2 * np.pi), e[:, None])
    x = a[:, None] * (np.cos(E) - e[:, None])
    y = a[:, None] * np.sqrt(1 - e[:, None] ** 2) * np.sin(E)
    X = fac[:, None] * (A[:, None] * x + F[:, None] * y)
    Y = fac[:, None] * (B[:, None] * x + G[:, None] * y)
    tc = t - t.mean()
    sx, sy = (X * tc).sum(1) / (tc**2).sum(), (Y * tc).sum(1) / (tc**2).sum()
    v[idx] = AUYR_KMS * np.column_stack([sx, sy])
    rx = X - X.mean(1, keepdims=True) - sx[:, None] * tc
    ry = Y - Y.mean(1, keepdims=True) - sy[:, None] * tc
    rms_mas = np.sqrt(0.5 * (rx**2 + ry**2).mean(1)) / d_pc[idx] * 1000.0
    ruwe[idx] = np.sqrt(1 + (rms_mas / sigma_al(g_app[idx])) ** 2)
    return v, ruwe


def simulate(sample, z, rel, draw_good, rng, c=SETTINGS):
    """Replicates of every pair; returns pair index, latent u, assumed mass, flags of companions and removal."""
    pair = np.repeat(np.arange(sample["u"].size), c["reps"])
    out = {k: [] for k in ("pair", "u_true", "m_assumed", "has", "removed")}
    for c0 in range(0, pair.size, c["chunk"]):
        pj = pair[c0:c0 + c["chunk"]]
        n = pj.size
        mg, zz, s, d, g = sample["absg"][pj], z[pj], sample["s_au"][pj], sample["d_pc"][pj], sample["g_app"][pj]
        m_ass = np.column_stack([rel.mass(mg[:, 0], zz), rel.mass(mg[:, 1], zz)])
        comps = [inner_companions(rng, m_ass[:, k], mg[:, k], zz, s, rel, c) for k in (0, 1)]
        m_true = sum(cp["m_h"] + cp["m_c"] for cp in comps)
        ut = draw_good(n)
        phi = rng.uniform(0, 2 * np.pi, n)
        dv = (ut * np.sqrt(m_true / s))[:, None] * np.column_stack([np.cos(phi), np.sin(phi)])
        v1, r1 = photocentre(rng, comps[0], c, d, g[:, 0])
        v2, r2 = photocentre(rng, comps[1], c, d, g[:, 1])
        u_true = np.hypot(*(dv + v1 - v2).T) * np.sqrt(s)
        rng.standard_normal((2, n))   # measurement noise of u: not used by the template (keeps the random stream)
        removed = (r1 > c["max_ruwe"]) | (r2 > c["max_ruwe"])
        for cp in comps:
            removed |= cp["has"] & ((cp["a_in"] / d > c["resolve_arcsec"]) | (cp["beta"] > c["beta_detect"]))
        for k, val in (("pair", pj), ("u_true", u_true), ("m_assumed", m_ass.sum(1)),
                       ("has", comps[0]["has"] | comps[1]["has"]), ("removed", removed)):
            out[k].append(val)
    return {k: np.concatenate(v) for k, v in out.items()}


def local_templates(sample, sim, c=SETTINGS):
    """Template h_j(u~) on u~ = 0, 0.25, ..., 250 and fraction r_j of retained systems with an inner companion, from
    the c['local_neighbours'] nearest pairs (including the pair itself)."""
    from scipy.ndimage import gaussian_filter1d
    from scipy.sparse import csr_matrix
    from scipy.spatial import cKDTree

    w = np.arange(0.0, 250.0 + 1e-9, 0.25)
    edges = np.concatenate([w - 0.125, [w[-1] + 0.125]]); edges[0] = 0.0
    x = sim["u_true"] / np.sqrt(sim["m_assumed"])
    trip = sim["has"] & ~sim["removed"]
    n = sample["u"].size
    feat = np.column_stack([np.log10(sample["d_pc"]), np.log10(sample["s_au"]), sample["absg"][:, 0]])
    feat = (feat - feat.mean(0)) / feat.std(0)
    k = min(int(c["local_neighbours"]), n)
    _, nb = cKDTree(feat).query(feat, k=k)
    A = csr_matrix((np.ones(n * k), (np.repeat(np.arange(n), k), nb.ravel())), shape=(n, n))
    kept = ~sim["removed"]
    n_kept = np.bincount(sim["pair"][kept], minlength=n).astype(float)
    n_trip = np.bincount(sim["pair"][trip], minlength=n).astype(float)
    bin_ = np.clip(np.searchsorted(edges, np.clip(x[trip], 0, w[-1]), side="right") - 1, 0, w.size - 1)
    counts = csr_matrix((np.ones(trip.sum()), (sim["pair"][trip], bin_)), shape=(n, w.size))
    hist = gaussian_filter1d((A @ counts).toarray(), c["template_smooth"] / 0.25, axis=1, mode="reflect")
    hist /= np.trapezoid(hist, w, axis=1)[:, None]
    return dict(w=w, pair_rows=np.asarray(sample["rows"], np.int64), pair_density=hist.astype(np.float32),
                pair_fraction=(A @ n_trip) / (A @ n_kept))


def good_sampler(shape, rng):
    w = np.linspace(0.0, 80.0, 40001)
    p = w * np.exp(-shape.B * w**2 - np.exp(np.minimum((w - shape.uc) / shape.C, 50)))
    cdf = np.cumsum(p); cdf /= cdf[-1]
    return lambda n: np.interp(rng.random(n), cdf, w)
