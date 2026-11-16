"""Monotone tensor-spline MLR relative to PARSEC and its NumPyro likelihood (Sect. 3.4-3.6 of the paper).

log10 M(M_G, Z) = sum_ih theta_ih B_i(M_G) C_h(Z) with cubic clamped B-splines (8 x 4 coefficients).  The
coefficients are built recursively from unconstrained variables so that the mass decreases with M_G and increases
with Z.  D = theta - Pi (Pi: least-squares projection of PARSEC) has a Gaussian shrinkage and smoothness prior, and a
solar anchor log10 M(4.67, 0) ~ N(0, sigma_sun).  For pair j at metallicity node Z_q with mass scale
k = sqrt(M_1 + M_2):

    L_jq = [(1 - f_j) L_good,j(k) + f_j L_out,j(k)] / [(1 - f_j) P_good,j(k) + f_j P_out,j(k)],
    logit f_j = logit f_0 + beta (logit r_j - <logit r>),

and the likelihood of the pair is sum_q P_jq L_jq with its metallicity posterior P_jq.
"""
from __future__ import annotations

import numpy as np
from scipy.special import expit

KNOTS_X = np.array([3.5, 3.5, 3.5, 3.5, 5.5, 7.5, 9.5, 11.5, 13.5, 13.5, 13.5, 13.5])
KNOTS_Z = np.array([-1.0, -1.0, -1.0, -1.0, 0.6, 0.6, 0.6, 0.6])
DEGREE = 3
TAU_D, TAU_X, TAU_Z, TAU_XZ = 0.10, 0.05, 0.05, 0.05
SOLAR_MG, SOLAR_SIGMA = 4.67, 0.05 / np.log(10.0)      # 5% solar mass anchor (0.0217 dex)
F0_PRIOR = (-1.386, 1.0)                               # logit f_0 ~ N(logit 0.2, 1)
BETA_PRIOR = (1.0, 0.5)                                # beta ~ N(1, 0.5)
PARAMS = ("c0", "a", "b", "r", "log_lambda_x", "log_lambda_z")


def bspline_basis(values, knots, degree=DEGREE):
    """Non-negative partition-of-unity B-spline basis (right end assigned to the last function)."""
    x = np.asarray(values, dtype=np.float64)
    t = np.asarray(knots, dtype=np.float64)
    n = t.size - degree - 1
    xc = np.clip(x, t[0], t[-1])
    basis = np.stack([((xc >= t[i]) & (xc < t[i + 1])).astype(np.float64) for i in range(t.size - 1)], axis=-1)
    basis = np.where((xc == t[-1])[..., None], np.eye(1, t.size - 1, t.size - 2)[0], basis)
    for order in range(1, degree + 1):
        terms = []
        for i in range(t.size - order - 1):
            ld, rd = t[i + order] - t[i], t[i + order + 1] - t[i + 1]
            left = np.zeros_like(xc) if ld == 0 else (xc - t[i]) / ld * basis[..., i]
            right = np.zeros_like(xc) if rd == 0 else (t[i + order + 1] - xc) / rd * basis[..., i + 1]
            terms.append(left + right)
        basis = np.stack(terms, axis=-1)
    end = np.zeros_like(basis[..., :n]); end[..., -1] = 1.0
    return np.where((xc == t[-1])[..., None], end, basis[..., :n])


def _softplus(x):
    return np.logaddexp(0.0, np.asarray(x, dtype=np.float64))


def _isotonic_increasing(values):
    levels, counts = [], []
    for v in np.asarray(values, dtype=np.float64):
        levels.append(float(v)); counts.append(1)
        while len(levels) >= 2 and levels[-2] > levels[-1]:
            c = counts[-2] + counts[-1]
            levels[-2:] = [(counts[-2] * levels[-2] + counts[-1] * levels[-1]) / c]
            counts[-2:] = [c]
    return np.concatenate([np.full(c, l) for l, c in zip(levels, counts)])


def lookup_interpolate(values, grid, table):
    """Row-wise linear interpolation of table (n, L) at values (n, Q); -inf outside the grid."""
    import jax.numpy as jnp
    grid = jnp.asarray(grid)
    upper = jnp.clip(jnp.searchsorted(grid, values, side="right"), 1, grid.size - 1)
    lower = upper - 1
    yl, yu = jnp.take_along_axis(table, lower, axis=1), jnp.take_along_axis(table, upper, axis=1)
    xl, xu = grid[lower], grid[upper]
    out = yl + (values - xl) / (xu - xl) * (yu - yl)
    return jnp.where((values >= grid[0]) & (values <= grid[-1]), out, -jnp.inf)


class MonotoneMLR:
    def __init__(self, parsec, solar_sigma=SOLAR_SIGMA):
        self.parsec = parsec
        self.Kx, self.Kz = KNOTS_X.size - DEGREE - 1, KNOTS_Z.size - DEGREE - 1
        self.solar_sigma = float(solar_sigma)
        self.greville_x = np.array([np.mean(KNOTS_X[i + 1:i + DEGREE + 1]) for i in range(self.Kx)])
        self.greville_z = np.array([np.mean(KNOTS_Z[i + 1:i + DEGREE + 1]) for i in range(self.Kz)])
        x, z = np.linspace(KNOTS_X[0], KNOTS_X[-1], 501), np.linspace(KNOTS_Z[0], KNOTS_Z[-1], 81)
        design = np.einsum("zh,xi->zxih", bspline_basis(z, KNOTS_Z), bspline_basis(x, KNOTS_X)).reshape(-1, self.Kx * self.Kz)
        gp = np.log10(parsec.mass(x[None, :], z[:, None]))
        self.Pi = np.linalg.lstsq(design, gp.reshape(-1), rcond=None)[0].reshape(self.Kx, self.Kz)

    # ----------------------------------------------------------------------------------- surface
    def lambda_references(self):
        dx = self.Pi[:-1] - self.Pi[1:]
        dz = self.Pi[:, 1:] - self.Pi[:, :-1]
        return float(np.median(dx[dx > 0]) / _softplus(0.0)), float(np.median(dz[dz > 0]) / _softplus(0.0))

    def theta_from_raw(self, p):
        a, b = np.asarray(p["a"], float), np.asarray(p["b"], float)
        r = np.asarray(p["r"], float).reshape(self.Kx - 1, self.Kz - 1)
        lx, lz = np.exp(float(p["log_lambda_x"])), np.exp(float(p["log_lambda_z"]))
        th = np.zeros((self.Kx, self.Kz))
        th[0, 0] = float(p["c0"])
        for h in range(1, self.Kz):
            th[0, h] = th[0, h - 1] + lz * _softplus(a[h - 1])
        for i in range(1, self.Kx):
            th[i, 0] = th[i - 1, 0] - lx * _softplus(b[i - 1])
        for total in range(2, self.Kx + self.Kz - 1):
            for i in range(1, self.Kx):
                h = total - i
                if 1 <= h < self.Kz:
                    lo, hi = th[i, h - 1], th[i - 1, h]
                    th[i, h] = lo + (hi - lo) * expit(r[i - 1, h - 1])
        return th

    def _theta_jax(self, c0, a, b, r, log_lambda_x, log_lambda_z):
        import jax.numpy as jnp
        from jax.nn import sigmoid
        th = jnp.zeros((self.Kx, self.Kz)).at[0, 0].set(c0)
        lx, lz = jnp.exp(log_lambda_x), jnp.exp(log_lambda_z)
        for h in range(1, self.Kz):
            th = th.at[0, h].set(th[0, h - 1] + lz * jnp.logaddexp(0.0, a[h - 1]))
        for i in range(1, self.Kx):
            th = th.at[i, 0].set(th[i - 1, 0] - lx * jnp.logaddexp(0.0, b[i - 1]))
        for total in range(2, self.Kx + self.Kz - 1):
            for i in range(1, self.Kx):
                h = total - i
                if 1 <= h < self.Kz:
                    lo, hi = th[i, h - 1], th[i - 1, h]
                    th = th.at[i, h].set(lo + (hi - lo) * sigmoid(r[i - 1, h - 1]))
        return th

    @staticmethod
    def log_mass(absg, z, theta):
        return np.einsum("...i,...h,ih->...", bspline_basis(absg, KNOTS_X), bspline_basis(z, KNOTS_Z), theta)

    def initial_parameters(self):
        """A feasible starting point near PARSEC: alternating isotonic projection of Pi, inverted through the
        recursion, with the solar anchor satisfied exactly."""
        target = self.Pi.copy()
        for _ in range(200):
            prev = target.copy()
            for h in range(self.Kz):
                target[:, h] = -_isotonic_increasing(-target[:, h])
            for i in range(self.Kx):
                target[i, :] = _isotonic_increasing(target[i, :])
            if np.max(np.abs(target - prev)) < 1e-12:
                break
        target += -float(self.log_mass(SOLAR_MG, 0.0, target))
        lx, lz = self.lambda_references()
        inv = lambda v: np.where(np.maximum(v, 1e-8) > 20.0, np.maximum(v, 1e-8), np.log(np.expm1(np.maximum(v, 1e-8))))
        p = dict(c0=float(target[0, 0]),
                 a=np.clip(inv(np.diff(target[0, :]) / lz), -4.0, 4.0),
                 b=np.clip(inv((target[:-1, 0] - target[1:, 0]) / lx), -4.0, 4.0),
                 r=np.zeros((self.Kx - 1, self.Kz - 1)), log_lambda_x=float(np.log(lx)), log_lambda_z=float(np.log(lz)),
                 f_outlier_logit=float(np.log(0.2 / 0.8)), f_outlier_slope=1.0)
        for total in range(2, self.Kx + self.Kz - 1):
            th = self.theta_from_raw(p)
            for i in range(1, self.Kx):
                h = total - i
                if 1 <= h < self.Kz:
                    lo, hi = th[i, h - 1], th[i - 1, h]
                    frac = float(np.clip((target[i, h] - lo) / max(hi - lo, 1e-12), expit(-4.0), expit(4.0)))
                    p["r"][i - 1, h - 1] = np.log(frac / (1.0 - frac))
        p["c0"] -= float(self.log_mass(SOLAR_MG, 0.0, self.theta_from_raw(p)))
        return p

    # ----------------------------------------------------------------------------------- likelihood
    def set_data(self, *, absg, z_grid, z_prob, sqrt_grid, log_good, log_pass_good, template_grid, log_out,
                 log_pass_out, f_offset):
        self.data = dict(absg=np.asarray(absg, float), z_grid=np.asarray(z_grid, float), z_prob=np.asarray(z_prob, float),
                         sqrt_grid=np.asarray(sqrt_grid, float), log_good=np.asarray(log_good, np.float32),
                         log_pass_good=np.asarray(log_pass_good, np.float32), template_grid=np.asarray(template_grid, float),
                         log_out=np.asarray(log_out, np.float32), log_pass_out=np.asarray(log_pass_out, np.float32),
                         f_offset=np.asarray(f_offset, float))

    def model(self, z_prob, log_good):
        import jax
        import jax.numpy as jnp
        import numpyro
        import numpyro.distributions as dist
        from jax.scipy.special import logsumexp

        d = self.data
        bx1, bx2 = (jnp.asarray(bspline_basis(d["absg"][:, k], KNOTS_X)) for k in (0, 1))
        bz = jnp.asarray(bspline_basis(d["z_grid"], KNOTS_Z))
        anchor_bx, anchor_bz = jnp.asarray(bspline_basis(SOLAR_MG, KNOTS_X)), jnp.asarray(bspline_basis(0.0, KNOTS_Z))
        sqrt_grid, t_grid = jnp.asarray(d["sqrt_grid"]), jnp.asarray(d["template_grid"])
        pass_good = jnp.asarray(d["log_pass_good"], dtype=jnp.float32)
        t_bad = jnp.asarray(d["log_out"], dtype=jnp.float32)
        t_pass = jnp.asarray(d["log_pass_out"], dtype=jnp.float32)
        f_offset = jnp.asarray(d["f_offset"])

        c0 = numpyro.sample("c0", dist.Normal(float(self.Pi[0, 0]), 0.10))
        a = numpyro.sample("a", dist.Normal(0.0, 1.0).expand([self.Kz - 1]))
        b = numpyro.sample("b", dist.Normal(0.0, 1.0).expand([self.Kx - 1]))
        r = numpyro.sample("r", dist.Normal(0.0, 1.0).expand([self.Kx - 1, self.Kz - 1]))
        lref_x, lref_z = self.lambda_references()
        log_lambda_x = numpyro.sample("log_lambda_x", dist.Normal(np.log(lref_x), 0.7))
        log_lambda_z = numpyro.sample("log_lambda_z", dist.Normal(np.log(lref_z), 0.7))
        f_logit = numpyro.sample("f_outlier_logit", dist.Normal(*F0_PRIOR))
        beta = numpyro.sample("f_outlier_slope", dist.Normal(*BETA_PRIOR))
        numpyro.deterministic("f_outlier", jax.nn.sigmoid(f_logit))
        f_j = jax.nn.sigmoid(f_logit + beta * f_offset)
        numpyro.deterministic("f_outlier_mean", jnp.mean(f_j))
        f = f_j[:, None]

        theta = self._theta_jax(c0, a, b, r, log_lambda_x, log_lambda_z)
        D = theta - jnp.asarray(self.Pi)
        hx, hz = jnp.asarray(np.diff(self.greville_x)), jnp.asarray(np.diff(self.greville_z))
        href_x = float((self.greville_x[-1] - self.greville_x[0]) / (self.Kx - 1))
        href_z = float((self.greville_z[-1] - self.greville_z[0]) / (self.Kz - 1))
        slope_x, slope_z = jnp.diff(D, axis=0) / hx[:, None], jnp.diff(D, axis=1) / hz[None, :]
        d2x = 2.0 * jnp.diff(slope_x, axis=0) / (hx[:-1, None] + hx[1:, None]) * href_x**2
        d2z = 2.0 * jnp.diff(slope_z, axis=1) / (hz[None, :-1] + hz[None, 1:]) * href_z**2
        dxz = (D[1:, 1:] - D[1:, :-1] - D[:-1, 1:] + D[:-1, :-1]) / (hx[:, None] * hz[None, :]) * href_x * href_z
        numpyro.factor("D_shrinkage", -0.5 * jnp.sum((D / TAU_D) ** 2))
        numpyro.factor("D_smoothness", -0.5 * (jnp.sum((d2x / TAU_X) ** 2) + jnp.sum((d2z / TAU_Z) ** 2)
                                               + jnp.sum((dxz / TAU_XZ) ** 2)))
        m1 = jnp.power(10.0, jnp.einsum("ni,qh,ih->nq", bx1, bz, theta))
        m2 = jnp.power(10.0, jnp.einsum("ni,qh,ih->nq", bx2, bz, theta))
        k = jnp.sqrt(jnp.maximum(m1 + m2, 1e-12))
        lg = lookup_interpolate(k, sqrt_grid, log_good)
        lb = lookup_interpolate(k, t_grid, t_bad).astype(lg.dtype)
        log_cond = jnp.logaddexp(jnp.log1p(-f) + lg, jnp.log(f) + lb)
        log_pass = jnp.logaddexp(jnp.log1p(-f) + lookup_interpolate(k, sqrt_grid, pass_good),
                                 jnp.log(f) + lookup_interpolate(k, t_grid, t_pass).astype(lg.dtype))
        log_cond = log_cond - log_pass
        numpyro.factor("dynamics", jnp.sum(logsumexp(jnp.log(jnp.maximum(z_prob, 1e-30)) + log_cond, axis=1)))
        solar_g = jnp.einsum("i,h,ih->", anchor_bx, anchor_bz, theta)
        numpyro.deterministic("solar_anchor_g", solar_g)
        numpyro.factor("solar_anchor", dist.Normal(0.0, self.solar_sigma).log_prob(solar_g))

    def model_args(self):
        import jax.numpy as jnp
        d = self.data
        return jnp.asarray(d["z_prob"]), jnp.asarray(d["log_good"])

    def run_mcmc(self, *, num_warmup=1000, num_samples=7000, num_chains=4, seed=20260820, target_accept_prob=0.97):
        import jax
        import jax.numpy as jnp
        from numpyro.infer import MCMC, NUTS
        from numpyro.infer.initialization import init_to_value

        init = {k: jnp.asarray(v) for k, v in self.initial_parameters().items()}
        kernel = NUTS(self.model, target_accept_prob=target_accept_prob, init_strategy=init_to_value(values=init))
        mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples, num_chains=num_chains)
        mcmc.run(jax.random.PRNGKey(seed), *self.model_args(),
                 extra_fields=("diverging", "energy", "potential_energy", "num_steps", "accept_prob"))
        self.mcmc = mcmc
        return mcmc
