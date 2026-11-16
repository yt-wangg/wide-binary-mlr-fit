"""System metallicities from J-CaPS (Sect. 3.2 of the paper).

Stage 1a fits the relative calibration of J-CaPS, a magnitude-dependent bias b_G, an extra scatter s_z and the
degrees of freedom nu of a Student-t measurement model, to the metallicity differences within pairs by maximum
likelihood.  Stage 1b fixes these, sets the zero point by b_G(M_G,sun) = 0, and infers the metallicity distribution
of the sample (six truncated Gaussians with free weights); it returns for every pair the posterior probability of
its metallicity on a grid of 81 nodes in -1.0 <= Z <= 0.6.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln, logsumexp, ndtr

BIAS_KNOTS = np.array([3.5, 6.0, 8.5, 11.0, 13.5])   # b_G knots in M_G; b_G(8.5) = 0 in the fit
BIAS_ZERO_MG = 4.66                                   # b_G is shifted to vanish at the solar M_G
Z_GRID = np.linspace(-1.0, 0.6, 81)
N_POP = 6
STAGE1A_SEED = 20260906


# ------------------------------------------------------------------------------------------------- Stage 1a
def stage1a_data(table):
    """Pairs with finite inputs, 3.5 <= M_G <= 13.5 for both stars (catalogue M_G) and no J-CaPS parameter at a
    boundary; a random 75% (seed 20260906, drawn before the boundary cut) is used for the fit."""
    z = np.column_stack([np.asarray(table[f"feh_jcaps_{j}"], float) for j in (1, 2)])
    e = np.column_stack([np.asarray(table[f"jc_sigma_m_h_{j}"], float) for j in (1, 2)])
    g = np.column_stack([np.asarray(table[f"absg{j}"], float) for j in (1, 2)])
    valid = (np.all(np.isfinite(np.column_stack([g, z, e])), axis=1)
             & np.all((g >= 3.5) & (g <= 13.5) & (e > 0), axis=1))
    rows0 = np.flatnonzero(valid)
    train0 = np.random.default_rng(STAGE1A_SEED).random(len(rows0)) < 0.75
    keep = (np.asarray(table["jc_at_bound_bits_1"])[rows0] == 0) & (np.asarray(table["jc_at_bound_bits_2"])[rows0] == 0)
    rows, train = rows0[keep], train0[keep]

    def basis(x):
        return np.column_stack([np.interp(x, BIAS_KNOTS, v) for v in np.eye(BIAS_KNOTS.size)])

    X = (basis(g[rows, 1]) - basis(g[rows, 0]))[:, [0, 1, 3, 4]]   # b_G(8.5) = 0
    return dict(rows=rows, train=train, d=z[rows, 1] - z[rows, 0], X=X, e=e[rows])


def _difference_logpdf(nodes):
    """log p(r) of the difference of two Student-t variables (scales a, b; common nu), by quadrature on two
    half-lines in the tangent coordinate.  Returns a JAX function of (p, d, X, e)."""
    import jax.numpy as jnp
    from jax.scipy.special import gammaln as jgammaln, logsumexp as jlogsumexp

    x, w = np.polynomial.legendre.leggauss(nodes)
    t, logw = jnp.asarray((x + 1) / 2), jnp.log(jnp.asarray(w / 2))

    def logpdf(x, scale, nu):
        return (jgammaln((nu + 1) / 2) - jgammaln(nu / 2) - 0.5 * jnp.log(nu * jnp.pi) - jnp.log(scale)
                - 0.5 * (nu + 1) * jnp.log1p((x / scale) ** 2 / nu))

    def conv(r, sa, sb, nu):
        radius = jnp.abs(r)[:, None]
        sa, sb = sa[:, None], sb[:, None]

        def half(a, b):
            span = jnp.arctan(radius / (2 * a)) + np.pi / 2
            theta = -np.pi / 2 + span * t[None, :]
            x = a * jnp.tan(theta)
            terms = (logpdf(x, a, nu) + logpdf(radius - x, b, nu) + jnp.log(a) - 2 * jnp.log(jnp.cos(theta))
                     + jnp.log(span) + logw[None, :])
            return jlogsumexp(terms, axis=1)

        return jnp.logaddexp(half(sa, sb), half(sb, sa))

    def f(p, d, X, e):
        s, nu = jnp.exp(p[4]), jnp.exp(p[5])
        a = jnp.sqrt(e**2 + s**2)
        return conv(d - X @ p[:4], a[:, 0], a[:, 1], nu)

    return f


def fit_stage1a(table) -> dict:
    """Maximum-likelihood fit of (b_G knots, s_z, nu) on the training pairs (L-BFGS-B from four starts, refined with
    96 quadrature nodes and checked with 160)."""
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    data = stage1a_data(table)
    tr = data["train"]
    args_tr = tuple(jnp.asarray(data[k][tr]) for k in ("d", "X", "e"))
    args_all = tuple(jnp.asarray(data[k]) for k in ("d", "X", "e"))

    def objective_for(nodes):
        lp = _difference_logpdf(nodes)
        vg = jax.jit(jax.value_and_grad(lambda p: -jnp.sum(lp(p, *args_tr))))

        def objective(p):
            v, g = vg(jnp.asarray(p))
            return float(v), np.asarray(g, dtype=np.float64)
        return objective, jax.jit(lambda p: lp(p, *args_all))

    starts = [np.r_[np.zeros(4), np.log(s), np.log(nu)] for s, nu in [(.02, 2.), (.06, 5.), (.12, 20.), (.25, 3.)]]
    bounds = [(-3, 3)] * 4 + [(-8, 1), (np.log(.5), np.log(300))]
    objective, _ = objective_for(48)
    fits = [minimize(objective, p0, jac=True, method="L-BFGS-B", bounds=bounds,
                     options={"maxiter": 700, "ftol": 2e-10, "gtol": 1e-5}) for p0 in starts]
    good = [f for f in fits if f.success]
    if not good:
        raise RuntimeError("No successful Stage-1a fit.")
    objective, _ = objective_for(96)
    refine = minimize(objective, min(good, key=lambda f: f.fun).x, jac=True, method="L-BFGS-B", bounds=bounds,
                      options={"maxiter": 400, "ftol": 2e-11, "gtol": 1e-5})
    if not refine.success:
        raise RuntimeError(str(refine.message))
    p = refine.x
    lp96 = np.asarray(objective_for(96)[1](jnp.asarray(p)))
    lp160 = np.asarray(objective_for(160)[1](jnp.asarray(p)))
    if np.max(np.abs(lp160 - lp96)) > 0.005:
        raise RuntimeError("Stage-1a quadrature did not converge.")
    return {
        "bias_knots": BIAS_KNOTS.tolist(),
        "bias_values": np.insert(p[:4], 2, 0.0).tolist(),
        "s_z": float(np.exp(p[4])),
        "nu": float(np.exp(p[5])),
        "n_train": int(tr.sum()),
        "n_test": int((~tr).sum()),
        "train_loglike": float(lp160[tr].sum()),
        "test_loglike": float(lp160[~tr].sum()),
    }


# ------------------------------------------------------------------------------------------------- Stage 1b
def student_t_logpdf(x, *, df, loc, scale):
    scale = np.maximum(np.asarray(scale, dtype=np.float64), 1e-12)
    z = (np.asarray(x, dtype=np.float64) - loc) / scale
    const = gammaln((df + 1.0) / 2.0) - gammaln(df / 2.0) - 0.5 * np.log(df * np.pi) - np.log(scale)
    return const - 0.5 * (df + 1.0) * np.log1p(z**2 / df)


def trapezoid_weights(grid):
    w = np.empty_like(grid)
    w[0], w[-1] = 0.5 * (grid[1] - grid[0]), 0.5 * (grid[-1] - grid[-2])
    w[1:-1] = 0.5 * (grid[2:] - grid[:-2])
    return w


class MetallicityPosterior:
    """Stage 1b: population of system metallicities and the posterior P(Z_q) of each pair."""

    def __init__(self, calibration: dict, z_grid=Z_GRID):
        self.z_grid = np.asarray(z_grid, dtype=np.float64)
        self.grid_weights = trapezoid_weights(self.z_grid)
        knots = np.asarray(calibration["bias_knots"], float)
        values = np.asarray(calibration["bias_values"], float)
        self.bias_knots, self.bias_values = knots, values - np.interp(BIAS_ZERO_MG, knots, values)
        self.s_z, self.nu = float(calibration["s_z"]), float(calibration["nu"])
        self.centers = np.linspace(self.z_grid[0], self.z_grid[-1], N_POP)
        self.sigma_pop = (self.z_grid[-1] - self.z_grid[0]) / N_POP

    def set_data(self, absg, feh, feh_sigma):
        self.absg, self.feh, self.feh_sigma = (np.asarray(v, dtype=np.float64) for v in (absg, feh, feh_sigma))
        self.bias = np.interp(self.absg, self.bias_knots, self.bias_values)

    def component_density(self):
        zs = (self.z_grid[:, None] - self.centers[None, :]) / self.sigma_pop
        norm = ndtr((self.z_grid[-1] - self.centers) / self.sigma_pop) - ndtr((self.z_grid[0] - self.centers) / self.sigma_pop)
        return np.exp(-0.5 * zs**2) / (np.sqrt(2.0 * np.pi) * self.sigma_pop * norm[None, :])

    def model(self, feh, feh_sigma, bias):
        import jax.numpy as jnp
        import numpyro
        import numpyro.distributions as dist
        from jax.scipy.special import logsumexp as jlogsumexp

        z = jnp.asarray(self.z_grid)
        comp = jnp.asarray(self.component_density())
        weights = numpyro.sample("population_weights", dist.Dirichlet(jnp.ones(N_POP)))
        log_joint = jnp.log(jnp.asarray(self.grid_weights))[None, :] + jnp.log(comp @ weights)[None, :]
        for k in range(2):
            scale = jnp.sqrt(feh_sigma[:, k, None] ** 2 + self.s_z**2)
            log_joint += dist.StudentT(self.nu, loc=z[None, :] + bias[:, k, None], scale=scale).log_prob(feh[:, k, None])
        numpyro.factor("observations", jnp.sum(jlogsumexp(log_joint, axis=1)))

    def run_mcmc(self, *, num_warmup=1000, num_samples=1000, num_chains=4, seed=20260819, target_accept_prob=0.9):
        import jax
        import jax.numpy as jnp
        from numpyro.infer import MCMC, NUTS

        mcmc = MCMC(NUTS(self.model, target_accept_prob=target_accept_prob), num_warmup=num_warmup,
                    num_samples=num_samples, num_chains=num_chains)
        mcmc.run(jax.random.PRNGKey(seed), jnp.asarray(self.feh), jnp.asarray(self.feh_sigma), jnp.asarray(self.bias))
        self.mcmc = mcmc
        self.weights = np.asarray(mcmc.get_samples()["population_weights"])
        return mcmc

    def posterior_grid(self, weights=None, *, max_draws=128, seed=20260819, chunk=256):
        """P(Z_q) of every pair averaged over max_draws posterior draws of the population weights."""
        weights = self.weights if weights is None else np.asarray(weights)
        n_draws = min(max_draws, weights.shape[0])
        idx = np.linspace(0, weights.shape[0] - 1, n_draws, dtype=int)
        if n_draws < weights.shape[0]:
            idx = np.sort(np.random.default_rng(seed).choice(weights.shape[0], size=n_draws, replace=False))
        pop = np.einsum("qk,dk->dq", self.component_density(), weights[idx])
        n = self.feh.shape[0]
        prob = np.empty((n, self.z_grid.size))
        for a in range(0, n, chunk):
            b = min(a + chunk, n)
            lj = np.log(self.grid_weights)[None, None, :] + np.log(pop)[:, None, :]
            lj = np.broadcast_to(lj, (n_draws, b - a, self.z_grid.size)).copy()
            for k in range(2):
                loc = self.z_grid[None, None, :] + self.bias[a:b, k][None, :, None]
                scale = np.sqrt(self.feh_sigma[a:b, k][None, :, None] ** 2 + self.s_z**2)
                lj += student_t_logpdf(self.feh[a:b, k][None, :, None], df=self.nu, loc=loc, scale=scale)
            p = np.exp(lj - logsumexp(lj, axis=2, keepdims=True))
            prob[a:b] = p.mean(axis=0)
        cum = np.cumsum(prob, axis=1)
        cum /= cum[:, -1, None]
        quant = np.array([[np.interp(t, cum[j], self.z_grid) for t in (0.16, 0.5, 0.84)] for j in range(n)])
        return prob, quant


def save_posterior(path, rows, z_grid, prob, quant, meta):
    np.savez_compressed(path, row_indices=np.asarray(rows, np.int64), z_grid=np.asarray(z_grid, np.float64),
                        probabilities=np.asarray(prob, np.float32), z_quantiles=np.asarray(quant, np.float32),
                        metadata_json=np.asarray(json.dumps(meta)))


def load_posterior(path):
    with np.load(path) as f:
        return dict(rows=f["row_indices"].astype(np.int64), z_grid=f["z_grid"].astype(np.float64),
                    prob=f["probabilities"].astype(np.float64), quant=f["z_quantiles"].astype(np.float64))
