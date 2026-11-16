"""Dynamical fit of the MLR.  Output: results/mlr_posterior.npz (posterior draws of all parameters and the spline
coefficients theta) and results/mlr_grid.csv (16/50/84% of the mass on a grid of M_G and Z).
NUTS, 4 chains x (1000 + 7000) iterations; GPU required in practice (several hours).
Usage: python scripts/6_fit_mlr.py [--samples N] [--warmup N] [--chains N]"""
import argparse
import json

import numpy as np

from _common import RESULTS, TEMPLATE_GRID
from wbfit import dynamics, metallicity, mlr, parsec, sample, shape

ap = argparse.ArgumentParser()
ap.add_argument("--warmup", type=int, default=1000)
ap.add_argument("--samples", type=int, default=7000)
ap.add_argument("--chains", type=int, default=4)
args = ap.parse_args()

post = metallicity.load_posterior(RESULTS / "metallicity_posterior.npz")
p = sample.mlr_sample(sample.read_table(), post["rows"])
good = shape.GoodShape(**json.loads((RESULTS / "good_shape.json").read_text())["adopted"])
lookup = np.load(RESULTS / "good_lookup.npz")
tmpl = np.load(RESULTS / "outlier_templates.npz")
if not (np.array_equal(lookup["rows"], p["rows"]) and np.array_equal(tmpl["pair_rows"], p["rows"])):
    raise SystemExit("The lookup or the templates belong to another sample; rerun steps 4 and 5.")
print(f"MLR sample: {p['rows'].size} pairs")

print("selection and outlier tables ...", flush=True)
sqrt_grid = lookup["sqrt_grid"]
log_pass_good = dynamics.good_pass(p["u_sigma"], sqrt_grid, good, cut=sample.SNR_CUT, upper=p["u_upper"])
t_grid = np.geomspace(*TEMPLATE_GRID)
log_out, log_pass_out = dynamics.template_tables(p["u"], p["u_sigma"], tmpl["w"], tmpl["pair_density"], t_grid,
                                                 cut=sample.SNR_CUT, upper=p["u_upper"])
lf = np.log(tmpl["pair_fraction"] / (1.0 - tmpl["pair_fraction"]))

surface = parsec.ParsecMassSurface()
model = mlr.MonotoneMLR(surface)
model.set_data(absg=p["absg"], z_grid=post["z_grid"], z_prob=post["prob"][p["pos"]], sqrt_grid=sqrt_grid,
               log_good=lookup["log_good"], log_pass_good=log_pass_good, template_grid=t_grid, log_out=log_out,
               log_pass_out=log_pass_out, f_offset=lf - lf.mean())
mcmc = model.run_mcmc(num_warmup=args.warmup, num_samples=args.samples, num_chains=args.chains)
mcmc.print_summary()
s = {k: np.asarray(v) for k, v in mcmc.get_samples().items()}
theta = np.stack([model.theta_from_raw({k: s[k][i] for k in mlr.PARAMS}) for i in range(s["c0"].shape[0])])
extra = mcmc.get_extra_fields()
np.savez_compressed(RESULTS / "mlr_posterior.npz", theta=theta, knots_x=mlr.KNOTS_X, knots_z=mlr.KNOTS_Z,
                    divergences=int(np.sum(extra["diverging"])), **s)

mg, zz = np.round(np.arange(3.5, 13.5 + 1e-9, 0.05), 2), np.round(np.arange(-1.0, 0.6 + 1e-9, 0.05), 2)
idx = np.linspace(0, theta.shape[0] - 1, min(4000, theta.shape[0])).astype(int)
mass = 10.0 ** np.stack([model.log_mass(mg[None, :], zz[:, None], theta[i]) for i in idx])
q = np.percentile(mass, [16, 50, 84], axis=0)
par = surface.mass(mg[None, :], zz[:, None])
lines = ["MG,Z,mass_p16,mass_p50,mass_p84,mass_parsec"]
lines += [f"{m:.2f},{z:.2f},{q[0, i, k]:.6f},{q[1, i, k]:.6f},{q[2, i, k]:.6f},{par[i, k]:.6f}"
          for i, z in enumerate(zz) for k, m in enumerate(mg)]
(RESULTS / "mlr_grid.csv").write_text("\n".join(lines) + "\n")
sun = np.median(10.0 ** np.array([model.log_mass(mlr.SOLAR_MG, 0.0, theta[i]) for i in idx]))
print(f"divergences: {int(np.sum(extra['diverging']))}; mass at (M_G=4.67, Z=0): {sun:.3f} Msun")
print("wrote", RESULTS / "mlr_posterior.npz", "and", RESULTS / "mlr_grid.csv")
