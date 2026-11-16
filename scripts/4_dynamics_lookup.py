"""Rice-convolved likelihood of the good component for every pair of the MLR sample on 1024 values of
k = sqrt(M_tot) between 0.25 and 2.5.  Output: results/good_lookup.npz.  GPU recommended (minutes)."""
import json

import numpy as np

from _common import RESULTS, SQRT_GRID
from wbfit import dynamics, metallicity, sample, shape

good = shape.GoodShape(**json.loads((RESULTS / "good_shape.json").read_text())["adopted"])
post = metallicity.load_posterior(RESULTS / "metallicity_posterior.npz")
p = sample.mlr_sample(sample.read_table(), post["rows"])
grid = np.geomspace(*SQRT_GRID)
log_good = dynamics.good_lookup(p["u"], p["u_sigma"], good, grid)
np.savez(RESULTS / "good_lookup.npz", rows=p["rows"], sqrt_grid=grid, log_good=log_good)
print(f"wrote {RESULTS / 'good_lookup.npz'} ({p['rows'].size} pairs)")
