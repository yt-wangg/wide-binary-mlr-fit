"""Forward model of hidden inner companions: local outlier templates h_j(u~) and fractions r_j of the pairs of the
MLR sample.  Output: results/outlier_templates.npz.  CPU, about ten minutes."""
import json

import numpy as np

from _common import RESULTS
from wbfit import companions, metallicity, parsec, sample, shape

c = companions.SETTINGS
post = metallicity.load_posterior(RESULTS / "metallicity_posterior.npz")
p = sample.mlr_sample(sample.read_table(), post["rows"])
z = post["quant"][p["pos"], 1]                       # median metallicity of each pair
good = shape.GoodShape(**json.loads((RESULTS / "good_shape.json").read_text())["adopted"])
rng = np.random.default_rng(c["seed"])
rel = companions.ParsecRelation(parsec.ParsecMassSurface())
sim = companions.simulate(p, z, rel, companions.good_sampler(good, rng), rng)
out = companions.local_templates(p, sim)
np.savez_compressed(RESULTS / "outlier_templates.npz", **out, settings=json.dumps(c))
f = out["pair_fraction"]
print(f"{p['rows'].size} pairs; local fraction r_j 5/50/95% = {np.round(np.percentile(f, [5, 50, 95]), 4)}")
