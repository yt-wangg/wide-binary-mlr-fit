"""Stage 1b: metallicity distribution of the sample and the metallicity posterior of every pair.
Output: results/metallicity_posterior.npz.  NUTS, 4 chains x 1000 draws; a few minutes on a GPU."""
import json

from _common import RESULTS
from wbfit import metallicity, sample

cal = json.loads((RESULTS / "stage1a.json").read_text())
p = sample.metallicity_sample(sample.read_table())
print(f"Stage 1b sample: {p['rows'].size} pairs")
model = metallicity.MetallicityPosterior(cal)
model.set_data(p["absg"], p["feh"], p["feh_sigma"])
model.run_mcmc()
model.mcmc.print_summary()
prob, quant = model.posterior_grid()
metallicity.save_posterior(RESULTS / "metallicity_posterior.npz", p["rows"], model.z_grid, prob, quant,
                           {"stage1a": cal, "population_weights_mean": model.weights.mean(0).tolist()})
print("wrote", RESULTS / "metallicity_posterior.npz")
