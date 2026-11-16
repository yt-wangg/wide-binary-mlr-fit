"""Shape of p(u~) for bound orbits: alpha_e = 1.233, dN/dln a ~ a^-0.6, reweighted to the observed (s, d).
Output: results/good_shape.json.  CPU, about a minute."""
import json

import numpy as np

from _common import RESULTS
from wbfit import sample, shape

t = sample.read_table()
fit = shape.fit_good_shape(np.asarray(t["sep_AU"], float), 1000.0 / np.asarray(t["parallax1"], float))
res = {"fit": fit, "adopted": shape.rounded(fit)}
(RESULTS / "good_shape.json").write_text(json.dumps(res, indent=2))
print(json.dumps(res, indent=2))
