"""Stage 1a: relative calibration of J-CaPS (b_G, s_z, nu) from the metallicity differences within pairs.
Output: results/stage1a.json.  CPU, about a minute."""
import json

from _common import RESULTS
from wbfit import metallicity, sample

res = metallicity.fit_stage1a(sample.read_table())
(RESULTS / "stage1a.json").write_text(json.dumps(res, indent=2))
print(json.dumps(res, indent=2))
