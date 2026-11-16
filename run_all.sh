#!/usr/bin/env bash
# Run the full pipeline; outputs go to results/.  Steps 2, 4 and 6 use JAX and run much faster on a GPU.
set -euo pipefail
cd "$(dirname "$0")"
for step in 1_metallicity_calibration 2_metallicity_posterior 3_good_shape 4_dynamics_lookup 5_hidden_companions 6_fit_mlr; do
    echo "=== ${step}"
    python "scripts/${step}.py"
done
