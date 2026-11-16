"""PARSEC reference mass surface M(M_G, [M/H]).

``data/parsec_mass`` holds the mass of PARSEC v1.2S isochrones of log10(age/yr) = 9.6 on a grid of M_G (smoothing
spline in M_G for each [M/H] = -1.0, -0.8, ..., 0.6).  The surface is interpolated bilinearly and restricted to
M_G <= 13.5.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from .sample import ROOT

PARSEC_DIR = ROOT / "data" / "parsec_mass"


class ParsecMassSurface:
    def __init__(self, data_dir=PARSEC_DIR, mg_max=13.5, mass_min=0.05):
        data_dir = Path(data_dir)
        x = np.load(data_dir / "gmag_grid.npy").astype(float)
        files = {float(re.match(r"mass_interp_MH_(-?\d+\.\d+)\.npy", f.name).group(1)): f
                 for f in data_dir.glob("mass_interp_MH_*.npy")}
        self.mh_grid = np.array(sorted(files))
        mass = np.vstack([np.load(files[mh]).astype(float) for mh in self.mh_grid])
        self.mass_min = float(mass_min)
        self.absg_grid, self.mass_grid = x, mass
        keep = x < mg_max - 1e-12
        last = np.array([self.mass(mg_max, mh) for mh in self.mh_grid])
        self.absg_grid = np.concatenate([x[keep], [mg_max]])
        self.mass_grid = np.column_stack([mass[:, keep], last])

    def mass(self, absg, mh):
        """Bilinear interpolation (linear in M_G, then in [M/H]), clipped to the grid."""
        absg, mh = np.broadcast_arrays(np.asarray(absg, float), np.asarray(mh, float))
        a = np.clip(absg, self.absg_grid[0], self.absg_grid[-1])
        m = np.clip(mh, self.mh_grid[0], self.mh_grid[-1]).reshape(-1)
        curves = np.stack([np.interp(a, self.absg_grid, row) for row in self.mass_grid]).reshape(self.mh_grid.size, -1)
        out = np.array([np.interp(m[j], self.mh_grid, curves[:, j]) for j in range(m.size)]).reshape(a.shape)
        return np.where(out > 0.0, out, self.mass_min)
