import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)
SQRT_GRID = (0.25, 2.5, 1024)        # mass-scale grid of the good-component tables, k = sqrt(M_tot)
TEMPLATE_GRID = (0.25, 2.5, 400)     # mass-scale grid of the outlier tables
