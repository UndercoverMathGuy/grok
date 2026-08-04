"""Efficiency scoreboard: realized relM / amplitude / norm per arm, vs tilt-5 wd-1 control."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from margin_analysis import committee, lp_relM, null_percentile, homeostat
from orthwe_analysis import discover
from grok.config import Config
from grok.model import Transformer
from grok.metrics import sum_sq_weights
from grok.train import checkpoint_epochs

ARMS = [("tilt5/wd1 (ctrl)", "runs/phase2-tilt/p-113"),
        ("B tilt5/wd2.5", "runs/eff-B/p-113"),
        ("A tilt15/wd1", "runs/eff-A/p-113"),
        ("C tilt15/wd2.5", "runs/eff-C/p-113")]
rng = np.random.default_rng(0)
print(f"{'arm':<17}{'seed':<7}{'K':<3}{'pct':<7}{'lp_relM':<9}{'relM*':<7}{'A_tot':<8}{'minGap':<8}{'norm':<6}")
for name, root in ARMS:
    rows = []
    for dseed, iseed, d in discover(Path(root)):
        z = np.load(d / "spectra.npz")
        comm, _ = committee(z["coeffs"][-1])
        lpM, _ = lp_relM(comm, 113)
        pct, _, _ = null_percentile(comm, 113, lp_relM, n=3000, rng=rng)
        A, relM, gap = homeostat(d, 113)
        cfg = Config.load(d / "config.json")
        m = Transformer(cfg); m.load_weights(str(checkpoint_epochs(d)[-1][1]))
        norm = sum_sq_weights(m)
        rows.append((pct, relM, A, gap, norm))
        print(f"{name:<17}{iseed:<7}{len(comm):<3}{pct:<7.1f}{lpM:<9.3f}{relM:<7.3f}{A:<8.1f}{gap:<8.2f}{norm:<6.0f}")
    if rows:
        r = np.array(rows)
        print(f"{name:<17}{'MEAN':<7}{'':<3}{r[:,0].mean():<7.1f}{'':<9}{r[:,1].mean():<7.3f}{r[:,2].mean():<8.1f}{r[:,3].mean():<8.2f}{r[:,4].mean():<6.0f}")
    print()
