"""initaudit_calib.py -- calibrate the count instruments on EXACT constructed K-frequency x^2 circuits at p = 71.
If the instrument is faithful, a circuit built on exactly K frequencies must read K."""
import json, math, sys
from pathlib import Path
import numpy as np, torch
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from nsum_rep import Data, analyse  # noqa: E402
KS = json.loads((ROOT / "kstar.json").read_text())
torch.set_default_dtype(torch.float64)
p = 71
data64 = Data(p, 2, 400_000, "cpu", dtype=torch.float64, prod=False)
print("  K_built | K_2pct K_5pct N_.05 | PR    | clean_frac | gamma")
for K in range(3, 13):
    S = KS[str(p)]["S"][str(K)]
    a = torch.arange(p) * 2 * math.pi / p
    ph = [(2 * math.pi * l / 6, 2 * math.pi * 4 * l / 6) for l in range(6)]
    m = len(S) * 6
    E = torch.zeros(2, m, p); W = torch.zeros(p, m); i = 0
    for k in S:
        for p1, p2 in ph:
            E[0, i] = torch.cos(k * a + p1); E[1, i] = torch.cos(k * a + p2)
            W[:, i] = torch.cos(k * a + p1 + p2); i += 1
    r = analyse(E, W, data64, "pow")
    en = np.array(r["amp"]) ** 2; en = en / en.sum()
    print(f"  {K:7d} | {r['K_2pct']:6d} {r['K_5pct']:6d} {r['N_05']:5d} | {1/(en**2).sum():5.2f} | "
          f"{r['clean_frac']:.4f} | {r['gamma']:.4e}")
