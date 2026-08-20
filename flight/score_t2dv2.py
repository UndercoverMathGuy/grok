"""Score T2-D v2 (and the v1 shadow ablation) per flight/PREREG_T2D_V2.md."""
import json
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
F = json.load(open(ROOT / "flight" / "forecasts_t2dv2.json"))
FE = F["meta"]["forecast_epoch"]


def fs(m):
    i = np.where(m[:-1] & m[1:])[0]
    return int(i[0]) if len(i) else -1


def realized(rd):
    z1 = np.load(rd / "spectra.npz")
    if fs(z1["test_acc"] >= 0.9) >= 0:
        return int(z1["epochs"][fs(z1["test_acc"] >= 0.9)])
    z2 = np.load(rd / "seg2" / "spectra.npz")
    ig = fs(z2["test_acc"] >= 0.9)
    return None if ig < 0 else FE + int(z2["epochs"][ig])


def scored(pred, act):
    if pred == "CENSORED":
        return act is None, None
    if act is None or pred is None:
        return False, None
    r = max(pred / act, act / pred)
    return r <= 1.5, r


res = {"v2": 0, "v1_shadow": 0}
pairs = []
print(f"{'run':34} {'v2':>9} {'v1':>9} {'realized':>9}  v2/v1")
for row in F["forecasts"]:
    act = realized(ROOT / "runs_flight" / row["run"])
    marks = []
    for key in ("v2", "v1_shadow"):
        ok, ratio = scored(row[key].get("forecast"), act)
        res[key] += ok
        marks.append("HIT " if ok else "MISS")
        if key == "v2" and isinstance(row[key].get("forecast"), (int, float)) \
                and act is not None:
            pairs.append((row[key]["forecast"], act))
    print(f"{row['run']:34} {str(row['v2'].get('forecast')):>9} "
          f"{str(row['v1_shadow'].get('forecast')):>9} "
          f"{str(act if act else 'CENS'):>9}  {marks[0]}/{marks[1]}")
n = len(F["forecasts"])
print(f"\nP-F1v2: {res['v2']}/{n} within bars (bar >= 9) -> "
      f"{'PASS' if res['v2'] >= 9 else 'FAIL'}")
if len(pairs) >= 4:
    a = np.array(pairs, float)
    ra, rb = np.argsort(np.argsort(a[:, 0])), np.argsort(np.argsort(a[:, 1]))
    rho = float(np.corrcoef(ra, rb)[0, 1])
    print(f"P-F2v2: Spearman {rho:.3f} (bar >= 0.8) -> "
          f"{'PASS' if rho >= 0.8 else 'FAIL'}")
print(f"P-F5 ablation: v2 {res['v2']} vs v1-shadow {res['v1_shadow']} "
      f"on identical runs -> {'PASS' if res['v2'] > res['v1_shadow'] else 'FAIL'}")
