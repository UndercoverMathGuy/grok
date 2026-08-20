"""Score the T2-D forecasts against realized t_gen (prereg P-F1..P-F4).
Stitches seg1 (0..500) + seg2 (500..30000, epochs stored relative)."""
import json
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
F = json.load(open(ROOT / "flight" / "forecasts_t2d.json"))
FORECAST_EPOCH = F["meta"]["forecast_epoch"]


def fs(m):
    i = np.where(m[:-1] & m[1:])[0]
    return int(i[0]) if len(i) else -1


def realized_t_gen(rd):
    z1 = np.load(rd / "spectra.npz")
    if fs(z1["test_acc"] >= 0.9) >= 0:
        return int(z1["epochs"][fs(z1["test_acc"] >= 0.9)])
    z2 = np.load(rd / "seg2" / "spectra.npz")
    ig = fs(z2["test_acc"] >= 0.9)
    return None if ig < 0 else FORECAST_EPOCH + int(z2["epochs"][ig])


hits = 0
pairs = []
print(f"{'run':38} {'forecast':>9} {'realized':>9} {'ratio':>6}  verdict")
for row in F["forecasts"]:
    rd = ROOT / "runs_flight" / row["run"]
    act = realized_t_gen(rd)
    pred = row["forecast"]
    if pred == "CENSORED":
        ok = act is None
        print(f"{row['run']:38} {'CENS':>9} {act if act else 'CENS':>9} "
          f"{'':>6}  {'HIT' if ok else 'MISS'}")
    elif act is None:
        ok = False
        print(f"{row['run']:38} {pred:>9} {'CENS':>9} {'':>6}  MISS")
    else:
        ratio = max(pred / act, act / pred)
        ok = ratio <= 1.5
        pairs.append((pred, act))
        print(f"{row['run']:38} {pred:>9} {act:>9} {ratio:>6.2f}  "
              f"{'HIT' if ok else 'MISS'}")
    hits += ok
n = len(F["forecasts"])
print(f"\nP-F1: {hits}/{n} within bars (prereg bar >= 11/15) -> "
      f"{'PASS' if hits >= 11 else 'FAIL'}")
if len(pairs) >= 4:
    a = np.array(pairs, float)
    ra = np.argsort(np.argsort(a[:, 0]))
    rb = np.argsort(np.argsort(a[:, 1]))
    rho = float(np.corrcoef(ra, rb)[0, 1])
    print(f"P-F2: Spearman {rho:.3f} over {len(pairs)} numeric pairs "
          f"(bar >= 0.8) -> {'PASS' if rho >= 0.8 else 'FAIL'}")

# P-F4 floor arm
print("\nfloor arm:")
ok4 = 0
tot4 = 0
for rd in sorted((ROOT / "runs_flight" / "t2d_floor").glob("*/*")):
    if not (rd / "spectra.npz").exists():
        continue
    z = np.load(rd / "spectra.npz")
    im, ig = fs(z["train_acc"] >= 0.99), fs(z["test_acc"] >= 0.9)
    tot4 += 1
    if im >= 0 and ig >= 0:
        wait = int(z["epochs"][ig]) - int(z["epochs"][im])
        good = wait <= 300
        ok4 += good
        print(f"  {rd.parent.name}/{rd.name}: wait {wait} "
              f"{'HIT' if good else 'MISS'}")
    else:
        print(f"  {rd.parent.name}/{rd.name}: no mem/gen -> MISS")
print(f"P-F4: {ok4}/{tot4} waits <= 300 (bar >= 7/8) -> "
      f"{'PASS' if ok4 >= 7 else 'FAIL'}")
