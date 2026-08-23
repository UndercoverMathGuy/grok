"""Analyzer for the mirror cohort (prereg: mirror/README.md, P-S1..P-S4).

Usage: python3 mirror/analyze_mirror.py [--runs runs_mirror]
           [--out notes/mirror_summary.json]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "dial"))
sys.path.insert(0, str(ROOT / "mirror"))
from analyze_dial import committee_from_coeffs  # noqa: E402
from make_mirror import freq_image  # noqa: E402


def load(rd):
    z = np.load(rd / "spectra.npz")
    ep = np.asarray(z["epochs"])
    ok = np.asarray(z["test_acc"]) >= 0.99
    h = np.where(ok[:-1] & ok[1:])[0]
    return {"committee": committee_from_coeffs(z["coeffs"][-1]),
            "t_gen": int(ep[h[0]]) if len(h) else -1,
            "acc": float(z["test_acc"][-1])}


def boot_ci(vals, stat=np.median, n=10_000, seed=0):
    v = np.asarray(vals, float)
    rng = np.random.default_rng(seed)
    bs = stat(v[rng.integers(0, len(v), (n, len(v)))], axis=1)
    return float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=str(ROOT / "runs_mirror"))
    ap.add_argument("--out", default=str(ROOT / "notes" / "mirror_summary.json"))
    args = ap.parse_args()
    R = Path(args.runs)
    man = json.loads((ROOT / "mirror" / "arms" / "mirror_manifest.json")
                     .read_text())
    g = man["g"]
    runs = {a["tag"]: load(R / a["tag"]) | a
            for a in man["arms"] if (R / a["tag"] / "spectra.npz").exists()}
    print(f"{len(runs)}/{len(man['arms'])} runs loaded")

    V = {}
    # P-S1 install gate: realized committee subset of installed targets
    inst = [r for r in runs.values() if r["kind"] == "install"]
    gate = {r["tag"]: set(r["committee"]) <= set(r["targets"]) for r in inst}
    frac = sum(gate.values()) / len(gate) if gate else 0
    V["P-S1"] = {"pass_frac": frac, "n": len(gate),
                 "failed": sorted(t for t, ok in gate.items() if not ok),
                 "result": ("VOID (<60%)" if frac < 0.6 else
                            "PASS" if frac >= 0.8 else "FAIL/partial")}

    # P-S3 twin floor (needed for the P-S2 band)
    tw_rel, tw_map = [], 0
    for r in runs.values():
        if r["kind"] != "twin":
            continue
        o = runs.get(r["twin_of"])
        if o is None or r["t_gen"] < 0 or o["t_gen"] < 0:
            continue
        tw_rel.append(abs(r["t_gen"] - o["t_gen"])
                      / np.mean([r["t_gen"], o["t_gen"]]))
        # token perm x -> g*x maps frequency k -> fold(g^-1 * k)
        tw_map += (sorted(r["committee"])
                   == sorted(freq_image(pow(g, -1, 113), k)
                             for k in o["committee"]))
    floor = float(np.median(tw_rel)) if tw_rel else None
    V["P-S3"] = {"twin_median_rel_dev": floor, "n": len(tw_rel),
                 "committee_maps": tw_map,
                 "result": "PASS" if tw_map >= 6 else "FAIL/partial"}

    # P-S2 mirror blindness: paired TOST
    by = {}
    for r in inst:
        by[(r["cell"]["base"], r["cell"]["content"], r["cell"]["side"])] = r
    rel, signed, dropped = [], [], []
    for b in {r["cell"]["base"] for r in inst}:
        for ci in range(len(man["contents"])):
            a, m = by.get((b, ci, "orig")), by.get((b, ci, "mirr"))
            if not a or not m:
                continue
            if not (gate[a["tag"]] and gate[m["tag"]]) \
                    or a["t_gen"] < 0 or m["t_gen"] < 0:
                dropped.append(f"{b}_c{ci}")
                continue
            mu = np.mean([a["t_gen"], m["t_gen"]])
            rel.append(abs(a["t_gen"] - m["t_gen"]) / mu)
            signed.append((a["t_gen"] - m["t_gen"]) / mu)
    med_tg = np.median([r["t_gen"] for r in inst if r["t_gen"] >= 0])
    band = max(2 * (floor or 0), 2 * 5 / med_tg) if med_tg else None
    lo, hi = boot_ci(rel) if len(rel) >= 8 else (None, None)
    slo, shi = boot_ci(signed) if len(signed) >= 8 else (None, None)
    detect = slo is not None and (slo > 0 or shi < 0)
    V["P-S2"] = {"n_pairs": len(rel), "dropped": dropped,
                 "median_rel_dev": float(np.median(rel)) if rel else None,
                 "rel_dev_95CI": [lo, hi], "signed_95CI": [slo, shi],
                 "band": band,
                 "result": ("DETECTED content effect (branch)" if detect else
                            "PASS (TOST)" if hi is not None and band
                            and hi <= band else "INCONCLUSIVE (UB > band)")}

    # P-S4 anchors
    ctl = [r for r in runs.values() if r["kind"] == "control"]
    ctl_tg = sorted(r["t_gen"] for r in ctl)
    acc = (np.median([r["t_gen"] for r in inst if r["t_gen"] >= 0])
           / np.median([t for t in ctl_tg if t >= 0])
           if any(t >= 0 for t in ctl_tg) else None)
    V["P-S4"] = {"control_t_gen": ctl_tg, "install_over_control": acc,
                 "result": "PASS" if all(t >= 0 for t in ctl_tg)
                 else "FAIL/partial"}

    for k, v in V.items():
        print(f"{k}: {json.dumps(v, default=str)}")
    o = Path(args.out)
    o.parent.mkdir(parents=True, exist_ok=True)
    o.write_text(json.dumps(
        {"runs": {t: {k: v for k, v in r.items() if k != "config"}
                  for t, r in runs.items()}, "verdicts": V},
        indent=1, default=str))
    print(f"wrote {o}")


if __name__ == "__main__":
    main()
