"""L-side content dissociation — the PRE-REGISTERED analysis.

Committed before Phase C trains.

The question, mirroring the S-side: does moving the LEARNING trajectory
rewrite WHAT was selected? The lever is a late data reveal — frac_train
0.3 -> 0.7 at donor epoch t, same data_seed. The split is a seeded shuffle
truncated at frac*N, so the larger training set is an exact SUPERSET of the
smaller one (verified: 0 examples change side, 5108 revealed, 2.33x). No
parameter, optimizer state, or hyperparameter is touched — data is only
revealed.

What Phase C fixes relative to `boost/` round 2, where this result
currently lives: donors are the 8 mainline natural runs at wd = 1.0, not 8
wd = 3.0 donors of which 2 never reached the threshold (effective n was 6);
resumes come from the donors' own save_every=500 checkpoints; and every
jump time carries its own self-restart gate, not just the last.

DONOR-RELATIVE TIME. The ladder is absolute (500..10000) but the analysis
is in phi = t / donor_t_gen, because a fixed epoch means different things
to different donors — t=4000 is mid-commitment for a 13400-epoch donor and
nearly at-grok for a 5100-epoch one. Binning on phi makes the arms
comparable and removes any data-dependent choice of which timepoint scores.

CONDITIONAL SPEEDUP, not raw ratio. The obvious statistic,
abs_t_gen / donor_t_gen, has an arithmetic floor of phi: a run resumed at t
cannot finish before t, so once phi > 0.7 a "<= 0.7" bar is unreachable no
matter how well the reveal works. Checked against the observed natural
t_gen spread, that floor is 0.72 at t=6000 and 0.94 at t=8000 — the bar
would have failed by arithmetic, not by physics. What the reveal can
actually change is the learning that REMAINS, so the scored quantity is

    (abs_t_gen - t) / (donor_t_gen - t)

which is 1.0 under no effect and has no floor. The old P-B8 pass (0.68)
was already skating its floor of ~0.53 and should be re-read this way.

THE GATE is what makes an arm interpretable: resuming at t with the data
UNCHANGED must reproduce the donor's own committee. If a gate fails, its
jump arm is VOID, not evidence — a cold-Adam restart that re-runs
selection by itself cannot tell us what the data reveal did.

Pre-registered endpoints, ONE primary:
  P-CL1  PRIMARY. Lock holds under the reveal: J(jump, donor) in the
         decisive phi band, scored by the lower end of a bootstrap CI
         against 0.8, with the band's gate median >= 0.8. A lower J with a
         clean gate is NOT a failure to hide — it means lock-in stays
         data-conditional, reported as a first-class finding (which is what
         happened at t=200 in boost round 1).
  P-CL2  VALIDITY CHECK. Conditional speedup <= 0.7 in the decisive band.
         Without it the reveal moved nothing and P-CL1 is vacuous.
  P-CL3  DESCRIPTIVE. Lock hardening: Spearman trend of J against phi.
         Scored as a trend, not exact monotonicity — seven medians in a row
         are never wiggle-free, and exact monotonicity would fail on noise.
  Post-grok arms (phi >= 1) are a stability check: reported, not scored.

Usage:
    uv run python compiler/analyze_lside.py
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "dial"))
from analyze_dial import committee_from_coeffs, jaccard  # noqa: E402

J_BAR = 0.8              # pre-registered lock-holds bar
SPEEDUP_BAR = 0.7        # pre-registered conditional-speedup bar
BAND = (0.4, 0.9)        # decisive phi band: post-lock, pre-grok
N_BOOT = 20_000


def read(run):
    f = Path(run) / "spectra.npz"
    if not f.exists():
        return None
    z = np.load(f)
    ep, acc = np.asarray(z["epochs"]), np.asarray(z["test_acc"])
    ok = acc >= 0.99
    h = np.where(ok[:-1] & ok[1:])[0]
    return dict(committee=set(committee_from_coeffs(z["coeffs"][-1])),
                t_gen=float(ep[h[0]]) if len(h) else None,
                acc=float(acc[-1]))


def med(v):
    v = [x for x in v if x is not None]
    return float(np.median(v)) if v else None


def boot_ci(v, lo=2.5, hi=97.5, seed=0):
    """Percentile bootstrap CI on the median."""
    v = np.array([x for x in v if x is not None], float)
    if len(v) < 2:
        return (None, None)
    rng = np.random.default_rng(seed)
    b = np.median(rng.choice(v, (N_BOOT, len(v)), replace=True), axis=1)
    return (float(np.percentile(b, lo)), float(np.percentile(b, hi)))


def spearman(x, y):
    """Rank correlation, without pulling in scipy."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3:
        return None
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    return float(np.corrcoef(rx, ry)[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest",
                    default=str(ROOT / "compiler/arms/phaseC_manifest.json"))
    ap.add_argument("--out", default=str(ROOT / "notes/lside_summary.json"))
    args = ap.parse_args()

    arms = [a for a in json.load(open(args.manifest))
            if a["kind"].startswith("lside")]
    if not arms:
        sys.exit("no L-side arms in manifest")

    donors = {}
    for a in arms:
        b = a["base"]
        if b not in donors:
            donors[b] = read(ROOT / "runs_torch" / b)
    missing = [b for b, d in donors.items() if d is None]
    if missing:
        sys.exit(f"missing donor runs: {missing}")
    censored = [b for b, d in donors.items() if d["t_gen"] is None]
    if censored:
        print(f"WARNING: {len(censored)} donor(s) never generalised; their "
              f"arms have no reference and are dropped: {censored}")

    rows = []
    for a in arms:
        r = read(a["run_dir"])
        d = donors[a["base"]]
        if r is None or d["t_gen"] is None:
            continue
        t = float(a["cell"]["t_jump"])
        abs_t = (t + r["t_gen"]) if r["t_gen"] is not None else None
        phi = t / d["t_gen"]
        rows.append(dict(
            base=a["base"], kind=a["kind"], t=t, phi=phi,
            J=jaccard(r["committee"], d["committee"]),
            abs_t_gen=abs_t, donor_t_gen=d["t_gen"], acc=r["acc"],
            speedup=(None if abs_t is None or phi >= 1.0
                     else (abs_t - t) / (d["t_gen"] - t))))
    if not rows:
        sys.exit(f"no completed L-side arms under {args.manifest}")

    dt = [d["t_gen"] for d in donors.values() if d["t_gen"] is not None]
    print(f"n={len(rows)} L-side arms, {len(dt)} usable donors "
          f"(t_gen median {np.median(dt):.0f}, range {min(dt):.0f}-{max(dt):.0f})")

    f = lambda v: "-" if v is None else f"{v:.3f}"
    print(f"\n{'phi band':>14} {'n':>4} {'J_jump':>7} {'J_gate':>7} "
          f"{'cond.speedup':>13}")
    edges = [0, .2, .4, .6, .8, 1.0, 99]
    for lo, hi in zip(edges, edges[1:]):
        j = [r for r in rows if r["kind"] == "lside_jump" and lo <= r["phi"] < hi]
        g = [r for r in rows if r["kind"] == "lside_gate" and lo <= r["phi"] < hi]
        if not j:
            continue
        lbl = f"{lo:.1f}-{hi:.1f}" if hi < 99 else f">={lo:.1f} (post-grok)"
        print(f"{lbl:>14} {len(j):>4} {f(med([r['J'] for r in j])):>7} "
              f"{f(med([r['J'] for r in g])):>7} "
              f"{f(med([r['speedup'] for r in j])):>13}")

    lo, hi = BAND
    dj = [r for r in rows if r["kind"] == "lside_jump" and lo <= r["phi"] < hi]
    dg = [r for r in rows if r["kind"] == "lside_gate" and lo <= r["phi"] < hi]
    print(f"\ndecisive band phi in [{lo}, {hi}): {len(dj)} jump / {len(dg)} gate"
          "  (post-lock, pre-grok: commitment hardest, speedup measurable)")

    V = {}
    if not dj or not dg:
        V["P-CL1"] = {"primary": True,
                      "result": "VOID (no arms landed in the decisive band)"}
    else:
        gj = med([r["J"] for r in dg])
        jj = med([r["J"] for r in dj])
        ci = boot_ci([r["J"] for r in dj])
        gate_ok = gj is not None and gj >= J_BAR
        V["P-CL1"] = {
            "primary": True, "band": list(BAND), "n": len(dj),
            "J_jump": jj, "J_jump_CI95": list(ci), "J_gate": gj,
            "result": ("VOID (gate failed — a cold restart re-runs selection "
                       "by itself here, so the reveal is unreadable)"
                       if not gate_ok else
                       "PASS" if ci[0] is not None and ci[0] >= J_BAR else
                       "FAIL -> lock-in stays data-conditional post-lock "
                       "(reported as a finding, not hidden)")}
        sp = med([r["speedup"] for r in dj])
        spci = boot_ci([r["speedup"] for r in dj])
        V["P-CL2"] = {"validity_check": True, "cond_speedup": sp,
                      "cond_speedup_CI95": list(spci),
                      "result": ("PASS" if spci[1] is not None
                                 and spci[1] <= SPEEDUP_BAR else
                                 "FAIL -> the reveal did not move learning; "
                                 "P-CL1 is vacuous in this cohort")}

    jr = [r for r in rows if r["kind"] == "lside_jump"]
    rho = spearman([r["phi"] for r in jr], [r["J"] for r in jr])
    V["P-CL3"] = {"descriptive": True, "spearman_rho_J_vs_phi": rho,
                  "result": ("PASS" if rho is not None and rho > 0
                             else "FAIL (no hardening trend)")}
    ps = [r["J"] for r in jr if r["phi"] >= 1.0]
    V["post_grok_stability"] = {
        "n": len(ps), "median_J": med(ps),
        "note": "reveal after the donor generalised: stability only, no "
                "speedup is measurable — reported, not scored"}

    print()
    for k, v in V.items():
        tag = ("PRIMARY  " if v.get("primary") else
               "validity " if v.get("validity_check") else
               "descript " if v.get("descriptive") else "         ")
        print(f"  {tag}{k}: {v.get('result', v.get('note'))}")
    print("\nL-SIDE VERDICT (primary endpoint P-CL1 alone): "
          f"{V.get('P-CL1', {}).get('result', 'n/a')}")

    o = Path(args.out)
    o.parent.mkdir(parents=True, exist_ok=True)
    o.write_text(json.dumps(
        {"rows": rows, "verdicts": V,
         "donor_t_gen": {b: d["t_gen"] for b, d in donors.items()}},
        indent=1, default=str))
    print(f"wrote {o}")


if __name__ == "__main__":
    main()
