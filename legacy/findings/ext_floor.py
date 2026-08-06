"""Paper-grade floor + depletion re-scoring on the full zoo (claim-4
re-review): LP-optimal relM percentiles (cached nulls per (p,K)), additive
depletion with mask-cluster bootstrap, and blind-vs-final repair provenance
— on every grokked spectra-logged selection run.

The dose_110 committee is corrected for the known detector artifact
(background f28 at |coeff| 172; real set {7,14,49,52}).
"""
import json, sys
from pathlib import Path
import numpy as np

ROOT = Path("/Users/ruhaanrajadhyaksha/projects/grok")
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "findings"))
sys.path.insert(0, str(ROOT / "scripts"))
from mask_lottery import committee_from_coeffs
from margin_analysis import lp_relM

SELECTION = {"og_seed0", "seed0", "seed1", "seed2", "p-113", "p-127", "p-157",
             "orthWE", "phase2-noise", "phase2-noise2", "phase2-tilt",
             "eff-A", "eff-B", "eff-C", "eff-D", "eff-E", "eff-G", "combined",
             "surgery", "surgery2", "transplant"}
NATURAL = {"og_seed0", "seed0", "seed1", "seed2", "p-113", "p-127", "p-157"}
FIX = {"surgery2/dose_110": [7, 14, 49, 52]}   # detector artifact correction

rng = np.random.default_rng(0)
NULLS = {}
def lp_pct(comm, p, n=2000):
    K = len(comm)
    if (p, K) not in NULLS:
        nf = p // 2
        NULLS[(p, K)] = np.sort([lp_relM(sorted(rng.choice(
            np.arange(1, nf + 1), K, replace=False).tolist()), p)[0]
            for _ in range(n)])
    null = NULLS[(p, K)]
    return 100.0 * np.searchsorted(null, lp_relM(sorted(comm), p)[0]) / len(null)

def fold(x, p):
    x %= p
    return min(x, p - x)

def nviol(S, p):
    S = sorted(S); ss = set(S); c = 0
    for i in range(len(S)):
        for j in range(i + 1, len(S)):
            if fold(S[i] + S[j], p) in ss or fold(S[i] - S[j], p) in ss:
                c += 1
    return c

EV = {}
def exp_viol(p, K, n=1500):
    if (p, K) not in EV:
        nf = p // 2
        EV[(p, K)] = np.mean([nviol(rng.choice(np.arange(1, nf + 1), K,
                              replace=False).tolist(), p) for _ in range(n)])
    return EV[(p, K)]

rows = []
for cj in sorted((ROOT / "runs").rglob("config.json")):
    d = cj.parent
    fam = str(d.relative_to(ROOT / "runs")).split("/")[0]
    if fam not in SELECTION or not (d / "spectra.npz").exists():
        continue
    c = json.loads(cj.read_text())
    z = np.load(d / "spectra.npz")
    if float(z["test_acc"][-1]) < 0.99:
        continue
    p = c["p"]
    rel = str(d.relative_to(ROOT / "runs"))
    comm = FIX.get(rel) or committee_from_coeffs(z["coeffs"][-1])
    K = len(comm)
    i3k = int(np.argmin(np.abs(z["epochs"] - 3000)))
    blind = sorted((np.argsort(np.abs(z["coeffs"][i3k]))[::-1][:K] + 1).tolist())
    rows.append(dict(rel=rel, fam=fam, p=p, ds=c["data_seed"], K=K,
                     natural=fam in NATURAL, pct=lp_pct(comm, p),
                     nv=nviol(comm, p), ev=exp_viol(p, K),
                     nv_blind=nviol(blind, p)))
    r = rows[-1]
    print(f"{rel:<40} K={K} lp-pct {r['pct']:5.1f} viol {r['nv']} "
          f"(blind {r['nv_blind']})", flush=True)

def agg(sub, label):
    pcts = np.array([r["pct"] for r in sub])
    obs = sum(r["nv"] for r in sub); exp = sum(r["ev"] for r in sub)
    ob = sum(r["nv_blind"] for r in sub)
    print(f"{label:<24} n={len(sub):>3}  LP-floor: {(pcts<25).sum()} below "
          f"25th (min {pcts.min():5.1f} mean {pcts.mean():5.1f})  "
          f"final viol {obs} vs exp {exp:.1f} (blind {ob})")

print()
agg(rows, "ALL")
agg([r for r in rows if r["natural"]], "natural")
agg([r for r in rows if not r["natural"]], "intervention")

# mask-cluster bootstrap for final-committee depletion (cluster = (p, ds))
clusters = {}
for r in rows:
    clusters.setdefault((r["p"], r["ds"]), []).append(r["nv"] - r["ev"])
means = [np.mean(v) for v in clusters.values()]
boot = []
cl = list(clusters.values())
for _ in range(20000):
    pick = rng.integers(0, len(cl), len(cl))
    boot.append(np.mean([np.mean(cl[i]) for i in pick]))
boot = np.array(boot)
print(f"\nmask-cluster bootstrap (clusters={len(cl)}): mean excess "
      f"{np.mean(means):+.3f}, p(excess >= 0) = {(boot >= 0).mean():.4f}")
below = [(r["rel"], round(r["pct"], 1)) for r in rows if r["pct"] < 25]
print(f"below-floor runs: {below}")
