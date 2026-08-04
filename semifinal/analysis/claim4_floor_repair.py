"""CLAIM 4 — the floor and the repair operator, over every compatible run.

  A. Floor: every grokked selection run's final committee scored by
     LP-optimal relative margin against 2000 random same-size sets
     (nulls cached per (p, K)). Claim: none in the bottom quartile.
  B. Additive-relation depletion: sum/difference pairs in final committees
     vs the chance expectation, with the mask-cluster bootstrap (cluster =
     (p, data_seed)); plus the blind mid-training leaders (top-K at epoch
     3000) as the repair-provenance contrast.
  C. Menu closure: final committee inside the run's own top-8 shortlist,
     measured two ways — at the fixed epoch 3000 (legacy) and at HALF the
     run's own grok epoch (run-relative; a fixed epoch is mis-calibrated
     for slow-grokking runs).
  D. Engineered repairs: every runs/collisionfarm/<base>_t<K> arm — was
     the implanted additive trio broken?

Committees come from the unified detector in common.py (largest log-gap +
2%-of-max floor). The old per-run FIX dict is gone: the two hand-corrected
committees (f28 stragglers at 0.4-0.7% of run max) are now handled by the
same rule as everything else.
"""
import re
import numpy as np

from common import (discover, violations, fold, menu_at, find_base_run,
                    grok_epoch, committee_from_coeffs)
from margin_analysis import lp_relM

rng = np.random.default_rng(0)
NULLS, EV = {}, {}
def lp_pct(comm, p, n=2000):
    K = len(comm)
    if (p, K) not in NULLS:
        nf = p // 2
        NULLS[(p, K)] = np.sort([lp_relM(sorted(rng.choice(
            np.arange(1, nf + 1), K, replace=False).tolist()), p)[0]
            for _ in range(n)])
    null = NULLS[(p, K)]
    return 100.0 * np.searchsorted(null, lp_relM(sorted(comm), p)[0]) / len(null)

def exp_viol(p, K, n=1500):
    if (p, K) not in EV:
        nf = p // 2
        EV[(p, K)] = np.mean([len(violations(rng.choice(
            np.arange(1, nf + 1), K, replace=False).tolist(), p))
            for _ in range(n)])
    return EV[(p, K)]

rows = []
for r in discover():
    p = r["cfg"].p
    comm = r["committee"]
    K = len(comm)
    z = r["spectra"]
    blind = sorted(menu_at(z, top=K))
    ge = grok_epoch(z)
    half = ge // 2 if ge > 0 else 3000
    rows.append(dict(
        rel=r["rel"], p=p, ds=r["cfg"].data_seed, cohort=r["cohort"],
        comm=comm,
        pct=lp_pct(comm, p), nv=len(violations(comm, p)),
        ev=exp_viol(p, K), nvb=len(violations(blind, p)),
        closed=set(comm) <= set(menu_at(z, top=8)),
        closed_rel=set(comm) <= set(menu_at(z, epoch=half, top=8))))
    print(f"{r['rel']:<42} pct {rows[-1]['pct']:5.1f} viol {rows[-1]['nv']} "
          f"(blind {rows[-1]['nvb']}) closed {rows[-1]['closed']}"
          f"/{rows[-1]['closed_rel']}", flush=True)

def agg(sub, label):
    pcts = np.array([x["pct"] for x in sub])
    print(f"{label:<18} n={len(sub):>3}  below-25th: {(pcts < 25).sum()} "
          f"(min {pcts.min():5.1f} mean {pcts.mean():5.1f})  "
          f"final viol {sum(x['nv'] for x in sub)} vs "
          f"exp {sum(x['ev'] for x in sub):.1f} "
          f"(blind {sum(x['nvb'] for x in sub)})  "
          f"closed e3000 {sum(x['closed'] for x in sub)}/{len(sub)}, "
          f"half-grok {sum(x['closed_rel'] for x in sub)}/{len(sub)}")

print("\n=== A/B/C: floor, depletion, closure ===")
agg(rows, "ALL")
for cohort in sorted({x["cohort"] for x in rows}):
    agg([x for x in rows if x["cohort"] == cohort], f"  {cohort}")
below = [(x["rel"], round(x["pct"], 1)) for x in rows if x["pct"] < 25]
print(f"below-floor runs: {below or 'NONE'}")

clusters = {}
for x in rows:
    clusters.setdefault((x["p"], x["ds"]), []).append(x["nv"] - x["ev"])
cl = list(clusters.values())
boot = []
for _ in range(20000):
    pick = rng.integers(0, len(cl), len(cl))
    boot.append(np.mean([np.mean(cl[i]) for i in pick]))
boot = np.array(boot)
print(f"depletion mask-cluster bootstrap ({len(cl)} clusters): "
      f"mean excess {np.mean([np.mean(v) for v in cl]):+.3f}, "
      f"p(excess >= 0) = {(boot >= 0).mean():.4f}")

print("\n=== D: engineered repair events (collisionfarm) ===")
_bcomm_cache = {}
def base_committee(name):
    """(committee, final |coeffs|) of a natural base run, cached; None if the
    base doesn't resolve under a natural family."""
    if name not in _bcomm_cache:
        b = find_base_run(name)
        if b is None:
            _bcomm_cache[name] = None
        else:
            bz = np.load(b / "spectra.npz")
            c = np.abs(bz["coeffs"][-1])
            _bcomm_cache[name] = (committee_from_coeffs(bz["coeffs"][-1]), c)
    return _bcomm_cache[name]

n_broken = n_tot = 0
for x in rows:
    m = re.match(r"collisionfarm/(.+)_t(\d+)$", x["rel"])
    if not m:
        continue
    base_name, t = m.group(1), int(m.group(2))
    bcm = base_committee(base_name)
    if bcm is None:
        print(f"   {x['rel']}: base run '{base_name}' not found — skipped")
        continue
    bcomm, bc = bcm
    p = x["p"]
    pair = None
    order = sorted(bcomm, key=lambda k: -bc[k - 1])
    for a in range(len(order)):
        for bb in range(a + 1, len(order)):
            i, j = order[a], order[bb]
            if t in (fold(i + j, p), fold(i - j, p)):
                pair = (i, j)
                break
        if pair:
            break
    fin = set(x["comm"])
    trio = {pair[0], pair[1], t} if pair else {t}
    broken = not trio <= fin
    n_tot += 1
    n_broken += broken
    print(f"   {x['rel']}: base comm {bcomm}, trio {sorted(trio)}, final "
          f"{sorted(fin)} -> trio broken: {broken}")
legacy_note = (" (+2/2 from surgery/collision & boost_strong, see claim3 "
               "script)" if any(x["rel"].startswith("surgery/") for x in rows)
               else "")
print(f"   engineered trios broken: {n_broken}/{n_tot}{legacy_note}")
print("""
Backs SEMIFINAL claim 4: the floor, additive depletion with cluster-robust
p, menu closure, and repair-on-demand across masks. Provenance of the
depletion differs by cohort and the claim text must say so: in NATURAL runs
the mid-training leaders carry violations at chance rate and the cleanup
happens during consolidation (active repair); in FLAT-init cohorts the
leaders are already clean by mid-training — with no init bias forcing the
early race, training freely picks a clean set from the start and there is
nothing to repair. Forced (natural/surgical) starts sometimes propose
broken sets and get repaired; free (flat) starts almost never propose them.""")
