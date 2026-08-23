"""O58 test: is the key set, conditioned on |K|=k, uniform over k-sets?  The
dilation symmetry zeta -> t*zeta (t in (Z/p)^x / ±1) is a theorem of the
ensemble, so the honest null is uniform over ORBITS of k-sets; within an orbit
equal mass is forced.  Also: collision (sum-closed trio) rate vs uniform null."""
import json, sys, itertools, math
from collections import Counter
import numpy as np
f = sys.argv[1]; d = json.load(open(f)); p = d["summary"]["p"]; nf = (p - 1) // 2
def fold(x): x %= p; return min(x, p - x)
def orbit_rep(S):
    return min(tuple(sorted(fold(t * s) for s in S)) for t in range(1, nf + 1))
rows = [r for r in d["rows"] if 0 < r["k"] < nf]
print(f"{f}: p={p}, {len(rows)} runs with 0<|K|<{nf}; k hist {Counter(r['k'] for r in rows)}")
for k in sorted(set(r["k"] for r in rows)):
    rr = [tuple(r["K"]) for r in rows if r["k"] == k]
    n_orb = math.comb(nf, k) // nf          # free action -> all orbits size nf (except k=0,nf)
    orb = Counter(orbit_rep(S) for S in rr)
    exp = len(rr) / n_orb
    # chi2 vs uniform over orbits (crude; many cells small)
    chi2 = sum((orb.get(o, 0) - exp) ** 2 / exp for o in set(orb)) + (n_orb - len(orb)) * exp
    # exact-set repeats vs within-orbit uniform: for each orbit, sets should be ~uniform over nf members
    rep = Counter(rr)
    print(f" k={k}: n={len(rr)}, orbits occupied {len(orb)}/{n_orb} (uniform expects ~{n_orb*(1-(1-1/n_orb)**len(rr)):.1f}), "
          f"chi2={chi2:.0f} on {n_orb-1} df; top orbits {orb.most_common(4)}; top exact sets {rep.most_common(3)}")
    # collisions
    def coll(S):
        for i, j in itertools.combinations(S, 2):
            for q in (fold(i + j), fold(i - j)):
                if q in S and q not in (i, j): return True
        return False
    rng = np.random.default_rng(0)
    null = np.mean([coll(set(rng.choice(np.arange(1, nf + 1), k, replace=False))) for _ in range(4000)])
    print(f"        collision rate obs {np.mean([coll(set(S)) for S in rr]):.3f} vs uniform null {null:.3f}")
