"""Adversarial re-analysis of T16 (additive-pair depletion, p=0.0018).

Attack A: dependence. 14/31 committees share data_seed 0 (p=113) and
committees within a mask overlap heavily. The null treats 31 committees as
independent. Redo inference with a mask-cluster bootstrap on per-committee
excess (obs - null mean).

Attack B: provenance. The claim is that depletion is the *repair operator's*
fingerprint (consolidation avoids additive pairs). If the blind e3000
amplitude top-K draws are ALREADY depleted, the fingerprint belongs to the
audition/amplification stage, not repair. Paired comparison blind vs final
on the 24 spectra runs + same census on blind draws.

Attack C: null mismatch. Real committees are amplitude winners; if amplitude
winners have a non-uniform marginal over frequencies, sums/differences may
fold into the committee at a different base rate. Popularity-matched null:
draw committees iid from the empirical popularity distribution of that mask.

Attack D: detector sensitivity. Recompute with the log-gap cut searched in
top-8 and top-16, and with a fixed K=4 top-amplitude rule.
"""
import sys
from pathlib import Path
import numpy as np

ROOT = Path("/Users/ruhaanrajadhyaksha/projects/grok")
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "findings"))
from mask_lottery import discover, final_coeffs_and_acc, committee_from_coeffs

def fold(x, p):
    x = x % p
    return min(x, p - x)

def n_additive(comm, p):
    s = set(comm); c = 0
    for i in range(len(comm)):
        for j in range(i + 1, len(comm)):
            if fold(comm[i] + comm[j], p) in s or fold(comm[i] - comm[j], p) in s:
                c += 1
    return c

def committee_cut(coeffs, top):
    a = np.abs(coeffs)
    order = np.argsort(a)[::-1]
    logs = np.log(a[order] + 1e-12)
    gaps = logs[:-1] - logs[1:]
    cut = int(np.argmax(gaps[:top])) + 1
    return sorted((order[:cut] + 1).tolist())

rows = []
for d, cfg in discover():
    coeffs, acc, z = final_coeffs_and_acc(d, cfg)
    if acc < 0.99:
        continue
    rows.append(dict(name=str(d.relative_to(ROOT/'runs')), p=cfg.p,
                     mask=(cfg.p, cfg.data_seed), coeffs=coeffs, z=z,
                     comm=committee_from_coeffs(coeffs)))

rng = np.random.default_rng(0)

def null_mean_sd(p, K, n=4000):
    vals = np.empty(n)
    for t in range(n):
        vals[t] = n_additive(rng.choice(np.arange(1, p//2+1), K, replace=False).tolist(), p)
    return vals.mean(), vals

print(f"{len(rows)} committees; mask sizes:")
from collections import Counter
print("  ", Counter(r["mask"] for r in rows))

# ---------------- Attack A: cluster bootstrap on per-committee excess
null_cache = {}
def nmean(p, K):
    if (p, K) not in null_cache:
        null_cache[(p, K)] = null_mean_sd(p, K)
    return null_cache[(p, K)][0]

excess = {}
for r in rows:
    e = n_additive(r["comm"], r["p"]) - nmean(r["p"], len(r["comm"]))
    excess.setdefault(r["mask"], []).append(e)

masks = list(excess)
per_mask_mean = {m: np.mean(v) for m, v in excess.items()}
print("\nper-mask mean excess (obs - null):")
for m, v in sorted(per_mask_mean.items()):
    print(f"  p={m[0]} ds={m[1]}: {v:+.3f} (n={len(excess[m])})")
obs_total = sum(np.sum(v) for v in excess.values())
print(f"total observed excess: {obs_total:+.2f} "
      f"(obs {sum(n_additive(r['comm'], r['p']) for r in rows)}, "
      f"null {sum(nmean(r['p'], len(r['comm'])) for r in rows):.2f})")

# cluster bootstrap: resample masks with replacement, mean of mask-mean excess
boot = np.empty(20000)
mm = np.array([per_mask_mean[m] for m in masks])
for t in range(20000):
    boot[t] = rng.choice(mm, len(mm), replace=True).mean()
p_boot = (boot >= 0).mean()
print(f"cluster bootstrap (mask as unit, n={len(masks)}): mean excess "
      f"{mm.mean():+.3f}, p(excess >= 0) = {p_boot:.4f}")

# also: original-style pooled permutation but drawing ONE committee per mask
pm = np.empty(20000)
by_mask = {}
for r in rows:
    by_mask.setdefault(r["mask"], []).append(r)
reps = [by_mask[m][0] for m in masks]  # first committee per mask
obs_rep = sum(n_additive(r["comm"], r["p"]) for r in reps)
for t in range(20000):
    tot = 0
    for r in reps:
        tot += n_additive(rng.choice(np.arange(1, r["p"]//2+1), len(r["comm"]),
                                     replace=False).tolist(), r["p"])
    pm[t] = tot
print(f"one-committee-per-mask (n={len(reps)}): obs {obs_rep}, null "
      f"{pm.mean():.2f} +/- {pm.std():.2f}, p(depleted) = {(pm <= obs_rep).mean():.4f}")

# ---------------- Attack B: blind draws vs final committees (24 spectra runs)
blind_obs, blind_null, final_obs, final_null = 0, 0.0, 0, 0.0
pairs = []
for r in rows:
    if r["z"] is None:
        continue
    K = len(r["comm"])
    i3000 = int(np.argmin(np.abs(r["z"]["epochs"] - 3000)))
    blind = sorted((np.argsort(np.abs(r["z"]["coeffs"][i3000]))[::-1][:K] + 1).tolist())
    nb, nf_ = n_additive(blind, r["p"]), n_additive(r["comm"], r["p"])
    blind_obs += nb; final_obs += nf_
    blind_null += nmean(r["p"], K); final_null += nmean(r["p"], K)
    pairs.append((nb, nf_))
print(f"\nblind e3000 top-K draws (n={len(pairs)}): obs {blind_obs}, "
      f"null {blind_null:.2f}")
print(f"final committees (same runs):            obs {final_obs}, "
      f"null {final_null:.2f}")
print(f"paired counts (blind, final): {pairs}")

# significance of blind depletion by itself (perm null, original style)
bt = np.empty(20000)
zrows = [r for r in rows if r["z"] is not None]
for t in range(20000):
    tot = 0
    for r in zrows:
        tot += n_additive(rng.choice(np.arange(1, r["p"]//2+1), len(r["comm"]),
                                     replace=False).tolist(), r["p"])
    bt[t] = tot
print(f"blind-draw depletion p (iid null, n=24): {(bt <= blind_obs).mean():.4f}; "
      f"final-committee p on same 24: {(bt <= final_obs).mean():.4f}")

# ---------------- Attack C: popularity-matched null (per mask, needs >=3 runs)
print("\npopularity-matched null (masks with >=3 committees):")
tot_obs, tot_null_mean = 0, []
null_draws = np.zeros(20000)
used = 0
for m, rs in by_mask.items():
    if len(rs) < 3:
        continue
    p = m[0]; nf = p // 2
    popw = np.zeros(nf)
    for r in rs:
        for k in r["comm"]:
            popw[k-1] += 1
    popw = popw + 0.25  # smoothing so zero-pop freqs remain drawable
    popw /= popw.sum()
    for r in rs:
        used += 1
        K = len(r["comm"])
        tot_obs += n_additive(r["comm"], p)
        for t in range(20000):
            draw = rng.choice(np.arange(1, nf+1), K, replace=False, p=popw)
            null_draws[t] += n_additive(draw.tolist(), p)
print(f"  n={used} committees: obs {tot_obs}, popularity-matched null "
      f"{null_draws.mean():.2f} +/- {null_draws.std():.2f}, "
      f"p(depleted) = {(null_draws <= tot_obs).mean():.4f}")

# ---------------- Attack D: detector sensitivity
for label, fn in [
    ("gap in top-8 ", lambda c: committee_cut(c, 8)),
    ("gap in top-12", lambda c: committee_cut(c, 12)),
    ("gap in top-16", lambda c: committee_cut(c, 16)),
    ("fixed top-4  ", lambda c: sorted((np.argsort(np.abs(c))[::-1][:4] + 1).tolist())),
]:
    obs = 0; nl = 0.0
    for r in rows:
        comm = fn(r["coeffs"])
        obs += n_additive(comm, r["p"])
        nl += nmean(r["p"], len(comm))
    print(f"detector {label}: obs {obs}, null mean {nl:.2f}")
