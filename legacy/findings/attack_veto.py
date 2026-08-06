"""Adversarial re-analysis of T23 (veto/repair test).

Attack 1: decomposition. For loyal runs, blind == final by definition, so
their blind percentile is the *survivor committee's* percentile — which the
floor claim already says is elevated. The only new content in "low margin
predicts reconfiguration" is whether RECONFIGURERS' blind draws are below
the random-draw median (50). Test that directly.

Attack 2: K-circularity — recompute with K-free blind sets (K in 3..6).

Attack 3: cluster structure — are reconfigurers concentrated in particular
masks? MW test assumes 24 independent runs.

Attack 4: epoch sensitivity — reclassify with blind draws at e2000/e4000;
also check grok epochs vs the e3000 snapshot (post-grok blind draws are
trivially loyal).
"""
import sys
from pathlib import Path
import numpy as np
from scipy.stats import mannwhitneyu, wilcoxon, ttest_1samp

ROOT = Path("/Users/ruhaanrajadhyaksha/projects/grok")
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "findings"))
from mask_lottery import discover, final_coeffs_and_acc, committee_from_coeffs
from margin_analysis import relM_equal

rng = np.random.default_rng(0)

runs = []
for d, cfg in discover():
    coeffs, acc, z = final_coeffs_and_acc(d, cfg)
    if acc < 0.99 or z is None:
        continue
    p = cfg.p
    final = committee_from_coeffs(coeffs)
    runs.append(dict(name=str(d.relative_to(ROOT / 'runs')), p=p,
                     dseed=cfg.data_seed, final=final, z=z))

# precompute nulls per (p, K)
null_cache = {}
def pctile(freqs, p):
    K = len(freqs)
    key = (p, K)
    if key not in null_cache:
        nf = p // 2
        null_cache[key] = np.array([
            relM_equal(rng.choice(np.arange(1, nf + 1), K, replace=False).tolist(), p)
            for _ in range(3000)])
    return 100.0 * (null_cache[key] < relM_equal(freqs, p)).mean()

def blind_at(z, epoch, K):
    i = int(np.argmin(np.abs(z["epochs"] - epoch)))
    return sorted((np.argsort(np.abs(z["coeffs"][i]))[::-1][:K] + 1).tolist())

def grok_epoch(z):
    i = np.argmax(z["test_acc"] >= 0.99)
    return int(z["epochs"][i]) if z["test_acc"][i] >= 0.99 else None

print("=== per-run detail ===")
rc_p, ok_p = [], []
for r in runs:
    K = len(r["final"])
    blind = blind_at(r["z"], 3000, K)
    reconf = set(blind) != set(r["final"])
    pct_blind = pctile(blind, r["p"])
    pct_final = pctile(r["final"], r["p"])
    r.update(blind=blind, reconf=reconf, pct_blind=pct_blind, pct_final=pct_final,
             grok=grok_epoch(r["z"]))
    (rc_p if reconf else ok_p).append(pct_blind)
    print(f"  {'RECONF' if reconf else 'loyal '}  blind-pct {pct_blind:5.1f}  "
          f"final-pct {pct_final:5.1f}  grok {r['grok']}  p={r['p']} ds={r['dseed']}  {r['name']}")

rc_p, ok_p = np.array(rc_p), np.array(ok_p)
print(f"\n=== Attack 1: decomposition ===")
print(f"reconf (n={len(rc_p)}): mean blind pct {rc_p.mean():.1f}")
print(f"loyal  (n={len(ok_p)}): mean blind pct {ok_p.mean():.1f} "
      f"(= their FINAL committee pct, i.e. the floor)")
u = mannwhitneyu(rc_p, ok_p, alternative="less")
print(f"original MW one-sided p = {u.pvalue:.4f}")
# the actually-new content: are reconf blind draws below the random median?
t = ttest_1samp(rc_p, 50)
w = wilcoxon(rc_p - 50, alternative="less")
# exact uniform-null: mean of n iid U(0,100)
nullm = np.array([rng.uniform(0, 100, len(rc_p)).mean() for _ in range(100000)])
p_unif = (nullm <= rc_p.mean()).mean()
print(f"reconf blind pct vs 50: t-test p(two-sided) = {t.pvalue:.3f}; "
      f"wilcoxon one-sided(less) p = {w.pvalue:.3f}; "
      f"uniform-null one-sided p = {p_unif:.3f}")
# and the loyal side alone (the floor restated):
t2 = ttest_1samp(ok_p, 50)
nullm2 = np.array([rng.uniform(0, 100, len(ok_p)).mean() for _ in range(100000)])
print(f"loyal  blind(=final) pct vs 50: t p = {t2.pvalue:.4f}, "
      f"uniform-null one-sided p = {(nullm2 >= ok_p.mean()).mean():.4f}")

print(f"\n=== Attack 2: K-free blind sets ===")
# classify reconf using K-free criterion: is final == top-K(e3000) for K=len(final)?
# alternative blind margin: percentile of the BEST top-K draw over K in 3..6
rc2, ok2 = [], []
for r in runs:
    pcts = [pctile(blind_at(r["z"], 3000, K), r["p"]) for K in (3, 4, 5, 6)]
    best = max(pcts)
    (rc2 if r["reconf"] else ok2).append(best)
u2 = mannwhitneyu(rc2, ok2, alternative="less")
print(f"best-over-K blind pct: reconf {np.mean(rc2):.1f} vs loyal {np.mean(ok2):.1f}, "
      f"MW one-sided p = {u2.pvalue:.4f}")

print(f"\n=== Attack 3: mask clustering of reconfigurer status ===")
from collections import Counter
by_mask = {}
for r in runs:
    by_mask.setdefault((r["p"], r["dseed"]), []).append(r["reconf"])
for k, v in sorted(by_mask.items()):
    print(f"  p={k[0]} dseed={k[1]}: {sum(v)}/{len(v)} reconfigured")

print(f"\n=== Attack 4: blind-epoch sensitivity ===")
for ep in (2000, 3000, 4000):
    flips = 0
    rc_e, ok_e = [], []
    for r in runs:
        K = len(r["final"])
        b = blind_at(r["z"], ep, K)
        rec = set(b) != set(r["final"])
        if rec != r["reconf"]:
            flips += 1
        (rc_e if rec else ok_e).append(pctile(b, r["p"]))
    u_e = mannwhitneyu(rc_e, ok_e, alternative="less")
    print(f"  e{ep}: reconf n={len(rc_e)} mean {np.mean(rc_e):.1f} vs "
          f"loyal {np.mean(ok_e):.1f}, MW p = {u_e.pvalue:.4f}, "
          f"classification flips vs e3000: {flips}")
print("\ngrok epochs:", sorted(r["grok"] for r in runs if r["grok"]))
