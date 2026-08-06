"""C1 core prediction: a pre-specified, margin-free, purely ARITHMETIC repair
algorithm applied to the blind e3000 amplitude draw should predict final
committees better than the blind draw itself (bar: blind Jaccard ~0.73).

Algorithm (fixed before looking at outcomes):
  S <- blind top-K by |coeff| at e3000; menu <- top-12 (recruit pool)
  repeat (max 10):
    V <- additive violations in S: unordered pairs (i,j) with fold(i+j) or
         fold(i-j) also in S  [note: 2k = k+k relations count via i=j pairs?
         NO — matching the census definition, only i<j pairs count]
    if none: stop
    evict the member with the lowest e3000 amplitude among all members
      involved in any violation
    recruit the highest-amplitude menu freq not in S whose addition creates
      no new violation (if all create violations, recruit the best anyway)
  predicted <- S

Scoring vs final committee: Jaccard, exact match; compared against blind
(paired). Also: in reconfigured runs where blind had violations, does the
algorithm's evicted freq match a freq that actually left, and does its
recruit match one that actually entered?

Honesty checks:
  - the algorithm does nothing on violation-free blind draws, so the 4
    clean-blind reconfigurers are guaranteed misses — reported;
  - a null variant that evicts the lowest-amplitude member REGARDLESS of
    violations and recruits next menu freq (amplitude churn, no arithmetic)
    — C1 must beat this too;
  - a random-evict variant (evict a random violation participant), averaged
    over 100 seeds, to show the lowest-amplitude choice matters.
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

def violations(S, p):
    S = sorted(S)
    out = []
    ss = set(S)
    for ii in range(len(S)):
        for jj in range(ii + 1, len(S)):
            i, j = S[ii], S[jj]
            if fold(i + j, p) in ss or fold(i - j, p) in ss:
                out.append((i, j))
    return out

def creates_violation(S, x, p):
    ss = set(S) | {x}
    for m in S:
        if fold(m + x, p) in ss or fold(m - x, p) in ss:
            return True
    return False

def repair(blind, menu, amp, p, evict_mode="lowest", rng=None):
    S = list(blind)
    evicted, recruited = [], []
    for _ in range(10):
        V = violations(S, p)
        if not V:
            break
        involved = sorted(set(f for pair in V for f in pair))
        if evict_mode == "lowest":
            out = min(involved, key=lambda f: amp[f - 1])
        else:
            out = involved[rng.integers(len(involved))]
        S.remove(out); evicted.append(out)
        cands = [m for m in menu if m not in S and m not in evicted]
        ok = [m for m in cands if not creates_violation(S, m, p)]
        pick = (ok or cands)[0] if (ok or cands) else None
        if pick is not None:
            S.append(pick); recruited.append(pick)
    return sorted(S), evicted, recruited

def churn(blind, menu, amp, p):
    """null: same number of moves as repair() made, but arithmetic-blind —
    evict globally lowest-amplitude member, recruit next menu freq."""
    S, evicted, recruited = repair(blind, menu, amp, p)
    n_moves = len(evicted)
    S2 = list(blind)
    ev2 = []
    for _ in range(n_moves):
        out = min(S2, key=lambda f: amp[f - 1])
        S2.remove(out); ev2.append(out)
        cands = [m for m in menu if m not in S2 and m not in ev2]
        if cands:
            S2.append(cands[0])
    return sorted(S2)

def jac(a, b):
    a, b = set(a), set(b)
    return len(a & b) / len(a | b)

rng = np.random.default_rng(0)
res = []
for d, cfg in discover():
    coeffs, acc, z = final_coeffs_and_acc(d, cfg)
    if acc < 0.99 or z is None:
        continue
    p = cfg.p
    final = committee_from_coeffs(coeffs)
    K = len(final)
    i3000 = int(np.argmin(np.abs(z["epochs"] - 3000)))
    amp = np.abs(z["coeffs"][i3000])
    order = (np.argsort(amp)[::-1] + 1).tolist()
    menu = order[:12]
    blind = sorted(order[:K])
    pred, ev, rec = repair(blind, menu, amp, p)
    pred_churn = churn(blind, menu, amp, p)
    # random-evict ensemble
    jr = []
    for s in range(100):
        pr, _, _ = repair(blind, menu, amp, p, "random",
                          np.random.default_rng(s))
        jr.append(jac(pr, final))
    reconf = set(blind) != set(final)
    res.append(dict(
        name=str(d.relative_to(ROOT / "runs")), p=p, reconf=reconf,
        nviol=len(violations(blind, p)),
        j_blind=jac(blind, final), j_pred=jac(pred, final),
        j_churn=jac(pred_churn, final), j_rand=float(np.mean(jr)),
        exact_blind=set(blind) == set(final), exact_pred=set(pred) == set(final),
        ev=ev, rec=rec,
        true_out=sorted(set(blind) - set(final)),
        true_in=sorted(set(final) - set(blind))))
    r = res[-1]
    print(f"{r['name']:<34} viol={r['nviol']} "
          f"jB={r['j_blind']:.2f} jP={r['j_pred']:.2f} "
          f"ev={ev} rec={rec} trueOut={r['true_out']} trueIn={r['true_in']}")

import numpy as np
jb = np.array([r["j_blind"] for r in res])
jp = np.array([r["j_pred"] for r in res])
jc = np.array([r["j_churn"] for r in res])
jrand = np.array([r["j_rand"] for r in res])
print(f"\nn={len(res)} runs")
print(f"Jaccard vs final: blind {jb.mean():.3f} | repair {jp.mean():.3f} | "
      f"churn-null {jc.mean():.3f} | random-evict {jrand.mean():.3f}")
print(f"exact match: blind {sum(r['exact_blind'] for r in res)}/24, "
      f"repair {sum(r['exact_pred'] for r in res)}/24")
from scipy.stats import wilcoxon
moved = jp != jb
if moved.any():
    w = wilcoxon(jp - jb)
    print(f"paired repair-vs-blind: mean diff {np.mean(jp-jb):+.3f}, "
          f"wilcoxon p = {w.pvalue:.4f} (runs changed: {moved.sum()})")
# eviction identity accuracy among runs where algorithm acted AND run reconfigured
hits, tries = 0, 0
rhits, rtries = 0, 0
for r in res:
    if r["ev"] and r["reconf"]:
        tries += 1
        if set(r["ev"]) & set(r["true_out"]):
            hits += 1
    if r["rec"] and r["reconf"]:
        rtries += 1
        if set(r["rec"]) & set(r["true_in"]):
            rhits += 1
print(f"evictee identity: {hits}/{tries} runs where an algorithm-evicted freq "
      f"actually left; recruit identity: {rhits}/{rtries}")
clean_miss = sum(1 for r in res if r["reconf"] and r["nviol"] == 0)
print(f"guaranteed misses (reconfigured but violation-free blind): {clean_miss}")
