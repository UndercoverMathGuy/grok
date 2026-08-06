"""Harmonic-channel mask test — the channel the symmetry argument forces.

Linear-response mask couplings depend only on j-k / j+k (translation-invariant
in frequency space -> cannot select individual frequencies; consistent with
all previous nulls). The only k-specific channels enter via the nonlinearity's
harmonics: circuit k's self-energy touches the mask at lattice points 2k and
3k. Score:

  H2_k = |M^(2k,0)|^2 + |M^(0,2k)|^2 + |M^(2k,2k)|^2 + |M^(2k,-2k)|^2
  H3_k = same at 3k;   Htot = H2 + H3   (all frequencies folded mod p)

Primary: H2 membership (two-sided). Secondary: popularity, audition, H3, Htot.
"""

import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path("/Users/ruhaanrajadhyaksha/projects/grok")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from grok.data import train_test_split
from mask_lottery import (discover, final_coeffs_and_acc, committee_from_coeffs,
                          audition_set, pctile)

RNG = np.random.default_rng(2)


def mask_2d_power(cfg):
    """|M^(u,v)|^2 for the full 2D mask spectrum, as a (p, p) array."""
    p = cfg.p
    is_train, _ = train_test_split(cfg)
    m = is_train.reshape(p, p).astype(float)
    return np.abs(np.fft.fft2(m)) ** 2


def harmonic_scores(cfg):
    p = cfg.p
    P = mask_2d_power(cfg)
    ks = np.arange(1, p // 2 + 1)

    def channel(mult):
        h = (mult * ks) % p           # fft indices handle folding automatically
        return (P[h, 0] + P[0, h] + P[h, h] + P[h, (p - h) % p])

    H2, H3 = channel(2), channel(3)
    return H2, H3, H2 + H3


def main():
    runs = discover()
    masks, rows = {}, []
    for d, cfg in runs:
        key = (cfg.p, cfg.data_seed)
        if key not in masks:
            masks[key] = harmonic_scores(cfg)
        coeffs, acc, z = final_coeffs_and_acc(d, cfg)
        if acc < 0.99:
            continue
        aud, _ = audition_set(z) if z is not None else (None, None)
        rows.append(dict(key=key, comm=committee_from_coeffs(coeffs), aud=aud))

    def perm_test(field, idx, n_iter=20000):
        active = [r for r in rows if r[field]]
        obs = np.mean([pctile(masks[r["key"]][idx], r[field]) for r in active])
        null = np.empty(n_iter)
        for i in range(n_iter):
            perms = {k: RNG.permutation(v[idx]) for k, v in masks.items()}
            null[i] = np.mean([pctile(perms[r["key"]], r[field]) for r in active])
        p_hi = (null >= obs).mean()
        p_two = min(1.0, 2 * min(p_hi, 1 - p_hi))
        return obs, null.mean(), null.std(), p_two

    print(f"{len(rows)} runs, {len(masks)} masks\n")
    for name, idx in [("H2 (2k channel)", 0), ("H3 (3k channel)", 1),
                      ("Htot", 2)]:
        o, mu, sd, p2 = perm_test("comm", idx)
        print(f"membership vs {name:<16}: {o:5.1f} (null {mu:.1f}+/-{sd:.1f}), "
              f"two-sided p = {p2:.4f}")
    print()
    for name, idx in [("H2", 0), ("Htot", 2)]:
        o, mu, sd, p2 = perm_test("aud", idx)
        print(f"audition top-8 vs {name:<5}: {o:5.1f} (null {mu:.1f}+/-{sd:.1f}), "
              f"two-sided p = {p2:.4f}")

    print("\npopularity Spearman (masks with >= 3 runs):")
    for key, sc in sorted(masks.items()):
        rs = [r for r in rows if r["key"] == key]
        if len(rs) < 3:
            continue
        pop = np.zeros(len(sc[0]))
        for r in rs:
            for k in r["comm"]:
                pop[k - 1] += 1
        rho2, pv2 = spearmanr(pop, sc[0])
        rhot, pvt = spearmanr(pop, sc[2])
        print(f"  p={key[0]} dseed={key[1]} (n={len(rs)}): "
              f"H2 rho={rho2:+.3f} (p={pv2:.3f})  Htot rho={rhot:+.3f} (p={pvt:.3f})")


if __name__ == "__main__":
    main()
