"""Extended mask-proxy dictionary, after |m_k| (sum-diagonal) came back null.

  4. row/column marginal spectra: |M-hat(k,0)|^2 + |M-hat(0,k)|^2 — uneven
     sampling of a-values / b-values at frequency k.
  5. pair-coupling test: two committee members j,k interfere on the train set
     through the mask's diagonal spectrum at fold(j-k) and fold(j+k).
     Statistic: mean over committee pairs of g_jk = |m_fold(j-k)|^2 +
     |m_fold(j+k)|^2, vs random size-matched committees. Two-sided.
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path("/Users/ruhaanrajadhyaksha/projects/grok")
sys.path.insert(0, str(ROOT))

from grok.config import Config
from grok.data import train_test_split

sys.path.insert(0, str(Path(__file__).parent))
from mask_lottery import discover, final_coeffs_and_acc, committee_from_coeffs, pctile

RNG = np.random.default_rng(1)


def spectra(cfg):
    p = cfg.p
    is_train, _ = train_test_split(cfg)
    a, b = np.divmod(np.arange(p * p), p)
    n_sum = np.bincount((a[is_train] + b[is_train]) % p, minlength=p).astype(float)
    n_a = np.bincount(a[is_train], minlength=p).astype(float)
    n_b = np.bincount(b[is_train], minlength=p).astype(float)
    ks = np.arange(1, p // 2 + 1)
    ph = np.exp(2j * np.pi * ks[:, None] * np.arange(p)[None, :] / p)
    m_sum = np.abs(ph @ n_sum)
    m_marg = np.abs(ph @ n_a) ** 2 + np.abs(ph @ n_b) ** 2
    return m_sum, m_marg


def fold(x, p):
    x = x % p
    return min(x, p - x)


def pair_stat(comm, m_sum, p):
    """Mean |m|^2 at the sum/difference channels over all member pairs."""
    vals = []
    for i in range(len(comm)):
        for j in range(i + 1, len(comm)):
            fd = fold(comm[j] - comm[i], p)
            fs = fold(comm[j] + comm[i], p)
            vals.append(m_sum[fd - 1] ** 2 + m_sum[fs - 1] ** 2)
    return float(np.mean(vals))


def main():
    runs = discover()
    masks, rows = {}, []
    for d, cfg in runs:
        key = (cfg.p, cfg.data_seed)
        if key not in masks:
            masks[key] = spectra(cfg)
        coeffs, acc, _ = final_coeffs_and_acc(d, cfg)
        if acc < 0.99:
            continue
        rows.append(dict(key=key, comm=committee_from_coeffs(coeffs),
                         name=str(d.relative_to(ROOT / "runs"))))

    # ---- 4. marginal spectra membership test (same permutation machinery)
    def perm_test(spec_idx, n_iter=20000):
        obs = np.mean([pctile(masks[r["key"]][spec_idx], r["comm"]) for r in rows])
        null = np.empty(n_iter)
        for i in range(n_iter):
            perms = {k: RNG.permutation(v[spec_idx]) for k, v in masks.items()}
            null[i] = np.mean([pctile(perms[r["key"]], r["comm"]) for r in rows])
        return obs, null.mean(), null.std(), (null >= obs).mean()

    print(f"{len(rows)} runs, {len(masks)} masks")
    print("\n=== 4. membership vs marginal spectra |M(k,0)|^2+|M(0,k)|^2 ===")
    obs, mu, sd, p = perm_test(1)
    print(f"mean member percentile {obs:.1f} (null {mu:.1f} +/- {sd:.1f}), "
          f"one-sided p = {p:.4f}")

    # ---- 5. pair-coupling test: observed vs random committees, per run,
    #         pooled as mean log-ratio to the null mean
    print("\n=== 5. committee pair-coupling g_jk vs size-matched null ===")
    logratios, pcts = [], []
    for r in rows:
        p_, m_sum = r["key"][0], masks[r["key"]][0]
        obs = pair_stat(r["comm"], m_sum, p_)
        K = len(r["comm"])
        pool = np.arange(1, p_ // 2 + 1)
        null = np.array([pair_stat(sorted(RNG.choice(pool, K, replace=False)
                                          .tolist()), m_sum, p_)
                         for _ in range(2000)])
        logratios.append(np.log(obs / null.mean()))
        pcts.append(100.0 * (null < obs).mean())
    logratios, pcts = np.array(logratios), np.array(pcts)
    print(f"mean percentile of observed pair-coupling: {pcts.mean():.1f} "
          f"(sd {pcts.std():.1f}), n = {len(pcts)}")
    print(f"mean log(obs/null): {logratios.mean():+.3f} "
          f"(sem {logratios.std()/np.sqrt(len(logratios)):.3f})")
    z = (pcts.mean() - 50) / np.sqrt(833.33 / len(pcts))
    from scipy.stats import norm
    print(f"uniform-null z = {z:+.2f}, two-sided p = {2*norm.sf(abs(z)):.4f}")
    lo = (pcts < 25).sum()
    hi = (pcts > 75).sum()
    print(f"runs in bottom quartile: {lo}/{len(pcts)}, top quartile: {hi}/{len(pcts)}")


if __name__ == "__main__":
    main()
