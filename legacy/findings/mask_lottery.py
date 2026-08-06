"""Test (a): does the mask's diagonal spectrum |m_k| predict committee identity?

m_k = sum_s n_M(s) exp(2*pi*i*k*s/p), where n_M(s) = #train pairs on the
sum-diagonal a+b=s. Model-free, computable from the data split alone.

Tests:
  1. membership: mean |m_k| percentile of final-committee members, pooled
     across runs, vs a within-mask permutation null.
  2. popularity: Spearman(pop_k, |m_k|) per mask (masks with >= 3 runs).
  3. audition: same as (1) for the top-8 audition set at memorization end.
  4. negative control: all of the above with the (a-b)-direction spectrum.
"""

import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve()))
ROOT = Path("/Users/ruhaanrajadhyaksha/projects/grok")
sys.path.insert(0, str(ROOT))

from grok.config import Config
from grok.data import make_dataset, train_test_split

RUNS = ROOT / "runs"
RNG = np.random.default_rng(0)


# --------------------------------------------------------------------------- #
# run discovery
# --------------------------------------------------------------------------- #

def discover():
    """[(run_dir, cfg)] for every run dir with a config + checkpoints."""
    dirs = []
    for pattern in ["og_seed0/*", "seed0/*", "seed1/*", "seed2/*",
                    "p-*/seed*/seed*"]:
        dirs += sorted(RUNS.glob(pattern))
    out = []
    for d in dirs:
        if not (d / "config.json").exists():
            continue
        if not list((d / "checkpoints").glob("epoch_*.safetensors")):
            continue
        out.append((d, Config.load(d / "config.json")))
    return out


# --------------------------------------------------------------------------- #
# committee + audition sets
# --------------------------------------------------------------------------- #

def committee_from_coeffs(coeffs):
    a = np.abs(coeffs)
    order = np.argsort(a)[::-1]
    logs = np.log(a[order] + 1e-12)
    gaps = logs[:-1] - logs[1:]
    cut = int(np.argmax(gaps[:12])) + 1
    return sorted((order[:cut] + 1).tolist())


def final_coeffs_and_acc(d, cfg):
    """(final coeffs over freqs, final test acc, spectra or None)."""
    sp = d / "spectra.npz"
    if sp.exists():
        z = np.load(sp)
        return z["coeffs"][-1], float(z["test_acc"][-1]), z
    # fall back to the final checkpoint
    import mlx.core as mx  # noqa: F401
    from grok.model import Transformer
    from grok.metrics import all_logits, freq_coeffs_and_energy, accuracy
    from grok.fourier import Fourier

    model = Transformer(cfg)
    ckpt = sorted((d / "checkpoints").glob("epoch_*.safetensors"))[-1]
    model.load_weights(str(ckpt))
    tokens, labels = make_dataset(cfg)
    logits = all_logits(model, tokens)
    labels = np.array(labels)
    _, is_test = train_test_split(cfg)
    acc = accuracy(logits, labels, is_test)
    coeffs, _ = freq_coeffs_and_energy(logits, Fourier(cfg.p))
    return coeffs, float(acc), None


def audition_set(z, n_top=8):
    """Top-n freqs by |coeff| at the first epoch with train_acc >= 0.995."""
    idx = np.argmax(z["train_acc"] >= 0.995)
    if z["train_acc"][idx] < 0.995:
        return None, None
    c = np.abs(z["coeffs"][idx])
    return sorted((np.argsort(c)[::-1][:n_top] + 1).tolist()), int(z["epochs"][idx])


# --------------------------------------------------------------------------- #
# mask spectra
# --------------------------------------------------------------------------- #

def mask_spectra(cfg):
    """(|m_k| sum-diagonal spectrum, |md_k| difference-diagonal control),
    k = 1..p//2."""
    p = cfg.p
    is_train, _ = train_test_split(cfg)
    a, b = np.divmod(np.arange(p * p), p)
    n_sum = np.bincount((a[is_train] + b[is_train]) % p, minlength=p).astype(float)
    n_dif = np.bincount((a[is_train] - b[is_train]) % p, minlength=p).astype(float)
    ks = np.arange(1, p // 2 + 1)
    ph = np.exp(2j * np.pi * ks[:, None] * np.arange(p)[None, :] / p)
    return np.abs(ph @ n_sum), np.abs(ph @ n_dif), (ph @ n_sum)


def pctile(values, members):
    """Mean percentile (0-100) of `members` (1-indexed freqs) within `values`."""
    ranks = np.argsort(np.argsort(values))  # 0..n-1
    n = len(values)
    return float(np.mean([100.0 * ranks[k - 1] / (n - 1) for k in members]))


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #

def main():
    runs = discover()
    masks = {}      # (p, data_seed) -> dict(m=, md=, mc=)
    rows = []       # per grokked run
    for d, cfg in runs:
        key = (cfg.p, cfg.data_seed)
        if key not in masks:
            m, md, mc = mask_spectra(cfg)
            masks[key] = dict(m=m, md=md, mc=mc)
        coeffs, acc, z = final_coeffs_and_acc(d, cfg)
        if acc < 0.99:
            print(f"skip (acc {acc:.3f}): {d.relative_to(RUNS)}")
            continue
        comm = committee_from_coeffs(coeffs)
        aud, aud_epoch = audition_set(z) if z is not None else (None, None)
        rows.append(dict(dir=d, key=key, comm=comm, aud=aud, aud_epoch=aud_epoch))

    # ------------------------------------------------------------------ table
    print(f"\n{len(rows)} grokked runs, {len(masks)} masks\n")
    hdr = f"{'run':<38}{'p':<5}{'dseed':<7}{'committee':<22}{'m_pct':>6}{'ctl_pct':>8}"
    print(hdr); print("-" * len(hdr))
    for r in rows:
        m = masks[r["key"]]
        r["m_pct"] = pctile(m["m"], r["comm"])
        r["c_pct"] = pctile(m["md"], r["comm"])
        cstr = "{" + ",".join(map(str, r["comm"])) + "}"
        print(f"{str(r['dir'].relative_to(RUNS)):<38}{r['key'][0]:<5}"
              f"{r['key'][1]:<7}{cstr:<22}{r['m_pct']:>6.1f}{r['c_pct']:>8.1f}")

    # -------------------------------------------------- 1. membership pooled
    def pooled(field, spec_key):
        return np.mean([pctile(masks[r["key"]][spec_key], r[field])
                        for r in rows if r[field]])

    def perm_test(field, spec_key, n_iter=20000):
        obs = pooled(field, spec_key)
        active = [r for r in rows if r[field]]
        null = np.empty(n_iter)
        for i in range(n_iter):
            perms = {k: RNG.permutation(v[spec_key]) for k, v in masks.items()}
            null[i] = np.mean([pctile(perms[r["key"]], r[field]) for r in active])
        p = (null >= obs).mean()
        return obs, null.mean(), null.std(), p, len(active)

    print("\n=== 1. final-committee membership vs |m_k| (sum-diagonal) ===")
    obs, mu, sd, p, n = perm_test("comm", "m")
    print(f"mean member percentile {obs:.1f} (null {mu:.1f} +/- {sd:.1f}), "
          f"one-sided p = {p:.4f}, n_runs = {n}")

    print("\n=== 1b. negative control: (a-b)-direction spectrum ===")
    obs, mu, sd, p, n = perm_test("comm", "md")
    print(f"mean member percentile {obs:.1f} (null {mu:.1f} +/- {sd:.1f}), "
          f"one-sided p = {p:.4f}")

    # -------------------------------------------------- 2. popularity per mask
    print("\n=== 2. per-mask popularity Spearman(pop_k, |m_k|) ===")
    for key, m in sorted(masks.items()):
        rs = [r for r in rows if r["key"] == key]
        if len(rs) < 3:
            continue
        nfreq = len(m["m"])
        pop = np.zeros(nfreq)
        for r in rs:
            for k in r["comm"]:
                pop[k - 1] += 1
        rho, pv = spearmanr(pop, m["m"])
        rho_c, pv_c = spearmanr(pop, m["md"])
        top_m = (np.argsort(m["m"])[::-1][:8] + 1).tolist()
        print(f"p={key[0]} dseed={key[1]} (n={len(rs)}): rho={rho:+.3f} "
              f"(p={pv:.3f})   ctl rho={rho_c:+.3f} (p={pv_c:.3f})")
        print(f"    top-8 |m_k| freqs: {top_m}")
        print(f"    committees: {[r['comm'] for r in rs]}")

    # -------------------------------------------------- 3. audition sets
    print("\n=== 3. audition top-8 (at memorization end) vs |m_k| ===")
    obs, mu, sd, p, n = perm_test("aud", "m")
    print(f"mean audition percentile {obs:.1f} (null {mu:.1f} +/- {sd:.1f}), "
          f"one-sided p = {p:.4f}, n_runs = {n}")
    epochs = [r["aud_epoch"] for r in rows if r["aud"]]
    if epochs:
        print(f"memorization epochs used: min {min(epochs)}, max {max(epochs)}")


if __name__ == "__main__":
    main()
