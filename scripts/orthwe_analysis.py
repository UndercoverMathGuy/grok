"""Did flattening the W_E init lottery (semi-orthogonal embed) change committees?

Compares runs/orthWE/p-113/** (embed_init='orthogonal') against the baseline
matrix cells runs/p-113/** (embed_init='normal'), using the same machinery as
scripts/margin_analysis.py:

  per run     committee, relM_equal, lp_relM, percentile vs random K-subsets
  paired      committee overlap (Jaccard) for (data_seed, init_seed) pairs that
              exist in both farms — same Gaussian draw, tilt surgically removed
  early lock  Jaccard(top-K logit freqs at epoch ~500, final committee): does
              the early-leader lock-in survive a flat start?
  pooled      orth vs baseline percentile distributions (Mann-Whitney)

Run: uv run python scripts/orthwe_analysis.py
"""

from pathlib import Path

import numpy as np
from scipy.stats import mannwhitneyu

import sys
sys.path.insert(0, str(Path(__file__).parent))
from margin_analysis import committee, relM_equal, lp_relM, null_percentile

RUNS = Path("runs")
P = 113


def discover(root):
    out = []
    for ddir in sorted(root.glob("seed*")):
        for rdir in sorted(ddir.glob("seed*")):
            if (rdir / "spectra.npz").exists() and (rdir / "metrics.json").exists():
                out.append((int(ddir.name[4:]), int(rdir.name[4:]), rdir))
    return out


def jaccard(a, b):
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if a | b else 1.0


def analyze_runs(runs, label, rng):
    rows = []
    print(f"\n=== {label} ===")
    header = (f"dseed  init    committee                      K  relM_eq  "
              f"lp_relM  pctile  acc     early-lock")
    print(header)
    print("-" * len(header))
    for dseed, iseed, d in runs:
        z = np.load(d / "spectra.npz")
        acc = float(z["test_acc"][-1])
        comm, _ = committee(z["coeffs"][-1])
        K = len(comm)
        rM = relM_equal(comm, P)
        lpM, _ = lp_relM(comm, P)
        pct, _, _ = null_percentile(comm, P, lp_relM, n=3000, rng=rng)
        # early lock-in: top-K |coeff| at the snapshot nearest epoch 500
        e500 = int(np.argmin(np.abs(z["epochs"] - 500)))
        early = np.sort(np.argsort(np.abs(z["coeffs"][e500]))[::-1][:K] + 1).tolist()
        lock = jaccard(early, comm)
        rows.append(dict(dseed=dseed, iseed=iseed, comm=comm, K=K, relM=rM,
                         lp=lpM, pct=pct, acc=acc, lock=lock))
        cstr = "{" + ",".join(map(str, comm)) + "}"
        print(f"{dseed:<7}{iseed:<8}{cstr:<30} {K}  {rM:6.3f}   {lpM:6.3f}   "
              f"{pct:5.1f}  {acc:6.3f}   {lock:.2f}")
    return rows


def main():
    rng = np.random.default_rng(0)
    orth = discover(RUNS / "orthWE" / "p-113")
    base = discover(RUNS / "p-113")
    orows = analyze_runs(orth, f"orthogonal W_E ({len(orth)} runs)", rng)
    brows = analyze_runs(base, f"baseline normal W_E ({len(base)} runs)", rng)

    print("\n=== paired (same data_seed + init_seed, tilt removed) ===")
    bmap = {(r["dseed"], r["iseed"]): r for r in brows}
    for r in orows:
        b = bmap.get((r["dseed"], r["iseed"]))
        if b is None:
            continue
        j = jaccard(r["comm"], b["comm"])
        print(f"  d{r['dseed']} i{r['iseed']}: orth {set(r['comm'])} vs "
              f"base {set(b['comm'])}  Jaccard {j:.2f}  "
              f"lp_relM {r['lp']:.3f} vs {b['lp']:.3f}")

    op = np.array([r["pct"] for r in orows])
    bp = np.array([r["pct"] for r in brows])
    print("\n=== pooled ===")
    print(f"orth     lp-relM percentile: mean {op.mean():.1f}  (n={len(op)})  "
          f"early-lock mean {np.mean([r['lock'] for r in orows]):.2f}")
    print(f"baseline lp-relM percentile: mean {bp.mean():.1f}  (n={len(bp)})  "
          f"early-lock mean {np.mean([r['lock'] for r in brows]):.2f}")
    if len(op) and len(bp):
        u = mannwhitneyu(op, bp, alternative="greater")
        print(f"Mann-Whitney orth > baseline: p={u.pvalue:.3f}")
    # is orth better than a coin flip vs the random-subset null?
    z = (op.mean() - 50) / np.sqrt(833.33 / len(op))
    from scipy.stats import norm
    print(f"orth vs uniform-null mean 50: z={z:.2f}, one-sided p={norm.sf(z):.2e}")


if __name__ == "__main__":
    main()
