"""Does the init readout beat a frequency-popularity prior that ignores the
init entirely?

The probe-ceiling run turned up an uncomfortable floor: scoring every run's
frequencies by "how often is k a committee member in the OTHER runs" reaches
AUC ~0.66 pooled -- indistinguishable from T_k. That prior uses no weights at
all, so any claim that T_k reads the init must be an *incremental* claim.

This script fits nested logistic models under leave-one-init-out CV:

  prior          the popularity prior alone
  tk             within-run z(log T_k) alone
  prior + tk     does T_k add anything on top of popularity?
  prior + ALL    does the full 123-feature epoch-0 bank add anything?

The prior for TRAINING rows is computed leave-one-run-out inside the training
set, so no run ever sees its own label. Analysis-only.
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

ROOT = Path("/Users/ruhaanrajadhyaksha/projects/grok")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "pivot"))
sys.path.insert(0, str(ROOT / "semifinal" / "analysis"))

from probe_features import build                        # noqa: E402
from probe_models import logreg_fit, logreg_pred        # noqa: E402
from probe_ceiling import auc1, per_run, summarize      # noqa: E402

CACHE = sys.argv[1] if len(sys.argv) > 1 else None


def main():
    d = build(cache=CACHE, verbose=False)
    X, y = d["X"], d["y"].astype(bool)
    names = [str(n) for n in d["names"]]
    runs = [str(r) for r in d["run"]]
    grp = np.array([str(g) for g in d["grp"]])
    ridx = d["runidx"].astype(int)
    uruns = np.unique(ridx)
    nf = int((ridx == uruns[0]).sum())
    runs_of = {int(i): runs[int(np.argmax(ridx == i))] for i in uruns}
    cohort_of = {runs[i]: str(d["cohort"][i]) for i in range(len(runs))}
    ugroups = sorted(set(grp.tolist()))

    # per-run label matrix (n_runs, nf)
    Y = np.stack([y[ridx == i] for i in uruns]).astype(float)
    tkz = np.zeros(len(y))
    tkr = np.zeros(len(y))
    j_tk = names.index("V_tk__z")
    j_tkr = names.index("V_tk__r")
    tkz, tkr = X[:, j_tk], X[:, j_tkr]

    print("global committee-membership popularity by frequency "
          "(96 runs, 385 memberships):")
    pk_all = Y.mean(0)
    order = np.argsort(-pk_all)
    print("  most popular:", ", ".join(f"k={o + 1}:{pk_all[o]:.3f}"
                                       for o in order[:10]))
    print("  never a member:", int((pk_all == 0).sum()), "of", nf,
          " max/mean ratio %.2f" % (pk_all.max() / pk_all.mean()))

    MODELS = {"prior": ["prior"], "tk": ["tk"], "prior+tk": ["prior", "tk"],
              "prior+ALL": ["prior", "ALL"], "ALL": ["ALL"]}
    scores = {k: np.full(len(y), np.nan) for k in MODELS}

    for u in ugroups:
        te_runs = [i for i in uruns if grp[ridx == i][0] == u]
        tr_runs = [i for i in uruns if i not in te_runs]
        Ytr = Y[tr_runs]
        pk_te = Ytr.mean(0)                                   # test-fold prior
        # leave-one-run-out prior inside the training set
        pk_tr = (Ytr.sum(0)[None, :] - Ytr) / (len(tr_runs) - 1)

        prior = np.zeros(len(y))
        for j, i in enumerate(tr_runs):
            prior[ridx == i] = pk_tr[j]
        for i in te_runs:
            prior[ridx == i] = pk_te
        prior = np.log(prior + 0.01)                          # log-odds-ish

        te = np.isin(ridx, te_runs)
        for tag, parts in MODELS.items():
            cols = []
            if "prior" in parts:
                cols.append(prior[:, None])
            if "tk" in parts:
                cols.append(np.stack([tkz, tkr], 1))
            if "ALL" in parts:
                cols.append(X)
            Z = np.concatenate(cols, 1)
            mu, sd = Z[~te].mean(0), Z[~te].std(0)
            sd = np.where(sd < 1e-9, 1.0, sd)
            Z = (Z - mu) / sd
            # inner leave-one-group-out ridge selection
            best, bl = -np.inf, 1.0
            for lam in ([1, 10, 100] if Z.shape[1] > 3 else [0.1, 1, 10]):
                a = []
                for hg in [g for g in ugroups if g != u]:
                    m = (~te) & (grp != hg)
                    v = (~te) & (grp == hg)
                    w = logreg_fit(Z[m], y[m], lam=lam)
                    s = logreg_pred(w, Z[v])
                    a += [auc1(s[ridx[v] == i], y[v][ridx[v] == i])
                          for i in np.unique(ridx[v])]
                if np.mean(a) > best:
                    best, bl = float(np.mean(a)), lam
            w = logreg_fit(Z[~te], y[~te], lam=bl)
            scores[tag][te] = logreg_pred(w, Z[te])

    out = {}
    prs = {}
    for tag in MODELS:
        pr = per_run(scores[tag], y, ridx, runs_of)
        prs[tag] = pr
        out[tag] = summarize(pr, cohort_of)
    print("\n=== leave-one-init-out, nested models ===")
    hdr = f"{'model':<12}" + "".join(f"{c:>16}" for c in
                                     ("all", "natural-normal", "surgical",
                                      "orth-flat", "double-flat"))
    print(hdr)
    for tag in ("prior", "tk", "prior+tk", "ALL", "prior+ALL"):
        row = f"{tag:<12}"
        for c in ("all", "natural-normal", "surgical", "orth-flat",
                  "double-flat"):
            v = out[tag].get(c)
            row += f"{v['auc']:>10.4f}±{v['auc_sd']:.2f}" if v else " " * 16
        print(row)
    print("\njaccard / exact-set (top-K, K = true K):")
    for tag in ("prior", "tk", "prior+tk", "ALL", "prior+ALL"):
        a = out[tag]["all"]
        n = out[tag]["natural-normal"]
        print(f"  {tag:<12} all jac {a['jac']:.3f} exact {a['exact']:.3f}   "
              f"natural jac {n['jac']:.3f} exact {n['exact']:.3f}")

    # paired incremental tests
    print("\nincremental (paired over runs, Wilcoxon):")
    tests = [("prior+tk", "prior"), ("prior+ALL", "prior"),
             ("prior+ALL", "prior+tk"), ("tk", "prior")]
    inc = {}
    for a, b in tests:
        for scope in ("all", "natural-normal"):
            keys = [k for k in prs[a] if scope == "all"
                    or cohort_of[k] == scope]
            va = np.array([prs[a][k]["auc"] for k in keys])
            vb = np.array([prs[b][k]["auc"] for k in keys])
            try:
                pv = float(wilcoxon(va, vb).pvalue) if np.any(va != vb) else 1.0
            except ValueError:
                pv = 1.0
            inc[f"{a} - {b} [{scope}]"] = dict(
                delta=float((va - vb).mean()), p=pv, n=len(keys))
            print(f"  {a:<10} - {b:<9} [{scope:<14}] "
                  f"delta {(va - vb).mean():+.4f}  p={pv:.3g}  n={len(keys)}")

    p = ROOT / "notes" / "pivot" / "probe_prior_control.json"
    p.write_text(json.dumps(dict(models=out, increments=inc,
                                 popularity=pk_all.tolist()),
                            indent=1, sort_keys=True))
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
