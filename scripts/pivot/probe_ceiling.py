"""PROBE CEILING — how much of the final Fourier committee is *learnably*
readable from the epoch-0 weights, versus the closed-form ticket T_k?

Baseline: T_k = sum_h ||W_O^h W_V^h W_E|_k||^2 (semifinal claim 1a), which
scores AUC 0.758 on the 8 natural-normal v2 runs.

This script trains supervised probes on a 123-feature epoch-0 bank (see
probe_features.py) with GROUPED cross-validation and compares them to T_k on
IDENTICAL folds.

CV groups: the independent init draw (data_seed, init_seed). All 96 kept v2
runs descend from only 8 such draws -- orth-flat variants and surgical arms
are deterministic edits of the same epoch-0 tensors -- so leave-one-run-out
would leak the init. Leave-one-init-out (8 folds) is primary; leave-one-run-out
is reported as an explicitly optimistic secondary.

Outputs: notes/pivot/probe_ceiling.json (+ probe_ceiling.md written by hand).
Analysis-only: reads runs_torch/*/checkpoints/epoch_00000.safetensors, trains
nothing, writes nothing outside notes/ and the scratch cache.
"""
import json
import sys
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, wilcoxon

ROOT = Path("/Users/ruhaanrajadhyaksha/projects/grok")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "pivot"))
sys.path.insert(0, str(ROOT / "semifinal" / "analysis"))

from probe_features import build                                # noqa: E402
from probe_models import (logreg_fit, logreg_pred, gbt_fit,      # noqa: E402
                          gbt_pred, mlp_fit, mlp_pred)

CACHE = Path(sys.argv[1]) if len(sys.argv) > 1 else None
OUT = ROOT / "notes" / "pivot" / "probe_ceiling.json"
RNG = np.random.default_rng(0)


# ----------------------------------------------------------------- metrics
def auc1(scores, lab):
    """Mann-Whitney AUC with midranks (identical to common.auc)."""
    n1, n0 = int(lab.sum()), int((~lab).sum())
    if n1 == 0 or n0 == 0:
        return np.nan
    r = rankdata(scores)
    return float((r[lab].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def per_run(scores, y, runidx, runs_of):
    """-> ordered dict run -> (auc, jaccard, exact, K)."""
    out = OrderedDict()
    for ri in np.unique(runidx):
        m = runidx == ri
        s, lab = scores[m], y[m]
        K = int(lab.sum())
        top = np.argsort(-s)[:K]
        pred = np.zeros(len(lab), bool)
        pred[top] = True
        inter = int((pred & lab).sum())
        out[runs_of[ri]] = dict(auc=auc1(s, lab),
                                jac=inter / (2 * K - inter) if K else 1.0,
                                exact=float(inter == K), K=K)
    return out


def summarize(pr, cohort_of):
    """Mean over runs, overall and per cohort."""
    def agg(rows):
        if not rows:
            return None
        return dict(n=len(rows),
                    auc=float(np.mean([r["auc"] for r in rows])),
                    auc_sd=float(np.std([r["auc"] for r in rows], ddof=1))
                    if len(rows) > 1 else 0.0,
                    jac=float(np.mean([r["jac"] for r in rows])),
                    exact=float(np.mean([r["exact"] for r in rows])))
    s = dict(all=agg(list(pr.values())))
    by = {}
    for run, r in pr.items():
        by.setdefault(cohort_of[run], []).append(r)
    for c, rows in by.items():
        s[c] = agg(rows)
    return s


def fold_stats(pr, fold_of):
    """Mean per-run AUC within each CV fold -> (mean, sd) across folds."""
    by = {}
    for run, r in pr.items():
        by.setdefault(fold_of[run], []).append(r["auc"])
    v = np.array([np.mean(x) for x in by.values()])
    return float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0


# ----------------------------------------------------------------- learners
def val_auc_score(yv, runidx_v):
    """Early-stopping objective: mean per-run AUC on the validation runs."""
    idx = [(runidx_v == ri) for ri in np.unique(runidx_v)]

    def f(scores):
        return float(np.mean([auc1(scores[m], yv[m]) for m in idx]))
    return f


def fit_predict(kind, Xtr, ytr, gtr, ridx_tr, Xte, seed=0):
    """Fit `kind` on the training fold (inner grouped split for tuning /
    early stopping) and return test scores."""
    mu, sd = Xtr.mean(0), Xtr.std(0)
    sd = np.where(sd < 1e-9, 1.0, sd)
    Ztr, Zte = (Xtr - mu) / sd, (Xte - mu) / sd
    ug = np.unique(gtr)

    if kind == "logreg":
        lams = [0.3, 1, 3, 10, 30, 100, 300, 1000]
        best, bl = -np.inf, lams[0]
        for lam in lams:
            aucs = []
            for hg in ug:                       # inner leave-one-group-out
                m = gtr != hg
                w = logreg_fit(Ztr[m], ytr[m], lam=lam)
                sv = logreg_pred(w, Ztr[~m])
                aucs += [auc1(sv[ridx_tr[~m] == ri], ytr[~m][ridx_tr[~m] == ri])
                         for ri in np.unique(ridx_tr[~m])]
            if np.mean(aucs) > best:
                best, bl = float(np.mean(aucs)), lam
        w = logreg_fit(Ztr, ytr, lam=bl)
        return logreg_pred(w, Zte), dict(lam=bl, inner_auc=best, w=w.tolist())

    # inner grouped split for early stopping: 2 held-out init groups
    hv = set(ug[RNG.permutation(len(ug))[:2]].tolist())
    m = np.array([g not in hv for g in gtr])
    sc = val_auc_score(ytr[~m], ridx_tr[~m])

    if kind == "gbt":
        _, nbest = gbt_fit(Ztr[m], ytr[m], Ztr[~m], ytr[~m], seed=seed,
                           score=sc)
        nbest = max(nbest, 20)
        model, _ = gbt_fit(Ztr, ytr, rounds=nbest, seed=seed)
        return gbt_pred(model, Zte), dict(rounds=int(nbest))

    if kind == "mlp":
        preds = []
        for s in range(3):                      # small ensemble, unstable fits
            mdl = mlp_fit(Ztr[m], ytr[m], Xval=Ztr[~m], yval=ytr[~m],
                          score=sc, seed=seed * 10 + s)
            preds.append(mlp_pred(mdl, Zte))
        return np.mean(preds, 0), dict(ens=3)
    raise ValueError(kind)


# --------------------------------------------------------------------- run
def main():
    t0 = time.time()
    d = build(cache=CACHE, verbose=False)
    X, y, tk = d["X"], d["y"].astype(bool), d["tk"]
    names = [str(n) for n in d["names"]]
    runs = [str(r) for r in d["run"]]
    grp = np.array([str(g) for g in d["grp"]])
    runidx = d["runidx"].astype(int)
    runs_of = {int(i): runs[int(np.argmax(runidx == i))]
               for i in np.unique(runidx)}
    cohort_of = {runs[i]: str(d["cohort"][i]) for i in range(len(runs))}
    grp_of = {runs[i]: str(d["grp"][i]) for i in range(len(runs))}
    ugroups = sorted(set(grp.tolist()))
    print(f"X {X.shape}  {y.sum()} positives  {len(runs_of)} runs  "
          f"{len(ugroups)} init groups", flush=True)

    # feature group membership (prefix before the first underscore)
    gof = np.array([n.split("_")[0] for n in names])
    SETS = OrderedDict([
        ("V(OV-only)", ["V"]),                   # T_k + its per-head siblings
        ("E", ["E"]),
        ("E+V", ["E", "V"]),
        ("E+V+U", ["E", "V", "U"]),
        ("E+V+U+M", ["E", "V", "U", "M"]),
        ("E+V+U+M+Q", ["E", "V", "U", "M", "Q"]),
        ("E+V+U+M+Q+P", ["E", "V", "U", "M", "Q", "P"]),
        ("ALL", ["E", "V", "U", "M", "Q", "P", "X", "K"]),
    ])
    ALLG = SETS["ALL"]
    for g in ALLG:                                # leave-one-group-out
        SETS[f"ALL-{g}"] = [x for x in ALLG if x != g]

    results = {"meta": dict(
        n_runs=len(runs_of), n_freq=int((runidx == 0).sum()),
        n_pos=int(y.sum()), n_feat=X.shape[1], n_groups=len(ugroups),
        groups=ugroups, n_by_cohort={c: sum(1 for r in runs_of.values()
                                            if cohort_of[r] == c)
                                     for c in set(cohort_of.values())})}

    # ---- T_k baseline, folds are irrelevant (no fitting) but reported the
    #      same way so the comparison is on identical test runs.
    pr_tk = per_run(tk, y, runidx, runs_of)
    results["tk_baseline"] = summarize(pr_tk, cohort_of)
    m, s = fold_stats(pr_tk, grp_of)
    results["tk_baseline"]["fold_mean"], results["tk_baseline"]["fold_sd"] = m, s
    print(f"T_k baseline: all {results['tk_baseline']['all']['auc']:.4f}  "
          f"natural {results['tk_baseline']['natural-normal']['auc']:.4f}",
          flush=True)

    # ---- floor: the run-invariant frequency popularity prior. Uses NO init
    #      weights at all -- just "how often is frequency k a member in the
    #      training runs". Any probe must clear this, not 0.5. (The permuted
    #      -label null below lands on the same value, which is why it is not
    #      0.5 either.)
    prior = np.full(len(y), np.nan)
    for u in ugroups:
        te = grp == u
        nfq = int((runidx == runidx[te][0]).sum())
        pk = np.zeros(nfq)
        for i in np.unique(runidx[~te]):
            pk += y[runidx == i]
        pk /= len(np.unique(runidx[~te]))
        for i in np.unique(runidx[te]):
            prior[runidx == i] = pk
    pr_pr = per_run(prior, y, runidx, runs_of)
    results["freq_prior_floor"] = summarize(pr_pr, cohort_of)
    m, s = fold_stats(pr_pr, grp_of)
    results["freq_prior_floor"]["fold_mean"] = m
    results["freq_prior_floor"]["fold_sd"] = s
    print(f"freq-prior floor (no init at all): all "
          f"{results['freq_prior_floor']['all']['auc']:.4f}  natural "
          f"{results['freq_prior_floor']['natural-normal']['auc']:.4f}",
          flush=True)

    # ---- primary: leave-one-init-group-out, all cohorts in training
    def run_cv(kind, keep, tag, folds="group", Xall=None, train_mask=None):
        Xs = X[:, np.isin(gof, keep)] if Xall is None else Xall
        scores = np.full(len(y), np.nan)
        extra = {}
        units = ugroups if folds == "group" else sorted(runs_of)
        for u in units:
            te = (grp == u) if folds == "group" else (runidx == u)
            tr = ~te
            if train_mask is not None:
                tr &= train_mask
            sc, info = fit_predict(kind, Xs[tr], y[tr], grp[tr], runidx[tr],
                                   Xs[te])
            scores[te] = sc
            extra[str(u)] = {k: v for k, v in info.items() if k != "w"}
        pr = per_run(scores, y, runidx, runs_of)
        out = summarize(pr, cohort_of)
        out["fold_mean"], out["fold_sd"] = fold_stats(pr, grp_of)
        out["n_feat"] = int(Xs.shape[1])
        out["tuning"] = extra
        out["_per_run"] = {r: v["auc"] for r, v in pr.items()}
        print(f"  {tag:<28} all {out['all']['auc']:.4f}  "
              f"nat {out.get('natural-normal', {}).get('auc', float('nan')):.4f}"
              f"  jac {out['all']['jac']:.3f} exact {out['all']['exact']:.3f}"
              f"  [{time.time() - t0:.0f}s]", flush=True)
        return out, pr

    print("\n=== primary: leave-one-init-out (8 folds), ALL features ===")
    main_res = {}
    for kind in ("logreg", "gbt", "mlp"):
        main_res[kind], _ = run_cv(kind, ALLG, kind)
    results["primary"] = main_res

    # paired significance vs T_k, per run, over the 96 runs and the 8 natural
    for kind, r in main_res.items():
        pr = r["_per_run"]
        for scope, keys in (("all", list(pr)),
                            ("natural-normal",
                             [k for k in pr if cohort_of[k] == "natural-normal"])):
            a = np.array([pr[k] for k in keys])
            b = np.array([pr_tk[k]["auc"] for k in keys])
            dd = a - b
            try:
                pv = float(wilcoxon(a, b).pvalue) if np.any(dd != 0) else 1.0
            except ValueError:
                pv = 1.0
            r.setdefault("vs_tk", {})[scope] = dict(
                delta=float(dd.mean()), n=len(keys), wilcoxon_p=pv,
                win=int((dd > 0).sum()), loss=int((dd < 0).sum()))

    print("\n=== ablations (leave-one-init-out) ===")
    abl = {}
    for tag, keep in SETS.items():
        for kind in ("logreg", "gbt"):
            r, _ = run_cv(kind, keep, f"{tag} [{kind}]")
            r.pop("tuning", None)
            r.pop("_per_run", None)
            abl[f"{tag}|{kind}"] = r
    results["ablations"] = abl

    print("\n=== transform ablation (which normalization carries it) ===")
    tabl = {}
    suf = np.array(["r" if n.endswith("__r") else
                    "z" if n.endswith("__z") else "abs" for n in names])
    for tag, keep in (("abs-log only", ["abs"]), ("within-run z only", ["z"]),
                      ("within-run rank only", ["r"]),
                      ("z+rank (scale-free)", ["z", "r"])):
        r, _ = run_cv("logreg", None, f"{tag} [logreg]",
                      Xall=X[:, np.isin(suf, keep)])
        r.pop("tuning", None)
        r.pop("_per_run", None)
        tabl[tag] = r
    results["transform_ablation"] = tabl

    print("\n=== expressivity check: IN-SAMPLE fit (train == test) ===")
    ins = {}
    for kind in ("logreg", "gbt"):
        mu, sd = X.mean(0), X.std(0)
        sd = np.where(sd < 1e-9, 1.0, sd)
        Z = (X - mu) / sd
        if kind == "logreg":
            sc = logreg_pred(logreg_fit(Z, y, lam=1.0), Z)
        else:
            mdl, _ = gbt_fit(Z, y, rounds=300, seed=0)
            sc = gbt_pred(mdl, Z)
        pr = per_run(sc, y, runidx, runs_of)
        s = summarize(pr, cohort_of)
        ins[kind] = s
        print(f"  {kind:<10} all {s['all']['auc']:.4f}  "
              f"nat {s['natural-normal']['auc']:.4f}  "
              f"jac {s['all']['jac']:.3f} exact {s['all']['exact']:.3f}",
              flush=True)
    results["in_sample"] = ins

    print("\n=== secondary: leave-one-RUN-out (optimistic, leaks init) ===")
    sec = {}
    r, _ = run_cv("logreg", ALLG, "logreg LORO", folds="run")
    r.pop("tuning", None)
    r.pop("_per_run", None)
    sec["logreg_loro"] = r
    results["secondary"] = sec

    print("\n=== control: train on normal-embed runs only ===")
    norm_runs = {r for r in runs_of.values()
                 if cohort_of[r] in ("natural-normal", "surgical")}
    tm = np.array([runs[i] in norm_runs for i in range(len(runs))])
    r, _ = run_cv("gbt", ALLG, "gbt normal-only train", train_mask=tm)
    r.pop("tuning", None)
    r.pop("_per_run", None)
    results["normal_only_train"] = r

    print("\n=== null: committee labels permuted across runs ===")
    nulls = []
    for rep in range(3):
        rr = np.random.default_rng(100 + rep)
        perm = rr.permutation(len(runs_of))
        yp = np.zeros_like(y)
        for i in np.unique(runidx):
            yp[runidx == i] = y[runidx == perm[i]]
        scores = np.full(len(yp), np.nan)
        for u in ugroups:
            te = grp == u
            sc, _ = fit_predict("logreg", X[~te], yp[~te], grp[~te],
                                runidx[~te], X[te])
            scores[te] = sc
        pr = per_run(scores, yp, runidx, runs_of)
        nulls.append(float(np.mean([v["auc"] for v in pr.values()])))
        print(f"  null rep {rep}: {nulls[-1]:.4f}", flush=True)
    results["null_permuted_labels"] = dict(mean=float(np.mean(nulls)),
                                           reps=nulls)

    # ---- univariate per-feature AUC (descriptive, no fitting)
    uni = {}
    for j, nm in enumerate(names):
        a = [auc1(X[runidx == i, j], y[runidx == i]) for i in np.unique(runidx)]
        uni[nm] = float(np.mean(a))
    results["univariate_run_auc"] = uni
    top = sorted(uni.items(), key=lambda kv: -abs(kv[1] - 0.5))[:20]
    print("\ntop univariate features (mean per-run AUC):")
    for nm, v in top:
        print(f"   {nm:<18} {v:.4f}")

    nc = OUT.parent / "probe_noise_ceiling.json"
    if nc.exists():
        results["noise_ceiling"] = json.loads(nc.read_text())

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, indent=1, sort_keys=True))
    print(f"\nwrote {OUT}   [{time.time() - t0:.0f}s]")


if __name__ == "__main__":
    main()
