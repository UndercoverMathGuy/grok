"""ANALYSIS B - does the compiled T_k margin collapse the s x K grid?

For every compiled arm (Phase A flat-substrate s=3 arms, Phase B 96-arm
dose x K sweep) we recompute the arrival-loudness spectrum T_k of the
COMPILED epoch-0 checkpoint and, for each target t in the predicted
committee S, its margin

    m_t  = log T_t - log max_{k not in S} T_k        (vs best background)
    m'_t = log T_t - log T_(K+1)                     (vs (K+1)-th overall)

Outcome: per-target adoption at the end of training (the scores json's
10x-median-amplitude criterion, cross-checked against membership in the
unified committee detector).

Theory: adoption is a function of the margin alone - the compiled dose s
and the committee size K should add nothing once margin is in the model.

Also overlays the single-target promotion arms from runs_torch
(dosefarm/gkrotate/collisionfarm/chaospair, doses x1.10 ... x2.25) whose
targets are recovered by diffing their epoch-0 T_k against the natural
base run's, giving the known coin-flip boundary cases in the same units.

Read-only.  Outputs notes/pivot/margin_collapse.json.
"""

import json
import re
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.stats import chi2

ROOT = Path("/Users/ruhaanrajadhyaksha/projects/grok")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "compiler"))
sys.path.insert(0, str(ROOT / "semifinal" / "analysis"))

from common import committee_from_coeffs             # noqa: E402
from core import load_ckpt, tk_profile               # noqa: E402

OUT = ROOT / "notes" / "pivot"
ARMS = ROOT / "compiler" / "arms"
RIDGE = 1e-6


# --------------------------------------------------------------- logistic

def _nll(beta, X, y, ridge=RIDGE):
    z = X @ beta
    # stable log(1+exp(z))
    ll = np.sum(y * z - np.logaddexp(0.0, z))
    return -ll + ridge * float(beta @ beta)


def _grad(beta, X, y, ridge=RIDGE):
    pr = 1 / (1 + np.exp(-(X @ beta)))
    return -X.T @ (y - pr) + 2 * ridge * beta


def logistic(X, y, names):
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    b0 = np.zeros(X.shape[1])
    res = minimize(_nll, b0, args=(X, y), jac=_grad, method="BFGS",
                   options=dict(maxiter=20000, gtol=1e-9))
    b = res.x
    # Newton polish with step halving (BFGS stalls on the separated cell-FE
    # models; the ridge keeps the Hessian invertible).
    for _ in range(200):
        pr = 1 / (1 + np.exp(-(X @ b)))
        g = _grad(b, X, y)
        H = X.T @ (X * (pr * (1 - pr))[:, None]) + 2 * RIDGE * np.eye(len(b))
        try:
            step = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            break
        f0, t = _nll(b, X, y), 1.0
        while t > 1e-8 and _nll(b - t * step, X, y) > f0:
            t /= 2
        b = b - t * step
        if np.max(np.abs(g)) < 1e-8:
            break
    gnorm = float(np.max(np.abs(_grad(b, X, y))))
    z = X @ b
    pr = 1 / (1 + np.exp(-z))
    ll = float(np.sum(y * z - np.logaddexp(0.0, z)))
    W = pr * (1 - pr)
    I = X.T @ (X * W[:, None]) + 2 * RIDGE * np.eye(X.shape[1])
    try:
        se = np.sqrt(np.diag(np.linalg.inv(I)))
    except np.linalg.LinAlgError:
        se = np.full(X.shape[1], np.nan)
    k = X.shape[1]
    return dict(names=names, coef={n: float(v) for n, v in zip(names, b)},
                se={n: float(v) for n, v in zip(names, se)},
                z={n: float(v / s) if s > 0 else None
                   for n, v, s in zip(names, b, se)},
                loglik=ll, k=k, aic=float(2 * k - 2 * ll),
                bic=float(k * np.log(len(y)) - 2 * ll),
                n=int(len(y)), converged=bool(gnorm < 1e-5), grad_inf=gnorm,
                fitted=pr.tolist())


def lr_test(big, small):
    d = 2 * (big["loglik"] - small["loglik"])
    df = big["k"] - small["k"]
    return dict(chi2=float(d), df=int(df),
                p=float(chi2.sf(max(d, 0.0), df)) if df > 0 else None,
                d_aic=float(small["aic"] - big["aic"]))


# ------------------------------------------------------------ compiled arms

def arm_rows():
    rows = []
    for phase, man_f, sc_f in [("A", "phaseA_manifest.json",
                                "phaseA_scores.json"),
                               ("B", "phaseB_manifest.json",
                                "phaseB_scores.json")]:
        man = json.loads((ARMS / man_f).read_text())
        sc = {r["tag"]: r for r in json.loads((ARMS / sc_f).read_text())}
        for a in man:
            S = a.get("predicted_committee")
            tag = a["tag"]
            if S is None or tag not in sc:
                continue
            s_row = sc[tag]
            p = a["config"]["p"]
            tk = np.asarray(tk_profile(load_ckpt(a["ckpt"]), p), float)
            nf = p // 2
            S = sorted(S)
            K = len(S)
            bg = [k for k in range(1, nf + 1) if k not in S]
            bg_max = max(tk[k - 1] for k in bg)
            ts = np.sort(tk)[::-1]
            t_k1 = ts[K]                                  # (K+1)-th overall
            cell = a.get("cell", {})
            s_dose = float(cell.get("s", a["report"]["safety"]))
            comm = s_row["committee"]
            edits = a["report"]["edits"]
            for t in S:
                # 'need' = T_k gain the compiler had to apply to t = how
                # quiet t was on the substrate before compilation; the only
                # per-target quantity that varies inside an arm.
                need = float(edits[str(t)]["need"])
                m = float(np.log(tk[t - 1]) - np.log(bg_max))
                m2 = float(np.log(tk[t - 1]) - np.log(t_k1))
                rows.append(dict(
                    phase=phase, tag=tag, cell_s=s_dose, K=K, target=int(t),
                    margin=m, margin_k1=m2, need=need,
                    log_need=float(np.log(need)),
                    # pre-compilation loudness of t, in margin units
                    pre_margin=float(m - np.log(need)),
                    tk=float(tk[t - 1]), bg_max=float(bg_max),
                    tk_rank=int(np.flatnonzero(np.argsort(tk)[::-1]
                                               == t - 1)[0]) + 1,
                    adopted=bool(s_row["adopted"][str(t)]),
                    in_committee=bool(t in comm),
                    amp=float(s_row["target_amps"][str(t)]),
                    arm_exact=bool(s_row.get("exact", False)),
                    arm_jaccard=float(s_row.get("jaccard", float("nan"))),
                    base=a["base"], substrate=a["report"]["substrate"],
                    realized_arm_margin=float(a["report"]["margin"]),
                    grok=int(s_row.get("grok", -1)),
                ))
    return rows


# ------------------------------------------- single-target promotion arms

BASES = {}


def base_tk(seed, p=113):
    if seed not in BASES:
        f = (ROOT / "runs_torch" / "p-113" / "seed3148" / f"seed{seed}"
             / "checkpoints" / "epoch_00000.safetensors")
        BASES[seed] = np.asarray(tk_profile(load_ckpt(f), p), float)
    return BASES[seed]


def promotion_rows():
    """dosefarm / gkrotate / collisionfarm / chaospair arms: one boosted
    frequency each.  Target + dose recovered by diffing T_k against the
    base natural init; adoption = target in the final committee."""
    rows = []
    p, nf = 113, 56
    for fam in ["dosefarm", "gkrotate", "collisionfarm", "chaospair"]:
        for cfgf in sorted((ROOT / "runs_torch" / fam).rglob("config.json")):
            d = cfgf.parent
            e0 = d / "checkpoints" / "epoch_00000.safetensors"
            sp = d / "spectra.npz"
            if not (e0.exists() and sp.exists()):
                continue
            rel = str(d.relative_to(ROOT / "runs_torch"))
            m = re.search(r"seed(\d+)", rel)
            seed = m.group(1)
            try:
                bt = base_tk(seed, p)
            except Exception:
                continue
            tk = np.asarray(tk_profile(load_ckpt(e0), p), float)
            ratio = tk / bt
            t = int(np.argmax(np.abs(np.log(ratio)))) + 1
            dose = float(ratio[t - 1])
            if dose < 1.0:            # suppression arm, not a promotion
                continue
            z = np.load(sp)
            if float(z["test_acc"][-1]) < 0.99:
                continue
            comm = committee_from_coeffs(z["coeffs"][-1])
            others = np.delete(tk, t - 1)
            marg = float(np.log(tk[t - 1]) - np.log(others.max()))
            amps = np.abs(z["coeffs"][-1])
            rows.append(dict(fam=fam, rel=rel, seed=seed, target=t,
                             dose=dose, margin=marg,
                             tk_rank=int(np.flatnonzero(
                                 np.argsort(tk)[::-1] == t - 1)[0]) + 1,
                             adopted=bool(t in comm),
                             adopted_10x=bool(amps[t - 1]
                                              >= 10 * np.median(amps)),
                             committee=comm))
    return rows


# --------------------------------------------------- joint regime summary

def write_joint(res):
    """Single spectral-spread ladder across both analyses (needs
    gap_determinism.json to have been produced first)."""
    gd_f = OUT / "gap_determinism.json"
    if not gd_f.exists():
        print("gap_determinism.json missing - skipping joint summary")
        return
    gd = json.loads(gd_f.read_text())
    rungs = []
    for c in ["double-flat", "orth-flat", "natural-normal", "surgical"]:
        sub = [r for r in gd["rows"] if r["cohort"] == c]
        if not sub:
            continue
        rungs.append(dict(
            regime=c, n=len(sub), unit="run",
            head_spread_median=float(np.median([r["head_spread"]
                                                for r in sub])),
            head_spread_max=float(np.max([r["head_spread"] for r in sub])),
            gap_at_cut_median=float(np.median([r["gap_log"] for r in sub])),
            mean_jaccard=float(np.mean([r["jaccard"] for r in sub])),
            exact_rate=float(np.mean([r["exact"] for r in sub])),
            top1_hit_rate=float(np.mean([r["top1_hit"] for r in sub])),
            membership_auc=float(np.mean([r["memb_auc"] for r in sub]))))
    rows = res["rows"]
    arms = {}
    for r in rows:
        arms[r["tag"]] = r
    for s in sorted({r["cell_s"] for r in rows}):
        sub = [r for r in arms.values() if r["cell_s"] == s]
        tr = [r for r in rows if r["cell_s"] == s]
        rungs.append(dict(
            regime=f"compiled s={s:g}", n=len(sub), unit="arm",
            head_spread_median=float(np.log(s)),
            head_spread_max=float(np.log(s)),
            gap_at_cut_median=float(np.log(s)),
            mean_jaccard=float(np.mean([r["arm_jaccard"] for r in sub])),
            exact_rate=float(np.mean([r["arm_exact"] for r in sub])),
            top1_hit_rate=None,
            adoption_rate=float(np.mean([r["adopted"] for r in tr])),
            # membership AUC is 1.0 by construction for compiled arms (the
            # targets ARE the top-K of T_k), so it is not reported.
            membership_auc=None))
    joint = dict(
        note="regime ladder: init T_k spectral spread vs predictability of "
             "the final committee.  Natural/flat inits occupy a narrow "
             "degenerate band; only compiled arms reach a gapped regime.",
        rungs=rungs,
        analysis_A=dict(
            n_runs=gd["n_runs"], exact_rate=gd["pooled"]["exact_rate"],
            mean_jaccard=gd["pooled"]["mean_jaccard"],
            mean_membership_auc=gd["pooled"]["mean_memb_auc"],
            gap_at_cut=gd["pooled"]["by_gap"]["gap_log"],
            head_spread=gd["pooled"]["by_gap"]["head_spread"],
            unedited=gd["by_cohort"]["unedited-inits"]["by_gap"],
            twin_tests=gd["twin_tests"],
            twin_groups=[{k: v for k, v in t.items() if k != "committees"}
                         for t in gd["twin_groups"]]),
        analysis_B=dict(
            margin_degeneracy=res["margin_degeneracy"],
            lr_tests=res["lr_tests"],
            models={k: dict(coef=v["coef"], z=v["z"], aic=v["aic"],
                            loglik=v["loglik"])
                    for k, v in res["models"].items()},
            cell_table=res["cell_table"],
            identified_threshold=res["identified_threshold"],
            eviction=res["eviction"]))
    (OUT / "gap_and_collapse.json").write_text(json.dumps(joint, indent=1))
    print(f"wrote {OUT / 'gap_and_collapse.json'}")
    for r in rungs:
        ma = r["membership_auc"]
        print(f"  {r['regime']:<18} n={r['n']:<4} spread~"
              f"{r['head_spread_median']:.3g} nats  J={r['mean_jaccard']:.3f} "
              f"exact={r['exact_rate']:.2f} "
              f"membAUC={f'{ma:.3f}' if ma else 'n/a (=1 by construction)'}")


# ------------------------------------------------------------------- main

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = arm_rows()
    print(f"{len(rows)} compiled targets from "
          f"{len({r['tag'] for r in rows})} arms")
    y = np.array([r["adopted"] for r in rows], float)
    m = np.array([r["margin"] for r in rows], float)
    m2 = np.array([r["margin_k1"] for r in rows], float)
    logs = np.log(np.array([r["cell_s"] for r in rows], float))
    K = np.array([r["K"] for r in rows], float)
    one = np.ones_like(y)
    agree = float(np.mean([r["adopted"] == r["in_committee"] for r in rows]))
    print(f"adoption rate {y.mean():.4f}; 10x-criterion vs committee-"
          f"membership agreement {agree:.4f}")
    print(f"corr(margin, log s) = {np.corrcoef(m, logs)[0,1]:.3f}; "
          f"corr(margin, K) = {np.corrcoef(m, K)[0,1]:.3f}")

    # --- how much independent variation does the margin actually have?
    by_tag = {}
    for r in rows:
        by_tag.setdefault(r["tag"], []).append(r)
    within_sd = [float(np.std([r["margin"] for r in v]))
                 for v in by_tag.values() if len(v) > 1]
    degeneracy = dict(
        n_distinct_margins=int(len({round(r["margin"], 6) for r in rows})),
        max_within_arm_sd=float(max(within_sd)),
        mean_within_arm_sd=float(np.mean(within_sd)),
        note="compile_init sets T_t = s * max_bg for EVERY target, so all "
             "targets inside an arm are exactly tied and margin == log s "
             "up to the Phase-A background renorm",
    )
    print("margin degeneracy:", degeneracy["n_distinct_margins"],
          "distinct values, max within-arm sd "
          f"{degeneracy['max_within_arm_sd']:.2e}")

    M = {}
    M["null"] = logistic(np.c_[one], y, ["const"])
    M["margin"] = logistic(np.c_[one, m], y, ["const", "margin"])
    M["margin_k1"] = logistic(np.c_[one, m2], y, ["const", "margin_k1"])
    M["margin+logs"] = logistic(np.c_[one, m, logs],
                                y, ["const", "margin", "log_s"])
    M["margin+K"] = logistic(np.c_[one, m, K], y, ["const", "margin", "K"])
    M["margin+logs+K"] = logistic(np.c_[one, m, logs, K], y,
                                  ["const", "margin", "log_s", "K"])
    M["logs+K"] = logistic(np.c_[one, logs, K], y, ["const", "log_s", "K"])
    M["logs"] = logistic(np.c_[one, logs], y, ["const", "log_s"])
    # cell fixed effects: the fully saturated s x K grid
    cells = sorted({(r["cell_s"], r["K"]) for r in rows})
    D = np.array([[1.0 if (r["cell_s"], r["K"]) == c else 0.0
                   for c in cells[1:]] for r in rows])
    ln = np.array([r["log_need"] for r in rows], float)
    M["logs+K+logneed"] = logistic(np.c_[one, logs, K, ln], y,
                                   ["const", "log_s", "K", "log_need"])
    M["margin+cellFE"] = logistic(np.c_[one, m, D], y,
                                  ["const", "margin"]
                                  + [f"cell{c}" for c in cells[1:]])
    M["cellFE"] = logistic(np.c_[one, D], y,
                           ["const"] + [f"cell{c}" for c in cells[1:]])

    tests = {
        "margin vs null": lr_test(M["margin"], M["null"]),
        "margin+logs vs margin": lr_test(M["margin+logs"], M["margin"]),
        "margin+K vs margin": lr_test(M["margin+K"], M["margin"]),
        "margin+logs+K vs margin": lr_test(M["margin+logs+K"], M["margin"]),
        "margin+cellFE vs margin": lr_test(M["margin+cellFE"], M["margin"]),
        "margin+logs+K vs logs+K": lr_test(M["margin+logs+K"], M["logs+K"]),
        "logs+K vs null": lr_test(M["logs+K"], M["null"]),
        "logs+K+logneed vs logs+K": lr_test(M["logs+K+logneed"], M["logs+K"]),
    }

    b = M["margin"]["coef"]
    m_star = -b["const"] / b["margin"]
    steep = b["margin"]
    m90 = (np.log(9) - b["const"]) / b["margin"]
    m10 = (np.log(1 / 9) - b["const"]) / b["margin"]

    # cluster bootstrap over arms for m* and slope
    tags = np.array([r["tag"] for r in rows])
    uniq = np.unique(tags)
    idx = {t: np.flatnonzero(tags == t) for t in uniq}
    rng = np.random.default_rng(0)
    boot_ms, boot_b = [], []
    for _ in range(2000):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        ii = np.concatenate([idx[t] for t in pick])
        if len(np.unique(y[ii])) < 2:
            continue
        f = logistic(np.c_[one[ii], m[ii]], y[ii], ["const", "margin"])
        if f["coef"]["margin"] > 1e-6:
            boot_ms.append(-f["coef"]["const"] / f["coef"]["margin"])
            boot_b.append(f["coef"]["margin"])
    ci = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]

    # per-cell observed vs margin-model-predicted adoption (collapse check)
    pr = np.array(M["margin"]["fitted"])
    cell_tab = []
    for c in cells:
        sel = np.array([(r["cell_s"], r["K"]) == c for r in rows])
        cell_tab.append(dict(s=c[0], K=c[1], n=int(sel.sum()),
                             observed=float(y[sel].mean()),
                             predicted_margin_only=float(pr[sel].mean()),
                             mean_margin=float(m[sel].mean()),
                             min_margin=float(m[sel].min())))
    resid = max(abs(c["observed"] - c["predicted_margin_only"])
                for c in cell_tab)

    # margin decile calibration
    dec = np.percentile(m, np.arange(0, 101, 10))
    calib = []
    for i in range(10):
        sel = (m >= dec[i]) & (m <= dec[i + 1] if i == 9 else m < dec[i + 1])
        if sel.sum() == 0:
            continue
        calib.append(dict(decile=i + 1, n=int(sel.sum()),
                          margin_lo=float(dec[i]), margin_hi=float(dec[i + 1]),
                          observed=float(y[sel].mean()),
                          predicted=float(pr[sel].mean())))

    # eviction: is the dropped target the min-margin one?
    by_arm = {}
    for r in rows:
        by_arm.setdefault(r["tag"], []).append(r)
    ev = dict(n_arms=len(by_arm), n_multi=0, n_with_drop=0, n_one_drop=0,
              one_drop_is_min=0, drops_are_bottom=0, n_full=0)
    ev_rows = []
    for tag, rs in by_arm.items():
        if len(rs) < 2:
            continue
        ev["n_multi"] += 1
        drop = [r for r in rs if not r["adopted"]]
        if not drop:
            ev["n_full"] += 1
            continue
        ev["n_with_drop"] += 1
        order = sorted(rs, key=lambda r: r["margin"])
        bottom = {r["target"] for r in order[:len(drop)]}
        is_bottom = bottom == {r["target"] for r in drop}
        ev["drops_are_bottom"] += int(is_bottom)
        if len(drop) == 1:
            ev["n_one_drop"] += 1
            hit = drop[0]["target"] == order[0]["target"]
            ev["one_drop_is_min"] += int(hit)
        # margins are exactly tied inside an arm, so also test the only
        # per-target quantity with real variation: the compiler's 'need'
        # (how quiet the target was on the substrate before the edit).
        order_need = sorted(rs, key=lambda r: -r["need"])
        if len(drop) == 1:
            ev["one_drop_is_max_need"] = ev.get("one_drop_is_max_need", 0) + \
                int(drop[0]["target"] == order_need[0]["target"])
        ev["need_top_set"] = ev.get("need_top_set", 0) + int(
            {r["target"] for r in order_need[:len(drop)]}
            == {r["target"] for r in drop})
        ev_rows.append(dict(tag=tag, K=len(rs), s=rs[0]["cell_s"],
                            dropped=[r["target"] for r in drop],
                            min_margin_target=order[0]["target"],
                            max_need_target=order_need[0]["target"],
                            margins={r["target"]: round(r["margin"], 4)
                                     for r in rs},
                            needs={r["target"]: round(r["need"], 3)
                                   for r in rs},
                            drops_are_bottom=bool(is_bottom)))
    ev["one_drop_min_rate"] = (ev["one_drop_is_min"] / ev["n_one_drop"]
                               if ev["n_one_drop"] else None)
    ev["bottom_set_rate"] = (ev["drops_are_bottom"] / ev["n_with_drop"]
                             if ev["n_with_drop"] else None)
    # chance rate for "the dropped one is the min-margin one" = mean 1/K
    ev["chance_one_drop"] = float(np.mean(
        [1 / len(by_arm[t]) for t in by_arm
         if sum(1 for r in by_arm[t] if not r["adopted"]) == 1]))

    ev["one_drop_max_need_rate"] = (ev.get("one_drop_is_max_need", 0)
                                    / ev["n_one_drop"]
                                    if ev["n_one_drop"] else None)

    prom = promotion_rows()
    for r in prom:
        r["pred_adopt_margin_model"] = float(
            1 / (1 + np.exp(-(b["const"] + b["margin"] * r["margin"]))))
    prom.sort(key=lambda r: r["margin"])

    # --- identified fit: the compiled arms all sit at margin >= 1.10 and
    # never cross 50%, so m* from them alone is pure extrapolation.  The
    # single-target promotion arms populate margin in [-0.24, +0.68] and
    # DO cross.  Fit them alone and pooled (with a substrate indicator).
    pm = np.array([r["margin"] for r in prom], float)
    py = np.array([r["adopted"] for r in prom], float)
    prom_fit = logistic(np.c_[np.ones_like(py), pm], py, ["const", "margin"])
    pooled_m = np.r_[m, pm]
    pooled_y = np.r_[y, py]
    flat_ind = np.r_[np.ones_like(m), np.zeros_like(pm)]
    pooled_fit = logistic(np.c_[np.ones_like(pooled_y), pooled_m],
                          pooled_y, ["const", "margin"])
    pooled_fit_sub = logistic(
        np.c_[np.ones_like(pooled_y), pooled_m, flat_ind], pooled_y,
        ["const", "margin", "compiled_flat_substrate"])
    ident = {}
    for nm, f in [("promotion_only", prom_fit), ("pooled", pooled_fit),
                  ("pooled+substrate", pooled_fit_sub)]:
        c = f["coef"]
        ident[nm] = dict(
            n=f["n"], loglik=f["loglik"], aic=f["aic"],
            coef=c, se=f["se"], z=f["z"],
            m_star=float(-c["const"] / c["margin"]) if c["margin"] else None,
            slope=float(c["margin"]),
            m_star_as_dose=float(np.exp(-c["const"] / c["margin"]))
            if c["margin"] else None)
    # non-parametric threshold on the promotion arms
    from scipy.stats import fisher_exact
    tbl = [[sum(1 for r in prom if r["margin"] > 0 and r["adopted"]),
            sum(1 for r in prom if r["margin"] > 0 and not r["adopted"])],
           [sum(1 for r in prom if r["margin"] < 0 and r["adopted"]),
            sum(1 for r in prom if r["margin"] < 0 and not r["adopted"])]]
    ident["promotion_nonparametric"] = dict(
        n=len(prom), table_pos_neg=tbl,
        fisher_p=float(fisher_exact(tbl, alternative="greater").pvalue),
        adopted_when_margin_gt0=[int(sum(1 for r in prom
                                         if r["margin"] > 0 and r["adopted"])),
                                 int(sum(1 for r in prom
                                         if r["margin"] > 0))],
        adopted_when_margin_lt0=[int(sum(1 for r in prom
                                         if r["margin"] < 0 and r["adopted"])),
                                 int(sum(1 for r in prom
                                         if r["margin"] < 0))],
        adopted_when_tk_rank1=[int(sum(1 for r in prom
                                       if r["tk_rank"] == 1 and r["adopted"])),
                               int(sum(1 for r in prom
                                       if r["tk_rank"] == 1))])

    res = dict(
        n_targets=len(rows), n_arms=len(by_arm),
        adoption_rate=float(y.mean()),
        criterion_agreement_10x_vs_committee=agree,
        corr_margin_logs=float(np.corrcoef(m, logs)[0, 1]),
        corr_margin_K=float(np.corrcoef(m, K)[0, 1]),
        margin_summary=dict(min=float(m.min()), p25=float(np.percentile(m, 25)),
                            median=float(np.median(m)),
                            p75=float(np.percentile(m, 75)),
                            max=float(m.max())),
        models={k: {kk: vv for kk, vv in v.items() if kk != "fitted"}
                for k, v in M.items()},
        lr_tests=tests,
        m_star=float(m_star), m_star_ci=ci(boot_ms) if boot_ms else None,
        slope=float(steep), slope_ci=ci(boot_b) if boot_b else None,
        slope_at_m_star_per_nat=float(steep / 4),
        m10=float(m10), m90=float(m90),
        s_equivalent=dict(m_star_as_dose=float(np.exp(m_star)),
                          m90_as_dose=float(np.exp(m90))),
        cell_table=cell_tab, max_cell_residual=float(resid),
        calibration_deciles=calib,
        margin_degeneracy=degeneracy,
        identified_threshold=ident,
        eviction=ev, eviction_rows=ev_rows,
        promotion_arms=prom,
        rows=rows,
    )
    (OUT / "margin_collapse.json").write_text(json.dumps(res, indent=1))
    print(f"\nwrote {OUT / 'margin_collapse.json'}")
    write_joint(res)

    print(f"\nm* = {m_star:.3f} nats (dose x{np.exp(m_star):.2f}), "
          f"slope {steep:.2f} logit/nat, m90 = {m90:.3f} "
          f"(x{np.exp(m90):.2f})")
    print("\nmodel comparison (lower AIC better):")
    for k, v in M.items():
        print(f"  {k:<16} k={v['k']:<3} LL={v['loglik']:9.3f} "
              f"AIC={v['aic']:8.2f} coefs="
              + ", ".join(f"{n}={v['coef'][n]:+.3f}" for n in v["names"][:4]))
    print("\nLR tests:")
    for k, v in tests.items():
        print(f"  {k:<28} chi2={v['chi2']:7.3f} df={v['df']} "
              f"p={v['p']:.4g} dAIC={v['d_aic']:+.2f}")
    print("\ncell table (observed vs margin-only prediction):")
    for c in cell_tab:
        print(f"  s={c['s']:<5g} K={c['K']} n={c['n']:<3} obs="
              f"{c['observed']:.3f} pred={c['predicted_margin_only']:.3f} "
              f"meanm={c['mean_margin']:.2f} minm={c['min_margin']:.2f}")
    print(f"\neviction: one-drop arms {ev['n_one_drop']}, dropped==min-margin "
          f"{ev['one_drop_is_min']} ({ev['one_drop_min_rate']:.3f}) "
          f"chance {ev['chance_one_drop']:.3f}; "
          f"dropped==max-need {ev.get('one_drop_is_max_need')} "
          f"({ev['one_drop_max_need_rate']:.3f}); "
          f"drops==bottom-|D| margin {ev['drops_are_bottom']}"
          f"/{ev['n_with_drop']}, ==top-|D| need {ev['need_top_set']}"
          f"/{ev['n_with_drop']}")
    print("\nidentified threshold fits:")
    for nm, v in ident.items():
        if "m_star" in v:
            print(f"  {nm:<22} n={v['n']:<4} m*={v['m_star']:+.3f} "
                  f"(dose x{v['m_star_as_dose']:.2f}) slope="
                  f"{v['slope']:.2f} z(margin)="
                  f"{v['z']['margin'] if v['z']['margin'] else float('nan'):.2f}")
        else:
            print(f"  {nm:<22} {v}")
    print("\npromotion arms (natural substrate, single target):")
    for r in prom:
        print(f"  {r['rel']:<32} k={r['target']:<3} dose x{r['dose']:.2f} "
              f"margin {r['margin']:+.3f} rank {r['tk_rank']:<3} "
              f"adopted={r['adopted']} pred={r['pred_adopt_margin_model']:.3f}")


if __name__ == "__main__":
    main()
