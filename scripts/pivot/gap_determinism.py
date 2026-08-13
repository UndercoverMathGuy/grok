"""ANALYSIS A - gap-conditional predictability of the init T_k readout.

Theory under test (eigenvalue-race framing): committee selection is a race
between frequencies for the top-K slots of the init arrival-loudness
readout T_k = sum_h ||W_O^h W_V^h W_E|_k||^2.  When the T_k spectrum has a
clear GAP at the committee cut, the race has a decided winner set and the
outcome should be predictable/deterministic; when the spectrum is
near-degenerate at the cut, the outcome should be a coin-flip.

Read-only: discovers every kept v2 run under runs_torch/ via
semifinal/analysis/common.discover(), loads each run's epoch-0 checkpoint
and its final Fourier spectrum.  Nothing is trained, nothing is modified.

Outputs notes/pivot/gap_determinism.json.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import mannwhitneyu, rankdata, spearmanr

ROOT = Path("/Users/ruhaanrajadhyaksha/projects/grok")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "compiler"))
sys.path.insert(0, str(ROOT / "semifinal" / "analysis"))

from common import discover, jaccard              # noqa: E402
from core import load_ckpt, tk_profile            # noqa: E402

OUT = ROOT / "notes" / "pivot"
NF = None                                          # set from p


# ------------------------------------------------------------------ stats

def auc(scores, labels):
    """Rank AUC with midranks for ties.  labels: bool array."""
    scores = np.asarray(scores, float)
    lab = np.asarray(labels, bool)
    n1, n0 = int(lab.sum()), int((~lab).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(scores)
    return float((r[lab].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def auc_ci(scores, labels, groups, n_boot=4000, seed=0):
    """Cluster bootstrap CI for the AUC (resample whole init-groups)."""
    scores, labels = np.asarray(scores, float), np.asarray(labels, bool)
    groups = np.asarray(groups)
    uniq = np.unique(groups)
    idx_by_g = {g: np.flatnonzero(groups == g) for g in uniq}
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        ii = np.concatenate([idx_by_g[g] for g in pick])
        a = auc(scores[ii], labels[ii])
        if not np.isnan(a):
            out.append(a)
    if not out:
        return (float("nan"), float("nan"))
    return (float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5)))


def mean_ci(vals, groups, n_boot=4000, seed=0):
    vals = np.asarray(vals, float)
    groups = np.asarray(groups)
    uniq = np.unique(groups)
    idx_by_g = {g: np.flatnonzero(groups == g) for g in uniq}
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        ii = np.concatenate([idx_by_g[g] for g in pick])
        out.append(vals[ii].mean())
    return (float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5)))


# ------------------------------------------------------------- per-run row

def run_row(r):
    p = r["cfg"].p
    nf = p // 2
    tk = np.asarray(tk_profile(load_ckpt(r["e0"]), p), float)
    comm = list(r["committee"])
    K = len(comm)
    order = np.argsort(tk)[::-1]                 # 0-based freq indices
    ts = tk[order]                               # sorted descending
    logts = np.log(ts + 1e-300)

    # --- gap at the true cut (K known from the outcome; "oracle-K" gap)
    gap_abs = float(ts[K - 1] - ts[K])
    gap_log = float(logts[K - 1] - logts[K])          # log-ratio gap
    gap_rel_tk = float(gap_abs / ts[K - 1])           # normalised by T_(K)
    gap_rel_med = float(gap_abs / np.median(tk))      # normalised by median T

    # --- cut-free: largest log gap in the head of the spectrum + location
    head = 12
    lg = logts[:head] - logts[1:head + 1]
    pos = int(np.argmax(lg)) + 1                      # cut position (1-based)
    gap_free_log = float(lg.max())
    gap_free_rel_med = float((ts[pos - 1] - ts[pos]) / np.median(tk))
    # spread of the head, a degeneracy scale that does not use K
    head_spread = float(logts[0] - logts[7])

    # --- prediction quality of T_k
    pred = sorted(int(i) + 1 for i in order[:K])
    jac = jaccard(pred, comm)
    exact = pred == sorted(comm)
    n_hit = len(set(pred) & set(comm))
    top1_hit = (int(order[0]) + 1) in comm
    any_hit = n_hit >= 1
    half_hit = n_hit >= (K + 1) // 2
    # degeneracy of the race at the cut: competitors sitting within 5% /20%
    # of the K-th place T value
    comp = ts[K:]
    n_within5 = int((comp >= 0.95 * ts[K - 1]).sum())
    n_within20 = int((comp >= 0.80 * ts[K - 1]).sum())
    lab = np.zeros(nf, bool)
    lab[np.array(comm) - 1] = True
    memb_auc = auc(tk, lab)
    ranks = {int(c): int(np.flatnonzero(order == c - 1)[0]) + 1 for c in comm}
    return dict(
        rel=r["rel"], fam=r["fam"], cohort=r["cohort"], p=p,
        init_seed=int(r["cfg"].init_seed), data_seed=int(r["cfg"].data_seed),
        lr=float(r["cfg"].lr), wd=float(r["cfg"].weight_decay),
        committee=comm, K=K, pred_topK=pred, jaccard=float(jac),
        exact=bool(exact), n_hit=n_hit, top1_hit=bool(top1_hit),
        any_hit=bool(any_hit), half_hit=bool(half_hit),
        n_within5=n_within5, n_within20=n_within20,
        memb_auc=float(memb_auc),
        member_tk_ranks=ranks, worst_member_rank=int(max(ranks.values())),
        gap_abs=gap_abs, gap_log=gap_log, gap_rel_tk=gap_rel_tk,
        gap_rel_med=gap_rel_med, gap_free_log=gap_free_log,
        gap_free_pos=pos, gap_free_rel_med=gap_free_rel_med,
        gap_free_pos_eq_K=bool(pos == K), head_spread=head_spread,
        tk_top12=[int(i) + 1 for i in order[:12]],
        tk_sorted_head=[float(x) for x in ts[:14]],
    )


# ------------------------------------------------------------------- tests

GAPS = ["gap_log", "gap_rel_med", "gap_rel_tk", "gap_free_log",
        "gap_free_rel_med", "head_spread"]


def quartile_table(rows, key):
    v = np.array([r[key] for r in rows], float)
    qs = np.percentile(v, [25, 50, 75])
    bins = np.digitize(v, qs)
    tab = []
    for b in range(4):
        m = bins == b
        if not m.any():
            continue
        tab.append(dict(quartile=b + 1, n=int(m.sum()),
                        gap_range=[float(v[m].min()), float(v[m].max())],
                        mean_jaccard=float(np.mean([r["jaccard"] for r, k
                                                    in zip(rows, m) if k])),
                        exact_rate=float(np.mean([r["exact"] for r, k
                                                  in zip(rows, m) if k])),
                        mean_memb_auc=float(np.mean([r["memb_auc"] for r, k
                                                     in zip(rows, m) if k]))))
    return tab


def partial_spearman(x, y, z):
    """Spearman of x,y with z rank-residualised out of both."""
    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)
    A = np.c_[np.ones_like(rz), rz]
    res = lambda a: a - A @ np.linalg.lstsq(A, a, rcond=None)[0]
    rho, pv = spearmanr(res(rx), res(ry))
    return rho, pv


BINARY = ["exact", "top1_hit", "any_hit", "half_hit"]


def gap_tests(rows, groups, label):
    out = dict(label=label, n=len(rows))
    ja = np.array([r["jaccard"] for r in rows], float)
    ma = np.array([r["memb_auc"] for r in rows], float)
    out["rates"] = {b: float(np.mean([r[b] for r in rows])) for b in BINARY}
    out["exact_rate"] = out["rates"]["exact"]
    out["mean_jaccard"] = float(ja.mean())
    out["mean_memb_auc"] = float(ma.mean())
    # scale of the race: how degenerate is the spectrum at the cut
    out["gap_scale"] = {
        k: dict(zip(["p10", "p50", "p90", "max"],
                    [float(x) for x in np.percentile(
                        [r[k] for r in rows], [10, 50, 90, 100])]))
        for k in ["gap_log", "gap_free_log", "head_spread"]}
    out["gap_scale"]["tk_ratio_at_cut_p50"] = float(
        np.exp(np.median([r["gap_log"] for r in rows])))
    out["mean_competitors_within_5pct"] = float(
        np.mean([r["n_within5"] for r in rows]))
    out["mean_competitors_within_20pct"] = float(
        np.mean([r["n_within20"] for r in rows]))
    out["by_gap"] = {}
    for key in GAPS:
        g = np.array([r[key] for r in rows], float)
        d = dict()
        for b in BINARY:
            y = np.array([r[b] for r in rows], bool)
            if 0 < y.sum() < len(y):
                d[f"auc_{b}"] = auc(g, y)
                d[f"auc_{b}_ci95"] = auc_ci(g, y, groups)
        rho, pv = spearmanr(g, ja)
        d["spearman_jaccard"] = [float(rho), float(pv)]
        rho2, pv2 = spearmanr(g, ma)
        d["spearman_memb_auc"] = [float(rho2), float(pv2)]
        # K is mechanically tied to Jaccard, so also report the partial
        # rank correlation with K residualised out.
        Kv = np.array([r["K"] for r in rows], float)
        d["partial_spearman_jaccard_given_K"] = list(
            map(float, partial_spearman(g, ja, Kv)))
        d["partial_spearman_memb_auc_given_K"] = list(
            map(float, partial_spearman(g, ma, Kv)))
        # top vs bottom quartile split
        lo, hi = np.percentile(g, [25, 75])
        bot, top = g <= lo, g >= hi
        d["quartile_split"] = dict(
            n_bottom=int(bot.sum()), n_top=int(top.sum()),
            memb_auc_bottom=float(ma[bot].mean()),
            memb_auc_top=float(ma[top].mean()),
            memb_auc_bottom_ci=mean_ci(ma[bot], np.asarray(groups)[bot]),
            memb_auc_top_ci=mean_ci(ma[top], np.asarray(groups)[top]),
            jaccard_bottom=float(ja[bot].mean()),
            jaccard_top=float(ja[top].mean()),
            rates_bottom={b: float(np.mean([r[b] for r, k
                                            in zip(rows, bot) if k]))
                          for b in BINARY},
            rates_top={b: float(np.mean([r[b] for r, k
                                         in zip(rows, top) if k]))
                       for b in BINARY},
            mannwhitney_memb_auc_p=float(
                mannwhitneyu(ma[top], ma[bot], alternative="greater").pvalue),
        )
        d["quartile_table"] = quartile_table(rows, key)
        out["by_gap"][key] = d
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    runs = list(discover(require_e0=True))
    print(f"discovered {len(runs)} kept runs with epoch-0 checkpoints")
    rows = []
    for r in runs:
        rows.append(run_row(r))
        print(f"  {rows[-1]['rel']:<45} K={rows[-1]['K']} "
              f"J={rows[-1]['jaccard']:.2f} exact={rows[-1]['exact']} "
              f"membAUC={rows[-1]['memb_auc']:.3f} "
              f"gap_log={rows[-1]['gap_log']:.3f}")

    # --- init twin groups: runs whose epoch-0 checkpoint is byte-identical
    import hashlib
    h_of = {}
    for r in runs:
        prm = load_ckpt(r["e0"])
        h_of[r["rel"]] = hashlib.sha1(
            b"".join(np.ascontiguousarray(prm[k].astype(np.float32)).tobytes()
                     for k in sorted(prm))).hexdigest()[:12]
    for row in rows:
        row["init_hash"] = h_of[row["rel"]]
    by_init = defaultdict(list)
    for row in rows:
        by_init[row["init_hash"]].append(row)

    DYN_ONLY = {"orthWE", "dyn-lr3", "dyn-lrlo", "dyn-wd04", "dyn-wd25"}
    twins = []
    for h, grp in by_init.items():
        if len(grp) < 2:
            continue
        pairs = [(a, b) for i, a in enumerate(grp) for b in grp[i + 1:]]
        jj = [jaccard(a["committee"], b["committee"]) for a, b in pairs]
        sub = [g for g in grp if g["fam"] in DYN_ONLY]
        pj = [jaccard(a["committee"], b["committee"])
              for i, a in enumerate(sub) for b in sub[i + 1:]]
        twins.append(dict(
            init_hash=h, n=len(grp), fams=sorted(g["fam"] for g in grp),
            init_seed=grp[0]["init_seed"], data_seed=grp[0]["data_seed"],
            cohort=grp[0]["cohort"],
            twin_jaccard_mean=float(np.mean(jj)),
            twin_jaccard_min=float(np.min(jj)),
            n_dyn_only=len(sub),
            twin_jaccard_dynonly=float(np.mean(pj)) if pj else None,
            # init-only (K-free) gap statistics: identical for all twins
            gap_free_log=grp[0]["gap_free_log"],
            gap_free_pos=grp[0]["gap_free_pos"],
            gap_free_rel_med=grp[0]["gap_free_rel_med"],
            head_spread=grp[0]["head_spread"],
            # outcome-side: mean per-run prediction quality inside the group
            mean_jaccard=float(np.mean([g["jaccard"] for g in grp])),
            mean_memb_auc=float(np.mean([g["memb_auc"] for g in grp])),
            exact_rate=float(np.mean([g["exact"] for g in grp])),
            Ks=sorted(g["K"] for g in grp),
            committees={g["rel"]: g["committee"] for g in grp},
        ))
    twins.sort(key=lambda t: -t["gap_free_log"])

    twin_tests = {}
    if len(twins) >= 4:
        for key in ["gap_free_log", "gap_free_rel_med", "head_spread"]:
            g = np.array([t[key] for t in twins], float)
            for ykey in ["twin_jaccard_mean", "twin_jaccard_dynonly",
                         "mean_jaccard", "mean_memb_auc"]:
                y = np.array([t[ykey] if t[ykey] is not None else np.nan
                              for t in twins], float)
                m = ~np.isnan(y)
                if m.sum() >= 4:
                    rho, pv = spearmanr(g[m], y[m])
                    twin_tests[f"{key}~{ykey}"] = [float(rho), float(pv),
                                                   int(m.sum())]

    groups_all = [r["init_hash"] for r in rows]
    res = dict(
        n_runs=len(rows),
        cohorts={c: sum(1 for r in rows if r["cohort"] == c)
                 for c in sorted({r["cohort"] for r in rows})},
        pooled=gap_tests(rows, groups_all, "all kept runs"),
        by_cohort={},
        twin_groups=twins,
        twin_tests=twin_tests,
        rows=rows,
    )
    for c in sorted({r["cohort"] for r in rows}):
        sub = [r for r in rows if r["cohort"] == c]
        if len(sub) >= 6:
            res["by_cohort"][c] = gap_tests(
                sub, [r["init_hash"] for r in sub], c)
    # unedited-init subset: natural + orth/double-flat (drop surgical arms
    # whose inits were deliberately edited)
    nat = [r for r in rows if r["cohort"] != "surgical"]
    res["by_cohort"]["unedited-inits"] = gap_tests(
        nat, [r["init_hash"] for r in nat], "natural+flat (no surgery)")

    (OUT / "gap_determinism.json").write_text(json.dumps(res, indent=1))
    print(f"\nwrote {OUT / 'gap_determinism.json'}")

    p = res["pooled"]
    print(f"\npooled n={p['n']} exact={p['exact_rate']:.3f} "
          f"meanJ={p['mean_jaccard']:.3f} membAUC={p['mean_memb_auc']:.3f}")
    print("rates", p["rates"])
    for k, d in p["by_gap"].items():
        qs = d["quartile_split"]
        print(f"  {k:<17} AUC(top1)={d.get('auc_top1_hit', float('nan')):.3f} "
              f"AUC(half)={d.get('auc_half_hit', float('nan')):.3f} "
              f"rho(J)={d['spearman_jaccard'][0]:+.3f} "
              f"p={d['spearman_jaccard'][1]:.3g} | membAUC top "
              f"{qs['memb_auc_top']:.3f} vs bottom {qs['memb_auc_bottom']:.3f}")


if __name__ == "__main__":
    main()
