"""Empirical NOISE CEILING for committee prediction from the init.

The probe-ceiling question ("is AUC 0.76 an information ceiling or a
functional problem?") has an upper bound that no probe can beat: if two runs
that share the SAME epoch-0 draw but differ in training dynamics land on
different committees, then the init simply does not contain the answer, and
the best possible init-only readout is bounded by that disagreement.

This script measures the bound directly. For every pair of kept v2 runs
sharing (data_seed, init_seed), it scores run A's committee using run B's
committee indicator as the "prediction" and reports AUC / Jaccard. The
sibling is a *better-than-any-probe* oracle: it has seen a full training run
of the same init. Pairs from different inits give the chance level.

Analysis-only.
"""
import json
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

ROOT = Path("/Users/ruhaanrajadhyaksha/projects/grok")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "semifinal" / "analysis"))
from common import discover  # noqa: E402


def auc_of_indicator(score_set, true_set, nf):
    s = np.zeros(nf)
    s[np.array(sorted(score_set)) - 1] = 1.0
    lab = np.zeros(nf, bool)
    lab[np.array(sorted(true_set)) - 1] = True
    n1, n0 = int(lab.sum()), int((~lab).sum())
    r = rankdata(s)
    return float((r[lab].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def jac(a, b):
    a, b = set(a), set(b)
    return len(a & b) / len(a | b)


def main():
    runs = []
    for r in discover():
        runs.append(dict(rel=r["rel"], cohort=r["cohort"], fam=r["fam"],
                         grp=f"{r['cfg'].data_seed}/{r['cfg'].init_seed}",
                         comm=r["committee"], nf=r["cfg"].p // 2))
    nf = runs[0]["nf"]
    same, diff = defaultdict(list), []
    for a, b in combinations(runs, 2):
        # symmetrized: predicting A from B and B from A
        m = dict(auc=0.5 * (auc_of_indicator(b["comm"], a["comm"], nf)
                            + auc_of_indicator(a["comm"], b["comm"], nf)),
                 jac=jac(a["comm"], b["comm"]))
        if a["grp"] != b["grp"]:
            # does the shared TRAIN MASK (data_seed) alone carry committee
            # information? If yes, the sibling oracle's edge is partly mask,
            # not init geometry.
            key = ("mask", "same" if a["grp"].split("/")[0] ==
                   b["grp"].split("/")[0] else "diff")
            same[key].append(m)
        if a["grp"] == b["grp"]:
            key = tuple(sorted([a["cohort"], b["cohort"]]))
            same[key].append(m)
            same[("ANY", "ANY")].append(m)
            if a["fam"] == b["fam"]:
                same[("samefam", a["fam"])].append(m)
        else:
            diff.append(m)

    def agg(rows):
        return dict(n=len(rows),
                    auc=float(np.mean([r["auc"] for r in rows])),
                    auc_sd=float(np.std([r["auc"] for r in rows], ddof=1)),
                    jac=float(np.mean([r["jac"] for r in rows])))

    out = {"different_init_pairs": agg(diff),
           "same_init_pairs": {"|".join(k): agg(v) for k, v in sorted(same.items())}}
    print(f"different-init pairs   n={out['different_init_pairs']['n']:5d}  "
          f"AUC {out['different_init_pairs']['auc']:.4f}  "
          f"Jac {out['different_init_pairs']['jac']:.3f}   (chance)")
    print("\nSAME-init sibling oracle (upper bound on any init-only probe):")
    for k, v in sorted(out["same_init_pairs"].items(),
                       key=lambda kv: -kv[1]["auc"]):
        print(f"  {k:<38} n={v['n']:4d}  AUC {v['auc']:.4f} "
              f"+-{v['auc_sd']:.3f}  Jac {v['jac']:.3f}")

    # ---- leave-one-out empirical posterior oracle -----------------------
    # A single sibling's committee is a hard Bernoulli sample of the
    # init-conditional probability q_k = P(k in committee | init), so its AUC
    # UNDER-states the ceiling. Averaging the other siblings' indicators
    # estimates q_k itself, which is exactly what the Bayes-optimal init-only
    # readout would score. This oracle also cheats (its siblings were trained),
    # so its AUC is an upper bound on any epoch-0 probe.
    # Families whose ONLY difference from their init-mates is optimizer
    # hyperparameters. Every other family (surgical arms, phase2-tilt) was
    # designed with knowledge of the base run's committee, so a sibling oracle
    # built from those leaks the label. This set does not.
    DYN_ONLY = {"orthWE", "dyn-lr3", "dyn-lrlo", "dyn-wd04", "dyn-wd25",
                "eff-G"}
    by_grp = defaultdict(list)
    for r in runs:
        by_grp[r["grp"]].append(r)
    post = defaultdict(list)
    for r in runs:
        sibs_all = [s for s in by_grp[r["grp"]] if s is not r]
        for tag, sibs in (("any-sibling", sibs_all),
                          ("same-cohort", [s for s in sibs_all
                                           if s["cohort"] == r["cohort"]]),
                          ("same-family", [s for s in sibs_all
                                           if s["fam"] == r["fam"]]),
                          # normal-W_E siblings only: for the 8 natural runs
                          # this is their surgical descendants, the closest
                          # available stand-in for a natural repeat (there are
                          # no natural repeats of one init in v2).
                          ("dynamics-only",
                           [s for s in sibs_all if s["fam"] in DYN_ONLY]
                           if r["fam"] in DYN_ONLY else []),
                          ("normalWE-sibling",
                           [s for s in sibs_all
                            if s["cohort"] in ("natural-normal", "surgical")])):
            if not sibs:
                continue
            q = np.zeros(nf)
            for s in sibs:
                q[np.array(s["comm"]) - 1] += 1.0 / len(sibs)
            lab = np.zeros(nf, bool)
            lab[np.array(r["comm"]) - 1] = True
            n1, n0 = int(lab.sum()), int((~lab).sum())
            rk = rankdata(q)
            a = float((rk[lab].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
            K = len(r["comm"])
            pred = set((np.argsort(-q)[:K] + 1).tolist())
            post[(tag, r["cohort"])].append(
                dict(auc=a, jac=jac(pred, r["comm"]),
                     exact=float(set(r["comm"]) == pred), n_sib=len(sibs)))
            post[(tag, "ALL")].append(post[(tag, r["cohort"])][-1])

    def agg2(rows):
        return dict(n=len(rows), n_sib=float(np.mean([r["n_sib"] for r in rows])),
                    auc=float(np.mean([r["auc"] for r in rows])),
                    auc_sd=float(np.std([r["auc"] for r in rows], ddof=1))
                    if len(rows) > 1 else 0.0,
                    jac=float(np.mean([r["jac"] for r in rows])),
                    exact=float(np.mean([r["exact"] for r in rows])))

    out["posterior_oracle"] = {"|".join(k): agg2(v) for k, v in post.items()}
    print("\nLOO mean-of-siblings posterior oracle (ceiling estimate):")
    for k, v in sorted(out["posterior_oracle"].items()):
        print(f"  {k:<32} n={v['n']:3d} sib={v['n_sib']:.1f}  "
              f"AUC {v['auc']:.4f} +-{v['auc_sd']:.3f}  Jac {v['jac']:.3f}  "
              f"exact {v['exact']:.3f}")

    p = ROOT / "notes" / "pivot" / "probe_noise_ceiling.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1, sort_keys=True))
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
