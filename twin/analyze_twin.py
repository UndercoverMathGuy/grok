"""twin/analyze_twin.py — the four twin analyses.

Writes twin/results/{fork,swap,dial,hold}_summary.json (NOT into notes/ — these
are suite outputs, not lab notes).

Shared definitions, applied identically to every cohort:
  t_gen       first LOGGED epoch with test acc >= 0.99 sustained over 2
              consecutive log points (a single-point touch of 0.99 is noise at
              the 10-epoch cadence); -1 if it never happens.
  committee   committee_from_coeffs on the FINAL logged coefficient vector.
  reference   every arm is compared to TWO references: `ctrl` — its own group's
              unedited / eps=0 / same-data twin, trained in the SAME vmap batch
              — and `base`, the stage-1 run it forked from, which lives in a
              batch of a different width.  Gate G3 fails (see gates.py): vmap
              width changes the last bit and grokking amplifies it, so `ctrl`
              is the ONLY exactly-controlled comparison and is the primary
              reference everywhere.  `base` is reported alongside so the size
              of the cross-batch offset is visible rather than hidden.

Usage:
    uv run --group convert python twin/analyze_twin.py --runs runs_twin --out twin/results
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from core import committee_from_coeffs, jaccard, write_json

THRESH, SUSTAIN = 0.99, 2


def t_gen(epochs, test_acc, thresh=THRESH, sustain=SUSTAIN):
    ok = np.asarray(test_acc) >= thresh
    for i in range(len(ok) - sustain + 1):
        if ok[i:i + sustain].all():
            return int(epochs[i])
    return -1


def load_run(rd):
    z = np.load(rd / "spectra.npz")
    cfg = json.loads((rd / "config.json").read_text())
    return {"name": cfg["job"]["name"], "dir": str(rd), "job": cfg["job"],
            "cert": cfg.get("edit_cert", {}), "batch_size": cfg.get("batch_size"),
            "compile_mode": cfg.get("compile_mode"), "device": cfg.get("device"),
            "epochs": z["epochs"], "test_acc": z["test_acc"],
            "train_acc": z["train_acc"], "test_loss": z["test_loss"],
            "train_loss": z["train_loss"], "coeff_epochs": z["coeff_epochs"],
            "coeffs": z["coeffs"], "energy": z["energy"],
            "committee": committee_from_coeffs(z["coeffs"][-1]),
            "t_gen": t_gen(z["epochs"], z["test_acc"]),
            "final_test_acc": float(z["test_acc"][-1])}


def load_all(runs_dir):
    out = {}
    root = Path(runs_dir)
    for f in sorted(root.glob("**/spectra.npz")):
        rd = f.parent
        # underscore-prefixed subtrees (_gates, _smoke) are harness runs, not
        # cohort data — a p=23 smoke row must never pollute a production summary
        if any(part.startswith("_") for part in f.relative_to(root).parts):
            continue
        if (rd / "config.json").exists():
            r = load_run(rd)
            out[r["name"]] = r
    return out


def logs_equal(a, b):
    """Bitwise equality of a fork's logs against its base on the shared grid."""
    ia, ib = np.isin(a["epochs"], b["epochs"]), np.isin(b["epochs"], a["epochs"])
    if not ia.any():
        return False
    ok = np.array_equal(a["epochs"][ia], b["epochs"][ib])
    for k in ("train_acc", "test_acc", "train_loss", "test_loss"):
        ok &= np.array_equal(a[k][ia], b[k][ib])
    ja, jb = (np.isin(a["coeff_epochs"], b["coeff_epochs"]),
              np.isin(b["coeff_epochs"], a["coeff_epochs"]))
    ok &= np.array_equal(a["coeffs"][ja], b["coeffs"][jb])
    ok &= np.array_equal(a["energy"][ja], b["energy"][jb])
    return bool(ok)


def curve_distance(a, b):
    """(max |dacc|, area between test-acc curves / epoch span) on shared grid."""
    ia, ib = np.isin(a["epochs"], b["epochs"]), np.isin(b["epochs"], a["epochs"])
    e = a["epochs"][ia]
    if len(e) < 2:
        return float("nan"), float("nan")
    d = np.abs(a["test_acc"][ia] - b["test_acc"][ib])
    return float(d.max()), float(np.trapezoid(d, e) / (e[-1] - e[0]))


def _stats(v):
    v = [x for x in v if x is not None and not (isinstance(x, float) and np.isnan(x))]
    if not v:
        return {"n": 0}
    return {"n": len(v), "mean": float(np.mean(v)), "median": float(np.median(v)),
            "min": float(np.min(v)), "max": float(np.max(v))}


# --------------------------------------------------------------------------
# FORK — when does selection causally lock?
# --------------------------------------------------------------------------

def analyze_fork(runs):
    forks = [r for r in runs.values() if r["job"].get("cohort") == "fork"]
    by_group = defaultdict(list)
    for r in forks:
        by_group[r["job"]["group"]].append(r)

    rows = []
    chi_c, chi_b = defaultdict(list), defaultdict(list)
    eps0_checked = eps0_ok = 0
    per_base_c, per_base_b = (defaultdict(lambda: defaultdict(list)),
                              defaultdict(lambda: defaultdict(list)))
    for grp, rs in sorted(by_group.items()):
        ctrl = next((r for r in rs if r["job"]["arm"] == "ctrl"), None)
        base = runs.get(rs[0]["job"]["base"])
        if ctrl is not None and base is not None:
            eps0_checked += 1
            eps0_ok += int(logs_equal(ctrl, base))
        for r in rs:
            if r["job"]["arm"] == "ctrl":
                continue
            j = r["job"]
            Jb = jaccard(r["committee"], base["committee"]) if base else float("nan")
            Jc = jaccard(r["committee"], ctrl["committee"]) if ctrl else float("nan")
            dmax, area = curve_distance(r, ctrl) if ctrl else (float("nan"),) * 2
            rows.append({"name": r["name"], "base": j["base"], "T": j["T"],
                         "eps": j["eps"], "draw": j["draw"],
                         "committee": r["committee"], "J_base": Jb, "J_ctrl": Jc,
                         "t_gen": r["t_gen"],
                         "t_gen_ctrl": ctrl["t_gen"] if ctrl else None,
                         "t_gen_base": base["t_gen"] if base else None,
                         "acc_max_dev_ctrl": dmax, "acc_area_ctrl": area})
            chi_c[(j["T"], j["eps"])].append(1.0 - Jc)
            chi_b[(j["T"], j["eps"])].append(1.0 - Jb)
            per_base_c[j["base"]][j["T"]].append(Jc)
            per_base_b[j["base"]][j["T"]].append(Jb)

    def lock_of(pb):
        return {b: next((T for T in sorted(d) if all(x == 1.0 for x in d[T])), -1)
                for b, d in pb.items()}

    chi_rows = [{"T": T, "eps": e, "chi": float(np.mean(v)),
                 "chi_base": float(np.mean(chi_b[(T, e)])), "n": len(v)}
                for (T, e), v in sorted(chi_c.items())]
    return {"n_forks": len(rows), "reference": "within-batch eps=0 control",
            "chi": chi_rows, "lock_epoch": lock_of(per_base_c),
            "lock_epoch_base": lock_of(per_base_b),
            "eps0_bitwise_vs_base": {"checked": eps0_checked, "matched": eps0_ok},
            "runs": rows}


# --------------------------------------------------------------------------
# SWAP — is the learning trajectory blind to WHICH committee?
# --------------------------------------------------------------------------

def analyze_swap(runs):
    swaps = [r for r in runs.values() if r["job"].get("cohort") == "swap"]
    by_group = defaultdict(list)
    for r in swaps:
        by_group[r["job"]["group"]].append(r)

    rows = []
    for grp, rs in sorted(by_group.items()):
        ctrl = next((r for r in rs if r["job"]["arm"] == "ctrl"), None)
        base = runs.get(rs[0]["job"]["base"])
        for r in rs:
            j = r["job"]
            if j["arm"] == "ctrl":
                continue
            comm = set(r["committee"])
            moved = None
            if j["arm"] in ("exchange", "sham"):
                i, k = j["freqs"]            # i swapped OUT, k swapped IN
                moved = bool(k in comm and i not in comm)
            dmax_b, area_b = curve_distance(r, base) if base else (float("nan"),) * 2
            dmax_c, area_c = curve_distance(r, ctrl) if ctrl else (float("nan"),) * 2
            rows.append({
                "name": r["name"], "base": j["base"], "arm": j["arm"],
                "freqs": j.get("freqs"), "committee": r["committee"],
                "base_committee": base["committee"] if base else None,
                "ctrl_committee": ctrl["committee"] if ctrl else None,
                "moved_as_commanded": moved,
                "J_base": jaccard(r["committee"], base["committee"]) if base else None,
                "J_ctrl": jaccard(r["committee"], ctrl["committee"]) if ctrl else None,
                "t_gen": r["t_gen"],
                "dt_gen_base": (r["t_gen"] - base["t_gen"]
                                if base and r["t_gen"] >= 0 and base["t_gen"] >= 0
                                else None),
                "dt_gen_ctrl": (r["t_gen"] - ctrl["t_gen"]
                                if ctrl and r["t_gen"] >= 0 and ctrl["t_gen"] >= 0
                                else None),
                "acc_max_dev_base": dmax_b, "acc_area_base": area_b,
                "acc_max_dev_ctrl": dmax_c, "acc_area_ctrl": area_c})

    by_arm = defaultdict(list)
    for r in rows:
        by_arm[r["arm"]].append(r)
    summary = {}
    for arm, rs in sorted(by_arm.items()):
        mv = [r["moved_as_commanded"] for r in rs if r["moved_as_commanded"] is not None]
        summary[arm] = {
            "n": len(rs),
            "moved_rate": float(np.mean(mv)) if mv else None,
            "abs_dt_gen_ctrl": _stats([abs(r["dt_gen_ctrl"]) for r in rs
                                       if r["dt_gen_ctrl"] is not None]),
            "abs_dt_gen_base": _stats([abs(r["dt_gen_base"]) for r in rs
                                       if r["dt_gen_base"] is not None]),
            "acc_max_dev_ctrl": _stats([r["acc_max_dev_ctrl"] for r in rs]),
            "acc_area_ctrl": _stats([r["acc_area_ctrl"] for r in rs]),
            "J_ctrl": _stats([r["J_ctrl"] for r in rs])}
    return {"n_arms": len(rows), "by_arm": summary, "runs": rows}


# --------------------------------------------------------------------------
# DIAL — can a single frequency be steered at init?
# --------------------------------------------------------------------------

def analyze_dial(runs):
    dials = [r for r in runs.values() if r["job"].get("cohort") == "dial"]
    by_group = defaultdict(list)
    for r in dials:
        by_group[r["job"]["group"]].append(r)

    rows = []
    for grp, rs in sorted(by_group.items()):
        ctrl = next((r for r in rs if r["job"]["arm"] == "ctrl"), None)
        base = runs.get(rs[0]["job"]["base"])
        ref = ctrl or base
        for r in rs:
            j = r["job"]
            if j["arm"] == "ctrl":
                continue
            k = j["target"]
            comm, refc = set(r["committee"]), set(ref["committee"]) if ref else set()
            bg_a, bg_b = comm - {k}, refc - {k}
            rows.append({
                "name": r["name"], "base": j["base"], "target": k,
                "target_class": j["target_class"], "dose": j["dose"],
                "committee": r["committee"],
                "ref_committee": sorted(refc),
                "target_in_committee": bool(k in comm),
                "target_in_ref": bool(k in refc),
                "background_J": jaccard(bg_a, bg_b),
                "background_flipped": int(len(bg_a ^ bg_b)),
                "t_gen": r["t_gen"],
                "t_gen_ref": ref["t_gen"] if ref else None})

    tables = defaultdict(lambda: defaultdict(list))
    for r in rows:
        tables[r["target_class"]][r["dose"]].append(r)
    memb, tg, bg = {}, {}, {}
    for cls, d in sorted(tables.items()):
        memb[cls] = {str(dose): {
            "n": len(rs),
            "in_committee_rate": float(np.mean([x["target_in_committee"] for x in rs])),
            "baseline_rate": float(np.mean([x["target_in_ref"] for x in rs]))}
            for dose, rs in sorted(d.items())}
        tg[cls] = {str(dose): _stats([x["t_gen"] for x in rs if x["t_gen"] >= 0])
                   for dose, rs in sorted(d.items())}
        bg[cls] = {str(dose): {"mean_background_J":
                               float(np.mean([x["background_J"] for x in rs])),
                               "mean_flipped":
                               float(np.mean([x["background_flipped"] for x in rs]))}
                   for dose, rs in sorted(d.items())}
    return {"n_arms": len(rows), "membership_vs_dose": memb,
            "t_gen_vs_dose": tg, "background_vs_dose": bg, "runs": rows}


# --------------------------------------------------------------------------
# HOLD — does mid-run data move learning without rewriting the committee?
# --------------------------------------------------------------------------

def analyze_hold(runs):
    holds = [r for r in runs.values() if r["job"].get("cohort") == "hold"]
    by_group = defaultdict(list)
    for r in holds:
        by_group[r["job"]["group"]].append(r)

    rows = []
    ctrl_checked = ctrl_ok = 0
    for grp, rs in sorted(by_group.items()):
        ctrl = next((r for r in rs if r["job"]["arm"] == "ctrl"), None)
        base = runs.get(rs[0]["job"]["base"])
        if ctrl is not None and base is not None:
            ctrl_checked += 1
            ctrl_ok += int(logs_equal(ctrl, base))
        for r in rs:
            j = r["job"]
            T = j["T"]
            rem = r["t_gen"] - T if r["t_gen"] >= 0 else None
            rem_b = (base["t_gen"] - T if base and base["t_gen"] >= 0 else None)
            rem_c = (ctrl["t_gen"] - T if ctrl and ctrl["t_gen"] >= 0 else None)
            rows.append({
                "name": r["name"], "base": j["base"], "T": T,
                "arm": j["arm"], "level": j["level"], "frac": j["frac"],
                "committee": r["committee"],
                "J_base": jaccard(r["committee"], base["committee"]) if base else None,
                "J_ctrl": jaccard(r["committee"], ctrl["committee"]) if ctrl else None,
                "t_gen": r["t_gen"], "remaining": rem,
                "remaining_base": rem_b, "remaining_ctrl": rem_c,
                # rem == 0 (instant grok at the fork) is a legitimate value;
                # only None (never grokked) is missing data
                "ratio_base": (rem / rem_b) if rem is not None and rem_b else None,
                "ratio_ctrl": (rem / rem_c) if rem is not None and rem_c else None})

    cells = defaultdict(list)
    for r in rows:
        cells[(r["T"], r["level"])].append(r)
    table = [{"T": T, "level": lv, "n": len(rs),
              "ratio_ctrl": _stats([x["ratio_ctrl"] for x in rs]),
              "ratio_base": _stats([x["ratio_base"] for x in rs]),
              "J_base": _stats([x["J_base"] for x in rs]),
              "J_ctrl": _stats([x["J_ctrl"] for x in rs]),
              "grok_rate": float(np.mean([x["t_gen"] >= 0 for x in rs]))}
             for (T, lv), rs in sorted(cells.items())]
    return {"n_arms": len(rows), "table": table,
            "ctrl_bitwise_vs_base": {"checked": ctrl_checked, "matched": ctrl_ok},
            "runs": rows}


# --------------------------------------------------------------------------

def analyze_bases(runs):
    out = []
    for r in sorted(runs.values(), key=lambda x: x["name"]):
        if r["job"].get("cohort") in ("basesA", "basesB"):
            out.append({"name": r["name"], "frac": r["job"]["frac"],
                        "committee": r["committee"], "t_gen": r["t_gen"],
                        "final_test_acc": r["final_test_acc"]})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs_twin")
    ap.add_argument("--out", default="twin/results")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    runs = load_all(args.runs)
    out = Path(args.out)
    # Provenance for the G3 fallback: batch width and compile mode are part of
    # the arithmetic, so record what each cohort was actually trained at.  A
    # cohort with more than one width is still internally valid (controls are
    # within-batch) but is NOT comparable lane-to-lane across chunks.
    prov = defaultdict(lambda: defaultdict(set))
    for r in runs.values():
        c = r["job"].get("cohort", "?")
        prov[c]["batch_width"].add(r.get("batch_size"))
        prov[c]["compile_mode"].add(r.get("compile_mode"))
        prov[c]["device"].add(r.get("device"))
    write_json(out / "provenance.json",
               {c: {k: sorted(v, key=str) for k, v in d.items()}
                for c, d in sorted(prov.items())})
    bases = analyze_bases(runs)
    write_json(out / "bases_summary.json", {"n": len(bases), "runs": bases})
    got = {}
    for cohort, fn in (("fork", analyze_fork), ("swap", analyze_swap),
                       ("dial", analyze_dial), ("hold", analyze_hold)):
        res = fn(runs)
        write_json(out / f"{cohort}_summary.json", res)
        got[cohort] = res
    if args.quiet:
        return got

    print(f"{len(runs)} runs under {args.runs}; {len(bases)} bases")
    for b in bases:
        print(f"  {b['name']:<28} frac {b['frac']}  t_gen {b['t_gen']:>7}  "
              f"committee {b['committee']}")
    f = got["fork"]
    print(f"\nFORK ({f['n_forks']} arms)  eps0-vs-base bitwise: "
          f"{f['eps0_bitwise_vs_base']['matched']}/{f['eps0_bitwise_vs_base']['checked']}")
    print("  chi(T, eps) = mean(1 - J) vs the within-batch eps=0 twin "
          "(chi_base = vs the stage-1 base):")
    for row in f["chi"]:
        print(f"    T {row['T']:>6}  eps {row['eps']:<5g}  chi {row['chi']:.3f}  "
              f"chi_base {row['chi_base']:.3f}  (n={row['n']})")
    print("  per-base lock epoch (first T with J==1 for every draw):")
    for b, T in sorted(f["lock_epoch"].items()):
        print(f"    {b:<28} {T}")
    s = got["swap"]
    print(f"\nSWAP ({s['n_arms']} arms)")
    for arm, v in s["by_arm"].items():
        print(f"  {arm:<9} n{v['n']:<4} moved {v['moved_rate']}  "
              f"|dt_gen|(ctrl) {v['abs_dt_gen_ctrl'].get('median')}  "
              f"maxdev {v['acc_max_dev_ctrl'].get('median')}  "
              f"J_ctrl {v['J_ctrl'].get('mean')}")
    d = got["dial"]
    print(f"\nDIAL ({d['n_arms']} arms)  target-in-committee rate vs dose")
    for cls, tab in d["membership_vs_dose"].items():
        cells = "  ".join(f"{k}:{v['in_committee_rate']:.2f}" for k, v in tab.items())
        print(f"  {cls:<7} {cells}")
    h = got["hold"]
    print(f"\nHOLD ({h['n_arms']} arms)  same-data ctrl bitwise vs base: "
          f"{h['ctrl_bitwise_vs_base']['matched']}/{h['ctrl_bitwise_vs_base']['checked']}")
    for row in h["table"]:
        print(f"  T {row['T']:>5}  level {row['level']:<5g}  "
              f"ratio {row['ratio_ctrl'].get('median')}  "
              f"J_base {row['J_base'].get('mean')}  grok {row['grok_rate']:.2f}")
    print(f"\nwrote {out}/*.json")
    return got


if __name__ == "__main__":
    main()
