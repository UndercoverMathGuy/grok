"""Score the D_n fixed-support cohort: measured gamma_hat(K) ladder.

For every run under --root (the fs_* arms written by
cloud/fixed_support_dihedral.py AND the free runs d<n>_f*_wd*), reload the
FINAL checkpoint and recompute on the full p^2 grid, in f64:

  margin   = min over all p^2 pairs of (correct logit - best wrong logit),
             over the p group classes (the '=' token column is excluded, as
             in the accuracy readout; margin_all keeps it, for reference)
  pnorm2   = sum of squares of every trained tensor (the 11 PARAM_KEYS)
  gamma    = margin / ||theta||^nu = margin / pnorm2^(nu/2),   nu = 6

gamma is the normalised margin whose constrained supremum over the class
"logits use at most K two-dim irreps" is the quantity the framework needs
when the closed form does not decouple (notes/dihedral_results.md). Per
arm x K we report median and IQR of gamma over seeds that GROKKED (test acc
>= 0.99) AND SEPARATE the full grid (margin > 0 — a run that groks by the
0.99 rule while still missing a few pairs has a negative min margin, which
is not a point of the constrained-sup problem; n/n_grok/n_sep are all
printed) and r_meas(K) = median gamma(K) / max_K median gamma(K) — the
MEASURED ladder to compare against the closed-form r(K) of dihedral.py. The
spectral readouts (K_thr, K_gap, sign share, top-6 relative amplitudes) come
from spectra.npz via score_dihedral.read_run.

CAVEAT printed with the table: the free runs were trained 30k epochs and the
fixed-support arms default to 20k; gamma still drifts up slowly under wd, so
free-vs-fixed gamma ratios are a lower bound on the free nets unless the
cohort is run with --epochs 30000.

    uv run python cloud/score_fixed_support_dihedral.py --root runs_dihedral \
        [--out notes/fixed_support_dihedral_scored.json] [--device cuda]
"""
import argparse, glob, json, re, sys
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

sys.path.insert(0, str(Path(__file__).resolve().parent))
from score_dihedral import read_run
from train_semifinal_torch import (Config, PARAM_KEYS, batched_final_logits,
                                   make_dataset, train_test_split)

NU = 6.0


def final_ckpt(rd):
    cks = sorted((rd / "checkpoints").glob("epoch_*.safetensors"))
    return cks[-1] if cks else None


@torch.no_grad()
def margin_and_norms(rd, device="cpu", chunk=8192):
    """f64 min-margin / per-tensor norms / accuracies from the final ckpt."""
    ck = final_ckpt(rd)
    if ck is None:
        return None
    cfg = Config.load(rd / "config.json")
    sd = load_file(str(ck))
    pr = {k: sd[k][None].to(device).double() for k in PARAM_KEYS}
    tok, lab = make_dataset(cfg)
    mask = train_test_split(cfg)
    tok_t = torch.from_numpy(tok).to(device)
    lab_t = torch.from_numpy(lab).to(device)
    p, N = cfg.p, tok.shape[0]
    marg, marg_all, correct = [], [], []
    for i in range(0, N, chunk):
        L = batched_final_logits(pr, tok_t[i:i + chunk], cfg)[0]   # (n, p+1)
        y = lab_t[i:i + chunk]
        hot = torch.nn.functional.one_hot(y, L.shape[-1]).bool()
        zy = L.gather(1, y[:, None]).squeeze(1)
        oth = L.masked_fill(hot, -float("inf"))
        marg.append(zy - oth[:, :p].max(1).values)
        marg_all.append(zy - oth.max(1).values)
        correct.append(L[:, :p].argmax(1) == y)
    marg = torch.cat(marg); marg_all = torch.cat(marg_all)
    correct = torch.cat(correct).cpu().numpy()
    pn = {k: float((pr[k] ** 2).sum()) for k in PARAM_KEYS}
    pn2 = float(sum(pn.values()))
    m = float(marg.min())
    return dict(epoch_ckpt=int(re.findall(r"\d+", ck.name)[-1]),
                margin=m, margin_all=float(marg_all.min()),
                margin_median=float(marg.median()), pnorm2=pn2,
                gamma=m / pn2 ** (NU / 2), pnorm2_per_tensor=pn,
                acc_ckpt=float(correct.mean()),
                train_acc_ckpt=float(correct[mask].mean()),
                test_acc_ckpt=float(correct[~mask].mean()))


def q(v):
    v = np.asarray(v, float)
    return [float(np.percentile(v, 25)), float(np.median(v)),
            float(np.percentile(v, 75))]


def score(root="runs_dihedral", out="notes/fixed_support_dihedral_scored.json",
          device="cpu"):
    rows, missing = [], []
    for f in sorted(glob.glob(f"{root}/*/*/spectra.npz")):
        rd = Path(f).parent
        r = read_run(rd)
        sup = (json.loads((rd / "support.json").read_text())
               if (rd / "support.json").exists() else None)
        cell = rd.parent.name
        r["arm"] = sup["arm"] if sup else "free30k"
        r["K"] = sup["K"] if sup else -1
        r["support"] = sup["rhos"] if sup else None
        r["rank"] = sup["rank"] if sup else None
        r["proj_E"] = sup["E"] if sup else False
        r["proj_U"] = sup["U"] if sup else False
        r["grokked"] = r["acc"] >= 0.99
        z = np.load(f)
        en = z["energy"][-1]                      # [sign, rho_1 .. rho_h]
        if sup and sup["arm"] != "free":
            on = [0] + list(sup["rhos"])
            r["offsupport_energy_frac"] = float(
                (en.sum() - en[on].sum()) / max(en.sum(), 1e-300))
        else:
            r["offsupport_energy_frac"] = None
        r["epochs_trained"] = int(z["epochs"][-1])
        mn = margin_and_norms(rd, device=device)
        if mn is None:
            missing.append(f"{cell}/{rd.name}")
        else:
            r.update(mn)
        rows.append(r)
        print(f"  scored {cell}/{rd.name}"
              + ("" if mn else "  [NO CHECKPOINT]"), flush=True)

    cells, arms = {}, {}
    for r in rows:
        cells.setdefault((r["arm"], r["K"]), []).append(r)
        arms.setdefault(r["arm"], []).append(r)
    summ = {}
    for (arm, K), rs in sorted(cells.items()):
        # gamma is only defined for a SEPARATING net: a run that groks by the
        # 0.99 rule but still misses a few pairs has min margin < 0, and a
        # negative gamma is not a point of the constrained-sup problem. The
        # gamma statistics are therefore over grokked AND separated runs, and
        # n_grok / n_sep are both reported so the exclusion is visible.
        g = [r for r in rs if r["grokked"] and r.get("margin", -1.0) > 0]
        summ[f"{arm}_K{K}"] = dict(
            arm=arm, K=K, n=len(rs), n_grok=len([r for r in rs if r["grokked"]]),
            n_sep=len(g), rank=rs[0]["rank"],
            gamma_iqr=q([r["gamma"] for r in g]) if g else None,
            gamma_max=float(max(r["gamma"] for r in g)) if g else None,
            margin_median=float(np.median([r["margin"] for r in g])) if g else None,
            pnorm2_median=float(np.median([r["pnorm2"] for r in g])) if g else None,
            K_thr=[r["K_thr"] for r in rs if r["grokked"]],
            sign_share_median=float(np.median([r["sign_share"] for r in rs
                                               if r["grokked"]])) if any(
                r["grokked"] for r in rs) else None,
            # spectral profile over GROKKED runs (score_dihedral convention,
            # directly comparable to notes/dihedral_scored.json); only the
            # gamma/margin/norm fields are restricted to separated runs
            top6_rel_median=(np.median([r["top6_rel"] for r in rs
                                        if r["grokked"]], 0).round(3).tolist()
                             if any(r["grokked"] for r in rs) else None),
            offsupport_max=(max([r["offsupport_energy_frac"] for r in rs
                                 if r["offsupport_energy_frac"] is not None],
                                default=None)))
    # measured ladder per arm: median gamma(K) / max over K within the arm
    ladder = {}
    for arm in sorted(arms):
        cs = {K: s for (a, K), s in
              [((s["arm"], s["K"]), s) for s in summ.values()]
              if a == arm and s["gamma_iqr"]}
        if not cs:
            continue
        gmax = max(s["gamma_iqr"][1] for s in cs.values())
        ladder[arm] = {K: dict(gamma_med=s["gamma_iqr"][1],
                               r_meas=s["gamma_iqr"][1] / gmax)
                       for K, s in sorted(cs.items())}
        for K in cs:
            summ[f"{arm}_K{K}"]["r_meas"] = cs[K]["gamma_iqr"][1] / gmax

    # ---- table
    def fmt(x, spec, w):
        return format("n/a" if x is None else format(x, spec), f">{w}")
    hdr = ("cell", "n/gr/sep", "rank", "margin", "|th|^2", "gamma med",
           "gamma IQR (q25, q75)", "r_meas", "K_thr", "sign", "offS")
    print("\n%-16s %8s %5s %10s %10s %11s %25s %7s %14s %6s %8s" % hdr)
    for key in sorted(summ, key=lambda k: (summ[k]["arm"], summ[k]["K"])):
        s = summ[key]
        gi = s["gamma_iqr"]
        n = f"{s['n']}/{s['n_grok']}/{s['n_sep']}"
        print(f"{key:16s} {n:>8s} "
              f"{fmt(s['rank'], 'd', 5)} "
              f"{fmt(s['margin_median'], '.3e', 10)} "
              f"{fmt(s['pnorm2_median'], '.3e', 10)} "
              f"{fmt(gi[1] if gi else None, '.4e', 11)} "
              f"{format('n/a' if not gi else f'[{gi[0]:.3e}, {gi[2]:.3e}]', '>25')} "
              f"{fmt(s.get('r_meas'), '.3f', 7)} "
              f"{str(sorted(s['K_thr'])):>14s} "
              f"{fmt(s['sign_share_median'], '.3f', 6)} "
              f"{fmt(s['offsupport_max'], '.1e', 8)}")
    for arm, lad in ladder.items():
        print(f"\nr_meas({arm}): " + "  ".join(
            f"K{K}:{v['r_meas']:.3f}" for K, v in lad.items()))
        rs = [r for r in arms[arm] if r["grokked"]]
        if rs:
            print(f"   hierarchy (median top-6 rel amplitude, {len(rs)} grokked): "
                  + str(np.median([r["top6_rel"] for r in rs], 0).round(3).tolist()))
    free = [r for r in rows if r["arm"] == "free30k" and r["grokked"]
            and r.get("margin", -1.0) > 0]
    if free:
        fq = q([r["gamma"] for r in free])
        fg = fq[1]
        print(f"\nfree runs (unconstrained, {len(free)} separated): gamma median "
              f"{fg:.4e}  IQR [{fq[0]:.3e}, {fq[2]:.3e}]")
        for arm, lad in ladder.items():
            if arm == "free30k":
                continue
            gmax = max(v["gamma_med"] for v in lad.values())
            kbest = max(lad, key=lambda K: lad[K]["gamma_med"])
            print(f"   free / max_K gamma({arm}) = {fg / gmax:.3f} "
                  f"(arm max at K={kbest})")
        print("   CAVEAT: free runs are 30k epochs, fixed-support default 20k; "
              "gamma still creeps up under wd, so this ratio is a lower bound.")
    if missing:
        print(f"\nNO FINAL CHECKPOINT for {len(missing)} run(s): {missing}")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(
        dict(nu=NU, root=root, summary=summ, ladder=ladder, runs=rows,
             missing_checkpoints=missing), indent=1))
    print("wrote", out)
    return summ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="runs_dihedral")
    ap.add_argument("--out", default="notes/fixed_support_dihedral_scored.json")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    dev = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    score(args.root, args.out, dev)


if __name__ == "__main__":
    main()
