"""Analyze the capacity-dial sweep against the prereg in dial/README.md.

numpy + safetensors only (no torch) so it runs locally on pulled run dirs.
The three helpers below are exact copies of cloud/train_semifinal_torch.py
(fourier_basis, committee_from_coeffs, grok_epoch) — that module imports
torch at module level, unavailable on the Mac.

Per run: k_eff + top4 share of final logit Fourier energy, committee label,
grok epoch, epoch-0 T_k readout AUC (W_E energy primary, OV-transmitted
secondary). Per cell: grok rate, means over groked runs. Trend tests: P-D1
(k_eff vs width at wd=1.0), P-D2 (k_eff vs wd at d_mlp=512), P-D3 (AUC vs
k_eff across cells), all Spearman with permutation p-values. Persistence:
committee Jaccard against the wd=1.0 twin (same width, same seeds).

Usage: python3 dial/analyze_dial.py [--runs runs_dial] [--out notes/dial_summary.json]
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file

P = 113
NF = P // 2


def fourier_basis(p):
    basis = [np.ones(p) / np.sqrt(p)]
    for k in range(1, p // 2 + 1):
        for trig in (np.cos, np.sin):
            v = trig(2 * np.pi * k * np.arange(p) / p)
            basis.append(v / np.linalg.norm(v))
    return np.stack(basis).astype(np.float64)


def committee_from_coeffs(coeffs, floor=0.02):
    a = np.abs(coeffs)
    order = np.argsort(a)[::-1]
    logs = np.log(a[order] + 1e-12)
    cut = int(np.argmax(logs[:-1][:12] - logs[1:][:12])) + 1
    mem = order[:cut] + 1
    mem = mem[a[mem - 1] >= floor * a.max()]
    return sorted(mem.tolist())


def grok_epoch(z):
    gi = int(np.argmax(z["test_acc"] >= 0.99))
    return int(z["epochs"][gi]) if z["test_acc"][gi] >= 0.99 else -1


def freq_energy(mat_pd):
    """(p, d) Fourier-basis-transformed matrix -> per-frequency energy (NF,)."""
    E = (mat_pd ** 2).sum(1)
    return E[1::2][:NF] + E[2::2][:NF]


BASIS = fourier_basis(P)


def t_k_readouts(ckpt0):
    """Epoch-0 T_k: W_E per-frequency energy, and OV-transmitted variant."""
    W_E = ckpt0["embed.W_E"].astype(np.float64)[:, :P]     # (d, p)
    t_we = freq_energy(BASIS @ W_E.T)
    W_V, W_O = (ckpt0["blocks.0.attn.W_V"].astype(np.float64),
                ckpt0["blocks.0.attn.W_O"].astype(np.float64))
    h, dh, _ = W_V.shape
    t_ov = np.zeros(NF)
    for i in range(h):
        OV = W_O[:, i * dh:(i + 1) * dh] @ W_V[i]
        t_ov += freq_energy(BASIS @ (OV @ W_E).T)
    return t_we, t_ov


def mw_auc(scores, pos_mask):
    """Mann-Whitney AUC of scores for pos vs rest (56 freqs: exact pairs)."""
    pos, neg = scores[pos_mask], scores[~pos_mask]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    gt = (pos[:, None] > neg[None, :]).mean()
    eq = (pos[:, None] == neg[None, :]).mean()
    return float(gt + 0.5 * eq)


def spearman_perm(x, y, n_perm=20_000, seed=0):
    """Spearman rho with two-sided permutation p-value (no scipy)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan"), float("nan")
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    r = float(np.corrcoef(rx, ry)[0, 1])
    rng = np.random.default_rng(seed)
    hits = sum(abs(float(np.corrcoef(rx, rng.permutation(ry))[0, 1])) >= abs(r)
               for _ in range(n_perm))
    return r, (hits + 1) / (n_perm + 1)


def jaccard(a, b):
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if a | b else float("nan")


def load_run(rd):
    cfg = json.loads((rd / "config.json").read_text())
    z = np.load(rd / "spectra.npz")
    E = z["energy"][-1]
    E = E / E.sum()
    ck0 = rd / "checkpoints" / "epoch_00000.safetensors"
    t_we, t_ov = t_k_readouts(load_file(str(ck0))) if ck0.exists() else (None, None)
    comm = committee_from_coeffs(z["coeffs"][-1])
    mask = np.zeros(NF, bool)
    mask[[k - 1 for k in comm]] = True
    return {
        "run": str(rd), "d_mlp": cfg["d_mlp"], "wd": cfg["weight_decay"],
        "data_seed": cfg["data_seed"], "init_seed": cfg["init_seed"],
        "final_test_acc": float(z["test_acc"][-1]),
        "groked": bool(z["test_acc"][-1] >= 0.99),
        "grok_epoch": grok_epoch(z),
        "k_eff": float(1.0 / (E ** 2).sum()),
        "top4_share": float(np.sort(E)[-4:].sum()),
        "committee": comm, "committee_size": len(comm),
        "auc_we": mw_auc(t_we, mask) if t_we is not None else float("nan"),
        "auc_ov": mw_auc(t_ov, mask) if t_ov is not None else float("nan"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=str(Path(__file__).resolve().parents[1]
                                          / "runs_dial"))
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1]
                                         / "notes" / "dial_summary.json"))
    args = ap.parse_args()

    run_dirs = sorted(p.parent for p in Path(args.runs).glob("*/*/spectra.npz"))
    runs = [load_run(rd) for rd in run_dirs]
    print(f"{len(runs)} finished runs under {args.runs}")
    if not runs:
        return

    # ---- per-cell aggregates -------------------------------------------
    cells = {}
    for r in runs:
        cells.setdefault((r["d_mlp"], r["wd"]), []).append(r)
    cell_rows = []
    for (w, wd), rs in sorted(cells.items()):
        g = [r for r in rs if r["groked"]]
        row = {"d_mlp": w, "wd": wd, "n": len(rs), "n_grok": len(g),
               "grok_rate": len(g) / len(rs)}
        for key in ("k_eff", "top4_share", "committee_size", "auc_we",
                    "auc_ov", "grok_epoch"):
            vals = [r[key] for r in g if not (isinstance(r[key], float)
                                              and math.isnan(r[key]))]
            row[key] = float(np.mean(vals)) if vals else None
            if key == "k_eff" and vals:
                row["k_eff_sd"] = float(np.std(vals))
        cell_rows.append(row)

    def fmt(v, spec):
        return format(v, spec) if v is not None else "-"

    print(f"\n{'d_mlp':>6} {'wd':>5} {'grok':>6} {'k_eff':>12} "
          f"{'top4':>6} {'|C|':>5} {'AUC_we':>7} {'AUC_ov':>7} {'grok_ep':>8}")
    for c in cell_rows:
        ke = (f"{c['k_eff']:.2f}±{c.get('k_eff_sd', 0):.2f}"
              if c["k_eff"] is not None else "-")
        print(f"{c['d_mlp']:>6} {c['wd']:>5g} {c['grok_rate']:>6.2f} "
              f"{ke:>12} {fmt(c['top4_share'], '6.2f'):>6} "
              f"{fmt(c['committee_size'], '5.1f'):>5} "
              f"{fmt(c['auc_we'], '7.3f'):>7} {fmt(c['auc_ov'], '7.3f'):>7} "
              f"{fmt(c['grok_epoch'], '8.0f'):>8}")

    # ---- prereg trend tests --------------------------------------------
    ok = [c for c in cell_rows if c["grok_rate"] >= 0.5 and c["k_eff"] is not None]

    def cells_at(fixed_key, fixed_val, x_key):
        sel = sorted((c for c in ok if c[fixed_key] == fixed_val),
                     key=lambda c: c[x_key])
        return ([c[x_key] for c in sel], [c["k_eff"] for c in sel])

    tests = {}
    xs, ys = cells_at("wd", 1.0, "d_mlp")
    r, p = spearman_perm(xs, ys)
    tests["P-D1_keff_vs_width_at_wd1"] = {
        "x": xs, "k_eff": ys, "spearman": r, "p_perm": p,
        "verdict": "PASS" if (not math.isnan(r) and r > 0 and p < 0.05)
        else "FAIL/insufficient"}
    xs, ys = cells_at("d_mlp", 512, "wd")
    r, p = spearman_perm(xs, ys)
    tests["P-D2_keff_vs_wd_at_512"] = {
        "x": xs, "k_eff": ys, "spearman": r, "p_perm": p,
        "verdict": "PASS" if (not math.isnan(r) and r < 0 and p < 0.05)
        else "FAIL/insufficient"}
    ke = [c["k_eff"] for c in ok if c["auc_we"] is not None]
    au = [c["auc_we"] for c in ok if c["auc_we"] is not None]
    r, p = spearman_perm(ke, au)
    tests["P-D3_auc_vs_keff"] = {
        "n_cells": len(ke), "spearman": r, "p_perm": p,
        "verdict": "PASS" if (not math.isnan(r) and r < 0 and p < 0.05)
        else "FAIL/insufficient"}

    print("\nprereg tests:")
    for name, t in tests.items():
        print(f"  {name}: rho {t['spearman']:.3f}  p {t['p_perm']:.4f}  "
              f"-> {t['verdict']}" if not math.isnan(t["spearman"])
              else f"  {name}: insufficient groked cells")

    # ---- persistence along wd (init-matched twins vs wd=1.0) ----------
    persist = {}
    by_key = {(r["d_mlp"], r["wd"], r["data_seed"], r["init_seed"]): r
              for r in runs}
    for r in runs:
        if r["wd"] == 1.0 or not r["groked"]:
            continue
        ref = by_key.get((r["d_mlp"], 1.0, r["data_seed"], r["init_seed"]))
        if ref is None or not ref["groked"]:
            continue
        persist.setdefault((r["d_mlp"], r["wd"]), []).append(
            jaccard(r["committee"], ref["committee"]))
    persist_rows = [{"d_mlp": w, "wd": wd, "n": len(v),
                     "jaccard_vs_wd1": float(np.mean(v))}
                    for (w, wd), v in sorted(persist.items())]
    if persist_rows:
        print("\ncommittee persistence vs wd=1.0 twin (groked pairs):")
        for pr in persist_rows:
            print(f"  w{pr['d_mlp']:>5} wd{pr['wd']:>4g}: "
                  f"J {pr['jaccard_vs_wd1']:.3f} (n={pr['n']})")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"runs": runs, "cells": cell_rows,
                               "tests": tests, "persistence": persist_rows},
                              indent=1))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
