"""nu_eff scaling test on runs_dial transformer checkpoints — auditable
replacement for the missing generator of notes/nu_eff_dial.json.

Per grokked run (final test acc >= 0.99), load the last checkpoint and measure

    nu_eff = d log f / d log t |_{t=1}

by central difference in log space, f evaluated at t = e^{+-eps} (eps 0.005):
  - f = correct-class logit z_y      -> nu_logit (per example, median over grid)
  - f = margin z_y - max_{c!=y} z_c  -> nu_margin
Two scaling variants:
  - all:          every parameter scaled by t (biases + pos embed included)
  - weights_only: biases (b_in, b_out) left unscaled
Extras the original JSON never reported:
  - bias share of the l2 norm (the model is NOT exactly homogeneous)
  - t-sensitivity: nu_logit at t0 in {0.8, 1.25} (softmax term is t-dependent)

Writes notes/nu_eff_replication.json and prints medians + per-run comparison
with notes/nu_eff_dial.json when present.

Usage: uv run python nudial/nu_eff.py [--limit N]
"""
import argparse, json, math, sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "cloud"))
from train_semifinal_torch import Config, batched_final_logits, make_dataset  # noqa: E402

EPS = 0.005
BIAS_KEYS = ("blocks.0.mlp.b_in", "blocks.0.mlp.b_out")


def load_final(rd):
    from safetensors.torch import load_file
    cks = sorted((rd / "checkpoints").glob("epoch_*.safetensors"),
                 key=lambda f: int(f.stem.split("_")[1]))
    return load_file(cks[-1]), int(cks[-1].stem.split("_")[1])


def nu_at(pr, tokens, labels, cfg, t0, scale_biases):
    """Central log-difference of z_y and margin at t0 (per-example medians)."""
    out = {}
    fs = {}
    for s in (t0 * math.exp(-EPS), t0 * math.exp(EPS)):
        prs = {k: (v * s if (scale_biases or k not in BIAS_KEYS) else v).unsqueeze(0)
               for k, v in pr.items()}
        L = batched_final_logits(prs, tokens, cfg)[0].double()
        z_y = L.gather(-1, labels[:, None]).squeeze(-1)
        others = L.masked_fill(torch.nn.functional.one_hot(
            labels, L.shape[-1]).bool(), float("-inf"))
        fs[s] = (z_y, z_y - others.max(-1).values)
    lo, hi = sorted(fs)
    dlt = math.log(hi) - math.log(lo)
    for name, i in (("logit", 0), ("margin", 1)):
        a, b = fs[lo][i], fs[hi][i]
        pos = (a > 0) & (b > 0)
        nu = (torch.log(b[pos]) - torch.log(a[pos])) / dlt
        out[f"nu_{name}_med"] = float(nu.median())
        out[f"nu_{name}_iqr"] = [float(nu.quantile(0.25)), float(nu.quantile(0.75))]
        out[f"frac_pos_{name}"] = float(pos.float().mean())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--runs", default=str(ROOT / "runs_dial"))
    args = ap.parse_args()
    torch.set_grad_enabled(False)
    rows = []
    run_dirs = sorted(Path(args.runs).glob("*/*"))
    if args.limit:
        run_dirs = run_dirs[:args.limit]
    for rd in run_dirs:
        if not (rd / "spectra.npz").exists():
            continue
        z = np.load(rd / "spectra.npz")
        if float(z["test_acc"][-1]) < 0.99:
            continue
        cfg = Config.load(rd / "config.json")
        pr, ep = load_final(rd)
        pr = {k: v.double() for k, v in pr.items()}
        tokens_np, labels_np = make_dataset(cfg)
        tokens = torch.from_numpy(tokens_np)
        labels = torch.from_numpy(labels_np)
        w2 = sum(float((v ** 2).sum()) for k, v in pr.items() if k not in BIAS_KEYS)
        b2 = sum(float((v ** 2).sum()) for k, v in pr.items() if k in BIAS_KEYS)
        row = dict(run=str(rd.relative_to(ROOT)), d_mlp=cfg.d_mlp,
                   wd=cfg.weight_decay, epoch=ep,
                   bias_norm2_share=b2 / (b2 + w2))
        for tag, sb in (("all", True), ("weights_only", False)):
            r = nu_at(pr, tokens, labels, cfg, 1.0, sb)
            row.update({f"{tag}_{k}": v for k, v in r.items()})
        for t0 in (0.8, 1.25):
            row[f"all_nu_logit_med_t{t0:g}"] = nu_at(
                pr, tokens, labels, cfg, t0, True)["nu_logit_med"]
        rows.append(row)
        print(f"{row['run']}: all logit {row['all_nu_logit_med']:.2f} "
              f"margin {row['all_nu_margin_med']:.2f} | w-only logit "
              f"{row['weights_only_nu_logit_med']:.2f} | bias share "
              f"{row['bias_norm2_share']:.2e} | t0.8 {row['all_nu_logit_med_t0.8']:.2f} "
              f"t1.25 {row['all_nu_logit_med_t1.25']:.2f}", flush=True)

    out = ROOT / "notes" / "nu_eff_replication.json"
    out.write_text(json.dumps(rows, indent=1))
    med = lambda f: float(np.median([r[f] for r in rows]))
    print(f"\n{len(rows)} grokked runs")
    for f in ("all_nu_logit_med", "all_nu_margin_med",
              "weights_only_nu_logit_med", "weights_only_nu_margin_med",
              "bias_norm2_share", "all_nu_logit_med_t0.8", "all_nu_logit_med_t1.25"):
        print(f"median {f}: {med(f):.4g}")

    ref_p = ROOT / "notes" / "nu_eff_dial.json"
    if ref_p.exists():
        ref = {r["run"]: r for r in json.loads(ref_p.read_text())}
        both = [(r, ref[r["run"]]) for r in rows if r["run"] in ref]
        if both:
            for f in ("all_nu_logit_med", "all_nu_margin_med"):
                a = np.array([x[0][f] for x in both]); b = np.array([x[1][f] for x in both])
                print(f"vs committed JSON, {f}: n={len(both)} corr="
                      f"{np.corrcoef(a, b)[0,1]:.4f} max|diff|={np.abs(a-b).max():.3g}")
    print("wrote", out)


if __name__ == "__main__":
    main()
