"""Margin redo (2026-09-05): trained networks vs the provable class ceiling.

Replaces the retracted "AdamW beats margin ascent 7-20x" comparison
(notes/nudial_results.md, round 3). No random-init margin ascent anywhere.

Arms (all relu, m=512, p=113, cohorts M / MG of train_nudial.py):
  M/relu{4,6}_f0.5   AdamW, frac 0.5, wd 1   (the paper recipe, grokked)
  M/relu{4,6}_f1     AdamW, full data, wd 1
  MG/relu{4,6}_gd    plain GD + coupled l2, full data, lr 0.3, wd 3e-4
Ceiling: constructive gamma_free(K) per nu from notes/construct_gamma.json
(phase-complete product-grid single-frequency circuits, optimal block
balance; construct_polish.json shows the monotone polish cannot improve it).

Per run: gamma_0 (exact full-grid min-margin / ||theta||^nu), gamma after
the verified-monotone polish (margin_polish.polish), K_gap, k_eff, and the
ratios gamma / gamma_ceiling(K*) and gamma / gamma_ceiling(K_run).
float32 on GPU (user: no TF32), exact-min acceptance with a 1e-6 relative
tolerance so float32 noise cannot count as an ascent step.

Usage (on the pod, after cohorts M and MG have trained):
  python3 nudial/margin_redo.py [--steps 400] [--n 32]
Writes notes/margin_redo.json.
"""
import argparse, json, sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "nudial"))
import margin_polish as mp                       # noqa: E402
from train_nudial import Config, forward, per_ex_margin, sq_norm  # noqa: E402
from safetensors.torch import load_file          # noqa: E402

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.set_grad_enabled(False)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
DT = torch.float32
KSTAR = {4: 4, 6: 3}


def load_group(runs):
    cfgs, prs = [], []
    for rd in runs:
        cfg = Config(**{k: v for k, v in json.loads((rd / "config.json").read_text()).items()
                        if k in Config.__dataclass_fields__})
        cfgs.append(cfg)
        prs.append({k: v.to(DT) for k, v in load_file(rd / "final.safetensors").items()})
    return cfgs, {k: torch.stack([p[k] for p in prs]).to(DEV) for k in prs[0]}


def polish(pr, cfg, nu, steps, lr0=3e-3, label="", rtol=1e-6):
    M = pr[next(iter(pr))].shape[0]; p = cfg.p
    a = torch.arange(p, device=DEV).repeat_interleave(p); b = torch.arange(p, device=DEV).repeat(p)
    labs = ((a + b) % p)[None].expand(M, -1)
    R2 = sq_norm(pr)
    keys = list(pr)

    def margins(q):
        return per_ex_margin(forward(q, a, b, cfg), labs)

    gmin = margins(pr).min(1).values
    gam0 = (gmin / R2 ** (nu / 2)).cpu().numpy()
    lr = torch.full((M,), lr0, device=DEV, dtype=DT)
    n_acc = torch.zeros(M, device=DEV)
    for step in range(steps):
        with torch.enable_grad():
            q = {k: v.clone().requires_grad_() for k, v in pr.items()}
            gq = margins(q)
            mn = gq.min(1, keepdim=True).values
            q10 = gq.kthvalue(max(1, gq.shape[1] // 10), dim=1, keepdim=True).values
            beta = 8.0 / (q10 - mn + 1e-30)
            sm = -(torch.logsumexp(-beta * (gq - mn), 1)) / beta.squeeze(1)
            grads = torch.autograd.grad(sm.sum(), [q[k] for k in keys])
        gn2 = sum((gr ** 2).flatten(1).sum(1) for gr in grads)
        sc = lr * (R2 / gn2.clamp(min=1e-30)).sqrt()
        prop = {k: pr[k] + sc.view(-1, *[1] * (pr[k].dim() - 1)) * gr for k, gr in zip(keys, grads)}
        fac = (R2 / sq_norm(prop)).sqrt()
        prop = {k: v * fac.view(-1, *[1] * (v.dim() - 1)) for k, v in prop.items()}
        newmin = margins(prop).min(1).values
        acc = newmin > gmin * (1 + rtol)
        for k in keys:
            am = acc.view(-1, *[1] * (pr[k].dim() - 1))
            pr[k] = torch.where(am, prop[k], pr[k])
        gmin = torch.where(acc, newmin, gmin)
        n_acc += acc.float()
        lr = torch.where(acc, lr * 1.15, lr * 0.5).clamp(1e-9, 0.3)
        if (step + 1) % 100 == 0:
            gam = gmin / R2 ** (nu / 2)
            print(f"  {label} step {step+1}: median gamma {gam.median().item():.3e} "
                  f"acc-rate {acc.float().mean().item():.2f}", flush=True)
    L = forward(pr, a, b, cfg).double().cpu()
    keff, Kg = mp.spectrum(L, p)
    gam = (gmin / R2 ** (nu / 2)).cpu().numpy()
    return gam0, gam, keff, Kg, n_acc.cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--out", default=str(ROOT / "notes" / "margin_redo.json"))
    args = ap.parse_args()
    R = ROOT / "runs_nudial"
    ceil = json.loads((ROOT / "notes" / "construct_gamma.json").read_text())
    groups = {}
    for nu in (4, 6):
        groups[f"AdamW f0.5 relu{nu}"] = sorted((R / "M" / f"relu{nu}_m512_p113_f0.5_train").glob("wd1_s*"))
        groups[f"AdamW f1 relu{nu}"] = sorted((R / "M" / f"relu{nu}_m512_p113_f1_train").glob("wd1_s*"))
        groups[f"GD f1 relu{nu}"] = sorted((R / "MG" / f"relu{nu}_m512_p113_train_gd").glob("wd0.0003_lr0.3_s*"))
    results = {"ceiling": {str(nu): {K: ceil[str(nu)][K]["gamma_free"] for K in ceil[str(nu)]} for nu in (4, 6)}}
    for name, runs in groups.items():
        runs = [r for r in runs if (r / "final.safetensors").exists()][:args.n]
        if not runs:
            print(f"{name}: no runs found"); continue
        cfgs, pr = load_group(runs)
        nu = cfgs[0].degree
        # grokked filter for the frac<1 arm (final test acc from spectra)
        acc = np.array([float(np.load(r / "spectra.npz")["test_acc"][-1]) for r in runs])
        tracc = np.array([float(np.load(r / "spectra.npz")["train_acc"][-1]) for r in runs])
        print(f"{name}: n={len(runs)} nu={nu} grokked={(acc >= 0.99).sum()} train_fit={(tracc >= 0.999).sum()}")
        g0, g1, keff, Kg, nacc = polish(pr, cfgs[0], nu, args.steps, label=name)
        cK = results["ceiling"][str(nu)]; gstar = cK[str(KSTAR[nu])]
        ratio_star = g1 / gstar
        ratio_own = np.array([g / cK[str(k)] if str(k) in cK else np.nan for g, k in zip(g1, Kg)])
        results[name] = dict(seeds=[r.name for r in runs], test_acc=acc.tolist(), train_acc=tracc.tolist(),
                             gamma_before=g0.tolist(), gamma_after=g1.tolist(),
                             polish_gain=(g1 / g0 - 1).tolist(), accepted_steps=nacc.tolist(),
                             keff=keff.tolist(), K_gap=[int(k) for k in Kg],
                             ratio_to_ceiling_Kstar=ratio_star.tolist(),
                             ratio_to_ceiling_ownK=ratio_own.tolist())
        ok = acc >= 0.99 if name.startswith("AdamW f0.5") else tracc >= 0.999
        print(f"  {name}: median gamma {np.median(g0[ok]):.3e} -> {np.median(g1[ok]):.3e} | "
              f"polish gain med {np.median(g1[ok]/g0[ok]-1)*100:.2f}% max {np.max(g1[ok]/g0[ok]-1)*100:.2f}% | "
              f"ratio to ceiling(K*) med {np.median(ratio_star[ok]):.3f} | "
              f"K hist {dict(zip(*np.unique(np.array(Kg)[ok], return_counts=True)))}", flush=True)
    Path(args.out).write_text(json.dumps(results, indent=1))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
