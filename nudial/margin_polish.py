"""Verified-monotone margin polish (W3's missing tool).

Projected ascent on the exact normalised margin gamma = min_i g_i / ||theta||^nu
for nudial MLP nets, on the fixed-radius sphere, full p^2 grid, f64:
  - ascent direction: grad of softmin_beta(g) with adaptive beta (concentrates
    on the worst decile)
  - a step is ACCEPTED only if the exact min margin increases (per run);
    otherwise reverted and that run's lr halved. Monotone by construction —
    unlike cohorts D/H/J, this cannot regress its warm start.

Warm starts: cohort G (AdamW frac-1.0 wd1) finals and cohort D (margin-ascent)
finals, relu-4 and relu-6. Prints gamma before/after and K/k_eff after.

Usage: uv run python nudial/margin_polish.py [--steps 400] [--n 8]
"""
import argparse, json, math, sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "nudial"))
from train_nudial import (Config, forward, per_ex_margin, sq_norm,  # noqa: E402
                          fourier_basis, committee_from_coeffs)
from safetensors.torch import load_file  # noqa: E402

torch.set_grad_enabled(False)


def load_group(runs):
    cfgs, prs = [], []
    for rd in runs:
        cfg = Config(**{k: v for k, v in json.loads(
            (rd / "config.json").read_text()).items()
            if k in Config.__dataclass_fields__})
        cfgs.append(cfg)
        prs.append({k: v.double() for k, v in
                    load_file(rd / "final.safetensors").items()})
    keys = list(prs[0])
    pr = {k: torch.stack([p[k] for p in prs]) for k in keys}
    return cfgs, pr


def spectrum(L, p):
    """Energy share over cos/sin(k(a+b-c)) products + K_gap, per run."""
    basis = fourier_basis(p)
    bc, bs = basis[1::2], basis[2::2]
    n_ = p // 2
    u_cos = torch.from_numpy((np.einsum("np,nq->npq", bc, bc)
                              - np.einsum("np,nq->npq", bs, bs))
                             .reshape(n_, -1) / np.sqrt(2))
    u_sin = torch.from_numpy((np.einsum("np,nq->npq", bs, bc)
                              + np.einsum("np,nq->npq", bc, bs))
                             .reshape(n_, -1) / np.sqrt(2))
    b_cos = torch.from_numpy(bc); b_sin = torch.from_numpy(bs)
    c = torch.einsum("fp,mpo->mfo", u_cos, L)
    s = torch.einsum("fp,mpo->mfo", u_sin, L)
    en = (c ** 2).sum(-1) + (s ** 2).sum(-1)
    co = ((c * b_cos).sum(-1) + (s * b_sin).sum(-1)) / math.sqrt(2)
    share = en / en.sum(1, keepdim=True)
    keff = 1.0 / (share ** 2).sum(1)
    Kg = [len(committee_from_coeffs(co[i].numpy())) for i in range(L.shape[0])]
    return keff.numpy(), Kg


def polish(pr, cfg, nu, steps, lr0=3e-3, label=""):
    M = pr[next(iter(pr))].shape[0]
    p = cfg.p
    a = torch.arange(p).repeat_interleave(p); b = torch.arange(p).repeat(p)
    lab = ((a + b) % p)
    R2 = sq_norm(pr)
    labs = lab[None].expand(M, -1)

    def margins(q):
        L = forward(q, a, b, cfg)
        return per_ex_margin(L, labs), L

    g, _ = margins(pr)
    gmin = g.min(1).values
    gam0 = (gmin / R2 ** (nu / 2)).numpy()
    lr = torch.full((M,), lr0, dtype=torch.float64)
    keys = list(pr)
    for step in range(steps):
        with torch.enable_grad():
            q = {k: v.clone().requires_grad_() for k, v in pr.items()}
            gq, _ = margins(q)
            mn = gq.min(1, keepdim=True).values
            q10 = gq.kthvalue(max(1, gq.shape[1] // 10), dim=1,
                              keepdim=True).values
            beta = 8.0 / (q10 - mn + 1e-30)
            sm = -(torch.logsumexp(-beta * (gq - mn), 1)) / beta.squeeze(1)
            grads = torch.autograd.grad(sm.sum(), [q[k] for k in keys])
        gn2 = sum((gr ** 2).flatten(1).sum(1) for gr in grads)
        sc = (lr * (R2 / gn2.clamp(min=1e-300)).sqrt())
        prop = {k: pr[k] + sc.view(-1, *[1] * (pr[k].dim() - 1)) * gr
                for k, gr in zip(keys, grads)}
        fac = (R2 / sq_norm(prop)).sqrt()
        prop = {k: v * fac.view(-1, *[1] * (v.dim() - 1))
                for k, v in prop.items()}
        gp, _ = margins(prop)
        newmin = gp.min(1).values
        acc = newmin > gmin
        for k in keys:
            am = acc.view(-1, *[1] * (pr[k].dim() - 1))
            pr[k] = torch.where(am, prop[k], pr[k])
        gmin = torch.where(acc, newmin, gmin)
        lr = torch.where(acc, lr * 1.15, lr * 0.5).clamp(1e-9, 0.3)
        if (step + 1) % 100 == 0:
            gam = gmin / R2 ** (nu / 2)
            print(f"  {label} step {step+1}: median gamma "
                  f"{np.median(gam.numpy()):.3e} acc {acc.float().mean():.2f}",
                  flush=True)
    _, L = margins(pr)
    keff, Kg = spectrum(L, p)
    gam = (gmin / R2 ** (nu / 2)).numpy()
    return gam0, gam, keff, Kg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--n", type=int, default=8)
    args = ap.parse_args()
    R = ROOT / "runs_nudial"
    groups = {
        "G-AdamW relu4": sorted((R / "G" / "relu4_m512_p113_f1_train").glob("wd1_s*"))[:args.n],
        "D-ascent relu4": sorted((R / "D" / "relu4_m512_p113_margin").glob("mm_s*"))[:args.n],
        "G-AdamW relu6": sorted((R / "G" / "relu6_m512_p113_f1_train").glob("wd1_s*"))[:args.n],
        "D-ascent relu6": sorted((R / "D" / "relu6_m512_p113_margin").glob("mm_s*"))[:args.n],
    }
    results = {}
    for name, runs in groups.items():
        runs = [r for r in runs if (r / "final.safetensors").exists()]
        if not runs:
            print(f"{name}: no runs found"); continue
        cfgs, pr = load_group(runs)
        nu = cfgs[0].degree
        print(f"{name}: n={len(runs)} nu={nu}")
        g0, g1, keff, Kg = polish(pr, cfgs[0], nu, args.steps, label=name)
        results[name] = dict(gamma_before=g0.tolist(), gamma_after=g1.tolist(),
                             keff=keff.tolist(), K_gap=Kg)
        print(f"  {name}: median gamma {np.median(g0):.3e} -> {np.median(g1):.3e} "
              f"(max after {g1.max():.3e}) | K_gap {Kg} | k_eff med {np.median(keff):.2f}")
    (ROOT / "notes" / "margin_polish.json").write_text(json.dumps(results, indent=1))
    print("wrote notes/margin_polish.json")


if __name__ == "__main__":
    main()
