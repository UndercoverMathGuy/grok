"""Linearized-twin cohort (prereg: lazy/README.md).

Arms: real x8 (ordinary training, in-cohort reference), lin-wd1 x8 and
lin-wd0 x8 (tangent model at the base's epoch-0 init, trained with the
exact MLX-AdamW arithmetic of the real trainer; gradients are exact via
one jvp + one vjp per step).

Run dirs: runs_lazy/{real,lin}_wd{wd:g}/<base>/

On the pod:  python3 lazy/train_lazy.py
Smoke (CPU, ~3 min): python3 lazy/train_lazy.py --smoke
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("RUNS_DIR", str(ROOT / "runs_lazy"))
sys.path.insert(0, str(ROOT / "cloud"))
sys.path.insert(0, str(ROOT / "mirror"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from safetensors.torch import load_file, save_file  # noqa: E402

from compile_torch import natural_bases  # noqa: E402
from train_semifinal_torch import (RUNS, Config, batched_final_logits,  # noqa: E402
                                   fourier_basis, make_dataset,
                                   per_ex_stable_f32, train_batched,
                                   train_test_split)


def cfg_from(base_cfg, **over):
    d = {k: v for k, v in base_cfg.items() if k in Config.__dataclass_fields__}
    d.update(over)
    return Config(**d)


def lazy_batch(cfgs, run_dirs, e0_paths, device, spectra_every=5, order=1):
    """Train the TRUNCATED-TAYLOR MODEL at each base's epoch-0 init.

    order=1: f_lin(theta) = f(theta0) + J delta         (tangent/kernel)
    order=2: f_quad = f_lin + 1/2 delta^T H delta       (bilinear regime)

    The order-2 directional term and J delta come from ONE nested jvp:
    jvp of g1(P) = J(P) delta at theta0 along delta returns
    (J(theta0) delta, delta^T H delta). Gradients are exact:
    dL/dtheta = J^T g (cached vjp at theta0) + [order 2] H[delta,.]^T g
    (a per-step vjp of g1). The parameter update is line-for-line the
    trainer's MLX-AdamW.
    """
    M, c0 = len(cfgs), cfgs[0]
    for c in cfgs:
        assert (c.p, c.frac_train, c.d_mlp) == (c0.p, c0.frac_train, c0.d_mlp)
    P = c0.p
    tokens_np, labels_np = make_dataset(c0)
    masks = [train_test_split(c) for c in cfgs]
    tr_tok = torch.from_numpy(np.stack([tokens_np[m] for m in masks])).to(device)
    tr_lab = torch.from_numpy(np.stack([labels_np[m] for m in masks])).to(device)
    grid = torch.from_numpy(tokens_np).to(device)
    lab_t = torch.from_numpy(labels_np).to(device)
    mask_t = torch.from_numpy(np.stack(masks)).to(device)

    from train_semifinal_torch import PARAM_KEYS
    params0 = {k: torch.stack([load_file(str(p))[k] for p in e0_paths])
               .to(device) for k in PARAM_KEYS}
    theta = {k: v.clone() for k, v in params0.items()}
    m_st = {k: torch.zeros_like(v) for k, v in params0.items()}
    v_st = {k: torch.zeros_like(v) for k, v in params0.items()}
    b1, b2 = c0.betas
    eps = c0.adam_eps
    lr_base = torch.tensor([c.lr for c in cfgs], device=device)
    wd_v = torch.tensor([c.weight_decay for c in cfgs], device=device)
    bshape = {k: (-1,) + (1,) * (v.dim() - 1) for k, v in params0.items()}

    f_tr = lambda p_: batched_final_logits(p_, tr_tok, c0)
    f_grid = lambda p_: batched_final_logits(p_, grid, c0)
    # one cached vjp at theta0: J^T g for any g, plus f(theta0) on train
    out0_tr, vjp_fn = torch.func.vjp(f_tr, params0)
    with torch.no_grad():
        out0_grid = f_grid(params0)

    # spectral instrument (identical math to the trainer's snapshot)
    basis = fourier_basis(P)
    n_ = P // 2
    bc, bs = basis[1::2], basis[2::2]
    u_cos = torch.from_numpy((np.einsum("np,nq->npq", bc, bc)
                              - np.einsum("np,nq->npq", bs, bs))
                             .reshape(n_, -1) / np.sqrt(2)).to(device)
    u_sin = torch.from_numpy((np.einsum("np,nq->npq", bs, bc)
                              + np.einsum("np,nq->npq", bc, bs))
                             .reshape(n_, -1) / np.sqrt(2)).to(device)
    b_cos = torch.from_numpy(bc).to(device)
    b_sin = torch.from_numpy(bs).to(device)
    spectra = [{"epochs": [], "coeffs": [], "energy": [],
                "train_acc": [], "test_acc": []} for _ in range(M)]

    def taylor_logits(f, out0, delta):
        if order == 1:
            _, jd = torch.func.jvp(f, (params0,), (delta,))
            return out0 + jd
        g1 = lambda p_: torch.func.jvp(f, (p_,), (delta,))[1]
        jd, hdd = torch.func.jvp(g1, (params0,), (delta,))
        return out0 + jd + 0.5 * hdd

    @torch.no_grad()
    def snapshot(epoch, delta):
        L = taylor_logits(f_grid, out0_grid, delta)[..., :P]
        acc = L.argmax(-1) == lab_t
        Ld = L.double()
        c = torch.einsum("fp,mpo->mfo", u_cos, Ld)
        s = torch.einsum("fp,mpo->mfo", u_sin, Ld)
        energy = (c ** 2).sum(-1) + (s ** 2).sum(-1)
        coeffs = ((c * b_cos).sum(-1) + (s * b_sin).sum(-1)) / math.sqrt(2)
        tr_a = (acc & mask_t).sum(-1) / mask_t.sum(-1)
        te_a = (acc & ~mask_t).sum(-1) / (~mask_t).sum(-1)
        for i in range(M):
            spectra[i]["epochs"].append(epoch)
            spectra[i]["coeffs"].append(coeffs[i].cpu().numpy())
            spectra[i]["energy"].append(energy[i].cpu().numpy())
            spectra[i]["train_acc"].append(float(tr_a[i]))
            spectra[i]["test_acc"].append(float(te_a[i]))

    for cfg, rd in zip(cfgs, run_dirs):
        rd.mkdir(parents=True, exist_ok=True)
        cfg.save(rd / "config.json")

    t0 = time.time()
    for epoch in range(c0.num_epochs):
        delta = {k: theta[k] - params0[k] for k in theta}
        if spectra_every and epoch % spectra_every == 0:
            snapshot(epoch, delta)
        lr = lr_base * min(epoch / c0.warmup_steps, 1.0)
        leaf = taylor_logits(f_tr, out0_tr, delta).detach().requires_grad_(True)
        per = per_ex_stable_f32(leaf, tr_lab).mean(dim=1)
        (g,) = torch.autograd.grad(per.sum(), leaf)
        (gd,) = vjp_fn(g)
        if order == 2:
            g1 = lambda p_: torch.func.jvp(f_tr, (p_,), (delta,))[1]
            _, vjp2 = torch.func.vjp(g1, params0)
            (gd2,) = vjp2(g)
            gd = {k: gd[k] + gd2[k] for k in gd}
        with torch.no_grad():
            for k in theta:
                sh = bshape[k]
                lr_, wd_ = lr.view(sh), wd_v.view(sh)
                m_st[k].mul_(b1).add_(gd[k], alpha=1 - b1)
                v_st[k].mul_(b2).addcmul_(gd[k], gd[k], value=1 - b2)
                theta[k].mul_(1 - lr_ * wd_)
                theta[k].sub_(lr_ * m_st[k] / (v_st[k].sqrt() + eps))
        if (epoch + 1) % 500 == 0:
            eps_s = (epoch + 1) / (time.time() - t0)
            print(f"lazy epoch {epoch + 1:6d}  train {per.mean():.4e}  "
                  f"({eps_s:.1f} eps/s x {M})", flush=True)

    delta = {k: theta[k] - params0[k] for k in theta}
    snapshot(c0.num_epochs, delta)
    for i, rd in enumerate(run_dirs):
        np.savez(rd / "spectra.npz",
                 **{k: np.array(v) for k, v in spectra[i].items()})
        save_file({k: theta[k][i].detach().cpu().contiguous() for k in theta},
                  str(rd / "theta_final.safetensors"))
        z = spectra[i]
        print(f"LAZY RESULT {rd.name} wd{cfgs[i].weight_decay:g}: "
              f"train_acc {z['train_acc'][-1]:.3f} "
              f"test_acc {z['test_acc'][-1]:.3f}", flush=True)
    return out0_tr, vjp_fn


def smoke(bases, device):
    """Exactness: at theta = theta0 the tangent model's loss gradient IS the
    real model's loss gradient (mathematical identity) — assert it."""
    from train_semifinal_torch import PARAM_KEYS
    (bdir, bcfg, e0) = bases[0]
    c = cfg_from(bcfg, num_epochs=40, save_every=40)
    tokens_np, labels_np = make_dataset(c)
    m = train_test_split(c)
    tr_tok = torch.from_numpy(tokens_np[m][None]).to(device)
    tr_lab = torch.from_numpy(labels_np[m][None]).to(device)
    params0 = {k: load_file(str(e0))[k][None].to(device) for k in PARAM_KEYS}

    # real gradient at theta0
    pr = {k: v.clone().requires_grad_(True) for k, v in params0.items()}
    per = per_ex_stable_f32(batched_final_logits(pr, tr_tok, c), tr_lab)
    g_real = torch.autograd.grad(per.mean(dim=1).sum(), list(pr.values()))

    # lazy gradient at theta0 (delta = 0)
    f = lambda p_: batched_final_logits(p_, tr_tok, c)
    out0, vjp_fn = torch.func.vjp(f, params0)
    zero = {k: torch.zeros_like(v) for k, v in params0.items()}
    _, jd = torch.func.jvp(f, (params0,), (zero,))
    assert jd.abs().max().item() == 0.0, "jvp of zero tangent must be zero"
    leaf = (out0 + jd).detach().requires_grad_(True)
    per = per_ex_stable_f32(leaf, tr_lab).mean(dim=1)
    (g,) = torch.autograd.grad(per.sum(), leaf)
    (gd,) = vjp_fn(g)
    devs = [(gd[k] - gr).abs().max().item() / (gr.abs().max().item() + 1e-12)
            for k, gr in zip(PARAM_KEYS, g_real)]
    print("grad identity at theta0, max rel dev per param:",
          [f"{d:.1e}" for d in devs])
    assert max(devs) < 1e-4, "lazy gradient != real gradient at theta0"
    print("SMOKE OK: tangent-model gradient exact at theta0")

    # order-2 machinery gate, run on a SMOOTH surrogate (GeLU): a ReLU net is
    # piecewise linear, so no O(a^3) truncation bound exists for it (kink
    # crossings give O(a^1.5) mean residual — measured) and FD of J(P)d
    # differentiates across jump discontinuities. On GeLU the same code must
    # show textbook behavior; passing validates the nested-jvp path used for
    # the (a.e.-Hessian) ReLU quadratic model.
    cg = cfg_from(bcfg, act_type="GeLU")
    fg = lambda p_: batched_final_logits(p_, tr_tok, cg)
    p64 = {k: v.double() for k, v in params0.items()}
    out64 = fg(p64)
    torch.manual_seed(0)
    b64 = {k: torch.randn_like(v) for k, v in p64.items()}
    with torch.no_grad():
        res = {}
        for a in (1.0, 0.5):
            da = {k: a * 0.01 * v for k, v in b64.items()}
            g1 = lambda p_: torch.func.jvp(fg, (p_,), (da,))[1]
            jd, hdd = torch.func.jvp(g1, (p64,), (da,))
            res[a] = (fg({k: p64[k] + da[k] for k in p64})
                      - (out64 + jd + 0.5 * hdd)).abs().mean().item()
        ratio = res[1.0] / max(res[0.5], 1e-30)
        d64 = {k: 0.01 * v for k, v in b64.items()}
        g1 = lambda p_: torch.func.jvp(fg, (p_,), (d64,))[1]
        _, hdd = torch.func.jvp(g1, (p64,), (d64,))
        eps_ = 1e-4
        fd = (g1({k: p64[k] + eps_ * d64[k] for k in p64})
              - g1({k: p64[k] - eps_ * d64[k] for k in p64})) / (2 * eps_)
        fdrel = (fd - hdd).norm().item() / hdd.norm().item()
    print(f"order-2 gate (GeLU surrogate): truncation ratio {ratio:.2f} "
          f"(expect ~8), FD rel dev {fdrel:.1e}")
    assert 6.0 < ratio < 10.0 and fdrel < 1e-6, "quadratic machinery wrong"
    print("SMOKE OK: quadratic twin machinery exact (smooth surrogate)")

    # short lazy run: loss must fall
    cfgs = [cfg_from(bcfg, num_epochs=40, save_every=40, weight_decay=wd)
            for wd in (1.0, 0.0)]
    rds = [RUNS / f"smoke_lin_wd{c_.weight_decay:g}" / bdir.name
           for c_ in cfgs]
    lazy_batch(cfgs, rds, [e0, e0], device, spectra_every=20)
    z = np.load(rds[0] / "spectra.npz")
    assert z["train_acc"][-1] > z["train_acc"][0], "lazy training not learning"
    print("SMOKE OK: lazy run trains (train_acc "
          f"{z['train_acc'][0]:.3f} -> {z['train_acc'][-1]:.3f})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real-epochs", type=int, default=20_000)
    ap.add_argument("--lin-epochs", type=int, default=10_000)
    ap.add_argument("--compile", default="default",
                    choices=["default", "reduce-overhead", "off"])
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    bases = natural_bases()[:8]
    if args.smoke:
        assert bases, "no natural bases found"
        smoke(bases, device)
        return
    assert len(bases) == 8, f"found {len(bases)} bases, need 8"

    # real arm (in-cohort reference), ordinary trainer
    jobs = []
    for bdir, bcfg, e0 in bases:
        rd = RUNS / "real_wd1" / bdir.name
        if not (rd / "spectra.npz").exists():
            jobs.append((cfg_from(bcfg, num_epochs=args.real_epochs,
                                  save_every=args.real_epochs), rd, e0))
    if jobs:
        print(f"real arm: {len(jobs)} runs", flush=True)
        train_batched([c for c, _, _ in jobs], [rd for _, rd, _ in jobs],
                      init_from=[str(e) for _, _, e in jobs],
                      compile_mode=args.compile, spectra_every=5)

    # taylor arms: one lockstep batch per order, wd per-run within it
    for name, order in (("lin", 1), ("quad", 2)):
        cfgs, rds, e0s = [], [], []
        for wd in (1.0, 0.0):
            for bdir, bcfg, e0 in bases:
                rd = RUNS / f"{name}_wd{wd:g}" / bdir.name
                if (rd / "spectra.npz").exists():
                    continue
                cfgs.append(cfg_from(bcfg, num_epochs=args.lin_epochs,
                                     save_every=args.lin_epochs,
                                     weight_decay=wd))
                rds.append(rd)
                e0s.append(e0)
        if cfgs:
            print(f"{name} arms (order {order}): {len(cfgs)} runs", flush=True)
            lazy_batch(cfgs, rds, e0s, device, spectra_every=5, order=order)
    print("lazy cohort done", flush=True)


if __name__ == "__main__":
    main()
