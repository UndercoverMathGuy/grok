"""heracles cohorts. Each job = (tag, [Config...]) trained as one batch (lr / wd / clip / q / maxnorm / seed
may differ per run; shape and code path must agree). Resume-safe: runs with spectra.npz are skipped.

Cohort R (recipe sweep, 2026-09-11): relu nu=4 (layout (1,1)), m=512, p=113, frac 0.5, 8 seeds, 20k epochs.
  R_gd     full-batch GD + l2:  lr {0.3, 1, 3} x wd {1e-4, 3e-4, 1e-3, 3e-3} x clip {0, 1}     192 runs
  R_gdm    heavy-ball (mu .9):   lr {0.03, 0.1, 0.3} x wd {3e-4, 1e-3, 3e-3}, clip 1            72
  R_sgd    minibatch 1024:       lr {0.3, 1, 3} x wd {3e-4, 1e-3, 3e-3}                          72
  R_lq3    l3 geometry + l3 pen: lr {0.01, 0.03, 0.1} x wd {0.03, 0.3, 3}                        72
  R_lq4    l4 geometry + l4 pen: same grid                                                        72
  R_sign   signSGD + box:        lr {1e-4, 3e-4, 1e-3} x maxnorm {0.05, 0.1, 0.2}                72
  R_signm  sign-momentum + box:  same grid                                                        72
  R_signl2 MIXED control (sign + l2 shrink, Lion-style wd): lr {1e-4, 3e-4} x wd {0.1, 1}        32

Usage (pod):   RUNS_DIR=/workspace/heracles_runs python -u heracles/jobs.py --cohort R
Local smoke:   uv run python heracles/jobs.py --smoke
"""
import argparse, os, sys, time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train import Config, train_batched, draw_seeds, RUNS

SEEDS = draw_seeds(16)
DATA_SEED = 911


def cfg(**kw):
    base = dict(p=113, nu=4, m=512, d=128, r=128, frac_train=0.5, data_seed=DATA_SEED, num_epochs=20_000, spectra_every=200)
    base.update(kw)
    return Config(**base)


def name(c, tag):
    parts = [tag, c.opt]
    if c.opt == "lq": parts.append(f"q{c.q:g}")
    parts += [f"lr{c.lr:g}"]
    parts.append(f"box{c.maxnorm:g}" if c.reg == "box" else f"wd{c.weight_decay:g}")
    if c.reg == "l2" and c.opt in ("sign", "signm"): parts.append("MIXED")
    if c.clip: parts.append(f"clip{c.clip:g}")
    if c.batch_size: parts.append(f"B{c.batch_size}")
    if c.nu != 4: parts.append(f"nu{c.nu}")
    if c.p != 113: parts.append(f"p{c.p}")
    if c.frac_train != 0.5: parts.append(f"f{c.frac_train:g}")
    if c.m != 512: parts.append(f"m{c.m}")
    return "/".join([parts[0], "_".join(parts[1:]), f"seed{c.init_seed}"])


def cohort_R():
    S = SEEDS[:8]; J = []
    gd = [cfg(opt="gd", lr=lr, weight_decay=wd, clip=cl, init_seed=s)
          for lr in (0.3, 1.0, 3.0) for wd in (1e-4, 3e-4, 1e-3, 3e-3) for cl in (0.0, 1.0) for s in S]
    J += [("R", gd)]
    gdm = [cfg(opt="gdm", lr=lr, weight_decay=wd, clip=1.0, momentum=0.9, init_seed=s)
           for lr in (0.03, 0.1, 0.3) for wd in (3e-4, 1e-3, 3e-3) for s in S]
    J += [("R", gdm)]
    sgd = [cfg(opt="sgd", batch_size=1024, lr=lr, weight_decay=wd, init_seed=s)
           for lr in (0.3, 1.0, 3.0) for wd in (3e-4, 1e-3, 3e-3) for s in S]
    J += [("R", sgd)]
    for q in (3.0, 4.0):
        lq = [cfg(opt="lq", q=q, reg="lq", lr=lr, weight_decay=wd, init_seed=s)
              for lr in (0.01, 0.03, 0.1) for wd in (0.03, 0.3, 3.0) for s in S]
        J += [("R", lq)]
    for opt in ("sign", "signm"):
        sg = [cfg(opt=opt, q=float("inf"), reg="box", lr=lr, maxnorm=c, momentum=0.9, init_seed=s)
              for lr in (1e-4, 3e-4, 1e-3) for c in (0.05, 0.1, 0.2) for s in S]
        J += [("R", sg)]
    mixed = [cfg(opt="signm", q=float("inf"), reg="l2", lr=lr, weight_decay=wd, momentum=0.9, init_seed=s)
             for lr in (1e-4, 3e-4) for wd in (0.1, 1.0) for s in S]
    J += [("R", mixed)]
    return J


def cohort_smoke():
    S = SEEDS[:2]; E = dict(p=29, m=64, d=16, r=16, num_epochs=30, spectra_every=10)
    return [("smoke", [cfg(opt="gd", lr=1.0, weight_decay=1e-3, clip=cl, init_seed=s, **E) for cl in (0.0, 1.0) for s in S]),
            ("smoke", [cfg(opt="gdm", lr=0.1, weight_decay=1e-3, init_seed=s, **E) for s in S]),
            ("smoke", [cfg(opt="sgd", batch_size=64, lr=1.0, weight_decay=1e-3, init_seed=s, **E) for s in S]),
            ("smoke", [cfg(opt="lq", q=q, lr=0.03, weight_decay=0.3, init_seed=s, **E) for q in (3.0, 4.0) for s in S]),
            ("smoke", [cfg(opt="sign", q=float("inf"), reg="box", lr=1e-3, maxnorm=0.2, init_seed=s, **E) for s in S]),
            ("smoke", [cfg(opt="signm", q=float("inf"), reg="l2", lr=1e-3, weight_decay=0.1, init_seed=s, **E) for s in S]),
            ("smoke", [cfg(opt="gd", nu=6, lr=1.0, weight_decay=1e-3, init_seed=s, **E) for s in S])]


def cohort_R2():
    """Follow-up to R (same day): l_q cells re-gridded 10-100x lower in wd after the R grid collapsed
    weights to zero (penalty-dominated early phase under the |v|^{1/(q-1)} map). q=3 and q=4 share one
    batch (q is per-run); two batches of 72 (144 OOMs the full-grid snapshot on a 24 GB card)."""
    S = SEEDS[:8]
    lq = [cfg(opt="lq", q=q, reg="lq", lr=lr, weight_decay=wd, init_seed=s)
          for q in (3.0, 4.0) for lr in (0.003, 0.01, 0.03) for wd in (1e-4, 1e-3, 1e-2) for s in S]
    return [("R", lq)]


# ---------------------------------------------------------------- 2026-09-12: pod died (credit), R2 + raw R lost.
REC = dict(opt="gd", lr=3.0, clip=1.0)     # record q=2 recipe from RECIPES.md (wd 3e-4; wd passed per cohort)


def cohort_Rkeep():
    """Provenance slice of R: the 14 cells behind RECIPES.md (scores/Rsides), retrained so raw trajectories
    exist behind every number in the recipe table. Same names as R (tag R). 112 runs."""
    S = SEEDS[:8]; J = []
    gd = [cfg(opt="gd", lr=lr, weight_decay=wd, clip=cl, init_seed=s)
          for lr, wd, cl in ((0.3, 1e-3, 0.0), (1.0, 3e-4, 0.0), (1.0, 1e-3, 1.0), (3.0, 3e-4, 1.0), (3.0, 1e-3, 1.0)) for s in S]
    J.append(("R", gd))
    J.append(("R", [cfg(opt="gdm", lr=lr, weight_decay=wd, clip=1.0, momentum=0.9, init_seed=s) for lr, wd in ((0.03, 1e-3), (0.3, 3e-4)) for s in S]))
    J.append(("R", [cfg(opt="lq", q=4.0, reg="lq", lr=lr, weight_decay=0.03, init_seed=s) for lr in (0.01, 0.1) for s in S]))
    J.append(("R", [cfg(opt="sgd", batch_size=1024, lr=1.0, weight_decay=3e-4, init_seed=s) for s in S]))
    J.append(("R", [cfg(opt="sign", q=float("inf"), reg="box", lr=1e-4, maxnorm=c, momentum=0.9, init_seed=s) for c in (0.1, 0.2) for s in S]))
    J.append(("R", [cfg(opt="signm", q=float("inf"), reg="box", lr=1e-4, maxnorm=0.2, momentum=0.9, init_seed=s) for s in S]))
    J.append(("R", [cfg(opt="signm", q=float("inf"), reg="l2", lr=3e-4, weight_decay=1.0, momentum=0.9, init_seed=s) for s in S]))
    return J


def cohort_H1():
    """p x nu grid, q = 2. Record recipe (lr 3, wd 3e-4, clip 1) is registered as primary; lr 1 (same wd, clip)
    rides in the same batch as the fallback in case lr 3 fails to grok >= 7/8 at another (p, nu). 20 cells x 16."""
    S = SEEDS[:8]
    return [("H1", [cfg(p=p, nu=nu, opt="gd", lr=lr, weight_decay=3e-4, clip=1.0, init_seed=s) for lr in (1.0, 3.0) for s in S])
            for p in (59, 71, 97, 113, 127) for nu in (3, 4, 5, 6)]


LONG = dict(num_epochs=60_000, spectra_every=600, ckpt_at=(20_000,))


def cohort_H2():
    """wd x frac at p 113, nu 4, record recipe: wd {1e-3, 3e-4} x frac {0.3, 0.7} x 8 seeds, 60k epochs with a
    20k checkpoint (so wd 1e-3 is read at 20k and 3e-4 at 60k, matched to H3's 1/(lr wd) budget rule).
    frac 0.5 is read from H3's 20k / 60k checkpoints. 2 batches x 16. (nu 6 dropped 09-12: throughput/budget.)"""
    S = SEEDS[:8]
    return [("H2", [cfg(nu=4, frac_train=f, weight_decay=wd, init_seed=s, **REC, **LONG) for wd in (1e-3, 3e-4) for s in S])
            for f in (0.3, 0.7)]


def cohort_H3():
    """The Wei limit: record recipe, 8 seeds, checkpoints at 20k / 60k / 200k so every wd is read at every matched
    budget (20k for 1e-3, 60k for 3e-4, 200k for 1e-4, 600k for 3e-5). nu 4: wd {1e-3, 3e-4, 1e-4, 3e-5}, 600k epochs
    (32 runs). nu 6: wd {1e-3, 3e-4, 1e-4}, 200k epochs (24 runs). The 20k / 60k checkpoints of the wd 1e-3 / 3e-4
    arms double as H2's frac 0.5 and H5's gd. (09-12: 16 seeds -> 8 and nu 6 capped at 200k for throughput/budget.)"""
    S = SEEDS[:8]
    E4 = dict(num_epochs=600_000, spectra_every=2000, ckpt_at=(20_000, 60_000, 200_000))
    E6 = dict(num_epochs=200_000, spectra_every=2000, ckpt_at=(20_000, 60_000))
    return [("H3", [cfg(nu=4, weight_decay=wd, init_seed=s, **REC, **E4) for wd in (1e-3, 3e-4, 1e-4, 3e-5) for s in S]),
            ("H3", [cfg(nu=6, weight_decay=wd, init_seed=s, **REC, **E6) for wd in (1e-3, 3e-4, 1e-4) for s in S])]


def cohort_H5():
    """Optimizer robustness at 60k: gdm and sgd record recipes at wd 3e-4, nu {4, 6}, 8 seeds (gd comes from H3's 60k ckpt)."""
    S = SEEDS[:8]; J = []
    for nu in (4, 6):
        J.append(("H5", [cfg(nu=nu, opt="gdm", lr=0.3, weight_decay=3e-4, clip=1.0, momentum=0.9, init_seed=s, **LONG) for s in S]))
        J.append(("H5", [cfg(nu=nu, opt="sgd", batch_size=1024, lr=1.0, weight_decay=3e-4, init_seed=s, **LONG) for s in S]))
    return J


def cohort_H():
    """Stage 1 of the H launch, cheapest first: R2 (l3 recipe), Rkeep, H1, H2, H3 (H5 deferred 09-12). H4 (geometry dial) launches
    separately once R2 is scored and the l3 / l6 recipes are chosen."""
    return cohort_R2() + cohort_Rkeep() + cohort_H1() + cohort_H2() + cohort_H3()   # H5 deferred (budget)


COHORTS = {"R": cohort_R, "R2": cohort_R2, "smoke": cohort_smoke, "Rkeep": cohort_Rkeep, "H1": cohort_H1, "H2": cohort_H2,
           "H3": cohort_H3, "H5": cohort_H5, "H": cohort_H}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", default="R")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--compile", default="reduce-overhead")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--log-every", type=int, default=500)
    ap.add_argument("--max-batch", type=int, default=144, help="split larger batches into chunks of at most this many runs")
    args = ap.parse_args()
    torch.set_float32_matmul_precision("highest")
    if args.smoke:
        args.cohort, args.compile, args.log_every = "smoke", "off", 10
    jobs = [(t, b[i:i + args.max_batch]) for t, b in COHORTS[args.cohort]() for i in range(0, len(b), args.max_batch)]
    total = sum(len(b) for _, b in jobs)
    print(f"cohort {args.cohort}: {len(jobs)} batches, {total} runs, RUNS={RUNS}", flush=True)
    for tag, cfgs in jobs:
        dirs = [RUNS / name(c, tag) for c in cfgs]
        keep = [(c, d) for c, d in zip(cfgs, dirs) if not (d / "spectra.npz").exists()]
        print(f"[{tag}/{cfgs[0].opt}] {len(keep)}/{len(cfgs)} runs to train: " + ", ".join(str(d.relative_to(RUNS)) for _, d in keep[:3]) + (" ..." if len(keep) > 3 else ""), flush=True)
        if args.dry_run or not keep:
            continue
        t = time.time()
        train_batched([c for c, _ in keep], [d for _, d in keep], compile_mode=args.compile, log_every=args.log_every)
        print(f"[{tag}/{cfgs[0].opt}] done in {time.time() - t:.0f}s", flush=True)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
