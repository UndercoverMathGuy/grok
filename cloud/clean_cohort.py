"""Clean cohort 2026-09-09: fresh p=113 transformer sweeps + S5, one script.

Star design around the record recipe (1L ReLU transformer, d_model 128,
4 heads, d_mlp 512, frac 0.3, wd 1, lr 1e-3, AdamW MLX semantics):
  base      8 seeds
  wd        0.4 / 2.5            3 seeds each   (same batch as base, ragged)
  frac      0.5 / 0.7            3 seeds each   (same batch as base, ragged)
  degree    n_pre_lin 1 | pre 1 + post 1 | pre 2 + post 2   6 / 6 / 4 seeds
  width     d_mlp 128 / 2048     3 seeds each
  opt       gd  lr 1.0 wd 1e-3 | lr 0.3 wd 3e-4           3 seeds each
            lion lr 1e-4 wd 1 | lr 3e-4 wd 1              3 seeds each
  S5        fn_name s5, p 120, frac 0.7, wd 1 / 2, 50k epochs, 4 seeds each
Checkpoints every 2500 epochs (save_every) so the ablation count can be run
at every stage. Runs are skipped if spectra.npz exists (resume-safe).

Usage (pod):  RUNS_DIR=/workspace/runs_clean python -u cloud/clean_cohort.py --part p113
              RUNS_DIR=/workspace/runs_clean python -u cloud/clean_cohort.py --part s5
Local smoke:  python cloud/clean_cohort.py --smoke
"""
import argparse, os, sys, time
from pathlib import Path

import numpy as np, torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_semifinal_torch import Config, train_batched, RUNS

SEEDS = [31831, 47904, 69089, 81412, 19078, 32610, 68549, 85697]   # the record's init seeds
DATA_SEED = 3148


def cfg(**kw):
    base = dict(p=113, fn_name="add", frac_train=0.3, data_seed=DATA_SEED, num_layers=1,
                d_model=128, num_heads=4, d_mlp=512, act_type="ReLU", lr=1e-3,
                weight_decay=1.0, num_epochs=25_000, save_every=2500, loss_dtype="f32stable")
    base.update(kw)
    return Config(**base)


def name(c, tag):
    parts = [tag, f"wd{c.weight_decay:g}", f"f{c.frac_train:g}"]
    if c.opt != "adamw": parts.append(f"{c.opt}_lr{c.lr:g}")
    if c.adam_eps != 1e-8: parts.append(f"eps{c.adam_eps:g}_lr{c.lr:g}")
    if c.n_pre_lin or c.n_post_lin: parts.append(f"L{c.n_pre_lin}{c.n_post_lin}")
    if c.d_mlp != 512: parts.append(f"m{c.d_mlp}")
    if c.init_scale != 1.0: parts.append(f"a{c.init_scale:g}")
    if c.clip: parts.append(f"clip{c.clip:g}")
    return "_".join(parts) + f"/seed{c.init_seed}"


def p113_jobs():
    S = SEEDS
    jobs = []
    # one ragged batch: base + wd + frac (all shape-bearing fields agree)
    b = [cfg(init_seed=s) for s in S[:8]]
    b += [cfg(init_seed=s, weight_decay=wd) for wd in (0.4, 2.5) for s in S[:3]]
    b += [cfg(init_seed=s, frac_train=fr) for fr in (0.5, 0.7) for s in S[:3]]
    jobs.append(("p113/base", b))
    for pre, post, n in ((1, 0, 6), (1, 1, 6), (2, 2, 4)):
        jobs.append((f"p113/degree", [cfg(init_seed=s, n_pre_lin=pre, n_post_lin=post) for s in S[:n]]))
    for m in (128, 2048):
        jobs.append(("p113/width", [cfg(init_seed=s, d_mlp=m) for s in S[:3]]))
    jobs.append(("p113/opt", [cfg(init_seed=s, opt="gd", lr=lr, weight_decay=wd, warmup_steps=1)
                              for lr, wd in ((1.0, 1e-3), (0.3, 3e-4)) for s in S[:3]]))
    jobs.append(("p113/opt", [cfg(init_seed=s, opt="lion", lr=lr, weight_decay=1.0, betas=(0.9, 0.99))
                              for lr in (1e-4, 3e-4) for s in S[:3]]))
    return jobs


def opt_jobs():
    """l2-geometry optimizers on the base transformer cell (frac 0.3), 3 seeds each, 50k epochs.
    Pass 2 (09-09): scaled to the plain-GD lr 0.3 cell (the only GD transformer that moved)."""
    S = SEEDS[:3]; jobs = []; E = dict(num_epochs=50_000, save_every=5000)
    jobs.append(("p113/opt3", [cfg(init_seed=s, opt="gd", lr=0.3, weight_decay=1e-3, warmup_steps=1, **E) for s in S]))
    jobs.append(("p113/opt3", [cfg(init_seed=s, opt="gdm", lr=0.03, weight_decay=1e-3, warmup_steps=1, **E) for s in S]
                 + [cfg(init_seed=s, opt="gdm", lr=0.01, weight_decay=1e-3, warmup_steps=1, **E) for s in S]))
    jobs.append(("p113/opt3", [cfg(init_seed=s, opt="ngd", lr=0.03, weight_decay=wd, warmup_steps=1, **E) for wd in (0.03, 0.1) for s in S]))
    jobs.append(("p113/opt3", [cfg(init_seed=s, adam_eps=1.0, lr=0.3, weight_decay=1e-3, **E) for s in S]
                 + [cfg(init_seed=s, adam_eps=1.0, lr=0.1, weight_decay=3e-3, **E) for s in S]))
    return jobs


def l2_jobs():
    """l2-geometry record cohort: momentum GD (mu 0.9, coupled l2 1e-3) at two lrs per cell, 50k epochs."""
    S = SEEDS; E = dict(num_epochs=50_000, save_every=5000, warmup_steps=1, opt="gdm", weight_decay=1e-3)
    G2 = lambda ss, **kw: ([cfg(init_seed=s, lr=0.03, **E, **kw) for s in ss] + [cfg(init_seed=s, lr=0.01, **E, **kw) for s in ss])
    jobs = []
    b = G2(S[:6]) + [cfg(init_seed=s, lr=0.03, **dict(E, weight_decay=wd)) for wd in (3e-4, 3e-3) for s in S[:3]] \
        + [cfg(init_seed=s, lr=0.03, frac_train=fr, **E) for fr in (0.5, 0.7) for s in S[:3]]
    jobs.append(("l2/base", b))
    for pre, post in ((1, 0), (1, 1), (2, 2)):
        jobs.append(("l2/degree", G2(S[:4], n_pre_lin=pre, n_post_lin=post)))
    for m in (128, 2048):
        jobs.append(("l2/width", G2(S[:3], d_mlp=m)))
    jobs.append(("l2/witness", [cfg(init_seed=s, adam_eps=1.0, lr=0.3, weight_decay=1e-3, num_epochs=50_000, save_every=5000) for s in S[:4]]))
    return jobs


def s5_l2_jobs():
    E = dict(num_epochs=100_000, save_every=10_000, warmup_steps=1, opt="gdm", weight_decay=1e-3, fn_name="s5", p=120, frac_train=0.7)
    return [("s5l2", [cfg(init_seed=s, lr=lr, **E) for lr in (0.03, 0.01) for s in SEEDS[:4]])]


def lyu_jobs():
    """Lyu-regime pilot: large init, small coupled l2, SGD-momentum + clip; the cells that died under wd 1e-3 at init 1."""
    E = dict(num_epochs=30_000, save_every=5000, warmup_steps=1, opt="gdm", lr=0.03, clip=1.0)
    jobs = []
    for a in (1.5, 2.5):
        b = [cfg(init_seed=s, init_scale=a, weight_decay=wd, frac_train=fr, **E)
             for wd in (1e-3, 1e-4) for fr in (0.3, 0.5, 0.7) for s in SEEDS[:3]]
        jobs.append(("lyu/base", b))
        jobs.append(("lyu/degree", [cfg(init_seed=s, init_scale=a, weight_decay=wd, n_pre_lin=1, n_post_lin=1, **E)
                                    for wd in (1e-3, 1e-4) for s in SEEDS[:3]]))
    # faithful Lyu: plain GD, no momentum, clipped
    jobs.append(("lyu/gd", [cfg(init_seed=s, init_scale=a, weight_decay=1e-3, opt="gd", lr=0.3, clip=1.0, num_epochs=30_000, save_every=5000, warmup_steps=1)
                            for a in (1.5, 2.5) for s in SEEDS[:3]]))
    return jobs


def s5_lyu_jobs():
    E = dict(num_epochs=60_000, save_every=10_000, warmup_steps=1, opt="gdm", lr=0.03, clip=1.0, fn_name="s5", p=120, frac_train=0.7)
    return [("s5lyu", [cfg(init_seed=s, init_scale=a, weight_decay=wd, **E) for a in (1.5, 2.5) for wd in (1e-3, 1e-4) for s in SEEDS[:2]])]


def s5_jobs():
    return [("s5", [cfg(fn_name="s5", p=120, frac_train=0.7, weight_decay=wd, num_epochs=50_000,
                        save_every=5000, init_seed=s) for wd in (1.0, 2.0) for s in SEEDS[:4]])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["p113", "s5", "opt", "l2", "s5l2", "lyu", "s5lyu", "all"], default="all")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--compile", default="reduce-overhead")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    jobs = ((p113_jobs() if args.part in ("p113", "all") else []) + (opt_jobs() if args.part in ("opt", "all") else [])
            + (l2_jobs() if args.part == "l2" else []) + (s5_l2_jobs() if args.part == "s5l2" else [])
            + (lyu_jobs() if args.part == "lyu" else []) + (s5_lyu_jobs() if args.part == "s5lyu" else [])
            + (s5_jobs() if args.part in ("s5", "all") else []))
    if args.smoke:
        # two tiny batches on CPU: a degree-dial AdamW batch and a lion batch, p=29
        jobs = [("smoke/deg", [cfg(p=29, init_seed=s, n_pre_lin=1, n_post_lin=1, num_epochs=40, save_every=20) for s in SEEDS[:2]]),
                ("smoke/lion", [cfg(p=29, init_seed=s, opt="lion", lr=1e-4, betas=(0.9, 0.99), num_epochs=40, save_every=20) for s in SEEDS[:2]]),
                ("smoke/gd", [cfg(p=29, init_seed=s, opt="gd", lr=0.3, weight_decay=1e-3, warmup_steps=1, num_epochs=40, save_every=20) for s in SEEDS[:2]]),
                ("smoke/gdm", [cfg(p=29, init_seed=s, opt="gdm", lr=0.1, weight_decay=1e-3, warmup_steps=1, num_epochs=40, save_every=20) for s in SEEDS[:2]]),
                ("smoke/ngd", [cfg(p=29, init_seed=s, opt="ngd", lr=0.1, weight_decay=1e-3, warmup_steps=1, num_epochs=40, save_every=20) for s in SEEDS[:2]]),
                ("smoke/eps", [cfg(p=29, init_seed=s, adam_eps=1.0, lr=0.3, weight_decay=3e-3, num_epochs=40, save_every=20) for s in SEEDS[:2]]),
                ("smoke/s5", [cfg(fn_name="s5", p=120, init_seed=s, num_epochs=20, save_every=20) for s in SEEDS[:1]])]
        args.compile = "off"
    total = sum(len(b) for _, b in jobs)
    print(f"{len(jobs)} batches, {total} runs, RUNS={RUNS}", flush=True)
    for tag, cfgs in jobs:
        dirs = [RUNS / name(c, tag) for c in cfgs]
        keep = [(c, d) for c, d in zip(cfgs, dirs) if not (d / "spectra.npz").exists()]
        print(f"[{tag}] {len(keep)}/{len(cfgs)} runs to train: "
              + ", ".join(str(d.relative_to(RUNS)) for _, d in keep[:4]) + (" ..." if len(keep) > 4 else ""), flush=True)
        if args.dry_run or not keep:
            continue
        t = time.time()
        train_batched([c for c, _ in keep], [d for _, d in keep], log_every=500 if not args.smoke else 10,
                      spectra_every=100 if not args.smoke else 10, compile_mode=args.compile)
        print(f"[{tag}] done in {time.time() - t:.0f}s", flush=True)


if __name__ == "__main__":
    main()
