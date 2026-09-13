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
    parts = [tag] + ([c.task] if c.task != "modadd" else []) + [c.opt]
    if c.support:
        from groups import get_group
        parts.append("sup-" + "-".join(get_group(c.task).irrep_names[i] for i in c.support))
    if c.opt == "lq": parts.append(f"q{c.q:g}")
    parts += [f"lr{c.lr:g}"]
    parts.append(f"box{c.maxnorm:g}" if c.reg == "box" else f"wd{c.weight_decay:g}")
    if c.reg == "l2" and c.opt in ("sign", "signm"): parts.append("MIXED")
    if c.clip: parts.append(f"clip{c.clip:g}")
    if c.batch_size: parts.append(f"B{c.batch_size}")
    if c.nu != 4: parts.append(f"nu{c.nu}")
    if c.p != 113 and c.task == "modadd": parts.append(f"p{c.p}")
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
    """The Wei limit: record recipe, 16 seeds (restored 09-12 07:15Z on the L40S: the card is underfilled at 32), checkpoints at 20k / 60k / 200k so every wd is read at every matched
    budget (20k for 1e-3, 60k for 3e-4, 200k for 1e-4, 600k for 3e-5). nu 4: wd {1e-3, 3e-4, 1e-4, 3e-5}, 600k epochs
    (64 runs). nu 6: wd {1e-3, 3e-4, 1e-4}, 200k epochs (48 runs). The 20k / 60k checkpoints of the wd 1e-3 / 3e-4
    arms double as H2's frac 0.5 and H5's gd. (09-12: 16 seeds -> 8 and nu 6 capped at 200k for throughput/budget.)"""
    S = SEEDS[:16]
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


LQREC = dict(lr=0.03, weight_decay=1e-4)     # R2 record for lq q = 3 and q = 4 at nu 4 (RECIPES.md)


def cohort_H4():
    """Geometry dial, p 113 (stage 2, after R2). nu 4: q3 / q4 are R2's record cells (reused, not retrained); q6 gets a
    3 x 3 (lr, wd) mini-grid; sign + box is R's record (Rkeep). nu 6: q3 / q4 / q6 mini-grids (3 x 3 each, sharing one
    batch since q is per-run) and a sign + box 2 x 2 grid; gd nu 6 comes from H3. Recipe selection inside each cell
    follows the RECIPES rule (>= 7/8 grok, finite, max median gamma_q, ties -> smaller wd), registered in PLAN.md."""
    S = SEEDS[:8]; J = []
    G = [(lr, wd) for lr in (0.01, 0.03, 0.1) for wd in (1e-5, 1e-4, 1e-3)]
    J.append(("H4", [cfg(nu=4, opt="lq", q=6.0, reg="lq", lr=lr, weight_decay=wd, init_seed=s) for lr, wd in G for s in S]))
    J.append(("H4", [cfg(nu=6, opt="lq", q=q, reg="lq", lr=lr, weight_decay=wd, init_seed=s) for q in (3.0, 4.0, 6.0) for lr, wd in G for s in S]))
    J.append(("H4", [cfg(nu=6, opt="sign", q=float("inf"), reg="box", lr=lr, maxnorm=c, momentum=0.9, init_seed=s)
                     for lr in (1e-4, 3e-4) for c in (0.2, 0.4) for s in S]))
    return J


def cohort_H2b():
    """H2's nu 6 half (deferred 09-12, restored in stage 2): wd {1e-3, 3e-4} x frac {0.3, 0.7} x 8 seeds, 60k, 20k ckpt."""
    S = SEEDS[:8]
    return [("H2", [cfg(nu=6, frac_train=f, weight_decay=wd, init_seed=s, **REC, **LONG) for wd in (1e-3, 3e-4) for s in S])
            for f in (0.3, 0.7)]


def cohort_S2():
    """Stage 2: H4 (geometry dial) + H5 (gdm / sgd at 60k) + H2 nu 6. ~384 runs."""
    return cohort_H4() + cohort_H5() + cohort_H2b()


def cohort_H():
    """Stage 1 of the H launch, cheapest first: R2 (l3 recipe), Rkeep, H1, H2, H3 (H5 deferred 09-12). H4 (geometry dial) launches
    separately once R2 is scored and the l3 / l6 recipes are chosen."""
    return cohort_R2() + cohort_Rkeep() + cohort_H1() + cohort_H2() + cohort_H3()   # H5 deferred (budget)


# ---------------------------------------------------------------- 2026-09-13: S5 programme (PLAN_S5.md)
def gcfg(task, **kw):
    """group-task cell: p = |G| (set by Config), m 512, d = r = 128, frac 0.8 (S5 memorises below 0.6), 20k epochs."""
    base = dict(task=task, nu=4, m=512, d=128, r=128, frac_train=0.8, data_seed=DATA_SEED, num_epochs=20_000, spectra_every=200)
    base.update(kw)
    return Config(**base)


S5REC = dict(opt="gd", lr=3.0, weight_decay=3e-4, clip=1.0)   # E1 winner goes here (mod-add record until E1 is scored)
S5_IRREPS = dict(sign=1, std=2, std_sign=3, a5=4, b5=5, six=6)


def cohort_SR():
    """E1: S5 recipe sweep. gd lr {1, 3} x wd {1e-3, 3e-4, 1e-4}, clip 1, frac {0.6, 0.8}, nu 4, 8 seeds, 20k. 96 runs, 2 batches."""
    S = SEEDS[:8]
    return [("SR", [gcfg("s5", opt="gd", lr=lr, weight_decay=wd, clip=1.0, frac_train=f, init_seed=s)
                    for lr in (1.0, 3.0) for wd in (1e-3, 3e-4, 1e-4) for s in S]) for f in (0.6, 0.8)]


def cohort_SRb():
    """E1b: gdm / sgd at the E1 winner (robustness), 8 seeds each."""
    S = SEEDS[:8]; wd = S5REC["weight_decay"]
    return [("SR", [gcfg("s5", opt="gdm", lr=0.3, weight_decay=wd, clip=1.0, momentum=0.9, init_seed=s) for s in S]),
            ("SR", [gcfg("s5", opt="sgd", batch_size=1024, lr=1.0, weight_decay=wd, init_seed=s) for s in S])]


def cohort_S0():
    """E0: the measured weighted ladder. Fixed-support runs (interfaces projected onto the isotypic span after every step):
    each faithful irrep alone and each faithful irrep + sign, nu {4, 6}, 4 seeds, S5 record recipe, 60k. Ceiling per
    support = best gamma_2 over seeds. 10 supports x 2 nu x 4 seeds = 80 runs."""
    S = SEEDS[:4]; sup = [(r,) for r in (2, 3, 4, 5, 6)] + [(r, 1) for r in (2, 3, 4, 5, 6)]
    return [("S0", [gcfg("s5", nu=nu, support=sp, init_seed=s, **S5REC, **LONG) for sp in sup for s in S]) for nu in (4, 6)]


def cohort_S1():
    """E2: S5 main. Record recipe x nu {4, 6} x frac {0.6, 0.8} x 16 seeds, 60k with a 20k checkpoint. 64 runs."""
    S = SEEDS[:16]
    return [("S1", [gcfg("s5", nu=nu, frac_train=f, init_seed=s, **S5REC, **LONG) for s in S]) for nu in (4, 6) for f in (0.6, 0.8)]


def cohort_S2c():
    """E3 (conditional): Wei limit at S5, nu 4, frac 0.8, wd {1e-3, 3e-4, 1e-4} x 8 seeds, 200k, ckpts 20k / 60k."""
    S = SEEDS[:8]; E = dict(num_epochs=200_000, spectra_every=2000, ckpt_at=(20_000, 60_000))
    rec = {k: v for k, v in S5REC.items() if k != "weight_decay"}
    return [("S2", [gcfg("s5", nu=4, weight_decay=wd, init_seed=s, **rec, **E) for wd in (1e-3, 3e-4, 1e-4) for s in S])]


def cohort_S3():
    """E4: other groups with the same ladder + free-run pair. S4 (faithful std, std_sign; sign patch), A5 (3a, 3b, 4, 5; no
    sign irrep), D_113 (56 two-dim irreps: one single + hierarchical pairs rho1 + rho_u, u in {2, 3, 5, 8, 27, 56}; the
    single-irrep ceilings are all equal by the r -> r^u automorphism). Ladders at 4 seeds (2 at D_113, nu 4 only);
    free runs 8 seeds x nu {4, 6}, 20k (60k at S4 / A5 is cheap: use LONG)."""
    S8, S4_ = SEEDS[:8], SEEDS[:4]; J = []
    lad = {"s4": [(3,), (4,), (3, 1), (4, 1)], "a5": [(1,), (2,), (3,), (4,)]}
    for task in ("s4", "a5"):
        for nu in (4, 6):
            J.append((f"S3", [gcfg(task, nu=nu, support=sp, init_seed=s, **S5REC, **LONG) for sp in lad[task] for s in S4_]
                             + [gcfg(task, nu=nu, init_seed=s, **S5REC, **LONG) for s in S8]))
    d_sup = [(2,)] + [(2, 1 + u) for u in (2, 3, 5, 8, 27, 56)]      # rho_j is irrep index 1 + j
    J.append(("S3", [gcfg("d113", nu=4, support=sp, init_seed=s, **S5REC) for sp in d_sup for s in SEEDS[:2]]))
    for nu in (4, 6):
        J.append(("S3", [gcfg("d113", nu=nu, init_seed=s, **S5REC) for s in S8]))
    return J


def cohort_S3b():
    """E4 addendum (after the first S3 read, 09-13 10:15Z): supports the free runs used that the singles ladder did not
    price. A5: the Galois-conjugate pair 3a + 3b (8/8 free runs at nu 4), plus 3a + 4 and 4 + 5 as controls. S4: std +
    sign + 2 and std_sign + sign + 2 (the free runs' third member is the non-faithful 2-dim irrep). 4 seeds, nu {4, 6}."""
    S4_ = SEEDS[:4]; J = []
    for nu in (4, 6):
        J.append(("S3", [gcfg("a5", nu=nu, support=sp, init_seed=s, **S5REC, **LONG) for sp in ((1, 2), (1, 3), (3, 4)) for s in S4_]))
        J.append(("S3", [gcfg("s4", nu=nu, support=sp, init_seed=s, **S5REC, **LONG) for sp in ((3, 1, 2), (4, 1, 2)) for s in S4_]))
    return J


def cohort_S3c():
    """D_113 retry (09-13 10:40Z): the record recipe collapses at D_113 frac 0.8 (0/8, weights -> 0, as wd 1e-3 did at
    mod-add frac 0.7). Lower train fraction + lower wd: nu 4, frac 0.5, wd {3e-4, 1e-4} x 4 seeds, 20k."""
    rec = {k: v for k, v in S5REC.items() if k != "weight_decay"}
    return [("S3", [gcfg("d113", nu=4, frac_train=0.5, weight_decay=wd, init_seed=s, **rec) for wd in (3e-4, 1e-4) for s in SEEDS[:4]])]


# ---------------------------------------------------------------- 2026-09-13 pm: abelian groups Z_120 and Z_11 x Z_11 (PLAN_Z.md)
ZREC = dict(opt="gd", lr=3.0, weight_decay=3e-4, clip=1.0)     # mod-add / S5 record; ZR checks it transfers (lr 1 rides along)
ZGROUPS = ("z120", "z11x11")


def zcfg(task, **kw):
    """abelian cell: p = |G|, frac 0.5 (mod-add convention; 7200 / 7320 train pairs), m 512, d = r = 128."""
    base = dict(task=task, nu=4, m=512, d=128, r=128, frac_train=0.5, data_seed=DATA_SEED, num_epochs=20_000, spectra_every=200)
    base.update(kw)
    return Config(**base)


def cohort_ZR():
    """Z-R: recipe check, lr {1, 3} x wd 3e-4 x clip 1, 8 seeds, 20k, both groups (one batch per group). 32 runs."""
    S = SEEDS[:8]
    return [("ZR", [zcfg(g, opt="gd", lr=lr, weight_decay=3e-4, clip=1.0, init_seed=s) for lr in (1.0, 3.0) for s in S]) for g in ZGROUPS]


def cohort_Z1():
    """Z-1: main, record recipe x nu {4, 6} x 16 seeds x both groups, 60k with a 20k checkpoint. 64 runs."""
    S = SEEDS[:16]
    return [("Z1", [zcfg(g, nu=nu, init_seed=s, **ZREC, **LONG) for s in S]) for g in ZGROUPS for nu in (4, 6)]


def cohort_ZL():
    """Z-L: fixed-support ladder at the kstar_ab.py sets S*_K for K in {K*-1 .. K*+2} (per nu), 4 seeds, 60k; the measured
    cross-check of the constructed ceilings (S5 methodology). ~2 groups x 4 K x 2 nu x 4 = 64 runs."""
    import json
    ks = json.load(open(Path(__file__).resolve().parent / "kstar_ab.json")); S4_ = SEEDS[:4]; J = []
    for g in ZGROUPS:
        for nu in (4, 6):
            Ks = ks[g][f"kstar_nu{nu}"]; sups = [tuple(ks[g]["ladder"][str(K)]["S"]) for K in range(max(1, Ks - 1), Ks + 3) if str(K) in ks[g]["ladder"]]
            J.append(("ZL", [zcfg(g, nu=nu, support=sp, init_seed=s, **ZREC, **LONG) for sp in sups for s in S4_]))
    return J


def cohort_Zsmoke():
    S = SEEDS[:2]; E = dict(m=64, d=16, r=16, num_epochs=30, spectra_every=10)
    return [("smoke", [zcfg("z120", opt="gd", lr=1.0, weight_decay=1e-3, clip=1.0, init_seed=s, support=sp, **E) for sp in ((), (1, 60)) for s in S]),
            ("smoke", [zcfg("z11x11", nu=6, opt="gd", lr=1.0, weight_decay=1e-3, init_seed=s, **E) for s in S])]


def cohort_S5smoke():
    S = SEEDS[:2]; E = dict(m=64, d=16, r=16, num_epochs=30, spectra_every=10)
    return [("smoke", [gcfg("s4", opt="gd", lr=1.0, weight_decay=1e-3, clip=1.0, init_seed=s, support=sp, **E) for sp in ((), (3,), (3, 1)) for s in S]),
            ("smoke", [gcfg("a5", nu=6, opt="gd", lr=1.0, weight_decay=1e-3, init_seed=s, **E) for s in S]),
            ("smoke", [gcfg("s5", nu=3, opt="gdm", lr=0.1, weight_decay=1e-3, init_seed=s, support=(2, 1), **E) for s in S])]


COHORTS = {"ZR": cohort_ZR, "Z1": cohort_Z1, "ZL": cohort_ZL, "Zsmoke": cohort_Zsmoke, "S3c": cohort_S3c, "S3b": cohort_S3b, "SR": cohort_SR, "SRb": cohort_SRb, "S0": cohort_S0, "S1": cohort_S1, "S2c": cohort_S2c, "S3": cohort_S3, "S5smoke": cohort_S5smoke,
           "R": cohort_R, "R2": cohort_R2, "smoke": cohort_smoke, "Rkeep": cohort_Rkeep, "H1": cohort_H1, "H2": cohort_H2,
           "H3": cohort_H3, "H5": cohort_H5, "H": cohort_H, "H4": cohort_H4, "H2b": cohort_H2b, "S2": cohort_S2}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", default="R")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--compile", default="default", help="reduce-overhead (CUDA graphs) crashes with a Dynamo guard error (ValueError: stoi) on the SECOND compile in one process, seen twice on 09-12 (nu-6 layouts after a nu-4 batch); default mode is ~10%% slower and safe")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--log-every", type=int, default=500)
    ap.add_argument("--max-batch", type=int, default=144, help="split larger batches into chunks of at most this many runs")
    args = ap.parse_args()
    torch.set_float32_matmul_precision("highest")
    if args.smoke:
        args.cohort, args.compile, args.log_every = (args.cohort if args.cohort.endswith("smoke") else "smoke"), "off", 10
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
