"""SEMIFINAL v2 — the whole evidence base, from scratch, in one script.

44 runs, all fresh seeds and fresh masks, nothing reused from the old
dataset, sized so every section of every analysis script in
semifinal/analysis/ produces its result from v2 runs alone. One uniform
protocol: p=113, 20k epochs (committees still drift at 15k in ~5% of the
old runs — measured), spectra every 100, checkpoints every 1000 (epoch 0
saved pre-update). Idempotent: any run with spectra.npz is skipped, so the
script can be killed and resumed and any prefix is a usable dataset.

Runs train in LOCKSTEP BATCHES (grok.batched): weights stacked with a
leading run axis, one graph, exact per-run AdamW — ~70 run-eps/s vs ~25
sequential on the 8GB M1. Width is capped well below Metal's working-set
limit (wider batches silently corrupt training there; grok.batched now
asserts on this) and past ~11 throughput falls off anyway — mem sweep
2026-08-04. Batched runs are NOT bit-identical to what sequential training
would produce (float op order, chaotically amplified); every run is still an
exact realization of its seed/mask/init, which is all the cohort-level
claims need. Don't mix batched and sequential runs within a cohort.
Loss: the stable-f32 GPU CE (fast_loss; softplus form) in place of the f64
CPU-stream CE — validated to 3e-6 rel of f64 down to loss ~3e-11, where
naive f32 underflows to 0 (the slingshot mechanism f64 was guarding).

Layout (execution order = priority order):
  1. cell-A naturals    p-113/seed4811/seed6100{1..4}     (claim 1 rows,
     claim 2B level, claim 4 census, steering bases)
  2. cell-A orthWE      orthWE/p-113/seed4811/...         (the contested
     ingredient-2 cohort — 4 independent fresh draws)
  3. cell-B orthWE      orthWE/p-113/seed7207/seed7200{1..4}  (same, on a
     second mask; 8 fresh independent inits total = the power target)
  4. cell-A doubleflat  doubleflat/p-113/seed4811/...     (nothing-readable
     cohort on a fresh mask; claim 2B third level)
  5. cell-B naturals, 6. cell-B doubleflat                (cells A and B
     identical by design: 4 nat + 4 orthWE + 4 df each)
  7-8. steering suite x2 on bases seed61001 and seed61002:
     dosefarm/<base>/dose_{110,120,150,225}   claim 3A dose-response
     suppress/<base>                          claim 3C (0.5x strongest winner)
     gkrotate/<base>/gain_{120,225}           claim 3D (energy-fixed rotation
                                              at gains matched to dose arms)
     chaospair/<base>/arm{A,B}                claim 3F (same 1.5x boost on
                                              two different dead frequencies)
     collisionfarm/<base>_t<K>                claim 4D (engineered trio)

Cut by design (kept only in the legacy dataset): transplants (claim 3E,
weak and uncited) and the tilt/noise/CVaR dynamics variants (claim 2A rests
on the legacy cohort, group-collapsed — see claim2_twins.py).

Run:  caffeinate -i nohup uv run python \
          semifinal/training/train_semifinal_v2.py \
          > runs/semifinal_v2.log 2>&1 &
One process only (Metal). DRY_RUN=1 prints the plan. ~3 h at ~80 run-eps/s.
"""
import os

import numpy as np

from _shared import (ROOT, committee_from_coeffs, fold, freq_energy,
                     grok_epoch, report, save_surgical_ckpt,
                     scale_freq_energy)

DRY_RUN = bool(os.environ.get("DRY_RUN"))

P = 113
NF = P // 2
EPOCHS = 20000
CELLS = {4811: [61001, 61002, 61003, 61004],
         7207: [72001, 72002, 72003, 72004]}
STEER_BASES = ["p-113/seed4811/seed61001", "p-113/seed4811/seed61002"]
DOSES = [1.10, 1.20, 1.50, 2.25]
GAINS = [1.20, 2.25]
# Lockstep widths (mem sweep 2026-08-04, 8GB M1): throughput plateaus at
# ~70 run-eps/s for M=8-11 and falls off beyond. 24 from-scratch runs
# split 8+8+8; each steering suite is one batch of its 10 arms.
SCRATCH_WIDTH = 8
STEER_WIDTH = 10


# --------------------------------------------------------------------------
# batched phase runner
# --------------------------------------------------------------------------

def base_cfg(ds, iseed, **kw):
    from grok.config import Config
    return Config(p=P, data_seed=ds, init_seed=iseed, num_epochs=EPOCHS,
                  save_every=1000, **kw)


def doubleflat_cfg(ds, iseed):
    cfg = base_cfg(ds, iseed, embed_init="orthogonal", attn_init="isometric")
    if DRY_RUN:
        return cfg
    import mlx.core as mx
    from grok.model import Transformer
    mx.random.seed(cfg.init_seed)
    t = _T_k(Transformer(cfg))
    spread = (t.max() - t.min()) / t.mean()
    print(f"doubleflat seed{iseed}: init T_k spread {spread:.2e}", flush=True)
    assert spread < 1e-5, "isometric init failed to flatten T_k"
    return cfg


def batch_train(jobs, width):
    """jobs: [(run_name, cfg, init_from-or-None)]. Skips finished runs and
    trains the rest in lockstep chunks of `width`, preserving job order (so
    any prefix of the plan is still a usable dataset)."""
    todo = []
    for name, cfg, ckpt in jobs:
        if (ROOT / "runs" / name / "spectra.npz").exists():
            print(f"skip {name} (exists)", flush=True)
        elif DRY_RUN:
            print(f"WOULD TRAIN {name} ({cfg.num_epochs} epochs)", flush=True)
        else:
            todo.append((name, cfg, ckpt))
    for i in range(0, len(todo), width):
        chunk = todo[i:i + width]
        print(f"=== batch x{len(chunk)}: {', '.join(n for n, _, _ in chunk)} ===",
              flush=True)
        from grok.batched import train_batched
        train_batched([c for _, c, _ in chunk],
                      [ROOT / "runs" / n for n, _, _ in chunk],
                      init_from=[ck for _, _, ck in chunk],
                      spectra_every=100, fast_loss=True)
        for n, _, _ in chunk:
            report(ROOT / "runs" / n)


def _T_k(model):
    from grok.fourier import Fourier
    f = Fourier(P)
    W_E = np.array(model.W_E, dtype=np.float64)[:, :P]
    at = model.blocks[0].attn
    W_V = np.array(at.W_V, dtype=np.float64)
    W_O = np.array(at.W_O, dtype=np.float64)
    h, dh, _ = W_V.shape
    t = np.zeros(NF)
    for i in range(h):
        OV = W_O[:, i * dh:(i + 1) * dh] @ W_V[i]
        t += freq_energy((OV @ W_E) @ f.basis.T, NF)
    return t


# --------------------------------------------------------------------------
# steering suite (constructions identical to the verified legacy scripts:
# train_dose_farm / train_gk_rotate / train_collision_farm)
# --------------------------------------------------------------------------

def pick_target(comm, menu, zfin):
    """Pre-registered dose-farm rule: the most-dead frequency (min final
    |coeff|) outside the menu, the committee, and their harmonic/additive
    relatives."""
    excl = set(menu) | set(comm)
    for k in comm:
        excl |= {fold(2 * k, P), fold(3 * k, P)}
        for j in comm:
            if j != k:
                excl |= {fold(k + j, P), fold(k - j, P)}
    cands = [k for k in range(1, NF + 1) if k not in excl]
    return sorted(cands, key=lambda k: abs(zfin[k - 1]))


def surgical_ckpt(base, scales=None, gk=None, tag=""):
    """Build an epoch-0 init checkpoint from `base` with W_E edited: energy
    scales {freq: factor} and/or an energy-preserving gk rotation
    (freq, gain). Returns (ckpt_path, training Config)."""
    from grok.config import Config
    from grok.model import Transformer
    from grok.fourier import Fourier
    f = Fourier(P)
    cfg = Config.load(base / "config.json")
    model = Transformer(cfg)
    e0 = base / "checkpoints" / "epoch_00000.safetensors"
    model.load_weights(str(e0))
    W_E = np.array(model.W_E, dtype=np.float64)
    F = W_E[:, :P] @ f.basis.T
    Fm = F.copy()
    if scales:
        Fm = scale_freq_energy(Fm, scales)
    if gk:
        k, g_target = gk
        at = model.blocks[0].attn
        W_V = np.array(at.W_V, dtype=np.float64)
        W_O = np.array(at.W_O, dtype=np.float64)
        h, dh, _ = W_V.shape
        OVs = [W_O[:, i * dh:(i + 1) * dh] @ W_V[i] for i in range(h)]
        M2 = sum(OV.T @ OV for OV in OVs)
        w_eig, V = np.linalg.eigh(M2)
        top = V[:, np.argsort(w_eig)[::-1][:2]]
        B0 = Fm[:, [2 * k - 1, 2 * k]].copy()

        def tk_of(B):
            return sum(((OV @ B) ** 2).sum() for OV in OVs)

        def rotated(alpha):
            B = (1 - alpha) * B0 + alpha * top * np.linalg.norm(B0, axis=0)
            B *= np.linalg.norm(B0, axis=0) / np.linalg.norm(B, axis=0)
            return B

        t0 = tk_of(B0)
        assert tk_of(rotated(1.0)) / t0 >= g_target, "gain unreachable"
        lo, hi = 0.0, 1.0
        for _ in range(60):
            mid = (lo + hi) / 2
            (lo, hi) = (mid, hi) if tk_of(rotated(mid)) / t0 < g_target \
                else (lo, mid)
        B = rotated((lo + hi) / 2)
        err = (np.abs(freq_energy(np.concatenate(
            [Fm[:, :2 * k - 1], B, Fm[:, 2 * k + 1:]], axis=1), NF)
            - freq_energy(Fm, NF)).max() / freq_energy(Fm, NF).mean())
        Fm[:, 2 * k - 1] = B[:, 0]
        Fm[:, 2 * k] = B[:, 1]
        print(f"    gk rotation f{k} gain {g_target}: energy err {err:.1e}",
              flush=True)
        assert err < 1e-7, "gk rotation moved energy"
    W = W_E.copy()
    W[:, :P] = Fm @ f.basis
    model.load_weights(str(e0))
    ck = save_surgical_ckpt(model, W, tag)
    c2 = Config.load(base / "config.json")
    c2.num_epochs = EPOCHS
    c2.save_every = 1000
    return ck, c2


def steering_suite(base_rel):
    base = ROOT / "runs" / base_rel
    bname = base.name                                    # e.g. seed61001
    if DRY_RUN:
        for d in DOSES:
            print(f"WOULD TRAIN dosefarm/{bname}/dose_{int(d*100):03d}")
        print(f"WOULD TRAIN suppress/{bname}")
        for g in GAINS:
            print(f"WOULD TRAIN gkrotate/{bname}/gain_{int(g*100):03d}")
        print(f"WOULD TRAIN chaospair/{bname}/armA + armB")
        print(f"WOULD TRAIN collisionfarm/{bname}_t<K>")
        return
    z = np.load(base / "spectra.npz")
    assert float(z["test_acc"][-1]) >= 0.99, f"base {base_rel} never grokked"
    zfin = z["coeffs"][-1]
    comm = committee_from_coeffs(zfin)
    i3 = int(np.argmin(np.abs(z["epochs"] - 3000)))
    menu = (np.argsort(np.abs(z["coeffs"][i3]))[::-1][:8] + 1).tolist()
    cands = pick_target(comm, menu, zfin)
    target, chaosA, chaosB = cands[0], cands[1], cands[2]
    strongest = max(comm, key=lambda k: abs(zfin[k - 1]))
    print(f"\n### steering suite on {base_rel}: committee {comm}, "
          f"dose/gk target f{target}, chaos f{chaosA}/f{chaosB}, "
          f"suppress f{strongest}", flush=True)

    jobs = []

    def add(run_name, scales=None, gk=None, tag=""):
        if (ROOT / "runs" / run_name / "spectra.npz").exists():
            print(f"skip {run_name} (exists)", flush=True)
            return
        ck, cfg = surgical_ckpt(base, scales, gk, tag)
        jobs.append((run_name, cfg, ck))

    for d in DOSES:
        add(f"dosefarm/{bname}/dose_{int(d*100):03d}",
            scales={target: d}, tag=f"v2dose_{bname}_{d}")
    add(f"suppress/{bname}", scales={strongest: 0.5}, tag=f"v2sup_{bname}")
    for g in GAINS:
        add(f"gkrotate/{bname}/gain_{int(g*100):03d}",
            gk=(target, g), tag=f"v2gk_{bname}_{g}")
    add(f"chaospair/{bname}/armA", scales={chaosA: 1.5}, tag=f"v2chA_{bname}")
    add(f"chaospair/{bname}/armB", scales={chaosB: 1.5}, tag=f"v2chB_{bname}")
    # collision: t = fold of the two strongest members, boosted x2.25
    order = sorted(comm, key=lambda k: -abs(zfin[k - 1]))
    trio_t = None
    for a in range(len(order)):
        for b in range(a + 1, len(order)):
            i, j = order[a], order[b]
            for cand in (fold(i + j, P), fold(i - j, P)):
                if cand not in (0, i, j) and cand not in comm:
                    trio_t = cand
                    break
            if trio_t:
                break
        if trio_t:
            break
    assert trio_t, f"no collision target on {base_rel}"
    add(f"collisionfarm/{bname}_t{trio_t}",
        scales={trio_t: 2.25}, tag=f"v2col_{bname}")
    batch_train(jobs, STEER_WIDTH)


# --------------------------------------------------------------------------
# the plan, in priority order
# --------------------------------------------------------------------------

print(__doc__, flush=True)
dsA, dsB = sorted(CELLS)
jobs = []
# 1. cell-A naturals (steering bases live here)
for s in CELLS[dsA]:
    jobs.append((f"p-113/seed{dsA}/seed{s}", base_cfg(dsA, s), None))
# 2-3. the contested cohort, both masks
for ds in (dsA, dsB):
    for s in CELLS[ds]:
        jobs.append((f"orthWE/p-113/seed{ds}/seed{s}",
                     base_cfg(ds, s, embed_init="orthogonal"), None))
# 4. cell-A doubleflat
for s in CELLS[dsA]:
    jobs.append((f"doubleflat/p-113/seed{dsA}/seed{s}",
                 doubleflat_cfg(dsA, s), None))
# 5-6. cell-B naturals + doubleflat (cells identical by design)
for s in CELLS[dsB]:
    jobs.append((f"p-113/seed{dsB}/seed{s}", base_cfg(dsB, s), None))
for s in CELLS[dsB]:
    jobs.append((f"doubleflat/p-113/seed{dsB}/seed{s}",
                 doubleflat_cfg(dsB, s), None))
batch_train(jobs, SCRATCH_WIDTH)
# 7-8. steering suites
for b in STEER_BASES:
    steering_suite(b)
print("SEMIFINAL V2 DATASET DONE", flush=True)
