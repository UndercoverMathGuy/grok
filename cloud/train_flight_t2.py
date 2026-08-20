"""Tier-2 flight-time cohorts: forecast showpiece (T2-D), lr*wd product grid
+ eps arm (T2-C), and the frac-matched Deff cohort (T2-E).

Prereg: flight/PREREG_T2.md (committed before launch). Stages:

  t2d-seg1      15 forecast runs, epochs 0..500, WARM ckpt at 500 (adam.*)
  t2d-forecast  compute per-run predicted t_gen from seg1 spectra ONLY,
                write flight/forecasts_t2d.json (refuses to overwrite)
  t2d-seg2      warm-resume 500 -> 30000 into <run>/seg2/ (requires the
                forecasts file to exist: prereg ordering is enforced)
  t2d-floor     8 floor-branch runs (ex-ante band call in the prereg),
                straight 10k, no pause
  t2c           36-run lr x wd grid (per-run lr/wd, shared batches) then
                2 x 4 eps arms (eps must agree within a batch)
  t2e           24 runs: {add113, add96, sub113} x frac {0.3, 0.6} x wd 1,
                50k epochs, save_every=500 (sparse ckpts for Deff)

All restarts are WARM (model + Adam moments + step offset) via
save_optim=True checkpoints; the only cold restarts anywhere in tier-2 are
the T2-B optimizer-reset control arms, which are not in this file.

    python -u cloud/train_flight_t2.py --stage t2d-seg1
"""

import argparse
import json
import sys
import zlib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "cloud"))

from train_semifinal_torch import Config, train_batched

META_SEED = 20260820
RUNS = ROOT / "runs_flight"
FORECAST_EPOCH = 500
TOTAL_EPOCHS = 30_000
FLOOR_EPOCHS = 10_000
T2E_EPOCHS = 50_000
SPECTRA_EVERY = 25
R_STAR = 10 ** -2.25          # preregistered: add113 task median (455-run sweep)
FLOOR_WAIT = 132              # preregistered: median floor-branch wait
FLOOR_GATE = np.e             # r_mem <= e * r_star -> floor-branch forecast

# (frac, wd) cells; every forecast cell's historical min t_gen >= 725
FORECAST_CELLS = [(0.2, 3.0), (0.25, 1.0), (0.3, 1.0), (0.3, 3.0), (0.4, 1.0)]
FLOOR_CELLS = [(0.6, 1.0), (0.7, 1.0), (0.9, 1.0), (0.9, 3.0)]
N_FORECAST_SEEDS = 3
N_FLOOR_SEEDS = 2

GRID_LRS = [3e-4, 1e-3, 3e-3]
GRID_WDS = [0.3, 1.0, 3.0]
GRID_SEEDS = 4
EPS_ARMS = [1e-6, 1e-10]      # 1e-8 covered by the grid
EPS_SEEDS = 4

T2E_TASKS = [("add", 113), ("add", 96), ("sub", 113)]
T2E_FRACS = [0.3, 0.6]
T2E_SEEDS = 4


def draw_seeds(family, n):
    """Fresh (data, init) pairs per family, fixed by META_SEED; drawn once,
    kept whatever comes out. crc32, not hash(): hash() is salted per process
    and would silently re-roll the cohort between invocations."""
    rng = np.random.default_rng([META_SEED, zlib.crc32(family.encode())])
    ds = rng.choice(np.arange(50_000, 60_000), n, replace=False)
    is_ = rng.choice(np.arange(10_000, 99_999), n, replace=False)
    return list(zip(ds.tolist(), is_.tolist()))


def mk_cfg(fn, p, frac, wd, ds, is_, epochs, lr=1e-3, eps=1e-8, save_every=500):
    return Config(p=p, fn_name=fn, frac_train=frac, data_seed=ds, init_seed=is_,
                  d_model=128, num_heads=4, d_mlp=512, act_type="ReLU",
                  lr=lr, weight_decay=wd, betas=(0.9, 0.98), adam_eps=eps,
                  num_epochs=epochs, save_every=save_every)


def fs(mask):
    idx = np.where(mask[:-1] & mask[1:])[0]
    return int(idx[0]) if len(idx) else -1


def committee_final(E):
    e = np.log10(np.clip(E[-1], 1e-12, None))
    o = np.argsort(e)[::-1]
    g = e[o[:-1]] - e[o[1:]]
    return o[:int(np.argmax(g[:min(10, len(g))])) + 1].tolist()


def t2d_jobs(epochs):
    jobs = []
    for frac, wd in FORECAST_CELLS:
        for ds, is_ in draw_seeds(f"t2d_f{frac}_wd{wd}", N_FORECAST_SEEDS):
            rd = RUNS / "t2d" / f"f{frac}_wd{wd}" / f"ds{ds}_is{is_}"
            jobs.append((mk_cfg("add", 113, frac, wd, ds, is_, epochs,
                                save_every=FORECAST_EPOCH), rd))
    return jobs


def stage_t2d_seg1(args):
    jobs = t2d_jobs(FORECAST_EPOCH)
    jobs = [(c, rd) for c, rd in jobs if not (rd / "spectra.npz").exists()]
    if not jobs:
        print("seg1 already complete")
        return
    print(f"t2d-seg1: {len(jobs)} runs to {FORECAST_EPOCH} ep (warm ckpts)")
    train_batched([c for c, _ in jobs], [rd for _, rd in jobs],
                  spectra_every=SPECTRA_EVERY, loss=args.loss,
                  compile_mode=args.compile_mode, save_optim=True)


def stage_t2d_forecast(args):
    out = (ROOT / "flight" / "forecasts_t2d.json"
           if RUNS.name == "runs_flight" else RUNS / "forecasts_t2d.json")
    if out.exists():
        sys.exit(f"REFUSING to overwrite existing forecasts: {out}")
    rows = []
    for _, rd in t2d_jobs(FORECAST_EPOCH):
        z = np.load(rd / "spectra.npz")
        ep, E, ta = z["epochs"], z["energy"], z["train_acc"]
        te = z["test_acc"]
        comm = committee_final(E)
        junk = [j for j in range(E.shape[1]) if j not in comm]
        r = E[:, junk].sum(1) / E.sum(1)
        im = fs(ta >= 0.99)
        row = {"run": str(rd.relative_to(RUNS)), "committee_ep500": comm}
        if im < 0:
            row["forecast"] = None
            row["why"] = "no memorization by forecast epoch"
        elif fs(te >= 0.9) >= 0:
            row["forecast"] = None
            row["why"] = "generalized before forecast epoch (window violated)"
        else:
            t_mem = int(ep[im])
            r_mem = float(r[im])
            iw = np.where(ep >= t_mem)[0]
            lam = -float(np.polyfit(ep[iw].astype(float),
                                    np.log(np.clip(r[iw], 1e-12, None)), 1)[0])
            row.update(t_mem=t_mem, r_mem=r_mem, lam_per_ep=lam)
            if r_mem <= FLOOR_GATE * R_STAR:
                row["branch"] = "floor"
                row["forecast"] = t_mem + FLOOR_WAIT
            elif lam <= 0:
                row["branch"] = "cleanup"
                row["forecast"] = "CENSORED"      # junk not decaying
            else:
                row["branch"] = "cleanup"
                pred = t_mem + np.log(r_mem / R_STAR) / lam
                row["forecast"] = "CENSORED" if pred > 30_000 else round(pred)
        rows.append(row)
        print(f'  {row["run"]}: forecast={row.get("forecast")} '
              f'(t_mem={row.get("t_mem")}, lam={row.get("lam_per_ep")})')
    out.write_text(json.dumps(
        {"meta": {"r_star": R_STAR, "floor_wait": FLOOR_WAIT,
                  "forecast_epoch": FORECAST_EPOCH,
                  "rule": "t_gen = t_mem + max(log(r_mem/r*)/lam, floor); "
                          "pred > 30000 -> CENSORED; scored vs first "
                          "sustained test_acc >= 0.9 of the stitched run"},
         "forecasts": rows}, indent=1))
    print(f"wrote {out} — COMMIT THIS BEFORE t2d-seg2")


def stage_t2d_seg2(args):
    fc = (ROOT / "flight" / "forecasts_t2d.json"
          if RUNS.name == "runs_flight" else RUNS / "forecasts_t2d.json")
    assert fc.exists(), \
        "forecasts_t2d.json missing: run t2d-forecast (and commit it) first"
    jobs = []
    for c, rd in t2d_jobs(TOTAL_EPOCHS - FORECAST_EPOCH):
        if (rd / "seg2" / "spectra.npz").exists():
            continue
        ck = rd / "checkpoints" / f"epoch_{FORECAST_EPOCH:05d}.safetensors"
        assert ck.exists(), f"missing warm checkpoint {ck}"
        jobs.append((c, rd / "seg2", ck))
    if not jobs:
        print("seg2 already complete")
        return
    print(f"t2d-seg2: {len(jobs)} warm resumes to 30k")
    train_batched([c for c, _, _ in jobs], [rd for _, rd, _ in jobs],
                  init_from=[ck for _, _, ck in jobs],
                  spectra_every=SPECTRA_EVERY, loss=args.loss,
                  compile_mode=args.compile_mode, save_optim=True)


def stage_t2d_floor(args):
    jobs = []
    for frac, wd in FLOOR_CELLS:
        for ds, is_ in draw_seeds(f"t2dfloor_f{frac}_wd{wd}", N_FLOOR_SEEDS):
            rd = RUNS / "t2d_floor" / f"f{frac}_wd{wd}" / f"ds{ds}_is{is_}"
            if (rd / "spectra.npz").exists():
                continue
            jobs.append((mk_cfg("add", 113, frac, wd, ds, is_, FLOOR_EPOCHS,
                                save_every=FLOOR_EPOCHS), rd))
    if not jobs:
        print("floor cohort complete")
        return
    print(f"t2d-floor: {len(jobs)} runs, 10k ep")
    train_batched([c for c, _ in jobs], [rd for _, rd in jobs],
                  spectra_every=SPECTRA_EVERY, loss=args.loss,
                  compile_mode=args.compile_mode)


def stage_t2c(args):
    grid = []
    for lr in GRID_LRS:
        for wd in GRID_WDS:
            for ds, is_ in draw_seeds(f"t2c_lr{lr}_wd{wd}", GRID_SEEDS):
                rd = RUNS / "t2c" / f"lr{lr}_wd{wd}" / f"ds{ds}_is{is_}"
                if (rd / "spectra.npz").exists():
                    continue
                grid.append((mk_cfg("add", 113, 0.3, wd, ds, is_,
                                    TOTAL_EPOCHS, lr=lr,
                                    save_every=TOTAL_EPOCHS), rd))
    for i in range(0, len(grid), args.batch):
        chunk = grid[i:i + args.batch]
        print(f"t2c grid batch {i // args.batch}: {len(chunk)} runs")
        train_batched([c for c, _ in chunk], [rd for _, rd in chunk],
                      spectra_every=SPECTRA_EVERY, loss=args.loss,
                      compile_mode=args.compile_mode)
    for eps in EPS_ARMS:                       # adam_eps must agree per batch
        jobs = []
        for ds, is_ in draw_seeds(f"t2c_eps{eps}", EPS_SEEDS):
            rd = RUNS / "t2c_eps" / f"eps{eps}" / f"ds{ds}_is{is_}"
            if (rd / "spectra.npz").exists():
                continue
            jobs.append((mk_cfg("add", 113, 0.3, 1.0, ds, is_, TOTAL_EPOCHS,
                                eps=eps, save_every=TOTAL_EPOCHS), rd))
        if jobs:
            print(f"t2c eps={eps}: {len(jobs)} runs")
            train_batched([c for c, _ in jobs], [rd for _, rd in jobs],
                          spectra_every=SPECTRA_EVERY, loss=args.loss,
                          compile_mode=args.compile_mode)


def stage_t2e(args):
    for fn, p in T2E_TASKS:                    # p / fn_name must agree per batch
        jobs = []
        for frac in T2E_FRACS:
            for ds, is_ in draw_seeds(f"t2e_{fn}{p}_f{frac}", T2E_SEEDS):
                rd = RUNS / "t2e" / f"{fn}{p}_f{frac}" / f"ds{ds}_is{is_}"
                if (rd / "spectra.npz").exists():
                    continue
                jobs.append((mk_cfg(fn, p, frac, 1.0, ds, is_, T2E_EPOCHS,
                                    save_every=500), rd))
        if jobs:
            print(f"t2e {fn}{p}: {len(jobs)} runs, 50k ep, sparse ckpts")
            train_batched([c for c, _ in jobs], [rd for _, rd in jobs],
                          spectra_every=SPECTRA_EVERY, loss=args.loss,
                          compile_mode=args.compile_mode)


STAGES = {"t2d-seg1": stage_t2d_seg1, "t2d-forecast": stage_t2d_forecast,
          "t2d-seg2": stage_t2d_seg2, "t2d-floor": stage_t2d_floor,
          "t2c": stage_t2c, "t2e": stage_t2e}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=sorted(STAGES))
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--loss", default="f32stable", choices=["f32stable", "f64"])
    ap.add_argument("--compile", dest="compile_mode", default="default")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    global FORECAST_EPOCH, TOTAL_EPOCHS, FLOOR_EPOCHS, T2E_EPOCHS, SPECTRA_EVERY
    if args.smoke:
        # tiny end-to-end wiring test (separate RUNS tree; forecasts are
        # numerically meaningless and written to a smoke path)
        args.compile_mode = "off"
        FORECAST_EPOCH, TOTAL_EPOCHS = 60, 160
        FLOOR_EPOCHS, T2E_EPOCHS, SPECTRA_EVERY = 80, 80, 10
        globals()["RUNS"] = ROOT / "runs_flight_smoke"
    STAGES[args.stage](args)


if __name__ == "__main__":
    main()
