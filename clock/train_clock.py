"""Clock-separation cohort: delay selection by freezing its substrate.

Prereg: clock/README.md (committed before training). One lockstep batch of
56 runs at width 512 / wd 3.0 / 5000 epochs: base + fz{50,200,500} (W_E
untouched until N) + on{50,200,500} (ONLY W_E trains until N), 8 seed
pairs shared with the dial/race cohorts (META_SEED=20260814).

Run dirs: runs_clock/{arm}/ds{data_seed}_is{init_seed}/
Idempotent: run dirs with spectra.npz are skipped (whole-batch skip only —
the cohort is one lockstep batch, so partial re-runs re-train the batch).

On the pod:
    GIT_LFS_SKIP_SMUDGE=1 git clone --filter=blob:none --sparse \
        https://github.com/UndercoverMathGuy/grok.git && cd grok \
        && git sparse-checkout set clock dial cloud
    pip install safetensors
    setsid nohup python3 -u clock/train_clock.py > clock.log 2>&1 < /dev/null &

Smoke (CPU, ~1 min, asserts gating mechanics + equivalence):
    python3 clock/train_clock.py --smoke
"""

import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# RUNS_DIR must be set before the trainer module reads it at import time.
os.environ.setdefault("RUNS_DIR", str(ROOT / "runs_clock"))
sys.path.insert(0, str(ROOT / "cloud"))
sys.path.insert(0, str(ROOT / "dial"))

from train_semifinal_torch import RUNS, Config, report, train_batched  # noqa: E402
from train_dial import draw_seeds  # noqa: E402  (same META_SEED seed draws)

NS = [50, 200, 500]
ARMS = ([("base", None)]
        + [(f"fz{n}", ("freeze_we", n)) for n in NS]
        + [(f"on{n}", ("only_we", n)) for n in NS])


def cohort(width=512, wd=3.0, epochs=5000, save_every=500):
    data_seeds, init_seeds = draw_seeds()
    jobs = []
    for arm, fz in ARMS:
        for ds in data_seeds:
            for is_ in init_seeds:
                cfg = Config(d_mlp=width, weight_decay=wd, data_seed=ds,
                             init_seed=is_, num_epochs=epochs,
                             save_every=save_every)
                jobs.append((cfg, RUNS / arm / f"ds{ds}_is{is_}", fz))
    return jobs


def smoke():
    """CPU mechanics check: (1) gated all-active path == ungated path;
    (2) freeze_we leaves W_E exactly at init through the window and only
    W_E; (3) only_we is the exact mirror."""
    import torch
    from safetensors.torch import load_file
    kw = dict(d_mlp=128, weight_decay=1.0, data_seed=15573, init_seed=135533,
              num_epochs=40, save_every=10)
    base = RUNS / "_smoke"
    arms = [("gated_base", None), ("fz", ("freeze_we", 20)),
            ("on", ("only_we", 20))]
    # batch A: SAME 3 cfgs through the untouched freeze=None code path;
    # batch B: gated path. Equal M so run 0 is comparable bitwise.
    cfgs = [Config(**kw) for _ in arms]
    train_batched(cfgs, [base / f"plain{i}" for i in range(len(arms))],
                  compile_mode="off", spectra_every=10)
    cfgs = [Config(**kw) for _ in arms]
    train_batched(cfgs, [base / a for a, _ in arms],
                  freeze=[f for _, f in arms], compile_mode="off",
                  spectra_every=10)

    ck = lambda a, e: load_file(str(base / a / "checkpoints" / f"epoch_{e:05d}.safetensors"))
    eq = lambda x, y: torch.equal(x, y)
    # (1) equivalence: gated batch's all-active run == ungated run, bitwise
    p40, g40 = ck("plain0", 40), ck("gated_base", 40)
    assert all(torch.allclose(p40[k], g40[k], atol=0, rtol=0) for k in p40), \
        "gated all-active path diverged from ungated path"
    # (2) freeze_we: W_E untouched through the window, trained after;
    #     everything else moving from the start
    f0, f10, f20, f30 = (ck("fz", e) for e in (0, 10, 20, 30))
    assert eq(f10["embed.W_E"], f0["embed.W_E"]) and eq(f20["embed.W_E"], f0["embed.W_E"])
    assert not eq(f30["embed.W_E"], f0["embed.W_E"])
    assert not eq(f10["unembed.W_U"], f0["unembed.W_U"])
    # (3) only_we: exact mirror
    o0, o10, o20, o30 = (ck("on", e) for e in (0, 10, 20, 30))
    assert not eq(o10["embed.W_E"], o0["embed.W_E"])
    for k in o0:
        if k != "embed.W_E":
            assert eq(o20[k], o0[k]), f"only_we let {k} move during the window"
    assert not eq(o30["unembed.W_U"], o0["unembed.W_U"])
    print("SMOKE OK: equivalence + freeze_we + only_we gating all verified")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=8000)
    ap.add_argument("--compile", default="default",
                    choices=["default", "reduce-overhead", "off"])
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        smoke()
        return

    jobs = cohort(epochs=args.epochs)
    pending = [(c, rd, fz) for c, rd, fz in jobs
               if not (rd / "spectra.npz").exists()]
    print(f"clock cohort: {len(jobs)} runs, {len(pending)} pending", flush=True)
    if not pending:
        return
    t0 = time.time()
    train_batched([c for c, _, _ in pending], [rd for _, rd, _ in pending],
                  freeze=[fz for _, _, fz in pending],
                  compile_mode=args.compile, spectra_every=5)
    for _, rd, _ in pending:
        report(rd)
    print(f"clock cohort done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
