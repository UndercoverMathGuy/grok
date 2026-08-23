"""Gradient-tap trajectory cohorts (Phase 2 of the gradient introspection).

C8  mirror decoherence: retrain all 64 mirror arms (installs + controls +
    twins) for 5k epochs with the grad tap on a dense grid. Readout: for each
    (C, gC) install pair, the epoch where the automorphism-mapped grad
    spectra exceed the twin-pair float-divergence envelope — is the unsigned
    ~20% timing dispersion injected at a phase, or chaotic from step 1?
D11 eviction watch: retrain the 32 Phase B K=5 install arms for 3k epochs
    with the tap. Readout: per installed member, the grad_dot trajectory —
    does the doomed (evicted) member receive net-decay pressure from epoch 0
    or only after competition develops?

Pod:   setsid nohup python3 -u gradtap/train_gradtap.py > gradtap.log 2>&1 &
Smoke: python3 gradtap/train_gradtap.py --smoke   (CPU, ~2 min)

If reduce-overhead trips on the eager tap between graph replays, rerun with
--compile default.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("RUNS_DIR", str(ROOT / "runs_gradtap"))
sys.path.insert(0, str(ROOT / "cloud"))

import numpy as np  # noqa: E402
from train_semifinal_torch import RUNS, Config, report, train_batched  # noqa: E402


def jobs_from(arms, sub, epochs, smoke):
    out = []
    for a in arms:
        cfg = {k: v for k, v in a["config"].items()
               if k in Config.__dataclass_fields__}
        cfg |= dict(num_epochs=60 if smoke else epochs,
                    save_every=60 if smoke else epochs)
        rd = RUNS / sub / a["tag"]
        if (rd / "spectra.npz").exists():
            continue
        ck = a["ckpt"]
        ck = ROOT / ck if not Path(ck).is_absolute() else Path(ck)
        assert ck.exists(), ck
        out.append((Config(**cfg), rd, str(ck)))
    return out


def run(jobs, label, args):
    if not jobs:
        print(f"{label}: nothing to do", flush=True)
        return
    print(f"{label}: {len(jobs)} runs x {jobs[0][0].num_epochs} ep", flush=True)
    train_batched([c for c, _, _ in jobs], [rd for _, rd, _ in jobs],
                  init_from=[ck for _, _, ck in jobs],
                  compile_mode="off" if args.smoke else args.compile,
                  spectra_every=args.spectra_every, grad_tap=True)
    for _, rd, _ in jobs:
        report(rd)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--compile", default="reduce-overhead",
                    choices=["default", "reduce-overhead", "off"])
    ap.add_argument("--spectra-every", type=int, default=10)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    t0 = time.time()

    mir = json.loads((ROOT / "mirror" / "arms" / "mirror_manifest.json")
                     .read_text())["arms"]
    phb = json.loads((ROOT / "compiler" / "arms" / "phaseB_manifest.json")
                     .read_text())
    k5 = [a for a in phb if "_K5_" in a["tag"]]
    assert len(k5) == 32, len(k5)
    for a in k5:  # ckpt paths in the B manifest are absolute from compile time
        a["ckpt"] = str(ROOT / "compiler" / "arms" / "ckpts_b"
                        / Path(a["ckpt"]).name)
    if args.smoke:
        b0 = mir[0]["base"]
        mir = [a for a in mir if a["tag"] in (f"{b0}_c0_orig", f"{b0}_twin")]
        k5 = k5[:2]

    run(jobs_from(mir, "c8", 5000, args.smoke), "c8 mirror", args)
    run(jobs_from(k5, "d11", 3000, args.smoke), "d11 eviction", args)

    if args.smoke:
        z = np.load(RUNS / "c8" / f"{mir[0]['tag']}" / "spectra.npz")
        assert "grad_energy" in z and "grad_dot" in z, list(z)
        ge, gd = z["grad_energy"], z["grad_dot"]
        assert ge.shape == gd.shape and ge.shape[1] == 56, ge.shape
        assert np.isfinite(ge).all() and np.isfinite(gd).all()
        assert (ge >= 0).all() and ge.max() > 0
        print(f"SMOKE OK: grad tap shapes {ge.shape}, "
              f"gE[0] top f{int(ge[0].argmax()) + 1}, "
              f"dot range [{gd.min():.2e}, {gd.max():.2e}]")
    print(f"gradtap done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
