"""Mirror cohort training (manifest-driven; prereg: mirror/README.md).

Local CPU or pod:  python3 mirror/train_mirror.py
Smoke (CPU, ~3 min): one install arm + its exact twin, 150 epochs in the
SAME lockstep batch, then verify the epoch-0 spectra are exactly the
g-permuted originals and report the float-divergence growth curve.
"""

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("RUNS_DIR", str(ROOT / "runs_mirror"))
sys.path.insert(0, str(ROOT / "cloud"))

import numpy as np  # noqa: E402
from train_semifinal_torch import RUNS, Config, report, train_batched  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--compile", default="reduce-overhead",
                    choices=["default", "reduce-overhead", "off"])
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    man = json.loads((ROOT / "mirror" / "arms" / "mirror_manifest.json")
                     .read_text())
    arms = man["arms"]
    if args.smoke:
        b0 = arms[0]["base"]
        arms = [a for a in arms if a["tag"] in (f"{b0}_c0_orig", f"{b0}_twin")]
        assert len(arms) == 2

    jobs = []
    for a in arms:
        cfg = {k: v for k, v in a["config"].items()
               if k in Config.__dataclass_fields__}
        if args.smoke:
            cfg |= dict(num_epochs=150, save_every=150)
        rd = RUNS / a["tag"]
        if (rd / "spectra.npz").exists():
            continue
        mask = np.load(ROOT / a["mask"]) if a.get("mask") else None
        jobs.append((Config(**cfg), rd, str(ROOT / a["ckpt"]), mask))

    for i in range(0, len(jobs), args.batch):
        ch = jobs[i:i + args.batch]
        print(f"batch {i // args.batch}: {len(ch)} runs", flush=True)
        train_batched([c for c, *_ in ch], [rd for _, rd, *_ in ch],
                      init_from=[ck for _, _, ck, _ in ch],
                      masks_override=[m for *_, m in ch],
                      compile_mode="off" if args.smoke else args.compile,
                      spectra_every=5)
        for _, rd, *_ in ch:
            report(rd)

    if args.smoke:
        sys.path.insert(0, str(ROOT / "mirror"))
        from make_mirror import freq_image
        # token perm x -> g*x maps frequency k -> fold(g^-1 * k)
        gi = pow(man["g"], -1, 113)
        b0 = man["arms"][0]["base"]
        zo = np.load(RUNS / f"{b0}_c0_orig" / "spectra.npz")
        zt = np.load(RUNS / f"{b0}_twin" / "spectra.npz")
        perm = np.array([freq_image(gi, k) - 1 for k in range(1, 57)])
        Et = np.asarray(zt["energy"])          # (S, 56), twin frequency space
        Eo = np.asarray(zo["energy"])
        Et_back = np.empty_like(Et)
        Et_back[:, :] = Et[:, perm]            # twin energy at fold(g*k) -> k
        d = (np.linalg.norm(Et_back - Eo, axis=1)
             / np.linalg.norm(Eo, axis=1))
        assert d[0] < 1e-4, f"epoch-0 spectra not permutation-equal: {d[0]}"
        assert abs(zt["test_acc"][0] - zo["test_acc"][0]) < 1e-6
        print("SMOKE OK: epoch-0 twin spectra == g-permuted originals "
              f"(rel {d[0]:.2e})")
        eps = np.asarray(zo["epochs"])
        print("float-divergence growth (rel spectra distance):")
        for j in range(0, len(eps), max(1, len(eps) // 8)):
            print(f"  epoch {int(eps[j]):>4}: {d[j]:.3e}")


if __name__ == "__main__":
    main()
