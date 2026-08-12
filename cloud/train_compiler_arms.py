"""Train the compiler Phase A arms on a CUDA box (torch only, no MLX).

Reads compiler/arms/phaseA_manifest.json (paths remapped relative to the
repo root, so a fresh clone works anywhere), stacks all arms into one
lockstep batch, and trains with the verified torch port.

On the pod:
    git clone https://github.com/UndercoverMathGuy/grok.git && cd grok
    git lfs pull --include "compiler/arms/ckpts/*"
    pip install safetensors numpy torch  # (runpod pytorch image: already there)
    nohup python -u cloud/train_compiler_arms.py > phaseA.log 2>&1 &

Outputs land in runs_compiler/phaseA/<base>/<set>/ — byte-compatible with
the MLX pipeline; tar and pull back, then score locally with
compiler/score_arms.py.
"""

import argparse
import dataclasses
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "cloud"))

from train_semifinal_torch import Config, train_batched

CFG_FIELDS = {f.name for f in dataclasses.fields(Config)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(ROOT / "compiler" / "arms" /
                                              "phaseA_manifest.json"))
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--spectra-every", type=int, default=100)
    ap.add_argument("--loss", default="f32stable", choices=["f32stable", "f64"])
    ap.add_argument("--compile", dest="compile_mode", default="default")
    args = ap.parse_args()

    arms = json.loads(Path(args.manifest).read_text())
    todo = []
    for a in arms:
        # remap the manifest's absolute local paths onto this clone
        rd = ROOT / "runs_compiler" / Path(a["run_dir"]).relative_to(
            Path(a["run_dir"]).parents[2])
        ck = ROOT / "compiler" / "arms" / "ckpts" / Path(a["ckpt"]).name
        assert ck.exists(), f"missing ckpt {ck} (git lfs pull?)"
        assert ck.stat().st_size > 1000, f"{ck} is an LFS pointer, not data"
        if (rd / "spectra.npz").exists():
            print(f"skip (done): {rd}")
            continue
        d = {k: v for k, v in a["config"].items() if k in CFG_FIELDS}
        d["betas"] = tuple(d["betas"])
        cfg = Config(**d)
        if args.epochs:
            cfg.num_epochs = args.epochs
        todo.append((cfg, rd, str(ck)))

    for i in range(0, len(todo), args.batch):
        chunk = todo[i:i + args.batch]
        print(f"=== batch {i // args.batch}: {len(chunk)} runs, "
              f"{chunk[0][0].num_epochs} epochs", flush=True)
        train_batched([c for c, _, _ in chunk],
                      [r for _, r, _ in chunk],
                      init_from=[k for _, _, k in chunk],
                      spectra_every=args.spectra_every,
                      loss=args.loss, compile_mode=args.compile_mode)
    print("ALL DONE")


if __name__ == "__main__":
    main()
