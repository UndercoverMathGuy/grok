"""Train the compiled arms from a manifest, batched on MLX (M<=8 lockstep).

Sequential batches only (Metal crashes under concurrent GPU processes);
launch detached:  nohup uv run python compiler/train_arms.py > phaseA.log &
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grok.batched import train_batched
from grok.config import Config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(ROOT / "compiler" / "arms" /
                                              "phaseA_manifest.json"))
    ap.add_argument("--batch", type=int, default=8, help="lockstep width M")
    ap.add_argument("--spectra-every", type=int, default=100)
    ap.add_argument("--epochs", type=int, default=None,
                    help="override manifest num_epochs (smoke tests)")
    ap.add_argument("--out-root", default=None,
                    help="override run_dir root (smoke tests)")
    args = ap.parse_args()

    arms = json.loads(Path(args.manifest).read_text())
    todo = []
    for a in arms:
        rd = Path(a["run_dir"])
        if args.out_root:
            rd = Path(args.out_root) / rd.relative_to(ROOT / "runs_compiler")
        if (rd / "spectra.npz").exists():
            print(f"skip (done): {rd}")
            continue
        cfg = Config(**{k: (tuple(v) if k == "betas" else v)
                        for k, v in a["config"].items()})
        if args.epochs:
            cfg.num_epochs = args.epochs
        todo.append((cfg, rd, a["ckpt"]))

    for i in range(0, len(todo), args.batch):
        chunk = todo[i:i + args.batch]
        cfgs = [c for c, _, _ in chunk]
        rds = [r for _, r, _ in chunk]
        cks = [k for _, _, k in chunk]
        print(f"=== batch {i // args.batch}: {len(chunk)} runs, "
              f"{cfgs[0].num_epochs} epochs")
        for r in rds:
            print(f"    {r}")
        train_batched(cfgs, rds, init_from=cks, fast_loss=True,
                      spectra_every=args.spectra_every)


if __name__ == "__main__":
    main()
