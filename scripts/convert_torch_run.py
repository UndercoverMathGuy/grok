"""Convert the original torch `full_run_data.pth` into our run-dir format.

The original grokking-paper run data (state dict snapshots every 100 epochs
plus loss curves) is downloadable from Google Drive:

    gdown 12pmgxpTHLDzSNMbMCuAMXP1lE_XiCQRy -O large_files/full_run_data.pth

Then:  python scripts/convert_torch_run.py large_files/full_run_data.pth runs/original

Parameter names in the torch state dict ('embed.W_E', 'blocks.0.attn.W_K',
...) match our MLX module tree exactly, so conversion is just torch -> numpy
-> safetensors, dropping the attention-mask buffer.
"""

import json
import sys
from pathlib import Path

import mlx.core as mx
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from grok.config import Config


def convert(pth_path: Path, run_dir: Path):
    data = torch.load(pth_path, map_location="cpu")
    run_dir = Path(run_dir)
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    Config().save(run_dir / "config.json")  # original run used the defaults
    (run_dir / "metrics.json").write_text(
        json.dumps(
            {
                "train_losses": [float(x) for x in data["train_losses"]],
                "test_losses": [float(x) for x in data["test_losses"]],
            }
        )
    )

    epochs = data["epochs"]
    for epoch, sd in zip(epochs, data["state_dicts"]):
        weights = {
            k: mx.array(v.numpy()) for k, v in sd.items() if not k.endswith("mask")
        }
        mx.save_safetensors(str(ckpt_dir / f"epoch_{epoch:05d}.safetensors"), weights)
    print(f"wrote {len(epochs)} checkpoints -> {run_dir}")


if __name__ == "__main__":
    convert(Path(sys.argv[1]), Path(sys.argv[2]))
