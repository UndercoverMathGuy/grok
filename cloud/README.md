# SEMIFINAL v2 on a cloud GPU

Self-contained torch port of the batched lockstep trainer + the v2 protocol
(44 runs identical to `semifinal/training/train_semifinal_v2.py`). No MLX.

## Flow

1. Commit → GitHub Actions builds `ghcr.io/<owner>/<repo>/semifinal-trainer`
   (workflow: `.github/workflows/build-image.yml`; runs a CPU smoke test).
   Make the package public once (repo → Packages → settings) or `docker login
   ghcr.io` on the box.
2. Rent a CUDA box (RunPod / vast / Lambda — image needs driver ≥ 570 for a
   5090; any Ampere+ card works too) and:

```bash
docker run --gpus all \
  -e WANDB_API_KEY=<your key> \
  -e WANDB_PROJECT=grok-semifinal-v2 \
  -v /workspace/runs:/runs \
  ghcr.io/<owner>/<repo>/semifinal-trainer:latest
```

That's it. Every finished run is uploaded to wandb as an artifact
(type `grok-run`, name e.g. `orthWE__p-113__seed4811__seed61001`) containing
`config.json`, `metrics.json`, `spectra.npz`, and all checkpoints — formats
byte-compatible with the MLX pipeline, so `semifinal/analysis/` works on the
downloaded dirs unchanged. Scalars (train/test CE, run-eps/s) stream live.

Resume: the volume makes it idempotent (runs with `spectra.npz` are
skipped). Steering suites derive from the from-scratch bases automatically,
same pre-registered rules.

## Options

```
--dry-run            print the 44-run plan
--smoke              tiny CPU sanity check (used by CI)
--loss f64           original f64 CE instead of stable-f32 (default f32stable)
--compile reduce-overhead|max-autotune|off   (default: torch.compile default)
--width-scratch 24 --width-steer 10          lockstep widths
--tf32               ~2x matmuls, NOT numerically faithful — leave off
```

## Faithfulness notes

- AdamW matches MLX semantics (NO bias correction, decoupled wd) — the
  homeostat eps-floor results depend on this; torch.optim.AdamW would not
  reproduce them.
- Data splits are bit-identical to MLX (stdlib `random(data_seed)`).
- Init distributions match; RNG streams don't (torch vs mx) — each run is a
  fresh, valid realization of its seed. Same cohort rule as always: don't
  mix backends within a cohort.
