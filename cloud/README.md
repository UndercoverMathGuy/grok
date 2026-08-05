# SEMIFINAL v2 on a cloud GPU

Self-contained torch port of the batched lockstep trainer + the v2 protocol
(44 runs, same as `semifinal/training/train_semifinal_v2.py`). No MLX.

One deliberate difference: **50k epochs here vs 20k in the MLX script** —
cloud compute is cheap and committees still drift late in a few percent of
runs. Everything else is numerically equivalent (same init distributions,
bit-identical data splits, MLX AdamW semantics; verified by training a
shared init in both backends).

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

## What lands in wandb

Nothing is auto-named. The dashboard gets **45 readable entries**:

| entry | what it is |
|---|---|
| `semifinal-v2-p113-50k-<timestamp>` | driver: live run-eps/s + per-batch aggregates (tag `driver`) |
| `p-113/seed4811/seed61001` | one run **per training run**, named for its path |
| `orthWE/p-113/seed4811/seed61001` | grouped by family, `job_type` = cohort |
| `doubleflat/p-113/seed7207/seed72003` | (`normal` / `orth-flat` / `double-flat`) |
| `dosefarm/seed61001/dose_110` | … 44 of these |

Each per-run entry carries its own `Config` (seeds, `embed_init`,
`attn_init`, every hyperparameter), `train_ce`/`test_ce`/`train_acc`/
`test_acc` curves, `grok_epoch` + `committee` + `committee_size` in the
summary, and its own artifact (type `grok-run`, name
`orthWE__p-113__seed4811__seed61001`) holding `config.json`,
`metrics.json`, `spectra.npz` and all checkpoints — formats byte-compatible
with the MLX pipeline, so `semifinal/analysis/` works on the downloaded
dirs unchanged.

So you can sort by `grok_epoch`, filter to `cohort=double-flat`, or grep a
seed, without decoding anything. Per-run entries publish as each lockstep
batch finishes; `WANDB_NAME` overrides the driver's name.

Resume: the volume makes it idempotent (runs with `spectra.npz` are
skipped). Steering suites derive from the from-scratch bases automatically,
same pre-registered rules.

## Options

```
--dry-run            print the 98-run plan
--only REGEX         train only matching run names; the 54 claim-2A/3E arms
                     added after the first 44 are
                     --only 'dyn-|phase2-tilt|eff-G|transplant/'
--skip REGEX         never train matching run names
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
