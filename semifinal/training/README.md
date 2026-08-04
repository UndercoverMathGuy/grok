# SEMIFINAL training scripts

Cleaned, self-contained versions of the scripts that TRAIN the runs the
`../analysis/` scripts read. Only the runs directly required by the claims
in `../../SEMIFINAL.md` are covered. Originals (with their full lab-notebook
history) live in `findings/` and `scripts/`; these copies are portable (no
hardcoded home paths), idempotent (existing runs are skipped), and free of
analysis code — the readout batteries live in `../analysis/`.

**Sequential only.** MLX/Metal on this machine crashes under concurrent GPU
processes. Run one script at a time, and nothing else on the GPU. Each farm
takes minutes-to-hours; launch detached, e.g.:

```sh
nohup uv run python semifinal/training/train_doubleflat.py \
  > semifinal/training/doubleflat.log 2>&1 & disown
```

## Order and dependencies

1. `train_natural_farm.py` — the natural cohort (claim 1's baseline; the
   floor census). Draws primes/seeds from OS entropy on first run and pins
   them in `runs/farm_matrix_spec.json`. NOTE: reproduces the cohort
   *statistically*; the exact named runs in `runs/` were one draw of this
   procedure. Also see `scripts/farm_seeds.py` for the per-mask zoos
   (`runs/og_seed0`, `runs/seed{0,1,2}`) the surgical scripts use as bases.
2. `train_orthflat_farm.py` — orth-flat cohort: `embed_init=orthogonal`
   baseline + ten training-dynamics variants (claims 1 and 2). New script:
   the original cohort was launched with ad-hoc shell loops; the grid here
   is recovered from the runs' `config.json` files. `DRY_RUN=1` prints
   what would train instead of launching.
3. `train_doubleflat.py` — 16 double-flat runs (orthogonal W_E + isometric
   attention; claim 1's "erase both carriers" arm). Verifies T_k flatness
   numerically per seed before training.
4. `train_surgery.py` — original five-arm surgery on seed27058 (claims 3, 4).
5. `train_c2c5_surgery.py` — triple-implant + fine dose arms (claims 3, 4).
6. `train_tilt_transplant.py` — energy-spectrum transplants (claim 3's
   "energy alone fails" result; kept with its refuted pre-registration).
7. `train_dose_farm.py` — cross-mask dose-response curves (claim 3).
8. `train_gk_rotate.py` — alignment-knob (G_k) arms at fixed energy
   (claim 3's cleanest experiment).
9. `train_collision_farm.py` — engineered additive-trio repairs (claim 4).

Scripts 4–9 need specific natural base runs with epoch-0 checkpoints
(hardcoded at the top of each script, e.g. `runs/og_seed0/seed27058`).
They reproduce the *exact* runs used in SEMIFINAL.md given those bases; if
you regrow the natural zoo from scratch, point the `BASE`/`BASES` constants
at your new runs — every target-selection rule is derived from the base at
runtime, so the design carries over.

`_shared.py` holds the common helpers (committee detector, Fourier energy,
surgical checkpoint writer). Surgical init checkpoints are written to
`_ckpt/` (safe to delete after the runs exist).

Pre-registered predictions are kept in each docstring, including the two
that FAILED (transplant P-TR1, dose-farm P-B2) — the failures are part of
the story in SEMIFINAL.md and the scripts are reproduced as run, not as
wished.

## Known incomplete cells (as of 2026-08-03)

Running the farms resumes these — expect real GPU time, not a no-op:

- `train_natural_farm.py`: 4 of the 12 matrix slots were never finished
  (the farm was stopped early; the 8 completed runs already form a
  balanced sub-matrix). Resuming trains 4 x 30k epochs.
- `train_orthflat_farm.py`: `combined/p-113/seed2034/seed777` was
  interrupted in the original campaign (checkpoints exist, no
  spectra.npz); the farm will retrain it from scratch. All other 41 grid
  cells are complete and skip.

Every surgical script (4–9) verified 2026-08-03 to skip everything and
reproduce its logged RESULT lines exactly from the existing runs.
