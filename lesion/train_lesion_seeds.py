"""Shared seed draws for the lesion-family cohorts (META_SEED=20260815,
identical to transplant/degrok donor draws)."""

import random

META_SEED = 20260815


def draw_seeds(n_masks=2, n_inits=4):
    rng = random.Random(META_SEED)
    data_seeds = sorted(rng.sample(range(20_000, 30_000), n_masks))
    init_seeds = sorted(rng.sample(range(100_000, 1_000_000), n_inits))
    return data_seeds, init_seeds
