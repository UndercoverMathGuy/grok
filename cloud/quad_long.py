"""Quad-cell extension: same 24 seeds as sweep_dist's quad cell, 150k epochs
(3x the sweep) to let quad cleanup finish. Runs land in RUNS/quad150k/."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_semifinal_torch import Config, train_batched, RUNS
from sweep_dist import cells

seeds, kw = cells()["quad"]
cfgs = [Config(data_seed=s, init_seed=s, save_every=5000,
               num_epochs=150_000, **kw) for s in seeds]
dirs = [RUNS / "quad150k" / f"s{s}" for s in seeds]
train_batched(cfgs, dirs, compile_mode="default")
