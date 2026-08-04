"""Hyperparameters for the mainline modular-addition grokking run."""

import json
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class Config:
    # Task: predict fn(a, b) from the sequence [a, b, =]. Token p is "=".
    p: int = 113
    fn_name: str = "add"  # 'add' | 'subtract' | 'x2xyy2'
    frac_train: float = 0.3
    # Two seeds so experiments can vary init while holding the train/test
    # split fixed (e.g. teacher vs student in the distillation conflict runs).
    data_seed: int = 0  # train/test split (Python random)
    init_seed: int = 0  # weight init (mx.random)

    # Architecture (mainline: 1 layer, no LayerNorm, no biases in attention)
    num_layers: int = 1
    d_model: int = 128
    # 'orthogonal' QR-orthonormalizes W_E's columns (needs d_model >= d_vocab):
    # every unit direction in vocab space gets energy exactly 1, so the
    # per-frequency init tilt (the committee lottery) is identically zero.
    # No Fourier analysis involved — flatness holds in every basis at once.
    embed_init: str = "normal"  # 'normal' | 'orthogonal'
    # 'isometric' makes the 4 heads' value subspaces jointly tile d_model:
    # stack the per-head W_V into one (h*d_head, d_model) matrix, QR it, and
    # set W_O to its transpose. Every direction of the residual stream then
    # passes through the head ENSEMBLE with total energy exactly 1, so with
    # embed_init='orthogonal' the per-frequency OV-transmitted energy (the
    # init-lottery ticket T_k) is identically flat. W_Q/W_K stay Gaussian.
    attn_init: str = "normal"  # 'normal' | 'isometric'
    num_heads: int = 4
    d_mlp: int = 512
    n_ctx: int = 3
    act_type: str = "ReLU"  # 'ReLU' | 'GeLU'

    # Optimization (full-batch AdamW; the large weight decay drives grokking)
    lr: float = 1e-3
    warmup_steps: int = 10  # linear LR warmup: factor min(step/warmup_steps, 1)
    weight_decay: float = 1.0
    betas: tuple = (0.9, 0.98)
    adam_eps: float = 1e-8
    # 'f64' computes the CE loss in float64 on the CPU stream. float32 CE
    # underflows to exactly 0 once the model is confident (true loss ~1e-7),
    # which starves Adam's second moment and causes periodic loss spikes
    # (slingshots). See train.cross_entropy_f64.
    loss_dtype: str = "f64"  # 'f32' | 'f64'
    # Tilted ERM: loss = (1/t)(log mean exp(t * ce_i)). t=0 is plain mean CE;
    # larger t upweights the worst examples, a smooth proxy for max-min-margin.
    # Frequency-blind: no reference to model internals, just example weighting.
    loss_tilt: float = 0.0
    # CVaR: mean CE over the worst loss_cvar fraction of examples (0 = off).
    # Stronger worst-case pressure than mild tilt, but a whole cohort votes
    # each step — none of tilt-15's single-example gradient thrash.
    loss_cvar: float = 0.0
    # SAM-lite: gradients are taken at w + sigma*eps (eps ~ N(0, rms(w)^2) per
    # tensor), the step applied at w. Selects for solutions whose loss — hence
    # margin — survives weight perturbation. sigma anneals linearly to 0 at
    # grad_noise_until, so late training / grokking runs clean.
    grad_noise: float = 0.0
    grad_noise_until: int = 0
    num_epochs: int = 50_000
    save_every: int = 100  # checkpoint interval, in epochs
    stopping_thresh: float = -1.0  # stop early if test loss drops below this

    @property
    def d_vocab(self) -> int:
        return self.p + 1  # 0..p-1 plus the "=" token

    @property
    def d_head(self) -> int:
        assert self.d_model % self.num_heads == 0
        return self.d_model // self.num_heads

    def fn(self, a, b):
        """The target function on Z_p (works on ints or numpy arrays)."""
        p = self.p
        return {
            "add": lambda: (a + b) % p,
            "subtract": lambda: (a - b) % p,
            "x2xyy2": lambda: (a**2 + a * b + b**2) % p,
        }[self.fn_name]()

    def save(self, path: Path):
        d = asdict(self)
        d["betas"] = list(d["betas"])
        Path(path).write_text(json.dumps(d, indent=2))

    @classmethod
    def load(cls, path: Path) -> "Config":
        d = json.loads(Path(path).read_text())
        d["betas"] = tuple(d["betas"])
        if "seed" in d:  # configs saved before the data/init seed split
            seed = d.pop("seed")
            d.setdefault("data_seed", seed)
            d.setdefault("init_seed", seed)
        return cls(**d)
