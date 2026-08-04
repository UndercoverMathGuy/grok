"""Full-batch training with checkpointing.

Run:  uv run python -m grok.train [--run-name mainline] [--num-epochs 50000]

Produces runs/<run-name>/ containing:
    config.json                         — the Config used
    metrics.json                        — per-epoch train/test loss
    checkpoints/epoch_00000.safetensors — weights every save_every epochs
                                          (saved *before* that epoch's update,
                                          so epoch_00000 is the init)
    spectra.npz                         — with --spectra-every N: per-frequency
                                          phase-locked coeffs + energies of the
                                          full-grid logits every N epochs, plus
                                          train/test accuracy (the spectral
                                          logger; same before-update convention
                                          as checkpoints)

Performance notes (tuned for small models where per-op dispatch dominates):
  - one compiled graph computes the test loss, train loss, and update
  - `unroll` epochs are traced into a single compiled call
  - losses are accumulated lazily and only synced every log_every epochs
"""

import argparse
import json
import math
import time
from functools import partial
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_map

from .config import Config
from .data import make_dataset, train_test_split
from .model import Transformer


def cross_entropy(logits, labels):
    """Mean CE over the batch in float32, computed stably via logsumexp.

    Warning: float32 log_softmax underflows to exactly 0 once the model is
    extremely confident (true loss ~1e-7). Training on this starves Adam's
    second moment and causes periodic loss spikes (slingshots) — prefer
    cross_entropy_f64 (Config.loss_dtype='f64', the default) for training.
    """
    logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    return -mx.take_along_axis(logprobs, labels[:, None], axis=-1).mean()


def cross_entropy_f64(logits, labels, tilt=0.0, cvar=0.0):
    """Mean CE in float64, matching the original torch codebase's precision.

    MLX supports float64 only on the CPU stream, so every op here pins
    stream=mx.cpu; gradients flow back through the dtype cast into the
    float32 GPU graph. The tensors involved are tiny (batch x p), so the
    CPU round-trip costs little.

    tilt > 0 gives tilted ERM, (1/t) log mean exp(t * ce_i): a smooth
    interpolation from mean CE (t -> 0) to the max per-example CE (t -> inf),
    i.e. increasing pressure on the lowest-margin examples.
    """
    z = logits.astype(mx.float64, stream=mx.cpu)
    lse = mx.logsumexp(z, axis=-1, keepdims=True, stream=mx.cpu)
    logprobs = mx.subtract(z, lse, stream=mx.cpu)
    picked = mx.take_along_axis(logprobs, labels[:, None], axis=-1, stream=mx.cpu)
    if cvar > 0.0:
        # CVaR-alpha: mean CE over the worst ceil(alpha*n) examples. Gradient
        # flows only into that cohort (sort is a gather), so every step is a
        # vote by the hardest examples — no single-example thrash.
        per_ex = mx.negative(mx.squeeze(picked, -1, stream=mx.cpu), stream=mx.cpu)  # (batch,)
        k = max(1, int(round(cvar * per_ex.shape[0])))
        srt = mx.sort(per_ex, stream=mx.cpu)  # ascending
        # Explicit cpu-stream take: a bare srt[-k:] slice would dispatch on
        # the default (GPU) device, where float64 does not exist.
        n = per_ex.shape[0]
        worst = mx.take(srt, mx.arange(n - k, n), stream=mx.cpu)
        loss = mx.mean(worst, stream=mx.cpu)
    elif tilt > 0.0:
        per_ex = mx.negative(picked, stream=mx.cpu)  # (batch, 1) ce_i >= 0
        n = math.log(per_ex.shape[0])
        lse_t = mx.logsumexp(mx.multiply(per_ex, tilt, stream=mx.cpu), stream=mx.cpu)
        loss = mx.divide(mx.subtract(lse_t, n, stream=mx.cpu), tilt, stream=mx.cpu)
    else:
        loss = mx.negative(mx.mean(picked, stream=mx.cpu), stream=mx.cpu)
    # Return float32 so downstream GPU ops (stacking, logging) accept it; the
    # value ~1e-7 is representable — only the *computation* needed float64.
    return loss.astype(mx.float32, stream=mx.cpu)


def loss_fn(model, tokens, labels, ce=cross_entropy):
    return ce(model.final_logits(tokens), labels)


def train(
    cfg: Config,
    run_dir: Path,
    log_every: int = 100,
    unroll: int = 10,
    init_from=None,
    spectra_every: int | None = None,
):
    assert log_every % unroll == 0 and cfg.save_every % unroll == 0
    assert spectra_every is None or spectra_every % unroll == 0
    run_dir = Path(run_dir)
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    cfg.save(run_dir / "config.json")

    mx.random.seed(cfg.init_seed)
    model = Transformer(cfg)
    if init_from is not None:  # warm-start from a checkpoint (fresh Adam state)
        model.load_weights(str(init_from))
    # Linear warmup over the first warmup_steps epochs (step 0 runs at lr=0),
    # matching the original's LambdaLR(lambda step: min(step/10, 1)).
    optimizer = optim.AdamW(
        learning_rate=optim.linear_schedule(0.0, cfg.lr, cfg.warmup_steps),
        betas=list(cfg.betas),
        eps=cfg.adam_eps,
        weight_decay=cfg.weight_decay,
    )
    ce = cross_entropy_f64 if cfg.loss_dtype == "f64" else cross_entropy
    if cfg.loss_tilt > 0.0 or cfg.loss_cvar > 0.0:
        assert cfg.loss_dtype == "f64", "tilt/CVaR are only implemented in f64"
        assert not (cfg.loss_tilt > 0.0 and cfg.loss_cvar > 0.0), "pick one of tilt/cvar"
        ce = partial(cross_entropy_f64, tilt=cfg.loss_tilt, cvar=cfg.loss_cvar)

    tokens, labels = make_dataset(cfg)
    is_train, is_test = train_test_split(cfg)
    train_idx, test_idx = mx.array(np.flatnonzero(is_train)), mx.array(np.flatnonzero(is_test))
    train_tokens, train_labels = tokens[train_idx], labels[train_idx]
    test_tokens, test_labels = tokens[test_idx], labels[test_idx]
    print(f"train: {train_tokens.shape[0]}  test: {test_tokens.shape[0]}")

    spectra = None
    if spectra_every is not None:
        from .fourier import Fourier
        from . import metrics

        fourier = Fourier(cfg.p)
        labels_np = np.array(labels)
        spectra = {"epochs": [], "coeffs": [], "energy": [], "train_acc": [], "test_acc": []}

        def take_snapshot(epoch):
            logits = metrics.all_logits(model, tokens)
            coeffs, energy = metrics.freq_coeffs_and_energy(logits, fourier)
            spectra["epochs"].append(epoch)
            spectra["coeffs"].append(coeffs)
            spectra["energy"].append(energy)
            spectra["train_acc"].append(metrics.accuracy(logits, labels_np, is_train))
            spectra["test_acc"].append(metrics.accuracy(logits, labels_np, is_test))

    loss_and_grad = nn.value_and_grad(model, partial(loss_fn, ce=ce))
    state = [model.state, optimizer.state]
    if cfg.grad_noise > 0.0:
        # mx.random is used inside the compiled graph; its key must be part
        # of the compile state or every call would reuse the same noise.
        state.append(mx.random.state)

    @partial(mx.compile, inputs=state, outputs=state)
    def train_steps(sigma):
        # `unroll` epochs in one graph. Both losses are measured at the
        # weights each epoch *starts* with, i.e. before that epoch's update.
        # sigma > 0 is the SAM-lite path: gradients at w + sigma*eps (eps
        # scaled to each tensor's rms), the update applied at the clean w.
        # The logged train loss is then the loss at the *perturbed* weights.
        train_losses, test_losses = [], []
        for _ in range(unroll):
            test_losses.append(loss_fn(model, test_tokens, test_labels, ce=ce))
            if cfg.grad_noise > 0.0:
                params = model.trainable_parameters()
                perturbed = tree_map(
                    lambda p: p + sigma * mx.sqrt(mx.mean(p * p)) * mx.random.normal(p.shape),
                    params,
                )
                model.update(perturbed)
                loss, grads = loss_and_grad(model, train_tokens, train_labels)
                model.update(params)
            else:
                loss, grads = loss_and_grad(model, train_tokens, train_labels)
            optimizer.update(model, grads)
            train_losses.append(loss)
        return mx.stack(train_losses), mx.stack(test_losses)

    train_hist, test_hist = [], []
    t0 = time.time()
    for epoch in range(0, cfg.num_epochs, unroll):
        if epoch % cfg.save_every == 0:
            model.save_weights(str(ckpt_dir / f"epoch_{epoch:05d}.safetensors"))
            save_optim(optimizer, ckpt_dir, epoch)
        if spectra is not None and epoch % spectra_every == 0:
            take_snapshot(epoch)
        sigma = 0.0
        if cfg.grad_noise > 0.0 and cfg.grad_noise_until > 0:
            sigma = cfg.grad_noise * max(0.0, 1.0 - epoch / cfg.grad_noise_until)
        tl, sl = train_steps(mx.array(sigma, mx.float32))
        mx.async_eval(tl, sl, state)
        train_hist.append(tl)
        test_hist.append(sl)

        if (epoch + unroll) % log_every == 0:
            train_loss, test_loss = train_hist[-1][-1].item(), test_hist[-1][-1].item()
            eps = (epoch + unroll) / (time.time() - t0)
            print(
                f"epoch {epoch + unroll:6d}  train {train_loss:.4e}  "
                f"test {test_loss:.4e}  ({eps:.0f} epochs/s)"
            )
            if test_loss < cfg.stopping_thresh:
                print(f"test loss below {cfg.stopping_thresh}, stopping")
                break

    epoch = len(train_hist) * unroll
    model.save_weights(str(ckpt_dir / f"epoch_{epoch:05d}.safetensors"))
    save_optim(optimizer, ckpt_dir, epoch)
    if spectra is not None:
        take_snapshot(epoch)
        np.savez(run_dir / "spectra.npz", **{k: np.array(v) for k, v in spectra.items()})
    (run_dir / "metrics.json").write_text(
        json.dumps(
            {
                "train_losses": np.concatenate([np.array(t) for t in train_hist]).tolist(),
                "test_losses": np.concatenate([np.array(t) for t in test_hist]).tolist(),
            }
        )
    )
    print(f"done: {epoch} epochs in {time.time() - t0:.0f}s -> {run_dir}")
    return model


def save_optim(optimizer, ckpt_dir: Path, epoch: int):
    """Save optimizer state next to a weight checkpoint (optim_XXXXX.safetensors)."""
    from mlx.utils import tree_flatten

    flat = dict(tree_flatten(optimizer.state))
    mx.save_safetensors(str(Path(ckpt_dir) / f"optim_{epoch:05d}.safetensors"), flat)


def load_optim(optimizer, run_dir: Path):
    """Warm-start an optimizer from a run's latest saved state, if any.

    Returns the epoch the state was saved at, or None if the run predates
    optimizer checkpointing. The caller's configured learning rate wins over
    the stored one. Call this *before* building any compiled graph that
    captures optimizer.state — loading replaces the state dict object.
    """
    from mlx.utils import tree_unflatten

    paths = sorted(Path(run_dir).glob("checkpoints/optim_*.safetensors"))
    if not paths:
        return None
    lr = optimizer.learning_rate
    optimizer.state = tree_unflatten(list(mx.load(str(paths[-1])).items()))
    optimizer.learning_rate = lr
    return int(paths[-1].stem.split("_")[1])


def checkpoint_epochs(run_dir: Path):
    """Sorted (epoch, path) pairs for every checkpoint in a run."""
    paths = sorted(Path(run_dir).glob("checkpoints/epoch_*.safetensors"))
    return [(int(p.stem.split("_")[1]), p) for p in paths]


def load_run(run_dir: Path):
    """Load a run's config, final-checkpoint model, and per-epoch losses."""
    run_dir = Path(run_dir)
    cfg = Config.load(run_dir / "config.json")
    model = Transformer(cfg)
    model.load_weights(str(checkpoint_epochs(run_dir)[-1][1]))
    metrics = json.loads((run_dir / "metrics.json").read_text())
    return cfg, model, metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", default="mainline")
    parser.add_argument("--p", type=int, default=None)
    parser.add_argument("--num-epochs", type=int, default=None)
    parser.add_argument("--init-seed", type=int, default=None)
    parser.add_argument("--embed-init", choices=["normal", "orthogonal"], default=None)
    parser.add_argument("--loss-tilt", type=float, default=None)
    parser.add_argument("--loss-cvar", type=float, default=None)
    parser.add_argument("--grad-noise", type=float, default=None)
    parser.add_argument("--grad-noise-until", type=int, default=None)
    parser.add_argument("--data-seed", type=int, default=None)
    parser.add_argument("--unroll", type=int, default=10)
    parser.add_argument("--loss-dtype", choices=["f32", "f64"], default=None)
    parser.add_argument("--adam-eps", type=float, default=None)
    parser.add_argument("--init-from", default=None, help="checkpoint to warm-start from")
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--save-every", type=int, default=None)
    parser.add_argument(
        "--spectra-every", type=int, default=None,
        help="log per-frequency coeffs/energies to spectra.npz every N epochs",
    )
    args = parser.parse_args()

    cfg = Config()
    if args.p is not None:
        cfg.p = args.p
    if args.num_epochs is not None:
        cfg.num_epochs = args.num_epochs
    if args.init_seed is not None:
        cfg.init_seed = args.init_seed
    if args.embed_init is not None:
        cfg.embed_init = args.embed_init
    if args.loss_tilt is not None:
        cfg.loss_tilt = args.loss_tilt
    if args.loss_cvar is not None:
        cfg.loss_cvar = args.loss_cvar
    if args.grad_noise is not None:
        cfg.grad_noise = args.grad_noise
    if args.grad_noise_until is not None:
        cfg.grad_noise_until = args.grad_noise_until
    if args.data_seed is not None:
        cfg.data_seed = args.data_seed
    if args.loss_dtype is not None:
        cfg.loss_dtype = args.loss_dtype
    if args.adam_eps is not None:
        cfg.adam_eps = args.adam_eps
    if args.lr is not None:
        cfg.lr = args.lr
    if args.weight_decay is not None:
        cfg.weight_decay = args.weight_decay
    if args.save_every is not None:
        cfg.save_every = args.save_every
    train(
        cfg,
        Path("runs") / args.run_name,
        unroll=args.unroll,
        init_from=args.init_from,
        spectra_every=args.spectra_every,
    )
