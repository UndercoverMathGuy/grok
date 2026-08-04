"""Lockstep batched training: M runs trained simultaneously as one model.

Every weight gains a leading run axis (M, ...), the forward pass becomes
broadcast matmuls, and the total loss is the SUM of per-run mean CEs — losses
are separable across runs, so the joint gradient is exactly the stack of
per-run gradients, and elementwise AdamW on stacked weights is exactly M
independent AdamW runs in lockstep. Numerically this reproduces sequential
`grok.train.train` (same seeds, same init, same schedule) up to float
reordering.

Why: these models are tiny, so sequential training is dominated by per-op
dispatch (~25 eps/s). Batching M runs pays that cost once per batch.

Constraints (asserted): 1 layer, identical architecture/optimizer/epochs
across the batch, no tilt/CVaR/grad-noise. Runs may differ in init_seed,
data_seed (per-run train/test masks), embed/attn init, and warm-start
checkpoints (init_from) — everything that only touches the initial weights
or the data split.

Outputs are per-run and byte-compatible with grok.train.train: each
run_dirs[i] gets config.json, metrics.json, checkpoints/epoch_*.safetensors
(+ optim_*), and spectra.npz.
"""

import json
import math
import time
from functools import partial
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten, tree_map

from .config import Config
from .data import make_dataset, train_test_split
from .model import Transformer


def build_stacked(cfgs, init_from=None):
    """Init each run exactly as train() would (seed -> Transformer -> optional
    warm-start), then stack all parameters along a new leading axis into a
    single Transformer-shaped module."""
    trees = []
    for i, cfg in enumerate(cfgs):
        mx.random.seed(cfg.init_seed)
        m = Transformer(cfg)
        if init_from is not None and init_from[i] is not None:
            m.load_weights(str(init_from[i]))
        trees.append(m.parameters())
    model = Transformer(cfgs[0])  # tree structure + cfg; weights replaced
    model.update(tree_map(lambda *xs: mx.stack(xs), *trees))
    return model


def batched_final_logits(model, tokens):
    """Final-position logits for stacked weights.

    tokens: (N, T) shared across runs, or (M, N, T) per-run. Returns
    (M, N, d_vocab). Mirrors Transformer.final_logits for num_layers=1.
    """
    cfg = model.cfg
    h, dh = cfg.num_heads, cfg.d_head
    M = model.W_E.shape[0]
    T = tokens.shape[-1]
    E = model.W_E.transpose(0, 2, 1)  # (M, v, d)
    if tokens.ndim == 2:
        x = E[:, tokens]  # shared grid (snapshots)
    else:
        # per-run tokens: take_along_axis is ~2x faster fwd+bwd than
        # advanced indexing on Metal (see notes/mlx-metal-quirks)
        x = mx.take_along_axis(E, tokens.reshape(M, -1)[..., None], axis=1)
        x = x.reshape(M, *tokens.shape[1:], -1)
    x = x + model.W_pos[:, None, :T, :]  # (M, N, T, d)

    N = x.shape[1]
    at = model.blocks[0].attn

    # Every matmul below keeps M as the ONLY batch dim (one big gemm per
    # run): batch dims like (M, N) dispatch thousands of tiny gemms and
    # materialize the broadcast weight — ~40x slower on Metal. The tiny
    # per-position attention contractions (T=3, single query) are done as
    # elementwise multiply-reduce for the same reason.
    # NOTE: fusing K/V/Q into one gemm was tried and is ~20% SLOWER — the
    # per-epoch weight concat, wasted Q rows, and strided output slices cost
    # more than the two saved kernel launches (bench 2026-08-04).
    def proj(y, W):  # (M, R, d) @ per-run (h*dh, d) -> (M, R, h*dh)
        return y @ W.reshape(M, h * dh, -1).transpose(0, 2, 1)

    x_flat = x.reshape(M, N * T, -1)
    k = proj(x_flat, at.W_K).reshape(M, N, T, h, dh)
    v = proj(x_flat, at.W_V).reshape(M, N, T, h, dh)
    q = proj(x[:, :, -1], at.W_Q).reshape(M, N, 1, h, dh)  # "=" query: no mask
    scores = (k * q).sum(-1) / math.sqrt(dh)  # (M, N, T, h)
    attn = mx.softmax(scores, axis=2)
    z = (attn[..., None] * v).sum(axis=2).reshape(M, N, h * dh)
    xf = x[:, :, -1] + z @ at.W_O.transpose(0, 2, 1)  # (M, N, d)

    mlp = model.blocks[0].mlp
    pre = xf @ mlp.W_in.transpose(0, 2, 1) + mlp.b_in[:, None]
    xf = xf + mlp.act(pre) @ mlp.W_out.transpose(0, 2, 1) + mlp.b_out[:, None]
    return xf @ model.W_U  # (M, N, v)


def ce_stable_f32_per_run(logits, labels):
    """(M,) mean CE per run, float32 entirely on GPU, stable where naive f32
    log_softmax cancels: per-example CE = softplus(lse_{j!=y}(z_j) - z_y).

    The label logit never enters a cancelling subtraction — in the confident
    regime (true loss ~1e-7, where f32 log_softmax underflows to exactly 0
    and starves Adam, see train.cross_entropy_f64) this reduces to
    log1p(sum of tiny exps), which f32 evaluates to full *relative*
    precision. In the diffuse/wrong regime softplus(u) ~ u, also exact.
    Removes the per-epoch CPU-stream round-trip of the f64 path.
    """
    z_y = mx.take_along_axis(logits, labels[..., None], axis=-1)  # (M, N, 1)
    others = mx.where(labels[..., None] == mx.arange(logits.shape[-1]),
                      -mx.array(np.inf, logits.dtype), logits)
    u = mx.logsumexp(others, axis=-1, keepdims=True) - z_y
    return mx.logaddexp(u, 0.0).mean(axis=(1, 2))  # softplus, per run


def ce_f64_per_run(logits, labels):
    """(M,) mean CE per run, float64 on the CPU stream (see train.cross_entropy_f64)."""
    z = logits.astype(mx.float64, stream=mx.cpu)
    lse = mx.logsumexp(z, axis=-1, keepdims=True, stream=mx.cpu)
    lp = mx.subtract(z, lse, stream=mx.cpu)
    picked = mx.take_along_axis(lp, labels[..., None], axis=-1, stream=mx.cpu)
    per = mx.negative(mx.mean(picked, axis=(1, 2), stream=mx.cpu), stream=mx.cpu)
    return per.astype(mx.float32, stream=mx.cpu)


def _save_ckpts(model, run_dirs, epoch, optimizer=None):
    """Slice run i out of every stacked tensor -> per-run safetensors with the
    exact key names Transformer.save_weights produces."""
    flat = tree_flatten(model.parameters())
    M = len(run_dirs)
    for i, rd in enumerate(run_dirs):
        d = {k: v[i] for k, v in flat}
        mx.save_safetensors(str(rd / "checkpoints" / f"epoch_{epoch:05d}.safetensors"), d)
    if optimizer is not None:
        oflat = tree_flatten(optimizer.state)
        for i, rd in enumerate(run_dirs):
            d = {k: (v[i] if v.ndim >= 1 and v.shape[0] == M else v) for k, v in oflat}
            mx.save_safetensors(str(rd / "checkpoints" / f"optim_{epoch:05d}.safetensors"), d)


def train_batched(
    cfgs: list,
    run_dirs: list,
    log_every: int = 100,
    unroll: int = 10,
    init_from: list | None = None,
    spectra_every: int | None = None,
    fast_loss: bool = False,
):
    M = len(cfgs)
    assert M == len(run_dirs)
    c0 = cfgs[0]
    for c in cfgs:
        assert c.num_layers == 1, "batched forward is 1-layer only"
        for f in ("p", "fn_name", "frac_train", "d_model", "num_heads", "d_mlp",
                  "n_ctx", "act_type", "lr", "warmup_steps", "weight_decay",
                  "betas", "adam_eps", "loss_dtype", "num_epochs", "save_every",
                  "stopping_thresh"):
            assert getattr(c, f) == getattr(c0, f), f"batch must agree on {f}"
        assert c.loss_dtype == "f64"
        assert c.loss_tilt == 0.0 and c.loss_cvar == 0.0 and c.grad_noise == 0.0
    assert log_every % unroll == 0 and c0.save_every % unroll == 0
    assert spectra_every is None or spectra_every % unroll == 0

    run_dirs = [Path(rd) for rd in run_dirs]
    for cfg, rd in zip(cfgs, run_dirs):
        (rd / "checkpoints").mkdir(parents=True, exist_ok=True)
        cfg.save(rd / "config.json")

    # fast_loss uses the GPU-resident stable-f32 CE instead of the f64
    # CPU-stream one — same regime coverage, no per-epoch CPU stall.
    ce = ce_stable_f32_per_run if fast_loss else ce_f64_per_run
    # Cap MLX's free-buffer cache. Do NOT raise the wired limit to the full
    # recommended working set on an 8GB machine: the cache then wires ~5.7GB,
    # the rest of the system swaps (3.4GB swap, 10% free observed mid-run),
    # and throughput decays 82 -> 27 run-eps/s over ~20 min.
    mx.set_cache_limit(2 << 30)

    model = build_stacked(cfgs, init_from)
    optimizer = optim.AdamW(
        learning_rate=optim.linear_schedule(0.0, c0.lr, c0.warmup_steps),
        betas=list(c0.betas),
        eps=c0.adam_eps,
        weight_decay=c0.weight_decay,
    )

    # Shared full grid (same p/fn for the whole batch); per-run split masks.
    tokens, labels = make_dataset(c0)
    tokens_np, labels_np = np.array(tokens), np.array(labels)
    splits = [train_test_split(c) for c in cfgs]
    train_tokens = mx.array(np.stack([tokens_np[tr] for tr, _ in splits]))
    train_labels = mx.array(np.stack([labels_np[tr] for tr, _ in splits]))
    test_tokens = mx.array(np.stack([tokens_np[te] for _, te in splits]))
    test_labels = mx.array(np.stack([labels_np[te] for _, te in splits]))
    print(f"batch {M}:  train {train_tokens.shape[1]}  test {test_tokens.shape[1]}")

    spectra = None
    if spectra_every is not None:
        from .fourier import Fourier
        from . import metrics

        fourier = Fourier(c0.p)
        spectra = [
            {"epochs": [], "coeffs": [], "energy": [], "train_acc": [], "test_acc": []}
            for _ in range(M)
        ]

        def take_snapshot(epoch):
            # (M, p^2, p) final-position logits over the full grid, "=" dropped
            L = batched_final_logits(model, tokens)[:, :, :-1]
            mx.eval(L)
            L = np.array(L, dtype=np.float64)
            for i in range(M):
                coeffs, energy = metrics.freq_coeffs_and_energy(L[i], fourier)
                s = spectra[i]
                s["epochs"].append(epoch)
                s["coeffs"].append(coeffs)
                s["energy"].append(energy)
                s["train_acc"].append(metrics.accuracy(L[i], labels_np, splits[i][0]))
                s["test_acc"].append(metrics.accuracy(L[i], labels_np, splits[i][1]))

    def loss_fn(model, toks, labs):
        per = ce(batched_final_logits(model, toks), labs)
        return per.sum(), per

    loss_and_grad = nn.value_and_grad(model, loss_fn)
    state = [model.state, optimizer.state]

    @partial(mx.compile, inputs=state, outputs=state)
    def train_steps():
        # `unroll` epochs in one graph; the train loss is measured at each
        # epoch's starting weights, as in grok.train.train. Unlike the
        # sequential loop, the test loss is NOT computed here: its f64 CE
        # runs on the CPU stream and stalls the GPU every epoch (spiky
        # bandwidth/utilization at M=44) — and it feeds nothing but logging,
        # so it is evaluated only at log points below.
        train_losses = []
        for _ in range(unroll):
            (_, per), grads = loss_and_grad(model, train_tokens, train_labels)
            optimizer.update(model, grads)
            train_losses.append(per)
        return mx.stack(train_losses)  # (unroll, M)

    # Peak-memory guard: exceeding Metal's recommended working set does not
    # fail — it silently corrupts compiled-graph buffers (observed at M=44 on
    # an 8GB M1: exact-zero and exact-repeat loss rows, then dead runs).
    # Throughput also peaks around M=8-11 there (~70 run-eps/s) and falls off
    # beyond, so wider batches are all cost and no gain. Fail loudly instead.
    mx.reset_peak_memory()
    budget = 0.8 * mx.device_info()["max_recommended_working_set_size"]

    train_hist, test_hist = [], []  # test at log_every resolution, post-update
    t0 = time.time()
    for epoch in range(0, c0.num_epochs, unroll):
        if epoch % c0.save_every == 0:
            _save_ckpts(model, run_dirs, epoch, optimizer)
        if spectra is not None and epoch % spectra_every == 0:
            take_snapshot(epoch)
        tl = train_steps()
        mx.async_eval(tl, state)
        train_hist.append(tl)
        if epoch == 0:
            mx.eval(tl, state)
            peak = mx.get_peak_memory()
            assert peak < budget, (
                f"peak Metal memory {peak / 1e9:.2f} GB > {budget / 1e9:.2f} GB "
                f"budget — reduce batch width (M={M}) or training silently corrupts")

        if (epoch + unroll) % log_every == 0:
            te = np.array(ce(
                batched_final_logits(model, test_tokens), test_labels))
            test_hist.append(te)
            tr = np.array(train_hist[-1][-1])
            eps = (epoch + unroll) / (time.time() - t0)
            print(
                f"epoch {epoch + unroll:6d}  train {tr.mean():.4e}  "
                f"test mean {te.mean():.4e} max {te.max():.4e}  "
                f"({eps:.1f} eps/s x {M} = {eps * M:.0f} run-eps/s)", flush=True,
            )
            if te.max() < c0.stopping_thresh:
                print(f"all test losses below {c0.stopping_thresh}, stopping")
                break

    epoch = len(train_hist) * unroll
    _save_ckpts(model, run_dirs, epoch, optimizer)
    if spectra is not None:
        take_snapshot(epoch)
    train_all = np.concatenate([np.array(t) for t in train_hist])  # (E, M)
    test_all = np.stack(test_hist) if test_hist else np.zeros((0, M))
    for i, rd in enumerate(run_dirs):
        if spectra is not None:
            np.savez(rd / "spectra.npz", **{k: np.array(v) for k, v in spectra[i].items()})
        (rd / "metrics.json").write_text(json.dumps({
            "train_losses": train_all[:, i].tolist(),
            # test CE is subsampled (see train_steps): entry j is measured
            # after epoch (j+1)*test_every's update.
            "test_losses": test_all[:, i].tolist(),
            "test_every": log_every,
        }))
    print(f"done: {epoch} epochs x {M} runs in {time.time() - t0:.0f}s")
    return model
