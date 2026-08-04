"""Knowledge distillation between grokked models.

Run:  uv run python -m grok.distill --student-run runs/s1 --teacher-run runs/mainline \\
          --run-name distill_m_to_s1 --temp 10

The student is initialized from its run's final checkpoint; the teacher is
frozen. Because the input space is finite (p^2 sequences), the teacher's
logits are precomputed once as a table — distillation training has no teacher
forward at all, so an epoch costs about the same as normal training.

Instead of --teacher-run you can pass --teacher-freqs 3,8,21,33,50 to distill
toward a *synthetic* teacher: idealized Fourier-algorithm logits
scale * sum_k cos(w_k(a+b-c)) for hand-chosen frequencies. This isolates
"change of frequency set" from "whatever noise a trained teacher carries".

The KD loss is computed over the p answer logits only (the "=" logit is never
a target and carries no frequency structure). Two modes:
    kl  — T^2 * KL(softmax(teacher/T) || softmax(student/T)), the standard
          Hinton loss; high T highlights the sub-dominant logit structure
    mse — mean squared error between row-centered logits, the T -> inf limit
--alpha mixes in hard-label cross entropy: (1-a)*kd + a*ce.

Output run dir has the same layout as train.py (so all analysis/sweep code
works on it) plus a distill.json recording the teacher spec, and metrics.json
gains a per-epoch kd_losses curve alongside the real-label train/test losses.
"""

import argparse
import json
import time
from functools import partial
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

from .config import Config
from .data import make_dataset, train_test_split
from .fourier import Fourier
from .model import Transformer
from .train import (
    checkpoint_epochs,
    cross_entropy,
    cross_entropy_f64,
    load_optim,
    loss_fn,
    save_optim,
)
from . import metrics as M


def synthetic_teacher_logits(cfg: Config, freqs, scale=10.0):
    """(p^2, p) idealized grokked logits: scale * sum_k cos(w_k(a+b-c))."""
    p = cfg.p
    a = np.arange(p)[:, None, None]
    b = np.arange(p)[None, :, None]
    c = np.arange(p)[None, None, :]
    logits = sum(np.cos(2 * np.pi * k * (a + b - c) / p) for k in freqs)
    return (scale * logits).reshape(p * p, p)


def teacher_logit_table(run_dir, tokens):
    """(p^2, p) float64 final-position logits of a run's last checkpoint."""
    cfg = Config.load(Path(run_dir) / "config.json")
    model = Transformer(cfg)
    model.load_weights(str(checkpoint_epochs(run_dir)[-1][1]))
    return M.all_logits(model, tokens)


def describe_teacher(teacher_logits, labels, fourier, top_k=8):
    ce = M.cross_entropy_high_precision(teacher_logits, labels)
    acc = M.accuracy(teacher_logits, labels)
    coeffs = M.cos_w_product_a_plus_b_minus_c_coeffs(teacher_logits, fourier)
    top = np.argsort(-np.abs(coeffs))[:top_k] + 1
    print(f"teacher: ce {ce:.3e}  acc {acc:.4f}  top freqs {top.tolist()}")


def kd_loss_fn(model, tokens, teacher, labels, temp, mode, alpha):
    student = model.final_logits(tokens)[:, :-1]  # drop the "=" logit
    if mode == "kl":
        s = student / temp
        t = teacher / temp
        s_logp = s - mx.logsumexp(s, axis=-1, keepdims=True)
        t_logp = t - mx.logsumexp(t, axis=-1, keepdims=True)
        kd = temp**2 * (mx.exp(t_logp) * (t_logp - s_logp)).sum(axis=-1).mean()
    elif mode == "mse":
        s = student - student.mean(axis=-1, keepdims=True)
        t = teacher - teacher.mean(axis=-1, keepdims=True)
        kd = ((s - t) ** 2).mean()
    else:
        raise ValueError(f"unknown mode {mode}")
    if alpha > 0:
        kd = (1 - alpha) * kd + alpha * cross_entropy(student, labels)
    return kd


def distill(
    student_run,
    run_dir,
    teacher_run=None,
    teacher_freqs=None,
    teacher_scale=10.0,
    temp=10.0,
    mode="kl",
    alpha=0.0,
    num_epochs=20_000,
    lr=None,
    weight_decay=None,
    distill_all=False,
    save_every=None,
    log_every=100,
    unroll=10,
):
    assert (teacher_run is None) != (teacher_freqs is None), "give exactly one teacher"
    assert log_every % unroll == 0

    run_dir = Path(run_dir)
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    cfg = Config.load(Path(student_run) / "config.json")
    cfg.num_epochs = num_epochs
    if lr is not None:
        cfg.lr = lr
    if weight_decay is not None:
        cfg.weight_decay = weight_decay
    if save_every is not None:
        cfg.save_every = save_every
    assert cfg.save_every % unroll == 0
    cfg.save(run_dir / "config.json")

    tokens, labels = make_dataset(cfg)
    labels_np = np.array(labels)
    is_train, is_test = train_test_split(cfg)
    fourier = Fourier(cfg.p)

    if teacher_run is not None:
        teacher_np = teacher_logit_table(teacher_run, tokens)
    else:
        teacher_np = synthetic_teacher_logits(cfg, teacher_freqs, teacher_scale)
    describe_teacher(teacher_np, labels_np, fourier)

    # Distill on the train split by default so the test set measures whether
    # the *mechanism* (not just the fitted outputs) moves to teacher freqs.
    distill_mask = np.ones(len(labels_np), bool) if distill_all else is_train
    d_idx = mx.array(np.flatnonzero(distill_mask))
    d_tokens, d_labels = tokens[d_idx], labels[d_idx]
    d_teacher = mx.array(teacher_np[distill_mask].astype(np.float32))
    train_idx, test_idx = mx.array(np.flatnonzero(is_train)), mx.array(np.flatnonzero(is_test))
    train_tokens, train_labels = tokens[train_idx], labels[train_idx]
    test_tokens, test_labels = tokens[test_idx], labels[test_idx]

    model = Transformer(cfg)
    model.load_weights(str(checkpoint_epochs(student_run)[-1][1]))
    optimizer = optim.AdamW(
        learning_rate=cfg.lr, betas=list(cfg.betas), weight_decay=cfg.weight_decay
    )
    warm = load_optim(optimizer, student_run)
    print(f"optimizer: {'warm-started from epoch ' + str(warm) if warm is not None else 'cold'}")

    def kd_loss(model):
        return kd_loss_fn(model, d_tokens, d_teacher, d_labels, temp, mode, alpha)

    loss_and_grad = nn.value_and_grad(model, kd_loss)
    state = [model.state, optimizer.state]

    # f64 for the real-label monitor curves so they don't floor/underflow;
    # the KD loss itself is float32 (soft targets at high T never underflow).
    ce = cross_entropy_f64 if cfg.loss_dtype == "f64" else cross_entropy

    @partial(mx.compile, inputs=state, outputs=state)
    def train_steps():
        # As in train.py: all three losses are measured at the weights each
        # epoch starts with, before that epoch's update.
        kd_hist, train_hist, test_hist = [], [], []
        for _ in range(unroll):
            train_hist.append(loss_fn(model, train_tokens, train_labels, ce=ce))
            test_hist.append(loss_fn(model, test_tokens, test_labels, ce=ce))
            kd, grads = loss_and_grad(model)
            optimizer.update(model, grads)
            kd_hist.append(kd)
        return mx.stack(kd_hist), mx.stack(train_hist), mx.stack(test_hist)

    (run_dir / "distill.json").write_text(
        json.dumps(
            {
                "student_run": str(student_run),
                "teacher_run": str(teacher_run) if teacher_run else None,
                "teacher_freqs": list(teacher_freqs) if teacher_freqs else None,
                "teacher_scale": teacher_scale,
                "temp": temp,
                "mode": mode,
                "alpha": alpha,
                "distill_all": distill_all,
            },
            indent=2,
        )
    )

    hists = {"kd_losses": [], "train_losses": [], "test_losses": []}
    t0 = time.time()
    for epoch in range(0, num_epochs, unroll):
        if epoch % cfg.save_every == 0:
            model.save_weights(str(ckpt_dir / f"epoch_{epoch:05d}.safetensors"))
            save_optim(optimizer, ckpt_dir, epoch)
        kd, tl, sl = train_steps()
        mx.async_eval(kd, tl, sl, state)
        hists["kd_losses"].append(kd)
        hists["train_losses"].append(tl)
        hists["test_losses"].append(sl)

        if (epoch + unroll) % log_every == 0:
            eps = (epoch + unroll) / (time.time() - t0)
            print(
                f"epoch {epoch + unroll:6d}  kd {kd[-1].item():.4e}  "
                f"train {tl[-1].item():.4e}  test {sl[-1].item():.4e}  ({eps:.0f} epochs/s)"
            )

    epoch = num_epochs
    model.save_weights(str(ckpt_dir / f"epoch_{epoch:05d}.safetensors"))
    save_optim(optimizer, ckpt_dir, epoch)
    (run_dir / "metrics.json").write_text(
        json.dumps({k: np.concatenate([np.array(t) for t in v]).tolist() for k, v in hists.items()})
    )
    print(f"done: {epoch} epochs in {time.time() - t0:.0f}s -> {run_dir}")
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--student-run", required=True)
    parser.add_argument("--teacher-run", default=None)
    parser.add_argument("--teacher-freqs", default=None, help="e.g. 3,8,21,33,50")
    parser.add_argument("--teacher-scale", type=float, default=10.0)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--temp", type=float, default=10.0)
    parser.add_argument("--mode", choices=["kl", "mse"], default="kl")
    parser.add_argument("--alpha", type=float, default=0.0)
    parser.add_argument("--num-epochs", type=int, default=20_000)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--distill-all", action="store_true")
    parser.add_argument("--save-every", type=int, default=None)
    args = parser.parse_args()

    freqs = [int(f) for f in args.teacher_freqs.split(",")] if args.teacher_freqs else None
    distill(
        student_run=args.student_run,
        run_dir=Path("runs") / args.run_name,
        teacher_run=args.teacher_run,
        teacher_freqs=freqs,
        teacher_scale=args.teacher_scale,
        temp=args.temp,
        mode=args.mode,
        alpha=args.alpha,
        num_epochs=args.num_epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        distill_all=args.distill_all,
        save_every=args.save_every,
    )
