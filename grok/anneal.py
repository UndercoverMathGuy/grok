"""Annealed distillation: walk the KD target along a path in logit space.

Run:  uv run python -m grok.anneal --student-run runs/seed37 --teacher-run runs/seed81 \\
          --run-name anneal_81_into_37 --schedule linear --num-steps 20

Direct distillation (grok/distill.py) breaks the student's frequencies almost
immediately — shrinking a circuit is downhill for both the KD loss and weight
decay — and then spends most of training rebuilding from a near-hash state.
Here the target is instead a *mixture* of two frozen logit tables, the
student's own final-checkpoint logits L_s and the teacher's L_t, whose
mixture weights move over training. Every intermediate target is a
mixed-frequency solution (each cos(w(a+b-c)) term peaks at c = a+b, so any
positive mixture does too), so the path never leaves the generalizing region
and the old circuits only lose their reward when the schedule says so.

Schedules (K = --num-steps targets; the epoch budget is split evenly):
    linear     target = (1-a) L_s + a L_t,  a: 1/K -> 1.
               Growth of teacher freqs is tied to destruction of student freqs.
    two-stage  grow:  target = L_s + a L_t,  a: -> 1  (old freqs held fixed)
               prune: target = (1-b) L_s + L_t,  b: -> 1
               Decouples circuit birth from circuit death, matching the order
               grokking itself uses: form under superposition, then clean up.

Pacing is either a fixed clock or gated. Without --gate-tol, the epoch budget
splits evenly across steps. With --gate-tol, each step (the last included)
runs until the gate metric drops below the tolerance — checked every
log_every epochs — and --num-epochs becomes a *global* cap so a step that
never converges can't run forever. --gate-metric picks what is watched:
    kd        the distillation loss against the current target
    train_ce  real-label CE on the train split (blind to internals *and* to
              the test set, which stays an untouched probe of the mechanism)
    test_ce   real-label CE on the test split
--final-epochs adds pure-teacher polish after the schedule finishes.

Output layout matches train.py/distill.py so all sweep/analysis code works,
plus anneal.json recording the spec and the realized step transitions, and
metrics.json gains kd_losses and the per-epoch target weights.
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
from .distill import (
    describe_teacher,
    kd_loss_fn,
    synthetic_teacher_logits,
    teacher_logit_table as run_logit_table,
)
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


def weight_schedule(kind, num_steps):
    """[(w_student, w_teacher)] target mixtures, excluding the trivial (1, 0)."""
    if kind == "linear":
        return [(1.0 - a, a) for a in np.linspace(0.0, 1.0, num_steps + 1)[1:]]
    if kind == "two-stage":
        n_grow = (num_steps + 1) // 2
        n_prune = num_steps - n_grow
        grow = [(1.0, a) for a in np.linspace(0.0, 1.0, n_grow + 1)[1:]]
        prune = [(1.0 - b, 1.0) for b in np.linspace(0.0, 1.0, n_prune + 1)[1:]]
        return grow + prune
    raise ValueError(f"unknown schedule {kind}")


def anneal(
    student_run,
    run_dir,
    teacher_run=None,
    teacher_freqs=None,
    teacher_scale=10.0,
    schedule="linear",
    num_steps=20,
    temp=10.0,
    mode="mse",
    alpha=0.0,
    num_epochs=40_000,
    final_epochs=0,
    gate_tol=None,
    gate_metric="kd",
    lr=None,
    weight_decay=None,
    distill_all=False,
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
    cfg.save(run_dir / "config.json")

    tokens, labels = make_dataset(cfg)
    labels_np = np.array(labels)
    is_train, is_test = train_test_split(cfg)
    fourier = Fourier(cfg.p)

    # Both path endpoints are frozen tables. Interpolating against the *live*
    # student would be degenerate: the target moves with the student and the
    # loss collapses to a down-weighted pull straight toward the teacher.
    student_np = run_logit_table(student_run, tokens)
    if teacher_run is not None:
        teacher_np = run_logit_table(teacher_run, tokens)
    else:
        teacher_np = synthetic_teacher_logits(cfg, teacher_freqs, teacher_scale)
    print("path start (frozen student logits) —", end=" ")
    describe_teacher(student_np, labels_np, fourier)
    print("path end   (teacher logits)        —", end=" ")
    describe_teacher(teacher_np, labels_np, fourier)

    # As in distill.py: distill on the train split by default so the test set
    # measures whether the mechanism (not the fitted outputs) moves.
    distill_mask = np.ones(len(labels_np), bool) if distill_all else is_train
    d_idx = mx.array(np.flatnonzero(distill_mask))
    d_tokens, d_labels = tokens[d_idx], labels[d_idx]
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
    ce = cross_entropy_f64 if cfg.loss_dtype == "f64" else cross_entropy

    steps = weight_schedule(schedule, num_steps)
    if gate_tol is None:
        epochs_per_step = max(unroll, num_epochs // len(steps) // unroll * unroll)
    else:
        epochs_per_step = num_epochs  # gated: each step runs until the gate;
        # num_epochs is a global cap shared across steps, enforced below.

    def make_train_steps(d_target):
        # Recompiled once per schedule step (the target table is baked into
        # the graph as a constant); ~num_steps compiles total, negligible.
        def kd_loss(model):
            return kd_loss_fn(model, d_tokens, d_target, d_labels, temp, mode, alpha)

        loss_and_grad = nn.value_and_grad(model, kd_loss)
        state = [model.state, optimizer.state]

        @partial(mx.compile, inputs=state, outputs=state)
        def train_steps():
            # As in train.py: losses are measured at the weights each epoch
            # starts with, before that epoch's update.
            kd_hist, train_hist, test_hist = [], [], []
            for _ in range(unroll):
                train_hist.append(loss_fn(model, train_tokens, train_labels, ce=ce))
                test_hist.append(loss_fn(model, test_tokens, test_labels, ce=ce))
                kd, grads = loss_and_grad(model)
                optimizer.update(model, grads)
                kd_hist.append(kd)
            return mx.stack(kd_hist), mx.stack(train_hist), mx.stack(test_hist)

        return train_steps, state

    spec = {
        "student_run": str(student_run),
        "teacher_run": str(teacher_run) if teacher_run else None,
        "teacher_freqs": list(teacher_freqs) if teacher_freqs else None,
        "teacher_scale": teacher_scale,
        "schedule": schedule,
        "num_steps": num_steps,
        "temp": temp,
        "mode": mode,
        "alpha": alpha,
        "gate_tol": gate_tol,
        "gate_metric": gate_metric,
        "final_epochs": final_epochs,
        "distill_all": distill_all,
    }
    (run_dir / "anneal.json").write_text(json.dumps(spec, indent=2))

    hists = {"kd_losses": [], "train_losses": [], "test_losses": []}
    w_student_hist, w_teacher_hist = [], []
    transitions = []
    epoch = 0
    t0 = time.time()
    for step_i, (w_s, w_t) in enumerate(steps):
        last = step_i == len(steps) - 1
        target_np = w_s * student_np + w_t * teacher_np
        d_target = mx.array(target_np[distill_mask].astype(np.float32))
        train_steps, state = make_train_steps(d_target)
        transitions.append({"epoch": epoch, "w_student": w_s, "w_teacher": w_t})
        print(f"step {step_i + 1:2d}/{len(steps)}  target = "
              f"{w_s:.2f}*student + {w_t:.2f}*teacher  (from epoch {epoch})")

        budget = epochs_per_step if gate_tol is not None else (
            epochs_per_step + (final_epochs if last else 0)
        )
        for _ in range(0, budget, unroll):
            if epoch >= num_epochs and gate_tol is not None:
                print(f"global cap {num_epochs} reached mid-schedule, stopping")
                break
            if epoch % cfg.save_every == 0:
                model.save_weights(str(ckpt_dir / f"epoch_{epoch:05d}.safetensors"))
                save_optim(optimizer, ckpt_dir, epoch)
            kd, tl, sl = train_steps()
            mx.async_eval(kd, tl, sl, state)
            hists["kd_losses"].append(kd)
            hists["train_losses"].append(tl)
            hists["test_losses"].append(sl)
            w_student_hist.extend([w_s] * unroll)
            w_teacher_hist.extend([w_t] * unroll)
            epoch += unroll

            if epoch % log_every == 0:
                eps = epoch / (time.time() - t0)
                print(
                    f"epoch {epoch:6d}  kd {kd[-1].item():.4e}  "
                    f"train {tl[-1].item():.4e}  test {sl[-1].item():.4e}  ({eps:.0f} epochs/s)"
                )
                if gate_tol is not None:
                    gate_val = {"kd": kd, "train_ce": tl, "test_ce": sl}[gate_metric][-1].item()
                    if gate_val < gate_tol:
                        print(f"  {gate_metric} {gate_val:.3e} below gate tol {gate_tol}, advancing")
                        break
        else:
            continue  # step exhausted its budget normally
        if epoch >= num_epochs and gate_tol is not None:
            break  # global cap: also exit the schedule loop

    model.save_weights(str(ckpt_dir / f"epoch_{epoch:05d}.safetensors"))
    save_optim(optimizer, ckpt_dir, epoch)
    spec["transitions"] = transitions
    (run_dir / "anneal.json").write_text(json.dumps(spec, indent=2))
    metrics = {k: np.concatenate([np.array(t) for t in v]).tolist() for k, v in hists.items()}
    metrics["w_student"] = w_student_hist
    metrics["w_teacher"] = w_teacher_hist
    (run_dir / "metrics.json").write_text(json.dumps(metrics))
    print(f"done: {epoch} epochs in {time.time() - t0:.0f}s -> {run_dir}")
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--student-run", required=True)
    parser.add_argument("--teacher-run", default=None)
    parser.add_argument("--teacher-freqs", default=None, help="e.g. 3,8,21,33,50")
    parser.add_argument("--teacher-scale", type=float, default=10.0)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--schedule", choices=["linear", "two-stage"], default="linear")
    parser.add_argument("--num-steps", type=int, default=20)
    parser.add_argument("--temp", type=float, default=10.0)
    parser.add_argument("--mode", choices=["kl", "mse"], default="mse")
    parser.add_argument("--alpha", type=float, default=0.0)
    parser.add_argument("--num-epochs", type=int, default=40_000)
    parser.add_argument("--final-epochs", type=int, default=0,
                        help="extra epochs on the pure-teacher target after the schedule")
    parser.add_argument("--gate-tol", type=float, default=None,
                        help="advance a step once the gate metric drops below this; "
                             "makes --num-epochs a global cap instead of a budget")
    parser.add_argument("--gate-metric", choices=["kd", "train_ce", "test_ce"], default="kd")
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--distill-all", action="store_true")
    args = parser.parse_args()

    freqs = [int(f) for f in args.teacher_freqs.split(",")] if args.teacher_freqs else None
    anneal(
        student_run=args.student_run,
        run_dir=Path("runs") / args.run_name,
        teacher_run=args.teacher_run,
        teacher_freqs=freqs,
        teacher_scale=args.teacher_scale,
        schedule=args.schedule,
        num_steps=args.num_steps,
        temp=args.temp,
        mode=args.mode,
        alpha=args.alpha,
        num_epochs=args.num_epochs,
        final_epochs=args.final_epochs,
        gate_tol=args.gate_tol,
        gate_metric=args.gate_metric,
        lr=args.lr,
        weight_decay=args.weight_decay,
        distill_all=args.distill_all,
    )
