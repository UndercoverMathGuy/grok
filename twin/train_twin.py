"""twin/train_twin.py — manifest-driven lockstep trainer for the twin suite.

One file, no framework.  Every job in a manifest is a dict describing a run:
its masks (data_seed + train fraction), where its parameters come from (a fresh
init seed, or a base checkpoint = a FORK), an optional edit applied at the fork
point, and the epoch window it covers.

    LOCKSTEP.  M runs are trained together: parameters stacked on a leading run
    axis, gradients via torch.func.vmap(grad(...)), and ONE torch.optim.AdamW
    over the stacked tensors.  NOTE: this is standard bias-corrected AdamW —
    prior cohorts (cloud/train_semifinal_torch.py) use MLX-semantics AdamW
    with NO bias correction, so twin epoch counts are NOT quantitatively
    comparable to earlier cohorts.  The twin suite is self-contained by
    design: every claim is a within-batch twin comparison.  AdamW applies lr / betas / eps / weight decay
    elementwise, so a single optimizer over stacked parameters is arithmetically
    identical to M independent optimizers as long as the runs share
    hyperparameters — which every run in this suite does.  Nothing about the
    batch couples the runs: the loss is a sum of per-run terms and the masks are
    per-run rows.

    BATCH GROUPS.  vmap kernels need not be bitwise identical to unbatched ones
    (gate G3).  The suite therefore never compares across batch sizes: each job
    carries a `group`, and a group is NEVER split across chunks, so a treatment
    arm and its unedited / epsilon=0 twin always sit in the same vmap batch and
    all within-cohort comparisons are within-batch.

    EPOCH CONVENTION.  A checkpoint written "at epoch e" is the state BEFORE the
    update of epoch e, and a fork with start_epoch = e replays epoch e onward.
    Metrics logged at epoch e are likewise pre-update at e.  A fork from e with
    no edit therefore reproduces the base's logs from e on, bitwise.

Usage:
    uv run --group convert python twin/train_twin.py --manifest twin/manifests/stage1.json \\
        --runs runs_twin --device cuda --width 40
"""
import argparse
import math
import re
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from torch.func import functional_call, grad, vmap

from core import (PARAM_KEYS, Recipe, SpectraOp, Transformer, apply_edit,
                  init_params, load_ckpt, make_dataset, per_ex_ce, read_json,
                  save_ckpt, setup_determinism, split_masks, write_json)


# --------------------------------------------------------------------------
# batching
# --------------------------------------------------------------------------

def batch_key(job):
    """Jobs may share a vmap batch only if their epoch window and cadences
    match.  Train FRACTION deliberately does NOT appear: the full-grid masked
    loss makes fraction a per-run mask row, which is exactly what lets a HOLD
    fork sit in the same batch as its same-data control."""
    return (job["start_epoch"], job["end_epoch"], job["log_every"],
            job["spectra_every"], tuple(job.get("ckpt_epochs", [])))


def chunk_jobs(jobs, width):
    """Pack whole groups into chunks of at most `width` lanes, never splitting a
    group, and never mixing incompatible epoch windows.

    When every group in a bucket is the same size (the normal case), the number
    of groups per chunk is reduced to a DIVISOR of the group count, so all
    chunks of a cohort have exactly the same width.  G3 fails, so batch width
    is part of a run's arithmetic: keeping it constant across a cohort means the
    cohort is internally comparable even outside a single group.
    """
    out = []
    by_key = OrderedDict()
    for j in jobs:
        by_key.setdefault(batch_key(j), []).append(j)
    for _, js in by_key.items():
        groups = OrderedDict()
        for j in js:
            groups.setdefault(j.get("group", j["name"]), []).append(j)
        gl = list(groups.values())
        sizes = {len(g) for g in gl}
        assert max(sizes) <= width, (
            f"group of {max(sizes)} lanes exceeds --width {width}; a group must "
            f"stay in one batch (G3 fallback rule)")
        if len(sizes) == 1:
            cap = max(1, width // sizes.pop())
            per = max(d for d in range(1, cap + 1) if len(gl) % d == 0)
            out += [[j for g in gl[i:i + per] for j in g]
                    for i in range(0, len(gl), per)]
        else:                                   # mixed group sizes: greedy
            cur = []
            for g in gl:
                if len(cur) + len(g) > width:
                    out.append(cur)
                    cur = []
                cur += g
            if cur:
                out.append(cur)
    return out


# --------------------------------------------------------------------------
# one lockstep batch
# --------------------------------------------------------------------------

def load_lane_params(job, recipe, runs_dir):
    """Parameters + optimizer state for one lane, with its edit applied."""
    src = job["init"]
    if src["kind"] == "fresh":
        w = init_params(recipe, job["init_seed"])
        opt = None
    elif src["kind"] == "ckpt":
        ck = load_ckpt(Path(runs_dir) / src["path"])
        assert int(ck["epoch"]) == int(job["start_epoch"]), (
            f"{job['name']}: ckpt at epoch {ck['epoch']} but job starts at "
            f"{job['start_epoch']}")
        w = {k: ck["weights"][k].clone() for k in PARAM_KEYS}
        opt = ck["opt"]
    else:
        raise ValueError(src["kind"])
    w, cert = apply_edit(w, job.get("edit", {"kind": "none"}), recipe.p)
    return w, opt, cert


def train_batch(jobs, recipe, runs_dir, device, compile_mode, verbose=True):
    """Train one chunk of jobs in lockstep.  Writes spectra.npz + config.json
    (+ checkpoints) into runs_dir/<name>/ for every lane."""
    M = len(jobs)
    runs_dir = Path(runs_dir)
    key = batch_key(jobs[0])
    assert all(batch_key(j) == key for j in jobs)
    start, end, log_every, spec_every, ckpt_epochs = key
    ckpt_set = set(ckpt_epochs)
    p = recipe.p

    tokens, labels = make_dataset(p)
    tokens, labels = tokens.to(device), labels.to(device)
    neg_inf = torch.nn.functional.one_hot(labels, recipe.d_vocab).bool()

    tr_m, te_m, certs = [], [], []
    per_run_w, per_run_opt = [], []
    for j in jobs:
        _, trm, tem = split_masks(p, j["data_seed"], j["frac"], recipe.test_frac)
        tr_m.append(trm)
        te_m.append(tem)
        w, o, c = load_lane_params(j, recipe, runs_dir)
        per_run_w.append(w)
        per_run_opt.append(o)
        certs.append(c)
    trm = torch.stack(tr_m).to(device).float()                 # (M, p^2)
    tem = torch.stack(te_m).to(device).float()

    params = [torch.stack([w[k] for w in per_run_w]).to(device).contiguous()
              for k in PARAM_KEYS]
    pdict = dict(zip(PARAM_KEYS, params))

    model = Transformer(recipe).to(device)
    model.set_onehot(tokens)
    opt = torch.optim.AdamW(params, lr=recipe.lr, betas=recipe.betas,
                            eps=recipe.adam_eps, weight_decay=recipe.weight_decay,
                            foreach=True)
    # Warm-start the optimizer.  torch.optim.AdamW lazily creates state on the
    # first step; pre-populating it with the stacked per-lane moments and the
    # shared step count is what makes a fork exact (bias correction depends on
    # the step count, so it MUST be restored, not reset).
    if any(o is not None for o in per_run_opt):
        assert all(o is not None for o in per_run_opt), "cannot mix fresh + forked lanes"
        steps = {int(o[PARAM_KEYS[0]]["step"]) for o in per_run_opt}
        assert len(steps) == 1, f"forked lanes disagree on step count: {steps}"
        st = steps.pop()
        assert st == start, f"ckpt step {st} != start epoch {start}"
        for k, pt in zip(PARAM_KEYS, params):
            opt.state[pt] = {
                "step": torch.tensor(float(st)),
                "exp_avg": torch.stack([o[k]["exp_avg"] for o in per_run_opt]).to(device),
                "exp_avg_sq": torch.stack([o[k]["exp_avg_sq"] for o in per_run_opt]).to(device)}
    else:
        assert start == 0, "fresh lanes must start at epoch 0"

    spec = SpectraOp(p, device)

    def fwd(pr):
        return functional_call(model, pr, (tokens,))

    def loss_aux(pr, mask):
        lg = fwd(pr)
        per = per_ex_ce(lg, labels, neg_inf)
        correct = (lg[:, :p].argmax(-1) == labels).float()
        return (per * mask).sum() / mask.sum(), (per, correct)

    grad_fn = vmap(grad(loss_aux, has_aux=True), in_dims=(0, 0))
    aux_fn = vmap(loss_aux, in_dims=(0, 0))
    fwd_fn = vmap(fwd)

    def step_fn(pr, mask):
        return grad_fn(pr, mask)

    if compile_mode != "off":
        step_fn = torch.compile(step_fn, mode=None if compile_mode == "default"
                                else compile_mode)

    def stats(per, correct):
        """(M, 4): train_acc, test_acc, train_loss, test_loss."""
        ntr, nte = trm.sum(1), tem.sum(1)
        return torch.stack([(correct * trm).sum(1) / ntr,
                            (correct * tem).sum(1) / nte,
                            (per * trm).sum(1) / ntr,
                            (per * tem).sum(1) / nte], dim=1)

    log_ep, log_rows, spec_ep, spec_co, spec_en = [], [], [], [], []

    @torch.no_grad()
    def snapshot(e):
        co, en = spec(fwd_fn(pdict))
        spec_ep.append(e)
        spec_co.append(co.cpu().numpy())
        spec_en.append(en.cpu().numpy())

    def save_all(e):
        ost = {k: opt.state[pt] for k, pt in zip(PARAM_KEYS, params)}
        for i, j in enumerate(jobs):
            save_ckpt(runs_dir / j["name"] / "checkpoints" / f"epoch_{e:06d}.pt",
                      {k: pdict[k][i] for k in PARAM_KEYS},
                      {k: {"step": ost[k]["step"], "exp_avg": ost[k]["exp_avg"][i],
                           "exp_avg_sq": ost[k]["exp_avg_sq"][i]} for k in PARAM_KEYS},
                      e, {"run": j["name"], "cohort": j.get("cohort")})

    # Materialise optimizer state up front: a checkpoint may be written at
    # epoch `start` before any step has run, and AdamW would otherwise create
    # its state lazily.  These are exactly the tensors AdamW itself would make.
    for pt in params:
        opt.state.setdefault(pt, {"step": torch.tensor(0.0),
                                  "exp_avg": torch.zeros_like(pt),
                                  "exp_avg_sq": torch.zeros_like(pt)})

    # Persistent destination buffers.  torch.compile(mode="reduce-overhead")
    # returns CUDA-graph-owned tensors that are recycled on the next call, so
    # nothing produced by step_fn may stay alive across iterations — copy out
    # immediately, keep references to nothing.
    gbuf = {k: torch.zeros_like(pdict[k]) for k in PARAM_KEYS}
    n_all = p * p
    perbuf = torch.zeros(M, n_all, device=device)
    corbuf = torch.zeros(M, n_all, device=device)

    def do_step():
        g, (per, correct) = step_fn(pdict, trm)
        for k in PARAM_KEYS:
            gbuf[k].copy_(g[k])
        perbuf.copy_(per)
        corbuf.copy_(correct)

    t0 = time.time()
    for e in range(start, end + 1):
        if e in ckpt_set:
            save_all(e)
        if e % spec_every == 0 or e == end:
            snapshot(e)
        if e == end:
            with torch.no_grad():
                _, (per, correct) = aux_fn(pdict, trm)
                perbuf.copy_(per)
                corbuf.copy_(correct)
            log_ep.append(e)
            log_rows.append(stats(perbuf, corbuf).clone())
            break
        lr = recipe.lr * min(e / recipe.warmup_steps, 1.0)
        for gp in opt.param_groups:
            gp["lr"] = lr
        do_step()
        if e % log_every == 0:
            log_ep.append(e)
            log_rows.append(stats(perbuf, corbuf).clone())
        for pt, k in zip(params, PARAM_KEYS):
            pt.grad = gbuf[k]
        opt.step()
        if verbose and log_rows and (e + 1) % (log_every * 100) == 0:
            r = log_rows[-1].cpu().numpy()
            eps_s = (e + 1 - start) / (time.time() - t0)
            print(f"  ep {e + 1:6d}  tr_loss {r[:, 2].mean():.3e}  "
                  f"te_acc {r[:, 1].mean():.4f}  "
                  f"({eps_s:.0f} eps/s x {M} = {eps_s * M:.0f} run-eps/s)",
                  flush=True)

    rows = torch.stack(log_rows).cpu().numpy()                  # (L, M, 4)
    co = np.stack(spec_co)                                      # (S, M, nf)
    en = np.stack(spec_en)
    for i, j in enumerate(jobs):
        rd = runs_dir / j["name"]
        rd.mkdir(parents=True, exist_ok=True)
        np.savez(rd / "spectra.npz",
                 epochs=np.array(log_ep), train_acc=rows[:, i, 0],
                 test_acc=rows[:, i, 1], train_loss=rows[:, i, 2],
                 test_loss=rows[:, i, 3], coeff_epochs=np.array(spec_ep),
                 coeffs=co[:, i], energy=en[:, i])
        # batch_size and compile_mode are recorded because they are part of the
        # ARITHMETIC (gate G3): a run is only comparable to another run trained
        # at the same width under the same compile mode.
        write_json(rd / "config.json",
                   {"job": j, "recipe": recipe.to_json(), "edit_cert": certs[i],
                    "batch_size": M, "compile_mode": compile_mode,
                    "device": str(device),
                    "batch_members": [x["name"] for x in jobs]})
    dt = time.time() - t0
    if verbose:
        print(f"  done {end - start} epochs x {M} lanes in {dt:.1f}s "
              f"({(end - start) * M / max(dt, 1e-9):.0f} run-eps/s)", flush=True)
    return rows, log_ep


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------

def run_manifest(manifest, runs_dir, device, compile_mode, width,
                 only=None, force=False, verbose=True):
    mf = read_json(manifest) if not isinstance(manifest, dict) else manifest
    recipe = Recipe.from_json(mf["recipe"])
    jobs = mf["jobs"]
    if only:
        jobs = [j for j in jobs if re.search(only, j["name"])]
    runs_dir = Path(runs_dir)
    chunks = chunk_jobs(jobs, width)
    n_done = 0
    for ci, chunk in enumerate(chunks):
        if not force and all((runs_dir / j["name"] / "spectra.npz").exists()
                             for j in chunk):
            # A done chunk must have been trained at the width the current
            # packing produces — G3 makes width part of the arithmetic, so a
            # resume with a different --width would silently mix arithmetics.
            for j in chunk:
                bs = read_json(runs_dir / j["name"] / "config.json").get("batch_size")
                assert bs == len(chunk), (
                    f"{j['name']} was trained at width {bs} but the current "
                    f"packing gives {len(chunk)}; resume with the original "
                    f"--width or delete this cohort's runs (G3 fallback rule)")
            if verbose:
                print(f"[{ci + 1}/{len(chunks)}] skip (done): "
                      f"{chunk[0]['name']} +{len(chunk) - 1}", flush=True)
            n_done += len(chunk)
            continue
        if verbose:
            print(f"[{ci + 1}/{len(chunks)}] batch x{len(chunk)} "
                  f"ep {chunk[0]['start_epoch']}->{chunk[0]['end_epoch']}: "
                  f"{chunk[0]['name']} ... {chunk[-1]['name']}", flush=True)
        train_batch(chunk, recipe, runs_dir, device, compile_mode, verbose)
        n_done += len(chunk)
    return n_done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--runs", default="runs_twin")
    ap.add_argument("--device", default=None, choices=["cpu", "cuda"])
    ap.add_argument("--width", type=int, default=40,
                    help="max lanes per vmap batch (~0.19 GB activations per "
                         "lane at p=113); groups are never split")
    ap.add_argument("--compile", default="reduce-overhead",
                    choices=["off", "default", "reduce-overhead", "max-autotune"])
    ap.add_argument("--no-compile", action="store_true")
    ap.add_argument("--only", default=None, metavar="REGEX")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    setup_determinism()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    cm = "off" if args.no_compile else args.compile
    t0 = time.time()
    n = run_manifest(args.manifest, args.runs, device, cm, args.width,
                     only=args.only, force=args.force)
    print(f"MANIFEST DONE: {n} runs in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
