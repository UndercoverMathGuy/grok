"""Induction-head lottery PILOT — self-contained torch, no imports from the
mod-add codebase. Lockstep-batched: ALL runs train simultaneously.

Question: in a 2-layer attention-only LM trained on a repeated-sequence
(in-context copy) task, which head pair becomes the induction circuit —
and is that identity a seed lottery (as frequency committees are in
mod-add)?

Setup: vocab V=64, seq length T=64 where the first 32 tokens are uniform
random and the last 32 are an exact repeat. Next-token loss ONLY on the
predictable positions (the second half): at query position i >= T/2-1 the
target seq[i+1] equals seq[i+1-T/2], solvable exactly by the induction
algorithm (L0 prev-token head composing into an L1 prefix-match+copy head).
Chance CE = ln(64) ~ 4.16.

Model: 2 layers, attention-only (no MLP, no LayerNorm), learned positional
embeddings, d_model=128, 8 heads x d_head=16 per layer, causal mask.
Residual: x = W_E[t] + W_pos;  x += attn_l(x) per layer;  logits = x @ W_U.
LN is omitted so the OV/QK composition readouts stay exactly linear.

Performance design (the mod-add trainer's recipe, ported):
  - lockstep batching: params stacked on a leading run axis M; loss =
    sum over runs of per-run mean CE (separable -> exact per-run grads);
    one backward + one elementwise AdamW step train every run at once.
  - training forward uses F.scaled_dot_product_attention (flash; never
    materializes the T x T pattern). Metrics evals use a separate manual
    path that DOES return patterns, chunked over the probe batch so the
    (M, chunk, H, T, T) tensor stays small.
  - torch.compile on the fused loss step (--no-compile to disable;
    auto-fallback to eager if compilation fails, e.g. no C++ toolchain —
    use the -devel base image on cloud, same as the mod-add trainer).
  - TF32 ON by default (--no-tf32 to disable). This pilot's claims are
    statistical, not bitwise; numerics caveats from the mod-add homeostat
    work do not apply here.
  - per-run data streams come from per-run numpy RNGs, identical to the
    sequential implementation; one small H2D copy per step.
  Sequential-vs-batched outputs are statistically equivalent but not
  bitwise identical (op order, SDPA).

Run grid: --init-seeds x --data-seeds (default 8 x 3 = 24 runs). Same init
across different data orders = twin probe for init-vs-SGD-noise attribution
(the mod-add twin design).

Per-eval metrics (every --eval-every steps): probe CE, per-head L1
induction score (attention mass from second-half queries to the induction
target j = i - T/2 + 1), per-head L0 prev-token score; per-head L1
ablation delta-CE every --ablate-every steps and at the final step.
Checkpoints: step 0 (init, saved BEFORE any update), every --ckpt-every,
and final. Layouts (per-run dirs, metrics.json, safetensors names) are
unchanged from the sequential version; analyze_pilot.py works as is.

Idempotent: runs whose directory contains metrics.json are excluded from
the lockstep batch. Detached-friendly: unbuffered one-line-per-eval prints.

Run:   python3 -u induction/train_pilot.py                    (24 runs)
       python3 -u induction/train_pilot.py --smoke            (2 tiny runs)
       python3 -u induction/train_pilot.py --out runs_induction/pilot
"""
import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import save_file


@dataclass
class Config:
    vocab: int = 64
    seq_len: int = 64          # first half random, second half exact repeat
    d_model: int = 128
    n_heads: int = 8
    d_head: int = 16
    n_layers: int = 2
    lr: float = 1e-3
    wd: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.999
    batch: int = 256
    steps: int = 4000
    eval_every: int = 50
    ablate_every: int = 250
    ckpt_every: int = 500
    probe_batch: int = 512
    init_seed: int = 0
    data_seed: int = 0
    probe_seed: int = 777      # shared probe set across all runs


# ---------------------------------------------------------------- params

def init_params_single(cfg: Config):
    """One run's init on CPU from its own generator — identical tensors to
    the sequential implementation for the same init_seed."""
    g = torch.Generator(device="cpu").manual_seed(cfg.init_seed)
    d, dh, H = cfg.d_model, cfg.d_head, cfg.n_heads

    def N(*shape, scale):
        return torch.randn(*shape, generator=g) * scale

    p = {
        "embed.W_E": N(cfg.vocab, d, scale=d ** -0.5),
        "pos.W_pos": N(cfg.seq_len, d, scale=d ** -0.5),
        "unembed.W_U": N(d, cfg.vocab, scale=d ** -0.5),
    }
    for l in range(cfg.n_layers):
        for name in ("W_Q", "W_K", "W_V"):
            p[f"blocks.{l}.attn.{name}"] = N(H, d, dh, scale=d ** -0.5)
        p[f"blocks.{l}.attn.W_O"] = N(H, dh, d, scale=d ** -0.5)
    return p


def stack_params(per_run, device):
    return {k: torch.stack([p[k] for p in per_run]).to(device)
                 .requires_grad_() for k in per_run[0]}


# --------------------------------------------------------------- forward

def forward_fast(P, tokens, cfg: Config, ablate=None):
    """Flash-attention training path. tokens (M,B,T) -> logits (M,B,T,V).
    No attention patterns materialized. ablate=(layer, head) zeroes that
    head's residual write (used by the ablation evals)."""
    M, B, T = tokens.shape
    ar = torch.arange(M, device=tokens.device)
    x = P["embed.W_E"][ar[:, None, None], tokens] \
        + P["pos.W_pos"][:, None, :T]
    for l in range(cfg.n_layers):
        q = torch.einsum("mbtd,mhde->mbhte", x, P[f"blocks.{l}.attn.W_Q"])
        k = torch.einsum("mbtd,mhde->mbhte", x, P[f"blocks.{l}.attn.W_K"])
        v = torch.einsum("mbtd,mhde->mbhte", x, P[f"blocks.{l}.attn.W_V"])
        H, dh = cfg.n_heads, cfg.d_head
        o = F.scaled_dot_product_attention(
            q.reshape(M * B, H, T, dh), k.reshape(M * B, H, T, dh),
            v.reshape(M * B, H, T, dh), is_causal=True,
        ).reshape(M, B, H, T, dh)
        if ablate is not None and ablate[0] == l:
            keep = torch.ones(H, device=x.device)
            keep[ablate[1]] = 0.0
            o = o * keep[None, None, :, None, None]
        x = x + torch.einsum("mbhte,mhed->mbtd", o, P[f"blocks.{l}.attn.W_O"])
    return torch.einsum("mbtd,mdv->mbtv", x, P["unembed.W_U"])


def forward_patterns(P, tokens, cfg: Config):
    """Manual path returning per-layer attention patterns
    [(M,B,H,T,T)] plus logits — evals only, callers chunk B."""
    M, B, T = tokens.shape
    ar = torch.arange(M, device=tokens.device)
    x = P["embed.W_E"][ar[:, None, None], tokens] \
        + P["pos.W_pos"][:, None, :T]
    mask = torch.triu(torch.ones(T, T, dtype=torch.bool,
                                 device=x.device), 1)
    attns = []
    for l in range(cfg.n_layers):
        q = torch.einsum("mbtd,mhde->mbhte", x, P[f"blocks.{l}.attn.W_Q"])
        k = torch.einsum("mbtd,mhde->mbhte", x, P[f"blocks.{l}.attn.W_K"])
        v = torch.einsum("mbtd,mhde->mbhte", x, P[f"blocks.{l}.attn.W_V"])
        s = torch.einsum("mbhqe,mbhke->mbhqk", q, k) / math.sqrt(cfg.d_head)
        a = s.masked_fill(mask, float("-inf")).softmax(-1)
        attns.append(a)
        o = torch.einsum("mbhqk,mbhke->mbhqe", a, v)
        x = x + torch.einsum("mbhte,mhed->mbtd", o, P[f"blocks.{l}.attn.W_O"])
    return torch.einsum("mbtd,mdv->mbtv", x, P["unembed.W_U"]), attns


def per_run_ce(logits, tokens, cfg: Config):
    """(M,) per-run mean CE on predictable positions i in [T/2-1, T-2]."""
    M = tokens.shape[0]
    half = cfg.seq_len // 2
    pred = logits[:, :, half - 1:-1]
    tgt = tokens[:, :, half:]
    ce = F.cross_entropy(pred.reshape(-1, cfg.vocab), tgt.reshape(-1),
                         reduction="none")
    return ce.view(M, -1).mean(1)


# ------------------------------------------------------------------ data

def make_batch_np(rngs, cfg: Config, n):
    half = cfg.seq_len // 2
    outs = []
    for rng in rngs:
        first = rng.integers(0, cfg.vocab, size=(n, half))
        outs.append(np.concatenate([first, first], axis=1))
    return torch.from_numpy(np.stack(outs))          # (M,n,T) int64, CPU


class GpuData:
    """Per-run torch generators ON DEVICE — no host->device copy per step.
    Same data_seed => same stream (that is all the twin design needs).
    Streams differ from the numpy path; --cpu-data restores it."""

    def __init__(self, cfgs, device):
        self.gens = [torch.Generator(device=device)
                     .manual_seed(c.data_seed) for c in cfgs]
        self.device = device

    def batch(self, cfg: Config, n, out=None):
        half = cfg.seq_len // 2
        M = len(self.gens)
        buf = out if out is not None else torch.empty(
            M, n, cfg.seq_len, dtype=torch.long, device=self.device)
        for m, g in enumerate(self.gens):
            first = torch.randint(0, cfg.vocab, (n, half), generator=g,
                                  device=self.device)
            buf[m, :, :half] = first
            buf[m, :, half:] = first
        return buf


# ----------------------------------------------------------------- evals

@torch.no_grad()
def eval_metrics(P, probe, cfg: Config, do_ablate, chunk_pat=64,
                 chunk_fast=256):
    """probe (Bp,T) shared across runs. Returns list of per-run dicts."""
    M = P["embed.W_E"].shape[0]
    half, T, H = cfg.seq_len // 2, cfg.seq_len, cfg.n_heads
    Bp = probe.shape[0]
    qs = torch.arange(half, T - 1, device=probe.device)
    qa = torch.arange(1, T, device=probe.device)

    ce_sum = torch.zeros(M, device=probe.device)
    ind_sum = torch.zeros(M, H, device=probe.device)
    prev_sum = torch.zeros(M, H, device=probe.device)
    for i in range(0, Bp, chunk_pat):
        tok = probe[i:i + chunk_pat].unsqueeze(0).expand(M, -1, -1)
        logits, attns = forward_patterns(P, tok, cfg)
        n = tok.shape[1]
        ce_sum += per_run_ce(logits, tok, cfg) * n
        ind_sum += attns[1][:, :, :, qs, qs - half + 1].mean(-1).sum(1)
        prev_sum += attns[0][:, :, :, qa, qa - 1].mean(-1).sum(1)
    ce = (ce_sum / Bp).tolist()
    ind = (ind_sum / Bp).tolist()
    prev = (prev_sum / Bp).tolist()
    out = [{"probe_ce": ce[m], "induction_l1": ind[m],
            "prevtoken_l0": prev[m]} for m in range(M)]

    if do_ablate:
        deltas = torch.zeros(M, H, device=probe.device)
        for h in range(H):
            tot = torch.zeros(M, device=probe.device)
            for i in range(0, Bp, chunk_fast):
                tok = probe[i:i + chunk_fast].unsqueeze(0).expand(M, -1, -1)
                lg = forward_fast(P, tok, cfg, ablate=(1, h))
                tot += per_run_ce(lg, tok, cfg) * tok.shape[1]
            deltas[:, h] = tot / Bp - ce_sum / Bp
        for m in range(M):
            out[m]["ablate_dce_l1"] = deltas[m].tolist()
    return out


# ------------------------------------------------------------- optimizer

def make_adamw(P, cfg: Config, names):
    """Fused multi-tensor AdamW (decoupled wd, NO bias correction — the
    mod-add trainer's optimizer semantics; elementwise on stacked tensors
    => exact lockstep of M independent per-run optimizers). _foreach_ ops
    collapse the 13-tensor python loop into ~6 multi-tensor kernels."""
    ws = [P[k] for k in names]
    ms = [torch.zeros_like(w) for w in ws]
    vs = [torch.zeros_like(w) for w in ws]

    def step(grads_list):
        torch._foreach_mul_(ms, cfg.beta1)
        torch._foreach_add_(ms, grads_list, alpha=1 - cfg.beta1)
        torch._foreach_mul_(vs, cfg.beta2)
        torch._foreach_addcmul_(vs, grads_list, grads_list,
                                value=1 - cfg.beta2)
        wdata = [w.data for w in ws]
        torch._foreach_mul_(wdata, 1 - cfg.lr * cfg.wd)
        denom = torch._foreach_sqrt(vs)
        torch._foreach_add_(denom, 1e-8)
        torch._foreach_addcdiv_(wdata, ms, denom, value=-cfg.lr)

    return step


# -------------------------------------------------------------- training

def save_ckpts(P, run_dirs, step):
    for m, rd in enumerate(run_dirs):
        path = rd / "checkpoints" / f"step_{step:05d}.safetensors"
        path.parent.mkdir(parents=True, exist_ok=True)
        save_file({k: v[m].detach().contiguous().cpu()
                   for k, v in P.items()}, str(path))


def train_batched(cfgs, run_dirs, device, use_compile=True,
                  cpu_data=False, bf16=False):
    """All runs share every Config field except init_seed/data_seed."""
    cfg = cfgs[0]
    M = len(cfgs)
    for rd, c in zip(run_dirs, cfgs):
        rd.mkdir(parents=True, exist_ok=True)
        (rd / "config.json").write_text(json.dumps(asdict(c), indent=1))

    P = stack_params([init_params_single(c) for c in cfgs], device)
    save_ckpts(P, run_dirs, 0)
    names = sorted(P.keys())
    params = [P[k] for k in names]
    opt_step = make_adamw(P, cfg, names)
    probe = make_batch_np([np.random.default_rng(cfg.probe_seed)], cfg,
                          cfg.probe_batch)[0].to(device)

    use_gpu_data = device == "cuda" and not cpu_data
    gpu_data = GpuData(cfgs, device) if use_gpu_data else None
    np_rngs = None if use_gpu_data else \
        [np.random.default_rng(c.data_seed) for c in cfgs]
    # static input buffer: same storage every step, so CUDA graphs
    # (compile mode reduce-overhead) can capture the whole fwd+loss
    tokens_buf = torch.empty(M, cfg.batch, cfg.seq_len, dtype=torch.long,
                             device=device)

    amp = torch.autocast(device_type="cuda", dtype=torch.bfloat16) \
        if (bf16 and device == "cuda") else None

    def loss_fn(tokens):
        if amp is not None:
            with amp:
                return per_run_ce(forward_fast(P, tokens, cfg),
                                  tokens, cfg).float().sum()
        return per_run_ce(forward_fast(P, tokens, cfg), tokens, cfg).sum()

    step_loss = loss_fn
    if use_compile:
        # mode ladder: CUDA-graph capture first, plain inductor second
        for mode in (("reduce-overhead" if device == "cuda" else None),
                     None):
            try:
                c = torch.compile(loss_fn, dynamic=False, **(
                    {"mode": mode} if mode else {}))
                tokens_buf.copy_(probe[:cfg.batch].unsqueeze(0)
                                 .expand(M, -1, -1))
                c(tokens_buf)        # errors surface on first call
                c(tokens_buf)        # second call exercises graph replay
                step_loss = c
                print(f"torch.compile: ok (mode={mode or 'default'})",
                      flush=True)
                break
            except Exception as e:                   # pragma: no cover
                print(f"torch.compile mode={mode or 'default'} failed "
                      f"({type(e).__name__}); trying next", flush=True)

    history = [[] for _ in range(M)]
    t0 = time.time()
    for step in range(cfg.steps + 1):
        last = step == cfg.steps
        if step % cfg.eval_every == 0 or last:
            do_abl = last or step % cfg.ablate_every == 0
            ms = eval_metrics(P, probe, cfg, do_ablate=do_abl)
            ces = []
            for m in range(M):
                ms[m]["step"] = step
                history[m].append(ms[m])
                ces.append(ms[m]["probe_ce"])
            tops = [int(np.argmax(x["induction_l1"])) for x in ms]
            wins = np.bincount(tops, minlength=cfg.n_heads).tolist()
            print(f"step {step:5d}  ce mean {np.mean(ces):.4f} "
                  f"[{min(ces):.3f},{max(ces):.3f}]  winners {wins}  "
                  f"({time.time()-t0:.0f}s)", flush=True)
        if step % cfg.ckpt_every == 0 and step > 0:
            save_ckpts(P, run_dirs, step)
        if last:
            break
        if use_gpu_data:
            gpu_data.batch(cfg, cfg.batch, out=tokens_buf)
        else:
            tokens_buf.copy_(make_batch_np(np_rngs, cfg, cfg.batch))
        loss = step_loss(tokens_buf)
        grads = torch.autograd.grad(loss, params)
        opt_step(list(grads))

    save_ckpts(P, run_dirs, cfg.steps)
    for m, rd in enumerate(run_dirs):
        (rd / "metrics.json").write_text(json.dumps(history[m]))
    print(f"done: {M} runs in {time.time()-t0:.0f}s  "
          f"final ce mean {np.mean([h[-1]['probe_ce'] for h in history]):.4f}",
          flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs_induction/pilot")
    ap.add_argument("--init-seeds", type=int, default=8)
    ap.add_argument("--data-seeds", type=int, default=3)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--device", default=None)
    ap.add_argument("--no-compile", action="store_true")
    ap.add_argument("--no-tf32", action="store_true")
    ap.add_argument("--cpu-data", action="store_true",
                    help="numpy data streams (reproduces the original "
                         "sequential implementation's batches)")
    ap.add_argument("--bf16", action="store_true",
                    help="bf16 autocast for fwd/bwd (cuda only)")
    ap.add_argument("--smoke", action="store_true",
                    help="2 tiny runs (200 steps, d=32) to verify end-to-end")
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available()
                             else "mps" if torch.backends.mps.is_available()
                             else "cpu")
    if device == "cuda" and not args.no_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    print(f"device: {device}  compile: {not args.no_compile}  "
          f"tf32: {device == 'cuda' and not args.no_tf32}", flush=True)

    if args.smoke:
        cfgs = [Config(init_seed=i, data_seed=0, steps=200, d_model=32,
                       d_head=8, n_heads=4, batch=64, probe_batch=128,
                       eval_every=50, ablate_every=100, ckpt_every=100)
                for i in (0, 1)]
        dirs = [Path(args.out + "_smoke") / f"init{c.init_seed}_data0"
                for c in cfgs]
        train_batched(cfgs, dirs, device,
                      use_compile=not args.no_compile,
                      cpu_data=args.cpu_data, bf16=args.bf16)
        return

    init_seeds = [1000 + 7 * i for i in range(args.init_seeds)]
    data_seeds = [500 + 11 * j for j in range(args.data_seeds)]
    print(f"init seeds: {init_seeds}\ndata seeds: {data_seeds}", flush=True)
    cfgs, dirs = [], []
    for iseed in init_seeds:
        for dseed in data_seeds:
            rd = Path(args.out) / f"init{iseed}_data{dseed}"
            if (rd / "metrics.json").exists():
                print(f"skip (done): {rd}")
                continue
            cfgs.append(Config(init_seed=iseed, data_seed=dseed,
                               steps=args.steps))
            dirs.append(rd)
    if not cfgs:
        print("nothing to train")
        return
    print(f"training {len(cfgs)} runs in lockstep", flush=True)
    train_batched(cfgs, dirs, device, use_compile=not args.no_compile,
                  cpu_data=args.cpu_data, bf16=args.bf16)


if __name__ == "__main__":
    main()
