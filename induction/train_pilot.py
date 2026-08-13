"""Induction-head lottery PILOT — self-contained torch, no imports from the
mod-add codebase. Lockstep-batched: ALL runs train simultaneously.

Question: in a 2-layer attention-only LM, which head pair becomes the
induction circuit — and is that identity a seed lottery (as frequency
committees are in mod-add)?

TASKS (--task):
  copy      v1. First 32 tokens uniform random, last 32 an exact repeat;
            loss on the second half. RESULT 2026-08-13: PQ0 FAILED —
            the lag is fixed at T/2, so a purely positional head solves
            it and induction smears uniformly over all heads
            (conc 0.131 ~ 1/8). Kept for the record.
  induction v2 (the fix). Zipfian background tokens; one segment of
            length seg_min..seg_max repeated at two RANDOM offsets per
            sequence. The lag varies per sequence, so only content-based
            prefix matching (real induction) predicts the second copy.
            Loss on the inducible positions of the second copy only.

Model: 2 layers, attention-only (no MLP, no LayerNorm), learned positional
embeddings, d_model=128, 8 heads x d_head=16 per layer, causal mask.
Residual: x = W_E[t] + W_pos;  x += attn_l(x) per layer;  logits = x @ W_U.
LN is omitted so the OV/QK composition readouts stay exactly linear.

Performance: lockstep run-batching (params stacked on leading axis M,
separable summed loss -> exact per-run grads, fused _foreach_ AdamW),
flash attention (SDPA) for training, torch.compile with a
reduce-overhead -> default -> eager fallback ladder, static input
buffers (CUDA-graph friendly), TF32 flag, optional bf16 autocast.
Pattern-returning evals are a separate manual path, chunked over the
probe batch. Sequential-vs-batched outputs are statistically equivalent,
not bitwise.

Unified batch representation for both tasks:
  tokens (M,B,T)   loss_mask (M,B,T-1)  over query positions 0..T-2
  qidx,kidx,valid (M,B,N)  per-sequence induction-metric index pairs:
  attention from query qidx to key kidx is the induction-correct edge.
Per-eval metrics: masked probe CE, per-head L1 induction score (mean
attention on the correct edges), per-head L0 prev-token score, per-head
L1 ablation delta-CE every --ablate-every and at the final step.
Checkpoints: step 0 (init, pre-update), every --ckpt-every, final.

Idempotent (runs with metrics.json are skipped); detached-friendly.

Run:   python3 -u induction/train_pilot.py --task induction   (24 runs)
       python3 -u induction/train_pilot.py --smoke             (2 tiny runs)
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
    task: str = "induction"    # "copy" (v1) | "induction" (v2)
    vocab: int = 64
    seq_len: int = 64
    d_model: int = 128
    n_heads: int = 8
    d_head: int = 16
    n_layers: int = 2
    zipf_alpha: float = 1.0    # induction task background distribution
    seg_min: int = 8           # repeated-segment length bounds (induction)
    seg_max: int = 16
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
    """One run's init on CPU from its own generator (reproducible)."""
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
    """Flash-attention path, no patterns materialized. (M,B,T)->(M,B,T,V).
    ablate=(layer, head) zeroes that head's residual write."""
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
    """Manual path returning per-layer patterns [(M,B,H,T,T)] + logits —
    evals only, callers chunk B."""
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


def per_run_ce(logits, tokens, loss_mask):
    """(M,) per-run mean CE over the masked query positions 0..T-2."""
    M, B, T, V = logits.shape
    ce = F.cross_entropy(logits[:, :, :-1].reshape(-1, V),
                         tokens[:, :, 1:].reshape(-1),
                         reduction="none").view(M, B, T - 1)
    return (ce * loss_mask).sum((1, 2)) / loss_mask.sum((1, 2))


# ------------------------------------------------------------------ data

def zipf_probs(cfg: Config):
    w = 1.0 / np.arange(1, cfg.vocab + 1) ** cfg.zipf_alpha
    return w / w.sum()


def gen_copy_np(rng, cfg: Config, n):
    """v1 fixed-lag repeat. Returns tokens plus the unified metadata."""
    half = cfg.seq_len // 2
    first = rng.integers(0, cfg.vocab, size=(n, half))
    toks = np.concatenate([first, first], axis=1)
    lm = np.zeros((n, cfg.seq_len - 1), bool)
    lm[:, half - 1:] = True
    q = np.arange(half, cfg.seq_len - 1)
    qidx = np.broadcast_to(q, (n, q.size)).copy()
    kidx = qidx - half + 1
    valid = np.ones_like(qidx, bool)
    return toks, lm, qidx, kidx, valid


def gen_induction_np(rng, cfg: Config, n, probs):
    """v2 variable-offset repeated segment over a zipfian background.
    Loss on second-copy positions o2+j, j in 0..L-2 (target = seg[j+1]);
    induction-correct attention edge: query o2+j -> key o1+j+1."""
    T = cfg.seq_len
    toks = rng.choice(cfg.vocab, size=(n, T), p=probs)
    L = rng.integers(cfg.seg_min, cfg.seg_max + 1, size=n)
    o1 = rng.integers(0, T - 2 * L + 1)
    o2 = rng.integers(o1 + L, T - L + 1)
    Nmax = cfg.seg_max
    j = np.broadcast_to(np.arange(Nmax), (n, Nmax))
    inseg = j < L[:, None]
    rows = np.broadcast_to(np.arange(n)[:, None], (n, Nmax))
    # write the repeat: toks[b, o2+j] = toks[b, o1+j] for j < L
    src = np.clip(o1[:, None] + j, 0, T - 1)
    dst = np.clip(o2[:, None] + j, 0, T - 1)
    toks[rows[inseg], dst[inseg]] = toks[rows[inseg], src[inseg]]
    # loss mask over query positions 0..T-2: q = o2+j, j <= L-2
    lm = np.zeros((n, T - 1), bool)
    qm = j <= L[:, None] - 2
    lm[rows[qm], dst[qm]] = True
    # induction metric edges: q = o2+j -> k = o1+j+1, j <= L-2
    qidx = dst
    kidx = np.clip(src + 1, 0, T - 1)
    return toks, lm, qidx, kidx, qm


def gen_batch(task, rng, cfg: Config, n, probs):
    if task == "copy":
        return gen_copy_np(rng, cfg, n)
    return gen_induction_np(rng, cfg, n, probs)


# ----------------------------------------------------------------- evals

@torch.no_grad()
def eval_metrics(P, probe, cfg: Config, do_ablate, chunk_pat=64,
                 chunk_fast=256):
    """probe = (tokens (Bp,T), loss_mask, qidx, kidx, valid) shared
    across runs. Returns list of per-run dicts."""
    ptok, plm, pq, pk, pv = probe
    M = P["embed.W_E"].shape[0]
    H, T = cfg.n_heads, cfg.seq_len
    Bp = ptok.shape[0]

    ce_w = torch.zeros(M, device=ptok.device)
    ind_sum = torch.zeros(M, H, device=ptok.device)
    prev_sum = torch.zeros(M, H, device=ptok.device)
    qa = torch.arange(1, T, device=ptok.device)
    n_edges = pv.sum().item()
    for i in range(0, Bp, chunk_pat):
        tok = ptok[i:i + chunk_pat].unsqueeze(0).expand(M, -1, -1)
        lm = plm[i:i + chunk_pat].unsqueeze(0).expand(M, -1, -1)
        logits, attns = forward_patterns(P, tok, cfg)
        C = tok.shape[1]
        ce_w += per_run_ce(logits, tok, lm) * lm[0].sum()
        # induction edges: permute to (M,H,C,T,T) then advanced-index
        a1 = attns[1].permute(0, 2, 1, 3, 4)
        ci = torch.arange(C, device=tok.device)[:, None]
        edges = a1[:, :, ci, pq[i:i + chunk_pat], pk[i:i + chunk_pat]]
        ind_sum += (edges * pv[i:i + chunk_pat]).sum((2, 3))
        prev_sum += attns[0][:, :, :, qa, qa - 1].mean(-1).sum(1)
    tot_mask = plm.sum()
    ce = (ce_w / tot_mask).tolist()
    ind = (ind_sum / n_edges).tolist()
    prev = (prev_sum / Bp).tolist()
    out = [{"probe_ce": ce[m], "induction_l1": ind[m],
            "prevtoken_l0": prev[m]} for m in range(M)]

    if do_ablate:
        deltas = torch.zeros(M, H, device=ptok.device)
        for h in range(H):
            ce_h = torch.zeros(M, device=ptok.device)
            for i in range(0, Bp, chunk_fast):
                tok = ptok[i:i + chunk_fast].unsqueeze(0).expand(M, -1, -1)
                lm = plm[i:i + chunk_fast].unsqueeze(0).expand(M, -1, -1)
                lg = forward_fast(P, tok, cfg, ablate=(1, h))
                ce_h += per_run_ce(lg, tok, lm) * lm[0].sum()
            deltas[:, h] = ce_h / tot_mask - ce_w / tot_mask
        for m in range(M):
            out[m]["ablate_dce_l1"] = deltas[m].tolist()
    return out


# ------------------------------------------------------------- optimizer

def make_adamw(P, cfg: Config, names):
    """Fused multi-tensor AdamW (decoupled wd, NO bias correction);
    elementwise on stacked tensors => exact lockstep of M per-run
    optimizers."""
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


def to_dev(arrs, device):
    return [torch.from_numpy(np.ascontiguousarray(a)).to(device)
            for a in arrs]


def train_batched(cfgs, run_dirs, device, use_compile=True, bf16=False):
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

    probs = zipf_probs(cfg)
    prng = np.random.default_rng(cfg.probe_seed)
    ptok, plm, pq, pk, pv = to_dev(
        gen_batch(cfg.task, prng, cfg, cfg.probe_batch, probs), device)
    probe = (ptok, plm.float(), pq, pk, pv.float())
    rngs = [np.random.default_rng(c.data_seed) for c in cfgs]

    # static buffers (same storage every step -> CUDA-graph capturable)
    tokens_buf = torch.empty(M, cfg.batch, cfg.seq_len, dtype=torch.long,
                             device=device)
    mask_buf = torch.empty(M, cfg.batch, cfg.seq_len - 1,
                           dtype=torch.float32, device=device)

    def fill_buffers():
        for m, rng in enumerate(rngs):
            toks, lm, _, _, _ = gen_batch(cfg.task, rng, cfg, cfg.batch,
                                          probs)
            tokens_buf[m].copy_(torch.from_numpy(
                np.ascontiguousarray(toks)), non_blocking=True)
            mask_buf[m].copy_(torch.from_numpy(lm.astype(np.float32)),
                              non_blocking=True)

    amp = torch.autocast(device_type="cuda", dtype=torch.bfloat16) \
        if (bf16 and device == "cuda") else None

    def loss_fn(tokens, lmask):
        if amp is not None:
            with amp:
                return per_run_ce(forward_fast(P, tokens, cfg),
                                  tokens, lmask).float().sum()
        return per_run_ce(forward_fast(P, tokens, cfg), tokens,
                          lmask).sum()

    step_loss = loss_fn
    if use_compile:
        for mode in (("reduce-overhead" if device == "cuda" else None),
                     None):
            try:
                c = torch.compile(loss_fn, dynamic=False, **(
                    {"mode": mode} if mode else {}))
                fill_buffers()
                c(tokens_buf, mask_buf)   # errors surface on first call
                c(tokens_buf, mask_buf)   # exercises CUDA-graph replay
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
        fill_buffers()
        loss = step_loss(tokens_buf, mask_buf)
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
    ap.add_argument("--task", default="induction",
                    choices=["copy", "induction"])
    ap.add_argument("--out", default=None,
                    help="default: runs_induction/pilot_<task>")
    ap.add_argument("--init-seeds", type=int, default=8)
    ap.add_argument("--data-seeds", type=int, default=3)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--wd", type=float, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--no-compile", action="store_true")
    ap.add_argument("--no-tf32", action="store_true")
    ap.add_argument("--bf16", action="store_true",
                    help="bf16 autocast for fwd/bwd (cuda only)")
    ap.add_argument("--smoke", action="store_true",
                    help="2 tiny runs (200 steps, d=32) to verify end-to-end")
    args = ap.parse_args()
    out = args.out or f"runs_induction/pilot_{args.task}"

    device = args.device or ("cuda" if torch.cuda.is_available()
                             else "mps" if torch.backends.mps.is_available()
                             else "cpu")
    if device == "cuda" and not args.no_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    print(f"device: {device}  task: {args.task}  "
          f"compile: {not args.no_compile}  "
          f"tf32: {device == 'cuda' and not args.no_tf32}", flush=True)

    over = {} if args.wd is None else {"wd": args.wd}
    if args.smoke:
        cfgs = [Config(task=args.task, init_seed=i, data_seed=0, steps=200,
                       d_model=32, d_head=8, n_heads=4, batch=64,
                       probe_batch=128, eval_every=50, ablate_every=100,
                       ckpt_every=100, **over) for i in (0, 1)]
        dirs = [Path(out + "_smoke") / f"init{c.init_seed}_data0"
                for c in cfgs]
        train_batched(cfgs, dirs, device,
                      use_compile=not args.no_compile, bf16=args.bf16)
        return

    init_seeds = [1000 + 7 * i for i in range(args.init_seeds)]
    data_seeds = [500 + 11 * j for j in range(args.data_seeds)]
    print(f"init seeds: {init_seeds}\ndata seeds: {data_seeds}", flush=True)
    cfgs, dirs = [], []
    for iseed in init_seeds:
        for dseed in data_seeds:
            rd = Path(out) / f"init{iseed}_data{dseed}"
            if (rd / "metrics.json").exists():
                print(f"skip (done): {rd}")
                continue
            cfgs.append(Config(task=args.task, init_seed=iseed,
                               data_seed=dseed, steps=args.steps, **over))
            dirs.append(rd)
    if not cfgs:
        print("nothing to train")
        return
    print(f"training {len(cfgs)} runs in lockstep", flush=True)
    train_batched(cfgs, dirs, device, use_compile=not args.no_compile,
                  bf16=args.bf16)


if __name__ == "__main__":
    main()
