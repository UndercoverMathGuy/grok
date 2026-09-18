"""relu-4 at p=1009: memory-safe full-batch trainer + FFT spectra instrument.

The p=113 cohort recipe (nudial/train_nudial.py, cohorts A/M: relu nu=4 =
layout (1,1), onehot -> E1(d=128) -> U,V(m) -> ReLU -> W1(r=128) -> W(p),
frac_train 0.5, wd 1.0, lr 1e-3, MLX-semantics AdamW, 10k epochs) scaled to
p=1009, where the problem is MEMORY, not flops: one run's full-grid logits
are (p^2, p) f32 = 4.1 GB and train_nudial's (nf, p^2) f64 basis matrices
u_cos/u_sin are 4 GB each.

Two changes, both exact:
  * full-batch gradient by ACCUMULATION over chunks of the train set (the
    mean over the whole train set is the sum of chunk sums / N_tr: identical
    objective, only the float summation order differs), plus a lean CE that
    avoids torch's int64 one-hot (8 bytes per logit at p=1009);
  * the spectra instrument via FFTs of the diagonal-collapsed logit table
    T[s,o] = sum_{a+b=s mod p} L[a,b,o]  (only p x p per run), so no basis is
    ever materialised.  With F = DFT_s(T)[k,o] = DFT2(L)[k,k,o]:
        energy_k = (2/p^2) sum_o |F[k,o]|^2
        coeff_k  = sqrt(2) p^-1.5 Re sum_o F[k,o] e^{+2 pi i k o / p}
                 = sqrt(2) p^-1.5 Re DFT3(L)[k,k,-k]
    which equal train_nudial.take_snapshot's einsum quantities exactly.

Run dirs (identical file format to train_nudial, so existing tooling reads them):
  $RUNS_DIR/P<p>/relu4_m<m>_p<p>_f0.5_train/wd1_s<seed>/
    config.json  spectra.npz  metrics.json  final.safetensors

Usage
  python3 -u nudial/p1009.py --width 512  --seeds 16 --mem-gb 28   # pod, 32 GB
  python3 -u nudial/p1009.py --width 2048 --seeds 8  --mem-gb 28
  uv run python nudial/p1009.py --smoke      # p=23 m=64 2 seeds 300 ep, CPU
  uv run python nudial/p1009.py --test       # the two required correctness tests
"""
import argparse, collections, json, math, os, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "nudial"))
from train_nudial import (Config, committee_from_coeffs, draw_seeds, forward,
                          init_params, param_shapes, sq_norm, train_test_split)

RUNS = Path(os.environ.get("RUNS_DIR", ROOT / "runs_nudial"))
torch.backends.cuda.matmul.allow_tf32 = False      # user veto: exact fp32
torch.backends.cudnn.allow_tf32 = False


# ---------------------------------------------------------------- lean CE ---
def per_ex_ce_lean(logits, labels):
    """Bit-for-bit train_nudial.per_ex_stable_f32 without the int64 one-hot.

    masked_fill(one_hot(y).bool(), -inf) and scatter(-inf) write -inf in the
    same places and have the same backward (zero gradient at y through the
    'others' path), but one_hot materialises (M, N, p) int64 = 8 bytes per
    logit, which at p=1009 is 8x the logits themselves.
    """
    idx = labels.unsqueeze(-1)
    z_y = logits.gather(-1, idx)
    others = logits.scatter(-1, idx, float("-inf"))
    return F.softplus(torch.logsumexp(others, -1, keepdim=True) - z_y).squeeze(-1)


def per_ex_margin_lean(logits, labels):
    """train_nudial.per_ex_margin without the int64 one-hot (same reason)."""
    idx = labels.unsqueeze(-1)
    z_y = logits.gather(-1, idx).squeeze(-1)
    return z_y - logits.scatter(-1, idx, float("-inf")).max(-1).values


# --------------------------------------------------------- spectra by FFT ---
def collapse_add(T, L, a0, p):
    """Accumulate T[s,o] += sum_{a in chunk, b} [a+b=s] L[a,b,o].

    L is (M, na*p, p) with the grid flattened a-major (train_nudial's order);
    a runs a0..a0+na-1.  For a fixed a the map b -> s = a+b is a cyclic shift,
    so the collapse is na rolls (no (M, na, p, p) gather buffer).
    """
    M, _, _ = L.shape
    L4 = L.view(M, -1, p, p)
    for i in range(L4.shape[1]):
        T.add_(torch.roll(L4[:, i], shifts=int(a0 + i), dims=1))


def spectra_from_T(T, p):
    """(M, p, p) diagonal-collapsed logits -> (coeffs, energy), both (M, p//2).

    Same quantities as train_nudial.take_snapshot: with the orthonormal rows
    bc_k = sqrt(2/p) cos(2 pi k x / p), bs_k = sqrt(2/p) sin(...),
      u_cos_k = (bc_k (x) bc_k - bs_k (x) bs_k)/sqrt2 = sqrt2/p cos(k(a+b))
      u_sin_k = (bs_k (x) bc_k + bc_k (x) bs_k)/sqrt2 = sqrt2/p sin(k(a+b))
      c = <u_cos_k, L>_(a,b),  s = <u_sin_k, L>_(a,b)   (per output o)
      energy_k = sum_o c^2 + s^2,  coeff_k = (<c, bc_k> + <s, bs_k>)/sqrt2.
    With F[k,o] = sum_s T[s,o] e^{-2 pi i k s / p} one has c = sqrt2/p Re F,
    s = -sqrt2/p Im F, hence the two lines below.
    """
    nf = p // 2
    Fk = torch.fft.fft(T, dim=1)[:, 1:nf + 1]                    # (M, nf, p)
    energy = (Fk.real ** 2 + Fk.imag ** 2).sum(-1) * (2.0 / p ** 2)
    k = torch.arange(1, nf + 1, device=T.device, dtype=torch.float64)
    o = torch.arange(p, device=T.device, dtype=torch.float64)
    ph = torch.exp(2j * math.pi * k[:, None] * o[None, :] / p)   # (nf, p)
    coeffs = torch.einsum("mko,ko->mk", Fk, ph).real * (math.sqrt(2.0) / p ** 1.5)
    return coeffs, energy


# ------------------------------------------------------------ memory plan ---
def bytes_per_ex(cfg, grad=True, safety=1.35):
    """f32 bytes of live/saved activation per (run, example).

    grad=True (fwd+bwd): ea/eb are saved for the U/V matmul backward, relu's
    output for its own, W1 h for the readout; the (.,.,p) tensors are logits,
    the -inf-scattered copy, and the two gradient buffers.  grad=False: no
    gradient buffers and activations die as they go, so only logits + the copy.
    """
    n_pre, n_post = cfg.layout
    p, d, m, r = cfg.p, cfg.d, cfg.m, cfg.r
    words = (2 * d * n_pre + 3 * m + 2 * r * n_post + 4 * p if grad else
             2 * d * n_pre + 2 * m + r * n_post + 2 * p)
    return 4 * words * safety


def usable(mem_gb, reserve_gb=0.8):
    return (mem_gb - min(reserve_gb, 0.25 * mem_gb)) * 1e9


def plan_chunks(cfg, M, n_ex, mem_gb, grad=True, reserve_gb=0.8):
    """Number of equal-ish chunks so M x chunk activations fit the budget.

    Prefers a chunk count that divides n_ex exactly (one compiled shape).
    """
    per_ex = bytes_per_ex(cfg, grad)
    cap = max(1, int(usable(mem_gb, reserve_gb) / (per_ex * M)))
    n_min = max(1, math.ceil(n_ex / cap))
    for n in range(n_min, n_min + 12):       # prefer an exact divisor
        if n_ex % n == 0:
            return n, n_ex // n, per_ex
    return n_min, math.ceil(n_ex / n_min), per_ex


def split_bounds(n_ex, n_chunks):
    e = np.linspace(0, n_ex, n_chunks + 1).round().astype(int)
    return list(zip(e[:-1].tolist(), e[1:].tolist()))


# ---------------------------------------------------------------- trainer ---
def train_chunked(cfgs, run_dirs, mem_gb=28.0, compile_mode="default",
                  log_every=500, spectra_every=100, device=None, save=True,
                  quiet=False):
    """Exact full-batch AdamW (train_nudial semantics) by gradient accumulation."""
    M = len(cfgs); c0 = cfgs[0]
    for c in cfgs:
        for f in ("p", "family", "m", "d", "r", "frac_train", "mode", "layout",
                  "opt", "warmup_steps", "betas", "adam_eps", "num_epochs"):
            assert getattr(c, f) == getattr(c0, f), f"batch must agree on {f}"
        assert c.mode == "train" and c.opt in ("adamw", "gd", "gdm", "ngd"), "p1009: train mode; adamw/gd/gdm/ngd"
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    run_dirs = [Path(rd) for rd in run_dirs]
    if save:
        for cfg, rd in zip(cfgs, run_dirs):
            rd.mkdir(parents=True, exist_ok=True); cfg.save(rd / "config.json")
    p, nu = c0.p, c0.degree
    keys = list(param_shapes(c0))
    per_run = [init_params(c) for c in cfgs]
    pr = {k: torch.stack([r[k] for r in per_run]).to(device).requires_grad_()
          for k in keys}
    params = [pr[k] for k in keys]

    masks = [train_test_split(c) for c in cfgs]
    assert len({int(mk.sum()) for mk in masks}) == 1, "equal-count batches only"
    tr_idx = torch.from_numpy(np.stack([np.flatnonzero(mk) for mk in masks]).astype(np.int32)).to(device)
    te_idx = torch.from_numpy(np.stack([np.flatnonzero(~mk) for mk in masks]).astype(np.int32)).to(device)
    mask_t = torch.from_numpy(np.stack(masks)).to(device)
    n_tr, n_te = tr_idx.shape[1], te_idx.shape[1]

    # ---- memory plan
    n_ch, c_tr, bpe = plan_chunks(c0, M, n_tr, mem_gb)
    bounds = split_bounds(n_tr, n_ch)
    n_ch_te, _, _ = plan_chunks(c0, M, max(n_te, 1), mem_gb, grad=False)
    bounds_te = split_bounds(n_te, max(n_ch_te, 1)) if n_te else []
    # full-grid pass: chunk over rows of a (p examples each); the (M,p,p) f64
    # collapse buffer and its f32 roll scratch are charged separately, and only
    # 60% of the budget is used -- snapshots are 1% of wall time, so headroom is free
    grid_free = max(0.6 * usable(mem_gb) - M * p * p * 12, 1e6)
    a_rows = max(1, min(p, int(grid_free / (bytes_per_ex(c0, grad=False) * M * p))))
    if not quiet:
        print(f"[plan] M={M} runs stacked | train {n_tr} -> {n_ch} chunks of ~{c_tr} "
              f"({bpe/1e3:.1f} kB/run-example incl. safety 1.35x -> "
              f"{M*c_tr*bpe/1e9:.1f} GB activations, budget {mem_gb} GB)", flush=True)
        print(f"[plan] test {n_te} -> {len(bounds_te)} chunks | full grid {p*p} -> "
              f"{math.ceil(p/a_rows)} chunks of {a_rows} a-rows x {p}", flush=True)

    # ---- optimiser state (MLX-semantics AdamW, no bias correction)
    m_st = [torch.zeros_like(q) for q in params]
    v_st = [torch.zeros_like(q) for q in params]
    acc = [torch.zeros_like(q) for q in params]
    b1, b2 = c0.betas; eps = c0.adam_eps; OPT = c0.opt; MU = c0.momentum; CLIP = c0.clip
    lr_base = torch.tensor([c.lr for c in cfgs], device=device)
    wd_v = torch.tensor([c.weight_decay for c in cfgs], device=device)
    lr_t = torch.zeros(M, device=device)
    bshape = {id(q): (-1,) + (1,) * (q.dim() - 1) for q in params}

    def unpack(idx32):
        idx = idx32.long(); A = idx // p; B = idx - A * p
        return A, B, (A + B) % p

    def chunk_fwd(A, B, lab):
        return per_ex_ce_lean(forward(pr, A, B, c0), lab).sum(1)     # (M,)

    if compile_mode != "off":
        chunk_fwd = torch.compile(chunk_fwd,
                                  mode=None if compile_mode == "default" else compile_mode)

    inv_n = 1.0 / n_tr

    def step():
        for a_ in acc:
            a_.zero_()
        tot = torch.zeros(M, device=device)
        for s, e in bounds:
            A, B, lab = unpack(tr_idx[:, s:e])
            per = chunk_fwd(A, B, lab)
            for a_, g in zip(acc, torch.autograd.grad(per.sum() * inv_n, params)):
                a_.add_(g)
            tot += per.detach()
        with torch.no_grad():
            if OPT == "ngd" or CLIP > 0:
                gn = torch.sqrt(sum((g ** 2).flatten(1).sum(1) for g in acc)) + 1e-12
            if CLIP > 0:
                sc = torch.clamp(CLIP / gn, max=1.0)
                for a_, q in zip(acc, params):
                    a_.mul_(sc.view(bshape[id(q)]))
            for q, g, m_, v_ in zip(params, acc, m_st, v_st):
                sh = bshape[id(q)]
                lr_, wd_ = lr_t.view(sh), wd_v.view(sh)
                if OPT == "gd":                       # plain GD, coupled l2
                    q.sub_(lr_ * (g + wd_ * q)); continue
                if OPT == "gdm":                      # heavy-ball GD, coupled l2
                    m_.mul_(MU).add_(g + wd_ * q); q.sub_(lr_ * m_); continue
                if OPT == "ngd":
                    q.sub_(lr_ * (g / gn.view(sh) + wd_ * q)); continue
                m_.mul_(b1).add_(g, alpha=1 - b1)
                v_.mul_(b2).addcmul_(g, g, value=1 - b2)
                q.mul_(1 - lr_ * wd_)
                q.sub_(lr_ * m_ / (v_.sqrt() + eps))
        return tot * inv_n

    # ---- chunked full-grid pass: accuracy, min margin, diagonal collapse
    snap = {"epochs": [], "coeffs": [], "energy": [], "train_acc": [], "test_acc": [],
            "pnorm2": [], "min_margin_norm": []}
    b_all = torch.arange(p, device=device)

    @torch.no_grad()
    def grid_pass(want_T=True):
        T = torch.zeros(M, p, p, dtype=torch.float64, device=device) if want_T else None
        ok = torch.zeros(M, p * p, dtype=torch.bool, device=device)
        gmin = torch.full((M,), float("inf"), device=device)
        for a0 in range(0, p, a_rows):
            na = min(a_rows, p - a0)
            A = torch.arange(a0, a0 + na, device=device).repeat_interleave(p)
            B = b_all.repeat(na)
            lab = (A + B) % p
            L = forward(pr, A, B, c0)                               # (M, na*p, p)
            ok[:, a0 * p:(a0 + na) * p] = L.argmax(-1) == lab
            g = per_ex_margin_lean(L, lab[None].expand(M, -1)).min(1).values
            gmin = torch.minimum(gmin, g)
            if want_T:
                collapse_add(T, L, a0, p)
            del L
        return T, ok, gmin

    @torch.no_grad()
    def take_snapshot(epoch):
        T, ok, gmin = grid_pass(True)
        coeffs, energy = spectra_from_T(T, p)
        pn2 = sq_norm(pr).detach()
        snap["epochs"].append(epoch)
        snap["train_acc"].append((ok & mask_t).sum(-1).double() / mask_t.sum(-1))
        snap["test_acc"].append((ok & ~mask_t).sum(-1).double() / (~mask_t).sum(-1).clamp(min=1))
        snap["coeffs"].append(coeffs); snap["energy"].append(energy)
        snap["pnorm2"].append(pn2)
        snap["min_margin_norm"].append(gmin.double() / pn2.double() ** (nu / 2))
        del T

    @torch.no_grad()
    def test_stats():
        if not n_te:
            return np.full(M, np.nan), np.full(M, np.nan)
        ce = torch.zeros(M, device=device); nc = torch.zeros(M, device=device)
        for s, e in bounds_te:
            A, B, lab = unpack(te_idx[:, s:e])
            L = forward(pr, A, B, c0)
            ce += per_ex_ce_lean(L, lab).sum(1)
            nc += (L.argmax(-1) == lab).sum(1)
        return (ce / n_te).cpu().numpy(), (nc / n_te).cpu().numpy()

    E = c0.num_epochs
    train_hist = torch.empty(E, M, device=device); test_hist = []
    print(f"batch {M} on {device}: {c0.family} nu={nu} m={c0.m} p={p} "
          f"layout={c0.layout} train {n_tr} test {n_te} epochs {E}", flush=True)
    t0 = time.time()
    for epoch in range(E):
        if spectra_every and epoch % spectra_every == 0:
            take_snapshot(epoch)
        lr_t.copy_(lr_base * min(epoch / c0.warmup_steps, 1.0))
        train_hist[epoch].copy_(step())
        if c0.ckpt_every and (epoch + 1) % c0.ckpt_every == 0 and epoch + 1 < E and save:
            for i, rd in enumerate(run_dirs):
                save_file({k: v[i].detach().cpu().contiguous() for k, v in pr.items()},
                          str(rd / f"ckpt_{epoch + 1:06d}.safetensors"))
        if (epoch + 1) % log_every == 0:
            tr = train_hist[epoch].cpu().numpy()
            te_ce, te_ac = test_stats()
            cf = snap["coeffs"][-1].abs() if snap["coeffs"] else None
            kt = ((cf >= 0.1 * cf.max(1, keepdim=True).values).sum(1).cpu().numpy()
                  if cf is not None else np.full(M, -1))
            eps_s = (epoch + 1) / (time.time() - t0)
            hist = dict(sorted(collections.Counter(kt.tolist()).items()))
            print(f"epoch {epoch+1:6d}  train CE {np.median(tr):.4e}  test CE {np.nanmedian(te_ce):.4e}"
                  f"  test acc {np.nanmedian(te_ac):.4f}  K_thr med {np.median(kt):.1f} "
                  f"(snap ep {snap['epochs'][-1] if snap['epochs'] else -1}, {hist})"
                  f"  ({eps_s:.2f} ep/s x {M} = {eps_s*M:.1f} run-eps/s)", flush=True)
            test_hist.append(te_ce)
    take_snapshot(E)
    if not save:
        return pr, snap
    train_all = train_hist.cpu().numpy()
    test_all = np.stack(test_hist) if test_hist else np.zeros((0, M))
    sp = {k: torch.stack(snap[k]).cpu().numpy() for k in snap if k != "epochs"}
    for i, rd in enumerate(run_dirs):
        save_file({k: v[i].detach().cpu().contiguous() for k, v in pr.items()},
                  str(rd / "final.safetensors"))
        np.savez(rd / "spectra.npz", epochs=np.array(snap["epochs"]),
                 **{k: v[:, i] for k, v in sp.items()})
        (rd / "metrics.json").write_text(json.dumps({
            "train_losses": train_all[:, i].tolist(),
            "test_losses": test_all[:, i].tolist(), "test_every": log_every,
            "final_K_gap": committee_from_coeffs(sp["coeffs"][-1, i]),
            "final_min_margin_norm": float(sp["min_margin_norm"][-1, i])}))
    print(f"done: {E} epochs x {M} runs in {time.time()-t0:.0f}s", flush=True)
    return pr, snap


# ------------------------------------------------------------------ tests ---
def _einsum_instrument(L, p):
    """train_nudial.take_snapshot's instrument, verbatim, on (M, p^2, p) f32."""
    from train_nudial import fourier_basis
    basis = fourier_basis(p); n_ = p // 2
    bc, bs = basis[1::2], basis[2::2]
    u_cos = torch.from_numpy((np.einsum("np,nq->npq", bc, bc) - np.einsum("np,nq->npq", bs, bs))
                             .reshape(n_, -1) / np.sqrt(2))
    u_sin = torch.from_numpy((np.einsum("np,nq->npq", bs, bc) + np.einsum("np,nq->npq", bc, bs))
                             .reshape(n_, -1) / np.sqrt(2))
    b_cos = torch.from_numpy(bc); b_sin = torch.from_numpy(bs)
    Ld = L.double()
    c = torch.einsum("fp,mpo->mfo", u_cos, Ld); s = torch.einsum("fp,mpo->mfo", u_sin, Ld)
    energy = (c ** 2).sum(-1) + (s ** 2).sum(-1)
    coeffs = ((c * b_cos).sum(-1) + (s * b_sin).sum(-1)) / math.sqrt(2)
    return coeffs, energy


def _fft_instrument(L, p):
    M = L.shape[0]
    T = torch.zeros(M, p, p, dtype=torch.float64)
    collapse_add(T, L, 0, p)
    return spectra_from_T(T, p)


def _rel(x, y):
    return float((x - y).abs().max() / y.abs().max().clamp(min=1e-300))


def test_fft(p=113):
    print(f"\n=== TEST 2: FFT spectra instrument vs train_nudial einsum (p={p}) ===")
    torch.manual_seed(0)
    L = torch.randn(2, p * p, p)
    ce, ee = _einsum_instrument(L, p); cf, ef = _fft_instrument(L, p)
    print(f"random logits     : coeffs rel {_rel(cf, ce):.3e}  energy rel {_rel(ef, ee):.3e}")
    ok = _rel(cf, ce) < 1e-6 and _rel(ef, ee) < 1e-6

    src = sorted((ROOT / "runs_nudial" / "M").glob("relu4_m*_p113_*/*_s*/final.safetensors"))
    cfgp = None
    if src:
        from safetensors.torch import load_file
        rd = src[0].parent
        cfgd = json.loads((rd / "config.json").read_text())
        cfg = Config(**{k: v for k, v in cfgd.items() if k in Config.__dataclass_fields__})
        pr = {k: v[None] for k, v in load_file(src[0]).items()}
        tag = f"cohort-M final {rd.parent.name}/{rd.name}"
    else:
        cfg = Config(p=p, family="relu", nu=4, m=512, init_seed=1)
        pr = {k: v[None] for k, v in init_params(cfg).items()}
        tag = "random-init relu4 net (no local cohort-M finals)"
    with torch.no_grad():
        a = torch.arange(p).repeat_interleave(p); b = torch.arange(p).repeat(p)
        L = forward(pr, a, b, cfg)
    ce, ee = _einsum_instrument(L, p); cf, ef = _fft_instrument(L, p)
    r1, r2 = _rel(cf, ce), _rel(ef, ee)
    print(f"{tag}: coeffs rel {r1:.3e}  energy rel {r2:.3e}")
    print(f"  top-6 |coeff| (fft)    {np.sort(cf[0].abs().numpy())[::-1][:6].round(6).tolist()}")
    print(f"  top-6 |coeff| (einsum) {np.sort(ce[0].abs().numpy())[::-1][:6].round(6).tolist()}")
    ok = ok and r1 < 1e-6 and r2 < 1e-6
    print("TEST 2:", "PASS" if ok else "FAIL")
    return ok


def test_grad(p=23, m=64, seeds=2, epochs=200):
    import shutil, tempfile
    from train_nudial import per_ex_stable_f32, train_batched
    print(f"\n=== TEST 1: chunked == unchunked (p={p} m={m} {seeds} runs) ===")
    ss = draw_seeds(32)[:seeds]
    cfgs = [Config(p=p, family="relu", nu=4, m=m, mode="train", frac_train=0.5,
                   weight_decay=1.0, init_seed=s, data_seed=s % 1000,
                   num_epochs=epochs) for s in ss]
    for c in cfgs:
        c.nu = c.degree
    c0 = cfgs[0]
    # ---- (a) gradients
    per_run = [init_params(c) for c in cfgs]
    keys = list(param_shapes(c0))
    masks = [train_test_split(c) for c in cfgs]
    tr = torch.from_numpy(np.stack([np.flatnonzero(mk) for mk in masks]))
    A, B = tr // p, tr % p; lab = (A + B) % p
    n_tr = tr.shape[1]

    def grads_of(fn):
        pr = {k: torch.stack([r[k] for r in per_run]).clone().requires_grad_() for k in keys}
        return [g.detach() for g in torch.autograd.grad(fn(pr), [pr[k] for k in keys])]

    def unchunked(pr):     # train_nudial's exact objective
        return per_ex_stable_f32(forward(pr, A, B, c0), lab).mean(1).sum()

    def chunked(pr):
        tot = 0.0
        for s, e in split_bounds(n_tr, 5):
            tot = tot + per_ex_ce_lean(forward(pr, A[:, s:e], B[:, s:e], c0),
                                       lab[:, s:e]).sum(1).sum() / n_tr
        return tot

    g0, g1 = grads_of(unchunked), grads_of(chunked)
    rels = [float((x - y).abs().max() / y.abs().max()) for x, y in zip(g1, g0)]
    print("  grad rel err per tensor: " +
          "  ".join(f"{k} {r:.2e}" for k, r in zip(keys, rels)))
    ok_g = max(rels) < 1e-6
    print(f"  (a) chunked (5 chunks) vs unchunked gradient: worst rel {max(rels):.2e} "
          f"{'<' if ok_g else '>='} 1e-6")

    # ---- (b) trajectory. In f64 the two trajectories are the SAME trajectory.
    # In f32 they are not, and cannot be: the 1e-7 summation-order difference in
    # the gradient is renormalised to O(1) by Adam's m/(sqrt(v)+eps) and grows
    # exponentially.  The control below runs OUR OWN trainer at two different
    # chunk counts: same drift => it is float arithmetic, not the port.
    from safetensors.torch import load_file
    tmp = Path(tempfile.mkdtemp())

    def traj(dtype):
        old = torch.get_default_dtype(); torch.set_default_dtype(dtype)
        try:
            tag = str(dtype).split(".")[-1]
            ref = [tmp / f"{tag}ref" / f"s{s}" for s in ss]
            n8 = [tmp / f"{tag}n8" / f"s{s}" for s in ss]
            n3 = [tmp / f"{tag}n3" / f"s{s}" for s in ss]
            train_batched(cfgs, ref, compile_mode="off", log_every=10 ** 9,
                          spectra_every=0, device="cpu")
            train_chunked(cfgs, n8, mem_gb=5e-4, compile_mode="off", log_every=10 ** 9,
                          spectra_every=0, device="cpu", quiet=True)   # 8 chunks
            train_chunked(cfgs, n3, mem_gb=2e-3, compile_mode="off", log_every=10 ** 9,
                          spectra_every=0, device="cpu", quiet=True)   # 2 chunks
        finally:
            torch.set_default_dtype(old)
        def cmp(x, y):
            w = l = 0.0
            for u, v in zip(x, y):
                a_, b_ = load_file(u / "final.safetensors"), load_file(v / "final.safetensors")
                w = max(w, max(float((a_[k] - b_[k]).abs().max() / a_[k].abs().max()) for k in a_))
                la = np.array(json.loads((u / "metrics.json").read_text())["train_losses"])
                lb = np.array(json.loads((v / "metrics.json").read_text())["train_losses"])
                l = max(l, float(np.abs(la - lb).max() / np.abs(la).max()))
            return w, l
        return cmp(ref, n8), cmp(n8, n3)

    (w64, l64), (wq, lq) = traj(torch.float64)
    (w32, l32), _ = traj(torch.float32)

    # Lyapunov control: train_batched against ITSELF from two inits differing by
    # exactly the f32 gradient difference measured in (a).
    from dataclasses import replace
    from safetensors.torch import save_file as _sf
    gsd = torch.Generator().manual_seed(7)
    for tag, eps_ in (("i0", 0.0), ("i1", float(max(rels)))):
        for c in cfgs:
            d = (tmp / tag / f"s{c.init_seed}"); d.mkdir(parents=True, exist_ok=True)
            q = init_params(c)
            _sf({k: (v * (1 + eps_ * torch.randn(v.shape, generator=gsd))).contiguous()
                 for k, v in q.items()}, str(d / "final.safetensors"))
    ctl = []
    for tag in ("i0", "i1"):
        cs = [replace(c, init_from=str(tmp / tag / f"s{c.init_seed}" / "final.safetensors"))
              for c in cfgs]
        out = [tmp / f"{tag}o" / f"s{c.init_seed}" for c in cfgs]
        train_batched(cs, out, compile_mode="off", log_every=10 ** 9, spectra_every=0, device="cpu")
        ctl.append(out)
    wc = max(max(float((load_file(u / "final.safetensors")[k] - load_file(v / "final.safetensors")[k]).abs().max()
                       / load_file(u / "final.safetensors")[k].abs().max()) for k in ("E1", "U", "V", "W1", "W"))
             for u, v in zip(*ctl))
    shutil.rmtree(tmp, ignore_errors=True)
    print(f"  (b) after {epochs} epochs, f64 (exact arithmetic regime):")
    print(f"        chunked vs train_batched   params rel {w64:.2e}  loss curve rel {l64:.2e}")
    print(f"        8 chunks vs 2 chunks       params rel {wq:.2e}  loss curve rel {lq:.2e}")
    print(f"      after {epochs} epochs, f32:")
    print(f"        chunked vs train_batched   params rel {w32:.2e}  loss curve rel {l32:.2e}")
    print(f"      f32 CONTROL train_batched vs itself from inits {max(rels):.0e} apart: "
          f"params rel {wc:.2e}")
    print(f"      -> Adam amplifies a gradient-sized {max(rels):.0e} perturbation to O({wc:.0e}) "
          f"in {epochs} epochs; the f32 drift is that, not a port difference.")
    ok = ok_g and max(w64, l64, wq, lq) < 1e-10 and w32 < 10 * max(wc, 1e-12)
    print("TEST 1:", "PASS" if ok else "FAIL")
    return ok


# ------------------------------------------------------------------- main ---
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p", type=int, default=1009)
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--seeds", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=10_000)
    ap.add_argument("--mem-gb", type=float, default=28.0)
    ap.add_argument("--batch", type=int, default=0, help="runs stacked at once (0 = all)")
    ap.add_argument("--compile", default="default")
    ap.add_argument("--log-every", type=int, default=500)
    ap.add_argument("--spectra-every", type=int, default=100)
    ap.add_argument("--device", default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--opt", default="adamw", choices=["adamw", "gd", "gdm", "ngd"])
    ap.add_argument("--lr", type=float, default=None, help="default 1e-3 adamw, 0.1 gdm")
    ap.add_argument("--wd", type=float, default=None, help="default 1.0 adamw, 1e-3 gd-family")
    ap.add_argument("--ckpt-every", type=int, default=0)
    ap.add_argument("--seed-offset", type=int, default=0)
    ap.add_argument("--clip", type=float, default=0.0)
    args = ap.parse_args()
    if args.test:
        ok = test_grad() & test_fft()
        sys.exit(0 if ok else 1)
    if args.smoke:
        args.p, args.width, args.seeds, args.epochs = 23, 64, 2, 300
        args.compile, args.device = "off", "cpu"
        args.log_every, args.spectra_every = 100, 100

    seeds = draw_seeds(32)[args.seed_offset:args.seed_offset + args.seeds]
    ad = args.opt == "adamw"
    lr = args.lr if args.lr is not None else (1e-3 if ad else 0.1)
    wd = args.wd if args.wd is not None else (1.0 if ad else 1e-3)
    cell = f"relu4_m{args.width}_p{args.p}_f0.5_train" + ("" if ad else f"_{args.opt}")
    cfgs, dirs = [], []
    for s in seeds:
        c = Config(p=args.p, family="relu", nu=4, m=args.width, d=128, r=128,
                   frac_train=0.5, mode="train", weight_decay=wd, lr=lr, opt=args.opt,
                   warmup_steps=10 if ad else 1, ckpt_every=args.ckpt_every, clip=args.clip,
                   init_scale=1.0, init_seed=s, data_seed=s % 1000,
                   num_epochs=args.epochs)
        c.nu = c.degree
        rd = RUNS / f"P{args.p}" / cell / (f"wd{c.weight_decay:g}_s{s}" if ad else f"wd{c.weight_decay:g}_lr{c.lr:g}" + (f"_clip{c.clip:g}" if c.clip else "") + f"_s{s}")
        if (rd / "spectra.npz").exists():
            print(f"skip {rd} (done)"); continue
        cfgs.append(c); dirs.append(rd)
    if not cfgs:
        print("nothing to do"); return
    B = args.batch or len(cfgs)
    for i in range(0, len(cfgs), B):
        print(f"[P{args.p}] {cell}: runs {i}..{i+len(cfgs[i:i+B])-1} of {len(cfgs)}", flush=True)
        train_chunked(cfgs[i:i + B], dirs[i:i + B], mem_gb=args.mem_gb,
                      compile_mode=args.compile, log_every=args.log_every,
                      spectra_every=args.spectra_every, device=args.device)


if __name__ == "__main__":
    main()
