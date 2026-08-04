"""SEMIFINAL v2 on CUDA — self-contained torch port of grok/batched.py plus
the train_semifinal_v2.py protocol. Imports NOTHING from the MLX codebase.

Same experiments as semifinal/training/train_semifinal_v2.py: 44 runs,
p=113, 20k epochs, spectra every 100, checkpoints every 1000 (epoch 0 saved
pre-update), idempotent (spectra.npz marks done), priority order preserved:
24 from-scratch (nat/orthWE/doubleflat, cells A+B) then 2 steering suites
(dose/suppress/gkrotate/chaospair/collision) on bases seed61001/seed61002.

Lockstep batched training, exactly as the MLX version: weights stacked on a
leading run axis, loss = sum of per-run mean CEs (separable -> exact per-run
grads), per-run AdamW in lockstep. Optimizer matches MLX semantics: NO bias
correction, decoupled wd as p*(1-lr*wd) — the homeostat eps-floor result
depends on this exact form; torch.optim.AdamW (bias-corrected) would NOT
reproduce it.

Loss: stable-f32 GPU CE softplus(lse_{j!=y}(z) - z_y) (full relative
precision to ~1e-38; the regime f64 guarded). --loss f64 for the original.
TF32 is OFF by default (precision-sensitive); --tf32 trades exactness for
~2x matmul speed.

Checkpoint/artifact formats are byte-compatible with the MLX pipeline:
safetensors with MLX key names (embed.W_E, blocks.0.attn.W_K, ...),
config.json with the full Config schema, spectra.npz, metrics.json.

wandb: set WANDB_API_KEY (and optionally WANDB_PROJECT / WANDB_ENTITY).
Metrics stream per batch; each finished run's directory is uploaded as an
artifact (type "grok-run"). No key -> wandb disabled, training unaffected.

Run:   python -u train_semifinal_torch.py            (the full 44)
       python -u train_semifinal_torch.py --dry-run
       python -u train_semifinal_torch.py --smoke    (tiny CPU sanity check)
"""
import argparse
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

try:
    import wandb
except ImportError:
    wandb = None

RUNS = Path(os.environ.get("RUNS_DIR", "runs"))
CKPT_DIR = RUNS / "_surgical_ckpt"

P = 113
NF = P // 2
EPOCHS = 20000
CELLS = {4811: [61001, 61002, 61003, 61004],
         7207: [72001, 72002, 72003, 72004]}
STEER_BASES = ["p-113/seed4811/seed61001", "p-113/seed4811/seed61002"]
DOSES = [1.10, 1.20, 1.50, 2.25]
GAINS = [1.20, 2.25]


# --------------------------------------------------------------------------
# config / data (schema- and split-identical to the MLX codebase)
# --------------------------------------------------------------------------

@dataclass
class Config:
    p: int = 113
    fn_name: str = "add"
    frac_train: float = 0.3
    data_seed: int = 0
    init_seed: int = 0
    num_layers: int = 1
    d_model: int = 128
    embed_init: str = "normal"      # 'normal' | 'orthogonal'
    attn_init: str = "normal"       # 'normal' | 'isometric'
    num_heads: int = 4
    d_mlp: int = 512
    n_ctx: int = 3
    act_type: str = "ReLU"
    lr: float = 1e-3
    warmup_steps: int = 10
    weight_decay: float = 1.0
    betas: tuple = (0.9, 0.98)
    adam_eps: float = 1e-8
    loss_dtype: str = "f64"
    loss_tilt: float = 0.0
    loss_cvar: float = 0.0
    grad_noise: float = 0.0
    grad_noise_until: int = 0
    num_epochs: int = 50_000
    save_every: int = 100
    stopping_thresh: float = -1.0

    @property
    def d_vocab(self):
        return self.p + 1

    @property
    def d_head(self):
        assert self.d_model % self.num_heads == 0
        return self.d_model // self.num_heads

    def save(self, path):
        d = asdict(self)
        d["betas"] = list(d["betas"])
        Path(path).write_text(json.dumps(d, indent=2))

    @classmethod
    def load(cls, path):
        d = json.loads(Path(path).read_text())
        d["betas"] = tuple(d["betas"])
        return cls(**d)


def make_dataset(cfg):
    """All p^2 pairs [a, b, =] lexicographic; labels (a+b)%p."""
    assert cfg.fn_name == "add"
    p = cfg.p
    a = np.repeat(np.arange(p), p)
    b = np.tile(np.arange(p), p)
    tokens = np.stack([a, b, np.full(p * p, p)], axis=1).astype(np.int64)
    labels = ((a + b) % p).astype(np.int64)
    return tokens, labels


def train_test_split(cfg):
    """Boolean is_train over the lexicographic batch — EXACT replica of the
    MLX/original-torch split (python stdlib random with data_seed)."""
    p = cfg.p
    pairs = [(i, j) for i in range(p) for j in range(p)]
    random.seed(cfg.data_seed)
    random.shuffle(pairs)
    train_set = set(pairs[:int(cfg.frac_train * len(pairs))])
    return np.array([(i, j) in train_set for i in range(p) for j in range(p)])


def fourier_basis(p):
    """(p, p) f64: rows [const, cos1, sin1, cos2, ...], each normalized."""
    basis = [np.ones(p) / np.sqrt(p)]
    for k in range(1, p // 2 + 1):
        for trig in (np.cos, np.sin):
            v = trig(2 * np.pi * k * np.arange(p) / p)
            basis.append(v / np.linalg.norm(v))
    return np.stack(basis).astype(np.float64)


# --------------------------------------------------------------------------
# init (same distributions as the MLX model; torch RNG stream, so draws are
# fresh — fine: every run is defined by its own seed within this backend)
# --------------------------------------------------------------------------

PARAM_KEYS = ["embed.W_E", "pos_embed.W_pos",
              "blocks.0.attn.W_K", "blocks.0.attn.W_Q",
              "blocks.0.attn.W_V", "blocks.0.attn.W_O",
              "blocks.0.mlp.W_in", "blocks.0.mlp.b_in",
              "blocks.0.mlp.W_out", "blocks.0.mlp.b_out",
              "unembed.W_U"]


def init_params(cfg):
    assert cfg.num_layers == 1
    g = torch.Generator().manual_seed(cfg.init_seed)
    d, v, h, dh, m = cfg.d_model, cfg.d_vocab, cfg.num_heads, cfg.d_head, cfg.d_mlp
    n = lambda *s: torch.randn(*s, generator=g) / math.sqrt(d)

    W_E = n(d, v)
    if cfg.embed_init == "orthogonal":
        # semi-orthogonal columns: W_E^T W_E = I_v (needs d >= v)
        q, _ = np.linalg.qr(W_E.double().numpy())
        W_E = torch.from_numpy(q.astype(np.float32))
    else:
        assert cfg.embed_init == "normal"
    W_K, W_Q, W_V = n(h, dh, d), n(h, dh, d), n(h, dh, d)
    W_O = n(d, h * dh)
    if cfg.attn_init == "isometric":
        # QR the W_V draw: per-head value subspaces tile d_model, W_O = Q^T
        gm = W_V.double().numpy().reshape(h * dh, d)
        q, _ = np.linalg.qr(gm.T)          # (d, h*dh)
        W_V = torch.from_numpy(q.T.astype(np.float32)).reshape(h, dh, d)
        W_O = torch.from_numpy(q.astype(np.float32))
    else:
        assert cfg.attn_init == "normal"
    return dict(zip(PARAM_KEYS, [
        W_E, n(cfg.n_ctx, d), W_K, W_Q, W_V, W_O,
        n(m, d), torch.zeros(m), n(d, m), torch.zeros(d),
        torch.randn(d, v, generator=g) / math.sqrt(v)]))


# --------------------------------------------------------------------------
# batched forward + losses (mirrors grok/batched.py: M is the ONLY batch dim
# in every matmul; tiny T=3 attention contractions stay elementwise)
# --------------------------------------------------------------------------

def batched_final_logits(pr, tokens, cfg):
    """pr: stacked params {key: (M, ...)}. tokens (N, T) shared or (M, N, T)
    per-run int64. Returns final-position logits (M, N, d_vocab)."""
    M = pr["embed.W_E"].shape[0]
    d, h, dh = cfg.d_model, cfg.num_heads, cfg.d_head
    T = tokens.shape[-1]
    E = pr["embed.W_E"].transpose(1, 2)                        # (M, v, d)
    if tokens.dim() == 2:
        x = E[:, tokens]                                       # (M, N, T, d)
    else:
        idx = tokens.reshape(M, -1, 1).expand(-1, -1, d)
        x = E.gather(1, idx).reshape(M, *tokens.shape[1:], d)
    x = x + pr["pos_embed.W_pos"][:, None, :T, :]
    N = x.shape[1]

    def proj(y, W):    # (M, R, d) @ per-run (h*dh, d)^T
        return y @ W.reshape(M, h * dh, d).transpose(1, 2)

    xf_ = x.reshape(M, N * T, d)
    k = proj(xf_, pr["blocks.0.attn.W_K"]).reshape(M, N, T, h, dh)
    v = proj(xf_, pr["blocks.0.attn.W_V"]).reshape(M, N, T, h, dh)
    q = proj(x[:, :, -1], pr["blocks.0.attn.W_Q"]).reshape(M, N, 1, h, dh)
    attn = torch.softmax((k * q).sum(-1) / math.sqrt(dh), dim=2)  # (M,N,T,h)
    z = (attn.unsqueeze(-1) * v).sum(2).reshape(M, N, h * dh)
    xf = x[:, :, -1] + z @ pr["blocks.0.attn.W_O"].transpose(1, 2)
    pre = xf @ pr["blocks.0.mlp.W_in"].transpose(1, 2) + pr["blocks.0.mlp.b_in"][:, None]
    act = F.relu(pre) if cfg.act_type == "ReLU" else F.gelu(pre)
    xf = xf + act @ pr["blocks.0.mlp.W_out"].transpose(1, 2) + pr["blocks.0.mlp.b_out"][:, None]
    return xf @ pr["unembed.W_U"]


def ce_stable_f32(logits, labels):
    """(M,) mean CE per run: softplus(lse_{j!=y} - z_y). Full f32 relative
    precision down to ~1e-38 (naive f32 log_softmax dies at ~1e-7)."""
    z_y = logits.gather(-1, labels.unsqueeze(-1))
    others = logits.masked_fill(
        F.one_hot(labels, logits.shape[-1]).bool(), float("-inf"))
    u = torch.logsumexp(others, dim=-1, keepdim=True) - z_y
    return F.softplus(u).mean(dim=(1, 2))


def ce_f64(logits, labels):
    z = logits.double()
    lp = z - torch.logsumexp(z, dim=-1, keepdim=True)
    return -lp.gather(-1, labels.unsqueeze(-1)).mean(dim=(1, 2)).float()


# --------------------------------------------------------------------------
# training
# --------------------------------------------------------------------------

def save_ckpts(pr, run_dirs, epoch):
    for i, rd in enumerate(run_dirs):
        save_file({k: v[i].detach().cpu().contiguous() for k, v in pr.items()},
                  str(rd / "checkpoints" / f"epoch_{epoch:05d}.safetensors"))


def train_batched(cfgs, run_dirs, init_from=None, log_every=100,
                  spectra_every=100, loss="f32stable", compile_mode="default",
                  device=None, wb=None):
    M = len(cfgs)
    c0 = cfgs[0]
    for c in cfgs:
        for f in ("p", "fn_name", "frac_train", "d_model", "num_heads", "d_mlp",
                  "n_ctx", "act_type", "lr", "warmup_steps", "weight_decay",
                  "betas", "adam_eps", "num_epochs", "save_every",
                  "stopping_thresh"):
            assert getattr(c, f) == getattr(c0, f), f"batch must agree on {f}"
        assert c.num_layers == 1
        assert c.loss_tilt == 0 and c.loss_cvar == 0 and c.grad_noise == 0
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    run_dirs = [Path(rd) for rd in run_dirs]
    for cfg, rd in zip(cfgs, run_dirs):
        (rd / "checkpoints").mkdir(parents=True, exist_ok=True)
        cfg.save(rd / "config.json")

    # stack per-run inits (fresh torch draws, or warm-start checkpoints)
    per_run = []
    for i, cfg in enumerate(cfgs):
        p = init_params(cfg)
        if init_from is not None and init_from[i] is not None:
            p = load_file(str(init_from[i]))
        per_run.append(p)
    pr = {k: torch.stack([r[k] for r in per_run]).to(device).requires_grad_()
          for k in PARAM_KEYS}
    params = list(pr.values())
    m_st = [torch.zeros_like(p) for p in params]
    v_st = [torch.zeros_like(p) for p in params]
    b1, b2 = c0.betas
    eps, wd = c0.adam_eps, c0.weight_decay
    lr_t = torch.zeros((), device=device)   # in-place updated: no recompiles

    tokens_np, labels_np = make_dataset(c0)
    masks = [train_test_split(c) for c in cfgs]
    tr_tok = torch.from_numpy(np.stack([tokens_np[m] for m in masks])).to(device)
    tr_lab = torch.from_numpy(np.stack([labels_np[m] for m in masks])).to(device)
    te_tok = torch.from_numpy(np.stack([tokens_np[~m] for m in masks])).to(device)
    te_lab = torch.from_numpy(np.stack([labels_np[~m] for m in masks])).to(device)
    grid = torch.from_numpy(tokens_np).to(device)
    ce = ce_stable_f32 if loss == "f32stable" else ce_f64
    print(f"batch {M} on {device}:  train {tr_tok.shape[1]}  test {te_tok.shape[1]}",
          flush=True)

    # spectra machinery (GPU f64; tiny matmuls)
    basis = fourier_basis(c0.p)
    n_ = c0.p // 2
    bc, bs = basis[1::2], basis[2::2]                       # (n, p)
    u_cos = torch.from_numpy((np.einsum("np,nq->npq", bc, bc)
                              - np.einsum("np,nq->npq", bs, bs))
                             .reshape(n_, -1) / np.sqrt(2)).to(device)
    u_sin = torch.from_numpy((np.einsum("np,nq->npq", bs, bc)
                              + np.einsum("np,nq->npq", bc, bs))
                             .reshape(n_, -1) / np.sqrt(2)).to(device)
    b_cos = torch.from_numpy(bc).to(device)
    b_sin = torch.from_numpy(bs).to(device)
    lab_t = torch.from_numpy(labels_np).to(device)
    mask_t = torch.from_numpy(np.stack(masks)).to(device)
    spectra = [{"epochs": [], "coeffs": [], "energy": [],
                "train_acc": [], "test_acc": []} for _ in range(M)]

    @torch.no_grad()
    def take_snapshot(epoch):
        L = batched_final_logits(pr, grid, c0)[..., :c0.p]
        acc = L.argmax(-1) == lab_t                          # (M, p^2)
        Ld = L.double()
        c = torch.einsum("fp,mpo->mfo", u_cos, Ld)           # (M, n, p)
        s = torch.einsum("fp,mpo->mfo", u_sin, Ld)
        energy = (c ** 2).sum(-1) + (s ** 2).sum(-1)
        coeffs = ((c * b_cos).sum(-1) + (s * b_sin).sum(-1)) / math.sqrt(2)
        tr_a = (acc & mask_t).sum(-1) / mask_t.sum(-1)
        te_a = (acc & ~mask_t).sum(-1) / (~mask_t).sum(-1)
        for i in range(M):
            spectra[i]["epochs"].append(epoch)
            spectra[i]["coeffs"].append(coeffs[i].cpu().numpy())
            spectra[i]["energy"].append(energy[i].cpu().numpy())
            spectra[i]["train_acc"].append(float(tr_a[i]))
            spectra[i]["test_acc"].append(float(te_a[i]))

    def step():
        per = ce(batched_final_logits(pr, tr_tok, c0), tr_lab)
        grads = torch.autograd.grad(per.sum(), params)
        with torch.no_grad():
            # MLX-semantics AdamW: NO bias correction, p <- p(1-lr*wd) - lr*m/(sqrt(v)+eps)
            for p, g, m_, v_ in zip(params, grads, m_st, v_st):
                m_.mul_(b1).add_(g, alpha=1 - b1)
                v_.mul_(b2).addcmul_(g, g, value=1 - b2)
                p.mul_(1 - lr_t * wd)
                p.sub_(lr_t * m_ / (v_.sqrt() + eps))
        return per.detach()

    if compile_mode != "off":
        step = torch.compile(step, mode=None if compile_mode == "default"
                             else compile_mode)

    train_hist, test_hist = [], []
    t0 = time.time()
    for epoch in range(c0.num_epochs):
        if epoch % c0.save_every == 0:
            save_ckpts(pr, run_dirs, epoch)
        if spectra_every and epoch % spectra_every == 0:
            take_snapshot(epoch)
        lr_t.fill_(c0.lr * min(epoch / c0.warmup_steps, 1.0))
        train_hist.append(step())

        if (epoch + 1) % log_every == 0:
            with torch.no_grad():
                te = ce(batched_final_logits(pr, te_tok, c0), te_lab).cpu().numpy()
            test_hist.append(te)
            tr = train_hist[-1].cpu().numpy()
            eps_s = (epoch + 1) / (time.time() - t0)
            print(f"epoch {epoch + 1:6d}  train {tr.mean():.4e}  "
                  f"test mean {te.mean():.4e} max {te.max():.4e}  "
                  f"({eps_s:.1f} eps/s x {M} = {eps_s * M:.0f} run-eps/s)",
                  flush=True)
            if wb:
                tag = Path(run_dirs[0]).name
                wb.log({"epoch": epoch + 1, f"{tag}/train_mean": float(tr.mean()),
                        f"{tag}/test_mean": float(te.mean()),
                        f"{tag}/test_max": float(te.max()),
                        "run_eps_per_s": eps_s * M})
            if te.max() < c0.stopping_thresh:
                break

    epoch = len(train_hist)
    save_ckpts(pr, run_dirs, epoch)
    if spectra_every:
        take_snapshot(epoch)
    train_all = torch.stack(train_hist).cpu().numpy()        # (E, M)
    test_all = np.stack(test_hist) if test_hist else np.zeros((0, M))
    for i, rd in enumerate(run_dirs):
        np.savez(rd / "spectra.npz",
                 **{k: np.array(v) for k, v in spectra[i].items()})
        (rd / "metrics.json").write_text(json.dumps({
            "train_losses": train_all[:, i].tolist(),
            "test_losses": test_all[:, i].tolist(),
            "test_every": log_every}))
    print(f"done: {epoch} epochs x {M} runs in {time.time() - t0:.0f}s", flush=True)


# --------------------------------------------------------------------------
# v2 protocol helpers (ported verbatim from _shared / train_semifinal_v2)
# --------------------------------------------------------------------------

def fold(x, p):
    x %= p
    return min(x, p - x)


def committee_from_coeffs(coeffs, floor=0.02):
    a = np.abs(coeffs)
    order = np.argsort(a)[::-1]
    logs = np.log(a[order] + 1e-12)
    cut = int(np.argmax(logs[:-1][:12] - logs[1:][:12])) + 1
    mem = order[:cut] + 1
    mem = mem[a[mem - 1] >= floor * a.max()]
    return sorted(mem.tolist())


def grok_epoch(z):
    gi = int(np.argmax(z["test_acc"] >= 0.99))
    return int(z["epochs"][gi]) if z["test_acc"][gi] >= 0.99 else -1


def report(run_dir, wb=None):
    z = np.load(run_dir / "spectra.npz")
    final = committee_from_coeffs(z["coeffs"][-1])
    name = str(run_dir.relative_to(RUNS))
    print(f"RESULT {name}: grok@{grok_epoch(z)}  committee {final}  "
          f"acc {z['test_acc'][-1]:.4f}", flush=True)
    if wb:
        art = wandb.Artifact(name.replace("/", "__"), type="grok-run")
        art.add_dir(str(run_dir))
        # wait() makes the upload synchronous: a run only counts as done
        # once its artifact is committed, so a dead box can't strand a
        # finished-locally-but-never-uploaded run (resume would skip it).
        handle = wb.log_artifact(art)
        try:
            handle.wait()
        except Exception as e:  # offline mode / transient — don't kill training
            print(f"    (artifact upload not confirmed for {name}: {e})", flush=True)


def freq_energy_F(Fm, nf):
    E = (Fm ** 2).sum(0)
    return E[1::2][:nf] + E[2::2][:nf]


def pick_target(comm, menu, zfin):
    excl = set(menu) | set(comm)
    for k in comm:
        excl |= {fold(2 * k, P), fold(3 * k, P)}
        for j in comm:
            if j != k:
                excl |= {fold(k + j, P), fold(k - j, P)}
    return sorted((k for k in range(1, NF + 1) if k not in excl),
                  key=lambda k: abs(zfin[k - 1]))


def surgical_ckpt(base, scales=None, gk=None, tag=""):
    """Epoch-0 ckpt of `base` with W_E's Fourier energies edited; returns
    (ckpt_path, training Config). Same constructions as the MLX scripts."""
    basis = fourier_basis(P)
    ck0 = load_file(str(base / "checkpoints" / "epoch_00000.safetensors"))
    W_E = ck0["embed.W_E"].double().numpy()
    Fm = W_E[:, :P] @ basis.T
    if scales:
        for k, s in scales.items():
            Fm[:, 2 * k - 1] *= np.sqrt(s)
            Fm[:, 2 * k] *= np.sqrt(s)
    if gk:
        k, g_target = gk
        W_V = ck0["blocks.0.attn.W_V"].double().numpy()
        W_O = ck0["blocks.0.attn.W_O"].double().numpy()
        h, dh, _ = W_V.shape
        OVs = [W_O[:, i * dh:(i + 1) * dh] @ W_V[i] for i in range(h)]
        M2 = sum(OV.T @ OV for OV in OVs)
        w_eig, V = np.linalg.eigh(M2)
        top = V[:, np.argsort(w_eig)[::-1][:2]]
        B0 = Fm[:, [2 * k - 1, 2 * k]].copy()
        tk_of = lambda B: sum(((OV @ B) ** 2).sum() for OV in OVs)

        def rotated(alpha):
            B = (1 - alpha) * B0 + alpha * top * np.linalg.norm(B0, axis=0)
            return B * np.linalg.norm(B0, axis=0) / np.linalg.norm(B, axis=0)

        t0 = tk_of(B0)
        assert tk_of(rotated(1.0)) / t0 >= g_target, "gain unreachable"
        lo, hi = 0.0, 1.0
        for _ in range(60):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if tk_of(rotated(mid)) / t0 < g_target else (lo, mid)
        B = rotated((lo + hi) / 2)
        e0 = freq_energy_F(Fm, NF)
        Fm2 = Fm.copy()
        Fm2[:, 2 * k - 1], Fm2[:, 2 * k] = B[:, 0], B[:, 1]
        err = np.abs(freq_energy_F(Fm2, NF) - e0).max() / e0.mean()
        print(f"    gk rotation f{k} gain {g_target}: energy err {err:.1e}",
              flush=True)
        assert err < 1e-7, "gk rotation moved energy"
        Fm = Fm2
    W = W_E.copy()
    W[:, :P] = Fm @ basis
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    out = dict(ck0)
    out["embed.W_E"] = torch.from_numpy(W.astype(np.float32))
    path = CKPT_DIR / f"{tag}.safetensors"
    save_file(out, str(path))
    cfg = Config.load(base / "config.json")
    cfg.num_epochs = EPOCHS
    cfg.save_every = 1000
    return path, cfg


def T_k_spread(p):
    """Doubleflat sanity: per-frequency OV-transmitted energy spread at init."""
    basis = fourier_basis(P)
    W_E = p["embed.W_E"].double().numpy()[:, :P]
    W_V = p["blocks.0.attn.W_V"].double().numpy()
    W_O = p["blocks.0.attn.W_O"].double().numpy()
    h, dh, _ = W_V.shape
    t = np.zeros(NF)
    for i in range(h):
        OV = W_O[:, i * dh:(i + 1) * dh] @ W_V[i]
        t += freq_energy_F((OV @ W_E) @ basis.T, NF)
    return (t.max() - t.min()) / t.mean()


# --------------------------------------------------------------------------
# the plan (identical runs/order to train_semifinal_v2.py)
# --------------------------------------------------------------------------

def batch_train(jobs, width, args, wb):
    todo = []
    for name, cfg, ckpt in jobs:
        if (RUNS / name / "spectra.npz").exists():
            print(f"skip {name} (exists)", flush=True)
        elif args.dry_run:
            print(f"WOULD TRAIN {name} ({cfg.num_epochs} epochs)", flush=True)
        else:
            todo.append((name, cfg, ckpt))
    for i in range(0, len(todo), width):
        chunk = todo[i:i + width]
        print(f"=== batch x{len(chunk)}: {', '.join(n for n, _, _ in chunk)} ===",
              flush=True)
        train_batched([c for _, c, _ in chunk], [RUNS / n for n, _, _ in chunk],
                      init_from=[ck for _, _, ck in chunk], loss=args.loss,
                      compile_mode=args.compile, wb=wb)
        for n, _, _ in chunk:
            report(RUNS / n, wb)


def base_cfg(ds, iseed, **kw):
    return Config(p=P, data_seed=ds, init_seed=iseed, num_epochs=EPOCHS,
                  save_every=1000, **kw)


def doubleflat_cfg(ds, iseed, dry):
    cfg = base_cfg(ds, iseed, embed_init="orthogonal", attn_init="isometric")
    if not dry:
        spread = T_k_spread(init_params(cfg))
        print(f"doubleflat seed{iseed}: init T_k spread {spread:.2e}", flush=True)
        assert spread < 1e-5, "isometric init failed to flatten T_k"
    return cfg


def steering_suite(base_rel, args, wb):
    base = RUNS / base_rel
    bname = base.name
    if args.dry_run:
        print(f"WOULD TRAIN steering suite on {base_rel} (10 arms)", flush=True)
        return
    z = np.load(base / "spectra.npz")
    assert float(z["test_acc"][-1]) >= 0.99, f"base {base_rel} never grokked"
    zfin = z["coeffs"][-1]
    comm = committee_from_coeffs(zfin)
    i3 = int(np.argmin(np.abs(z["epochs"] - 3000)))
    menu = (np.argsort(np.abs(z["coeffs"][i3]))[::-1][:8] + 1).tolist()
    cands = pick_target(comm, menu, zfin)
    target, chaosA, chaosB = cands[0], cands[1], cands[2]
    strongest = max(comm, key=lambda k: abs(zfin[k - 1]))
    print(f"\n### steering suite on {base_rel}: committee {comm}, "
          f"dose/gk target f{target}, chaos f{chaosA}/f{chaosB}, "
          f"suppress f{strongest}", flush=True)

    jobs = []

    def add(run_name, scales=None, gk=None, tag=""):
        if (RUNS / run_name / "spectra.npz").exists():
            print(f"skip {run_name} (exists)", flush=True)
            return
        ck, cfg = surgical_ckpt(base, scales, gk, tag)
        jobs.append((run_name, cfg, ck))

    for d in DOSES:
        add(f"dosefarm/{bname}/dose_{int(d*100):03d}",
            scales={target: d}, tag=f"v2dose_{bname}_{d}")
    add(f"suppress/{bname}", scales={strongest: 0.5}, tag=f"v2sup_{bname}")
    for g in GAINS:
        add(f"gkrotate/{bname}/gain_{int(g*100):03d}",
            gk=(target, g), tag=f"v2gk_{bname}_{g}")
    add(f"chaospair/{bname}/armA", scales={chaosA: 1.5}, tag=f"v2chA_{bname}")
    add(f"chaospair/{bname}/armB", scales={chaosB: 1.5}, tag=f"v2chB_{bname}")
    order = sorted(comm, key=lambda k: -abs(zfin[k - 1]))
    trio_t = next((cand for a in range(len(order)) for b in range(a + 1, len(order))
                   for cand in (fold(order[a] + order[b], P),
                                fold(order[a] - order[b], P))
                   if cand not in (0, order[a], order[b]) and cand not in comm),
                  None)
    assert trio_t, f"no collision target on {base_rel}"
    add(f"collisionfarm/{bname}_t{trio_t}",
        scales={trio_t: 2.25}, tag=f"v2col_{bname}")
    batch_train(jobs, args.width_steer, args, wb)


def smoke(args, wb):
    """Tiny end-to-end sanity: 3 runs (normal/orth/doubleflat), 60 epochs."""
    global P, NF, EPOCHS
    P, NF, EPOCHS = 29, 14, 60
    cfgs = [base_cfg(1, 11), base_cfg(1, 12, embed_init="orthogonal"),
            doubleflat_cfg(1, 13, dry=False)]
    for c in cfgs:
        c.p, c.num_epochs, c.save_every = P, EPOCHS, 60
    dirs = [RUNS / f"smoke/run{i}" for i in range(3)]
    train_batched(cfgs, dirs, log_every=20, spectra_every=20, loss=args.loss,
                  compile_mode=args.compile, wb=wb)
    for d in dirs:
        z = np.load(d / "spectra.npz")
        ck = load_file(str(d / "checkpoints" / "epoch_00000.safetensors"))
        assert set(ck) == set(PARAM_KEYS) and len(z["coeffs"][0]) == P // 2
        tr = json.loads((d / "metrics.json").read_text())["train_losses"]
        assert tr[-1] < tr[0], "loss did not decrease"
        report(d, wb)
    print("SMOKE OK", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--loss", choices=["f32stable", "f64"], default="f32stable")
    ap.add_argument("--compile", default="default",
                    choices=["default", "reduce-overhead", "max-autotune", "off"])
    ap.add_argument("--width-scratch", type=int, default=24)
    ap.add_argument("--width-steer", type=int, default=10)
    ap.add_argument("--tf32", action="store_true",
                    help="~2x matmul speed, NOT numerically faithful")
    args = ap.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = args.tf32
    torch.backends.cudnn.allow_tf32 = args.tf32
    torch.set_float32_matmul_precision("high" if args.tf32 else "highest")

    wb = None
    if wandb and os.environ.get("WANDB_API_KEY") and not args.dry_run:
        wb = wandb.init(project=os.environ.get("WANDB_PROJECT", "grok-semifinal-v2"),
                        config={"loss": args.loss, "tf32": args.tf32})

    if args.smoke:
        smoke(args, wb)
    else:
        print(__doc__, flush=True)
        dsA, dsB = sorted(CELLS)
        jobs = [(f"p-113/seed{dsA}/seed{s}", base_cfg(dsA, s), None)
                for s in CELLS[dsA]]
        jobs += [(f"orthWE/p-113/seed{ds}/seed{s}",
                  base_cfg(ds, s, embed_init="orthogonal"), None)
                 for ds in (dsA, dsB) for s in CELLS[ds]]
        jobs += [(f"doubleflat/p-113/seed{dsA}/seed{s}",
                  doubleflat_cfg(dsA, s, args.dry_run), None) for s in CELLS[dsA]]
        jobs += [(f"p-113/seed{dsB}/seed{s}", base_cfg(dsB, s), None)
                 for s in CELLS[dsB]]
        jobs += [(f"doubleflat/p-113/seed{dsB}/seed{s}",
                  doubleflat_cfg(dsB, s, args.dry_run), None) for s in CELLS[dsB]]
        batch_train(jobs, args.width_scratch, args, wb)
        for b in STEER_BASES:
            steering_suite(b, args, wb)
        print("SEMIFINAL V2 DATASET DONE", flush=True)
    if wb:
        wb.finish()
