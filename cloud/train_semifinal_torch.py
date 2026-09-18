"""SEMIFINAL v2 on CUDA — self-contained torch port of grok/batched.py plus
the train_semifinal_v2.py protocol. Imports NOTHING from the MLX codebase.

Same experiments as semifinal/training/train_semifinal_v2.py, p=113, 50k
epochs, spectra every 100, checkpoints every 1000 (epoch 0 saved pre-update),
idempotent (spectra.npz marks done), priority order preserved: 24
from-scratch (nat/orthWE/doubleflat, cells A+B) then 2 steering suites
(dose/suppress/gkrotate/chaospair/collision) on the first two cell-A
naturals. Seeds are drawn from SEED_DRAW and printed at startup.

Lockstep batched training, exactly as the MLX version: weights stacked on a
leading run axis, loss = sum of per-run mean CEs (separable -> exact per-run
grads), per-run AdamW in lockstep. Perf notes (2026-08-18): batches may be
RAGGED in frac_train (per-run example indices padded mode=edge with zero
weights — padded slots contribute exactly zero loss/grad; equal-count
batches take the historical bit-identical path); compile_mode
"reduce-overhead" (CUDA graphs) is safe — the history buffer copies the
graph-owned step output each epoch; f64 spectra projections are dispatched
to a side CUDA stream (identical math/values, overlapped scheduling). Optimizer matches MLX semantics: NO bias
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

EPOCHS is 50k here vs 20k in the MLX script — the only protocol difference,
taken because cloud compute is cheap and committees still drift late in a
few percent of runs. Everything else is identical (verified numerically).

wandb: set WANDB_API_KEY (and optionally WANDB_PROJECT / WANDB_ENTITY /
WANDB_NAME). NOTHING is left to wandb's animal-name generator. The dashboard
gets one entry per run plus a driver, all readable and greppable:

  semifinal-v2-p113-50k-<timestamp>   one "driver" run: live run-eps/s and
                                      per-batch aggregates while training
  p-113/seed<ds>/seed<is>             one run PER training run, named for
  orthWE/p-113/seed<ds>/seed<is>      its path, grouped by family, carrying
  doubleflat/p-113/seed<ds>/seed<is>  its own Config, train/test CE + acc
  dosefarm/seed<is>/dose_110          curves, committee + grok epoch in the
  gkrotate/seed<is>/gain_225          summary, and its own "grok-run"
  collisionfarm/seed<is>_t<k>         artifact (config/metrics/spectra/ckpts)

Per-run entries are published by report() as each lockstep batch finishes.
No key -> wandb disabled entirely, training unaffected.

Beyond the original 44 the plan also trains (98 runs total):
  claim 2A  6 orth-flat dynamics families x 8 (both cells): dyn-wd25/wd04
            (wd 2.5/0.4), dyn-lr3/lrlo (lr 3e-3/3e-4), phase2-tilt (tilted
            ERM t=5), eff-G (CVaR 0.05). Family names are the ones
            claim2_twins.GROUPS maps, giving 5 recipe groups with orthWE.
            tilt/CVaR are validated against grok.train.cross_entropy_f64 to
            1.3e-7. SAM-lite grad_noise is NOT ported.
  claim 3E  6 transplants, all ordered pairs of three cell-A naturals: the
            recipient's epoch-0 W_E with every frequency's energy rescaled
            to the donor's.

Resume: runs with spectra.npz are skipped, so pointing RUNS_DIR at a volume
that already holds finished runs trains only what is missing. --only/--skip
REGEX filter by run name on top of that (--only 'dyn-|phase2-tilt|eff-G|
transplant/' is exactly the 54 arms added after the first 44).

Run:   python -u train_semifinal_torch.py            (all 98)
       python -u train_semifinal_torch.py --dry-run
       python -u train_semifinal_torch.py --smoke    (tiny CPU sanity check)
"""
import argparse
import json
import math
import os
import random
import re
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
EPOCHS = 50000
# Cohort seeds are DRAWN, not hand-picked: an arbitrary sample from one fixed
# meta-seed, so they carry no author preference yet stay fully reproducible
# (the same SEED_DRAW always yields the same cohort — required for the
# spectra.npz resume to match, and for anyone to rebuild this dataset).
#
# PRE-REGISTERED: SEED_DRAW is set before the runs exist and the cohort is
# kept whatever comes out. The previous set was retired WHOLESALE (not
# filtered) after one orth-flat run landed on a partial generalizer. Retiring
# a whole set is legitimate; dropping individual runs by outcome — or bumping
# SEED_DRAW again because a re-draw also produced one — is survivorship
# selection. Changing this number is a visible one-line commit, which is the
# point: re-rolling after seeing results should be hard to do silently.
# Expect partial generalizers (~1/8 of orth-flat in the last cohort); their
# RATE is a finding to report, not contamination to remove.
SEED_DRAW = 20260805
_rng = random.Random(SEED_DRAW)
# data_seeds fix the train/test masks, so keep them off every mask already
# spent in runs/, runs_torch50k/ and legacy/runs/ — that is what makes this
# cohort independent. init_seed collisions are harmless: a run is identified
# by the (data_seed, init_seed) pair, and the data_seeds here are all fresh.
_SPENT_MASKS = {0, 1, 2, 2034, 3604, 4811, 7207}
CELLS = {ds: sorted(_rng.sample(range(10_000, 100_000), 4))
         for ds in sorted(_rng.sample(
             [s for s in range(1000, 10_000) if s not in _SPENT_MASKS], 2))}
# Derived, so refreshing seeds means editing CELLS and nothing else.
_DSA = sorted(CELLS)[0]
STEER_BASES = [f"p-113/seed{_DSA}/seed{s}" for s in CELLS[_DSA][:2]]
DOSES = [1.10, 1.20, 1.50, 2.25]
GAINS = [1.20, 2.25]

# claim-2A across-dynamics twins: orth-flat runs under qualitatively distinct
# dynamics. Family names are the ones claim2_twins.GROUPS already maps, so the
# recipe-group collapse works with no analysis edit: plain(orthWE) / wd / lr /
# tilt / cvar = 5 groups. tilt and cvar values are the legacy ones
# (phase2-tilt t=5.0, eff-G alpha=0.05). SAM-lite noise is deliberately absent
# — grad_noise is not ported to the batched trainer.
DYNAMICS = [("dyn-wd25", dict(weight_decay=2.5)),
            ("dyn-wd04", dict(weight_decay=0.4)),
            ("dyn-lr3", dict(lr=3e-3)),
            ("dyn-lrlo", dict(lr=3e-4)),
            ("phase2-tilt", dict(loss_tilt=5.0)),
            ("eff-G", dict(loss_cvar=0.05))]
# claim-3E transplants: all ordered pairs of the first three cell-A naturals.
TRANSPLANT_SEEDS = CELLS[_DSA][:3]


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
    # Shortcut pathway (shortcut/): logits += table_scale * T[a, b, :] with a
    # zero-init learnable per-pair table T (p*p, d_vocab) — an explicit,
    # degree-1 memorization route beside the degree->=4 circuit. 0 = off
    # (T then has exactly zero gradient and stays zero: baseline arithmetic
    # unchanged). table_wd_mult scales the decoupled wd applied to T.
    table_scale: float = 0.0
    table_wd_mult: float = 1.0
    # 2026-09-09 clean cohort: optimizer family and homogeneity-degree dial.
    # opt: adamw (MLX-semantics, the record) | gd (plain full-batch GD with
    # coupled l2 wd/2||theta||^2) | lion (sign-momentum, decoupled wd).
    # n_pre_lin extra (d, d) linear maps applied to the token embedding
    # before the positional add; n_post_lin extra (d, d) maps applied to the
    # final residual before unembedding. Each raises the path degree by 1.
    opt: str = "adamw"
    momentum: float = 0.9       # gdm (heavy-ball GD) only
    clip: float = 0.0           # >0: per-run global gradient-norm clipping (direction-preserving)
    init_scale: float = 1.0     # Lyu et al. 2024 regime: every weight matrix x init_scale (biases stay 0)
    n_pre_lin: int = 0
    n_post_lin: int = 0

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


def group_mod(fn_name):
    """Group task module for fn_name: 's5' -> groups (S5, p=120); 'd<n>' ->
    dihedral D_n (p = 2n). None for 'add'/'sub'. Exposes MULT, PROJ (isotypic
    projectors, trivial irrep first), IRREP_NAMES, IRREP_DIMS."""
    if fn_name == "s5":
        import groups
        return groups
    if fn_name.startswith("d") and fn_name[1:].isdigit():
        import types
        from dihedral import build
        return types.SimpleNamespace(**build(int(fn_name[1:])))
    return None


def make_dataset(cfg):
    """All p^2 pairs [a, b, =] lexicographic; labels (a+b)%p for 'add',
    (a-b)%p for 'sub', group product for 's5' (p must be 120; token
    structure unchanged)."""
    p = cfg.p
    a = np.repeat(np.arange(p), p)
    b = np.tile(np.arange(p), p)
    tokens = np.stack([a, b, np.full(p * p, p)], axis=1).astype(np.int64)
    gm = group_mod(cfg.fn_name)
    if gm is not None:
        assert p == gm.MULT.shape[0], (cfg.fn_name, p, gm.MULT.shape)
        labels = gm.MULT[a, b].astype(np.int64)
    elif cfg.fn_name == "sub":
        labels = ((a - b) % p).astype(np.int64)
    else:
        assert cfg.fn_name == "add"
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

TTAP_STATS = ["g_q10", "g_q50", "g_q90", "sqrtv_q50", "frac_g_lt_eps",
              "frac_sqrtv_lt_eps", "l1", "l2"]
def param_keys(cfg):
    return (PARAM_KEYS[:1] + [f"embed.L{i}" for i in range(cfg.n_pre_lin)]
            + PARAM_KEYS[1:-1] + [f"unembed.L{i}" for i in range(cfg.n_post_lin)]
            + PARAM_KEYS[-1:])


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
    out = dict(zip(PARAM_KEYS, [
        W_E, n(cfg.n_ctx, d), W_K, W_Q, W_V, W_O,
        n(m, d), torch.zeros(m), n(d, m), torch.zeros(d),
        torch.randn(d, v, generator=g) / math.sqrt(v)]))
    for i in range(cfg.n_pre_lin):
        out[f"embed.L{i}"] = n(d, d)
    for i in range(cfg.n_post_lin):
        out[f"unembed.L{i}"] = n(d, d)
    if cfg.init_scale != 1.0:
        out = {k: (v * cfg.init_scale if v.dim() > 1 else v) for k, v in out.items()}
    return out


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
    for i in range(cfg.n_pre_lin):                            # degree dial
        x = x @ pr[f"embed.L{i}"].transpose(1, 2)[:, None]
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
    act = (F.relu(pre) if cfg.act_type == "ReLU"
           else pre * pre if cfg.act_type == "Quad" else F.gelu(pre))
    xf = xf + act @ pr["blocks.0.mlp.W_out"].transpose(1, 2) + pr["blocks.0.mlp.b_out"][:, None]
    for i in range(cfg.n_post_lin):                           # degree dial
        xf = xf @ pr[f"unembed.L{i}"].transpose(1, 2)
    return xf @ pr["unembed.W_U"]


def per_ex_stable_f32(logits, labels):
    """(M, N) per-example CE: softplus(lse_{j!=y} - z_y). Full f32 relative
    precision down to ~1e-38 (naive f32 log_softmax dies at ~1e-7)."""
    z_y = logits.gather(-1, labels.unsqueeze(-1))
    others = logits.masked_fill(
        F.one_hot(labels, logits.shape[-1]).bool(), float("-inf"))
    u = torch.logsumexp(others, dim=-1, keepdim=True) - z_y
    return F.softplus(u).squeeze(-1)


def per_ex_f64(logits, labels):
    z = logits.double()
    lp = z - torch.logsumexp(z, dim=-1, keepdim=True)
    return -lp.gather(-1, labels.unsqueeze(-1)).squeeze(-1)


def aggregate(per_ex, tilt=0.0, cvar=0.0):
    """(M, N) per-example CE -> (M,) per-run objective.

    Branch-for-branch the same as grok.train.cross_entropy_f64:
      cvar > 0  mean over the worst round(cvar*N) examples (MLX sorts
                ascending and takes the last k; topk(k) is the same set)
      tilt > 0  (1/t)(logsumexp(t*ce_i) - log N) = (1/t) log mean exp(t*ce_i)
      else      plain mean
    """
    if cvar > 0.0:
        n = per_ex.shape[1]
        k = max(1, int(round(cvar * n)))
        return per_ex.topk(k, dim=1).values.mean(dim=1)
    if tilt > 0.0:
        return (torch.logsumexp(per_ex * tilt, dim=1)
                - math.log(per_ex.shape[1])) / tilt
    return per_ex.mean(dim=1)


def make_ce(loss, cfgs):
    """Per-run objective, allowing DIFFERENT tilt/CVaR within one batch.

    The batch loss is a sum of independent per-run terms, so runs are
    partitioned by their (tilt, cvar) recipe, each group is aggregated on its
    own slice, and the results are scattered back into run order. With a
    single recipe this is exactly aggregate() on the whole batch.

    NOTE: as in grok.train, the SAME objective is used for the reported test
    loss — so metrics.json test_losses of a tilt/CVaR family are tilt/CVaR
    values, NOT comparable to a plain family's. test_acc in spectra.npz is
    argmax-based and unaffected.
    """
    base = per_ex_stable_f32 if loss == "f32stable" else per_ex_f64
    groups = {}
    for i, c in enumerate(cfgs):
        groups.setdefault((c.loss_tilt, c.loss_cvar), []).append(i)

    if len(groups) == 1:
        (tilt, cvar), _ = next(iter(groups.items()))
        return lambda lg, lb: aggregate(base(lg, lb), tilt, cvar).float()

    def ce(lg, lb):
        per_ex = base(lg, lb)
        idx, vals = [], []
        for (tilt, cvar), ii in groups.items():
            t = torch.as_tensor(ii, device=per_ex.device)
            idx.append(t)
            vals.append(aggregate(per_ex.index_select(0, t), tilt, cvar))
        order = torch.argsort(torch.cat(idx))
        return torch.cat(vals)[order].float()
    return ce


# --------------------------------------------------------------------------
# training
# --------------------------------------------------------------------------

def save_ckpts(pr, run_dirs, epoch, optim=None):
    """optim=(m_st, v_st, step_off) additionally stores the Adam moments and
    absolute step count per run (keys adam.m.<param>, adam.v.<param>,
    adam.step), enabling WARM resumes: init_from auto-detects these keys and
    restores the exact optimizer state instead of a cold restart."""
    for i, rd in enumerate(run_dirs):
        d = {k: v[i].detach().cpu().contiguous() for k, v in pr.items()}
        if optim is not None:
            m_st, v_st, step_off = optim
            for j, k in enumerate(pr):
                d[f"adam.m.{k}"] = m_st[j][i].detach().cpu().contiguous()
                d[f"adam.v.{k}"] = v_st[j][i].detach().cpu().contiguous()
            d["adam.step"] = torch.tensor(float(epoch + step_off[i].item()))
        save_file(d, str(rd / "checkpoints" / f"epoch_{epoch:05d}.safetensors"))


def train_batched(cfgs, run_dirs, init_from=None, log_every=100,
                  spectra_every=100, loss="f32stable", compile_mode="default",
                  device=None, wb=None, wb_tag=None, freeze=None,
                  masks_override=None, save_optim=False, grad_tap=False,
                  weight_tap=False, tensor_tap=False, vocab_proj=None):
    M = len(cfgs)
    c0 = cfgs[0]
    # lr, weight_decay, loss_tilt and loss_cvar are PER-RUN (broadcast vectors
    # in the optimizer / grouped in the aggregator), so dynamics families can
    # share one lockstep batch. Everything below still has to agree: it either
    # changes tensor shapes or the epoch grid.
    # frac_train may differ within a batch (ragged path pads per-run example
    # sets); everything shape- or schedule-bearing below still must agree.
    for c in cfgs:
        for f in ("p", "fn_name", "d_model", "num_heads", "d_mlp",
                  "n_ctx", "act_type", "warmup_steps",
                  "betas", "adam_eps", "num_epochs", "save_every",
                  "stopping_thresh", "opt", "momentum", "clip", "n_pre_lin", "n_post_lin"):
            assert getattr(c, f) == getattr(c0, f), f"batch must agree on {f}"
        assert c.num_layers == 1
        # tilt/CVaR are supported (they only reshape the per-example
        # aggregation); SAM-lite grad_noise is NOT ported — it needs a second
        # perturbed forward per step.
        assert c.grad_noise == 0, "grad_noise (SAM-lite) is not implemented here"
        assert not (c.loss_tilt > 0 and c.loss_cvar > 0), "pick one of tilt/cvar"
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    run_dirs = [Path(rd) for rd in run_dirs]

    def _rel(rd):   # run path relative to RUNS: the wandb scalar key prefix
        try:
            return str(rd.relative_to(RUNS))
        except ValueError:
            return rd.name

    rel_names = [_rel(rd) for rd in run_dirs]
    # Label the batch aggregates by what the batch IS, not by whichever run
    # happened to sort first (that read as "seed61001/train_mean" for a mean
    # over all 24 runs).
    tag = wb_tag or f"batch/{rel_names[0]}"
    for cfg, rd in zip(cfgs, run_dirs):
        (rd / "checkpoints").mkdir(parents=True, exist_ok=True)
        cfg.save(rd / "config.json")

    # stack per-run inits (fresh torch draws, or checkpoints). A checkpoint
    # carrying adam.* keys (written via save_optim=True) is resumed WARM:
    # its Adam moments and step count are restored exactly. Otherwise the
    # resume is cold (m=v=0, warmup restarts) — the historical behavior.
    per_run = []
    for i, cfg in enumerate(cfgs):
        p = init_params(cfg)
        if init_from is not None and init_from[i] is not None:
            p = load_file(str(init_from[i]))
        per_run.append(p)
    KEYS = param_keys(c0)
    pr = {k: torch.stack([r[k] for r in per_run]).to(device).requires_grad_()
          for k in KEYS}
    # optional per-pair shortcut table (see Config.table_scale): one extra
    # stacked param, zero-init, per-run gain ts_v and wd multiplier twm_v
    table = any(c.table_scale > 0 for c in cfgs)
    keys = list(KEYS)
    if table:
        pr["table.T"] = torch.zeros(M, c0.p * c0.p, c0.d_vocab, device=device,
                                    requires_grad=True)
        keys.append("table.T")
        ts_v = torch.tensor([c.table_scale for c in cfgs], dtype=torch.float32,
                            device=device)[:, None, None]
        twm_v = torch.tensor([c.table_wd_mult for c in cfgs],
                             dtype=torch.float32, device=device)
    params = [pr[k] for k in keys]

    # ---- fixed-support constraint (projected AdamW on the vocab axis) -----
    # vocab_proj = {"E": bool, "U": bool, "P": (p, p) or (M, p, p)} with P a
    # symmetric idempotent projector on the GROUP-token axis (vocab columns
    # 0..p-1; the '=' column p is never touched and stays free). W_E and/or
    # W_U have shape (M, d, v); the length-p slice of each d-row is the
    # token-indexed function whose isotypic content P restricts. The init is
    # projected once and the projection is re-applied after every AdamW
    # update, so out-of-subspace coordinates are zeroed every step (to f32
    # roundoff of the projected weight, ~5e-8 relative — it does not
    # accumulate, and the observable it controls, the logits' off-support
    # isotypic energy, comes out at ~1e-14 of the total). Gradients and Adam
    # moments are NOT projected: only the weights are constrained. With
    # P = I this is a no-op matmul, bit-identical to vocab_proj=None.
    if vocab_proj is not None:
        vp_keys = [k for k, on in (("embed.W_E", vocab_proj.get("E")),
                                   ("unembed.W_U", vocab_proj.get("U"))) if on]
        P_vp = torch.as_tensor(np.asarray(vocab_proj["P"]),
                               dtype=torch.float32, device=device)
        assert P_vp.shape[-2:] == (c0.p, c0.p), P_vp.shape
        assert P_vp.dim() in (2, 3) and (P_vp.dim() == 2 or P_vp.shape[0] == M)
        pv = c0.p

        def _project_vocab():
            for k in vp_keys:
                w = pr[k]
                w[:, :, :pv].copy_(w[:, :, :pv] @ P_vp)
        if vp_keys:
            with torch.no_grad():
                _project_vocab()
            print(f"vocab_proj on {vp_keys}: P {tuple(P_vp.shape)} "
                  f"rank {int(round(float(P_vp.reshape(-1, pv, pv)[0].diagonal().sum())))}",
                  flush=True)
    else:
        vp_keys = []

    def logits_fn(toks):
        L = batched_final_logits(pr, toks, c0)
        if table:
            T = pr["table.T"]
            if toks.dim() == 2:
                L = L + ts_v * T[:, toks[:, 0] * c0.p + toks[:, 1]]
            else:
                idx = (toks[..., 0] * c0.p + toks[..., 1]).unsqueeze(-1)
                L = L + ts_v * T.gather(1, idx.expand(-1, -1, T.shape[-1]))
        return L
    m_st = [torch.zeros_like(p) for p in params]
    v_st = [torch.zeros_like(p) for p in params]
    step_off = torch.zeros(M, device=device)
    warm_any = False
    for i, p in enumerate(per_run):
        if "adam.step" not in p:
            continue
        missing = [k for k in KEYS
                   if f"adam.m.{k}" not in p or f"adam.v.{k}" not in p]
        assert not missing, f"partial optimizer state in ckpt: {missing}"
        for j, k in enumerate(KEYS):
            m_st[j][i] = p[f"adam.m.{k}"].to(device)
            v_st[j][i] = p[f"adam.v.{k}"].to(device)
        step_off[i] = float(p["adam.step"].item())
        warm_any = True
    if warm_any:
        print(f"warm resume: {int((step_off > 0).sum())} run(s) with restored "
              f"Adam state, step offsets {step_off.cpu().numpy().tolist()}",
              flush=True)
    b1, b2 = c0.betas
    eps = c0.adam_eps
    OPT = c0.opt; MU = c0.momentum; CLIP = c0.clip
    assert OPT in ("adamw", "gd", "gdm", "ngd", "lion"), OPT
    assert OPT == "adamw" or freeze is None, "freeze gating is AdamW-only"
    # Per-run lr / weight decay as (M,) vectors, broadcast against each
    # stacked param's trailing dims. With one distinct value this is
    # arithmetically identical to the old scalar form.
    lr_base = torch.tensor([c.lr for c in cfgs], dtype=torch.float32, device=device)
    wd_v = torch.tensor([c.weight_decay for c in cfgs], dtype=torch.float32,
                        device=device)
    wd_k = {k: wd_v for k in keys}
    if table:
        wd_k["table.T"] = wd_v * twm_v
    lr_t = torch.zeros(M, device=device)    # in-place updated: no recompiles
    bshape = {id(p): (-1,) + (1,) * (p.dim() - 1) for p in params}

    # freeze: optional per-run gating of which params update early. Entry i is
    # None or (mode, until_epoch): "freeze_we" gates embed.W_E, "only_we"
    # gates everything BUT embed.W_E, for epochs < until_epoch. A gated param
    # is left completely untouched — no Adam step, no moment accumulation,
    # and no weight-decay multiply (a frozen tilt must not decay). Masks are
    # (M,) tensors updated in place only at schedule boundaries: no recompiles.
    if freeze is not None and any(f is not None for f in freeze):
        assert len(freeze) == M
        act = {k: torch.ones(M, device=device) for k in keys}
        f_bounds = sorted({u for f in freeze if f for u in (f[1],)})

        def set_active(epoch):
            for k in KEYS:
                col = [1.0 if (f is None or epoch >= f[1]
                               or (f[0] == "freeze_we") != (k == "embed.W_E"))
                       else 0.0 for f in freeze]
                act[k].copy_(torch.tensor(col, device=device))
        set_active(0)
    else:
        act, f_bounds = None, []

    tokens_np, labels_np = make_dataset(c0)
    # masks_override: per-run explicit train masks (np bool, p^2), None entries
    # fall back to the seeded split. Runs whose masks differ in COUNT (e.g.
    # different frac_train stacked into one lockstep batch) are handled by the
    # ragged path: per-run example indices padded (mode=edge) to the batch max
    # and a per-run weight vector (1/count on real slots, 0 on pads), so the
    # padded slots contribute exactly zero loss and zero gradient. With equal
    # counts the historical compact path runs unchanged (bit-identical).
    masks = [train_test_split(c) if masks_override is None
             or masks_override[i] is None else np.asarray(masks_override[i])
             for i, c in enumerate(cfgs)]
    ragged = len({int(m.sum()) for m in masks}) > 1
    if ragged:
        assert all(c.loss_tilt == 0 and c.loss_cvar == 0 for c in cfgs), \
            "ragged batches support the plain-mean objective only"

        def pack(sels):
            n = max(len(s) for s in sels)
            idx = np.stack([np.pad(s, (0, n - len(s)), mode="edge")
                            for s in sels])
            w = np.stack([np.concatenate([np.full(len(s), 1.0 / len(s),
                                                  dtype=np.float32),
                                          np.zeros(n - len(s), np.float32)])
                          for s in sels])
            return idx, w

        tr_idx, tr_w = pack([np.flatnonzero(m) for m in masks])
        te_idx, te_w = pack([np.flatnonzero(~m) for m in masks])
        tr_tok = torch.from_numpy(tokens_np[tr_idx]).to(device)
        tr_lab = torch.from_numpy(labels_np[tr_idx]).to(device)
        te_tok = torch.from_numpy(tokens_np[te_idx]).to(device)
        te_lab = torch.from_numpy(labels_np[te_idx]).to(device)
        w_tr = torch.from_numpy(tr_w).to(device)
        w_te = torch.from_numpy(te_w).to(device)
        base = per_ex_stable_f32 if loss == "f32stable" else per_ex_f64
        ce_tr = lambda lg, lb: (base(lg, lb) * w_tr).sum(1).float()
        ce_te = lambda lg, lb: (base(lg, lb) * w_te).sum(1).float()
    else:
        tr_tok = torch.from_numpy(np.stack([tokens_np[m] for m in masks])).to(device)
        tr_lab = torch.from_numpy(np.stack([labels_np[m] for m in masks])).to(device)
        te_tok = torch.from_numpy(np.stack([tokens_np[~m] for m in masks])).to(device)
        te_lab = torch.from_numpy(np.stack([labels_np[~m] for m in masks])).to(device)
        ce_tr = ce_te = make_ce(loss, cfgs)
    grid = torch.from_numpy(tokens_np).to(device)
    recipes = sorted({(c.lr, c.weight_decay, c.loss_tilt, c.loss_cvar)
                      for c in cfgs})
    counts = sorted({int(m.sum()) for m in masks})
    print(f"batch {M} on {device}:  train {counts[0]}"
          + (f"-{counts[-1]} padded to {tr_tok.shape[1]} (ragged)"
             if ragged else "")
          + f"  test {te_tok.shape[1]}  {len(recipes)} recipe(s) "
          f"(lr, wd, tilt, cvar): {recipes}", flush=True)

    # spectra machinery (GPU f64; tiny matmuls). For 's5' the per-frequency
    # Fourier decomposition is replaced by per-irrep isotypic energy over
    # the 6 NONTRIVIAL irreps (trivial = constant offset, excluded exactly
    # like the constant Fourier row); coeffs = sqrt(energy) so the
    # gap-detector committee machinery works unchanged.
    lab_t = torch.from_numpy(labels_np).to(device)
    mask_t = torch.from_numpy(np.stack(masks)).to(device)
    # snapshots accumulate ON DEVICE (lists of (M, ...) tensors) and transfer
    # once at save time — the old per-run .cpu()/float() pattern cost M*4 host
    # syncs per snapshot, which dominates at spectra_every=5.
    snap = {"epochs": [], "coeffs": [], "energy": [],
            "train_acc": [], "test_acc": []}
    if table:
        # circuit-only accs/spectra above; the FULL (circuit + table) accs and
        # the two pathways' squared norms alongside
        for k in ("train_acc_full", "test_acc_full", "table_norm2",
                  "circuit_norm2"):
            snap[k] = []
    if grad_tap:
        snap["grad_energy"], snap["grad_dot"] = [], []
    if tensor_tap:
        # eps-crossing chronology (nudial/eps_chronology.py, E5): per weight
        # tensor, per snapshot: quantiles of the full-batch |grad| Adam sees,
        # of sqrt(v) (its actual denominator), fractions below adam_eps, and
        # the tensor's l1/l2 norms. (S, M, n_keys, len(TTAP_STATS)).
        snap["ttap"] = []
    if weight_tap:
        # committee-agnostic decomposition, so any committee can be applied
        # post hoc: full 2D logit spectrum over (a, b) per output, grouped
        # by input-frequency pair; Fourier energy of W_E / W_U over the p
        # token axis; per-matrix squared norms. All f32, main stream.
        assert group_mod(c0.fn_name) is None, "weight tap is Fourier-only"
        for k in ("diag2d", "off2d", "single2d", "const2d", "total2d",
                  "wE_energy", "wU_energy", "pnorm"):
            snap[k] = []
        F_t = torch.from_numpy(fourier_basis(c0.p)).float().to(device)
        kidx = torch.tensor([(i + 1) // 2 for i in range(c0.p)],
                            device=device)                    # row -> freq, 0=const
        same_k = (kidx[:, None] == kidx[None, :]) & (kidx[:, None] > 0)
    GM = group_mod(c0.fn_name)
    if GM is not None:
        proj_t = torch.from_numpy(GM.PROJ[1:]).to(device)     # (n_irreps-1, p, p), trivial dropped
    else:
        basis = fourier_basis(c0.p)
        n_ = c0.p // 2
        bc, bs = basis[1::2], basis[2::2]                    # (n, p)
        # 'add' solutions live on cos/sin(w(a+b)); 'sub' on cos/sin(w(a-b)).
        # sgn flips the cross terms so the projectors match the task.
        sgn = -1.0 if c0.fn_name == "sub" else 1.0
        u_cos = torch.from_numpy((np.einsum("np,nq->npq", bc, bc)
                                  - sgn * np.einsum("np,nq->npq", bs, bs))
                                 .reshape(n_, -1) / np.sqrt(2)).to(device)
        u_sin = torch.from_numpy((np.einsum("np,nq->npq", bs, bc)
                                  + sgn * np.einsum("np,nq->npq", bc, bs))
                                 .reshape(n_, -1) / np.sqrt(2)).to(device)
        b_cos = torch.from_numpy(bc).to(device)
        b_sin = torch.from_numpy(bs).to(device)

    # The f64 projection runs at 1/64 rate on GeForce and would stall the
    # fp32 training stream ~10ms per snapshot. On CUDA it is dispatched to a
    # SIDE STREAM against a cloned logit buffer: the f64 units grind away
    # while the next training steps proceed on the fp32 pipeline. The math,
    # dtypes and stored values are IDENTICAL to the synchronous form — only
    # the scheduling changes. Non-CUDA devices take the synchronous path.
    side = (torch.cuda.Stream() if str(device).startswith("cuda") else None)

    def project(Ld):
        if GM is not None:
            energy = torch.stack([((Ld @ proj_t[r].T) ** 2).sum((-1, -2))
                                  for r in range(proj_t.shape[0])], -1)
            return energy.sqrt(), energy
        c = torch.einsum("fp,mpo->mfo", u_cos, Ld)           # (M, n, p)
        s = torch.einsum("fp,mpo->mfo", u_sin, Ld)
        energy = (c ** 2).sum(-1) + (s ** 2).sum(-1)
        coeffs = ((c * b_cos).sum(-1) + (s * b_sin).sum(-1)) / math.sqrt(2)
        return coeffs, energy

    @torch.no_grad()
    def weight_snapshot(L):
        p, nf = c0.p, c0.p // 2 + 1
        Lc = (L - L.mean(-1, keepdim=True)).reshape(M, p, p, p)   # softmax gauge
        H = torch.einsum("ia,jb,mabo->mijo", F_t, F_t, Lc)
        E2 = (H ** 2).sum(-1)                                      # (M, p, p)
        diag = torch.zeros(M, nf, device=device).index_add_(
            1, kidx, (E2 * same_k).sum(2))                         # (M, nf): k=k' blocks
        const = E2[:, 0, 0]
        single = E2[:, 0, 1:].sum(1) + E2[:, 1:, 0].sum(1)
        total = E2.sum((1, 2))
        snap["diag2d"].append(diag[:, 1:])                         # freqs 1..p//2
        snap["const2d"].append(const)
        snap["single2d"].append(single)
        snap["off2d"].append(total - const - single - diag[:, 1:].sum(1))
        snap["total2d"].append(total)
        for key, name in (("embed.W_E", "wE_energy"), ("unembed.W_U", "wU_energy")):
            W = pr[key].detach()[:, :, :p]                         # (M, d, p)
            C = W @ F_t.T                                          # (M, d, p)
            e = torch.zeros(M, nf, device=device).index_add_(
                1, kidx, (C ** 2).sum(1))
            snap[name].append(e)                                   # index 0 = const
        snap["pnorm"].append(torch.stack(
            [(pr[k].detach() ** 2).flatten(1).sum(1) for k in KEYS], 1))

    @torch.no_grad()
    def take_snapshot(epoch):
        L = batched_final_logits(pr, grid, c0)[..., :c0.p]
        if weight_tap:
            weight_snapshot(L)
        acc = L.argmax(-1) == lab_t                          # (M, p^2)
        if table:
            accf = logits_fn(grid)[..., :c0.p].argmax(-1) == lab_t
            snap["train_acc_full"].append((accf & mask_t).sum(-1).double()
                                          / mask_t.sum(-1))
            snap["test_acc_full"].append((accf & ~mask_t).sum(-1).double()
                                         / (~mask_t).sum(-1))
            snap["table_norm2"].append((pr["table.T"].detach() ** 2).sum((1, 2)))
            snap["circuit_norm2"].append(sum(
                (pr[k].detach() ** 2).flatten(1).sum(1) for k in KEYS))
        snap["epochs"].append(epoch)
        snap["train_acc"].append((acc & mask_t).sum(-1).double()
                                 / mask_t.sum(-1))
        snap["test_acc"].append((acc & ~mask_t).sum(-1).double()
                                / (~mask_t).sum(-1))
        if side is None:
            coeffs, energy = project(L.double())
        else:
            Lc = L.detach().clone()
            ready = torch.cuda.Event()
            ready.record()
            with torch.cuda.stream(side):
                side.wait_event(ready)
                Lc.record_stream(side)
                coeffs, energy = project(Lc.double())
        snap["coeffs"].append(coeffs)
        snap["energy"].append(energy)

    def grad_snapshot():
        """Per-frequency (or per-irrep) energy of grad(W_E) and its dot with
        W_E's own components, on the training objective. One extra eager
        fwd/bwd per snapshot epoch — the mechanistic 'gradient tap'. Under
        plain GD dE_k/dt = -2*lr*dot_k: dot_k > 0 means the loss is decaying
        frequency k, dot_k < 0 means it is growing it."""
        per = ce_tr(logits_fn(tr_tok), tr_lab)
        (gW,) = torch.autograd.grad(per.sum(), pr["embed.W_E"])
        gF = gW[:, :, :c0.p].double()
        wF = pr["embed.W_E"].detach()[:, :, :c0.p].double()
        if GM is not None:
            gE = torch.stack([((gF @ proj_t[r].T) ** 2).sum((1, 2))
                              for r in range(proj_t.shape[0])], -1)
            gd = torch.stack([((gF @ proj_t[r].T) * (wF @ proj_t[r].T))
                              .sum((1, 2)) for r in range(proj_t.shape[0])], -1)
        else:
            gc, gs = gF @ b_cos.T, gF @ b_sin.T              # (M, d, n)
            wc, ws = wF @ b_cos.T, wF @ b_sin.T
            gE = (gc ** 2 + gs ** 2).sum(1)
            gd = (gc * wc + gs * ws).sum(1)
        snap["grad_energy"].append(gE)
        snap["grad_dot"].append(gd)

    def _q(a, qs):
        """Quantiles of a (M, n) tensor along n via sort (device-agnostic)."""
        s = a.sort(dim=1).values
        idx = [int(round(q * (a.shape[1] - 1))) for q in qs]
        return s[:, idx]                                   # (M, len(qs))

    def tensor_snapshot():
        """One extra eager fwd/bwd; per-tensor stats of what Adam sees."""
        per = ce_tr(logits_fn(tr_tok), tr_lab)
        grads = torch.autograd.grad(per.sum(), [pr[k] for k in KEYS])
        rows = []
        with torch.no_grad():
            for j, (k, g) in enumerate(zip(KEYS, grads)):
                a = g.abs().flatten(1)                     # (M, n)
                sv = v_st[j].sqrt().flatten(1)
                w = pr[k].detach().flatten(1)
                rows.append(torch.cat([
                    _q(a, (0.1, 0.5, 0.9)),
                    _q(sv, (0.5,)),
                    (a < eps).float().mean(1, keepdim=True),
                    (sv < eps).float().mean(1, keepdim=True),
                    w.abs().sum(1, keepdim=True),
                    w.norm(dim=1, keepdim=True)], 1))      # (M, 8)
        snap["ttap"].append(torch.stack(rows, 1))          # (M, n_keys, 8)

    def step():
        per = ce_tr(logits_fn(tr_tok), tr_lab)
        grads = torch.autograd.grad(per.sum(), params)
        with torch.no_grad():
            if OPT == "ngd" or CLIP > 0:
                gn = torch.sqrt(sum((g ** 2).flatten(1).sum(1) for g in grads)) + 1e-12
            if CLIP > 0:
                sc = torch.clamp(CLIP / gn, max=1.0)
                grads = [g * sc.view(bshape[id(p)]) for p, g in zip(params, grads)]
            # MLX-semantics AdamW: NO bias correction, p <- p(1-lr*wd) - lr*m/(sqrt(v)+eps)
            for k, p, g, m_, v_ in zip(keys, params, grads, m_st, v_st):
                sh = bshape[id(p)]
                lr_, wd_ = lr_t.view(sh), wd_k[k].view(sh)
                if OPT == "gd":                       # plain GD, coupled l2
                    p.sub_(lr_ * (g + wd_ * p))
                    continue
                if OPT == "gdm":                      # heavy-ball GD, coupled l2
                    m_.mul_(MU).add_(g + wd_ * p)
                    p.sub_(lr_ * m_)
                    continue
                if OPT == "ngd":                      # normalized GD, coupled l2
                    p.sub_(lr_ * (g / gn.view(sh) + wd_ * p))
                    continue
                if OPT == "lion":                     # sign momentum, decoupled wd
                    u = torch.sign(m_ * b1 + g * (1 - b1))
                    m_.mul_(b2).add_(g, alpha=1 - b2)
                    p.sub_(lr_ * (u + wd_ * p))
                    continue
                if act is None:
                    m_.mul_(b1).add_(g, alpha=1 - b1)
                    v_.mul_(b2).addcmul_(g, g, value=1 - b2)
                    p.mul_(1 - lr_ * wd_)
                    p.sub_(lr_ * m_ / (v_.sqrt() + eps))
                else:
                    # same arithmetic (kernel-for-kernel, so active runs are
                    # bit-identical to the ungated path), gated per run:
                    # inactive runs keep p, m, v exactly — no step, no
                    # moment accumulation, no weight decay
                    ab = act[k].view(sh) > 0
                    m_new = torch.add(m_ * b1, g, alpha=1 - b1)
                    v_new = torch.addcmul(v_ * b2, g, g, value=1 - b2)
                    p_new = torch.sub(p * (1 - lr_ * wd_),
                                      lr_ * m_new / (v_new.sqrt() + eps))
                    m_.copy_(torch.where(ab, m_new, m_))
                    v_.copy_(torch.where(ab, v_new, v_))
                    p.copy_(torch.where(ab, p_new, p))
            if vp_keys:
                _project_vocab()          # fixed support: re-zero off-support
        return per.detach()

    if compile_mode != "off":
        step = torch.compile(step, mode=None if compile_mode == "default"
                             else compile_mode)

    # Preallocated device history: under reduce-overhead (CUDA graphs) the
    # tensor step() returns is graph-owned and overwritten on the next replay,
    # so it must be copied out before the next call — appending the returned
    # tensor to a list would silently alias every epoch to the same buffer.
    train_hist = torch.empty(c0.num_epochs, M, device=device)
    test_hist = []
    t0 = time.time()
    for epoch in range(c0.num_epochs):
        if epoch % c0.save_every == 0:
            save_ckpts(pr, run_dirs, epoch,
                       optim=(m_st, v_st, step_off) if save_optim else None)
        if spectra_every and epoch % spectra_every == 0:
            take_snapshot(epoch)
            if grad_tap:
                grad_snapshot()
            if tensor_tap:
                tensor_snapshot()
        if warm_any:
            # per-run step offsets: warm-resumed runs skip the lr warmup they
            # already served. Cold runs (offset 0) follow the same curve; the
            # scalar path below is kept for pure-cold batches so historical
            # cohorts stay bit-identical.
            lr_t.copy_(lr_base * ((epoch + step_off)
                                  / c0.warmup_steps).clamp(max=1.0))
        else:
            lr_t.copy_(lr_base * min(epoch / c0.warmup_steps, 1.0))
        if act is not None and epoch in f_bounds:
            set_active(epoch)
        train_hist[epoch].copy_(step())

        if (epoch + 1) % log_every == 0:
            with torch.no_grad():
                te = ce_te(logits_fn(te_tok), te_lab).cpu().numpy()
            test_hist.append(te)
            tr = train_hist[epoch].cpu().numpy()
            eps_s = (epoch + 1) / (time.time() - t0)
            # median as well as mean: the test distribution is heavy-tailed
            # (one late-grokking straggler can be 100x the median and then
            # owns most of the mean) — see the 16-run MLX cohort.
            print(f"epoch {epoch + 1:6d}  train {tr.mean():.4e}  "
                  f"test med {np.median(te):.4e} mean {te.mean():.4e} "
                  f"max {te.max():.4e}  "
                  f"({eps_s:.1f} eps/s x {M} = {eps_s * M:.0f} run-eps/s)",
                  flush=True)
            if wb:
                d = {"epoch": epoch + 1, "run_eps_per_s": eps_s * M,
                     f"{tag}/train_mean": float(tr.mean()),
                     f"{tag}/test_mean": float(te.mean()),
                     f"{tag}/test_median": float(np.median(te)),
                     f"{tag}/test_max": float(te.max())}
                for i, n in enumerate(rel_names):   # per-run curves, named
                    d[f"{n}/train"] = float(tr[i])
                    d[f"{n}/test"] = float(te[i])
                wb.log(d)
            if te.max() < c0.stopping_thresh:
                break

    epoch = epoch + 1 if c0.num_epochs else 0        # epochs actually trained
    save_ckpts(pr, run_dirs, epoch,
               optim=(m_st, v_st, step_off) if save_optim else None)
    if spectra_every:
        take_snapshot(epoch)
        if grad_tap:
            grad_snapshot()
        if tensor_tap:
            tensor_snapshot()
    if side is not None:
        torch.cuda.synchronize()             # drain the side-stream spectra
    train_all = train_hist[:epoch].cpu().numpy()             # (E, M)
    test_all = np.stack(test_hist) if test_hist else np.zeros((0, M))
    # assemble per-run spectra from the device-side accumulators (S, M, ...)
    sp_ep = np.array(snap["epochs"])
    sp = {k: torch.stack(snap[k]).cpu().numpy() if snap[k] else
          np.zeros((0, M)) for k in snap if k != "epochs"}
    extra = ({"ttap_keys": np.array(KEYS), "ttap_stats": np.array(TTAP_STATS)}
             if tensor_tap else {})
    for i, rd in enumerate(run_dirs):
        np.savez(rd / "spectra.npz", epochs=sp_ep,
                 **{k: v[:, i] for k, v in sp.items()}, **extra)
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
    """Print the run summary and, if wandb is on, publish the run as its OWN
    wandb run named for its path (e.g. orthWE/p-113/seed4811/seed61001).

    One wandb run per training run is the whole point: the dashboard list is
    then a list of real run names you can read and grep, instead of 44 runs
    hidden inside one 'nosy-aardvark'. Each carries its own Config, its own
    train/test/test_acc curves, its committee in the summary, and its
    artifact — so filtering by cohort or sorting by grok epoch works in the
    UI with no decoding.
    """
    z = np.load(run_dir / "spectra.npz")
    final = committee_from_coeffs(z["coeffs"][-1])
    name = str(run_dir.relative_to(RUNS))
    ge = grok_epoch(z)
    print(f"RESULT {name}: grok@{ge}  committee {final}  "
          f"acc {z['test_acc'][-1]:.4f}", flush=True)
    if not wb:
        return
    cfg = Config.load(run_dir / "config.json")
    conf = asdict(cfg)
    conf["betas"] = list(conf["betas"])
    fam = name.split("/")[0]
    cohort = ("double-flat" if cfg.embed_init == "orthogonal"
              and cfg.attn_init == "isometric" else
              "orth-flat" if cfg.embed_init == "orthogonal" else "normal")
    r = wandb.init(
        project=os.environ.get("WANDB_PROJECT", "grok-semifinal-v2"),
        name=name, group=fam, job_type=cohort, reinit="create_new",
        tags=[fam, cohort, f"p{cfg.p}", f"{cfg.num_epochs // 1000}k"],
        config={**conf, "run": name, "family": fam, "cohort": cohort})
    try:
        m = json.loads((run_dir / "metrics.json").read_text())
        tr, te, every = m["train_losses"], m["test_losses"], m["test_every"]
        acc = {int(e): (float(a), float(b)) for e, a, b in
               zip(z["epochs"], z["train_acc"], z["test_acc"])}
        for j, t in enumerate(te):          # train subsampled to test cadence
            ep = (j + 1) * every
            row = {"epoch": ep, "test_ce": t}
            if ep - 1 < len(tr):            # short if stopping_thresh fired
                row["train_ce"] = tr[ep - 1]
            if ep in acc:
                row["train_acc"], row["test_acc"] = acc[ep]
            r.log(row)
        r.summary.update({
            "grok_epoch": ge, "committee": final, "committee_size": len(final),
            "final_train_ce": tr[-1], "final_test_ce": te[-1] if te else None,
            "final_test_acc": float(z["test_acc"][-1])})
        art = wandb.Artifact(name.replace("/", "__"), type="grok-run")
        art.add_dir(str(run_dir))
        # wait() makes the upload synchronous: a run only counts as done
        # once its artifact is committed, so a dead box can't strand a
        # finished-locally-but-never-uploaded run (resume would skip it).
        try:
            r.log_artifact(art).wait()
        except Exception as e:  # offline mode / transient — don't kill training
            print(f"    (artifact upload not confirmed for {name}: {e})", flush=True)
    finally:
        r.finish()


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


def e0_freq_energy(run_dir):
    """Per-frequency energy of a run's epoch-0 W_E."""
    W = load_file(str(run_dir / "checkpoints" / "epoch_00000.safetensors"))
    return freq_energy_F(W["embed.W_E"].double().numpy()[:, :P]
                         @ fourier_basis(P).T, NF)


def transplant_ckpt(recip, donor, tag):
    """Claim 3E: the recipient's epoch-0 W_E with EVERY frequency's energy
    rescaled to the donor's — a full-spectrum energy copy that keeps the
    recipient's direction geometry. If identity travelled with energy alone,
    the recipient would adopt the donor's committee."""
    eR, eD = e0_freq_energy(recip), e0_freq_energy(donor)
    return surgical_ckpt(recip, scales={k: float(eD[k - 1] / eR[k - 1])
                                        for k in range(1, NF + 1)}, tag=tag)


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

def wanted(name, args):
    """--only / --skip run-name filters (regex, re.search). Independent of the
    spectra.npz resume check: --only picks what to CONSIDER, spectra.npz still
    skips whatever is already finished on disk."""
    if args.only and not re.search(args.only, name):
        return False
    if args.skip and re.search(args.skip, name):
        return False
    return True


def batch_train(jobs, width, args, wb, label="batch"):
    todo, filtered = [], 0
    for name, cfg, ckpt in jobs:
        if not wanted(name, args):
            filtered += 1
        elif (RUNS / name / "spectra.npz").exists():
            print(f"skip {name} (exists)", flush=True)
        elif args.dry_run:
            print(f"WOULD TRAIN {name} ({cfg.num_epochs} epochs)", flush=True)
        else:
            todo.append((name, cfg, ckpt))
    if filtered:
        print(f"({filtered} filtered out by --only/--skip)", flush=True)
    for i in range(0, len(todo), width):
        chunk = todo[i:i + width]
        print(f"=== batch x{len(chunk)}: {', '.join(n for n, _, _ in chunk)} ===",
              flush=True)
        train_batched([c for _, c, _ in chunk], [RUNS / n for n, _, _ in chunk],
                      init_from=[ck for _, _, ck in chunk], loss=args.loss,
                      compile_mode=args.compile, wb=wb,
                      wb_tag=f"_batch/{label}_{i // width:02d}_x{len(chunk)}")
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


def steering_jobs(base_rel, args, wb):
    """Build (not train) this base's 10 steering arms, so both suites and the
    transplants — which all share lr/wd/no-tilt and depend only on the
    from-scratch runs — can go into ONE lockstep batch."""
    base = RUNS / base_rel
    bname = base.name
    # Bail before touching the base's spectra if every arm is filtered out —
    # otherwise --only 'dyn-' would still demand a grokked base on a box that
    # has never trained one.
    if not any(wanted(f"{f}/{bname}", args) for f in
               ("dosefarm", "suppress", "gkrotate", "chaospair", "collisionfarm")):
        print(f"steering suite on {base_rel}: all arms filtered out", flush=True)
        return []
    if args.dry_run:
        print(f"WOULD TRAIN steering suite on {base_rel} (10 arms)", flush=True)
        return []
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
    return jobs


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
    ap.add_argument("--only", default=None, metavar="REGEX",
                    help="train only runs whose name matches. The claim-2A/3E "
                         "arms added after the first 44-run dataset are: "
                         "--only 'dyn-|phase2-tilt|eff-G|transplant/'")
    ap.add_argument("--skip", default=None, metavar="REGEX",
                    help="never train runs whose name matches")
    # Two batches is the floor: phase 1 is every independent run (72), phase 2
    # is everything whose epoch-0 ckpt is derived from a finished phase-1 run
    # (26). Lower these if a box OOMs — activations scale linearly in M.
    ap.add_argument("--width-scratch", type=int, default=72)
    ap.add_argument("--width-steer", type=int, default=26)
    ap.add_argument("--tf32", action="store_true",
                    help="~2x matmul speed, NOT numerically faithful")
    args = ap.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = args.tf32
    torch.backends.cudnn.allow_tf32 = args.tf32
    torch.set_float32_matmul_precision("high" if args.tf32 else "highest")

    wb = None
    if wandb and os.environ.get("WANDB_API_KEY") and not args.dry_run:
        # Name the run ourselves. wandb's auto-generated "nosy-aardvark" tells
        # you nothing about which sweep it was; this is greppable and sorts
        # chronologically. WANDB_NAME overrides.
        stamp = time.strftime("%Y%m%d-%H%M%S")
        kind = "smoke" if args.smoke else "semifinal-v2"
        dsA_ = sorted(CELLS)[0]
        ref = asdict(base_cfg(dsA_, CELLS[dsA_][0]))
        ref["betas"] = list(ref["betas"])
        ref.update(
            n_runs=len(STEER_BASES) * (len(DOSES) + len(GAINS) + 4)
                   + 3 * sum(len(v) for v in CELLS.values()),
            epochs=EPOCHS, spectra_every=100, cells={str(k): v for k, v in CELLS.items()},
            steer_bases=STEER_BASES, doses=DOSES, gains=GAINS,
            width_scratch=args.width_scratch, width_steer=args.width_steer,
            loss=args.loss, tf32=args.tf32, compile=args.compile,
            torch_version=torch.__version__,
            gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            runs_dir=str(RUNS),
        )
        wb = wandb.init(
            project=os.environ.get("WANDB_PROJECT", "grok-semifinal-v2"),
            name=os.environ.get("WANDB_NAME")
                 or f"{kind}-p{P}-{EPOCHS // 1000}k-{stamp}",
            tags=["driver", kind, f"p{P}", f"{EPOCHS // 1000}k", args.loss]
                 + (["tf32"] if args.tf32 else []),
            config=ref)

    if args.smoke:
        smoke(args, wb)
    else:
        print(__doc__, flush=True)
        print(f"cohort seeds (SEED_DRAW={SEED_DRAW}): "
              + "   ".join(f"cell {ds} -> {CELLS[ds]}" for ds in sorted(CELLS)),
              flush=True)
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
        # 9. claim-2A dynamics variants ride in the SAME batch as the
        #    from-scratch runs: lr/wd/tilt/cvar are per-run now, so all 72
        #    independent runs are one lockstep batch.
        for fam, kw in DYNAMICS:
            jobs += [(f"{fam}/p-113/seed{ds}/seed{s}",
                      base_cfg(ds, s, embed_init="orthogonal", **kw), None)
                     for ds in sorted(CELLS) for s in CELLS[ds]]
        batch_train(jobs, args.width_scratch, args, wb, label="scratch")

        # Phase 2 — everything that needs a FINISHED from-scratch run to build
        # its epoch-0 checkpoint: both steering suites plus the transplants.
        # No config constraint binds them, only this dependency.
        jobs2 = []
        for b in STEER_BASES:
            jobs2 += steering_jobs(b, args, wb)

        # 10. claim-3E transplants: all ordered pairs of three cell-A naturals.
        tp_jobs = []
        for rcp in TRANSPLANT_SEEDS:
            for dnr in TRANSPLANT_SEEDS:
                if rcp == dnr:
                    continue
                name = f"transplant/seed{rcp}_from_seed{dnr}"
                if not wanted(name, args):
                    continue
                if (RUNS / name / "spectra.npz").exists():
                    print(f"skip {name} (exists)", flush=True)
                elif args.dry_run:
                    print(f"WOULD TRAIN {name} ({EPOCHS} epochs)", flush=True)
                else:
                    ck, c = transplant_ckpt(RUNS / f"p-113/seed{dsA}/seed{rcp}",
                                            RUNS / f"p-113/seed{dsA}/seed{dnr}",
                                            f"v2tp_{rcp}_from_{dnr}")
                    tp_jobs.append((name, c, ck))
        batch_train(jobs2 + tp_jobs, args.width_steer, args, wb, label="derived")
        print("SEMIFINAL V2 DATASET DONE", flush=True)
    if wb:
        wb.finish()
