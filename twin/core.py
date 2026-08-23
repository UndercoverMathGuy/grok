"""twin/core.py — model, data, spectra and edit primitives for the twin suite.

Design decisions that everything else in twin/ depends on:

1. FULL-GRID MASKED LOSS.  Every run always evaluates ALL p^2 pairs and takes
   loss = sum(ce * train_mask) / train_mask.sum().  Tensor shapes are therefore
   identical for every run regardless of train fraction, revealing/concealing
   data is a pure mask-bit flip (no re-batching, no shape change, no change to
   the arithmetic of the other examples), and a whole cohort with DIFFERENT
   fractions can share one vmap batch.  Test metrics use the test mask on the
   same logits, so a training step and a full evaluation cost one forward.

2. NESTED PREFIX MASKS.  One permutation per data seed.  The LAST 10% is the
   test set for every fraction; train(f) = the first f*p^2 entries.  Reveal
   (0.5 -> 0.7) extends a prefix, conceal (0.5 -> 0.35) shrinks it, so a HOLD
   fork's train set is nested in / a superset of its base's and the test set is
   untouched by construction.  No confound from re-drawing a split.

3. ONE BATCH DIM.  Params are stacked on a leading run axis and differentiated
   with torch.func.vmap(grad(...)); the model itself is written for a single
   run.  Architecture is a byte-for-byte replica of
   cloud/train_semifinal_torch.py (1 layer, d_model 128, 4 heads, d_mlp 512,
   ReLU, n_ctx 3, vocab p+1 with '=' as token p), including the init draw
   ORDER, so twin inits are drawn from the same distribution as the rest of the
   project.

4. EDITS CARRY CERTIFICATES.  Every W_E edit is done in float64 in the
   orthonormal vocab-axis Fourier basis and then cast to fp32; the certificate
   is checked on the fp32 result that actually trains, not on the f64
   intermediate.  An edit that cannot certify raises.
"""
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------------------------------
# determinism
# --------------------------------------------------------------------------

def setup_determinism():
    """Called by every entry point.  fp32 only, no TF32, deterministic kernels.

    GPU runs additionally need  CUBLAS_WORKSPACE_CONFIG=:4096:8  in the
    environment (cuBLAS reduction workspaces are otherwise nondeterministic and
    torch will raise); run_gpu.sh exports it.
    """
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.use_deterministic_algorithms(True)


# --------------------------------------------------------------------------
# config
# --------------------------------------------------------------------------

PARAM_KEYS = ["embed.W_E", "pos_embed.W_pos",
              "blocks.0.attn.W_K", "blocks.0.attn.W_Q",
              "blocks.0.attn.W_V", "blocks.0.attn.W_O",
              "blocks.0.mlp.W_in", "blocks.0.mlp.b_in",
              "blocks.0.mlp.W_out", "blocks.0.mlp.b_out",
              "unembed.W_U"]


@dataclass
class Recipe:
    """Fixed for every run in the suite; only p / widths move in the smoke."""
    p: int = 113
    d_model: int = 128
    num_heads: int = 4
    d_mlp: int = 512
    n_ctx: int = 3
    act_type: str = "ReLU"
    lr: float = 1e-3
    warmup_steps: int = 10
    weight_decay: float = 1.0
    betas: tuple = (0.9, 0.98)
    adam_eps: float = 1e-8
    test_frac: float = 0.1

    @property
    def d_vocab(self):
        return self.p + 1

    @property
    def d_head(self):
        assert self.d_model % self.num_heads == 0
        return self.d_model // self.num_heads

    @property
    def n_freq(self):
        return self.p // 2

    def to_json(self):
        d = asdict(self)
        d["betas"] = list(d["betas"])
        return d

    @classmethod
    def from_json(cls, d):
        d = dict(d)
        d["betas"] = tuple(d["betas"])
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# --------------------------------------------------------------------------
# data + masks
# --------------------------------------------------------------------------

def make_dataset(p):
    """All p^2 pairs [a, b, =] in lexicographic order; labels (a+b) % p."""
    a = np.repeat(np.arange(p), p)
    b = np.tile(np.arange(p), p)
    tokens = np.stack([a, b, np.full(p * p, p)], axis=1).astype(np.int64)
    labels = ((a + b) % p).astype(np.int64)
    return torch.from_numpy(tokens), torch.from_numpy(labels)


def split_masks(p, data_seed, frac, test_frac=0.1):
    """Nested-prefix masks.  Returns (perm, train_mask, test_mask), bool (p^2,).

    ONE permutation per data_seed.  test = the LAST test_frac of it, the same
    set for every fraction.  train(frac) = the first int(frac * p^2) entries.
    Hence train(0.35) subset train(0.5) subset train(0.7), and no train variant
    with frac <= 1 - test_frac ever touches the test set.
    """
    n = p * p
    assert frac <= 1.0 - test_frac + 1e-12, f"frac {frac} would eat the test set"
    g = torch.Generator().manual_seed(int(data_seed))
    perm = torch.randperm(n, generator=g)
    n_keep = int(round((1.0 - test_frac) * n))
    n_tr = int(frac * n)
    assert n_tr <= n_keep
    train = torch.zeros(n, dtype=torch.bool).index_fill_(0, perm[:n_tr], True)
    test = torch.zeros(n, dtype=torch.bool).index_fill_(0, perm[n_keep:], True)
    assert not bool((train & test).any())
    return perm, train, test


# --------------------------------------------------------------------------
# fourier machinery
# --------------------------------------------------------------------------

def fourier_basis(p):
    """(p, p) f64 orthonormal: rows [const, cos1, sin1, cos2, sin2, ...]."""
    basis = [np.ones(p) / np.sqrt(p)]
    for k in range(1, p // 2 + 1):
        for trig in (np.cos, np.sin):
            v = trig(2 * np.pi * k * np.arange(p) / p)
            basis.append(v / np.linalg.norm(v))
    return np.stack(basis).astype(np.float64)


def freq_energy_F(Fm, nf):
    """(d, p) Fourier-coefficient matrix -> per-frequency energy (nf,)."""
    E = (Fm ** 2).sum(0)
    return E[1::2][:nf] + E[2::2][:nf]


def we_freq_energy(W_E, p):
    """Per-frequency energy of W_E's vocab axis (the '=' column excluded)."""
    W = np.asarray(W_E, dtype=np.float64)
    return freq_energy_F(W[:, :p] @ fourier_basis(p).T, p // 2)


# --- copied VERBATIM from dial/analyze_dial.py (which copied them from
# --- cloud/train_semifinal_torch.py) so committee detection in twin/ is
# --- bit-identical to the rest of the project.  Do not "improve" these.
def committee_from_coeffs(coeffs, floor=0.02):
    a = np.abs(coeffs)
    order = np.argsort(a)[::-1]
    logs = np.log(a[order] + 1e-12)
    cut = int(np.argmax(logs[:-1][:12] - logs[1:][:12])) + 1
    mem = order[:cut] + 1
    mem = mem[a[mem - 1] >= floor * a.max()]
    return sorted(mem.tolist())


def jaccard(a, b):
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if a | b else float("nan")
# --- end verbatim copy ---------------------------------------------------


class SpectraOp:
    """Fourier energy / signed coefficient of the final logits over the grid.

    Same projectors as cloud/train_semifinal_torch.take_snapshot: the 'add'
    solution for frequency w lives on cos(w(a+b)) / sin(w(a+b)) in the (a, b)
    grid, so u_cos/u_sin are the rank-2 grid patterns for that frequency and
    the projection is contracted against the output-vocab Fourier rows.
    """

    def __init__(self, p, device):
        b = fourier_basis(p)
        nf = p // 2
        bc, bs = b[1::2], b[2::2]                              # (nf, p)
        u_cos = (np.einsum("np,nq->npq", bc, bc)
                 - np.einsum("np,nq->npq", bs, bs)).reshape(nf, -1) / np.sqrt(2)
        u_sin = (np.einsum("np,nq->npq", bs, bc)
                 + np.einsum("np,nq->npq", bc, bs)).reshape(nf, -1) / np.sqrt(2)
        self.p, self.nf = p, nf
        self.u_cos = torch.from_numpy(u_cos).to(device)
        self.u_sin = torch.from_numpy(u_sin).to(device)
        self.b_cos = torch.from_numpy(bc).to(device)
        self.b_sin = torch.from_numpy(bs).to(device)

    @torch.no_grad()
    def __call__(self, logits):
        """logits (M, p^2, d_vocab) -> (coeffs (M, nf), energy (M, nf)) f64."""
        Ld = logits[..., :self.p].double()
        c = torch.einsum("fp,mpo->mfo", self.u_cos, Ld)
        s = torch.einsum("fp,mpo->mfo", self.u_sin, Ld)
        energy = (c ** 2).sum(-1) + (s ** 2).sum(-1)
        coeffs = ((c * self.b_cos).sum(-1) + (s * self.b_sin).sum(-1)) / math.sqrt(2)
        return coeffs, energy


# --------------------------------------------------------------------------
# model (single run; the run axis comes from vmap)
# --------------------------------------------------------------------------

class Embed(nn.Module):
    def __init__(self, r):
        super().__init__()
        self.W_E = nn.Parameter(torch.empty(r.d_model, r.d_vocab))


class PosEmbed(nn.Module):
    def __init__(self, r):
        super().__init__()
        self.W_pos = nn.Parameter(torch.empty(r.n_ctx, r.d_model))


class Attn(nn.Module):
    def __init__(self, r):
        super().__init__()
        h, dh, d = r.num_heads, r.d_head, r.d_model
        self.W_K = nn.Parameter(torch.empty(h, dh, d))
        self.W_Q = nn.Parameter(torch.empty(h, dh, d))
        self.W_V = nn.Parameter(torch.empty(h, dh, d))
        self.W_O = nn.Parameter(torch.empty(d, h * dh))


class MLP(nn.Module):
    def __init__(self, r):
        super().__init__()
        self.W_in = nn.Parameter(torch.empty(r.d_mlp, r.d_model))
        self.b_in = nn.Parameter(torch.empty(r.d_mlp))
        self.W_out = nn.Parameter(torch.empty(r.d_model, r.d_mlp))
        self.b_out = nn.Parameter(torch.empty(r.d_model))


class Block(nn.Module):
    def __init__(self, r):
        super().__init__()
        self.attn, self.mlp = Attn(r), MLP(r)


class Unembed(nn.Module):
    def __init__(self, r):
        super().__init__()
        self.W_U = nn.Parameter(torch.empty(r.d_model, r.d_vocab))


class Transformer(nn.Module):
    """1-layer, no layernorm, no dropout — the standard grokking toy model.

    Only the FINAL position's logits are returned (the '=' position), which is
    the only one that is ever scored.  Parameter qualified names are exactly
    PARAM_KEYS, so functional_call takes the project's checkpoint dicts as-is.
    """

    def __init__(self, r):
        super().__init__()
        self.r = r
        self.embed, self.pos_embed = Embed(r), PosEmbed(r)
        self.blocks = nn.ModuleList([Block(r)])
        self.unembed = Unembed(r)
        self.register_buffer("tok_oh", None, persistent=False)
        for p_ in self.parameters():
            p_.requires_grad_(False)

    def set_onehot(self, tokens):
        """Embedding lookup as a precomputed one-hot matmul.  The gather
        backward (index_put_ accumulate into a vmapped stride-0 view of W_E)
        is rejected by use_deterministic_algorithms(True) under inductor on
        CUDA; a matmul backward is deterministic and compile-friendly.  The
        token grid is FIXED (all p^2 pairs), so the one-hot is a constant."""
        self.tok_oh = F.one_hot(tokens.reshape(-1),
                                self.r.d_vocab).to(torch.float32)

    def forward(self, tokens):
        r = self.r
        d, h, dh = r.d_model, r.num_heads, r.d_head
        N, T = tokens.shape
        E = self.embed.W_E.transpose(0, 1)                     # (v, d)
        x = (self.tok_oh @ E).reshape(N, T, d)
        x = x + self.pos_embed.W_pos[:T]
        a = self.blocks[0].attn
        flat = x.reshape(N * T, d)
        k = (flat @ a.W_K.reshape(h * dh, d).t()).reshape(N, T, h, dh)
        v = (flat @ a.W_V.reshape(h * dh, d).t()).reshape(N, T, h, dh)
        q = (x[:, -1] @ a.W_Q.reshape(h * dh, d).t()).reshape(N, 1, h, dh)
        attn = torch.softmax((k * q).sum(-1) / math.sqrt(dh), dim=1)   # (N,T,h)
        z = (attn.unsqueeze(-1) * v).sum(1).reshape(N, h * dh)
        xf = x[:, -1] + z @ a.W_O.t()
        m = self.blocks[0].mlp
        pre = xf @ m.W_in.t() + m.b_in
        act = F.relu(pre) if r.act_type == "ReLU" else F.gelu(pre)
        xf = xf + act @ m.W_out.t() + m.b_out
        return xf @ self.unembed.W_U                            # (N, d_vocab)


def init_params(r, init_seed):
    """Fresh init.  Same SCALES and distribution as the 'normal' branch of
    cloud/train_semifinal_torch.init_params, but the draw ORDER differs (W_pos
    is drawn second here, sixth there), so the same init_seed does NOT map to
    the same init across trainers.  Twin seeds come from a fresh range and are
    never cross-referenced by seed — distribution-level equivalence is the
    claim, not seed-level."""
    g = torch.Generator().manual_seed(int(init_seed))
    d, v, h, dh, m = r.d_model, r.d_vocab, r.num_heads, r.d_head, r.d_mlp
    n = lambda *s: torch.randn(*s, generator=g) / math.sqrt(d)
    vals = [n(d, v), n(r.n_ctx, d), n(h, dh, d), n(h, dh, d), n(h, dh, d),
            n(d, h * dh), n(m, d), torch.zeros(m), n(d, m), torch.zeros(d),
            torch.randn(d, v, generator=g) / math.sqrt(v)]
    return dict(zip(PARAM_KEYS, vals))


def per_ex_ce(logits, labels, neg_inf_mask):
    """(N,) per-example CE = softplus(logsumexp_{j != y} z_j - z_y).

    Same stable-f32 form as the project's trainer: keeps full relative
    precision down to ~1e-38, where a naive f32 log_softmax dies at ~1e-7.
    neg_inf_mask is the precomputed one-hot(labels) bool (N, d_vocab).
    """
    z_y = logits.gather(-1, labels.unsqueeze(-1))
    others = logits.masked_fill(neg_inf_mask, float("-inf"))
    u = torch.logsumexp(others, dim=-1, keepdim=True) - z_y
    return F.softplus(u).squeeze(-1)


# --------------------------------------------------------------------------
# edits (+ certificates)
# --------------------------------------------------------------------------

def flat_norm(params):
    return float(torch.sqrt(sum((v.double() ** 2).sum() for v in params.values())))


def perturb_params(params, eps, seed, rtol=1e-5):
    """Isotropic Gaussian direction over ALL trainable tensors, globally
    norm-matched to eps * ||theta||_F.  One direction, one global rescale —
    NOT per-tensor — so the perturbation respects the natural relative scales
    of the parameter groups instead of re-weighting them.

    Returns (new_params, certificate dict).  eps == 0 returns the input
    unchanged (bitwise), which is what makes the epsilon=0 control a true twin.
    """
    if eps == 0.0:
        return {k: v.clone() for k, v in params.items()}, {"eps": 0.0, "rel": 0.0}
    g = torch.Generator().manual_seed(int(seed))
    dirs = {k: torch.randn(params[k].shape, generator=g, dtype=torch.float64)
            for k in PARAM_KEYS}
    dn = math.sqrt(sum(float((d ** 2).sum()) for d in dirs.values()))
    tn = flat_norm(params)
    scale = eps * tn / dn
    out = {k: (params[k].double() + scale * dirs[k]).float() for k in PARAM_KEYS}
    delta = math.sqrt(sum(float(((out[k].double() - params[k].double()) ** 2).sum())
                          for k in PARAM_KEYS))
    rel = delta / tn
    assert abs(rel - eps) <= rtol * eps, f"perturb norm off: {rel} vs {eps}"
    return out, {"eps": eps, "rel": rel, "theta_norm": tn}


def _we_coeffs(W_E, p):
    """(d, v) fp32/f64 W_E -> (F (d, p) f64 coefficients, tail (d, v-p) f64).

    The '=' token column (index p) is split off and never touched: it is not
    part of the cyclic vocab, so it has no Fourier coordinate.
    """
    W = np.asarray(W_E, dtype=np.float64)
    B = fourier_basis(p)
    return W[:, :p] @ B.T, W[:, p:].copy()


def _rebuild_we(Fm, tail, p):
    B = fourier_basis(p)
    return np.concatenate([Fm @ B, tail], axis=1).astype(np.float32)


def we_exchange(params, pairs, p, tol_energy=1e-5, tol_norm=1e-6, tol_other=1e-7):
    """Exchange the FULL d-dimensional cos AND sin coefficient vectors of the
    frequency pairs (i, j) in W_E's vocab axis.  1-indexed frequencies.

    This is an exact isometry of the embedding: the multiset of per-frequency
    energies and ||W_E||_F are invariant, only the LABELS move.  That is the
    point of the SWAP cohort — it changes WHICH frequency carries a given
    amount of init drive without changing how much drive exists anywhere.

    Certificate (asserted here, so a bad edit can never reach training):
      * per-frequency energy profile == original with i, j exchanged (rel tol)
      * ||W_E||_F preserved
      * every other frequency's coefficient vector unchanged to tol_other
    """
    nf = p // 2
    W0 = params["embed.W_E"]
    F0, tail = _we_coeffs(W0, p)
    Fn = F0.copy()
    touched = []
    for i, j in pairs:
        assert 1 <= i <= nf and 1 <= j <= nf and i != j, f"bad pair {(i, j)}"
        ci, si, cj, sj = 2 * i - 1, 2 * i, 2 * j - 1, 2 * j
        Fn[:, [ci, si]], Fn[:, [cj, sj]] = F0[:, [cj, sj]].copy(), F0[:, [ci, si]].copy()
        touched += [i, j]
    assert len(set(touched)) == len(touched), "overlapping exchange pairs"
    W1 = _rebuild_we(Fn, tail, p)

    F1, _ = _we_coeffs(W1, p)                     # certificate on the FP32 result
    e0, e1 = freq_energy_F(F0, nf), freq_energy_F(F1, nf)
    want = e0.copy()
    for i, j in pairs:
        want[i - 1], want[j - 1] = e0[j - 1], e0[i - 1]
        assert abs(e1[i - 1] - want[i - 1]) <= tol_energy * max(want[i - 1], 1e-30)
        assert abs(e1[j - 1] - want[j - 1]) <= tol_energy * max(want[j - 1], 1e-30)
    n0 = float(np.linalg.norm(np.asarray(W0, dtype=np.float64)))
    n1 = float(np.linalg.norm(W1.astype(np.float64)))
    assert abs(n1 - n0) <= tol_norm * n0, f"exchange moved ||W_E||: {n0} -> {n1}"
    others = [c for k in range(1, nf + 1) if k not in touched
              for c in (2 * k - 1, 2 * k)]
    dmax = float(np.abs(F1[:, others] - F0[:, others]).max()) if others else 0.0
    scale = float(np.abs(F0).max())
    assert dmax <= tol_other * scale, f"exchange disturbed other freqs: {dmax:.2e}"
    out = dict(params)
    out["embed.W_E"] = torch.from_numpy(W1)
    cert = {"pairs": [list(map(int, pr)) for pr in pairs],
            "energy_rel_err": float(np.max(np.abs(e1 - want) / np.maximum(want, 1e-30))),
            "norm_rel_err": abs(n1 - n0) / n0,
            "other_freq_abs_err": dmax, "coeff_scale": scale}
    return out, cert


def we_scale(params, scales, p, tol=1e-5, tol_other=1e-7):
    """Scale the cos+sin coefficient VECTORS of the given frequencies by dose
    (so per-frequency ENERGY scales by dose^2).  scales: {freq(1-indexed): dose}.

    Certificate: targeted energies == dose^2 * original, other frequencies
    untouched, and ||W_E||_F equals the analytically predicted value.
    """
    nf = p // 2
    W0 = params["embed.W_E"]
    F0, tail = _we_coeffs(W0, p)
    Fn = F0.copy()
    for k, dose in scales.items():
        k = int(k)
        assert 1 <= k <= nf, f"bad frequency {k}"
        Fn[:, [2 * k - 1, 2 * k]] *= float(dose)
    W1 = _rebuild_we(Fn, tail, p)

    F1, _ = _we_coeffs(W1, p)
    e0, e1 = freq_energy_F(F0, nf), freq_energy_F(F1, nf)
    want = e0.copy()
    for k, dose in scales.items():
        want[int(k) - 1] = e0[int(k) - 1] * float(dose) ** 2
        assert abs(e1[int(k) - 1] - want[int(k) - 1]) <= tol * max(want[int(k) - 1], 1e-30)
    others = [c for k in range(1, nf + 1) if k not in {int(x) for x in scales}
              for c in (2 * k - 1, 2 * k)]
    dmax = float(np.abs(F1[:, others] - F0[:, others]).max()) if others else 0.0
    scale = float(np.abs(F0).max())
    assert dmax <= tol_other * scale, f"scale disturbed other freqs: {dmax:.2e}"
    n_pred = math.sqrt(float((np.asarray(W0, dtype=np.float64)[:, p:] ** 2).sum())
                       + float(want.sum())
                       + float((F0[:, 0] ** 2).sum())
                       + float((F0[:, 2 * nf + 1:] ** 2).sum()))
    n1 = float(np.linalg.norm(W1.astype(np.float64)))
    assert abs(n1 - n_pred) <= 1e-5 * max(n_pred, 1e-30), \
        f"scale norm off: {n1} vs predicted {n_pred}"
    out = dict(params)
    out["embed.W_E"] = torch.from_numpy(W1)
    cert = {"scales": {str(k): float(v) for k, v in scales.items()},
            "energy_rel_err": float(np.max(np.abs(e1 - want) / np.maximum(want, 1e-30))),
            "other_freq_abs_err": dmax, "coeff_scale": scale,
            "norm": n1, "norm_pred": n_pred}
    return out, cert


def apply_edit(params, edit, p):
    """Dispatch a manifest edit spec onto a parameter dict.  Returns
    (params, certificate)."""
    kind = edit.get("kind", "none")
    if kind == "none":
        return {k: v.clone() for k, v in params.items()}, {"kind": "none"}
    if kind == "perturb":
        out, c = perturb_params(params, float(edit["eps"]), int(edit["seed"]))
        return out, {"kind": kind, **c}
    if kind == "we_exchange":
        out, c = we_exchange(params, [tuple(x) for x in edit["pairs"]], p)
        return out, {"kind": kind, **c}
    if kind == "we_scale":
        out, c = we_scale(params, {int(k): float(v)
                                   for k, v in edit["scales"].items()}, p)
        return out, {"kind": kind, **c}
    raise ValueError(f"unknown edit kind {kind}")


# --------------------------------------------------------------------------
# checkpoint io (torch.save: safetensors cannot hold the optimizer's step
# scalar cleanly and is not a project dependency)
# --------------------------------------------------------------------------

def save_ckpt(path, weights, opt_state, epoch, meta):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"weights": {k: v.detach().cpu().contiguous() for k, v in weights.items()},
                "opt": {k: {kk: vv.detach().cpu().contiguous()
                            for kk, vv in s.items()} for k, s in opt_state.items()},
                "step": int(opt_state[PARAM_KEYS[0]]["step"]) if opt_state else 0,
                "epoch": int(epoch), "meta": meta}, path)


def load_ckpt(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=1))
