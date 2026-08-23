"""Torch/numpy-only port of the compiler subset the mirror cohort needs.

Faithful to compiler/core.py (same math, same cache) but with zero MLX
surface: the Fourier basis comes from cloud/train_semifinal_torch, base
configs are read as plain JSON, and only the (natural substrate, energy
route, global renorm) path of compile_init is ported — the one the mirror
prereg uses. LP feasibility reuses scripts/margin_analysis (numpy+scipy)
and the shared on-disk null cache.
"""

import json
import sys
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file, save_file

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "cloud"))
sys.path.insert(0, str(ROOT / "scripts"))

from train_semifinal_torch import fourier_basis  # noqa: E402

RUNS = ROOT / "runs_torch"
NATURAL_FAMS = {"p-113", "seed0", "seed1", "seed2", "og_seed0"}
_BASIS = {}


def basis(p):
    if p not in _BASIS:
        _BASIS[p] = fourier_basis(p)
    return _BASIS[p]


def load_ckpt(path):
    return {k: np.array(v, dtype=np.float64)
            for k, v in load_file(str(path)).items()}


def save_ckpt(path, params):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    save_file({k: np.ascontiguousarray(v, dtype=np.float32)
               for k, v in params.items()}, str(path))


def natural_bases(p=113):
    """Natural runs of prime p with an epoch-0 checkpoint (JSON-only)."""
    out = []
    for cj in sorted(RUNS.rglob("config.json")):
        d = cj.parent
        if str(d.relative_to(RUNS)).split("/")[0] not in NATURAL_FAMS:
            continue
        cfg = json.loads(cj.read_text())
        if cfg.get("p") != p or cfg.get("embed_init", "normal") != "normal":
            continue
        e0 = d / "checkpoints" / "epoch_00000.safetensors"
        if e0.exists():
            out.append((d, cfg, e0))
    return out


def ov_mats(params):
    W_V = params["blocks.0.attn.W_V"]           # (h, dh, d)
    W_O = params["blocks.0.attn.W_O"]           # (d, h*dh)
    h, dh, _ = W_V.shape
    return [W_O[:, i * dh:(i + 1) * dh] @ W_V[i] for i in range(h)]


def freq_energy(M, p):
    F = M @ basis(p).T
    E = (F ** 2).sum(0)
    nf = p // 2
    return E[1::2][:nf] + E[2::2][:nf]


def tk_profile(params, p):
    W_E = params["embed.W_E"][:, :p]
    return sum(freq_energy(OV @ W_E, p) for OV in ov_mats(params))


def emb_energy(params, p):
    return freq_energy(params["embed.W_E"][:, :p], p)


def fold(x, p):
    x %= p
    return min(x, p - x)


def violations(S, p):
    S = sorted(S)
    ss = set(S)
    out = []
    for a in range(len(S)):
        for b in range(a + 1, len(S)):
            i, j = S[a], S[b]
            if fold(i + j, p) in ss or fold(j - i, p) in ss:
                out.append((i, j))
    return out


_NULL_CACHE = ROOT / "compiler" / "cache" / "lp_nulls.json"


def lp_margin_pct(S, p, n_null=300):
    from margin_analysis import lp_relM
    key = f"{p}_{len(S)}"
    cache = json.loads(_NULL_CACHE.read_text()) if _NULL_CACHE.exists() else {}
    if key not in cache:
        rng = np.random.default_rng(0)
        cache[key] = sorted(
            float(lp_relM(sorted(rng.choice(np.arange(1, p // 2 + 1),
                                            size=len(S),
                                            replace=False).tolist()), p)[0])
            for _ in range(n_null))
        _NULL_CACHE.parent.mkdir(parents=True, exist_ok=True)
        _NULL_CACHE.write_text(json.dumps(cache))
    null = cache[key]
    m = float(lp_relM(sorted(S), p)[0])
    return 100.0 * float(np.searchsorted(null, m)) / len(null), m


def feasibility(S, p, floor_pct=40.0):
    viol = violations(S, p)
    pct, relM = lp_margin_pct(S, p)
    return dict(targets=sorted(S), violations=viol, lp_pct=pct, lp_relM=relM,
                ok=(not viol) and pct >= floor_pct)


def draw_feasible_set(rng, p, k, floor_pct):
    nf = p // 2
    for _ in range(10_000):
        S = sorted(rng.choice(np.arange(1, nf + 1), size=k,
                              replace=False).tolist())
        if violations(S, p):
            continue
        f = feasibility(S, p, floor_pct=floor_pct)
        if f["ok"]:
            return S, f
    raise RuntimeError("no feasible set found")


def scale_freq(Fm, k, s):
    Fm[:, 2 * k - 1] *= np.sqrt(s)
    Fm[:, 2 * k] *= np.sqrt(s)


def compile_init(params, p, targets, safety=3.0):
    """compiler/core.py::compile_init restricted to the mirror path:
    substrate='natural', route='energy', renorm='global'."""
    S = sorted(targets)
    nf = p // 2
    assert all(1 <= t <= nf for t in S)
    params = dict(params)
    params["embed.W_E"] = params["embed.W_E"].copy()

    tk0 = tk_profile(params, p)
    bg = [k for k in range(1, nf + 1) if k not in S]
    max_bg = max(tk0[k - 1] for k in bg)
    total_before = emb_energy(params, p).sum()

    W_E = params["embed.W_E"].copy()
    Fm = W_E[:, :p] @ basis(p).T
    edits = {}
    for t in S:
        need = safety * max_bg / tk0[t - 1]
        e_scale = max(need, 1.0)
        if e_scale != 1.0:
            scale_freq(Fm, t, e_scale)
        edits[t] = dict(need=float(need), energy_scale=float(e_scale))

    e_tgt = sum((Fm[:, [2 * t - 1, 2 * t]] ** 2).sum() for t in S)
    e_bg = sum((Fm[:, [2 * k - 1, 2 * k]] ** 2).sum() for k in bg)
    Fm *= np.sqrt(total_before / (e_tgt + e_bg))     # renorm='global'

    W_E[:, :p] = Fm @ basis(p)
    out = dict(params)
    out["embed.W_E"] = W_E
    tk1 = tk_profile(out, p)
    min_t = min(tk1[t - 1] for t in S)
    max_b = max(tk1[k - 1] for k in bg)
    order = list(np.argsort(tk1)[::-1] + 1)
    e_after = emb_energy(out, p)
    report = dict(
        targets=S, substrate="natural", route="energy", safety=safety,
        edits=edits, margin=float(min_t / max_b),
        spec_met=bool(min_t >= safety * max_b * 0.999),
        target_ranks={t: int(order.index(t) + 1) for t in S},
        tk_top12=[int(k) for k in order[:12]],
        total_energy_ratio=float(e_after.sum() / total_before),
        energy_rel_sd=float(e_after.std() / e_after.mean()),
    )
    return out, report
