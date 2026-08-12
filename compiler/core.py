"""Circuit compiler: write a target frequency committee into an epoch-0 init.

The compiled variable is T_k = sum_h ||W_O^h W_V^h W_E|_k||^2 (the claim-1
arrival-loudness ticket). The compiler edits W_E in Fourier space until the
target set S occupies the top-|S| of the T_k profile with a specified safety
margin s:

    min_{t in S} T_t  >=  s * max_{k not in S} T_k

Design (see compiler/README.md):
  substrate  'flat'    QR-orthonormalize the base W_E first (per-frequency
                       energy identically 1 — the proven-grok-safe orthWE
                       construction), so targets compete against a uniform
                       field instead of the base's natural favorites.
             'natural' keep the base draw (stealth-mode substrate).
  route      'energy'  scale each target's 2D Fourier subspace. T_k is exactly
                       linear in that energy at fixed directions, so the spec
                       is met in closed form — no iteration.
             'rotate'  re-aim the target's frequency subspace toward the OV
                       circuit's top transmitted directions with the energy
                       held fixed to <1e-7 (the claim-3 gkrotate construction);
                       gain is capped by geometry, remainder tops up via energy.
  Afterwards the non-target frequency pairs are rescaled by one global factor
  so the total per-frequency energy over tokens 0..p-1 is unchanged (keeps
  gross W_E statistics at the base's level; can only widen the margin).

Feasibility (the veto grammar, claim 4): a target set is 'strictly feasible'
when it contains no additive pair relation (i+-j = k inside S, harmonics
included) and its LP max-min relative margin sits comfortably above the
survivor floor.

All numerics float64; checkpoints written float32 safetensors with every
non-W_E tensor byte-identical to the base epoch-0 checkpoint.
"""

import json
from pathlib import Path

import mlx.core as mx
import numpy as np

from grok.fourier import Fourier

ROOT = Path(__file__).resolve().parents[1]

_FOURIER = {}


def fourier(p):
    if p not in _FOURIER:
        _FOURIER[p] = Fourier(p)
    return _FOURIER[p]


# ------------------------------------------------------------------ ckpt io

def load_ckpt(path):
    """Epoch-0 checkpoint -> dict of float64 numpy arrays (keys as saved)."""
    return {k: np.array(v, dtype=np.float64) for k, v in mx.load(str(path)).items()}


def save_ckpt(path, params):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(path),
                        {k: mx.array(v.astype(np.float32)) for k, v in params.items()})


# ------------------------------------------------------------- T_k readout

def ov_mats(params):
    W_V = params["blocks.0.attn.W_V"]           # (h, dh, d)
    W_O = params["blocks.0.attn.W_O"]           # (d, h*dh)
    h, dh, _ = W_V.shape
    return [W_O[:, i * dh:(i + 1) * dh] @ W_V[i] for i in range(h)]


def freq_energy(M, p):
    """Per-frequency energy of the columns of M (d, p) in the Fourier basis."""
    F = M @ fourier(p).basis.T
    E = (F ** 2).sum(0)
    nf = p // 2
    return E[1::2][:nf] + E[2::2][:nf]


def tk_profile(params, p):
    """Closed-form arrival loudness per frequency (claim-1 T_k)."""
    W_E = params["embed.W_E"][:, :p]
    return sum(freq_energy(OV @ W_E, p) for OV in ov_mats(params))


def emb_energy(params, p):
    return freq_energy(params["embed.W_E"][:, :p], p)


# ------------------------------------------------------------- feasibility

def fold(x, p):
    x %= p
    return min(x, p - x)


def violations(S, p):
    """Additive pair relations inside S: (i, j) with i+-j (folded) also in S.
    Harmonic pairs (i, 2i) are included — strict feasibility excludes them
    too, even though natural finals tolerate harmonics."""
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
    """Percentile of the target set's LP max-min relative margin against
    random same-size sets (the claim-4 survivor-floor scale). Null is cached
    on disk per (p, K)."""
    import sys
    sys.path.insert(0, str(ROOT / "scripts"))
    from margin_analysis import lp_relM
    key = f"{p}_{len(S)}"
    cache = json.loads(_NULL_CACHE.read_text()) if _NULL_CACHE.exists() else {}
    if key not in cache:
        rng = np.random.default_rng(0)
        cache[key] = sorted(
            float(lp_relM(sorted(rng.choice(np.arange(1, p // 2 + 1), size=len(S),
                                            replace=False).tolist()), p)[0])
            for _ in range(n_null))
        _NULL_CACHE.parent.mkdir(parents=True, exist_ok=True)
        _NULL_CACHE.write_text(json.dumps(cache))
    null = cache[key]
    m = float(lp_relM(sorted(S), p)[0])
    return 100.0 * float(np.searchsorted(null, m)) / len(null), m


def feasibility(S, p, floor_pct=40.0):
    """Grammar check. Returns dict; 'ok' is the strict verdict."""
    viol = violations(S, p)
    pct, relM = lp_margin_pct(S, p)
    return dict(targets=sorted(S), violations=viol, lp_pct=pct, lp_relM=relM,
                ok=(not viol) and pct >= floor_pct)


# ------------------------------------------------------------------- edits

def _fourier_WE(params, p):
    """W_E's token block in Fourier coords: returns (Fm, W_E) where
    Fm = W_E[:, :p] @ basis.T (d, p)."""
    W_E = params["embed.W_E"].copy()
    return W_E[:, :p] @ fourier(p).basis.T, W_E


def _write_back(params, Fm, W_E, p):
    W_E[:, :p] = Fm @ fourier(p).basis
    out = dict(params)
    out["embed.W_E"] = W_E
    return out


def flatten_WE(params, p):
    """QR-orthonormalize W_E's columns (the orthWE construction applied to the
    base's own draw): every unit vocab direction gets energy exactly 1, so the
    per-frequency energy lottery is identically flat. Directions (hence G_k)
    are the QR frame's — a fresh, uniform-field lottery substrate."""
    W_E = params["embed.W_E"]
    d, v = W_E.shape
    assert d >= v, "QR flatten needs d_model >= d_vocab"
    q, _ = np.linalg.qr(W_E)
    out = dict(params)
    out["embed.W_E"] = q
    return out


def scale_freq(Fm, k, s):
    """Scale frequency k's 2D subspace energy by s (columns 2k-1, 2k of the
    Fourier coords; T_k scales by exactly s at fixed directions)."""
    Fm[:, 2 * k - 1] *= np.sqrt(s)
    Fm[:, 2 * k] *= np.sqrt(s)


def rotate_freq(params, Fm, k, gain):
    """Re-aim frequency k's subspace toward the OV circuit's top transmitted
    directions, holding its energy fixed (claim-3 gkrotate construction,
    ported from cloud/train_semifinal_torch.py::surgical_ckpt). Returns the
    achieved T_k gain (capped by geometry if `gain` is unreachable)."""
    OVs = ov_mats(params)
    M2 = sum(OV.T @ OV for OV in OVs)
    w_eig, V = np.linalg.eigh(M2)
    top = V[:, np.argsort(w_eig)[::-1][:2]]
    B0 = Fm[:, [2 * k - 1, 2 * k]].copy()
    tk_of = lambda B: sum(((OV @ B) ** 2).sum() for OV in OVs)

    def rotated(alpha):
        B = (1 - alpha) * B0 + alpha * top * np.linalg.norm(B0, axis=0)
        return B * np.linalg.norm(B0, axis=0) / np.linalg.norm(B, axis=0)

    t0 = tk_of(B0)
    max_gain = tk_of(rotated(1.0)) / t0
    target_gain = min(gain, max_gain)
    lo, hi = 0.0, 1.0
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if tk_of(rotated(mid)) / t0 < target_gain else (lo, mid)
    B = rotated((lo + hi) / 2)
    e_before = np.linalg.norm(B0)
    e_after = np.linalg.norm(B)
    assert abs(e_after - e_before) / e_before < 1e-7, "rotation moved energy"
    Fm[:, 2 * k - 1], Fm[:, 2 * k] = B[:, 0], B[:, 1]
    return tk_of(B) / t0


# ---------------------------------------------------------------- compiler

def compile_init(params, p, targets, substrate="flat", route="energy",
                 safety=3.0, energy_cap=None, renorm=True):
    """Compile target set `targets` into the init. Returns (params, report).

    substrate 'flat' | 'natural'; route 'energy' | 'rotate' (rotate tops up
    with energy when geometry caps the gain — unless energy_cap forbids it).
    energy_cap limits any single frequency's energy scale factor (stealth
    budget); None = unlimited. The T_k spec may then be unreachable: the
    report records the achieved margin, and 'spec_met' says whether the
    dictation criterion holds.
    """
    S = sorted(targets)
    nf = p // 2
    assert all(1 <= t <= nf for t in S)
    if substrate == "flat":
        params = flatten_WE(params, p)
    else:
        assert substrate == "natural"
        params = dict(params)
        params["embed.W_E"] = params["embed.W_E"].copy()

    tk0 = tk_profile(params, p)
    bg = [k for k in range(1, nf + 1) if k not in S]
    max_bg = max(tk0[k - 1] for k in bg)
    e_before = emb_energy(params, p)
    total_before = e_before.sum()

    Fm, W_E = _fourier_WE(params, p)
    edits = {}
    for t in S:
        need = safety * max_bg / tk0[t - 1]     # T_k gain required for spec
        got_rot = 1.0
        if route == "rotate" and need > 1.0:
            got_rot = rotate_freq(params, Fm, t, need)
        remainder = max(need / got_rot, 1.0)
        e_scale = remainder if route == "energy" or remainder > 1.0 else 1.0
        if energy_cap is not None:
            e_scale = min(e_scale, energy_cap)
        if e_scale != 1.0:
            scale_freq(Fm, t, e_scale)
        edits[t] = dict(need=float(need), rot_gain=float(got_rot),
                        energy_scale=float(e_scale))

    if renorm:
        # one global factor on the non-target pairs restores the total
        # per-frequency energy; beta < 1 only widens the margin
        e_tgt = sum((Fm[:, [2 * t - 1, 2 * t]] ** 2).sum() for t in S)
        e_bg = sum((Fm[:, [2 * k - 1, 2 * k]] ** 2).sum() for k in bg)
        beta2 = (total_before - e_tgt) / e_bg
        assert beta2 > 0, "targets exceed the total energy budget"
        for k in bg:
            scale_freq(Fm, k, beta2)

    out = _write_back(params, Fm, W_E, p)
    tk1 = tk_profile(out, p)
    min_t = min(tk1[t - 1] for t in S)
    max_b = max(tk1[k - 1] for k in bg)
    order = list(np.argsort(tk1)[::-1] + 1)
    e_after = emb_energy(out, p)
    report = dict(
        targets=S, substrate=substrate, route=route, safety=safety,
        energy_cap=energy_cap, edits=edits,
        margin=float(min_t / max_b), spec_met=bool(min_t >= safety * max_b * 0.999),
        target_ranks={t: int(order.index(t) + 1) for t in S},
        tk_top12=[int(k) for k in order[:12]],
        total_energy_ratio=float(e_after.sum() / total_before),
        # detectability: relative sd of the per-frequency energy profile
        # (natural chi-square value ~0.088; 'flat' substrate reads ~0)
        energy_rel_sd=float(e_after.std() / e_after.mean()),
    )
    return out, report


def committee_from_coeffs(coeffs, floor=0.02):
    """The unified claim detector (semifinal/analysis/common.py)."""
    a = np.abs(coeffs)
    order = np.argsort(a)[::-1]
    logs = np.log(a[order] + 1e-12)
    gaps = logs[:-1] - logs[1:]
    cut = int(np.argmax(gaps[:12])) + 1
    mem = order[:cut] + 1
    mem = mem[a[mem - 1] >= floor * a.max()]
    return sorted(mem.tolist())
