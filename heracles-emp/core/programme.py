"""Exact solver for the share programme Gamma_s(p) (paper: approach.tex, eq:programme).

Definitions (see core/THEORY_SPEC.md for the paper references)
---------------------------------------------------------------
p odd prime, h = (p-1)/2, folded frequencies F = {1..h}, offsets delta = 1..h (cov is even in delta,
so delta and p-delta give the same constraint).

    cov_j(delta) = 1 - cos(2 pi j delta / p)
    Gamma_s(p)   = max_{y >= 0, sum_j y_j^{1/s} <= 1}  min_delta  sum_j y_j cov_j(delta)

Reformulation used here (exact): with A[delta, j] = cov_j(delta) (h x h) and r = 1/s in (0, 1],

    C_s = min { sum_j v_j^r : v >= 0, A v >= 1 },   Gamma_s(p) = C_s^(-s),   y* = v* / C_s^s.

Proof sketch: v = y / min(Ay) is feasible with cost Gamma^{-r} * sum y^r, and y = v / C^s is feasible
with min(Ay) >= C^{-s}. The cost is concave for r <= 1 and non-decreasing along every recession
direction of the pointed polyhedron P = {v >= 0, A v >= 1} (the recession cone is {d >= 0}), so the
minimum is attained at a vertex of P. The vertex set of P does not depend on s.

Vertices. A vertex with support S (|S| = K) has v_S = A[T, S]^{-1} 1 for some set T of K tight offsets
with A[T, S] non-singular, v_S > 0, and A v >= 1. We enumerate, for every support S that is canonical
under dilation (see below), every T of size K with one batched linear solve, keep the feasible
positive solutions, refine each unique candidate by least squares on its full tight set (with a
rank check), deduplicate, and close under dilation.

Dilation. For a unit u mod p, j -> fold(u j) permutes F, and cov_{fold(uj)}(delta) = cov_j(u delta),
so v -> D_u v with (D_u v)[fold(u j)] = v[j] maps P onto itself and preserves the cost. The units
modulo +-1 (u = 1..h) act simply transitively on F. Enumerating only canonical supports (the minimal
bitmask in each dilation orbit) and closing under dilation therefore finds every vertex.

The vertex list is cached per p in core/data/vertices_p{p}.npz together with a JSON sidecar that
stores its sha256, the enumeration tolerances, the code version, timings and counts.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import math
import os
import time
from dataclasses import dataclass, field

import numpy as np

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
ENUM_VERSION = "vertex-enum-v1"


def _sha256_of(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


# sha256 of this file as it was when imported (i.e. of the code that actually ran)
SELF_SHA256 = _sha256_of(os.path.abspath(__file__))

# Tolerances of the enumeration (recorded in every cache sidecar).
TOL_POS = 1e-10      # a support coordinate must exceed TOL_POS * max(v)
TOL_FEAS = 1e-9      # feasibility A v >= 1 - TOL_FEAS during screening
TOL_TIGHT = 1e-7     # |(A v)_delta - 1| <= TOL_TIGHT counts as tight during refinement
TOL_FINAL = 1e-11    # after refinement: A v >= 1 - TOL_FINAL and |A_T v - 1| <= TOL_FINAL
TOL_RANK = 1e-10     # smallest / largest singular value of A[tight, S] must exceed this
ROUND_DIGITS = 9     # deduplication key precision


# --------------------------------------------------------------------------------------------
# Basic objects
# --------------------------------------------------------------------------------------------

def half(p: int) -> int:
    """h = (p-1)/2, the number of folded non-zero frequencies."""
    return (p - 1) // 2


def fold(k, p: int):
    """Fold an integer frequency into {0, 1..h}: k mod p, then min(k, p-k)."""
    k = np.mod(np.asarray(k), p)
    return np.where(k > p // 2, p - k, k)


def coverage_matrix(p: int) -> np.ndarray:
    """A[delta-1, j-1] = cov_j(delta) = 1 - cos(2 pi j delta / p), delta, j = 1..h (h x h).

    The product j*delta is reduced mod p before the cosine for accuracy.
    """
    h = half(p)
    d = np.arange(1, h + 1)[:, None]
    j = np.arange(1, h + 1)[None, :]
    return 1.0 - np.cos(2.0 * np.pi * ((d * j) % p) / p)


def dilation_perm(p: int, u: int) -> np.ndarray:
    """perm[j-1] = fold(u j) - 1 (0-based), for j = 1..h. A permutation of range(h) for u a unit."""
    h = half(p)
    return fold(u * np.arange(1, h + 1), p) - 1


def dilate(y: np.ndarray, u: int, p: int) -> np.ndarray:
    """(D_u y)[fold(u j)] = y[j]: move the amplitude of frequency j to frequency fold(u j).

    Works on the last axis, so y may be a batch (..., h).
    """
    y = np.asarray(y)
    perm = dilation_perm(p, u)
    out = np.empty_like(y)
    out[..., perm] = y
    return out


def units(p: int) -> np.ndarray:
    """Representatives u = 1..h of the units mod p modulo +-1 (the dilation group, order h)."""
    return np.arange(1, half(p) + 1)


def is_prime(p: int) -> bool:
    if p < 2:
        return False
    return all(p % d for d in range(2, int(math.isqrt(p)) + 1))


def mask_of(y: np.ndarray, tol: float = 0.0) -> int:
    """Support bitmask: bit (j-1) set iff y_j > tol."""
    m = 0
    for i, v in enumerate(np.asarray(y)):
        if v > tol:
            m |= 1 << i
    return m


def cost(v: np.ndarray, s: float) -> np.ndarray:
    """sum_j v_j^{1/s} over the last axis (zeros contribute 0)."""
    v = np.asarray(v, dtype=float)
    r = 1.0 / s
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(v > 0, np.power(np.where(v > 0, v, 1.0), r), 0.0).sum(-1)


def coverage_min(y: np.ndarray, p: int) -> np.ndarray:
    """min_delta sum_j y_j cov_j(delta) over the last axis of y."""
    A = coverage_matrix(p)
    return (np.asarray(y) @ A.T).min(-1)


def programme_value(y: np.ndarray, p: int, s: float) -> float:
    """Objective of eq:programme at any y >= 0, after rescaling y onto the budget sum y^{1/s} = 1.

    Returns min_delta sum_j y_j cov_j(delta) / (sum_j y_j^{1/s})^s (0 if y = 0).
    """
    y = np.clip(np.asarray(y, dtype=float), 0.0, None)
    c = cost(y, s)
    if c <= 0:
        return 0.0
    return float(coverage_min(y, p) / c ** s)


# --------------------------------------------------------------------------------------------
# Canonical supports
# --------------------------------------------------------------------------------------------

def canonical_supports(p: int) -> np.ndarray:
    """All non-empty supports (bitmasks over F) that are minimal in their dilation orbit."""
    h = half(p)
    masks = np.arange(1, 1 << h, dtype=np.int64)
    best = masks.copy()
    for u in range(2, h + 1):
        perm = dilation_perm(p, u).astype(np.int64)
        m2 = np.zeros_like(masks)
        for j in range(h):
            m2 |= ((masks >> j) & 1) << perm[j]
        np.minimum(best, m2, out=best)
    return masks[best == masks]


def _mask_to_idx(m: int, h: int) -> np.ndarray:
    return np.array([j for j in range(h) if (m >> j) & 1], dtype=np.int64)


def popcount(masks: np.ndarray) -> np.ndarray:
    masks = np.asarray(masks, dtype=np.int64)
    c = np.zeros(masks.shape, dtype=np.int64)
    x = masks.copy()
    while np.any(x):
        c += x & 1
        x >>= 1
    return c


# --------------------------------------------------------------------------------------------
# Vertex enumeration
# --------------------------------------------------------------------------------------------

def _refine(A: np.ndarray, S: np.ndarray, vS: np.ndarray):
    """Refine a candidate vertex: least squares on its whole tight set, with rank/feasibility checks.

    Returns the refined dense vertex (length h) or None if it is not a genuine vertex.
    """
    h = A.shape[0]
    full = np.zeros(h)
    full[S] = vS
    Av = A @ full
    tight = np.flatnonzero(np.abs(Av - 1.0) <= TOL_TIGHT)
    K = len(S)
    if len(tight) < K:
        return None
    M = A[np.ix_(tight, S)]
    sv = np.linalg.svd(M, compute_uv=False)
    if sv.min() <= TOL_RANK * sv.max():
        return None
    v2, *_ = np.linalg.lstsq(M, np.ones(len(tight)), rcond=None)
    if np.any(v2 <= TOL_POS * v2.max()):
        return None
    full = np.zeros(h)
    full[S] = v2
    Av = A @ full
    if Av.min() < 1.0 - TOL_FINAL:
        return None
    if np.abs(Av[tight] - 1.0).max() > TOL_FINAL:
        return None
    return full


def enumerate_vertices_raw(p: int, canonical_only: bool = True, chunk_elems: int = 4_000_000,
                           verbose: bool = False, torch_threads: int | None = None):
    """Enumerate the vertices of P = {v >= 0, A v >= 1} whose support is canonical (or all supports).

    Returns (V, stats) with V an (n, h) array of refined, deduplicated vertices (NOT closed under
    dilation when canonical_only=True) and a stats dict with counts and timings per support size.
    """
    import torch

    if torch_threads is not None:
        torch.set_num_threads(torch_threads)
    h = half(p)
    A = coverage_matrix(p)
    supports = canonical_supports(p) if canonical_only else np.arange(1, 1 << h, dtype=np.int64)
    pc = popcount(supports)
    stats = {"p": p, "h": h, "n_supports": int(len(supports)), "per_K": {}}
    found = {}
    t_all = time.time()
    for K in range(1, h + 1):
        tK = time.time()
        SK = supports[pc == K]
        if len(SK) == 0:
            continue
        S_idx = np.stack([_mask_to_idx(int(m), h) for m in SK])  # (nS, K)
        T_all = np.array(list(itertools.combinations(range(h), K)), dtype=np.int64)  # (nT, K)
        nT = len(T_all)
        A_T = A[T_all]  # (nT, K, h)
        per_S = nT * K * K
        bs = max(1, chunk_elems // max(per_S, 1))
        n_solved = 0
        n_cand = 0
        for c0 in range(0, len(S_idx), bs):
            Sc = S_idx[c0:c0 + bs]  # (nSc, K)
            nSc = len(Sc)
            M = A_T[:, :, Sc]  # (nT, K, nSc, K)
            M = np.ascontiguousarray(M.transpose(2, 0, 1, 3)).reshape(nSc * nT, K, K)
            Mt = torch.from_numpy(M)
            rhs = torch.ones(nSc * nT, K, 1, dtype=torch.float64)
            sol, info = torch.linalg.solve_ex(Mt, rhs)
            v = sol[..., 0].numpy()
            ok = (info.numpy() == 0) & np.all(np.isfinite(v), axis=1)
            vmax = np.where(ok, np.abs(v).max(1), 1.0)
            ok &= np.all(v > TOL_POS * vmax[:, None], axis=1)
            n_solved += nSc * nT
            if not ok.any():
                continue
            idx = np.flatnonzero(ok)
            s_of = idx // nT
            vv = v[idx]
            # feasibility: A[:, S] @ v_S >= 1 - TOL_FEAS
            AS = A[:, Sc[s_of]]  # (h, n, K)
            cover = np.einsum("hnk,nk->nh", AS, vv)
            feas = cover.min(1) >= 1.0 - TOL_FEAS
            if not feas.any():
                continue
            for i in np.flatnonzero(feas):
                S = Sc[s_of[i]]
                key = (tuple(S.tolist()), tuple(np.round(vv[i], 7).tolist()))
                if key not in found:
                    found[key] = (S, vv[i].copy())
                    n_cand += 1
        stats["per_K"][K] = {"n_supports": int(len(SK)), "n_T": int(nT), "n_solved": int(n_solved),
                             "n_candidates": int(n_cand), "seconds": time.time() - tK}
        if verbose:
            print(f"p={p} K={K}: supports={len(SK)} T={nT} solves={n_solved} cand={n_cand} "
                  f"t={time.time() - tK:.1f}s", flush=True)
    # refine + deduplicate
    tR = time.time()
    uniq = {}
    n_rejected = 0
    for S, vS in found.values():
        full = _refine(A, S, vS)
        if full is None:
            n_rejected += 1
            continue
        key = tuple(np.round(full, ROUND_DIGITS).tolist())
        uniq.setdefault(key, full)
    V = np.array(list(uniq.values())) if uniq else np.zeros((0, h))
    stats["n_candidates_total"] = len(found)
    stats["n_rejected_in_refinement"] = n_rejected
    stats["n_vertices_raw"] = int(len(V))
    stats["seconds_refine"] = time.time() - tR
    stats["seconds_total"] = time.time() - t_all
    return V, stats


def orbit_canonical(V: np.ndarray, p: int):
    """Canonical representative of each vertex's dilation orbit, and the orbit size.

    Canonical = lexicographically smallest (support bitmask, rounded values) among the h dilates.
    Returns (reps, orbit_sizes, rep_index_of_each_input_row).
    """
    h = half(p)
    V = np.asarray(V, dtype=float)
    keys_all = []
    for u in units(p):
        D = dilate(V, int(u), p)
        masks = np.array([mask_of(d) for d in D], dtype=np.int64)
        keys_all.append((masks, np.round(D, ROUND_DIGITS), D))
    n = len(V)
    reps = np.zeros_like(V)
    rep_key = []
    for i in range(n):
        best = None
        for masks, R, D in keys_all:
            k = (int(masks[i]), tuple(R[i].tolist()))
            if best is None or k < best[0]:
                best = (k, D[i])
        rep_key.append(best[0])
        reps[i] = best[1]
    uniq = {}
    order = []
    rep_of = np.zeros(n, dtype=np.int64)
    for i, k in enumerate(rep_key):
        if k not in uniq:
            uniq[k] = len(order)
            order.append(i)
        rep_of[i] = uniq[k]
    R = reps[order]
    # orbit size = number of distinct dilates of the representative
    sizes = np.array([len({tuple(np.round(dilate(r, int(u), p), ROUND_DIGITS).tolist()) for u in units(p)})
                      for r in R], dtype=np.int64)
    return R, sizes, rep_of


def expand_orbits(reps: np.ndarray, p: int) -> np.ndarray:
    """All distinct dilates of the given representatives (the full vertex list)."""
    out = {}
    for r in reps:
        for u in units(p):
            d = dilate(r, int(u), p)
            out.setdefault(tuple(np.round(d, ROUND_DIGITS).tolist()), d)
    return np.array(list(out.values())) if out else np.zeros((0, half(p)))


# --------------------------------------------------------------------------------------------
# Cache
# --------------------------------------------------------------------------------------------

def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def cache_paths(p: int):
    return (os.path.join(DATA_DIR, f"vertices_p{p}.npz"), os.path.join(DATA_DIR, f"vertices_p{p}.json"))


def build_cache(p: int, verbose: bool = False, torch_threads: int | None = 1) -> dict:
    """Enumerate the vertices at p and write core/data/vertices_p{p}.npz + .json (sha256 sidecar)."""
    assert is_prime(p) and p > 2, p
    os.makedirs(DATA_DIR, exist_ok=True)
    t0 = time.time()
    V, stats = enumerate_vertices_raw(p, canonical_only=True, verbose=verbose, torch_threads=torch_threads)
    t1 = time.time()
    reps, sizes, _ = orbit_canonical(V, p)
    stats["seconds_orbits"] = time.time() - t1
    npz, js = cache_paths(p)
    tmp = npz + ".tmp.npz"
    np.savez(tmp, reps=reps, orbit_sizes=sizes)
    os.replace(tmp, npz)
    meta = {
        "p": p,
        "h": half(p),
        "enum_version": ENUM_VERSION,
        "tolerances": {"TOL_POS": TOL_POS, "TOL_FEAS": TOL_FEAS, "TOL_TIGHT": TOL_TIGHT,
                       "TOL_FINAL": TOL_FINAL, "TOL_RANK": TOL_RANK, "ROUND_DIGITS": ROUND_DIGITS},
        "n_orbit_reps": int(len(reps)),
        "n_vertices": int(sizes.sum()),
        "npz_sha256": _sha256_file(npz),
        "programme_py_sha256_at_import": SELF_SHA256,
        "stats": stats,
        "seconds_wall": time.time() - t0,
        "created": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
    }
    with open(js, "w") as f:
        json.dump(meta, f, indent=1, default=str)
    return meta


_CACHE: dict = {}


def load_vertices(p: int, build_if_missing: bool = False):
    """Return (reps, orbit_sizes, meta) for p, verifying the sha256 of the cached npz."""
    if p in _CACHE:
        return _CACHE[p]
    npz, js = cache_paths(p)
    if not os.path.exists(npz) or not os.path.exists(js):
        if not build_if_missing:
            raise FileNotFoundError(f"no vertex cache for p={p}; run `python -m core.programme build {p}`")
        build_cache(p)
    with open(js) as f:
        meta = json.load(f)
    sha = _sha256_file(npz)
    if sha != meta["npz_sha256"]:
        raise RuntimeError(f"checksum mismatch for {npz}: {sha} != {meta['npz_sha256']}")
    d = np.load(npz)
    out = (d["reps"], d["orbit_sizes"], meta)
    _CACHE[p] = out
    return out


def cached_primes() -> list[int]:
    if not os.path.isdir(DATA_DIR):
        return []
    ps = []
    for fn in os.listdir(DATA_DIR):
        if fn.startswith("vertices_p") and fn.endswith(".json"):
            ps.append(int(fn[len("vertices_p"):-len(".json")]))
    return sorted(ps)


# --------------------------------------------------------------------------------------------
# Solving the programme
# --------------------------------------------------------------------------------------------

@dataclass
class ProgrammeSolution:
    """Solution of eq:programme at (p, s).

    gamma        : Gamma_s(p)
    y_all        : (n_opt, h) every optimal vertex (within rel_tol), normalised to sum y^{1/s} = 1
    kstar_all    : support size of each optimal vertex
    kstar        : the common support size if all agree, else None (see kstar_all)
    tight_all    : for each optimal vertex, the tight offsets delta in 1..h (coverage within tight_tol of gamma)
    orbit_reps   : canonical (dilation) representatives of the optimal vertices, normalised
    """
    p: int
    s: float
    gamma: float
    y_all: np.ndarray
    kstar_all: np.ndarray
    kstar: int | None
    tight_all: list
    orbit_reps: np.ndarray
    rel_tol: float
    extra: dict = field(default_factory=dict)

    @property
    def y(self) -> np.ndarray:
        """The canonical representative of the first optimal orbit (normalised)."""
        return self.orbit_reps[0]


def tight_offsets(y: np.ndarray, p: int, tol: float = 1e-9) -> np.ndarray:
    """Offsets delta in 1..h with sum_j y_j cov_j(delta) within tol (relative) of the minimum."""
    A = coverage_matrix(p)
    c = A @ np.asarray(y)
    return np.flatnonzero(c <= c.min() * (1 + tol) + tol * 1e-3) + 1


def solve(p: int, s: float, rel_tol: float = 1e-9, tight_tol: float = 1e-9) -> ProgrammeSolution:
    """Exact Gamma_s(p) and all optimal vertices, from the cached vertex list of p."""
    if s < 1.0:
        raise ValueError("s must be >= 1 (the vertex argument needs a concave cost, r = 1/s <= 1)")
    reps, sizes, meta = load_vertices(p)
    c = cost(reps, s)
    cmin = c.min()
    gamma = cmin ** (-s)
    opt = np.flatnonzero(c <= cmin * (1 + rel_tol))
    opt_reps = reps[opt] / cmin ** s
    y_all = expand_orbits(opt_reps, p)
    kst = (y_all > 0).sum(1)
    tights = [tight_offsets(y, p, tight_tol) for y in y_all]
    order = np.argsort(c[opt])
    return ProgrammeSolution(p=p, s=s, gamma=float(gamma), y_all=y_all, kstar_all=kst,
                             kstar=int(kst[0]) if np.all(kst == kst[0]) else None, tight_all=tights,
                             orbit_reps=opt_reps[order], rel_tol=rel_tol,
                             extra={"n_opt_orbits": int(len(opt)), "n_vertices": int(sizes.sum())})


def gamma(p: int, s: float) -> float:
    """Gamma_s(p) (exact, from the vertex cache)."""
    return solve(p, s).gamma


def kstar(p: int, s: float):
    """K* = |spt y*|: an int if all optimal vertices agree, else the sorted list of distinct values."""
    sol = solve(p, s)
    return sol.kstar if sol.kstar is not None else sorted(set(sol.kstar_all.tolist()))


def _support_subset_mask(S) -> int:
    m = 0
    for j in S:
        m |= 1 << (int(j) - 1)
    return m


def gamma_restricted(p: int, S, s: float, use_cache: bool = True, max_solves: int = 200_000_000):
    """Gamma_s(S): the programme with support restricted to S (a set of frequencies in 1..h).

    Vertices of {v >= 0, supp v in S, A v >= 1} are exactly the vertices of P with support in S,
    so with a cache this is a minimum over cached vertices (all dilates checked). Without a cache
    (or use_cache=False) it enumerates subsets S' of S and tight sets T directly.
    Returns (Gamma_s(S), y_opt normalised, support of y_opt as a sorted list).
    """
    h = half(p)
    S = sorted(set(int(j) for j in S))
    assert all(1 <= j <= h for j in S), S
    smask = _support_subset_mask(S)
    if use_cache and p in cached_primes():
        reps, sizes, meta = load_vertices(p)
        best = None
        for u in units(p):
            D = dilate(reps, int(u), p)
            masks = np.array([mask_of(d) for d in D], dtype=np.int64)
            ok = (masks & ~smask) == 0
            if not ok.any():
                continue
            c = cost(D[ok], s)
            i = int(np.argmin(c))
            if best is None or c[i] < best[0]:
                best = (c[i], D[ok][i])
        cmin, v = best
    else:
        A = coverage_matrix(p)
        Sidx = np.array(S) - 1
        n_total = sum(math.comb(len(S), K) * math.comb(h, K) for K in range(1, len(S) + 1))
        if n_total > max_solves:
            raise ValueError(f"direct enumeration needs {n_total} solves > max_solves")
        best = None
        for K in range(1, len(S) + 1):
            T_all = np.array(list(itertools.combinations(range(h), K)), dtype=np.int64)
            for Ssub in itertools.combinations(Sidx, K):
                Ssub = np.array(Ssub)
                M = A[T_all][:, :, Ssub]
                det_ok = np.abs(np.linalg.det(M)) > 1e-13
                if not det_ok.any():
                    continue
                vs = np.linalg.solve(M[det_ok], np.ones((det_ok.sum(), K, 1)))[..., 0]
                pos = np.all(vs > TOL_POS * np.abs(vs).max(1, keepdims=True), axis=1)
                if not pos.any():
                    continue
                vs = vs[pos]
                cover = vs @ A[:, Ssub].T
                feas = cover.min(1) >= 1 - TOL_FEAS
                if not feas.any():
                    continue
                vs = vs[feas]
                c = (vs ** (1.0 / s)).sum(1)
                i = int(np.argmin(c))
                if best is None or c[i] < best[0]:
                    full = np.zeros(h)
                    full[Ssub] = vs[i]
                    best = (c[i], full)
        cmin, v = best
    y = v / cmin ** s
    return float(cmin ** (-s)), y, [int(j) + 1 for j in np.flatnonzero(y > 0)]


# --------------------------------------------------------------------------------------------
# Switch points in s
# --------------------------------------------------------------------------------------------

def cost_classes(reps: np.ndarray):
    """Group vertices whose sorted non-zero values coincide (identical cost for every s).

    Returns (class_reps, class_members) where class_members[i] lists the input indices.
    """
    keys = {}
    members = []
    creps = []
    for i, r in enumerate(reps):
        k = tuple(sorted(np.round(r[r > 0], ROUND_DIGITS).tolist()))
        if k not in keys:
            keys[k] = len(creps)
            creps.append(r)
            members.append([i])
        else:
            members[keys[k]].append(i)
    return np.array(creps), members


def switch_points(p: int, s_lo: float = 1.0, s_hi: float = 4.0, n_grid: int = 4001,
                  min_width: float = 1e-12):
    """Exact switch points of the optimal vertex class on s in [s_lo, s_hi].

    Method (in r = 1/s, cost_v(r) = sum_j v_j^r, a smooth function):
      1. evaluate every cost class on a uniform r-grid;
      2. certify each grid interval: class c is excluded from interval I if a Lipschitz lower bound
         of cost_c on I exceeds an upper bound of the grid-winner's cost on I
         (|d/dr cost_c| <= max_j |log v_j| * (cost_c(r0) + cost_c(r1)) on I);
      3. intervals with more than one surviving class are bisected recursively with the survivors
         only, down to width min_width; the final crossing is located by brentq on the cost
         difference of the two neighbouring winners.
    Returns a dict with 'plateaus' (list of {s_from, s_to, rep (normalised at s_mid), support,
    ratios}) and 'switches' (list of s values), both ordered by increasing s.
    """
    from scipy.optimize import brentq

    reps, sizes, meta = load_vertices(p)
    creps, members = cost_classes(reps)
    n = len(creps)
    L = np.where(creps > 0, np.log(np.where(creps > 0, creps, 1.0)), -np.inf)
    Lmax = np.max(np.where(np.isfinite(L), np.abs(L), 0.0), axis=1)

    def costs(idx, r):
        r = np.atleast_1d(r)
        LL = L[idx][:, None, :] * r[None, :, None]
        return np.where(np.isfinite(LL), np.exp(np.where(np.isfinite(LL), LL, 0.0)), 0.0).sum(-1)

    r_lo, r_hi = 1.0 / s_hi, 1.0 / s_lo
    grid = np.linspace(r_lo, r_hi, n_grid)
    all_idx = np.arange(n)
    # pass 1: envelope on the grid
    C = np.empty((n, n_grid))
    ch = max(1, 2_000_000 // n_grid)
    for c0 in range(0, n, ch):
        C[c0:c0 + ch] = costs(all_idx[c0:c0 + ch], grid)
    win = np.argmin(C, axis=0)
    # upper bound of the envelope on each interval: min over the two endpoint winners of their max
    D = Lmax[:, None] * (C[:, :-1] + C[:, 1:])  # Lipschitz constants per (class, interval)
    dr = np.diff(grid)
    upper_w0 = (C[win[:-1], np.arange(n_grid - 1)] + C[win[:-1], np.arange(1, n_grid)]
                + D[win[:-1], np.arange(n_grid - 1)] * dr) / 2
    upper_w1 = (C[win[1:], np.arange(n_grid - 1)] + C[win[1:], np.arange(1, n_grid)]
                + D[win[1:], np.arange(n_grid - 1)] * dr) / 2
    upper = np.minimum(upper_w0, upper_w1)
    lower = (C[:, :-1] + C[:, 1:] - D * dr) / 2  # (n, G-1)
    surv = lower <= upper[None, :]  # candidate classes per interval

    segments = []  # (r0, r1, class) with a single certified winner

    def resolve(r0, r1, cand):
        if len(cand) == 1:
            segments.append((r0, r1, int(cand[0])))
            return
        c01 = costs(cand, np.array([r0, r1]))
        w = cand[np.argmin(c01, axis=0)]
        if r1 - r0 < min_width:
            segments.append((r0, r1, int(w[0])))
            if w[1] != w[0]:
                segments.append((r1, r1, int(w[1])))
            return
        Dc = Lmax[cand] * (c01[:, 0] + c01[:, 1])
        lo = (c01[:, 0] + c01[:, 1] - Dc * (r1 - r0)) / 2
        up = np.inf
        for ww in set(w.tolist()):
            k = int(np.flatnonzero(cand == ww)[0])
            up = min(up, (c01[k, 0] + c01[k, 1] + Dc[k] * (r1 - r0)) / 2)
        keep = cand[lo <= up]
        if len(keep) == 1:
            segments.append((r0, r1, int(keep[0])))
            return
        rm = (r0 + r1) / 2
        resolve(r0, rm, keep)
        resolve(rm, r1, keep)

    for k in range(n_grid - 1):
        cand = np.flatnonzero(surv[:, k])
        if len(cand) == 1:
            segments.append((grid[k], grid[k + 1], int(cand[0])))
        else:
            resolve(grid[k], grid[k + 1], cand)

    # merge consecutive segments with the same winner (segments are in increasing r order)
    segments.sort(key=lambda t: (t[0], t[1]))
    merged = []
    for r0, r1, c in segments:
        if merged and merged[-1][2] == c:
            merged[-1] = (merged[-1][0], max(merged[-1][1], r1), c)
        else:
            merged.append((r0, r1, c))
    # locate crossings exactly
    bounds = []
    for (a0, a1, ca), (b0, b1, cb) in zip(merged[:-1], merged[1:]):
        f = lambda r: costs(np.array([ca]), r)[0, 0] - costs(np.array([cb]), r)[0, 0]
        lo_, hi_ = a0, b1
        try:
            rstar = brentq(f, lo_, hi_, xtol=1e-15, rtol=1e-15, maxiter=500) if f(lo_) * f(hi_) < 0 else a1
        except ValueError:
            rstar = a1
        bounds.append(rstar)
    edges = [r_lo] + bounds + [r_hi]
    plateaus = []
    for i, (r0, r1, c) in enumerate(merged):
        ra, rb = edges[i], edges[i + 1]
        rmid = (ra + rb) / 2
        smid = 1.0 / rmid
        v = creps[c]
        cm = cost(v, smid)
        # every orbit rep in this cost class
        mem = members[c]
        plateaus.append({
            "s_from": 1.0 / rb, "s_to": 1.0 / ra,
            "rep": v / cm ** smid,
            "support": [int(j) + 1 for j in np.flatnonzero(v > 0)],
            "K": int((v > 0).sum()),
            "ratios": (v / v.max()),
            "class_index": int(c),
            "n_orbits_in_class": len(mem),
        })
    plateaus = plateaus[::-1]  # increasing s
    switches = sorted(1.0 / b for b in bounds)
    return {"p": p, "s_lo": s_lo, "s_hi": s_hi, "plateaus": plateaus, "switches": switches,
            "n_cost_classes": n, "n_grid": n_grid}


# --------------------------------------------------------------------------------------------
# Alignment helpers used by metrics
# --------------------------------------------------------------------------------------------

def canonicalize(y: np.ndarray, p: int, tol: float = 0.0):
    """Canonical dilate of y: the dilate with the smallest (support bitmask, rounded values) key.

    Meaningful for exactly sparse vectors (vertices, constructed networks). Returns (y_canon, u)
    with y_canon = dilate(y, u, p).
    """
    best = None
    for u in units(p):
        d = dilate(y, int(u), p)
        k = (mask_of(d, tol), tuple(np.round(d, ROUND_DIGITS).tolist()))
        if best is None or k < best[0]:
            best = (k, d, int(u))
    return best[1], best[2]


def all_dilates(y: np.ndarray, p: int) -> np.ndarray:
    """(h, h) array whose row u-1 is dilate(y, u, p)."""
    return np.stack([dilate(y, int(u), p) for u in units(p)])


# --------------------------------------------------------------------------------------------
# Beyond the exact cache: certified K cap, capped exact enumeration, best-found solver
# --------------------------------------------------------------------------------------------

def kcap(m_lb: float, s: float) -> dict:
    """Certified upper bounds on the support size of EVERY maximiser, given m_lb <= Gamma_s(p).

    Deletion argument (CONSEQUENCES §4, astra): removing frequency j from an optimum y (margin m,
    budget 1) lowers every coverage by at most 2 y_j, and rescaling the rest to budget 1 multiplies by
    (1 - w_j)^{-s} (w_j = y_j^{1/s}); optimality gives 2 w_j^s >= m (1 - (1 - w_j)^s)   (*).
      - 'paper': (*) with 1 - (1-w)^s >= w gives w_j >= (m/2)^{1/(s-1)}, so K <= (2/m)^{1/(s-1)}.
      - 'sharp': w_j >= w0, the first positive root of phi(w) = 2 w^s - m (1 - (1-w)^s), so K <= 1/w0.
    Both are monotone in m, so any lower bound m_lb <= Gamma_s(p) gives valid caps. None for s = 1.
    """
    from scipy.optimize import brentq

    if s <= 1.0 + 1e-12:
        return {"paper": None, "sharp": None, "w0": None}
    paper = (2.0 / m_lb) ** (1.0 / (s - 1.0))
    phi = lambda w: 2.0 * w ** s - m_lb * (1.0 - (1.0 - w) ** s)
    # phi < 0 just above 0 (phi'(0) = -m s) and phi(1) = 2 - m > 0; scan for the first sign change
    ws = np.logspace(-12, 0, 4001)
    vals = phi(ws)
    k = int(np.argmax(vals >= 0))
    w0 = brentq(phi, ws[k - 1], ws[k], xtol=1e-16) if k > 0 else ws[0]
    # guard: the certified statement is w_j >= w0 only if phi < 0 on (0, w0); the scan checks it
    assert np.all(vals[:k] < 0)
    return {"paper": paper, "paper_int": int(math.floor(paper + 1e-9)), "sharp": 1.0 / w0,
            "sharp_int": int(math.floor(1.0 / w0 + 1e-9)), "w0": w0}


def canonical_supports_upto(p: int, kmax: int):
    """Dilation-canonical supports of size <= kmax (as sorted 0-based index tuples), for any p.

    Every orbit contains a support that includes frequency 1 (dilate by j^{-1}); among those we keep
    the lexicographically smallest sorted tuple of the orbit.
    """
    h = half(p)
    perms = [dilation_perm(p, int(u)) for u in units(p)]
    out = {}
    for K in range(1, kmax + 1):
        res = []
        for rest in itertools.combinations(range(1, h), K - 1):
            S = (0,) + rest
            best = min(tuple(sorted(int(pm[j]) for j in S)) for pm in perms)
            if best == S:
                res.append(S)
        out[K] = res
    return out


def capped_enumeration_cost(p: int, kmax: int) -> int:
    """Approximate number of K x K solves needed by exact_capped(p, kmax)."""
    h = half(p)
    return int(sum(math.ceil(math.comb(h, K) / h) * math.comb(h, K) for K in range(1, kmax + 1)))


def exact_capped(p: int, kmax: int, chunk_elems: int = 4_000_000, verbose: bool = False):
    """Every vertex of P with support size <= kmax (dilation-closed), by exhaustive (S, T) solves.

    Exact for Gamma_s(p) whenever a certified cap (kcap) guarantees |spt y*| <= kmax.
    Returns (reps, orbit_sizes).
    """
    import torch

    h = half(p)
    A = coverage_matrix(p)
    sup = canonical_supports_upto(p, kmax)
    found = {}
    for K in range(1, kmax + 1):
        S_idx = np.array(sup[K], dtype=np.int64).reshape(-1, K)
        if len(S_idx) == 0:
            continue
        T_all = np.array(list(itertools.combinations(range(h), K)), dtype=np.int64)
        nT = len(T_all)
        A_T = A[T_all]
        bs = max(1, chunk_elems // (nT * K * K))
        t0 = time.time()
        for c0 in range(0, len(S_idx), bs):
            Sc = S_idx[c0:c0 + bs]
            nSc = len(Sc)
            Mx = np.ascontiguousarray(A_T[:, :, Sc].transpose(2, 0, 1, 3)).reshape(nSc * nT, K, K)
            sol, info = torch.linalg.solve_ex(torch.from_numpy(Mx), torch.ones(nSc * nT, K, 1, dtype=torch.float64))
            v = sol[..., 0].numpy()
            ok = (info.numpy() == 0) & np.all(np.isfinite(v), axis=1)
            vmax = np.where(ok, np.abs(v).max(1), 1.0)
            ok &= np.all(v > TOL_POS * vmax[:, None], axis=1)
            if not ok.any():
                continue
            idx = np.flatnonzero(ok)
            s_of = idx // nT
            vv = v[idx]
            cover = np.einsum("hnk,nk->nh", A[:, Sc[s_of]], vv)
            for i in np.flatnonzero(cover.min(1) >= 1.0 - TOL_FEAS):
                S = Sc[s_of[i]]
                key = (tuple(S.tolist()), tuple(np.round(vv[i], 7).tolist()))
                found.setdefault(key, (S, vv[i].copy()))
        if verbose:
            print(f"capped p={p} K={K}: supports={len(S_idx)} T={nT} t={time.time() - t0:.1f}s", flush=True)
    uniq = {}
    for S, vS in found.values():
        full = _refine(A, S, vS)
        if full is not None:
            uniq.setdefault(tuple(np.round(full, ROUND_DIGITS).tolist()), full)
    V = np.array(list(uniq.values())) if uniq else np.zeros((0, h))
    reps, sizes, _ = orbit_canonical(V, p)
    return reps, sizes


def _lp_min(A: np.ndarray, c: np.ndarray, ub: np.ndarray | None = None):
    """min c.v s.t. A v >= 1, 0 <= v <= ub (HiGHS dual simplex: returns a basic solution)."""
    from scipy.optimize import linprog

    h = A.shape[1]
    bounds = [(0.0, None if ub is None or np.isinf(ub[j]) else float(ub[j])) for j in range(h)]
    res = linprog(c, A_ub=-A, b_ub=-np.ones(A.shape[0]), bounds=bounds, method="highs-ds")
    if res.status != 0:
        return None
    return np.clip(res.x, 0.0, None)


def mm_descent(A: np.ndarray, v0: np.ndarray, r: float, max_iter: int = 200):
    """Majorise-minimise descent of the concave cost sum v^r over P = {v >= 0, A v >= 1}.

    Each step minimises the tangent (linear) upper bound of the cost at the current point by an LP,
    so the cost never increases; frequencies at exactly 0 stay at 0 (infinite slope). Ends at a basic
    solution (a vertex of P restricted to the current support). Returns (v, cost).
    """
    v = np.asarray(v0, dtype=float)
    cur = float((v[v > 0] ** r).sum())
    for _ in range(max_iter):
        ub = np.where(v > 0, np.inf, 0.0)
        c = np.where(v > 0, r * np.where(v > 0, v, 1.0) ** (r - 1.0), 1.0)
        vn = _lp_min(A, c, ub)
        if vn is None:
            break
        vn[vn < 1e-14 * max(vn.max(), 1e-300)] = 0.0
        new = float((vn[vn > 0] ** r).sum())
        if new >= cur * (1 - 1e-13):
            if new <= cur:
                v, cur = vn, new
            break
        v, cur = vn, new
    return v, cur


def best_found(p: int, s: float, n_starts: int = 400, seed: int = 0, n_swap_rounds: int = 2,
               top_keep: int = 6, verbose: bool = False) -> dict:
    """Best-found (NOT certified) solution of eq:programme at any p, by multistart MM + swap moves.

    1. random strictly positive feasible starts (log-normal coordinates of several spreads, and
       random-subset starts), each descended by mm_descent to a vertex;
    2. for the best `top_keep` distinct vertices (up to dilation), swap moves: restart from the vertex
       plus a positive bump on each unused frequency (lets one frequency enter), repeated
       n_swap_rounds times;
    3. every local minimum is refined exactly (tight-set least squares, rank check) and canonicalised
       modulo dilation.
    Returns {'gamma', 'y' (canonical, budget 1), 'support', 'K', 'n_local_minima', 'distinct' (list of
    (gamma, K) of the best distinct local minima), 'kcap', 'label': 'best-found (not certified)'}.
    """
    rng = np.random.default_rng(seed)
    h = half(p)
    A = coverage_matrix(p)
    r = 1.0 / s
    found = {}

    def add(v):
        if v is None or not np.any(v > 0):
            return
        S = np.flatnonzero(v > 0)
        full = _refine(A, S, v[S])
        if full is None:
            full = v / (A @ v).min()
        c, _u = canonicalize(full, p)
        key = tuple(np.round(c, 8).tolist())
        if key not in found:
            found[key] = c

    def feasible_scale(v):
        return v / (A @ v).min()

    t0 = time.time()
    for k in range(n_starts):
        kind = k % 3
        if kind == 0:
            v0 = np.exp(rng.normal(0, rng.choice([0.3, 1.0, 2.0, 3.0]), h))
        elif kind == 1:
            size = int(rng.integers(1, min(h, 12) + 1))
            v0 = np.full(h, 1e-3)
            v0[rng.choice(h, size, replace=False)] = 1.0 + rng.random(size)
        else:
            v0 = rng.random(h) ** rng.choice([1.0, 3.0, 8.0]) + 1e-6
        v, _ = mm_descent(A, feasible_scale(v0), r)
        add(v)
    for _round in range(n_swap_rounds):
        items = sorted(found.values(), key=lambda v: cost(v, s))[:top_keep]
        for v in items:
            act = np.flatnonzero(v > 0)
            for j in range(h):
                if v[j] > 0:
                    continue
                # add move: let frequency j enter
                for bump in (0.3, 1.0):
                    v0 = v + 1e-4 * v.max()
                    v0[j] += bump * v.max()
                    vv, _ = mm_descent(A, feasible_scale(v0), r)
                    add(vv)
                # swap move: frequency i leaves (set to exactly 0, so it cannot return), j enters
                for i in act:
                    v0 = v + 1e-4 * v.max()
                    v0[i] = 0.0
                    v0[j] += v.max()
                    vv, _ = mm_descent(A, feasible_scale(v0), r)
                    add(vv)
    allv = np.array(list(found.values()))
    cs = cost(allv, s)
    order = np.argsort(cs)
    best = allv[order[0]]
    cmin = cs[order[0]]
    y = best / cmin ** s
    gam = float(cmin ** (-s))
    distinct = [(float(cs[i] ** (-s)), int((allv[i] > 0).sum())) for i in order[:10]]
    out = {"p": p, "s": s, "gamma": gam, "y": y, "support": [int(j) + 1 for j in np.flatnonzero(y > 0)],
           "K": int((y > 0).sum()), "n_local_minima": int(len(allv)), "distinct": distinct,
           "kcap": kcap(gam, s), "label": "best-found (not certified)", "seconds": time.time() - t0,
           "n_starts": n_starts, "seed": seed}
    if verbose:
        print(f"best_found p={p} s={s}: gamma={gam:.10f} K={out['K']} minima={len(allv)} "
              f"t={out['seconds']:.1f}s", flush=True)
    return out


def solve_any(p: int, s: float, allow_capped: bool = True, capped_budget: int = 300_000_000,
              best_found_kwargs: dict | None = None) -> dict:
    """Gamma_s(p) with an explicit certification label.

    'exact (vertex cache)'            : p has a full vertex cache.
    'exact (certified K cap)'         : best-found gives m_lb, kcap(m_lb, s) gives K_max, and all
                                        vertices with support <= K_max are enumerated.
    'best-found (not certified)'      : otherwise.
    """
    if p in cached_primes():
        sol = solve(p, s)
        return {"p": p, "s": s, "gamma": sol.gamma, "y": sol.y, "y_all": sol.y_all, "K": sol.kstar,
                "K_all": sorted(set(sol.kstar_all.tolist())), "label": "exact (vertex cache)"}
    bf = best_found(p, s, **(best_found_kwargs or {}))
    cap = bf["kcap"]
    if allow_capped and cap["sharp_int"] is not None:
        kmax = cap["sharp_int"]
        if capped_enumeration_cost(p, kmax) <= capped_budget:
            reps, sizes = exact_capped(p, kmax)
            c = cost(reps, s)
            cmin = c.min()
            opt = np.flatnonzero(c <= cmin * (1 + 1e-9))
            y_all = expand_orbits(reps[opt] / cmin ** s, p)
            kk = sorted(set((y_all > 0).sum(1).tolist()))
            return {"p": p, "s": s, "gamma": float(cmin ** (-s)), "y": reps[opt[0]] / cmin ** s, "y_all": y_all,
                    "K": kk[0] if len(kk) == 1 else None, "K_all": kk, "label": "exact (certified K cap)",
                    "kcap": cap, "best_found_gamma": bf["gamma"]}
    return {"p": p, "s": s, "gamma": bf["gamma"], "y": bf["y"], "y_all": all_dilates(bf["y"], p), "K": bf["K"],
            "K_all": [bf["K"]], "label": bf["label"], "kcap": cap, "best_found": bf}


# --------------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    cmd = sys.argv[1]
    if cmd == "build":
        for p in map(int, sys.argv[2:]):
            meta = build_cache(p, verbose=True, torch_threads=1)
            print(json.dumps({k: meta[k] for k in ("p", "n_orbit_reps", "n_vertices", "seconds_wall")}),
                  flush=True)
    elif cmd == "show":
        p = int(sys.argv[2])
        for s in map(float, sys.argv[3:]):
            sol = solve(p, s)
            print(p, s, sol.gamma, sol.kstar, sol.kstar_all[:5], np.round(sol.y, 4))
