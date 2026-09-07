"""Score the p=1009 relu-4 cohort trained by nudial/p1009.py.

Per run, from final.safetensors (chunked over the a-rows of the p^2 grid, f64):
  five committee counts, generalised from nudial/count_definitions.py to any p
    K_gap    committee_from_coeffs on the clean cos coefficients (log-gap cut)
    K_thr10  #{k : amp_k >= 0.10 max amp}      amp_k = sqrt(cc_k^2 + ss_k^2)
    K_thr05  #{k : amp_k >= 0.05 max amp}
    k_eff    1 / sum_k share_k^2 over amplitude shares (as the audited code
             computes it -- amp shares, not spectra 'energy')
    K_load90 smallest K whose top-K clean truncation reaches 90% of the
             min-margin of the full clean table
  exact full-grid min margin, gamma = min margin / ||theta||_2^4, accuracies,
  grokked flag (test acc >= 0.99), top-8 relative amplitude profile.

Two things make this cheap at p=1009 where count_definitions.py's dense
(nf, p^2, p) cos/sin tables (504 x 1e6 x 1009 f64) are impossible:
  * cc_k / ss_k come from FFTs of the diagonal-collapsed logits (p1009.py),
    never from a basis matrix;
  * the clean table sum_k cc_k cos(k(a+b-c)) + ss_k sin(k(a+b-c)) is a
    function of n = (a+b-c) mod p ALONE, so it is a length-p vector, not a
    (p^2, p) table.  Its min margin over the whole grid is exactly
    f(0) - max_{n != 0} f(n), independent of (a,b).  Only (nf, p) cos/sin
    tables (4 MB) are ever built.

Theory side (plain l2, nu = 4, the paper's ladder):
  M_K(p)      = max_{|S|=K} min_{Delta != 0} sum_{k in S} (1 - cos 2pi k Delta/p)
                (kstar_table.best_subset; exhaustive while feasible, else local)
  gamma_hat(K)= M_K / K^2      equal-share normalised margin of a K committee
  r(K)        = gamma_hat(K) / max_K gamma_hat(K)
  plateau sets {K : r(K) >= 0.74} and {K : r(K) >= 0.84}
  --construct additionally builds the explicit phase-complete K-committee
  circuits of construct_gamma.py at this p (equal per-frequency amplitude,
  closed-form block balance) to get the class ceiling.

Usage
  uv run python nudial/score_p1009.py --out /tmp/p1009_scored.json
  uv run python nudial/score_p1009.py --check          # vs the audited p=113 ledger
  uv run python nudial/score_p1009.py --p 23 --widths 64 --out /tmp/smoke.json
  ... --construct        # + constructive class ceiling for K in 5..9 (slower)
"""
import argparse, collections, json, math, os, sys, time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "nudial"))
from train_nudial import Config, committee_from_coeffs, forward
from kstar_table import best_subset
from p1009 import collapse_add
from safetensors.torch import load_file

RUNS = Path(os.environ.get("RUNS_DIR", ROOT / "runs_nudial"))
torch.set_grad_enabled(False)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


# ------------------------------------------------------------- grid sweep ---
def grid_stats(pr, cfg, device, a_rows):
    """One chunked f64 pass over the p^2 grid.

    Returns T (diagonal-collapsed logits, p x p), min margin, train/test
    correctness counts per split mask, and the centred logit energy.
    """
    p = cfg.p
    T = torch.zeros(p, p, dtype=torch.float64, device=device)
    ok = torch.zeros(p * p, dtype=torch.bool, device=device)
    gmin = torch.tensor(float("inf"), dtype=torch.float64, device=device)
    sq = torch.zeros((), dtype=torch.float64, device=device)
    b_all = torch.arange(p, device=device)
    for a0 in range(0, p, a_rows):
        na = min(a_rows, p - a0)
        A = torch.arange(a0, a0 + na, device=device).repeat_interleave(p)
        B = b_all.repeat(na)
        lab = (A + B) % p
        L = forward(pr, A, B, cfg)[0]                        # (na*p, p) f64
        ok[a0 * p:(a0 + na) * p] = L.argmax(-1) == lab
        Lc = L - L.mean(-1, keepdim=True)                    # count_definitions centres
        sq += (Lc ** 2).sum()
        zy = L.gather(-1, lab[:, None])
        o = L.scatter(-1, lab[:, None], float("-inf"))
        gmin = torch.minimum(gmin, (zy.squeeze(-1) - o.max(-1).values).min())
        collapse_add(T[None], L[None], a0, p)
        del L, Lc, o
    return T, float(gmin), ok.cpu().numpy(), float(sq)


def clean_coeffs(T, p):
    """cc_k, ss_k in count_definitions.py's normalisation (2/p^3 * <L, cos/sin>).

    3D DFT at (k, k, -k) = sum_o F[k,o] e^{+2 pi i k o/p} with F = DFT_s(T);
    its real part is sum_{abc} L cos(k(a+b-c)), minus its imaginary part is
    sum_{abc} L sin(k(a+b-c)).
    """
    nf = p // 2
    Fk = torch.fft.fft(T.double(), dim=0)[1:nf + 1]                    # (nf, p)
    k = torch.arange(1, nf + 1, dtype=torch.float64, device=T.device)
    o = torch.arange(p, dtype=torch.float64, device=T.device)
    ph = torch.exp(2j * math.pi * k[:, None] * o[None, :] / p)
    z = (Fk * ph).sum(-1) * (2.0 / p ** 3)
    return z.real.cpu().numpy(), -z.imag.cpu().numpy()


def counts(cc, ss, p):
    """The five counts + the extras count_definitions.py reports."""
    nf = p // 2
    amp = np.sqrt(cc ** 2 + ss ** 2)
    order = np.argsort(amp)[::-1]
    n = np.arange(p); k = np.arange(1, nf + 1)
    C = np.cos(2 * np.pi * np.outer(k, n) / p)          # (nf, p) -- 4 MB at p=1009
    S = np.sin(2 * np.pi * np.outer(k, n) / p)
    prof_full = cc @ C + ss @ S                          # clean table as f(n)
    mm = lambda f: f[0] - f[1:].max()                    # exact grid min margin
    m_full = mm(prof_full)
    K_load = K_acc = None
    f = np.zeros(p)
    for i, j in enumerate(order):
        f = f + cc[j] * C[j] + ss[j] * S[j]
        if K_acc is None and mm(f) > 0:                  # truncation is 100% or 0% correct
            K_acc = i + 1
        if K_load is None and mm(f) >= 0.9 * m_full:
            K_load = i + 1
        if K_load and K_acc:
            break
    share = amp ** 2 / (amp ** 2).sum()
    return dict(K_gap=len(committee_from_coeffs(cc)),
                K_gap_amp=len(committee_from_coeffs(amp)),
                K_thr10=int((amp >= 0.1 * amp.max()).sum()),
                K_thr05=int((amp >= 0.05 * amp.max()).sum()),
                k_eff=float(1 / (share ** 2).sum()), K_load90=K_load, K_acc99=K_acc,
                m_clean=float(m_full), sin_frac=float((ss ** 2).sum() / (amp ** 2).sum()),
                amp_sq=float((amp ** 2).sum()),
                top_k=[int(x) + 1 for x in order[:8]],
                top8=[round(float(x), 3) for x in amp[order[:8]] / amp.max()])


def score_run(rd, device, mem_gb):
    cfgd = json.loads((rd / "config.json").read_text())
    cfg = Config(**{k: v for k, v in cfgd.items() if k in Config.__dataclass_fields__})
    pr = {k: v.double()[None].to(device) for k, v in load_file(rd / "final.safetensors").items()}
    if not all(torch.isfinite(v).all() for v in pr.values()):
        return None
    p = cfg.p
    # f64 grid pass: logits + the scattered copy + activations, per (a-row x p)
    per_row = 8 * p * (3 * p + 2 * cfg.d + 2 * cfg.m + cfg.r) * 1.35
    a_rows = max(1, min(p, int((mem_gb * 0.75e9 - 8 * p * p) / per_row)))
    T, gmin, ok, sq = grid_stats(pr, cfg, device, a_rows)
    cc, ss = clean_coeffs(T, p)
    r = counts(cc, ss, p)
    pn2 = float(sum((v ** 2).sum() for v in pr.values()))
    from train_nudial import train_test_split
    mask = train_test_split(cfg)
    r.update(cell=rd.parent.name, run=rd.name, p=p, m=cfg.m, seed=cfg.init_seed,
             pnorm2=pn2, min_margin=gmin, gamma=gmin / pn2 ** (cfg.degree / 2),
             train_acc=float(ok[mask].mean()), test_acc=float(ok[~mask].mean()),
             a_rows=a_rows)
    r["grokked"] = r["test_acc"] >= 0.99
    r["m_net_over_clean"] = gmin / r["m_clean"] if r["m_clean"] else float("nan")
    r["clean_frac"] = r.pop("amp_sq") * p ** 3 / 2 / sq
    return r


# ------------------------------------------------------------ theory side ---
def ladder(p, nu=4, kmax=16, restarts=400, reps=3, cache=None):
    """gamma_hat(K) = M_K / K^(nu/2) equal-share ladder + r(K) = gamma_hat/max.

    best_subset is exhaustive only while C(nf-1, K-1) <= 4e6 (K <= 3 at p=1009),
    so M_K for K >= 4 is a LOCAL SEARCH lower bound.  `reps` independent runs are
    taken and the best kept; the spread across reps is recorded, because near the
    top of the ladder the K-to-K gaps (~1%) are the size of the search noise.
    """
    if cache and Path(cache).exists():
        d = json.loads(Path(cache).read_text())
        if (d.get("p") == p and d.get("kmax", 0) >= kmax
                and d.get("restarts", 0) >= restarts and d.get("reps", 0) >= reps):
            return d
    rows = {}
    for K in range(1, min(kmax, p // 2) + 1):
        t = time.time()
        v = [best_subset(p, K, restarts=restarts) for _ in range(reps)]
        M, S, how = max(v, key=lambda x: x[0])
        spread = M - min(x[0] for x in v)
        rows[K] = dict(M_K=M, S=list(S), how=how, spread=spread,
                       sec=round(time.time() - t, 1))
        print(f"  M_{K}(p={p}) = {M:.4f}  ({how}, rep-spread {spread:.4f}, "
              f"{rows[K]['sec']}s)  S={S}", flush=True)
    gh = {K: rows[K]["M_K"] / K ** (nu / 2) for K in rows}
    mx = max(gh.values())
    d = dict(p=p, nu=nu, kmax=kmax, restarts=restarts, reps=reps,
             M_K={str(K): rows[K]["M_K"] for K in rows},
             M_K_spread={str(K): rows[K]["spread"] for K in rows},
             how={str(K): rows[K]["how"] for K in rows},
             S={str(K): rows[K]["S"] for K in rows},
             gamma_hat={str(K): gh[K] for K in rows},
             r={str(K): gh[K] / mx for K in rows},
             K_star=int(max(gh, key=gh.get)))
    if cache:
        Path(cache).write_text(json.dumps(d, indent=1))
    return d


# --- constructive class ceiling: construct_gamma.py's circuit, any p, chunked
def committee_net(p, S, amps, m, n_a, n_b):
    """construct_gamma.committee_net with P a parameter (layout (1,1))."""
    import itertools
    K = len(S)
    pairs = np.array([[2 * np.pi * i / n_a, 2 * np.pi * j / n_b]
                      for i, j in itertools.product(range(n_a), range(n_b))])
    n = len(pairs); assert K * n <= m, (K, n, m)
    x = np.arange(p); d = 2 * K
    E = np.zeros((d, p)); U = np.zeros((m, d)); V = np.zeros((m, d))
    W1 = np.zeros((d, m)); W = np.zeros((p, d))
    for j, k in enumerate(S):
        s = amps[j] ** 0.25
        rows = slice(j * n, (j + 1) * n); phi, psi = pairs[:, 0], pairs[:, 1]
        E[2 * j] = s * np.cos(2 * np.pi * k * x / p)
        E[2 * j + 1] = s * np.sin(2 * np.pi * k * x / p)
        U[rows, 2 * j] = s * np.cos(phi); U[rows, 2 * j + 1] = -s * np.sin(phi)
        V[rows, 2 * j] = s * np.cos(psi); V[rows, 2 * j + 1] = -s * np.sin(psi)
        W1[2 * j, rows] = s * np.cos(phi + psi) / n
        W1[2 * j + 1, rows] = -s * np.sin(phi + psi) / n
        W[:, 2 * j] = s * np.cos(2 * np.pi * k * x / p)
        W[:, 2 * j + 1] = s * np.sin(2 * np.pi * k * x / p)
    return E, U, V, W1, W


def construct_gamma_K(p, S, m, nu=4, device="cpu", a_rows=8):
    """Exact full-grid min margin of the K-committee construction / optimal balance.

    gamma = g / (B^(B/2) sqrt(prod_i c_i)) for B=4 scale groups (E; U,V; W1; W)
    -- construct_gamma.gamma_balanced, chunked over a so no (m, p^2) is built.
    """
    K = len(S); n_a = n_b = max(2, int(math.floor(math.sqrt(m // K))))
    E, U, V, W1, W = committee_net(p, S, np.ones(K), m, n_a, n_b)
    t = lambda x: torch.from_numpy(x).to(device).double()
    UE, VE, W1t, Wt = t(U @ E), t(V @ E), t(W1), t(W)
    g = torch.tensor(float("inf"), dtype=torch.float64, device=device)
    for a0 in range(0, p, a_rows):
        na = min(a_rows, p - a0)
        H = torch.relu(UE[:, a0:a0 + na, None] + VE[:, None, :])      # (m, na, p)
        L = (Wt @ (W1t @ H.reshape(m, -1))).T.reshape(na, p, p)       # (a, b, c)
        lab = (torch.arange(a0, a0 + na, device=device)[:, None]
               + torch.arange(p, device=device)[None, :]) % p
        zy = L.gather(-1, lab[..., None]).squeeze(-1)
        L.scatter_(-1, lab[..., None], float("-inf"))
        g = torch.minimum(g, (zy - L.max(-1).values).min())
        del H, L
    costs = [float((E ** 2).sum()), float((U ** 2).sum() + (V ** 2).sum()),
             float((W1 ** 2).sum()), float((W ** 2).sum())]
    B = 4
    return float(g) / (B ** (B / 2) * math.sqrt(float(np.prod(costs)))), float(g), n_a


# ------------------------------------------------------------------- main ---
def check_113(n=6, device="cpu"):
    """Validate every field against the audited p=113 ledger notes/count_definitions.json
    (which was produced by the dense (nf, p^2, p) instrument of count_definitions.py)."""
    ref = {(r["cell"], r["run"]): r for r in
           json.loads((ROOT / "notes" / "count_definitions.json").read_text())}
    tgt = [k for k in ref if k[0].startswith("relu4_m512_p113_f0.5")][:n]
    keys = ("K_gap", "K_gap_amp", "K_thr10", "K_thr05", "k_eff", "K_load90", "K_acc99",
            "sin_frac", "clean_frac", "m_net_over_clean", "top8")
    bad = 0
    for cell, run in tgt:
        r = score_run(ROOT / "runs_nudial" / "M" / cell / run, device, 8.0)
        g, diffs = ref[(cell, run)], []
        for k in keys:
            a, b = r[k], g[k]
            e = (max(abs(x - y) for x, y in zip(a, b)) if isinstance(a, list)
                 else abs(a - b) / max(abs(b), 1e-30) if isinstance(a, float) else abs(a - b))
            if e >= 2e-3:
                diffs.append(f"{k}: {a} vs {b}")
            bad += e >= 2e-3
        print(f"  {run}: {'; '.join(diffs) or f'all {len(keys)} fields match'} "
              f"| gamma {r['gamma']:.4e} test {r['test_acc']:.4f}")
    print(f"CHECK p=113 vs count_definitions.json: {'PASS' if not bad else f'{bad} MISMATCHES'}")
    return not bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p", type=int, default=1009)
    ap.add_argument("--widths", type=int, nargs="+", default=[512, 2048])
    ap.add_argument("--group", default=None, help="run group dir (default P<p>)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--check", action="store_true",
                    help="validate all fields against notes/count_definitions.json (p=113)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--mem-gb", type=float, default=28.0)
    ap.add_argument("--kmax", type=int, default=16)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--restarts", type=int, default=2000,
                    help="best_subset local-search restarts (K>=4 at p=1009 is not exhaustive)")
    ap.add_argument("--ladder-cache", default=None,
                    help="default $RUNS_DIR/p<p>_ladder_nu4.json (never notes/)")
    ap.add_argument("--construct", action="store_true")
    ap.add_argument("--construct-ks", type=int, nargs="+", default=[5, 6, 7, 8, 9])
    ap.add_argument("--construct-m", type=int, default=512)
    args = ap.parse_args()
    dev = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if args.check:
        sys.exit(0 if check_113(device=dev) else 1)
    assert args.out, "--out is required"
    p = args.p
    group = RUNS / (args.group or f"P{p}")

    # ---- runs
    rows = []
    for w in args.widths:
        for rd in sorted((group / f"relu4_m{w}_p{p}_f0.5_train").glob("wd*_s*")):
            if not (rd / "final.safetensors").exists():
                continue
            t = time.time(); r = score_run(rd, dev, args.mem_gb)
            if r is None:
                print(f"  {rd.name}: NON-FINITE params, skipped"); continue
            rows.append(r)
            print(f"  m={r['m']} {r['run']}: test {r['test_acc']:.4f} gamma {r['gamma']:.3e} "
                  f"K_gap {r['K_gap']} K_thr10 {r['K_thr10']} K_load90 {r['K_load90']} "
                  f"k_eff {r['k_eff']:.2f}  ({time.time()-t:.1f}s, a_rows {r['a_rows']})",
                  flush=True)
    if not rows:
        print(f"no runs under {group}")

    # ---- theory ladder
    print(f"\nplain-l2 ladder at p={p}, nu=4 (kstar_table.best_subset):", flush=True)
    lad = ladder(p, 4, args.kmax, args.restarts, args.reps,
                 cache=args.ladder_cache or str(RUNS / f"p{p}_ladder_nu4.json"))
    r_of = {int(k): v for k, v in lad["r"].items()}
    gh_of = {int(k): v for k, v in lad["gamma_hat"].items()}
    sp = lad.get("M_K_spread", {})
    hw = lad.get("how", {})
    print(f"  K*={lad['K_star']}   (M_K: {sum(v != 'local' for v in hw.values())} exhaustive, "
          f"{sum(v == 'local' for v in hw.values())} local search; worst rep-spread "
          f"{max([sp[k] / max(lad['M_K'][k], 1e-30) for k in sp], default=0.0):.3f} relative)")
    print("  K        " + " ".join(f"{K:8d}" for K in sorted(gh_of)))
    print("  gamma_hat" + " ".join(f"{gh_of[K]:8.4f}" for K in sorted(gh_of)))
    print("  r(K)     " + " ".join(f"{r_of[K]:8.3f}" for K in sorted(r_of)))
    plat74 = sorted(K for K in r_of if r_of[K] >= 0.74)
    plat84 = sorted(K for K in r_of if r_of[K] >= 0.84)
    print(f"  plateau r>=0.74: {plat74}    r>=0.84: {plat84}")
    if r_of.get(max(r_of), 0) >= 0.74:
        print(f"  WARNING: r(K={max(r_of)}) = {r_of[max(r_of)]:.3f} >= 0.74 -- the plateau "
              f"is truncated by --kmax; raise it to bracket the whole plateau.")

    # ---- constructive ceiling
    ceil = {}
    if args.construct:
        print(f"\nconstructive class ceiling (equal amplitude, m={args.construct_m}):",
              flush=True)
        for K in args.construct_ks:
            if K > p // 2:
                continue
            S = lad["S"][str(K)]
            t = time.time()
            gc, g_raw, n_a = construct_gamma_K(p, S, args.construct_m, 4, dev)
            ceil[K] = dict(gamma=gc, margin=g_raw, n_a=n_a, S=S)
            print(f"  K={K} S={S[:6]}{'...' if len(S) > 6 else ''} grid {n_a}x{n_a}: "
                  f"gamma_eq {gc:.4e}  ({time.time()-t:.1f}s)", flush=True)
        print("  NOTE: equal per-frequency amplitude only; construct_gamma.py's "
              "free-amplitude polish (60 coordinate-ascent sweeps x an exact full-grid\n"
              "  margin each) is ~700 grid sweeps per K and is skipped here -- these "
              "are LOWER BOUNDS on the class ceiling (at p=113 the polish adds ~1-3%).")

    out = dict(p=p, widths=args.widths, group=str(group), runs=rows, ladder=lad,
               plateau_74=plat74, plateau_84=plat84,
               ceiling={str(k): v for k, v in ceil.items()})
    Path(args.out).write_text(json.dumps(out, indent=1))

    # ---- table
    ceil_max = max((v["gamma"] for v in ceil.values()), default=None)
    print(f"\n{'='*100}\np={p}   K*={lad['K_star']}   plateau r>=0.74 {plat74}   r>=0.84 {plat84}")
    for w in args.widths:
        rs = [r for r in rows if r["m"] == w]
        if not rs:
            continue
        gk = [r for r in rs if r["grokked"]]
        h = lambda key, ss: dict(sorted(collections.Counter(r[key] for r in ss).items()))
        print(f"\n-- m={w}: n={len(rs)}, grokked {len(gk)}/{len(rs)} "
              f"(test acc med {np.median([r['test_acc'] for r in rs]):.4f})")
        for key in ("K_gap", "K_gap_amp", "K_thr10", "K_thr05", "K_load90"):
            print(f"   {key:9s} all {h(key, rs)}   grokked {h(key, gk)}")
        print(f"   k_eff med {np.median([r['k_eff'] for r in rs]):.2f} | "
              f"clean/total energy med {np.median([r['clean_frac'] for r in rs]):.3f} | "
              f"net/clean margin med {np.median([r['m_net_over_clean'] for r in rs]):.3f}")
        gm = float(np.median([r["gamma"] for r in rs]))
        q = np.percentile([r["gamma"] for r in rs], [25, 75])
        best_K = max(ceil, key=lambda k: ceil[k]["gamma"]) if ceil else None
        print(f"   gamma median {gm:.4e} (IQR {q[0]:.3e}-{q[1]:.3e})" +
              (f"   / ceiling(best K={best_K}) {gm/ceil_max:.3f}" if ceil_max
               else "   / ceiling n/a (--construct not run)"))
        if ceil:
            own = [r["gamma"] / ceil[r["K_thr10"]]["gamma"] for r in gk
                   if r["K_thr10"] in ceil]
            if own:
                print(f"   gamma / ceiling(own K_thr10) median {np.median(own):.3f} "
                      f"(n={len(own)})")
        if gk:
            f74 = np.mean([r["K_thr10"] in plat74 for r in gk])
            f84 = np.mean([r["K_thr10"] in plat84 for r in gk])
            print(f"   grokked runs with K_thr10 in plateau: r>=0.74 {f74:.3f} "
                  f"({int(f74*len(gk))}/{len(gk)}) | r>=0.84 {f84:.3f} "
                  f"({int(f84*len(gk))}/{len(gk)})")
            print(f"   top-8 relative amplitude median (grokked) "
                  f"{np.median([r['top8'] for r in gk], 0).round(3).tolist()}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
