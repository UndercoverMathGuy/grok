"""ladder_abl.py -- score free-training endpoints of the gated chain against the exact ceiling.

Runs ON THE POD after training (or locally on pulled runs).  Per seed it reports

  gamma_q          = margin / P_q^{nu/q}                      (exact, fp64, full table)
  quality          = gamma_q / gamma_ceiling,  gamma_ceiling = mu N^{-(nu-q)/q} B Gamma_{nu/q}(p)
                     (the exact share-weighted block on the programme's optimal support -- the
                     highest normalized margin any junk-free block can reach)
  clean_frac       = fraction of the centred-logit energy that is a function of D = sum a - c;
                     junk_frac = 1 - clean_frac
  K_5pct, N_.05    = Fourier-energy instruments from nsum.py's metrics.json.  N_.05 counts every
                     frequency with >= 5% of the top frequency's amplitude: it is a TAIL statistic
                     proportional to the number of frequencies present, NOT "the count".
  K_abl            = INTERFACE-ABLATION count (nudial/ablation_count.py's definition, re-implemented
                     here for the chain): project cos/sin of frequency k out of every token-indexed
                     interface (the n embeddings on the input side, the readout on the output side),
                     re-run the net's own forward pass, and count the frequencies whose min-margin
                     drop beats a NULL FLOOR = the largest drop from cutting any of 32 random
                     frequencies outside the top 32 by interface energy.  A frequency must matter
                     when cut on the input side alone AND on the output side alone ("read-and-write"),
                     and a small multiple of an already-counted frequency carrying < 25% of its
                     interface energy is folded in as a harmonic corrector.

Usage (pod):  python3 -u heracles/ladder_abl.py --runs heracles/runs/LAD/* --ceil heracles/FINAL_RUNS/LADDER/json/ceilings.json --out heracles/FINAL_RUNS/LADDER/train/scored.json
Usage (validate the instrument on a constructed net):  python3 -u heracles/ladder_abl.py --validate 59
"""
import argparse
import itertools
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
DEV = os.environ.get("NSUM_DEV") or ("cuda" if torch.cuda.is_available() else "cpu")
torch.set_grad_enabled(False)
DT = torch.float64


# ------------------------------------------------------------------ table + forward (chain only)
def rows_of(p, n, tied):
    if not tied:
        r = torch.arange(p ** n)
        return torch.stack([(r // p ** j) % p for j in range(n)], 1)
    combs = np.fromiter(itertools.chain.from_iterable(itertools.combinations_with_replacement(range(p), n)),
                        dtype=np.int64).reshape(-1, n)
    return torch.from_numpy(combs)


def min_margin(E, W, ops, p, n, chunk=200_000):
    """Exact min over rows and wrong classes of z_correct - z_wrong.  E (nE, m, p), W (p, m)."""
    y = ops.sum(1) % p
    worst = math.inf
    for s in range(0, len(ops), chunk):
        o = ops[s:s + chunk].to(E.device)
        h = torch.ones(o.shape[0], E.shape[1], dtype=E.dtype, device=E.device)
        for j in range(n):
            h = h * E[0 if E.shape[0] == 1 else j][:, o[:, j]].T
        z = h @ W.T
        yy = y[s:s + chunk].to(E.device)
        zy = z.gather(1, yy[:, None]).squeeze(1)
        z = z.scatter(1, yy[:, None], -math.inf)
        worst = min(worst, float((zy - z.max(1).values).min()))
    return worst


def clean_spectrum_of(E, W, p, n):
    """ghat(k) = sum_i prod_j ehat_ij(k) * conj(what_i(k)) -- the only part of the logit table that
    is a function of D = sum a - c (THEORY_nsum L3).  Exact, O(m p), no table needed."""
    a = torch.arange(p, dtype=DT, device=E.device)
    ks = torch.arange(1, (p - 1) // 2 + 1, dtype=DT, device=E.device)
    ph = torch.exp(-2j * math.pi * ks[:, None] * a[None, :] / p) / p       # (nf, p)
    Eh = torch.stack([ph @ E[0 if E.shape[0] == 1 else j].to(torch.complex128).T for j in range(n)])  # (n, nf, m)
    Wh = ph @ W.to(torch.complex128)                                      # (nf, p) @ (p, m)
    g = (Eh.prod(0) * Wh.conj()).sum(1)                                   # (nf,)
    return g.abs().cpu().numpy()


def interfaces(E, W, p):
    """Token-indexed interfaces: the n embedding tables (input side) and the readout (output side)."""
    return [("in", E), ("out", W)]


def cut(E, W, p, ks, sides):
    E2, W2 = E.clone(), W.clone()
    x = torch.arange(p, dtype=E.dtype, device=E.device)
    for k in ks:
        c = torch.cos(2 * math.pi * k * x / p); c = c / c.norm()
        s = torch.sin(2 * math.pi * k * x / p); s = s / s.norm()
        if "in" in sides:
            for j in range(E2.shape[0]):
                E2[j] = E2[j] - (E2[j] @ c)[:, None] * c[None, :] - (E2[j] @ s)[:, None] * s[None, :]
        if "out" in sides:
            W2 = W2 - c[:, None] * (c[None, :] @ W2) - s[:, None] * (s[None, :] @ W2)
    return E2, W2


def iface_energy(E, W, p):
    """Per-frequency energy of the token-indexed interfaces (input tables averaged, output table)."""
    x = torch.arange(p, dtype=E.dtype, device=E.device)
    nf = (p - 1) // 2
    ei = torch.zeros(nf, dtype=E.dtype, device=E.device)
    eo = torch.zeros(nf, dtype=E.dtype, device=E.device)
    for k in range(1, nf + 1):
        c = torch.cos(2 * math.pi * k * x / p); c = c / c.norm()
        s = torch.sin(2 * math.pi * k * x / p); s = s / s.norm()
        ei[k - 1] = sum(float(((E[j] @ c) ** 2).sum() + ((E[j] @ s) ** 2).sum()) for j in range(E.shape[0])) / E.shape[0]
        eo[k - 1] = float(((c @ W) ** 2).sum() + ((s @ W) ** 2).sum())
    return ei.cpu().numpy(), eo.cpu().numpy()


def ablation_count(E, W, p, n, ops, n_cand=16, n_null=32, null_seed=0, fold_n=5, fold_e=0.25, log=print):
    nf = (p - 1) // 2
    m0 = min_margin(E, W, ops, p, n)
    if m0 <= 0:
        # the instrument measures the RELATIVE loss of min-margin when a frequency is cut; with a
        # non-positive margin to start from that ratio has no meaning, so we report it as undefined
        # rather than as a count of 0.
        log(f"    ablation UNDEFINED: min-margin {m0:.4f} <= 0 (the net does not separate the table)")
        return dict(K_abl=None, counted=[], K_raw=None, raw=[], null_floor=None, m_min=m0,
                    profile=[], folded={}, K_d10=None, K_d25=None, K_d50=None, K_d90=None,
                    undefined="min-margin <= 0")
    ei, eo = iface_energy(E, W, p)
    eb = (ei / max(ei.max(), 1e-300) + eo / max(eo.max(), 1e-300)) / 2
    order = np.argsort(-eb)
    # CANDIDATES: the n_cand frequencies with the most interface energy.  NULL POOL: everything
    # ranked below them.  nudial/ablation_count.py hardwires "top 32" for the null cut, which leaves
    # an EMPTY null pool whenever nf = (p-1)/2 is small (p = 59: nf = 29 < 32 + 16), and then the
    # floor degenerates to 1e-6 and every candidate is counted.  Here the cut scales with p.
    n_cand = max(4, min(n_cand, nf // 2))
    cand = sorted(int(k) + 1 for k in order[:n_cand])
    rng = np.random.default_rng(null_seed)
    pool = [int(k) + 1 for k in order[n_cand:]]
    null = sorted(rng.choice(pool, size=min(n_null, len(pool)), replace=False).tolist()) if pool else []

    def drop(ks, sides):
        E2, W2 = cut(E, W, p, ks, sides)
        return 1 - min_margin(E2, W2, ops, p, n) / m0 if m0 > 0 else float("nan")

    per = {k: dict(both=drop([k], ("in", "out")), inn=drop([k], ("in",)), out=drop([k], ("out",)),
                   e=float(ei[k - 1] + eo[k - 1])) for k in cand + null}
    EPS = 1e-6
    fl = max([per[k]["both"] for k in null] + [EPS])
    fi = max([per[k]["inn"] for k in null] + [EPS])
    fo = max([per[k]["out"] for k in null] + [EPS])
    raw = [k for k in cand if per[k]["both"] > fl]
    counted, classes = [], {}
    for k in sorted(raw, key=lambda k: -per[k]["e"]):
        owner = None
        for f in counted:
            if any((t * f) % p in (k, p - k) for t in range(2, fold_n + 1)):
                owner = f
                break
        if owner is not None and per[k]["e"] < fold_e * per[owner]["e"]:
            classes.setdefault(owner, []).append(k)
        elif per[k]["inn"] > fi and per[k]["out"] > fo:
            counted.append(k)
    counted.sort(key=lambda k: -per[k]["both"])
    # fixed-threshold counts alongside the null-floor one: how many frequencies cost at least
    # t of the min-margin when cut.  On a constructed K-net every support frequency costs > 50%.
    fixed = {f"K_d{int(t * 100):02d}": sum(1 for k in cand if per[k]["both"] >= t)
             for t in (0.10, 0.25, 0.50, 0.90)}
    out = dict(K_abl=len(counted), counted=counted, K_raw=len(raw), raw=raw, null_floor=fl,
               m_min=m0, profile=[(k, round(per[k]["both"], 4)) for k in sorted(cand, key=lambda k: -per[k]["both"])],
               folded={str(k): v for k, v in classes.items()}, **fixed)
    log(f"    K_abl {out['K_abl']} (raw {out['K_raw']}, floor {fl:.4f}) K_d10/25/50/90 "
        f"{fixed['K_d10']}/{fixed['K_d25']}/{fixed['K_d50']}/{fixed['K_d90']} counted {counted[:8]} m_min {m0:.4f}")
    return out


# ------------------------------------------------------------------ scoring one run
def score_run(sd, ceil, n_cand, n_null, log=print):
    st = __import__("safetensors.torch", fromlist=["load_file"]).load_file(str(sd / "final.safetensors"))
    mm = json.loads((sd / "metrics.json").read_text())
    E = st["E"].to(DT).to(DEV)
    W = st["W"].to(DT).to(DEV)
    p, n = mm["p"], mm["n"]
    nu = n + 1
    q = mm.get("q") or (nu if mm.get("penalty") == "l2nu" else 2.0)
    tied = bool(mm.get("tied", False))
    ops = rows_of(p, n, tied)
    n2 = (E ** 2).sum((0, 2)) * (n if E.shape[0] == 1 else 1) + (W ** 2).sum(0)
    Pq = float((n2 ** (q / 2)).sum())
    mg = min_margin(E, W, ops, p, n)
    gam = mg / Pq ** (nu / q)
    key = f"chain_n{n}_p{p}_q{q:.4g}"
    cl = ceil.get(key, {})
    g = clean_spectrum_of(E, W, p, n)
    top = np.sort(g)[::-1]
    row = dict(run=sd.parent.name, seed=sd.name, p=p, n=n, nu=nu, q=q, lam=mm.get("lam"), tied=tied,
               acc=mm.get("acc"), margin=mg, Pq=Pq, gamma=gam,
               ceiling=cl.get("gamma_ceiling"), quality=(gam / cl["gamma_ceiling"]) if cl else None,
               K_star=cl.get("K_star"), clean_frac=mm.get("clean_frac"),
               junk_frac=(1 - mm["clean_frac"]) if mm.get("clean_frac") is not None else None,
               K_5pct=mm.get("K_5pct"), K_2pct=mm.get("K_2pct"), N_05=mm.get("N_05"),
               ghat_top=[float(x) for x in top[:10]],
               K_ghat_5pct=int((g >= 0.05 * g.max()).sum()), steps=mm.get("steps"))
    log(f"  {sd.parent.name}/{sd.name}: acc {row['acc']} margin {mg:.4f} gamma {gam:.4e} "
        f"quality {row['quality'] if row['quality'] is None else round(row['quality'], 4)} clean {row['clean_frac']} "
        f"K_5pct {row['K_5pct']} N_.05 {row['N_05']}")
    row["abl"] = ablation_count(E, W, p, n, ops, n_cand=n_cand, n_null=n_null, log=log)
    return row


def validate(p, n, K, q, log=print):
    """Instrument validation: an exactly junk-free constructed K-frequency chain must give K_abl = K."""
    from ladder_core import best_K_support, build_net, gmat
    g = gmat(p)
    bk = best_K_support(g, K, (n + 1) / q, shortlist=40, polish=False)
    E, W = build_net("chain", p, n, bk["S"], bk["b"], q)
    E = torch.tensor(E, dtype=DT, device=DEV)
    W = torch.tensor(W, dtype=DT, device=DEV)
    ops = rows_of(p, n, n >= 4)
    r = ablation_count(E, W, p, n, ops, n_cand=min(16, (p - 1) // 2), n_null=24, log=log)
    act = [k for k, bb in zip(bk["S"], bk["b"]) if bb > 1e-4]            # the net's TRUE frequency count
    log(f"  constructed p {p} n {n} q {q:g}: asked K {K}, programme made {len(act)} active {act} "
        f"-> K_abl {r['K_abl']} {'OK' if r['K_abl'] == len(act) else 'MISMATCH'}")
    return dict(p=p, n=n, K_asked=K, q=q, K_true=len(act), K_abl=r["K_abl"], K_raw=r["K_raw"],
                counted=sorted(r["counted"]), S_active=act)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=[])
    ap.add_argument("--ceil", default=str(ROOT / "FINAL_RUNS" / "LADDER" / "json" / "ceilings.json"))
    ap.add_argument("--out", default=str(ROOT / "FINAL_RUNS" / "LADDER" / "train" / "scored.json"))
    ap.add_argument("--n-cand", type=int, default=16)
    ap.add_argument("--n-null", type=int, default=32)
    ap.add_argument("--validate", type=int, default=0, help="prime: run the instrument validation only")
    a = ap.parse_args()
    if a.validate:
        out = [validate(a.validate, n, K, q) for n, K, q in [(2, 3, 2.0), (2, 5, 2.0), (3, 4, 2.0), (3, 3, 8 / 3)]]
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump(out, open(Path(a.out).with_name("validate.json"), "w"), indent=1)
        return
    ceil = json.loads(Path(a.ceil).read_text()) if Path(a.ceil).exists() else {}
    rows = []
    for r in a.runs:
        rp = Path(r)
        if not rp.is_dir():
            continue
        for sd in sorted(rp.glob("seed*")):
            if (sd / "final.safetensors").exists():
                try:
                    rows.append(score_run(sd, ceil, a.n_cand, a.n_null))
                except Exception as e:                                   # noqa: BLE001
                    print(f"  !! {sd}: {e}", flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(rows, open(a.out, "w"), indent=1)
    print("wrote", a.out, len(rows), "rows")


if __name__ == "__main__":
    main()
