"""heracles scorer: count + margin instruments for one run directory (heracles/train.py outputs).

Count by INTERFACE ABLATION (port of nudial/ablation_count.py, MLP adapter only): a frequency k is
removed by projecting cos/sin(2 pi k x / p) out of the token axis of the input table (E1, or U and V
when there is no embedding) and of the output table W, and the model's own forward pass is re-run on
the full p^2 grid.  drop_k = 1 - minmargin(cut) / minmargin(full).  A frequency COUNTS if its
both-sides drop clears the null floor (largest drop from 32 random non-top-32 frequencies), it is read
AND written (input-only and output-only cuts each clear their own floor), and it is not a low-energy
harmonic corrector of an already-counted fundamental.  Legacy logit-side counts (K_thr10, K_thr05,
K_gap, K_load90, k_eff, clean_frac) are computed on the same forward pass.

Margins: m_min over the full grid; gamma2 = m_min / ||theta||_2^nu; gammaq = m_min / ||theta||_q^nu for
the run's own q (sign runs: q = inf, ||theta||_inf).

Usage:  uv run python heracles/score.py --runs runs/R/gd_* --out heracles/scores/R.json [--no-sides]
"""
import argparse, collections, json, math, sys, time
from pathlib import Path

import numpy as np, torch
from safetensors.torch import load_file

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train import Config, forward
from groups import get_group
torch.set_grad_enabled(False)
DT = torch.float64
FLOOR_FIXED = 0.02      # group tasks with too few irreps for a null pool (S5, S4, A5): an irrep counts if every cut drops m_min by >= 2%


def trig(p, k):
    x = torch.arange(p, dtype=DT)
    c = torch.cos(2 * math.pi * k * x / p); s = torch.sin(2 * math.pi * k * x / p)
    return c / c.norm(), s / s.norm()


def cut_freq(T, axis, p, ks):
    Tm = T.movedim(axis, -1).clone(); tok = Tm[..., :p]
    for k in ks:
        c, s = trig(p, k)
        tok = tok - (tok @ c)[..., None] * c - (tok @ s)[..., None] * s
    Tm[..., :p] = tok
    return Tm.movedim(-1, axis)


def interface_energy(T, axis, p):
    tok = T.movedim(axis, -1)[..., :p].reshape(-1, p)
    Fq = torch.fft.rfft(tok, dim=-1)[:, 1:p // 2 + 1]
    return (Fq.abs() ** 2).sum(0)


def cut_irrep(T, axis, PROJ, rs):
    """remove the isotypic components rs from the token axis: tok <- tok - tok @ P_r (P_r symmetric)."""
    Tm = T.movedim(axis, -1).clone(); tok = Tm
    for r in rs:
        tok = tok - tok @ PROJ[r]
    return tok.movedim(-1, axis)


def interface_energy_irrep(T, axis, PROJ):
    tok = T.movedim(axis, -1).reshape(-1, T.shape[axis])
    return torch.stack([((tok @ PROJ[r]) ** 2).sum() for r in range(PROJ.shape[0])])


def committee_from_coeffs(coeffs, floor=0.02):
    a = np.abs(coeffs); order = np.argsort(a)[::-1]
    logs = np.log(a[order] + 1e-12)
    cut = int(np.argmax(logs[:-1][:12] - logs[1:][:12])) + 1
    mem = order[:cut] + 1
    mem = mem[a[mem - 1] >= floor * a.max()]
    return sorted(mem.tolist())


class MLP:
    def __init__(self, rd, ckpt=None):
        rd = Path(rd)
        self.cfg = Config.load(rd / "config.json"); self.p = self.cfg.p
        self.G = None if self.cfg.task == "modadd" else get_group(self.cfg.task)
        self.PROJ = None if self.G is None else torch.from_numpy(self.G.PROJ).to(DT)
        f = rd / "final.safetensors" if ckpt is None else rd / ckpt
        self.pr = {k: v.to(DT)[None] for k, v in load_file(str(f)).items()}
        self.name = f"{rd.parent.name}/{rd.name}" + ("" if ckpt is None else f"@{ckpt}")
        n_pre, _ = self.cfg.layout
        self.interfaces = ([("U", 2, "in"), ("V", 2, "in"), ("W", 1, "out")] if n_pre == 0
                           else [("E1", 2, "in"), ("W", 1, "out")])

    def logits(self, pr, a, b):
        return forward(pr, a, b, self.cfg)[0]

    def cut(self, ks, sides=("in", "out")):
        """ks: frequencies (modadd) or irrep indices (group tasks)."""
        pr = dict(self.pr)
        for key, axis, side in self.interfaces:
            if side in sides:
                pr[key] = cut_freq(pr[key], axis, self.p, ks) if self.G is None else cut_irrep(pr[key], axis, self.PROJ, ks)
        return pr

    def energy(self, sides=("in", "out")):
        """modadd: (p//2,) per-frequency share, index k-1; group: (R,) per-irrep share, index = irrep index."""
        e = torch.zeros(self.p // 2 if self.G is None else self.G.R, dtype=DT)
        for key, axis, side in self.interfaces:
            if side in sides:
                ek = interface_energy(self.pr[key], axis, self.p) if self.G is None else interface_energy_irrep(self.pr[key], axis, self.PROJ)
                e += ek / ek.sum()
        return e / e.sum()

    def norms(self):
        n2 = float(sum((v ** 2).sum() for v in self.pr.values()))
        q = self.cfg.q
        if math.isinf(q):
            nq = float(max(v.abs().max() for v in self.pr.values()))
        else:
            nq = float(sum((v.abs() ** q).sum() for v in self.pr.values())) ** (1 / q)
        return n2, nq


class Grid:
    def __init__(self, p, G=None):
        self.p = p; self.a_rows = p if p <= 200 else max(1, 20_000_000 // (p * p)); self.n_idx = torch.arange(p)
        self.G = G
        if G is not None:
            self.MULT = torch.from_numpy(G.MULT); self.INV = torch.from_numpy(G.INV)

    def stats(self, model, pr, want_spectrum=False):
        p = self.p; mins = []; sums = 0.0; cnt = 0; accs = 0
        g = torch.zeros(p, dtype=DT); sq = 0.0; qs = []
        for a0 in range(0, p, self.a_rows):
            a = torch.arange(a0, min(p, a0 + self.a_rows)).repeat_interleave(p)
            b = torch.arange(p).repeat(len(a) // p)
            L = model.logits(pr, a, b).to(DT); L = L - L.mean(1, keepdim=True)
            y = (a + b) % p if self.G is None else self.MULT[a, b]; idx = torch.arange(len(a))
            zy = L[idx, y]; o = L.clone(); o[idx, y] = -1e30
            mg = zy - o.max(1).values
            mins.append(float(mg.min())); sums += float(mg.sum()); cnt += len(a)
            accs += int((L.argmax(1) == y).sum()); qs.append(mg)
            if want_spectrum:
                n = ((a + b)[:, None] - self.n_idx[None]) % p if self.G is None else self.MULT[y[:, None], self.INV[None, :]]
                g.index_add_(0, n.reshape(-1), L.reshape(-1)); sq += float((L ** 2).sum())
        allq = torch.cat(qs)
        out = dict(m_min=min(mins), m_mean=sums / cnt, m_q01=float(torch.quantile(allq, 0.01)), acc=accs / cnt)
        if want_spectrum:
            out["g"] = g; out["sq"] = sq
        return out


def legacy_counts(g, sq, p):
    nf = p // 2; n = torch.arange(p, dtype=DT); k = torch.arange(1, nf + 1, dtype=DT)
    C = torch.cos(2 * math.pi * k[:, None] * n[None] / p); S = torch.sin(2 * math.pi * k[:, None] * n[None] / p)
    cc = (C @ g) * 2 / p ** 3; ss = (S @ g) * 2 / p ** 3
    amp = torch.sqrt(cc ** 2 + ss ** 2); order = torch.argsort(amp, descending=True)
    f = lambda idx: (cc[idx][:, None] * C[idx] + ss[idx][:, None] * S[idx]).sum(0)
    mm = lambda fn: float(fn[0] - fn[1:].max())
    m_full = mm(f(torch.arange(nf)))
    K_load = None
    for i in range(1, nf + 1):
        if mm(f(order[:i])) >= 0.9 * m_full:
            K_load = i; break
    share = amp ** 2 / (amp ** 2).sum()
    return dict(K_gap=len(committee_from_coeffs(cc.numpy())),
                K_thr10=int((amp >= 0.1 * amp.max()).sum()), K_thr05=int((amp >= 0.05 * amp.max()).sum()),
                K_load90=K_load, k_eff=float(1 / (share ** 2).sum()),
                clean_frac=float((amp ** 2).sum() * p ** 3 / 2 / sq), m_clean=m_full,
                top_k=[int(x) + 1 for x in order[:8]], top8=[round(float(x), 3) for x in (amp[order][:8] / amp.max())])


def legacy_counts_group(g, sq, G):
    """group analogue of legacy_counts on the folded clean function f = g / N^2: amplitudes A_r = d_r sqrt(E_r / N),
    m_clean = f(e) - max_{x != e} f(x), K_load90 by adding irreps in amplitude order, clean_frac = ||g||^2 / (N^2 sq)."""
    N = G.N; f = (g / N ** 2).numpy(); PROJ = G.PROJ
    amp_all = G.amplitudes(f); nt = np.arange(1, G.R)                     # nontrivial irreps (the trivial one is a constant shift)
    amp = torch.from_numpy(amp_all[nt]); order = torch.argsort(amp, descending=True)
    comp = np.stack([PROJ[r] @ f for r in range(G.R)])
    mm = lambda fn: float(fn[G.e] - np.max(np.delete(fn, G.e)))
    m_full = mm(f); K_load = None
    for i in range(1, len(nt) + 1):
        if mm(comp[nt[order[:i].numpy()]].sum(0)) >= 0.9 * m_full:
            K_load = i; break
    share = amp ** 2 / (amp ** 2).sum()
    names = [G.irrep_names[int(nt[i])] for i in order]
    return dict(K_gap=len(committee_from_coeffs(amp.numpy())),
                K_thr10=int((amp >= 0.1 * amp.max()).sum()), K_thr05=int((amp >= 0.05 * amp.max()).sum()),
                K_load90=K_load, k_eff=float(1 / (share ** 2).sum()),
                clean_frac=float((g ** 2).sum() / N ** 2 / sq), m_clean=m_full,
                top_k=[int(nt[i]) for i in order[:8]], top8=[round(float(x), 3) for x in (amp[order][:8] / amp.max())],
                top_names=names[:8], amp={G.irrep_names[r]: float(amp_all[r]) for r in range(G.R)})


def ablation_count(model, n_cand=16, n_null=32, null_seed=0, sides_all=True, fold_n=5, fold_e=0.25, log=print):
    p = model.p; nf = p // 2; G = model.G; grid = Grid(p, G); t0 = time.time()
    base = grid.stats(model, model.pr, want_spectrum=True)
    e_in, e_out = model.energy(("in",)), model.energy(("out",)); e_both = (e_in + e_out) / 2
    rng = np.random.default_rng(null_seed)
    if G is None:
        leg = legacy_counts(base.pop("g"), base.pop("sq"), p)
        order = torch.argsort(e_both, descending=True)
        cand = sorted(set(int(k) + 1 for k in order[:n_cand]) | set(leg["top_k"][:8]))
        top32 = set(int(k) + 1 for k in order[:32])
        pool = [k for k in range(1, nf + 1) if k not in top32 and k not in cand]
    else:                                     # candidates = every nontrivial irrep (small groups) or the top n_cand by energy (D_113)
        leg = legacy_counts_group(base.pop("g"), base.pop("sq"), G)
        eb = e_both.clone(); eb[0] = -1; order = torch.argsort(eb, descending=True)
        cand = sorted(set(int(r) for r in order[:n_cand] if r != 0) | set(leg["top_k"][:8]))
        top32 = set(int(r) for r in order[:32])
        pool = [r for r in range(1, G.R) if r not in top32 and r not in cand]
    null = sorted(int(k) for k in rng.choice(pool, size=min(n_null, len(pool)), replace=False)) if pool else []
    m0 = base["m_min"]
    n2, nq = model.norms(); nu = model.cfg.nu
    res = dict(name=model.name, task=model.cfg.task, p=p, nu=nu, q=model.cfg.q, opt=model.cfg.opt, support=list(model.cfg.support),
               m_min=m0, m_mean=base["m_mean"], m_q01=base["m_q01"],
               acc=base["acc"], pnorm2=n2, qnorm=nq, gamma2=m0 / n2 ** (nu / 2), gammaq=m0 / nq ** nu, legacy=leg)
    if G is not None:                         # leader readout for the registered S5 statements (a)-(c)
        A = leg["amp"]; fa = [G.irrep_names[r] for r in np.flatnonzero(G.faithful)]
        tot = sum(A[n] ** 2 for n in fa); lead = max(fa, key=lambda n: A[n])
        res.update(leader=lead, lead_share=A[lead] ** 2 / tot if tot > 0 else None,
                   sign_rel=(A[G.irrep_names[G.sign]] / A[lead] if G.sign is not None and A[lead] > 0 else None),
                   n_faithful_02=int(sum(A[n] >= 0.2 * A[lead] for n in fa)))
    if m0 <= 0:                                    # not separated: counts are meaningless
        res.update(K_abl=None, counted=[], profile=[], null_floor=None, secs=time.time() - t0)
        log(f"  {model.name}: NOT separated (m_min {m0:.3f}, acc {base['acc']:.3f}) | K_thr10 {leg['K_thr10']} clean {leg['clean_frac']:.2f}")
        return res

    def drops(ks, sides):
        st = grid.stats(model, model.cut(ks, sides))
        return dict(d_min=1 - st["m_min"] / m0, d_mean=1 - st["m_mean"] / base["m_mean"], acc=st["acc"])

    per = {}
    for k in cand + null:
        row = dict(both=drops([k], ("in", "out")))
        if sides_all:
            row["in"] = drops([k], ("in",)); row["out"] = drops([k], ("out",))
        row["e_in"] = float(e_in[k - 1]); row["e_out"] = float(e_out[k - 1]); per[k] = row
    EPS = 1e-6 if (G is None or null) else FLOOR_FIXED      # mod-add unchanged; small groups use the fixed 2% floor
    null_floor = max([per[k]["both"]["d_min"] for k in null] + [EPS])
    prof = sorted(((per[k]["both"]["d_min"], k) for k in cand), reverse=True)
    raw = [k for d, k in prof if d > null_floor]
    e_tot = {k: per[k]["e_in"] + per[k]["e_out"] for k in cand}

    def harmonic_of(k, fnds):
        if G is not None:
            return None
        for f1 in fnds:
            if any((n * f1) % p in (k, p - k) for n in range(2, fold_n + 1)):
                return f1
        for i, f1 in enumerate(fnds):
            for f2 in fnds[i + 1:]:
                for n1 in range(-2, 3):
                    for n2 in range(-2, 3):
                        if n1 and n2 and (n1 * f1 + n2 * f2) % p in (k, p - k):
                            return f1 if e_tot[f1] >= e_tot[f2] else f2
        return None

    if sides_all:
        fl_in = max([per[k]["in"]["d_min"] for k in null] + [EPS]); fl_out = max([per[k]["out"]["d_min"] for k in null] + [EPS])
        is_rw = lambda k: per[k]["in"]["d_min"] > fl_in and per[k]["out"]["d_min"] > fl_out
    else:
        fl_in = fl_out = None; is_rw = lambda k: True
    counted, classes, dropped = [], {}, []
    for k in sorted(raw, key=lambda k: -e_tot[k]):
        owner = harmonic_of(k, counted)
        if owner is not None and e_tot[k] < fold_e * e_tot[owner]:
            classes[owner].append(k)
        elif not is_rw(k):
            dropped.append(k)
        else:
            counted.append(k); classes[k] = []
    counted.sort(key=lambda k: -per[k]["both"]["d_min"])
    fixed = {f"K_d{int(t*100):02d}": sum(1 for d, _ in prof if d >= t) for t in (0.02, 0.05, 0.10)}
    res.update(K_abl=len(counted), counted=counted, K_abl_raw=len(raw), raw=raw, rw_dropped=dropped,
               classes={str(k): v for k, v in classes.items()}, null_floor=null_floor, floor_in=fl_in, floor_out=fl_out,
               profile=[(k, round(d, 4)) for d, k in prof], per=per, cand=cand, null=null, **fixed, secs=time.time() - t0)
    log(f"  {model.name}: m_min {m0:.3f} acc {base['acc']:.4f} gamma2 {res['gamma2']:.3g} | K_abl {len(counted)} raw {len(raw)} "
        f"(floor {null_floor:.4f}) vs K_thr10 {leg['K_thr10']} K_gap {leg['K_gap']} K_load90 {leg['K_load90']} clean {leg['clean_frac']:.3f} "
        f"| top {res['profile'][:6]} | {res['secs']:.0f}s"
        + (f" | leader {res['leader']} share {res['lead_share']:.2f} sign {res['sign_rel']}" if G is not None else ""))
    return res


def summarize(rows, key=lambda r: r["cell"]):
    out = {}
    for cell in dict.fromkeys(key(r) for r in rows):
        rs = [r for r in rows if key(r) == cell]
        g = [r for r in rs if r["acc"] >= 0.99 and r["K_abl"] is not None]
        h = lambda f: dict(sorted(collections.Counter(f(r) for r in g).items(), key=lambda kv: str(kv[0])))
        med = lambda f: float(np.median([f(r) for r in g])) if g else None
        out[cell] = dict(n=len(rs), grok=len(g), K_abl=h(lambda r: r["K_abl"]), K_thr10=h(lambda r: r["legacy"]["K_thr10"]),
                         leader=h(lambda r: r.get("leader")) if g and "leader" in g[0] else None,
                         lead_share_med=med(lambda r: r["lead_share"]) if g and "leader" in g[0] else None,
                         sign_rel_med=(float(np.median([r["sign_rel"] for r in g if r.get("sign_rel") is not None]))
                                       if g and any(r.get("sign_rel") is not None for r in g) else None),
                         K_abl_med=med(lambda r: r["K_abl"]), gamma2_med=med(lambda r: r["gamma2"]), gammaq_med=med(lambda r: r["gammaq"]),
                         clean_med=med(lambda r: r["legacy"]["clean_frac"]), m_min_med=med(lambda r: r["m_min"]),
                         nonfinite=sum(1 for r in rs if not math.isfinite(r["m_min"])))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="run dirs or cell dirs (each holding seed*/ run dirs)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--no-sides", action="store_true")
    ap.add_argument("--n-cand", type=int, default=16)
    args = ap.parse_args()
    rds = []; wfile = args.ckpt or "final.safetensors"     # run dirs are those holding the requested weight file
    for r in args.runs:
        r = Path(r)
        if (r / wfile).exists():
            rds.append(r)
        else:
            rds += sorted(d for d in r.glob("*") if (d / wfile).exists())
    rows = []; Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    save = lambda: Path(args.out).write_text(json.dumps(dict(rows=rows, summary=summarize(rows)), indent=0, default=float))
    for rd in rds:
        try:
            m = MLP(rd, args.ckpt)
        except Exception as e:
            print(f"  skip {rd}: {e}"); continue
        if not all(torch.isfinite(v).all() for v in m.pr.values()):
            rows.append(dict(cell=rd.parent.name, name=str(rd), acc=0.0, m_min=float("nan"), K_abl=None, gamma2=float("nan"), gammaq=float("nan"), legacy=dict(K_thr10=None, clean_frac=float("nan"))))
            print(f"  {rd}: non-finite weights"); save(); continue
        rows.append(dict(cell=rd.parent.name, **ablation_count(m, args.n_cand, sides_all=not args.no_sides)))
        save()
    print("\n==== SUMMARY")
    for cell, s in summarize(rows).items():
        print(f"{cell}: n={s['n']} grok={s['grok']} K_abl={s['K_abl']} K_thr10={s['K_thr10']} gamma2_med={s['gamma2_med']} clean={s['clean_med']} nonfinite={s['nonfinite']}"
              + (f" leader={s['leader']} share={s['lead_share_med']} sign_rel={s['sign_rel_med']}" if s.get('leader') else ""))
    save(); print("wrote", args.out)


if __name__ == "__main__":
    main()
