"""Architecture-agnostic frequency count by INTERFACE ABLATION.

Every model on a Z_p token task has exactly two token-indexed interfaces: an
input table (one row/column per token) and an output table (one per logit).
A frequency k is removed SURGICALLY by projecting cos(2 pi k x/p), sin(2 pi k x/p)
out of those tables along the token axis, and the model's OWN unmodified
forward pass is re-run on the full p^2 grid. Nothing inside the box is
touched or modelled (attention, MLP, extra linear layers all run as-is).

Per run and per candidate frequency k (three cuts: input only, output only,
both):
    drop_k = 1 - minmargin(theta with k cut) / minmargin(theta)
plus the mean-margin and 1st-percentile-margin drops.
The COUNT is the number of candidate frequencies whose both-sides drop clears
a NULL FLOOR: the largest drop produced by cutting any of 32 random frequencies
outside the top-32 by interface energy. The threshold is therefore set by the
run itself, in margin units, and does not depend on p. Fixed-threshold counts
(drop >= 2%, 5%, 10%) and an additivity residual (sum of individual drops vs
cutting the whole counted set at once) are reported alongside.

Legacy logit-side instruments (K_gap, K_thr10, K_thr05, K_load90, k_eff,
clean_frac; nudial/count_definitions.py) are recomputed on the same forward
pass so every run carries both counts.

Adapters: nudial relu/quad MLPs (train_nudial.forward), the 1-layer
transformer (cloud/train_semifinal_torch.batched_final_logits), and the
explicit K-committee constructions (construct_gamma / score_p1009).

Usage
  uv run python nudial/ablation_count.py --mlp runs_nudial/M/relu4_m512_p113_f0.5_train \
      --tf runs_torch/p-113 --construct 113 --out notes/ablation_count_p113.json
  uv run python nudial/ablation_count.py --construct 1009 --out notes/ablation_count_c1009.json
"""
import argparse, collections, json, math, sys, time
from dataclasses import replace
from pathlib import Path

import numpy as np, torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "nudial")); sys.path.insert(0, str(ROOT / "cloud"))
from safetensors.torch import load_file
from train_nudial import Config as MLPConfig, forward as mlp_forward, committee_from_coeffs
torch.set_grad_enabled(False)
DT = torch.float64


# ------------------------------------------------------------------ Fourier
def trig(p, k, device="cpu"):
    x = torch.arange(p, dtype=DT, device=device)
    c = torch.cos(2 * math.pi * k * x / p); s = torch.sin(2 * math.pi * k * x / p)
    return c / c.norm(), s / s.norm()


def cut_freq(T, axis, p, ks):
    """Return T with frequencies ks projected out along `axis` (first p entries
    of that axis are the tokens; any extra entries, e.g. the '=' token, untouched)."""
    Tm = T.movedim(axis, -1).clone()
    tok = Tm[..., :p]
    for k in ks:
        c, s = trig(p, k, T.device)
        tok = tok - (tok @ c)[..., None] * c - (tok @ s)[..., None] * s
    Tm[..., :p] = tok
    return Tm.movedim(-1, axis)


def interface_energy(T, axis, p):
    """(nf,) energy of T's token axis at each frequency 1..p//2, summed over the other axes."""
    tok = T.movedim(axis, -1)[..., :p].reshape(-1, p)
    F = torch.fft.rfft(tok, dim=-1)[:, 1:p // 2 + 1]          # drop DC
    return (F.abs() ** 2).sum(0)


def cut_irrep(T, axis, p, rs, PROJ):
    """Group version of cut_freq: project isotypic components rs (indices into
    PROJ, (n_irr, p, p) symmetric idempotents) out of the token axis."""
    Tm = T.movedim(axis, -1).clone()
    tok = Tm[..., :p]
    for r in rs:
        tok = tok - tok @ PROJ[r]
    Tm[..., :p] = tok
    return Tm.movedim(-1, axis)


def interface_irrep_energy(T, axis, p, PROJ):
    tok = T.movedim(axis, -1)[..., :p].reshape(-1, p)
    return torch.stack([((tok @ PROJ[r]) ** 2).sum() for r in range(PROJ.shape[0])])


# ------------------------------------------------------------------ models
class Model:
    """cfg/p/pr plus: interfaces = [(key, axis, side)], logits(pr, a, b) -> (N, p).
    Z_p models: features are frequencies 1..p//2. Group models (self.group set):
    features are the nontrivial irreps, indexed into self.PROJ."""
    name = "?"
    group = None          # None (Z_p) or a groups-like module (MULT, PROJ, IRREP_NAMES)

    def labels(self, a, b):
        if self.group is None:
            return (a + b) % self.p
        return self.MULT[a, b]

    def feature_ids(self):
        return list(range(1, self.p // 2 + 1)) if self.group is None else list(range(1, self.PROJ.shape[0]))

    def cut(self, ks, sides=("in", "out")):
        pr = dict(self.pr)
        for key, axis, side in self.interfaces:
            if side in sides:
                pr[key] = (cut_freq(pr[key], axis, self.p, ks) if self.group is None
                           else cut_irrep(pr[key], axis, self.p, ks, self.PROJ))
        return pr

    def energy(self, sides=("in", "out")):
        """per-feature interface energy share, indexed by feature id - 1."""
        n = self.p // 2 if self.group is None else self.PROJ.shape[0] - 1
        e = torch.zeros(n, dtype=DT)
        for key, axis, side in self.interfaces:
            if side in sides:
                ek = (interface_energy(self.pr[key], axis, self.p) if self.group is None
                      else interface_irrep_energy(self.pr[key], axis, self.p, self.PROJ)[1:])
                e += ek / ek.sum()
        return e / e.sum()

    def set_group(self, fn_name):
        if fn_name == "s5":
            import groups
            self.group = groups
            self.MULT = torch.from_numpy(groups.MULT.astype(np.int64))
            self.PROJ = torch.from_numpy(groups.PROJ).to(DT)
            self.irrep_names = list(groups.IRREP_NAMES)

    def pnorm2(self, pr):
        return float(sum((v ** 2).sum() for v in pr.values()))


class MLP(Model):
    def __init__(self, rd):
        rd = Path(rd)
        cfgd = json.loads((rd / "config.json").read_text())
        self.cfg = MLPConfig(**{k: v for k, v in cfgd.items() if k in MLPConfig.__dataclass_fields__})
        self.p = self.cfg.p
        self.pr = {k: v.to(DT)[None] for k, v in load_file(rd / "final.safetensors").items()}
        self.name = f"mlp:{rd.parent.name}/{rd.name}"
        self._set_interfaces()

    def _set_interfaces(self):
        n_pre, _ = self.cfg.layout
        if n_pre == 0:
            self.interfaces = [("U", 2, "in"), ("V", 2, "in"), ("W", 1, "out")]
        else:
            self.interfaces = [("E1", 2, "in"), ("W", 1, "out")]

    def logits(self, pr, a, b):
        return mlp_forward(pr, a, b, self.cfg)[0]


class ConstructedMLP(MLP):
    def __init__(self, p, S, m=512, nu=4):
        assert nu == 4, "layout (1,1) constructions only"
        if p == 113:
            from construct_gamma import committee_net
            build = lambda: committee_net(S, np.ones(len(S)), m, n_a, n_b)
        else:
            from score_p1009 import committee_net
            build = lambda: committee_net(p, S, np.ones(len(S)), m, n_a, n_b)
        K = len(S); n_a = n_b = max(2, int(math.floor(math.sqrt(m // K))))
        E, U, V, W1, W = build()
        self.p = p; self.cfg = MLPConfig(p=p, m=m, d=2 * K, r=2 * K, family="relu", nu=4, layout_pre=1, layout_post=1)
        t = lambda x: torch.from_numpy(np.asarray(x)).to(DT)[None]
        self.pr = dict(E1=t(E), U=t(U), V=t(V), W1=t(W1), W=t(W))
        self.name = f"construct:p{p}_K{K}"
        self.S = list(S)
        self._set_interfaces()


class Transformer(Model):
    def __init__(self, rd, epoch=None):
        from train_semifinal_torch import Config as TFConfig, batched_final_logits
        rd = Path(rd)
        self.cfg = TFConfig.load(rd / "config.json"); self.p = self.cfg.p
        ck = sorted((rd / "checkpoints").glob("epoch_*.safetensors"),
                    key=lambda f: int(f.stem.split("_")[1]))
        f = ck[-1] if epoch is None else rd / "checkpoints" / f"epoch_{epoch}.safetensors"
        self.pr = {k: v.to(DT)[None] for k, v in load_file(f).items()}
        try:
            rel = rd.resolve().relative_to(ROOT.resolve())
        except ValueError:
            rel = rd
        self.name = f"tf:{rel}@{f.stem}"
        self.interfaces = [("embed.W_E", 2, "in"), ("unembed.W_U", 2, "out")]
        self._fwd = batched_final_logits
        self.set_group(self.cfg.fn_name)

    def logits(self, pr, a, b):
        tok = torch.stack([a, b, torch.full_like(a, self.p)], 1)
        return self._fwd(pr, tok, self.cfg)[0][:, :self.p]


# ------------------------------------------------------------------ grid stats
class Grid:
    def __init__(self, p, a_rows=None):
        self.p = p
        self.a_rows = a_rows or (p if p <= 200 else max(1, 20_000_000 // (p * p)))
        self.n_idx = torch.arange(p)

    def stats(self, model, pr, want_spectrum=False):
        """min / mean / q01 margin, accuracy; optionally the collapsed clean
        table g(n) = sum_{a+b-c=n} L and total energy."""
        p = self.p; mins = []; sums = 0.0; cnt = 0; accs = 0
        g = torch.zeros(p, dtype=DT); sq = 0.0
        qs = []
        for a0 in range(0, p, self.a_rows):
            a = torch.arange(a0, min(p, a0 + self.a_rows)).repeat_interleave(p)
            b = torch.arange(p).repeat(len(a) // p)
            L = model.logits(pr, a, b).to(DT)                      # (N, p)
            L = L - L.mean(1, keepdim=True)
            y = model.labels(a, b); idx = torch.arange(len(a))
            zy = L[idx, y]; o = L.clone(); o[idx, y] = -1e30
            mg = zy - o.max(1).values
            mins.append(float(mg.min())); sums += float(mg.sum()); cnt += len(a)
            accs += int((L.argmax(1) == y).sum()); qs.append(mg)
            if want_spectrum and model.group is None:
                n = ((a + b)[:, None] - self.n_idx[None]) % p          # (N, p)
                g.index_add_(0, n.reshape(-1), L.reshape(-1)); sq += float((L ** 2).sum())
            elif want_spectrum:                                        # group: per-irrep logit energy over c
                g = g + torch.stack([((L @ model.PROJ[r]) ** 2).sum() for r in range(model.PROJ.shape[0])]) if g.shape[0] == model.PROJ.shape[0] \
                    else torch.stack([((L @ model.PROJ[r]) ** 2).sum() for r in range(model.PROJ.shape[0])])
                sq += float((L ** 2).sum())
        allq = torch.cat(qs)
        out = dict(m_min=min(mins), m_mean=sums / cnt, m_q01=float(torch.quantile(allq, 0.01)),
                   acc=accs / cnt)
        if want_spectrum:
            out["g"] = g; out["sq"] = sq
        return out


def legacy_counts_group(e, sq, model):
    """Group analogue of the logit-side instruments: per-irrep logit energy
    (trivial excluded), amplitude = sqrt(energy), 10% / 5% cuts, k_eff."""
    amp = e[1:].sqrt(); order = torch.argsort(amp, descending=True)
    share = amp ** 2 / (amp ** 2).sum()
    return dict(K_gap=None, K_thr10=int((amp >= 0.1 * amp.max()).sum()), K_thr05=int((amp >= 0.05 * amp.max()).sum()),
                K_load90=None, k_eff=float(1 / (share ** 2).sum()), clean_frac=float(e[1:].sum() / sq), m_clean=None,
                top_k=[int(x) + 1 for x in order], top8=[round(float(x), 3) for x in (amp[order] / amp.max())],
                irrep_names=[model.irrep_names[int(x) + 1] for x in order])


def legacy_counts(g, sq, p):
    """The logit-side instruments from the collapsed clean table g(n)."""
    nf = p // 2; n = torch.arange(p, dtype=DT); k = torch.arange(1, nf + 1, dtype=DT)
    C = torch.cos(2 * math.pi * k[:, None] * n[None] / p); S = torch.sin(2 * math.pi * k[:, None] * n[None] / p)
    cc = (C @ g) * 2 / p ** 3; ss = (S @ g) * 2 / p ** 3
    amp = torch.sqrt(cc ** 2 + ss ** 2); order = torch.argsort(amp, descending=True)
    f = lambda idx: (cc[idx][:, None] * C[idx] + ss[idx][:, None] * S[idx]).sum(0)   # clean f(n)
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
                top_k=[int(x) + 1 for x in order[:8]],
                top8=[round(float(x), 3) for x in (amp[order][:8] / amp.max())])


# ------------------------------------------------------------------ the instrument
def ablation_count(model, n_cand=16, n_null=32, null_seed=0, sides_all=True, fold_n=5, fold_e=0.25, group_floor=0.05, log=print):
    p = model.p; nf = p // 2; grid = Grid(p)
    t0 = time.time()
    base = grid.stats(model, model.pr, want_spectrum=True)
    if model.group is None:
        leg = legacy_counts(base.pop("g"), base.pop("sq"), p)
    else:
        leg = legacy_counts_group(base.pop("g"), base.pop("sq"), model)
    e_in, e_out = model.energy(("in",)), model.energy(("out",))
    e_both = (e_in + e_out) / 2
    order = torch.argsort(e_both, descending=True)
    if model.group is None:
        cand = sorted(set(int(k) + 1 for k in order[:n_cand]) | set(leg["top_k"][:8]))
        top32 = set(int(k) + 1 for k in order[:32])
        rng = np.random.default_rng(null_seed)
        pool = [k for k in range(1, nf + 1) if k not in top32 and k not in cand]
        null = sorted(int(k) for k in rng.choice(pool, size=min(n_null, len(pool)), replace=False))
    else:
        cand = model.feature_ids(); null = []          # 6 nontrivial irreps, no null pool: fixed floor
    m0 = base["m_min"]

    def drops(ks, sides):
        st = grid.stats(model, model.cut(ks, sides))
        return dict(d_min=1 - st["m_min"] / m0, d_mean=1 - st["m_mean"] / base["m_mean"],
                    d_q01=1 - st["m_q01"] / base["m_q01"], acc=st["acc"])

    per = {}
    for k in cand + null:
        row = dict(both=drops([k], ("in", "out")))
        if sides_all:
            row["in"] = drops([k], ("in",)); row["out"] = drops([k], ("out",))
        row["e_in"] = float(e_in[k - 1]); row["e_out"] = float(e_out[k - 1])
        per[k] = row
    EPS = 1e-6 if model.group is None else group_floor   # Z_p: numerical floor; groups: fixed fraction
    null_floor = max([per[k]["both"]["d_min"] for k in null] + [EPS])
    null_floor_mean = max([per[k]["both"]["d_mean"] for k in null] + [EPS])
    prof = sorted(((per[k]["both"]["d_min"], k) for k in cand), reverse=True)
    raw = [k for d, k in prof if d > null_floor]
    # harmonic fold (group-level, model-free): a significant k that is a small
    # multiple n*f (n <= fold_n, up to sign, mod p) of an already-counted
    # fundamental f AND carries < fold_e of f's interface energy is a
    # harmonic-cancellation CORRECTOR of f, not an independent feature.
    e_tot = {k: per[k]["e_in"] + per[k]["e_out"] for k in cand}
    def harmonic_of(k, fnds):
        """fundamental f (or pair) such that k = n f or n1 f1 + n2 f2 (mod p, up to sign)."""
        if model.group is not None:
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
    # read-AND-write: k must matter when cut on the input side alone AND on the
    # output side alone (each against its own null floor). Harmonic correctors
    # are read-only, output sidelobes are write-only; features do both.
    if sides_all:
        fl_in = max([per[k]["in"]["d_min"] for k in null] + [EPS]); fl_out = max([per[k]["out"]["d_min"] for k in null] + [EPS])
        is_rw = lambda k: per[k]["in"]["d_min"] > fl_in and per[k]["out"]["d_min"] > fl_out
    else:
        fl_in = fl_out = None; is_rw = lambda k: True
    rw = [k for k in raw if is_rw(k)]
    # fold in interface-ENERGY order so a fundamental is always counted before
    # its harmonics (drop order can put a corrector ahead of its fundamental)
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
    counted_mean = [k for k in cand if per[k]["both"]["d_mean"] > null_floor_mean]
    fixed = {f"K_d{int(t*100):02d}": sum(1 for d, _ in prof if d >= t) for t in (0.02, 0.05, 0.10)}
    joint = drops(counted, ("in", "out"))["d_min"] if counted else 0.0
    pair = counted[-2:] if len(counted) >= 2 else counted
    pair_joint = drops(pair, ("in", "out"))["d_min"] if pair else 0.0
    sum_ind = sum(per[k]["both"]["d_min"] for k in pair)
    res = dict(name=model.name, p=p, m_min=m0, m_mean=base["m_mean"], m_q01=base["m_q01"], acc=base["acc"],
               pnorm2=model.pnorm2(model.pr),
               K_abl=len(counted), counted=counted, K_abl_raw=len(raw), raw=raw, K_rw=len(rw), rw=rw, rw_dropped=dropped, floor_in=fl_in, floor_out=fl_out, classes={str(k): v for k, v in classes.items()},
               K_abl_mean=len(counted_mean),
               null_floor=null_floor, null_floor_mean=null_floor_mean,
               joint_drop=joint, pair=pair, pair_joint=pair_joint, pair_sum=sum_ind, additivity=(pair_joint / sum_ind if sum_ind else float("nan")),
               profile=[(k, round(d, 4)) for d, k in prof],
               per=per, cand=cand, null=null, **fixed, legacy=leg, secs=time.time() - t0)
    if model.group is not None:
        res["counted_names"] = [model.irrep_names[k] for k in counted]
        res["profile_names"] = [(model.irrep_names[k], d) for k, d in res["profile"]]
    log(f"  {model.name}: m_min {m0:.3f} acc {base['acc']:.4f} | K_abl {len(counted)} raw {len(raw)} fold {dict((k, v) for k, v in classes.items() if v)} "
        f"(null floor {null_floor:.4f}) vs K_thr10 {leg['K_thr10']} K_gap {leg['K_gap']} K_load90 {leg['K_load90']} "
        f"| profile {[(k, round(d, 3)) for k, d in res['profile'][:8]]} | add {res['additivity']:.2f} | {res['secs']:.0f}s")
    return res


# ------------------------------------------------------------------ driver
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mlp", nargs="*", default=[], help="cell dirs (each holds wd*_s* run dirs)")
    ap.add_argument("--tf", nargs="*", default=[], help="cohort roots under runs_torch (recursed for config.json)")
    ap.add_argument("--construct", type=int, nargs="*", default=[], help="p values for constructed K-circuits")
    ap.add_argument("--construct-ks", type=int, nargs="+", default=None)
    ap.add_argument("--limit", type=int, default=None, help="max runs per cell/cohort")
    ap.add_argument("--tf-epoch", type=int, default=None, help="transformer checkpoint epoch (default: last)")
    ap.add_argument("--group-floor", type=float, default=0.05, help="fixed drop floor for group (irrep) models")
    ap.add_argument("--n-cand", type=int, default=16)
    ap.add_argument("--n-null", type=int, default=32)
    ap.add_argument("--no-sides", action="store_true", help="skip the single-side cuts (3x faster)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rows = []
    save = lambda: Path(args.out).write_text(json.dumps(rows, indent=0, default=float))

    for cell in args.mlp:
        cell = Path(cell); print(f"\n== MLP cell {cell}")
        rds = sorted(d for d in cell.glob("*_s*") if (d / "final.safetensors").exists())[:args.limit]
        for rd in rds:
            try:
                m = MLP(rd)
            except Exception as e:
                print(f"  skip {rd}: {e}"); continue
            if not all(torch.isfinite(v).all() for v in m.pr.values()):
                print(f"  skip {rd}: non-finite"); continue
            rows.append(dict(kind="mlp", cell=str(cell), **ablation_count(m, args.n_cand, args.n_null, sides_all=not args.no_sides)))
            save()

    for root in args.tf:
        root = Path(root); print(f"\n== transformer cohort {root}")
        rds = sorted(f.parent for f in root.rglob("config.json") if (f.parent / "checkpoints").exists())[:args.limit]
        for rd in rds:
            try:
                m = Transformer(rd, epoch=args.tf_epoch)
            except Exception as e:
                print(f"  skip {rd}: {e}"); continue
            rows.append(dict(kind="tf", cell=str(root), **ablation_count(m, args.n_cand, args.n_null, sides_all=not args.no_sides, group_floor=args.group_floor)))
            save()

    for p in args.construct:
        print(f"\n== constructed circuits p={p}")
        if p == 113:
            from construct_gamma import SETS
            sets = {K: list(S) for K, S in SETS.items()}
        else:
            lad = json.load(open(ROOT / "notes" / "p1009_scored_m512.json"))["ladder"]["S"]
            sets = {int(K): S for K, S in lad.items()}
        ks = [K for K in (args.construct_ks or sorted(sets)) if K in sets]
        for K in ks:
            m = ConstructedMLP(p, sets[K])
            rows.append(dict(kind="construct", cell=f"construct_p{p}", K_true=K, S=sets[K],
                             **ablation_count(m, args.n_cand, min(args.n_null, 8 if p > 200 else args.n_null),
                                              sides_all=not args.no_sides)))
            save()

    # ---- summary
    print("\n==== SUMMARY (count instruments per cohort)")
    for cell in dict.fromkeys(r["cell"] for r in rows):
        rs = [r for r in rows if r["cell"] == cell]
        h = lambda f: dict(sorted(collections.Counter(f(r) for r in rs).items(), key=lambda kv: str(kv[0])))
        print(f"\n{cell}  n={len(rs)}")
        print(f"  K_abl      {h(lambda r: r['K_abl'])}   raw(unfolded) {h(lambda r: r['K_abl_raw'])}   K_rw(read&write) {h(lambda r: r['K_rw'])}")
        print(f"  K_abl_mean {h(lambda r: r['K_abl_mean'])}")
        print(f"  K_d05      {h(lambda r: r['K_d05'])}   K_d10 {h(lambda r: r['K_d10'])}")
        print(f"  K_thr10    {h(lambda r: r['legacy']['K_thr10'])}   K_gap {h(lambda r: r['legacy']['K_gap'])}   K_load90 {h(lambda r: r['legacy']['K_load90'])}")
        if "K_true" in rs[0]:
            print(f"  K_true     {h(lambda r: r['K_true'])}  exact: {sum(r['K_abl'] == r['K_true'] for r in rs)}/{len(rs)}")
        else:
            ag = np.mean([r["K_abl"] == r["legacy"]["K_thr10"] for r in rs]); w1 = np.mean([abs(r["K_abl"] - r["legacy"]["K_thr10"]) <= 1 for r in rs])
            print(f"  K_abl == K_thr10: {ag:.2f}  within 1: {w1:.2f} | null floor med {np.median([r['null_floor'] for r in rs]):.4f} "
                  f"| additivity med {np.median([r['additivity'] for r in rs]):.2f} | clean_frac med {np.median([r['legacy']['clean_frac'] for r in rs]):.3f}")
    save(); print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
