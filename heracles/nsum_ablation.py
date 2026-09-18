"""Interface-ablation count (nudial/ablation_count.py) for nsum.py networks, so every cohort is scored by one instrument.

A frequency k is cut by projecting cos/sin(2 pi k x / p) out of the token axis of every embedding table (input side) and of
the readout W (output side). k is COUNTED if the margin drop clears the null floor (max drop over 32 random low-energy
frequencies) with both sides cut, and on each side alone (read-and-write rule); harmonic correctors are folded into their
fundamental. Margin statistic: the minimum margin when it is positive, else the mean margin (flagged `stat`).

    uv run python heracles/nsum_ablation.py [--glob 'n2_*'] [--out heracles/nsum_ablation.json]
"""
import argparse, json, math, re, sys, time
from pathlib import Path
import numpy as np, torch
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT.parent / "nudial"))
from nsum import table
torch.set_grad_enabled(False)
DT = torch.float64


def trig(p, k):
    x = torch.arange(p, dtype=DT)
    c = torch.cos(2 * math.pi * k * x / p); s = torch.sin(2 * math.pi * k * x / p)
    return c / c.norm(), s / s.norm()


def cut_freq(T, axis, p, ks):
    Tm = T.movedim(axis, -1).clone()
    for k in ks:
        c, s = trig(p, k)
        Tm = Tm - (Tm @ c)[..., None] * c - (Tm @ s)[..., None] * s
    return Tm.movedim(-1, axis)


def energy(T, axis, p):
    tok = T.movedim(axis, -1).reshape(-1, p)
    F = torch.fft.rfft(tok, dim=-1)[:, 1:p // 2 + 1]
    e = (F.abs() ** 2).sum(0)
    return e / e.sum()


class Net:
    def __init__(self, rd, chunk=60000):
        rd = Path(rd); mt = json.loads((rd / "metrics.json").read_text())
        g = re.match(r"n(\d+)_([a-z]+)_p(\d+)_", rd.parent.name)
        self.n, self.act, self.p = int(g.group(1)), g.group(2), int(g.group(3))
        self.tied = "_tied" in rd.parent.name
        sd = load_file(rd / "final.safetensors")
        self.E, self.W = sd["E"].to(DT), sd["W"].to(DT)              # (n_emb, m, p), (p, m)
        self.ops, self.w = table(self.p, self.n, self.tied)
        self.y = self.ops.sum(1) % self.p
        self.chunk = chunk; self.name = f"{rd.parent.name}/{rd.name}"; self.metrics = mt

    def logits(self, E, W, o):
        emb = lambda j: E[0 if self.tied else j][:, o[:, j]]          # (m, Nc)
        if self.act == "prod":
            h = emb(0)
            for j in range(1, self.n):
                h = h * emb(j)
        else:
            x = emb(0)
            for j in range(1, self.n):
                x = x + emb(j)
            h = torch.relu(x) if self.act == "relu" else x ** self.n
        return (W @ h).T

    def stats(self, E, W):
        mn, sm, acc = float("inf"), 0.0, 0.0
        for s in range(0, len(self.ops), self.chunk):
            o, y, w = self.ops[s:s + self.chunk], self.y[s:s + self.chunk], self.w[s:s + self.chunk]
            z = self.logits(E, W, o); idx = torch.arange(len(y))
            zy = z[idx, y]; z[idx, y] = -float("inf")
            mg = zy - z.max(1).values
            mn = min(mn, float(mg.min())); sm += float((mg * w).sum()); acc += float(((mg > 0).double() * w).sum())
        N = float(self.w.sum())
        return dict(m_min=mn, m_mean=sm / N, acc=acc / N)

    def cut(self, ks, sides):
        E = cut_freq(self.E, 2, self.p, ks) if "in" in sides else self.E
        W = cut_freq(self.W, 0, self.p, ks) if "out" in sides else self.W
        return E, W


def ablation_count(net, n_cand=16, n_null=32, fold_n=5, fold_e=0.25, log=print):
    p, nf = net.p, net.p // 2; t0 = time.time()
    base = net.stats(net.E, net.W)
    stat = "m_min" if base["m_min"] > 0 else "m_mean"
    e_in, e_out = energy(net.E, 2, p), energy(net.W, 0, p)
    order = torch.argsort(e_in + e_out, descending=True)
    b2 = np.asarray(net.metrics["b2"]); top_logit = [int(i) + 1 for i in np.argsort(-b2)[:8]]
    cand = sorted(set(int(k) + 1 for k in order[:n_cand]) | set(top_logit))
    top = set(int(k) + 1 for k in order[:max(n_cand, nf - n_null)]) | set(cand)
    pool = [k for k in range(1, nf + 1) if k not in top]
    rng = np.random.default_rng(0)
    null = sorted(int(k) for k in rng.choice(pool, size=min(n_null, len(pool)), replace=False)) if pool else []
    drop = lambda ks, sides: 1 - net.stats(*net.cut(ks, sides))[stat] / base[stat]
    per = {k: dict(both=drop([k], ("in", "out")), inn=drop([k], ("in",)), out=drop([k], ("out",)),
                   e=float(e_in[k - 1] + e_out[k - 1])) for k in cand + null}
    EPS = 1e-6
    fl = {s: max([per[k][s] for k in null] + [EPS]) for s in ("both", "inn", "out")}
    raw = [k for k in cand if per[k]["both"] > fl["both"]]
    is_rw = lambda k: per[k]["inn"] > fl["inn"] and per[k]["out"] > fl["out"]

    def harmonic_of(k, fnds):
        for f1 in fnds:
            if any((n * f1) % p in (k, p - k) for n in range(2, fold_n + 1)):
                return f1
        for i, f1 in enumerate(fnds):
            for f2 in fnds[i + 1:]:
                for n1 in range(-2, 3):
                    for n2 in range(-2, 3):
                        if n1 and n2 and (n1 * f1 + n2 * f2) % p in (k, p - k):
                            return f1 if per[f1]["e"] >= per[f2]["e"] else f2
        return None

    counted, folded, dropped = [], {}, []
    for k in sorted(raw, key=lambda k: -per[k]["e"]):
        owner = harmonic_of(k, counted)
        if owner is not None and per[k]["e"] < fold_e * per[owner]["e"]:
            folded.setdefault(owner, []).append(k)
        elif not is_rw(k):
            dropped.append(k)
        else:
            counted.append(k)
    prof = sorted(((round(per[k]["both"], 4), k) for k in cand), reverse=True)
    res = dict(name=net.name, p=p, n=net.n, act=net.act, stat=stat, **base, K_abl=len(counted), counted=sorted(counted),
               K_raw=len(raw), K_rw=sum(is_rw(k) for k in raw), folded={str(k): v for k, v in folded.items()}, rw_dropped=dropped,
               floors=fl, n_null=len(null), K_d05=sum(d >= 0.05 for d, _ in prof), K_d10=sum(d >= 0.10 for d, _ in prof),
               profile=[(k, d) for d, k in prof[:12]],
               N_05=net.metrics["N_05"], K_5pct=net.metrics["K_5pct"], K_2pct=net.metrics["K_2pct"],
               clean_frac=net.metrics["clean_frac"], q=net.metrics.get("q"), penalty=net.metrics.get("penalty"),
               lam=net.metrics.get("lam"), seed=net.metrics.get("seed"), secs=time.time() - t0)
    log(f"{net.name}: acc {base['acc']:.3f} {stat} {base[stat]:.3f} | K_abl {res['K_abl']} raw {res['K_raw']} rw {res['K_rw']} "
        f"K_d05 {res['K_d05']} | N_05 {res['N_05']} K_5pct {res['K_5pct']} K_2pct {res['K_2pct']} clean {res['clean_frac']:.2f} "
        f"| floor {fl['both']:.4f} (null {len(null)}) | {res['secs']:.0f}s")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=str(ROOT / "runs" / "NS"))
    ap.add_argument("--glob", default="*")
    ap.add_argument("--out", default=str(ROOT / "nsum_ablation.json"))
    a = ap.parse_args()
    out = Path(a.out); done = {r["name"]: r for r in json.loads(out.read_text())} if out.exists() else {}
    for rd in sorted(Path(a.runs).glob(f"{a.glob}/seed*")):
        name = f"{rd.parent.name}/{rd.name}"
        if name in done or not (rd / "final.safetensors").exists():
            continue
        try:
            done[name] = ablation_count(Net(rd), log=lambda s: print(s, flush=True))
        except Exception as ex:
            print(f"FAIL {name}: {ex!r}", flush=True); continue
        out.write_text(json.dumps(list(done.values()), indent=1))


if __name__ == "__main__":
    main()
