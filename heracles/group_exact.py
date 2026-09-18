"""group_exact.py -- the group (numerator M_K) axis on the EXACT-theory families (nu = 3, plain l2, q = 2).

Task: a . b = c on a finite group G (S5, S4, A5, Z_120, Z_11xZ_11; Z_113, Z_71 cyclic controls), one-hot inputs,
one hidden layer, no biases, no pre-layer, width m:
    quad : z(c; a, b) = sum_i w_i(c) (e_i(a) + f_i(b))^2           (Morwani's network)
    bil  : z(c; a, b) = sum_i w_i(c)  e_i(a) f_i(b)                (gated / CP factorisation, nsum.py --act prod, n = 2)
Recipe = nsum.py / quad_morwani.py: full table (or a per-seed random --frac of it), full-batch GD on
mean CE + lam * sum(theta^2) (penalty in the loss, NOT decoupled decay), lr 0.05 doubled at steps 1e3..10e3 capped at
--lrmax, init = nsum's uniform init x --init_scale (same generator order: a cyclic group at seed s starts from
nsum.py's seed-s weights), optional per-run global grad-norm clip. Seeds are BATCHED on a leading dim (independent
runs, one bmm), gradients are hand-written (checked against autograd with --check).

Scoring (fp64, full table): score.py's group instruments via a duck-typed adapter -- folded clean class function,
per-irrep amplitudes A_r, leader / lead_share / sign_rel / n_faithful_02, K_thr10, K_load90, clean_frac, and the
interface-ablation count K_abl over isotypic components (cut irrep r from e, f and w; read-and-write rule).
gamma = m_min / ||theta||^3.

`predict` subcommand: the nu = 3 ladders (see RESULTS_GX.md for what is theorem and what is ansatz).
Usage:  uv run python heracles/group_exact.py predict
        python3 heracles/group_exact.py train --group s5 --family bil --seeds 0 1 2 3 --steps 40000
"""
import argparse, itertools, json, math, os, sys, time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import groups as groups_mod
from groups import get_group

OUT = Path(os.environ.get("GX_OUT", HERE / "FINAL_RUNS" / "GX"))
DEV = os.environ.get("GX_DEV") or ("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
LR_DOUBLE_STEPS = (1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000)


def group(name):
    if name == "z71":
        if "z71" not in groups_mod._CACHE:
            groups_mod._CACHE["z71"] = groups_mod.abelian((71,), "z71")
        return groups_mod._CACHE["z71"]
    return get_group(name)


# ------------------------------------------------------------------------------------------------ model
def forward(Ea, Eb, W, fam):
    """Ea, Eb (S, m, N), W (S, N, m) -> h (S, m, N*N) rows ordered a*N + b, logits (S, N, N*N)."""
    S, m, N = Ea.shape
    if fam == "quad":
        x = Ea[:, :, :, None] + Eb[:, :, None, :]; h = (x * x).reshape(S, m, N * N)
    else:
        h = (Ea[:, :, :, None] * Eb[:, :, None, :]).reshape(S, m, N * N)
    return h, torch.bmm(W, h)


def loss_and_grads(Ea, Eb, W, fam, y, wrow):
    """mean (weighted) CE per seed and its exact gradients. y (R,), wrow (S, R) normalised to sum 1 per seed."""
    S, m, N = Ea.shape
    h, Z = forward(Ea, Eb, W, fam)
    lse = torch.logsumexp(Z, 1)                                           # (S, R)
    zy = Z.gather(1, y[None, None, :].expand(S, 1, -1)).squeeze(1)
    ce = ((lse - zy) * wrow).sum(1)                                       # (S,)
    P = torch.softmax(Z, 1); P.scatter_add_(1, y[None, None, :].expand(S, 1, -1), -torch.ones_like(P[:, :1]))
    P *= wrow[:, None, :]                                                 # dL/dZ
    gW = torch.bmm(P, h.transpose(1, 2))
    gh = torch.bmm(W.transpose(1, 2), P).reshape(S, m, N, N)              # (S, m, a, b)
    if fam == "quad":                                                     # dh/dx = 2 (Ea[a] + Eb[b])
        ra, rb = gh.sum(3), gh.sum(2)
        gEa = 2 * (Ea * ra + torch.einsum("smab,smb->sma", gh, Eb))
        gEb = 2 * (Eb * rb + torch.einsum("smab,sma->smb", gh, Ea))
    else:
        gEa = torch.einsum("smab,smb->sma", gh, Eb)
        gEb = torch.einsum("smab,sma->smb", gh, Ea)
    correct = (Z.argmax(1) == y[None]).float().mean(1)
    return ce, gEa, gEb, gW, correct


def init(seeds, m, N, scale):
    Es, Ws = [], []
    for s in seeds:
        gen = torch.Generator().manual_seed(s)
        E = (torch.rand(2, m, N, generator=gen) * 2 - 1) / math.sqrt(2 * N)
        W = (torch.rand(N, m, generator=gen) * 2 - 1) / math.sqrt(m)
        Es.append(scale * E); Ws.append(scale * W)
    E = torch.stack(Es); return E[:, 0].contiguous(), E[:, 1].contiguous(), torch.stack(Ws)


# ------------------------------------------------------------------------------------------------ scoring adapter
class Net:
    """duck-typed score.py model: irreps cut from the token axes of e, f ('in') and w ('out')."""
    def __init__(self, G, fam, Ea, Eb, W, name, dev):
        import score
        self.score = score; self.G = G; self.p = G.N; self.fam = fam; self.name = name; self.dev = dev
        self.PROJ = torch.from_numpy(G.PROJ).to(torch.float64).to(dev)
        self.pr = {"Ea": Ea.double().to(dev), "Eb": Eb.double().to(dev), "W": W.double().to(dev)}
        self.cfg = SimpleNamespace(nu=3, task=G.name, q=2.0, opt="gd", support=[])
        self.interfaces = [("Ea", 1, "in"), ("Eb", 1, "in"), ("W", 0, "out")]

    def logits(self, pr, a, b):
        a, b = a.to(self.dev), b.to(self.dev)
        ea, eb = pr["Ea"][:, a], pr["Eb"][:, b]
        h = (ea + eb) ** 2 if self.fam == "quad" else ea * eb
        return (pr["W"] @ h).T.cpu()

    def cut(self, ks, sides=("in", "out")):
        pr = dict(self.pr)
        for key, axis, side in self.interfaces:
            if side in sides:
                pr[key] = self.score.cut_irrep(pr[key], axis, self.PROJ, ks)
        return pr

    def energy(self, sides=("in", "out")):
        e = torch.zeros(self.G.R, dtype=torch.float64)
        for key, axis, side in self.interfaces:
            if side in sides:
                ek = self.score.interface_energy_irrep(self.pr[key], axis, self.PROJ).cpu(); e += ek / ek.sum()
        return e / e.sum()

    def norms(self):
        n2 = float(sum((v ** 2).sum() for v in self.pr.values())); return n2, n2 ** 0.5


def kappa_measured(G, fam, Ea, Eb, W, amp):
    """measured price of irrep r in a trained net: (squared weight norm on the r-blocks of e, f, w) / (3 |G| (A_r / c_fam)^{2/3}),
    c_bil = 1, c_quad = 2. Comparable to kappa(): bound d^{1/3} <= kappa_meas; naive junk-free construction d^{4/3}."""
    mass = sum(G.irrep_energy(np.asarray(T, dtype=np.float64)).sum(0) for T in (Ea.numpy(), Eb.numpy(), W.numpy().T))
    c = 2.0 if fam == "quad" else 1.0; mx = max(v for k, v in amp.items() if k != "trivial")
    return {n: float(mass[r] / (3 * G.N * (amp[n] / c) ** (2 / 3))) for r, n in enumerate(G.irrep_names) if n != "trivial" and amp[n] >= 0.2 * mx}


def score_run(G, fam, Ea, Eb, W, name, sides=True):
    import score
    net = Net(G, fam, Ea, Eb, W, name, os.environ.get("GX_SCORE_DEV", "cpu"))   # CPU: fp64 scoring does not queue behind the training kernels
    with torch.no_grad():
        res = score.ablation_count(net, sides_all=sides, log=lambda *_: None)
    res.pop("per", None)
    # per-neuron irrep concentration (share of each neuron's e-energy in its top irrep, energy-weighted)
    en = G.irrep_energy(Ea.double().cpu().numpy())[:, 1:]; tot = en.sum(1) + 1e-30
    res["neuron_top_share"] = float(((en.max(1) / tot) * (tot / tot.sum())).sum())
    res["counted_names"] = [G.irrep_names[r] for r in res.get("counted", [])]
    res["kappa_meas"] = kappa_measured(G, fam, Ea, Eb, W, res["legacy"]["amp"])
    return res


# ------------------------------------------------------------------------------------------------ training
def train(a):
    G = group(a.group); N = G.N; S = len(a.seeds); fam = a.family
    tag = a.tag or f"{a.group}_{fam}_frac{a.frac:g}_lam{a.lam:g}_init{a.init_scale:g}_lrmax{a.lrmax:g}_clip{a.clip:g}_m{a.m}_st{a.steps}" + (f"_wdw{a.wd_warmup}" if a.wd_warmup else "") + (f"_sup-{'-'.join(a.support)}" if a.support else "")
    out = OUT / tag
    todo = [s for s in a.seeds if not (out / f"seed{s}" / "metrics.json").exists()]
    if not todo:
        print("skip", tag); return
    a.seeds = todo; S = len(todo)
    dt = torch.float64 if a.f64 else torch.float32
    Ea, Eb, W = [t.to(DEV, dt) for t in init(a.seeds, a.m, N, a.init_scale)]
    y = torch.from_numpy(G.MULT.reshape(-1)).to(DEV)
    Psup = None
    if a.support:                                                         # fixed-support ladder run: weights projected onto the named irreps after every step
        Psup = torch.from_numpy(sum(G.PROJ[G.irrep_names.index(n)] for n in a.support)).to(DEV, dt)
        Ea, Eb, W = Ea @ Psup, Eb @ Psup, Psup @ W
    R = N * N; wrow = torch.ones(S, R, dtype=dt)
    if a.frac < 1:
        for i, s in enumerate(a.seeds):
            gen = torch.Generator().manual_seed(10_000 + s); keep = torch.randperm(R, generator=gen)[:int(round(a.frac * R))]
            wrow[i] = 0; wrow[i, keep] = 1
    train_mask = wrow.clone().bool()
    wrow = (wrow / wrow.sum(1, keepdim=True)).to(DEV)
    if a.check:
        Ea.requires_grad_(True); Eb.requires_grad_(True); W.requires_grad_(True)
        _, Z = forward(Ea, Eb, W, fam); ce = (torch.nn.functional.cross_entropy(Z, y[None].expand(S, -1), reduction="none") * wrow).sum()
        ga = torch.autograd.grad(ce, (Ea, Eb, W))
        with torch.no_grad():
            _, gEa, gEb, gW, _ = loss_and_grads(Ea, Eb, W, fam, y, wrow)
        print("grad check (max abs diff / max abs):", [(float((x - z).abs().max()), float(z.abs().max())) for x, z in zip((gEa, gEb, gW), ga)]); return
    lr, t0, log = a.lr0, time.time(), []
    lnN = math.log(N); warm = a.wd_warmup > 0; lrcap = a.lrmax
    wd_on = torch.full((S,), not warm, dtype=torch.bool, device=DEV); esc_step = torch.full((S,), -1 if warm else 0, device=DEV)
    print(f"{tag}: |G| {N}, {S} seeds, dev {DEV}", flush=True)
    with torch.no_grad():
        for step in range(1, a.steps + 1):
            if step in LR_DOUBLE_STEPS:
                lr = min(lr * 2, lrcap)
            ce, gEa, gEb, gW, acc = loss_and_grads(Ea, Eb, W, fam, y, wrow)
            if warm:                                                      # per-seed decay-free warm-up (nsum.py --wd_warmup): wd switches on once CE < escape_frac ln|G|
                wd_on |= ce < a.escape_frac * lnN
                if step >= a.wd_warmup: wd_on[:] = True
                newly = wd_on & (esc_step < 0)
                if newly.any(): esc_step[newly] = step
                if bool(wd_on.all()):
                    warm = False; print(f"  warm-up over for every seed at step {step}: {esc_step.tolist()}", flush=True)
            lam_s = (2 * a.lam * wd_on.to(dt))[:, None, None]
            gEa += lam_s * Ea; gEb += lam_s * Eb; gW += lam_s * W
            if a.clip:
                gn = torch.sqrt(gEa.square().sum((1, 2)) + gEb.square().sum((1, 2)) + gW.square().sum((1, 2)))
                f = torch.clamp(a.clip / (gn + 1e-30), max=1.0)[:, None, None]
                gEa *= f; gEb *= f; gW *= f
            Ea -= lr * gEa; Eb -= lr * gEb; W -= lr * gW
            if Psup is not None:
                Ea, Eb, W = Ea @ Psup, Eb @ Psup, Psup @ W
            if step % a.log_every == 0 or step == a.steps:
                n2 = (Ea.square().sum((1, 2)) + Eb.square().sum((1, 2)) + W.square().sum((1, 2)))
                row = dict(step=step, lr=lr, ce=ce.tolist(), acc=acc.tolist(), norm2=n2.tolist(), t=time.time() - t0); log.append(row)
                print(f"  step {step:6d} lr {lr:5.2f} ce med {float(ce.median()):.3e} max {float(ce.max()):.3e} acc(all rows) min {float(acc.min()):.4f} "
                      f"med {float(acc.median()):.4f} |theta|^2 med {float(n2.median()):.1f} [{time.time() - t0:.0f}s]", flush=True)
    train_time = time.time() - t0
    from safetensors.torch import save_file
    for i, s in enumerate(a.seeds):                                       # weights first: a killed scorer loses nothing
        d = out / f"seed{s}"; d.mkdir(parents=True, exist_ok=True)
        save_file({"Ea": Ea[i].float().cpu(), "Eb": Eb[i].float().cpu(), "W": W[i].float().cpu()}, str(d / "final.safetensors"))
    torch.set_num_threads(int(os.environ.get("GX_SCORE_THREADS", "4")))
    for i, s in enumerate(a.seeds):
        d = out / f"seed{s}"; d.mkdir(parents=True, exist_ok=True)
        ea, eb, w = Ea[i].float().cpu(), Eb[i].float().cpu(), W[i].float().cpu()
        if not all(torch.isfinite(t).all() for t in (ea, eb, w)):
            res = dict(acc=0.0, m_min=float("nan"), K_abl=None, nonfinite=True)
        else:
            res = score_run(G, fam, ea, eb, w, f"{tag}/seed{s}", sides=not a.no_sides)
            with torch.no_grad():                                       # train / held-out accuracy for the frac arm
                _, Z = forward(Ea[i:i + 1], Eb[i:i + 1], W[i:i + 1], fam); ok = (Z.argmax(1)[0] == y).cpu()
                tm = train_mask[i]; res["acc_train"] = float(ok[tm].float().mean()); res["acc_test"] = float(ok[~tm].float().mean()) if (~tm).any() else None
        res.update(group=a.group, family=fam, seed=s, frac=a.frac, lam=a.lam, m=a.m, steps=a.steps, lrmax=a.lrmax, init_scale=a.init_scale,
                   clip=a.clip, support=a.support, wd_warmup=a.wd_warmup, escape_step=int(esc_step[i]), train_time=train_time, final_ce=log[-1]["ce"][i], cell=tag,
                   log=[dict(step=r["step"], ce=r["ce"][i], acc=r["acc"][i], norm2=r["norm2"][i]) for r in log])
        (d / "metrics.json").write_text(json.dumps(res, indent=0, default=float))
        save_file({"Ea": ea, "Eb": eb, "W": w}, str(d / "final.safetensors"))
        print(f"DONE {tag} seed{s}: acc {res.get('acc', 0):.4f} m_min {res.get('m_min', float('nan')):.4f} gamma {res.get('gamma2', float('nan')):.3e} "
              f"K_abl {res.get('K_abl')} {res.get('counted_names')} leader {res.get('leader')} share {res.get('lead_share')} sign_rel {res.get('sign_rel')} "
              f"K_thr10 {res.get('legacy', {}).get('K_thr10')} clean {res.get('legacy', {}).get('clean_frac', float('nan')):.3f}", flush=True)


# ------------------------------------------------------------------------------------------------ predictions (nu = 3)
def kappa(G, r, mode):
    """cost weight of irrep r: reaching clean amplitude A_r (coefficient of chi_r / d_r) costs kappa_r A_r^{2/3} (x 3|G|).
    'bound'  : per-irrep AM-GM upper bound, d^{1/3} (attained by ONE neuron with rank-1 Fourier blocks; junk not cancelled)
    'naive'  : the junk-free matrix-coefficient construction (d^3 neurons rho_ij (x) rho_jk (x) rho_ik): d^{4/3} for
               real-type irreps; 2 for the complex-type 2-dim real irreps of abelian groups (4 cos/sin neurons)."""
    d = float(G.irrep_dims[r]); cplx = G.schur[r] == 2
    if mode == "bound":
        return d ** (1 / 3)
    return 2.0 if cplx else d ** (4 / 3)


def best_amplitudes(cov, kap, restarts=6, rng=None):
    """max over A >= 0 of min_x (cov^T A) / (sum kap A^{2/3})^{3/2} on a fixed support. cov (K, X)."""
    from scipy.optimize import minimize
    K = cov.shape[0]
    if K == 1:
        return float(cov[0].min() / kap[0] ** 1.5), np.ones(1)
    rng = rng or np.random.default_rng(0)
    def neg(la):
        A = np.exp(la - la.max()); return -float((A @ cov).min() / (kap @ A ** (2 / 3)) ** 1.5)
    best, bA = -1, None
    for t in range(restarts):
        x0 = np.zeros(K) if t == 0 else rng.normal(0, 1.0, K)
        r = minimize(neg, x0, method="Nelder-Mead", options=dict(xatol=1e-6, fatol=1e-12, maxiter=4000))
        if -r.fun > best: best, bA = -r.fun, np.exp(r.x - r.x.max())
    return best, bA / bA.max()


def predict(a):
    out = {}
    for gname in a.groups.split(","):
        G = group(gname); cov_full = G.coverage(); nt = [r for r in range(G.R) if G.irrep_names[r] != "trivial"]
        cls = np.unique(G.class_of[np.arange(G.N) != G.e]) if G.R <= 10 else None
        xs = [int(np.flatnonzero(G.class_of == c)[0]) for c in cls] if cls is not None else [i for i in range(G.N) if i != G.e]
        cov = cov_full[:, xs]; res = {}
        print(f"\n== {gname}: |G| {G.N}, nontrivial irreps {len(nt)}")
        for mode in ("bound", "naive"):
            kap = np.array([kappa(G, r, mode) for r in range(G.R)]); rows = []
            if len(nt) <= 8:                                          # weighted ladder: every support, free amplitudes
                for K in range(1, len(nt) + 1):
                    for Sup in itertools.combinations(nt, K):
                        Sup = list(Sup)
                        if cov[Sup].sum(0).min() <= 1e-12: continue
                        g, A = best_amplitudes(cov[Sup], kap[Sup])
                        if A.min() < 1e-3: continue                   # a member at zero amplitude = a smaller support
                        rows.append(dict(support=[G.irrep_names[r] for r in Sup], K=K, gamma_hat=g, amps=[round(float(x), 3) for x in A]))
            else:                                                     # abelian: equal-share sets by local search (kstar_ab), then free amplitudes
                from kstar_ab import best_subset
                rng = np.random.default_rng(0); gnt = cov[nt]
                for K in range(1, a.kmax + 1):
                    MK, Sidx, how = best_subset(gnt, K, rng, restarts=a.restarts); Sup = [nt[i] for i in Sidx]
                    if MK <= 1e-9: continue
                    g_eq = MK / (kap[Sup].sum()) ** 1.5
                    g, A = best_amplitudes(cov[Sup], kap[Sup], restarts=3)
                    rows.append(dict(support=[G.irrep_names[r] for r in Sup], K=K, M_K=MK, gamma_eq=g_eq, gamma_hat=max(g, g_eq), amps=[round(float(x), 3) for x in A], how=how))
            rows.sort(key=lambda r: -r["gamma_hat"]); top = rows[0]["gamma_hat"]
            byK = {}
            for r in rows: byK.setdefault(r["K"], r)
            plateau = sorted(K for K, r in byK.items() if r["gamma_hat"] >= 0.74 * top)
            res[mode] = dict(argmax=rows[0], plateau74=plateau, top=rows[:12], byK={K: byK[K] for K in sorted(byK)})
            print(f"  [{mode}] argmax K = {rows[0]['K']} {rows[0]['support']} amps {rows[0]['amps']} gamma_hat {top:.4f}; plateau(0.74) K in {plateau}")
            for r in rows[:8]:
                print(f"      {r['gamma_hat'] / top:5.3f}  K {r['K']}  {r['support']}  amps {r['amps']}")
            print("      best per K: " + "  ".join(f"{K}:{byK[K]['gamma_hat'] / top:.2f}" for K in sorted(byK)))
        out[gname] = res
    Path(a.out).write_text(json.dumps(out, indent=1, default=float)); print("wrote", a.out)


def kappa_of(r, G):
    if "kappa_meas" in r: return r["kappa_meas"]
    from safetensors.torch import load_file
    f = OUT_TABLE / r["cell"] / f"seed{r['seed']}" / "final.safetensors"
    if not f.exists(): return {}
    st = load_file(str(f)); return kappa_measured(G, r["family"], st["Ea"], st["Eb"], st["W"], r["legacy"]["amp"])


def table(a):
    global OUT_TABLE; OUT_TABLE = Path(a.runs)
    import collections
    med = lambda xs: float(np.median(xs)) if len(xs) else float("nan")
    cells = collections.OrderedDict()
    for f in sorted(Path(a.runs).glob("*/seed*/metrics.json")):
        cells.setdefault(f.parent.parent.name, []).append(json.load(open(f)))
    for cell, rs in cells.items():
        G = group(rs[0]["group"]); nonab = G.R <= 10
        sep = [r for r in rs if r.get("K_abl") is not None]; n = len(rs)
        H = lambda key, src=sep: dict(sorted(collections.Counter(r[key] for r in src).items(), key=lambda kv: str(kv[0])))
        print(f"\n== {cell}: n {n}, acc=1 {sum(r.get('acc', 0) >= 0.9999 for r in rs)}, separated {len(sep)}, acc med {med([r.get('acc', 0) for r in rs]):.4f}"
              + (f", test acc med {med([r['acc_test'] for r in rs if r.get('acc_test') is not None]):.3f}" if rs[0]["frac"] < 1 else "")
              + (f", escape step med {med([r.get('escape_step', 0) for r in rs]):.0f}" if rs[0].get("wd_warmup") else ""))
        if not sep: continue
        print(f"   K_abl {H('K_abl')}  K_thr10 {dict(sorted(collections.Counter(r['legacy']['K_thr10'] for r in sep).items()))}  K_load90 {dict(sorted(collections.Counter(r['legacy']['K_load90'] for r in sep).items()))}"
              f"  k_eff med {med([r['legacy']['k_eff'] for r in sep]):.2f}")
        print(f"   gamma med {med([r['gamma2'] for r in sep]):.3e} max {max(r['gamma2'] for r in sep):.3e}  m_min med {med([r['m_min'] for r in sep]):.3f}  clean med {med([r['legacy']['clean_frac'] for r in sep]):.3f}"
              f"  neuron top-irrep share med {med([r['neuron_top_share'] for r in sep]):.3f}")
        def Ke(r, t):                                                     # RESULTS_QM instrument: irreps holding >= t of the clean-logit energy (A_r^2 s_r / d_r^2 x d_r^2 = ||P_r f||^2)
            e = np.array([(v / G.irrep_dims[G.irrep_names.index(k)]) ** 2 * G.schur[G.irrep_names.index(k)] * G.irrep_dims[G.irrep_names.index(k)] ** 2 for k, v in r["legacy"]["amp"].items() if k != "trivial"])
            return int((e / e.sum() >= t).sum())
        print(f"   K_2pct {dict(sorted(collections.Counter(Ke(r, 0.02) for r in sep).items()))}  K_5pct {dict(sorted(collections.Counter(Ke(r, 0.05) for r in sep).items()))}  secs/score med {med([r.get('secs', 0) for r in sep]):.0f}")
        if nonab:
            print(f"   leaders {H('leader')}  one-irrep>=0.9: {sum(r['lead_share'] >= 0.9 for r in sep)}/{len(sep)}  >=2 faithful above 0.2: {sum(r['n_faithful_02'] >= 2 for r in sep)}/{len(sep)}")
            for L in sorted(set(r["leader"] for r in sep)):
                g = [r for r in sep if r["leader"] == L]
                print(f"      {L:9} n {len(g):2d}  gamma med {med([r['gamma2'] for r in g]):.3e} max {max(r['gamma2'] for r in g):.3e}  share med {med([r['lead_share'] for r in g]):.2f}"
                      f"  sign_rel med {med([r['sign_rel'] for r in g if r.get('sign_rel') is not None]):.3f}  K_abl {H('K_abl', g)}")
            km = collections.defaultdict(list)
            for r in sep:
                for k, v in kappa_of(r, G).items(): km[k].append(v)
            print("      kappa_meas (median; bound d^1/3, naive d^4/3): " + "  ".join(f"{k}: {med(v):.2f} (n {len(v)}; {G.irrep_dims[G.irrep_names.index(k)] ** (1 / 3):.2f}, {G.irrep_dims[G.irrep_names.index(k)] ** (4 / 3):.2f})" for k, v in km.items()))
            for r in sorted(sep, key=lambda r: -r["gamma2"]):
                A = r["legacy"]["amp"]; mx = max(v for k, v in A.items() if k != "trivial")
                print(f"      s{r['seed']:<2d} gamma {r['gamma2']:.3e} K_abl {r['K_abl']} {r['counted_names']}  amps " + " ".join(f"{k}:{v / mx:.2f}" for k, v in A.items() if k != "trivial"))
        else:
            names = collections.Counter(nm for r in sep for nm in r["counted_names"])
            print(f"   sign-character members: {sum(any(nm.startswith('sgn') for nm in r['counted_names']) for r in sep)}/{len(sep)}")
            if G.name == "z120":
                gc = collections.Counter(math.gcd(int(nm[1:]), 120) if nm.startswith("f") else 60 for r in sep for nm in r["counted_names"])
                print(f"   gcd(k, 120) of counted members {dict(sorted(gc.items()))}; runs with a gcd>1 member {sum(any((not nm.startswith('f')) or math.gcd(int(nm[1:]), 120) > 1 for nm in r['counted_names']) for r in sep)}/{len(sep)}")
            for r in sorted(sep, key=lambda r: -r["gamma2"])[:4]:
                print(f"      s{r['seed']:<2d} gamma {r['gamma2']:.3e} K_abl {r['K_abl']} {r['counted_names']} top8 {r['legacy']['top8']}")


def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--group", required=True); t.add_argument("--family", required=True, choices=["quad", "bil"])
    t.add_argument("--seeds", type=int, nargs="+", default=list(range(8)))
    t.add_argument("--frac", type=float, default=1.0); t.add_argument("--lam", type=float, default=1e-4)
    t.add_argument("--m", type=int, default=500); t.add_argument("--steps", type=int, default=40000)
    t.add_argument("--lr0", type=float, default=0.05); t.add_argument("--lrmax", type=float, default=3.2)
    t.add_argument("--init_scale", type=float, default=4.0); t.add_argument("--clip", type=float, default=0.0)
    t.add_argument("--wd_warmup", type=int, default=0); t.add_argument("--escape_frac", type=float, default=0.8)
    t.add_argument("--support", nargs="*", default=None, help="fixed-support run: irrep names"); t.add_argument("--log_every", type=int, default=2000); t.add_argument("--tag", default=None)
    t.add_argument("--no_sides", action="store_true"); t.add_argument("--check", action="store_true"); t.add_argument("--f64", action="store_true")
    p = sub.add_parser("predict")
    p.add_argument("--groups", default="s5,s4,a5,z120,z11x11,z113,z71"); p.add_argument("--kmax", type=int, default=16)
    p.add_argument("--restarts", type=int, default=200); p.add_argument("--out", default=str(HERE / "kstar_gx.json"))
    tb = sub.add_parser("table"); tb.add_argument("--runs", default=str(OUT))
    a = ap.parse_args()
    {"train": train, "predict": predict, "table": table}[a.cmd](a)


if __name__ == "__main__":
    main()
