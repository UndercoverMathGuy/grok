"""nu-dial cohort: does the Fourier committee size track homogeneity degree?

Batched lockstep trainer for one-hidden-layer MLPs on a+b mod p whose
homogeneity degree nu is dialled by inserting LINEAR layers (prereg:
nudial/README.md). Same conventions as cloud/train_semifinal_torch.py:
M is the only batch dim, MLX-semantics AdamW (no bias correction, decoupled
wd before the step, per-run lr/wd vectors), stable-f32 CE, torch.compile
reduce-overhead (CUDA graphs), and the IDENTICAL final-logit Fourier
instrument (coeffs of cos(k(a+b-c)); committee_from_coeffs gap detector).

Architectures (shared embedding E for both tokens; act after U/V):
  relu  nu = 2 + n_pre + n_post      onehot -> [E1 -> E2] -> U,V -> ReLU -> W [-> W2 -> W3]
  quad  nu = 2(1 + n_pre) + 1 + n_post   same with x^2
Modes:
  train   CE + decoupled l2 weight decay (cohorts A, B-control)
  l2nu    CE + lam * sum_j ||neuron_j||^nu  (Morwani's L_{2,nu} regulariser;
          1-hidden-layer nets only)                                  (cohort B)
  margin  no data split: maximise the softmin normalised margin
          min_i g_i(theta) / ||theta||^nu directly                    (cohort C)

Run dirs: runs_nudial/<cohort>/<family><nu>_m<m>_p<p>_<mode>/wd<wd>_s<seed>/
  config.json, spectra.npz (epochs, coeffs, energy, train_acc, test_acc),
  metrics.json (train/test curves, final normalised margin), final.safetensors
Smoke (CPU, ~1 min): python3 nudial/train_nudial.py --smoke
"""
import argparse, json, math, os, random, sys, time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[1]
RUNS = Path(os.environ.get("RUNS_DIR", ROOT / "runs_nudial"))
META_SEED = 20260822
torch.backends.cuda.matmul.allow_tf32 = False      # user veto: exact fp32
torch.backends.cudnn.allow_tf32 = False


# --------------------------------------------------------------------------
@dataclass
class Config:
    p: int = 113
    family: str = "relu"        # relu | quad
    nu: int = 2
    m: int = 512                # hidden width
    d: int = 128                # embedding width (pre layers)
    r: int = 128                # readout-stack width (post layers)
    frac_train: float = 0.5
    data_seed: int = 0
    init_seed: int = 0
    mode: str = "train"         # train | l2nu | margin
    lr: float = 1e-3
    weight_decay: float = 1.0   # decoupled (mode train); ignored otherwise
    lam: float = 0.0            # L_{2,nu} coefficient (mode l2nu)
    beta_max: float = 200.0     # softmin sharpness at end of run (mode margin)
    warmup_steps: int = 10
    betas: tuple = (0.9, 0.98)
    adam_eps: float = 1e-8
    num_epochs: int = 20_000
    opt: str = "adamw"          # adamw | gd  (gd: plain full-batch GD, coupled l2 penalty wd/2*||theta||^2)
    init_from: str = ""         # path to a final.safetensors to warm-start from (e.g. cohort J)
    layout_pre: int = -1        # >=0: explicit (n_pre, n_post) override (nu is then derived)
    layout_post: int = -1
    init_scale: float = 1.0     # std = init_scale / sqrt(fan_in)

    @property
    def layout(self):
        """(n_pre, n_post) realising nu for the family."""
        if self.layout_pre >= 0:
            return (self.layout_pre, self.layout_post)
        if self.family == "relu":
            table = {2: (0, 0), 3: (1, 0), 4: (1, 1), 5: (2, 1), 6: (2, 2)}
        else:
            table = {3: (0, 0), 4: (0, 1), 5: (0, 2), 6: (1, 1), 7: (1, 2)}
        return table[self.nu]

    @property
    def degree(self):
        n_pre, n_post = self.layout
        return (2 + n_pre + n_post) if self.family == "relu" else (2 * (1 + n_pre) + 1 + n_post)

    def save(self, path):
        dd = asdict(self); dd["betas"] = list(dd["betas"])
        Path(path).write_text(json.dumps(dd, indent=2))


def train_test_split(cfg):
    """Same stdlib-random split as the transformer trainer."""
    p = cfg.p
    pairs = [(i, j) for i in range(p) for j in range(p)]
    random.seed(cfg.data_seed); random.shuffle(pairs)
    tr = set(pairs[:int(cfg.frac_train * len(pairs))])
    return np.array([(i, j) in tr for i in range(p) for j in range(p)])


def fourier_basis(p):
    basis = [np.ones(p) / np.sqrt(p)]
    for k in range(1, p // 2 + 1):
        for trig in (np.cos, np.sin):
            v = trig(2 * np.pi * k * np.arange(p) / p)
            basis.append(v / np.linalg.norm(v))
    return np.stack(basis).astype(np.float64)


def committee_from_coeffs(coeffs, floor=0.02):
    a = np.abs(coeffs); order = np.argsort(a)[::-1]
    logs = np.log(a[order] + 1e-12)
    cut = int(np.argmax(logs[:-1][:12] - logs[1:][:12])) + 1
    mem = order[:cut] + 1
    mem = mem[a[mem - 1] >= floor * a.max()]
    return sorted(mem.tolist())


# --------------------------------------------------------------------------
def param_shapes(cfg):
    n_pre, n_post = cfg.layout
    p, m, d, r = cfg.p, cfg.m, cfg.d, cfg.r
    sh = {}
    fan = p
    for i in range(n_pre):
        sh[f"E{i+1}"] = (d, fan); fan = d
    sh["U"] = (m, fan); sh["V"] = (m, fan)
    fan = m
    for i in range(n_post):
        sh[f"W{i+1}"] = (r, fan); fan = r
    sh["W"] = (p, fan)
    return sh


def init_params(cfg):
    g = torch.Generator().manual_seed(cfg.init_seed)
    return {k: cfg.init_scale * torch.randn(*s, generator=g) / math.sqrt(s[1])
            for k, s in param_shapes(cfg).items()}


def _cols(W, idx):
    """W (M, out, p); idx (N,) shared or (M, N) per-run -> (M, out, N)."""
    if idx.dim() == 1:
        return W[:, :, idx]
    return W.gather(2, idx[:, None, :].expand(-1, W.shape[1], -1))


def forward(pr, A, B, cfg):
    """pr {key: (M, out, in)}; A, B (N,) or (M, N) int64 -> logits (M, N, p)."""
    n_pre, n_post = cfg.layout
    if n_pre == 0:
        xa, xb = _cols(pr["U"], A), _cols(pr["V"], B)          # (M, m, N)
    else:
        ea, eb = _cols(pr["E1"], A), _cols(pr["E1"], B)        # (M, d, N)
        for i in range(1, n_pre):
            ea, eb = pr[f"E{i+1}"] @ ea, pr[f"E{i+1}"] @ eb
        xa, xb = pr["U"] @ ea, pr["V"] @ eb
    h = xa + xb
    h = F.relu(h) if cfg.family == "relu" else h * h
    for i in range(n_post):
        h = pr[f"W{i+1}"] @ h
    return (pr["W"] @ h).transpose(1, 2)                       # (M, N, p)


def per_ex_stable_f32(logits, labels):
    z_y = logits.gather(-1, labels.unsqueeze(-1))
    others = logits.masked_fill(
        F.one_hot(labels, logits.shape[-1]).bool(), float("-inf"))
    u = torch.logsumexp(others, dim=-1, keepdim=True) - z_y
    return F.softplus(u).squeeze(-1)


def per_ex_margin(logits, labels):
    z_y = logits.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    others = logits.masked_fill(
        F.one_hot(labels, logits.shape[-1]).bool(), float("-inf"))
    return z_y - others.max(-1).values                          # (M, N)


def sq_norm(pr):
    return sum((v ** 2).flatten(1).sum(1) for v in pr.values())   # (M,)


def l2nu_penalty(pr, nu):
    """Morwani L_{2,nu}^nu: sum over hidden neurons j of ||(u_j, v_j, w_j)||^nu."""
    n2 = (pr["U"] ** 2).sum(2) + (pr["V"] ** 2).sum(2) + (pr["W"] ** 2).sum(1)
    return (n2 ** (nu / 2)).sum(1)                              # (M,)


# --------------------------------------------------------------------------
def train_batched(cfgs, run_dirs, compile_mode="reduce-overhead",
                  log_every=500, spectra_every=100, device=None):
    M = len(cfgs); c0 = cfgs[0]
    for c in cfgs:
        for f in ("p", "family", "m", "d", "r", "frac_train", "mode", "layout", "opt",
                  "warmup_steps", "betas", "adam_eps", "num_epochs"):
            assert getattr(c, f) == getattr(c0, f), f"batch must agree on {f}"
        if c.mode == "l2nu":
            assert c.layout == (0, 0), "L_{2,nu} is defined for 1-hidden-layer nets"
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    run_dirs = [Path(rd) for rd in run_dirs]
    for cfg, rd in zip(cfgs, run_dirs):
        rd.mkdir(parents=True, exist_ok=True); cfg.save(rd / "config.json")
    p, nu = c0.p, c0.degree
    assert all(c.degree == nu for c in cfgs)
    keys = list(param_shapes(c0))
    per_run = []
    for c in cfgs:
        if c.init_from:
            from safetensors.torch import load_file
            per_run.append({k: v.float() for k, v in load_file(c.init_from).items()})
        else:
            per_run.append(init_params(c))
    pr = {k: torch.stack([r[k] for r in per_run]).to(device).requires_grad_()
          for k in keys}
    params = [pr[k] for k in keys]
    if c0.mode == "margin":
        # constrain theta to the sphere of its INITIAL radius (per run): the
        # normalised margin g/||theta||^nu is radius-independent, and keeping
        # the natural 1/sqrt(fan_in) coordinate scale keeps Adam's lr sane
        # (on the unit sphere every coordinate is ~1e-3 and ReLUs die).
        R2 = sq_norm(pr).detach()

    # data
    a_np = np.repeat(np.arange(p), p); b_np = np.tile(np.arange(p), p)
    lab_np = (a_np + b_np) % p
    A_all = torch.from_numpy(a_np).to(device); B_all = torch.from_numpy(b_np).to(device)
    lab_all = torch.from_numpy(lab_np).to(device)
    if c0.mode == "margin" and c0.frac_train >= 1.0:
        masks = [np.ones(p * p, bool) for _ in cfgs]
    else:
        masks = [train_test_split(c) for c in cfgs]
    n_tr = {int(mk.sum()) for mk in masks}
    assert len(n_tr) == 1, "equal-count batches only"
    tr_idx = torch.from_numpy(np.stack([np.flatnonzero(mk) for mk in masks])).to(device)
    te_idx = torch.from_numpy(np.stack([np.flatnonzero(~mk) for mk in masks])).to(device)
    mask_t = torch.from_numpy(np.stack(masks)).to(device)
    has_test = te_idx.shape[1] > 0

    # optimiser state (MLX-semantics AdamW)
    m_st = [torch.zeros_like(q) for q in params]
    v_st = [torch.zeros_like(q) for q in params]
    b1, b2 = c0.betas; eps = c0.adam_eps
    lr_base = torch.tensor([c.lr for c in cfgs], device=device)
    wd_v = torch.tensor([c.weight_decay if c.mode == "train" else 0.0
                         for c in cfgs], device=device)
    GD = c0.opt == "gd"
    lam_v = torch.tensor([c.lam for c in cfgs], device=device)
    beta_v = torch.tensor([c.beta_max for c in cfgs], device=device)
    lr_t = torch.zeros(M, device=device); beta_t = torch.ones(M, device=device)
    phase_t = torch.zeros(M, device=device)
    bshape = {id(q): (-1,) + (1,) * (q.dim() - 1) for q in params}

    A_tr, B_tr, lab_tr = A_all[tr_idx], B_all[tr_idx], lab_all[tr_idx]   # (M, N_tr)

    def objective():
        if c0.mode == "margin":
            # params live on a fixed-radius sphere (renormalised every step), so
            # g is the normalised margin up to a constant; softmin on a
            # scale-free copy so beta
            # means the same thing at every nu (margins are ~1e-7 at nu=6)
            # two phases (phase_t 0 -> 1 at 60% of the run): CE on the sphere
            # first (its minimiser direction is the max-margin direction and it
            # is well conditioned from random init), then softmin-margin
            # refinement warm-started from the separated solution.
            L = forward(pr, A_tr, B_tr, c0)                     # (M, N_tr, p): full grid iff frac=1
            labs = lab_tr
            ce = per_ex_stable_f32(L, labs).mean(1)
            g = per_ex_margin(L, labs)
            scale = g.abs().mean(1, keepdim=True).detach() + 1e-30
            gn = g / scale
            sm = (torch.logsumexp(-beta_t[:, None] * gn, 1)
                  - math.log(g.shape[1])) / beta_t              # soft -min margin
            return (1 - phase_t) * ce + phase_t * sm
        Ltr = forward(pr, A_tr, B_tr, c0)                       # train examples only
        per = per_ex_stable_f32(Ltr, lab_tr).mean(1)
        if c0.mode == "l2nu":
            per = per + lam_v * l2nu_penalty(pr, nu)
        return per

    def step():
        per = objective()
        grads = torch.autograd.grad(per.sum(), params)
        with torch.no_grad():
            for q, g, m_, v_ in zip(params, grads, m_st, v_st):
                sh = bshape[id(q)]
                lr_, wd_ = lr_t.view(sh), wd_v.view(sh)
                if GD:                                   # plain GD on CE + wd/2 ||theta||^2
                    q.sub_(lr_ * (g + wd_ * q))
                    continue
                m_.mul_(b1).add_(g, alpha=1 - b1)
                v_.mul_(b2).addcmul_(g, g, value=1 - b2)
                q.mul_(1 - lr_ * wd_)
                q.sub_(lr_ * m_ / (v_.sqrt() + eps))
            if c0.mode == "margin":                         # project to the sphere
                fac = (R2 / sq_norm(pr)).sqrt()
                for q in params:
                    q.mul_(fac.view(bshape[id(q)]))
        return per.detach()

    if compile_mode != "off":
        step = torch.compile(step, mode=None if compile_mode == "default" else compile_mode)

    # spectra instrument (identical to the transformer trainer's 'add' path)
    basis = fourier_basis(p); n_ = p // 2
    bc, bs = basis[1::2], basis[2::2]
    u_cos = torch.from_numpy((np.einsum("np,nq->npq", bc, bc) - np.einsum("np,nq->npq", bs, bs))
                             .reshape(n_, -1) / np.sqrt(2)).to(device)
    u_sin = torch.from_numpy((np.einsum("np,nq->npq", bs, bc) + np.einsum("np,nq->npq", bc, bs))
                             .reshape(n_, -1) / np.sqrt(2)).to(device)
    b_cos = torch.from_numpy(bc).to(device); b_sin = torch.from_numpy(bs).to(device)
    snap = {"epochs": [], "coeffs": [], "energy": [], "train_acc": [], "test_acc": [],
            "pnorm2": [], "min_margin_norm": []}

    @torch.no_grad()
    def take_snapshot(epoch):
        L = forward(pr, A_all, B_all, c0)                       # (M, p^2, p)
        acc = L.argmax(-1) == lab_all
        snap["epochs"].append(epoch)
        snap["train_acc"].append((acc & mask_t).sum(-1).double() / mask_t.sum(-1))
        snap["test_acc"].append((acc & ~mask_t).sum(-1).double() / (~mask_t).sum(-1).clamp(min=1))
        Ld = L.double()
        c = torch.einsum("fp,mpo->mfo", u_cos, Ld); s = torch.einsum("fp,mpo->mfo", u_sin, Ld)
        snap["energy"].append((c ** 2).sum(-1) + (s ** 2).sum(-1))
        snap["coeffs"].append(((c * b_cos).sum(-1) + (s * b_sin).sum(-1)) / math.sqrt(2))
        snap["pnorm2"].append(sq_norm(pr).detach())
        g = per_ex_margin(L, lab_all[None].expand(M, -1)).min(1).values
        snap["min_margin_norm"].append(g / sq_norm(pr).detach() ** (nu / 2))

    E = c0.num_epochs
    train_hist = torch.empty(E, M, device=device); test_hist = []
    print(f"batch {M} on {device}: {c0.family} nu={nu} m={c0.m} p={p} mode={c0.mode} "
          f"layout={c0.layout} train {tr_idx.shape[1]} test {te_idx.shape[1]}", flush=True)
    t0 = time.time()
    for epoch in range(E):
        if spectra_every and epoch % spectra_every == 0:
            take_snapshot(epoch)
        lr_t.copy_(lr_base * min(epoch / c0.warmup_steps, 1.0))
        ph = max(0.0, (epoch - 0.6 * E) / (0.4 * E))          # 0 until 60%, then ramps
        phase_t.fill_(1.0 if epoch >= 0.6 * E else 0.0)
        beta_t.copy_(1.0 + (beta_v - 1.0) * min(ph, 1.0))
        train_hist[epoch].copy_(step())
        if (epoch + 1) % log_every == 0:
            with torch.no_grad():
                tr = train_hist[epoch].cpu().numpy()
                if has_test and c0.mode != "margin":
                    L = forward(pr, A_all, B_all, c0)
                    Lte = L.gather(1, te_idx.unsqueeze(-1).expand(-1, -1, p))
                    te = per_ex_stable_f32(Lte, lab_all[te_idx]).mean(1).cpu().numpy()
                else:
                    te = np.full(M, np.nan)
            test_hist.append(te)
            eps_s = (epoch + 1) / (time.time() - t0)
            print(f"epoch {epoch+1:6d}  obj {tr.mean():.4e}  test med {np.nanmedian(te):.4e} "
                  f"max {np.nanmax(te):.4e}  ({eps_s:.1f} eps/s x {M} = {eps_s*M:.0f} run-eps/s)",
                  flush=True)
    take_snapshot(E)
    train_all = train_hist.cpu().numpy(); test_all = np.stack(test_hist) if test_hist else np.zeros((0, M))
    sp = {k: torch.stack(snap[k]).cpu().numpy() for k in snap if k != "epochs"}
    for i, rd in enumerate(run_dirs):
        save_file({k: v[i].detach().cpu().contiguous() for k, v in pr.items()},
                  str(rd / "final.safetensors"))
        np.savez(rd / "spectra.npz", epochs=np.array(snap["epochs"]),
                 **{k: v[:, i] for k, v in sp.items()})
        (rd / "metrics.json").write_text(json.dumps({
            "train_losses": train_all[:, i].tolist(),
            "test_losses": test_all[:, i].tolist(), "test_every": log_every,
            "final_K_gap": committee_from_coeffs(sp["coeffs"][-1, i]),
            "final_min_margin_norm": float(sp["min_margin_norm"][-1, i])}))
    print(f"done: {E} epochs x {M} runs in {time.time()-t0:.0f}s", flush=True)


# --------------------------------------------------------------------------
def draw_seeds(n):
    rng = random.Random(META_SEED)
    return sorted(rng.sample(range(200_000, 300_000), n))


def plan(args):
    """[(cohort, cfg_template_kwargs, per-run list of (wd, lam, seed))]"""
    seeds = draw_seeds(args.seeds)
    jobs = []
    if args.cohort in ("A", "all"):
        for m in args.widths:                    # width-outer: m=512 cells land first
            for fam, nus in (("relu", [2, 3, 4, 5, 6]), ("quad", [3, 4, 5])):
                for nu in nus:
                    ss = seeds[:args.seeds_2048] if m >= 2048 else seeds
                    runs = [dict(weight_decay=wd, init_seed=s, data_seed=s % 1000)
                            for wd in args.wds for s in ss]
                    jobs.append(("A", dict(family=fam, nu=nu, m=m, mode="train"), runs))
    if args.cohort in ("B", "all"):
        runs = [dict(lam=lam, init_seed=s, data_seed=s % 1000)
                for lam in args.lams for s in seeds]
        jobs.append(("B", dict(family="quad", nu=3, m=512, mode="l2nu"), runs))
    if args.cohort in ("C", "all"):
        for fam, nu in (("relu", 2), ("quad", 3), ("relu", 4)):
            for p in args.margin_ps:
                runs = [dict(init_seed=s) for s in seeds]
                jobs.append(("C", dict(family=fam, nu=nu, m=512, p=p, mode="margin",
                                       frac_train=1.0), runs))
    if args.cohort in ("D", "all2"):          # C rerun with the sphere-projected optimiser
        for fam, nu in (("relu", 2), ("relu", 4), ("relu", 5), ("relu", 6), ("quad", 3), ("quad", 4)):
            runs = [dict(init_seed=s) for s in seeds]
            jobs.append(("D", dict(family=fam, nu=nu, m=512, p=113, mode="margin",
                                   frac_train=1.0), runs))
    if args.cohort in ("E", "all2"):          # degree placement / sharing at fixed nu
        for fam, pre, post in (("quad", 1, 0), ("quad", 0, 2),            # nu=5 both
                               ("relu", 2, 0), ("relu", 0, 2), ("relu", 1, 1)):  # nu=4 all
            runs = [dict(weight_decay=wd, init_seed=s, data_seed=s % 1000)
                    for wd in args.wds for s in seeds]
            jobs.append(("E", dict(family=fam, nu=0, layout_pre=pre, layout_post=post,
                                   m=512, mode="train"), runs))
    if args.cohort in ("G", "all3"):          # +1 vs data fraction (memorisation-remnant test)
        for fam, nu in (("relu", 4), ("relu", 6)):
            for frac in args.fracs:
                runs = [dict(weight_decay=wd, init_seed=s, data_seed=s % 1000)
                        for wd in args.wds for s in seeds]
                jobs.append(("G", dict(family=fam, nu=nu, m=512, mode="train", frac_train=frac), runs))
    if args.cohort in ("H", "all3"):          # optimum on the frac=0.5 SUBSET
        for fam, nu in (("relu", 4), ("relu", 6)):
            runs = [dict(init_seed=s, data_seed=s % 1000) for s in seeds]
            jobs.append(("H", dict(family=fam, nu=nu, m=512, p=113, mode="margin", frac_train=0.5), runs))
    if args.cohort in ("I", "all4"):          # plain GD + coupled l2, full data (is the +1 Adam's geometry?)
        for fam, nu in (("relu", 4), ("relu", 6)):
            runs = [dict(opt="gd", lr=lr, weight_decay=wd, init_seed=s, data_seed=s % 1000)
                    for lr in args.gd_lrs for wd in args.gd_wds for s in seeds[:8]]
            jobs.append(("I", dict(family=fam, nu=nu, m=512, mode="train", frac_train=1.0,
                                   warmup_steps=1), runs))
    if args.cohort in ("J", "all4"):          # margin refinement warm-started from trained AdamW nets (G frac 1.0)
        for fam, nu in (("relu", 4), ("relu", 6)):
            src = sorted((RUNS / "G" / f"{fam}{nu}_m512_p113_f1_train").glob("wd1_s*/final.safetensors"))
            runs = [dict(init_from=str(f), init_seed=int(f.parent.name.split("_s")[1])) for f in src[:32]]
            if runs:
                jobs.append(("J", dict(family=fam, nu=nu, m=512, p=113, mode="margin", frac_train=1.0), runs))
    if args.cohort in ("F", "all2"):          # init-scale sweep at wd=1
        for fam, nu in (("relu", 4), ("quad", 3)):
            runs = [dict(init_scale=sc, weight_decay=1.0, init_seed=s, data_seed=s % 1000)
                    for sc in args.scales for s in seeds[:24]]
            jobs.append(("F", dict(family=fam, nu=nu, m=512, mode="train"), runs))
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", default="all", choices=["A", "B", "C", "all", "D", "E", "F", "all2", "G", "H", "all3", "I", "J", "all4"])
    ap.add_argument("--gd-lrs", type=float, nargs="+", default=[1.0, 3.0, 10.0])
    ap.add_argument("--gd-wds", type=float, nargs="+", default=[3e-4, 1e-3, 3e-3])
    ap.add_argument("--fracs", type=float, nargs="+", default=[0.3, 0.5, 0.7, 0.9, 1.0])
    ap.add_argument("--scales", type=float, nargs="+", default=[0.1, 0.3, 1.0, 3.0])
    ap.add_argument("--batch", type=int, default=100)
    ap.add_argument("--seeds", type=int, default=32)
    ap.add_argument("--widths", type=int, nargs="+", default=[512, 128, 2048])
    ap.add_argument("--seeds-2048", type=int, default=16, help="seeds per wd at m>=2048")
    ap.add_argument("--wds", type=float, nargs="+", default=[0.3, 1.0, 3.0])
    ap.add_argument("--lams", type=float, nargs="+", default=[1e-4, 1e-3, 1e-2])
    ap.add_argument("--margin-ps", type=int, nargs="+", default=[23, 113])
    ap.add_argument("--epochs", type=int, default=10_000)
    ap.add_argument("--compile", default="reduce-overhead")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.seeds, args.widths, args.wds, args.lams = 2, [64], [1.0], [1e-3]
        args.margin_ps, args.epochs, args.compile = [23], 60, "off"
        args.scales = [0.3, 1.0]; args.fracs = [0.5, 1.0]; args.gd_lrs = [1.0]; args.gd_wds = [1e-3]
    for cohort, tmpl, runs in plan(args):
        if args.smoke:
            tmpl = dict(tmpl, p=min(tmpl.get("p", 113), 23))
        # VRAM: the (M, m, p^2) activation dominates; cap M by width
        cap = args.batch if tmpl["m"] <= 512 else max(8, args.batch // 4)
        for i in range(0, len(runs), cap):
            chunk = runs[i:i + cap]
            cfgs, dirs = [], []
            for rk in chunk:
                c = Config(**tmpl, **rk, num_epochs=args.epochs)
                c.nu = c.degree
                lay = f"_L{c.layout[0]}{c.layout[1]}" if c.layout_pre >= 0 else ""
                fr = f"_f{c.frac_train:g}" if cohort in ("G", "H") else ""
                cell = f"{c.family}{c.nu}{lay}_m{c.m}_p{c.p}{fr}_{c.mode}"
                sc = f"_sc{c.init_scale:g}" if c.init_scale != 1.0 else ""
                if c.opt == "gd": cell += "_gd"
                if c.init_from: cell += "_warm"
                tag = (f"wd{c.weight_decay:g}_lr{c.lr:g}{sc}" if c.opt == "gd" else
                       f"wd{c.weight_decay:g}{sc}" if c.mode == "train" else
                       f"lam{c.lam:g}" if c.mode == "l2nu" else "mm")
                rd = RUNS / cohort / cell / f"{tag}_s{c.init_seed}"
                if (rd / "spectra.npz").exists():
                    continue
                cfgs.append(c); dirs.append(rd)
            if not cfgs:
                continue
            print(f"[{cohort}] {cell}: {len(cfgs)} runs", flush=True)
            train_batched(cfgs, dirs, compile_mode=args.compile,
                          spectra_every=5 if args.smoke else 100)


if __name__ == "__main__":
    main()
