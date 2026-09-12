"""heracles trainer: exactly homogeneous ReLU MLP family on a + b mod p, batched over runs,
optimizers with a CONSTANT l_q geometry throughout training.

Architecture (family "relu", degree nu = 2 + n_pre + n_post, layout table below):
    tokens (one-hot) -> E1 [-> E2 ..] -> (U a-branch, V b-branch) -> ReLU -> W1 [-> W2 ..] -> W -> logits
    no biases, no normalisation: f(t theta) = t^nu f(theta) exactly.

Optimizers (all coupled: they descend CE + penalty in ONE fixed geometry):
    gd    q=2   theta -= lr * (g + wd * theta)                     plain full-batch gradient descent + l2
    gdm   q=2   heavy-ball on the same direction (momentum mu)
    sgd   q=2   gd on one random minibatch (batch_size) per iteration
    lq    q     theta -= lr * map_q(g + wd * sign(theta)|theta|^(q-1)),   map_q(v) = sign(v)|v|^(1/(q-1))
                q = 2 reduces to gd; the penalty is wd/q * ||theta||_q^q  (same norm as the geometry)
    sign  inf   theta -= lr * sign(g); then project every entry to [-maxnorm, maxnorm]   (l_inf penalty = box)
    signm inf   sign of the heavy-ball momentum (Lion-like), same projection
    reg="l2" on sign/signm gives the MIXED control (sign steps + multiplicative l2 shrink, Lion's actual wd).
clip > 0: per-run global l2 norm clip of the (mapped) direction — direction-preserving.
warmup: linear lr warmup over warmup_steps epochs. No other schedule.

Per-run fields that may differ inside one batch: init_seed, data_seed, lr, weight_decay, clip, q, maxnorm,
momentum. Everything shape- or code-path-bearing must agree (asserted).

Outputs per run dir: config.json, final.safetensors, spectra.npz (coeffs/energy of the clean logit table,
train/test acc, ||theta||_2^2, ||theta||_q^q, min margin normalised by ||theta||_2^nu, at spectra_every),
metrics.json (train loss every epoch, test loss/acc every log_every, t_train99, t_test99).
Resume-safe: a run whose spectra.npz exists is skipped by jobs.py.
"""
import json, math, os, random, time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parent
RUNS = Path(os.environ.get("RUNS_DIR", ROOT / "runs"))
META_SEED = 20260911
torch.backends.cuda.matmul.allow_tf32 = False      # exact fp32 (user veto on TF32)
torch.backends.cudnn.allow_tf32 = False

LAYOUT = {2: (0, 0), 3: (1, 0), 4: (1, 1), 5: (2, 1), 6: (2, 2), 7: (3, 2), 8: (3, 3)}
SHAPE_FIELDS = ("p", "nu", "m", "d", "r", "frac_train", "opt", "reg", "num_epochs", "warmup_steps",
                "batch_size", "spectra_every")


@dataclass
class Config:
    p: int = 113
    nu: int = 4
    m: int = 512                # hidden width
    d: int = 128                # embedding width (pre layers)
    r: int = 128                # readout-stack width (post layers)
    frac_train: float = 0.5
    data_seed: int = 0
    init_seed: int = 0
    init_scale: float = 1.0     # std = init_scale / sqrt(fan_in)
    opt: str = "gd"             # gd | gdm | sgd | lq | sign | signm
    q: float = 2.0              # geometry exponent for opt=lq (2 = gd); recorded as inf for sign/signm
    reg: str = "lq"             # lq: penalty wd/q ||theta||_q^q in the SAME q (l2 for gd/gdm/sgd);
                                # box: sign/signm project to [-maxnorm, maxnorm];  l2: mixed control (shrink)
    lr: float = 1.0
    weight_decay: float = 1e-3
    maxnorm: float = 0.1        # reg=box radius
    momentum: float = 0.9       # gdm / signm
    clip: float = 0.0           # >0: clip per-run global l2 norm of the update direction
    batch_size: int = 0         # sgd only: minibatch size per iteration (0 = full batch)
    warmup_steps: int = 1
    num_epochs: int = 20_000
    spectra_every: int = 200
    ckpt_every: int = 0
    ckpt_at: tuple = ()          # explicit epochs at which to save ckpt_XXXXXX.safetensors (in addition to ckpt_every)

    @property
    def layout(self):
        return LAYOUT[self.nu]

    def save(self, path):
        Path(path).write_text(json.dumps(asdict(self), indent=2))

    @staticmethod
    def load(path):
        dd = json.loads(Path(path).read_text())
        dd = {k: v for k, v in dd.items() if k in Config.__dataclass_fields__}
        if "ckpt_at" in dd: dd["ckpt_at"] = tuple(dd["ckpt_at"])
        return Config(**dd)


# ------------------------------------------------------------------ data / model
def train_test_split(cfg):
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


def param_shapes(cfg):
    n_pre, n_post = cfg.layout
    p, m, d, r = cfg.p, cfg.m, cfg.d, cfg.r
    sh = {}; fan = p
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
        xa, xb = _cols(pr["U"], A), _cols(pr["V"], B)
    else:
        ea, eb = _cols(pr["E1"], A), _cols(pr["E1"], B)
        for i in range(1, n_pre):
            ea, eb = pr[f"E{i+1}"] @ ea, pr[f"E{i+1}"] @ eb
        xa, xb = pr["U"] @ ea, pr["V"] @ eb
    h = F.relu(xa + xb)
    for i in range(n_post):
        h = pr[f"W{i+1}"] @ h
    return (pr["W"] @ h).transpose(1, 2)


def per_ex_ce(logits, labels):
    z_y = logits.gather(-1, labels.unsqueeze(-1))
    others = logits.masked_fill(F.one_hot(labels, logits.shape[-1]).bool(), float("-inf"))
    return F.softplus(torch.logsumexp(others, dim=-1, keepdim=True) - z_y).squeeze(-1)


def per_ex_margin(logits, labels):
    z_y = logits.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    others = logits.masked_fill(F.one_hot(labels, logits.shape[-1]).bool(), float("-inf"))
    return z_y - others.max(-1).values


def sq_norm(pr):
    return sum((v ** 2).flatten(1).sum(1) for v in pr.values())


def q_norm_q(pr, q):
    """(M,) sum |theta|^q with per-run q (M,) tensor."""
    return sum((v.abs() ** q.view((-1,) + (1,) * (v.dim() - 1))).flatten(1).sum(1) for v in pr.values())


# ------------------------------------------------------------------ training
def train_batched(cfgs, run_dirs, compile_mode="reduce-overhead", log_every=500, device=None):
    M = len(cfgs); c0 = cfgs[0]
    for c in cfgs:
        for f in SHAPE_FIELDS:
            assert getattr(c, f) == getattr(c0, f), f"batch must agree on {f}: {getattr(c, f)} vs {getattr(c0, f)}"
    assert c0.opt in ("gd", "gdm", "sgd", "lq", "sign", "signm"), c0.opt
    if c0.opt in ("gd", "gdm", "sgd"):
        for c in cfgs: assert c.q == 2.0 and c.reg == "lq", "gd-family is l2 geometry with the l2 penalty"
    if c0.opt in ("sign", "signm"):
        for c in cfgs: assert c.reg in ("box", "l2"), "sign geometry: reg=box (l_inf) or l2 (mixed control)"
    if c0.opt == "sgd":
        assert c0.batch_size > 0
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    run_dirs = [Path(rd) for rd in run_dirs]
    for cfg, rd in zip(cfgs, run_dirs):
        rd.mkdir(parents=True, exist_ok=True); cfg.save(rd / "config.json")
    p, nu = c0.p, c0.nu
    keys = list(param_shapes(c0))
    pr = {k: torch.stack([init_params(c)[k] for c in cfgs]).to(device).requires_grad_() for k in keys}
    params = [pr[k] for k in keys]
    bshape = {id(x): (-1,) + (1,) * (x.dim() - 1) for x in params}

    a_np = np.repeat(np.arange(p), p); b_np = np.tile(np.arange(p), p); lab_np = (a_np + b_np) % p
    A_all = torch.from_numpy(a_np).to(device); B_all = torch.from_numpy(b_np).to(device)
    lab_all = torch.from_numpy(lab_np).to(device)
    masks = [train_test_split(c) for c in cfgs]
    assert len({int(mk.sum()) for mk in masks}) == 1, "equal-count batches only"
    tr_idx = torch.from_numpy(np.stack([np.flatnonzero(mk) for mk in masks])).to(device)
    te_idx = torch.from_numpy(np.stack([np.flatnonzero(~mk) for mk in masks])).to(device)
    mask_t = torch.from_numpy(np.stack(masks)).to(device)
    A_tr, B_tr, lab_tr = A_all[tr_idx], B_all[tr_idx], lab_all[tr_idx]
    N_tr = tr_idx.shape[1]

    lr_base = torch.tensor([c.lr for c in cfgs], device=device)
    wd_v = torch.tensor([c.weight_decay for c in cfgs], device=device)
    clip_v = torch.tensor([c.clip for c in cfgs], device=device)
    q_v = torch.tensor([c.q for c in cfgs], device=device)
    qm1 = (q_v - 1).clamp(min=1e-6)
    box_v = torch.tensor([c.maxnorm for c in cfgs], device=device)
    mu_v = torch.tensor([c.momentum for c in cfgs], device=device)
    lr_t = torch.zeros(M, device=device)
    mom = [torch.zeros_like(x) for x in params]
    OPT, REG = c0.opt, c0.reg
    use_clip = any(c.clip > 0 for c in cfgs)
    gen = torch.Generator(device=device).manual_seed(c0.data_seed * 7919 + 1)

    def step():
        if OPT == "sgd":
            sel = torch.randint(0, N_tr, (M, c0.batch_size), device=device, generator=gen)
            A, B, lab = A_tr.gather(1, sel), B_tr.gather(1, sel), lab_tr.gather(1, sel)
        else:
            A, B, lab = A_tr, B_tr, lab_tr
        per = per_ex_ce(forward(pr, A, B, c0), lab).mean(1)
        grads = torch.autograd.grad(per.sum(), params)
        with torch.no_grad():
            dirs = []
            for i_, (x, g) in enumerate(zip(params, grads)):
                sh = bshape[id(x)]
                if OPT in ("sign", "signm"):
                    v = g
                else:                                    # coupled penalty in the same geometry
                    if OPT == "lq":
                        pen = torch.sign(x) * x.abs() ** qm1.view(sh)
                        v = g + wd_v.view(sh) * pen
                        v = torch.sign(v) * v.abs() ** (1.0 / qm1).view(sh)
                    else:
                        v = g + wd_v.view(sh) * x
                if OPT in ("gdm", "signm"):
                    mo = mom[i_]
                    mo.mul_(mu_v.view(sh)).add_(v); v = mo
                if OPT == "signm":
                    v = torch.sign(v)
                elif OPT == "sign":
                    v = torch.sign(v)
                dirs.append(v)
            if use_clip:
                gn = torch.sqrt(sum((v ** 2).flatten(1).sum(1) for v in dirs)) + 1e-12
                sc = torch.where(clip_v > 0, torch.clamp(clip_v / gn, max=1.0), torch.ones_like(gn))
                dirs = [v * sc.view(bshape[id(x)]) for x, v in zip(params, dirs)]
            for x, v in zip(params, dirs):
                sh = bshape[id(x)]
                if OPT in ("sign", "signm") and REG == "l2":
                    x.mul_(1 - lr_t.view(sh) * wd_v.view(sh))
                x.sub_(lr_t.view(sh) * v)
                if OPT in ("sign", "signm") and REG == "box":
                    b = box_v.view(sh)
                    x.copy_(torch.maximum(torch.minimum(x, b), -b))
        return per.detach()

    if compile_mode != "off":
        step = torch.compile(step, mode=None if compile_mode == "default" else compile_mode)

    basis = fourier_basis(p); n_ = p // 2
    bc, bs = basis[1::2], basis[2::2]
    u_cos = torch.from_numpy((np.einsum("np,nq->npq", bc, bc) - np.einsum("np,nq->npq", bs, bs)).reshape(n_, -1) / np.sqrt(2)).to(device)
    u_sin = torch.from_numpy((np.einsum("np,nq->npq", bs, bc) + np.einsum("np,nq->npq", bc, bs)).reshape(n_, -1) / np.sqrt(2)).to(device)
    b_cos = torch.from_numpy(bc).to(device); b_sin = torch.from_numpy(bs).to(device)
    snap = {k: [] for k in ("epochs", "coeffs", "energy", "train_acc", "test_acc", "pnorm2", "qnormq", "min_margin", "min_margin_norm")}
    q_eff = torch.where(torch.isinf(q_v), torch.full_like(q_v, 2.0), q_v)   # sign: record l2 too; l_inf via max below
    first99 = {"train": [None] * M, "test": [None] * M}

    @torch.no_grad()
    def take_snapshot(epoch):
        L = forward(pr, A_all, B_all, c0)
        acc = L.argmax(-1) == lab_all
        tr_acc = (acc & mask_t).sum(-1).double() / mask_t.sum(-1)
        te_acc = (acc & ~mask_t).sum(-1).double() / (~mask_t).sum(-1).clamp(min=1)
        for i in range(M):
            if first99["train"][i] is None and tr_acc[i] >= 0.99: first99["train"][i] = epoch
            if first99["test"][i] is None and te_acc[i] >= 0.99: first99["test"][i] = epoch
        snap["epochs"].append(epoch); snap["train_acc"].append(tr_acc); snap["test_acc"].append(te_acc)
        Ld = L.double()
        c = torch.einsum("fp,mpo->mfo", u_cos, Ld); s = torch.einsum("fp,mpo->mfo", u_sin, Ld)
        snap["energy"].append((c ** 2).sum(-1) + (s ** 2).sum(-1))
        snap["coeffs"].append(((c * b_cos).sum(-1) + (s * b_sin).sum(-1)) / math.sqrt(2))
        n2 = sq_norm(pr).detach(); snap["pnorm2"].append(n2)
        snap["qnormq"].append(q_norm_q(pr, q_eff).detach() if OPT != "sign" and OPT != "signm"
                              else torch.stack([max(v[i].abs().max() for v in pr.values()) for i in range(M)]).detach())
        g = per_ex_margin(L, lab_all[None].expand(M, -1)).min(1).values
        snap["min_margin"].append(g); snap["min_margin_norm"].append(g / n2 ** (nu / 2))

    E = c0.num_epochs
    train_hist = torch.empty(E, M, device=device); test_hist = []
    print(f"batch {M} on {device}: nu={nu} layout={c0.layout} m={c0.m} p={p} opt={OPT} reg={REG} "
          f"train {N_tr} test {te_idx.shape[1]} | lr {sorted(set(c.lr for c in cfgs))} wd {sorted(set(c.weight_decay for c in cfgs))}", flush=True)
    t0 = time.time()
    for epoch in range(E):
        if c0.spectra_every and epoch % c0.spectra_every == 0:
            take_snapshot(epoch)
        lr_t.copy_(lr_base * min((epoch + 1) / c0.warmup_steps, 1.0))
        train_hist[epoch].copy_(step())
        if ((c0.ckpt_every and (epoch + 1) % c0.ckpt_every == 0) or (epoch + 1) in c0.ckpt_at) and epoch + 1 < E:
            for i, rd in enumerate(run_dirs):
                save_file({k: v[i].detach().cpu().contiguous() for k, v in pr.items()}, str(rd / f"ckpt_{epoch + 1:06d}.safetensors"))
        if (epoch + 1) % log_every == 0:
            with torch.no_grad():
                tr = train_hist[epoch].cpu().numpy()
                L = forward(pr, A_all, B_all, c0)
                Lte = L.gather(1, te_idx.unsqueeze(-1).expand(-1, -1, p))
                te = per_ex_ce(Lte, lab_all[te_idx]).mean(1).cpu().numpy()
                te_acc = ((L.argmax(-1) == lab_all) & ~mask_t).sum(-1).double() / (~mask_t).sum(-1)
            test_hist.append(te)
            eps_s = (epoch + 1) / (time.time() - t0)
            bad = int((~np.isfinite(tr)).sum())
            print(f"epoch {epoch+1:6d}  train med {np.nanmedian(tr):.3e} max {np.nanmax(tr):.3e}  test CE med {np.nanmedian(te):.3e}  "
                  f"test acc med {te_acc.median():.3f} min {te_acc.min():.3f}  n>=.99 {int((te_acc>=0.99).sum())}/{M}  nonfinite {bad}  "
                  f"({eps_s:.1f} eps/s x {M} = {eps_s*M:.0f} run-eps/s)", flush=True)
    take_snapshot(E)
    train_all = train_hist.cpu().numpy(); test_all = np.stack(test_hist) if test_hist else np.zeros((0, M))
    train_every = 10 if E > 100_000 else 1     # long budgets: thin the per-epoch train-loss log (finite flag uses the last epoch)
    sp = {k: torch.stack(snap[k]).cpu().numpy() for k in snap if k != "epochs"}
    for i, rd in enumerate(run_dirs):
        save_file({k: v[i].detach().cpu().contiguous() for k, v in pr.items()}, str(rd / "final.safetensors"))
        np.savez(rd / "spectra.npz", epochs=np.array(snap["epochs"]), **{k: v[:, i] for k, v in sp.items()})
        (rd / "metrics.json").write_text(json.dumps({
            "train_losses": train_all[::train_every, i].tolist(), "train_every": train_every, "test_losses": test_all[:, i].tolist(), "test_every": log_every,
            "final_train_acc": float(sp["train_acc"][-1, i]), "final_test_acc": float(sp["test_acc"][-1, i]),
            "t_train99": first99["train"][i], "t_test99": first99["test"][i],
            "final_min_margin": float(sp["min_margin"][-1, i]), "final_min_margin_norm": float(sp["min_margin_norm"][-1, i]),
            "final_pnorm2": float(sp["pnorm2"][-1, i]), "finite": bool(np.isfinite(train_all[-1, i]))}))
    print(f"done: {E} epochs x {M} runs in {time.time()-t0:.0f}s", flush=True)


def draw_seeds(n):
    rng = random.Random(META_SEED)
    return sorted(rng.sample(range(300_000, 400_000), n))
