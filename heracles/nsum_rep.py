"""nsum_rep.py -- replication trainer for the two-operand exact family (copy of nsum.py's recipe, n = 2 untied only),
with a leading batched-seeds dimension.  Same init draws (per-seed torch.Generator, same order), same schedule
(lr 0.05 doubled at steps 1e3..10e3 capped at --lrmax), same penalties, same fp64 instruments (analyse() is nsum.py's,
verbatim).  The only numerical difference: the full p x p table is evaluated on the grid by broadcasting
x[i, a, b] = E0[i, a] + E1[i, b]  (pow / relu)  or  h = E0[i, a] * E1[i, b]  (prod) instead of one-hot GEMMs -- the
same function and gradient, different summation order.  Seeds are independent problems: the summed loss gives each
seed its own gradient; lr, warm-up state and clipping are per seed.

Runs -> heracles/FINAL_RUNS/REP/<tag>/seed<k>/{metrics.json, final.safetensors}; tag scheme = nsum.py's.
Usage: uv run python heracles/nsum_rep.py --act pow --p 71 --penalty l2q --q 2 --seeds 100 101 102 103
"""
import argparse, json, math, os, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import save_file, load_file

ROOT = Path(__file__).resolve().parent
OUT = Path(os.environ.get("REP_OUT") or (ROOT / "FINAL_RUNS" / "REP"))
DEV = os.environ.get("NSUM_DEV") or ("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False   # exact fp32 (user veto on TF32)
LR_DOUBLE_STEPS = (1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000)


# ---------------------------------------------------------------- nsum.py verbatim (instruments) ----------------
def table(p, n):
    r = torch.arange(p ** n)
    ops = torch.stack([(r // (p ** j)) % p for j in range(n)], 1)
    return ops, torch.ones(len(ops), dtype=torch.float64)


class Data:
    def __init__(self, p, n, chunk, device, dtype=torch.float32, prod=False):
        ops, w = table(p, n)
        self.p, self.n, self.tied, self.N_full, self.prod = p, n, False, p ** n, prod
        self.N_rows = len(ops)
        self.chunks = []
        for s in range(0, len(ops), chunk):
            o = ops[s:s + chunk]; y = (o.sum(1) % p).to(device)
            slabs = [F.one_hot(o[:, j], p).to(dtype).T.contiguous().to(device) for j in range(n)]
            self.chunks.append((slabs, y, w[s:s + chunk].to(device)))

    def forward(self, E, W, slabs, act):
        if act == "prod":
            h = E[0] @ slabs[0]
            for j in range(1, len(slabs)):
                h = h * (E[j] @ slabs[j])
            return W @ h
        x = E[0] @ slabs[0]
        for j in range(1, len(slabs)):
            x = x + E[j] @ slabs[j]
        if act == "relu":
            h = torch.relu(x)
        else:
            h = x
            for _ in range(self.n - 1):
                h = h * x
        return W @ h


def analyse(E, W, data_f64, act):
    p, n, N = data_f64.p, data_f64.n, data_f64.N_full
    dev = E.device; nf = (p - 1) // 2
    E, W = E.detach().double(), W.detach().double()
    Dgrid = torch.arange(p, device=dev)
    g = torch.zeros(p, dtype=torch.float64, device=dev)
    margin, correct, cen_energy = float("inf"), 0.0, 0.0
    with torch.no_grad():
        for slabs, y, w in data_f64.chunks:
            z = data_f64.forward(E, W, slabs, act).T
            zy = z.gather(1, y[:, None]).squeeze(1)
            zo = z.clone(); zo[torch.arange(len(y), device=dev), y] = -float("inf")
            margin = min(margin, float((zy - zo.max(1).values).min()))
            correct += float(((z.argmax(1) == y).double() * w).sum())
            zc = z - z.mean(1, keepdim=True)
            cen_energy += float((zc.square().sum(1) * w).sum())
            idx = (y[:, None] - Dgrid[None, :]) % p
            g += (z.gather(1, idx) * w[:, None]).sum(0)
    g = (g / N).cpu().numpy()
    gt = g - g.mean()
    amp = np.abs(np.fft.fft(gt) / p)[1:nf + 1]
    en = amp ** 2 / (amp ** 2).sum()
    b2 = 2 * N * p * amp ** 2 / cen_energy
    srt = np.sort(b2)[::-1]; cum = np.cumsum(srt) / srt.sum()
    N05 = int(np.searchsorted(cum, 0.95) + 1); N10 = int(np.searchsorted(cum, 0.90) + 1)
    n2 = float(E.square().sum() + W.square().sum())
    nu = 2 if act == "relu" else n + 1
    En = E.cpu().numpy(); shares, hists = [], []
    for j in range(En.shape[0]):
        Fu = np.abs(np.fft.rfft(En[j], axis=1)[:, 1:nf + 1]) ** 2
        tot = Fu.sum(1) + 1e-30; wts = tot / tot.sum()
        shares.append(float(((Fu.max(1) / tot) * wts).sum()))
        hists.append(np.bincount(Fu.argmax(1) + 1, weights=wts, minlength=nf + 1)[1:])
    hist = np.mean(hists, 0)
    return dict(margin=margin, acc=correct / N, norm2=n2, nu=nu, gamma=margin / n2 ** (nu / 2),
                amp=amp.tolist(), b2=b2.tolist(), clean_frac=float(b2.sum()),
                K_2pct=int((en >= 0.02).sum()), K_5pct=int((en >= 0.05).sum()), N_05=N05, N_10=N10,
                top_share=float(np.mean(shares)), top_share_per_emb=shares,
                neuron_hist=hist.tolist(), n_freq_hist_1pct=int((hist >= 0.01).sum()))


# ---------------------------------------------------------------- batched trainer ------------------------------
def grid_forward(E, W, act, a0, a1):
    """E (S, 2, m, p), W (S, p, m); rows a in [a0, a1), all b.  Returns logits (S, p, (a1-a0) * p)."""
    ea = E[:, 0, :, a0:a1, None]; eb = E[:, 1, :, None, :]
    if act == "prod":
        h = ea * eb
    else:
        x = ea + eb
        h = torch.relu(x) if act == "relu" else x * x
    return W @ h.reshape(h.shape[0], h.shape[1], -1)


def penalty(E, W, kind, lam, nu, q):
    """Summed over seeds (independent problems)."""
    if kind == "l2":
        return lam * (E.square().sum() + W.square().sum())
    n2 = E.square().sum((1, 3)) + W.square().sum(1)          # (S, m) per neuron
    ex = (q if kind == "l2q" else nu) / 2
    if ex < 1:
        n2 = n2.clamp_min(1e-12)
    return lam * n2.pow(ex).sum()


def tag_of(a, p, kind):
    return (f"n2_{a.act}_p{p}_{kind}" + (f"{a.q:g}" if kind == "l2q" else "") + f"_lam{a.lam:g}_lrmax{a.lrmax:g}_init{a.init_scale:g}_m{a.m}_st{a.steps}"
            + (f"_from{a.resume_tag}" if a.resume else "") + (f"_clip{a.clip:g}" if a.clip else "")
            + (f"_wdw{a.wd_warmup}_esc{a.escape_frac:g}_lr2{a.lrmax2:g}" if a.wd_warmup else ""))


def run_cell(a, p, kind):
    n, act, m = 2, a.act, a.m
    nu = 2 if act == "relu" else 3
    tag = tag_of(a, p, kind); cell = OUT / tag
    seeds = [s for s in a.seeds if not (cell / f"seed{s}" / "metrics.json").exists()]
    if not seeds:
        print("skip", cell); return
    S = len(seeds)
    Es, Ws = [], []
    for s in seeds:
        if a.resume:
            st = load_file(str(Path(a.resume) / f"seed{s}" / "final.safetensors")); Es.append(st["E"]); Ws.append(st["W"])
        else:
            gen = torch.Generator().manual_seed(s)
            E = (torch.rand(n, m, p, generator=gen) * 2 - 1) / math.sqrt(n * p)
            W = (torch.rand(p, m, generator=gen) * 2 - 1) / math.sqrt(m)
            Es.append(a.init_scale * E); Ws.append(a.init_scale * W)
    E = torch.stack(Es).to(DEV).requires_grad_(True); W = torch.stack(Ws).to(DEV).requires_grad_(True)
    N = p * p
    ar = torch.arange(p, device=DEV)
    ach = a.achunk or p
    ychunks = [((ar[a0:a0 + ach, None] + ar[None, :]) % p).reshape(-1) for a0 in range(0, p, ach)]
    lr0 = a.lrmax2 if a.resume else a.lr0
    lr = torch.full((S, 1, 1), lr0, device=DEV)
    lrcap = torch.full((S, 1, 1), a.lrmax2 if a.resume else a.lrmax, device=DEV)
    warm = torch.full((S,), a.wd_warmup > 0, device=DEV)
    escape = torch.full((S,), -1, device=DEV, dtype=torch.long)
    thr = a.escape_frac * math.log(p)
    logs = [[] for _ in seeds]; t0 = time.time()
    for step in range(1, a.steps + 1):
        if step in LR_DOUBLE_STEPS:
            lr = torch.minimum(lr * 2, lrcap)
        report = step % a.report == 0 or step == a.steps
        gE = torch.zeros_like(E); gW = torch.zeros_like(W)
        ce = torch.zeros(S, device=DEV); correct = torch.zeros(S, device=DEV)
        for ci, a0 in enumerate(range(0, p, ach)):
            z = grid_forward(E, W, act, a0, min(a0 + ach, p))               # (S, p, Nc)
            y = ychunks[ci]
            l = F.cross_entropy(z, y[None].expand(S, -1), reduction="none").sum(1) / N   # (S,)
            dE, dW = torch.autograd.grad(l.sum(), (E, W))
            gE += dE; gW += dW; ce += l.detach()
            if report:
                correct += (z.argmax(1) == y[None]).float().sum(1)
        if a.wd_warmup:
            ended = warm & ((ce < thr) | (step >= a.wd_warmup))
            escape = torch.where(ended, torch.full_like(escape, step), escape)
            warm = warm & ~ended
            lrcap = torch.where(ended[:, None, None], torch.full_like(lrcap, a.lrmax2), lrcap)
            lr = torch.minimum(lr, lrcap)
        pen = penalty(E, W, kind, a.lam, nu, a.q)
        dE, dW = torch.autograd.grad(pen, (E, W))
        on = (~warm).float()
        gE += dE * on[:, None, None, None]; gW += dW * on[:, None, None]
        with torch.no_grad():
            if a.clip:
                gn = torch.sqrt(gE.square().sum((1, 2, 3)) + gW.square().sum((1, 2)))
                sc = torch.clamp(a.clip / gn, max=1.0)
                gE *= sc[:, None, None, None]; gW *= sc[:, None, None]
            E -= lr[:, :, :, None] * gE; W -= lr * gW
        if report:
            cel = ce.tolist(); accl = (correct / N).tolist(); lrl = lr.flatten().tolist()
            for i in range(S):
                logs[i].append(dict(step=step, ce=cel[i], acc=accl[i], lr=lrl[i], t=time.time() - t0))
            print(f"  {tag} step {step:6d} ce " + " ".join(f"{c:.2e}" for c in cel) + " acc " + " ".join(f"{x:.3f}" for x in accl)
                  + f"  [{time.time() - t0:.0f}s]", flush=True)
            if not any(math.isfinite(c) for c in cel):
                break
    train_time = time.time() - t0
    data64 = Data(p, n, a.chunk, DEV, dtype=torch.float64, prod=(act == "prod"))
    esc = escape.tolist()
    for i, s in enumerate(seeds):
        out = cell / f"seed{s}"; out.mkdir(parents=True, exist_ok=True)
        Ei, Wi = E[i].detach(), W[i].detach()
        finite = bool(torch.isfinite(Ei).all() and torch.isfinite(Wi).all())
        if finite:
            res = analyse(Ei, Wi, data64, act)
        else:
            res = dict(diverged=True, acc=0.0, margin=float("nan"), nu=nu, gamma=float("nan"), K_2pct=-1, K_5pct=-1, N_05=-1, N_10=-1,
                       clean_frac=float("nan"), top_share=float("nan"), n_freq_hist_1pct=-1)
        res.update(p=p, n=n, act=act, penalty=kind, q=(a.q if kind == "l2q" else (nu if kind == "l2nu" else 2.0)), lam=a.lam, seed=s, m=m,
                   steps=a.steps, lrmax=a.lrmax, init_scale=a.init_scale, clip=a.clip, tied=False, wd_warmup=a.wd_warmup,
                   escape_frac=a.escape_frac, lrmax2=a.lrmax2, escape_step=(esc[i] if esc[i] >= 0 else None),
                   train_time=train_time, batch_seeds=S, final_ce=logs[i][-1]["ce"] if logs[i] else None, log=logs[i], trainer="nsum_rep")
        (out / "metrics.json").write_text(json.dumps(res, indent=1))
        save_file({"E": Ei.cpu().contiguous(), "W": Wi.cpu().contiguous()}, str(out / "final.safetensors"))
        print(f"DONE {tag} seed{s}: acc {res['acc']:.4f} margin {res['margin']:.4f} K_2pct {res['K_2pct']} K_5pct {res['K_5pct']} "
              f"N_05 {res['N_05']} clean {res['clean_frac']:.3f}  [{train_time:.0f}s / {S} seeds]", flush=True)
    print("CELL DONE", tag, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--act", default="pow", choices=["pow", "relu", "prod"])
    ap.add_argument("--p", type=int, nargs="+", default=[71])
    ap.add_argument("--penalty", nargs="+", default=["l2"], choices=["l2", "l2nu", "l2q"])
    ap.add_argument("--q", type=float, default=None)
    ap.add_argument("--seeds", type=int, nargs="+", default=[100, 101, 102, 103])
    ap.add_argument("--lam", type=float, default=1e-4)
    ap.add_argument("--m", type=int, default=500)
    ap.add_argument("--steps", type=int, default=40000)
    ap.add_argument("--lr0", type=float, default=0.05)
    ap.add_argument("--lrmax", type=float, default=12.8)
    ap.add_argument("--init_scale", type=float, default=1.0)
    ap.add_argument("--clip", type=float, default=0.0)
    ap.add_argument("--wd_warmup", type=int, default=0)
    ap.add_argument("--escape_frac", type=float, default=0.97)
    ap.add_argument("--lrmax2", type=float, default=None)
    ap.add_argument("--resume", default=None, help="cell dir whose seed{seed}/final.safetensors to start from; lr = --lrmax2 from step 1")
    ap.add_argument("--chunk", type=int, default=400_000, help="fp64 analysis rows per chunk")
    ap.add_argument("--achunk", type=int, default=0, help="training: rows of a per chunk (0 = whole grid)")
    ap.add_argument("--report", type=int, default=2000)
    args = ap.parse_args()
    if "l2q" in args.penalty and args.q is None:
        ap.error("--penalty l2q needs --q")
    if args.lrmax2 is None:
        args.lrmax2 = args.lrmax
    args.resume_tag = Path(args.resume).name.split("_lam")[1].split("_")[0] if args.resume else ""
    print("device", DEV, "args", vars(args), flush=True)
    for p in args.p:
        for kind in args.penalty:
            run_cell(args, p, kind)
