"""nsum.py -- the exact nu ladder: n-operand modular sum a_1 + ... + a_n = c (mod p) with activation x^n
(homogeneity degree nu = n + 1), one hidden layer, no biases, FULL table (p^n rows), full-batch GD (chunked, exact),
Morwani App. B.1 recipe otherwise (lr 0.05 doubled at steps 1e3..10e3 capped at --lrmax, penalty 1e-4).
Theory: heracles/THEORY_nsum.md.  --act relu gives the nu = 2 dense anchor on the same task.

Penalties: l2 = lam sum(theta^2) (ours: K* frequencies);  l2nu = lam sum_i (||theta_i||^2)^{nu/2}
(Li et al. 2025 / Morwani L_{2,nu}: dense).

Implementation: the input gather is written as a one-hot matmul  x = sum_j E_j @ H_j^T  with the one-hot slabs
H_j^T (p, rows) precomputed per chunk and cached on the GPU, so the backward is a GEMM (no scatter atomics).
--tied shares one embedding across the n operands; then the net is symmetric in the operands and the full table
is EXACTLY equivalent to the multiset table (C(p+n-1, n) rows) with multinomial weights: same loss, same gradient,
same margin, same clean projection. (x = E @ C^T with C the count matrix; nu unchanged; the clean construction
exists with tied embeddings, see THEORY_nsum.md.)  p^n / n! fewer rows for the 4-operand cells.

Per run (fp64): margin, acc, ||theta||^2, gamma = margin / ||theta||^nu, clean projection g(D) over D = sum a - c,
K at 2% / 5% of clean energy, Levine's spectral weights b_k^2, N_delta (.05, .10), clean fraction,
per-embedding single-frequency share, neuron histogram.
Usage: uv run python heracles/nsum.py --n 3 --p 71 --penalty l2 --seeds 0 1 2 3
"""
import argparse, itertools, json, math, os, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "runs" / "NS"          # overridden by --out
DEV = os.environ.get("NSUM_DEV") or ("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False   # exact fp32 (user veto on TF32)
LR_DOUBLE_STEPS = (1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000)


def table(p, n, tied):
    """Rows of the task as (ops (N, n) int64, weights (N,) float64). Full table (weights 1) or, if tied, the multiset
    table with multinomial multiplicities. N_full = p^n always."""
    if not tied:
        r = torch.arange(p ** n)
        ops = torch.stack([(r // (p ** j)) % p for j in range(n)], 1)
        return ops, torch.ones(len(ops), dtype=torch.float64)
    combs = np.fromiter(itertools.chain.from_iterable(itertools.combinations_with_replacement(range(p), n)),
                        dtype=np.int64).reshape(-1, n)
    fact = [math.factorial(i) for i in range(n + 1)]
    w = np.empty(len(combs))
    for r_, row in enumerate(combs):
        _, cnt = np.unique(row, return_counts=True)
        w[r_] = fact[n] / np.prod([fact[c] for c in cnt])
    return torch.from_numpy(combs), torch.from_numpy(w)


class Data:
    """Chunked table cached on the device: per chunk the one-hot slabs (or count slab if tied), labels, weights."""
    def __init__(self, p, n, tied, chunk, device, dtype=torch.float32, prod=False, grid=False):
        ops, w = table(p, n, tied)
        self.p, self.n, self.tied, self.N_full, self.prod = p, n, tied, p ** n, prod
        self.N_rows = len(ops)
        self.chunks = []
        self.gather = False
        self.grid = bool(grid) and not tied      # --grid: untied full table as a broadcast grid (no one-hot GEMMs); same rows, same order
        if self.grid:
            self.chunks.append(([], (ops.sum(1) % p).to(device), w.to(device)))
            return
        self.gather = bool(grid) and tied and prod    # --grid with --tied --act prod: index gathers instead of one-hot GEMMs
        if self.gather:
            for s in range(0, len(ops), chunk):
                o = ops[s:s + chunk]
                self.chunks.append(([o[:, j].contiguous().to(device) for j in range(n)], (o.sum(1) % p).to(device), w[s:s + chunk].to(device)))
            return
        for s in range(0, len(ops), chunk):
            o = ops[s:s + chunk]; y = (o.sum(1) % p).to(device)
            if tied and not prod:
                C = torch.zeros(len(o), p, dtype=dtype)
                C.scatter_add_(1, o, torch.ones_like(o, dtype=dtype))
                slabs = [C.T.contiguous().to(device)]                         # (p, Nc)
            else:
                slabs = [F.one_hot(o[:, j], p).to(dtype).T.contiguous().to(device) for j in range(n)]
            self.chunks.append((slabs, y, w[s:s + chunk].to(device)))

    def forward(self, E, W, slabs, act):
        if self.grid and act != "relu":                 # row r = sum_j a_j p^j: operand 0 varies fastest
            x = E[0]
            for j in range(1, self.n):
                x = (E[j][:, :, None] * x[:, None, :] if act == "prod" else E[j][:, :, None] + x[:, None, :]).reshape(x.shape[0], -1)
            if act == "prod":
                return W @ x
            h = x
            for _ in range(self.n - 1):
                h = h * x
            return W @ h
        if getattr(self, "gather", False):
            h = E[0].index_select(1, slabs[0])              # index_select: backward is index_add_ (fast); fancy indexing sorts
            for j in range(1, len(slabs)):
                h = h * E[0].index_select(1, slabs[j])
            return W @ h
        if act == "prod":                               # gated chain: h_i = prod_j e_ij(a_j)  (CP factorisation of the logit tensor)
            h = E[0] @ slabs[0]
            for j in range(1, len(slabs)):
                h = h * (E[0 if self.tied else j] @ slabs[j])
            return W @ h
        x = E[0] @ slabs[0]
        for j in range(1, len(slabs)):
            x = x + E[j] @ slabs[j]                     # (m, Nc)
        if act == "relu":
            h = torch.relu(x)
        else:
            h = x
            for _ in range(self.n - 1):
                h = h * x
        return W @ h                                    # (p, Nc)


def penalty(E, W, kind, lam, nu, q=None):
    """l2: lam sum theta^2;  l2nu: lam sum_i (||theta_i||^2)^{nu/2} (Morwani, flat);  l2q: lam sum_i (||theta_i||^2)^{q/2}
    (per-neuron group norm to the power q; the sparse program has exponent nu/q, so q = 2 is plain l2 and q = nu is l2nu)."""
    if kind == "l2":
        return lam * (E.square().sum() + W.square().sum())
    n2 = E.square().sum((0, 2)) + W.square().sum(0)          # per neuron
    ex = (q if kind == "l2q" else nu) / 2
    if ex < 1:                                            # d/dtheta n2^ex = 2 ex n2^(ex-1) theta: 0 * inf at n2 = 0 -> guard
        n2 = n2.clamp_min(1e-12)
    return lam * n2.pow(ex).sum()


def analyse(E, W, data_f64, act):
    """All fp64. Weighted over the (possibly multiset) table; N = p^n throughout."""
    p, n, N = data_f64.p, data_f64.n, data_f64.N_full
    dev = E.device; nf = (p - 1) // 2
    E, W = E.detach().double(), W.detach().double()
    Dgrid = torch.arange(p, device=dev)
    g = torch.zeros(p, dtype=torch.float64, device=dev)
    margin, correct, cen_energy = float("inf"), 0.0, 0.0
    with torch.no_grad():
        for slabs, y, w in data_f64.chunks:
            z = data_f64.forward(E, W, slabs, act).T                # (Nc, p)
            zy = z.gather(1, y[:, None]).squeeze(1)
            zo = z.clone(); zo[torch.arange(len(y), device=dev), y] = -float("inf")
            margin = min(margin, float((zy - zo.max(1).values).min()))
            correct += float(((z.argmax(1) == y).double() * w).sum())
            zc = z - z.mean(1, keepdim=True)
            cen_energy += float((zc.square().sum(1) * w).sum())
            idx = (y[:, None] - Dgrid[None, :]) % p                 # z[row, s - D]
            g += (z.gather(1, idx) * w[:, None]).sum(0)
    g = (g / N).cpu().numpy()
    gt = g - g.mean()
    amp = np.abs(np.fft.fft(gt) / p)[1:nf + 1]
    en = amp ** 2 / (amp ** 2).sum()
    b2 = 2 * N * p * amp ** 2 / cen_energy                        # Levine spectral weights; sum = clean fraction
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


def run(a, p, kind, seed, data=None):
    n, act, m, tied = a.n, a.act, a.m, a.tied
    nu = 2 if act == "relu" else n + 1
    tag = (f"n{n}_{act}_p{p}_{kind}" + (f"{a.q:g}" if kind == "l2q" else "") + f"_lam{a.lam:g}_lrmax{a.lrmax:g}_init{a.init_scale:g}_m{m}_st{a.steps}" + (f"_from{a.resume_tag}" if a.resume else "")
           + ("_tied" if tied else "") + (f"_clip{a.clip:g}" if a.clip else "") + (f"_wdw{a.wd_warmup}_esc{a.escape_frac:g}_lr2{a.lrmax2:g}" if a.wd_warmup else ""))
    out = OUT / tag / f"seed{seed}"
    if (out / "metrics.json").exists():
        print("skip", out); return data
    out.mkdir(parents=True, exist_ok=True)
    if data is None:
        t0 = time.time(); data = Data(p, n, tied, a.chunk, DEV, prod=(act == "prod"), grid=a.grid)
        print(f"  table p{p} n{n} tied={tied}: {data.N_rows} rows in {len(data.chunks)} chunks [{time.time()-t0:.0f}s]", flush=True)
    ne = 1 if tied else n
    gen = torch.Generator().manual_seed(seed)
    E = (torch.rand(ne, m, p, generator=gen) * 2 - 1) / math.sqrt(n * p)
    W = (torch.rand(p, m, generator=gen) * 2 - 1) / math.sqrt(m)
    if a.resume:
        from safetensors.torch import load_file
        src = Path(a.resume) / f"seed{seed}" / "final.safetensors"
        st = load_file(str(src)); E, W = st["E"], st["W"]
        print(f"  resumed from {src}", flush=True)
    else:
        E, W = a.init_scale * E, a.init_scale * W
    E = E.to(DEV).requires_grad_(True); W = W.to(DEV).requires_grad_(True)
    N = data.N_full
    lr, log, t0 = a.lr0, [], time.time()
    trace, trace_data = [], None
    warm = a.wd_warmup > 0; escape_step = None; lrcap = a.lrmax; lnp = math.log(p)
    if a.resume:
        lr = lrcap = a.lrmax2
    for step in range(1, a.steps + 1):
        if step in LR_DOUBLE_STEPS:
            lr = min(lr * 2, lrcap)
        gE = torch.zeros_like(E); gW = torch.zeros_like(W); loss_ce = 0.0; correct = 0.0
        report = step % 2000 == 0 or step == a.steps
        for slabs, y, w in data.chunks:
            z = data.forward(E, W, slabs, act).T                       # (Nc, p)
            l = (F.cross_entropy(z, y, reduction="none") * w).sum() / N
            dE, dW = torch.autograd.grad(l, (E, W))
            gE += dE; gW += dW; loss_ce += float(l.detach())
            if report:
                correct += float(((z.argmax(1) == y).float() * w).sum())
        if warm and (loss_ce < a.escape_frac * lnp or step >= a.wd_warmup):
            warm = False; escape_step = step; lrcap = a.lrmax2; lr = min(lr, lrcap)
            print(f"  warm-up ends at step {step} (ce {loss_ce:.3f} vs {a.escape_frac:g} ln p = {a.escape_frac * lnp:.3f}); wd on, lr cap {lrcap}", flush=True)
        pen = penalty(E, W, kind, a.lam, nu, a.q)
        if not warm:
            dE, dW = torch.autograd.grad(pen, (E, W))
            gE += dE; gW += dW
        with torch.no_grad():
            if a.clip:
                gn = torch.sqrt(gE.square().sum() + gW.square().sum())
                if gn > a.clip:
                    gE *= a.clip / gn; gW *= a.clip / gn
            E -= lr * gE; W -= lr * gW
        if report:
            tot = loss_ce + float(pen)
            log.append(dict(step=step, loss=tot, ce=loss_ce, acc=correct / N, lr=lr, t=time.time() - t0))
            print(f"  n{n} {act} p{p} {kind} s{seed} step {step:6d} lr {lr:6.2f} loss {tot:.3e} ce {loss_ce:.3e} "
                  f"acc {correct / N:.4f}  [{time.time() - t0:.0f}s]", flush=True)
            if not math.isfinite(tot):
                break
        if a.trace_every and step % a.trace_every == 0 and step < a.steps:
            if trace_data is None:
                trace_data = Data(p, n, tied, a.chunk, DEV, dtype=torch.float64, prod=(act == "prod"), grid=a.grid)
            tr = analyse(E, W, trace_data, act)
            trace.append({k: tr[k] for k in ("margin", "acc", "norm2", "gamma", "clean_frac", "K_2pct", "K_5pct", "N_05", "N_10")} | dict(step=step))
            print(f"  TRACE step {step} margin {tr['margin']:.3f} gamma {tr['gamma']:.3e} K5 {tr['K_5pct']} K2 {tr['K_2pct']} N05 {tr['N_05']} clean {tr['clean_frac']:.3f}", flush=True)
    data64 = trace_data if trace_data is not None else Data(p, n, tied, a.chunk, DEV, dtype=torch.float64, prod=(act == "prod"), grid=a.grid)
    res = analyse(E, W, data64, act); del data64, trace_data
    res["trace"] = trace
    res.update(p=p, n=n, act=act, penalty=kind, q=(a.q if kind == "l2q" else (nu if kind == "l2nu" else 2.0)), lam=a.lam, seed=seed, m=m, steps=a.steps, lrmax=a.lrmax,
               init_scale=a.init_scale, clip=a.clip, tied=tied, wd_warmup=a.wd_warmup, escape_frac=a.escape_frac, lrmax2=a.lrmax2, escape_step=escape_step, train_time=time.time() - t0, log=log)
    (out / "metrics.json").write_text(json.dumps(res, indent=1))
    save_file({"E": E.detach().cpu(), "W": W.detach().cpu()}, str(out / "final.safetensors"))
    print(f"DONE {tag} seed{seed}: acc {res['acc']:.4f} margin {res['margin']:.4f} gamma {res['gamma']:.3e} "
          f"K_2pct {res['K_2pct']} K_5pct {res['K_5pct']} N_05 {res['N_05']} clean {res['clean_frac']:.3f} "
          f"top-share {res['top_share']:.3f} freqs>=1% {res['n_freq_hist_1pct']}  [{res['train_time']:.0f}s]", flush=True)
    return data


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3, help="operands; activation x^n unless --act relu")
    ap.add_argument("--act", default="pow", choices=["pow", "relu", "prod"])
    ap.add_argument("--p", type=int, nargs="+", default=[71])
    ap.add_argument("--penalty", nargs="+", default=["l2"], choices=["l2", "l2nu", "l2q"])
    ap.add_argument("--q", type=float, default=None, help="penalty exponent for --penalty l2q: lam sum_i ||theta_i||^q (program exponent nu/q)")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--lam", type=float, default=1e-4)
    ap.add_argument("--m", type=int, default=500)
    ap.add_argument("--steps", type=int, default=40000)
    ap.add_argument("--lr0", type=float, default=0.05)
    ap.add_argument("--lrmax", type=float, default=12.8)
    ap.add_argument("--init_scale", type=float, default=1.0)
    ap.add_argument("--clip", type=float, default=0.0, help="global grad-norm clip (0 = off)")
    ap.add_argument("--wd_warmup", type=int, default=0, help="max steps with the penalty off at the start (0 = no warm-up)")
    ap.add_argument("--escape_frac", type=float, default=0.97, help="warm-up ends early once train CE < escape_frac * ln p")
    ap.add_argument("--lrmax2", type=float, default=None, help="lr cap after the warm-up (default: same as --lrmax)")
    ap.add_argument("--resume", default=None, help="run dir (cell) whose seed{seed}/final.safetensors to start from; lr = --lrmax2 from step 1")
    ap.add_argument("--tied", action="store_true", help="one shared embedding; exact multiset reduction of the table")
    ap.add_argument("--chunk", type=int, default=400_000, help="table rows per chunk")
    ap.add_argument("--grid", action="store_true", help="untied only: broadcast-grid forward instead of one-hot GEMMs (same maths, ~3x faster; tag unchanged)")
    ap.add_argument("--trace_every", type=int, default=0, help="fp64 analysis every this many steps, stored in metrics['trace']")
    ap.add_argument("--out", default=None, help="runs directory (default heracles/runs/NS)")
    args = ap.parse_args()
    if args.out:
        OUT = Path(args.out)
    if "l2q" in args.penalty and args.q is None:
        ap.error("--penalty l2q needs --q")
    if args.lrmax2 is None:
        args.lrmax2 = args.lrmax
    args.resume_tag = Path(args.resume).name.split("_lam")[1].split("_")[0] if args.resume else ""
    print("device", DEV, "args", vars(args), flush=True)
    for p in args.p:
        data = None
        for kind in args.penalty:
            for s in args.seeds:
                data = run(args, p, kind, s, data)
