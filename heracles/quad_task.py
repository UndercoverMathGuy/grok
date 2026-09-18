"""quad_task.py (from quad_morwani.py) -- Morwani et al. (ICLR 2024, App. B.1) cyclic-group recipe, quadratic activation, with the
penalty as the only dial: plain l2 (our theorem: K* frequencies) vs L_{2,3} (theirs: all frequencies).

Recipe copied: 1 hidden layer, width m = 500, x^2 activation, no biases, FULL dataset (all p^2 pairs, no split),
full-batch gradient descent, 40000 steps, lr 0.05 doubled at steps 1e3..10e3 (their final 51.2; plain l2 diverges there, so --lrmax caps it), penalty coefficient 1e-4.
Input = one-hot(a) (+) one-hot(b); neuron i = (u_i, v_i, w_i). Degree nu = 3 (x^2 doubles the first layer).
    l2 : lam * sum(theta^2)                 l23: lam * sum_i (||u_i||^2 + ||v_i||^2 + ||w_i||^2)^{3/2}
Per run: clean-logit Fourier amplitudes, frequency count at 2% / 5% energy share, Morwani-style neuron histogram
over top frequencies, per-neuron top-frequency share, min margin, ||theta||^2, gamma = margin / ||theta||^3.
Usage: uv run python heracles/quad_morwani.py --p 71 113 --penalty l2 l23 --seeds 0 1 2 3
"""
import argparse, json, math, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "runs" / "QT"
TASKS = {"add": lambda a, b, p: (a + b) % p, "sq": lambda a, b, p: (a * a + b * b) % p, "norm": lambda a, b, p: (a * a + a * b + b * b) % p}
DEV = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False   # exact fp32 (user veto on TF32)


def full_data(p, task="add"):
    a, b = torch.meshgrid(torch.arange(p), torch.arange(p), indexing="ij")
    return a.reshape(-1), b.reshape(-1), TASKS[task](a, b, p).reshape(-1)


def forward(U, V, W, a, b):
    x = U[:, a] + V[:, b]                      # (m, N)
    return (W @ (x * x)).T                     # (N, p)


def penalty(U, V, W, kind, lam):
    if kind == "l2":
        return lam * (U.square().sum() + V.square().sum() + W.square().sum())
    n2 = U.square().sum(1) + V.square().sum(1) + W.square().sum(0)
    return lam * n2.pow(1.5).sum()


def analyse(U, V, W, p, task="add"):
    U, V, W = (t.detach().double().cpu() for t in (U, V, W))
    a, b, y = full_data(p, task)
    z = forward(U, V, W, a, b)                                   # (p^2, p)
    zy = z.gather(1, y[:, None]).squeeze(1)
    zo = z.clone(); zo[torch.arange(p * p), y] = -float("inf")
    margin = float((zy - zo.max(1).values).min())
    acc = float((z.argmax(1) == y).float().mean())
    n2 = float(U.square().sum() + V.square().sum() + W.square().sum())
    Z = z.reshape(p, p, p).numpy()                                # [a, b, c]
    A, B = np.meshgrid(np.arange(p), np.arange(p), indexing="ij")
    Y = y.reshape(p, p).numpy()
    # clean projection: average of z[c = f(a,b) - D] over the level set, indexed by offset D from the label
    g = np.array([Z[A, B, (Y - D) % p].mean() for D in range(p)])
    amp = np.abs(np.fft.fft(g) / p)[1:(p - 1) // 2 + 1]
    en = amp ** 2 / (amp ** 2).sum()
    # output-side spectrum of the readout (rows W[c, i] over c): which output frequencies are used
    Fw = np.abs(np.fft.rfft(W.numpy(), axis=0)[1:, :]) ** 2      # (nf, m)
    wtot = Fw.sum(0) + 1e-30; wwts = wtot / wtot.sum()
    w_top_share = float(((Fw.max(0) / wtot) * wwts).sum())
    w_hist = np.bincount(Fw.argmax(0) + 1, weights=wwts, minlength=(p - 1) // 2 + 1)[1:]
    # input-side: energy of u_i in the cos(k a) basis vs the cos(k a^2) basis (both normalised, non-orthogonal for a^2)
    Un = U.numpy(); th = 2 * np.pi * np.arange(p) / p; sq = (np.arange(p) ** 2) % p
    Fu = np.abs(np.fft.rfft(Un, axis=1)[:, 1:]) ** 2
    tot = Fu.sum(1) + 1e-30; wts = tot / tot.sum()
    top_share = float(((Fu.max(1) / tot) * wts).sum())
    hist = np.bincount(Fu.argmax(1) + 1, weights=wts, minlength=(p - 1) // 2 + 1)[1:]
    nf = (p - 1) // 2
    Bsq = np.stack([np.cos(k * th[sq]) for k in range(1, nf + 1)] + [np.sin(k * th[sq]) for k in range(1, nf + 1)], 1)
    Bsq = Bsq / np.linalg.norm(Bsq, axis=0, keepdims=True)
    coef, *_ = np.linalg.lstsq(Bsq, Un.T, rcond=None)              # (2nf, m)
    fit = Bsq @ coef
    r2_sq = float(1 - ((Un.T - fit) ** 2).sum() / (Un.T ** 2).sum())
    esq = coef[:nf] ** 2 + coef[nf:] ** 2                           # (nf, m) energy per a^2-frequency
    sq_top_share = float(((esq.max(0) / (esq.sum(0) + 1e-30)) * wts).sum())
    sq_hist = np.bincount(esq.argmax(0) + 1, weights=wts, minlength=nf + 1)[1:]
    return dict(margin=margin, acc=acc, norm2=n2, gamma=margin / n2 ** 1.5, amp=amp.tolist(),
                K_2pct=int((en >= 0.02).sum()), K_5pct=int((en >= 0.05).sum()),
                top_share=top_share, neuron_hist=hist.tolist(), n_freq_hist_1pct=int((hist >= 0.01).sum()),
                w_top_share=w_top_share, w_hist=w_hist.tolist(), n_wfreq_1pct=int((w_hist >= 0.01).sum()),
                r2_sq_basis=r2_sq, sq_top_share=sq_top_share, sq_hist=sq_hist.tolist(), n_sqfreq_1pct=int((sq_hist >= 0.01).sum()))


def run(p, kind, seed, lam, m, steps, lr0, lrmax, init_scale=1.0, task="add"):
    out = OUT / f"{task}_p{p}_{kind}_lam{lam:g}_lrmax{lrmax:g}_init{init_scale:g}" / f"seed{seed}"
    if (out / "metrics.json").exists():
        print("skip", out); return
    out.mkdir(parents=True, exist_ok=True)
    g = torch.Generator().manual_seed(seed)
    U = (torch.rand(m, p, generator=g) * 2 - 1) / math.sqrt(2 * p)
    V = (torch.rand(m, p, generator=g) * 2 - 1) / math.sqrt(2 * p)
    W = (torch.rand(p, m, generator=g) * 2 - 1) / math.sqrt(m)
    U, V, W = ((init_scale * t).to(DEV).requires_grad_(True) for t in (U, V, W))
    a, b, y = (t.to(DEV) for t in full_data(p, task))
    lr, log, t0 = lr0, [], time.time()
    for step in range(1, steps + 1):
        if step in (1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000):
            lr = min(lr * 2, lrmax)
        z = forward(U, V, W, a, b)
        loss = F.cross_entropy(z, y) + penalty(U, V, W, kind, lam)
        grads = torch.autograd.grad(loss, (U, V, W))
        with torch.no_grad():
            for t, gr in zip((U, V, W), grads):
                t -= lr * gr
        if step % 2000 == 0 or step == steps:
            with torch.no_grad():
                acc = float((z.argmax(1) == y).float().mean()); l = float(loss)
            log.append(dict(step=step, loss=l, acc=acc, lr=lr))
            print(f"  p{p} {kind} s{seed} step {step:5d} lr {lr:6.2f} loss {l:.3e} acc {acc:.4f}  [{time.time()-t0:.0f}s]", flush=True)
            if not math.isfinite(l):
                break
    res = analyse(U, V, W, p, task)
    res.update(task=task, p=p, penalty=kind, lam=lam, seed=seed, m=m, steps=steps, lrmax=lrmax, init_scale=init_scale, log=log)
    (out / "metrics.json").write_text(json.dumps(res, indent=1))
    save_file({"U": U.detach().cpu(), "V": V.detach().cpu(), "W": W.detach().cpu()}, str(out / "final.safetensors"))
    print(f"DONE {task} p{p} {kind} seed{seed}: acc {res['acc']:.4f} margin {res['margin']:.4f} gamma {res['gamma']:.3e} "
          f"K_2pct {res['K_2pct']} K_5pct {res['K_5pct']} | u: cos(ka) top-share {res['top_share']:.3f} #f {res['n_freq_hist_1pct']}, "
          f"cos(ka^2) R2 {res['r2_sq_basis']:.3f} top-share {res['sq_top_share']:.3f} #f {res['n_sqfreq_1pct']} | w: top-share {res['w_top_share']:.3f} #f {res['n_wfreq_1pct']}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--p", type=int, nargs="+", default=[71, 113])
    ap.add_argument("--penalty", nargs="+", default=["l2", "l23"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--lam", type=float, default=1e-4)
    ap.add_argument("--m", type=int, default=500)
    ap.add_argument("--steps", type=int, default=40000)
    ap.add_argument("--lr0", type=float, default=0.05)
    ap.add_argument("--lrmax", type=float, nargs="+", default=[12.8])
    ap.add_argument("--init_scale", type=float, default=1.0)
    ap.add_argument("--task", nargs="+", default=["add"])
    args = ap.parse_args()
    print("device", DEV)
    for task in args.task:
        for p in args.p:
            for lrmax in args.lrmax:
                for kind in args.penalty:
                    for s in args.seeds:
                        run(p, kind, s, args.lam, args.m, args.steps, args.lr0, lrmax, args.init_scale, task)
