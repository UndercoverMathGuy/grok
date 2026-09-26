"""pax_run.py -- prime-axis diagnosis / recipe trainer (LOCAL CPU).

Same net, same task, same init draw and the same fp64 instruments as `nsum_rep.py` (imported verbatim), plus
three things that trainer does not have:

  * a per-step trace of  ||theta||^2  and of the PROGRESS TIME
        tau(t) = sum_{s<=t} lr_s * ||theta_s||^(nu-2)          (nu = 3 here, so lr * ||theta||)
    which is the natural clock of the escape-from-zero flow (see PRIME_AXIS.md sec 2);
  * a constant-lr option (`--lr_const`) instead of nsum.py's 0.05-doubled-to-12.8 ramp;
  * a `--stop_ce` early exit, so an escape-time measurement costs only the steps up to escape.

Everything else (net, loss, penalty, clipping, warm-up semantics) is nsum_rep's.
Outputs -> heracles/FINAL_RUNS/PAX/<tag>/seed<k>/metrics.json (+ final.safetensors unless --nosave).
"""
import argparse, json, math, os, sys, time
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import save_file, load_file

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from nsum_rep import Data, analyse, grid_forward, penalty  # noqa: E402

OUT = Path(os.environ.get("PAX_OUT") or (ROOT / "FINAL_RUNS" / "PAX"))
DEV = os.environ.get("PAX_DEV") or ("cuda" if torch.cuda.is_available() else "cpu")
LR_DOUBLE_STEPS = (1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000)


def init_weights(seeds, p, m, n, scale):
    """nsum_rep.py's draw, verbatim: uniform(-1,1)/sqrt(n p) for the embeddings, /sqrt(m) for the readout."""
    Es, Ws = [], []
    for s in seeds:
        gen = torch.Generator().manual_seed(s)
        E = (torch.rand(n, m, p, generator=gen) * 2 - 1) / math.sqrt(n * p)
        W = (torch.rand(p, m, generator=gen) * 2 - 1) / math.sqrt(m)
        Es.append(scale * E); Ws.append(scale * W)
    return torch.stack(Es), torch.stack(Ws)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--act", default="pow", choices=["pow", "relu", "prod"])
    ap.add_argument("--p", type=int, required=True)
    ap.add_argument("--m", type=int, default=500)
    ap.add_argument("--seeds", type=int, nargs="+", default=[100])
    ap.add_argument("--init_scale", type=float, default=4.0)
    ap.add_argument("--lam", type=float, default=1e-4)
    ap.add_argument("--lr0", type=float, default=0.05)
    ap.add_argument("--lrmax", type=float, default=12.8)
    ap.add_argument("--lr_const", type=float, default=0.0, help="if > 0: constant lr, no doubling ramp")
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--clip", type=float, default=0.0)
    ap.add_argument("--wd_warmup", type=int, default=0, help="max steps with lambda = 0; 0 = penalty on from step 1")
    ap.add_argument("--escape_frac", type=float, default=0.8, help="warm-up ends when CE < escape_frac * ln p")
    ap.add_argument("--lr_after", type=float, default=0.0, help="lr cap after the warm-up ends (0 = keep)")
    ap.add_argument("--stop_ce", type=float, default=0.0, help="stop as soon as CE < stop_ce * ln p (0 = never)")
    ap.add_argument("--achunk", type=int, default=0)
    ap.add_argument("--report", type=int, default=250)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--nosave", action="store_true")
    ap.add_argument("--threads", type=int, default=3)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)

    p, m, n = a.p, a.m, 2
    nu = 2 if a.act == "relu" else 3
    tag = a.tag or (f"pax_{a.act}_p{p}_m{m}_init{a.init_scale:g}_lam{a.lam:g}"
                    + (f"_lrc{a.lr_const:g}" if a.lr_const else f"_ramp{a.lrmax:g}")
                    + (f"_wdw{a.wd_warmup}e{a.escape_frac:g}" if a.wd_warmup else "")
                    + (f"_clip{a.clip:g}" if a.clip else "") + f"_st{a.steps}")
    cell = OUT / tag
    seeds = a.seeds
    S = len(seeds)
    E, W = init_weights(seeds, p, m, n, a.init_scale)
    E = E.to(DEV).requires_grad_(True); W = W.to(DEV).requires_grad_(True)
    N = p * p
    ar = torch.arange(p, device=DEV)
    ach = a.achunk or p
    ychunks = [((ar[a0:a0 + ach, None] + ar[None, :]) % p).reshape(-1) for a0 in range(0, p, ach)]

    lr = torch.full((S, 1, 1), a.lr_const if a.lr_const else a.lr0, device=DEV)
    lrcap = torch.full((S, 1, 1), a.lr_const if a.lr_const else a.lrmax, device=DEV)
    warm = torch.full((S,), a.wd_warmup > 0, device=DEV)
    escape = torch.full((S,), -1, device=DEV, dtype=torch.long)
    thr = a.escape_frac * math.log(p)
    stop_thr = a.stop_ce * math.log(p) if a.stop_ce else -1.0
    tau = torch.zeros(S, device=DEV)                     # progress time  sum lr * ||theta||^(nu-2)
    first = {k: [None] * S for k in ("ce80", "ce50", "acc99")}
    logs = [[] for _ in seeds]; t0 = time.time()
    lnp = math.log(p)
    print(f"start {tag} seeds {seeds} ln p {lnp:.4f} thr {thr:.4f}", flush=True)

    for step in range(1, a.steps + 1):
        if not a.lr_const and step in LR_DOUBLE_STEPS:
            lr = torch.minimum(lr * 2, lrcap)
        report = step % a.report == 0 or step == a.steps or step == 1
        gE = torch.zeros_like(E); gW = torch.zeros_like(W)
        ce = torch.zeros(S, device=DEV); correct = torch.zeros(S, device=DEV)
        for ci, a0 in enumerate(range(0, p, ach)):
            z = grid_forward(E, W, a.act, a0, min(a0 + ach, p))
            y = ychunks[ci]
            l = F.cross_entropy(z, y[None].expand(S, -1), reduction="none").sum(1) / N
            dE, dW = torch.autograd.grad(l.sum(), (E, W))
            gE += dE; gW += dW; ce += l.detach()
            if report:
                correct += (z.argmax(1) == y[None]).float().sum(1)
        with torch.no_grad():
            n2 = E.square().sum((1, 2, 3)) + W.square().sum((1, 2))          # (S,)
            tau += lr.flatten() * n2.sqrt() ** (nu - 2)
        if a.wd_warmup:
            ended = warm & ((ce < thr) | (step >= a.wd_warmup))
            escape = torch.where(ended, torch.full_like(escape, step), escape)
            warm = warm & ~ended
            if a.lr_after:
                lrcap = torch.where(ended[:, None, None], torch.full_like(lrcap, a.lr_after), lrcap)
                lr = torch.where(ended[:, None, None], torch.full_like(lr, a.lr_after), torch.minimum(lr, lrcap))
        pen = penalty(E, W, "l2", a.lam, nu, 2.0)
        dE, dW = torch.autograd.grad(pen, (E, W))
        on = (~warm).float()
        gE += dE * on[:, None, None, None]; gW += dW * on[:, None, None]
        with torch.no_grad():
            if a.clip:
                gn = torch.sqrt(gE.square().sum((1, 2, 3)) + gW.square().sum((1, 2)))
                sc = torch.clamp(a.clip / gn, max=1.0)
                gE *= sc[:, None, None, None]; gW *= sc[:, None, None]
            E -= lr[:, :, :, None] * gE; W -= lr * gW
        cel = ce.tolist()
        for i in range(S):
            if first["ce80"][i] is None and cel[i] < 0.8 * lnp:
                first["ce80"][i] = step
            if first["ce50"][i] is None and cel[i] < 0.5 * lnp:
                first["ce50"][i] = step
        if report:
            n2l = n2.tolist(); taul = tau.tolist(); accl = (correct / N).tolist()
            for i in range(S):
                if first["acc99"][i] is None and accl[i] >= 0.99:
                    first["acc99"][i] = step
                logs[i].append(dict(step=step, ce=cel[i], acc=accl[i], norm2=n2l[i], lr=float(lr[i, 0, 0]),
                                    tau=taul[i], wd=bool(not warm[i]), t=time.time() - t0))
            print(f"  {step:6d} ce " + " ".join(f"{c:.3f}" for c in cel) + " acc " + " ".join(f"{x:.3f}" for x in accl)
                  + " n2 " + " ".join(f"{x:.0f}" for x in n2l) + " tau " + " ".join(f"{x:.3g}" for x in taul)
                  + f"  [{time.time() - t0:.0f}s]", flush=True)
            if not all(math.isfinite(c) for c in cel):
                print("NONFINITE -- stop", flush=True); break
        if stop_thr > 0 and all(c < stop_thr for c in cel):
            print(f"  stop_ce hit at step {step}", flush=True); break
    train_time = time.time() - t0

    data64 = Data(p, n, max(2000, 4_000_000 // p), DEV, dtype=torch.float64, prod=(a.act == "prod"))
    for i, s in enumerate(seeds):
        out = cell / f"seed{s}"; out.mkdir(parents=True, exist_ok=True)
        Ei, Wi = E[i].detach(), W[i].detach()
        if bool(torch.isfinite(Ei).all() and torch.isfinite(Wi).all()):
            res = analyse(Ei, Wi, data64, a.act)
        else:
            res = dict(diverged=True, acc=0.0, margin=float("nan"), gamma=float("nan"), K_2pct=-1, K_5pct=-1,
                       N_05=-1, N_10=-1, clean_frac=float("nan"), n_freq_hist_1pct=-1, nu=nu)
        res.update(p=p, m=m, act=a.act, lam=a.lam, seed=s, steps=step, init_scale=a.init_scale, clip=a.clip,
                   lr_const=a.lr_const, lrmax=a.lrmax, wd_warmup=a.wd_warmup, escape_frac=a.escape_frac,
                   lr_after=a.lr_after, escape_step=(int(escape[i]) if int(escape[i]) >= 0 else None),
                   tau_final=float(tau[i]), t_ce80=first["ce80"][i], t_ce50=first["ce50"][i],
                   t_acc99=first["acc99"][i], train_time=train_time, log=logs[i], trainer="pax_run")
        (out / "metrics.json").write_text(json.dumps(res, indent=1))
        if not a.nosave:
            save_file({"E": Ei.cpu().contiguous(), "W": Wi.cpu().contiguous()}, str(out / "final.safetensors"))
        print(f"DONE {tag} seed{s}: acc {res['acc']:.5f} margin {res['margin']:.4f} gamma {res['gamma']:.4e} "
              f"K_5pct {res['K_5pct']} K_2pct {res['K_2pct']} N_05 {res['N_05']} clean {res['clean_frac']:.3f} "
              f"norm2 {res.get('norm2', float('nan')):.0f} t_ce80 {res['t_ce80']} tau {res['tau_final']:.4g}", flush=True)
    print("CELL DONE", tag, f"{train_time:.0f}s", flush=True)


if __name__ == "__main__":
    main()
