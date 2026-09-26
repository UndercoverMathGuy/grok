"""initaudit_cont.py -- init-scale audit: batched-seed x^2 trainer with a periodic fp64 instrument TRACE.

Two modes, both on the two-operand task a + b = c (mod p), activation x^2 (homogeneity nu = 3), width m, no biases,
full-batch GD on the whole p^2 table, penalty lam * sum(theta^2) (q = 2, plain l2).

  --resume CELLDIR : start from the saved endpoints of an archived cell (continuation test, H2).
                     lr is held at --lrmax for the whole continuation (the archived runs ended at lr 12.8).
  (no --resume)    : fresh init, weights = init_scale * U(-1,1)/sqrt(2p) (embeddings) and /sqrt(m) (readout),
                     lr 0.05 doubled at steps 1e3..1e4 capped at --lrmax  (the archived recipe, H6).

Every --trace_every steps the fp64 analyser of nsum_rep.py is run and margin / gamma / norm2 / clean_frac /
K_2pct / K_5pct / N_05 are appended to the trace, so the COUNT TRAJECTORY is recorded, not just the endpoint.

Usage:
  uv run python heracles/initaudit_cont.py --p 71 --seeds 100 101 102 103 --steps 160000 --trace_every 10000 \
      --resume heracles/FINAL_RUNS/REP/n2_pow_p71_l2q2_lam0.0001_lrmax12.8_init1_m500_st40000 --tag cont_init1
"""
import argparse, json, math, os, time
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import save_file, load_file

import sys
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from nsum_rep import Data, analyse, grid_forward, penalty, LR_DOUBLE_STEPS, DEV  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--p", type=int, default=71)
ap.add_argument("--seeds", type=int, nargs="+", default=[100, 101, 102, 103])
ap.add_argument("--m", type=int, default=500)
ap.add_argument("--lam", type=float, default=1e-4)
ap.add_argument("--q", type=float, default=2.0)
ap.add_argument("--steps", type=int, default=160000)
ap.add_argument("--lr0", type=float, default=0.05)
ap.add_argument("--lrmax", type=float, default=12.8)
ap.add_argument("--init_scale", type=float, default=1.0)
ap.add_argument("--resume", default=None)
ap.add_argument("--trace_every", type=int, default=10000)
ap.add_argument("--report", type=int, default=5000)
ap.add_argument("--tag", required=True)
ap.add_argument("--out", default=str(ROOT / "FINAL_RUNS" / "INITAUDIT"))
a = ap.parse_args()

p, m, S = a.p, a.m, len(a.seeds)
OUT = Path(a.out) / a.tag
OUT.mkdir(parents=True, exist_ok=True)
print(f"device {DEV} | {vars(a)}", flush=True)

Es, Ws = [], []
for s in a.seeds:
    if a.resume:
        st = load_file(str(Path(a.resume) / f"seed{s}" / "final.safetensors"))
        Es.append(st["E"]); Ws.append(st["W"])
    else:
        gen = torch.Generator().manual_seed(s)
        E = (torch.rand(2, m, p, generator=gen) * 2 - 1) / math.sqrt(2 * p)
        W = (torch.rand(p, m, generator=gen) * 2 - 1) / math.sqrt(m)
        Es.append(a.init_scale * E); Ws.append(a.init_scale * W)
E = torch.stack(Es).to(DEV).requires_grad_(True)
W = torch.stack(Ws).to(DEV).requires_grad_(True)

N = p * p
ar = torch.arange(p, device=DEV)
y = ((ar[:, None] + ar[None, :]) % p).reshape(-1)
lr = torch.full((S, 1, 1), a.lrmax if a.resume else a.lr0, device=DEV)
lrcap = torch.full((S, 1, 1), a.lrmax, device=DEV)
ADEV = "cpu"   # fp64 analysis always on CPU (MPS has no float64)
data64 = Data(p, 2, 400_000, ADEV, dtype=torch.float64, prod=False)
traces = [[] for _ in a.seeds]
t0 = time.time()


def trace(step):
    for i, s in enumerate(a.seeds):
        r = analyse(E[i].detach().to(ADEV), W[i].detach().to(ADEV), data64, "pow")
        traces[i].append({k: r[k] for k in ("margin", "acc", "norm2", "gamma", "clean_frac",
                                            "K_2pct", "K_5pct", "N_05", "N_10")} | dict(step=step, amp=r["amp"], b2=r["b2"]))
    print("  TRACE step %6d | " % step + " || ".join(
        f"s{s} K2 {traces[i][-1]['K_2pct']:2d} K5 {traces[i][-1]['K_5pct']:2d} N05 {traces[i][-1]['N_05']:2d} "
        f"gam {traces[i][-1]['gamma']:.3e} n2 {traces[i][-1]['norm2']:7.1f} mar {traces[i][-1]['margin']:6.3f} "
        f"cl {traces[i][-1]['clean_frac']:.3f}" for i, s in enumerate(a.seeds)), flush=True)


trace(0)
logs = [[] for _ in a.seeds]
for step in range(1, a.steps + 1):
    if not a.resume and step in LR_DOUBLE_STEPS:
        lr = torch.minimum(lr * 2, lrcap)
    report = step % a.report == 0 or step == a.steps
    z = grid_forward(E, W, "pow", 0, p)
    l = F.cross_entropy(z, y[None].expand(S, -1), reduction="none").sum(1) / N
    pen = penalty(E, W, "l2q", a.lam, 3, a.q)
    gE, gW = torch.autograd.grad(l.sum() + pen, (E, W))
    with torch.no_grad():
        E -= lr[:, :, :, None] * gE
        W -= lr * gW
    if report:
        with torch.no_grad():
            acc = (z.argmax(1) == y[None]).float().mean(1)
        cel = l.detach().tolist(); accl = acc.tolist()
        for i in range(S):
            logs[i].append(dict(step=step, ce=cel[i], acc=accl[i], t=time.time() - t0))
        print(f"  {a.tag} step {step:6d} ce " + " ".join(f"{c:.2e}" for c in cel)
              + " acc " + " ".join(f"{x:.3f}" for x in accl) + f"  [{time.time()-t0:.0f}s]", flush=True)
    if a.trace_every and step % a.trace_every == 0 and step != a.steps:
        trace(step)
trace(a.steps)

for i, s in enumerate(a.seeds):
    d = OUT / f"seed{s}"; d.mkdir(parents=True, exist_ok=True)
    r = analyse(E[i].detach().to(ADEV), W[i].detach().to(ADEV), data64, "pow")
    r.update(p=p, n=2, act="pow", penalty="l2q", q=a.q, lam=a.lam, seed=s, m=m, steps=a.steps, lrmax=a.lrmax,
             init_scale=a.init_scale, resume=a.resume, tag=a.tag, train_time=time.time() - t0,
             final_ce=logs[i][-1]["ce"], log=logs[i], trace=traces[i], trainer="initaudit_cont")
    (d / "metrics.json").write_text(json.dumps(r, indent=1))
    save_file({"E": E[i].detach().cpu().contiguous(), "W": W[i].detach().cpu().contiguous()}, str(d / "final.safetensors"))
    print(f"DONE {a.tag} seed{s}: acc {r['acc']:.4f} margin {r['margin']:.4f} gamma {r['gamma']:.3e} "
          f"K_2pct {r['K_2pct']} K_5pct {r['K_5pct']} N_05 {r['N_05']} clean {r['clean_frac']:.3f}", flush=True)
print("CELL DONE", a.tag, flush=True)
