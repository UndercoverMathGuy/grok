"""Round-4 prereg cohort S: fixed-support gamma(K) for relu-4 (P-N12/P-N13).

relu-4 (onehot -> E1 -> U,V -> relu -> W1 -> W), cohort-A recipe, except
E1 = A @ B_S with B_S the fixed orthonormal cos/sin rows of an
exhaustively-optimal support S (K in {3,4,5,6}), so only frequencies in S
can enter. ||A @ B_S||_F = ||A||_F: wd accounting unchanged, nu stays 4.
A is initialised as the PROJECTION of the standard E1 init onto span(B_S),
so per-frequency init statistics match the normal lottery. The same 12
seeds are used for every K (paired). K=0 = unmasked control: full 113-row
Fourier basis (orthogonal reparameterization only; P-N13 checks Adam's
coordinate-dependence is inert).

Train: `uv run python nudial/fixed_support.py`         (all groups, sequential)
Score: `uv run python nudial/fixed_support.py --score`
"""
import argparse, json, math, time
from pathlib import Path
import numpy as np
import torch
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_nudial import fourier_basis

P, D, M_HID, R = 113, 128, 512, 128
NU = 4
LR, WD, (B1, B2), EPS = 1e-3, 1.0, (0.9, 0.98), 1e-8
EPOCHS, WARMUP = 20_000, 10
INIT_SCALE = 1.0
SEEDS = np.random.default_rng(20260828).integers(200000, 300000, 12).tolist()
SUPPORTS = {3: (1, 6, 43), 4: (1, 7, 11, 43), 5: (1, 14, 20, 44, 52),
            6: (1, 5, 12, 15, 38, 46), 0: ()}          # 0 = unmasked control
OUT = Path(__file__).resolve().parent.parent / "runs_nudial" / "S"
DEV = ("cuda" if torch.cuda.is_available()
       else "mps" if torch.backends.mps.is_available() else "cpu")

BASIS = fourier_basis(P)                                # (p, p) orthonormal

def basis_rows(K):
    if K == 0:
        return torch.from_numpy(BASIS).float()          # full square basis
    rows = []
    for k in SUPPORTS[K]:
        rows += [BASIS[2 * k - 1], BASIS[2 * k]]
    return torch.from_numpy(np.stack(rows)).float()     # (2K, p)

def splits():
    """Per-seed train/test index sets (frac 0.5, data_seed = seed % 1000,
    same convention as train_nudial)."""
    tr, te = [], []
    n_tr = (P * P) // 2
    for s in SEEDS:
        perm = np.random.default_rng(s % 1000).permutation(P * P)
        tr.append(perm[:n_tr]); te.append(perm[n_tr:])
    return (torch.tensor(np.stack(tr)), torch.tensor(np.stack(te)))

def init_group(K, B):
    """Stacked params for 12 seeds. A = E1_init @ B^T (projection init)."""
    pr = {}
    def draw(shape, fan, gens):
        return torch.stack([INIT_SCALE * torch.randn(*shape, generator=g)
                            / math.sqrt(fan) for g in gens])
    gens = [torch.Generator().manual_seed(s) for s in SEEDS]
    E1 = draw((D, P), P, gens)
    pr["A"] = E1 @ B.T.double().float() if B.dtype == torch.float32 else E1 @ B.T
    gens = [torch.Generator().manual_seed(s + 1) for s in SEEDS]
    pr["U"] = draw((M_HID, D), D, gens)
    gens = [torch.Generator().manual_seed(s + 2) for s in SEEDS]
    pr["V"] = draw((M_HID, D), D, gens)
    gens = [torch.Generator().manual_seed(s + 3) for s in SEEDS]
    pr["W1"] = draw((R, M_HID), M_HID, gens)
    gens = [torch.Generator().manual_seed(s + 4) for s in SEEDS]
    pr["W"] = draw((P, R), R, gens)
    return {k: v.to(DEV).requires_grad_(True) for k, v in pr.items()}

def forward(pr, B, a_idx, b_idx):
    """a_idx/b_idx (S, N) per-run token ids -> logits (S, N, p)."""
    E = pr["A"] @ B.to(DEV)                             # (S, d, p)
    ga = E.gather(2, a_idx[:, None, :].expand(-1, D, -1))
    gb = E.gather(2, b_idx[:, None, :].expand(-1, D, -1))
    h = torch.relu(pr["U"] @ ga + pr["V"] @ gb)         # (S, m, N)
    return (pr["W"] @ (pr["W1"] @ h)).transpose(1, 2)   # (S, N, p)

def ce(logits, labels):
    zy = logits.gather(2, labels[:, :, None]).squeeze(2)
    others = logits.masked_fill(
        torch.nn.functional.one_hot(labels, P).bool(), float("-inf"))
    u = torch.logsumexp(others, 2) - zy
    return torch.nn.functional.softplus(u).mean(1)      # (S,)

def run_group(K):
    B = basis_rows(K)
    pr = init_group(K, B)
    tr_idx, te_idx = splits()
    a_tr = (tr_idx // P).to(DEV); b_tr = (tr_idx % P).to(DEV)
    lab_tr = ((a_tr + b_tr) % P)
    m_st = {k: torch.zeros_like(v) for k, v in pr.items()}
    v_st = {k: torch.zeros_like(v) for k, v in pr.items()}
    t0 = time.time()
    for ep in range(EPOCHS):
        lr = LR * min((ep + 1) / WARMUP, 1.0)
        loss = ce(forward(pr, B, a_tr, b_tr), lab_tr)
        grads = torch.autograd.grad(loss.sum(), list(pr.values()))
        with torch.no_grad():
            for (k, q), g in zip(pr.items(), grads):
                m_st[k].mul_(B1).add_(g, alpha=1 - B1)
                v_st[k].mul_(B2).addcmul_(g, g, value=1 - B2)
                q.mul_(1 - lr * WD)
                q.sub_(lr * m_st[k] / (v_st[k].sqrt() + EPS))
        if (ep + 1) % 2000 == 0:
            with torch.no_grad():
                a_te = (te_idx // P).to(DEV); b_te = (te_idx % P).to(DEV)
                acc = (forward(pr, B, a_te, b_te).argmax(2)
                       == (a_te + b_te) % P).float().mean(1)
            print(f"  K={K} ep {ep+1}: loss {loss.median():.2e} "
                  f"test acc med {acc.median():.4f} "
                  f"({(ep+1)/(time.time()-t0):.1f} ep/s)", flush=True)
    # ---- final scoring in f64 on CPU
    prc = {k: v.detach().cpu().double() for k, v in pr.items()}
    Bc = B.double().cpu()
    a = torch.arange(P).repeat_interleave(P); b = torch.arange(P).repeat(P)
    lab = (a + b) % P
    E = prc["A"] @ Bc
    h = torch.relu(prc["U"] @ E[:, :, a] + prc["V"] @ E[:, :, b])
    logits = (prc["W"] @ (prc["W1"] @ h)).transpose(1, 2)   # (S, p^2, p)
    zy = logits.gather(2, lab[None, :, None].expand(12, -1, -1)).squeeze(2)
    others = logits.masked_fill(
        torch.nn.functional.one_hot(lab, P)[None].bool(), float("-inf"))
    marg = (zy - others.max(2).values)                       # (S, p^2)
    g_min = marg.min(1).values
    pn2 = sum((v ** 2).flatten(1).sum(1) for v in prc.values())
    gamma = g_min / pn2 ** (NU / 2)
    # committee energy of the final logits (diag Fourier)
    bc, bs = BASIS[1::2], BASIS[2::2]
    u_cos = torch.from_numpy((np.einsum("np,nq->npq", bc, bc)
                              - np.einsum("np,nq->npq", bs, bs))
                             .reshape(56, -1) / np.sqrt(2))
    u_sin = torch.from_numpy((np.einsum("np,nq->npq", bs, bc)
                              + np.einsum("np,nq->npq", bc, bs))
                             .reshape(56, -1) / np.sqrt(2))
    Lc = (logits - logits.mean(2, keepdim=True))
    en = torch.stack([(u_cos @ Lc[i].reshape(P * P, P)).pow(2).sum(1)
                      + (u_sin @ Lc[i].reshape(P * P, P)).pow(2).sum(1)
                      for i in range(12)])                   # (S, 56)
    for i, s in enumerate(SEEDS):
        te = te_idx[i]
        acc = float((logits[i, te].argmax(1) == lab[te]).double().mean())
        tracc = float((logits[i, tr_idx[i]].argmax(1)
                       == lab[tr_idx[i]]).double().mean())
        rd = OUT / f"K{K}" / f"s{s}"
        rd.mkdir(parents=True, exist_ok=True)
        json.dump({"K": K, "support": SUPPORTS[K], "seed": s,
                   "recipe": dict(p=P, m=M_HID, d=D, r=R, nu=NU, lr=LR,
                                  wd=WD, betas=[B1, B2], eps=EPS,
                                  epochs=EPOCHS, frac=0.5,
                                  init_scale=INIT_SCALE,
                                  init="projected-E1", device=DEV),
                   "test_acc": acc, "train_acc": tracc,
                   "margin": float(g_min[i]), "pnorm2": float(pn2[i]),
                   "gamma": float(gamma[i]),
                   "energy": en[i].tolist()},
                  open(rd / "result.json", "w"), indent=1)
        torch.save(prc and {k: v[i] for k, v in prc.items()}, rd / "final.pt")
    print(f"K={K} done: gamma med {gamma.median():.3e} "
          f"grokked {(marg.min(1).values > 0).sum().item()}/12", flush=True)

def score():
    import glob
    from collections import defaultdict
    rows = defaultdict(list)
    for f in glob.glob(str(OUT / "K*" / "s*" / "result.json")):
        r = json.load(open(f))
        rows[r["K"]].append(r)
    meds = {}
    for K in sorted(rows):
        ok = [r for r in rows[K] if r["test_acc"] >= 0.99]
        if not ok: print(f"K={K}: 0 grokked of {len(rows[K])}"); continue
        gm = float(np.median([r["gamma"] for r in ok]))
        meds[K] = gm
        onS = [sum(r["energy"][k - 1] for k in r["support"])
               / sum(r["energy"]) for r in ok] if K else [float("nan")]
        print(f"K={K}: n_grok {len(ok)}/{len(rows[K])} gamma med {gm:.3e} "
              f"support-energy med {np.median(onS):.3f}")
    if 4 in meds and 5 in meds:
        ratio = meds[5] / meds[4]
        verdict = ("CONFIRM C8/C9" if ratio >= 1.1 else
                   "KILL C8/C9" if meds[4] >= meds[5] else "UNDERPOWERED")
        print(f"P-N12: gamma(5)/gamma(4) = {ratio:.2f} -> {verdict}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--groups", type=int, nargs="+", default=[4, 5, 3, 6, 0])
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    args = ap.parse_args()
    EPOCHS = args.epochs
    if args.score:
        score()
    else:
        print(f"device {DEV}; seeds {SEEDS}")
        for K in args.groups:
            run_group(K)
