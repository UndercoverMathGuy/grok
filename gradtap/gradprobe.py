"""Static gradient-response probes over existing checkpoints (no training).

Each probe loads a saved weight state, optionally applies a surgery (boost /
isometric swap / chimera), runs ONE backward pass on the run's own train set,
and projects grad(W_E) onto the task's Fourier (or S5 irrep) basis. The
readouts are first-order: the raw loss gradient, not the Adam step (the
historical surgeries all cold-restarted Adam, so g is the honest signal at
the surgery point).

Per-frequency stats on W_E columns (f64):
  E_k      weight energy of frequency k
  gE_k     grad energy of frequency k
  dot_k    <g_k, w_k>  — under plain GD, dE_k/dt = -2*lr*dot_k, so
           dot_k > 0 means the loss itself DECAYS the frequency (restoring),
           dot_k < 0 means the loss grows it.

Probes: a1 boost restoring force, a2 swap gradient response, a3 chimera
whose-lottery, d10 committee gradient interference vs width, e14 new-data
gradient content, e13 S5 irrep grad profile by wd, b0 epoch-0 grad-vs-
amplitude committee AUC (free by-product).

Usage:  python3 gradtap/gradprobe.py [--probe a1,a3,...] [--out notes/...]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "cloud"))

from safetensors.torch import load_file  # noqa: E402
from train_semifinal_torch import (  # noqa: E402
    PARAM_KEYS, Config, batched_final_logits, committee_from_coeffs,
    fourier_basis, grok_epoch, per_ex_stable_f32, train_test_split)

torch.set_grad_enabled(True)
DG2 = ROOT / "runs_degrok2"
B113 = fourier_basis(113)
NF = 56


# ---------------------------------------------------------------- core

def load_run(rd, epoch):
    cfg = Config.load(rd / "config.json")
    ck = load_file(str(rd / "checkpoints" / f"epoch_{epoch:05d}.safetensors"))
    z = np.load(rd / "spectra.npz")
    return cfg, {k: ck[k] for k in PARAM_KEYS}, z


def grad_we(params, cfg, sel_tokens, sel_labels):
    """Mean-CE grad of W_E at this state. Returns (g, W) as (d, v) f64 numpy."""
    pr = {k: v.detach().clone().unsqueeze(0) for k, v in params.items()}
    pr["embed.W_E"].requires_grad_()
    lg = batched_final_logits(pr, torch.from_numpy(sel_tokens), cfg)
    per = per_ex_stable_f32(lg, torch.from_numpy(sel_labels).unsqueeze(0))
    (g,) = torch.autograd.grad(per.mean(), pr["embed.W_E"])
    return g[0].double().numpy(), params["embed.W_E"].double().numpy()


def train_sel(cfg):
    from train_semifinal_torch import make_dataset
    tok, lab = make_dataset(cfg)
    m = train_test_split(cfg)
    return tok, lab, m


def fstats(g, W, p=113):
    """Per-frequency (E_k, gE_k, dot_k) over W_E columns."""
    gF, wF = g[:, :p] @ B113.T, W[:, :p] @ B113.T
    c, s = slice(1, None, 2), slice(2, None, 2)
    E = (wF[:, c] ** 2 + wF[:, s] ** 2).sum(0)[:NF]
    gE = (gF[:, c] ** 2 + gF[:, s] ** 2).sum(0)[:NF]
    dot = (gF[:, c] * wF[:, c] + gF[:, s] * wF[:, s]).sum(0)[:NF]
    return E, gE, dot


def freq_vecs(g, p=113):
    """(NF, 2d) flattened per-frequency grad component vectors."""
    gF = g[:, :p] @ B113.T
    return np.concatenate([gF[:, 1::2][:, :NF], gF[:, 2::2][:, :NF]], 0).T


def auc(score, members):
    """Rank AUC of score for committee membership."""
    lab = np.zeros(len(score), bool)
    lab[np.asarray(members) - 1] = True
    if lab.all() or not lab.any():
        return float("nan")
    r = np.argsort(np.argsort(score)) + 1
    return (r[lab].sum() - lab.sum() * (lab.sum() + 1) / 2) \
        / (lab.sum() * (~lab).sum())


def boost(W, k, alpha, p=113):
    Pk = np.outer(B113[2 * k - 1], B113[2 * k - 1]) \
        + np.outer(B113[2 * k], B113[2 * k])
    W2 = W.copy()
    W2[:, :p] = W[:, :p] + (alpha - 1.0) * (W[:, :p] @ Pk)
    return W2


def swap(W, k1, k2, p=113):
    W2 = W.copy()
    Wp = W2[:, :p]
    for a, b in ((2 * k1 - 1, 2 * k2 - 1), (2 * k1, 2 * k2)):
        va, vb = B113[a], B113[b]
        ca, cb = Wp @ va, Wp @ vb
        Wp = Wp - np.outer(ca, va) - np.outer(cb, vb) \
            + np.outer(ca, vb) + np.outer(cb, va)
    W2[:, :p] = Wp
    return W2


def runs_in(cell, min_tgen=0):
    out = []
    for rd in sorted((DG2 / cell).iterdir()):
        if not (rd / "spectra.npz").exists():
            continue
        z = np.load(rd / "spectra.npz")
        ge = grok_epoch(z)
        if ge < min_tgen:
            continue
        out.append((rd, committee_from_coeffs(z["coeffs"][-1]), ge))
    return out


def with_we(params, W):
    q = dict(params)
    q["embed.W_E"] = torch.from_numpy(W.astype(np.float32))
    return q


# ---------------------------------------------------------------- probes

def probe_a1():
    """Boost the run's own winner at alpha 1..8 at a mid-race state: does the
    loss gradient turn against the boosted frequency (dot_k flips +)?"""
    rows = []
    for rd, comm, ge in runs_in("add113_m512_f0.3_wd1", min_tgen=1500)[:4]:
        cfg, params, _ = load_run(rd, 1000)
        tok, lab, m = train_sel(cfg)
        st, sl = tok[m], lab[m]
        W0 = params["embed.W_E"].double().numpy()
        E0 = fstats(np.zeros_like(W0), W0)[0]
        k = max(comm, key=lambda q: E0[q - 1])           # current top member
        ctrl = int(np.argsort(E0)[NF // 2]) + 1          # median non-member
        if ctrl in comm:
            ctrl += 1
        for tag, kk in (("winner", k), ("nonmember", ctrl)):
            for a in (1.0, 1.5, 2.0, 3.0, 4.0, 8.0):
                Wb = boost(W0, kk, a)
                g, W = grad_we(with_we(params, Wb), cfg, st, sl)
                E, gE, dot = fstats(g, W)
                i = kk - 1
                rows.append(dict(run=rd.name, freq=kk, kind=tag, alpha=a,
                                 E=E[i], dot=dot[i],
                                 rel_rate=2 * dot[i] / E[i],
                                 gE_share=gE[i] / gE.sum()))
                print(f"a1 {rd.name} {tag} f{kk} a={a:<3g} "
                      f"dot={dot[i]:+.3e}  relrate={2*dot[i]/E[i]:+.3e}  "
                      f"gE%={100*gE[i]/gE.sum():.1f}", flush=True)
    return rows


def probe_a2():
    """Isometric swap leader<->non-member at a mid-race state: is the grad's
    frequency ranking preserved (state intact) or rerandomized?"""
    from scipy.stats import spearmanr
    rows = []
    for rd, comm, ge in runs_in("add113_m512_f0.3_wd1", min_tgen=1500)[:4]:
        cfg, params, _ = load_run(rd, 1000)
        tok, lab, m = train_sel(cfg)
        st, sl = tok[m], lab[m]
        g0, W0 = grad_we(params, cfg, st, sl)
        E0, gE0, dot0 = fstats(g0, W0)
        k1 = max(comm, key=lambda q: E0[q - 1])
        k2 = int(np.argsort(E0)[NF // 4]) + 1
        if k2 in comm:
            k2 += 1
        Ws = swap(W0, k1, k2)
        g1, W1 = grad_we(with_we(params, Ws), cfg, st, sl)
        E1, gE1, dot1 = fstats(g1, W1)
        rest = [i for i in range(NF) if i + 1 not in (k1, k2)]
        rho = float(spearmanr(gE0[rest], gE1[rest]).statistic)
        rows.append(dict(
            run=rd.name, leader=k1, target=k2, rho_rest=rho,
            leader_dot=(dot0[k1 - 1], dot1[k1 - 1]),
            target_dot=(dot0[k2 - 1], dot1[k2 - 1]),
            leader_gE=(gE0[k1 - 1], gE1[k1 - 1]),
            target_gE=(gE0[k2 - 1], gE1[k2 - 1])))
        print(f"a2 {rd.name} swap f{k1}<->f{k2}: rest-rho {rho:.3f}  "
              f"leader gE {gE0[k1-1]:.2e}->{gE1[k1-1]:.2e} "
              f"dot {dot0[k1-1]:+.2e}->{dot1[k1-1]:+.2e}  "
              f"target gE {gE0[k2-1]:.2e}->{gE1[k2-1]:.2e} "
              f"dot {dot0[k2-1]:+.2e}->{dot1[k2-1]:+.2e}", flush=True)
    return rows


def probe_a3():
    """Epoch-0 chimera (A's W_E + B's downstream): whose committee does the
    first gradient rank higher? Natural nets included as anchors."""
    cells = runs_in("add113_m512_f0.3_wd1") + runs_in("add113_m512_f0.3_wd3")
    byds = {}
    for rd, comm, ge in cells:
        byds.setdefault(rd.name.split("_")[0], []).append((rd, comm))
    rows = []
    for ds, rs in byds.items():
        loaded = []
        for rd, comm in rs:
            cfg, params, _ = load_run(rd, 0)
            loaded.append((rd, comm, cfg, params))
        tok, lab, m = train_sel(loaded[0][2])
        st, sl = tok[m], lab[m]
        for i, (rdA, commA, cfgA, pA) in enumerate(loaded):
            gN, _ = grad_we(pA, cfgA, st, sl)
            gEN = fstats(gN, pA["embed.W_E"].double().numpy())[1]
            rows.append(dict(kind="natural", run=rdA.name, ds=ds,
                             auc_own=auc(gEN, commA)))
            for j, (rdB, commB, cfgB, pB) in enumerate(loaded):
                if i == j:
                    continue
                chim = dict(pB)
                chim["embed.W_E"] = pA["embed.W_E"]
                g, W = grad_we(chim, cfgB, st, sl)
                gE = fstats(g, W)[1]
                rows.append(dict(kind="chimera", ds=ds,
                                 we=rdA.name, rest=rdB.name,
                                 auc_we_owner=auc(gE, commA),
                                 auc_rest_owner=auc(gE, commB)))
    nat = [r["auc_own"] for r in rows if r["kind"] == "natural"]
    cwe = [r["auc_we_owner"] for r in rows if r["kind"] == "chimera"]
    cre = [r["auc_rest_owner"] for r in rows if r["kind"] == "chimera"]
    print(f"a3 natural grad-AUC(own committee): med {np.median(nat):.3f} "
          f"(n={len(nat)})", flush=True)
    print(f"a3 chimera AUC vs W_E-owner comm: med {np.median(cwe):.3f}  "
          f"vs rest-owner comm: med {np.median(cre):.3f} (n={len(cwe)})",
          flush=True)
    return rows


def probe_d10():
    """Committee-pair gradient interference vs width at a mid-race state."""
    rows = []
    for cell in ("add113_m128_f0.3_wd3", "add113_m512_f0.3_wd3",
                 "add113_m2048_f0.3_wd3"):
        mm, mn = [], []
        for rd, comm, ge in runs_in(cell, min_tgen=1200):
            cfg, params, _ = load_run(rd, 1000)
            tok, lab, m = train_sel(cfg)
            g, W = grad_we(params, cfg, tok[m], lab[m])
            V = freq_vecs(g)
            V = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-30)
            mem = [k - 1 for k in comm]
            non = [i for i in np.argsort(-fstats(g, W)[1])[:12]
                   if i not in mem][:len(mem)]
            for x in range(len(mem)):
                for y in range(x + 1, len(mem)):
                    mm.append(abs(V[mem[x]] @ V[mem[y]]))
                for y in non:
                    mn.append(abs(V[mem[x]] @ V[y]))
        rows.append(dict(cell=cell, mem_mem=float(np.mean(mm)),
                         mem_non=float(np.mean(mn)),
                         n_mm=len(mm), n_mn=len(mn)))
        print(f"d10 {cell}: |cos| mem-mem {np.mean(mm):.4f} (n={len(mm)})  "
              f"mem-non {np.mean(mn):.4f} (n={len(mn)})", flush=True)
    return rows


def probe_e14():
    """Mid-race: gradient of UNSEEN examples vs train examples — how much of
    each grad's W_E energy sits on the run's committee?"""
    rows = []
    rng = np.random.default_rng(0)
    for rd, comm, ge in runs_in("add113_m512_f0.3_wd1", min_tgen=1500)[:4]:
        cfg, params, _ = load_run(rd, 1000)
        tok, lab, m = train_sel(cfg)
        new = rng.permutation(np.flatnonzero(~m))[:int(m.sum())]
        out = {}
        for tag, sel in (("train", np.flatnonzero(m)), ("new", new)):
            g, W = grad_we(params, cfg, tok[sel], lab[sel])
            gE = fstats(g, W)[1]
            out[tag] = dict(on=float(gE[[k - 1 for k in comm]].sum()
                                     / gE.sum()),
                            auc=auc(gE, comm))
        rows.append(dict(run=rd.name, comm=comm, **out))
        print(f"e14 {rd.name} comm{comm}: on-comm grad share "
              f"train {out['train']['on']:.3f} (auc {out['train']['auc']:.2f})"
              f"  new {out['new']['on']:.3f} (auc {out['new']['auc']:.2f})",
              flush=True)
    return rows


def probe_e13():
    """S5: per-irrep grad-energy profile of W_E, wd1 vs wd2 cells."""
    from groups import PROJ
    Pr = PROJ[1:]
    rows = []
    for cell in sorted(p.name for p in DG2.glob("s5120_m512_f0.7_wd*")):
        profs = {}
        for rd in sorted((DG2 / cell).iterdir()):
            if not (rd / "spectra.npz").exists():
                continue
            cfg, params, _ = load_run(rd, 1000)
            tok, lab, m = train_sel(cfg)
            g, W = grad_we(params, cfg, tok[m], lab[m])
            gE = np.array([((g[:, :120] @ Pr[r].T) ** 2).sum() for r in range(6)])
            wE = np.array([((W[:, :120] @ Pr[r].T) ** 2).sum() for r in range(6)])
            profs[rd.name] = dict(gE=(gE / gE.sum()).round(4).tolist(),
                                  wE=(wE / wE.sum()).round(4).tolist())
        rows.append(dict(cell=cell, profiles=profs))
        gm = np.mean([p["gE"] for p in profs.values()], 0)
        print(f"e13 {cell}: mean grad irrep profile {gm.round(3).tolist()}",
              flush=True)
    return rows


def probe_b0():
    """Free by-product: epoch-0 grad energy vs epoch-0 amplitude as committee
    predictors, per run."""
    rows = []
    for cell in ("add113_m512_f0.3_wd1", "add113_m512_f0.3_wd3",
                 "add113_m512_f0.9_wd1", "add113_m512_f0.9_wd3"):
        for rd, comm, ge in runs_in(cell):
            cfg, params, _ = load_run(rd, 0)
            tok, lab, m = train_sel(cfg)
            g, W = grad_we(params, cfg, tok[m], lab[m])
            E, gE, _ = fstats(g, W)
            rows.append(dict(cell=cell, run=rd.name,
                             auc_grad=auc(gE, comm), auc_amp=auc(E, comm)))
    ag = [r["auc_grad"] for r in rows]
    aa = [r["auc_amp"] for r in rows]
    print(f"b0 epoch-0 committee AUC: grad med {np.median(ag):.3f}  "
          f"amplitude med {np.median(aa):.3f}  (n={len(ag)})", flush=True)
    return rows


PROBES = dict(a1=probe_a1, a2=probe_a2, a3=probe_a3, d10=probe_d10,
              e14=probe_e14, e13=probe_e13, b0=probe_b0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", default="a1,a2,a3,d10,e14,e13,b0")
    ap.add_argument("--out", default=str(ROOT / "notes" / "gradprobe1.json"))
    args = ap.parse_args()
    out = {}
    for name in args.probe.split(","):
        print(f"=== probe {name} ===", flush=True)
        out[name] = PROBES[name]()
    Path(args.out).write_text(json.dumps(out, indent=1, default=float))
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
