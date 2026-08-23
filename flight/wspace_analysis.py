"""Deletion vs outcompetition, measured in Nanda's coordinates.
Run ON the pod (needs torch + checkpoints). Per run: wait window
[t_mem, t_gen]; trends of diagonal junk, off-diagonal (memorizer) logit
energy, off-committee W_E / W_U Fourier energy, parameter norms; Nanda
excluded / restricted train loss from the dense checkpoints with the
FINAL committee. Writes notes/wspace_summary.json (+ per-run series).
"""
import json, sys, glob
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "cloud"))
from train_semifinal_torch import (Config, fourier_basis, make_dataset,
                                   train_test_split, batched_final_logits,
                                   PARAM_KEYS)
from safetensors.torch import load_file

RUNS = ROOT / "runs_flight" / (sys.argv[1] if len(sys.argv) > 1 else "wspace")
dev = "cuda" if torch.cuda.is_available() else "cpu"


def fs(m):
    i = np.where(m[:-1] & m[1:])[0]
    return int(i[0]) if len(i) else -1


def committee_final(E):
    e = np.log10(np.clip(E[-1], 1e-12, None))
    o = np.argsort(e)[::-1]
    g = e[o[:-1]] - e[o[1:]]
    return o[:int(np.argmax(g[:min(10, len(g))])) + 1].tolist()


def slope(ep, y):
    """log-slope per kep over the window (nan if <3 pts)."""
    y = np.clip(y, 1e-12, None)
    return float(np.polyfit(ep / 1e3, np.log(y), 1)[0]) if len(ep) >= 3 else np.nan


@torch.no_grad()
def nanda_losses(rd, cfg, comm, ckpt_epochs):
    """excluded / restricted train CE at each checkpoint, final committee."""
    p = cfg.p
    F = torch.from_numpy(fourier_basis(p)).float().to(dev)
    kidx = torch.tensor([(i + 1) // 2 for i in range(p)], device=dev)
    inC = torch.zeros(p, dtype=torch.bool, device=dev)
    for k in comm:
        inC |= kidx == (k + 1)                      # energy index k -> freq k+1
    excl_mask = ~(inC[:, None] | inC[None, :])      # drop any coeff touching C
    restr_mask = (inC[:, None] & inC[None, :] & (kidx[:, None] == kidx[None, :]))
    restr_mask[0, 0] = True
    tok, lab = make_dataset(cfg)
    tr = train_test_split(cfg)
    tok_t = torch.from_numpy(tok).to(dev)
    lab_t = torch.from_numpy(lab[tr]).to(dev)
    tr_t = torch.from_numpy(tr).to(dev)
    out = {"epochs": [], "full": [], "excluded": [], "restricted": []}
    for e in ckpt_epochs:
        f = rd / "checkpoints" / f"epoch_{e:05d}.safetensors"
        if not f.exists():
            continue
        ck = load_file(str(f))
        pr = {k: ck[k][None].to(dev) for k in PARAM_KEYS}
        L = batched_final_logits(pr, tok_t, cfg)[0, :, :p].float()   # (p^2, p)
        Lc = (L - L.mean(-1, keepdim=True)).reshape(p, p, p)
        H = torch.einsum("ia,jb,abo->ijo", F, F, Lc)

        def ce(Hm):
            Lx = torch.einsum("ia,jb,ijo->abo", F, F, Hm).reshape(p * p, p)[tr_t]
            return float(torch.nn.functional.cross_entropy(Lx.double(), lab_t))
        out["epochs"].append(int(e))
        out["full"].append(ce(H))
        out["excluded"].append(ce(H * excl_mask[:, :, None]))
        out["restricted"].append(ce(H * restr_mask[:, :, None]))
    return out


def analyze(rd):
    z = np.load(rd / "spectra.npz")
    cfg = Config.load(rd / "config.json") if hasattr(Config, "load") else None
    if cfg is None:
        cfg = Config(**{k: v for k, v in json.load(open(rd / "config.json")).items()
                        if k in Config.__dataclass_fields__})
    ep, E = z["epochs"], z["energy"]
    comm = committee_final(E)
    im, ig = fs(z["train_acc"] >= 0.99), fs(z["test_acc"] >= 0.9)
    res = dict(run=str(rd.relative_to(RUNS)), frac=cfg.frac_train, wd=cfg.weight_decay,
               committee=comm, t_mem=int(ep[im]) if im >= 0 else None,
               t_gen=int(ep[ig]) if ig >= 0 else None)
    if im < 0 or ig < 0 or ig <= im + 2:
        res["status"] = "no_mem" if im < 0 else ("censored" if ig < 0 else "no_wait")
        return res
    res["status"] = "ok"
    w = slice(im, ig + 1)
    junk = [j for j in range(E.shape[1]) if j not in comm]
    d2 = z["diag2d"]
    series = {
        "diag_junk": E[:, junk].sum(1), "diag_comm": E[:, comm].sum(1),
        "diag2d_junk": d2[:, junk].sum(1), "diag2d_comm": d2[:, comm].sum(1),
        "off2d": z["off2d"], "single2d": z["single2d"], "total2d": z["total2d"],
        "wE_junk": z["wE_energy"][:, 1:][:, junk].sum(1),
        "wE_comm": z["wE_energy"][:, 1:][:, comm].sum(1),
        "wU_junk": z["wU_energy"][:, 1:][:, junk].sum(1),
        "wU_comm": z["wU_energy"][:, 1:][:, comm].sum(1),
        "pnorm_total": z["pnorm"].sum(1),
        "pnorm_WE": z["pnorm"][:, 0], "pnorm_WU": z["pnorm"][:, -1],
    }
    res["trend"] = {k: dict(slope=slope(ep[w].astype(float), v[w]),
                            ratio=float(v[ig] / max(v[im], 1e-12)),
                            at_mem=float(v[im]), at_gen=float(v[ig]))
                    for k, v in series.items()}
    # memorizer share of 2D logit energy at t_mem / t_gen
    res["mem_share"] = {"at_mem": float(z["off2d"][im] / z["total2d"][im]),
                        "at_gen": float(z["off2d"][ig] / z["total2d"][ig])}
    ck = sorted(int(Path(f).stem.split("_")[1])
                for f in glob.glob(str(rd / "checkpoints" / "epoch_*.safetensors")))
    ck = [e for e in ck if ep[im] - 250 <= e <= ep[ig] + 500]
    nl = nanda_losses(rd, cfg, comm, ck)
    res["nanda"] = nl
    ex = np.array(nl["excluded"]); eps_ = np.array(nl["epochs"])
    if len(ex) >= 2:
        base = ex[np.argmin(np.abs(eps_ - ep[im]))]
        hit = np.where((ex >= 2 * base) & (eps_ >= ep[im]))[0]
        t2x = int(eps_[hit[0]]) if len(hit) else None
        res["excl_double_frac"] = ((t2x - ep[im]) / (ep[ig] - ep[im])
                                   if t2x is not None else None)
    # downsampled series for figures
    keep = np.arange(0, len(ep), max(1, len(ep) // 400))
    res["series"] = {"epochs": ep[keep].tolist(),
                     **{k: v[keep].astype(float).tolist() for k, v in series.items()}}
    return res


def main():
    out = []
    for rd in sorted(RUNS.glob("*/*")):
        if not (rd / "spectra.npz").exists():
            continue
        r = analyze(rd)
        out.append(r)
        print(r["run"], r["status"], r.get("t_mem"), r.get("t_gen"),
              {k: round(r["trend"][k]["ratio"], 3) for k in
               ("diag_junk", "off2d", "wE_junk", "wU_junk")} if r["status"] == "ok" else "",
              "excl2x@", r.get("excl_double_frac"), flush=True)
    (ROOT / "notes").mkdir(exist_ok=True)
    json.dump(out, open(ROOT / "notes" / "wspace_summary.json", "w"))
    # per-wd table
    print("\nwd   n  off2d_up  wE_junk_up  wU_junk_up  diag_junk_up  excl2x<0.5")
    for wd in sorted({r["wd"] for r in out}):
        rs = [r for r in out if r["wd"] == wd and r["status"] == "ok"]
        if not rs:
            print(f"{wd:<4} 0"); continue
        up = lambda k: np.mean([r["trend"][k]["ratio"] > 1 for r in rs])
        ex = [r["excl_double_frac"] for r in rs if r.get("excl_double_frac") is not None]
        print(f"{wd:<4} {len(rs):2d}  {up('off2d'):.2f}      {up('wE_junk'):.2f}        "
              f"{up('wU_junk'):.2f}        {up('diag_junk'):.2f}          "
              f"{np.mean(np.array(ex) < 0.5) if ex else float('nan'):.2f}")


if __name__ == "__main__":
    main()
