"""Deletion vs outcompetition on EXISTING checkpointed runs (numpy, no torch).
For every run with spectra + >=3 checkpoints inside [t_mem, t_gen]:
  - full 2D logit spectrum over (a,b) per output (softmax gauge removed):
    committee diagonal blocks, other diagonal blocks, off-diagonal
    (memorizer), single-input terms
  - Fourier energy of W_E / W_U over the token axis, committee vs other
  - per-matrix squared norms
  - Nanda excluded / restricted TRAIN loss with the FINAL committee
Writes notes/wspace_local.json.
"""
import glob, json, os, random, sys
from pathlib import Path
import numpy as np
from safetensors.numpy import load_file

ROOT = Path(__file__).resolve().parents[1]
ONLY = sys.argv[1] if len(sys.argv) > 1 else None


def fs(m):
    i = np.where(m[:-1] & m[1:])[0]
    return int(i[0]) if len(i) else -1


def committee_final(E):
    e = np.log10(np.clip(E[-1], 1e-12, None))
    o = np.argsort(e)[::-1]
    g = e[o[:-1]] - e[o[1:]]
    return o[:int(np.argmax(g[:min(10, len(g))])) + 1].tolist()


def fourier_basis(p):
    b = [np.ones(p) / np.sqrt(p)]
    for k in range(1, p // 2 + 1):
        for trig in (np.cos, np.sin):
            v = trig(2 * np.pi * k * np.arange(p) / p)
            b.append(v / np.linalg.norm(v))
    return np.stack(b)


def split(p, frac, seed):
    pairs = [(i, j) for i in range(p) for j in range(p)]
    random.seed(seed); random.shuffle(pairs)
    tr = set(pairs[:int(frac * len(pairs))])
    return np.array([(i, j) in tr for i in range(p) for j in range(p)])


def logits(W, cfg):
    p, d, h = cfg["p"], cfg["d_model"], cfg["num_heads"]
    dh = d // h
    a = np.repeat(np.arange(p), p); b = np.tile(np.arange(p), p)
    tok = np.stack([a, b, np.full(p * p, p)], 1)
    E = W["embed.W_E"].T                                  # (v, d)
    x = E[tok] + W["pos_embed.W_pos"][None, :3]           # (N, 3, d)
    N = x.shape[0]
    def proj(y, M): return y @ M.reshape(h * dh, d).T
    k = proj(x.reshape(N * 3, d), W["blocks.0.attn.W_K"]).reshape(N, 3, h, dh)
    v = proj(x.reshape(N * 3, d), W["blocks.0.attn.W_V"]).reshape(N, 3, h, dh)
    q = proj(x[:, -1], W["blocks.0.attn.W_Q"]).reshape(N, 1, h, dh)
    s = (k * q).sum(-1) / np.sqrt(dh)                    # (N, 3, h)
    s = np.exp(s - s.max(1, keepdims=True)); att = s / s.sum(1, keepdims=True)
    z = (att[..., None] * v).sum(1).reshape(N, h * dh)
    xf = x[:, -1] + z @ W["blocks.0.attn.W_O"].T
    pre = xf @ W["blocks.0.mlp.W_in"].T + W["blocks.0.mlp.b_in"]
    act = np.maximum(pre, 0) if cfg["act_type"] == "ReLU" else \
        0.5 * pre * (1 + np.tanh(0.79788456 * (pre + 0.044715 * pre ** 3)))
    xf = xf + act @ W["blocks.0.mlp.W_out"].T + W["blocks.0.mlp.b_out"]
    return xf @ W["unembed.W_U"][:, :p]                  # (N, p)


def ce(L, lab):
    L = L.astype(np.float64)
    m = L.max(1, keepdims=True)
    return float(np.mean(np.log(np.exp(L - m).sum(1)) + m[:, 0] - L[np.arange(len(lab)), lab]))


def analyze(rd, F, kidx):
    cfg = json.load(open(rd / "config.json"))
    if cfg["fn_name"] not in ("add", "sub") or cfg.get("num_layers", 1) != 1:
        return None
    p = cfg["p"]
    z = np.load(rd / "spectra.npz"); ep, E = z["epochs"], z["energy"]
    im, ig = fs(z["train_acc"] >= 0.99), fs(z["test_acc"] >= 0.9)
    if im < 0 or ig < 0 or ig <= im:
        return None
    t_mem, t_gen = int(ep[im]), int(ep[ig])
    cks = sorted((int(Path(f).stem.split("_")[1]), f) for f in
                 glob.glob(str(rd / "checkpoints" / "epoch_*.safetensors")))
    inwin = [(e, f) for e, f in cks if t_mem <= e <= t_gen]
    if len(inwin) < 3:
        return None
    # one checkpoint before and after the window for context
    before = [(e, f) for e, f in cks if e < t_mem][-1:]
    after = [(e, f) for e, f in cks if e > t_gen][:1]
    use = before + inwin + after
    comm = committee_final(E)
    inC = np.zeros(p, bool)
    for k in comm: inC |= kidx == k + 1
    excl = ~(inC[:, None] | inC[None, :])
    restr = inC[:, None] & inC[None, :] & (kidx[:, None] == kidx[None, :]); restr[0, 0] = True
    same = (kidx[:, None] == kidx[None, :]) & (kidx[:, None] > 0)
    a = np.repeat(np.arange(p), p); b = np.tile(np.arange(p), p)
    lab = ((a + b) if cfg["fn_name"] == "add" else (a - b)) % p
    tr = split(p, cfg["frac_train"], cfg["data_seed"])
    nf = p // 2 + 1
    rows = []
    for e, f in use:
        W = load_file(f)
        L = logits(W, cfg).astype(np.float64)
        Lc = (L - L.mean(1, keepdims=True)).reshape(p, p, p)
        H = np.einsum("ia,jb,abo->ijo", F, F, Lc, optimize=True)
        E2 = (H ** 2).sum(-1)
        diag = np.zeros(nf); np.add.at(diag, kidx, (E2 * same).sum(1))
        const = E2[0, 0]; single = E2[0, 1:].sum() + E2[1:, 0].sum()
        total = E2.sum()
        dk = diag[1:]
        def wE(key):
            C = W[key][:, :p] @ F.T; e_ = np.zeros(nf); np.add.at(e_, kidx, (C ** 2).sum(0)); return e_[1:]
        we, wu = wE("embed.W_E"), wE("unembed.W_U")
        Lx = lambda m: np.einsum("ia,jb,ijo->abo", F, F, H * m[:, :, None], optimize=True).reshape(p * p, p)[tr]
        rows.append(dict(
            epoch=e, in_wait=t_mem <= e <= t_gen,
            diag_comm=float(dk[comm].sum()), diag_junk=float(np.delete(dk, comm).sum()),
            off2d=float(total - const - single - dk.sum()), single2d=float(single),
            total2d=float(total),
            wE_comm=float(we[comm].sum()), wE_junk=float(np.delete(we, comm).sum()),
            wU_comm=float(wu[comm].sum()), wU_junk=float(np.delete(wu, comm).sum()),
            pnorm=float(sum((v ** 2).sum() for k, v in W.items() if not k.startswith("adam"))),
            pnorm_WE=float((W["embed.W_E"] ** 2).sum()), pnorm_WU=float((W["unembed.W_U"] ** 2).sum()),
            loss_full=ce(L[tr], lab[tr]), loss_excl=ce(Lx(excl), lab[tr]),
            loss_restr=ce(Lx(restr), lab[tr]),
            train_acc=float((L.argmax(1) == lab)[tr].mean()),
            test_acc=float((L.argmax(1) == lab)[~tr].mean())))
    return dict(run=str(rd), cohort=str(rd).split("/")[1], task=cfg["fn_name"] + str(p),
                frac=cfg["frac_train"], wd=cfg["weight_decay"], lr=cfg["lr"],
                width=cfg["d_mlp"], committee=comm, t_mem=t_mem, t_gen=t_gen, rows=rows)


def main():
    out, cache = [], {}
    dirs = sorted({os.path.dirname(c) for c in glob.glob("runs_*/**/checkpoints", recursive=True)
                   if os.path.exists(os.path.join(os.path.dirname(c), "spectra.npz"))})
    if ONLY: dirs = [d for d in dirs if ONLY in d]
    for d in dirs:
        rd = Path(d)
        try:
            cfg = json.load(open(rd / "config.json"))
        except Exception:
            continue
        if "p" not in cfg: continue
        p = cfg["p"]
        if p not in cache:
            cache[p] = (fourier_basis(p), np.array([(i + 1) // 2 for i in range(p)]))
        try:
            r = analyze(rd, *cache[p])
        except Exception as ex:
            print("FAIL", d, ex, flush=True); continue
        if r is None: continue
        out.append(r)
        w = [x for x in r["rows"] if x["in_wait"]]
        rat = lambda k: w[-1][k] / max(w[0][k], 1e-12)
        print(f"{d} wd={r['wd']} wait {r['t_mem']}-{r['t_gen']} n={len(w)} "
              f"off2d x{rat('off2d'):.2f} wE_junk x{rat('wE_junk'):.2f} wU_junk x{rat('wU_junk'):.2f} "
              f"diag_junk x{rat('diag_junk'):.2f} excl {w[0]['loss_excl']:.2f}->{w[-1]['loss_excl']:.2f}", flush=True)
        json.dump(out, open(ROOT / "notes" / "wspace_local.json", "w"))
    print("done", len(out))


if __name__ == "__main__":
    main()
