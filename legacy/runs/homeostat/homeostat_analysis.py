"""Homeostat test harness — self-contained, for adversarial review.

Reproduces every number in runs/HOMEOSTAT_BRIEF.md from the saved runs. All
the load-bearing math (committee detection, the margin functional, the
amplitude read-off, the symmetric/phase decomposition) is inlined below with
comments so a reviewer can audit the exact computation rather than trust an
import. Only the model definition (grok.model / grok.data / grok.config) is
imported — that is the trusted artifact that produced the checkpoints.

Run:  uv run python runs/homeostat/homeostat_analysis.py   (from the repo root)
Needs: only the project env (numpy + mlx); no scipy. Consumes the runs in
runs/og_seed0/seed*/ and runs/p-*/seed*/seed*/ (config.json, checkpoints,
spectra.npz).

Data: 15 grokked models (test acc = 1.0). 7 at p=113 (runs/og_seed0, the
exploratory zoo, 20-30k epochs) + 8 from the matrix farm (runs/p-{113,127,157}
/seed{2034,3604}/seed*, 30k epochs). Every run: 1-layer transformer, d_model
128, 4 heads, d_mlp 512, ReLU, no LayerNorm; full-batch AdamW lr 1e-3,
weight_decay 1.0, betas (0.9,0.98), warmup 10, float64 CE loss; frac_train 0.3.
"""

import glob
from pathlib import Path

import numpy as np

from grok.config import Config
from grok.model import Transformer
from grok.data import make_dataset, train_test_split


# --------------------------------------------------------------------------- #
# run discovery
# --------------------------------------------------------------------------- #

def grokked_runs():
    """(p, run_dir) for every run with a final checkpoint and test acc > 0.9."""
    out = []
    for f in sorted(glob.glob("runs/og_seed0/seed*/spectra.npz")):
        z = np.load(f)
        if (z["test_acc"] > 0.9).any():
            out.append((113, Path(f).parent))
    for pdir in sorted(glob.glob("runs/p-*")):
        p = int(Path(pdir).name.split("-")[1])
        for f in sorted(glob.glob(f"{pdir}/seed*/seed*/spectra.npz")):
            if (np.load(f)["test_acc"] > 0.9).any():
                out.append((p, Path(f).parent))
    return out


# --------------------------------------------------------------------------- #
# inlined primitives (auditable)
# --------------------------------------------------------------------------- #

def final_logits(run_dir, p):
    """(p^2, p) float64 logits at the final position, '=' logit dropped.

    Loads the model's last checkpoint and runs it on all p^2 inputs.
    """
    import mlx.core as mx
    cfg = Config.load(run_dir / "config.json")
    ckpt = sorted((run_dir / "checkpoints").glob("epoch_*.safetensors"))[-1]
    model = Transformer(cfg)
    model.load_weights(str(ckpt))
    tokens, labels = make_dataset(cfg)
    logits = model(tokens)[:, -1, :-1]
    mx.eval(logits)
    is_train, is_test = train_test_split(cfg)
    weight_norm = _sum_sq_weights(model)
    return np.array(logits, dtype=np.float64), np.array(labels), is_train, is_test, weight_norm

def _sum_sq_weights(model):
    from mlx.utils import tree_flatten
    return sum(float((v ** 2).sum()) for _, v in tree_flatten(model.parameters()))

def committee(coeffs_final):
    """Frequencies (1-indexed) above the largest log-gap in sorted |coeff|.

    coeffs_final = the last row of spectra.npz['coeffs']: the cos(w(a+b-c))
    phase-locked coefficient for each frequency k=1..p//2.
    """
    a = np.abs(coeffs_final)
    order = np.argsort(a)[::-1]
    logs = np.log(a[order] + 1e-12)
    gaps = logs[:-1] - logs[1:]
    cut = int(np.argmax(gaps[:12])) + 1
    return sorted((order[:cut] + 1).tolist())

def relM_equal(freqs, p):
    """Equal-amplitude relative min-margin: 1 - max_{x!=0} mean_k cos(2*pi*k*x/p)."""
    x = np.arange(1, p)[:, None]
    k = np.asarray(freqs)[None, :]
    return 1.0 - np.cos(2 * np.pi * k * x / p).mean(axis=1).max()

def amplitudes_and_gap(logits, p):
    """Read cosine amplitudes off the translation-averaged logits.

    The logit profile is averaged over the p 'miss' diagonals x = a+b-c mod p,
    giving L(x). Its cosine amplitude at frequency k is
        a_k = (2/p) * sum_x L(x) cos(2*pi*k*x/p).
    The correct answer (x=0) has logit A_tot = sum_k a_k; a wrong answer at
    miss x has A_tot - gap(x) with gap(x) = sum_k a_k (1 - cos(2*pi*k*x/p)).
    Returns (a_k array, A_tot over positive amps, minGap, L(x) profile).
    """
    a, b = np.divmod(np.arange(p * p), p)
    x = (a[:, None] + b[:, None] - np.arange(p)[None, :]) % p      # (p^2, p)
    Lx = np.zeros(p)
    np.add.at(Lx, x.ravel(), logits.ravel())
    Lx /= p * p
    ks = np.arange(1, p // 2 + 1)
    amp = (2.0 / p) * (Lx[None, :] * np.cos(2 * np.pi * ks[:, None]
                       * np.arange(p)[None, :] / p)).sum(axis=1)
    gap = (amp[:, None] * (1 - np.cos(2 * np.pi * ks[:, None]
                           * np.arange(1, p)[None, :] / p))).sum(axis=0)
    A_tot = amp[amp > 0].sum()
    return amp, A_tot, gap.min(), Lx

def _log_softmax(z):
    z = z - z.max(axis=-1, keepdims=True)
    return z - np.log(np.exp(z).sum(axis=-1, keepdims=True))

def ce(logits, labels, mask):
    lp = _log_softmax(logits[mask])
    return -lp[np.arange(mask.sum()), labels[mask]].mean()


# --------------------------------------------------------------------------- #
# harness
# --------------------------------------------------------------------------- #

def main():
    runs = grokked_runs()
    R = []
    for p, d in runs:
        logits, labels, is_tr, is_te, norm = final_logits(d, p)
        amp, A_tot, minGap, Lx = amplitudes_and_gap(logits, p)
        relM = minGap / max(A_tot, 1e-9)
        comm = committee(np.load(d / "spectra.npz")["coeffs"][-1])
        # symmetric (translation-averaged) logits, and the phase-noise factor
        a, b = np.divmod(np.arange(p * p), p)
        xmat = (a[:, None] + b[:, None] - np.arange(p)[None, :]) % p
        Lsym = Lx[xmat]
        R.append(dict(
            p=p, name=d.name, K=len(comm), A=A_tot, relM=relM, minGap=minGap,
            norm=norm,
            ce_act_te=ce(logits, labels, is_te), ce_sym_te=ce(Lsym, labels, is_te),
            ce_act_tr=ce(logits, labels, is_tr), ce_sym_tr=ce(Lsym, labels, is_tr),
        ))

    p_=np.array([r["p"] for r in R]); K=np.array([r["K"] for r in R])
    A=np.array([r["A"] for r in R]); relM=np.array([r["relM"] for r in R])
    mg=np.array([r["minGap"] for r in R]); norm=np.array([r["norm"] for r in R])
    ph_te=np.array([r["ce_act_te"]/r["ce_sym_te"] for r in R])
    ce_sym=np.array([r["ce_sym_te"] for r in R])

    print(f"{len(R)} grokked runs\n")
    h=f'{"run":14} {"p":4} {"K":2} {"A_tot":7} {"relM":6} {"minGap":7} {"norm":6} {"CEsym_te":10} {"phase_te":8}'
    print(h); print("-"*len(h))
    for r in R:
        print(f'{r["name"]:14} {r["p"]:<4} {r["K"]:<2} {r["A"]:6.1f} {r["relM"]:6.3f} '
              f'{r["minGap"]:7.2f} {r["norm"]:6.0f} {r["ce_sym_te"]:.2e} '
              f'{r["ce_act_te"]/r["ce_sym_te"]:7.1f}')

    cv=lambda v: v.std()/v.mean()*100
    print(f"\n[H1] HOMEOSTAT  minGap = A_tot*relM")
    print(f"     mean {mg.mean():.2f}  sd {mg.std():.2f}  CV {cv(mg):.1f}%  range [{mg.min():.2f},{mg.max():.2f}]")
    print(f"     factor spreads: A_tot CV {cv(A):.0f}%, relM CV {cv(relM):.0f}%; corr(A,relM) {np.corrcoef(A,relM)[0,1]:+.2f}")
    neglog = -np.log(ce_sym)
    print(f"     TRIVIALITY CONTROL — is minGap more than 'all models reach the same loss'?")
    print(f"       sd(minGap) {mg.std():.2f} nats vs sd(-log CEsym_te) {neglog.std():.2f} nats;"
          f" corr {np.corrcoef(mg,neglog)[0,1]:+.2f}")
    print(f"       (if these match, H1 largely restates constant symmetric loss; the"
          f" non-trivial content is the 27% factor compensation + H2/H4/H5.)")

    print(f"\n[H2] p-INDEPENDENCE (optimizer constant, not task)")
    print(f"     corr(minGap, p) = {np.corrcoef(mg,p_)[0,1]:+.2f}")
    for pv in sorted(set(p_)):
        v=mg[p_==pv]; print(f"       p={pv}: minGap mean {v.mean():.2f} (n={len(v)})")

    print(f"\n[H3] K-SLOPE and PHASE NOISE")
    sl,ic=np.polyfit(K,mg,1)
    print(f"     minGap = {ic:.1f} + {sl:.2f}*K   corr(minGap,K)={np.corrcoef(mg,K)[0,1]:+.2f}")
    for k in sorted(set(K)):
        print(f"       K={k}: minGap {mg[K==k].mean():.2f}, phase_te median {np.median(ph_te[K==k]):.1f}")
    print(f"     corr(log phase_te, K) = {np.corrcoef(np.log(ph_te),K)[0,1]:+.2f}")
    print(f"     phase is a TEST effect: median phase_train {np.median([r['ce_act_tr']/max(r['ce_sym_tr'],1e-30) for r in R]):.1f} vs phase_test {np.median(ph_te):.1f}")

    print(f"\n[H4] LOSS LAW  actual CE = phase * symmetric CE")
    clean=ph_te[K<=4]
    print(f"     phase_te median {np.median(ph_te):.1f}; clean (K<=4) runs {clean.min():.1f}-{clean.max():.1f} (claim: ~3-5x)")

    print(f"\n[H5] AMPLITUDE CHEAP, NORM PRICES K")
    print(f"     slope d log(norm)/d log(A_tot) = {np.polyfit(np.log(A),np.log(norm),1)[0]:+.2f} (claim ~0)")
    m113=(p_==113)
    for k in sorted(set(K[m113])):
        print(f"       p113 K={k}: norm {norm[m113&(K==k)].mean():.0f}")

    print(f"\n[H6] UNTESTED PREDICTION: minGap ~ C - log(lr*wd).")
    print(f"     Needs a weight-decay sweep. Prediction: -log(2)={-np.log(2):.2f} nats per wd doubling.")


if __name__ == "__main__":
    main()
