"""Epoch-0 per-(run, frequency) feature bank for the PROBE CEILING study.

Reads ONLY the init checkpoint (epoch_00000.safetensors) of every kept v2 run
discovered by semifinal/analysis/common.discover, and emits a feature matrix
X (n_run*nf, F) with labels y = "frequency k is in the run's final committee"
(unified largest-log-gap + floor detector, same exclusions as every other
analysis in the repo).

The point of the feature bank is to cover every path from the init weights to
frequency k that the closed-form ticket T_k = sum_h ||W_O^h W_V^h W_E|_k||^2
throws away:

  E   embedding      per-freq energy / anisotropy of W_E|_k
  U   unembedding    per-freq energy / anisotropy of W_U|_k
  V   OV path        T_k itself, per-head spread, OV gain (T_k / emb)
  Q   QK path        query/key energy of W_E|_k, the freq-k self-attention
                     bilinear term, and the '=' -> token-a attention logit
  M   MLP path       W_in @ OV @ W_E|_k, the direct (skip) read W_in @ W_E|_k,
                     the full linearized loop W_U^T W_out W_in OV W_E|_k
                     projected back onto output frequency k, and a per-neuron
                     read/write matched filter
  P   phase/geometry principal angles between W_E|_k and W_U|_k, and the
                     fraction of W_E|_k / W_U|_k lying in the OV circuit's top
                     transmitted subspaces (the compiler's `rotate` direction)
  X   cross terms    log-sums the linear model could form anyway, kept so the
                     ablation can attribute them
  K   frequency id   k / nf, as a control for any bare frequency prior

Each raw feature is emitted in three transforms, because the lottery is
relative, not absolute:
  <name>            global z-score of log(f)      (absolute scale)
  <name>__z         within-run z-score of log(f)  (relative scale)
  <name>__r         within-run percentile rank    (rank-only, monotone-invariant)

Analysis-only: never trains, never writes into runs_torch / runs_compiler.
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/Users/ruhaanrajadhyaksha/projects/grok")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "semifinal" / "analysis"))

from common import discover, fourier  # noqa: E402
from compiler.core import load_ckpt   # noqa: E402

EPS = 1e-30


def _sub(F, k):
    """Columns (2k-1, 2k) of a Fourier-coordinate matrix -> the freq-k plane."""
    return F[:, [2 * k - 1, 2 * k]]


def _cos_angles(A, B):
    """Cosines of the principal angles between the column spans of A and B."""
    qa, _ = np.linalg.qr(A)
    qb, _ = np.linalg.qr(B)
    s = np.linalg.svd(qa.T @ qb, compute_uv=False)
    return np.clip(s, 0.0, 1.0)


def _aniso(A):
    """sv1^2 / sv2^2 of a (d, 2) block: how elongated the 2D plane is."""
    s = np.linalg.svd(A, compute_uv=False)
    return (s[0] ** 2 + EPS) / (s[-1] ** 2 + EPS)


def _pr(v):
    """Participation ratio of a nonneg vector: (sum v)^2 / sum v^2."""
    v = np.asarray(v, float)
    return (v.sum() ** 2 + EPS) / ((v ** 2).sum() + EPS)


def run_features(params, p):
    """-> (names, raw) with raw of shape (nf, n_raw_features)."""
    nf = p // 2
    Bas = fourier(p).basis                                   # (p, p)

    W_E = params["embed.W_E"]                                # (d, vocab)
    W_U = params["unembed.W_U"]
    W_pos = params["pos_embed.W_pos"]                        # (n_ctx, d)
    W_V = params["blocks.0.attn.W_V"]                        # (h, dh, d)
    W_K = params["blocks.0.attn.W_K"]
    W_Q = params["blocks.0.attn.W_Q"]
    W_O = params["blocks.0.attn.W_O"]                        # (d, h*dh)
    W_in = params["blocks.0.mlp.W_in"]                       # (d_mlp, d)
    W_out = params["blocks.0.mlp.W_out"]                     # (d, d_mlp)
    h, dh, d = W_V.shape

    FE = W_E[:, :p] @ Bas.T                                  # (d, p)
    FU = W_U[:, :p] @ Bas.T                                  # (d, p)
    OV = [W_O[:, i * dh:(i + 1) * dh] @ W_V[i] for i in range(h)]

    # query vector at the final ('=') position, the only position that reads
    q_vec = W_E[:, p] + W_pos[-1]

    OVF = [O @ FE for O in OV]                               # (d, p) each
    MOV = [W_in @ X for X in OVF]                            # (d_mlp, p)
    WUt = W_U[:, :p].T                                       # (p, d)
    # full linearized loop: input fourier coord -> output token -> output freq
    LOOP = [Bas @ (WUt @ (W_out @ X)) for X in MOV]          # (p, p) each
    MDIR = W_in @ FE                                         # (d_mlp, p)
    QB = [W_Q[i] @ FE for i in range(h)]                     # (dh, p)
    KB = [W_K[i] @ FE for i in range(h)]
    qq = [W_Q[i] @ q_vec for i in range(h)]                  # (dh,)
    # per-neuron output-side write strength for each frequency (claim-1 out_nk)
    FO = (W_out.T @ W_U[:, :p]) @ Bas.T                      # (d_mlp, p)

    # OV circuit's preferred input / output directions
    M2i = sum(O.T @ O for O in OV)
    M2o = sum(O @ O.T for O in OV)
    ei, Vi = np.linalg.eigh(M2i)
    eo, Vo = np.linalg.eigh(M2o)
    Vi = Vi[:, np.argsort(ei)[::-1]]
    Vo = Vo[:, np.argsort(eo)[::-1]]

    names, rows = None, []
    for k in range(1, nf + 1):
        Bk = _sub(FE, k)                                     # (d, 2)
        Uk = _sub(FU, k)
        e_emb = float((Bk ** 2).sum())
        e_unb = float((Uk ** 2).sum())

        tk_h = np.array([float((_sub(X, k) ** 2).sum()) for X in OVF])
        tk = float(tk_h.sum())
        ovk = np.concatenate([_sub(X, k) for X in OVF], axis=1)   # (d, 2h)
        sv_ov = np.linalg.svd(ovk, compute_uv=False)

        q_e = float(sum((_sub(X, k) ** 2).sum() for X in QB))
        k_e = float(sum((_sub(X, k) ** 2).sum() for X in KB))
        qk_self = float(sum(np.abs(_sub(QB[i], k).T @ _sub(KB[i], k)).sum()
                            for i in range(h)))
        qk_pos = float(sum(np.abs(qq[i] @ _sub(KB[i], k)).sum()
                           for i in range(h)))

        mlp_ov = float(sum((_sub(X, k) ** 2).sum() for X in MOV))
        mlp_dir = float((_sub(MDIR, k) ** 2).sum())
        loop_kk = float(sum((L[[2 * k - 1, 2 * k]][:, [2 * k - 1, 2 * k]] ** 2).sum()
                            for L in LOOP))
        loop_row = float(sum((L[[2 * k - 1, 2 * k]] ** 2).sum() for L in LOOP))

        read_n = np.sqrt(sum(_sub(X, k) ** 2 for X in MOV).sum(1))    # (d_mlp,)
        write_n = np.sqrt((_sub(FO, k) ** 2).sum(1))
        match = float((read_n * write_n).sum())
        pr_read = float(_pr(read_n ** 2))
        pr_write = float(_pr(write_n ** 2))

        c_eu = _cos_angles(Bk, Uk)
        c_ov_u = _cos_angles(sum(_sub(X, k) for X in OVF), Uk)
        prj = lambda V, A, m: float(((V[:, :m].T @ A) ** 2).sum()
                                    / ((A ** 2).sum() + EPS))

        f = dict(
            # --- E: embedding ------------------------------------------------
            E_emb=e_emb,
            E_aniso=_aniso(Bk),
            E_top=float(np.linalg.svd(Bk, compute_uv=False)[0] ** 2),
            # --- U: unembedding ----------------------------------------------
            U_emb=e_unb,
            U_aniso=_aniso(Uk),
            U_top=float(np.linalg.svd(Uk, compute_uv=False)[0] ** 2),
            # --- V: OV path ---------------------------------------------------
            V_tk=tk,
            V_tk_max=float(tk_h.max()),
            V_tk_pr=float(_pr(tk_h)),
            V_gain=tk / (e_emb + EPS),
            V_top_sv=float(sv_ov[0] ** 2),
            V_sv_pr=float(_pr(sv_ov ** 2)),
            # --- Q: QK path ---------------------------------------------------
            Q_q=q_e,
            Q_k=k_e,
            Q_self=qk_self,
            Q_pos=qk_pos,
            Q_gain=(q_e + k_e) / (e_emb + EPS),
            # --- M: MLP path ---------------------------------------------------
            M_ov=mlp_ov,
            M_dir=mlp_dir,
            M_gain=mlp_ov / (tk + EPS),
            M_loop=loop_kk,
            M_loop_row=loop_row,
            M_loop_purity=loop_kk / (loop_row + EPS),
            M_match=match,
            M_pr_read=pr_read,
            M_pr_write=pr_write,
            # --- P: phase / geometry -------------------------------------------
            P_eu_sum=float((c_eu ** 2).sum()),
            P_eu_max=float(c_eu[0]),
            P_eu_min=float(c_eu[-1]),
            P_ovu_sum=float((c_ov_u ** 2).sum()),
            P_e_in2=prj(Vi, Bk, 2),
            P_e_in8=prj(Vi, Bk, 8),
            P_e_in32=prj(Vi, Bk, 32),
            P_u_out2=prj(Vo, Uk, 2),
            P_u_out8=prj(Vo, Uk, 8),
            P_u_out32=prj(Vo, Uk, 32),
            # --- X: explicit cross terms ----------------------------------------
            X_tk_u=tk * e_unb,
            X_tk_match=tk * match,
            X_emb_u=e_emb * e_unb,
            X_loop_eu=loop_kk * float((c_eu ** 2).sum() + EPS),
            # --- K: bare frequency index ------------------------------------------
            K_idx=float(k) / nf,
        )
        if names is None:
            names = list(f.keys())
        rows.append([f[n] for n in names])
    return names, np.array(rows, float)


GROUP_OF = dict(E="E", U="U", V="V", Q="Q", M="M", P="P", X="X", K="K")


def expand(raw, names):
    """Raw (nf, F) -> (nf, 3F) with within-run z and rank transforms."""
    lg = np.log(np.abs(raw) + 1e-12)
    mu, sd = lg.mean(0), lg.std(0)
    z = (lg - mu) / np.where(sd < 1e-12, 1.0, sd)
    nf = raw.shape[0]
    r = np.argsort(np.argsort(lg, axis=0), axis=0) / (nf - 1.0)
    out = np.concatenate([lg, z, r], axis=1)
    nm = names + [n + "__z" for n in names] + [n + "__r" for n in names]
    return out, nm


def build(cache=None, verbose=True):
    cache = Path(cache) if cache else None
    if cache and cache.exists():
        z = np.load(cache, allow_pickle=True)
        return dict(X=z["X"], y=z["y"], tk=z["tk"], names=list(z["names"]),
                    run=list(z["run"]), cohort=list(z["cohort"]),
                    grp=list(z["grp"]), fam=list(z["fam"]),
                    kidx=z["kidx"], runidx=z["runidx"])

    Xs, ys, tks, runs, cohorts, grps, fams, kidx, runidx = ([] for _ in range(9))
    names_out = None
    for i, r in enumerate(discover(require_e0=True)):
        if r["e0"].stat().st_size < 1000:
            print(f"SKIP (LFS pointer) {r['rel']}", file=sys.stderr)
            continue
        p = r["cfg"].p
        nf = p // 2
        params = load_ckpt(r["e0"])
        names, raw = run_features(params, p)
        feat, nm = expand(raw, names)
        names_out = nm
        lab = np.zeros(nf, bool)
        lab[np.array(r["committee"]) - 1] = True
        Xs.append(feat)
        ys.append(lab)
        tks.append(raw[:, names.index("V_tk")])
        runs += [r["rel"]] * nf
        cohorts += [r["cohort"]] * nf
        fams += [r["fam"]] * nf
        # CV group = the independent init draw. All 96 v2 runs come from just
        # 8 (data_seed, init_seed) draws; surgical arms and orth-flat variants
        # are deterministic edits of the SAME epoch-0 draw, so grouping by run
        # (or by common.cluster_key) would leak the init across folds.
        grps += [f"{r['cfg'].data_seed}/{r['cfg'].init_seed}"] * nf
        kidx.append(np.arange(1, nf + 1))
        runidx.append(np.full(nf, i))
        if verbose:
            print(f"  {r['rel']:<44} K={len(r['committee'])}", flush=True)

    out = dict(X=np.concatenate(Xs), y=np.concatenate(ys),
               tk=np.concatenate(tks), names=names_out, run=runs,
               cohort=cohorts, grp=grps, fam=fams,
               kidx=np.concatenate(kidx), runidx=np.concatenate(runidx))
    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, **{k: np.array(v, dtype=object) if
                                      isinstance(v, list) else v
                                      for k, v in out.items()})
    return out


if __name__ == "__main__":
    d = build(cache=sys.argv[1] if len(sys.argv) > 1 else None)
    print(f"\nX {d['X'].shape}  y {d['y'].sum()}/{len(d['y'])} positives  "
          f"{len(set(d['run']))} runs  {len(set(d['grp']))} init groups")
