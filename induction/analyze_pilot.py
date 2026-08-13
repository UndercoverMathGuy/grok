"""Induction-head lottery pilot — analysis.

Reads runs_induction/pilot/*/  (metrics.json + checkpoints/step_00000)
and answers the pre-registered pilot questions (induction/README.md):

  PQ0 concentration : is the induction circuit sparse (one dominant L1
                      head), or distributed? -> top-1 share of induction
                      score mass and of ablation delta-CE mass.
  PQ1 lottery       : does winner identity vary across init seeds, and is
                      it stable across data seeds at fixed init?
                      -> distinct winners, twin agreement matrix.
  PQ2 init readout  : is the winner predictable from the INIT weights
                      alone? Candidate closed forms, all Elhage-style and
                      weights-only:
                        prev0[h]   L0 positional prev-token QK bias
                        kcomp[g,h] K-composition ||W_QK^h(L1)^T W_OV^g(L0)||_F
                                   / (||W_QK^h||_F ||W_OV^g||_F)
                        copy1[h]   L1 OV copying score, eigvals of
                                   W_E W_OV W_U (sum Re / sum abs)
                        pair score = z(prev0[g]) + z(max-over-g' kcomp) ...
                      reported as top-1 hit rate (chance 1/H), mean rank,
                      pooled AUC, permutation p.

Usage: python3 -u induction/analyze_pilot.py [--runs runs_induction/pilot]
Writes notes/pivot/induction_pilot.json and prints the summary.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file


def load_run(rd: Path):
    hist = json.loads((rd / "metrics.json").read_text())
    cfg = json.loads((rd / "config.json").read_text())
    init = load_file(str(rd / "checkpoints" / "step_00000.safetensors"))
    return cfg, hist, init


def winner_stats(hist, H):
    fin = hist[-1]
    ind = np.array(fin["induction_l1"])
    abl = np.array(fin.get("ablate_dce_l1", [np.nan] * H))
    w_ind = int(ind.argmax())
    w_abl = int(abl.argmax()) if np.isfinite(abl).all() else w_ind
    conc_ind = float(ind.max() / max(ind.sum(), 1e-12))
    pos = np.clip(abl, 0, None)
    conc_abl = float(pos.max() / max(pos.sum(), 1e-12))
    # formation step: first eval where winner's induction score crosses
    # half its final value
    half = ind[w_ind] / 2
    t_form = next((h["step"] for h in hist
                   if h["induction_l1"][w_ind] >= half), None)
    return dict(winner_ind=w_ind, winner_abl=w_abl, conc_ind=conc_ind,
                conc_abl=conc_abl, t_form=t_form,
                final_ce=fin["probe_ce"], ind=ind.tolist(), abl=abl.tolist())


def init_readouts(init, cfg):
    H, d = cfg["n_heads"], cfg["d_model"]
    WE, Wpos, WU = init["embed.W_E"], init["pos.W_pos"], init["unembed.W_U"]
    out = {}
    # L0 prev-token positional bias: pre-softmax logit margin of key i-1
    WQ0, WK0 = init["blocks.0.attn.W_Q"], init["blocks.0.attn.W_K"]
    prev0 = []
    T = Wpos.shape[0]
    for h in range(H):
        S = (Wpos @ WQ0[h]) @ (Wpos @ WK0[h]).T / np.sqrt(cfg["d_head"])
        margins = [S[i, i - 1] - S[i, :i + 1].mean() for i in range(1, T)]
        prev0.append(float(np.mean(margins)))
    out["prev0"] = prev0
    # OV and QK per head/layer
    def ov(l, h):
        return init[f"blocks.{l}.attn.W_V"][h] @ init[f"blocks.{l}.attn.W_O"][h]
    def qk(l, h):
        return init[f"blocks.{l}.attn.W_Q"][h] @ init[f"blocks.{l}.attn.W_K"][h].T
    # K-composition L0 g -> L1 h
    kcomp = np.zeros((H, H))
    for g in range(H):
        OVg = ov(0, g)
        nOV = np.linalg.norm(OVg)
        for h in range(H):
            QKh = qk(1, h)
            kcomp[g, h] = (np.linalg.norm(QKh.T @ OVg)
                           / (np.linalg.norm(QKh) * nOV + 1e-30))
    out["kcomp"] = kcomp.tolist()
    # L1 copy score: eigvals of W_E W_OV W_U (vocab x vocab)
    copy1 = []
    for h in range(H):
        M = WE @ ov(1, h) @ WU
        ev = np.linalg.eigvals(M)
        copy1.append(float(ev.real.sum() / (np.abs(ev).sum() + 1e-30)))
    out["copy1"] = copy1
    # candidate L1-winner scores
    z = lambda a: (np.asarray(a) - np.mean(a)) / (np.std(a) + 1e-30)
    out["score_kmax"] = kcomp.max(axis=0).tolist()          # best-composed
    out["score_kmax_copy"] = (z(kcomp.max(axis=0)) + z(copy1)).tolist()
    return out


def committee_from_deltas(deltas, floor_frac=0.05):
    """Mod-add-style set detector on final ablation delta-CEs: sort desc,
    cut at the largest gap among positive deltas; members must exceed
    floor_frac * max delta. Returns sorted member list."""
    d = np.asarray(deltas)
    order = np.argsort(d)[::-1]
    s = d[order]
    pos = s > max(s.max(), 1e-9) * floor_frac
    n_pos = int(pos.sum())
    if n_pos <= 1:
        return sorted(order[:1].tolist())
    gaps = s[:n_pos - 1] - s[1:n_pos]
    cut = int(np.argmax(gaps)) + 1
    return sorted(order[:cut].tolist())


def jaccard(a, b):
    a, b = set(a), set(b)
    return len(a & b) / max(len(a | b), 1)


def rank_of(scores, winner):
    order = np.argsort(np.asarray(scores))[::-1]
    return int(np.where(order == winner)[0][0]) + 1


def auc(labels, scores):
    labels, scores = np.asarray(labels), np.asarray(scores)
    pos, neg = scores[labels == 1], scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    return float((pos[:, None] > neg[None, :]).mean()
                 + 0.5 * (pos[:, None] == neg[None, :]).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs_induction/pilot_induction")
    args = ap.parse_args()
    run_dirs = sorted(d for d in Path(args.runs).iterdir()
                      if (d / "metrics.json").exists())
    if not run_dirs:
        raise SystemExit(f"no finished runs under {args.runs}")

    rows = []
    for rd in run_dirs:
        cfg, hist, init = load_run(rd)
        H = cfg["n_heads"]
        row = dict(run=rd.name, init_seed=cfg["init_seed"],
                   data_seed=cfg["data_seed"], **winner_stats(hist, H))
        row["prev_final"] = hist[-1]["prevtoken_l0"]
        row["readouts"] = init_readouts(init, cfg)
        rows.append(row)
    H = json.loads((run_dirs[0] / "config.json").read_text())["n_heads"]

    # ---- PQ0 concentration
    conc = np.array([r["conc_ind"] for r in rows])
    conc_abl = np.array([r["conc_abl"] for r in rows])
    agree = np.mean([r["winner_ind"] == r["winner_abl"] for r in rows])

    # ---- PQ1 lottery
    winners = np.array([r["winner_ind"] for r in rows])
    distinct = len(set(winners.tolist()))
    by_init, same_init_agree = {}, []
    for r in rows:
        by_init.setdefault(r["init_seed"], []).append(r["winner_ind"])
    for ws in by_init.values():
        if len(ws) > 1:
            same_init_agree.append(
                float(np.mean([a == b for i, a in enumerate(ws)
                               for b in ws[i + 1:]])))
    cross = [a == b for i, a in enumerate(winners) for j, b in
             enumerate(winners) if i < j
             and rows[i]["init_seed"] != rows[j]["init_seed"]]

    # ---- PQ2 init readout
    pq2 = {}
    for name in ("score_kmax", "score_kmax_copy", "copy1"):
        ranks = [rank_of(r["readouts"][name], r["winner_ind"]) for r in rows]
        labels = [1 if h == r["winner_ind"] else 0
                  for r in rows for h in range(H)]
        scores = [r["readouts"][name][h] for r in rows for h in range(H)]
        hits = sum(1 for k in ranks if k == 1)
        # one-sided binomial vs chance 1/H via normal approx + exact-ish sum
        n = len(ranks)
        from math import comb
        pval = sum(comb(n, k) * (1 / H) ** k * (1 - 1 / H) ** (n - k)
                   for k in range(hits, n + 1))
        pq2[name] = dict(top1=hits / n, mean_rank=float(np.mean(ranks)),
                         auc=auc(labels, scores), binom_p=float(pval))

    # ---- committee-level analysis (prereg fallback branch: the identity
    # variable is the SET of induction heads by ablation, mod-add style)
    for r in rows:
        r["committee"] = committee_from_deltas(r["abl"]) \
            if np.isfinite(r["abl"]).all() else [r["winner_ind"]]
    ks = [len(r["committee"]) for r in rows]
    same_j, cross_j = [], []
    for i, a in enumerate(rows):
        for b in rows[i + 1:]:
            j = jaccard(a["committee"], b["committee"])
            (same_j if a["init_seed"] == b["init_seed"] else cross_j).append(j)
    # popularity prior (leave-one-out): score head h by membership rate in
    # other runs; the init readout must beat THIS, not chance
    mem = np.array([[1 if h in r["committee"] else 0 for h in range(H)]
                    for r in rows])
    pop_scores, pop_labels = [], []
    kc_scores, kc_labels = [], []
    for i, r in enumerate(rows):
        prior = (mem.sum(0) - mem[i]) / (len(rows) - 1)
        kmax = np.asarray(r["readouts"]["score_kmax"])
        for h in range(H):
            pop_scores.append(prior[h])
            pop_labels.append(mem[i, h])
            kc_scores.append(kmax[h])
            kc_labels.append(mem[i, h])
    # coupled-race test: do L1 committee members compose preferentially
    # with the WINNING L0 head (argmax prev-token score at final)?
    coupled = []
    for r in rows:
        kcomp = np.asarray(r["readouts"]["kcomp"])       # (g,h) L0 x L1
        g_win = int(np.argmax(r["prev_final"]))
        in_c = np.array([h in r["committee"] for h in range(H)])
        if in_c.all() or not in_c.any():
            continue
        coupled.append(float(kcomp[g_win, in_c].mean()
                             - kcomp[g_win, ~in_c].mean()))
    # detector-independent effective committee size: participation ratio
    # of positive ablation deltas — robust where the gap detector isn't
    prs = []
    for r in rows:
        d = np.clip(np.asarray(r["abl"], dtype=float), 0, None)
        prs.append(float(d.sum() ** 2 / max((d ** 2).sum(), 1e-30)))
    pq_committee = dict(
        k_mean=float(np.mean(ks)), k_counts={str(k): ks.count(k)
                                             for k in sorted(set(ks))},
        k_eff_mean=float(np.mean(prs)), k_eff_sd=float(np.std(prs)),
        head_popularity=(mem.mean(0)).tolist(),
        twin_jaccard=float(np.mean(same_j)) if same_j else None,
        cross_jaccard=float(np.mean(cross_j)) if cross_j else None,
        membership_auc_kcomp=auc(kc_labels, kc_scores),
        membership_auc_popularity_prior=auc(pop_labels, pop_scores),
        kcomp_to_winning_l0_delta=(float(np.mean(coupled))
                                   if coupled else None),
    )

    summary = dict(
        n_runs=len(rows), n_heads=H,
        pq_committee=pq_committee,
        pq0=dict(conc_ind_mean=float(conc.mean()),
                 conc_ind_min=float(conc.min()),
                 conc_abl_mean=float(np.nanmean(conc_abl)),
                 winner_metric_agreement=float(agree)),
        pq1=dict(distinct_winners=distinct,
                 winner_counts={str(h): int((winners == h).sum())
                                for h in sorted(set(winners.tolist()))},
                 same_init_twin_agreement=(float(np.mean(same_init_agree))
                                           if same_init_agree else None),
                 cross_init_agreement=(float(np.mean(cross))
                                       if cross else None)),
        pq2=pq2,
        final_ce=dict(mean=float(np.mean([r["final_ce"] for r in rows])),
                      max=float(np.max([r["final_ce"] for r in rows]))),
        t_form=[r["t_form"] for r in rows],
    )
    out = Path("notes/pivot/induction_pilot.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(summary=summary, runs=rows), indent=1,
                              default=str))
    print(json.dumps(summary, indent=1))
    print(f"\nwritten: {out}")


if __name__ == "__main__":
    main()
