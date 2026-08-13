"""CE floor for the v2 induction task: a longest-suffix-match oracle.

At each scored position (second-copy q = o2+j, j <= L-2) the oracle finds
every earlier position whose preceding k-gram matches the current one
(largest k with any match, k <= KMAX), predicts the empirical distribution
of the next token over those matches (zipf prior as backoff/smoothing),
and we score its CE on the same positions the trainer scores. A trained
model cannot beat this by much — it approximates the Bayes predictor of
the generator. Model CE ~ oracle CE  =>  the circuit is at the task floor.

Usage: python3 induction/oracle_floor.py [--n 2048] [--kmax 6]
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from train_pilot import Config, gen_induction_np, zipf_probs  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2048)
    ap.add_argument("--kmax", type=int, default=6)
    ap.add_argument("--seed", type=int, default=777)
    ap.add_argument("--alpha", type=float, default=0.1,
                    help="zipf-prior smoothing weight")
    args = ap.parse_args()

    cfg = Config()
    probs = zipf_probs(cfg)
    rng = np.random.default_rng(args.seed)
    toks, lm, qidx, kidx, valid = gen_induction_np(rng, cfg, args.n, probs)

    ces, by_j = [], {}
    for b in range(args.n):
        seq = toks[b]
        qs = np.where(lm[b])[0]
        for j, q in enumerate(qs):
            pred = None
            for k in range(min(args.kmax, q + 1), 0, -1):
                pat = seq[q - k + 1:q + 1]
                # positions p in [k-1, q-1] with seq[p-k+1..p] == pat
                hits = [p for p in range(k - 1, q)
                        if np.array_equal(seq[p - k + 1:p + 1], pat)]
                if hits:
                    counts = np.zeros(cfg.vocab)
                    for p in hits:
                        counts[seq[p + 1]] += 1
                    pred = ((1 - args.alpha) * counts / counts.sum()
                            + args.alpha * probs)
                    break
            if pred is None:
                pred = probs
            ce = -np.log(pred[seq[q + 1]] + 1e-12)
            ces.append(ce)
            by_j.setdefault(j, []).append(ce)

    ces = np.array(ces)
    print(f"n={args.n} seqs, {len(ces)} scored positions, kmax={args.kmax}")
    print(f"oracle CE mean {ces.mean():.4f}   median {np.median(ces):.4f}")
    print("by position-in-segment j (match length j+1):")
    for j in sorted(by_j)[:10]:
        v = np.array(by_j[j])
        print(f"  j={j:2d}  n={len(v):5d}  ce {v.mean():.4f}")


if __name__ == "__main__":
    main()
