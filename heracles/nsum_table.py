"""nsum_table.py -- aggregate heracles/runs/NS/*/seed*/metrics.json into the ladder table, next to the K* prediction
(heracles/kstar.py ladder at q = 2).  Usage: uv run python heracles/nsum_table.py [--runs heracles/runs/NS]"""
import argparse, json, sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kstar import ladder  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--runs", default=str(Path(__file__).resolve().parent / "runs" / "NS"))
ap.add_argument("--kmax", type=int, default=10)
a = ap.parse_args()

cells = defaultdict(list)
for f in sorted(Path(a.runs).glob("*/seed*/metrics.json")):
    r = json.loads(f.read_text()); cells[f.parent.parent.name].append(r)

pred_cache = {}
def pred(p, nu, q=2.0):
    """argmax_K M_K / K^(nu/q) with the dense reference (nf + 1/2) nf^(-nu/q); q = 2 is plain l2, q = nu is L_{2,nu}."""
    if p not in pred_cache:
        pred_cache[p] = ladder(p, a.kmax)
    M = pred_cache[p]["M_K"]; nf = pred_cache[p]["nf"]; s = nu / q
    gh = {K: M[K] * K ** (-s) for K in M}; gmax = max(gh.values()); ks = max(gh, key=gh.get)
    dense = (nf + 0.5) * nf ** (-s); plat = [K for K in gh if gh[K] / gmax >= 0.74]
    verdict = "dense" if dense >= gmax or ks == a.kmax else ks
    return f"{verdict} ({plat[0]}..{plat[-1]})" if plat else str(verdict)

print("| cell | nu | q | seeds | acc | margin | gamma | K_5pct | K_2pct | N_.05 | clean | top-share | freqs>=1% | K* (plateau) |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for name, rs in cells.items():
    rs.sort(key=lambda r: r["seed"])
    if "nu" not in rs[0]:
        continue                      # foreign metrics (e.g. converted QM cell)
    nu = rs[0]["nu"]; p = rs[0]["p"]
    def col(k, fmt="{}"):
        return ", ".join(fmt.format(r[k]) for r in rs)
    q = rs[0].get("q", 2.0)
    kp = pred(p, nu, q) if rs[0]["act"] != "relu" else "dense"
    print(f"| {name} | {nu} | {q:g} | {len(rs)} | {col('acc', '{:.3f}')} | {col('margin', '{:.2f}')} | {col('gamma', '{:.1e}')} | "
          f"{col('K_5pct')} | {col('K_2pct')} | {col('N_05')} | {col('clean_frac', '{:.2f}')} | {col('top_share', '{:.2f}')} | "
          f"{col('n_freq_hist_1pct')} | {kp} |")
