"""Agreement metrics (core/metrics.py): a constructed network must score as a perfect match."""
import numpy as np
import pytest

from core import construct as C
from core import measure as MS
from core import metrics as MT
from core import programme as P


@pytest.mark.parametrize("p,n,q", [(11, 2, 2.0), (13, 2, 2.5), (13, 3, 2.0), (11, 2, 3.0)])
def test_constructed_network_is_perfect_match(p, n, q):
    s = (n + 1) / q
    sol = P.solve(p, s)
    y = sol.y_all[-1]
    u = 2 if (p - 1) // 2 >= 2 else 1
    yd = P.dilate(y, u, p)  # a dilate: the alignment must find it
    U, W = C.nu_block(p, n, yd)
    meas = MS.full_measurement("product", U, W, q, with_ablation=False)
    sc = MT.score_run(meas)
    assert sc["prediction_label"].startswith("exact")
    # Tolerance note: off-support amplitudes of a constructed table are ~1e-17 (float64 FFT noise), and the
    # budget normalisation raises them to the power 1/s (e.g. (1e-17)^0.4 ~ 1e-7), so budget-normalised
    # quantities agree to ~1e-6, while beta itself measures back to 1e-12 (test_nu_block_attains_class_optimum).
    assert abs(sc["amp"]["max_abs_error"]) < 1e-6
    assert abs(sc["share"]["tv"]) < 1e-5
    assert sc["amp"]["cosine"] > 1 - 1e-12
    if np.std(yd) > 0 and len(set(np.round(yd, 12))) > 1:
        assert sc["amp"]["r2_identity"] > 1 - 1e-9 and sc["amp"]["r2_linear_fit"] > 1 - 1e-9
    for t in MT.THRESHOLDS:
        assert sc[f"jaccard_thr_{t}"] == 1.0 or yd[yd > 0].min() < t / 100 * yd.max()
    for k in MS.COUNT_KEYS:
        assert sc["counts"][k]["exact"]
    assert sc["counts"]["mass_99"]["meas"] <= sc["kstar"]
    assert abs(sc["gamma_over_ceiling"]["class_IF"] - 1) < 1e-10


def test_elementary_metrics():
    a = np.array([1.0, 2.0, 3.0, 4.0])
    assert MT.r2_identity(a, a) == 1.0
    assert abs(MT.r2_linear_fit(2 * a + 1, a) - 1) < 1e-12
    assert MT.r2_identity(2 * a + 1, a) < 0.0  # identity-line R^2 penalises slope and offset
    assert abs(MT.tv_distance([0.5, 0.5], [1.0, 0.0]) - 0.5) < 1e-12
    assert MT.jaccard({1, 2}, {2, 3}) == 1 / 3
    assert abs(MT.spearman(a, a ** 3) - 1) < 1e-12 and abs(MT.kendall(a, -a) + 1) < 1e-12
    assert MT.frac_within([1, 2, 5], [1, 3, 3]) == 2 / 3
    fit = MT.linfit_ci(a, 2 * a + 1 + np.array([0.01, -0.01, 0.02, -0.02]))
    assert fit["slope_ci"][0] < 2 < fit["slope_ci"][1]
    bs = MT.bootstrap_over_seeds([1.0, 2.0, 3.0, 4.0], n_boot=2000)
    assert bs["ci"][0] <= 2.5 <= bs["ci"][1]
    sw = MT.sweep_summary([3, 4, 5], [3, 4, 6])
    assert sw["frac_within_tol"] == 1.0 and abs(sw["frac_exact"] - 2 / 3) < 1e-12
