"""Models, measurements and constructions (core/models.py, core/measure.py, core/construct.py)."""
import math

import numpy as np
import pytest
import torch

from core import construct as C
from core import measure as MS
from core import models as M
from core import programme as P


def rand_net(model, n, m, p, seed, scale=1.0):
    g = torch.Generator().manual_seed(seed)
    U = torch.randn((1, n, m, p), generator=g, dtype=torch.float64) * scale / math.sqrt(p)
    W = torch.randn((1, m, p), generator=g, dtype=torch.float64) * scale / math.sqrt(p)
    return U, W


@pytest.mark.parametrize("model", ["product", "power"])
@pytest.mark.parametrize("n", [2, 3])
def test_logits_match_definition(model, n):
    p, m = 5, 3
    U, W = rand_net(model, n, m, p, 0)
    F = M.logits(model, U, W)[0].numpy().reshape((p,) * (n + 1))
    Un, Wn = U[0].numpy(), W[0].numpy()
    for x in np.ndindex(*(p,) * n):
        for c in range(p):
            if model == "product":
                v = sum(np.prod([Un[t, i, x[t]] for t in range(n)]) * Wn[i, c] for i in range(m))
            else:
                v = sum(sum(Un[t, i, x[t]] for t in range(n)) ** n * Wn[i, c] for i in range(m))
            assert abs(F[x + (c,)] - v) < 1e-12
    y = M.labels(p, n).numpy()
    X = M.inputs(p, n).numpy()
    assert np.all(y == X.sum(1) % p)


@pytest.mark.parametrize("model", ["product", "power"])
@pytest.mark.parametrize("q", [2.0, 2.5, 3.0])
def test_autograd_vs_finite_differences(model, q):
    """d/dtheta [mean CE + lam * sum_i ||omega_i||^q] by autograd vs central differences (float64)."""
    n, m, p, lam = 2, 3, 5, 0.03
    U, W = rand_net(model, n, m, p, 1, scale=1.5)
    y = M.labels(p, n)

    def loss(U_, W_):
        return M.ce_margin_form(M.logits(model, U_, W_), y)[0] + lam * M.penalty(U_, W_, q)[0]

    U.requires_grad_(True)
    W.requires_grad_(True)
    gU, gW = torch.autograd.grad(loss(U, W), (U, W))
    rng = np.random.default_rng(0)
    eps = 1e-6
    with torch.no_grad():
        for _ in range(25):
            if rng.random() < 0.5:
                idx = tuple(int(rng.integers(0, s)) for s in U.shape)
                Up, Um = U.clone(), U.clone()
                Up[idx] += eps
                Um[idx] -= eps
                fd = (loss(Up, W) - loss(Um, W)) / (2 * eps)
                assert abs(fd - gU[idx]) < 1e-7 * max(1.0, abs(fd)), (idx, fd, gU[idx])
            else:
                idx = tuple(int(rng.integers(0, s)) for s in W.shape)
                Wp, Wm = W.clone(), W.clone()
                Wp[idx] += eps
                Wm[idx] -= eps
                fd = (loss(U, Wp) - loss(U, Wm)) / (2 * eps)
                assert abs(fd - gW[idx]) < 1e-7 * max(1.0, abs(fd)), (idx, fd, gW[idx])
    # analytic penalty gradient
    rU, rW = M.penalty_grad(U.detach(), W.detach(), q)
    aU, aW = torch.autograd.grad(M.penalty(U, W, q)[0], (U, W))
    assert torch.allclose(rU, aU, atol=1e-12) and torch.allclose(rW, aW, atol=1e-12)


def test_ce_margin_form_equals_standard():
    torch.manual_seed(0)
    F = torch.randn(3, 125, 5, dtype=torch.float64) * 4
    y = M.labels(5, 3)
    F.requires_grad_(True)
    a = M.ce_margin_form(F, y)
    b = M.ce_standard(F, y)
    assert torch.allclose(a, b, atol=1e-13)
    ga = torch.autograd.grad(a.sum(), F)[0]
    gb = torch.autograd.grad(b.sum(), F)[0]
    assert torch.allclose(ga, gb, atol=1e-14)


def test_ce_margin_form_float32_small_loss_gradient():
    """In float32 the margin form keeps the correct-class gradient when the wrong-class mass is ~1e-9."""
    p, n = 5, 1
    y = M.labels(p, n)
    F = torch.full((1, p, p), -20.0, dtype=torch.float64)
    F[0, torch.arange(p), y] = 0.0  # S = 4 e^{-20} ~ 8e-9
    F32 = F.float().requires_grad_(True)
    g32 = torch.autograd.grad(M.ce_margin_form(F32, y).sum(), F32)[0]
    F64 = F.clone().requires_grad_(True)
    g64 = torch.autograd.grad(M.ce_margin_form(F64, y).sum(), F64)[0]
    rel = ((g32.double() - g64).abs().max() / g64.abs().max()).item()
    assert rel < 1e-5, rel
    # the standard form loses the correct-class gradient in float32 here
    F32b = F.float().requires_grad_(True)
    gs = torch.autograd.grad(torch.nn.functional.cross_entropy(F32b[0], y), F32b)[0]
    assert gs[0, torch.arange(p), y].abs().max().item() == 0.0


@pytest.mark.parametrize("model", ["product", "power"])
@pytest.mark.parametrize("n", [2, 3])
def test_gamma_scale_invariance(model, n):
    p, m = 7, 6
    U, W = rand_net(model, n, m, p, 3)
    for q in [2.0, (2 + n + 1) / 2, n + 1.0]:
        T = MS.table(model, U, W)
        g = MS.normalized_margin(T, U, W, q)
        for c in [0.1, 3.7, 50.0]:
            T2 = MS.table(model, c * U, c * W)
            g2 = MS.normalized_margin(T2, c * U, c * W, q)
            assert abs(g2 - g) <= 1e-10 * abs(g)


def test_fft_normalisation_and_modes():
    p, n = 7, 2
    idx = np.indices((p,) * (n + 1))
    a1, a2, c = idx
    for j in [1, 2, 3]:
        for phi in [0.0, 0.7]:
            Cst = 1.7
            T = Cst * np.cos(2 * np.pi * j * (a1 + a2 - c) / p + phi)
            sp = MS.spectrum(T)
            assert abs(sp["beta"][j - 1] - Cst * np.exp(1j * phi)) < 1e-12
            assert np.abs(np.delete(sp["beta"], j - 1)).max() < 1e-12
            assert max(np.abs(v).max() for v in sp["junk"].values()) < 1e-12
            assert abs(sp["frac"]["clean"] - 1) < 1e-12
        # a junk table: operand 2 flipped
        T = 0.9 * np.cos(2 * np.pi * j * (a1 - a2 - c) / p)
        sp = MS.spectrum(T)
        assert np.abs(sp["beta"]).max() < 1e-12
        assert abs(sp["junk"][(1, -1)][j - 1] - 0.9) < 1e-12
        assert abs(sp["frac"]["junk"] - 1) < 1e-12
    # averaged-margin identity (app:spectrum): eta(0) - (eta(d)+eta(-d))/2 = sum_j Re(beta_j) cov_j(d)
    rng = np.random.default_rng(0)
    T = rng.normal(size=(p,) * (n + 1))
    sp = MS.spectrum(T)
    y = MS.label_table(p, n)
    flat = T.reshape(-1, p)
    yy = y.reshape(-1)
    eta = np.array([flat[np.arange(len(yy)), (yy - d) % p].mean() for d in range(p)])
    A = P.coverage_matrix(p)
    lhs = eta[0] - 0.5 * (eta[1:(p - 1) // 2 + 1] + eta[::-1][:(p - 1) // 2])
    assert np.allclose(lhs, A @ sp["beta"].real, atol=1e-12)
    # phase lemma: margin <= sum_j Re(beta_j) cov_j(delta) for every delta
    assert MS.margin(T) <= (A @ sp["beta"].real).min() + 1e-12


@pytest.mark.parametrize("p", [5, 7, 11, 13])
@pytest.mark.parametrize("n", [2, 3, 4])
def test_nu_block_attains_class_optimum(p, n):
    if p ** (n + 1) > 400_000:
        pytest.skip("table too large for the test budget")
    nu = n + 1
    for q in [2.0, (2 + nu) / 2, float(nu)]:
        s = nu / q
        sol = P.solve(p, s)
        for y in sol.y_all[:3]:
            U, W = C.nu_block(p, n, y, width=nu * int((y > 0).sum()) + 2)
            T = MS.table("product", U, W)
            assert np.abs(T - C.target_table(p, n, y)).max() < 1e-12
            sp = MS.spectrum(T)
            assert np.abs(sp["beta"] - y).max() < 1e-12  # measures back exactly
            assert sp["junk_abs"].max() < 1e-12
            g = MS.normalized_margin(T, U, W, q)
            ce = MS.ceilings(p, n, q, sol.gamma)
            assert abs(g / ce["class_IF"] - 1) < 1e-10
            assert g <= ce["universal_product"] * (1 + 1e-12)
            # B_p from the definition: a balanced single-frequency neuron attains |beta|/||omega||^nu = B_p
            nf = MS.neuron_fourier(U[:, :, :1], W[:, :1])
            j = int(np.flatnonzero(y > 0)[0])
            sq = nf["sqnorm"][0]
            assert abs(abs(nf["beta_neuron"][0, j]) / sq ** (nu / 2) - MS.B_p(p, nu)) < 1e-12 * MS.B_p(p, nu)


@pytest.mark.parametrize("p", [5, 7, 11, 13, 17])
def test_morwani_dense_attains_power_ceiling(p):
    U, W = C.morwani_dense(p)
    T = MS.table("power", U, W)
    y = np.full((p - 1) // 2, 2 / (p - 1))
    assert np.abs(T - C.target_table(p, 2, y)).max() < 1e-12
    sp = MS.spectrum(T)
    assert np.abs(sp["beta"] - y).max() < 1e-12
    g = MS.normalized_margin(T, U, W, 3.0)
    ce = MS.ceilings(p, 2, 3.0, P.gamma(p, 1.0))
    assert abs(g / ce["universal_power"] - 1) < 1e-10
    assert abs(g - math.sqrt(2 / 27) / (p ** 0.5 * (p - 1))) < 1e-10 * g  # cor:dense closed form


def test_single_neuron_budget_bound():
    """Random product neurons never exceed B_p: sum_j |beta_j|^{q/nu} <= C_q ||omega||^q with C_q^{nu/q} = B_p."""
    rng = np.random.default_rng(0)
    for p, n in [(7, 2), (11, 2), (7, 3)]:
        nu = n + 1
        for _ in range(200):
            U = rng.normal(size=(1, n, 1, p))
            W = rng.normal(size=(1, 1, p))
            nf = MS.neuron_fourier(U, W)
            b = np.abs(nf["beta_neuron"][0])
            sq = nf["sqnorm"][0]
            for q in [2.0, 2.5, float(nu)]:
                lhs = (b ** (q / nu)).sum()
                rhs = MS.B_p(p, nu) ** (q / nu) * sq ** (q / 2)
                assert lhs <= rhs * (1 + 1e-12)


@pytest.mark.parametrize("n", [2, 3])
def test_ablation_on_constructed_network(n):
    p = 11
    nu = n + 1
    q = 2.0
    s = nu / q
    sol = P.solve(p, s)
    y = sol.y_all[0]
    U, W = C.nu_block(p, n, y)
    ab = MS.ablation_margins("product", U, W)
    A = P.coverage_matrix(p)
    assert abs(ab["h_base"] - sol.gamma) < 1e-12
    w = MS.budget_shares(y, s)
    for j in range(1, (p - 1) // 2 + 1):
        y2 = y.copy()
        y2[j - 1] = 0
        expect = (A @ y2).min()
        assert abs(ab["h_ablated"][j - 1] - expect) < 1e-12
        if y[j - 1] == 0:
            assert abs(ab["rel_loss"][j - 1]) < 1e-12
        else:  # CONSEQUENCES §3 lower bound for the clean allocation
            assert ab["rel_loss"][j - 1] >= 1 - (1 - w[j - 1]) ** s - 1e-12
    # ablate_frequency is a projection: ablating twice = once, and it removes exactly +-j
    U1, W1 = MS.ablate_frequency(U, W, 2)
    U2, W2 = MS.ablate_frequency(U1, W1, 2)
    assert np.abs(U1 - U2).max() < 1e-13 and np.abs(W1 - W2).max() < 1e-13
    assert np.abs(np.fft.fft(U1, axis=-1)[..., [2, p - 2]]).max() < 1e-12


def test_counts_definitions():
    A = np.array([1.0, 0.5, 0.15, 0.08, 0.0])
    c = MS.counts(A, 1.5)
    assert c["thr_1"] == 4 and c["thr_5"] == 4 and c["thr_10"] == 3 and c["thr_20"] == 2
    assert abs(c["pr_amp"] - A.sum() ** 2 / (A ** 2).sum()) < 1e-12
    a = A ** (1 / 1.5)
    assert abs(c["pr_budget"] - a.sum() ** 2 / (a ** 2).sum()) < 1e-12
    w = np.sort(a / a.sum())[::-1]
    for t in (90, 95, 99):
        k = c[f"mass_{t}"]
        assert w[:k].sum() >= t / 100 - 1e-12 and (k == 1 or w[:k - 1].sum() < t / 100)
    flat = MS.counts(np.ones(6), 2.0)
    assert all(flat[k] == 6 for k in ("thr_1", "thr_20", "mass_99")) and abs(flat["pr_amp"] - 6) < 1e-12
