"""P2: Theorem P2-2 (quantitative Hadamard) and the kill-point moment identities.

THEOREM P2-2.  For n+1 unit-modulus v_1..v_{n+1} with power sums P_k = sum_l v_l^k,

    V^2 = prod_{l<m}|v_l - v_m|^2 <= (n+1)^{n+1} exp( -(n+1)^{-3} sum_{k=1..n}(n+1-k)|P_k|^2 ).

Proof.  Let W_{a,l} = v_l^a (a = 0..n); then det(W W*) = |det W|^2 = V^2 and
(W W*)_{ab} = P_{a-b}.  Put A = W W*/(n+1): PSD, unit diagonal, trace n+1, so every
eigenvalue lies in (0, n+1].  For lambda in (0, Lambda],
    log lambda <= (lambda - 1) - (lambda - 1)^2/(2 Lambda)
(both sides agree with derivative 0 at lambda = 1 and at lambda = Lambda, and the difference
is monotone on each side).  Summing over eigenvalues, with sum(lambda - 1) = 0,
    log det A <= -||A - I||_F^2 / (2(n+1)) = -(n+1)^{-3} sum_k (n+1-k)|P_k|^2 .   QED

KILL-POINT MOMENTS.  For N = n+1, the kill point of atom l0 under a bijection sigma is
u_j = -conj(v_{sigma(j),j}).  m_T := average over all (n+1)! (l0, sigma) of u^T.  Proved here
by direct expansion at n = 2:
    m_{j}    = -conj(S_j)/(n+1),          S_j = sum_l v_lj                       (every n)
    m_{1,2}  = (conj(S_1) conj(S_2) - Z)/6,   Z = sum_l conj(v_l1 v_l2)          (n = 2)
"""
import json, os
import numpy as np
from itertools import permutations
import p2_core as P


def had_check(n, trials=4000, seed=0):
    rng = np.random.default_rng(seed)
    nb = n + 1
    worst = -np.inf
    for _ in range(trials):
        v = np.exp(1j * rng.uniform(0, 2 * np.pi, nb))
        V2 = np.prod([abs(v[l] - v[m]) ** 2 for l in range(nb) for m in range(l + 1, nb)])
        Pk = np.array([np.sum(v ** k) for k in range(1, nb)])
        rhs = nb ** nb * np.exp(-sum((nb - k) * abs(Pk[k - 1]) ** 2
                                     for k in range(1, nb)) / nb ** 3)
        worst = max(worst, V2 / rhs)
    return worst


def moments(V):
    """m_T by brute force over all (n+1)! kill points."""
    N, n = V.shape
    U, _ = P.kill_points(V)
    M = P.masks(n)
    out = []
    for t in range(M.shape[0]):
        col = np.where(M[t][None, :], U, 1.0).prod(axis=1)
        out.append(col.mean())
    return np.array(out)


def main():
    res = {}
    for n in (1, 2, 3, 4):
        res[f"hadamard_n{n}"] = float(had_check(n))
        print(f"Theorem P2-2, n={n}: max V^2/bound = {res[f'hadamard_n{n}']:.9f} (must be <= 1)")

    rng = np.random.default_rng(7)
    e1 = e2 = 0.0
    for _ in range(400):
        V = np.exp(1j * rng.uniform(0, 2 * np.pi, (3, 2)))
        m = moments(V)
        S = V.sum(axis=0)
        Z = np.conj(V[:, 0] * V[:, 1]).sum()
        e1 = max(e1, abs(m[1] + np.conj(S[0]) / 3), abs(m[2] + np.conj(S[1]) / 3))
        e2 = max(e2, abs(m[3] - (np.conj(S[0]) * np.conj(S[1]) - Z) / 6))
    res["m_slot_identity_err"] = float(e1)
    res["m_pair_identity_err"] = float(e2)
    print(f"m_{{j}} = -conj(S_j)/(n+1)           : max error {e1:.3e}")
    print(f"m_{{1,2}} = (conj(S1)conj(S2) - Z)/6 : max error {e2:.3e}")

    # the configuration with zero Hadamard defect and a large cross moment
    w = np.exp(2j * np.pi / 3)
    V = np.array([[w ** (-l), w ** (l)] for l in range(3)], dtype=complex)
    m = moments(V)
    Mk = P.masks(2)
    print("\nzero-Hadamard-defect / misaligned-slots configuration v_l = (w^-l, w^+l):")
    print("  power sums per slot:", [complex(np.sum(V[:, j] ** k)) for j in (0, 1) for k in (1, 2)])
    print("  |m_T| =", np.abs(m).round(6))
    # its junk is forced: C_{1,2} = sum_l c_l = C_empty = 1 for ANY c
    rng2 = np.random.default_rng(3)
    mins = []
    for _ in range(200):
        c = rng2.standard_normal(3) + 1j * rng2.standard_normal(3)
        c = c / np.sum(c)
        C = P.amps(c, V, Mk)
        mins.append(float(np.sum(np.abs(C[1:]) ** 2)) ** 0.5)
    res["misaligned_min_eta"] = float(np.min(mins))
    print(f"  junk of that configuration: eta >= {np.min(mins):.6f} "
          f"(exactly: C_{{1,2}} = sum_l c_l = 1, so eta >= 1)")

    os.makedirs("FINAL_RUNS/P2", exist_ok=True)
    json.dump(res, open("FINAL_RUNS/P2/hadamard.json", "w"), indent=1)
    print("\nwrote FINAL_RUNS/P2/hadamard.json")


if __name__ == "__main__":
    main()
