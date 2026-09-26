# Best curve the fable0923 matrix route can give: kappa_PSD(a) = min over e >= 1, 0 <= eta <= sqrt3 e of
#   L(e, eta) / (1 - a eta)_+^{2/3},  L = max(e^{2/3}, 3^{1/3} e^{4/3} (3/B)^{1/3}),  B = 3e^2 + 2eta^2 + 4 eta u + 2u^2, u = sqrt(2e(e-1)).
# Compare with the lone-atom conjecture kappa(a) = min(3^{1/3}, (1 - sqrt3 a)^{-2/3}).
import numpy as np
def L(e, h):
    u = np.sqrt(2 * e * (e - 1)); B = 3 * e ** 2 + 2 * h ** 2 + 4 * h * u + 2 * u ** 2
    return np.maximum(e ** (2 / 3), 3 ** (1 / 3) * e ** (4 / 3) * (3 / B) ** (1 / 3))
E = np.linspace(1, np.sqrt(3), 1201)[:, None]; H = np.linspace(0, 1, 2001)[None, :] * np.sqrt(3) * E
for a in [0, 0.02, 0.05, 0.08, 0.1, 0.12, 0.15, 0.18, 0.2, 0.22, 0.24, 0.244, 0.25, 0.26, 0.27, 0.272]:
    rhs = np.clip(1 - a * H, 0, None) ** (2 / 3)
    with np.errstate(divide='ignore'):
        ratio = np.where(rhs > 0, L(E, H) / rhs, np.inf)
    i = np.unravel_index(np.argmin(ratio), ratio.shape)
    kap = min(3 ** (1 / 3), (1 - np.sqrt(3) * a) ** (-2 / 3))
    print(f"a={a:.3f}  kappa_PSD={ratio[i]:.5f}  lone-atom kappa={kap:.5f}  ratio={ratio[i] / kap:.5f}  argmin e={E[i[0], 0]:.4f} eta={H[i]:.4f}")
