"""Turn the ansatz gamma(S) = M(S)/K^(nu/2) into a probability distribution
over final committees and fit it to the 69-run vanilla cohort.

Models (each with ONE fitted parameter):
  A  Boltzmann over K only:   P(K) ~ exp(gamma*(K)/T), gamma* = best set of size K
     (expected to fail: mode pinned at K*=3 for all T)
  B  Gibbs over SETS:         P(S) ~ exp(gamma(S)/T), all subsets of {1..56}
  C  Gibbs + veto law:        P(S) ~ 1[S collision-free] exp(gamma(S)/T)
     (collision = k_i +- k_j hits a third member mod p; 0/70 observed finals
      carry one, so vetoed sets get weight 0)
  B2/C2  power-law variants:  P(S) ~ gamma(S)^beta  (scale-free selection)

Exact enumeration K=2..5 (4.2M sets); MC (2M samples) for K=6..9.
Fit T (or beta) by maximum likelihood on the observed K histogram; then check
set-level predictions (per-frequency inclusion prob vs observed popularity,
quantile of observed sets' gamma inside their size class).
"""
import sys, json, glob, itertools, math
from collections import Counter
import numpy as np

sys.path.insert(0, "/Users/ruhaanrajadhyaksha/projects/grok")
from nudial.train_nudial import committee_from_coeffs

P, NU = 113, 6.0
NF = 56
M_K = {1: 0.0015, 2: 0.1875, 3: 0.8446, 4: 1.8842, 5: 2.8943, 6: 3.9060,
       7: 4.745, 8: 5.831, 9: 6.658}
KS = list(range(2, 10))
MC_N = 2_000_000
rng = np.random.default_rng(0)

# ---------------- observed cohort ------------------------------------------
GROK = "/Users/ruhaanrajadhyaksha/projects/grok"
EXCLUDE = ("dose", "transplant", "compiler", "gradtap", "boost", "lesion",
           "engram", "clock", "mirror", "twin", "shortcut", "lazy", "race",
           "degrok", "tasks")
def vanilla(cfg):
    return (cfg.get("p") == 113 and cfg.get("frac_train") == 0.3
            and cfg.get("d_mlp") == 512 and cfg.get("num_layers") == 1
            and cfg.get("weight_decay") == 1.0 and cfg.get("lr") == 1e-3
            and cfg.get("fn_name", "add") == "add"
            and cfg.get("embed_init", "normal") == "normal"
            and cfg.get("attn_init", "normal") == "normal"
            and not cfg.get("init_from", "") and cfg.get("table_scale", 0) == 0
            and cfg.get("grad_noise", 0) == 0 and cfg.get("loss_tilt", 0) == 0
            and cfg.get("loss_cvar", 0) == 0)

obs_sets = []
for cf in glob.glob(f"{GROK}/runs_*/**/config.json", recursive=True):
    if any(x in cf.lower() for x in EXCLUDE): continue
    try: cfg = json.load(open(cf))
    except Exception: continue
    if not vanilla(cfg): continue
    try: z = np.load(cf.replace("config.json", "spectra.npz"))
    except Exception: continue
    if z["test_acc"][-1] < 0.99: continue
    obs_sets.append(tuple(sorted(committee_from_coeffs(z["coeffs"][-1]))))
obs_K = Counter(len(s) for s in obs_sets)
n_obs = len(obs_sets)
pop_obs = Counter(k for s in obs_sets for k in s)
print(f"observed: n={n_obs}  K hist {dict(sorted(obs_K.items()))}")

# ---------------- coverage machinery ---------------------------------------
d = np.arange(1, P)
g = (1 - np.cos(2 * np.pi * np.outer(np.arange(1, NF + 1), d) / P)) \
    .astype(np.float32)                                   # (56, 112)
FOLD = np.minimum(np.arange(P), P - np.arange(P))         # v -> freq 0..56

def M_of(sets):
    """sets (N, K) int frequencies 1..56 -> M(S) (N,) float32."""
    acc = np.zeros((sets.shape[0], P - 1), np.float32)
    for j in range(sets.shape[1]):
        acc += g[sets[:, j] - 1]
    return acc.min(1)

def collision_free(sets):
    """True where NO pair sum/diff (mod p, folded) hits a third member."""
    N, K = sets.shape
    memb = np.zeros((N, NF + 1), bool)
    memb[np.arange(N)[:, None], sets] = True
    rows = np.arange(N)
    coll = np.zeros(N, bool)
    for a, b in itertools.combinations(range(K), 2):
        sa, sb = sets[:, a], sets[:, b]
        for v in (sa + sb, sa - sb):
            f = FOLD[v % P]
            coll |= memb[rows, f] & (f != sa) & (f != sb) & (f > 0)
    return ~coll

# ---------------- enumerate / sample every size class ----------------------
CACHE = f"{GROK}/runs_nudial/gibbs_cache.npz"
CHUNK = 200_000
Mvals, cf_flags, all_sets, mult = {}, {}, {}, {}
import os
if os.path.exists(CACHE):
    z = np.load(CACHE)
    for K in KS:
        Mvals[K] = z[f"M{K}"]; cf_flags[K] = z[f"cf{K}"]
        all_sets[K] = z[f"S{K}"]
        mult[K] = 1.0 if math.comb(NF, K) <= 4_000_000 \
            else math.comb(NF, K) / MC_N
    print("loaded cache")
for K in [] if os.path.exists(CACHE) else KS:
    total = math.comb(NF, K)
    ms, fl, st = [], [], []
    if total <= 4_000_000:                                # exact enumeration
        it = itertools.combinations(range(1, NF + 1), K)
        while True:
            block = list(itertools.islice(it, CHUNK))
            if not block: break
            sets = np.array(block, np.int16)
            ms.append(M_of(sets)); fl.append(collision_free(sets))
            st.append(sets.astype(np.int8))
        mult[K] = 1.0
        how = "exact"
    else:                                                 # uniform MC
        done = 0
        while done < MC_N:
            n = min(CHUNK, MC_N - done); done += n
            r = rng.random((n, NF), np.float32)
            sets = np.sort(r.argpartition(K - 1, 1)[:, :K] + 1).astype(np.int16)
            ms.append(M_of(sets)); fl.append(collision_free(sets))
            st.append(sets.astype(np.int8))
        mult[K] = total / MC_N
        how = f"MC x{MC_N}"
    Mvals[K] = np.concatenate(ms)
    cf_flags[K] = np.concatenate(fl)
    all_sets[K] = np.concatenate(st)
    print(f"K={K}: {how:11s} n={len(Mvals[K]):>9,} mult={mult[K]:.3g} "
          f"cf frac {cf_flags[K].mean():.4f} "
          f"M max {Mvals[K].max():.4f} (table {M_K[K]:.4f})", flush=True)

if not os.path.exists(CACHE):
    np.savez(CACHE, **{f"M{K}": Mvals[K] for K in KS},
             **{f"cf{K}": cf_flags[K] for K in KS},
             **{f"S{K}": all_sets[K] for K in KS})

# ---------------- histograms over gamma (per K, per model) -----------------
def gamma_hists(nu):
    gmax = max(M_K[K] * K ** (-nu / 2) for K in KS)
    edges = np.linspace(0, gmax * 1.001, 3201)
    cen = 0.5 * (edges[:-1] + edges[1:])
    hb, hc = {}, {}
    for K in KS:
        gam = Mvals[K] * K ** (-nu / 2)
        hb[K] = np.histogram(gam, edges)[0] * mult[K]
        hc[K] = np.histogram(gam[cf_flags[K]], edges)[0] * mult[K]
    return cen, gmax, hb, hc

def fit_gibbs(cen, gmax, hists, Ts):
    """log Z_K(T) via binned log-sum-exp; return (bestT, LL, P(K) at best)."""
    best = (None, -np.inf, None)
    for T in Ts:
        w = np.exp((cen - gmax) / T)
        zk = np.array([float(hists[K] @ w) for K in KS])
        if zk.sum() <= 0: continue
        pk = zk / zk.sum()
        ll = sum(obs_K.get(K, 0) * math.log(max(pk[i], 1e-300))
                 for i, K in enumerate(KS))
        if ll > best[1]: best = (T, ll, pk)
    return best

def fit_power(cen, gmax, hists, betas):
    best = (None, -np.inf, None)
    lg = np.log(np.maximum(cen / gmax, 1e-30))
    for b in betas:
        w = np.exp(b * lg)
        zk = np.array([float(hists[K] @ w) for K in KS])
        if zk.sum() <= 0: continue
        pk = zk / zk.sum()
        ll = sum(obs_K.get(K, 0) * math.log(max(pk[i], 1e-300))
                 for i, K in enumerate(KS))
        if ll > best[1]: best = (b, ll, pk)
    return best

def show(name, param, ll, pk):
    exp_cnt = " ".join(f"{K}:{pk[i]*n_obs:5.1f}" for i, K in enumerate(KS))
    print(f" {name:28s} param={param:.3g}  LL={ll:8.2f}  "
          f"mode={KS[int(np.argmax(pk))]}  E[n_K]= {exp_cnt}")

Ts = np.logspace(-4.5, -0.5, 240)
betas = np.logspace(0.3, 4.2, 240)
cen, gmax, hb, hc = gamma_hists(NU)

# saturated + baselines for reference
p_emp = np.array([obs_K.get(K, 0) / n_obs for K in KS])
ll_sat = sum(obs_K[K] * math.log(obs_K[K] / n_obs) for K in obs_K)
print(f"\nnu={NU}  observed {dict(sorted(obs_K.items()))}   "
      f"saturated LL={ll_sat:.2f}")

# Model A: Boltzmann over K on best-set gamma
gstar = np.array([M_K[K] * K ** (-NU / 2) for K in KS])
bestA = (None, -np.inf, None)
for T in Ts:
    pk = np.exp((gstar - gstar.max()) / T); pk /= pk.sum()
    ll = sum(obs_K.get(K, 0) * math.log(max(pk[i], 1e-300))
             for i, K in enumerate(KS))
    if ll > bestA[1]: bestA = (T, ll, pk)
show("A  Boltzmann over K", *bestA)

TB, llB, pkB = fit_gibbs(cen, gmax, hb, Ts);   show("B  Gibbs over sets", TB, llB, pkB)
TC, llC, pkC = fit_gibbs(cen, gmax, hc, Ts);   show("C  Gibbs + veto", TC, llC, pkC)
b2, ll2, pk2 = fit_power(cen, gmax, hb, betas); show("B2 power-law sets", b2, ll2, pk2)
c2, llc2, pkc2 = fit_power(cen, gmax, hc, betas); show("C2 power-law + veto", c2, llc2, pkc2)

# nu sensitivity for the best structural model
print("\nnu sensitivity (model C):")
for nu in (5.0, 5.5, 6.0, 6.5, 7.0):
    cenx, gmx, _, hcx = gamma_hists(nu)
    t, ll, pk = fit_gibbs(cenx, gmx, hcx, Ts)
    print(f"  nu={nu}: T*={t:.3g} LL={ll:.2f} mode={KS[int(np.argmax(pk))]} "
          f"E[n_K]=" + " ".join(f"{K}:{pk[i]*n_obs:.1f}" for i, K in enumerate(KS)))

# ---------------- set-level checks at the winning model --------------------
winner = max([("C", TC, llC, hc, "gibbs"), ("C2", c2, llc2, hc, "power"),
              ("B", TB, llB, hb, "gibbs"), ("B2", b2, ll2, hb, "power")],
             key=lambda r: r[2])
name, par, llw, hw, kind = winner
use_cf = name.startswith("C")
print(f"\n== set-level checks under winning model {name} (param {par:.3g}) ==")

pop_pred = np.zeros(NF + 1)
Ztot = 0.0
qtiles = []
for K in KS:
    gam = Mvals[K] * K ** (-NU / 2)
    ok = cf_flags[K] if use_cf else np.ones(len(gam), bool)
    w = (np.exp((gam - gmax) / par) if kind == "gibbs"
         else np.maximum(gam / gmax, 1e-30) ** par)
    w = w * ok
    Ztot += w.sum() * mult[K]
    sets = all_sets[K].astype(np.int64)
    np.add.at(pop_pred, sets.ravel(),
              np.repeat(w * mult[K], sets.shape[1]))
    for s in obs_sets:                      # quantile of observed sets' gamma
        if len(s) == K:
            gs = float(g[[k - 1 for k in s]].sum(0).min()) * K ** (-NU / 2)
            qtiles.append(float((gam < gs).mean()))
pop_pred = pop_pred[1:] / Ztot              # P(k in S) per frequency
obs_p = np.array([pop_obs.get(k, 0) / n_obs for k in range(1, NF + 1)])
from scipy.stats import spearmanr
rho = spearmanr(pop_pred, obs_p).statistic
print(f" per-frequency inclusion prob: Spearman(model, observed) = {rho:.3f}")
print(f" observed sets' gamma quantile in their size class: "
      f"median {np.median(qtiles):.4f}  min {np.min(qtiles):.4f}")
top = np.argsort(pop_pred)[::-1][:8] + 1
print(f" model's 8 most-included freqs: {top.tolist()}")
print(f" cohort's 8 most-popular freqs: "
      f"{[k for k, _ in pop_obs.most_common(8)]}")

# ---------------- money check: same beta, second prediction -----------------
# Within a size class K, model C2 implies gamma is drawn with density
# ~ gamma^beta over the collision-free class. Map gamma -> its quantile u
# in the class; compare the model-implied distribution of u with the 69
# observed sets' quantiles. ONE parameter (beta, fit on counts only) must
# now also reproduce HOW GOOD the chosen sets are. Never fit on this.
print(f"\n== within-class quality distribution (beta from K-counts only) ==")
beta = c2
u_obs, u_pred_draws = [], []
for K in (3, 4, 5, 6):
    gam = np.sort(Mvals[K][cf_flags[K]] * K ** (-NU / 2))
    w = (gam / gam[-1]) ** beta
    cdf_w = np.cumsum(w) / w.sum()                # model CDF over rank-quantile
    uu = np.arange(1, len(gam) + 1) / len(gam)    # unweighted quantile of rank
    # sample model-implied quantiles for this class, in class-count proportion
    n_draw = 4000 * obs_K.get(K, 0)
    if n_draw:
        r = rng.random(n_draw)
        u_pred_draws.append(uu[np.searchsorted(cdf_w, r)])
    for s in obs_sets:
        if len(s) != K: continue
        gs = float(g[[k - 1 for k in s]].sum(0).min()) * K ** (-NU / 2)
        u_obs.append(float(np.searchsorted(gam, gs) / len(gam)))
u_obs = np.array(u_obs); u_pred = np.concatenate(u_pred_draws)
from scipy.stats import ks_2samp
ks = ks_2samp(u_obs, u_pred)
print(f" observed quantiles: median {np.median(u_obs):.3f}  "
      f"[10%,90%] = [{np.quantile(u_obs,.1):.3f},{np.quantile(u_obs,.9):.3f}]")
print(f" model-implied     : median {np.median(u_pred):.3f}  "
      f"[10%,90%] = [{np.quantile(u_pred,.1):.3f},{np.quantile(u_pred,.9):.3f}]")
print(f" KS 2-sample: D={ks.statistic:.3f}  p={ks.pvalue:.3f}  (n_obs={len(u_obs)})")

# ---------------- parametric bootstrap for the K-histogram fit --------------
def refit_power(counts, hists, bgrid, cen, gmax):
    lg = np.log(np.maximum(cen / gmax, 1e-30))
    best = -np.inf
    for b in bgrid:
        w = np.exp(b * lg)
        zk = np.array([float(hists[K] @ w) for K in KS])
        pk = zk / zk.sum()
        ll = float(np.sum(counts * np.log(np.maximum(pk, 1e-300))))
        if ll > best: best = ll
    return best

def sat_ll(counts):
    n = counts.sum()
    nz = counts[counts > 0]
    return float(np.sum(nz * np.log(nz / n)))

bgrid = np.logspace(0.8, 2.5, 60)
counts_obs = np.array([obs_K.get(K, 0) for K in KS], float)
G2_obs = 2 * (sat_ll(counts_obs) - refit_power(counts_obs, hc, bgrid, cen, gmax))
G2_sim = []
for _ in range(1500):
    sim = rng.multinomial(n_obs, pkc2).astype(float)
    G2_sim.append(2 * (sat_ll(sim) - refit_power(sim, hc, bgrid, cen, gmax)))
G2_sim = np.array(G2_sim)
print(f"\n== parametric bootstrap, model C2 on K histogram ==")
print(f" deviance G2 obs {G2_obs:.2f}  sim median {np.median(G2_sim):.2f}  "
      f"p = {np.mean(G2_sim >= G2_obs):.3f}")
