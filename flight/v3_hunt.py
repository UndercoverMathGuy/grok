"""v3 observable hunt: predict the law's residual from forecast-time
features. Leave-cell-out CV (GroupKFold over (cohort, cell)); models:
ridge, gradient boosting, small MLP. Importance = permutation on held-out
folds. Run twice: with and without config features (frac/wd/lr/width)."""
import json
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
from sklearn.pipeline import make_pipeline

rows = json.load(open("notes/v3_features.json"))
feat_names = [k for k in rows[0] if not k.startswith("_")]
# fill missing test-loss features with column median
X0 = np.array([[r.get(k, np.nan) for k in feat_names] for r in rows], float)
for j in range(X0.shape[1]):
    col = X0[:, j]
    col[np.isnan(col)] = np.nanmedian(col)
y = np.array([r["_y"] for r in rows])
ylog = np.array([r["_log_t_gen"] for r in rows])
groups = np.array([r["_cohort"] + "/" + r["_cell"] for r in rows])
cfgf = ["frac", "wd", "lr", "width"]
Xc = np.array([[r["_cfg"].get(k) or 0 for k in cfgf] for r in rows], float)
print(f"{len(rows)} runs, {len(set(groups))} cells, "
      f"residual var {y.var():.4f} (law leaves this), "
      f"log t_gen var {ylog.var():.4f}")

def run(X, names, label):
    gkf = GroupKFold(n_splits=min(8, len(set(groups))))
    models = {
        "ridge": make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 13))),
        "gbm": HistGradientBoostingRegressor(max_depth=3, max_iter=300,
                                             learning_rate=0.05,
                                             l2_regularization=1.0),
        "mlp": make_pipeline(StandardScaler(),
                             MLPRegressor(hidden_layer_sizes=(32, 16),
                                          alpha=1.0, max_iter=3000,
                                          random_state=0)),
    }
    print(f"\n== {label} ({len(names)} features)")
    best = None
    for mname, model in models.items():
        oof = np.full(len(y), np.nan)
        for tr, tefold in gkf.split(X, y, groups):
            model.fit(X[tr], y[tr])
            oof[tefold] = model.predict(X[tefold])
        r2 = 1 - np.nanvar(y - oof) / y.var()
        med = np.median(np.abs(y - oof))
        print(f"  {mname:6}: held-out-cell R2(resid) {r2:+.3f}   "
              f"median |miss| {med:.3f} dex (x{10**med:.2f})")
        if best is None or r2 > best[1]:
            best = (mname, r2, model)
    # permutation importance for the best model, on held-out folds
    mname, r2, model = best
    imp = np.zeros(X.shape[1])
    for tr, tefold in gkf.split(X, y, groups):
        model.fit(X[tr], y[tr])
        pi = permutation_importance(model, X[tefold], y[tefold],
                                    n_repeats=10, random_state=0)
        imp += pi.importances_mean
    order = np.argsort(imp)[::-1]
    print(f"  top features ({mname}, held-out permutation):")
    for i in order[:8]:
        print(f"    {names[i]:<18} {imp[i]:+.4f}")
    return r2

r2_obs = run(X0, feat_names, "observables only")
Xall = np.hstack([X0, Xc])
r2_all = run(Xall, feat_names + cfgf, "observables + config")
print(f"\nlaw baseline: residual is 100% unexplained by definition; "
      f"floor says ~96% is closable.")
print(f"closed by observables only: {max(r2_obs,0)*100:.0f}%   "
      f"with config: {max(r2_all,0)*100:.0f}%")
