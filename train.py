import warnings; warnings.filterwarnings("ignore")
import json, joblib, numpy as np, pandas as pd
from scipy.optimize import nnls
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.model_selection import KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, TargetEncoder
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from features import load_raw, engineer, city_of

CAT = ["region", "city", "locality", "type", "status", "age", "city_status", "region_bhk"]
NUM = ["bhk", "area", "log_area", "area_per_bhk", "is_ready"]

# ---------- data cleaning ----------
raw = load_raw()
n0 = len(raw)
raw = raw.drop_duplicates(["bhk", "type", "locality", "area", "price_inr", "region", "status", "age"])  # repeat listings leak across splits
raw["ppsf"] = raw.price_inr / raw.area
lo, hi = raw.ppsf.quantile([0.005, 0.995])
raw = raw[raw.ppsf.between(lo, hi) & raw.area.between(150, 6000)].reset_index(drop=True)
print(f"rows {n0} -> {len(raw)} after de-dup + outlier trimming (₹{lo:,.0f}-{hi:,.0f}/sqft)")
raw["city"] = raw.region.map(lambda r: city_of(" ".join(str(r).split()).title()))
y = np.log(raw.price_inr.values)

X = engineer(raw)
# rare projects -> "Other" so the app can handle unseen builders/projects
cnt = X.locality.value_counts()
KEEP_LOC = set(cnt[cnt >= 3].index)
X["locality"] = X.locality.where(X.locality.isin(KEEP_LOC), "Other")
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=X.city)

def rmse(a, b): return float(np.sqrt(mean_squared_error(a, b)))

def linear_prep():
    return ColumnTransformer([
        ("te", TargetEncoder(random_state=0), ["region", "locality", "region_bhk", "city_status"]),
        ("oh", OneHotEncoder(handle_unknown="ignore", min_frequency=20), ["city", "type", "status", "age"]),
        ("num", StandardScaler(), NUM)])

class Cat(CatBoostRegressor):  # thin wrapper so the pipeline can pass cat feature names
    pass

def mk_cat(seed, depth):
    return CatBoostRegressor(iterations=2500, learning_rate=0.06, depth=depth, l2_leaf_reg=3, random_seed=seed,
                             verbose=0, cat_features=CAT, thread_count=-1, early_stopping_rounds=None)

models = {
    "Ridge": Pipeline([("p", linear_prep()), ("m", RidgeCV(alphas=np.logspace(-2, 3, 20)))]),
    "LightGBM": Pipeline([("p", ColumnTransformer([("te", TargetEncoder(random_state=0), ["region", "locality", "region_bhk", "city_status"]),
                                                   ("oh", OneHotEncoder(handle_unknown="ignore", min_frequency=20), ["city", "type", "status", "age"]),
                                                   ("n", "passthrough", NUM)])),
                          ("m", LGBMRegressor(n_estimators=1500, learning_rate=0.03, num_leaves=63, min_child_samples=15,
                                              subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=1, verbose=-1, n_jobs=-1))]),
    "XGBoost": Pipeline([("p", ColumnTransformer([("te", TargetEncoder(random_state=0), ["region", "locality", "region_bhk", "city_status"]),
                                                  ("oh", OneHotEncoder(handle_unknown="ignore", min_frequency=20), ["city", "type", "status", "age"]),
                                                  ("n", "passthrough", NUM)])),
                         ("m", XGBRegressor(n_estimators=1500, learning_rate=0.03, max_depth=7, subsample=0.8, colsample_bytree=0.7,
                                            min_child_weight=3, reg_lambda=2, n_jobs=-1, random_state=0))]),
    "CatBoost-d6": mk_cat(1, 6),
    "CatBoost-d8": mk_cat(2, 8),
}

def fit(m, Xa, ya): return m.fit(Xa, ya) if isinstance(m, Pipeline) else m.fit(Xa[CAT + NUM], ya)
def pred(m, Xa): return m.predict(Xa) if isinstance(m, Pipeline) else m.predict(Xa[CAT + NUM])
from copy import deepcopy as clone  # CatBoost is not sklearn-clonable

# ---------- 5-fold CV (out-of-fold) on the training split ----------
kf = KFold(5, shuffle=True, random_state=42)
oof = {k: np.zeros(len(ytr)) for k in models}
for k, m in models.items():
    for tr, va in kf.split(Xtr):
        mm = fit(clone(m), Xtr.iloc[tr], ytr[tr]); oof[k][va] = pred(mm, Xtr.iloc[va])
    print(f"{k:12s} CV RMSE(log)={rmse(ytr, oof[k]):.4f}  R2(log)={r2_score(ytr, oof[k]):.4f}", flush=True)
names = list(models); A = np.column_stack([oof[k] for k in names])
w, _ = nnls(A, ytr); w /= w.sum()
print("blend weights", {k: round(float(x), 3) for k, x in zip(names, w)}, " CV RMSE", round(rmse(ytr, A @ w), 4))

# ---------- hold-out ----------
fitted = {k: fit(clone(m), Xtr, ytr) for k, m in models.items()}
P = {k: pred(f, Xte) for k, f in fitted.items()}
blend = np.column_stack([P[k] for k in names]) @ w
act, pr = np.exp(yte), np.exp(blend)
metrics = {"n_listings": int(len(raw)), "holdout_n": int(len(yte)),
           "holdout_rmse_log": rmse(yte, blend), "holdout_r2_log": float(r2_score(yte, blend)),
           "holdout_r2_price": float(r2_score(act, pr)), "holdout_mae_inr": float(mean_absolute_error(act, pr)),
           "holdout_mape_pct": float(np.mean(np.abs(act - pr) / act) * 100),
           "median_ape_pct": float(np.median(np.abs(act - pr) / act) * 100),
           "within_10pct": float(np.mean(np.abs(act - pr) / act < 0.10) * 100),
           "within_20pct": float(np.mean(np.abs(act - pr) / act < 0.20) * 100),
           "cv_rmse_log": rmse(ytr, A @ w), "weights": {k: float(x) for k, x in zip(names, w)},
           "single_model_holdout_log": {k: rmse(yte, P[k]) for k in names}}
by_city = {}
for c in sorted(Xte.city.unique()):
    mk = (Xte.city == c).values
    by_city[c] = {"n": int(mk.sum()), "r2": float(r2_score(yte[mk], blend[mk])),
                  "mape_pct": float(np.mean(np.abs(act[mk] - pr[mk]) / act[mk]) * 100)}
metrics["by_city"] = by_city
print(json.dumps(metrics, indent=1))

# ---------- final refit on all data ----------
final = {k: fit(clone(m), X, y) for k, m in models.items()}
meta = {"regions": {c: sorted(X[X.city == c].region.unique().tolist()) for c in sorted(X.city.unique())},
        "projects": {r: sorted(g.locality[g.locality != "Other"].value_counts().index.tolist()[:300])
                     for r, g in X.groupby("region")},
        "median_ppsf": raw.groupby("region").ppsf.median().round(0).to_dict()}
joblib.dump({"models": final, "weights": dict(zip(names, map(float, w))), "keep_loc": KEEP_LOC, "cat": CAT, "num": NUM, "meta": meta},
            "models/model.joblib", compress=3)
json.dump(metrics, open("models/metrics.json", "w"), indent=1)
