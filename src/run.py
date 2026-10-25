"""Replicate the paper, add new model families + tuning + robust evaluation, make tables/plots."""
import argparse, os, warnings
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV
import pipeline as P, models as M

warnings.filterwarnings("ignore")
RES = os.path.join(os.path.dirname(__file__), "..", "results")


def score_of(m, X):
    return m.predict_proba(X)[:, 1] if hasattr(m, "predict_proba") else m.decision_function(X)


def metrics(m, X, y, rng):
    pred = m.predict(X); ok = (pred == y).astype(float)
    ci = [ok[rng.randint(0, len(ok), len(ok))].mean() for _ in range(300)]
    try: auc = roc_auc_score(y, score_of(m, X))
    except Exception: auc = np.nan
    return dict(acc=ok.mean(), ci_lo=np.percentile(ci, 2.5), ci_hi=np.percentile(ci, 97.5),
                bal_acc=balanced_accuracy_score(y, pred), f1=f1_score(y, pred, zero_division=0), auc=auc)


def run_one(df, stock, task, split, a, rows, hist):
    tr, va, te = P.split_idx(df["day"], split)
    thr = df["absret"].iloc[tr].median()                       # "significant shift" threshold from train only
    y = (df["direction"] if task == "direction" else (df["absret"] > thr).astype(int)).values
    fit_idx = np.concatenate([tr, va])                          # tune with time-ordered CV on train+val
    Xf, yf, Xt, yt = df.iloc[fit_idx], y[fit_idx], df.iloc[te], y[te]
    rng = np.random.RandomState(0)
    cv = P.make_cv(df["day"].iloc[fit_idx], split)        # day-based, leakage-free tuning folds
    tag = dict(stock=stock, task=task, split=split, n_train=len(fit_idx), n_test=len(te))

    def add(name, m, family, cvs=np.nan):
        rows.append({**tag, "model": name, "family": family, "cv_auc": cvs, **metrics(m, Xt, yt, rng)})

    # ---- baselines
    maj = np.bincount(yf).argmax()
    ok = (yt == maj).astype(float)
    rows.append({**tag, "model": "Majority baseline", "family": "baseline", "acc": ok.mean(),
                 "ci_lo": np.nan, "ci_hi": np.nan, "bal_acc": .5, "f1": np.nan, "auc": .5, "cv_auc": np.nan})
    rnd = np.random.RandomState(1).randint(0, 2, len(yt))
    rows.append({**tag, "model": "Random baseline", "family": "baseline", "acc": (rnd == yt).mean(),
                 "ci_lo": np.nan, "ci_hi": np.nan, "bal_acc": balanced_accuracy_score(yt, rnd), "f1": f1_score(yt, rnd, zero_division=0),
                 "auc": .5, "cv_auc": np.nan})
    def add_nn(name, feats, family):
        m = M.pipe(feats, M.SeedAvgNN(a.seeds, a.epochs)).fit(Xf, yf)
        accs = [(s.predict(m.named_steps["prep"].transform(Xt)) == yt).mean() for s in m.named_steps["clf"].models_]
        add(name, m, family)                                    # metrics of the seed-ensemble ...
        rows[-1].update(acc_ens=rows[-1]["acc"], acc=float(np.mean(accs)), acc_std=float(np.std(accs)),
                        ci_lo=np.nan, ci_hi=np.nan)             # ... but `acc` = mean single-net accuracy over seeds
    # ---- paper replication (6 features + bias, fixed settings)
    lr = M.pipe("paper", M.NewtonLR(l2=1e-6)).fit(Xf, yf); add("LogReg-Newton (paper)", lr, "paper")
    hist[(stock, task, split)] = lr.named_steps["clf"].history_
    add_nn("NN 3x256 (paper)", "paper", "paper")
    # ---- ablation: new algorithms on the paper's 6 features only
    for n, k in [("RandomForest", "RandomForest"), ("HistGradBoost", "HistGradBoost")]:
        add(f"{n} (paper feats)", M.pipe("paper", clone(M.TUNED[k][0])).fit(Xf, yf), "ablation")
    # ---- improved: extended features + tuned models
    add_nn("NN 3x256 (ext feats)", "ext", "improved")
    fitted, cands = {}, {}
    for name, (clf, grid) in M.TUNED.items():
        s = RandomizedSearchCV(M.pipe("ext", clf), grid, n_iter=a.n_iter, cv=cv, scoring="roc_auc",
                               random_state=0, error_score=np.nan, n_jobs=-1).fit(Xf, yf)
        fitted[name] = s; add(name, s.best_estimator_, "improved", s.best_score_); cands[name] = s.best_score_
    # ---- model selection WITHOUT looking at test data
    best = max(cands, key=lambda k: -1 if np.isnan(cands[k]) else cands[k])
    pick = fitted[best].best_estimator_
    add(f"BestByCV [{best}]", pick, "selected", cands[best])


def plots(df, hist, posts):
    cnt = P.mention_counts(posts)
    plt.figure(figsize=(5, 3)); plt.bar(cnt.keys(), cnt.values(), color="c"); plt.title("Mentions per stock")
    plt.tight_layout(); plt.savefig(f"{RES}/fig_mentions.png", dpi=130); plt.close()
    sp0 = next(iter(hist))[2]
    plt.figure(figsize=(6, 4))
    for (s, t, sp), h in hist.items():
        if t == "direction" and sp == sp0: plt.semilogy(range(1, len(h) + 1), np.maximum(h, 1e-12), "o-", label=s)
    plt.xlabel("Newton iteration"); plt.ylabel("||delta w||"); plt.title("Newton convergence"); plt.legend()
    plt.tight_layout(); plt.savefig(f"{RES}/fig_convergence.png", dpi=130); plt.close()
    for (t, sp), g in df.groupby(["task", "split"]):
        pv = g.pivot_table(index="model", columns="stock", values="acc")
        pv = pv.loc[g.drop_duplicates("model").model.values]
        pv["MEAN"] = pv.mean(axis=1)
        pv.plot.bar(figsize=(13, 5), width=.85); plt.ylim(.3, 1); plt.ylabel("test accuracy")
        plt.title(f"Test accuracy - task={t}, split={sp}"); plt.xticks(rotation=40, ha="right")
        plt.tight_layout(); plt.savefig(f"{RES}/fig_acc_{t}_{sp}.png", dpi=130); plt.close()
    # improvement of the CV-selected model over the better of the two paper models
    rows = []
    for k, g in df.groupby(["stock", "task", "split"]):
        pap = g[g.family == "paper"].acc.max(); sel = g[g.family == "selected"].acc.iloc[0]
        rows.append(dict(zip(["stock", "task", "split"], k), paper_best=pap, selected=sel, gain=sel - pap))
    imp = pd.DataFrame(rows); imp.to_csv(f"{RES}/improvement.csv", index=False)
    pv = imp.pivot_table(index="stock", columns=["task", "split"], values="gain")
    pv.plot.bar(figsize=(9, 4)); plt.axhline(0, c="k", lw=.8); plt.ylabel("accuracy gain vs best paper model")
    plt.title("CV-selected model minus best paper model (test set)"); plt.tight_layout()
    plt.savefig(f"{RES}/fig_improvement.png", dpi=130); plt.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stocks", nargs="+", default=list(P.STOCKS))
    ap.add_argument("--tasks", nargs="+", default=["direction", "volatility"])
    ap.add_argument("--splits", nargs="+", default=["grouped", "chrono"],
                    help="grouped/chrono = split by day (leak-free); random = paper-style post shuffle (leaky, for comparison)")
    ap.add_argument("--n_iter", type=int, default=15, help="random-search iterations per model")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--seeds", type=int, default=5, help="random seeds for the neural nets")
    ap.add_argument("--fast", action="store_true", help="n_iter=4, epochs=10, seeds=2 (smoke test)")
    a = ap.parse_args()
    if a.fast: a.n_iter, a.epochs, a.seeds = 4, 10, 2
    os.makedirs(RES, exist_ok=True)
    posts = P.load_posts(); rows, hist = [], {}
    for s in a.stocks:
        df = P.build_stock(posts, s); print(f"{s}: {len(df)} labelled posts", flush=True)
        if len(df) < 200: print("  too few posts, skipped"); continue
        for t in a.tasks:
            for sp in a.splits:
                print(f"  {t}/{sp}", flush=True); run_one(df, s, t, sp, a, rows, hist)
    df = pd.DataFrame(rows); df.to_csv(f"{RES}/results.csv", index=False)
    plots(df, hist, posts)
    summ = df.pivot_table(index="model", columns=["task", "split"], values="acc").round(3)
    summ.to_csv(f"{RES}/summary_mean_accuracy.csv"); print(summ.to_string())
    print(pd.read_csv(f"{RES}/improvement.csv").round(3).to_string())