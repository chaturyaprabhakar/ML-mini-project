"""Walk-forward long/flat backtest (direction task). Expanding window, retrained every `--block` days,
1-day embargo, per-day mean probability -> long if > 0.5 else flat, transaction costs, compared with
buy-and-hold on the same days. Significance: block-bootstrap CI of excess return + permutation test."""
import argparse, os, warnings
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
import pipeline as P, models as M

warnings.filterwarnings("ignore")
RES = os.path.join(os.path.dirname(__file__), "..", "results")
WF = {
    "LogReg-Newton (paper)": lambda: M.pipe("paper", M.NewtonLR(l2=1e-6)),
    "RandomForest (new)": lambda: M.pipe("ext", RandomForestClassifier(300, min_samples_leaf=10, max_features="sqrt",
                                                                       random_state=0, n_jobs=-1)),
    "HistGradBoost (new)": lambda: M.pipe("ext", HistGradientBoostingClassifier(max_depth=3, learning_rate=.05, max_iter=150,
                                                                                l2_regularization=1.0, random_state=0)),
}


def walk_forward(df, make, start, block, gap=1):
    u = np.array(sorted(df["day"].unique())); out = []
    for k in range(int(start * len(u)), len(u), block):
        tr = df[df["day"].isin(u[:max(k - gap, 1)])]; te = df[df["day"].isin(u[k:k + block])]
        if len(tr) < 200 or tr["direction"].nunique() < 2 or te.empty: continue
        p = make().fit(tr, tr["direction"].values).predict_proba(te)[:, 1]
        out.append(pd.DataFrame({"day": te["day"].values, "p": p}))
    pr = pd.concat(out).groupby("day")["p"].mean()
    return pd.DataFrame({"p": pr, "ret": df.groupby("day")["ret_next"].first().loc[pr.index]})


def stats(x):
    eq = np.cumprod(1 + x)
    return (eq[-1] - 1, x.mean() / x.std(ddof=1) * np.sqrt(252) if x.std() > 0 else np.nan,
            (eq / np.maximum.accumulate(eq) - 1).min(), eq)


def evaluate(d, cost, rng, nboot=2000, nperm=2000, blk=5):
    pos = (d["p"] > .5).astype(float).values; r = d["ret"].values
    net = pos * r - np.abs(np.diff(np.r_[0., pos])) * cost          # cost on every entry/exit
    bh = r.copy(); bh[0] -= cost                                     # buy & hold on the same days
    ex = net - bh; n = len(ex); nb = int(np.ceil(n / blk))
    boots = [ex[((rng.randint(0, n, nb)[:, None] + np.arange(blk)).ravel()[:n]) % n].mean() for _ in range(nboot)]
    obs = (pos * r).mean(); perm = np.array([(rng.permutation(pos) * r).mean() for _ in range(nperm)])
    (tr_, sh, dd, eq), (btr, bsh, bdd, beq) = stats(net), stats(bh)
    return dict(n_days=n, day_acc=((pos == 1) == (r > 0)).mean(), exposure=pos.mean(),
                n_trades=int(np.abs(np.diff(np.r_[0., pos])).sum()), total_ret=tr_, bh_total_ret=btr,
                sharpe=sh, bh_sharpe=bsh, max_dd=dd, bh_max_dd=bdd, excess_mean_daily=ex.mean(),
                excess_ci_lo=np.percentile(boots, 2.5), excess_ci_hi=np.percentile(boots, 97.5),
                perm_p=(1 + (perm >= obs).sum()) / (nperm + 1)), eq, beq


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stocks", nargs="+", default=list(P.STOCKS))
    ap.add_argument("--start", type=float, default=.5, help="fraction of days used as the initial training window")
    ap.add_argument("--block", type=int, default=10, help="retrain every N days")
    ap.add_argument("--cost_bps", type=float, default=10, help="cost per entry/exit in basis points")
    ap.add_argument("--nn", action="store_true", help="also backtest the paper NN (slow)")
    a = ap.parse_args()
    if a.nn: WF["NN 3x256 (paper)"] = lambda: M.pipe("paper", M.SeedAvgNN(3, 30))
    os.makedirs(RES, exist_ok=True); posts = P.load_posts(); rng = np.random.RandomState(0)
    rows, fig = [], plt.subplots(2, 2, figsize=(12, 7))
    for ax, s in zip(fig[1].ravel(), a.stocks):
        df = P.build_stock(posts, s); print(s, len(df), "posts", flush=True)
        for name, make in WF.items():
            d = walk_forward(df, make, a.start, a.block)
            res, eq, beq = evaluate(d, a.cost_bps / 1e4, rng)
            rows.append(dict(stock=s, model=name, **res)); ax.plot(eq, label=name)
        ax.plot(beq, "k--", label="Buy & hold (same days)"); ax.set_title(f"{s} walk-forward ({a.cost_bps:.0f} bps cost)")
        ax.set_xlabel("traded day"); ax.set_ylabel("growth of 1"); ax.legend(fontsize=7)
    plt.tight_layout(); plt.savefig(f"{RES}/fig_backtest.png", dpi=130)
    out = pd.DataFrame(rows); out.to_csv(f"{RES}/backtest_summary.csv", index=False)
    print(out.round(3).to_string())