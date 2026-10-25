"""Data loading, cleaning, sentiment, features and labels for the WSB project."""
import os, re, shutil
import numpy as np, pandas as pd

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
STOCKS = {"GME": r"\b(?:gme|gamestop)\b", "AMC": r"\bamc\b",
          "TSLA": r"\b(?:tsla|tesla)\b", "FB": r"\b(?:fb|facebook)\b"}
YF = {"GME": "GME", "AMC": "AMC", "TSLA": "TSLA", "FB": "META"}  # FB is META on Yahoo now
PAPER = ["post_score", "total_comms_num", "len_of_post",
         "positive_sentiment", "neutral_sentiment", "negative_sentiment"]  # + bias (added by model)
EXTRA = ["log_score", "log_comms", "n_exclaim", "caps_ratio", "hour", "dow",
         "daily_mentions", "ret_prev", "vol5"]


def _clean(t):
    t = re.sub(r"http\S+|www\.\S+", " ", t.lower())
    t = re.sub(r"@\w+", " ", t)
    t = re.sub(r"[^a-z0-9\s]", " ", t)
    t = re.sub(r"\b\w\b", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def load_posts():
    cache = os.path.join(DATA, "posts_processed.pkl")
    if os.path.exists(cache):
        return pd.read_pickle(cache)
    os.makedirs(DATA, exist_ok=True)
    csv = os.path.join(DATA, "reddit_wsb.csv")
    if not os.path.exists(csv):
        import kagglehub
        p = kagglehub.dataset_download("gpreda/reddit-wallstreetsbets-posts")
        shutil.copy(os.path.join(p, "reddit_wsb.csv"), csv)
    df = pd.read_csv(csv)
    df = df.dropna(subset=["title"]).copy()          # body is mostly empty, so only title is required
    df["body"] = df["body"].fillna("")
    ts = pd.to_datetime(df["timestamp"], errors="coerce")
    if ts.isna().all():
        ts = pd.to_datetime(df["created"], unit="s")
    df["ts"] = ts
    df = df.dropna(subset=["ts"])
    df["raw"] = (df["title"] + " " + df["body"]).str.lower()
    df["text"] = (df["title"] + " " + df["body"]).map(_clean)
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    sia = SentimentIntensityAnalyzer()
    sc = df["text"].map(sia.polarity_scores)
    df["positive_sentiment"] = sc.map(lambda s: s["pos"])
    df["neutral_sentiment"] = sc.map(lambda s: s["neu"])
    df["negative_sentiment"] = sc.map(lambda s: s["neg"])
    df["post_score"] = df["score"]
    df["total_comms_num"] = df["comms_num"]
    df["len_of_post"] = df["text"].str.len()
    df["n_exclaim"] = df["title"].str.count("!")
    df["caps_ratio"] = df["title"].map(lambda s: sum(c.isupper() for c in s) / max(len(s), 1))
    df = df[["title", "ts", "raw", "text", "n_exclaim", "caps_ratio", "post_score", "total_comms_num",
             "len_of_post", "positive_sentiment", "neutral_sentiment", "negative_sentiment"]]
    df.to_pickle(cache)
    return df


def load_prices(stock):
    f = os.path.join(DATA, f"prices_{stock}.csv")
    if not os.path.exists(f):
        import yfinance as yf
        d = yf.download(YF[stock], start="2020-08-01", end="2021-09-15", auto_adjust=False, progress=False)
        if isinstance(d.columns, pd.MultiIndex):
            d.columns = d.columns.get_level_values(0)
        d.to_csv(f)
    d = pd.read_csv(f, index_col=0, parse_dates=True).sort_index()
    d.index = d.index.tz_localize(None).normalize() if d.index.tz else d.index.normalize()
    return d


def mention_counts(posts):
    return {s: int(posts["raw"].str.contains(r, regex=True).sum()) for s, r in STOCKS.items()}


def build_stock(posts, stock):
    """One row per post mentioning `stock`. Label: open(t1) vs open(t0) where t0 is the first
    trading day on/after the post date and t1 the next trading day (paper's P_k < P_{k+1})."""
    px = load_prices(stock)["Open"].dropna()
    r = px.pct_change()
    vol5 = r.rolling(5).std()
    m = posts[posts["raw"].str.contains(STOCKS[stock], regex=True)].copy()
    day = m["ts"].dt.normalize()
    i0 = px.index.searchsorted(day.values)
    ok = (i0 >= 6) & (i0 + 1 < len(px))
    m, i0 = m[ok].copy(), i0[ok]
    # group key = the TRADING day t0 the post maps to (weekend/holiday posts share Monday's label, so they
    # must share its group too, otherwise same-label posts leak across train/test)
    m["day"] = px.index[i0].values
    m["daily_mentions"] = m.groupby("day")["day"].transform("size").values
    o0, o1 = px.values[i0], px.values[i0 + 1]
    m["ret_next"] = o1 / o0 - 1
    m["absret"] = m["ret_next"].abs()
    m["direction"] = (m["ret_next"] > 0).astype(int)
    m["ret_prev"] = r.values[i0]       # known at the open of t0
    m["vol5"] = vol5.values[i0]
    m["log_score"] = np.log1p(m["post_score"].clip(lower=0))
    m["log_comms"] = np.log1p(m["total_comms_num"].clip(lower=0))
    m["hour"] = m["ts"].dt.hour
    m["dow"] = m["ts"].dt.dayofweek
    return m.sort_values("ts").reset_index(drop=True)


def split_idx(days, mode, seed=0):
    """80/10/10 split -> positional (train, val, test) indices.
    'random'  : paper-style shuffle of POSTS (leaky: same-day posts, which share one label, land in train AND test)
    'grouped' : shuffle whole DAYS - every post of a day goes to the same part (no same-day leakage)
    'chrono'  : DAYS in time order, test = latest days, 1-day embargo before test (strictest, no look-ahead)"""
    d = pd.to_datetime(pd.Series(np.asarray(days))).values
    if mode == "random":
        idx = np.random.RandomState(seed).permutation(len(d))
        a, b = int(.8 * len(d)), int(.9 * len(d))
        return np.sort(idx[:a]), np.sort(idx[a:b]), np.sort(idx[b:])
    u = np.unique(d)
    if mode == "grouped":
        u = np.random.RandomState(seed).permutation(u)
    a, b = int(.8 * len(u)), int(.9 * len(u))
    te_days = u[b + 1:] if mode == "chrono" else u[b:]          # embargo: skip first test day
    pick = lambda ds: np.where(np.isin(d, ds))[0]
    return pick(u[:a]), pick(u[a:b]), pick(te_days)


def make_cv(days, mode, k=3, gap=1):
    """Leakage-free CV folds (list of (train_pos, valid_pos)) split by DAY.
    chrono: forward-chaining over day blocks with a `gap`-day embargo; otherwise GroupKFold by day."""
    d = pd.to_datetime(pd.Series(np.asarray(days))).values
    if mode == "chrono":
        u = np.unique(d); e = np.linspace(0, len(u), k + 2).astype(int)
        return [(np.where(np.isin(d, u[:max(e[i] - gap, 1)]))[0], np.where(np.isin(d, u[e[i]:e[i + 1]]))[0])
                for i in range(1, k + 1)]
    from sklearn.model_selection import GroupKFold
    return list(GroupKFold(k).split(np.zeros(len(d)), groups=d.astype("int64")))