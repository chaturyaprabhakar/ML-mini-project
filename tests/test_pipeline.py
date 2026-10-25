import os, sys, unittest
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import pipeline as P


def fake_prices():
    days = pd.bdate_range("2021-01-04", periods=40)            # Mon-Fri only
    return pd.DataFrame({"Open": 100 + np.arange(40) * 1.0 + (np.arange(40) % 3)}, index=days)


def fake_posts(stamps):
    n = len(stamps)
    return pd.DataFrame({"ts": pd.to_datetime(stamps), "raw": ["gme to the moon"] * n,
                         "post_score": np.arange(n) + 1, "total_comms_num": np.arange(n)})


class TextCleaningTests(unittest.TestCase):
    def test_strips_urls_handles_symbols_and_single_letters(self):
        out = P._clean("Check https://x.com/a @user GME!!! to the moon a b")
        self.assertEqual(out, "check gme to the moon")


class LabelTests(unittest.TestCase):
    def setUp(self):
        self._orig = P.load_prices
        P.load_prices = lambda stock: fake_prices()

    def tearDown(self):
        P.load_prices = self._orig

    def test_label_is_next_trading_open_vs_open_of_post_day(self):
        px = fake_prices()["Open"]; day = px.index[10]
        df = P.build_stock(fake_posts([day + pd.Timedelta(hours=15)]), "GME")
        self.assertAlmostEqual(df["ret_next"].iloc[0], px.iloc[11] / px.iloc[10] - 1)
        self.assertEqual(df["direction"].iloc[0], int(px.iloc[11] > px.iloc[10]))

    def test_weekend_posts_share_mondays_group_and_label(self):
        px = fake_prices()["Open"]; fri = px.index[9]             # Friday 15 Jan 2021
        sat, mon = fri + pd.Timedelta(days=1), fri + pd.Timedelta(days=3)
        df = P.build_stock(fake_posts([sat, mon]), "GME")
        self.assertEqual(df["day"].nunique(), 1)                   # Saturday and Monday -> same trading day
        self.assertEqual(df["day"].iloc[0], mon.normalize())
        self.assertEqual(df["direction"].nunique(), 1)

    def test_drops_posts_without_next_trading_day_or_history(self):
        px = fake_prices()["Open"]
        stamps = [px.index[2], px.index[-1], px.index[15]]         # too early, no next day, valid
        df = P.build_stock(fake_posts(stamps), "GME")
        self.assertEqual(len(df), 1)
        self.assertEqual(df["day"].iloc[0], px.index[15])


class SplitTests(unittest.TestCase):
    days = np.repeat(pd.bdate_range("2021-01-04", periods=50).values, 4)   # 4 posts/day, time ordered

    def parts(self, mode):
        return P.split_idx(self.days, mode)

    def test_every_mode_is_disjoint(self):
        for mode in ("random", "grouped", "chrono"):
            tr, va, te = self.parts(mode)
            self.assertEqual(len(set(tr) & set(va)) + len(set(tr) & set(te)) + len(set(va) & set(te)), 0)

    def test_grouped_never_splits_a_day_across_parts(self):
        tr, va, te = self.parts("grouped")
        sets = [set(self.days[i]) for i in (tr, va, te)]
        self.assertFalse(sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])

    def test_random_does_split_days(self):
        tr, _, te = self.parts("random")
        self.assertTrue(set(self.days[tr]) & set(self.days[te]))   # the leak we demonstrate

    def test_chrono_test_days_are_strictly_later_with_embargo(self):
        tr, va, te = self.parts("chrono")
        last_fit = max(self.days[np.concatenate([tr, va])]); first_test = min(self.days[te])
        self.assertGreater(first_test, last_fit)
        self.assertGreater(len(np.unique(self.days[(self.days > last_fit) & (self.days < first_test)])), 0)

    def test_sizes_are_roughly_80_10_10_by_days(self):
        tr, va, te = self.parts("grouped")
        self.assertEqual(len(np.unique(self.days[tr])), 40)
        self.assertEqual(len(np.unique(self.days[va])), 5)


class CvTests(unittest.TestCase):
    days = np.repeat(pd.bdate_range("2021-01-04", periods=60).values, 3)

    def test_grouped_cv_keeps_days_together(self):
        for tr, va in P.make_cv(self.days, "grouped"):
            self.assertFalse(set(self.days[tr]) & set(self.days[va]))

    def test_chrono_cv_validates_on_the_future_with_gap(self):
        folds = P.make_cv(self.days, "chrono")
        self.assertEqual(len(folds), 3)
        for tr, va in folds:
            self.assertGreater(min(self.days[va]), max(self.days[tr]))


if __name__ == "__main__":
    unittest.main()