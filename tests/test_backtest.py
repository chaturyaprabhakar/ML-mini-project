import os, sys, unittest
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import backtest as B


def toy(p, r):
    return pd.DataFrame({"p": p, "ret": r}, index=pd.bdate_range("2021-03-01", periods=len(p)))


class EvaluateTests(unittest.TestCase):
    r = np.array([.02, -.01, .03, -.02, .01, .02, -.03, .01, .02, -.01])

    def test_always_long_matches_buy_and_hold(self):
        res, eq, beq = B.evaluate(toy(np.ones(10), self.r), 0.001, np.random.RandomState(0), nboot=50, nperm=50)
        self.assertAlmostEqual(res["excess_mean_daily"], 0.0)
        np.testing.assert_allclose(eq, beq)
        self.assertEqual(res["n_trades"], 1)
        self.assertEqual(res["exposure"], 1.0)

    def test_always_flat_earns_nothing_and_pays_no_cost(self):
        res, _, _ = B.evaluate(toy(np.zeros(10), self.r), 0.01, np.random.RandomState(0), nboot=50, nperm=50)
        self.assertEqual(res["total_ret"], 0.0)
        self.assertEqual(res["n_trades"], 0)

    def test_costs_reduce_returns(self):
        d = toy(np.array([1, 0] * 5, float), self.r)
        free, _, _ = B.evaluate(d, 0.0, np.random.RandomState(0), nboot=50, nperm=50)
        paid, _, _ = B.evaluate(d, 0.005, np.random.RandomState(0), nboot=50, nperm=50)
        self.assertLess(paid["total_ret"], free["total_ret"])

    def test_perfect_foresight_has_small_permutation_p_value(self):
        r = np.random.RandomState(1).randn(40) * .02
        res, _, _ = B.evaluate(toy((r > 0).astype(float), r), 0.0, np.random.RandomState(0), nboot=50, nperm=500)
        self.assertLess(res["perm_p"], .05)
        self.assertEqual(res["day_acc"], 1.0)


class WalkForwardTests(unittest.TestCase):
    def test_never_trains_on_the_future(self):
        n_days = 60
        days = np.repeat(pd.bdate_range("2021-01-04", periods=n_days).values, 10)
        rng = np.random.RandomState(0)
        df = pd.DataFrame({"day": days, "direction": rng.randint(0, 2, len(days)),
                           "ret_next": np.repeat(rng.randn(n_days) * .02, 10)})
        seen = []

        class Spy:
            def fit(self, X, y):
                self.train_last = X["day"].max(); return self
            def predict_proba(self, X):
                seen.append((self.train_last, X["day"].min()))
                return np.c_[np.full(len(X), .5), np.full(len(X), .5)]

        out = B.walk_forward(df, Spy, start=.5, block=5)
        self.assertEqual(len(seen), 6)
        for last_train, first_test in seen:
            self.assertLess(last_train, first_test)
        self.assertEqual(len(out), n_days // 2)


if __name__ == "__main__":
    unittest.main()