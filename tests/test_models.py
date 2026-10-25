import os, sys, unittest
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import models as M, pipeline as P


class NewtonLRTests(unittest.TestCase):
    def data(self, n=300, seed=0):
        rng = np.random.RandomState(seed); X = rng.randn(n, 3)
        return X, (X @ np.array([2., -1., .5]) + .3 * rng.randn(n) > 0).astype(int)

    def test_converges_quickly_and_learns(self):
        X, y = self.data()
        m = M.NewtonLR(l2=1e-3).fit(X, y)
        self.assertLess(m.history_[-1], 1e-5)
        self.assertLess(len(m.history_), 20)
        self.assertGreater((m.predict(X) == y).mean(), .9)

    def test_probabilities_are_valid(self):
        X, y = self.data(); p = M.NewtonLR().fit(X, y).predict_proba(X)
        self.assertEqual(p.shape, (len(X), 2))
        np.testing.assert_allclose(p.sum(1), 1.0)
        self.assertTrue(((p >= 0) & (p <= 1)).all())

    def test_stronger_l2_shrinks_weights(self):
        X, y = self.data()
        small, big = M.NewtonLR(l2=1e-6).fit(X, y), M.NewtonLR(l2=1e3).fit(X, y)
        self.assertLess(np.linalg.norm(big.w_[:-1]), np.linalg.norm(small.w_[:-1]))


class PreprocessingTests(unittest.TestCase):
    def frame(self):
        rng = np.random.RandomState(1); n = 120
        d = {c: rng.rand(n) for c in P.PAPER + P.EXTRA}
        vocab = [f"word{i}" for i in range(40)]
        d["text"] = [" ".join(rng.choice(vocab, 8)) for _ in range(n)]
        return pd.DataFrame(d)

    def test_paper_features_are_six_standardised_columns(self):
        X = M.make_prep("paper").fit_transform(self.frame())
        self.assertEqual(X.shape[1], 6)
        np.testing.assert_allclose(X.mean(0), 0, atol=1e-8)

    def test_extended_features_add_text_components(self):
        X = M.make_prep("ext").fit_transform(self.frame())
        self.assertEqual(X.shape[1], len(P.PAPER) + len(P.EXTRA) + 20)


class ModelRosterTests(unittest.TestCase):
    def test_tuned_roster(self):
        self.assertEqual(set(M.TUNED), {"LogReg-Newton (tuned L2)", "RandomForest", "HistGradBoost", "SVM-RBF", "KNN"})

    def test_search_spaces_target_the_classifier_step(self):
        for _, grid in M.TUNED.values():
            self.assertTrue(all(k.startswith("clf__") for k in grid))


class SeedAvgNNTests(unittest.TestCase):
    def test_trains_one_net_per_seed_and_averages_probabilities(self):
        rng = np.random.RandomState(0); X = rng.randn(120, 4); y = (X[:, 0] > 0).astype(int)
        m = M.SeedAvgNN(n_seeds=2, epochs=3).fit(X, y)
        self.assertEqual(len(m.models_), 2)
        expected = np.mean([n.predict_proba(X) for n in m.models_], axis=0)
        np.testing.assert_allclose(m.predict_proba(X), expected)
        self.assertEqual(m.predict(X).shape, (120,))


if __name__ == "__main__":
    unittest.main()