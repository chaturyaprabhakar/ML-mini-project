"""Models: Newton-method logistic regression (from scratch), paper NN, and extra model families."""
import numpy as np
from scipy.stats import loguniform, randint
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from joblib import Parallel, delayed
from pipeline import PAPER, EXTRA


class NewtonLR(ClassifierMixin, BaseEstimator):
    """Logistic regression solved with Newton's method (IRLS); bias column added internally."""
    def __init__(self, l2=1e-6, max_iter=50, tol=1e-6):
        self.l2, self.max_iter, self.tol = l2, max_iter, tol

    def fit(self, X, y):
        X = np.asarray(X, float); y = np.asarray(y, float)
        Xb = np.hstack([X, np.ones((len(X), 1))])
        reg = np.full(Xb.shape[1], self.l2); reg[-1] = 0.0       # do not penalise bias
        w = np.zeros(Xb.shape[1]); self.history_ = []
        for _ in range(self.max_iter):
            p = 1 / (1 + np.exp(-np.clip(Xb @ w, -30, 30)))
            g = Xb.T @ (p - y) + reg * w
            H = (Xb * (p * (1 - p))[:, None]).T @ Xb + np.diag(reg) + 1e-8 * np.eye(len(w))
            step = np.linalg.solve(H, g); w -= step
            self.history_.append(float(np.linalg.norm(step)))
            if self.history_[-1] < self.tol:
                break
        self.w_, self.classes_ = w, np.array([0, 1])
        return self

    def decision_function(self, X):
        return np.hstack([np.asarray(X, float), np.ones((len(X), 1))]) @ self.w_

    def predict_proba(self, X):
        p = 1 / (1 + np.exp(-np.clip(self.decision_function(X), -30, 30)))
        return np.c_[1 - p, p]

    def predict(self, X):
        return (self.decision_function(X) > 0).astype(int)


def make_prep(feats):
    if feats == "paper":
        return ColumnTransformer([("num", StandardScaler(), PAPER)])
    txt = Pipeline([("tfidf", TfidfVectorizer(max_features=3000, ngram_range=(1, 2), min_df=3)),
                    ("svd", TruncatedSVD(20, random_state=0)), ("sc", StandardScaler())])
    return ColumnTransformer([("num", StandardScaler(), PAPER + EXTRA), ("txt", txt, "text")])


def pipe(feats, clf):
    return Pipeline([("prep", make_prep(feats)), ("clf", clf)])


def paper_nn(seed=0, epochs=30):
    """3 x 256 ReLU layers, sigmoid output, Adam lr=1e-3, batch 16, 30 epochs (as in the report)."""
    return MLPClassifier((256, 256, 256), activation="relu", solver="adam", learning_rate_init=1e-3,
                         batch_size=16, max_iter=epochs, random_state=seed)


# name -> (estimator, search space). Extra families beyond the paper: RF, HGB, SVM, KNN (+ tuned LR and the extended-feature NN).
_tree = {"clf__n_estimators": randint(100, 400), "clf__max_depth": [3, 5, 8, 12, None],
         "clf__min_samples_leaf": randint(2, 30), "clf__max_features": ["sqrt", 0.5]}
TUNED = {
    "LogReg-Newton (tuned L2)": (NewtonLR(), {"clf__l2": loguniform(1e-3, 1e3)}),
    "RandomForest": (RandomForestClassifier(random_state=0, n_jobs=1), _tree),
    "HistGradBoost": (HistGradientBoostingClassifier(random_state=0),
                      {"clf__learning_rate": loguniform(.01, .3), "clf__max_depth": [2, 3, 5, None],
                       "clf__max_leaf_nodes": randint(5, 31), "clf__l2_regularization": loguniform(1e-3, 10),
                       "clf__min_samples_leaf": randint(10, 60), "clf__max_iter": randint(50, 300)}),
    "SVM-RBF": (SVC(kernel="rbf"), {"clf__C": loguniform(.1, 100), "clf__gamma": loguniform(1e-4, 1e-1)}),
    "KNN": (KNeighborsClassifier(), {"clf__n_neighbors": randint(5, 80),
                                     "clf__weights": ["uniform", "distance"], "clf__p": [1, 2]}),
}


def _fit_nn(seed, epochs, X, y):
    return paper_nn(seed, epochs).fit(X, y)


class SeedAvgNN(ClassifierMixin, BaseEstimator):
    """The paper NN trained with `n_seeds` random seeds (in parallel). `models_` holds the individual nets
    (for mean/std of single-net accuracy); predict_proba averages them (seed ensemble)."""
    def __init__(self, n_seeds=5, epochs=30):
        self.n_seeds, self.epochs = n_seeds, epochs

    def fit(self, X, y):
        self.models_ = Parallel(n_jobs=-1)(delayed(_fit_nn)(s, self.epochs, X, y) for s in range(self.n_seeds))
        self.classes_ = self.models_[0].classes_
        return self

    def predict_proba(self, X):
        return np.mean([m.predict_proba(X) for m in self.models_], axis=0)

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(1)]