
# WSB market prediction – replication + improvements (UE24CS352A ML mini-project)

Replicates *"Predicting market volatility and building short-term trading strategies using data from Reddit's WallStreetBets"* (CS229) and improves on it.

## Setup (Python 3.10+)

```bash
pip install -r requirements.txt
python src/run.py            # downloads data on first run (Kaggle gpreda/reddit-wallstreetsbets-posts + Yahoo prices)
```

If Kaggle download fails, put `reddit_wsb.csv` in `data/` manually. Quick check: `python src/run.py --fast --stocks TSLA --tasks direction`.
Other flags: `--stocks GME AMC TSLA FB`, `--tasks direction volatility`, `--splits random chrono`, `--n_iter 15`.
Outputs go to `results/` (results.csv, summary_mean_accuracy.csv, improvement.csv, figures).

## What is run

| Group                                     | Models                                                                                             |
| ----------------------------------------- | -------------------------------------------------------------------------------------------------- |
| Baselines                                 | random, majority class                                                                             |
| Paper (6 features + bias, fixed settings) | Newton's-method logistic regression (from scratch), 3x256 ReLU NN (Adam 1e-3, batch 16, 30 epochs) |
| Ablation                                  | RandomForest, HistGradBoost on the paper's features only                                           |
| New (extended features, tuned)            | NN, LogReg (tuned L2), RandomForest, HistGradBoost, SVM-RBF, KNN                                   |

**Improvements over the paper:** (1) extra features – log score/comments, caps ratio, exclamations, hour, weekday, same-day mention count, previous return, 5-day volatility, TF-IDF+SVD of text; (2) extra model families (RF, HGB, SVM, KNN, tuned LR, extended-feature NN); (3) randomized hyper-parameter search with time-ordered CV (tuning never touches the test set); (4) model chosen by CV AUC ("BestByCV"), not by test score; (5) both a paper-style random split and a **chronological** split (no look-ahead leakage); (6) 95% bootstrap CI, balanced accuracy, F1, AUC, majority baseline; (7) two tasks: next-day direction and "significant shift" (volatility).

Labels: t0 = first trading day on/after the post, t1 = next trading day; direction = open(t1) > open(t0); volatility = |open(t1)/open(t0) − 1| > train median.

## v2: leakage-free evaluation (split by DAY, not by post)

All posts of one trading day share one label (and several features), so a post-level shuffle lets the model memorise days.
Posts are therefore grouped by the **trading day t0** they map to (weekend/holiday posts join Monday's group).

```bash
python src/run.py                          # default splits: grouped (shuffled days) + chrono (future days, 1-day embargo)
python src/run.py --splits random grouped chrono   # add the paper-style post shuffle to demonstrate the leakage
python src/run.py --seeds 5                # neural nets: mean/std over seeds (acc_std, acc_ens columns in results.csv)
python src/backtest.py                     # walk-forward long/flat backtest -> results/backtest_summary.csv, fig_backtest.png
```

* **Splits:** `random` (paper, leaky) / `grouped` / `chrono`.
* **Leakage-free tuning:** hyper-parameter search uses day-based CV folds (GroupKFold by day, or forward-chaining with a 1-day embargo for chrono) - never same-day posts in train and validation.
* **Seed averaging:** paper NN and extended-feature NN are trained with `--seeds` seeds; `acc` = mean single-net test accuracy, `acc_std` = std over seeds, `acc_ens` = accuracy of the averaged ensemble.
* **Backtest:** expanding-window walk-forward (retrain every `--block` days, 1-day embargo), daily mean probability > 0.5 -> long else flat, `--cost_bps` per entry/exit, compared with buy-and-hold on the same days. Reports return, Sharpe, max drawdown, hit rate, exposure, trades, block-bootstrap 95% CI of excess return and a permutation-test p-value (is the timing better than random timing at the same exposure?).

## Tests and documents

```bash
python -m unittest discover -s tests -t . -v     # 24 unit tests: labels, day-based splits/CV, Newton LR, seed-averaged NN, backtest logic
python docs/build_docs.py                        # builds docs/writeup.pdf (2 pages) and docs/presentation.pptx from results_*/ folders
```

Edit `TEAM` and `PROBLEM` at the top of `docs/build_docs.py` (and the contributions slide) before submitting.
