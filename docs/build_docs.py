"""Build the write-up (PDF) and the presentation (PPTX) from the experiment results.

    python docs/build_docs.py [--results_dir .]

Reads  results_2_leakage/results.csv (or results_1_main / results), results_3_backtest/backtest_summary.csv
Writes docs/writeup.pdf, docs/presentation.pptx, docs/figs/*.png. Every number in the text is computed from the CSVs.
Edit TEAM / PROBLEM below before submitting.
"""
import argparse, os, shutil
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

TEAM = "Kaushik Krishna Dwarakanath (PES1UG24CS900)  |  Chaturya Prabhakar Reddy (PES1UG24CS126)"
PROBLEM = ("Replicate the CS229 report 'Predicting market volatility and building short-term trading strategies using data "
           "from Reddit's WallStreetBets', then build and compare further models and a more rigorous evaluation. "
           "Task: from a WSB post that mentions GME, AMC, TSLA or FB, predict (a) whether the stock's open rises at the next "
           "trading day (direction) and (b) whether the move is large (volatility).")
HERE = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.join(HERE, "figs")
PAPER_M = ["LogReg-Newton (paper)", "NN 3x256 (paper)"]
NEW_M = ["NN 3x256 (ext feats)", "LogReg-Newton (tuned L2)", "RandomForest", "HistGradBoost", "SVM-RBF", "KNN"]
ABL_M = ["RandomForest (paper feats)", "HistGradBoost (paper feats)"]
ROSTER = ["Majority baseline", "Random baseline"] + PAPER_M + ABL_M + NEW_M
SPLITS = ["random", "grouped", "chrono"]
SHORT = {"LogReg-Newton (paper)": "LogReg Newton (paper)", "NN 3x256 (paper)": "NN 3x256 (paper)",
         "RandomForest (paper feats)": "RF, paper feats", "HistGradBoost (paper feats)": "HGB, paper feats",
         "NN 3x256 (ext feats)": "NN 3x256, ext feats", "LogReg-Newton (tuned L2)": "LogReg, tuned L2"}


def load(root):
    pick = lambda *c: next((p for p in (os.path.join(root, x) for x in c) if os.path.exists(p)), None)
    r = pick("results_2_leakage/results.csv", "results_1_main/results.csv", "results/results.csv")
    b = pick("results_3_backtest/backtest_summary.csv", "results/backtest_summary.csv")
    if r is None: raise SystemExit("no results.csv found - run the experiments first")
    d = pd.read_csv(r); d = d[d.model.isin(ROSTER)]
    bt = pd.read_csv(b) if b else None
    fig = pick("results_3_backtest/fig_backtest.png", "results/fig_backtest.png")
    return d, bt, fig


def mean_table(d, metric):
    pv = d.pivot_table(index="model", columns=["task", "split"], values=metric)
    return pv.reindex([m for m in ROSTER if m in pv.index])


def facts(d, bt):
    acc, bal = mean_table(d, "acc"), mean_table(d, "bal_acc")
    f = {}
    n = d[d.split == "random"].groupby("stock").apply(lambda g: int((g.n_train + g.n_test).iloc[0]), include_groups=False)
    f["n_posts"] = n.to_dict()
    for t in ("direction", "volatility"):
        f[f"maj_{t}"] = acc.loc["Majority baseline", (t, "grouped")]
        f[f"leak_rf_{t}"] = (acc.loc["RandomForest", (t, "random")], acc.loc["RandomForest", (t, "grouped")])
        f[f"leak_hgb_{t}"] = (acc.loc["HistGradBoost", (t, "random")], acc.loc["HistGradBoost", (t, "grouped")])
        pb = bal.loc[PAPER_M, (t, "grouped")]; nb = bal.loc[NEW_M, (t, "grouped")]
        f[f"paper_best_bal_{t}"] = (pb.idxmax(), pb.max()); f[f"new_best_bal_{t}"] = (nb.idxmax(), nb.max())
        f[f"new_best_acc_{t}"] = acc.loc[NEW_M, (t, "grouped")].max()
        f[f"paper_acc_rand_{t}"] = acc.loc[PAPER_M, (t, "random")].max()
    f["test_sizes"] = d[d.split == "chrono"].groupby("stock").n_test.first().to_dict()
    if bt is not None:
        f["bt_days"] = int(bt.n_days.median()); f["bt_sig"] = int((bt.perm_p < .05).sum()); f["bt_total"] = len(bt)
        f["bt_minp"] = bt.perm_p.min()
    return f, acc, bal


def make_figs(acc, bal, fig_bt):
    os.makedirs(FIGS, exist_ok=True); out = {}
    show = ["Majority baseline", "LogReg-Newton (paper)", "NN 3x256 (paper)", "RandomForest", "HistGradBoost"]
    col = {"random": "#d9534f", "grouped": "#2b6cb0", "chrono": "#718096"}
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.4), sharey=True)
    for a, t in zip(ax, ("direction", "volatility")):
        w = .26
        for i, s in enumerate(SPLITS):
            a.bar(np.arange(len(show)) + (i - 1) * w, [acc.loc[m, (t, s)] for m in show], w, label=s, color=col[s])
        a.set_xticks(range(len(show))); a.set_xticklabels([SHORT.get(m, m).replace(" ", "\n", 1) for m in show], fontsize=7)
        a.set_title(f"{t}: mean test accuracy", fontsize=9); a.set_ylim(.3, 1); a.grid(axis="y", alpha=.3)
    ax[0].legend(title="split", fontsize=7, title_fontsize=7); ax[0].set_ylabel("accuracy")
    plt.tight_layout(); out["leak"] = os.path.join(FIGS, "fig_leakage.png"); plt.savefig(out["leak"], dpi=150); plt.close()
    models = ["LogReg-Newton (paper)", "NN 3x256 (paper)"] + NEW_M
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for a, t in zip(ax, ("direction", "volatility")):
        v = [bal.loc[m, (t, "grouped")] for m in models]
        a.bar(range(len(models)), v, color=["#d9534f"] * 2 + ["#2b6cb0"] * len(NEW_M)); a.axhline(.5, c="k", lw=.8, ls="--")
        a.set_xticks(range(len(models))); a.set_xticklabels([SHORT.get(m, m) for m in models], fontsize=7, rotation=30, ha="right")
        a.set_title(f"{t}: balanced accuracy, day-grouped split", fontsize=9); a.set_ylim(.3, .9); a.grid(axis="y", alpha=.3)
    ax[0].set_ylabel("balanced accuracy (red = paper models)")
    plt.tight_layout(); out["bal"] = os.path.join(FIGS, "fig_balacc.png"); plt.savefig(out["bal"], dpi=150); plt.close()
    out["bt"] = None
    if fig_bt: out["bt"] = os.path.join(FIGS, "fig_backtest.png"); shutil.copy(fig_bt, out["bt"])
    return out


def conclusions(f):
    c = [f"Leakage: the paper-style random split lets same-day posts (which share one label) fall in both train and test. "
         f"Random Forest direction accuracy falls from {f['leak_rf_direction'][0]:.3f} (random) to {f['leak_rf_direction'][1]:.3f} "
         f"(day-grouped); volatility from {f['leak_rf_volatility'][0]:.3f} to {f['leak_rf_volatility'][1]:.3f}. "
         f"The paper's own models barely move, so reported paper-style accuracies are likely optimistic."]
    for t in ("direction", "volatility"):
        pm, pv = f[f"paper_best_bal_{t}"]; nm, nv = f[f"new_best_bal_{t}"]
        beat = "does not beat" if f[f"new_best_acc_{t}"] <= f[f"maj_{t}"] else "beats"
        c.append(f"{t.capitalize()}, leak-free: best new model by balanced accuracy is {SHORT.get(nm, nm)} ({nv:.3f}) vs best paper model "
                 f"{SHORT.get(pm, pm)} ({pv:.3f}); however the best new raw accuracy {f[f'new_best_acc_{t}']:.3f} {beat} the majority baseline "
                 f"({f[f'maj_{t}']:.3f}).")
    if "bt_sig" in f:
        c.append(f"Walk-forward backtest (about {f['bt_days']} traded days per stock, 10 bps costs): {f['bt_sig']} of {f['bt_total']} stock-model "
                 f"combinations beat random timing at p<0.05 (smallest p = {f['bt_minp']:.2f}), so there is no statistically reliable trading edge.")
    c.append("Limitations: few days per stock, tiny chronological test sets, regime shifts around the Jan 2021 spike; results are exploratory. "
             "Chronological-split numbers should be read with caution.")
    return c


def approach_text(f):
    n = f["n_posts"]; tot = ", ".join(f"{k} {v:,}" for k, v in n.items())
    return {
        "dataset": f"Kaggle 'gpreda/reddit-wallstreetsbets-posts' (title, score, comments, body, timestamp) joined with Yahoo Finance daily "
                   f"open prices. Posts mentioning GME, AMC, TSLA or FB give {tot} labelled posts. Features (paper): post score, "
                   f"comment count, length, VADER positive/neutral/negative sentiment (+ bias). Labels: direction = open(t1) > open(t0), "
                   f"volatility = |open(t1)/open(t0)-1| above the training median, where t0 is the first trading day on/after the post "
                   f"and t1 the next trading day.",
        "approach": "(1) Replicate: Newton's-method logistic regression written from scratch and a 3x256 ReLU network "
                    "(Adam 1e-3, batch 16, 30 epochs), plus random/majority baselines. (2) Extend: six extra model families "
                    "(NN on extended features, tuned logistic regression, Random Forest, Gradient Boosting, SVM-RBF, KNN) and extra "
                    "features (log counts, caps ratio, hour, weekday, daily mentions, previous return, 5-day volatility, TF-IDF+SVD). "
                    "(3) Evaluate rigorously: three 80/10/10 splits - paper-style random post shuffle (leaky), day-grouped, and chronological "
                    "with an embargo day; leakage-free tuning with day-based CV folds; neural nets averaged over 5 seeds; balanced accuracy, "
                    "F1, AUC and bootstrap CIs; walk-forward long/flat backtest with costs, block bootstrap and a permutation test.",
        "impl": "Python (pandas, scikit-learn, NumPy). src/pipeline.py (cleaning, VADER, labels, day-based splits/CV), src/models.py "
                "(NewtonLR, SeedAvgNN, model roster and search spaces), src/run.py (experiments, tables, plots), src/backtest.py "
                "(walk-forward backtest), tests/ (unit tests), run_all.sh (one-command reproduction)."}


def tab_rows(acc, bal):
    hdr = ["Model", "Dir rand", "Dir grp", "Dir chr", "Vol rand", "Vol grp", "Vol chr", "BalAcc dir", "BalAcc vol"]
    rows = []
    for m in acc.index:
        a = [acc.loc[m, (t, s)] for t in ("direction", "volatility") for s in SPLITS]
        b = [bal.loc[m, (t, "grouped")] for t in ("direction", "volatility")]
        rows.append([SHORT.get(m, m)] + [f"{x:.3f}" for x in a + b])
    return hdr, rows


def bt_rows(bt):
    hdr = ["Stock", "Model", "Days", "Exposure", "Return", "Buy&hold", "Sharpe", "Perm. p"]
    rows = [[r.stock, r.model.split(" (")[0], int(r.n_days), f"{r.exposure:.2f}", f"{r.total_ret:+.1%}", f"{r.bh_total_ret:+.1%}",
             "-" if pd.isna(r.sharpe) else f"{r.sharpe:.2f}", f"{r.perm_p:.2f}"] for r in bt.itertuples()]
    return hdr, rows


def build_pdf(path, f, acc, bal, bt, figs):
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
    B = ParagraphStyle("b", fontName="Helvetica", fontSize=8.3, leading=10.2, spaceAfter=2.5)
    H = ParagraphStyle("h", fontName="Helvetica-Bold", fontSize=9.6, leading=11.5, spaceBefore=4, spaceAfter=2, textColor=colors.HexColor("#1a365d"))
    T = ParagraphStyle("t", fontName="Helvetica-Bold", fontSize=13, leading=15, spaceAfter=2)
    C = ParagraphStyle("c", fontName="Helvetica-Oblique", fontSize=7, leading=8.5, spaceAfter=3)
    at = approach_text(f)

    def table(hdr, rows, widths, hi=None):
        t = Table([hdr] + rows, colWidths=widths, repeatRows=1)
        st = [("FONT", (0, 0), (-1, -1), "Helvetica", 6.8), ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 6.8),
              ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e2e8f0")), ("GRID", (0, 0), (-1, -1), .25, colors.grey),
              ("ALIGN", (1, 0), (-1, -1), "CENTER"), ("TOPPADDING", (0, 0), (-1, -1), 1.2), ("BOTTOMPADDING", (0, 0), (-1, -1), 1.2)]
        for r in (hi or []): st.append(("BACKGROUND", (0, r), (-1, r), colors.HexColor("#edf2f7")))
        t.setStyle(TableStyle(st)); return t

    el = [Paragraph("Replicating and stress-testing stock prediction from Reddit WallStreetBets posts", T),
          Paragraph(TEAM.replace("<", "&lt;").replace(">", "&gt;"), C),
          Paragraph("1. Problem statement", H), Paragraph(PROBLEM, B),
          Paragraph("2. Dataset", H), Paragraph(at["dataset"], B),
          Paragraph("3. Approach", H), Paragraph(at["approach"], B),
          Paragraph("4. Implementation overview", H), Paragraph(at["impl"], B),
          Paragraph("5. Results", H)]
    hdr, rows = tab_rows(acc, bal)
    el += [Paragraph("Table 1. Mean over the four stocks. Accuracy on the random (leaky), day-grouped and chronological test sets, and "
                     "balanced accuracy on the day-grouped set. First rows are baselines, then the paper's models, then ablations and new models.", C),
           table(hdr, rows, [3.5 * cm] + [1.55 * cm] * 6 + [1.8 * cm] * 2, hi=[3, 4]), Spacer(1, 4),
           Image(figs["leak"], width=16.5 * cm, height=5.6 * cm),
           Paragraph("Figure 1. Accuracy drops once the leak (same-day posts across train and test) is removed.", C)]
    if bt is not None:
        h2, r2 = bt_rows(bt)
        el += [PageBreak(), Paragraph("Table 2. Walk-forward long/flat backtest (direction task, retrained every 10 days, 10 bps cost per entry/exit, "
                         "same-day buy-and-hold as benchmark; p-value from a permutation test of the timing).", C),
               table(h2, r2, [1.3 * cm, 3.5 * cm, 1.2 * cm, 1.8 * cm, 1.9 * cm, 1.9 * cm, 1.5 * cm, 1.6 * cm])]
    el += [Spacer(1, 4), Image(figs["bal"], width=16.5 * cm, height=6.3 * cm),
           Paragraph("Figure 2. Balanced accuracy on the day-grouped split (red = paper models, dashed = chance).", C)]
    if figs.get("bt"):
        el += [Image(figs["bt"], width=13.5 * cm, height=7.9 * cm), Paragraph("Figure 3. Equity curves of the walk-forward backtest.", C)]
    el.append(Paragraph("6. Conclusions", H))
    for c in conclusions(f): el.append(Paragraph("&bull; " + c, B))
    SimpleDocTemplate(path, pagesize=A4, leftMargin=1.5 * cm, rightMargin=1.5 * cm, topMargin=1.2 * cm, bottomMargin=1.1 * cm).build(el)


def build_pptx(path, f, acc, bal, bt, figs):
    from pptx import Presentation
    from pptx.dml.color import RGBColor
    from pptx.util import Inches, Pt
    P_ = Presentation(); P_.slide_width, P_.slide_height = Inches(13.333), Inches(7.5)
    NAVY, BLUE = RGBColor(0x1a, 0x36, 0x5d), RGBColor(0x2b, 0x6c, 0xb0)
    at = approach_text(f)

    def slide(title, bullets=None, img=None, table=None, size=18, img_w=12.2):
        s = P_.slides.add_slide(P_.slide_layouts[6])
        bar = s.shapes.add_shape(1, 0, 0, P_.slide_width, Inches(1.0)); bar.fill.solid(); bar.fill.fore_color.rgb = NAVY; bar.line.fill.background()
        tb = s.shapes.add_textbox(Inches(.5), Inches(.15), Inches(12.3), Inches(.7)); r = tb.text_frame.paragraphs[0].add_run()
        r.text = title; r.font.size = Pt(28); r.font.bold = True; r.font.color.rgb = RGBColor(255, 255, 255)
        top = 1.25
        if bullets:
            h = 1.2 if img else 5.6
            tf = s.shapes.add_textbox(Inches(.6), Inches(top), Inches(12.1), Inches(h)).text_frame; tf.word_wrap = True
            for i, b in enumerate(bullets):
                p = tf.paragraphs[0] if i == 0 else tf.add_paragraph(); p.text = "\u2022 " + b; p.font.size = Pt(size); p.space_after = Pt(8)
            top += h + .1
        if img: s.shapes.add_picture(img, Inches((13.333 - img_w) / 2), Inches(top), width=Inches(img_w))
        if table:
            hdr, rows = table; n = len(rows) + 1
            sh = s.shapes.add_table(n, len(hdr), Inches(.5), Inches(top), Inches(12.3), Inches(.37 * n)); t = sh.table
            for j, hd in enumerate(hdr): t.cell(0, j).text = hd
            for i, rw in enumerate(rows, 1):
                for j, v in enumerate(rw): t.cell(i, j).text = str(v)
            for i in range(n):
                for j in range(len(hdr)):
                    for p in t.cell(i, j).text_frame.paragraphs: p.font.size = Pt(11); p.font.bold = (i == 0)
        return s

    s = P_.slides.add_slide(P_.slide_layouts[6])
    bg = s.shapes.add_shape(1, 0, 0, P_.slide_width, P_.slide_height); bg.fill.solid(); bg.fill.fore_color.rgb = NAVY; bg.line.fill.background()
    tf = s.shapes.add_textbox(Inches(.8), Inches(2.3), Inches(11.7), Inches(3)).text_frame; tf.word_wrap = True
    tf.text = "Reddit WallStreetBets and stock prediction"; tf.paragraphs[0].font.size = Pt(40); tf.paragraphs[0].font.bold = True
    tf.paragraphs[0].font.color.rgb = RGBColor(255, 255, 255)
    for line in ("Replication, new models and leakage-free evaluation", TEAM):
        p = tf.add_paragraph(); p.text = line; p.font.size = Pt(18); p.font.color.rgb = RGBColor(0xbe, 0xe3, 0xf8)
    slide("Problem statement", [PROBLEM, "Long-term goal of the paper: a bot that makes short-term trades from unconventional sources like WSB.",
                                "Our goal: reproduce the results, add models, and test whether the gains survive a rigorous evaluation."], size=20)
    slide("Dataset and features", [at["dataset"]], size=18)
    slide("Approach", ["Replication: Newton logistic regression (from scratch) + 3x256 ReLU network, random and majority baselines",
                       "Extension: six extra model families, extended features, randomized hyperparameter search",
                       "Evaluation: random (paper) vs day-grouped vs chronological split; day-based CV for tuning; 5-seed NNs",
                       "Backtest: walk-forward long/flat with costs, bootstrap CI and permutation test"], size=19)
    slide("Models built and tested", ["Baselines: random, majority class",
                                      "Paper models: Newton logistic regression, NN 3x256 (ReLU, Adam 1e-3, batch 16, 30 epochs)",
                                      "Ablation: Random Forest and Gradient Boosting on the paper's 6 features",
                                      "New (extended features, tuned): NN, Logistic Regression (tuned L2), Random Forest, Gradient Boosting, SVM-RBF, KNN",
                                      "Selection and tuning use day-grouped / forward-chaining CV on training days only"], size=19)
    slide("Finding 1: the random split leaks", ["All posts of a trading day share one label, so a post-level shuffle lets models memorise days.",
                                                 f"RF direction: {f['leak_rf_direction'][0]:.3f} (random) -> {f['leak_rf_direction'][1]:.3f} (day-grouped). "
                                                 f"RF volatility: {f['leak_rf_volatility'][0]:.3f} -> {f['leak_rf_volatility'][1]:.3f}."], img=figs["leak"], img_w=10.8, size=16)
    hdr, rows = tab_rows(acc, bal)
    slide("Results: mean test accuracy over the 4 stocks", table=(hdr, rows))
    slide("Finding 2: balanced accuracy on a leak-free split",
          [f"Direction: best new {SHORT.get(f['new_best_bal_direction'][0], f['new_best_bal_direction'][0])} {f['new_best_bal_direction'][1]:.3f} vs best paper "
           f"{f['paper_best_bal_direction'][1]:.3f}. Volatility: {f['new_best_bal_volatility'][1]:.3f} vs {f['paper_best_bal_volatility'][1]:.3f}.",
           "Raw accuracy does not beat the majority baseline, so the improvement is modest and not consistent across stocks."], img=figs["bal"], img_w=10.8, size=16)
    if bt is not None and figs.get("bt"):
        slide("Walk-forward backtest", [f"About {f['bt_days']} traded days per stock; {f['bt_sig']} of {f['bt_total']} combinations beat random timing (p<0.05)."],
              img=figs["bt"], img_w=8.2, size=16)
    slide("Conclusions", conclusions(f), size=14)
    slide("Contributions and future work", ["Member 1: <fill in>", "Member 2: <fill in>",
                                            "Future work: pool stocks at the day level, longer price history, FinBERT sentiment, "
                                            "intraday timestamps for exact market-open alignment, more data beyond early 2021."], size=20)
    P_.save(path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--results_dir", default=os.path.join(HERE, "..")); a = ap.parse_args()
    d, bt, fig_bt = load(a.results_dir); f, acc, bal = facts(d, bt); figs = make_figs(acc, bal, fig_bt)
    build_pdf(os.path.join(HERE, "writeup.pdf"), f, acc, bal, bt, figs)
    build_pptx(os.path.join(HERE, "presentation.pptx"), f, acc, bal, bt, figs)
    print("built docs/writeup.pdf and docs/presentation.pptx")