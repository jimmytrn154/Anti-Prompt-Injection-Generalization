"""M1/M2: TF-IDF (word 1-2-grams + char 3-5-grams) + Logistic Regression / linear SVM.

    python -m src.models.m12_tfidf                                  # M1 and M2 on all 11 splits
    python -m src.models.m12_tfidf --models m1 --splits loso        # M1 on loso_S1..S3
    python -m src.models.m12_tfidf --save-model                     # also keep the split_random models (coefficients)

Deterministic, so one run per split (seed 0). The vectorizer is fitted on the training rows only.
C is chosen on the split's validation rows (AUROC). Scores: LR probability / SVM decision value;
only the ranking matters because the threshold is set on the calibration pool (common.py).
"""
from __future__ import annotations

import argparse
import time

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

from src.models.common import Results, eval_rows, load_data, parse_splits, split_rows

C_GRID = [1.0, 0.1, 10.0]          # first value wins ties
SEED = 0


def features() -> FeatureUnion:
    return FeatureUnion([
        ("word", TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True)),
    ])


def classifier(name: str, C: float):
    if name == "m1":
        return LogisticRegression(C=C, max_iter=2000, random_state=SEED)
    return LinearSVC(C=C, random_state=SEED)


def score(clf, X) -> np.ndarray:
    return clf.predict_proba(X)[:, 1] if hasattr(clf, "predict_proba") else clf.decision_function(X)


def run(name: str, df, split: str, res: Results, save_model: bool) -> None:
    parts = split_rows(df, split)
    t0 = time.time()
    vec = features()
    X_tr = vec.fit_transform(parts["train"]["text"])
    X_va = vec.transform(parts["val"]["text"])
    y_tr, y_va = parts["train"]["label"].to_numpy(), parts["val"]["label"].to_numpy()

    best = None
    for C in C_GRID:
        clf = classifier(name, C).fit(X_tr, y_tr)
        auc = roc_auc_score(y_va, score(clf, X_va))
        if best is None or auc > best[0] + 1e-9:
            best = (auc, C, clf)
    val_auc, C, clf = best
    train_time = time.time() - t0

    preds = eval_rows(parts)
    preds["score"] = score(clf, vec.transform(preds["text"]))
    pipe = Pipeline([("tfidf", vec), ("clf", clf)])
    t1 = time.perf_counter()
    for text in preds["text"].head(200):          # batch size 1, like a real guardrail
        score(clf, vec.transform([text]))
    latency = 1000 * (time.perf_counter() - t1) / min(len(preds), 200)
    res.save(split, SEED, preds, {"n_train": len(y_tr), "C": C, "val_auroc": val_auc,
                                  "n_features": X_tr.shape[1], "train_time_s": round(train_time, 1),
                                  "latency_ms": round(latency, 2)})
    if save_model and split == "split_random":
        path = res.root / "models" / f"{name}_split_random.joblib"
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipe, path)
        print(f"  saved {path} (for coefficient analysis)")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", help="apig_v1.parquet (default: data/processed/apig_v1.parquet)")
    p.add_argument("--out", help="results folder (default: results/)")
    p.add_argument("--models", nargs="+", default=["m1", "m2"], choices=["m1", "m2"])
    p.add_argument("--splits", nargs="+", default=["all"], help="split columns or prefixes (default: all 11)")
    p.add_argument("--save-model", action="store_true")
    args = p.parse_args()

    df, data_path = load_data(args.data)
    for name in args.models:
        res = Results(args.out, data_path, name)
        res.save_config({**vars(args), "C_grid": C_GRID})
        for split in parse_splits(args.splits):
            run(name, df, split, res, args.save_model)
        print(f"Metrics: {res.metrics_file}")


if __name__ == "__main__":
    main()
