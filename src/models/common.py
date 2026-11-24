"""Shared helpers for all detectors (M1-M4): data loading, splits, threshold, metrics, saving.

Every model is evaluated the same way:
  * train/val/test rows come from one split column (e.g. `loso_S2`);
  * the threshold is the one giving 1% FPR on the benign calibration pool (`subset == "calibration"`);
  * the same threshold is applied to the split's test rows and to NotInject (RQ3).

Outputs (one folder per data file, so FAKE results never mix with real ones):
  results/<data stem>/preds/<model>/<split>_seed<k>.parquet   id, role, label, score  (no text)
  results/<data stem>/metrics/<model>.csv                     one row per (split, seed)
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

from src.data.schema import SPLIT_COLUMNS

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = REPO_ROOT / "data" / "processed" / "apig_v1.parquet"
DEFAULT_OUT = REPO_ROOT / "results"

TARGET_FPR = 0.01          # operating point: 1% FPR on the calibration pool
COST_FN, COST_FP = 10, 1   # cost-weighted error c_FN*FNR + c_FP*FPR (proposal: 10:1)
NEEDED = ["id", "text", "label", "source", "subset", "type", "strategy_primary", *SPLIT_COLUMNS]


# ------------------------------------------------------------------ data
def load_data(path: str | Path | None) -> tuple[pd.DataFrame, Path]:
    path = Path(path) if path else DEFAULT_DATA
    if not path.exists():
        raise SystemExit(f"Data file not found: {path}")
    df = pd.read_parquet(path)
    missing = [c for c in NEEDED if c not in df.columns]
    if missing:
        raise SystemExit(f"{path.name} is missing columns {missing}")
    df = df.reset_index(drop=True)
    df["label"] = df["label"].astype(int)
    if "FAKE" in path.name.upper():
        print("=" * 72 + "\n  WARNING: FAKE strategy labels. Use only to test code; never report.\n" + "=" * 72)
    print(f"Loaded {path.name}: {len(df):,} rows")
    return df, path


def split_rows(df: pd.DataFrame, split: str) -> dict[str, pd.DataFrame]:
    """Rows a model needs for one split column. calibration/notinject are the same for every split."""
    if split not in SPLIT_COLUMNS:
        raise SystemExit(f"Unknown split {split!r}; choose from {SPLIT_COLUMNS}")
    col = df[split]
    parts = {
        "train": df[col == "train"],
        "val": df[col == "val"],
        "test": df[col == "test"],
        "calibration": df[df["subset"] == "calibration"],
        "notinject": df[df["subset"] == "notinject"],
    }
    for name in ("train", "val", "test"):
        if parts[name]["label"].nunique() < 2:
            raise SystemExit(f"{split}/{name} does not contain both classes")
    return parts


def eval_rows(parts: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Rows to score after training: test + calibration + notinject, tagged with their role."""
    return pd.concat([parts[r].assign(role=r) for r in ("test", "calibration", "notinject")], ignore_index=True)


def parse_splits(values: list[str] | None) -> list[str]:
    if not values or values == ["all"]:
        return list(SPLIT_COLUMNS)
    out = []
    for v in values:                      # allow prefixes: "loso" -> loso_S1..S3
        match = [s for s in SPLIT_COLUMNS if s == v or s.startswith(v + "_")]
        if not match:
            raise SystemExit(f"Unknown split {v!r}; choose from {SPLIT_COLUMNS}")
        out += [m for m in match if m not in out]
    return out


# ------------------------------------------------------------------ threshold + metrics
def threshold_at_fpr(calib_scores: np.ndarray, fpr: float = TARGET_FPR) -> float:
    """Smallest threshold t with mean(calib_scores > t) <= fpr. A row is flagged as attack if score > t."""
    return float(np.quantile(np.asarray(calib_scores, dtype=float), 1 - fpr, method="higher"))


def _rates(y: np.ndarray, pred: np.ndarray) -> tuple[float, float]:
    pos, neg = y == 1, y == 0
    fnr = float((pred[pos] == 0).mean()) if pos.any() else float("nan")
    fpr = float((pred[neg] == 1).mean()) if neg.any() else float("nan")
    return fnr, fpr


def compute_metrics(preds: pd.DataFrame) -> dict:
    """preds: output of eval_rows() plus a `score` column (higher = more likely attack)."""
    calib = preds[preds.role == "calibration"]
    test = preds[preds.role == "test"]
    ni = preds[preds.role == "notinject"]
    t = threshold_at_fpr(calib["score"].to_numpy())

    y, s = test["label"].to_numpy(), test["score"].to_numpy()
    pred = (s > t).astype(int)
    fnr, fpr = _rates(y, pred)
    m = {
        "threshold": t,
        "n_test_attack": int((y == 1).sum()), "n_test_benign": int((y == 0).sum()),
        "auroc": float(roc_auc_score(y, s)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "fnr": fnr, "fpr": fpr,
        "cost_10to1": COST_FN * fnr + COST_FP * fpr,
        "fpr_calibration": float((calib["score"].to_numpy() > t).mean()),   # sanity check, ~0.01
        "fpr_notinject": float((ni["score"].to_numpy() > t).mean()) if len(ni) else float("nan"),
    }
    # FNR per attack type and per primary strategy (only groups present in this test set)
    att = test[test.label == 1]
    for col, prefix in (("type", "fnr_"), ("strategy_primary", "fnr_")):
        for g, grp in att.groupby(col, dropna=True):
            m[f"{prefix}{g}"] = float((grp["score"].to_numpy() <= t).mean())
    return m


# ------------------------------------------------------------------ saving
class Results:
    """Writes per-run predictions and appends a metrics row; lets scripts skip finished runs."""

    def __init__(self, out_dir: str | Path | None, data_path: Path, model: str):
        self.root = Path(out_dir or DEFAULT_OUT) / data_path.stem
        self.model = model
        self.pred_dir = self.root / "preds" / model
        self.metrics_file = self.root / "metrics" / f"{model}.csv"
        self.pred_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_file.parent.mkdir(parents=True, exist_ok=True)
        self.data_name = data_path.name

    def pred_path(self, split: str, seed: int) -> Path:
        return self.pred_dir / f"{split}_seed{seed}.parquet"

    def done(self, split: str, seed: int) -> bool:
        return self.pred_path(split, seed).exists()

    def save(self, split: str, seed: int, preds: pd.DataFrame, extra: dict | None = None) -> dict:
        cols = ["id", "role", "label", "score"]
        preds[cols].to_parquet(self.pred_path(split, seed), index=False)
        row = {"model": self.model, "split": split, "seed": seed, **compute_metrics(preds),
               **(extra or {}), "data": self.data_name, "time": time.strftime("%Y-%m-%d %H:%M")}
        old = pd.read_csv(self.metrics_file) if self.metrics_file.exists() else pd.DataFrame()
        if len(old):
            old = old[~((old.split == split) & (old.seed == seed))]
        pd.concat([old, pd.DataFrame([row])], ignore_index=True).to_csv(self.metrics_file, index=False)
        print(f"  [{self.model} | {split} | seed {seed}] AUROC {row['auroc']:.3f}  FNR {row['fnr']:.3f}  "
              f"FPR {row['fpr']:.3f}  FPR(NotInject) {row['fpr_notinject']:.3f}")
        return row

    def save_config(self, config: dict) -> None:
        (self.root / "metrics" / f"{self.model}_config.json").write_text(json.dumps(config, indent=2, default=str))
