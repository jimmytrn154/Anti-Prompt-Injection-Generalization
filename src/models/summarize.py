"""Collect every model's metrics into one table (mean +- std over seeds).

    python -m src.models.summarize                         # results/apig_v1/
    python -m src.models.summarize --results results/apig_v1_FAKE

Writes <results>/summary.csv and prints FNR / FPR / FPR(NotInject) / AUROC per model x split.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.data.schema import SPLIT_COLUMNS
from src.models.common import DEFAULT_OUT

KEY = ["auroc", "f1", "fnr", "fpr", "fpr_notinject", "cost_10to1", "fnr_injection", "fnr_jailbreak", "latency_ms"]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", default=str(DEFAULT_OUT / "apig_v1"))
    args = p.parse_args()
    root = Path(args.results)
    files = sorted(f for f in (root / "metrics").glob("*.csv") if "smoke" not in f.stem)
    if not files:
        raise SystemExit(f"No metrics in {root / 'metrics'}")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    cols = [c for c in KEY if c in df.columns]
    g = df.groupby(["model", "split"])[cols]
    summary = g.mean().add_suffix("_mean").join(g.std().add_suffix("_std")).join(g.size().rename("n_seeds"))
    summary = summary.reset_index()
    summary["split"] = pd.Categorical(summary["split"], SPLIT_COLUMNS, ordered=True)
    summary = summary.sort_values(["model", "split"])
    summary.to_csv(root / "summary.csv", index=False)

    def fmt(r, m):
        sd = r[f"{m}_std"]
        return f"{r[f'{m}_mean']:.3f}" + (f"±{sd:.3f}" if pd.notna(sd) and r["n_seeds"] > 1 else "")

    view = summary[["model", "split", "n_seeds"]].copy()
    for m in ["fnr", "fpr", "fpr_notinject", "auroc"]:
        view[m.upper()] = [fmt(r, m) for _, r in summary.iterrows()]
    print(view.to_string(index=False))
    print(f"\nSaved {root / 'summary.csv'}")


if __name__ == "__main__":
    main()
