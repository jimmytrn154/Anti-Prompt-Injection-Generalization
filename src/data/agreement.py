"""300-sample double annotation for Cohen's kappa.  Run from the repo root:

  1) after B's automatic labels exist, draw a sample stratified by source x strategy
     (optionally with ~30 of B's NOT_ATTACK rows mixed in, to check the exclusion decision too):
       python -m src.data.agreement sample --labels data/processed/labels_v1.parquet \
              --excluded data/processed/not_attack_exclusions.csv
     -> data/processed/agreement/agreement_300_annotator1.csv   (B)
        data/processed/agreement/agreement_300_annotator2.csv   (A)
     Both files are BLIND: they contain the text but not the automatic label, so neither
     annotator is anchored by it. Each person fills only their own file, without looking at the other.

  2) when both files are filled in:
       python -m src.data.agreement kappa --labels data/processed/labels_v1.parquet \
              --excluded data/processed/not_attack_exclusions.csv
     -> reports/data/agreement_report.md                         (kappa, confusion matrix; counts only)
        data/processed/agreement/agreement_300_disagreements.csv (rows to discuss and adjudicate)

After adjudication, B writes the agreed label into labels_v1 for those ids, with
verified_by = the annotators' handles (label_method = "manual" if the label changed, else unchanged).
"""
from __future__ import annotations

import argparse
import warnings

import numpy as np
import pandas as pd

from src.data.schema import STRATEGIES, parse_secondary, validate_labels
from src.data.sources import REPO_ROOT

PROC_DIR = REPO_ROOT / "data" / "processed"
OUT_DIR = PROC_DIR / "agreement"
REPORT = REPO_ROOT / "reports" / "data" / "agreement_report.md"
N_SAMPLE = 300
SEED = 7
FILL = ["strategy_primary", "strategy_secondary", "ambiguous", "obfuscated", "notes"]


def _load_labels(path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    from pathlib import Path
    clean = pd.read_parquet(PROC_DIR / "clean_v0.parquet")
    p = Path(path)
    p = p if p.is_absolute() else REPO_ROOT / p
    labels = pd.read_csv(p, encoding="utf-8-sig") if p.suffix == ".csv" else pd.read_parquet(p)
    labels = validate_labels(labels, clean)
    labels = labels.merge(clean[["id", "source", "text"]], on="id", how="left")
    return labels, clean


def allocate(cells: pd.Series, n: int) -> pd.Series:
    """Spread n samples over cells: an equal floor per cell (capped by cell size), rest proportional."""
    alloc = np.minimum(cells, n // len(cells))
    while alloc.sum() < n and (alloc < cells).any():
        room = cells - alloc
        share = room / room.sum() * (n - alloc.sum())
        add = np.minimum(room, np.maximum(1, np.floor(share))).astype(int)
        alloc = alloc + add
        if alloc.sum() > n:                              # trim the overshoot from the largest cells
            for idx in alloc.sort_values(ascending=False).index:
                if alloc.sum() == n:
                    break
                alloc[idx] -= 1
    return alloc.astype(int)


def _load_excluded(path: str | None, clean: pd.DataFrame) -> pd.DataFrame:
    """B's NOT_ATTACK list (id, source, reason) with texts; empty if no path is given."""
    if not path:
        return pd.DataFrame(columns=["id", "source", "text"])
    from pathlib import Path
    p = Path(path)
    p = p if p.is_absolute() else REPO_ROOT / p
    ex = pd.read_csv(p, encoding="utf-8-sig")
    return ex[["id"]].merge(clean[["id", "source", "text"]], on="id", how="inner")


def sample(labels_path: str, excluded_path: str | None = None, n_excluded: int = 0) -> None:
    labels, clean = _load_labels(labels_path)
    rng = np.random.default_rng(SEED)
    n_attack = N_SAMPLE - n_excluded if excluded_path else N_SAMPLE
    cells = labels.groupby(["source", "strategy_primary"]).size()
    alloc = allocate(cells, min(n_attack, len(labels)))
    picked = []
    for (src, strat), k in alloc.items():
        idx = labels.index[(labels["source"] == src) & (labels["strategy_primary"] == strat)]
        picked.extend(rng.choice(idx, size=int(k), replace=False))
    parts = [labels.loc[picked, ["id", "source", "text"]]]
    if excluded_path and n_excluded:
        ex = _load_excluded(excluded_path, clean)          # B's NOT_ATTACK rows: checks the exclusion decision
        ex_alloc = allocate(ex.groupby("source").size(), min(n_excluded, len(ex)))
        for src, k in ex_alloc.items():
            parts.append(ex[ex["source"] == src].sample(n=int(k), random_state=SEED)[["id", "source", "text"]])
        print("NOT_ATTACK rows mixed in (blind):", ex_alloc.to_dict())
    s = pd.concat(parts).sample(frac=1, random_state=SEED).reset_index(drop=True)
    for c in FILL:
        s[c] = ""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for who in ("annotator1", "annotator2"):
        s.to_csv(OUT_DIR / f"agreement_{len(s)}_{who}.csv", index=False, encoding="utf-8-sig")
    print(f"Drew {len(s)} rows (source x strategy of the automatic labels):\n")
    print(alloc.unstack(fill_value=0).to_string())
    print(f"\nWrote blind files to {OUT_DIR}. Fill: {FILL}. strategy_primary: S1 / S2 / S3 / NOT_ATTACK; "
          "strategy_secondary e.g. 'S2;S3'; ambiguous/obfuscated: TRUE/FALSE.")


def _read_annot(who: str) -> pd.DataFrame:
    f = sorted(OUT_DIR.glob(f"agreement_*_{who}.csv"))
    if not f:
        raise SystemExit(f"no file for {who} in {OUT_DIR}; run the `sample` step first")
    d = pd.read_csv(f[-1], encoding="utf-8-sig", dtype=str).fillna("")
    d["strategy_primary"] = d["strategy_primary"].str.strip().str.upper()
    for c in ("ambiguous", "obfuscated"):
        d[c] = d[c].str.strip().str.lower().isin(["true", "1", "yes", "y"])
    d["strategy_secondary"] = d["strategy_secondary"].map(lambda v: ";".join(sorted(parse_secondary(v))))
    missing = d["strategy_primary"].eq("") | ~d["strategy_primary"].isin(STRATEGIES + ["NOT_ATTACK"])
    if missing.any():
        raise SystemExit(f"{who}: {int(missing.sum())} rows have no valid strategy_primary (S1/S2/S3/NOT_ATTACK), "
                         f"e.g. ids {d.loc[missing, 'id'].head(3).tolist()}")
    return d


def kappa(labels_path: str, excluded_path: str | None = None) -> None:
    from sklearn.metrics import cohen_kappa_score
    warnings.filterwarnings("ignore", module="sklearn")   # constant fields give kappa = nan; reported as such
    a, b = _read_annot("annotator1"), _read_annot("annotator2")
    m = a.merge(b, on="id", suffixes=("_1", "_2"))
    if len(m) != len(a) or len(m) != len(b):
        raise SystemExit("the two files do not contain the same ids")
    labels, clean = _load_labels(labels_path)
    m = m.merge(labels[["id", "strategy_primary", "label_method"]].rename(
        columns={"strategy_primary": "auto"}), on="id", how="left")
    excluded_ids = set(_load_excluded(excluded_path, clean)["id"])
    m.loc[m["id"].isin(excluded_ids), "auto"] = "NOT_ATTACK"

    def k(c: str) -> float:
        return cohen_kappa_score(m[f"{c}_1"].astype(str), m[f"{c}_2"].astype(str))

    L = ["# Inter-annotator agreement (300-sample check)\n",
         f"Generated by `python -m src.data.agreement kappa` on {pd.Timestamp.now():%Y-%m-%d %H:%M}. "
         f"{len(m)} attacks annotated independently by two people (blind to the automatic labels).\n",
         "## 1. Cohen's kappa\n", "| Field | Kappa | Raw agreement |", "|---|---:|---:|"]
    for c in ["strategy_primary", "ambiguous", "obfuscated"]:
        L.append(f"| {c} | {k(c):.2f} | {(m[f'{c}_1'] == m[f'{c}_2']).mean():.0%} |")
    att1, att2 = m["strategy_primary_1"] != "NOT_ATTACK", m["strategy_primary_2"] != "NOT_ATTACK"
    if (~att1).any() or (~att2).any():
        L.append(f"| attack vs NOT_ATTACK | {cohen_kappa_score(att1, att2):.2f} | {(att1 == att2).mean():.0%} |")
    L.append("\nRule of thumb (Landis & Koch): 0.61-0.80 substantial, > 0.80 almost perfect.\n")

    L.append("## 2. strategy_primary: annotator 1 (rows) vs annotator 2 (columns)\n")
    cats = STRATEGIES + (["NOT_ATTACK"] if (~att1).any() or (~att2).any() else [])
    ct = pd.crosstab(m["strategy_primary_1"], m["strategy_primary_2"]).reindex(
        index=cats, columns=cats, fill_value=0)
    L.append("| | " + " | ".join(ct.columns) + " |")
    L.append("|---|" + "---:|" * len(ct.columns))
    for s, r in ct.iterrows():
        L.append(f"| **{s}** | " + " | ".join(str(v) for v in r) + " |")
    L.append("")

    L.append("## 3. Kappa per source\n")
    L.append("| Source | Rows | Kappa (strategy_primary) |")
    L.append("|---|---:|---:|")
    for src, g in m.groupby("source_1"):
        kk = cohen_kappa_score(g["strategy_primary_1"], g["strategy_primary_2"]) \
            if g["strategy_primary_1"].nunique() > 1 or g["strategy_primary_2"].nunique() > 1 else float("nan")
        L.append(f"| {src} | {len(g)} | {kk:.2f} |")
    L.append("")

    agree = m[m["strategy_primary_1"] == m["strategy_primary_2"]]
    L.append("## 4. Automatic labels vs. human agreement\n")
    L.append(f"On the {len(agree)} rows where both annotators agree, the automatic label matches them in "
             f"{(agree['auto'] == agree['strategy_primary_1']).mean():.0%} of cases.\n")
    if "label_method" in agree:
        L.append("| Method | Rows | Matches humans |")
        L.append("|---|---:|---:|")
        for meth, g in agree.groupby(agree["label_method"].fillna("unknown")):
            L.append(f"| {meth} | {len(g)} | {(g['auto'] == g['strategy_primary_1']).mean():.0%} |")
        L.append("")

    dis = m[(m["strategy_primary_1"] != m["strategy_primary_2"]) | (m["ambiguous_1"] != m["ambiguous_2"])]
    dis = dis[["id", "source_1", "text_1", "strategy_primary_1", "strategy_primary_2", "ambiguous_1", "ambiguous_2",
               "notes_1", "notes_2", "auto"]].rename(columns={"source_1": "source", "text_1": "text"})
    dis["agreed_strategy"] = ""
    dis.to_csv(OUT_DIR / "agreement_300_disagreements.csv", index=False, encoding="utf-8-sig")
    L.append(f"## 5. Disagreements\n\n{len(dis)} rows written to "
             "`data/processed/agreement/agreement_300_disagreements.csv` for discussion; fill `agreed_strategy`.\n")
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("step", choices=["sample", "kappa"])
    p.add_argument("--labels", required=True, help="B's labels file (.parquet or .csv)")
    p.add_argument("--excluded", help="B's NOT_ATTACK list (csv with id); mixes such rows into the blind sample")
    p.add_argument("--n-excluded", type=int, default=30, help="NOT_ATTACK rows to mix in (default 30)")
    a = p.parse_args()
    sample(a.labels, a.excluded, a.n_excluded) if a.step == "sample" else kappa(a.labels, a.excluded)


if __name__ == "__main__":
    main()
