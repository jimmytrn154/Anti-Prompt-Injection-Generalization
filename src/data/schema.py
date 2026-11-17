"""Shared data schema for APIG (proposal v0.1, to be agreed by A and B).

Every script reads and writes one table: data/processed/apig_v<N>.parquet.
Run `validate(df)` before saving or after loading; it raises on any violation.

See data/SCHEMA.md for the human-readable description.
"""
from __future__ import annotations

import pandas as pd

# ---------------------------------------------------------------- vocabularies
SOURCES = [
    "deepset",       # deepset/prompt-injections
    "gandalf",       # Lakera/gandalf_ignore_instructions
    "jailbreak",     # jackhhao/jailbreak-classification
    "tensortrust",   # Tensor Trust raw dump
    "dolly",         # databricks/databricks-dolly-15k
    "alpaca",        # tatsu-lab/alpaca
    "notinject",     # leolee99/NotInject
]
ATTACK_SOURCES = ["deepset", "gandalf", "jailbreak", "tensortrust"]
STRATEGIES = ["S1", "S2", "S3"]
TYPES = ["injection", "jailbreak"]
SUBSETS = ["main", "calibration", "notinject"]
SPLIT_VALUES = ["train", "val", "test", "excluded"]

# One column per split scheme / fold. Value = role of the row in that fold.
SPLIT_COLUMNS = (
    ["split_random"]
    + [f"loso_{s}" for s in STRATEGIES]
    + [f"wsloso_{s}" for s in STRATEGIES]          # within-source LOSO (Tensor Trust)
    + [f"lodo_{src}" for src in ATTACK_SOURCES]
)

# ---------------------------------------------------------------- columns
# name -> (pandas dtype, filled by, description)
COLUMNS = {
    "id":                 ("string",  "A", "Stable unique ID: <source>_<native id or row index>"),
    "text":               ("string",  "A", "User-turn text (Tensor Trust: 'access granted' masked)"),
    "label":              ("int8",    "A", "1 = attack, 0 = benign"),
    "source":             ("string",  "A", f"One of {SOURCES}"),
    "subset":             ("string",  "A", f"One of {SUBSETS}; calibration/notinject are never trained on"),
    "original_split":     ("string",  "A", "Split name in the source dataset (provenance only)"),
    "language":           ("string",  "A", "ISO 639-1 code, e.g. 'en', 'de'"),
    "n_tokens":           ("Int32",   "A", "Token count with the shared tokenizer (see check_env.py)"),
    "category":           ("string",  "A", "Category given by the source (Dolly, NotInject); else empty"),
    "type":               ("string",  "A→B", "injection / jailbreak (attacks only; A pre-fills, B confirms)"),
    "strategy_primary":   ("string",  "B", "S1 / S2 / S3 (attacks only)"),
    "strategy_secondary": ("object",  "B", "List of zero or more of S1-S3 (attacks only)"),
    "ambiguous":          ("boolean", "B", "True if the primary strategy is unclear (excluded from LOSO tests)"),
    "obfuscated":         ("boolean", "B", "Natural obfuscation present in the original text (attacks only)"),
    **{c: ("string", "A", f"Role in split '{c}': one of {SPLIT_VALUES}") for c in SPLIT_COLUMNS},
}
ATTACK_ONLY = ["type", "strategy_primary", "strategy_secondary", "ambiguous", "obfuscated"]


# ---------------------------------------------------------------- helpers
def empty_frame() -> pd.DataFrame:
    """An empty table with every schema column and dtype."""
    return pd.DataFrame({c: pd.Series(dtype=d) for c, (d, _, _) in COLUMNS.items()})


def make_id(source: str, native_id) -> str:
    """Build a stable ID, e.g. make_id('gandalf', 42) -> 'gandalf_00042'."""
    if source not in SOURCES:
        raise ValueError(f"unknown source {source!r}")
    if isinstance(native_id, int):
        return f"{source}_{native_id:05d}"
    return f"{source}_{native_id}"


def _check_values(df: pd.DataFrame, col: str, allowed, errors: list[str]) -> None:
    bad = df[col].dropna()
    bad = bad[~bad.isin(allowed)]
    if len(bad):
        errors.append(f"{col}: {len(bad)} invalid values, e.g. {bad.unique()[:5].tolist()}")


def validate(df: pd.DataFrame, labelled: bool = False, split: bool = False) -> None:
    """Check a table against the schema. Raises ValueError listing every problem.

    labelled=True  -> also require B's strategy labels on every attack.
    split=True     -> also require every split column to be filled.
    """
    errors: list[str] = []

    missing = [c for c in COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"missing columns: {missing}")
    extra = [c for c in df.columns if c not in COLUMNS]
    if extra:
        errors.append(f"unexpected columns (add them to schema.py first): {extra}")

    # identity
    if df["id"].isna().any():
        errors.append("id: null values")
    dup = df["id"][df["id"].duplicated()]
    if len(dup):
        errors.append(f"id: {len(dup)} duplicates, e.g. {dup.head(3).tolist()}")
    if df["text"].isna().any() or (df["text"].str.strip() == "").any():
        errors.append("text: null or empty values")

    # vocabularies
    _check_values(df, "label", [0, 1], errors)
    _check_values(df, "source", SOURCES, errors)
    _check_values(df, "subset", SUBSETS, errors)
    _check_values(df, "type", TYPES, errors)
    _check_values(df, "strategy_primary", STRATEGIES, errors)
    for c in SPLIT_COLUMNS:
        _check_values(df, c, SPLIT_VALUES, errors)

    sec = df["strategy_secondary"].dropna()
    bad_sec = sec[~sec.apply(lambda v: isinstance(v, (list, tuple)) or hasattr(v, "tolist"))]
    if len(bad_sec):
        errors.append("strategy_secondary: must be a list (use [] for none)")
    else:
        flat = {s for v in sec for s in list(v)}
        if flat - set(STRATEGIES):
            errors.append(f"strategy_secondary: invalid values {sorted(flat - set(STRATEGIES))}")

    # logical consistency
    benign = df["label"] == 0
    for c in ATTACK_ONLY:
        if df.loc[benign, c].notna().any():
            errors.append(f"{c}: must be empty for benign rows")
    if (df.loc[df["subset"].isin(["calibration", "notinject"]), "label"] != 0).any():
        errors.append("calibration/notinject rows must all be benign (label 0)")
    if (df.loc[df["source"] == "notinject", "subset"] != "notinject").any():
        errors.append("NotInject rows must have subset='notinject'")

    # leakage guard: held-out pools may never be trained or validated on
    held_out = df["subset"].isin(["calibration", "notinject"])
    for c in SPLIT_COLUMNS:
        if df.loc[held_out, c].isin(["train", "val"]).any():
            errors.append(f"{c}: calibration/notinject rows assigned to train/val")

    if labelled:
        attacks = df["label"] == 1
        if df.loc[attacks, "strategy_primary"].isna().any():
            errors.append("strategy_primary: missing on some attacks")
        if df.loc[attacks, "type"].isna().any():
            errors.append("type: missing on some attacks")

    if split:
        for c in SPLIT_COLUMNS:
            if df[c].isna().any():
                errors.append(f"{c}: {df[c].isna().sum()} rows without a split role")

    if errors:
        raise ValueError("schema validation failed:\n  - " + "\n  - ".join(errors))


if __name__ == "__main__":
    # Print the schema as a table (handy for the README / SCHEMA.md).
    print(f"{'column':<20} {'dtype':<8} {'by':<4} description")
    for c, (d, who, desc) in COLUMNS.items():
        print(f"{c:<20} {d:<8} {who:<4} {desc}")
