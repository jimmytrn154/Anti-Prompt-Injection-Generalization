"""Registry of APIG data sources and loaders for the raw downloads.

`download.py` saves every source untouched under data/raw/<source>/.
`load_raw(source)` reads it back as one tidy table:

    native_id | original_split | text | label

so later steps (inspection, unification, de-duplication) never need to know
each dataset's own column names.
"""
from __future__ import annotations

import bz2
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = REPO_ROOT / "data" / "raw"

TENSORTRUST_URL = (
    "https://raw.githubusercontent.com/HumanCompatibleAI/tensor-trust-data/main/"
    "raw-data/v2/raw_dump_attacks.jsonl.bz2"
)
TENSORTRUST_FILE = "raw_dump_attacks_v2.jsonl.bz2"


@dataclass(frozen=True)
class Source:
    name: str          # our short name (matches schema.SOURCES)
    origin: str        # Hugging Face dataset id or URL
    kind: str          # "hf" or "url"
    classes: str       # "A", "B" or "A+B"
    licence: str
    citation: str      # reference number in the proposal


SOURCES: dict[str, Source] = {
    s.name: s
    for s in [
        Source("deepset", "deepset/prompt-injections", "hf", "A+B", "Apache-2.0", "[11]"),
        Source("gandalf", "Lakera/gandalf_ignore_instructions", "hf", "A", "MIT", "[12]"),
        Source("jailbreak", "jackhhao/jailbreak-classification", "hf", "A+B", "Apache-2.0", "[13]"),
        Source("tensortrust", TENSORTRUST_URL, "url", "A", "see github.com/HumanCompatibleAI/tensor-trust-data", "[4]"),
        Source("dolly", "databricks/databricks-dolly-15k", "hf", "B", "CC-BY-SA-3.0", "[14]"),
        Source("alpaca", "tatsu-lab/alpaca", "hf", "B", "CC-BY-NC-4.0", "[15]"),
        Source("notinject", "leolee99/NotInject", "hf", "B (hard)", "MIT", "[16]"),
    ]
}

_SPLIT_ORDER = {"train": 0, "validation": 1, "test": 2}


# ------------------------------------------------------------------ HF sources
def _read_hf_splits(name: str) -> pd.DataFrame:
    """Concatenate data/raw/<name>/<split>.parquet in a fixed split order."""
    folder = RAW_DIR / name
    files = sorted(folder.glob("*.parquet"), key=lambda p: (_SPLIT_ORDER.get(p.stem, 9), p.stem))
    if not files:
        raise FileNotFoundError(f"no raw files for {name!r} in {folder}. Run: python -m src.data.download")
    parts = []
    for f in files:
        df = pd.read_parquet(f)
        df.insert(0, "original_split", f.stem)
        parts.append(df)
    df = pd.concat(parts, ignore_index=True)
    df.insert(0, "native_id", range(len(df)))  # stable: fixed file order + fixed row order
    return df


def _tidy(df: pd.DataFrame, text: pd.Series, label) -> pd.DataFrame:
    return pd.DataFrame({
        "native_id": df["native_id"],
        "original_split": df["original_split"],
        "text": text.astype("string"),
        "label": pd.Series(label, index=df.index).astype("int8"),
    })


def _load_deepset() -> pd.DataFrame:
    df = _read_hf_splits("deepset")
    return _tidy(df, df["text"], df["label"])          # label: 1 = injection, 0 = benign


def _load_gandalf() -> pd.DataFrame:
    df = _read_hf_splits("gandalf")
    return _tidy(df, df["text"], 1)                     # all attacks


def _load_jailbreak() -> pd.DataFrame:
    df = _read_hf_splits("jailbreak")
    return _tidy(df, df["prompt"], (df["type"] == "jailbreak").astype(int))


def _load_dolly() -> pd.DataFrame:
    df = _read_hf_splits("dolly")
    # instruction only; the optional `context` is long reference text (team decision pending)
    out = _tidy(df, df["instruction"], 0)
    out["category"] = df["category"].astype("string")
    return out


def _load_alpaca() -> pd.DataFrame:
    df = _read_hf_splits("alpaca")
    inp = df["input"].fillna("").str.strip()
    text = df["instruction"].str.strip() + inp.map(lambda s: f"\n{s}" if s else "")
    return _tidy(df, text, 0)


def _load_notinject() -> pd.DataFrame:
    df = _read_hf_splits("notinject")
    out = _tidy(df, df["prompt"], 0)
    out["category"] = df["category"].astype("string")
    return out


# ------------------------------------------------------------------ Tensor Trust
def tensortrust_funnel_and_attacks(path: Path | None = None) -> tuple[list[tuple[str, int]], pd.DataFrame]:
    """Stream the raw Tensor Trust dump and keep successful attacks only.

    Filters, in order (counts after each step are returned for the data card):
      1. attacker_input present (ToS-violating text was removed by the authors)
      2. not a self-attack (players testing their own defense / sandbox)
      3. successful (`output_is_access_granted` is True)
      4. not a plain password entry (input is not, and does not contain, the access code)
      5. distinct attack text (first occurrence kept, by attack_id order)

    Language filtering and "access granted" masking happen later, in preprocessing.
    """
    path = path or RAW_DIR / "tensortrust" / TENSORTRUST_FILE
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run: python -m src.data.download --only tensortrust")

    counts = {"all attacks": 0, "has attacker_input": 0, "not self-attack": 0,
              "successful": 0, "not a password entry": 0}
    ids, texts, seen = [], [], set()
    with bz2.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            counts["all attacks"] += 1
            r = json.loads(line)
            text = r.get("attacker_input")
            if not text or not str(text).strip():
                continue
            counts["has attacker_input"] += 1
            if r.get("is_self_attack"):
                continue
            counts["not self-attack"] += 1
            if not r.get("output_is_access_granted"):
                continue
            counts["successful"] += 1
            code = (r.get("access_code") or "").strip().lower()
            low = text.strip().lower()
            # exact password, or a longer code pasted inside (very short codes would match by chance)
            if code and (low == code or (len(code) >= 5 and code in low)):
                continue
            counts["not a password entry"] += 1
            if text in seen:
                continue
            seen.add(text)
            ids.append(int(r["attack_id"]))
            texts.append(text)

    funnel = list(counts.items()) + [("distinct text", len(texts))]
    df = pd.DataFrame({
        "native_id": ids,
        "original_split": "raw_v2",
        "text": pd.Series(texts, dtype="string"),
        "label": pd.Series(1, index=range(len(texts)), dtype="int8"),
    })
    return funnel, df


def _load_tensortrust() -> pd.DataFrame:
    return tensortrust_funnel_and_attacks()[1]


_LOADERS = {
    "deepset": _load_deepset, "gandalf": _load_gandalf, "jailbreak": _load_jailbreak,
    "tensortrust": _load_tensortrust, "dolly": _load_dolly, "alpaca": _load_alpaca,
    "notinject": _load_notinject,
}


def load_raw(name: str) -> pd.DataFrame:
    """Load one raw source as columns: native_id, original_split, text, label (+ category if any)."""
    if name not in _LOADERS:
        raise ValueError(f"unknown source {name!r}; choose from {list(_LOADERS)}")
    return _LOADERS[name]()
