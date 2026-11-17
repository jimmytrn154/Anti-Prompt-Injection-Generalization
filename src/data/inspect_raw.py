"""Inspect the raw downloads and write a report for the team.  Run from the repo root:

    python -m src.data.inspect_raw                 # all sources
    python -m src.data.inspect_raw --lang-sample 0 # skip language detection (faster)

Writes reports/data/raw_inspection.md (post this in the group chat) and
reports/data/raw_inspection.csv. Both contain counts only, no dataset text,
so they can be committed.
"""
from __future__ import annotations

import argparse
import logging
import re
import time
from collections import Counter

import numpy as np
import pandas as pd

from src.data.check_env import TOKENIZER_NAME
from src.data.sources import REPO_ROOT, SOURCES, load_raw, tensortrust_funnel_and_attacks

REPORT_DIR = REPO_ROOT / "reports" / "data"
MIN_TOK, MAX_TOK = 5, 300          # proposal's length filter
SEED = 42


# ------------------------------------------------------------------ helpers
def normalize(s: str) -> str:
    """Comparison key for exact-duplicate counting (stored text is never changed)."""
    return re.sub(r"\s+", " ", s).strip().lower()


def count_tokens(texts: list[str]) -> np.ndarray:
    from transformers import AutoTokenizer
    logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)  # silence >512 warnings
    tok = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    out = []
    for i in range(0, len(texts), 2048):
        enc = tok(texts[i:i + 2048], add_special_tokens=False, truncation=False)["input_ids"]
        out.extend(len(e) for e in enc)
    return np.array(out, dtype=np.int32)


_DETECTOR = None


def detect_languages(texts: list[str]) -> list[str]:
    global _DETECTOR
    from lingua import Language, LanguageDetectorBuilder
    if _DETECTOR is None:
        langs = [Language.ENGLISH, Language.GERMAN, Language.FRENCH, Language.SPANISH, Language.ITALIAN,
                 Language.PORTUGUESE, Language.DUTCH, Language.RUSSIAN, Language.CHINESE, Language.JAPANESE,
                 Language.KOREAN, Language.VIETNAMESE, Language.ARABIC, Language.HINDI, Language.TURKISH]
        _DETECTOR = LanguageDetectorBuilder.from_languages(*langs).build()
    det = _DETECTOR
    if hasattr(det, "detect_languages_in_parallel_of"):
        res = det.detect_languages_in_parallel_of(texts)
    else:
        res = [det.detect_language_of(t) for t in texts]
    return [r.iso_code_639_1.name.lower() if r is not None else "unknown" for r in res]


def describe(name: str, cls: str, df: pd.DataFrame, lang_sample: int) -> dict:
    n = len(df)
    tok = df["n_tokens"].to_numpy()
    keys = df["text"].map(normalize)
    row = {
        "source": name, "class": cls, "rows": n,
        "empty": int((df["text"].str.strip() == "").sum()),
        "exact_dups": int(keys.duplicated().sum()),
        "tok_median": int(np.median(tok)) if n else 0,
        "tok_p95": int(np.percentile(tok, 95)) if n else 0,
        "tok_max": int(tok.max()) if n else 0,
        "in_5_300": int(((tok >= MIN_TOK) & (tok <= MAX_TOK)).sum()),
        "under_5": int((tok < MIN_TOK).sum()),
        "over_300": int((tok > MAX_TOK).sum()),
        "over_512": int((tok > 512).sum()),
        "languages": "",
    }
    if lang_sample and n:
        sample = df["text"].sample(n=min(lang_sample, n), random_state=SEED).tolist()
        c = Counter(detect_languages(sample))
        row["languages"] = ", ".join(f"{k} {v / len(sample):.0%}" for k, v in c.most_common(4))
        if len(sample) < n:
            row["languages"] += f" (sample of {len(sample)})"
    return row


def pct(a: int, b: int) -> str:
    return f"{a / b:.0%}" if b else "-"


# ------------------------------------------------------------------ report
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--lang-sample", type=int, default=3000,
                   help="texts per source/class for language detection (0 = skip)")
    p.add_argument("--only", nargs="+", choices=list(SOURCES), help="inspect only these sources")
    args = p.parse_args()

    t0 = time.time()
    rows, extras, tt_funnel = [], {}, None
    for name in args.only or list(SOURCES):
        print(f"[{name}] loading ...", flush=True)
        try:
            if name == "tensortrust":
                tt_funnel, df = tensortrust_funnel_and_attacks()
            else:
                df = load_raw(name)
        except FileNotFoundError as e:
            print(f"  skipped: {e}")
            continue
        df = df.copy()
        df["n_tokens"] = count_tokens(df["text"].fillna("").tolist())
        for label, cls in [(1, "attack"), (0, "benign")]:
            part = df[df["label"] == label]
            if len(part):
                rows.append(describe(name, cls, part, args.lang_sample))
        if "category" in df:
            extras[name] = df["category"].value_counts()
        if name == "tensortrust":
            extras["tt_access_granted"] = int(df["text"].str.contains(r"access\s*granted", case=False, regex=True).sum())
            extras["tt_kept"] = len(df)
        print(f"  {len(df):,} rows", flush=True)

    if not rows:
        raise SystemExit("Nothing to inspect. Run: python -m src.data.download")
    stats = pd.DataFrame(rows)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    stats.to_csv(REPORT_DIR / "raw_inspection.csv", index=False)

    L = []
    L.append("# Raw data inspection\n")
    L.append(f"Generated by `python -m src.data.inspect_raw` on {pd.Timestamp.now():%Y-%m-%d %H:%M}. "
             f"Token counts use `{TOKENIZER_NAME}` without special tokens. Counts only; no dataset text.\n")

    L.append("## 1. Size and label balance\n")
    L.append("| Source | Class | Rows | Exact dups | Empty | Languages |")
    L.append("|---|---|---:|---:|---:|---|")
    for r in rows:
        L.append(f"| {r['source']} | {r['class']} | {r['rows']:,} | {r['exact_dups']:,} | {r['empty']} | {r['languages'] or '-'} |")
    L.append("\n*Exact dups* = rows whose text repeats another row in the same source and class "
             "(after lowercasing and collapsing whitespace). Near-duplicates are not counted yet.\n")

    L.append("## 2. Length (tokens) vs. the 5-300 token filter\n")
    L.append("| Source | Class | Median | 95th pct | Max | Kept by 5-300 | < 5 | > 300 | > 512 |")
    L.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        L.append(f"| {r['source']} | {r['class']} | {r['tok_median']} | {r['tok_p95']} | {r['tok_max']:,} | "
                 f"{r['in_5_300']:,} ({pct(r['in_5_300'], r['rows'])}) | {r['under_5']:,} | "
                 f"{r['over_300']:,} ({pct(r['over_300'], r['rows'])}) | {r['over_512']:,} |")
    L.append("")

    jb = next((r for r in rows if r["source"] == "jailbreak" and r["class"] == "attack"), None)
    L.append("## 3. Questions for the team\n")
    if jb:
        L.append(f"- **Token cap:** {jb['over_300']:,} of {jb['rows']:,} jailbreak attacks "
                 f"({pct(jb['over_300'], jb['rows'])}) are longer than 300 tokens and would be removed by the "
                 f"proposal's filter; {jb['over_512']:,} are longer than 512 (DistilBERT's limit). "
                 "Options: keep 300, raise to 512, or truncate.")
    if tt_funnel:
        L.append(f"- **Tensor Trust:** {extras.get('tt_kept', 0):,} distinct successful attacks remain before "
                 f"language filtering and sampling down to ~1.2k; {extras.get('tt_access_granted', 0):,} of them "
                 "contain the phrase \"access granted\", which preprocessing will mask.")
    att = sum(r["in_5_300"] for r in rows if r["class"] == "attack" and r["source"] != "tensortrust")
    ben = sum(r["in_5_300"] for r in rows if r["class"] == "benign" and r["source"] in ("deepset", "jailbreak"))
    L.append(f"- **Balance check:** with the 5-300 filter (before de-duplication), deepset + Gandalf + "
             f"jailbreak give {att:,} attacks (Tensor Trust adds up to ~1.2k) and deepset + jailbreak give "
             f"{ben:,} benign; Dolly/Alpaca fill the rest of the ~2.7k benign target plus the ~3k calibration pool.\n")

    if tt_funnel:
        L.append("## 4. Tensor Trust filtering funnel (raw dump v2)\n")
        L.append("| Step | Remaining |")
        L.append("|---|---:|")
        for step, n in tt_funnel:
            L.append(f"| {step} | {n:,} |")
        L.append("")

    if "dolly" in extras:
        L.append("## 5. Dolly categories (for category matching)\n")
        L.append("| Category | Rows |")
        L.append("|---|---:|")
        for k, v in extras["dolly"].items():
            L.append(f"| {k} | {v:,} |")
        L.append("")

    report = "\n".join(L)
    (REPORT_DIR / "raw_inspection.md").write_text(report, encoding="utf-8")
    print("\n" + report)
    print(f"\nSaved {REPORT_DIR / 'raw_inspection.md'} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
