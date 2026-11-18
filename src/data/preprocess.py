"""Phase 3: clean, filter and de-duplicate all sources into one table.  Run from the repo root:

    python -m src.data.preprocess              # full run (~15 min on CPU)
    python -m src.data.preprocess --pool-only  # only rebuild the labelling pool from saved outputs

Steps (counts after each are written to the report):
  1. load every raw source (Dolly = instruction + context passage)
  2. Tensor Trust: remove the game phrase "access granted"
  3. length filter: MIN_TOK-MAX_TOK tokens (team decision: 5-500)
  4. Alpaca: random pre-sample (we need only a few thousand benign prompts)
  5. language filter (EN everywhere, EN+DE for deepset; NotInject kept as published)
  6. exact duplicates across all sources (same text with different labels -> all copies dropped)
  7. near-duplicates (embedding cosine > NEAR_DUP), within each class
  8. Tensor Trust candidate pool for labelling, by k-means cluster sampling

Outputs (data/processed/ is git-ignored; reports/ can be committed):
  data/processed/clean_v0.parquet       every surviving row, schema-valid (no strategy labels or splits yet)
  data/processed/labeling_pool_v0.*     attacks for B to label (.parquet + Excel-friendly .csv)
  data/processed/removal_log.csv        every removed row: id, step, reason, duplicate_of
  reports/data/preprocess_report.md     funnel, length bins, language checks (counts only)
"""
from __future__ import annotations

import logging
import re
import time

import numpy as np
import pandas as pd

from src.data.check_env import EMBEDDING_MODEL, TOKENIZER_NAME
from src.data.schema import COLUMNS, empty_frame, make_id, validate
from src.data.sources import REPO_ROOT, load_raw

# ------------------------------------------------------------------ settings (change here only)
MIN_TOK, MAX_TOK = 5, 500        # team decision Nov 17: 300 -> 500 (keeps 2/3 of jailbreaks, < 512 for every tokenizer)
NEAR_DUP = 0.90                  # embedding cosine above which two texts count as duplicates (proposal)
ALPACA_POOL = 20_000             # Alpaca rows kept before language filtering / dedup
TT_POOL = 3_000                  # Tensor Trust candidates handed to B for labelling (final sample <= ~1.2k)
TT_CLUSTERS = 300
SEED = 42
MASK_WITH = ""                   # replacement for "access granted"; a placeholder token would itself become a shortcut
ALLOWED_LANG = {"deepset": {"en", "de"}}   # every other source: {"en"}; NotInject: no filter
NO_LANG_FILTER = {"notinject"}
# priority when the same text appears in several sources: keep the earliest in this list
PRIORITY = ["notinject", "deepset", "gandalf", "jailbreak", "tensortrust", "dolly", "alpaca"]

PROC_DIR = REPO_ROOT / "data" / "processed"
REPORT_DIR = REPO_ROOT / "reports" / "data"
LENGTH_BINS = [MIN_TOK, 50, 100, 200, 300, MAX_TOK + 1]


# ------------------------------------------------------------------ building blocks (patchable in tests)
def count_tokens(texts: list[str]) -> np.ndarray:
    from transformers import AutoTokenizer
    logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)
    tok = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    out = []
    for i in range(0, len(texts), 2048):
        out.extend(len(e) for e in tok(texts[i:i + 2048], add_special_tokens=False, truncation=False)["input_ids"])
    return np.array(out, dtype=np.int32)


_REPEAT_TOKEN = re.compile(r"(\S+)(?:\s+\1){2,}")       # "ü ü ü ü" -> "ü"
_REPEAT_CHUNK = re.compile(r"(\w{2,}?)\1{2,}")          # "artisanlibartisanlib..." -> "artisanlib"
_LATIN = re.compile(r"[A-Za-z]")
_OTHER_SCRIPT = re.compile(r"[Ͱ-ϿЀ-ӿ]")  # Greek / Cyrillic look-alikes


def language_view(text: str) -> str | None:
    """Text as the language detector should see it, or None if it should be 'und'.

    Tensor Trust attacks often pad with repeated characters or glitch tokens ("ü ü ü", "artisanlib..."),
    which fools the detector into German/Italian. We collapse repeats before detecting, and treat
    words mixing Latin with Cyrillic/Greek letters as homoglyph obfuscation (language undetermined,
    kept) instead of Russian. The stored text is never changed."""
    if any(_LATIN.search(w) and _OTHER_SCRIPT.search(w) for w in text.split()):
        return None
    v = _REPEAT_CHUNK.sub(r"\1", _REPEAT_TOKEN.sub(r"\1", text))
    return v if sum(ch.isalpha() for ch in v) >= 20 else None


def detect_languages(texts: list[str]) -> list[str]:
    """ISO 639-1 code, or 'und' when the detector is not confident (short, padded or obfuscated text)."""
    views = [language_view(t) for t in texts]
    todo = [i for i, v in enumerate(views) if v is not None]
    codes = _detect([views[i] for i in todo])
    out = ["und"] * len(texts)
    for i, c in zip(todo, codes):
        out[i] = c
    return out


def _detect(texts: list[str]) -> list[str]:
    from lingua import Language, LanguageDetectorBuilder
    langs = [Language.ENGLISH, Language.GERMAN, Language.FRENCH, Language.SPANISH, Language.ITALIAN,
             Language.PORTUGUESE, Language.DUTCH, Language.RUSSIAN, Language.CHINESE, Language.JAPANESE,
             Language.KOREAN, Language.VIETNAMESE, Language.ARABIC, Language.HINDI, Language.TURKISH]
    det = LanguageDetectorBuilder.from_languages(*langs).with_minimum_relative_distance(0.25).build()
    res = det.detect_languages_in_parallel_of(texts) if hasattr(det, "detect_languages_in_parallel_of") \
        else [det.detect_language_of(t) for t in texts]
    return [r.iso_code_639_1.name.lower() if r is not None else "und" for r in res]


def embed(texts: list[str]) -> np.ndarray:
    """L2-normalised sentence embeddings (float32)."""
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(EMBEDDING_MODEL)
    return model.encode(texts, batch_size=64, normalize_embeddings=True, show_progress_bar=True,
                        convert_to_numpy=True).astype(np.float32)


def norm_key(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


def near_dup_groups(emb: np.ndarray, threshold: float, chunk: int = 1024) -> np.ndarray:
    """Greedy de-duplication in row order (rows are pre-sorted by source priority).

    A row is dropped only if it is directly similar (cosine > threshold) to a row already kept;
    unlike union-find this avoids "chaining" (A~B, B~C merging A with a dissimilar C).
    Returns, for each row, the index of the kept row it duplicates (itself if kept)."""
    n = len(emb)
    rep = np.arange(n)
    kept: list[int] = []
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        block = emb[start:stop]
        if kept:                                               # against rows kept in earlier blocks
            s = block @ emb[kept].T
            best = s.argmax(axis=1)
            hit = s[np.arange(len(block)), best] > threshold
            rep[start:stop][hit] = np.array(kept)[best[hit]]
        else:
            hit = np.zeros(len(block), dtype=bool)
        inner = block @ block.T                                # within the block, sequentially
        block_kept: list[int] = []
        for i in range(len(block)):
            if hit[i]:
                continue
            if block_kept:
                j = int(np.argmax(inner[i, block_kept]))
                if inner[i, block_kept[j]] > threshold:
                    rep[start + i] = start + block_kept[j]
                    continue
            block_kept.append(i)
        kept.extend(start + i for i in block_kept)
    return rep


def cluster_sample(emb: np.ndarray, k: int, n_clusters: int, seed: int) -> np.ndarray:
    """Pick k diverse rows: k-means on embeddings, then an equal share from every cluster."""
    from sklearn.cluster import MiniBatchKMeans
    n = len(emb)
    if n <= k:
        return np.arange(n)
    labels = MiniBatchKMeans(n_clusters=min(n_clusters, n), random_state=seed, n_init=3).fit_predict(emb)
    rng = np.random.default_rng(seed)
    per = int(np.ceil(k / (labels.max() + 1)))
    chosen = []
    for c in rng.permutation(np.unique(labels)):
        idx = np.flatnonzero(labels == c)
        chosen.extend(rng.choice(idx, size=min(per, len(idx)), replace=False))
    chosen = np.array(chosen)
    if len(chosen) > k:
        chosen = rng.choice(chosen, size=k, replace=False)
    elif len(chosen) < k:
        rest = np.setdiff1d(np.arange(n), chosen)
        chosen = np.concatenate([chosen, rng.choice(rest, size=k - len(chosen), replace=False)])
    return np.sort(chosen)


# ------------------------------------------------------------------ pipeline
class Funnel:
    def __init__(self) -> None:
        self.rows: list[dict] = []
        self.removed: list[pd.DataFrame] = []

    def record(self, step: str, df: pd.DataFrame) -> None:
        c = df.groupby(["source", "label"]).size()
        self.rows.append({"step": step, **{f"{s}|{l}": int(v) for (s, l), v in c.items()}})
        print(f"  {step:<28} {len(df):>8,} rows", flush=True)

    def drop(self, df: pd.DataFrame, mask: pd.Series, step: str, reason, duplicate_of=None) -> pd.DataFrame:
        gone = df.loc[mask, ["id", "source", "label"]].copy()
        gone["step"], gone["reason"] = step, reason if isinstance(reason, str) else reason[mask]
        gone["duplicate_of"] = None if duplicate_of is None else duplicate_of[mask]
        self.removed.append(gone)
        return df.loc[~mask].reset_index(drop=True)


def run() -> None:
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    f = Funnel()

    print("[1] loading raw sources", flush=True)
    parts = []
    for src in PRIORITY:
        d = load_raw(src)
        d["source"] = src
        parts.append(d)
    df = pd.concat(parts, ignore_index=True)
    df["id"] = [make_id(s, int(n)) for s, n in zip(df["source"], df["native_id"])]
    df["text"] = df["text"].fillna("").astype(str)
    if "category" not in df:
        df["category"] = pd.NA
    df = f.drop(df, df["text"].str.strip() == "", "load", "empty text")
    f.record("raw", df)

    print("[2] masking 'access granted' in Tensor Trust", flush=True)
    tt = df["source"] == "tensortrust"
    pat = re.compile(r"access\s*granted", re.IGNORECASE)
    n_masked = int(df.loc[tt, "text"].str.contains(pat).sum())
    df.loc[tt, "text"] = (df.loc[tt, "text"].str.replace(pat, MASK_WITH, regex=True)
                          .str.replace(r"[ \t]{2,}", " ", regex=True).str.strip())
    df = f.drop(df, df["text"].str.strip() == "", "mask", "empty after masking")

    print("[3] token counts and length filter", flush=True)
    df["n_tokens"] = count_tokens(df["text"].tolist())
    df = f.drop(df, df["n_tokens"] < MIN_TOK, "length", f"< {MIN_TOK} tokens")
    df = f.drop(df, df["n_tokens"] > MAX_TOK, "length", f"> {MAX_TOK} tokens")
    f.record(f"length {MIN_TOK}-{MAX_TOK}", df)

    print(f"[4] pre-sampling Alpaca to {ALPACA_POOL:,}", flush=True)
    alp = np.flatnonzero(df["source"].to_numpy() == "alpaca")
    if len(alp) > ALPACA_POOL:
        drop_idx = np.setdiff1d(alp, rng.choice(alp, size=ALPACA_POOL, replace=False))
        m = pd.Series(False, index=df.index); m.iloc[drop_idx] = True
        df = f.drop(df, m, "alpaca_sample", "not in random Alpaca pool")
    f.record("alpaca pre-sample", df)

    print("[5] language detection and filter", flush=True)
    df["language"] = detect_languages(df["text"].tolist())
    allowed = df["source"].map(lambda s: ALLOWED_LANG.get(s, {"en"}))
    ok = [lang in a or lang == "und" or s in NO_LANG_FILTER
          for lang, a, s in zip(df["language"], allowed, df["source"])]
    df = f.drop(df, ~pd.Series(ok, index=df.index), "language", "language: " + df["language"])
    f.record("language", df)

    print("[6] exact duplicates", flush=True)
    df["source"] = pd.Categorical(df["source"], categories=PRIORITY, ordered=True)
    df = df.sort_values(["source", "native_id"], kind="stable").reset_index(drop=True)
    df["source"] = df["source"].astype(str)
    key = df["text"].map(norm_key)
    conflict = key.map(df.groupby(key)["label"].nunique()) > 1
    n_conflict_rows = int(conflict.sum())
    df = f.drop(df, conflict, "exact_dup", "same text with both labels")
    key = df["text"].map(norm_key)
    first_id = df.groupby(key)["id"].transform("first")
    df = f.drop(df, key.duplicated(), "exact_dup", "exact duplicate", duplicate_of=first_id)
    f.record("exact dedup", df)

    print(f"[7] near-duplicates (cosine > {NEAR_DUP}) - embedding {len(df):,} texts", flush=True)
    emb_all = embed(df["text"].tolist())
    keep = np.ones(len(df), dtype=bool)
    rep_id = pd.Series([None] * len(df), dtype=object)
    cross_pairs = 0
    for label in (1, 0):
        idx = np.flatnonzero(df["label"].to_numpy() == label)
        roots = near_dup_groups(emb_all[idx], NEAR_DUP)
        dup = roots != np.arange(len(idx))
        keep[idx[dup]] = False
        rep_id.iloc[idx[dup]] = df["id"].to_numpy()[idx[roots[dup]]]
    # attack/benign pairs above the threshold are kept (labels differ) but counted for review
    a_idx, b_idx = np.flatnonzero(df["label"] == 1), np.flatnonzero(df["label"] == 0)
    for s in range(0, len(a_idx), 1024):
        cross_pairs += int((emb_all[a_idx[s:s + 1024]] @ emb_all[b_idx].T > NEAR_DUP).sum())
    m = pd.Series(~keep, index=df.index)
    df = f.drop(df, m, "near_dup", f"cosine > {NEAR_DUP}", duplicate_of=rep_id)
    emb_all = emb_all[keep]
    f.record("near dedup", df)

    print("[8] building the clean table", flush=True)
    out = empty_frame()
    clean = pd.DataFrame({
        "id": df["id"], "text": df["text"], "label": df["label"], "source": df["source"],
        "subset": np.where(df["source"] == "notinject", "notinject", "main"),
        "original_split": df["original_split"], "language": df["language"], "n_tokens": df["n_tokens"],
        "category": df["category"],
        "type": np.where(df["label"] == 1, np.where(df["source"] == "jailbreak", "jailbreak", "injection"), None),
    })
    clean = pd.concat([out, clean], ignore_index=True)
    clean = clean.astype({c: d for c, (d, _, _) in COLUMNS.items() if d != "object"})
    validate(clean)
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    clean.to_parquet(PROC_DIR / "clean_v0.parquet", index=False)
    np.save(PROC_DIR / "clean_v0_embeddings.npy", emb_all)   # reused for k-means / benign matching

    print(f"[9] Tensor Trust candidate pool ({TT_POOL:,}) by cluster sampling", flush=True)
    pool = make_labeling_pool(clean, emb_all)

    removed = pd.concat(f.removed, ignore_index=True)
    removed.to_csv(PROC_DIR / "removal_log.csv", index=False, encoding="utf-8")

    write_report(f, clean, pool, removed, n_masked, n_conflict_rows, cross_pairs, time.time() - t0)


def make_labeling_pool(clean: pd.DataFrame, emb_all: np.ndarray) -> pd.DataFrame:
    """All cleaned non-Tensor-Trust attacks + TT_POOL diverse Tensor Trust attacks, with empty label columns."""
    from src.data.schema import LABEL_COLUMNS
    tt_idx = np.flatnonzero(clean["source"] == "tensortrust")
    pick = tt_idx[cluster_sample(emb_all[tt_idx], TT_POOL, TT_CLUSTERS, SEED)]
    pool = pd.concat([clean[(clean["label"] == 1) & (clean["source"] != "tensortrust")], clean.iloc[pick]])
    pool = pool[["id", "source", "language", "n_tokens", "type", "text"]].reset_index(drop=True)
    for c in LABEL_COLUMNS:
        if c != "type":
            pool[c] = None
    pool = pool[["id", "source", "language", "n_tokens", "text", *LABEL_COLUMNS]]
    pool.to_parquet(PROC_DIR / "labeling_pool_v0.parquet", index=False)
    pool.to_csv(PROC_DIR / "labeling_pool_v0.csv", index=False, encoding="utf-8-sig")
    return pool


def pool_only() -> None:
    """Rebuild the labelling pool from saved outputs (seconds; no re-embedding)."""
    clean = pd.read_parquet(PROC_DIR / "clean_v0.parquet")
    emb = np.load(PROC_DIR / "clean_v0_embeddings.npy")
    pool = make_labeling_pool(clean, emb)
    print(f"Rebuilt {PROC_DIR / 'labeling_pool_v0.csv'}: {len(pool):,} rows, columns: {list(pool.columns)}")


# ------------------------------------------------------------------ report
def write_report(f: Funnel, clean, pool, removed, n_masked, n_conflict, cross_pairs, secs) -> None:
    L = ["# Preprocessing report (Phase 3)\n",
         f"Generated by `python -m src.data.preprocess` on {pd.Timestamp.now():%Y-%m-%d %H:%M} in {secs / 60:.1f} min. "
         f"Length {MIN_TOK}-{MAX_TOK} tokens (`{TOKENIZER_NAME}`, no special tokens); near-duplicate threshold "
         f"cosine > {NEAR_DUP} (`{EMBEDDING_MODEL}`); seed {SEED}. Counts only, no dataset text.\n"]

    L.append("## 1. Funnel (rows remaining after each step)\n")
    fun = pd.DataFrame(f.rows).set_index("step").fillna(0).astype(int).T
    fun.index = [i.replace("|1", " attack").replace("|0", " benign") for i in fun.index]
    fun = fun.sort_index()
    L.append("| Source / class | " + " | ".join(fun.columns) + " |")
    L.append("|---|" + "---:|" * len(fun.columns))
    for name, r in fun.iterrows():
        L.append(f"| {name} | " + " | ".join(f"{v:,}" for v in r) + " |")
    L.append("")
    L.append(f"- Tensor Trust attacks with \"access granted\" removed: {n_masked:,}")
    L.append(f"- Rows dropped because the same text appeared with both labels: {n_conflict:,}")
    L.append(f"- Attack/benign pairs above the near-duplicate threshold (kept, worth a look): {cross_pairs:,}\n")

    L.append("## 2. Removals by reason\n")
    L.append("| Step | Reason | Rows |")
    L.append("|---|---|---:|")
    by = removed.copy()
    by["reason"] = by["reason"].where(~by["reason"].astype(str).str.startswith("language:"), "other language")
    for (s, r), n in by.groupby(["step", "reason"], sort=False).size().items():
        L.append(f"| {s} | {r} | {n:,} |")
    L.append("")

    L.append("## 3. Length bins: can benign data be length-matched to attacks?\n")
    bins = pd.cut(clean["n_tokens"], LENGTH_BINS, right=False,
                  labels=[f"{a}-{b - 1}" for a, b in zip(LENGTH_BINS[:-1], LENGTH_BINS[1:])])
    main = clean["subset"] == "main"
    groups = {
        "Attacks (all)": main & (clean["label"] == 1),
        "Attacks (jailbreak)": main & (clean["type"] == "jailbreak"),
        "Benign (all main)": main & (clean["label"] == 0),
        "Benign (Dolly)": clean["source"] == "dolly",
    }
    L.append("| Tokens | " + " | ".join(groups) + " |")
    L.append("|---|" + "---:|" * len(groups))
    for b in bins.cat.categories:
        L.append(f"| {b} | " + " | ".join(f"{int(((bins == b) & m).sum()):,}" for m in groups.values()) + " |")
    L.append("\nPhase 4 samples benign rows so each length bin has about as many benign as attack rows; "
             "every attack bin needs enough benign candidates.\n")

    L.append("## 4. Language\n")
    lang = clean.groupby(["source", "language"]).size().unstack(fill_value=0)
    L.append("| Source | " + " | ".join(lang.columns) + " |")
    L.append("|---|" + "---:|" * len(lang.columns))
    for s, r in lang.iterrows():
        L.append(f"| {s} | " + " | ".join(f"{v:,}" for v in r) + " |")
    L.append("\n`und` = detector not confident (short or symbol-heavy text); kept.\n")
    ni = clean[clean["source"] == "notinject"]
    if len(ni) and ni["category"].notna().any():
        L.append("**NotInject: category x language** (checks whether non-English rows are one category)\n")
        ct = ni.groupby(["category", "language"]).size().unstack(fill_value=0)
        L.append("| Category | " + " | ".join(ct.columns) + " |")
        L.append("|---|" + "---:|" * len(ct.columns))
        for c, r in ct.iterrows():
            L.append(f"| {c} | " + " | ".join(str(v) for v in r) + " |")
        L.append("")

    L.append("## 5. Labeling pool for B\n")
    L.append(f"`data/processed/labeling_pool_v0.parquet` / `.csv`: {len(pool):,} attacks to label "
             f"(all cleaned deepset, Gandalf and jailbreak attacks + {int((pool.source == 'tensortrust').sum()):,} "
             "Tensor Trust candidates chosen by k-means cluster sampling for diversity). Columns for B: "
             "`type` (pre-filled), `strategy_primary`, `strategy_secondary`, `ambiguous`, `obfuscated`, "
             "`label_method`, `verified_by`, `label_notes`.\n")
    L.append("| Source | Rows |")
    L.append("|---|---:|")
    for s, n in pool["source"].value_counts().items():
        L.append(f"| {s} | {n:,} |")
    L.append("")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    text = "\n".join(L)
    (REPORT_DIR / "preprocess_report.md").write_text(text, encoding="utf-8")
    print("\n" + text)
    print(f"\nSaved: {PROC_DIR / 'clean_v0.parquet'}\n       {PROC_DIR / 'labeling_pool_v0.csv'}\n"
          f"       {REPORT_DIR / 'preprocess_report.md'}")


if __name__ == "__main__":
    import sys
    pool_only() if "--pool-only" in sys.argv else run()
