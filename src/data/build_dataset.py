"""Phases 4-5: merge B's labels, sample the final dataset, and assign every split.  Run from the repo root:

    python -m src.data.build_dataset --labels data/processed/labels_v1.parquet   # real run
    python -m src.data.build_dataset --fake-labels                                # format preview only

Inputs : data/processed/clean_v0.parquet (+ clean_v0_embeddings.npy) from preprocess.py,
         and B's labels file (`id` + schema.LABEL_COLUMNS; .parquet or .csv).
Outputs: data/processed/apig_v1.parquet           final table (schema-valid, labelled, all splits)
         reports/data/dataset_card.md             counts per fold, strategy x source, length check

--fake-labels assigns RANDOM strategies so C and D can write code against the final format before
the real labels exist. Its output is apig_v1_FAKE.parquet. Never report results from it.

Steps
  1. merge labels by id; attacks without labels are dropped (Tensor Trust rows outside the pool);
     text fixes: Tensor Trust "access granted" replaced by varied phrases, short CJK texts removed
  2. Tensor Trust: per-strategy quotas (equal shares, unused share redistributed), non-ambiguous first
  3. benign: keep all deepset/jailbreak benign (needed for LODO), fill up to a 1:1 ratio from
     Dolly/Alpaca so every length bin has as many benign as attack rows
  4. calibration pool: ~CALIB_SIZE more Dolly/Alpaca rows with the same length profile
  5. splits: random 70/15/15, LOSO, within-source LOSO (Tensor Trust), LODO
  6. leak checks (raise on failure) + length-only baseline (AUROC should be close to 0.5)
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from src.data.schema import (ATTACK_SOURCES, COLUMNS, LABEL_COLUMNS, SPLIT_COLUMNS, STRATEGIES,
                             validate, validate_labels)
from src.data.sources import REPO_ROOT, RAW_DIR, TENSORTRUST_FILE
from src.data.textfix import MARKERS, cjk_language, mask_access_granted

PROC_DIR = REPO_ROOT / "data" / "processed"
REPORT_DIR = REPO_ROOT / "reports" / "data"

SEED = 42
TT_TOTAL = 1_300            # final Tensor Trust attacks (proposal: <= ~1.2k; 1.3k brings attacks to ~2.6k)
CALIB_SIZE = 3_000          # benign calibration pool for the 1% FPR threshold
VAL_FRAC = 0.15             # share of each training pool held out for validation / early stopping
FILLER = ["dolly", "alpaca"]
# length bins for matching benign to attacks (tokens); finer at the short end where most rows are
MATCH_BINS = [5, 10, 15, 20, 30, 40, 50, 75, 100, 150, 200, 300, 400, 501]


# ------------------------------------------------------------------ 1. labels
def load_inputs(labels_path: str | None, fake: bool) -> pd.DataFrame:
    clean = pd.read_parquet(PROC_DIR / "clean_v0.parquet")
    for c, (dtype, _, _) in COLUMNS.items():          # upgrade tables written with an older schema
        if c not in clean.columns:
            clean[c] = pd.Series([None] * len(clean), dtype="object" if dtype == "object" else None)
    attacks = clean[clean["label"] == 1]
    if fake:
        labels = fake_labels(attacks)
    else:
        from pathlib import Path
        path = Path(labels_path)
        path = path if path.is_absolute() else REPO_ROOT / path
        labels = pd.read_csv(path, encoding="utf-8-sig") if path.suffix == ".csv" else pd.read_parquet(path)
    labels = validate_labels(labels, clean)

    prefilled_type = clean.set_index("id")["type"]
    clean = clean.drop(columns=LABEL_COLUMNS).merge(labels, on="id", how="left")
    clean["type"] = clean["type"].fillna(clean["id"].map(prefilled_type))   # B may leave type blank
    clean["type"] = clean["type"].where(clean["label"] == 1, None)
    unlabelled = (clean["label"] == 1) & clean["strategy_primary"].isna()
    missing_non_tt = unlabelled & (clean["source"] != "tensortrust")
    if missing_non_tt.any():
        print(f"  WARNING: {int(missing_non_tt.sum())} non-Tensor-Trust attacks have no label and are dropped "
              f"(e.g. {clean.loc[missing_non_tt, 'id'].head(3).tolist()})")
    clean = clean[~unlabelled].reset_index(drop=True)
    for c in ["ambiguous", "obfuscated"]:
        clean.loc[clean["label"] == 1, c] = clean.loc[clean["label"] == 1, c].fillna(False)
    return clean


def fake_labels(attacks: pd.DataFrame) -> pd.DataFrame:
    """Random but plausible labels, ONLY for previewing the format."""
    rng = np.random.default_rng(SEED)
    pool_ids = set(pd.read_parquet(PROC_DIR / "labeling_pool_v0.parquet")["id"])
    a = attacks[attacks["id"].isin(pool_ids)]
    probs = {"tensortrust": [0.5, 0.15, 0.35], "jailbreak": [0.1, 0.8, 0.1],
             "gandalf": [0.6, 0.3, 0.1], "deepset": [0.6, 0.2, 0.2]}
    prim = [rng.choice(STRATEGIES, p=probs[s]) for s in a["source"]]
    sec = [[x for x in STRATEGIES if x != p and rng.random() < 0.1] for p in prim]
    return pd.DataFrame({
        "id": a["id"].values, "type": a["type"].values, "strategy_primary": prim, "strategy_secondary": sec,
        "ambiguous": rng.random(len(a)) < 0.05, "obfuscated": rng.random(len(a)) < 0.1,
        "label_method": "rule", "verified_by": None, "label_notes": "FAKE LABEL",
    })


# ------------------------------------------------------------------ 1b. text fixes (Nov 19)
def fix_texts(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Apply text fixes found after preprocessing, without changing any id (so labels stay valid).

    a) Tensor Trust: rebuild each text from the raw dump and REPLACE "access granted" with an ordinary
       phrase (clean_v0 deleted it, leaving empty quotes in ~26% of Tensor Trust attacks: a shortcut).
    b) Short Chinese/Japanese/Korean texts that the language detector could not decide are removed
       from sources where only English (deepset: English/German) is allowed. NotInject is untouched.
    """
    import bz2
    import json

    from src.data.preprocess import ALLOWED_LANG, MAX_TOK, MIN_TOK, NO_LANG_FILTER, count_tokens

    info = {}
    tt = df["source"] == "tensortrust"
    want = {int(i.split("_", 1)[1]): i for i in df.loc[tt, "id"]}
    raw = {}
    path = RAW_DIR / "tensortrust" / TENSORTRUST_FILE
    with bz2.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            aid = r.get("attack_id")
            if aid in want and aid not in raw:
                raw[aid] = r.get("attacker_input") or ""
    missing = set(want) - set(raw)
    if missing:
        raise SystemExit(f"{len(missing)} Tensor Trust ids not found in the raw dump, e.g. {sorted(missing)[:3]}")
    new_text = df.loc[tt, "id"].map(lambda i: mask_access_granted(raw[int(i.split("_", 1)[1])], i))
    changed = tt & (df["text"] != new_text.reindex(df.index))
    df.loc[tt, "text"] = new_text
    if changed.any():
        df.loc[changed, "n_tokens"] = count_tokens(df.loc[changed, "text"].tolist())
    too_long = changed & ((df["n_tokens"] > MAX_TOK) | (df["n_tokens"] < MIN_TOK))   # keep the 5-500 range
    info["tt_rewritten"], info["tt_too_long"] = int(changed.sum()), int(too_long.sum())
    df = df[~too_long]

    cjk = pd.Series([cjk_language(str(t)) for t in df["text"]], index=df.index, dtype=object)
    allowed = df["source"].map(lambda s: ALLOWED_LANG.get(s, {"en"}))
    ok = pd.Series([isinstance(c, str) and c in a for c, a in zip(cjk, allowed)], index=df.index)
    drop = cjk.notna() & ~df["source"].isin(NO_LANG_FILTER) & ~ok
    df.loc[cjk.notna(), "language"] = cjk[cjk.notna()]
    info["cjk_removed"] = df.loc[drop, "source"].value_counts().to_dict()
    return df[~drop].reset_index(drop=True), info


def marker_table(df: pd.DataFrame) -> pd.DataFrame:
    """How often each marker pattern appears in main attacks vs main benign texts."""
    main = df[df["subset"] == "main"]
    a, b = main[main["label"] == 1]["text"], main[main["label"] == 0]["text"]
    rows = []
    for name, pat in MARKERS.items():
        pa, pb = a.str.contains(pat).mean(), b.str.contains(pat).mean()
        rows.append({"marker": name, "attacks": pa, "benign": pb,
                     "flag": pa >= 0.02 and pa >= 5 * max(pb, 0.001)})
    return pd.DataFrame(rows).sort_values("attacks", ascending=False)


# ------------------------------------------------------------------ 2-4. sampling
def sample_tensortrust(df: pd.DataFrame, total: int, rng) -> pd.Index:
    tt = df[(df["source"] == "tensortrust") & (df["label"] == 1)]
    clear = tt[~_flag(tt["ambiguous"])]
    by_s = {s: clear.index[clear["strategy_primary"] == s] for s in STRATEGIES}
    quota = {s: total // len(STRATEGIES) for s in STRATEGIES}
    # redistribute unused quota from rare strategies to the others
    for _ in range(3):
        spare = sum(max(0, quota[s] - len(by_s[s])) for s in STRATEGIES)
        quota = {s: min(quota[s], len(by_s[s])) for s in STRATEGIES}
        open_ = [s for s in STRATEGIES if quota[s] < len(by_s[s])]
        for s in open_:
            quota[s] += spare // max(1, len(open_))
    chosen = np.concatenate([rng.choice(by_s[s], size=min(quota[s], len(by_s[s])), replace=False)
                             for s in STRATEGIES])
    if len(chosen) < total:                            # top up with ambiguous rows (train-only in LOSO)
        amb = tt.index.difference(chosen)
        chosen = np.concatenate([chosen, rng.choice(amb, size=min(total - len(chosen), len(amb)), replace=False)])
    return pd.Index(chosen)


def length_bin(n_tokens: pd.Series) -> pd.Series:
    return pd.cut(n_tokens, MATCH_BINS, right=False, labels=False)


def match_by_length(target_counts: pd.Series, candidates: pd.DataFrame, rng) -> tuple[pd.Index, int]:
    """Sample candidates so each length bin reaches its target count. Returns (chosen index, shortfall)."""
    chosen, short = [], 0
    bins = length_bin(candidates["n_tokens"])
    for b, need in target_counts.items():
        if need <= 0:
            continue
        pool = candidates.index[bins == b]
        take = min(int(need), len(pool))
        short += int(need) - take
        if take:
            chosen.append(rng.choice(pool, size=take, replace=False))
    return (pd.Index(np.concatenate(chosen)) if chosen else pd.Index([])), short


# ------------------------------------------------------------------ 5. splits
def _is_list(v) -> bool:
    return isinstance(v, (list, tuple, np.ndarray))


def _flag(s: pd.Series) -> pd.Series:
    """Nullable boolean -> plain bool (missing = False)."""
    return s.astype("boolean").fillna(False).astype(bool)


def stratified_parts(index: pd.Index, strata: pd.Series, fracs: dict[str, float], rng) -> pd.Series:
    """Assign each row in `index` to a part (e.g. train/val/test) with the given fractions per stratum."""
    out = pd.Series(index=index, dtype=object)
    names, cum = list(fracs), np.cumsum(list(fracs.values()))
    for _, idx in strata.loc[index].groupby(strata.loc[index]).groups.items():
        idx = rng.permutation(np.array(idx))
        cuts = np.round(cum * len(idx)).astype(int)
        start = 0
        for name, stop in zip(names, cuts):
            out.loc[idx[start:stop]] = name
            start = stop
    return out


def assign_splits(df: pd.DataFrame, rng) -> pd.DataFrame:
    main = df["subset"] == "main"
    atk, ben = main & (df["label"] == 1), main & (df["label"] == 0)
    strat = df["label"].astype(str) + "|" + df["source"] + "|" + df["strategy_primary"].fillna("-")
    for c in SPLIT_COLUMNS:
        df[c] = "excluded"

    # (i) random 70/15/15, stratified by label x source x strategy
    df.loc[main, "split_random"] = stratified_parts(df.index[main], strat, {"train": .70, "val": .15, "test": .15}, rng)

    # fixed benign slices reused by every fold (so folds differ only in their attacks)
    ben_role = df["split_random"].where(ben)                      # train / val / test for benign rows

    def finish(col: str, train_mask: pd.Series, test_mask: pd.Series) -> None:
        """test rows are given; train pool is split into train/val (val from training groups only)."""
        df.loc[test_mask, col] = "test"
        parts = stratified_parts(df.index[train_mask], strat, {"train": 1 - VAL_FRAC, "val": VAL_FRAC}, rng)
        df.loc[parts.index, col] = parts

    sec_has = lambda s: df["strategy_secondary"].map(lambda v: _is_list(v) and s in list(v))  # noqa: E731
    amb = _flag(df["ambiguous"])

    # (ii) LOSO and (iii) within-source LOSO
    for s in STRATEGIES:
        involved = atk & ((df["strategy_primary"] == s) | sec_has(s))
        test_atk = atk & (df["strategy_primary"] == s) & ~amb
        ben_test, ben_train = ben & (ben_role == "test"), ben & (ben_role != "test")
        finish(f"loso_{s}", train_mask=(atk & ~involved) | ben_train, test_mask=test_atk | ben_test)
        tt_atk = atk & (df["source"] == "tensortrust")
        finish(f"wsloso_{s}", train_mask=(tt_atk & ~involved) | ben_train, test_mask=(test_atk & tt_atk) | ben_test)

    # (iv) LODO: hold out each attack source with its own benign data. The test benign is length-matched
    # to the held-out attacks (otherwise length alone separates e.g. long jailbreak attacks from short
    # jailbreak benign prompts): own benign rows first, topped up with Dolly/Alpaca rows.
    filler_ben = df[ben & df["source"].isin(FILLER)]
    for src in ATTACK_SOURCES:
        test_atk = atk & (df["source"] == src)
        own_ben = df[ben & (df["source"] == src)]
        a_cnt = length_bin(df.loc[test_atk, "n_tokens"]).value_counts()
        # per length bin: the source's own benign rows first, Dolly/Alpaca for the rest. Own benign rows
        # beyond the attacks' length profile are left out of this fold (not trained on, not tested).
        own_pick, _ = match_by_length(a_cnt, own_ben, rng)
        got = length_bin(df.loc[own_pick, "n_tokens"]).value_counts().reindex(a_cnt.index, fill_value=0)
        top_up, _ = match_by_length((a_cnt - got).clip(lower=0), filler_ben, rng)
        test_mask = test_atk | df.index.isin(own_pick) | df.index.isin(top_up)
        train_mask = main & ~test_mask & (df["source"] != src)
        finish(f"lodo_{src}", train_mask=train_mask, test_mask=test_mask)
    return df


# ------------------------------------------------------------------ 6. checks
def leak_checks(df: pd.DataFrame) -> list[str]:
    """Raise if any split breaks its definition; return human-readable confirmations."""
    ok = []
    sec_has = lambda s: df["strategy_secondary"].map(lambda v: _is_list(v) and s in list(v))  # noqa: E731
    atk = df["label"] == 1
    for c in SPLIT_COLUMNS:
        tr, te = df[c].isin(["train", "val"]), df[c] == "test"
        assert not (df["subset"].isin(["calibration", "notinject"]) & df[c].ne("excluded")).any(), f"{c}: held-out pool used"
        for lab in (0, 1):
            assert ((df["label"] == lab) & te).any(), f"{c}: test set has no label {lab}"
            assert ((df["label"] == lab) & (df[c] == "train")).any(), f"{c}: train set has no label {lab}"
    ok.append("calibration and NotInject rows are never in train/val/test of any split")
    for s in STRATEGIES:
        for pre in ("loso", "wsloso"):
            c = f"{pre}_{s}"
            leaked = atk & df[c].isin(["train", "val"]) & ((df["strategy_primary"] == s) | sec_has(s))
            assert not leaked.any(), f"{c}: {int(leaked.sum())} training attacks use {s}"
            bad_test = atk & (df[c] == "test") & ((df["strategy_primary"] != s) | _flag(df["ambiguous"]))
            assert not bad_test.any(), f"{c}: test attacks with another strategy or ambiguous"
        non_tt = atk & (df["source"] != "tensortrust") & df[f"wsloso_{s}"].ne("excluded")
        assert not non_tt.any(), f"wsloso_{s}: non-Tensor-Trust attacks used"
    ok.append("LOSO: the held-out strategy (primary or secondary) never appears in training; tests contain only that strategy, no ambiguous rows")
    ok.append("Within-source LOSO uses Tensor Trust attacks only")
    for src in ATTACK_SOURCES:
        c = f"lodo_{src}"
        assert not ((df["source"] == src) & df[c].isin(["train", "val"])).any(), f"{c}: held-out source in training"
        assert not (atk & (df[c] == "test") & (df["source"] != src)).any(), f"{c}: other-source attacks in test"
    ok.append("LODO: the held-out source (attacks and its benign data) never appears in training")
    for c in SPLIT_COLUMNS:
        te_ids = set(df.loc[df[c] == "test", "id"])
        assert not (set(df.loc[df[c].isin(['train', 'val']), 'id']) & te_ids), f"{c}: id in train and test"
    ok.append("no id is in both training and test of any split (near-duplicates were removed globally in Phase 3)")
    return ok


def length_only_auroc(df: pd.DataFrame, col: str) -> float | None:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    tr, te = df[df[col] == "train"], df[df[col] == "test"]
    if tr["label"].nunique() < 2 or te["label"].nunique() < 2:
        return None
    f = lambda d: np.c_[np.log1p(d["n_tokens"].astype(float)), np.log1p(d["n_tokens"].astype(float)) ** 2]  # noqa: E731
    m = LogisticRegression().fit(f(tr), tr["label"])
    return float(roc_auc_score(te["label"], m.predict_proba(f(te))[:, 1]))


# ------------------------------------------------------------------ main
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--labels", help="B's labels file (.parquet or .csv), path relative to the repo root")
    g.add_argument("--fake-labels", action="store_true", help="random labels, for previewing the format only")
    args = p.parse_args()
    t0, rng = time.time(), np.random.default_rng(SEED)
    fake = args.fake_labels

    print("[1] loading clean table and labels", flush=True)
    df = load_inputs(args.labels, fake)
    print("    text fixes: Tensor Trust 'access granted' replacement, CJK language check", flush=True)
    df, fix_info = fix_texts(df)

    print("[2] sampling Tensor Trust with per-strategy quotas", flush=True)
    tt_keep = sample_tensortrust(df, TT_TOTAL, rng)
    df = df[(df["source"] != "tensortrust") | df.index.isin(tt_keep)].reset_index(drop=True)
    attacks = df[(df["label"] == 1)]

    print("[3] length-matched benign data", flush=True)
    fixed_ben = df[(df["label"] == 0) & df["source"].isin(["deepset", "jailbreak"])]
    target = length_bin(attacks["n_tokens"]).value_counts().sort_index()
    have = length_bin(fixed_ben["n_tokens"]).value_counts().reindex(target.index, fill_value=0)
    filler_pool = df[(df["label"] == 0) & df["source"].isin(FILLER)]
    main_fill, short_main = match_by_length((target - have).clip(lower=0), filler_pool, rng)

    print("[4] calibration pool", flush=True)
    rest = filler_pool.drop(index=main_fill)
    main_ben_bins = length_bin(pd.concat([fixed_ben, df.loc[main_fill]])["n_tokens"]).value_counts(normalize=True)
    calib_target = (main_ben_bins * CALIB_SIZE).round().astype(int)
    calib, short_calib = match_by_length(calib_target, rest, rng)

    keep = df["label"].eq(1) | df["source"].isin(["deepset", "jailbreak", "notinject"]) | df.index.isin(main_fill) \
        | df.index.isin(calib)
    df.loc[df.index.isin(calib), "subset"] = "calibration"
    df = df[keep].reset_index(drop=True)

    print("[5] assigning splits", flush=True)
    df = assign_splits(df, rng)

    print("[6] checks", flush=True)
    df = df[list(COLUMNS)]
    df = df.astype({c: d for c, (d, _, _) in COLUMNS.items() if d != "object"})
    validate(df, labelled=True, split=True)
    confirmations = leak_checks(df)
    aurocs = {c: length_only_auroc(df, c) for c in SPLIT_COLUMNS}
    markers = marker_table(df)
    if markers["flag"].any():
        print("  WARNING: marker patterns much more common in attacks than benign: "
              f"{markers.loc[markers['flag'], 'marker'].tolist()}")

    name = "apig_v1_FAKE.parquet" if fake else "apig_v1.parquet"
    df.to_parquet(PROC_DIR / name, index=False)
    write_card(df, name, fake, confirmations, aurocs, short_main, short_calib, time.time() - t0, fix_info, markers)


def write_card(df, name, fake, confirmations, aurocs, short_main, short_calib, secs, fix_info, markers) -> None:
    L = [f"# Dataset card: `{name}`\n"]
    if fake:
        L.append("> **FAKE LABELS.** Strategies were assigned at random to preview the file format. "
                 "Do not report any result computed on this file.\n")
    L.append(f"Built by `python -m src.data.build_dataset` on {pd.Timestamp.now():%Y-%m-%d %H:%M} ({secs:.0f}s), "
             f"seed {SEED}. Counts only, no dataset text.\n")
    main = df[df["subset"] == "main"]
    L.append("## 1. Contents\n")
    L.append("| Subset | Attacks | Benign |")
    L.append("|---|---:|---:|")
    for s in ["main", "calibration", "notinject"]:
        d = df[df["subset"] == s]
        L.append(f"| {s} | {int((d.label == 1).sum()):,} | {int((d.label == 0).sum()):,} |")
    L.append("")
    if short_main or short_calib:
        L.append(f"Length matching shortfall: {short_main} benign (main), {short_calib} (calibration) rows "
                 "could not be matched in their length bin.\n")

    L.append("## 2. Strategy x source (main attacks)\n")
    ct = pd.crosstab(main.loc[main.label == 1, "strategy_primary"], main.loc[main.label == 1, "source"],
                     margins=True, margins_name="total")
    L.append("| Strategy | " + " | ".join(ct.columns) + " |")
    L.append("|---|" + "---:|" * len(ct.columns))
    for s, r in ct.iterrows():
        L.append(f"| {s} | " + " | ".join(f"{v:,}" for v in r) + " |")
    L.append("")
    t = main[main.label == 1]
    L.append(f"Type: {int((t.type == 'injection').sum()):,} injection, {int((t.type == 'jailbreak').sum()):,} jailbreak. "
             f"Ambiguous: {int(_flag(t.ambiguous).sum()):,}. Obfuscated: {int(_flag(t.obfuscated).sum()):,}.\n")

    L.append("## 3. Split sizes\n")
    L.append("| Split | Train (atk/ben) | Val (atk/ben) | Test (atk/ben) | Excluded | Length-only AUROC |")
    L.append("|---|---|---|---|---:|---:|")
    for c in SPLIT_COLUMNS:
        cell = lambda part: f"{int(((df[c] == part) & (df.label == 1)).sum()):,} / {int(((df[c] == part) & (df.label == 0)).sum()):,}"  # noqa: E731
        a = aurocs.get(c)
        L.append(f"| {c} | {cell('train')} | {cell('val')} | {cell('test')} | {int((df[c] == 'excluded').sum()):,} | "
                 f"{'-' if a is None else f'{a:.2f}'} |")
    L.append("\n*Length-only AUROC*: a logistic regression that sees only the token count, trained on each split's "
             "train part and scored on its test part. Close to 0.5 = length does not give the label away; far from 0.5 in *either* direction "
             "(e.g. 0.1 or 0.9) = length separates the classes.\n")

    L.append("## 4. Shortcut markers (main split)\n")
    L.append(f"Text fixes applied at build time: {fix_info['tt_rewritten']:,} Tensor Trust texts had \"access granted\" "
             f"replaced by an ordinary phrase ({fix_info['tt_too_long']} then fell outside the token range and were dropped); "
             f"short CJK texts removed: {fix_info['cjk_removed'] or 'none'}.\n")
    L.append("| Pattern | In attacks | In benign | Flag |")
    L.append("|---|---:|---:|---|")
    for r in markers.itertuples():
        if r.attacks > 0 or r.benign > 0:
            L.append(f"| {r.marker} | {r.attacks:.1%} | {r.benign:.1%} | {'⚠' if r.flag else ''} |")
    L.append("\n⚠ = at least 2% of attacks and 5x more common than in benign text (a possible shortcut).\n")
    L.append("## 5. Checks passed\n")
    L.extend(f"- {c}" for c in confirmations)
    L.append("")
    L.append("## 6. How to use\n")
    L.append("```python\nimport pandas as pd\n"
             f"df = pd.read_parquet('data/processed/{name}')\n"
             "train = df[df.loso_S2 == 'train']; val = df[df.loso_S2 == 'val']; test = df[df.loso_S2 == 'test']\n"
             "calib = df[df.subset == 'calibration']   # threshold = 99th percentile of benign scores here\n"
             "notinject = df[df.subset == 'notinject'] # RQ3 test only\n```\n")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    card = "dataset_card_FAKE.md" if fake else "dataset_card.md"
    text = "\n".join(L)
    (REPORT_DIR / card).write_text(text, encoding="utf-8")
    print("\n" + text)
    print(f"\nSaved {PROC_DIR / name}\n      {REPORT_DIR / card}")


if __name__ == "__main__":
    main()
