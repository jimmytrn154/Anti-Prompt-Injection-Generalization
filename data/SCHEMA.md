# APIG data schema — proposal v0.1

> **Status:** draft by A (Allaberdi) for agreement with B (Sangah). The code version is `src/data/schema.py`; run `validate(df)` before saving or after loading. If you change a column, change it in both places.

## File

- **One table for everything:** `data/processed/apig_v<N>.parquet` (v1, v1.1, …). Never overwrite a released version silently.
- **Parquet**, not CSV: keeps dtypes and stores `strategy_secondary` as a real list.
- Data is **not committed to Git** (licences). Shared via Google Drive / private HF dataset; scripts in `src/data/` regenerate it.

## Columns

| Column | Type | Filled by | Description |
|---|---|---|---|
| `id` | string | A | Stable unique ID: `<source>_<native id or 5-digit row index>`, e.g. `gandalf_00042` |
| `text` | string | A | User-turn text. Tensor Trust: "access granted" masked. Not Unicode-normalized (natural obfuscation is kept) |
| `label` | int | A | `1` attack, `0` benign |
| `source` | string | A | `deepset`, `gandalf`, `jailbreak`, `tensortrust`, `dolly`, `alpaca`, `notinject` |
| `subset` | string | A | `main` · `calibration` (≈3k benign for the 1% FPR threshold) · `notinject` (test only) |
| `original_split` | string | A | Split name in the source dataset (provenance only, never used) |
| `language` | string | A | ISO 639-1 code (`en`, `de`), or `und` when undetermined (short, padded or homoglyph-obfuscated text) |
| `n_tokens` | int | A | Length with the shared tokenizer (`TOKENIZER_NAME` in `check_env.py`) |
| `category` | string | A | Category given by the source (Dolly, NotInject); empty otherwise |
| `type` | string | A → B | `injection` / `jailbreak`. Attacks only. A pre-fills by source, B confirms |
| `strategy_primary` | string | B | `S1` / `S2` / `S3`. Attacks only |
| `strategy_secondary` | list | B | Zero or more of `S1`–`S3` (`[]` for none). Attacks only |
| `ambiguous` | bool | B | Primary strategy unclear → excluded from LOSO test sets |
| `obfuscated` | bool | B | Natural obfuscation in the original text. Attacks only |
| `label_method` | string | B | Method that determined the **final** strategy label: `rule` / `llm` / `manual` |
| `verified_by` | string | B | GitHub handle(s) of whoever hand-checked the label; empty = not checked |
| `label_notes` | string | B | Free-text reason for borderline cases |
| split columns | string | A | See below |

Attack-only columns are empty (null) for benign rows.

## Labelling workflow (agreed A + B, Nov 17)

1. A shares `labeling_pool_v0.csv` (attacks only, label columns empty; `type` pre-filled by source).
2. B labels with rules + an LLM pass and delivers **`labels_v1.parquet`**: `id` + the 8 label columns. A's `build_dataset.py` merges it by `id`; neither edits the other's files. `schema.validate_labels()` checks it first.
3. **`label_method` vs `verified_by`** are independent:

   | Case | `label_method` | `verified_by` |
   |---|---|---|
   | LLM label, nobody checked it | `llm` | empty |
   | LLM label, a person checked it and agreed | `llm` | `ivory4326` |
   | A person checked it and changed it | `manual` | `ivory4326` |
   | Adjudicated in the 300-sample check | `manual` if it changed, else unchanged | both handles, e.g. `ivory4326;<A's handle>` |

4. **300-sample agreement check** (`src/data/agreement.py`): drawn **stratified by source x automatic strategy** so every source and strategy is covered; two blind files (no automatic label shown), one per annotator; Cohen's κ, confusion matrix, κ per source and automatic-vs-human accuracy go to `reports/data/agreement_report.md`; disagreements are adjudicated together.

## Split columns

One column per split scheme or fold. Each cell is the row's role in that fold: `train`, `val`, `test` or `excluded`.

| Column(s) | Rule |
|---|---|
| `split_random` | Stratified 70/15/15 by label × source |
| `loso_S1`, `loso_S2`, `loso_S3` | **test:** non-ambiguous attacks with primary = Si, plus a fixed held-out benign slice. **excluded:** other attacks with Si as primary or secondary. **train/val:** the rest; val only from training strategies |
| `wsloso_S1`, `wsloso_S2`, `wsloso_S3` | Same rule, attacks from Tensor Trust only; same benign slices in every fold |
| `lodo_deepset`, `lodo_gandalf`, `lodo_jailbreak`, `lodo_tensortrust` | **test:** that source's attacks + its own benign data. **train/val:** other sources; val only from training sources |

`calibration` and `notinject` rows are never `train` or `val` in any column (the validator enforces this).

Example — train M1 on the LOSO fold that holds out S2:

```python
df = pd.read_parquet("data/processed/apig_v1.parquet")
train = df[df.loso_S2 == "train"]; val = df[df.loso_S2 == "val"]; test = df[df.loso_S2 == "test"]
calib = df[df.subset == "calibration"]   # threshold at 1% FPR
```

## Open questions (A + B, this week)

1. ~~Token cap~~ **Decided (Nov 17): 5–500 tokens**, plus long benign prompts (Dolly instruction + context) and length matching, verified by a length-only baseline.
2. **Tokenizer:** confirm `TOKENIZER_NAME` with C so lengths match M3.
3. **LODO benign for attack-only sources:** Gandalf and Tensor Trust have no benign data. Proposal: use a fixed held-out Dolly/Alpaca slice.
4. **Ambiguous attacks in training:** allowed when the held-out strategy is not among their possible strategies?
5. **Type pre-fill:** jailbreak-classification → `jailbreak`; deepset, Gandalf, Tensor Trust → `injection`. B to check exceptions.
6. **Extra columns B needs** (e.g. `label_method` = rule/LLM/manual, `verified_by`)? Add them to `schema.py` now so the file format stays fixed.
