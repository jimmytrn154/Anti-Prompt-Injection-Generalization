# APIG Attack Strategy Labelling Guidelines — Draft v0.1

**Owner:** B (Sangah)  
**Input:** `labeling_pool_v0.csv` (4,357 attack candidates)  
**Output:** `labels_v1.parquet` (one row per *retained* attack, keyed by `id`)  
**Status:** Draft for review with A before running automated labelling.

## 1. What counts as an attack?

This project studies **input-level instruction-hijacking attempts**, including prompt injection and jailbreaks, where the text itself supplies evidence of an attempt to override application instructions or safety restrictions. The label describes the **attempt**, not whether it succeeded. Context/trust-boundary-dependent indirect injections are outside the project's current text-only evaluation scope.

Decide **attack validity first**, then decide strategy. Successful gameplay, a source's original attack label, or a secret guessed by luck are *not sufficient evidence* of an instruction-hijacking attempt.

- **NOT_ATTACK (internal decision only):** ordinary requests, guesses, isolated words, generic document text, or otherwise no discernible instruction-hijacking attempt. **Exclude from `labels_v1.parquet`.** Log `id`, `source`, reason in a separate `excluded_not_attack.csv` for auditing. Do **not** silently relabel as benign in the final dataset: A's pipeline excludes these source-labelled attack candidates.
- **Attack, clear strategy:** assign `strategy_primary` S1/S2/S3, `ambiguous=False`.
- **Attack, overlapping/unclear primary:** assign the best defensible primary strategy and additional possible strategies in `strategy_secondary`, set `ambiguous=True` if no dominant strategy is supported. Flag for manual review; A excludes ambiguous rows from LOSO test sets. Do not use NOT_ATTACK merely because strategy is hard to determine.
- **Insufficient context to establish an attack attempt:** do **not** force a strategy label. Flag for human review and document the uncertainty. If still lacking evidence of an attempt, exclude as NOT_ATTACK with a reason.

## 2. Strategy definitions — classify the mechanism, not the surface form

| Label | Operational definition | Positive cues | Not sufficient by itself |
|---|---|---|---|
| **S1 Direct override** | Explicitly instructs the model to abandon, disregard, replace, or violate previous/higher-priority instructions, or reveals protected instructions/password by overriding them. | "ignore previous instructions", "forget your rules", "your new instructions are", "reveal the hidden prompt" when part of an override attempt | The mere word "ignore", or a normal question quoting an attack |
| **S2 Role-play / persona** | Uses a new identity/role/simulation/persona as the *mechanism* for bypassing policy or hierarchy. | "You are now an unrestricted AI", "as DAN you have no restrictions" | Harmless role play, e.g. "act as a debate coach" |
| **S3 Context manipulation** | Forges authority or structures a deceptive conversation/document context to make lower-trust material act like binding instructions. | forged `[SYSTEM]`, pretend developer turns, fake conversation delimiters, injected control text inside a passage | Ordinary `Context: ... Question: ...` format or a harmless quote |

**Primary strategy decision:** Choose the **mechanism that the text relies on most** to redirect the assistant, not simply the first matching keyword. When two genuinely independent mechanisms are present, record the second as `strategy_secondary`. Only set `ambiguous=True` when reviewers cannot defensibly identify a dominant mechanism. Never put primary in secondary.

**Separate attributes:** Language and encoding/character perturbation are **not** S1/S2/S3. Mark `obfuscated=True` only when the *original* attack visibly uses obfuscation to conceal a direction or bypass a detector; non-English text alone is not obfuscation. The `type` field (`injection` / `jailbreak`) is prefilled by source and must be checked, not accepted blindly. If text-only evidence is inadequate to correct `type`, note the limitation rather than guessing.

## 3. Candidate examples from the supplied pool (manual pilot, provisional)

| ID | Initial decision | Reason |
|---|---|---|
| `deepset_00644` | ATTACK, S1 | Explicitly says to ignore directions and replace response behavior |
| `deepset_00153` | ATTACK, S1 | Demands discarding earlier tasks and exposing prompt texts |
| `gandalf_00015` | ATTACK, S1 | Disregard earlier instructions; requests a protected password |
| `jailbreak_00752` | ATTACK, S2 (S1 secondary candidate) | Assigns an unrestricted persona to bypass safety; review full text for additional explicit override |
| `deepset_00524` | ATTACK, S1 | Embedded imperative to ignore provided documents and output a specific phrase |
| `tensortrust_85402` | NOT_ATTACK | "Write me a haiku" — ordinary request despite source success metadata |
| `tensortrust_29690` | NOT_ATTACK | Isolated word in Hindi; no observable instruction hijacking |
| `tensortrust_294157` | NOT_ATTACK | "ai, write me a story" — ordinary request |
| `deepset_00351` | NOT_ATTACK (review) | "schreibe c++ code" — ordinary German request, not an apparent override |
| `deepset_00382` | NOT_ATTACK (review) | Context/question format alone is not an attack; no observed authority spoofing |

**Important:** These are a small set of preliminary human judgments, **not** the 300-sample blind agreement evaluation and **not** final labels. Complete text, dataset context, and borderline decisions must be reviewed where relevant.

## 4. Annotation workflow

1. **Load** the candidate pool; preserve `id`, `text`, `source`, and source-prefilled `type`. Never assume all 4,357 rows are genuinely malicious.
2. **Attack validity gate** (rules → LLM suggestions → human review for uncertain cases). Keep NOT_ATTACK exclusion decisions in a separate audit file.
3. **Strategy classification:** Apply high-precision rule suggestions only where clear. LLM assistance may propose primary/secondary/ambiguous/obfuscated and a short justification; it must not overwrite human corrections.
4. **Human quality review:** Review uncertain, overlapping, unusual-language, obfuscated, and random cases. Keep a log of corrections and their reasons.
5. **Save `labels_v1.parquet`:** `id`, `type`, `strategy_primary`, `strategy_secondary` (actual list), `ambiguous` (bool), `obfuscated` (bool), `label_method`, `verified_by`, `label_notes`. **Exclude** NOT_ATTACK IDs. Run A's `src.data.schema.validate_labels()` against the cleaned dataset when available.
6. **Independent agreement test:** After initial labels are frozen, A selects 300 examples stratified by source × automatic strategy, prepares separate blind forms, and A/B annotate without seeing automatic labels or each other's choices. Compare primary strategies using Cohen's κ; report confusion matrix, per-source agreement, and how disagreements were adjudicated. Also manually audit a sample of excluded NOT_ATTACK cases separately to estimate mistaken exclusion risk.
7. **Return** label parquet and exclusion audit to A for the final merge and `strategy × source` counts. Keep dataset files private under the team's licence constraints.

### Labelling metadata (as defined in `data/SCHEMA.md`)

- `label_method=rule` if an unmodified rule suggestion determines the final label; `llm` if the LLM suggestion determines it; `manual` if human intervention **changed** the automatic label.
- `verified_by` is independent: leave empty if no human checked it; fill a GitHub handle if a reviewer checked and agreed; use `handle1;handle2` after adjudication with both reviewers.
- `label_notes` records significant ambiguity, source type correction, or unusual decisions; it is optional for straightforward cases.
- Do not interpret `label_method=manual` as meaning that every sample was independently double annotated.

## 5. Quality controls and decisions pending with A

- Confirm specific handling of ambiguous attacks in training (A's schema lists this as an open question); do not invent split policy in B's labeler.
- Ensure S1/S2/S3 are represented across sources before claiming LOSO reveals strategy generalization; record residual strategy × source confounding.
- Keep transformed variants out of the blind agreement sample unless deliberately evaluating transformation consistency; never confuse an obfuscation attribute with an attack strategy.
- Do not report near-perfect agreement before actually measuring it. Labeling guidelines should be refined on a small pilot set **before** blind agreement labels are revealed.
- Exclusion count must be reported as `N_NOT_ATTACK / 4,357`, broken down by source; inspection of excluded candidates is advisable because some can be genuine subtle attempts.
