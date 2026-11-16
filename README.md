# APIG: Do prompt-injection detectors generalize to unseen and evasive attacks?

**Anti Prompt-Injection Generalization (APIG)** is our team project for IE30601 Applied Machine Learning (UNIST, Fall 2026), Group 7. We evaluate lightweight ML guardrails on attack strategies, sources and rewrites they never saw in training.

> **Status:** proposal submitted; implementation not started. Everything below describes the *plan* from [IE30601_Proposal_Group7.pdf](IE30601_Proposal_Group7.pdf), which is the source of truth. Items marked **planned** or **TBD** do not exist yet.

## Overview

LLM-integrated applications (chatbots, retrieval-augmented and agentic systems) cannot reliably separate the developer's instructions from text supplied by users. A common defense is a lightweight **guardrail classifier** that screens every input before it reaches the LLM.

These detectors look near-perfect on random train/test splits, but attackers invent new styles and rewrite old attacks. A missed attack (false negative) is a security failure; blocking legitimate users (false positive) gets the guardrail switched off. Prior work shows deployed guardrails can be evaded by character-level perturbations; we extend this to **unseen attack strategies and sources**, and test whether augmentation closes the gap and at what false-alarm cost.

**Task.** Binary text classification.

- Input `X`: a single user-turn text.
- Target `y`: `1` if the text attempts to hijack the model's instructions (direct prompt injection and jailbreaks), `0` if benign.
- Injection and jailbreak results are reported separately.
- **Out of scope:** indirect injection via retrieved content. Its maliciousness depends on context and trust boundaries that a single-text detector cannot observe.

## Research questions

| RQ | Question | Main metric |
|---|---|---|
| **RQ1 Generalization** | What fraction of attacks from a strategy or source absent from training is missed? | FNR |
| **RQ2 Robustness** | How often do simple rewrites turn a detected attack into a missed one, and does augmentation reduce this? | Evasion rate (= induced FNR) |
| **RQ3 False alarms** | How often are benign but suspicious-looking inputs (e.g. "How do I ignore whitespace in Python?") wrongly blocked? | FPR |

## Datasets

A = attack (`y=1`), B = benign (`y=0`). Strategy sets are *expected* and will be confirmed by labeling.

| Source | Class | Expected strategies | Original → used | Handling |
|---|---|---|---|---|
| [deepset/prompt-injections](https://huggingface.co/datasets/deepset/prompt-injections) | A+B | S1, S3 | 662 (263 A / 399 B) → all | EN/DE; natural German attacks and benign |
| Lakera Gandalf (`gandalf_ignore_instructions`) | A | S1, S2 | 1,000 → ≤1,000 | Real attempts; de-duplicated |
| [jackhhao/jailbreak-classification](https://huggingface.co/datasets/jackhhao/jailbreak-classification) | A+B | S2 (S3) | 1,306 (666 A / 640 B) → all | Jailbreak type; reported separately |
| Tensor Trust | A | S1–S3 | 563k (295k distinct) → ≤1.2k | Successful, English; cluster sampling; game phrase "access granted" masked |
| Dolly-15k / Alpaca | B | – | 15,011 / 52,002 → ≈1.7k (+3k calib.) | Subsampled to balance; length-matched; separate calibration pool |
| NotInject | B (hard) | – | 339 → all | Benign text with trigger words; **test only** (RQ3) |

Parentheses mark a secondary strategy.

- **Expected size:** ≈2.7k attacks and ≈2.7k benign (≈1:1), plus 339 NotInject hard negatives (test only). Exact counts follow de-duplication.
- **Calibration pool:** a separate, held-out pool of ≈3k ordinary benign prompts (from Dolly/Alpaca) used only to set the detection threshold.
- **Licences:** check each dataset's licence and terms before downloading, redistributing or committing any data. Do not commit raw data to the repo until this is done.

**Preprocessing (planned):**

1. Filter by language and length (5–300 tokens).
2. Remove exact and near-duplicates (embedding cosine > 0.9) *before* splitting.
3. Subsample large sources by k-means on sentence embeddings, with per-strategy quotas.
4. Length- and category-match benign data to attacks so classes cannot be separated by dataset style.
5. Fixed seeds; IDs logged.

## Attack taxonomy

No public dataset groups attacks by style, so we label them ourselves.

| Code | Strategy | Description |
|---|---|---|
| **S1** | Direct override | Tells the model to ignore or replace its instructions |
| **S2** | Role-play / persona | Asks the model to adopt a persona or role that bypasses its rules |
| **S3** | Context manipulation | Fake system or dialogue delimiters, or instructions embedded in quoted text supplied by the user |

**Primary vs. secondary.** Each attack gets exactly one **primary** strategy (S1–S3) and zero or more **secondary** strategies. Secondary labels matter for LOSO: an attack whose secondary strategy is the held-out one is removed from training, so the held-out strategy is truly unseen.

**Attributes** (per attack):

- `type`: injection or jailbreak
- `language`
- `obfuscated`: natural obfuscation present in the original text

Strategy and surface form are separate axes. RQ1 holds out *strategies*; surface changes (obfuscation, encoding, language) are studied in RQ2.

**Labeling process:** keyword rules plus an LLM pass; two members independently verify 300 samples and we report Cohen's κ. Ambiguous attacks are excluded from LOSO test sets.

## Data schema

> **Proposed; to be confirmed by the team.** One shared table format for all sources, so every script reads the same file.

| Column | Description |
|---|---|
| `id` | Unique, stable example ID (logged for reproducibility) |
| `text` | The user-turn text |
| `label` | `1` attack, `0` benign |
| `source` | Originating dataset |
| `strategy_primary` | `S1` / `S2` / `S3` (attacks only) |
| `strategy_secondary` | Zero or more of `S1`–`S3` (attacks only) |
| `type` | `injection` / `jailbreak` |
| `language` | Language code |
| `obfuscated` | Natural obfuscation present (bool) |
| `split_random` | Role in the random 70/15/15 split |
| `split_loso` | Leave-one-strategy-out assignment |
| `split_within_source_loso` | Within-source LOSO assignment (Tensor Trust only) |
| `split_lodo` | Leave-one-dataset-out assignment |

How fold-based split columns encode "which fold" is **TBD** (e.g. the fold in which each example is a test example, with the training set derived by rule).

## Evaluation splits

| Split | Definition |
|---|---|
| **Random** | Stratified 70/15/15 (in-distribution baseline) |
| **LOSO** (leave-one-strategy-out) | For each Si: train on the other strategies plus benign data; test on Si plus held-out benign. Training examples whose *secondary* strategy is Si are removed |
| **Within-source LOSO** | LOSO on Tensor Trust only, the only source spanning S1–S3. Holds the source fixed to isolate the strategy effect |
| **LODO** (leave-one-dataset-out) | Hold out each source, including that source's benign data |
| **NotInject, evasion sets** | Test-only; never used for training |

- Strategies and sources are correlated, so LOSO vs. LODO only helps *disentangle* the two effects; we report a **strategy × source table**.
- **Validation and early stopping use training strategies/sources only.**

## Models

All detectors are cheap enough for a real-time guardrail.

| ID | Model | Notes |
|---|---|---|
| **M1** | TF-IDF (word 1–2-grams + char 3–5-grams) + Logistic Regression | |
| **M2** | Same TF-IDF + linear SVM | |
| **M3** | Fine-tuned DistilBERT (or DeBERTa-v3-small) | AdamW, early stopping, 3 seeds (mean ± std) |
| **M4** | Off-the-shelf ProtectAI DeBERTa-v3 and Meta Prompt Guard | **Reference only**, not a held-out test: training data may overlap our sources, ProtectAI is English-only, and it is not designed for jailbreaks. Injection/jailbreak labels merged to 1 |

## Evasion attacks and defense

**Evasion (RQ2).** Label-preserving rewrites of attacks that a detector already catches:

| ID | Transform |
|---|---|
| **T1** | Character perturbation (leetspeak, spaces, homoglyphs, zero-width characters) |
| **T2** | Base64 / ROT13 with a "decode and follow" wrapper |
| **T3** | Paraphrase by a small LLM |
| **T4** | Translation into Korean / Vietnamese |

**Defense.** Retrain M1–M3 with T1–T4-augmented attacks. T3–T4 are also applied to *benign* prompts so the detector cannot learn "non-English or unusual = attack". Translated benign test prompts are disjoint from those used in training. We measure the recovered LOSO/evasion gap and the FPR cost on NotInject and translated benign data.

## Metrics

- **One operating point for all models:** the threshold giving **1% FPR on the held-out benign calibration pool**. FNR and FPR are always reported together.
- AUROC, F1, FNR, FPR; evasion rate per transform; ROC/DET curves.
- Cost-weighted error `c_FN·FNR + c_FP·FPR` with `c_FN : c_FP = 10:1` (sensitivity analysis 1:1 to 50:1).
- Defense effect: ΔFNR (LOSO), Δevasion rate, ΔFPR.
- Bootstrap 95% CIs; folds averaged; M3 over 3 seeds.
- All results split by `type` (injection vs. jailbreak) and by attribute (language, natural obfuscation).
- Latency and model size.
- **Interpretability:** M1 coefficients and Integrated Gradients on M3, to test the hypothesis that detectors key on a few trigger words.

| Question | Setting | Metrics |
|---|---|---|
| In-distribution | Random split | AUROC, F1, FNR, FPR |
| RQ1 | LOSO; within-source LOSO; LODO | FNR on unseen strategy/source; AUROC; drop vs. random split |
| RQ2 | T1–T4 on detected attacks | Evasion rate per transform |
| RQ3 | NotInject; translated benign | FPR at the same operating point |
| Trade-off | All test sets | ROC/DET; cost-weighted error |
| Defense | Augmented vs. original training | ΔFNR (LOSO), Δevasion rate, ΔFPR |

## Repository structure

**Current** (what exists today):

```
.
├── IE30601_Proposal_Group7.pdf
└── README.md
```

**Planned** (not created yet; subject to change):

```
.
├── data/
│   ├── raw/            # downloaded source datasets (check licences first)
│   └── processed/      # unified table following the data schema
├── src/
│   ├── data/           # loading, preprocessing, de-duplication, splits
│   ├── labeling/       # keyword rules, LLM pass, agreement check
│   ├── models/         # M1–M4
│   ├── attacks/        # T1–T4 evasion transforms and augmentation
│   └── eval/           # shared evaluation script, metrics, plots
├── notebooks/
├── results/
└── reports/
```

## Setup

**TBD.** There is no requirements file, environment or code in the repo yet. Expected tools:

- Python
- scikit-learn
- PyTorch
- Hugging Face `transformers` and `datasets`

Exact versions and install commands will be added once the environment is agreed.

## Team and roles

Group 7: Duong Hien Chi Kien, Tran Anh Chuong, Sangah Park, Allaberdi Zinkhaydarov.

Roles are not final.

| Workstream | Scope | Owner |
|---|---|---|
| **A. Data & splits** | Source datasets, preprocessing, de-duplication, benign matching, all splits | TBD |
| **B. Labeling** | Strategy/attribute labels, labeling rules, 300-sample agreement check (Cohen's κ) | TBD |
| **C. Models** | M1–M4 training, threshold calibration | TBD |
| **D. Robustness & evaluation** | T1–T4, augmentation defense, shared eval script, metrics, interpretability, figures | TBD |

## Timeline

| Dates | Milestones |
|---|---|
| Nov 16–22 | Data collection, labeling rules, shared evaluation script |
| Nov 23–28 | Labels; 300-sample check (Cohen's κ); all splits; M1–M4 baselines. *Note: Quiz II on Nov 26* |
| Nov 29 – Dec 5 | Within-source LOSO, evasion attacks, augmentation, interpretability, figures |
| **Dec 8** | Slides due |
| **Dec 10** | Presentation |
| **Dec 15** | After-presentation report |

## Results

*To be added. No results have been produced yet.*

## Limitations

- **Single-text input ignores trust boundaries**, so indirect injection is out of scope.
- **Strategy and source remain partly confounded across datasets.** LODO and within-source LOSO quantify this but do not fully remove it.

## References

1. K. Greshake et al. "Not What You've Signed Up For: Compromising Real-World LLM-Integrated Applications with Indirect Prompt Injection." AISec Workshop, ACM CCS, 2023.
2. F. Perez and I. Ribeiro. "Ignore Previous Prompt: Attack Techniques for Language Models." NeurIPS ML Safety Workshop, 2022.
3. Y. Liu et al. "Formalizing and Benchmarking Prompt Injection Attacks and Defenses." USENIX Security, 2024.
4. S. Toyer et al. "Tensor Trust: Interpretable Prompt Injection Attacks from an Online Game." ICLR, 2024.
5. J. Yi et al. "Benchmarking and Defending Against Indirect Prompt Injection Attacks on Large Language Models." arXiv:2312.14197, 2023.
6. V. Sanh et al. "DistilBERT, a Distilled Version of BERT: Smaller, Faster, Cheaper and Lighter." arXiv:1910.01108, 2019.
7. M. Sundararajan, A. Taly, and Q. Yan. "Axiomatic Attribution for Deep Networks." ICML, 2017.
8. Meta. "Prompt Guard (Llama Prompt Guard 86M) Model Card." Hugging Face, 2024.
9. ProtectAI. "deberta-v3-base-prompt-injection-v2." Hugging Face model card, 2024.
10. N. Jain et al. "Baseline Defenses for Adversarial Attacks Against Aligned Language Models." arXiv:2309.00614, 2023.
11. deepset. "prompt-injections" dataset. Hugging Face.
12. Lakera. "gandalf_ignore_instructions" dataset. Hugging Face.
13. jackhhao. "jailbreak-classification" dataset. Hugging Face.
14. Databricks. "databricks-dolly-15k" dataset. Hugging Face, 2023.
15. R. Taori et al. "Stanford Alpaca: An Instruction-Following LLaMA Model." GitHub, 2023.
16. H. Li and X. Liu. "InjecGuard: Benchmarking and Mitigating Over-defense in Prompt Injection Guardrail Models." arXiv:2410.22770, 2024. *(NotInject benchmark)*
17. W. Hackett et al. "Bypassing LLM Guardrails: An Empirical Analysis of Evasion Attacks against Prompt Injection and Jailbreak Detection Systems." arXiv:2504.11168, 2025.

## Ethics note

Attack prompts in this project are used only to evaluate and improve defenses. They are kept generic and are not aimed at any real deployed system.
