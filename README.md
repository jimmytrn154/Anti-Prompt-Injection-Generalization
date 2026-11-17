# APIG: Do prompt-injection detectors generalize to unseen and evasive attacks?

**Anti Prompt-Injection Generalization (APIG)** is our team project for IE30601 Applied Machine Learning (UNIST, Fall 2026), Group 7. We evaluate lightweight ML guardrails on attack strategies, sources and rewrites they never saw in training.

> **Status (Nov 17):** the **data pipeline is implemented** (download → preprocessing → dataset build with all splits). **Strategy labelling is in progress.** Models and evasion experiments have not started. The plan follows [IE30601_Proposal_Group7.pdf](IE30601_Proposal_Group7.pdf); deviations agreed since submission are marked **(changed)**.

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

A = attack (`y=1`), B = benign (`y=0`). Counts are from the current build; final numbers can shift slightly once strategy labels are in.

| Source | Class | Raw | After cleaning | In final dataset | Notes |
|---|---|---:|---:|---:|---|
| [deepset/prompt-injections](https://huggingface.co/datasets/deepset/prompt-injections) | A+B | 263 A / 399 B | 206 A / 247 B | all A; B length-matched | EN + DE (~40% German) |
| [Lakera Gandalf](https://huggingface.co/datasets/Lakera/gandalf_ignore_instructions) | A | 1,000 | 814 | 814 | |
| [jackhhao/jailbreak-classification](https://huggingface.co/datasets/jackhhao/jailbreak-classification) | A+B | 666 A / 640 B | 337 A / 607 B | all A; B length-matched | Jailbreak type; reported separately |
| [Tensor Trust](https://github.com/HumanCompatibleAI/tensor-trust-data) (raw dump v2) | A | 563,349 | 4,256 | 1,300 **(changed: ≤1.2k → 1.3k)** | Successful, non-self, distinct attacks; "access granted" removed; per-strategy quotas |
| [Dolly-15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k) / [Alpaca](https://huggingface.co/datasets/tatsu-lab/alpaca) | B | 15,011 / 52,002 | 13,349 / 19,172 | length-matched filler + calibration pool | Dolly = instruction **+ context** **(changed)** for long benign prompts |
| [NotInject](https://huggingface.co/datasets/leolee99/NotInject) | B (hard) | 339 | 337 | 337, **test only** (RQ3) | Includes a *Multilingual* category (~82 rows, mostly Chinese) |

**Current build:** 2,657 attacks + 2,657 length-matched benign (1:1), a 2,999-prompt benign **calibration pool** (only for setting the threshold), and 337 NotInject hard negatives. Data is **never committed** (licences, see [data/LICENSES.md](data/LICENSES.md)); raw data is rebuilt with the scripts below, and the processed dataset is shared privately within the team.

**Preprocessing** (`src/data/preprocess.py`; report in [reports/data/preprocess_report.md](reports/data/preprocess_report.md)):

1. Tensor Trust: keep successful, non-self, non-password, distinct attacks (563,349 → 9,829); delete the game phrase "access granted" (a placeholder token would itself be a shortcut).
2. Length filter **5–500 tokens (changed from 5–300)**: 58% of jailbreak attacks are > 300 tokens; 500 keeps ~2/3 of them and stays under every model's 512-token limit.
3. Language filter: English; English + German for deepset; NotInject kept as published. Repeated padding is collapsed and homoglyph words are treated as obfuscation before detection, so padded English attacks are not misread as German/Italian/Russian.
4. Exact duplicates across all sources; near-duplicates (embedding cosine > 0.9) within each class, greedy so that dissimilar texts are never chained together.
5. Tensor Trust candidate pool for labelling by k-means cluster sampling (3,000 diverse attacks).
6. Fixed seeds; every removed row is logged with its reason.

## Attack taxonomy

No public dataset groups attacks by style, so we label them ourselves.

| Code | Strategy | Description |
|---|---|---|
| **S1** | Direct override | Tells the model to ignore or replace its instructions |
| **S2** | Role-play / persona | Asks the model to adopt a persona or role that bypasses its rules |
| **S3** | Context manipulation | Fake system or dialogue delimiters, or instructions embedded in quoted text supplied by the user |

**Primary vs. secondary.** Each attack gets exactly one **primary** strategy (S1–S3) and zero or more **secondary** strategies. An attack whose secondary strategy is the held-out one is removed from LOSO training, so the held-out strategy is truly unseen.

**Attributes** (per attack): `type` (injection / jailbreak), `language`, `obfuscated` (natural obfuscation in the original text). Strategy and surface form are separate axes: RQ1 holds out *strategies*; surface changes are studied in RQ2.

**Labelling process:** keyword rules plus an LLM pass, delivered as `labels_v1.parquet` (`id` + label columns) and merged by `id`. `label_method` records how the final label was decided (`rule` / `llm` / `manual`) and `verified_by` records manual checks. **300-sample agreement check:** drawn stratified by source × strategy; two members annotate blind (without the automatic label); we report Cohen's κ (`src/data/agreement.py`). Ambiguous attacks are excluded from LOSO test sets. Full workflow: [data/SCHEMA.md](data/SCHEMA.md).

## Data format

One table, `data/processed/apig_v1.parquet`, for all rows (attacks, benign, calibration, NotInject); definition and validator in [`src/data/schema.py`](src/data/schema.py), description in [data/SCHEMA.md](data/SCHEMA.md). Instead of separate train/test files, **every split is a column** whose value is `train` / `val` / `test` / `excluded`:

```python
import pandas as pd
df = pd.read_parquet("data/processed/apig_v1.parquet")
train, val, test = (df[df.loso_S2 == p] for p in ("train", "val", "test"))
calib = df[df.subset == "calibration"]   # 1% FPR threshold
notinject = df[df.subset == "notinject"] # RQ3, test only
```

## Evaluation splits

11 split columns, built by `src/data/build_dataset.py`:

| Column(s) | Definition |
|---|---|
| `split_random` | Stratified 70/15/15 (in-distribution baseline) |
| `loso_S1`, `loso_S2`, `loso_S3` | Train on the other strategies plus benign data; test on Si plus held-out benign. Training attacks with Si as *secondary* strategy are removed; ambiguous Si attacks are excluded |
| `wsloso_S1`, `wsloso_S2`, `wsloso_S3` | LOSO on Tensor Trust attacks only (the only source spanning S1–S3), to isolate the strategy effect |
| `lodo_deepset`, `lodo_gandalf`, `lodo_jailbreak`, `lodo_tensortrust` | Hold out each attack source together with its own benign data |
| `subset = calibration / notinject` | Never used for training; evasion sets are test-only too |

- Strategies and sources are correlated, so LOSO vs. LODO only helps *disentangle* the two effects; we report a **strategy × source table** and use within-source LOSO as the control.
- **Validation and early stopping use training strategies/sources only.**
- **Length matching:** raw sources differ in length between classes (e.g. jailbreak attacks median 354 tokens vs. benign 11–14). Benign data and every test set are length-matched to the attacks; a **length-only baseline scores 0.44–0.54 AUROC on all 11 splits**, so length does not give the label away. Automatic leak checks run on every build (see the dataset card in `reports/data/`).

## Models

All detectors are cheap enough for a real-time guardrail.

| ID | Model | Notes |
|---|---|---|
| **M1** | TF-IDF (word 1–2-grams + char 3–5-grams) + Logistic Regression | |
| **M2** | Same TF-IDF + linear SVM | |
| **M3** | Fine-tuned DistilBERT (or DeBERTa-v3-small) | AdamW, early stopping, 3 seeds (mean ± std). ~40% of deepset is German: consider a multilingual checkpoint or report German separately |
| **M4** | Off-the-shelf ProtectAI DeBERTa-v3 and Meta Prompt Guard | **Reference only**, not a held-out test: training data may overlap our sources, ProtectAI is English-only, and it is not designed for jailbreaks. Injection/jailbreak labels merged to 1 |

## Evasion attacks and defense

**Evasion (RQ2).** Label-preserving rewrites of attacks that a detector already catches:

| ID | Transform |
|---|---|
| **T1** | Character perturbation (leetspeak, spaces, homoglyphs, zero-width characters) |
| **T2** | Base64 / ROT13 with a "decode and follow" wrapper |
| **T3** | Paraphrase by a small LLM |
| **T4** | Translation into Korean / Vietnamese |

Note: T1, T2 and T4 lengthen texts (Base64 ~2–4× in tokens), so some rewritten attacks exceed 512 tokens; report or limit these so that evasion is not caused by truncation.

**Defense.** Retrain M1–M3 with T1–T4-augmented attacks. T3–T4 are also applied to *benign* prompts so the detector cannot learn "non-English or unusual = attack". Translated benign test prompts are disjoint from those used in training. We measure the recovered LOSO/evasion gap and the FPR cost on NotInject and translated benign data.

## Metrics

- **One operating point for all models:** the threshold giving **1% FPR on the held-out benign calibration pool**. FNR and FPR are always reported together.
- AUROC, F1, FNR, FPR; evasion rate per transform; ROC/DET curves.
- Cost-weighted error `c_FN·FNR + c_FP·FPR` with `c_FN : c_FP = 10:1` (sensitivity analysis 1:1 to 50:1).
- Defense effect: ΔFNR (LOSO), Δevasion rate, ΔFPR.
- Bootstrap 95% CIs; folds averaged; M3 over 3 seeds.
- All results split by `type` (injection vs. jailbreak) and by attribute (language, natural obfuscation); RQ3 also per NotInject category.
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

```
.
├── IE30601_Proposal_Group7.pdf
├── README.md
├── SETUP.md                  # environment + data commands
├── requirements.txt
├── data/                     # data files are git-ignored
│   ├── SCHEMA.md             # table format + labelling workflow
│   ├── LICENSES.md
│   ├── raw/                  # downloads (local only)
│   └── processed/            # clean_v0, labelling pool, apig_v1 (local / shared privately)
├── src/data/
│   ├── check_env.py          # verify the environment
│   ├── schema.py             # table definition + validators
│   ├── sources.py            # where each dataset comes from, how it is read
│   ├── download.py           # Phase 1: download everything
│   ├── inspect_raw.py        # Phase 2: raw data report
│   ├── preprocess.py         # Phase 3: filter, mask, de-duplicate, labelling pool
│   ├── agreement.py          # 300-sample double annotation + Cohen's κ
│   └── build_dataset.py      # Phases 4–5: sampling, length matching, splits, checks
└── reports/data/             # count-only reports (committed)
```

Planned: `src/labeling/` (B), `src/models/` (C), `src/attacks/` and `src/eval/` (D), `results/`.

## Setup

Python 3.11+ (tested on 3.13, Windows). Full instructions in [SETUP.md](SETUP.md).

```powershell
py -3.13 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m src.data.check_env --models          # should end with "All checks passed."

python -m src.data.download                    # raw data -> data/raw/ (~10 min)
python -m src.data.inspect_raw                 # reports/data/raw_inspection.md
python -m src.data.preprocess                  # data/processed/clean_v0.parquet + labelling pool (~15 min)
python -m src.data.build_dataset --labels data/processed/labels_v1.parquet   # final apig_v1.parquet
```

`python -m src.data.build_dataset --fake-labels` builds `apig_v1_FAKE.parquet` (random strategy labels) so model and evaluation code can be written before the real labels exist. Never report results from it.

## Team and roles

Group 7: Duong Hien Chi Kien, Tran Anh Chuong, Sangah Park, Allaberdi Zinkhaydarov.

| Workstream | Scope | Owner |
|---|---|---|
| **A. Data & splits** | Datasets, preprocessing, de-duplication, benign matching, all splits, dataset build | Allaberdi Zinkhaydarov |
| **B. Labelling + M1/M2** | S1–S3 guidelines, rule/LLM labelling, 300-sample agreement check (with A), M1/M2, M1 coefficient analysis | Sangah Park |
| **C. Models** | M3, M4, Integrated Gradients, demo | TBD |
| **D. Robustness & evaluation** | Shared evaluation script, T1–T4, augmentation defense, figures | TBD |

## Timeline

| Dates | Milestones | Status |
|---|---|---|
| Nov 16–22 | Data collection, labelling rules, shared evaluation script | Data pipeline done; labelling started |
| Nov 23–28 | Labels; 300-sample check (Cohen's κ); all splits; M1–M4 baselines. *Quiz II on Nov 26* | Split code ready (waiting for labels) |
| Nov 29 – Dec 5 | Within-source LOSO, evasion attacks, augmentation, interpretability, figures | |
| **Dec 8** | Slides due | |
| **Dec 10** | Presentation | |
| **Dec 15** | After-presentation report | |

## Results

*To be added. No results have been produced yet.*

## Limitations

- **Single-text input ignores trust boundaries**, so indirect injection is out of scope.
- **Strategy and source remain partly confounded across datasets.** LODO and within-source LOSO quantify this but do not fully remove it.
- **Label noise in public datasets:** some deepset "attacks" are harmless requests behind a task switch, i.e. attacks only relative to an unstated application; these are flagged as ambiguous where unclear.
- **Length matching** removes a real-world difference between classes on purpose; LODO test benign data is partly drawn from Dolly/Alpaca rather than only the held-out source.

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
