# Dataset card: `apig_v1_FAKE.parquet`

> **FAKE LABELS.** Strategies were assigned at random to preview the file format. Do not report any result computed on this file.

Built by `python -m src.data.build_dataset` on 2026-11-19 14:30 (125s), seed 42. Counts only, no dataset text.

## 1. Contents

| Subset | Attacks | Benign |
|---|---:|---:|
| main | 2,657 | 2,657 |
| calibration | 0 | 3,001 |
| notinject | 0 | 337 |

## 2. Strategy x source (main attacks)

| Strategy | deepset | gandalf | jailbreak | tensortrust | total |
|---|---:|---:|---:|---:|---:|
| S1 | 127 | 489 | 30 | 439 | 1,085 |
| S2 | 43 | 242 | 275 | 421 | 981 |
| S3 | 36 | 83 | 32 | 440 | 591 |
| total | 206 | 814 | 337 | 1,300 | 2,657 |

Type: 2,320 injection, 337 jailbreak. Ambiguous: 58. Obfuscated: 275.

## 3. Split sizes

| Split | Train (atk/ben) | Val (atk/ben) | Test (atk/ben) | Excluded | Length-only AUROC |
|---|---|---|---|---:|---:|
| split_random | 1,858 / 1,860 | 403 / 399 | 396 / 398 | 3,338 | 0.49 |
| loso_S1 | 1,202 / 1,920 | 210 / 339 | 1,057 / 398 | 3,526 | 0.46 |
| loso_S2 | 1,285 / 1,920 | 226 / 339 | 954 / 398 | 3,530 | 0.46 |
| loso_S3 | 1,589 / 1,920 | 281 / 339 | 588 / 398 | 3,537 | 0.45 |
| wsloso_S1 | 656 / 1,920 | 115 / 339 | 439 / 398 | 4,785 | 0.51 |
| wsloso_S2 | 674 / 1,920 | 119 / 339 | 421 / 398 | 4,781 | 0.52 |
| wsloso_S3 | 657 / 1,920 | 116 / 339 | 440 / 398 | 4,782 | 0.51 |
| lodo_deepset | 2,085 / 1,959 | 366 / 346 | 206 / 206 | 3,484 | 0.49 |
| lodo_gandalf | 1,568 / 1,567 | 275 / 276 | 814 / 814 | 3,338 | 0.50 |
| lodo_jailbreak | 1,974 / 1,635 | 346 / 289 | 337 / 337 | 3,734 | 0.49 |
| lodo_tensortrust | 1,156 / 1,219 | 201 / 215 | 1,300 / 1,223 | 3,338 | 0.48 |

*Length-only AUROC*: a logistic regression that sees only the token count, trained on each split's train part and scored on its test part. Close to 0.5 = length does not give the label away; far from 0.5 in *either* direction (e.g. 0.1 or 0.9) = length separates the classes.

## 4. Shortcut markers (main split)

Text fixes applied at build time: 1,554 Tensor Trust texts had "access granted" replaced by an ordinary phrase (4 then fell outside the token range and were dropped); short CJK texts removed: {'tensortrust': 24}.

| Pattern | In attacks | In benign | Flag |
|---|---:|---:|---|
| 'Understood' | 1.8% | 0.0% |  |
| 'Welcome' | 1.5% | 0.1% |  |
| 'Great' | 1.5% | 1.8% |  |
| 'Done' | 1.4% | 0.4% |  |
| 'Verified' | 1.4% | 0.1% |  |
| 'Authorized' | 1.4% | 0.0% |  |
| 'Thank you' | 1.2% | 0.5% |  |
| 'Yes' | 1.1% | 1.8% |  |
| empty quotes | 1.1% | 0.2% |  |
| 'Got it' | 1.1% | 0.0% |  |
| 'Ready' | 1.0% | 0.5% |  |
| 'Unlocked' | 0.9% | 0.0% |  |
| 'Fine' | 0.9% | 0.3% |  |
| 'Accepted' | 0.9% | 0.0% |  |
| 'Of course' | 0.9% | 0.3% |  |
| 'Very well' | 0.8% | 0.0% |  |
| 'Welcome back' | 0.8% | 0.0% |  |
| 'Confirmed' | 0.8% | 0.2% |  |
| 'Go ahead' | 0.8% | 0.0% |  |
| 'Excellent' | 0.7% | 0.2% |  |
| 'Entry allowed' | 0.7% | 0.0% |  |
| 'All clear' | 0.7% | 0.0% |  |
| 'Success' | 0.6% | 0.6% |  |
| 'Noted' | 0.6% | 0.2% |  |
| 'Perfect' | 0.6% | 0.3% |  |
| 'Sounds good' | 0.6% | 0.0% |  |
| 'Alright' | 0.6% | 0.1% |  |
| 'Certainly' | 0.6% | 0.1% |  |
| 'I have been PWNED' | 0.6% | 0.0% |  |
| 'Approved' | 0.5% | 0.3% |  |
| 'Good morning' | 0.5% | 0.0% |  |
| 'Permission given' | 0.5% | 0.0% |  |
| 'Hello world' | 0.5% | 0.1% |  |
| 'Hi there' | 0.5% | 0.0% |  |
| 'Sure thing' | 0.4% | 0.0% |  |

⚠ = at least 2% of attacks and 5x more common than in benign text (a possible shortcut).

## 5. Checks passed

- calibration and NotInject rows are never in train/val/test of any split
- LOSO: the held-out strategy (primary or secondary) never appears in training; tests contain only that strategy, no ambiguous rows
- Within-source LOSO uses Tensor Trust attacks only
- LODO: the held-out source (attacks and its benign data) never appears in training
- no id is in both training and test of any split (near-duplicates were removed globally in Phase 3)

## 6. How to use

```python
import pandas as pd
df = pd.read_parquet('data/processed/apig_v1_FAKE.parquet')
train = df[df.loso_S2 == 'train']; val = df[df.loso_S2 == 'val']; test = df[df.loso_S2 == 'test']
calib = df[df.subset == 'calibration']   # threshold = 99th percentile of benign scores here
notinject = df[df.subset == 'notinject'] # RQ3 test only
```
