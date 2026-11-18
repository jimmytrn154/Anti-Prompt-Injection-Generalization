# Dataset card: `apig_v1_FAKE.parquet`

> **FAKE LABELS.** Strategies were assigned at random to preview the file format. Do not report any result computed on this file.

Built by `python -m src.data.build_dataset` on 2026-11-17 15:23 (3s), seed 42. Counts only, no dataset text.

## 1. Contents

| Subset | Attacks | Benign |
|---|---:|---:|
| main | 2,657 | 2,657 |
| calibration | 0 | 2,999 |
| notinject | 0 | 337 |

## 2. Strategy x source (main attacks)

| Strategy | deepset | gandalf | jailbreak | tensortrust | total |
|---|---:|---:|---:|---:|---:|
| S1 | 127 | 489 | 30 | 438 | 1,084 |
| S2 | 43 | 242 | 275 | 426 | 986 |
| S3 | 36 | 83 | 32 | 436 | 587 |
| total | 206 | 814 | 337 | 1,300 | 2,657 |

Type: 2,320 injection, 337 jailbreak. Ambiguous: 58. Obfuscated: 269.

## 3. Split sizes

| Split | Train (atk/ben) | Val (atk/ben) | Test (atk/ben) | Excluded | Length-only AUROC |
|---|---|---|---|---:|---:|
| split_random | 1,858 / 1,860 | 403 / 399 | 396 / 398 | 3,336 | 0.51 |
| loso_S1 | 1,208 / 1,920 | 212 / 339 | 1,056 / 398 | 3,517 | 0.46 |
| loso_S2 | 1,286 / 1,920 | 226 / 339 | 959 / 398 | 3,522 | 0.44 |
| loso_S3 | 1,599 / 1,920 | 283 / 339 | 584 / 398 | 3,527 | 0.50 |
| wsloso_S1 | 662 / 1,920 | 117 / 339 | 438 / 398 | 4,776 | 0.53 |
| wsloso_S2 | 675 / 1,920 | 119 / 339 | 426 / 398 | 4,773 | 0.54 |
| wsloso_S3 | 667 / 1,920 | 118 / 339 | 436 / 398 | 4,772 | 0.54 |
| lodo_deepset | 2,085 / 1,960 | 366 / 345 | 206 / 206 | 3,482 | 0.49 |
| lodo_gandalf | 1,568 / 1,566 | 275 / 277 | 814 / 814 | 3,336 | 0.51 |
| lodo_jailbreak | 1,974 / 1,636 | 346 / 288 | 337 / 337 | 3,732 | 0.49 |
| lodo_tensortrust | 1,156 / 1,219 | 201 / 215 | 1,300 / 1,223 | 3,336 | 0.48 |

*Length-only AUROC*: a logistic regression that sees only the token count, trained on each split's train part and scored on its test part. Close to 0.5 = length does not give the label away; far from 0.5 in *either* direction (e.g. 0.1 or 0.9) = length separates the classes.

## 4. Checks passed

- calibration and NotInject rows are never in train/val/test of any split
- LOSO: the held-out strategy (primary or secondary) never appears in training; tests contain only that strategy, no ambiguous rows
- Within-source LOSO uses Tensor Trust attacks only
- LODO: the held-out source (attacks and its benign data) never appears in training
- no id is in both training and test of any split (near-duplicates were removed globally in Phase 3)

## 5. How to use

```python
import pandas as pd
df = pd.read_parquet('data/processed/apig_v1_FAKE.parquet')
train = df[df.loso_S2 == 'train']; val = df[df.loso_S2 == 'val']; test = df[df.loso_S2 == 'test']
calib = df[df.subset == 'calibration']   # threshold = 99th percentile of benign scores here
notinject = df[df.subset == 'notinject'] # RQ3 test only
```
