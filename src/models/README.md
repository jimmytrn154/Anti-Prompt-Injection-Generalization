# Detectors M1–M4 (workstream C)

All four detectors share one evaluation path (`common.py`). For each split column:
- train on its `train` rows and use `val` for early stopping or C selection;
- set the threshold at **1% FPR on the calibration pool**;
- apply that threshold to the split's `test` rows and to NotInject.

| File | Model | Where to run | Runs |
|---|---|---|---|
| `m12_tfidf.py` | M1 TF-IDF + LR, M2 TF-IDF + linear SVM | laptop, ~1 min for everything | 11 splits each |
| `m3_finetune.py` | M3 DistilBERT (`--model-name` for others) | Kaggle GPU (`notebooks/kaggle_m3.ipynb`), or Mac MPS (~50 s/epoch) | 11 splits × 3 seeds |
| `m4_guardrails.py` | M4 ProtectAI DeBERTa-v3, Meta Prompt Guard | laptop, ~5–10 min per guardrail | scored once, evaluated on 11 splits |
| `summarize.py` | mean ± std over seeds, all models | anywhere | — |

## Commands (from the repo root)

```bash
python -m src.models.m12_tfidf --save-model           # M1 + M2
python -m src.models.m4_guardrails                     # M4 (Prompt Guard: accept licence on HF, then `hf auth login`)
python -m src.models.m3_finetune --smoke               # 1-minute code test
python -m src.models.m3_finetune --save-model          # M3, all splits x seeds 0 1 2
python -m src.models.summarize                         # -> results/apig_v1/summary.csv
```

Every script takes `--data <file>` (default `data/processed/apig_v1.parquet`) and `--splits` (column names or prefixes, such as `loso` or `lodo`). Finished M3 runs are skipped, so an interrupted job can be restarted.

## M4 attack score (changed from the proposal)

The proposal merges injection and jailbreak labels into one score. That works for ProtectAI, which has the classes SAFE and INJECTION. It does not work for **Prompt Guard v1**:
- Its INJECTION class is meant for third-party data such as documents and web pages, not for user turns.
- It gives P(INJECTION) ≈ 1.0 to almost every ordinary user prompt in our calibration pool.
- With the merged score, AUROC is below 0.3.

Meta's model card recommends the JAILBREAK class alone for user input, so `m4_promptguard` uses **P(JAILBREAK)**. `--promptguard-merged` also writes the proposal's version as `m4_promptguard_merged` for the report. The cache keeps every class probability, so switching definitions needs no re-scoring.

## Outputs (`results/<data file name>/`)

| Path | Content |
|---|---|
| `metrics/<model>.csv` | One row per split × seed: AUROC, F1, FNR, FPR, FPR on NotInject, cost (10:1), FNR per type and per strategy, threshold, latency, training time |
| `preds/<model>/<split>_seed<k>.parquet` | `id`, `role` (test/calibration/notinject), `label`, `score`. **No text.** D can join these with the dataset to compute bootstrap CIs, ROC/DET curves and per-attribute results |
| `models/` | split_random models (`--save-model`) for M1 coefficients, M3 Integrated Gradients and the demo (git-ignored) |

Results from `apig_v1_FAKE.parquet` go to `results/apig_v1_FAKE/` (git-ignored). Never report them.
