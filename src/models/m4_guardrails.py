"""M4: off-the-shelf guardrails (reference only, no training).

    python -m src.models.m4_guardrails                               # both guardrails, all 11 splits
    python -m src.models.m4_guardrails --models protectai            # one guardrail

Each guardrail scores every row once (cached in results/<data>/preds/<model>/_all_rows.parquet);
metrics are then computed per split on that split's test rows, with the same 1% FPR calibration
threshold as M1-M3. Attack score = sum of the probabilities of the guardrail's attack classes:
  protectai   : P(INJECTION)
  promptguard : P(JAILBREAK) by default. Prompt Guard v1's INJECTION class is meant for third-party
                data (documents, web pages) and fires on almost every ordinary user prompt (P~1.0 on our
                benign calibration pool), so Meta's model card recommends JAILBREAK alone for user input.
                `--promptguard-merged` gives the proposal's INJECTION+JAILBREAK score as m4_promptguard_merged.

Meta Prompt Guard is a gated model: accept its licence on huggingface.co and run `hf auth login` first.
M4 is a reference, not a held-out test: its training data may overlap our sources.
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.models.common import Results, eval_rows, load_data, parse_splits, split_rows
from src.models.m3_finetune import pick_device

GUARDRAILS = {
    "protectai": "protectai/deberta-v3-base-prompt-injection-v2",
    "promptguard": "meta-llama/Prompt-Guard-86M",
}
ATTACK_CLASSES = {"protectai": ["INJECTION"], "promptguard": ["JAILBREAK"]}
MERGED_CLASSES = ["INJECTION", "JAILBREAK"]


@torch.no_grad()
def score_all(model_id: str, texts: list[str], device, batch_size: int, max_len: int) -> tuple[pd.DataFrame, float]:
    """Probability of every class (columns p_<LABEL>) for every text, plus latency in ms at batch size 1."""
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(model_id).to(device).eval()
    labels = [str(model.config.id2label[i]).upper() for i in range(model.config.num_labels)]
    print(f"  classes: {labels}")

    lengths = np.array([len(t) for t in texts])
    order = np.argsort(lengths)
    out = np.empty((len(texts), len(labels)), dtype=np.float32)
    for i in range(0, len(texts), batch_size):
        b = order[i:i + batch_size]
        enc = tok([texts[j] for j in b], truncation=True, max_length=max_len, padding=True,
                  return_tensors="pt").to(device)
        out[b] = torch.softmax(model(**enc).logits.float(), dim=-1).cpu().numpy()

    t0 = time.perf_counter()                       # latency at batch size 1
    for t in texts[:100]:
        model(**tok([t], truncation=True, max_length=max_len, return_tensors="pt").to(device))
    latency = 1000 * (time.perf_counter() - t0) / min(len(texts), 100)
    return pd.DataFrame(out, columns=[f"p_{n}" for n in labels]), latency


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", help="apig_v1.parquet (default: data/processed/apig_v1.parquet)")
    p.add_argument("--out", help="results folder (default: results/)")
    p.add_argument("--models", nargs="+", default=list(GUARDRAILS), choices=list(GUARDRAILS))
    p.add_argument("--splits", nargs="+", default=["all"])
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-len", type=int, default=512)
    p.add_argument("--device")
    p.add_argument("--force", action="store_true", help="re-score even if cached")
    p.add_argument("--promptguard-merged", action="store_true",
                   help="also report Prompt Guard with INJECTION+JAILBREAK merged (proposal's definition)")
    args = p.parse_args()

    df, data_path = load_data(args.data)
    device = pick_device(args.device)
    for name in args.models:
        model_id = GUARDRAILS[name]
        cache_dir = Results(args.out, data_path, f"m4_{name}").pred_dir
        cache = cache_dir / "_all_rows.parquet"
        scored = pd.read_parquet(cache) if cache.exists() and not args.force else None
        if scored is not None and any(c.startswith("p_") for c in scored.columns):
            print(f"[M4 {name}] using cached class probabilities {cache}")
        else:
            print(f"[M4 {name}] scoring {len(df):,} rows with {model_id} on {device}")
            probs, latency = score_all(model_id, df["text"].tolist(), device, args.batch_size, args.max_len)
            scored = pd.concat([df[["id"]].reset_index(drop=True), probs], axis=1).assign(latency_ms=latency)
            scored.to_parquet(cache, index=False)
        latency = float(scored["latency_ms"].iloc[0])

        variants = [(f"m4_{name}", ATTACK_CLASSES[name])]
        if name == "promptguard" and args.promptguard_merged:
            variants.append(("m4_promptguard_merged", MERGED_CLASSES))
        for tag, classes in variants:
            res = Results(args.out, data_path, tag)
            score_by_id = dict(zip(scored["id"], scored[[f"p_{c}" for c in classes]].sum(axis=1)))
            res.save_config({**vars(args), "model_id": model_id, "attack_classes": classes})
            for split in parse_splits(args.splits):
                preds = eval_rows(split_rows(df, split))
                preds["score"] = preds["id"].map(score_by_id)
                res.save(split, 0, preds, {"n_train": 0, "model_name": model_id, "attack_classes": "+".join(classes),
                                           "latency_ms": round(latency, 2)})
            print(f"Metrics: {res.metrics_file}")


if __name__ == "__main__":
    main()
