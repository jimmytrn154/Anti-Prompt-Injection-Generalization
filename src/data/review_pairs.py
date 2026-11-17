"""For each attack, its closest benign text if they are almost identical (possible label errors).  Run from the repo root:

    python -m src.data.review_pairs              # cosine > 0.90 (the Phase 3 threshold)
    python -m src.data.review_pairs --min 0.85   # a looser look

Reads data/processed/clean_v0.parquet + clean_v0_embeddings.npy (from preprocess.py) and writes
data/processed/review_pairs.csv (git-ignored: contains text). Open it in Excel and fill `decision`:
  ok            genuinely different despite the similarity (keep both)
  attack_wrong  the "attack" is really benign      -> tell B (ambiguous / relabel)
  benign_wrong  the "benign" text is really an attack -> drop it in preprocessing
  unsure
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from src.data.sources import REPO_ROOT

PROC = REPO_ROOT / "data" / "processed"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--min", type=float, default=0.90, help="minimum cosine similarity (default 0.90)")
    args = p.parse_args()

    df = pd.read_parquet(PROC / "clean_v0.parquet")
    emb = np.load(PROC / "clean_v0_embeddings.npy")
    if len(df) != len(emb):
        raise SystemExit("clean_v0.parquet and clean_v0_embeddings.npy do not match; re-run preprocess.py")

    a_idx = np.flatnonzero(df["label"].to_numpy() == 1)
    b_idx = np.flatnonzero(df["label"].to_numpy() == 0)
    rows = []
    for s in range(0, len(a_idx), 1024):
        block = a_idx[s:s + 1024]
        sims = emb[block] @ emb[b_idx].T
        best = sims.argmax(axis=1)                       # closest benign text for each attack
        top = sims[np.arange(len(block)), best]
        rows += [(block[i], b_idx[best[i]], float(top[i])) for i in np.flatnonzero(top > args.min)]

    if not rows:
        print(f"No attack/benign pairs above cosine {args.min}.")
        return
    a, b, sim = map(np.array, zip(*rows))
    A, B = df.iloc[a].reset_index(drop=True), df.iloc[b].reset_index(drop=True)
    out = pd.DataFrame({
        "cosine": np.round(sim, 3),
        "attack_id": A["id"], "attack_source": A["source"], "attack_text": A["text"],
        "benign_id": B["id"], "benign_source": B["source"], "benign_text": B["text"],
        "decision": "", "comment": "",
    }).sort_values("cosine", ascending=False)
    out.to_csv(PROC / "review_pairs.csv", index=False, encoding="utf-8-sig")

    print(f"{len(out)} attacks have a benign text with cosine > {args.min} (closest one shown)\n")
    print("By source (attack source -> benign source):")
    print(out.groupby(["attack_source", "benign_source"]).size().to_string(), "\n")
    for _, r in out.head(15).iterrows():
        print(f"[{r.cosine:.3f}]  ATTACK {r.attack_id}: {r.attack_text[:110]!r}")
        print(f"         BENIGN {r.benign_id}: {r.benign_text[:110]!r}\n")
    print(f"Saved {PROC / 'review_pairs.csv'}. Close it in Excel before re-running.")


if __name__ == "__main__":
    main()
