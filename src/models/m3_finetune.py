"""M3: fine-tuned DistilBERT (or another Hugging Face encoder) for every split x seed.

    python -m src.models.m3_finetune --data data/processed/apig_v1.parquet           # all 11 splits x 3 seeds
    python -m src.models.m3_finetune --splits split_random loso --seeds 0            # a subset ("loso" = loso_S1..S3)
    python -m src.models.m3_finetune --data <FAKE file> --smoke                      # 1-minute code test

Runs on CUDA (Kaggle/Colab, fp16), Apple MPS or CPU. Finished runs are skipped, so an interrupted job
can simply be restarted. Training: AdamW, linear warmup/decay, early stopping on validation loss
(validation rows come only from training strategies/sources). The best epoch's weights are scored on
test + calibration + NotInject; see common.py for the threshold and metrics.
"""
from __future__ import annotations

import argparse
import copy
import math
import random
import time

import numpy as np
import pandas as pd
import torch
from torch.nn.functional import cross_entropy
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from src.models.common import Results, eval_rows, load_data, parse_splits, split_rows

DEFAULT_MODEL = "distilbert-base-uncased"   # = TOKENIZER_NAME used by A for the 5-500 token filter


def pick_device(name: str | None) -> torch.device:
    if name:
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ------------------------------------------------------------------ batching
def length_batches(lengths: np.ndarray, batch_size: int, rng: np.random.Generator | None) -> list[np.ndarray]:
    """Group texts of similar length so short prompts are not padded to 500 tokens.
    Training (rng given): shuffle, sort inside chunks of 50 batches, shuffle batch order.
    Inference (rng None): sort everything by length."""
    n = len(lengths)
    if rng is None:
        order = np.argsort(lengths, kind="stable")
        return [order[i:i + batch_size] for i in range(0, n, batch_size)]
    idx = rng.permutation(n)
    chunk = batch_size * 50
    batches = []
    for i in range(0, n, chunk):
        part = idx[i:i + chunk]
        part = part[np.argsort(lengths[part], kind="stable")]
        batches += [part[j:j + batch_size] for j in range(0, len(part), batch_size)]
    rng.shuffle(batches)
    return batches


def collate(ids: list[list[int]], pad_id: int, device: torch.device) -> dict[str, torch.Tensor]:
    width = max(len(x) for x in ids)
    input_ids = torch.full((len(ids), width), pad_id, dtype=torch.long)
    mask = torch.zeros((len(ids), width), dtype=torch.long)
    for i, x in enumerate(ids):
        input_ids[i, :len(x)] = torch.tensor(x)
        mask[i, :len(x)] = 1
    return {"input_ids": input_ids.to(device), "attention_mask": mask.to(device)}


class Encoded:
    """Token ids for every row of the table, computed once and shared by all splits/seeds."""

    def __init__(self, df, tokenizer, max_len: int):
        enc = tokenizer(df["text"].tolist(), truncation=True, max_length=max_len)
        self.ids = enc["input_ids"]
        self.lengths = np.array([len(x) for x in self.ids])
        self.pad_id = tokenizer.pad_token_id
        n_trunc = int((self.lengths >= max_len).sum())
        if n_trunc:
            print(f"  note: {n_trunc} rows reach {max_len} tokens and may be truncated")

    def take(self, index: np.ndarray) -> tuple[list[list[int]], np.ndarray]:
        return [self.ids[i] for i in index], self.lengths[index]


@torch.no_grad()
def predict(model, enc: Encoded, index: np.ndarray, device, batch_size: int, amp: bool) -> np.ndarray:
    """P(attack) for the given table rows."""
    model.eval()
    ids, lengths = enc.take(index)
    out = np.empty(len(ids), dtype=np.float32)
    for b in length_batches(lengths, batch_size, None):
        batch = collate([ids[i] for i in b], enc.pad_id, device)
        with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
            logits = model(**batch).logits
        out[b] = torch.softmax(logits.float(), dim=-1)[:, 1].cpu().numpy()
    return out


@torch.no_grad()
def val_loss(model, enc: Encoded, index: np.ndarray, labels: np.ndarray, device, batch_size: int, amp: bool) -> float:
    model.eval()
    ids, lengths = enc.take(index)
    total = 0.0
    for b in length_batches(lengths, batch_size, None):
        batch = collate([ids[i] for i in b], enc.pad_id, device)
        y = torch.tensor(labels[b], device=device)
        with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
            logits = model(**batch).logits
        total += cross_entropy(logits.float(), y, reduction="sum").item()
    return total / len(index)


# ------------------------------------------------------------------ one run
def train_one(args, df, enc: Encoded, split: str, seed: int, device, res: Results) -> None:
    set_seed(seed)
    parts = split_rows(df, split)
    tr, va = parts["train"], parts["val"]
    if args.smoke:
        tr = pd.concat([g.sample(min(len(g), 100), random_state=seed) for _, g in tr.groupby("label")])
    tr_idx, va_idx = tr.index.to_numpy(), va.index.to_numpy()
    y_tr, y_va = tr["label"].to_numpy(), va["label"].to_numpy()
    print(f"\n[M3 | {split} | seed {seed}] train {len(tr_idx):,}  val {len(va_idx):,}  "
          f"test {len(parts['test']):,}  on {device}")

    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=2).to(device)
    amp = device.type == "cuda" and not args.no_fp16
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = math.ceil(len(tr_idx) / args.batch_size)
    total_steps = steps_per_epoch * args.epochs
    sched = get_linear_schedule_with_warmup(opt, int(args.warmup * total_steps), total_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    rng = np.random.default_rng(seed)
    tr_ids, tr_len = enc.take(tr_idx)

    best, best_loss, best_epoch, bad, history = None, float("inf"), 0, 0, []
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for b in length_batches(tr_len, args.batch_size, rng):
            batch = collate([tr_ids[i] for i in b], enc.pad_id, device)
            y = torch.tensor(y_tr[b], device=device)
            with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
                loss = cross_entropy(model(**batch).logits.float(), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            running += loss.item() * len(b)
        vl = val_loss(model, enc, va_idx, y_va, device, args.eval_batch_size, amp)
        history.append({"epoch": epoch, "train_loss": running / len(tr_idx), "val_loss": vl})
        print(f"  epoch {epoch}: train loss {running / len(tr_idx):.4f}  val loss {vl:.4f}  "
              f"({time.time() - t0:.0f}s)")
        if vl < best_loss - 1e-4:
            best_loss, best_epoch, bad = vl, epoch, 0
            best = {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= args.patience:
                print(f"  early stop (best epoch {best_epoch})")
                break
    train_time = time.time() - t0
    model.load_state_dict(best)

    preds = eval_rows(parts)
    rows = preds["id"].map(dict(zip(df["id"], df.index))).to_numpy()
    preds["score"] = predict(model, enc, rows, device, args.eval_batch_size, amp)
    extra = {"n_train": len(tr_idx), "best_epoch": best_epoch, "epochs_run": len(history),
             "best_val_loss": best_loss, "train_time_s": round(train_time, 1),
             "latency_ms": latency_ms(model, enc, rows[:100], device),
             "params_m": round(sum(p.numel() for p in model.parameters()) / 1e6, 1),
             "model_name": args.model_name, "device": device.type}
    res.save(split, seed, preds, extra)
    if args.save_model and split == "split_random" and seed == args.seeds[0]:
        path = res.root / "models" / "m3"
        model.save_pretrained(path)
        AutoTokenizer.from_pretrained(args.model_name).save_pretrained(path)
        print(f"  saved model to {path} (for Integrated Gradients / demo)")
    del model, opt, best
    if device.type == "cuda":
        torch.cuda.empty_cache()


@torch.no_grad()
def latency_ms(model, enc: Encoded, rows: np.ndarray, device) -> float:
    """Mean ms per prompt at batch size 1 (a guardrail scores one input at a time)."""
    model.eval()
    ids, _ = enc.take(rows)
    for x in ids[:5]:                                   # warm-up
        model(**collate([x], enc.pad_id, device))
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for x in ids:
        model(**collate([x], enc.pad_id, device))
    if device.type == "cuda":
        torch.cuda.synchronize()
    return round(1000 * (time.perf_counter() - t0) / max(len(ids), 1), 2)


# ------------------------------------------------------------------ main
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", help="apig_v1.parquet (default: data/processed/apig_v1.parquet)")
    p.add_argument("--out", help="results folder (default: results/)")
    p.add_argument("--splits", nargs="+", default=["all"], help="split columns or prefixes (default: all 11)")
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--model-name", default=DEFAULT_MODEL)
    p.add_argument("--max-len", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--eval-batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup", type=float, default=0.1, help="fraction of steps for linear warmup")
    p.add_argument("--epochs", type=int, default=5, help="maximum epochs")
    p.add_argument("--patience", type=int, default=2, help="early stopping patience (epochs)")
    p.add_argument("--device", help="cuda / mps / cpu (default: auto)")
    p.add_argument("--no-fp16", action="store_true", help="disable mixed precision on CUDA")
    p.add_argument("--save-model", action="store_true", help="save split_random weights (for IG / demo)")
    p.add_argument("--force", action="store_true", help="re-run finished split/seed pairs")
    p.add_argument("--smoke", action="store_true", help="code test: 200 train rows, 1 epoch, split_random, seed 0")
    args = p.parse_args()
    if args.smoke:
        args.splits, args.seeds, args.epochs = ["split_random"], [0], 1

    df, data_path = load_data(args.data)
    splits = parse_splits(args.splits)
    device = pick_device(args.device)
    model_tag = "m3" if args.model_name == DEFAULT_MODEL else "m3_" + args.model_name.split("/")[-1]
    res = Results(args.out, data_path, model_tag + ("_smoke" if args.smoke else ""))
    res.save_config(vars(args))

    print(f"Tokenizing with {args.model_name} ...")
    enc = Encoded(df, AutoTokenizer.from_pretrained(args.model_name), args.max_len)

    todo = [(s, k) for s in splits for k in args.seeds if args.force or not res.done(s, k)]
    print(f"{len(todo)} run(s) to do, {len(splits) * len(args.seeds) - len(todo)} already finished")
    t0 = time.time()
    for i, (split, seed) in enumerate(todo, 1):
        train_one(args, df, enc, split, seed, device, res)
        print(f"  progress {i}/{len(todo)}, elapsed {(time.time() - t0) / 60:.1f} min")
    print(f"\nDone. Metrics: {res.metrics_file}")


if __name__ == "__main__":
    main()
