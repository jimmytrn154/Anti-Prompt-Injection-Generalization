"""Download every APIG source, untouched, into data/raw/.  Run from the repo root:

    python -m src.data.download                      # everything (skips what is already there)
    python -m src.data.download --only deepset gandalf
    python -m src.data.download --force              # re-download

Hugging Face sources -> data/raw/<source>/<split>.parquet
Tensor Trust         -> data/raw/tensortrust/raw_dump_attacks_v2.jsonl.bz2  (~100 MB)
A manifest with row counts, dataset revisions and timestamps -> data/raw/MANIFEST.json

data/raw/ is git-ignored: never commit it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from datetime import datetime, timezone

from tqdm import tqdm

from src.data.sources import RAW_DIR, SOURCES, TENSORTRUST_FILE, TENSORTRUST_URL

MANIFEST = RAW_DIR / "MANIFEST.json"


def _load_manifest() -> dict:
    if MANIFEST.exists():
        return json.loads(MANIFEST.read_text(encoding="utf-8"))
    return {}


def _save_manifest(m: dict) -> None:
    MANIFEST.write_text(json.dumps(m, indent=2, ensure_ascii=False), encoding="utf-8")


def _hf_revision(repo_id: str) -> str | None:
    """Commit hash of the dataset on the Hub, so we can report exactly which version we used."""
    try:
        from huggingface_hub import HfApi
        return HfApi().dataset_info(repo_id).sha
    except Exception:  # noqa: BLE001 - revision is nice to have, not required
        return None


def download_hf(name: str, force: bool) -> dict:
    from datasets import load_dataset

    src = SOURCES[name]
    out = RAW_DIR / name
    if out.exists() and any(out.glob("*.parquet")) and not force:
        print(f"  skip {name}: already in {out.relative_to(RAW_DIR.parents[1])} (use --force to refresh)")
        return {}
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("*.parquet"):
        old.unlink()

    revision = _hf_revision(src.origin)
    ds = load_dataset(src.origin, revision=revision) if revision else load_dataset(src.origin)
    rows = {}
    for split, d in ds.items():
        d.to_parquet(str(out / f"{split}.parquet"))
        rows[split] = len(d)
    print(f"  done {name}: {rows}")
    return {"origin": src.origin, "revision": revision, "rows": rows}


def download_tensortrust(force: bool) -> dict:
    out = RAW_DIR / "tensortrust" / TENSORTRUST_FILE
    if out.exists() and not force:
        print(f"  skip tensortrust: already in {out.relative_to(RAW_DIR.parents[1])} (use --force to refresh)")
        return {}
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".part")
    sha = hashlib.sha256()
    req = urllib.request.Request(TENSORTRUST_URL, headers={"User-Agent": "apig-data-download/1.0"})
    with urllib.request.urlopen(req) as resp, open(tmp, "wb") as f, tqdm(
        total=int(resp.headers.get("Content-Length") or 0), unit="B", unit_scale=True, desc="  tensortrust"
    ) as bar:
        while chunk := resp.read(1 << 20):
            f.write(chunk)
            sha.update(chunk)
            bar.update(len(chunk))
    tmp.replace(out)
    print(f"  done tensortrust: {out.stat().st_size / 1e6:.1f} MB")
    return {"origin": TENSORTRUST_URL, "version": "raw dump v2 (2024-02-16)", "sha256": sha.hexdigest()}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--only", nargs="+", choices=list(SOURCES), help="download only these sources")
    p.add_argument("--force", action="store_true", help="re-download even if files exist")
    args = p.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    manifest = _load_manifest()
    failed = []
    for name in args.only or list(SOURCES):
        print(f"[{name}] {SOURCES[name].origin}")
        try:
            info = download_tensortrust(args.force) if SOURCES[name].kind == "url" else download_hf(name, args.force)
        except Exception as e:  # noqa: BLE001 - keep going, report at the end
            failed.append(name)
            print(f"  FAILED {name}: {type(e).__name__}: {e}")
            continue
        if info:
            info.update(licence=SOURCES[name].licence,
                        downloaded_at=datetime.now(timezone.utc).isoformat(timespec="seconds"))
            manifest[name] = info
            _save_manifest(manifest)

    if failed:
        print(f"\nFailed: {failed}. Re-run with: python -m src.data.download --only {' '.join(failed)}")
        sys.exit(1)
    print(f"\nAll sources are in {RAW_DIR}. Next: python -m src.data.inspect_raw")


if __name__ == "__main__":
    main()
