# Environment setup

Python 3.11 or newer. Run everything from the repo root.

## Windows (PowerShell)

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1          # if blocked: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m src.data.check_env --models
```

## macOS / Linux

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m src.data.check_env --models
```

The last line should end with `All checks passed.` The first run downloads the tokenizer and the embedding model (~500 MB) into the Hugging Face cache.

**GPU (for M3 fine-tuning):** install the CUDA build of PyTorch from <https://pytorch.org/get-started/locally/> *before* `pip install -r requirements.txt`.

**VS Code:** `Ctrl+Shift+P` → "Python: Select Interpreter" → pick `.venv`.

## Data (workstream A)

```powershell
python -m src.data.download      # ~10 min first time; saves to data/raw/ (git-ignored)
python -m src.data.inspect_raw   # writes reports/data/raw_inspection.md
python -m src.data.preprocess    # ~10-20 min on CPU; writes data/processed/clean_v0.parquet + labeling pool
python -m src.data.build_dataset --labels data/processed/labels_v1.parquet   # final apig_v1.parquet + dataset card
python -m src.data.build_dataset --fake-labels   # preview of the final format (random labels; never report results)
```

Raw data never goes into Git. Licences are listed in `data/LICENSES.md`.
