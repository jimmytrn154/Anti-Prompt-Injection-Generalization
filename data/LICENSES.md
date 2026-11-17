# Data sources and licences

Raw data is downloaded by `python -m src.data.download` into `data/raw/` and is **never committed**. `data/raw/MANIFEST.json` records the exact dataset revisions and download dates.

| Source | Where | Licence | Proposal ref. | Notes |
|---|---|---|---|---|
| deepset prompt-injections | HF `deepset/prompt-injections` | Apache-2.0 | [11] | |
| Gandalf | HF `Lakera/gandalf_ignore_instructions` | MIT | [12] | |
| jailbreak-classification | HF `jackhhao/jailbreak-classification` | Apache-2.0 | [13] | |
| Tensor Trust raw dump v2 | GitHub `HumanCompatibleAI/tensor-trust-data` | not stated in the repo; cite Toyer et al. (ICLR 2024) | [4] | Report the version (v2) when using it |
| Dolly-15k | HF `databricks/databricks-dolly-15k` | CC BY-SA 3.0 | [14] | Share-alike: derived data shared under the same licence |
| Stanford Alpaca | HF `tatsu-lab/alpaca` | CC BY-NC 4.0 | [15] | Non-commercial use only (fine for a class project) |
| NotInject | HF `leolee99/NotInject` | MIT | [16] | Test only |

**Rules for the team:** use the data for this course project only; cite every source in the report; share the processed dataset privately (Google Drive / private HF dataset), not publicly.
