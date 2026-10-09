# Fairness and service expectations in airline online reviews — data and code

Replication materials for the article *"Fairness and service expectations in airline online reviews: A validated large-language-model analysis of a full-service and a low-cost carrier"* (submitted to the *Journal of Air Transport Management*). This repository is anonymized for double-blind peer review.

## Contents

All files are in the main folder.

| File | Description |
|---|---|
| `analysis_dataset.csv` | 4,225 reviews: carrier, source, review date/period, star rating (1–5; blank for Facebook), language, length (words, characters), text sentiment, zero-shot labels (`zs_*`) |
| `validation_sample.csv` | the 250 reviews in the human validation sample (Code, dataset index, source, zero-shot labels) |
| `human_coding_validation.csv` | independent codes of Coder A and Coder B for the 250 reviews (codes only) |
| `classify_llm.py` | codebook-guided LLM classifier (full prompt and codebook inside the script) |
| `llm_labels_all.csv` | LLM labels for all 4,225 reviews, with the raw JSON output of the model |
| `llm_labels_validation.csv` | LLM labels for the 250 validation reviews (separate run) |
| `run_info_all.json`, `run_info_validation.json` | run metadata: model, revision hash, quantization, decoding, prompt hash, software, GPU |
| `00_prepare.py` | builds the working dataset |
| `01_human_reliability.py` | inter-coder agreement (κ, weighted κ, AC1, bootstrap CIs) |
| `02_llm_validation.py` | LLM and zero-shot labels vs. each coder and the agreed subset; pre-specified criterion |
| `03_validation_supplement.py` | supplementary validation tables |
| `04_hypothesis_models.py` | proportional-odds ordinal models (Models 1–6), LR and Wald tests, Holm adjustment |
| `05_robustness.py` | Brant test, excerpt-excluded and human-coded robustness checks, descriptives |
| `06_tables.py` | article and supplementary tables |
| `07_test_retest.py` | agreement between the validation run and the full-corpus run of the LLM |
| `ologit.py`, `hyp_helpers.py` | ordinal-logit estimator and helpers |
| `run_all.sh` | runs the full pipeline |
| `requirements.txt` | Python package versions |

## Reproducing the results

```
pip install -r requirements.txt
bash run_all.sh
```

Results are written to the main folder as `.pkl` and `.csv` files; all figures reported in the article and supplement are produced deterministically (bootstrap seed 20261008).

## LLM classification

`llm/classify_llm.py` runs `Qwen/Qwen2.5-7B-Instruct` (revision `a09a35458c702b33eeacc393d103063234e8bc28`) with 4-bit NF4 quantization and greedy decoding (prompt SHA-256 prefix `0669816d907e`). It expects a CSV with columns `index` and `text`. The labels used in the article are provided in `llm_labels_all.csv` and `llm_labels_validation.csv`, so the statistical analysis can be reproduced without re-running the model.

## Data and privacy

Only publicly available reviews were analyzed. Review texts, reviewer names, profile links and review URLs are **not** included, to protect reviewers' privacy and in line with the platforms' terms of use. Review texts can be made available to the editor and reviewers for verification on reasonable request.
