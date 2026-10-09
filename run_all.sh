#!/usr/bin/env bash
# Reproduces all validation, model, robustness and test-retest results reported in the article and supplement.
set -e
cd "$(dirname "$0")"
for s in 00_prepare 01_human_reliability 02_llm_validation 03_validation_supplement 04_hypothesis_models 05_robustness 06_tables 07_test_retest; do
  echo "== $s"; python3 "$s.py"
done
