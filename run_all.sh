#!/usr/bin/env bash
# Reproduce every preliminary result. Run from the repository root.
set -e
python src/prepare_data.py --raw data/raw --out data/processed
python src/train.py --model lr  --task binary
python src/train.py --model xgb --task binary
python src/train.py --model lr  --task family
python src/train.py --model xgb --task family
python src/train.py --model lr  --task family --no-weights
python src/train.py --model xgb --task family --no-weights
python src/train.py --model xgb --task family --drop IAT
python src/summarize.py
