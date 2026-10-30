#!/usr/bin/env bash
# Download the first 8 of the 63 merged CICIoT2023 CSV files (~1.2 GB) from a
# public Hugging Face mirror of the official UNB release.
# Official source: https://www.unb.ca/cic/datasets/iotdataset-2023.html
set -e
mkdir -p data/raw
BASE="https://huggingface.co/datasets/bencorn/CIC-IoT-2023/resolve/main/CSV/MERGED_CSV"
for i in 01 02 03 04 05 06 07 08; do
  curl -L --retry 5 --retry-all-errors -o "data/raw/Merged$i.csv" "$BASE/Merged$i.csv"
done
