# Network Intrusion Detection on IoT Traffic (CICIoT2023)

CPS 536 Machine Learning, Fall 2026 course project. Logistic Regression and
XGBoost classifiers that detect attacks in IoT network traffic, evaluated on
benign-vs-attack (binary) and benign + 7 attack families (8-class) tasks.

The milestone report is in [`report/report.md`](report/report.md).

**The whole pipeline is in one notebook:
[`iot_intrusion_detection.ipynb`](iot_intrusion_detection.ipynb)**, saved with
its outputs. It downloads the data, cleans it, trains every model, and shows
the tables and figures. The scripts below do the same thing from the command
line.

## Layout

```
iot_intrusion_detection.ipynb   full pipeline in one notebook (with outputs)
download_data.sh      fetch 8 of the 63 merged CICIoT2023 CSVs (~1.2 GB)
run_all.sh            clean data + run every experiment + build the summary
src/prepare_data.py   cleaning, de-duplication, label mapping, stratified split
src/train.py          train/tune/evaluate one model on one task
src/summarize.py      collect results/*/metrics.json into results/summary.md
results/              metrics, confusion matrices, feature importances
report/               milestone report
```

## Reproduce

```bash
python -m pip install -r requirements.txt jupyter
jupyter notebook iot_intrusion_detection.ipynb
```

or, without Jupyter:

```bash
bash download_data.sh
bash run_all.sh
```

Everything uses `random_state=42`. On a 12-core laptop with 16 GB RAM the
cleaning step takes about 30 s and the full set of experiments about 30-40 min.

## Data

CICIoT2023 (Neto et al., *Sensors* 2023), from the Canadian Institute for
Cybersecurity, University of New Brunswick:
<https://www.unb.ca/cic/datasets/iotdataset-2023.html>. The raw CSVs are not
committed (see `.gitignore`); `download_data.sh` fetches them from a public
Hugging Face mirror of the official release.
