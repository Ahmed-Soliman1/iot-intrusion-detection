"""Clean CICIoT2023 merged CSVs and build a stratified train/val/test split.

Steps:
  1. Read each raw CSV as float32 (keeps memory low), normalise label strings.
  2. Drop rows with NaN / +-inf values.
  3. Drop exact duplicate rows (features + label), first within each file,
     then across files.
  4. Drop feature vectors that appear with more than one label (ambiguous).
  5. Drop constant (zero-variance) features.
  6. Map the 34 fine-grained labels to 8 classes (benign + 7 attack families)
     and to a binary benign/attack label.
  7. Cap every fine-grained label at CAP rows (random sample) so the very
     large DDoS/DoS classes do not swamp the data; rare classes are kept whole.
  8. Stratified 70/15/15 split on the fine-grained label.

Usage:
  python src/prepare_data.py --raw data/raw --out data/processed
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42

FAMILY = {
    "BENIGN": "Benign",
    "DICTIONARYBRUTEFORCE": "BruteForce",
    "MITM-ARPSPOOFING": "Spoofing",
    "DNS_SPOOFING": "Spoofing",
    "RECON-HOSTDISCOVERY": "Recon",
    "RECON-OSSCAN": "Recon",
    "RECON-PORTSCAN": "Recon",
    "RECON-PINGSWEEP": "Recon",
    "VULNERABILITYSCAN": "Recon",
    "SQLINJECTION": "Web",
    "COMMANDINJECTION": "Web",
    "XSS": "Web",
    "BACKDOOR_MALWARE": "Web",
    "UPLOADING_ATTACK": "Web",
    "BROWSERHIJACKING": "Web",
}


def family(label):
    if label in FAMILY:
        return FAMILY[label]
    if label.startswith("DDOS"):
        return "DDoS"
    if label.startswith("DOS"):
        return "DoS"
    if label.startswith("MIRAI"):
        return "Mirai"
    raise ValueError(f"unknown label {label}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--out", default="data/processed")
    ap.add_argument("--cap", type=int, default=100_000,
                    help="max rows kept per fine-grained label")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    files = sorted(glob.glob(os.path.join(args.raw, "*.csv")))
    stats = {"files": [os.path.basename(f) for f in files],
             "rows_raw": 0, "rows_nan_inf": 0, "rows_dup_within_file": 0}
    parts = []
    for f in files:
        df = pd.read_csv(f, engine="pyarrow")
        stats["rows_raw"] += len(df)
        feats = [c for c in df.columns if c != "Label"]
        df[feats] = df[feats].astype(np.float32)
        df["Label"] = df["Label"].str.strip().str.upper()

        bad = ~np.isfinite(df[feats].to_numpy()).all(axis=1)
        stats["rows_nan_inf"] += int(bad.sum())
        df = df[~bad]

        n = len(df)
        df = df.drop_duplicates()
        stats["rows_dup_within_file"] += n - len(df)
        parts.append(df)
        print(f"{os.path.basename(f)}: kept {len(df):,}")

    df = pd.concat(parts, ignore_index=True)
    del parts
    n = len(df)
    df = df.drop_duplicates(ignore_index=True)
    stats["rows_dup_across_files"] = n - len(df)

    feats = [c for c in df.columns if c != "Label"]
    ambiguous = df.duplicated(subset=feats, keep=False)
    stats["rows_ambiguous_label"] = int(ambiguous.sum())
    stats["ambiguous_label_counts"] = (
        df.loc[ambiguous, "Label"].value_counts().to_dict())
    df = df[~ambiguous].reset_index(drop=True)

    constant = [c for c in feats if df[c].nunique() <= 1]
    stats["constant_features_dropped"] = constant
    df = df.drop(columns=constant)
    feats = [c for c in feats if c not in constant]
    stats["n_features"] = len(feats)
    stats["features"] = feats
    stats["rows_clean"] = len(df)
    stats["clean_label_counts"] = df["Label"].value_counts().to_dict()

    # Per-label cap: keeps every rare class whole, trims the huge flood classes.
    rng = np.random.default_rng(SEED)
    df = df.loc[df.groupby("Label")["Label"].transform(
        lambda s: pd.Series(rng.permutation(len(s)) < args.cap, index=s.index))]
    df = df.reset_index(drop=True)
    df["Family"] = df["Label"].map(family)
    df["Binary"] = (df["Family"] != "Benign").astype(np.int8)
    stats["rows_sampled"] = len(df)
    stats["sampled_family_counts"] = df["Family"].value_counts().to_dict()

    train, rest = train_test_split(df, test_size=0.30, stratify=df["Label"],
                                   random_state=SEED)
    val, test = train_test_split(rest, test_size=0.50, stratify=rest["Label"],
                                 random_state=SEED)
    for name, part in [("train", train), ("val", val), ("test", test)]:
        part.to_parquet(os.path.join(args.out, f"{name}.parquet"), index=False)
        stats[f"rows_{name}"] = len(part)

    with open(os.path.join(args.out, "cleaning_stats.json"), "w") as fh:
        json.dump(stats, fh, indent=2)
    print(json.dumps({k: v for k, v in stats.items()
                      if k not in ("features", "clean_label_counts")}, indent=2))


if __name__ == "__main__":
    main()
