"""Train and evaluate one model on the processed CICIoT2023 split.

Models
  lr  : Logistic Regression on signed-log1p + standardised features.
        C is tuned on the validation set (macro-F1).
  xgb : XGBoost (hist trees). max_depth is tuned on the validation set,
        number of trees chosen by early stopping on validation log-loss.

Tasks
  binary : benign vs attack
  family : benign + 7 attack families (8 classes)

Examples
  python src/train.py --model lr  --task family
  python src/train.py --model xgb --task family --no-weights
  python src/train.py --model xgb --task family --drop IAT
"""
import argparse
import json
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             classification_report, confusion_matrix,
                             f1_score, roc_auc_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

SEED = 42
FAMILIES = ["Benign", "DDoS", "DoS", "Mirai", "Recon", "Spoofing", "Web",
            "BruteForce"]


def signed_log1p(x):
    return np.sign(x) * np.log1p(np.abs(x))


def load(data_dir, task, drop):
    parts = {s: pd.read_parquet(os.path.join(data_dir, f"{s}.parquet"))
             for s in ("train", "val", "test")}
    feats = [c for c in parts["train"].columns
             if c not in ("Label", "Family", "Binary") and c not in drop]
    if task == "binary":
        names = ["Benign", "Attack"]
        enc = lambda df: df["Binary"].to_numpy()
    else:
        names = FAMILIES
        idx = {n: i for i, n in enumerate(FAMILIES)}
        enc = lambda df: df["Family"].map(idx).to_numpy()
    out = {s: (df[feats].to_numpy(np.float32), enc(df)) for s, df in parts.items()}
    return out, feats, names


def make_lr(C, weighted):
    return make_pipeline(
        FunctionTransformer(signed_log1p),
        StandardScaler(),
        LogisticRegression(C=C, max_iter=2000, tol=1e-4,
                           class_weight="balanced" if weighted else None))


def make_xgb(depth, n_classes):
    params = dict(n_estimators=1500, learning_rate=0.1, max_depth=depth,
                  subsample=0.8, colsample_bytree=0.8, tree_method="hist",
                  early_stopping_rounds=30, n_jobs=-1, random_state=SEED)
    if n_classes == 2:
        params.update(objective="binary:logistic", eval_metric="logloss")
    else:
        params.update(objective="multi:softprob", eval_metric="mlogloss")
    return XGBClassifier(**params)


def fit(model_name, hp, data, n_classes, weighted):
    Xtr, ytr = data["train"]
    Xva, yva = data["val"]
    sw = compute_sample_weight("balanced", ytr) if weighted else None
    if model_name == "lr":
        m = make_lr(hp, weighted)
        m.fit(Xtr, ytr)
    else:
        m = make_xgb(hp, n_classes)
        sw_va = compute_sample_weight("balanced", yva) if weighted else None
        m.fit(Xtr, ytr, sample_weight=sw, eval_set=[(Xva, yva)],
              sample_weight_eval_set=[sw_va] if weighted else None,
              verbose=False)
    return m


def evaluate(m, X, y, names):
    proba = m.predict_proba(X)
    pred = proba.argmax(axis=1)
    if len(names) == 2:
        auc = roc_auc_score(y, proba[:, 1])
    else:
        auc = roc_auc_score(y, proba, multi_class="ovr", average="macro")
    return pred, {
        "accuracy": accuracy_score(y, pred),
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "macro_f1": f1_score(y, pred, average="macro"),
        "weighted_f1": f1_score(y, pred, average="weighted"),
        "roc_auc": auc,
    }


def plot_confusion(y, pred, names, title, path):
    cm = confusion_matrix(y, pred, labels=range(len(names)))
    cmn = cm / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(1.0 + 0.8 * len(names), 0.6 + 0.7 * len(names)))
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(names)), names, rotation=45, ha="right")
    ax.set_yticks(range(len(names)), names)
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, f"{cmn[i, j]:.2f}", ha="center", va="center",
                    fontsize=8, color="white" if cmn[i, j] > 0.5 else "black")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True (row-normalised = recall)")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return cm.tolist()


def plot_importance(m, feats, path, top=15):
    imp = m.get_booster().get_score(importance_type="gain")
    s = pd.Series({feats[int(k[1:])]: v for k, v in imp.items()})
    s = (s / s.sum()).sort_values().tail(top)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.barh(s.index, s.values, color="#3b6ea5")
    ax.set_xlabel("Share of total gain")
    ax.set_title("XGBoost feature importance (gain)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return s.sort_values(ascending=False).to_dict()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["lr", "xgb"], required=True)
    ap.add_argument("--task", choices=["binary", "family"], required=True)
    ap.add_argument("--no-weights", action="store_true")
    ap.add_argument("--drop", nargs="*", default=[], help="features to drop")
    ap.add_argument("--data", default="data/processed")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    weighted = not args.no_weights
    tag = f"{args.model}_{args.task}" + ("" if weighted else "_noweights")
    if args.drop:
        tag += "_drop-" + "-".join(args.drop)
    out = os.path.join(args.out, tag)
    os.makedirs(out, exist_ok=True)

    data, feats, names = load(args.data, args.task, args.drop)
    grid = [0.01, 0.1, 1.0, 10.0] if args.model == "lr" else [6, 10]

    tuning, best = [], None
    for hp in grid:
        t0 = time.time()
        m = fit(args.model, hp, data, len(names), weighted)
        _, val = evaluate(m, *data["val"], names)
        row = {"hp": hp, "val_macro_f1": val["macro_f1"],
               "fit_seconds": round(time.time() - t0, 1)}
        if args.model == "xgb":
            row["best_iteration"] = int(m.best_iteration)
        tuning.append(row)
        print(tag, row, flush=True)
        if best is None or val["macro_f1"] > best[1]:
            best = (m, val["macro_f1"], hp)
    m, _, best_hp = best

    pred, test = evaluate(m, *data["test"], names)
    yte = data["test"][1]
    report = classification_report(yte, pred, labels=range(len(names)),
                                   target_names=names, digits=4,
                                   output_dict=True, zero_division=0)
    cm = plot_confusion(yte, pred, names, tag, os.path.join(out, "confusion.png"))
    result = {"tag": tag, "model": args.model, "task": args.task,
              "weighted": weighted, "dropped": args.drop,
              "n_features": len(feats), "best_hp": best_hp,
              "tuning": tuning, "test": test, "per_class": report,
              "confusion_matrix": cm}
    if args.model == "xgb":
        result["importance"] = plot_importance(m, feats,
                                               os.path.join(out, "importance.png"))
    with open(os.path.join(out, "metrics.json"), "w") as fh:
        json.dump(result, fh, indent=2)
    print(tag, "TEST", {k: round(v, 4) for k, v in test.items()}, flush=True)


if __name__ == "__main__":
    main()
