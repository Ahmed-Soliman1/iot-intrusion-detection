"""Collect results/*/metrics.json into Markdown tables (results/summary.md)."""
import glob
import json
import os

ORDER = ["lr_binary", "xgb_binary", "lr_family", "xgb_family",
         "lr_family_noweights", "xgb_family_noweights", "xgb_family_drop-IAT"]


def main(results="results"):
    runs = {}
    for f in glob.glob(os.path.join(results, "*", "metrics.json")):
        with open(f) as fh:
            r = json.load(fh)
        runs[r["tag"]] = r
    tags = [t for t in ORDER if t in runs] + sorted(set(runs) - set(ORDER))

    lines = ["## Test-set metrics", "",
             "| Run | Best hp | Accuracy | Balanced acc. | Macro-F1 | Weighted-F1 | ROC-AUC |",
             "|---|---|---|---|---|---|---|"]
    for t in tags:
        m = runs[t]["test"]
        lines.append(f"| {t} | {runs[t]['best_hp']} | {m['accuracy']:.4f} | "
                     f"{m['balanced_accuracy']:.4f} | {m['macro_f1']:.4f} | "
                     f"{m['weighted_f1']:.4f} | {m['roc_auc']:.4f} |")

    fam = [t for t in tags if runs[t]["task"] == "family"]
    if fam:
        classes = [k for k in runs[fam[0]]["per_class"]
                   if k not in ("accuracy", "macro avg", "weighted avg")]
        lines += ["", "## Per-class recall / F1 (8-class task, test set)", "",
                  "| Class | Support | " + " | ".join(fam) + " |",
                  "|---|---|" + "---|" * len(fam)]
        for c in classes:
            sup = int(runs[fam[0]]["per_class"][c]["support"])
            cells = [f"{runs[t]['per_class'][c]['recall']:.3f} / "
                     f"{runs[t]['per_class'][c]['f1-score']:.3f}" for t in fam]
            lines.append(f"| {c} | {sup:,} | " + " | ".join(cells) + " |")

    out = os.path.join(results, "summary.md")
    with open(out, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
