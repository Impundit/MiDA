import re
import numpy as np
import pandas as pd
import glob

# -----------------------------------
# Extract metrics from one file
# -----------------------------------
def extract_metrics_from_file(file_path):

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    text = re.sub(r"[\r\n\t]+", " ", content)
    text = re.sub(r"\s+", " ", text)

    def safe_find(pattern):
        m = re.search(pattern, text)
        return float(m.group(1)) if m else np.nan

    # macro or weighted averages
    macro = re.search(r"macro avg\s+([0-9\.]+)\s+([0-9\.]+)\s+([0-9\.]+)", text)
    weighted = re.search(r"weighted avg\s+([0-9\.]+)\s+([0-9\.]+)\s+([0-9\.]+)", text)

    if macro:
        precision, recall, fscore = map(float, macro.groups())
    elif weighted:
        precision, recall, fscore = map(float, weighted.groups())
    else:
        precision = recall = fscore = np.nan

    accuracy = safe_find(r"accuracy\s*([0-9]*\.[0-9]+)")
    auc = safe_find(r"AUC[:=]\s*([0-9]*\.[0-9]+)")
    prauc = safe_find(r"PRAUC[:=]\s*([0-9]*\.[0-9]+)")

    return accuracy, precision, recall, fscore, auc, prauc


# -----------------------------------
# Summarize folds for one dataset
# -----------------------------------
def summarize_results(eventlog_prefix):

    # Find all result text files for this dataset
    all_files = sorted(glob.glob(f"{eventlog_prefix}*.txt"))
    if not all_files:
        return None

    metrics = [extract_metrics_from_file(f) for f in all_files]
    arr = np.array(metrics)

    mean = np.nanmean(arr, axis=0)
    std = np.nanstd(arr, axis=0)

    labels = ["Accuracy", "Precision", "Recall", "Fscore", "AUC", "AUCPR"]

    df = pd.DataFrame({
        "Metric": labels,
        "Mean": mean.round(3),
        "Std": std.round(3),
        "Final": [f"{m:.3f} ± {s:.3f}" for m, s in zip(mean, std)]
    })

    return df


# -----------------------------------
# Run for all datasets
# -----------------------------------
eventlogs = [
    "receipt",
    "bpi12_all_complete",
    "bpi13_incidents",
    "bpi13_problems"
]

for ev in eventlogs:
    df_summary = summarize_results(ev)
    if df_summary is not None:
        print(f"\n===== Final Results: {ev} =====")
        print(df_summary.to_string(index=False))
        print()
