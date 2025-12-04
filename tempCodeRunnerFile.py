import re
import numpy as np
import pandas as pd
import glob

def extract_metrics_from_file(file_path):
    """Read one result file and extract metrics: Accuracy, Precision, Recall, F1, AUC, PRAUC"""
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # استانداردسازی فاصله‌ها
    text = re.sub(r"\s+", " ", content)

    # استخراج macro avg
    macro_match = re.search(r"macro avg\s+([0-9\.]+)\s+([0-9\.]+)\s+([0-9\.]+)", text)
    if macro_match:
        precision = float(macro_match.group(1))
        recall = float(macro_match.group(2))
        fscore = float(macro_match.group(3))
    else:
        precision = recall = fscore = np.nan

    # استخراج accuracy
    acc_match = re.search(r"accuracy\s+([0-9\.]+)", text)
    accuracy = float(acc_match.group(1)) if acc_match else np.nan

    # استخراج AUC و PRAUC
    auc_match = re.search(r"AUC[:=]\s*([0-9\.]+)", text)
    prauc_match = re.search(r"PRAUC[:=]\s*([0-9\.]+)", text)
    auc = float(auc_match.group(1)) if auc_match else np.nan
    prauc = float(prauc_match.group(1)) if prauc_match else np.nan

    print(f"📄 {file_path} -> acc={accuracy}, prec={precision}, rec={recall}, f1={fscore}, auc={auc}, prauc={prauc}")
    return accuracy, precision, recall, fscore, auc, prauc


def summarize_results(eventlog_prefix):
    """Read all folds for one event log"""
    all_files = glob.glob(f"{eventlog_prefix}*.txt")
    if not all_files:
        print("⚠️ هیچ فایل .txt با این پیشوند پیدا نشد.")
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
        "Final": [f"{m:.3f}±{s:.3f}" for m, s in zip(mean, std)]
    })
    return df


# 🔹 مثال اجرا:
eventlog = "receipt"  # نام دیتاستت
df_summary = summarize_results(eventlog)

if df_summary is not None:
    print("\n📊 جدول نهایی:")
    print(df_summary.to_string(index=False))
