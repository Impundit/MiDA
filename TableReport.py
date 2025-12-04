import re
import numpy as np
import pandas as pd
import glob

def extract_metrics_from_file(file_path):
    """Read one result file and extract metrics robustly: Accuracy, Precision, Recall, F1, AUC, PRAUC"""
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # استانداردسازی فاصله‌ها
    text = re.sub(r"[\r\n\t]+", " ", content)
    text = re.sub(r"\s+", " ", text)

    def safe_find(pattern):
        m = re.search(pattern, text)
        return float(m.group(1)) if m else np.nan

    # تلاش برای پیدا کردن macro avg یا weighted avg
    macro = re.search(r"macro avg\s+([0-9\.]+)\s+([0-9\.]+)\s+([0-9\.]+)", text)
    weighted = re.search(r"weighted avg\s+([0-9\.]+)\s+([0-9\.]+)\s+([0-9\.]+)", text)

    if macro:
        precision, recall, fscore = map(float, macro.groups())
    elif weighted:
        precision, recall, fscore = map(float, weighted.groups())
    else:
        precision = recall = fscore = np.nan

    # استخراج accuracy با فرمت‌های مختلف
    accuracy = safe_find(r"accuracy\s*([0-9]*\.[0-9]+)") or safe_find(r"Accuracy[:=]?\s*([0-9]*\.[0-9]+)")

    # استخراج AUC و PRAUC
    auc = safe_find(r"AUC[:=]\s*([0-9]*\.[0-9]+)")
    prauc = safe_find(r"PRAUC[:=]\s*([0-9]*\.[0-9]+)")

    print(f"📄 {file_path} → acc={accuracy}, prec={precision}, rec={recall}, f1={fscore}, auc={auc}, prauc={prauc}")
    return accuracy, precision, recall, fscore, auc, prauc


def summarize_results(eventlog_prefix):
    """Summarize mean and std across folds"""
    all_files = sorted(glob.glob(f"{eventlog_prefix}*.txt"))
    if not all_files:
        print(f"⚠️ هیچ فایل خروجی با پیشوند '{eventlog_prefix}' یافت نشد.")
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
eventlog = "bpi12_all_complete"  # نام دیتاست
df_summary = summarize_results(eventlog)

if df_summary is not None:
    print("\n📊 جدول نهایی:")
    print(df_summary.to_string(index=False))
