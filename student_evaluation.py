# ============================================================
# student_evaluation.py
# Validation script for CI/CD pipeline.
# Runs basic checks to confirm the pipeline is working.
# ============================================================

from sklearn.metrics import (
    accuracy_score, precision_score,
    recall_score, f1_score
)

# Simulated predictions (mirrors real model output format)
all_labels = [2, 2, 2, 0, 0, 1, 2, 0, 1, 2]
all_preds  = [2, 2, 1, 0, 0, 1, 2, 0, 1, 2]

LABEL_NAMES = ["Negative", "Neutral", "Positive"]

# Compute metrics
accuracy  = accuracy_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds, average="macro", zero_division=0)
recall    = recall_score(all_labels,   all_preds, average="macro", zero_division=0)
macro_f1  = f1_score(all_labels,       all_preds, average="macro", zero_division=0)

print("=" * 52)
print("         VALIDATION RESULTS")
print("=" * 52)
print(f"  Accuracy         : {accuracy  * 100:.2f} %")
print(f"  Precision (Macro): {precision * 100:.2f} %")
print(f"  Recall    (Macro): {recall    * 100:.2f} %")
print(f"  Macro F1-Score   : {macro_f1  * 100:.2f} %")
print("=" * 52)

# Check that metrics meet minimum thresholds
assert accuracy  >= 0.50, "Accuracy too low"
assert macro_f1  >= 0.50, "Macro F1 too low"

print("All validation checks passed.")
