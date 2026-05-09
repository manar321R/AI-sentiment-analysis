# 3.6
# Compute and Print Evaluation Metrics
# ============================================================
accuracy  = accuracy_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
recall    = recall_score(all_labels,   all_preds, average='macro', zero_division=0)
macro_f1  = f1_score(all_labels,       all_preds, average='macro', zero_division=0)

print("=" * 52)
print("         MODEL EVALUATION RESULTS")
print("=" * 52)
print(f"  Accuracy         : {accuracy  * 100:.2f} %")
print(f"  Precision (Macro): {precision * 100:.2f} %")
print(f"  Recall    (Macro): {recall    * 100:.2f} %")
print(f"  Macro F1-Score   : {macro_f1  * 100:.2f} %")
print("=" * 52)

print("
Detailed Classification Report:")
print("-" * 52)
print(classification_report(
    all_labels, all_preds,
    target_names  = LABEL_NAMES,
    zero_division = 0
))

# 3.7
# Plot Confusion Matrix
# ============================================================

cm = confusion_matrix(all_labels, all_preds)

fig, ax = plt.subplots(figsize=(7, 5))
sns.heatmap(
    cm,
    annot       = True,
    fmt         = 'd',
    cmap        = 'Blues',
    xticklabels = LABEL_NAMES,
    yticklabels = LABEL_NAMES,
    ax          = ax
)
ax.set_title('Confusion Matrix — Test Set', fontsize=14, fontweight='bold', pad=12)
ax.set_xlabel('Predicted Label', fontsize=12)
ax.set_ylabel('True Label',      fontsize=12)

plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150)
plt.show()
print("✅ Saved: confusion_matrix.png")

# ============================================================
# Cell 3.8
# Plot Per-Class Metrics Bar Chart
# ============================================================

per_class_precision = precision_score(all_labels, all_preds, average=None, zero_division=0)
per_class_recall    = recall_score(all_labels,    all_preds, average=None, zero_division=0)
per_class_f1        = f1_score(all_labels,        all_preds, average=None, zero_division=0)

x     = np.arange(len(LABEL_NAMES))
width = 0.25

fig, ax = plt.subplots(figsize=(8, 5))
bars1 = ax.bar(x - width, per_class_precision, width, label='Precision', color='steelblue')
bars2 = ax.bar(x,         per_class_recall,    width, label='Recall',    color='seagreen')
bars3 = ax.bar(x + width, per_class_f1,        width, label='F1-Score',  color='darkorange')

for bars in [bars1, bars2, bars3]:
    for bar in bars:
        ax.annotate(
            f"{bar.get_height():.2f}",
            xy       = (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext   = (0, 3),
            textcoords = "offset points",
            ha='center', va='bottom', fontsize=8
        )

ax.set_title('Per-class Metrics — Test Set', fontsize=13, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(LABEL_NAMES, fontsize=11)
ax.set_ylim(0, 1.1)
ax.set_ylabel('Score')
ax.legend()
ax.grid(axis='y', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig('per_class_metrics.png', dpi=150)
plt.show()
print("✅ Saved: per_class_metrics.png")

# 3.9
# Save All Results to Google Drive
# ============================================================

import pandas as pd

# Save metrics as CSV
metrics_df = pd.DataFrame({
    'Metric': ['Accuracy', 'Precision (Macro)', 'Recall (Macro)', 'Macro F1-Score'],
    'Score' : [accuracy,    precision,            recall,           macro_f1]
})
metrics_df.to_csv('evaluation_metrics.csv', index=False)

# Copy everything to Drive
shutil.copy('evaluation_metrics.csv', '/content/drive/MyDrive/evaluation_metrics.csv')
shutil.copy('confusion_matrix.png',   '/content/drive/MyDrive/confusion_matrix.png')
shutil.copy('per_class_metrics.png',  '/content/drive/MyDrive/per_class_metrics.png')

print(metrics_df.to_string(index=False))
print("
✅ All results saved to Google Drive")

