# 3.1
# Mount Drive and Install Libraries
# ============================================================
from google.colab import drive
drive.mount('/content/drive')

# Install required libraries (skip if already installed in your env)
!pip install -q transformers torch scikit-learn pandas seaborn matplotlib

# 3.2
# Import Evaluation Libraries
# ============================================================
import os
import shutil

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns

from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LABEL_NAMES = ['Negative', 'Neutral', 'Positive']

print(f"✅ Using device: {DEVICE}")

# 3.3
# Load Test Data
# ============================================================
TEST_CSV = '/content/test_clean.csv'

# Restore from Drive if missing
if not os.path.exists(TEST_CSV):
    shutil.copy('/content/drive/MyDrive/test_clean.csv', TEST_CSV)
    print("♻️  Restored test_clean.csv from Drive")

test_df = pd.read_csv(TEST_CSV, lineterminator='
').dropna().reset_index(drop=True)

print(f"✅ Test set size: {len(test_df):,} samples")
print("
Class distribution:")
print(test_df['label'].value_counts().rename(
    {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
))

# 3.4
# Define Dataset Class and Load Trained Model
# ============================================================

# Load trained model and tokenizer
MODEL_DIR = '/content/drive/MyDrive/sentiment_bert_model_best'
MAX_LEN    = 128    # must match training
BATCH_SIZE = 32

assert os.path.isdir(MODEL_DIR), f"❌ Model not found: {MODEL_DIR}"


# Data preparation class (same class used for training to ensure compatibility)
class SentimentDataset(Dataset):
    def _init_(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts.values if hasattr(texts, 'values') else texts
        self.labels = labels.values if hasattr(labels, 'values') else labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def _len_(self):
        return len(self.texts)

    def _getitem_(self, item):
        text = str(self.texts[item])
        label = self.labels[item]

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


# Load tokenizer and model
tokenizer = BertTokenizer.from_pretrained(MODEL_DIR)
model     = BertForSequenceClassification.from_pretrained(MODEL_DIR, num_labels=NUM_LABELS)
model     = model.to(DEVICE)
model.eval()   # ← disable dropout for inference

print("✅ Model loaded and set to evaluation mode.")

# 3.5
# Run Inference on Test Set
# ============================================================

test_dataset = SentimentDataset(
    texts     = test_df['clean_text'],
    labels    = test_df['label'],
    tokenizer = tokenizer,
    max_len   = MAX_LEN
)

test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

all_preds  = []
all_labels = []

print(f"Running inference on {len(test_dataset):,} samples …")

with torch.no_grad():
    for batch_idx, batch in enumerate(test_loader):
        input_ids      = batch['input_ids'].to(DEVICE)
        attention_mask = batch['attention_mask'].to(DEVICE)
        labels         = batch['labels']

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        preds   = torch.argmax(outputs.logits, dim=1).cpu().numpy()

        all_preds.extend(preds)
        all_labels.extend(labels.numpy())

        if (batch_idx + 1) % 50 == 0:
            done = (batch_idx + 1) * BATCH_SIZE
            print(f"  … {min(done, len(test_dataset)):,} / {len(test_dataset):,} done")

all_preds  = np.array(all_preds)
all_labels = np.array(all_labels)

print("✅ Inference complete.")
print("Inference file ready")
