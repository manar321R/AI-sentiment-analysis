# 2.1
# Mount Drive and Install Libraries
# ============================================================
from google.colab import drive
drive.mount('/content/drive')
# 2.2
# Import Training Libraries
# ============================================================
import os
import shutil
import csv


import pandas as pd
import numpy as np
import torch

from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import Dataset

from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments

from transformers.trainer_utils import get_last_checkpoint


from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report

# 2.3
# Load Train and Validation Data
# ============================================================


# Update paths based on files in your environment
train_path = '/content/train_clean.csv'
val_path = '/content/val_clean.csv'

# Check if files exist before loading
if os.path.exists(train_path) and os.path.exists(val_path):
    try:
        # Read files in the standard way, specifying line terminator to avoid text interference
        train_df = pd.read_csv(train_path, lineterminator='\n')
        val_df = pd.read_csv(val_path, lineterminator='\n')

    except Exception as e:
        print(f"⚠️ Error while reading: {e}")

    train_df = train_df.dropna().reset_index(drop=True)
    val_df = val_df.dropna().reset_index(drop=True)

    print(f"✅ Data loaded successfully: {len(train_df)} rows")
else:
    print(f"❌ Error: Files not found in specified paths.")
# 2.4
# Oversample Minority Classes
# ============================================================
from imblearn.over_sampling import RandomOverSampler

print(f"📊 Data size before balancing: {len(train_df)}")

# Separate texts and labels
X = train_df[['clean_text']]
y = train_df['label']

# Balance data (equalizing neutral and negative with positive)
ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(X, y);

# Merge into a new dataframe
train_df = pd.DataFrame(X_resampled, columns=['clean_text'])
train_df['label'] = y_resampled

print(f"✅ Data size after balancing: {len(train_df)}")
print("Number of reviews is now perfectly balanced for each class!")
# 2.5
# Define SentimentDataset Class
# Convert raw text strings into PyTorch tensors that BERT can process
# ============================================================

class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts.values if hasattr(texts, 'values') else texts
        self.labels = labels.values if hasattr(labels, 'values') else labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
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



