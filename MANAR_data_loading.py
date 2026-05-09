# 1.1
# Mount Google Drive to access the dataset
# ============================================================
from google.colab import drive
drive.mount('/content/drive')

# 1.2 
# Import all libraries needed 
# ============================================================
import re, html, json
import shutil
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# 1.3 
# Load Dataset
# ============================================================

# Path to the raw JSON dataset stored in Google Drive
FILE_PATH = "/content/drive/MyDrive/Cell_Phones_and_Accessories.json"

# Load only 100,000 reviews for efficient training (full dataset is 4.6 GB)
MAX_ROWS  = 100_000

records = []
with open(FILE_PATH, "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        if len(records) >= MAX_ROWS:
            break
        try:
            obj = json.loads(line.strip())
        except:
            continue  # Skip invalid JSON lines

        # Extract rating (overall) and combine review text with title (summary)
        rating = obj.get("overall", 0)
        text   = (obj.get("reviewText","") or "") + " " + (obj.get("summary","") or "")
        text   = text.strip()

        if not text:
            continue          # Skip empty reviews

        # Convert star rating into three sentiment classes
        if rating >= 4:
            label = 2    # Positive
        elif rating == 3:
            label = 1    # Neutral
        else:
            label = 0    # Negative

        records.append({"review_text": text, "label": label})

# Create DataFrame and display class distribution
df = pd.DataFrame(records)
print(f"Loaded: {len(df):,}")
print(df["label"].value_counts().rename({0:"Negative", 1:"Neutral", 2:"Positive"}))
