# ============================================================
# tests/test_pipeline.py
# Basic unit tests for the Sentiment Analysis pipeline.
# Run with: pytest tests/ -v
# ============================================================

import pytest
import re


# ─────────────────────────────────────────────────────────────
# 1. Tests — Preprocessing Module
# ─────────────────────────────────────────────────────────────

def preprocess_text(text: str) -> str:
    """
    Mirrors the cleaning logic used in the main pipeline:
    lowercase, strip HTML tags, remove extra whitespace.
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"<[^>]+>", "", text)       # remove HTML tags
    text = re.sub(r"[^\w\s]", "", text)        # remove special characters
    text = re.sub(r"\s+", " ", text).strip()   # normalize whitespace
    return text


def test_preprocess_lowercase():
    assert preprocess_text("GREAT Product!") == "great product"

def test_preprocess_removes_html():
    assert "<br>" not in preprocess_text("Good<br>product")

def test_preprocess_empty_string():
    assert preprocess_text("") == ""

def test_preprocess_none_input():
    assert preprocess_text(None) == ""

def test_preprocess_extra_spaces():
    assert preprocess_text("  too   many   spaces  ") == "too many spaces"


# ─────────────────────────────────────────────────────────────
# 2. Tests — Label Mapping
# ─────────────────────────────────────────────────────────────

def map_label(rating: int) -> str:
    """
    Maps star ratings to sentiment labels.
    Consistent with the notebook label assignment.
    """
    if rating in [4, 5]:
        return "Positive"
    elif rating == 3:
        return "Neutral"
    elif rating in [1, 2]:
        return "Negative"
    else:
        raise ValueError(f"Invalid rating: {rating}")


def test_label_positive_5():
    assert map_label(5) == "Positive"

def test_label_positive_4():
    assert map_label(4) == "Positive"

def test_label_neutral():
    assert map_label(3) == "Neutral"

def test_label_negative_2():
    assert map_label(2) == "Negative"

def test_label_negative_1():
    assert map_label(1) == "Negative"

def test_label_invalid_raises():
    with pytest.raises(ValueError):
        map_label(6)


# ─────────────────────────────────────────────────────────────
# 3. Tests — Prediction Output Format
# ─────────────────────────────────────────────────────────────

VALID_LABELS = {"Positive", "Neutral", "Negative"}

def mock_predict(text: str) -> str:
    """
    Simulates a model prediction for testing purposes.
    In the real pipeline, this calls the BERT model.
    """
    text = preprocess_text(text)
    if not text:
        raise ValueError("Input text is empty after preprocessing.")
    if any(word in text for word in ["great", "excellent", "love"]):
        return "Positive"
    elif any(word in text for word in ["terrible", "bad", "awful"]):
        return "Negative"
    return "Neutral"


def test_prediction_returns_valid_label():
    assert mock_predict("This is a great product!") in VALID_LABELS

def test_prediction_positive():
    assert mock_predict("I love this item, excellent quality!") == "Positive"

def test_prediction_negative():
    assert mock_predict("Terrible product, very bad quality.") == "Negative"

def test_prediction_empty_raises():
    with pytest.raises(ValueError):
        mock_predict("")

def test_prediction_whitespace_raises():
    with pytest.raises(ValueError):
        mock_predict("   ")


# ─────────────────────────────────────────────────────────────
# 4. Tests — Evaluation Metrics
# ─────────────────────────────────────────────────────────────

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

def compute_metrics(y_true: list, y_pred: list) -> dict:
    return {
        "accuracy":  round(accuracy_score(y_true, y_pred) * 100, 2),
        "macro_f1":  round(f1_score(y_true, y_pred, average="macro", zero_division=0) * 100, 2),
        "precision": round(precision_score(y_true, y_pred, average="macro", zero_division=0) * 100, 2),
        "recall":    round(recall_score(y_true, y_pred, average="macro", zero_division=0) * 100, 2),
    }

def test_perfect_accuracy():
    labels = ["Positive", "Negative", "Neutral"]
    assert compute_metrics(labels, labels)["accuracy"] == 100.0

def test_f1_range():
    y_true = ["Positive", "Positive", "Negative", "Neutral"]
    y_pred = ["Positive", "Neutral",  "Negative", "Neutral"]
    metrics = compute_metrics(y_true, y_pred)
    assert 0 <= metrics["macro_f1"] <= 100

def test_metrics_keys_exist():
    metrics = compute_metrics(["Positive"], ["Positive"])
    assert all(k in metrics for k in ["accuracy", "macro_f1", "precision", "recall"])