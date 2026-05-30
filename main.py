# ============================================================
# main.py  —  FastAPI Backend
# AI-Based Sentiment Analysis System
# ============================================================
# يحمّل الموديل من Hugging Face Hub تلقائياً
# لا يحتاج saved_model/ محلياً
# ============================================================

import re
import html
import time
import os
import io
from typing import List

import torch
import pandas as pd
from torch.nn.functional import softmax
from transformers import BertTokenizer, BertForSequenceClassification

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


# ─────────────────────────────────────────────
# Configuration
# غيّري HF_MODEL_NAME لاسم الموديل بعد ما ترفعيه
# مثال: "lamar123/sentiment-bert-model"
# ─────────────────────────────────────────────
HF_MODEL_NAME = os.getenv("HF_MODEL_NAME", "YOUR_HF_USERNAME/sentiment-bert-model")
MAX_LEN       = 128
NUM_LABELS    = 3
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LABEL_MAP     = {0: "Negative", 1: "Neutral", 2: "Positive"}


# ─────────────────────────────────────────────
# preprocess_text — من كود لمار
# ─────────────────────────────────────────────
def preprocess_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = html.unescape(text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[^\x00-\x7F]+", " ", text)
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ─────────────────────────────────────────────
# Model Loading من Hugging Face Hub
# ─────────────────────────────────────────────
print(f"⏳ Loading model from Hugging Face: {HF_MODEL_NAME}")
try:
    tokenizer = BertTokenizer.from_pretrained(HF_MODEL_NAME)
    model     = BertForSequenceClassification.from_pretrained(
        HF_MODEL_NAME, num_labels=NUM_LABELS
    )
    model.to(DEVICE)
    model.eval()
    print(f"✅ Model loaded successfully on {DEVICE}")
except Exception as e:
    print(f"❌ Could not load model: {e}")
    print("   → تأكدي أن HF_MODEL_NAME صحيح وأن الموديل public")
    tokenizer = None
    model     = None


# ─────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────
def run_inference(text: str):
    if model is None or tokenizer is None:
        raise HTTPException(
            status_code=503,
            detail=f"Model not loaded from '{HF_MODEL_NAME}'. Check HF_MODEL_NAME env variable."
        )
    clean = preprocess_text(text)
    if not clean:
        return "Neutral", 0.33, {"Negative": 0.33, "Neutral": 0.34, "Positive": 0.33}

    enc = tokenizer(
        clean, max_length=MAX_LEN, truncation=True,
        padding="max_length", return_tensors="pt"
    )
    enc = {k: v.to(DEVICE) for k, v in enc.items()}

    with torch.no_grad():
        logits = model(**enc).logits

    probs      = softmax(logits, dim=1).squeeze().tolist()
    pred_idx   = int(torch.argmax(logits, dim=1))
    label      = LABEL_MAP[pred_idx]
    confidence = probs[pred_idx]
    probs_dict = {LABEL_MAP[i]: round(probs[i], 4) for i in range(NUM_LABELS)}
    return label, round(confidence, 4), probs_dict


# ─────────────────────────────────────────────
# FastAPI App
# ─────────────────────────────────────────────
app = FastAPI(
    title="Sentiment Analysis API",
    description="AI-Based Sentiment Classification — UQU Project",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────
# Schemas
# ─────────────────────────────────────────────
class SingleRequest(BaseModel):
    text: str

class MultiRequest(BaseModel):
    texts: List[str]

class PredictionResult(BaseModel):
    text_preview: str
    sentiment: str
    confidence: float
    probabilities: dict
    processing_time_ms: float


# ─────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "status": "running",
        "model": HF_MODEL_NAME,
        "model_loaded": model is not None,
        "device": str(DEVICE),
        "endpoints": ["/predict", "/predict-batch", "/predict-csv", "/docs"]
    }

@app.get("/health")
def health():
    return {"status": "ok", "model_ready": model is not None}


# ── 1. Single ──────────────────────────────
@app.post("/predict", response_model=PredictionResult)
def predict_single(req: SingleRequest):
    """تحليل مراجعة واحدة"""
    if not req.text.strip():
        raise HTTPException(status_code=422, detail="text cannot be empty.")
    t0 = time.time()
    label, conf, probs = run_inference(req.text)
    elapsed = round((time.time() - t0) * 1000, 2)
    return PredictionResult(
        text_preview=req.text[:100],
        sentiment=label,
        confidence=conf,
        probabilities=probs,
        processing_time_ms=elapsed
    )


# ── 2. Batch JSON ───────────────────────────
@app.post("/predict-batch")
def predict_batch(req: MultiRequest):
    """تحليل عدة مراجعات دفعة واحدة"""
    if not req.texts:
        raise HTTPException(status_code=422, detail="texts list cannot be empty.")
    if len(req.texts) > 1000:
        raise HTTPException(status_code=422, detail="Max 1000 reviews per request.")

    t0 = time.time()
    results = []
    for text in req.texts:
        label, conf, probs = run_inference(text)
        results.append({
            "text_preview": text[:80],
            "sentiment": label,
            "confidence": conf,
            "probabilities": probs
        })
    total_ms = round((time.time() - t0) * 1000, 2)
    sentiments = [r["sentiment"] for r in results]

    return {
        "summary": {
            "total": len(results),
            "positive": sentiments.count("Positive"),
            "neutral":  sentiments.count("Neutral"),
            "negative": sentiments.count("Negative"),
            "total_processing_time_ms": total_ms,
            "avg_time_per_review_ms": round(total_ms / len(results), 2)
        },
        "results": results
    }


# ── 3. CSV Upload ───────────────────────────
@app.post("/predict-csv")
async def predict_csv(
    file: UploadFile = File(...),
    text_column: str = "review_text"
):
    """رفع ملف CSV وتحليل كل المراجعات"""
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=422, detail="Only .csv files accepted.")
    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
    except Exception:
        raise HTTPException(status_code=422, detail="Could not parse CSV file.")

    if text_column not in df.columns:
        raise HTTPException(
            status_code=422,
            detail=f"Column '{text_column}' not found. Available: {list(df.columns)}"
        )

    texts = df[text_column].fillna("").astype(str).tolist()
    if len(texts) > 5000:
        raise HTTPException(status_code=422, detail="Max 5000 rows per CSV.")

    t0 = time.time()
    results = []
    for text in texts:
        label, conf, probs = run_inference(text)
        results.append({"text_preview": text[:80], "sentiment": label, "confidence": conf})

    total_ms  = round((time.time() - t0) * 1000, 2)
    sentiments = [r["sentiment"] for r in results]

    return {
        "filename": file.filename,
        "rows_analyzed": len(results),
        "summary": {
            "positive": sentiments.count("Positive"),
            "neutral":  sentiments.count("Neutral"),
            "negative": sentiments.count("Negative"),
            "total_processing_time_ms": total_ms
        },
        "results": results
    }
