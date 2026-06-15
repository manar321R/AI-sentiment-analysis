from fastapi.responses import FileResponse
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from transformers import AutoTokenizer, BertForSequenceClassification
import pandas as pd
import torch
from pydantic import BaseModel # new

app = FastAPI()

MODEL_DIR = "sentiment_bert_model_best"

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = BertForSequenceClassification.from_pretrained(MODEL_DIR)

# Label mapping
label_map = {
    0: "Negative",
    1: "Neutral",
    2: "Positive"
}


@app.get("/")
def dashboard():
    return FileResponse("index.html")


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read CSV
        df = pd.read_csv(file.file)

        # Change this if your text column has another name
        text_column = df.columns[0]

        texts = df[text_column].astype(str).tolist()

        predictions = []

        for text in texts:
            inputs = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=128
            )

            with torch.no_grad():
                outputs = model(**inputs)

            pred = torch.argmax(outputs.logits, dim=1).item()

            predictions.append(label_map[pred])

        df["Prediction"] = predictions

        sentiment_counts = df["Prediction"].value_counts().to_dict()

        return JSONResponse(
            content={
                "total_records": len(df),
                "sentiment_distribution": sentiment_counts,
                "results": df.head(20).to_dict(orient="records")
            }
        )

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )


# Define the expected input data schema
class SingleReview(BaseModel):
    text: str

# New endpoint for single text sentiment analysis
@app.post("/predict_text")
async def predict_single_text(review: SingleReview):
    try:
        # Prepare the text using the existing tokenizer configurations
        inputs = tokenizer(
            review.text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128
        )

        # Perform prediction
        with torch.no_grad():
            outputs = model(**inputs)

        pred = torch.argmax(outputs.logits, dim=1).item()
        sentiment = label_map[pred]

        return {"sentiment": sentiment}

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )