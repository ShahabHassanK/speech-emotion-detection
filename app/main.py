"""
main.py
-------
FastAPI application for the Speech Emotion Recognition API.
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from predict import predict_emotion, le

ALLOWED_EXTENSIONS = {
    # Audio
    ".wav", ".mp3", ".ogg", ".flac", ".m4a",
    # Video
    ".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v",
}

app = FastAPI(
    title="Speech Emotion Recognition API",
    description=(
        "Hybrid CNN-LSTM model trained on the RAVDESS dataset. "
        "Classifies 8 emotions from audio or video input."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["Health"])
def health_check():
    """API health check — confirms the model is loaded and ready."""
    return {
        "status": "running",
        "model": "CNN-LSTM (PyTorch)",
        "dataset": "RAVDESS",
        "test_accuracy": "88%",
        "emotions": list(le.classes_),
        "supported_formats": sorted(ALLOWED_EXTENSIONS),
    }


@app.post("/predict", tags=["Inference"])
async def predict(file: UploadFile = File(...)):
    """
    Predict emotion from an uploaded audio or video file.

    Returns the top predicted emotion, its confidence score,
    probability scores for all 8 emotion classes, clip duration,
    and which segment of the audio was analysed.
    """
    suffix = ("." + file.filename.rsplit(".", 1)[-1].lower()) if "." in file.filename else ""

    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported format '{suffix}'. "
                f"Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
            ),
        )

    content = await file.read()

    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        result = predict_emotion(content, file.filename)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference error: {exc}")

    return result
