"""
run.py
------
Entry point for the Speech Emotion Recognition FastAPI backend.

Usage (from project root or app/ directory):
    cd "e:\\Personal Projects\\Speech Recognition\\app"
    python run.py
"""

import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
