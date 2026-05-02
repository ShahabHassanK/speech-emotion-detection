"""
predict.py
----------
Audio/video preprocessing and emotion inference pipeline.

Supported inputs:
  Audio : WAV, MP3, OGG, FLAC, M4A  (via librosa — uses imageio-ffmpeg for MP3)
  Video : MP4, MOV, AVI, MKV, WEBM  (audio extracted via moviepy)

Analysis window logic:
  < 3.5s clips  → skip first 0.5 s, use remaining audio
  >= 3.5s clips → extract a 3-second window from the middle
"""

import io
import os
import pickle
import tempfile
import warnings
from pathlib import Path

import librosa
import numpy as np
import torch
import torch.nn.functional as F

from model import CNNLSTM

warnings.filterwarnings("ignore")

# ── Paths ───────────────────────────────────────────────────────────────────
BASE_DIR     = Path(__file__).resolve().parent.parent
MODEL_PATH   = BASE_DIR / "models" / "ser_cnn_lstm.pth"
ENCODER_PATH = BASE_DIR / "models" / "label_encoder.pkl"

# ── Constants — must match training exactly ──────────────────────────────────
SAMPLE_RATE = 22050
DURATION    = 3.0    # seconds to analyse
OFFSET      = 0.5    # skip first 0.5 s (silence / breath)
N_MFCC      = 40
MAX_PAD_LEN = 174    # time frames after STFT

# ── Format sets ─────────────────────────────────────────────────────────────
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".flv"}
AUDIO_EXTENSIONS = {".wav", ".mp3", ".ogg", ".flac", ".m4a", ".aiff"}

# ── Load model + encoder once at startup ────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CNNLSTM(num_classes=8)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

with open(ENCODER_PATH, "rb") as f:
    le = pickle.load(f)


# ── Audio loading ────────────────────────────────────────────────────────────

def _extract_audio_from_video(video_bytes: bytes, suffix: str) -> np.ndarray:
    """Write video to a temp file, extract audio with moviepy, return waveform."""
    from moviepy.editor import VideoFileClip

    tmp_audio_path = None
    tmp_video_path = None

    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(video_bytes)
            tmp_video_path = tmp.name

        with VideoFileClip(tmp_video_path) as clip:
            if clip.audio is None:
                raise ValueError("This video file has no audio track.")

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_a:
                tmp_audio_path = tmp_a.name

            clip.audio.write_audiofile(tmp_audio_path, fps=SAMPLE_RATE, logger=None)

        waveform, _ = librosa.load(tmp_audio_path, sr=SAMPLE_RATE, mono=True)
        return waveform

    finally:
        if tmp_video_path and os.path.exists(tmp_video_path):
            os.unlink(tmp_video_path)
        if tmp_audio_path and os.path.exists(tmp_audio_path):
            os.unlink(tmp_audio_path)


def load_audio(file_bytes: bytes, filename: str) -> tuple:
    """
    Load audio from raw bytes for any supported format.
    Returns (waveform: np.ndarray, total_duration: float).

    librosa.load handles WAV / OGG / FLAC / M4A / MP3 transparently
    once imageio-ffmpeg is installed (pulled in by moviepy).
    """
    suffix = Path(filename).suffix.lower()

    if suffix in VIDEO_EXTENSIONS:
        waveform = _extract_audio_from_video(file_bytes, suffix)
    else:
        # Works for WAV, MP3, OGG, FLAC, M4A
        waveform, _ = librosa.load(io.BytesIO(file_bytes), sr=SAMPLE_RATE, mono=True)

    total_duration = len(waveform) / SAMPLE_RATE
    return waveform, total_duration


# ── Window extraction ────────────────────────────────────────────────────────

def get_analysis_window(waveform: np.ndarray, total_duration: float) -> tuple:
    """
    Extract the best 3-second segment for emotion analysis.

    Short clips (< 3.5 s) → skip first 0.5 s, use everything else.
    Long clips  (>= 3.5 s) → take 3 s from the middle (most representative).

    Returns (segment: np.ndarray, window_description: str).
    """
    min_samples = int(DURATION * SAMPLE_RATE)

    if total_duration < (DURATION + OFFSET):
        start = min(int(OFFSET * SAMPLE_RATE), len(waveform) // 4)
        segment = waveform[start:]
        window_desc = f"Full clip ({total_duration:.1f}s)"
    else:
        mid = len(waveform) // 2
        half = int((DURATION / 2) * SAMPLE_RATE)
        start = max(0, mid - half)
        segment = waveform[start: start + int(DURATION * SAMPLE_RATE)]
        window_desc = f"Middle 3s of {total_duration:.1f}s clip"

    # Pad if shorter than expected, truncate if longer
    if len(segment) < min_samples:
        segment = np.pad(segment, (0, min_samples - len(segment)))
    else:
        segment = segment[:min_samples]

    return segment, window_desc


# ── Feature extraction ───────────────────────────────────────────────────────

def extract_features(segment: np.ndarray) -> torch.Tensor:
    """
    MFCC + delta + delta-delta  →  shape (1, 1, 120, 174).
    Must exactly match the training pipeline in the Kaggle notebook.
    """
    mfcc   = librosa.feature.mfcc(y=segment, sr=SAMPLE_RATE, n_mfcc=N_MFCC)
    delta  = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)

    combined = np.vstack([mfcc, delta, delta2])  # (120, T)

    if combined.shape[1] < MAX_PAD_LEN:
        combined = np.pad(combined, ((0, 0), (0, MAX_PAD_LEN - combined.shape[1])))
    else:
        combined = combined[:, :MAX_PAD_LEN]

    return torch.FloatTensor(combined).unsqueeze(0).unsqueeze(0)  # (1, 1, 120, 174)


# ── Full inference pipeline ──────────────────────────────────────────────────

def predict_emotion(file_bytes: bytes, filename: str) -> dict:
    """
    End-to-end inference: bytes → emotion result dict.

    Returns:
        {
            "emotion":    str,
            "confidence": float,           # percentage, e.g. 87.34
            "all_scores": dict[str, float],
            "duration":   float,           # total clip duration in seconds
            "window":     str,             # description of analysed segment
        }
    """
    waveform, total_duration = load_audio(file_bytes, filename)
    segment, window_desc     = get_analysis_window(waveform, total_duration)
    features                 = extract_features(segment).to(device)

    with torch.no_grad():
        logits = model(features)
        probs  = F.softmax(logits, dim=1).squeeze().cpu().numpy()

    emotions   = le.classes_
    all_scores = {e: round(float(p) * 100, 2) for e, p in zip(emotions, probs)}
    top_emotion = max(all_scores, key=all_scores.get)

    return {
        "emotion":    top_emotion,
        "confidence": all_scores[top_emotion],
        "all_scores": all_scores,
        "duration":   round(total_duration, 2),
        "window":     window_desc,
    }
