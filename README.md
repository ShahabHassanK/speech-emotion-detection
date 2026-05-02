# 🎙️ Speech Emotion Recognition — Hybrid CNN-LSTM

<p align="left">
  <img src="https://img.shields.io/badge/Python-3.11-3776AB?style=flat&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/PyTorch-2.3-EE4C2C?style=flat&logo=pytorch&logoColor=white" />
  <img src="https://img.shields.io/badge/FastAPI-0.111-009688?style=flat&logo=fastapi&logoColor=white" />
  <img src="https://img.shields.io/badge/Streamlit-1.33-FF4B4B?style=flat&logo=streamlit&logoColor=white" />
  <img src="https://img.shields.io/badge/Dataset-RAVDESS-6B46C1?style=flat" />
  <img src="https://img.shields.io/badge/Test%20Accuracy-88%25-16A34A?style=flat" />
</p>

A deep learning system that classifies **8 distinct human emotions** from raw vocal audio or video input. Built with a Hybrid CNN-LSTM architecture trained on the RAVDESS dataset, deployed as a real-time web application with a FastAPI backend and Streamlit frontend.

---

## 📸 Application Highlights

### Upload Interface
![Upload Page](viz/uploadpage.PNG)

### Detailed Emotion Analysis Result
![Results Page](viz/results.png)

---

## 📑 Table of Contents
1. [System Architecture & Workflow](#-system-architecture--workflow)
2. [Dataset Details & Label Encoding](#-dataset-details--label-encoding)
3. [Feature Engineering Details](#-feature-engineering-details)
4. [Model Building & Architecture](#-model-building--architecture)
5. [Model Training Details](#-model-training-details)
6. [Model Evaluation & Saving](#-model-evaluation--saving)
7. [Backend Implementation (FastAPI)](#-backend-implementation-fastapi)
8. [Frontend Implementation (Streamlit)](#-frontend-implementation-streamlit)
9. [Project Structure](#-project-structure)
10. [Setup & Installation](#️-setup--installation)
11. [API Reference](#-api-reference)

---

## ⚙️ System Architecture & Workflow

The pipeline is designed to be seamless, handling both audio and video files robustly. Here is the step-by-step breakdown of how the system processes an incoming file:

1. **File Ingestion:** The user uploads a media file (Audio: WAV, MP3, OGG, FLAC, M4A or Video: MP4, MOV, AVI, MKV, WEBM).
2. **Audio Extraction (If Video):** If the file is a video, the system uses `moviepy` to extract the underlying audio track and saves it to a temporary `.wav` file.
3. **Audio Loading:** `librosa` loads the audio into a mono waveform array at a sampling rate of `22,050 Hz`.
4. **Window Selection:** 
   - **Short Clips (< 3.5s):** Skips the first 0.5 seconds (often silence or breathing) and processes the rest.
   - **Long Clips (>= 3.5s):** Extracts a precise 3-second window exactly from the middle of the audio, capturing the most representative sustained emotion.
5. **Feature Engineering:** Extracts 40 MFCCs (Mel-Frequency Cepstral Coefficients) along with their first (delta) and second (delta-delta) derivatives. This produces a raw feature map.
6. **Padding/Truncation:** The feature map is padded or truncated to a fixed size of `(120, 174)`.
7. **Inference:** The processed feature tensor is passed through the pre-trained PyTorch CNN-LSTM model.
8. **Softmax Output:** The network outputs logits, which are passed through a Softmax function to produce a probability distribution across the 8 emotion classes.
9. **Response Delivery:** The backend formats the confidence scores, durations, and top emotion into a JSON response which the Streamlit frontend visualizes beautifully.

---

## 📊 Dataset Details & Label Encoding

The system is trained exclusively on the **RAVDESS** (Ryerson Audio-Visual Database of Emotional Speech and Song) dataset.

### Dataset Specifications
| Property | Specification |
|---|---|
| **Total Files** | 2,880 WAV audio files |
| **Actors** | 24 professional actors (12 male, 12 female) |
| **Emotions** | 8 specific emotional states |
| **Sample Rate** | 22,050 Hz (Standardized) |
| **Data Split** | 80% Training / 10% Validation / 10% Testing |

### Label Encoding
RAVDESS filenames contain standardized identifiers for emotions. We utilize `scikit-learn`'s `LabelEncoder` to map these string/numeric categories to contiguous integers `[0-7]` for cross-entropy loss computation.

| Identifier Code | Emotion | Identifier Code | Emotion |
|---|---|---|---|
| `01` | **Neutral** | `05` | **Angry** |
| `02` | **Calm** | `06` | **Fearful** |
| `03` | **Happy** | `07` | **Disgust** |
| `04` | **Sad** | `08` | **Surprised** |

---

## 🔬 Feature Engineering Details

Emotion in speech is carried in both the frequency spectrum (timbre, pitch) and how that spectrum changes over time (cadence, attack, decay). We capture this using a 3-channel-equivalent MFCC approach.

**Feature Pipeline Parameters:**
- `SAMPLE_RATE`: 22,050 Hz
- `DURATION`: 3.0 seconds
- `OFFSET`: 0.5 seconds
- `N_MFCC`: 40 coefficients
- `MAX_PAD_LEN`: 174 time frames

**Step-by-Step Transformation:**
1. **MFCC Extraction:** `librosa.feature.mfcc` extracts 40 coefficients over the time frames -> Shape: `(40, T)`.
2. **Delta Features (Velocity):** Computes the first derivative of the MFCCs to measure how the frequency spectrum is changing -> Shape: `(40, T)`.
3. **Delta-Delta Features (Acceleration):** Computes the second derivative to measure the rate of change of the spectrum -> Shape: `(40, T)`.
4. **Stacking:** The three matrices are stacked vertically resulting in a unified feature map of `120` features -> Shape: `(120, T)`.
5. **Standardization (Padding/Truncating):** To feed into the neural network, the time axis `T` must be constant. We pad with zeros if `T < 174` and truncate if `T > 174`.
6. **Formatting:** A channel dimension is added to make it compatible with PyTorch 2D Convolutions -> Final Tensor Shape: `(1, 120, 174)`.

---

## 🧠 Model Building & Architecture

### Why a Hybrid CNN-LSTM?
Emotion recognition requires understanding two different aspects of sound:
1. **Spatial/Spectral Features:** Identifying local patterns in the frequency domain (e.g., sharp harsh frequencies for anger, smooth low frequencies for calm). CNNs excel at this by treating the MFCC feature map as a 2D image.
2. **Temporal Dynamics:** Emotion evolves over time; a laugh builds up and fades away. LSTMs (Long Short-Term Memory networks) excel at remembering sequence context and understanding how these spectral features progress temporally.

By combining them, the CNN extracts high-level spectral features, and the LSTM analyzes the sequential flow of those features.

### Layer-by-Layer Architecture

**Input:** Tensor of shape `(Batch, 1, 120, 174)`

**1. CNN Feature Extractor:**
- **Block 1:** `Conv2D(1→32)` + `BatchNorm` + `ReLU` + `MaxPool2D(2x2)` + `Dropout(0.25)` -> Out: `(32, 60, 87)`
- **Block 2:** `Conv2D(32→64)` + `BatchNorm` + `ReLU` + `MaxPool2D(2x2)` + `Dropout(0.25)` -> Out: `(64, 30, 43)`
- **Block 3:** `Conv2D(64→128)` + `BatchNorm` + `ReLU` + `MaxPool2D(2x2)` + `Dropout(0.30)` -> Out: `(128, 15, 21)`

**2. Bridging CNN to LSTM:**
- The spatial output `(128 channels, 15 height, 21 width/time)` is flattened along the channel and height dimensions: `128 * 15 = 1920`.
- The tensor is permuted to treat the width `(21)` as the sequence length: `(Batch, 21, 1920)`.

**3. LSTM Temporal Modeler:**
- **LSTM Layer 1:** Input size `1920` → Hidden size `128` (with `dropout=0.3`) -> Out: `(Batch, 21, 128)`
- **LSTM Layer 2:** Input size `128` → Hidden size `64` (with `dropout=0.3`) -> Out: `(Batch, 21, 64)`
- *Sequence Truncation:* We take only the output of the **last time step** `[:, -1, :]` -> Out: `(Batch, 64)`.

**4. Classifier Head:**
- **Fully Connected 1:** `Linear(64 → 64)` + `ReLU` + `Dropout(0.40)`
- **Fully Connected 2:** `Linear(64 → 8)` -> Raw Logits for 8 emotions.

---

## ⏳ Model Training Details

The model was trained on Kaggle using an NVIDIA Tesla T4 GPU. The training script is available in `notebooks/cnnlstm-emotion-detection.ipynb`.

**Training Configuration:**
- **Optimizer:** `Adam` (Initial Learning Rate = 1e-3)
- **Loss Function:** `CrossEntropyLoss` (ideal for multi-class classification).
- **Batch Size:** `32`
- **Total Epochs Run:** `100`

**Callbacks & Schedulers:**
- **Learning Rate Scheduler:** `ReduceLROnPlateau` (factor=0.5, patience=7). If validation loss stagnates for 7 epochs, the learning rate is halved to fine-tune weights.
- **Early Stopping:** Monitored validation loss with a patience of 15 epochs. This prevents over-fitting by stopping training if the model stops generalizing.

---

## 📈 Model Evaluation & Saving

### Evaluation Metrics
The model achieved highly robust metrics across all emotion classes, successfully distinguishing closely related emotions like *calm* and *neutral*.

- **Best Validation Accuracy:** `83.98%`
- **Final Test Set Accuracy:** `88%` (evaluated on an unseen set of 576 samples).

| Emotion | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| 😠 Angry | 0.90 | 0.92 | **0.91** | 76 |
| 😌 Calm | 0.89 | 0.91 | **0.90** | 77 |
| 🤢 Disgust | 0.92 | 0.91 | **0.92** | 77 |
| 😨 Fearful | 0.87 | 0.88 | **0.88** | 77 |
| 😄 Happy | 0.83 | 0.81 | **0.82** | 77 |
| 😐 Neutral | 0.81 | 0.76 | **0.78** | 38 |
| 😢 Sad | 0.80 | 0.83 | **0.82** | 77 |
| 😲 Surprised | 0.97 | 0.94 | **0.95** | 77 |
| **Macro Avg** | **0.87** | **0.87** | **0.87** | 576 |

**Confusion Matrix**
![Confusion Matrix](viz/confusion_matrix.png)

### Saving the Model and Label Encoder
Upon completing training and achieving the best validation score, the system's state was serialized to ensure deterministic inference in production:
1. **Model Weights:** The PyTorch model's state dictionary (`model.state_dict()`) was saved as `ser_cnn_lstm.pth`.
2. **Label Encoder:** The fitted `scikit-learn` `LabelEncoder` object, which maps the network's `0-7` output indices back to human-readable strings like `happy` or `sad`, was serialized using Python's `pickle` library and saved as `label_encoder.pkl`.

Both artifacts are stored in the `/models` directory and loaded into memory on backend startup.

---

## 🚀 Backend Implementation (FastAPI)

The inference server is built using **FastAPI** (`app/main.py` and `app/predict.py`), providing a lightning-fast, asynchronous API for the frontend.

**Key Implementation Details:**
1. **Startup Initialization:** The PyTorch model and the pickled Label Encoder are loaded into RAM immediately when the Uvicorn server starts, preventing cold-start delays during API calls. The model is explicitly set to `model.eval()` to disable dropout layers.
2. **CORS Middleware:** Implemented to allow cross-origin requests from the Streamlit frontend.
3. **Format Handling:** The `/predict` endpoint accepts `multipart/form-data`. It gracefully handles validation, ensuring only supported audio/video extensions are processed.
4. **Temporary File Management:** For video files, the payload is temporarily saved to disk, audio is extracted via `moviepy`, and both temp files are securely unlinked (deleted) in a `finally` block to prevent disk space leaks.
5. **Inference Pipeline:** Calls the `predict_emotion()` function, which runs the waveform through the identical feature extraction pipeline used during training, executes a forward pass under `torch.no_grad()`, and applies `F.softmax` to return a standardized JSON response containing the confidence of all 8 emotions.

---

## 🎨 Frontend Implementation (Streamlit)

The user interface (`frontend/app.py`) is built using **Streamlit**, heavily customized with CSS to deliver a premium, modern web experience.

**Key Implementation Details:**
1. **Custom Styling:** Vanilla Streamlit elements are overridden using injected CSS to utilize the 'Inter' font, clean white backgrounds, pill-shaped buttons, and dynamic hover effects.
2. **Interactive Audio/Video:** Utilizes `st.audio` and `st.video` components so users can preview their uploads directly.
3. **Waveform Visualization:** Generates a real-time amplitude waveform of the uploaded audio using `plotly.graph_objects`, giving the user immediate visual feedback of their audio's structure.
4. **Emotion Breakdown Bar Chart:** The JSON response from the API is parsed to construct a responsive, horizontal bar chart highlighting the dominant emotion in a vibrant color while muting the secondary emotions.
5. **Session State:** Utilizes `st.session_state` to persist the inference results across UI reruns, allowing the user to view results without the app losing context.

---

## 📁 Project Structure

```text
Speech Recognition/
│
├── models/
│   ├── ser_cnn_lstm.pth                  # Trained PyTorch model weights
│   └── label_encoder.pkl                 # Pickled Scikit-learn LabelEncoder
│
├── notebooks/
│   └── cnnlstm-emotion-detection.ipynb   # Full training notebook (Kaggle)
│
├── app/
│   ├── model.py                          # CNN-LSTM PyTorch class definition
│   ├── predict.py                        # Audio/video loading & inference pipeline
│   ├── main.py                           # FastAPI routes and middleware
│   └── run.py                            # Uvicorn entry point
│
├── frontend/
│   └── app.py                            # Streamlit UI with custom CSS
│
├── viz/
│   ├── confusion_matrix.png              # Test set evaluation matrix
│   ├── uploadpage.PNG                    # UI screenshot (Upload state)
│   └── results.png                       # UI screenshot (Result state)
│
├── requirements.txt                      # Project dependencies
└── README.md                             # Comprehensive project documentation
```

---

## ⚙️ Setup & Installation

### Prerequisites
- Python 3.10 or 3.11
- Git

### 1. Clone the repository

```bash
git clone https://github.com/shahabhassank/speech-emotion-recognition.git
cd speech-emotion-recognition
```

### 2. Create a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** On first use of the application, `moviepy` will download a bundled `ffmpeg` binary (~60 MB). This is a one-time download required for robust video and MP3 extraction support.

---

## 🏁 Running the Application

You need **two terminals**, both with the virtual environment activated.

### Terminal 1 — Start the API Backend
```bash
cd app
python run.py
```
The API will be available at `http://localhost:8000`.
Interactive Swagger API docs are available at `http://localhost:8000/docs`.

### Terminal 2 — Start the Streamlit Frontend
```bash
cd frontend
streamlit run app.py
```
The User Interface will open automatically in your browser at `http://localhost:8501`.

---

## 🌐 API Reference

### `GET /` (Health Check)
Confirms the model is loaded and the API is ready to accept requests.

**Response:**
```json
{
  "status": "running",
  "model": "CNN-LSTM (PyTorch)",
  "dataset": "RAVDESS",
  "test_accuracy": "88%",
  "emotions": ["angry", "calm", "disgust", "fearful", "happy", "neutral", "sad", "surprised"],
  "supported_formats": [".avi", ".flac", ".m4a", ".m4v", ".mkv", ".mov", ".mp3", ".mp4", ".ogg", ".wav", ".webm"]
}
```

### `POST /predict` (Inference Endpoint)
Analyzes the emotion from an uploaded audio or video file.

**Request:** `multipart/form-data` containing the file under the key `file`.

**Response:**
```json
{
  "emotion": "happy",
  "confidence": 91.4,
  "all_scores": {
    "angry": 0.8,
    "calm": 1.2,
    "disgust": 0.3,
    "fearful": 0.5,
    "happy": 91.4,
    "neutral": 2.1,
    "sad": 1.9,
    "surprised": 1.8
  },
  "duration": 4.2,
  "window": "Middle 3s of 4.2s clip"
}
```

---

## 📄 License
This project is licensed under the MIT License.
