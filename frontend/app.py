"""
app.py — Vocal Emotion AI
Run: streamlit run app.py
"""
import io
import streamlit as st
import requests
import plotly.graph_objects as go
import librosa
import numpy as np

API_URL = "http://localhost:8000/predict"

EMOTIONS = {
    "angry":     {"emoji": "😠", "color": "#E53E3E", "bg": "#FFF5F5", "label": "Angry"},
    "calm":      {"emoji": "😌", "color": "#0987A0", "bg": "#EDFDFD", "label": "Calm"},
    "disgust":   {"emoji": "🤢", "color": "#276749", "bg": "#F0FFF4", "label": "Disgust"},
    "fearful":   {"emoji": "😨", "color": "#6B46C1", "bg": "#FAF5FF", "label": "Fearful"},
    "happy":     {"emoji": "😄", "color": "#C05621", "bg": "#FFFAF0", "label": "Happy"},
    "neutral":   {"emoji": "😐", "color": "#4A5568", "bg": "#F7FAFC", "label": "Neutral"},
    "sad":       {"emoji": "😢", "color": "#3730A3", "bg": "#EEF2FF", "label": "Sad"},
    "surprised": {"emoji": "😲", "color": "#C4370A", "bg": "#FFF8F1", "label": "Surprised"},
}

st.set_page_config(page_title="Vocal Emotion AI", page_icon="🎙️", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"], * {
    font-family: 'Inter', system-ui, sans-serif !important;
}

/* ── Base ── */
.stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
    background: #FFFFFF !important;
}
.block-container {
    padding: 0 2.5rem 4rem !important;
    max-width: 1080px !important;
}
header[data-testid="stHeader"], footer, [data-testid="stDecoration"] {
    display: none !important;
}

/* ── Text ── */
[data-testid="stMarkdownContainer"] h1,
[data-testid="stMarkdownContainer"] h2,
[data-testid="stMarkdownContainer"] h3,
[data-testid="stMarkdownContainer"] h4,
[data-testid="stMarkdownContainer"] h5,
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] span {
    color: #1A202C !important;
}
[data-testid="stCaptionContainer"] p, .stCaption p {
    color: #718096 !important;
    font-size: 0.85rem !important;
}

/* ── Upload zone — force white, clean ── */
[data-testid="stFileUploadDropzone"] {
    background: #F7F8FA !important;
    border: 2px dashed #CBD5E0 !important;
    border-radius: 14px !important;
    padding: 2rem 1.5rem !important;
    transition: all 0.2s !important;
}
[data-testid="stFileUploadDropzone"]:hover {
    border-color: #4F46E5 !important;
    background: #F5F3FF !important;
}
[data-testid="stFileUploadDropzone"] p,
[data-testid="stFileUploadDropzone"] span {
    color: #718096 !important;
}
[data-testid="stFileUploadDropzone"] button {
    background: #4F46E5 !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
}

/* ── Primary button ── */
.stButton > button {
    background: #4F46E5 !important;
    color: #fff !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    width: 100% !important;
    padding: 0.65rem 1.5rem !important;
    transition: all 0.2s !important;
    box-shadow: 0 4px 14px rgba(79,70,229,0.35) !important;
    letter-spacing: 0.01em !important;
}
.stButton > button:hover {
    background: #4338CA !important;
    box-shadow: 0 6px 20px rgba(79,70,229,0.45) !important;
    transform: translateY(-1px) !important;
}
.stButton > button:active { transform: translateY(0) !important; }

/* ── Metrics ── */
[data-testid="stMetricLabel"] p {
    color: #A0AEC0 !important;
    font-size: 0.65rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
}
[data-testid="stMetricValue"] {
    color: #1A202C !important;
    font-size: 1rem !important;
    font-weight: 700 !important;
}
[data-testid="stMetricDelta"] { display: none !important; }

/* ── Cards ── */
[data-testid="stVerticalBlockBorderWrapper"] {
    border: 1px solid #EDF2F7 !important;
    border-radius: 16px !important;
    background: #FFFFFF !important;
    box-shadow: 0 1px 4px rgba(0,0,0,0.05) !important;
}

/* ── Progress ── */
[data-testid="stProgress"] > div {
    background: #EDF2F7 !important;
    border-radius: 999px !important;
    height: 6px !important;
}
[data-testid="stProgress"] > div > div {
    background: linear-gradient(90deg, #4F46E5, #7C3AED) !important;
    border-radius: 999px !important;
}

/* ── Media ── */
audio { width: 100%; border-radius: 10px; }
video { border-radius: 12px; width: 100%; }

/* ── Misc ── */
.stSpinner > div { border-top-color: #4F46E5 !important; }
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-thumb { background: #E2E8F0; border-radius: 4px; }
hr { border-color: #EDF2F7 !important; margin: 1.25rem 0 !important; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def render_waveform(audio_bytes):
    try:
        wv, sr = librosa.load(io.BytesIO(audio_bytes), sr=22050, mono=True, duration=30)
        step = max(1, len(wv) // 700)
        wv_ds = wv[::step][:700]
        t = np.linspace(0, len(wv) / sr, len(wv_ds))
        fig = go.Figure(go.Scatter(
            x=t, y=wv_ds, fill="tozeroy",
            line=dict(color="#4F46E5", width=1.0),
            fillcolor="rgba(79,70,229,0.07)",
            hovertemplate="%{x:.1f}s<extra></extra>",
        ))
        fig.update_layout(
            height=64, margin=dict(l=0, r=0, t=0, b=0),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
            yaxis=dict(showgrid=False, showticklabels=False, zeroline=False, range=[-1, 1]),
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    except Exception:
        pass


def render_breakdown(all_scores, top_emotion):
    items = sorted(all_scores.items(), key=lambda x: x[1], reverse=True)
    labels = [f"{EMOTIONS.get(e, {}).get('emoji', '')}  {e.capitalize()}" for e, _ in items]
    values = [s for _, s in items]
    bar_colors = []
    text_colors = []
    for e, _ in items:
        if e == top_emotion:
            bar_colors.append(EMOTIONS.get(e, {}).get("color", "#4F46E5"))
            text_colors.append(EMOTIONS.get(e, {}).get("color", "#4F46E5"))
        else:
            bar_colors.append("#EDF2F7")
            text_colors.append("#A0AEC0")

    fig = go.Figure(go.Bar(
        x=values, y=labels, orientation="h",
        marker=dict(color=bar_colors, line=dict(width=0)),
        text=[f"  {v:.1f}%" for v in values],
        textposition="outside",
        textfont=dict(size=11, color=text_colors, family="Inter"),
        hovertemplate="%{y}: %{x:.1f}%<extra></extra>",
        cliponaxis=False,
    ))
    fig.update_layout(
        height=260, margin=dict(l=0, r=55, t=0, b=0),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(range=[0, 120], showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, autorange="reversed",
                   tickfont=dict(size=12, color="#4A5568", family="Inter")),
        showlegend=False, bargap=0.38,
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


# ── Top accent bar ────────────────────────────────────────────────────────────
st.markdown(
    '<div style="height:4px;background:linear-gradient(90deg,#4F46E5 0%,#7C3AED 50%,#EC4899 100%);'
    'margin:-1px -9999px 0;"></div>',
    unsafe_allow_html=True
)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("<div style='height:2rem'></div>", unsafe_allow_html=True)
st.markdown(
    '<div style="display:flex;align-items:center;gap:.8rem;margin-bottom:.3rem;">'
    '<span style="font-size:2rem;font-weight:800;color:#1A202C;letter-spacing:-.03em;">'
    '🎙️ Vocal Emotion AI'
    '</span>'
    '<span style="background:linear-gradient(135deg,#EEF2FF,#F5F3FF);color:#4F46E5;'
    'font-size:.68rem;font-weight:700;padding:.22rem .75rem;border-radius:999px;'
    'letter-spacing:.05em;border:1px solid #C7D2FE;">'
    'CNN-LSTM · 88% ACC'
    '</span>'
    '</div>',
    unsafe_allow_html=True
)
st.caption("Detect emotion from any voice recording or video clip.")
st.markdown("<div style='height:.75rem'></div>", unsafe_allow_html=True)
st.divider()

# ── Layout ────────────────────────────────────────────────────────────────────
left, right = st.columns([1, 1], gap="large")

# ══ LEFT ══════════════════════════════════════════════════════════════════════
with left:
    st.markdown(
        '<p style="font-size:.7rem;font-weight:700;color:#A0AEC0;text-transform:uppercase;'
        'letter-spacing:.1em;margin-bottom:.5rem;">Input</p>',
        unsafe_allow_html=True
    )

    uploaded = st.file_uploader(
        "Drop a file or browse",
        type=["wav", "mp3", "ogg", "flac", "m4a", "mp4", "mov", "avi", "mkv", "webm"],
        label_visibility="collapsed",
    )

    if uploaded:
        ext = uploaded.name.rsplit(".", 1)[-1].lower()
        is_audio = ext in {"wav", "mp3", "ogg", "flac", "m4a"}

        st.markdown("<div style='height:.6rem'></div>", unsafe_allow_html=True)

        with st.container(border=True):
            if is_audio:
                st.audio(uploaded)
                render_waveform(uploaded.getvalue())
            else:
                st.video(uploaded)

            st.markdown("<div style='height:.5rem'></div>", unsafe_allow_html=True)
            mc1, mc2 = st.columns(2)
            mc1.metric("Format", ext.upper())
            mc2.metric("Size", f"{uploaded.size / 1e6:.1f} MB")

        st.markdown("<div style='height:.6rem'></div>", unsafe_allow_html=True)

        if st.button("Analyse Emotion →"):
            msg = (
                "Extracting audio from video…"
                if not is_audio else
                "Running emotion analysis…"
            )
            with st.spinner(msg):
                try:
                    r = requests.post(
                        API_URL,
                        files={"file": (uploaded.name, uploaded.getvalue())},
                        timeout=300,
                    )
                    r.raise_for_status()
                    st.session_state["result"] = r.json()
                    st.rerun()
                except requests.exceptions.ConnectionError:
                    st.error("Backend not running. Start it with:\n```\ncd app\npython run.py\n```")
                except requests.exceptions.Timeout:
                    st.error("Request timed out — please try again.")
                except requests.exceptions.HTTPError as e:
                    try:
                        detail = e.response.json().get("detail", str(e))
                    except Exception:
                        detail = str(e)
                    st.error(f"Error: {detail}")
                except Exception as e:
                    st.error(f"Unexpected error: {e}")

    else:
        # Format pills when no file uploaded
        st.markdown("<div style='height:.4rem'></div>", unsafe_allow_html=True)
        st.markdown(
            '<div style="display:flex;flex-wrap:wrap;gap:.45rem;margin-top:.25rem;">'
            + "".join(
                f'<span style="background:#F7F8FA;border:1px solid #E2E8F0;color:#718096;'
                f'font-size:.72rem;font-weight:500;padding:.2rem .65rem;border-radius:999px;">{f}</span>'
                for f in ["WAV", "MP3", "OGG", "FLAC", "M4A", "MP4", "MOV", "AVI", "MKV", "WEBM"]
            )
            + '</div>',
            unsafe_allow_html=True
        )


# ══ RIGHT ═════════════════════════════════════════════════════════════════════
with right:
    st.markdown(
        '<p style="font-size:.7rem;font-weight:700;color:#A0AEC0;text-transform:uppercase;'
        'letter-spacing:.1em;margin-bottom:.5rem;">Result</p>',
        unsafe_allow_html=True
    )

    with st.container(border=True):
        if "result" not in st.session_state:
            st.markdown(
                '<div style="text-align:center;padding:3.5rem 1rem 3rem;">'
                '<div style="font-size:3rem;margin-bottom:.9rem;opacity:.2;">🎵</div>'
                '<p style="font-weight:600;color:#A0AEC0;margin:0 0 .25rem;font-size:.95rem;">No result yet</p>'
                '<p style="font-size:.82rem;color:#CBD5E0;margin:0;">Upload a file and click Analyse</p>'
                '</div>',
                unsafe_allow_html=True
            )
        else:
            result  = st.session_state["result"]
            emotion = result["emotion"]
            m       = EMOTIONS.get(emotion, EMOTIONS["neutral"])
            conf    = result["confidence"]

            # Emotion hero row
            ec1, ec2 = st.columns([3, 1])
            with ec1:
                st.markdown(f"### {m['emoji']} {m['label']}")
            with ec2:
                st.metric("Confidence", f"{conf:.1f}%")

            st.progress(conf / 100)
            st.divider()

            st.markdown(
                '<p style="font-size:.7rem;font-weight:700;color:#A0AEC0;'
                'text-transform:uppercase;letter-spacing:.1em;margin-bottom:0;">Breakdown</p>',
                unsafe_allow_html=True
            )
            render_breakdown(result["all_scores"], emotion)

            st.divider()
            d1, d2 = st.columns(2)
            d1.metric("Duration", f"{result.get('duration', 0):.1f}s")
            d2.metric("Model", "CNN-LSTM")

    if "result" in st.session_state:
        st.markdown("<div style='height:.4rem'></div>", unsafe_allow_html=True)
        if st.button("Clear"):
            del st.session_state["result"]
            st.rerun()
