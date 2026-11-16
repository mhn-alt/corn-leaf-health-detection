import numpy as np
import streamlit as st
from PIL import Image, ImageOps
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

MODEL_PATH = "corn_leaf_disease_model.h5"
CLASS_NAMES = ["diseased", "healthy"]   # alphabetical, matches training; sigmoid = P(healthy)
IMG_SIZE = (224, 224)
LOW_CONF = 0.70

st.set_page_config(page_title="Corn Leaf Doctor", page_icon="🌽", layout="centered")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Poppins', sans-serif; }
.stApp { background: linear-gradient(160deg, #f1f8e9 0%, #fffde7 100%); }
#MainMenu, footer { visibility: hidden; }
.hero { text-align: center; padding: 1.2rem 0 0.4rem; }
.hero h1 { font-size: 2.6rem; font-weight: 700; color: #2e5d1e; margin-bottom: 0; }
.hero p { color: #6b7a5e; font-size: 1.05rem; }
div[role="radiogroup"] { justify-content: center; gap: 10px; flex-wrap: wrap; }
div[role="radiogroup"] > label {
    background: #ffffffcc; border-radius: 30px; padding: 6px 18px;
    box-shadow: 0 2px 8px rgba(60,90,40,0.10);
}
.card {
    background: white; border-radius: 22px; padding: 1.4rem 1.6rem; margin-top: 1rem;
    box-shadow: 0 10px 30px rgba(60, 90, 40, 0.12); text-align: center;
}
.badge { display: inline-block; padding: 6px 18px; border-radius: 30px;
         font-weight: 700; font-size: 1.3rem; color: white; }
.healthy { background: linear-gradient(90deg, #43a047, #7cb342); }
.diseased { background: linear-gradient(90deg, #e53935, #fb8c00); }
.uncertain { background: linear-gradient(90deg, #8d8d8d, #b0a99f); }
.bar-bg { background: #e8ecdf; border-radius: 20px; height: 14px; margin: 14px 0 4px; overflow: hidden; }
.bar-fill { height: 100%; border-radius: 20px; }
.muted { color: #7a8570; font-size: 0.9rem; }
[data-testid="stImage"] img { border-radius: 18px; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading model...")
def get_model():
    return load_model(MODEL_PATH, compile=False)


def predict(img: Image.Image):
    img = ImageOps.exif_transpose(img).convert("RGB").resize(IMG_SIZE)
    x = preprocess_input(np.array(img, dtype="float32"))[None, ...]
    p_healthy = float(get_model().predict(x, verbose=0)[0][0])
    label = CLASS_NAMES[int(p_healthy > 0.5)]
    conf = p_healthy if p_healthy > 0.5 else 1 - p_healthy
    return label, conf


def show_result(img: Image.Image):
    with st.spinner("Analysing leaf..."):
        label, conf = predict(img)

    if conf < LOW_CONF:
        cls, text, color = "uncertain", "Not sure 🤔", "#9e9e9e"
        note = "Low confidence. Try a clearer, well-lit photo of a single corn leaf."
    elif label == "healthy":
        cls, text, color = "healthy", "Healthy 🌱", "#558b2f"
        note = "No signs of disease detected."
    else:
        cls, text, color = "diseased", "Diseased 🍂", "#e53935"
        note = "Signs of leaf disease detected (dataset: Northern Leaf Blight)."

    col1, col2 = st.columns(2)
    with col1:
        st.image(img, use_container_width=True)
    with col2:
        st.markdown(f"""
        <div class="card">
            <span class="badge {cls}">{text}</span>
            <div class="bar-bg"><div class="bar-fill" style="width:{conf*100:.0f}%; background:{color};"></div></div>
            <div class="muted">Confidence: <b>{conf:.1%}</b></div>
            <p class="muted" style="margin-top:10px;">{note}</p>
        </div>
        """, unsafe_allow_html=True)


st.markdown("""
<div class="hero">
    <h1>🌽 Corn Leaf Doctor</h1>
    <p>Upload a leaf photo or snap one with your camera to check its health.</p>
</div>
""", unsafe_allow_html=True)

mode = st.radio(
    "Input method",
    ["📁 Upload photo", "📷 Webcam"],
    horizontal=True,
    label_visibility="collapsed",
)

if mode == "📁 Upload photo":
    file = st.file_uploader("Choose a corn leaf image", type=["jpg", "jpeg", "png"],
                            label_visibility="collapsed")
    if file:
        show_result(Image.open(file))

else:
    st.caption("Allow camera access when your browser asks. Works on localhost or HTTPS.")
    shot = st.camera_input("Take a photo of the leaf", label_visibility="collapsed")
    if shot:
        show_result(Image.open(shot))
    else:
        st.info("Camera not showing? Check the browser's camera permission (lock icon in the "
                "address bar) and refresh the page.")

st.markdown("<p class='muted' style='text-align:center;margin-top:2rem;'>Trained for corn leaves only. "
            "Results are a guide, not a substitute for an agronomist.</p>", unsafe_allow_html=True)
