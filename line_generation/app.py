# app.py — Streamlit UI to generate handwriting from a chosen style
# - Pick a style from a precomputed library (PKL)
# - OR upload/provide a path to a reference line image, extract its style, and emulate it

import os
import json
import pickle
from collections import defaultdict

import numpy as np
import torch
import cv2
import streamlit as st

# ---- Backward-compatible cache shims (Streamlit old/new) ----
cache_resource = getattr(st, "cache_resource", getattr(st, "experimental_singleton"))
cache_data     = getattr(st, "cache_data",     getattr(st, "experimental_memo"))

from utils import string_utils
from model import *  # your repo's arch via eval(config['arch'])

# ======================== CONFIGURE THESE PATHS ========================
CKPT = "/home/woody/iwi5/iwi5333h/handwriting_line_generation/saved2/IAMslant_noMask_charSpecSingleAppend_GANMedMT_autoAEMoPrcp2tightNewCTCUseGen_balB_hCF0.75_sMG/checkpoint-latest.pth"
STYLE_PKL = "/home/woody/iwi5/iwi5333h/handwriting_line_generation/styless/val_styles_175000.pkl"   # or train_* pkl
CHARSET_JSON = "/home/woody/iwi5/iwi5333h/handwriting_line_generation/data/IAM_char_set.json"
# ======================================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================ LOADERS ==================================
@cache_resource(show_spinner=False)
def load_model_and_charset():
    ckpt = torch.load(CKPT, map_location="cpu")
    config = ckpt.get("config")
    assert config is not None, "Checkpoint missing 'config'."

    # Disable training-only bits
    config['model']['RUN'] = True
    config['optimizer_type'] = "none"
    config['trainer']['use_learning_schedule'] = False
    config['trainer']['swa'] = False
    config['cuda'] = (DEVICE == "cuda")
    if DEVICE == "cuda":
        config['gpu'] = 0

    model = eval(config['arch'])(config['model'])
    model.load_state_dict(ckpt['state_dict'])
    model.eval().to(DEVICE)

    with open(CHARSET_JSON) as f:
        char_to_idx = json.load(f)['char_to_idx']

    return model, char_to_idx


@cache_data(show_spinner=False)
def load_styles():
    with open(STYLE_PKL, "rb") as f:
        blob = pickle.load(f)

    by_author = defaultdict(list)
    for a, s in zip(blob['authors'], blob['styles']):
        by_author[a].append(s)

    authors = sorted(by_author.keys())
    counts = {a: len(by_author[a]) for a in authors}

    # Optional: if your PKL saved a mapping (nice labels like a01)
    idx_to_author = blob.get('idx_to_author', None)

    return authors, by_author, counts, idx_to_author


# ========================= CORE HELPERS ================================
def npstyle_to_tensor(style_np, device):
    """Convert a saved style (numpy or tuple-of-numpy) into tensors with batch dim."""
    if isinstance(style_np, (tuple, list)):
        s0 = torch.from_numpy(style_np[0])[None, ...].to(device)
        s1 = torch.from_numpy(style_np[1])[None, ...].to(device)
        s2 = torch.from_numpy(style_np[2])[None, ...].to(device)
        return (s0, s1, s2)
    else:
        return torch.from_numpy(style_np).to(device)[None, ...]


def generate_line(model, text, char_to_idx, style_tensor, device):
    """Generate a single line image for the given text and style."""
    label = string_utils.str2label_single(text, char_to_idx)
    label = torch.from_numpy(label.astype(np.int32))[:, None].expand(-1, 1).to(device).long()
    label_len = torch.IntTensor(1).fill_(label.size(0)).to(device)
    with torch.no_grad():
        im = model(label, label_len, style_tensor)[0]  # (C,H,W)
    im = ((1 - im.permute(1, 2, 0)) * 127.5).cpu().numpy().astype(np.uint8)
    return im


# ---------- reference-image helpers ----------
def _prep_line_image(gray, target_h=None):
    """Resize (optional) + normalize exactly like training (1 - img/128). Returns (1,H,W) np."""
    if target_h and gray.shape[0] != target_h:
        scale = float(target_h) / float(gray.shape[0])
        gray = cv2.resize(gray, (int(gray.shape[1] * scale), target_h), interpolation=cv2.INTER_CUBIC)
    im = gray.astype(np.float32)
    im = 1.0 - im / 128.0
    im = im[None, ...]  # (C=1,H,W)
    return im


def load_line_from_upload(uploaded_file, target_h=None):
    file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
    gray = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise ValueError("Could not decode uploaded image.")
    return _prep_line_image(gray, target_h)


def load_line_from_path(path, target_h=None):
    gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise FileNotFoundError(f"Could not read {path}")
    return _prep_line_image(gray, target_h)


def extract_style_from_image(model, im_chw, ref_text, char_to_idx, device):
    """
    Extract a style embedding from a reference image.
    If ref_text is provided, use it (better style/content disentanglement).
    """
    img_t = torch.from_numpy(im_chw).to(device)[None, ...]  # (B=1,C=1,H,W)

    if ref_text and len(ref_text) > 0:
        lab_np = string_utils.str2label_single(ref_text, char_to_idx)
        lab_t = torch.from_numpy(lab_np.astype(np.int32))[:, None].to(device).long()  # (L,B=1)
        a_bs = 1
        with torch.no_grad():
            style = model.extract_style(img_t, lab_t, a_bs)
    else:
        with torch.no_grad():
            if hasattr(model, "extract_style"):
                style = model.extract_style(img_t, None, 1)
            else:
                style = model.style_extractor(img_t)
    return style
# ---------- end helpers ----------


# =============================== UI ====================================
st.set_page_config(page_title="Handwriting Generator", page_icon="✍️", layout="centered")
st.title("✍️ Generate handwriting from a specific style")

model, char_to_idx = load_model_and_charset()
authors, by_author, counts, idx_to_author = load_styles()

st.subheader("Choose how to get the style")
mode = st.radio("Style source", ["Library (author/index)", "Reference image"], horizontal=True)

# Keep the selected/extracted style alive across interactions
if "active_style" not in st.session_state:
    st.session_state.active_style = None
if "style_caption" not in st.session_state:
    st.session_state.style_caption = ""

if mode == "Library (author/index)":
    def fmt_author(a):
        if idx_to_author is not None and int(a) in idx_to_author:
            label = idx_to_author[int(a)]
        else:
            label = str(a)
        return f"{label}  — {counts[a]} styles"

    colA, colB = st.columns([2, 1])
    with colA:
        author = st.selectbox("Author (writer id)", authors, format_func=fmt_author)
    with colB:
        max_idx = max(0, counts[author] - 1)
        style_idx = st.number_input("Style index", min_value=0, max_value=max_idx, value=0, step=1)

    if st.button("Use selected library style"):
        style_np = by_author[author][style_idx]
        st.session_state.active_style = npstyle_to_tensor(style_np, DEVICE)
        if idx_to_author is not None and int(author) in idx_to_author:
            human = idx_to_author[int(author)]
            st.session_state.style_caption = f"Author {human} • style #{style_idx}"
        else:
            st.session_state.style_caption = f"Author {author} • style #{style_idx}"
        st.success("Style loaded from library.")

else:
    st.markdown("Upload a **reference line image** to imitate its handwriting style.")
    target_h = st.number_input("Resize height for reference (px)", 16, 256, 64, 1)
    up = st.file_uploader("Upload PNG/JPG", type=["png", "jpg", "jpeg"])
    path = st.text_input("…or local path to an image", value="")
    ref_text = st.text_input("Reference transcription (recommended)", value="")

    if st.button("Extract style from image"):
        try:
            if up is not None:
                im_chw = load_line_from_upload(up, target_h)
            elif path.strip():
                im_chw = load_line_from_path(path.strip(), target_h)
            else:
                st.error("Please upload a file or enter a local path.")
                im_chw = None

            if im_chw is not None:
                style = extract_style_from_image(model, im_chw, ref_text, char_to_idx, DEVICE)
                if isinstance(style, (tuple, list)):
                    st.session_state.active_style = tuple(s.to(DEVICE) for s in style)
                else:
                    st.session_state.active_style = style.to(DEVICE)
                st.session_state.style_caption = "Style extracted from reference image"
                st.success("Style extracted.")
        except Exception as e:
            st.error(f"Extraction failed: {e}")

# -------- Generate section --------
text = st.text_input("Text to render", value="God is the greatest of all time.")
if st.button("Generate"):
    if st.session_state.active_style is None:
        st.error("Please choose a style from the library or extract from a reference image first.")
    else:
        img = generate_line(model, text, char_to_idx, st.session_state.active_style, DEVICE)
        st.image(img, caption=st.session_state.style_caption or "Generated", clamp=True)
        ok, png = cv2.imencode(".png", img)
        if ok:
            st.download_button("Download PNG", png.tobytes(), file_name="generated.png", mime="image/png")
