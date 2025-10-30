# app.py — Streamlit UI to generate handwriting from a chosen style
# - Backend switch: GAN (PKL styles or extracted style) OR EmuRU (HF)
# - GAN: pick a style from PKL OR extract from a reference line image
# - EmuRU: upload/path a reference line image + provide its exact transcription
# - EmuRU loads strictly from local cache (no downloads)

import io
import json
import pickle
import inspect
from importlib import import_module
from collections import defaultdict
import os
import unicodedata, re

# ==== POINT HF CACHE TO YOUR LOCAL COPY (parent folder of models--... and blobs/) ====
os.environ["HUGGINGFACE_HUB_CACHE"] = "/home/woody/iwi5/iwi5333h/.cache/huggingface/hub"
os.environ["HF_HUB_OFFLINE"] = "1"  # hard offline


import numpy as np
import torch
import cv2
import streamlit as st
from PIL import Image
import torchvision.transforms.functional as VF
# HF card code sometimes references deprecated name; add a safe alias.
try:
    import huggingface_hub as _hfhub
    if not hasattr(_hfhub, "cached_download") and hasattr(_hfhub, "hf_hub_download"):
        _hfhub.cached_download = _hfhub.hf_hub_download
except Exception:
    pass

# ---- Backward-compatible cache shims (Streamlit old/new) ----
cache_resource = getattr(st, "cache_resource", getattr(st, "experimental_singleton"))
cache_data     = getattr(st, "cache_data",     getattr(st, "experimental_memo"))

# ======================== CONFIGURE THESE PATHS ========================
CKPT = "/home/woody/iwi5/iwi5333h/handwriting_line_generation/saved2/IAMslant_noMask_charSpecSingleAppend_GANMedMT_autoAEMoPrcp2tightNewCTCUseGen_balB_hCF0.75_sMG/checkpoint-latest.pth"
STYLE_PKL = "/home/woody/iwi5/iwi5333h/handwriting_line_generation/styless/val_styles_175000.pkl"   # or train_* pkl
CHARSET_JSON = "/home/woody/iwi5/iwi5333h/handwriting_line_generation/data/IAM_char_set.json"
# ======================================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

# ============================ YOUR REPO IMPORTS ========================
from utils import string_utils
from model import *  # your repo's arch via eval(config['arch'])

# ============================ GAN LOADERS ===============================
def _safe_torch_load(path):
    """
    Robust torch.load for PyTorch >= 2.6:
    - Allowlist custom logger.Logger if present.
    - Try weights_only=True first (safer), fall back to False (trusted file only).
    """
    try:
        try:
            mod = import_module("logger.logger")
            Logger = getattr(mod, "Logger", None)
            if Logger is not None:
                from torch.serialization import add_safe_globals
                add_safe_globals([Logger])
        except Exception:
            pass

        if "weights_only" in inspect.signature(torch.load).parameters:
            return torch.load(path, map_location="cpu", weights_only=True)
        return torch.load(path, map_location="cpu")
    except Exception:
        if "weights_only" in inspect.signature(torch.load).parameters:
            return torch.load(path, map_location="cpu", weights_only=False)
        return torch.load(path, map_location="cpu")

@cache_resource(show_spinner=False)
def load_model_and_charset():
    ckpt = _safe_torch_load(CKPT)
    config = ckpt.get("config")
    assert config is not None, "Checkpoint missing 'config'."

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
    idx_to_author = blob.get('idx_to_author', None)
    return authors, by_author, counts, idx_to_author

# ========================= GAN HELPERS =================================
def npstyle_to_tensor(style_np, device):
    if isinstance(style_np, (tuple, list)):
        s0 = torch.from_numpy(style_np[0])[None, ...].to(device)
        s1 = torch.from_numpy(style_np[1])[None, ...].to(device)
        s2 = torch.from_numpy(style_np[2])[None, ...].to(device)
        return (s0, s1, s2)
    else:
        return torch.from_numpy(style_np).to(device)[None, ...]

def generate_line(model, text, char_to_idx, style_tensor, device):
    label = string_utils.str2label_single(text, char_to_idx)
    label = torch.from_numpy(label.astype(np.int32))[:, None].expand(-1, 1).to(device).long()
    label_len = torch.IntTensor(1).fill_(label.size(0)).to(device)
    with torch.no_grad():
        im = model(label, label_len, style_tensor)[0]  # (C,H,W)
    im = ((1 - im.permute(1, 2, 0)) * 127.5).cpu().numpy().astype(np.uint8)
    return im

def _prep_line_image(gray, target_h=None):
    if target_h and gray.shape[0] != target_h:
        scale = float(target_h) / float(gray.shape[0])
        gray = cv2.resize(gray, (int(gray.shape[1] * scale), target_h), interpolation=cv2.INTER_CUBIC)
    im = gray.astype(np.float32)
    im = 1.0 - im / 128.0
    im = im[None, ...]
    return im

def load_line_from_upload_bytes(file_bytes, target_h=None):
    arr = np.frombuffer(file_bytes, np.uint8)
    gray = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise ValueError("Could not decode uploaded image.")
    return _prep_line_image(gray, target_h)

def load_line_from_path(path, target_h=None):
    gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise FileNotFoundError(f"Could not read {path}")
    return _prep_line_image(gray, target_h)

def extract_style_from_image(model, im_chw, ref_text, char_to_idx, device):
    img_t = torch.from_numpy(im_chw).to(device)[None, ...]
    if ref_text and len(ref_text) > 0:
        lab_np = string_utils.str2label_single(ref_text, char_to_idx)
        lab_t = torch.from_numpy(lab_np.astype(np.int32))[:, None].to(device).long()
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

# ========================= EmuRU BACKEND ===============================
EMURU_AVAILABLE = True
_emuru_import_error = ""
try:
    from torchvision.transforms import functional as VF
    from transformers import AutoModel
    import diffusers, einops, accelerate  # noqa: F401
except Exception as e:
    EMURU_AVAILABLE = False
    _emuru_import_error = str(e)

def _find_local_emuru_snapshot(cache_root: str) -> str:
    """Return a valid snapshot dir that contains config.json and model.safetensors."""
    base = os.path.join(cache_root, "models--blowing-up-groundhogs--emuru", "snapshots")
    if not os.path.isdir(base):
        raise FileNotFoundError(f"No snapshots dir at: {base}")
    # collect candidate subdirs
    candidates = []
    for name in os.listdir(base):
        p = os.path.join(base, name)
        if os.path.isdir(p):
            cfg = os.path.join(p, "config.json")
            safes = os.path.join(p, "model.safetensors")
            if os.path.isfile(cfg) and os.path.isfile(safes):
                candidates.append((os.path.getmtime(p), p))
    if not candidates:
        raise FileNotFoundError("No complete EmuRU snapshot found (config.json + model.safetensors). "
                                "Make sure you copied both 'models--...' and 'blobs/' from your local PC.")
    # pick most recent
    candidates.sort(reverse=True)
    return candidates[0][1]

@cache_resource(show_spinner=False)
def load_emuru():
    from transformers import AutoModel
    cache_root = os.environ.get("HUGGINGFACE_HUB_CACHE", os.path.expanduser("~/.cache/huggingface/hub"))
    local_snapshot = _find_local_emuru_snapshot(cache_root)
    # Fail fast if anything is missing, and stay offline.
    model = AutoModel.from_pretrained(
        local_snapshot,
        trust_remote_code=True,
        local_files_only=True
    ).to(DEVICE).eval()
    return model

def _prep_style_image_emuru(pil: Image.Image):
    img = pil.convert("RGB")
    h = 64
    w = max(1, img.width * h // img.height)
    img = img.resize((w, h))
    t = VF.to_tensor(img)
    t = VF.normalize(t, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    return t

def _normalize_for_emuru(text: str) -> str:
    s = unicodedata.normalize("NFKC", text)           # unify punctuation/width
    s = s.replace("’","'").replace("‘","'")           # smart → straight quotes
    s = s.replace("“",'"').replace("”",'"')
    s = re.sub(r"\s+"," ", s).strip()                 # collapse spaces
    return s

# @torch.inference_mode()
# def generate_emuru(style_img_pil: Image.Image, style_text: str, gen_text: str, max_tokens=64):
#     model = load_emuru()
#     style_img = _prep_style_image_emuru(style_img_pil).unsqueeze(0).to(DEVICE)
#     out_pil = model.generate(
#         style_text=style_text,
#         gen_text=gen_text,
#         style_img=style_img,
#         max_new_tokens=max_tokens
#     )
#     return out_pil

@torch.inference_mode()
def generate_emuru(style_img_pil: Image.Image, style_text: str, gen_text: str, max_tokens=64):
    model = load_emuru()
    style_img = _prep_style_image_emuru(style_img_pil).unsqueeze(0).to(DEVICE)

    # >>> new: normalize inputs
    style_text = _normalize_for_emuru(style_text)
    gen_text   = _normalize_for_emuru(gen_text)
    # <<<
    max_tokens = max(max_tokens, int(len(gen_text) * 3))
    
    out_pil = model.generate(
        style_text=style_text,
        gen_text=gen_text,
        style_img=style_img,
        max_new_tokens=max_tokens
    )
    target_h = 96  # you can tune between 80–112 for best match
    if out_pil.height != target_h:
        new_w = int(out_pil.width * (target_h / out_pil.height))
        out_pil = out_pil.resize((new_w, target_h), Image.BICUBIC)
    return out_pil


# =============================== UI ====================================
st.set_page_config(page_title="Handwriting Generator", page_icon="✍️", layout="centered")
st.title("✍️ Generate handwriting from a specific style")

# Backend switch
st.subheader("")
backend = st.radio("Choose generator", ["GAN (PKL styles)", "EmuRU (HF)"], horizontal=True)

if backend == "EmuRU (HF)" and not EMURU_AVAILABLE:
    st.error(
        "EmuRU dependencies not available. Please install:\n\n"
        "`pip install -U diffusers einops accelerate safetensors transformers huggingface_hub pillow torchvision`\n\n"
        f"Import error: {_emuru_import_error}"
    )

# Keep state across interactions
if "active_style" not in st.session_state:
    st.session_state.active_style = None
if "style_caption" not in st.session_state:
    st.session_state.style_caption = ""
if "pil_for_emuru" not in st.session_state:
    st.session_state.pil_for_emuru = None

# ------------------------------- Style selection & model loading ------------------------

# Only load GAN model/charset if that backend is selected
model = None
char_to_idx = None
if backend == "GAN (PKL styles)":
    try:
        model, char_to_idx = load_model_and_charset()
    except Exception as e:
        st.error(f"Failed to load GAN checkpoint: {e}")

# Only load style PKL for GAN backend
if backend == "GAN (PKL styles)" and model is not None:
    authors, by_author, counts, idx_to_author = load_styles()

# --------------------------------- UI per backend ---------------------------------------

if backend == "GAN (PKL styles)":
    st.subheader("Choose how to get the style")
    mode = st.radio("Style source", ["Library (author/index)", "Reference image"], horizontal=True)

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
        ref_text = st.text_input("Reference transcription (optional)", value="")

        if st.button("Extract style from image"):
            try:
                pil_for_emuru = None
                im_chw = None

                if up is not None:
                    file_bytes = up.read()
                    if not file_bytes:
                        raise ValueError("Empty upload.")
                    pil_for_emuru = Image.open(io.BytesIO(file_bytes)).convert("RGB")
                    im_chw = load_line_from_upload_bytes(file_bytes, target_h)
                elif path.strip():
                    pil_for_emuru = Image.open(path.strip()).convert("RGB")
                    im_chw = load_line_from_path(path.strip(), target_h)
                else:
                    st.error("Please upload a file or enter a local path.")
                    raise RuntimeError("No image provided.")

                style = extract_style_from_image(model, im_chw, ref_text, char_to_idx, DEVICE)
                if isinstance(style, (tuple, list)):
                    st.session_state.active_style = tuple(s.to(DEVICE) for s in style)
                else:
                    st.session_state.active_style = style.to(DEVICE)

                st.session_state.style_caption = "Style extracted from reference image (GAN)"
                st.session_state.pil_for_emuru = pil_for_emuru  # useful if user switches to EmuRU later
                st.success("Style extracted.")
            except Exception as e:
                st.error(f"Extraction failed: {e}")

else:
    st.subheader("EmuRU requires a reference line image & transcription")
    up_e = st.file_uploader("Upload PNG/JPG (EmuRU)", type=["png", "jpg", "jpeg"], key="emuru_uploader")
    path_e = st.text_input("…or local path to an image (EmuRU)", value="", key="emuru_path")
    style_text_emuru = st.text_input("Exact text in the reference image (required)", value="", key="emuru_style_text")
    

    if st.button("Load reference for EmuRU"):
        try:
            if up_e is not None:
                file_bytes = up_e.read()
                if not file_bytes:
                    raise ValueError("Empty upload.")
                st.session_state.pil_for_emuru = Image.open(io.BytesIO(file_bytes)).convert("RGB")
                st.success("Reference loaded for EmuRU.")
            elif path_e.strip():
                st.session_state.pil_for_emuru = Image.open(path_e.strip()).convert("RGB")
                st.success("Reference loaded for EmuRU.")
            else:
                st.error("Please upload a file or enter a local path.")
        except Exception as e:
            st.error(f"Failed to load reference: {e}")

# ------------------------------- Generate -------------------------------
text = st.text_input("Text to render", value="God is the greatest of all time.")
gan_out_h = st.number_input("Output height", 32, 256, 96, 1)
if st.button("Generate"):
    if backend == "GAN (PKL styles)":
        if (model is None) or (char_to_idx is None):
            st.error("GAN model not loaded.")
        elif st.session_state.active_style is None:
            st.error("Please choose a library style or extract from a reference image first (GAN).")
        else:
            try:
                img = generate_line(model, text, char_to_idx, st.session_state.active_style, DEVICE)
                if img.shape[0] != gan_out_h:
                    scale = gan_out_h / img.shape[0]
                    img = cv2.resize(img, (int(img.shape[1] * scale), gan_out_h), interpolation=cv2.INTER_CUBIC)
                    
                st.image(img, caption=st.session_state.style_caption or "Generated (GAN)", clamp=True)
                ok, png = cv2.imencode(".png", img)
                if ok:
                    st.download_button("Download PNG", png.tobytes(), file_name="generated_gan.png", mime="image/png")
            except Exception as e:
                st.error(f"GAN generation failed: {e}")

    else:
        if not EMURU_AVAILABLE:
            st.error(
                "EmuRU not available — install required packages:\n"
                "`pip install -U diffusers einops accelerate safetensors transformers huggingface_hub pillow torchvision`"
            )
        elif st.session_state.pil_for_emuru is None:
            st.error("Load a reference image for EmuRU (see above).")
        else:
            style_text_val = st.session_state.get("emuru_style_text", "")
            if not style_text_val:
                st.error("Enter the exact transcription of the reference image for EmuRU.")
            else:
                try:
                    with st.spinner("EmuRU generating…"):
                        out_pil = generate_emuru(st.session_state.pil_for_emuru, style_text_val, text, max_tokens=64)
                    st.image(out_pil, caption="EmuRU output", use_column_width=True)
                    buf = io.BytesIO()
                    out_pil.save(buf, format="PNG")
                    st.download_button("Download PNG", buf.getvalue(), file_name="generated_emuru.png", mime="image/png")
                except Exception as e:
                    st.error(f"EmuRU generation failed: {e}")
