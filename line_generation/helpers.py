# ---------- reference-image helpers ----------
import io

def _prep_line_image(gray, target_h=None):
    if target_h and gray.shape[0] != target_h:
        scale = float(target_h) / float(gray.shape[0])
        gray = cv2.resize(gray, (int(gray.shape[1]*scale), target_h), interpolation=cv2.INTER_CUBIC)
    im = gray.astype(np.float32)
    im = 1.0 - im / 128.0          # same normalization as training
    im = im[None, ...]             # (C=1,H,W)
    return im

def load_line_from_upload(uploaded_file, target_h=None):
    file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
    gray = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    if gray is None: raise ValueError("Could not decode uploaded image.")
    return _prep_line_image(gray, target_h)

def load_line_from_path(path, target_h=None):
    gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if gray is None: raise FileNotFoundError(f"Could not read {path}")
    return _prep_line_image(gray, target_h)

def extract_style_from_image(model, im_chw, ref_text, char_to_idx, device):
    """
    im_chw: np.ndarray of shape (1,H,W) already normalized.
    ref_text: transcription of the reference line (recommended)
    """
    img_t = torch.from_numpy(im_chw).to(device)[None, ...]  # (B=1,C=1,H,W)
    if ref_text and len(ref_text) > 0:
        lab_np = string_utils.str2label_single(ref_text, char_to_idx)
        lab_t  = torch.from_numpy(lab_np.astype(np.int32))[:, None].to(device).long()  # (L,B=1)
        a_bs   = 1
        with torch.no_grad():
            style = model.extract_style(img_t, lab_t, a_bs)
    else:
        # fallback if you don't have transcription
        with torch.no_grad():
            # most repos expose either extract_style(image, None, a_bs) or style_extractor(image)
            if hasattr(model, "extract_style"):
                style = model.extract_style(img_t, None, 1)
            else:
                style = model.style_extractor(img_t)
    return style
# ---------------------------------------------
