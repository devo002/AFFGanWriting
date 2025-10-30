# emuru_infer.py
import torch
from PIL import Image
from torchvision.transforms import functional as F
from transformers import AutoModel

_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_MODEL = None

def _prep_style_image(pil: Image.Image):
    # EmuRU expects RGB, fixed height=64, normalized to [-1,1] with mean=0.5/std=0.5
    img = pil.convert("RGB")
    h = 64
    w = img.width * h // img.height
    img = img.resize((w, h))
    t = F.to_tensor(img)               # (C,H,W) in [0,1]
    t = F.normalize(t, [0.5,0.5,0.5], [0.5,0.5,0.5])  # -> [-1,1]
    return t

def load_emuru():
    global _MODEL
    if _MODEL is None:
        _MODEL = AutoModel.from_pretrained(
            "blowing-up-groundhogs/emuru", trust_remote_code=True
        ).to(_DEVICE).eval()
    return _MODEL, _DEVICE

@torch.inference_mode()
def generate_emuru(style_img_pil: Image.Image, style_text: str, gen_text: str, max_tokens=64):
    model, device = load_emuru()
    style_img = _prep_style_image(style_img_pil).unsqueeze(0).to(device)  # (1,C,H,W)
    out_pil = model.generate(
        style_text=style_text,
        gen_text=gen_text,
        style_img=style_img,
        max_new_tokens=max_tokens
    )
    return out_pil
