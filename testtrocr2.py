import os
import re
import csv
import glob
import math
import torch
import numpy as np
import cv2
from typing import List, Tuple

from trocr_teacher import TrocrTeacher
from helpers import trocr_predict_best_polarity

# ---------- small utils ----------
def _clean_pred(s: str) -> str:
    if not isinstance(s, str):
        return s
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"\s+([.,!?;:])", r"\1", s)
    s = re.sub(r"[.,!?;:]+$", "", s)
    return s

def _read_image_gray_uint8(path: str) -> np.ndarray:
    """Read image as grayscale uint8 [H,W] in [0,255]."""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError(f"Failed to read image: {path}")
    return img

def _to_minus1_1(img_u8: np.ndarray) -> np.ndarray:
    """Convert uint8 [0,255] -> float32 [-1,1]."""
    return (img_u8.astype(np.float32) / 127.5) - 1.0

def _batchify(items: List[str], batch_size: int) -> List[List[str]]:
    return [items[i:i + batch_size] for i in range(0, len(items), batch_size)]

# ---------- main ----------
@torch.no_grad()
def trocr_predict_folder(
    image_dir: str,
    output_csv: str,
    trocr_repo_or_path: str = "/home/woody/iwi5/iwi5333h/model/trocr-base-handwritten",
    batch_size: int = 16,
    max_length: int = 48,
    num_beams: int = 1,
    device_str: str | None = None,
) -> str:
    """
    Run TrOCR on all images in `image_dir` and write predictions to CSV.

    Columns: filename, text, confidence

    Assumes you have:
      - TrocrTeacher
      - helpers.trocr_predict_best_polarity(teacher, xg, ...)
    where xg is a tensor [B,1,H,W] in [-1,1].
    """
    # collect images
    exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff", "*.gif")
    paths = []
    for e in exts:
        paths.extend(glob.glob(os.path.join(image_dir, e)))
    paths = sorted(paths)

    if not paths:
        raise FileNotFoundError(f"No images found in: {image_dir}")

    # device
    if device_str is not None:
        device = torch.device(device_str)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # init teacher
    teacher = TrocrTeacher(trocr_repo_or_path, device=device)

    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    rows: List[dict] = []

    # process in batches
    for chunk in _batchify(paths, batch_size):
        # read + convert
        imgs_u8: List[np.ndarray] = []
        for p in chunk:
            try:
                imgs_u8.append(_read_image_gray_uint8(p))
            except Exception as e:
                print(f"[WARN] Skipping {p}: {e}")

        if not imgs_u8:
            continue

        # convert to [-1,1], stack to [B,1,H,W]
        # variable sizes are okay if your OCR handles dynamic resizing internally
        # (helpers.trocr_predict_best_polarity typically handles preprocessing)
        tensors = []
        for img in imgs_u8:
            x = _to_minus1_1(img)          # [H,W] float32 [-1,1]
            x = np.expand_dims(x, axis=0)  # [1,H,W]
            tensors.append(x)

        xg = torch.from_numpy(np.stack(tensors, axis=0)).float().to(device)  # [B,1,H,W]

        # predict with polarity search
        texts, confs, _ = trocr_predict_best_polarity(
            teacher,
            xg,
        )

        # clean + record
        for path_img, text, conf in zip(chunk, texts, confs):
            rows.append({
                "filename": os.path.basename(path_img),
                "text": _clean_pred(text),
                "confidence": float(conf),
            })

    # write CSV
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["filename", "text", "confidence"])
        w.writeheader()
        w.writerows(rows)

    print(f"[DONE] Wrote {len(rows)} predictions to: {output_csv}")
    return output_csv


# ---------- CLI example ----------
if __name__ == "__main__":
    # change these to your paths
    IMAGE_DIR = "/home/woody/iwi5/iwi5333h/trocrimages"
    OUTPUT_CSV = "//home/woody/iwi5/iwi5333h/trocrimages/predictions.csv"

    trocr_predict_folder(
        image_dir=IMAGE_DIR,
        output_csv=OUTPUT_CSV,
        trocr_repo_or_path="/home/woody/iwi5/iwi5333h/model/trocr-base-handwritten",
        batch_size=8,
        max_length=100,
        num_beams=1,            # try 5 for stronger decoding at some speed cost
        device_str=None,        # or "cuda"/"cpu"
    )
