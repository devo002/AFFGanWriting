import os
import re
import torch
import numpy as np
from datetime import datetime
from modules_tro import normalize
from load_data import loadData as load_data_func
from load_data import NUM_WRITERS
from network_tro import ConTranModel
from helpers import (
    generate_from_words_like_writertest,
    trocr_predict_best_polarity,
    TARGET_WORDS,
)
from trocr_teacher import TrocrTeacher
import cv2
from PIL import Image


# ---------------- Utility: normalize & save images ----------------
def _san(s: str) -> str:
    """Sanitize text for filenames"""
    return re.sub(r"[^a-zA-Z0-9_\-]+", "_", s)[:60]


# ✅ NEW: Cleaning function for predictions
def clean_pred(s: str) -> str:
    """Clean text predictions: remove trailing punctuation and fix spaces."""
    if not isinstance(s, str):
        return s
    s = re.sub(r"\s+", " ", s).strip()           # collapse multiple spaces
    s = re.sub(r"\s+([.,!?;:])", r"\1", s)       # remove spaces before punctuation
    s = re.sub(r"[.,!?;:]+$", "", s)             # remove trailing punctuation
    return s


def save_teacher_samples(
    xg, trocr_texts, target_texts, out_dir, epoch=0, step=0, max_n=11, invert=True
):
    """
    Save generated handwriting images as PNGs.
    Args:
        xg: [B,1,H,W] in [-1,1]
        trocr_texts: list[str]  (predicted text from TrOCR)
        target_texts: list[str] (ground-truth / prompted words)
    """
    os.makedirs(out_dir, exist_ok=True)
    n = min(xg.size(0), len(trocr_texts), len(target_texts), max_n)
    for i in range(n):
        arr = xg[i, 0].detach().cpu().numpy()
        arr = normalize(arr)  # convert [-1,1] to [0,255]
        if invert:
            arr = 255 - arr
        arr = np.clip(arr, 0, 255).astype("uint8")

        # ✅ clean predictions before saving
        tgt = _san(clean_pred(target_texts[i]))
        pred = _san(clean_pred(trocr_texts[i]))

        fname = f"ep{epoch:04d}_st{step:04d}_{i:02d}__tgt_{tgt}__pred_{pred}.png"
        cv2.imwrite(os.path.join(out_dir, fname), arr)


# ---------------- Main Inference Function ----------------
@torch.no_grad()
def infer_words_with_weights(
    weights_path: str,
    out_dir: str,
    trocr_repo_or_path: str = "/home/woody/iwi5/iwi5333h/model/trocr-base-handwritten",
    max_words: int | None = None,
    use_rec_filter: bool = True,
    max_edit: int = 100,
    invert_for_png: bool = True,
):
    """
    1. Load ConTran model from given weights
    2. Generate handwriting images for TARGET_WORDS
    3. Save generated images in 'out_dir'
    4. Use TrOCR to predict text from generated images
    5. Save results as CSV

    Returns: Path to results.csv
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(out_dir, exist_ok=True)

    # ---- Load model ----
    print(f"[Load model] {weights_path}")
    model = ConTranModel(NUM_WRITERS, show_iter_num=500, oov=True).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    # ---- Load test data to get style images ----
    _, test_loader = load_data_func(True)
    test_loader = torch.utils.data.DataLoader(
        test_loader,
        batch_size=8,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        collate_fn=lambda x: (
            np.array([a[0] for a in x]),
            torch.tensor([a[1] for a in x]),
            np.array([a[2] for a in x]),
            torch.tensor(np.array([a[3] for a in x]), dtype=torch.float32),
            torch.tensor(np.array([a[4] for a in x]), dtype=torch.int64),
            torch.tensor(np.array([a[5] for a in x]), dtype=torch.int64),
            torch.tensor(np.array([a[6] for a in x]), dtype=torch.float32),
            torch.tensor(np.array([a[7] for a in x]), dtype=torch.int64),
            torch.tensor(np.array([a[8] for a in x]), dtype=torch.int64),
        ),
    )

    try:
        batch = next(iter(test_loader))
    except StopIteration:
        raise RuntimeError("No data found in test_loader!")

    style_imgs = batch[3].to(device)  # [B,1,H,W] in [-1,1]

    # ---- Select words to generate ----
    words = TARGET_WORDS[:]
    if max_words is not None:
        words = words[:max_words]
    if len(words) == 0:
        raise ValueError("TARGET_WORDS list is empty in helpers.py")

    if style_imgs.size(0) < len(words):
        reps = (len(words) + style_imgs.size(0) - 1) // style_imgs.size(0)
        style_imgs = style_imgs.repeat(reps, 1, 1, 1)[:len(words)]

    # ---- Generate images ----
    print(f"[Generate] Generating {len(words)} samples...")
    xg, words_used, label_ids_wt, img_widths_wt = generate_from_words_like_writertest(
        model,
        style_imgs,
        words,
        use_rec_filter=use_rec_filter,
        max_edit=max_edit,
    )
    if xg is None:
        raise RuntimeError("No images were generated (filtered or invalid).")

    # ---- TrOCR inference ----
    print("[TrOCR] Running OCR on generated images...")
    trocr_teacher = TrocrTeacher(trocr_repo_or_path, device=device)
    texts, conf, _ = trocr_predict_best_polarity(trocr_teacher, xg)

    # ✅ Clean TrOCR predictions
    texts = [clean_pred(t) for t in texts]

    # ---- Save generated images ----
    save_teacher_samples(
        xg=xg,
        trocr_texts=texts,
        target_texts=words_used,
        out_dir=out_dir,
        epoch=0,
        step=0,
        max_n=9999,
        invert=invert_for_png,
    )

    # ---- Save CSV results ----
    import csv
    csv_path = os.path.join(out_dir, "results.csv")
    rows = []
    for i, (tgt, pred, c) in enumerate(zip(words_used, texts, conf)):
        pred_clean = clean_pred(pred)
        fname = f"ep0000_st0000_{i:02d}__tgt_{_san(tgt)}__pred_{_san(pred_clean)}.png"
        rows.append({
            "target": tgt,
            "pred": pred_clean,
            "conf": float(c),
            "filename": fname,
        })

    with open(csv_path, "w", newline="") as f:
        writer_csv = csv.DictWriter(f, fieldnames=["target", "pred", "conf", "filename"])
        writer_csv.writeheader()
        writer_csv.writerows(rows)

    print(f"\n[Done] {len(rows)} images saved in: {out_dir}")
    print(f"Results CSV: {csv_path}")

    for r in rows:
        print(f"tgt='{r['target']}'  pred='{r['pred']}'  conf={r['conf']:.3f}")

    return csv_path


# ---------------- Example Run ----------------
if __name__ == "__main__":
    weights = "/home/vault/iwi5/iwi5333h/bestmodel/contran-5000.model"
    out_dir = "/home/woody/iwi5/iwi5333h/infer2"

    infer_words_with_weights(
        weights_path=weights,
        out_dir=out_dir,
        trocr_repo_or_path="/home/woody/iwi5/iwi5333h/model/trocr-base-handwritten",
        max_words=150,
        use_rec_filter=True,
        max_edit=100,
        invert_for_png=True,
    )



# import os
# import re
# import torch
# import numpy as np
# from datetime import datetime
# from modules_tro import normalize
# from load_data import loadData as load_data_func
# from load_data import NUM_WRITERS
# from network_tro import ConTranModel
# from helpers import (
#     generate_from_words_like_writertest,
#     trocr_predict_best_polarity,
#     TARGET_WORDS,
# )
# from trocr_teacher import TrocrTeacher
# import cv2
# from PIL import Image

# # ---------------- Utility: normalize & save images ----------------
# def _san(s: str) -> str:
#     """Sanitize text for filenames"""
#     return re.sub(r"[^a-zA-Z0-9_\-]+", "_", s)[:60]

# def save_teacher_samples(
#     xg, trocr_texts, target_texts, out_dir, epoch=0, step=0, max_n=11, invert=True
# ):
#     """
#     Save generated handwriting images as PNGs.
#     Args:
#         xg: [B,1,H,W] in [-1,1]
#         trocr_texts: list[str]  (predicted text from TrOCR)
#         target_texts: list[str] (ground-truth / prompted words)
#     """
#     os.makedirs(out_dir, exist_ok=True)
#     n = min(xg.size(0), len(trocr_texts), len(target_texts), max_n)
#     for i in range(n):
#         arr = xg[i, 0].detach().cpu().numpy()
#         arr = normalize(arr)  # convert [-1,1] to [0,255]
#         if invert:
#             arr = 255 - arr
#         arr = np.clip(arr, 0, 255).astype("uint8")

#         tgt = _san(target_texts[i])
#         pred = _san(trocr_texts[i])
#         fname = f"ep{epoch:04d}_st{step:04d}_{i:02d}__tgt_{tgt}__pred_{pred}.png"
#         cv2.imwrite(os.path.join(out_dir, fname), arr)


# # ---------------- Main Inference Function ----------------
# @torch.no_grad()
# def infer_words_with_weights(
#     weights_path: str,
#     out_dir: str,
#     trocr_repo_or_path: str = "/home/woody/iwi5/iwi5333h/model/trocr-base-handwritten",
#     max_words: int | None = None,
#     use_rec_filter: bool = True,
#     max_edit: int = 100,
#     invert_for_png: bool = True,
#     # --- NEW: beam search size for TrOCR decoding
#     beam_size: int = 1,
# ):
#     """
#     1. Load ConTran model from given weights
#     2. Generate handwriting images for TARGET_WORDS
#     3. Save generated images in 'out_dir'
#     4. Use TrOCR to predict text from generated images
#     5. Save results as CSV

#     Returns: Path to results.csv
#     """
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     os.makedirs(out_dir, exist_ok=True)

#     # ---- Load model ----
#     print(f"[Load model] {weights_path}")
#     # NOTE: If your ConTranModel expects OOV (capital), flip the arg name accordingly.
#     model = ConTranModel(NUM_WRITERS, show_iter_num=500, oov=True).to(device)
#     model.load_state_dict(torch.load(weights_path, map_location=device))
#     model.eval()

#     # ---- Load test data to get style images ----
#     _, test_loader = load_data_func(True)
#     test_loader = torch.utils.data.DataLoader(
#         test_loader,
#         batch_size=8,
#         shuffle=False,
#         num_workers=2,
#         pin_memory=True,
#         collate_fn=lambda x: (
#             np.array([a[0] for a in x]),
#             torch.tensor([a[1] for a in x]),
#             np.array([a[2] for a in x]),
#             torch.tensor(np.array([a[3] for a in x]), dtype=torch.float32),
#             torch.tensor(np.array([a[4] for a in x]), dtype=torch.int64),
#             torch.tensor(np.array([a[5] for a in x]), dtype=torch.int64),
#             torch.tensor(np.array([a[6] for a in x]), dtype=torch.float32),
#             torch.tensor(np.array([a[7] for a in x]), dtype=torch.int64),
#             torch.tensor(np.array([a[8] for a in x]), dtype=torch.int64),
#         ),
#     )

#     try:
#         batch = next(iter(test_loader))
#     except StopIteration:
#         raise RuntimeError("No data found in test_loader!")

#     style_imgs = batch[3].to(device)  # [B,1,H,W] in [-1,1]

#     # ---- Select words to generate ----
#     words = TARGET_WORDS[:]
#     if max_words is not None:
#         words = words[:max_words]
#     if len(words) == 0:
#         raise ValueError("TARGET_WORDS list is empty in helpers.py")

#     if style_imgs.size(0) < len(words):
#         reps = (len(words) + style_imgs.size(0) - 1) // style_imgs.size(0)
#         style_imgs = style_imgs.repeat(reps, 1, 1, 1)[:len(words)]

#     # ---- Generate images ----
#     print(f"[Generate] Generating {len(words)} samples...")
#     xg, words_used, label_ids_wt, img_widths_wt = generate_from_words_like_writertest(
#         model,
#         style_imgs,
#         words,
#         use_rec_filter=use_rec_filter,
#         max_edit=max_edit,
#     )
#     if xg is None:
#         raise RuntimeError("No images were generated (filtered or invalid).")

#     # ---- TrOCR inference (with beam search) ----
#     print(f"[TrOCR] Running OCR on generated images... (beam_size={beam_size})")
#     trocr_teacher = TrocrTeacher(trocr_repo_or_path, device=device)
#     # IMPORTANT: this assumes your helpers.trocr_predict_best_polarity forwards **num_beams**
#     # to TrocrTeacher.predict(...). If not, add that parameter in your helper/teacher.
#     texts, conf, _ = trocr_predict_best_polarity(
#         trocr_teacher,
#         xg,
#         num_beams=beam_size,   # <-- NEW
#         max_length=48
#     )

#     # ---- Save generated images ----
#     save_teacher_samples(
#         xg=xg,
#         trocr_texts=texts,
#         target_texts=words_used,
#         out_dir=out_dir,
#         epoch=0,
#         step=0,
#         max_n=9999,
#         invert=invert_for_png,
#     )

#     # ---- Save CSV results ----
#     import csv
#     csv_path = os.path.join(out_dir, "results.csv")
#     rows = []
#     for i, (tgt, pred, c) in enumerate(zip(words_used, texts, conf)):
#         fname = f"ep0000_st0000_{i:02d}__tgt_{_san(tgt)}__pred_{_san(pred)}.png"
#         rows.append({
#             "target": tgt,
#             "pred": pred,
#             "conf": float(c),
#             "filename": fname,
#         })

#     with open(csv_path, "w", newline="") as f:
#         writer_csv = csv.DictWriter(f, fieldnames=["target", "pred", "conf", "filename"])
#         writer_csv.writeheader()
#         writer_csv.writerows(rows)

#     print(f"\n[Done] {len(rows)} images saved in: {out_dir}")
#     print(f"Results CSV: {csv_path}")

#     for r in rows:
#         print(f"tgt='{r['target']}'  pred='{r['pred']}'  conf={r['conf']:.3f}")

#     return csv_path


# # ---------------- Example Run ----------------
# if __name__ == "__main__":
#     weights = "/home/vault/iwi5/iwi5333h/bestmodel/contran-5000.model"
#     out_dir = "/home/woody/iwi5/iwi5333h/infer2"

#     infer_words_with_weights(
#         weights_path=weights,
#         out_dir=out_dir,
#         trocr_repo_or_path="/home/woody/iwi5/iwi5333h/model/trocr-base-handwritten",
#         max_words=120,
#         use_rec_filter=True,
#         max_edit=100,
#         invert_for_png=True,
#         # --- NEW: try beam search
#         beam_size=5,
#     )
