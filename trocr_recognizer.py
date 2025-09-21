
"""
TrOCR-based recognizer that plugs into your pipeline as a drop‑in
replacement for your RecModel: it returns logits shaped
[B, T, vocab_size] so your existing loss computation keeps working.

Key ideas
- We freeze TrOCR weights (encoder+decoder) so recognition loss still
  backprops to the *image* (training your generator) but not into TrOCR.
- We project TrOCR tokenizer logits back onto *your* vocabulary using a
  robust char→token mapping that handles ints in index2letter.
- Time dimension is padded/truncated to OUTPUT_MAX_LEN for compatibility.

Defaults
- ckpt points to your local HPC path (change if you use the large model):
  /home/woody/iwi5/iwi5333h/model/trocr-base-handwritten

Requires
    pip install transformers>=4.41.0
"""
from __future__ import annotations
import os
import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
except Exception as e:
    raise ImportError("Please `pip install transformers` to use TrOCRRecModel") from e

# ---- Import your project globals (same as your existing recognizer expects) ----
try:
    # must provide: OUTPUT_MAX_LEN, IMG_HEIGHT, vocab_size, index2letter, num_tokens, tokens
    from load_data import OUTPUT_MAX_LEN, IMG_HEIGHT, vocab_size, index2letter, num_tokens, tokens
except Exception as e:
    raise RuntimeError(
        "trocr_recognizer.py expects load_data to define OUTPUT_MAX_LEN, IMG_HEIGHT, "
        "vocab_size, index2letter, num_tokens, tokens"
    ) from e


# ------------------------ image utilities ------------------------

def _ensure_rgb_and_square(x: torch.Tensor, size: int = 384) -> torch.Tensor:
    """
    x: [B, C, H, W] in [0,1] or [-1,1]
    Returns: [B, 3, size, size] by aspect-preserving resize + right/bottom pad.
    Differentiable (uses torch ops only).
    """
    if x.dtype != torch.float32:
        x = x.float()
    if torch.min(x) < 0:  # convert [-1,1] -> [0,1]
        x = (x + 1.0) * 0.5
    x = x.clamp(0, 1)

    # grayscale -> rgb
    if x.shape[1] == 1:
        x = x.repeat(1, 3, 1, 1)

    B, C, H, W = x.shape
    # scale so that both dims <= size, keep aspect
    scale = min(size / max(1, H), size / max(1, W))
    new_h = max(1, int(round(H * scale)))
    new_w = max(1, int(round(W * scale)))
    x = F.interpolate(x, size=(new_h, new_w), mode="bilinear", align_corners=False)

    # pad to square (right/bottom)
    pad_h = size - new_h
    pad_w = size - new_w
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, pad_w, 0, pad_h), mode="constant", value=0.0)  # (left,right,top,bottom)
    return x

# ------------------------ tokenizer mapping helpers ------------------------

def _to_char(sym) -> Optional[str]:
    """Convert entries from index2letter to a single-character string when possible.
    Handles str, int (Unicode codepoint), and bytes.
    Returns None if the symbol shouldn't be mapped (specials).
    """
    if isinstance(sym, str):
        return sym
    if isinstance(sym, bytes):
        try:
            return sym.decode("utf-8")
        except Exception:
            return sym.decode("latin-1", errors="ignore")
    if isinstance(sym, int):
        if 0 <= sym <= 0x10FFFF:
            try:
                return chr(sym)
            except Exception:
                return None
    return None


class TrOCRRecModel(nn.Module):
    def __init__(
        self,
        ckpt: str = "/home/woody/iwi5/iwi5333h/model/trocr-base-handwritten",
        local_only: bool = True,
    ):
        super().__init__()

        # Detect local vs hub id if user passes a repo name
        if os.path.isdir(ckpt):
            _local_only = True if local_only is None else local_only
        else:
            _local_only = False if local_only is None else local_only

        # Load processor+model
        self.processor = TrOCRProcessor.from_pretrained(ckpt, local_files_only=_local_only)
        self.model = VisionEncoderDecoderModel.from_pretrained(ckpt, local_files_only=_local_only)

        # Freeze TrOCR params
        
        
        #for p in self.model.parameters():
        #    p.requires_grad = False
            
            
        # Freeze the encoder (vision part)
        #for name, p in self.model.encoder.named_parameters():
        #    p.requires_grad = False
        # UNfreeze the decoder (text generation part)
        # for name, p in self.model.decoder.named_parameters():
            #p.requires_grad = True

        # Avoid internal resizing; we normalize only
        if hasattr(self.processor, "image_processor"):
            self.processor.image_processor.do_resize = False
            self.processor.image_processor.do_normalize = True

        # Build mapping from your vocab indices -> TrOCR tokenizer ids
        tok = self.processor.tokenizer
        V_you = int(vocab_size)
        map_list = [0] * V_you
        valid_mask = torch.zeros(V_you, dtype=torch.bool)

        def _first_token_id_for_char(ch_str: str) -> Optional[int]:
            # Tokenize a single-character string and grab its first id
            ids = tok(ch_str, add_special_tokens=False)["input_ids"]
            return int(ids[0]) if len(ids) > 0 else None

        # 1) regular printable chars from your index2letter
        for i, sym in enumerate(index2letter):
            your_idx = num_tokens + i
            if your_idx >= V_you:
                break
            ch = _to_char(sym)
            if not ch:
                continue
            tro_id = _first_token_id_for_char(ch)
            if tro_id is not None:
                map_list[your_idx] = tro_id
                valid_mask[your_idx] = True

        # 2) (optional) explicit mapping for a SPACE token in your vocab dict
        try:
            space_idx = tokens.get("SPACE_TOKEN", None)
            if space_idx is not None and 0 <= space_idx < V_you:
                sp_id = _first_token_id_for_char(" ")
                if sp_id is not None:
                    map_list[space_idx] = sp_id
                    valid_mask[space_idx] = True
        except Exception:
            pass

        self.register_buffer("_map_vec", torch.tensor(map_list, dtype=torch.long), persistent=False)
        self.register_buffer("_valid_mask", valid_mask, persistent=False)

    # ------------------------ label → decoder inputs ------------------------
    def _labels_to_trocr_inputs(self, labels_you: torch.Tensor) -> torch.Tensor:
        """Convert your [B,T] label indices -> decoder_input_ids for TrOCR.
        We first build per-sample strings using index2letter, then tokenize.
        """
        strings: List[str] = []
        B, T = labels_you.shape
        for b in range(B):
            seq_chars: List[str] = []
            for idx in labels_you[b].tolist():
                j = idx - num_tokens
                if 0 <= j < len(index2letter):
                    ch = _to_char(index2letter[j])
                    if ch:
                        seq_chars.append(ch)
            strings.append("".join(seq_chars))
        tok = self.processor.tokenizer
        enc = tok(strings, padding=True, add_special_tokens=True, return_tensors="pt")
        return enc["input_ids"].to(labels_you.device)

    # ------------------------ forward ------------------------
    def forward(self, img: torch.Tensor, label_you: torch.Tensor, img_width=None) -> torch.Tensor:
        """Return logits in your vocab space: [B, OUTPUT_MAX_LEN, vocab_size]."""
        x = _ensure_rgb_and_square(img, 384)  # [B,3,H,~W]

        # Normalize using processor stats (differentiable)
        ip = self.processor.image_processor
        mean = torch.tensor(ip.image_mean, dtype=x.dtype, device=x.device).view(1, 3, 1, 1)
        std = torch.tensor(ip.image_std, dtype=x.dtype, device=x.device).view(1, 3, 1, 1)
        x = (x - mean) / std

        with torch.no_grad():
            dec_in = self._labels_to_trocr_inputs(label_you)

        outputs = self.model(pixel_values=x, decoder_input_ids=dec_in)
        logits_tro = outputs.logits  # [B, T, V_tro]
        B, T, V_tro = logits_tro.shape

        # Project to your vocab
        map_vec = self._map_vec  # [V_you]
        valid = self._valid_mask  # [V_you]
        V_you = map_vec.numel()
        idx = map_vec.view(1, 1, V_you).expand(B, T, V_you)
        logits_you = torch.gather(logits_tro, dim=2, index=idx.clamp(0, V_tro - 1))
        logits_you = logits_you.masked_fill(~valid.view(1, 1, V_you), -1e9)

        # Pad/trim time dim to OUTPUT_MAX_LEN for drop-in compatibility
        if T < OUTPUT_MAX_LEN:
            pad = logits_you.new_full((B, OUTPUT_MAX_LEN - T, V_you), -1e9)
            logits_you = torch.cat([logits_you, pad], dim=1)
        elif T > OUTPUT_MAX_LEN:
            logits_you = logits_you[:, :OUTPUT_MAX_LEN, :]
            
        #logits_you = logits_you / 100000000.0

        return logits_you

    # ------------------------ decode helper (greedy/beam) ------------------------
    @torch.no_grad()
    def decode(self, img: torch.Tensor, beam_size: int = 5, max_new_tokens: int = 128) -> List[str]:
        x = _ensure_rgb_and_square(img, IMG_HEIGHT)
        ip = self.processor.image_processor
        mean = torch.tensor(ip.image_mean, dtype=x.dtype, device=x.device).view(1, 3, 1, 1)
        std = torch.tensor(ip.image_std, dtype=x.dtype, device=x.device).view(1, 3, 1, 1)
        x = (x - mean) / std
        gen = self.model.generate(
            x,
            num_beams=beam_size,
            max_new_tokens=max_new_tokens,
            early_stopping=True,
        )
        return self.processor.batch_decode(gen, skip_special_tokens=True)