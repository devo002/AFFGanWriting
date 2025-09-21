"""
Drop-in TrOCR recognizer wrapper that matches your RecModel interface
and returns logits with shape [B, T, vocab_size] so your existing
training code can keep computing cross-entropy against your current
labels. Gradients flow through TrOCR to the input image (recognition
loss still trains your generator), while TrOCR weights stay frozen.

Assumptions from your codebase:
- You pass grayscale images shaped [B, 1, IMG_HEIGHT, W].
- You already have: vocab_size, index2letter, tokens, OUTPUT_MAX_LEN, IMG_HEIGHT.
- labels are LongTensor of shape [B, T] in your current vocabulary where
  valid characters live at indices [num_tokens .. num_tokens+len(index2letter)-1].

Usage:
    from trocr_recognizer import TrOCRRecModel
    rec = TrOCRRecModel(ckpt="/home/woody/iwi5/iwi5333h/model/trocr-base-handwritten", local_only=True).to(device)
    logits = rec(img, label, img_width)  # -> [B,T,vocab_size]

Dependencies:
    pip install transformers>=4.41.0
"""
from __future__ import annotations

import math
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
except Exception as e:  # nicer error if missing
    raise ImportError("Please `pip install transformers` to use TrOCRRecModel") from e

# These should be imported from your project where you define them
try:
    from load_data import OUTPUT_MAX_LEN, IMG_HEIGHT, vocab_size, index2letter, num_tokens, tokens
except Exception as e:
    raise RuntimeError(
        "trocr_recognizer.py expects load_data to define OUTPUT_MAX_LEN, IMG_HEIGHT, "
        "vocab_size, index2letter, num_tokens, tokens"
    ) from e


__all__ = ["TrOCRRecModel"]


def _ensure_rgb_and_resize(x: torch.Tensor, target_h: int) -> torch.Tensor:
    """x: [B, C, H, W] in [0,1] or [-1,1]. Returns [B, 3, target_h, W*]."""
    if x.dtype != torch.float32:
        x = x.float()
    # bring to [0,1]
    if torch.min(x) < 0:  # assume [-1,1]
        x = (x + 1.0) * 0.5
    x = x.clamp(0, 1)
    # grayscale -> rgb
    if x.shape[1] == 1:
        x = x.repeat(1, 3, 1, 1)
    B, C, H, W = x.shape
    if H != target_h:
        new_w = math.floor(W * (target_h / H))
        x = F.interpolate(x, size=(target_h, new_w), mode="bilinear", align_corners=False)
    return x


class TrOCRRecModel(nn.Module):
    def __init__(self, ckpt: str = "microsoft/trocr-base-handwritten", local_only: bool = False):
        """
        Args:
            ckpt: HF model id or local directory path
            local_only: if True, load only from local files (no internet)
        """
        super().__init__()
        self.processor = TrOCRProcessor.from_pretrained(ckpt, local_files_only=local_only)
        self.model = VisionEncoderDecoderModel.from_pretrained(ckpt, local_files_only=local_only)

        # Freeze TrOCR weights; we still backprop to the IMAGE
        self.model.eval()
        self.model.requires_grad_(False)

        # Avoid the processor resizing for us (we resize in-tensor for autograd safety)
        image_proc = getattr(self.processor, "image_processor", getattr(self.processor, "feature_extractor", None))
        if image_proc is not None:
            # we'll resize ourselves; keep normalization enabled (we'll apply same stats manually)
            if hasattr(image_proc, "do_resize"):
                image_proc.do_resize = False
            if hasattr(image_proc, "do_normalize"):
                image_proc.do_normalize = True

        # Build mapping from your vocab indices -> TrOCR tokenizer ids
        tok = self.processor.tokenizer
        V_you = vocab_size
        map_list = [0] * V_you
        valid_mask = torch.zeros(V_you, dtype=torch.bool)

        # Your printable chars start at offset num_tokens, following index2letter ordering
        for i, ch in enumerate(index2letter):
            your_idx = num_tokens + i
            ch_py = str(ch)  # ensure plain Python str (numpy.str_ etc. would confuse HF)
            # normalize your space symbol to real space for TrOCR
            if ch_py in ("<space>", "␣"):
                ch_py = " "
            # Always use list API so we deterministically get a list[int]
            ids = tok.convert_tokens_to_ids([ch_py])
            trocr_id = ids[0] if ids and ids[0] is not None else tok.unk_token_id
            if trocr_id is not None and trocr_id != tok.unk_token_id:
                map_list[your_idx] = int(trocr_id)
                valid_mask[your_idx] = True

        # (optional) map a few specials from your tokens dict, e.g., SPACE_TOKEN
        if hasattr(tokens, "get") and "SPACE_TOKEN" in tokens:
            your_idx = tokens["SPACE_TOKEN"]
            ids = tok.convert_tokens_to_ids([" "])
            trocr_id = ids[0] if ids and ids[0] is not None else tok.unk_token_id
            if trocr_id is not None and trocr_id != tok.unk_token_id:
                map_list[your_idx] = int(trocr_id)
                valid_mask[your_idx] = True

        self.register_buffer("_map_vec", torch.tensor(map_list, dtype=torch.long), persistent=False)
        self.register_buffer("_valid_mask", valid_mask.to(torch.bool), persistent=False)


    def _labels_to_trocr_inputs(self, labels_you: torch.Tensor) -> torch.Tensor:
        """Convert your [B,T] label indices -> decoder_input_ids for TrOCR.
        We first turn indices into a string using index2letter, then tokenize.
        """
        strings: List[str] = []
        B, T = labels_you.shape
        for b in range(B):
            seq_chars = []
            for idx in labels_you[b].tolist():
                off = int(idx) - int(num_tokens)
                if 0 <= off < len(index2letter):
                    ch_py = str(index2letter[off])
                    if ch_py in ("<space>", "␣"):
                        ch_py = " "
                    seq_chars.append(ch_py)
                # else: skip non-printable specials
            strings.append("".join(seq_chars))
        # Tokenize into decoder inputs (teacher forcing; HF will handle shifting internally)
        tok = self.processor.tokenizer
        enc = tok(strings, padding=True, add_special_tokens=True, return_tensors="pt")
        return enc["input_ids"].to(labels_you.device)

    def forward(self, img: torch.Tensor, label_you: torch.Tensor, img_width=None) -> torch.Tensor:
        """Return logits in your vocab space: [B, T, vocab_size].
        The time dimension T matches the tokenizer length for the given label,
        then is padded/truncated to OUTPUT_MAX_LEN to match your pipeline.
        """
        # Prepare image: to RGB, resize height, normalize with processor stats
        x = _ensure_rgb_and_resize(img, IMG_HEIGHT)  # [B,3,H,~W]
        image_proc = getattr(self.processor, "image_processor", getattr(self.processor, "feature_extractor", None))
        if image_proc is not None and hasattr(image_proc, "image_mean") and hasattr(image_proc, "image_std"):
            mean = torch.tensor(image_proc.image_mean, dtype=x.dtype, device=x.device).view(1, 3, 1, 1)
            std = torch.tensor(image_proc.image_std, dtype=x.dtype, device=x.device).view(1, 3, 1, 1)
        else:
            # Sensible defaults if stats are missing
            mean = torch.tensor([0.5, 0.5, 0.5], dtype=x.dtype, device=x.device).view(1, 3, 1, 1)
            std = torch.tensor([0.5, 0.5, 0.5], dtype=x.dtype, device=x.device).view(1, 3, 1, 1)
        x = (x - mean) / std
        pixel_values = x  # [B,3,H,W]

        # Build teacher-forced decoder inputs from your labels (no grads needed here)
        with torch.no_grad():
            dec_in = self._labels_to_trocr_inputs(label_you)

        # Run the model to obtain decoder logits (no LM loss computed)
        outputs = self.model(pixel_values=pixel_values, decoder_input_ids=dec_in)
        logits_tro = outputs.logits  # [B, T, V_tro]
        B, T, V_tro = logits_tro.shape

        # Project to your vocab by gathering the corresponding TrOCR logits for overlapping chars
        map_vec = self._map_vec  # [V_you]
        valid = self._valid_mask  # [V_you]
        V_you = map_vec.numel()

        # Gather: for each of your vocab slots, fetch the TrOCR logit at the mapped id
        # idx tensor shape [B,T,V_you] with trocr ids per your vocab slot (broadcasted)
        idx = map_vec.view(1, 1, V_you).expand(B, T, V_you)
        # Clamp indices to valid range (unmapped positions are 0; will be masked below)
        idx = idx.clamp(0, max(0, V_tro - 1))
        logits_you = torch.gather(logits_tro, dim=2, index=idx)

        # Mask out unmapped vocab positions with a large negative value so CE ignores them
        if valid is not None:
            logits_you = logits_you.masked_fill(~valid.view(1, 1, V_you), -1e9)

        # Pad/trim time dim to your expected OUTPUT_MAX_LEN for drop-in compatibility
        if T < OUTPUT_MAX_LEN:
            pad = logits_you.new_full((B, OUTPUT_MAX_LEN - T, V_you), -1e9)
            logits_you = torch.cat([logits_you, pad], dim=1)
        elif T > OUTPUT_MAX_LEN:
            logits_you = logits_you[:, :OUTPUT_MAX_LEN, :]

        return logits_you  # raw logits in your vocab

    @torch.no_grad()
    def decode(self, img: torch.Tensor, beam_size: int = 5, max_new_tokens: int = 128) -> List[str]:
        """Greedy/beam decode to strings using TrOCR's tokenizer."""
        x = _ensure_rgb_and_resize(img, IMG_HEIGHT)
        image_proc = getattr(self.processor, "image_processor", getattr(self.processor, "feature_extractor", None))
        if image_proc is not None and hasattr(image_proc, "image_mean") and hasattr(image_proc, "image_std"):
            mean = torch.tensor(image_proc.image_mean, dtype=x.dtype, device=x.device).view(1, 3, 1, 1)
            std = torch.tensor(image_proc.image_std, dtype=x.dtype, device=x.device).view(1, 3, 1, 1)
        else:
            mean = torch.tensor([0.5, 0.5, 0.5], dtype=x.dtype, device=x.device).view(1, 3, 1, 1)
            std = torch.tensor([0.5, 0.5, 0.5], dtype=x.dtype, device=x.device).view(1, 3, 1, 1)
        x = (x - mean) / std

        generated = self.model.generate(
            x,
            num_beams=beam_size,
            max_new_tokens=max_new_tokens,
            early_stopping=True,
        )
        return self.processor.batch_decode(generated, skip_special_tokens=True)
