# trocr_teacher.py
import torch
from transformers import VisionEncoderDecoderModel, TrOCRProcessor

class TrocrTeacher(torch.nn.Module):
    def __init__(self, name="/home/woody/iwi5/iwi5333h/model/trocr-base-handwritten", device="cuda"):
        super().__init__()
        self.processor = TrOCRProcessor.from_pretrained(name)
        self.model = VisionEncoderDecoderModel.from_pretrained(name).to(device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        self.device = device

        if self.model.config.decoder_start_token_id is None:
            self.model.config.decoder_start_token_id = self.processor.tokenizer.cls_token_id
        if self.model.config.pad_token_id is None:
            self.model.config.pad_token_id = self.processor.tokenizer.pad_token_id

    @torch.no_grad()
    def predict(self, imgs_tensor):
        """
        imgs_tensor: [B,1,H,W] or [B,3,H,W] or [B,H,W] or [H,W] or [3,H,W] etc.
        Returns:
      texts: list[str] length B
      conf : tensor[B] in [0,1]
    """
    # ---- 1) normalize to [B,3,H,W] uint8 on CPU ----
    # ensure a batch dimension
        if imgs_tensor.ndim == 2:            # [H,W]
            imgs_tensor = imgs_tensor.unsqueeze(0).unsqueeze(0)   # -> [1,1,H,W]
        elif imgs_tensor.ndim == 3:
            if imgs_tensor.shape[0] in (1, 3):                    # [C,H,W]
                imgs_tensor = imgs_tensor.unsqueeze(0)            # -> [1,C,H,W]
            else:                                                 # [B,H,W]
                imgs_tensor = imgs_tensor.unsqueeze(1)            # -> [B,1,H,W]
        elif imgs_tensor.ndim != 4:
            raise ValueError(f"Unexpected image ndim={imgs_tensor.ndim}")

    # move to CPU for processor and clamp/scale
        x = imgs_tensor.detach().float().cpu()                    # [B,C,H,W], float
    # many of your tensors are in [-1,1]; map to [0,1] if needed
        if x.min() < 0:
            x = (x + 1.0) / 2.0
            x = x.clamp(0, 1)

    # ensure 3 channels
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)                              # -> [B,3,H,W]
        elif x.shape[1] != 3:
        # very rare, but keep a hard fail if channels are odd
            raise ValueError(f"Unexpected channel count: {x.shape[1]}")

    # convert to uint8 channel-last numpy for the processor
        x = (x * 255.0).round().to(torch.uint8).contiguous()      # [B,3,H,W]
        pil_list = [img.permute(1, 2, 0).numpy() for img in x]    # [B,H,W,3] uint8

    # ---- 2) run TrOCR ----
        inputs = self.processor(images=pil_list, return_tensors="pt").to(self.device)

        gen = self.model.generate(
            **inputs,
            max_new_tokens=64,
            num_beams=getattr(self, "num_beams", 1),  # support greedy or beam
            do_sample=False,
            return_dict_in_generate=False,
            output_scores=False,
            )
        texts = self.processor.batch_decode(gen, skip_special_tokens=True)

    # ---- 3) confidence via teacher-forced pass on predicted texts ----
        labels = self.processor.tokenizer(texts, padding=True, return_tensors="pt")["input_ids"].to(self.device)
        out = self.model(pixel_values=inputs["pixel_values"], labels=labels)
        logits = out.logits                                  # [B,T,V]

        mask = (labels != -100) & (labels != self.processor.tokenizer.pad_token_id)
        probs = torch.softmax(logits, dim=-1)
        token_prob = probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)  # [B,T]
        token_prob = torch.where(mask, token_prob, torch.ones_like(token_prob))
        conf = token_prob.sum(dim=1) / mask.sum(dim=1).clamp_min(1)      # [B]

        return texts, conf
