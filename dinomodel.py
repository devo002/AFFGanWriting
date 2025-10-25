import os
import torch
import torch.nn as nn
import torch.nn.functional as F


class ImageEncoderDINOv2(nn.Module):
    """
    DINOv2 ViT encoder loaded from a LOCAL torch.hub repo that:
      • accepts arbitrary in_channels (e.g., 50)
      • auto-pads inputs so H,W are multiples of the patch size (14)
      • returns 5 spatial feature maps, each reduced to 512 channels
      • resizes the LAST map to `final_size` for your decoder

    Typical hub names in the local repo:
      "dinov2_vits14", "dinov2_vitb14", "dinov2_vitl14", "dinov2_vitg14"
    Use `arch="vitl14"` for ViT-L/14, etc.
    """

    def __init__(
        self,
        repo_dir: str,
        arch: str = "vitl14",                 # "vits14" | "vitb14" | "vitl14" | "vitg14"
        ckpt_path: str | None = None,
        in_channels: int = 50,
        final_size: tuple[int, int] = (8, 27),
        tap_blocks: list[int] | None = None,  # which transformer block indices to tap
        probe_size: tuple[int, int] = (48, 540),
    ):
        super().__init__()
        self.output_dim = 512
        self.final_size = final_size

        # ---- Load backbone from the local repo (no internet) ----
        hub_name = f"dinov2_{arch}"
        self.model = torch.hub.load(repo_dir, hub_name, source="local", pretrained=False)

        # Optional: remove head if present
        if hasattr(self.model, "reset_classifier"):
            self.model.reset_classifier(0)

        # ---- Load local checkpoint (lenient) ----
        if ckpt_path is not None and os.path.isfile(ckpt_path):
            sd = torch.load(ckpt_path, map_location="cpu")
            if isinstance(sd, dict) and "state_dict" in sd:
                sd = sd["state_dict"]
            elif isinstance(sd, dict) and "model" in sd:
                sd = sd["model"]
            sd = { (k[7:] if k.startswith("module.") else k): v for k, v in sd.items() }
            self.model.load_state_dict(sd, strict=False)

        # ---- Rewrite first conv so we can use in_channels != 3 ----
        patch = self.model.patch_embed                      # PatchEmbed
        old_proj = patch.proj                               # nn.Conv2d
        new_proj = nn.Conv2d(
            in_channels,
            old_proj.out_channels,
            kernel_size=old_proj.kernel_size,
            stride=old_proj.stride,
            padding=old_proj.padding,
            bias=(old_proj.bias is not None),
        )
        with torch.no_grad():
            if old_proj.weight.shape[1] == 3:
                new_proj.weight[:, :3] = old_proj.weight
                if in_channels > 3:
                    rep = old_proj.weight[:, :1].repeat(1, in_channels - 3, 1, 1)
                    new_proj.weight[:, 3:] = rep
            else:
                nn.init.kaiming_normal_(new_proj.weight, mode="fan_out", nonlinearity="relu")
                if new_proj.bias is not None:
                    nn.init.zeros_(new_proj.bias)
        patch.proj = new_proj

        self.embed_dim = self.model.embed_dim
        self.patch_size = (
            patch.patch_size if isinstance(patch.patch_size, tuple)
            else (patch.patch_size, patch.patch_size)
        )

        # ---- Decide which blocks to tap ----
        num_blocks = len(self.model.blocks)
        if tap_blocks is None:
            # 4 roughly even blocks + stem → total 5 taps
            # (works across S/B/L/G; for ViT-L/14 a nice choice is [4, 8, 16, 23])
            idxs = torch.linspace(0, num_blocks - 1, steps=4).round().to(torch.int64).tolist()
            tap_blocks = sorted(set(idxs))
        self.tap_blocks = tap_blocks

        # 1 stem + len(tap_blocks) reducers (1x1 conv to 512)
        num_taps = 1 + len(self.tap_blocks)
        self.reduce_layers = nn.ModuleList(
            [nn.Conv2d(self.embed_dim, 512, kernel_size=1) for _ in range(num_taps)]
        )

        # ---- Quick probe to ensure forward works with padding ----
        with torch.no_grad():
            H, W = probe_size
            dummy = torch.zeros(1, in_channels, H, W)
            _ = self.encode_with_intermediate(dummy)

    # ---------- helpers ----------
    def _pos_embed_tokens(self, x):
        """Use model's internal positional embedding util if available."""
        if hasattr(self.model, "_pos_embed"):
            return self.model._pos_embed(x)
        # Fallback: add cls token + pos without interpolation (usually fine)
        B, N, C = x.shape
        if hasattr(self.model, "cls_token") and self.model.cls_token is not None:
            cls_tok = self.model.cls_token.expand(B, -1, -1)
            x = torch.cat((cls_tok, x), dim=1)
        if hasattr(self.model, "pos_embed") and self.model.pos_embed is not None:
            if self.model.pos_embed.shape[1] == x.shape[1]:
                x = x + self.model.pos_embed
        if hasattr(self.model, "pos_drop"):
            x = self.model.pos_drop(x)
        return x

    @staticmethod
    def _tokens_to_map(x_tokens, Hp, Wp):
        """(B, Hp*Wp, C) -> (B, C, Hp, Wp)"""
        B, HW, C = x_tokens.shape
        x_tokens = x_tokens.transpose(1, 2).contiguous()
        return x_tokens.view(B, C, Hp, Wp)

    # ---------- main ----------
    def encode_with_intermediate(self, x: torch.Tensor):
        B, _, H, W = x.shape
        ph, pw = self.patch_size

        # ---- PAD to multiples of patch size (right & bottom) ----
        pad_h = (ph - (H % ph)) % ph
        pad_w = (pw - (W % pw)) % pw
        if pad_h or pad_w:
            # pad = (left, right, top, bottom)
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")
        Hp, Wp = x.shape[-2] // ph, x.shape[-1] // pw

        # Stem: patchify → tokens → add pos + cls
        tokens = self.model.patch_embed(x)           # (B, N, C)
        tokens = self._pos_embed_tokens(tokens)      # (B, 1+N, C)

        results = []

        # Tap 0: "stem" (spatial tokens only, discard cls)
        stem_tokens = tokens[:, 1:, :]
        stem_map = self._tokens_to_map(stem_tokens, Hp, Wp)         # (B, C, Hp, Wp)
        results.append(self.reduce_layers[0](stem_map))             # -> (B, 512, Hp, Wp)

        # Transformer blocks, collect at specified indices
        red = 1
        for i, blk in enumerate(self.model.blocks):
            tokens = blk(tokens)
            if i in self.tap_blocks:
                spatial = tokens[:, 1:, :]
                fmap = self._tokens_to_map(spatial, Hp, Wp)         # (B, C, Hp, Wp)
                results.append(self.reduce_layers[red](fmap))       # -> (B, 512, Hp, Wp)
                red += 1

        # Last map → fixed size for your decoder
        results[-1] = F.interpolate(results[-1], size=self.final_size,
                                    mode="bilinear", align_corners=False)
        return results  # list of 5 tensors: [B, 512, H_i, W_i]

    def forward(self, x: torch.Tensor):
        return self.encode_with_intermediate(x)