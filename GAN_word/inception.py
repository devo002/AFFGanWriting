import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.inception import inception_v3


class ImageEncoderInceptionV3(nn.Module):
    """
    Inception v3 encoder that returns 5 intermediate feature maps reduced to 512 channels:
      taps = [Mixed_5c, Mixed_5d, Mixed_6b, Mixed_6e, Mixed_7c]
    It supports arbitrary in_channels (e.g., 50) and short-height inputs (e.g., 48xW).
    """
    def __init__(self, weight_path=None, in_channels=50, aux_logits=False,
                 final_size=(8, 27), soften_downsampling=True,
                 probe_size=(48, 540)):
        super().__init__()
        self.output_dim = 512
        self.final_size = final_size
        self.probe_size = probe_size  # (H, W) used once at init to infer channel sizes

        # Backbone
        self.model = inception_v3(weights=None, aux_logits=aux_logits, transform_input=False)

        # Optional weights (lenient to allow first-conv change)
        if weight_path:
            sd = torch.load(weight_path, map_location="cpu")
            if isinstance(sd, dict) and "state_dict" in sd:
                sd = sd["state_dict"]
            elif isinstance(sd, dict) and "model" in sd:
                sd = sd["model"]
            sd = { (k[7:] if k.startswith("module.") else k): v for k, v in sd.items() }
            self.model.load_state_dict(sd, strict=False)

        # First conv → accept arbitrary in_channels
        first = self.model.Conv2d_1a_3x3.conv
        new_first = nn.Conv2d(
            in_channels, first.out_channels,
            kernel_size=first.kernel_size, stride=first.stride,
            padding=first.padding, bias=(first.bias is not None)
        )
        with torch.no_grad():
            if first.weight.shape[1] == 3:
                new_first.weight[:, :3] = first.weight
                if in_channels > 3:
                    rep = first.weight[:, :1].repeat(1, in_channels-3, 1, 1)
                    new_first.weight[:, 3:] = rep
            else:
                nn.init.kaiming_normal_(new_first.weight, mode="fan_out", nonlinearity="relu")
                if new_first.bias is not None:
                    nn.init.zeros_(new_first.bias)
        self.model.Conv2d_1a_3x3.conv = new_first

        # Handle naming differences across torchvision versions (pool wrappers vs direct)
        pool1 = getattr(self.model, "MaxPool_3a_3x3", None) or getattr(self.model, "maxpool1")
        pool2 = getattr(self.model, "MaxPool_5a_3x3", None) or getattr(self.model, "maxpool2")
        self.pool1 = pool1
        self.pool2 = pool2

        # Optionally soften early downsampling for short-height inputs
        if soften_downsampling:
            self.model.Conv2d_1a_3x3.conv.stride = (1, 1)
            if hasattr(self.pool1, "pool"):   # newer wrapper
                self.pool1.pool.stride = (1, 1)
            else:                              # older direct MaxPool2d
                self.pool1.stride = (1, 1)
            # Leave pool2 at (2,2). Relax if you still lose too much height.

        # Build a flat, ordered dict following official forward() sequence
        self.blocks = nn.ModuleDict({
            "Conv2d_1a_3x3": self.model.Conv2d_1a_3x3,
            "Conv2d_2a_3x3": self.model.Conv2d_2a_3x3,
            "Conv2d_2b_3x3": self.model.Conv2d_2b_3x3,
            "pool1": self.pool1,                          # name normalized
            "Conv2d_3b_1x1": self.model.Conv2d_3b_1x1,
            "Conv2d_4a_3x3": self.model.Conv2d_4a_3x3,
            "pool2": self.pool2,                          # name normalized
            "Mixed_5b": self.model.Mixed_5b,
            "Mixed_5c": self.model.Mixed_5c,
            "Mixed_5d": self.model.Mixed_5d,
            "Mixed_6a": self.model.Mixed_6a,
            "Mixed_6b": self.model.Mixed_6b,
            "Mixed_6c": self.model.Mixed_6c,
            "Mixed_6d": self.model.Mixed_6d,
            "Mixed_6e": self.model.Mixed_6e,
            "Mixed_7a": self.model.Mixed_7a,
            "Mixed_7b": self.model.Mixed_7b,
            "Mixed_7c": self.model.Mixed_7c,
        })
        self.block_order = list(self.blocks.keys())

        # We will collect features at these 5 points (consistent across versions)
        self.collect_points = ["Mixed_5c", "Mixed_5d", "Mixed_6b", "Mixed_6e", "Mixed_7c"]

        # ---- Infer reducer in_channels with a one-time dummy pass (robust to concat) ----
        with torch.no_grad():
            H, W = self.probe_size
            dummy = torch.zeros(1, in_channels, H, W)
            ch_list = []
            x = dummy
            for name in self.block_order:
                block = self.blocks[name]
                x = block(x)
                if name in self.collect_points:
                    ch_list.append(x.shape[1])  # record channel count at tap
            # Safety check
            if len(ch_list) != len(self.collect_points):
                raise RuntimeError(f"Expected to collect {len(self.collect_points)} points, "
                                   f"but got {len(ch_list)}. Check collect_points.")
        # Create reducers now that we know exact channels
        self.reduce_layers = nn.ModuleList([
            nn.Conv2d(c, 512, kernel_size=1) for c in ch_list
        ])

    def encode_with_intermediate(self, x: torch.Tensor):
        results = []
        red_idx = 0
        h = x
        for name in self.block_order:
            h = self.blocks[name](h)
            if name in self.collect_points:
                results.append(self.reduce_layers[red_idx](h))
                red_idx += 1

        # Resize the LAST map to match your VGG path
        results[-1] = F.interpolate(results[-1], size=self.final_size,
                                    mode='bilinear', align_corners=False)
        return results  # 5 tensors: [B, 512, H_i, W_i]

    def forward(self, x: torch.Tensor):
        return self.encode_with_intermediate(x)
