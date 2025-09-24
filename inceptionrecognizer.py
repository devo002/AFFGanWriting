import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torchvision.models import inception_v3
import numpy as np

# If you already define these elsewhere, remove these defaults.

DROP_OUT = True
LSTM = False
SUM_UP = False

def _flex_load_state_dict(module, ckpt_path, map_location="cpu", strict=False, prefix_to_strip=""):
    """Load state_dict from various checkpoint formats, tolerating missing/extra keys."""
    sd = torch.load(ckpt_path, map_location=map_location)
    if isinstance(sd, dict):
        if "state_dict" in sd and isinstance(sd["state_dict"], dict):
            sd = sd["state_dict"]
        elif "model" in sd and isinstance(sd["model"], dict):
            sd = sd["model"]
    if prefix_to_strip:
        sd = {k[len(prefix_to_strip):] if k.startswith(prefix_to_strip) else k: v for k, v in sd.items()}
    missing, unexpected = module.load_state_dict(sd, strict=strict)
    if missing or unexpected:
        print(f"[Inception load] missing: {len(missing)}, unexpected: {len(unexpected)}")
    return module


# ---------- backbone ----------

class InceptionV3Backbone(nn.Module):
    """
    Returns a conv feature map from InceptionV3.

    Args:
        weights_path: path to a .pth checkpoint (optional).
        in_channels:  number of input channels your data has (1, 3, 50, ...).
        output_stride: 16 (Mixed_6e, 768ch) or 32 (Mixed_7c, 2048ch).
        map_location: torch.load map_location for the weights.
    """
    def __init__(self, weights_path: str = None, in_channels: int = 3,
                 output_stride: int = 16, map_location: str = "cpu"):
        super().__init__()
        assert output_stride in (16, 32), "output_stride must be 16 or 32"
        self.output_stride = output_stride

        m = inception_v3(weights=None, aux_logits=False)
        if weights_path:
            _flex_load_state_dict(m, weights_path, map_location=map_location, strict=False)

        # Replace first conv to accept arbitrary in_channels
        old = m.Conv2d_1a_3x3.conv
        new_first = nn.Conv2d(in_channels, old.out_channels,
                              kernel_size=old.kernel_size,
                              stride=old.stride,
                              padding=old.padding,
                              bias=(old.bias is not None))

        with torch.no_grad():
            if in_channels == 3 and old.weight.shape[1] == 3:
                # keep pretrained RGB weights
                new_first.weight.copy_(old.weight)
                if old.bias is not None:
                    new_first.bias.copy_(old.bias)
            elif in_channels == 1 and old.weight.shape[1] == 3:
                # average RGB to single channel
                new_first.weight.copy_(old.weight.mean(dim=1, keepdim=True))
                if old.bias is not None:
                    new_first.bias.copy_(old.bias)
            else:
                # generic init for arbitrary channel counts (e.g., 50)
                nn.init.kaiming_normal_(new_first.weight, mode="fan_out", nonlinearity="relu")
                if new_first.bias is not None:
                    nn.init.zeros_(new_first.bias)

        m.Conv2d_1a_3x3.conv = new_first
        self.m = m

    def forward(self, x):
        m = self.m
        # Stem
        x = m.Conv2d_1a_3x3(x)
        x = m.Conv2d_2a_3x3(x)
        x = m.Conv2d_2b_3x3(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        x = m.Conv2d_3b_1x1(x)
        x = m.Conv2d_4a_3x3(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # Mixed 5 (35x35)
        x = m.Mixed_5b(x); x = m.Mixed_5c(x); x = m.Mixed_5d(x)
        # Mixed 6 (17x17)
        x = m.Mixed_6a(x); x = m.Mixed_6b(x); x = m.Mixed_6c(x); x = m.Mixed_6d(x); x = m.Mixed_6e(x)

        if self.output_stride == 32:
            # Mixed 7 (8x8)
            x = m.Mixed_7a(x); x = m.Mixed_7b(x); x = m.Mixed_7c(x)
        return x  # OS=16: (B, 768, H/16, W/16) ; OS=32: (B, 2048, H/32, W/32)


# ---------- encoder wrapper (VGG-compatible API) ----------

class EncoderInception(nn.Module):
    """
    Mirrors your VGG Encoder API:
        __init__(hidden_size, height, width, bgru, step, flip, ...)

    Uses InceptionV3 conv features as the sequence source (time = width').
    """
    def __init__(self, hidden_size, height, width, bgru, step, flip,
                 weights_path: str = None, in_channels: int = 3,
                 output_stride: int = 16, map_location: str = "cpu"):
        super().__init__()
        self.hidden_size = hidden_size
        self.height = height
        self.width = width
        self.bi = bgru
        self.step = step
        self.flip = flip
        self.n_layers = 2
        self.dropout = 0.5

        # Backbone
        self.layer = InceptionV3Backbone(
            weights_path=weights_path,
            in_channels=in_channels,
            output_stride=output_stride,
            map_location=map_location
        )

        if DROP_OUT:
            self.layer_dropout = nn.Dropout2d(p=0.5)

        # Feature geometry
        if output_stride == 16:
            self.spatial_down = 16
            self.feat_channels = 768
        else:
            self.spatial_down = 32
            self.feat_channels = 2048

        self.flat_feat_per_t = (self.height // self.spatial_down) * self.feat_channels

        # Optional temporal stacking (step)
        if self.step is not None:
            self.output_proj = nn.Linear(self.flat_feat_per_t * self.step, self.flat_feat_per_t)

        # RNN choice
        RNN = nn.LSTM if LSTM else nn.GRU
        self.rnn = RNN(self.flat_feat_per_t, self.hidden_size, self.n_layers,
                       dropout=self.dropout, bidirectional=self.bi)

        if self.bi and SUM_UP:
            self.enc_out_merge = lambda x: x[:, :, : x.shape[-1] // 2] + x[:, :, x.shape[-1] // 2 :]
            self.enc_hidden_merge = lambda x: (x[0] + x[1]).unsqueeze(0)

    def forward(self, in_data, in_data_len, hidden=None):
        """
        in_data: (B, C, H, W)  with C == backbone in_channels
        in_data_len: per-sample original width (Tensor or np array), like your VGG encoder.
        """
        device = in_data.device
        B = in_data.size(0)

        # Safety check: channels must match what backbone expects
        assert in_data.shape[1] == self.layer.m.Conv2d_1a_3x3.conv.in_channels, \
            f"Input has {in_data.shape[1]} channels, but backbone expects {self.layer.m.Conv2d_1a_3x3.conv.in_channels}"

        # Conv features
        out = self.layer(in_data)  # (B, C', H', W')
        if DROP_OUT and self.training:
            out = self.layer_dropout(out)

        # To (T, B, F) where T is W'
        out = out.permute(3, 0, 2, 1)                  # (W', B, H', C')
        out = out.reshape(-1, B, self.flat_feat_per_t) # (T, B, F)

        # Optional temporal stacking
        if self.step is not None:
            T, B, Ff = out.shape
            t_new = T // self.step
            out_short = torch.zeros(t_new, B, Ff * self.step, requires_grad=True, device=device)
            for i in range(t_new):
                part_out = [out[j] for j in range(i * self.step, (i + 1) * self.step)]
                out_short[i] = torch.cat(part_out, dim=1)
            out = self.output_proj(out_short)  # (t_new, B, F)

        # Compute src_len in the new temporal grid
        width_prime = out.shape[0]
        src_len = (in_data_len.detach().cpu().numpy() * (width_prime / self.width)) + 0.999
        src_len = src_len.astype('int').tolist()

        # Pack & RNN
        packed = pack_padded_sequence(out, src_len, batch_first=False, enforce_sorted=False)
        output, hidden = self.rnn(packed, hidden)
        output, _ = pad_packed_sequence(output, batch_first=False)

        if self.bi and SUM_UP:
            output = self.enc_out_merge(output)
            # hidden = self.enc_hidden_merge(hidden)  # keep same behavior as your VGG file (commented there too)

        # Mimic your odd-index selection for final_hidden
        odd_idx = [1, 3, 5, 7, 9, 11][:self.n_layers]
        final_hidden = hidden[odd_idx]
        return output, final_hidden  # output: (T, B, F{*2 if bi}); final_hidden: (n_layers, B, hidden)

    # matrix: (B, C, H', W'), lens: list of original widths
    def conv_mask(self, matrix, lens):
        device = matrix.device
        lens = np.array(lens)
        width = matrix.shape[-1]
        lens2 = (lens * (width / self.width) + 0.999).astype('int')
        matrix_new = matrix.permute(0, 3, 1, 2)  # (B, W', C, H')
        matrix_out = torch.zeros_like(matrix_new, requires_grad=True, device=device)
        for i, le in enumerate(lens2):
            if self.flip:
                matrix_out[i, -le:] = matrix_new[i, -le:]
            else:
                matrix_out[i, :le] = matrix_new[i, :le]
        return matrix_out.permute(0, 2, 3, 1)  # (B, C, H', W')



