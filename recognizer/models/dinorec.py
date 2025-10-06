import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from dinomodel import ImageEncoderDINOv2   # use `from .dinomodel ...` if this is a package

class RecDecoderDINOv2(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        height: int,
        width: int,
        bgru: bool = True,
        step: int | None = None,
        flip: bool = False,
        *,
        use_lstm: bool = False,
        drop_out: bool = False,                 # enable spatial dropout on CNN features
        sum_up: bool = False,

        repo_dir: str = "/home/woody/iwi5/iwi5333h/facebookresearch_dinov2_main",
        #ckpt_path: str | None = "/home/woody/iwi5/iwi5333h/model/dinov2_vitl14_pretrain.pth",
        ckpt_path = "/home/woody/iwi5/iwi5333h/model/dinov2_vits14_pretrain.pth",
        arch: str = "vits14",
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.height = height
        self.width = width
        self.bi = bgru
        self.step = step
        self.flip = flip

        self.n_layers = 2
        self.dropout_p = 0.5                   # RNN inter-layer dropout prob
        self.use_lstm = use_lstm
        self.drop_out = drop_out               # flag for 2D feature dropout
        self.sum_up = sum_up

        # ---- ViT feature extractor -> (B, 512, H/16, W/16) on the last map ----
        self.backbone = ImageEncoderDINOv2(
            repo_dir=repo_dir,
            arch=arch,
            ckpt_path=ckpt_path,
            in_channels=3,
            final_size=(height // 16, width // 16),
            #tap_blocks=[4, 8, 16, 23] if "vitl" in arch else None
            tap_blocks=None
        )

        if self.drop_out:
            self.layer_dropout = nn.Dropout2d(p=0.5)

        # Number of features per time step before the RNN
        self.n_feat_per_step = (self.height // 16) * 512

        # Optional STEP grouping (concat then project)
        if self.step is not None:
            self.output_proj = nn.Linear(self.n_feat_per_step * self.step, self.n_feat_per_step)

        RNN = nn.LSTM if use_lstm else nn.GRU
        rnn_input_size = self.n_feat_per_step
        rnn_dropout = self.dropout_p if self.n_layers > 1 else 0.0

        if self.bi:
            self.rnn = RNN(rnn_input_size, self.hidden_size, self.n_layers,
                           dropout=rnn_dropout, bidirectional=True)
            if 'SUM_UP' in globals() and self.sum_up:
                self.enc_out_merge = lambda x: x[:, :, :x.shape[-1] // 2] + x[:, :, x.shape[-1] // 2:]
                self.enc_hidden_merge = lambda x: (x[0] + x[1]).unsqueeze(0)
        else:
            self.rnn = RNN(rnn_input_size, self.hidden_size, self.n_layers,
                           dropout=rnn_dropout, bidirectional=False)

    def forward(self, in_data, in_data_len, hidden=None):
        """
        in_data: (B, 3, H, W)
        in_data_len: CPU 1D tensor with original widths (pixels)
        """
        B = in_data.size(0)

        maps = self.backbone(in_data)      # list of [B, 512, h_i, w_i]
        feat = maps[-1]                    # (B, 512, H/16, W/16)

        if self.drop_out and self.training:
            feat = self.layer_dropout(feat)

        # (B, C, Hs, Ws) -> (Ws, B, Hs*C)
        feat = feat.permute(0, 2, 3, 1).contiguous()   # (B, Hs, Ws, C)
        Hs, Ws = feat.shape[1], feat.shape[2]
        feat = feat.view(B, Hs * 512, Ws)              # (B, Hs*C, Ws)
        out = feat.permute(2, 0, 1).contiguous()       # (t=Ws, B, Hs*C)

        # Optional STEP grouping
        if self.step is not None and self.step > 1:
            t, b, f = out.shape
            t_trim = (t // self.step) * self.step
            if t_trim != t:
                out = out[:t_trim]
                t = t_trim
            out = out.view(self.step, t // self.step, b, f)     # (step, t//step, B, F)
            out = out.permute(1, 2, 0, 3).contiguous()          # (t//step, B, step, F)
            out = out.view(t // self.step, b, f * self.step)    # (t//step, B, step*F)
            out = self.output_proj(out)                         # (t//step, B, F)
            Ws_eff = t // self.step
        else:
            Ws_eff = out.size(0)

        # Scale src_len to recognizer time steps (width // 16, with STEP if used)
        step_div = (self.step if self.step else 1)
        denom = (self.width // step_div) // 16
        src_len = (in_data_len.numpy() * (Ws_eff / denom) + 0.999).astype('int')

        # Pack, RNN, unpack
        packed = pack_padded_sequence(out, src_len.tolist(), batch_first=False, enforce_sorted=False)
        output, hidden = self.rnn(packed, hidden)
        output, _ = pad_packed_sequence(output, batch_first=False)

        if self.bi and 'SUM_UP' in globals() and self.sum_up:
            output = self.enc_out_merge(output)

        # LSTM returns tuple; GRU returns Tensor
        if isinstance(hidden, tuple):
            h_n, _ = hidden
        else:
            h_n = hidden

        odd_idx = [1, 3, 5, 7, 9, 11][:self.n_layers]
        final_hidden = h_n[odd_idx]

        return output, final_hidden
