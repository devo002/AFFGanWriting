import math
import random
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg19_bn

from utils.util import getGroupSize


# -------------------------
# Basic building blocks
# -------------------------
class Conv2dBlock(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, stride,
                 padding=0, norm='none', activation='relu', pad_type='zero',
                 transpose=False, reverse=False):
        super(Conv2dBlock, self).__init__()
        self.reverse = reverse
        self.use_bias = True

        # padding
        if transpose:
            self.pad = lambda x: x
        elif pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        else:
            raise AssertionError(f"Unsupported padding type: {pad_type}")

        # normalization
        norm_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(norm_dim)
        elif norm == 'in':
            self.norm = nn.InstanceNorm2d(norm_dim)
        elif norm == 'ln':
            raise NotImplementedError("LayerNorm not wired in this repo")
        elif norm == 'adain':
            raise NotImplementedError("AdaIN not wired in this repo")
        elif norm == 'none' or norm == 'sn':
            self.norm = None
        elif norm == 'group':
            self.norm = nn.GroupNorm(getGroupSize(norm_dim), norm_dim)
        else:
            raise AssertionError(f"Unsupported normalization: {norm}")

        # activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'logsoftmax':
            self.activation = nn.LogSoftmax(dim=1)
        elif activation == 'none':
            self.activation = None
        else:
            raise AssertionError(f"Unsupported activation: {activation}")

        # convolution
        if transpose:
            self.conv = nn.ConvTranspose2d(input_dim, output_dim, kernel_size, stride,
                                           bias=self.use_bias, padding=padding)
        elif norm == 'sn':
            raise NotImplementedError("SpectralNorm path not implemented here")
        else:
            self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride,
                                  bias=self.use_bias, padding=0)

    def forward(self, x):
        if not self.reverse:
            x = self.conv(self.pad(x))
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        if self.reverse:
            x = self.conv(self.pad(x))
        return x


class CharExtractor(nn.Module):
    def __init__(self, input_dim, dim, style_dim, num_fc=1, small=False):
        super(CharExtractor, self).__init__()
        self.conv1 = nn.Sequential(
            nn.ReLU(),
            nn.Conv1d(input_dim, dim, 3, padding=1),
            nn.GroupNorm(getGroupSize(dim), dim),
            nn.ReLU(),
            nn.Conv1d(dim, input_dim, 3, padding=1),
        )
        if small:
            self.conv2 = nn.Sequential(
                nn.ReLU(),
                nn.Conv1d(input_dim, 2 * dim, 1),
                nn.GroupNorm(getGroupSize(2 * dim), 2 * dim),
                nn.ReLU(),
            )
        else:
            self.conv2 = nn.Sequential(
                nn.ReLU(),
                nn.MaxPool1d(2),
                nn.Conv1d(input_dim, 2 * dim, 3),
                nn.GroupNorm(getGroupSize(2 * dim), 2 * dim),
                nn.ReLU(),
            )

        fc = [nn.Linear(2 * dim, 2 * dim), nn.ReLU(True)]
        for _ in range(num_fc - 1):
            fc += [nn.Linear(2 * dim, 2 * dim), nn.Dropout(0.25, True), nn.ReLU(True)]
        fc.append(nn.Linear(2 * dim, style_dim))
        self.fc = nn.Sequential(*fc)

    def forward(self, x):
        res = x
        b = x.size(0)
        x = self.conv1(x)
        x = self.conv2(x + res)
        x = F.adaptive_avg_pool1d(x, 1).view(b, -1)
        return self.fc(x)


VGG19_BN_WEIGHTS = '/home/woody/iwi5/iwi5333h/model/vgg19_bn-c79401a0.pth'


# -------------------------
# CharStyleEncoder (with VGG19-BN option)
# -------------------------
class CharStyleEncoder(nn.Module):
    def __init__(
        self, input_dim, dim, style_dim, char_dim, char_style_dim, norm, activ, pad_type,
        n_class, global_pool=False, average_found_char_style=0, num_final_g_spacing_style=1,
        num_char_fc=1, vae=False, window=6, small=False, use_vgg19bn=True,
        vgg_weights_path: str = VGG19_BN_WEIGHTS, freeze_to_block: int = 1, bn_eval_early=True
    ):
        super(CharStyleEncoder, self).__init__()

        # ---- VAE flags ----
        if vae:
            self.vae = True
            style_dim *= 2
            char_style_dim *= 2
        else:
            self.vae = False

        self.n_class = n_class
        if char_style_dim > 0:
            self.char_style_dim = char_style_dim
            self.average_found_char_style = average_found_char_style if isinstance(average_found_char_style, float) else 0.0
            self.single_style = False
        else:
            # single-style mode: per-character path disabled, but we still use a style
            assert not self.vae
            self.char_style_dim = style_dim   # IMPORTANT: used below for concat size
            char_style_dim = style_dim
            self.single_style = True

        self.window = window
        small_char_ex = window < 3

        # ---- Feature trunk (VGG19-BN or fallback) ----
        self.use_vgg19bn = use_vgg19bn
        if self.use_vgg19bn:
            self.features, in_c_for_adapter = self._build_vgg19bn(
                in_ch=input_dim, weights_path=vgg_weights_path,
                freeze_to_block=freeze_to_block, bn_eval_early=bn_eval_early
            )
            self.adapter = nn.Conv1d(in_c_for_adapter, dim, kernel_size=1)
            prepped_size = dim
            print("[S] Using VGG19-BN backbone")
            print(f"[S] 1D adapter: 512 -> {dim}")
        else:
            down = []
            down += [Conv2dBlock(input_dim, dim, 5, 1, 2, norm=norm, activation=activ, pad_type=pad_type)]
            for i in range(2):
                if i == 0 and small:
                    down += [Conv2dBlock(dim, 2 * dim, 3, 1, 1, norm=norm, activation=activ, pad_type=pad_type)]
                else:
                    down += [Conv2dBlock(dim, 2 * dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)]
                dim *= 2
                down += [Conv2dBlock(dim, dim, 3, 1, (1, 1, 0, 0), norm=norm, activation=activ, pad_type=pad_type)]
            down += [Conv2dBlock(dim, dim, 4, (2, 1), (1, 1, 0, 0), norm=norm, activation=activ, pad_type=pad_type)]
            down += [Conv2dBlock(dim, dim, 4, (2, 1), (1, 1, 0, 0), norm='none', activation='none', pad_type=pad_type)]
            self.down = nn.Sequential(*down)
            prepped_size = dim

        # ---- 1D prep trunk ----
        self.prep = nn.Sequential(
            nn.Conv1d(prepped_size + n_class, prepped_size, 5, 1, 2),
            nn.ReLU(True),
            nn.MaxPool1d(2, 2),
            nn.Conv1d(prepped_size, prepped_size, 3, 1, 1),
            nn.GroupNorm(getGroupSize(prepped_size), prepped_size),
            nn.ReLU(True),
            nn.Conv1d(prepped_size, prepped_size, 3, 1, 1),
            nn.ReLU(True),
        )

        # ---- final style heads (FIXED: in_features must include avg_char_style) ----
        # We always concatenate xr ([B, prepped_size]) with avg_char_style ([B, self.char_style_dim]).
        in_features = prepped_size + self.char_style_dim
        final_g_spacing_style = [nn.Linear(in_features, prepped_size), nn.ReLU(True)]
        for _ in range(num_final_g_spacing_style - 1):
            final_g_spacing_style += [nn.Linear(prepped_size, prepped_size), nn.Dropout(0.25, True), nn.ReLU(True)]
        if self.single_style:
            final_g_spacing_style.append(nn.Linear(prepped_size, style_dim))
        else:
            final_g_spacing_style.append(nn.Linear(prepped_size, style_dim + char_style_dim))
        self.final_g_spacing_style = nn.Sequential(*final_g_spacing_style)

        # ---- character-specific heads ----
        self.char_extractor = nn.ModuleList()
        if not self.single_style:
            self.fill_pred = nn.ModuleList()

        char_in_dim = prepped_size  # equals `dim` when VGG is used (after adapter)
        for _ in range(n_class):
            self.char_extractor.append(CharExtractor(char_in_dim, char_dim, self.char_style_dim, num_char_fc, small_char_ex))
            if not self.single_style:
                self.fill_pred.append(
                    nn.Sequential(
                        nn.Linear(self.char_style_dim, 2 * self.char_style_dim),
                        nn.ReLU(True),
                        nn.Linear(2 * self.char_style_dim, self.char_style_dim * n_class),
                    )
                )

    # ---------------- VGG builder ----------------
    def _build_vgg19bn_backbone(self):
        return vgg19_bn()

    def _build_vgg19bn(self, in_ch: int, weights_path: str, freeze_to_block: int = 1, bn_eval_early: bool = True):
        vgg = self._build_vgg19bn_backbone()
        # load weights
        try:
            sd = torch.load(weights_path, map_location='cpu')
            vgg.load_state_dict(sd)
            print("[S] Loaded VGG19-BN weights from:", weights_path)
        except Exception as e:
            print(f"[S] Warning: failed to load VGG19-BN weights from {weights_path}: {e}")
            print("[S] Proceeding with randomly initialized VGG19-BN.")

        # adapt first conv to in_ch
        old0 = vgg.features[0]
        new0 = nn.Conv2d(in_ch, old0.out_channels, kernel_size=old0.kernel_size,
                         stride=old0.stride, padding=old0.padding, bias=False)
        with torch.no_grad():
            w = old0.weight.data  # [64, 3, 3, 3]
            if in_ch == 1:
                new0.weight.copy_(w.mean(dim=1, keepdim=True))
            elif in_ch == 3:
                new0.weight.copy_(w)
            else:
                # for unusual in_ch we leave random init
                pass
        vgg.features[0] = new0

        features = vgg.features

        # freeze early blocks if desired (indices for VGG19-BN feature seq)
        block_ends = [6, 13, 26, 39, 52]
        if freeze_to_block >= 0:
            freeze_upto = block_ends[min(max(freeze_to_block, 0), len(block_ends) - 1)]
            for i, m in enumerate(features):
                if i <= freeze_upto:
                    for p in m.parameters():
                        p.requires_grad = False
                    if isinstance(m, nn.BatchNorm2d):
                        m.eval()

        # keep early BN in eval mode to stabilize tiny batches
        if bn_eval_early:
            for i, m in enumerate(features):
                if isinstance(m, nn.BatchNorm2d) and i <= block_ends[1]:
                    m.eval()

        return features, 512

    # ---------------- forward ----------------
    def forward(self, x, recog):
        b = x.size(0)

        # feature extraction -> B×C×W
        if self.use_vgg19bn:
            f2d = self.features(x)     # B×512×H'×W'
            x = f2d.mean(dim=2)        # vertical GAP -> B×512×W'
            x = self.adapter(x)        # B×dim×W'
        else:
            x = self.down(x)           # B×C×1×W
            x = x.view(b, x.size(1), x.size(3))

        # width align with recognizer logits
        diff = x.size(2) - recog.size(2)
        if diff > 0:
            recog = F.pad(recog, (diff // 2, (diff // 2) + diff % 2), mode='replicate')
        elif diff < 0:
            x = F.pad(x, (-diff // 2, (-diff // 2) + (-diff) % 2), mode='replicate')

        # character-aligned pooling
        recogPred = torch.argmax(recog, dim=1)
        fill_styles = [[] for _ in range(b)]
        found_chars_style = {}

        if self.single_style:
            b_sum = torch.zeros(b, device=x.device, dtype=x.dtype)
            total_style = torch.zeros(b, self.char_style_dim, device=x.device, dtype=x.dtype)

        for char_n in range(1, self.n_class):
            locs = recogPred == char_n
            if locs.any():
                patches = []
                b_weight = []
                for bi in range(b):
                    horz_cents = locs[bi].nonzero()
                    for hc in horz_cents:
                        c = hc.item()
                        left = max(0, c - self.window)
                        pad_left = left - (c - self.window)
                        right = min(x.size(2) - 1, c + self.window)
                        pad_right = (c + self.window) - right
                        wind = x[bi:bi + 1, :, left:right + 1]
                        if pad_left > 0 or pad_right > 0:
                            wind = F.pad(wind, (pad_left, pad_right))
                        assert wind.size(2) == self.window * 2 + 1
                        patches.append(wind)
                        b_weight.append((bi, math.exp(recog[bi, char_n, c].item())))
                if not patches:
                    continue
                patches = torch.cat(patches, dim=0)

                char_styles = self.char_extractor[char_n](patches)

                if self.single_style:
                    for i, (bi, score) in enumerate(b_weight):
                        total_style[bi] += score * char_styles[i]
                        b_sum[bi] += score
                else:
                    b_sum_local = defaultdict(lambda: 0.0)
                    found_chars_style[char_n] = defaultdict(
                        lambda: torch.zeros(self.char_style_dim, device=x.device, dtype=x.dtype)
                    )
                    for i, (bi, score) in enumerate(b_weight):
                        found_chars_style[char_n][bi] += score * char_styles[i]
                        b_sum_local[bi] += score
                    bs_of_interest = list(found_chars_style[char_n].keys())
                    for bi in bs_of_interest:
                        assert b_sum_local[bi] != 0
                        found_chars_style[char_n][bi] /= b_sum_local[bi]
                    char_style_batch = torch.stack([found_chars_style[char_n][bi] for bi in bs_of_interest], dim=0)
                    fill_pred = self.fill_pred[char_n](char_style_batch)
                    for i, bi in enumerate(bs_of_interest):
                        fill_styles[bi].append(fill_pred[i])

        if not self.single_style:
            fill_bs = []
            for bi in range(b):
                if len(fill_styles[bi]) > 0:
                    fill_bs.append(torch.stack(fill_styles[bi], dim=0).mean(dim=0))
                else:
                    fill_bs.append(torch.zeros(self.n_class * self.char_style_dim, device=x.device, dtype=x.dtype))
            all_char_style = [list(torch.chunk(styles, self.n_class, dim=0)) for styles in fill_bs]

            for char_n, char_style in found_chars_style.items():
                for bi in char_style:
                    if self.average_found_char_style > 0:
                        all_char_style[bi][char_n] = (
                            char_style[bi] * (1 - self.average_found_char_style)
                            + all_char_style[bi][char_n] * (self.average_found_char_style)
                        )
                    elif self.average_found_char_style < 0:
                        mix = random.random() * (-self.average_found_char_style) if self.training else 0.1
                        all_char_style[bi][char_n] = char_style[bi] * (1 - mix) + all_char_style[bi][char_n] * (mix)
                    else:
                        all_char_style[bi][char_n] = char_style[bi]
            all_char_style = [torch.stack(styles, dim=0) for styles in all_char_style]
            all_char_style = torch.stack(all_char_style, dim=0)  # B×n_class×char_style_dim
            avg_char_style = all_char_style.sum(dim=1) / self.n_class
        else:
            avg_char_style = torch.where(b_sum[..., None] != 0, total_style / b_sum[..., None], total_style)

        # global + avg-char concat
        xr = torch.cat((F.relu(x), recog), dim=1)  # B×(dim+n_class)×W
        xr = self.prep(xr)
        xr = F.adaptive_avg_pool1d(xr, 1).view(b, -1)

        comb_style = torch.cat((xr, avg_char_style), dim=1)  # size: prepped_size + self.char_style_dim
        comb_style = self.final_g_spacing_style(comb_style)

        if self.single_style:
            return comb_style

        g_style = comb_style[:, self.char_style_dim:]
        spacing_style = comb_style[:, :self.char_style_dim]

        if self.vae:
            g_mu, g_log_sigma = g_style.chunk(2, dim=1)
            spacing_mu, spacing_log_sigma = spacing_style.chunk(2, dim=1)
            all_char_mu, all_char_log_sigma = all_char_style.chunk(2, dim=2)
            return g_mu, g_log_sigma, spacing_mu, spacing_log_sigma, all_char_mu, all_char_log_sigma
        else:
            return g_style, spacing_style, all_char_style
