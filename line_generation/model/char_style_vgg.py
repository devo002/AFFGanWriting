import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg19_bn

class VGG19BNBackbone1D(nn.Module):
    """
    Loads VGG19-BN weights from a local .pth, adapts the first conv to 1-channel,
    returns a B×out_dim×W sequence via vertical average pooling + 1D adapter.
    """
    def __init__(self, weights_path: str, in_ch: int = 1, out_dim: int = 256,
                 freeze_to_block: int = 1, bn_eval_early: bool = True):
        """
        freeze_to_block: -1 (no freeze) or 0..4 (freeze up to that VGG block)
          VGG19-BN blocks (features indices roughly):
            block1: 0..6, block2: 7..13, block3: 14..26, block4: 27..39, block5: 40..52
        """
        super().__init__()

        # 1) Build vanilla VGG19-BN and load weights
        vgg = vgg19_bn()  # no internet used
        sd = torch.load(weights_path, map_location="cpu")
        vgg.load_state_dict(sd)

        # 2) Adapt first conv to 1-channel
        #    - create new conv with same hyperparams, copy averaged RGB weights
        old0 = vgg.features[0]  # Conv2d(3, 64, kernel_size=3, padding=1, bias=False)
        new0 = nn.Conv2d(in_ch, old0.out_channels,
                         kernel_size=old0.kernel_size, stride=old0.stride,
                         padding=old0.padding, bias=False)
        with torch.no_grad():
            w = old0.weight.data  # [64, 3, 3, 3]
            if in_ch == 1:
                new0.weight.copy_(w.mean(dim=1, keepdim=True))  # average RGB -> 1ch
            else:
                # in case you ever pass in_ch=3, keep original weights
                new0.weight.copy_(w)
        vgg.features[0] = new0

        self.features = vgg.features  # up to conv5_4 + pool5 (index ~52)

        # 3) Optional freezing of early blocks (helps stability on small batches)
        block_ends = [6, 13, 26, 39, 52]  # inclusive end indices of each VGG block
        if freeze_to_block >= 0:
            freeze_upto = block_ends[min(freeze_to_block, len(block_ends)-1)]
            for i, m in enumerate(self.features):
                if i <= freeze_upto:
                    for p in m.parameters():
                        p.requires_grad = False
                    # keep BN in eval for frozen layers
                    if isinstance(m, nn.BatchNorm2d):
                        m.eval()

        # 4) Optionally set early BNs to eval (stabilizes tiny batch sizes)
        if bn_eval_early:
            for i, m in enumerate(self.features):
                if isinstance(m, nn.BatchNorm2d) and i <= block_ends[1]:  # blocks 0..1
                    m.eval()

        # 5) 1D adapter to your encoder dim (VGG last block is 512 channels)
        self.adapter = nn.Conv1d(512, out_dim, kernel_size=1)

    def forward(self, x):  # x: B×1×H×W
        f = self.features(x)            # B×512×H'×W'
        seq = f.mean(dim=2)             # vertical GAP -> B×512×W'
        seq = self.adapter(seq)         # B×out_dim×W'
        return seq
