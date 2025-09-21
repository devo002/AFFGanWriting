from torch import nn
#from torch.autograd import Variable
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np
#from models.vgg_tro_channel1 import vgg16_bn
from recognizer.models.vgg_tro_channel3 import vgg16_bn, vgg19_bn
#from Resnet18 import ResNet18
#from torchvision.models import efficientnet_v2_s
#from torchvision.models import efficientnet_v2_l
#from torchvision.models import resnet50,resnet18, convnext_tiny
#from torchvision.models import vit_b_16, ViT_B_16_Weights
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torchvision.models import efficientnet_b7, EfficientNet_B7_Weights
#from torchvision.models import vit_b_16

SUM_UP = True
DROP_OUT = True
LSTM = False  # Set True if you want to use LSTM instead of GRU

#torch.cuda.set_device(1)
cuda = torch.device('cuda')

DROP_OUT = True
LSTM = False
SUM_UP = True
PRE_TRAIN_VGG = True


# weights_path = "/home/woody/iwi5/iwi5333h/model/efficientnet_v2_l-59c71312.pth"


# class EfficientNetEncoder(nn.Module):
#     def __init__(self, hidden_size, height, width, bgru=True, step=None, flip=False, weight_path=None):
#         super(EfficientNetEncoder, self).__init__()
#         self.hidden_size = hidden_size
#         self.height = height
#         self.width = width
#         self.bi = bgru
#         self.step = step
#         self.flip = flip
#         self.n_layers = 2
#         self.dropout = 0.5

#         # Load EfficientNetV2S backbone
#         self.backbone = efficientnet_v2_l(weights=None)
        
#         # Load pretrained weights if path is provided
#         if weight_path:
#             try:
#                 state_dict = torch.load(weight_path, map_location="cpu")
#                 # Handle possible state_dict formats
#                 if 'state_dict' in state_dict:  # If weights are saved with model wrapper
#                     state_dict = state_dict['state_dict']
#                 # Remove any 'module.' prefix from parameter names if present
#                 state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
#                 self.backbone.load_state_dict(state_dict)
#                 print(f"Successfully loaded EfficientNetV2 weights from: {weight_path}")
#             except Exception as e:
#                 print(f"Error loading weights from {weight_path}: {str(e)}")
#                 print("Using randomly initialized weights instead")

#         # Remove the classifier head
#         self.features = self.backbone.features
        
#         # For EfficientNetV2S, the final feature map is [B, 1280, H/32, W/32]
#         # We'll add a projection to match expected dimensions
#         self.feature_height = height // 32
#         self.feature_proj = nn.Sequential(
#             nn.Conv2d(1280, 512, kernel_size=1),  # Reduce channels to match VGG
#             nn.BatchNorm2d(512),
#             nn.ReLU(inplace=True),
#             nn.AdaptiveAvgPool2d((self.feature_height, None))  # Maintain consistent height
#         )
        
#         if DROP_OUT:
#             self.layer_dropout = nn.Dropout2d(p=0.5)
            
#         if self.step is not None:
#             self.output_proj = nn.Linear(self.feature_height*512*self.step, self.feature_height*512)

#         if LSTM:
#             RNN = nn.LSTM
#         else:
#             RNN = nn.GRU

#         # RNN input size is feature_height * 512
#         rnn_input_size = self.feature_height * 512
        
#         if self.bi:
#             self.rnn = RNN(rnn_input_size, self.hidden_size, self.n_layers,
#                          dropout=self.dropout, bidirectional=True)
#             if SUM_UP:
#                 self.enc_out_merge = lambda x: x[:,:,:x.shape[-1]//2] + x[:,:,x.shape[-1]//2:]
#                 self.enc_hidden_merge = lambda x: (x[0] + x[1]).unsqueeze(0)
#         else:
#             self.rnn = RNN(rnn_input_size, self.hidden_size, self.n_layers,
#                          dropout=self.dropout, bidirectional=False)

#     def forward(self, in_data, in_data_len, hidden=None):
#         batch_size = in_data.shape[0]
        
#         # Convert grayscale to RGB by repeating channels if needed
#         if in_data.shape[1] == 1:
#             in_data = in_data.repeat(1, 3, 1, 1)
        
#         # Extract features
#         out = self.features(in_data)  # [B, 1280, H/32, W/32]
#         out = self.feature_proj(out)  # [B, 512, feature_height, W']
        
#         if DROP_OUT and self.training:
#             out = self.layer_dropout(out)
            
#         # Prepare for RNN - [W', B, feature_height*512]
#         out = out.permute(3, 0, 2, 1)  # [W', B, feature_height, 512]
#         out = out.reshape(out.shape[0], batch_size, -1)  # [W', B, feature_height*512]
        
#         if self.step is not None:
#             time_step = out.shape[0]
#             out_short = torch.zeros(time_step//self.step, batch_size, 
#                                  self.feature_height*512*self.step,
#                                  device=out.device)
#             for i in range(time_step//self.step):
#                 part_out = [out[j] for j in range(i*self.step, (i+1)*self.step)]
#                 out_short[i] = torch.cat(part_out, dim=1)
#             out = self.output_proj(out_short)

#         # Handle variable length sequences
#         width = out.shape[0]
#         src_len = in_data_len.cpu().numpy()*(width/self.width)
#         src_len = src_len + 0.999  # in case of 0 length value from float to int
#         src_len = src_len.astype('int')
#         out = pack_padded_sequence(out, src_len.tolist(), batch_first=False)
        
#         # RNN processing
#         output, hidden = self.rnn(out, hidden)
#         output, _ = pad_packed_sequence(output, batch_first=False)
        
#         if self.bi and SUM_UP:
#             output = self.enc_out_merge(output)
            
#         # Handle hidden states
#         if isinstance(hidden, tuple):  # LSTM case
#             hidden = hidden[0]
#         odd_idx = [1, 3, 5, 7, 9, 11]
#         hidden_idx = odd_idx[:self.n_layers]
#         final_hidden = hidden[hidden_idx]
        
#         return output, final_hidden

#     def conv_mask(self, matrix, lens):
#         lens = np.array(lens)
#         width = matrix.shape[-1]
#         lens2 = lens * (width / self.width)
#         lens2 = lens2 + 0.999  # in case le == 0
#         lens2 = lens2.astype('int')
#         matrix_new = matrix.permute(0, 3, 1, 2)  # b, w, c, h
#         matrix_out = torch.zeros(matrix_new.shape, device=matrix.device)
#         for i, le in enumerate(lens2):
#             if self.flip:
#                 matrix_out[i, -le:] = matrix_new[i, -le:]
#             else:
#                 matrix_out[i, :le] = matrix_new[i, :le]
#         matrix_out = matrix_out.permute(0, 2, 3, 1)  # b, c, h, w
#         return matrix_out


# class EfficientNetB7Encoder(nn.Module):
#     def __init__(self, hidden_size, height, width, bgru=True, step=None, flip=False, weight_path=None):
#         super().__init__()
#         self.hidden_size = hidden_size
#         self.height = height
#         self.width = width
#         self.bi = bgru
#         self.step = step
#         self.flip = flip
#         self.n_layers = 2
#         self.dropout = 0.5

#         # 1) Swap backbone here
#         self.backbone = efficientnet_b7(weights=None)   # <- change made

#         # 2) (Optional) try loading weights (must match B7!)
#         if weight_path:
#             try:
#                 state_dict = torch.load(weight_path, map_location="cpu")
#                 if 'state_dict' in state_dict:
#                     state_dict = state_dict['state_dict']
#                 state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
#                 self.backbone.load_state_dict(state_dict, strict=True)  # keep strict=True for safety
#                 print(f"Loaded EfficientNet-B7 weights from: {weight_path}")
#             except Exception as e:
#                 print(f"Error loading B7 weights from {weight_path}: {e}")
#                 print("Using randomly initialized weights instead")

#         self.features = self.backbone.features

#         # 3) Infer feature channels and downsample factor dynamically
#         with torch.no_grad():
#             dummy = torch.zeros(1, 3, self.height, self.width)
#             feat = self.features(dummy)  # [1, C_out, H', W']
#             C_out = feat.shape[1]
#             H_out = feat.shape[2]

#         self.feature_height = H_out  # don’t assume //32

#         # 4) Use detected C_out instead of hard-coded 1280
#         self.feature_proj = nn.Sequential(
#             nn.Conv2d(C_out, 512, kernel_size=1),
#             nn.BatchNorm2d(512),
#             nn.ReLU(inplace=True),
#             nn.AdaptiveAvgPool2d((self.feature_height, None))  # keep height consistent
#         )

#         if DROP_OUT:
#             self.layer_dropout = nn.Dropout2d(p=0.5)

#         if self.step is not None:
#             self.output_proj = nn.Linear(self.feature_height*512*self.step, self.feature_height*512)

#         RNN = nn.LSTM if LSTM else nn.GRU
#         rnn_input_size = self.feature_height * 512

#         if self.bi:
#             self.rnn = RNN(rnn_input_size, self.hidden_size, self.n_layers,
#                            dropout=self.dropout, bidirectional=True)
#             if SUM_UP:
#                 self.enc_out_merge = lambda x: x[:, :, :x.shape[-1]//2] + x[:, :, x.shape[-1]//2:]
#                 self.enc_hidden_merge = lambda x: (x[0] + x[1]).unsqueeze(0)
#         else:
#             self.rnn = RNN(rnn_input_size, self.hidden_size, self.n_layers,
#                            dropout=self.dropout, bidirectional=False)
            
            
#     def forward(self, in_data, in_data_len, hidden=None):
#         batch_size = in_data.shape[0]
        
#         # Convert grayscale to RGB by repeating channels if needed
#         if in_data.shape[1] == 1:
#             in_data = in_data.repeat(1, 3, 1, 1)
        
#         # Extract features
#         out = self.features(in_data)  # [B, 1280, H/32, W/32]
#         out = self.feature_proj(out)  # [B, 512, feature_height, W']
        
#         if DROP_OUT and self.training:
#             out = self.layer_dropout(out)
            
#         # Prepare for RNN - [W', B, feature_height*512]
#         out = out.permute(3, 0, 2, 1)  # [W', B, feature_height, 512]
#         out = out.reshape(out.shape[0], batch_size, -1)  # [W', B, feature_height*512]
        
#         if self.step is not None:
#             time_step = out.shape[0]
#             out_short = torch.zeros(time_step//self.step, batch_size, 
#                                  self.feature_height*512*self.step,
#                                  device=out.device)
#             for i in range(time_step//self.step):
#                 part_out = [out[j] for j in range(i*self.step, (i+1)*self.step)]
#                 out_short[i] = torch.cat(part_out, dim=1)
#             out = self.output_proj(out_short)

#         # Handle variable length sequences
#         width = out.shape[0]
#         src_len = in_data_len.cpu().numpy()*(width/self.width)
#         src_len = src_len + 0.999  # in case of 0 length value from float to int
#         src_len = src_len.astype('int')
#         out = pack_padded_sequence(out, src_len.tolist(), batch_first=False)
        
#         # RNN processing
#         output, hidden = self.rnn(out, hidden)
#         output, _ = pad_packed_sequence(output, batch_first=False)
        
#         if self.bi and SUM_UP:
#             output = self.enc_out_merge(output)
            
#         # Handle hidden states
#         if isinstance(hidden, tuple):  # LSTM case
#             hidden = hidden[0]
#         odd_idx = [1, 3, 5, 7, 9, 11]
#         hidden_idx = odd_idx[:self.n_layers]
#         final_hidden = hidden[hidden_idx]
        
#         return output, final_hidden

#     def conv_mask(self, matrix, lens):
#         lens = np.array(lens)
#         width = matrix.shape[-1]
#         lens2 = lens * (width / self.width)
#         lens2 = lens2 + 0.999  # in case le == 0
#         lens2 = lens2.astype('int')
#         matrix_new = matrix.permute(0, 3, 1, 2)  # b, w, c, h
#         matrix_out = torch.zeros(matrix_new.shape, device=matrix.device)
#         for i, le in enumerate(lens2):
#             if self.flip:
#                 matrix_out[i, -le:] = matrix_new[i, -le:]
#             else:
#                 matrix_out[i, :le] = matrix_new[i, :le]
#         matrix_out = matrix_out.permute(0, 2, 3, 1)  # b, c, h, w
#         return matrix_out



#import torch
#import torch.nn as nn
#from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
#from torchvision.models import vgg19_bn, VGG19_BN_Weights
# assumes global flags exist somewhere in your code:
# DROP_OUT, LSTM, SUM_UP

# class VGG19Encoder(nn.Module):
#     def __init__(self, hidden_size, height, width, bgru=True, step=None, flip=False,
#                  weight_path=None, use_imagenet=False):
#         super().__init__()
#         self.hidden_size = hidden_size
#         self.height = height
#         self.width = width
#         self.bi = bgru
#         self.step = step
#         self.flip = flip
#         self.n_layers = 2
#         self.dropout = 0.5

#         # 1) Backbone: VGG19 with batch-norm
#         if use_imagenet:
#             self.backbone = vgg19_bn(weights=VGG19_BN_Weights.IMAGENET1K_V1)
#         else:
#             self.backbone = vgg19_bn(weights=None)

#         # 2) (Optional) load your own weights (must match VGG19-BN)
#         if weight_path:
#             try:
#                 sd = torch.load(weight_path, map_location="cpu")
#                 if "state_dict" in sd:
#                     sd = sd["state_dict"]
#                 sd = {k.replace("module.", ""): v for k, v in sd.items()}
#                 self.backbone.load_state_dict(sd, strict=True)
#                 print(f"[VGG19Encoder] Loaded weights from {weight_path}")
#             except Exception as e:
#                 print(f"[VGG19Encoder] Could not load weights: {e}")

#         # VGG exposes conv blocks as .features
#         self.features = self.backbone.features  # -> [B, 512, H', W']

#         # 3) Infer output channels/height at runtime (works for any backbone)
#         with torch.no_grad():
#             dummy = torch.zeros(1, 3, self.height, self.width)
#             feat = self.features(dummy)
#             C_out = feat.shape[1]          # 512 for VGG19
#             H_out = feat.shape[2]          # typically height//32

#         self.feature_height = H_out

#         # 4) Project to 512 channels and keep feature_height; width is free
#         self.feature_proj = nn.Sequential(
#             nn.Conv2d(C_out, 512, kernel_size=1),
#             nn.BatchNorm2d(512),
#             nn.ReLU(inplace=True),
#             nn.AdaptiveAvgPool2d((self.feature_height, None))  # keep height
#         )

#         if DROP_OUT:
#             self.layer_dropout = nn.Dropout2d(p=0.5)

#         if self.step is not None:
#             self.output_proj = nn.Linear(self.feature_height * 512 * self.step,
#                                          self.feature_height * 512)

#         # 5) RNN stack
#         RNN = nn.LSTM if LSTM else nn.GRU
#         rnn_input_size = self.feature_height * 512
#         if self.bi:
#             self.rnn = RNN(rnn_input_size, self.hidden_size, self.n_layers,
#                            dropout=self.dropout, bidirectional=True)
#             if SUM_UP:
#                 self.enc_out_merge = lambda x: x[:, :, :x.shape[-1]//2] + x[:, :, x.shape[-1]//2:]
#                 self.enc_hidden_merge = lambda x: (x[0] + x[1]).unsqueeze(0)
#         else:
#             self.rnn = RNN(rnn_input_size, self.hidden_size, self.n_layers,
#                            dropout=self.dropout, bidirectional=False)

#     def forward(self, in_data, in_data_len, hidden=None):
#         B = in_data.shape[0]

#         # VGG expects 3ch; repeat grayscale if needed
#         if in_data.shape[1] == 1:
#             in_data = in_data.repeat(1, 3, 1, 1)

#         # If you used ImageNet pretrained weights, consider normalizing here:
#         # mean = torch.tensor([0.485, 0.456, 0.406], device=in_data.device).view(1,3,1,1)
#         # std  = torch.tensor([0.229, 0.224, 0.225], device=in_data.device).view(1,3,1,1)
#         # in_data = (in_data + 1) / 2  # if inputs were normalized to mean=0,std=1 with 0.5/0.5 earlier
#         # in_data = (in_data - mean) / std

#         out = self.features(in_data)      # [B, C, H', W']
#         out = self.feature_proj(out)      # [B, 512, Hf, W']

#         if DROP_OUT and self.training:
#             out = self.layer_dropout(out)

#         # [W', B, Hf*512]
#         out = out.permute(3, 0, 2, 1).contiguous()
#         out = out.view(out.shape[0], B, -1)

#         # optional step aggregation
#         if self.step is not None:
#             T = out.shape[0]
#             out_short = torch.zeros(T//self.step, B, self.feature_height*512*self.step,
#                                     device=out.device)
#             for i in range(T//self.step):
#                 part = [out[j] for j in range(i*self.step, (i+1)*self.step)]
#                 out_short[i] = torch.cat(part, dim=1)
#             out = self.output_proj(out_short)

#         # variable-length packing
#         width = out.shape[0]
#         src_len = in_data_len.cpu().numpy() * (width / self.width)
#         src_len = (src_len + 0.999).astype('int')
#         out = pack_padded_sequence(out, src_len.tolist(), batch_first=False, enforce_sorted=False)

#         output, hidden = self.rnn(out, hidden)
#         output, _ = pad_packed_sequence(output, batch_first=False)

#         if self.bi and SUM_UP:
#             output = self.enc_out_merge(output)

#         # handle hidden (GRU or LSTM cell state)
#         if isinstance(hidden, tuple):
#             hidden = hidden[0]
#         odd_idx = [1, 3, 5, 7, 9, 11][:self.n_layers]
#         final_hidden = hidden[odd_idx]
#         return output, final_hidden

#     def conv_mask(self, matrix, lens):
#         lens = np.array(lens)
#         width = matrix.shape[-1]
#         lens2 = lens * (width / self.width)
#         lens2 = (lens2 + 0.999).astype('int')
#         matrix_new = matrix.permute(0, 3, 1, 2)  # b,w,c,h
#         matrix_out = torch.zeros_like(matrix_new)
#         for i, le in enumerate(lens2):
#             if self.flip:
#                 matrix_out[i, -le:] = matrix_new[i, -le:]
#             else:
#                 matrix_out[i, :le] = matrix_new[i, :le]
#         return matrix_out.permute(0, 2, 3, 1)



# class ResNet50Encoder(nn.Module):
#     def __init__(self, hidden_size, height, width, bgru=True, step=None, flip=False, weight_path=None):
#         super(ResNet50Encoder, self).__init__()
#         self.hidden_size = hidden_size
#         self.height = height
#         self.width = width
#         self.bi = bgru
#         self.step = step
#         self.flip = flip
#         self.n_layers = 2
#         self.dropout = 0.5

#         # Load ResNet50 backbone
#         self.model = resnet50(weights=None)
        
#         if weight_path:
#             try:
#                 state_dict = torch.load(weight_path, map_location="cpu")
#                 if 'state_dict' in state_dict:
#                     state_dict = state_dict['state_dict']
#                 state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
#                 self.model.load_state_dict(state_dict)
#                 print(f"Successfully loaded ResNet50 weights from: {weight_path}")
#             except Exception as e:
#                 print(f"Error loading weights from {weight_path}: {str(e)}")
#                 print("Using randomly initialized weights instead")

#         # Remove the classifier head (avgpool + fc)
#         self.features = nn.Sequential(*list(self.model.children())[:-2])  # Output: [B, 2048, H/32, W/32]

        
        
        
#         # Project to 512 channels and fixed height
#         self.feature_height = height // 32   # change to 32 for layer 4
        
#         #use for resnet18
#         # self.feature_proj = nn.Sequential(
#         #      nn.Conv2d(512, 512, kernel_size=1),  # change 2048 -> 512 for ResNet18
#         #      nn.BatchNorm2d(512),
#         #      nn.ReLU(inplace=True),
#         #      nn.AdaptiveAvgPool2d((self.feature_height, None))
#         #         )

#         self.feature_proj = nn.Sequential(
#             nn.Conv2d(2048, 512, kernel_size=1), # change to 2048 for layer 4
#             nn.BatchNorm2d(512),
#             nn.ReLU(inplace=True),
#             nn.AdaptiveAvgPool2d((self.feature_height, None))
#         )

#         if DROP_OUT:
#             self.layer_dropout = nn.Dropout2d(p=0.5)

#         if self.step is not None:
#             self.output_proj = nn.Linear(self.feature_height*512*self.step, self.feature_height*512)

#         RNN = nn.LSTM if LSTM else nn.GRU
#         rnn_input_size = self.feature_height * 512

#         if self.bi:
#             self.rnn = RNN(rnn_input_size, self.hidden_size, self.n_layers,
#                            dropout=self.dropout, bidirectional=True)
#             if SUM_UP:
#                 self.enc_out_merge = lambda x: x[:,:,:x.shape[-1]//2] + x[:,:,x.shape[-1]//2:]
#                 self.enc_hidden_merge = lambda x: (x[0] + x[1]).unsqueeze(0)
#         else:
#             self.rnn = RNN(rnn_input_size, self.hidden_size, self.n_layers,
#                            dropout=self.dropout, bidirectional=False)

#     def forward(self, in_data, in_data_len, hidden=None):
#         batch_size = in_data.shape[0]

#         if in_data.shape[1] == 1:
#             in_data = in_data.repeat(1, 3, 1, 1)

#         out = self.features(in_data)                    # [B, 2048, H/32, W/32]
#         out = self.feature_proj(out)                    # [B, 512, feature_height, W']

#         if DROP_OUT and self.training:
#             out = self.layer_dropout(out)

#         out = out.permute(3, 0, 2, 1)                   # [W', B, feature_height, 512]
#         out = out.reshape(out.shape[0], batch_size, -1) # [W', B, feature_height*512]

#         if self.step is not None:
#             time_step = out.shape[0]
#             out_short = torch.zeros(time_step//self.step, batch_size, 
#                                  self.feature_height*512*self.step,
#                                  device=out.device)
#             for i in range(time_step//self.step):
#                 part_out = [out[j] for j in range(i*self.step, (i+1)*self.step)]
#                 out_short[i] = torch.cat(part_out, dim=1)
#             out = self.output_proj(out_short)

#         width = out.shape[0]
#         src_len = in_data_len.detach().cpu().numpy() * (width / self.width)
#         src_len = src_len + 0.999
#         src_len = src_len.astype('int')
#         out = pack_padded_sequence(out, src_len.tolist(), batch_first=False)

#         output, hidden = self.rnn(out, hidden)
#         output, _ = pad_packed_sequence(output, batch_first=False)

#         if self.bi and SUM_UP:
#             output = self.enc_out_merge(output)

#         if isinstance(hidden, tuple):
#             hidden = hidden[0]
#         hidden_idx = [1, 3, 5, 7, 9, 11][:self.n_layers]
#         final_hidden = hidden[hidden_idx]

#         return output, final_hidden

#     def conv_mask(self, matrix, lens):
#         lens = np.array(lens)
#         width = matrix.shape[-1]
#         lens2 = lens * (width / self.width)
#         lens2 = lens2 + 0.999
#         lens2 = lens2.astype('int')
#         matrix_new = matrix.permute(0, 3, 1, 2)  # b, w, c, h
#         matrix_out = torch.zeros(matrix_new.shape, device=matrix.device)
#         for i, le in enumerate(lens2):
#             if self.flip:
#                 matrix_out[i, -le:] = matrix_new[i, -le:]
#             else:
#                 matrix_out[i, :le] = matrix_new[i, :le]
#         matrix_out = matrix_out.permute(0, 2, 3, 1)  # b, c, h, w
#         return matrix_out
    
    

    
# class EncoderResnet(nn.Module):
#     def __init__(self, hidden_size, height, width, bgru, step, flip):
#         super(EncoderResnet, self).__init__()
#         self.hidden_size = hidden_size
#         self.height = height
#         self.width = width
#         self.bi = bgru
#         self.step = step
#         self.flip = flip
#         self.n_layers = 2
#         self.dropout = 0.5

#         self.layer = resnet50(nb_feat=384)  # Using WriteViT ResNet18 with 384 channels

#         if DROP_OUT:
#             self.layer_dropout = nn.Dropout2d(p=0.5)

#         feature_dim = (self.height // 8) * 384
#         if self.step is not None:
#             self.output_proj = nn.Linear(feature_dim * self.step, feature_dim)

#         RNN = nn.LSTM if LSTM else nn.GRU

#         if self.bi:
#             self.rnn = RNN(feature_dim, self.hidden_size, self.n_layers, dropout=self.dropout, bidirectional=True)
#             if SUM_UP:
#                 self.enc_out_merge = lambda x: x[:, :, :x.shape[-1] // 2] + x[:, :, x.shape[-1] // 2:]
#                 self.enc_hidden_merge = lambda x: (x[0] + x[1]).unsqueeze(0)
#         else:
#             self.rnn = RNN(feature_dim, self.hidden_size, self.n_layers, dropout=self.dropout, bidirectional=False)

#     def forward(self, in_data, in_data_len, hidden=None):
#         batch_size = in_data.shape[0]
#         out = self.layer(in_data)

#         if DROP_OUT and self.training:
#             out = self.layer_dropout(out)

#         out = out.permute(3, 0, 2, 1)  # (W, B, H, C)
#         out = out.reshape(-1, batch_size, (self.height // 8) * 384)

#         if self.step is not None:
#             time_step = out.shape[0]
#             out_short = torch.zeros(time_step // self.step, batch_size, out.shape[2] * self.step, requires_grad=True).to(in_data.device)
#             for i in range(0, time_step // self.step):
#                 part_out = [out[j] for j in range(i * self.step, (i + 1) * self.step)]
#                 out_short[i] = torch.cat(part_out, 1)
#             out = self.output_proj(out_short)

#         width = out.shape[0]
#         src_len = in_data_len.numpy() * (width / self.width)
#         src_len = (src_len + 0.999).astype('int')

#         out = pack_padded_sequence(out, src_len.tolist(), batch_first=False)
#         output, hidden = self.rnn(out, hidden)
#         output, _ = pad_packed_sequence(output, batch_first=False)

#         if self.bi and SUM_UP:
#             output = self.enc_out_merge(output)

#         odd_idx = [1, 3, 5, 7, 9, 11]
#         hidden_idx = odd_idx[:self.n_layers]
#         final_hidden = hidden[hidden_idx]

#         return output, final_hidden

#     def conv_mask(self, matrix, lens):
#         lens = np.array(lens)
#         width = matrix.shape[-1]
#         lens2 = (lens * (width / self.width) + 0.999).astype('int')
#         matrix_new = matrix.permute(0, 3, 1, 2)
#         matrix_out = torch.zeros_like(matrix_new, requires_grad=True).to(matrix.device)
#         for i, le in enumerate(lens2):
#             if self.flip:
#                 matrix_out[i, -le:] = matrix_new[i, -le:]
#             else:
#                 matrix_out[i, :le] = matrix_new[i, :le]
#         matrix_out = matrix_out.permute(0, 2, 3, 1)
#         return matrix_out    
    



class Encoder(nn.Module):
    def __init__(self, hidden_size, height, width, bgru, step, flip):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.height = height
        self.width = width
        self.bi = bgru
        self.step = step
        self.flip = flip
        self.n_layers = 2
        self.dropout = 0.5

        #self.layer = vgg16_bn(PRE_TRAIN_VGG)
        self.layer = vgg19_bn(PRE_TRAIN_VGG)
       



        if DROP_OUT:
            self.layer_dropout = nn.Dropout2d(p=0.5)
        if self.step is not None:
            #self.output_proj = nn.Linear((((((self.height-2)//2)-2)//2-2-2-2)//2)*128*self.step, self.hidden_size)
            self.output_proj = nn.Linear(self.height//16*512*self.step, self.height//16*512)

        if LSTM:
            RNN = nn.LSTM
        else:
            RNN = nn.GRU

        if self.bi: #8: 3 MaxPool->2**3    128: last hidden_size of layer4
            self.rnn = RNN(self.height//16*512, self.hidden_size, self.n_layers, dropout=self.dropout, bidirectional=True)
            if SUM_UP:
                self.enc_out_merge = lambda x: x[:,:,:x.shape[-1]//2] + x[:,:,x.shape[-1]//2:]
                self.enc_hidden_merge = lambda x: (x[0] + x[1]).unsqueeze(0)
        else:
            self.rnn = RNN(self.height//16*512, self.hidden_size, self.n_layers, dropout=self.dropout, bidirectional=False)

    # (32, 1, 80, 1400)
    def forward(self, in_data, in_data_len, hidden=None):
        batch_size = in_data.shape[0]
        out = self.layer(in_data) # torch.Size([32, 512, 4, 63])
        if DROP_OUT and self.training:
            out = self.layer_dropout(out)
        #out.register_hook(print)
        out = out.permute(3, 0, 2, 1) # (width, batch, height, channels)
        #out = out.view(-1, batch_size, (((((self.height-2)//2)-2)//2-2-2-2)//2)*128) # (t, b, f) (173, 32, 1024)
        out = out.reshape(-1, batch_size, self.height//16*512)
        if self.step is not None:
            time_step, batch_size, n_feature = out.shape[0], out.shape[1], out.shape[2]
            #out_short = Variable(torch.zeros(time_step//self.step, batch_size, n_feature*self.step)).cuda() # t//STEP, b, f*STEP
            out_short = torch.zeros(time_step//self.step, batch_size, n_feature*self.step, requires_grad=True).to(cuda) # t//STEP, b, f*STEP
            for i in range(0, time_step//self.step):
                part_out = [out[j] for j in range(i*self.step, (i+1)*self.step)]
                # reverse the image feature map
                out_short[i] = torch.cat(part_out, 1) # b, f*STEP

            out = self.output_proj(out_short) # t//STEP, b, hidden_size
        width = out.shape[0]
        src_len = in_data_len.numpy()*(width/self.width)
        src_len = src_len + 0.999 # in case of 0 length value from float to int
        src_len = src_len.astype('int')
        out = pack_padded_sequence(out, src_len.tolist(), batch_first=False)
        output, hidden = self.rnn(out, hidden)
        # output: t, b, f*2  hidden: 2, b, f
        output, output_len = pad_packed_sequence(output, batch_first=False)
        if self.bi and SUM_UP:
            output = self.enc_out_merge(output)
            #hidden = self.enc_hidden_merge(hidden)
       # # output: t, b, f    hidden:  b, f
        odd_idx = [1, 3, 5, 7, 9, 11]
        hidden_idx = odd_idx[:self.n_layers]
        final_hidden = hidden[hidden_idx]
        #if self.flip:
        #    hidden = output[-1]
        #    #hidden = hidden.permute(1, 0, 2) # b, 2, f
        #    #hidden = hidden.contiguous().view(batch_size, -1) # b, f*2
        #else:
        #    hidden = output[0] # b, f*2
        return output, final_hidden # t, b, f*2    b, f*2

    # matrix: b, c, h, w    lens: list size of batch_size
    def conv_mask(self, matrix, lens):
        lens = np.array(lens)
        width = matrix.shape[-1]
        lens2 = lens * (width / self.width)
        lens2 = lens2 + 0.999 # in case le == 0
        lens2 = lens2.astype('int')
        matrix_new = matrix.permute(0, 3, 1, 2) # b, w, c, h
        #matrix_out = Variable(torch.zeros(matrix_new.shape)).cuda()
        matrix_out = torch.zeros(matrix_new.shape, requires_grad=True).to(cuda)
        for i, le in enumerate(lens2):
            if self.flip:
                matrix_out[i, -le:] = matrix_new[i, -le:]
            else:
                matrix_out[i, :le] = matrix_new[i, :le]
        matrix_out = matrix_out.permute(0, 2, 3, 1) # b, c, h, w
        return matrix_out

if __name__ == '__main__':
    print(vgg16_bn())
    
    
    
    
    
    
    

