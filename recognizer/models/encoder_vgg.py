from torch import nn
#from torch.autograd import Variable
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np
#from models.vgg_tro_channel1 import vgg16_bn
from recognizer.models.vgg_tro_channel3 import vgg16_bn, vgg19_bn
from torchvision.models import efficientnet_v2_s

#torch.cuda.set_device(1)
cuda = torch.device('cuda')

DROP_OUT = False
LSTM = False
SUM_UP = True
PRE_TRAIN_VGG = True

class EfficientNetEncoder(nn.Module):
    def __init__(self, hidden_size, height, width, bgru=True, step=None, flip=False, weight_path=None):
        super(EfficientNetEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.height = height
        self.width = width
        self.bi = bgru
        self.step = step
        self.flip = flip
        self.n_layers = 2
        self.dropout = 0.5

        # Load EfficientNetV2S backbone
        self.backbone = efficientnet_v2_s(weights=None)
        
        # Load pretrained weights if path is provided
        if weight_path:
            try:
                state_dict = torch.load(weight_path, map_location="cpu")
                # Handle possible state_dict formats
                if 'state_dict' in state_dict:  # If weights are saved with model wrapper
                    state_dict = state_dict['state_dict']
                # Remove any 'module.' prefix from parameter names if present
                state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
                self.backbone.load_state_dict(state_dict)
                print(f"Successfully loaded EfficientNetV2S weights from: {weight_path}")
            except Exception as e:
                print(f"Error loading weights from {weight_path}: {str(e)}")
                print("Using randomly initialized weights instead")

        # Remove the classifier head
        self.features = self.backbone.features
        
        # For EfficientNetV2S, the final feature map is [B, 1280, H/32, W/32]
        # We'll add a projection to match expected dimensions
        self.feature_height = height // 32
        self.feature_proj = nn.Sequential(
            nn.Conv2d(1280, 512, kernel_size=1),  # Reduce channels to match VGG
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((self.feature_height, None))  # Maintain consistent height
        )
        
        if DROP_OUT:
            self.layer_dropout = nn.Dropout2d(p=0.5)
            
        if self.step is not None:
            self.output_proj = nn.Linear(self.feature_height*512*self.step, self.feature_height*512)

        if LSTM:
            RNN = nn.LSTM
        else:
            RNN = nn.GRU

        # RNN input size is feature_height * 512
        rnn_input_size = self.feature_height * 512
        
        if self.bi:
            self.rnn = RNN(rnn_input_size, self.hidden_size, self.n_layers,
                         dropout=self.dropout, bidirectional=True)
            if SUM_UP:
                self.enc_out_merge = lambda x: x[:,:,:x.shape[-1]//2] + x[:,:,x.shape[-1]//2:]
                self.enc_hidden_merge = lambda x: (x[0] + x[1]).unsqueeze(0)
        else:
            self.rnn = RNN(rnn_input_size, self.hidden_size, self.n_layers,
                         dropout=self.dropout, bidirectional=False)

    def forward(self, in_data, in_data_len, hidden=None):
        batch_size = in_data.shape[0]
        
        # Convert grayscale to RGB by repeating channels if needed
        if in_data.shape[1] == 1:
            in_data = in_data.repeat(1, 3, 1, 1)
        
        # Extract features
        out = self.features(in_data)  # [B, 1280, H/32, W/32]
        out = self.feature_proj(out)  # [B, 512, feature_height, W']
        
        if DROP_OUT and self.training:
            out = self.layer_dropout(out)
            
        # Prepare for RNN - [W', B, feature_height*512]
        out = out.permute(3, 0, 2, 1)  # [W', B, feature_height, 512]
        out = out.reshape(out.shape[0], batch_size, -1)  # [W', B, feature_height*512]
        
        if self.step is not None:
            time_step = out.shape[0]
            out_short = torch.zeros(time_step//self.step, batch_size, 
                                 self.feature_height*512*self.step,
                                 device=out.device)
            for i in range(time_step//self.step):
                part_out = [out[j] for j in range(i*self.step, (i+1)*self.step)]
                out_short[i] = torch.cat(part_out, dim=1)
            out = self.output_proj(out_short)

        # Handle variable length sequences
        width = out.shape[0]
        src_len = in_data_len.cpu().numpy()*(width/self.width)
        src_len = src_len + 0.999  # in case of 0 length value from float to int
        src_len = src_len.astype('int')
        out = pack_padded_sequence(out, src_len.tolist(), batch_first=False)
        
        # RNN processing
        output, hidden = self.rnn(out, hidden)
        output, _ = pad_packed_sequence(output, batch_first=False)
        
        if self.bi and SUM_UP:
            output = self.enc_out_merge(output)
            
        # Handle hidden states
        if isinstance(hidden, tuple):  # LSTM case
            hidden = hidden[0]
        odd_idx = [1, 3, 5, 7, 9, 11]
        hidden_idx = odd_idx[:self.n_layers]
        final_hidden = hidden[hidden_idx]
        
        return output, final_hidden

    def conv_mask(self, matrix, lens):
        lens = np.array(lens)
        width = matrix.shape[-1]
        lens2 = lens * (width / self.width)
        lens2 = lens2 + 0.999  # in case le == 0
        lens2 = lens2.astype('int')
        matrix_new = matrix.permute(0, 3, 1, 2)  # b, w, c, h
        matrix_out = torch.zeros(matrix_new.shape, device=matrix.device)
        for i, le in enumerate(lens2):
            if self.flip:
                matrix_out[i, -le:] = matrix_new[i, -le:]
            else:
                matrix_out[i, :le] = matrix_new[i, :le]
        matrix_out = matrix_out.permute(0, 2, 3, 1)  # b, c, h, w
        return matrix_out

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
