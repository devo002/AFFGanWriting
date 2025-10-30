import numpy as np
import os
import torch
from torch import nn
from blocks import LinearBlock, Conv2dBlock, ResBlocks, ActFirstResBlock
from vgg_tro_channel3_modi import vgg19_bn
#from Resnet18 import ResNet18
from recognizer.models.encoder_vgg import Encoder as rec_encoder
from recognizer.models.dinorec import RecDecoderDINOv2
#from recognizer.models.encoder_vgg import EncoderResnet
#from recognizer.models.encoder_vgg import EfficientNetEncoder
#from recognizer.models.encoder_vgg import ResNet50Encoder
from recognizer.models.decoder import Decoder as rec_decoder
#from recognizer.models.encoder_vgg import VGG19Encoder
from recognizer.models.seq2seqnew2 import Seq2Seq as rec_seq2seq
from recognizer.models.attention import locationAttention as rec_attention
from load_data import OUTPUT_MAX_LEN, IMG_HEIGHT, IMG_WIDTH, vocab_size, index2letter, num_tokens, tokens
import cv2
#from torchvision.models import efficientnet_v2_s
from torchvision.models import efficientnet_v2_l
import torch.nn.functional as F
from torchvision.models import resnet50, resnet18, convnext_tiny 
from torchvision.models.feature_extraction import create_feature_extractor
#from torchvision.models.vision_transformer import vit_b_16, ViT_B_16_Weights
from trocr_recognizer import TrOCRRecModel
from inception import ImageEncoderInceptionV3
from dinomodel import ImageEncoderDINOv2
from inceptionrecognizer import EncoderInception
from cnn import ImageEncoderStyleCNN
#from recognizer.models.encoder_vgg import EfficientNetB7Encoder


gpu = torch.device('cuda')

def normalize(tar):
    tar = (tar - tar.min())/(tar.max()-tar.min())
    tar = tar * 255
    tar = tar.astype(np.uint8)
    return tar

def fine(label_list):
    if type(label_list) != type([]):
        return [label_list]
    else:
        return label_list

def write_image(xg, pred_label, gt_img, gt_label, tr_imgs, xg_swap, pred_label_swap, gt_label_swap, title, num_tr=2):
    folder = '/home/woody/iwi5/iwi5333h/img3'
    if not os.path.exists(folder):
        os.makedirs(folder)
    batch_size = gt_label.shape[0]
    tr_imgs = tr_imgs.cpu().numpy()
    xg = xg.cpu().numpy()
    xg_swap = xg_swap.cpu().numpy()
    gt_img = gt_img.cpu().numpy()
    gt_label = gt_label.cpu().numpy()
    gt_label_swap = gt_label_swap.cpu().numpy()
    pred_label = torch.topk(pred_label, 1, dim=-1)[1].squeeze(-1) # b,t,83 -> b,t,1 -> b,t
    pred_label = pred_label.cpu().numpy()
    pred_label_swap = torch.topk(pred_label_swap, 1, dim=-1)[1].squeeze(-1) # b,t,83 -> b,t,1 -> b,t
    pred_label_swap = pred_label_swap.cpu().numpy()
    tr_imgs = tr_imgs[:, :num_tr, :, :]
    outs = list()
    for i in range(batch_size):
        src = tr_imgs[i].reshape(num_tr*IMG_HEIGHT, -1)
        gt = gt_img[i].squeeze()
        tar = xg[i].squeeze()
        tar_swap = xg_swap[i].squeeze()
        src = normalize(src)
        gt = normalize(gt)
        tar = normalize(tar)
        tar_swap = normalize(tar_swap)
        gt_text = gt_label[i].tolist()
        gt_text_swap = gt_label_swap[i].tolist()
        pred_text = pred_label[i].tolist()
        pred_text_swap = pred_label_swap[i].tolist()

        gt_text = fine(gt_text)
        gt_text_swap = fine(gt_text_swap)
        pred_text = fine(pred_text)
        pred_text_swap = fine(pred_text_swap)

        for j in range(num_tokens):
            gt_text = list(filter(lambda x: x!=j, gt_text))
            gt_text_swap = list(filter(lambda x: x!=j, gt_text_swap))
            pred_text = list(filter(lambda x: x!=j, pred_text))
            pred_text_swap = list(filter(lambda x: x!=j, pred_text_swap))


        gt_text = ''.join([index2letter[c-num_tokens] for c in gt_text])
        gt_text_swap = ''.join([index2letter[c-num_tokens] for c in gt_text_swap])
        pred_text = ''.join([index2letter[c-num_tokens] for c in pred_text])
        pred_text_swap = ''.join([index2letter[c-num_tokens] for c in pred_text_swap])
        gt_text_img = np.zeros_like(tar)
        gt_text_img_swap = np.zeros_like(tar)
        pred_text_img = np.zeros_like(tar)
        pred_text_img_swap = np.zeros_like(tar)
        cv2.putText(gt_text_img, gt_text, (5, 55), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        cv2.putText(gt_text_img_swap, gt_text_swap, (5, 55), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        cv2.putText(pred_text_img, pred_text, (5, 55), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        cv2.putText(pred_text_img_swap, pred_text_swap, (5, 55), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        out = np.vstack([src, gt, gt_text_img, tar, pred_text_img, gt_text_img_swap, tar_swap, pred_text_img_swap])
        outs.append(out)
    final_out = np.hstack(outs)
    cv2.imwrite(folder+'/'+title+'.png', final_out)




def get_num_adain_params(model):
    # return the number of AdaIN parameters needed by the model
    num_adain_params = 0
    for m in model.modules():
        if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
            num_adain_params += 2*m.num_features
    return num_adain_params


class DisModel(nn.Module):
    def __init__(self):
        super(DisModel, self).__init__()
        self.n_layers = 6
        self.final_size = 1024
        nf = 16
        cnn_f = [Conv2dBlock(1, nf, 7, 1, 3,
                             pad_type='reflect',
                             norm='none',
                             activation='none')]
        for i in range(self.n_layers - 1):
            nf_out = np.min([nf * 2, 1024])
            cnn_f += [ActFirstResBlock(nf, nf, None, 'lrelu', 'none')]
            cnn_f += [ActFirstResBlock(nf, nf_out, None, 'lrelu', 'none')]
            cnn_f += [nn.ReflectionPad2d(1)]
            cnn_f += [nn.AvgPool2d(kernel_size=3, stride=2)]
            nf = np.min([nf * 2, 1024])
        nf_out = np.min([nf * 2, 1024])
        cnn_f += [ActFirstResBlock(nf, nf, None, 'lrelu', 'none')]
        cnn_f += [ActFirstResBlock(nf, nf_out, None, 'lrelu', 'none')]
        cnn_c = [Conv2dBlock(nf_out, self.final_size, IMG_HEIGHT//(2**(self.n_layers-1)), IMG_WIDTH//(2**(self.n_layers-1))+1,
                             norm='none',
                             activation='lrelu',
                             activation_first=True)]
        self.cnn_f = nn.Sequential(*cnn_f)
        self.cnn_c = nn.Sequential(*cnn_c)
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, x):
        feat = self.cnn_f(x)
        out = self.cnn_c(feat)
        return out.squeeze(-1).squeeze(-1) # b,1024   maybe b is also 1, so cannnot out.squeeze()

    def calc_dis_fake_loss(self, input_fake):
        label = torch.zeros(input_fake.shape[0], self.final_size).to(gpu)
        resp_fake = self.forward(input_fake)
        fake_loss = self.bce(resp_fake, label)
        return fake_loss

    def calc_dis_real_loss(self, input_real):
        label = torch.ones(input_real.shape[0], self.final_size).to(gpu)
        resp_real = self.forward(input_real)
        real_loss = self.bce(resp_real, label)
        return real_loss

    def calc_gen_loss(self, input_fake):
        label = torch.ones(input_fake.shape[0], self.final_size).to(gpu)
        resp_fake = self.forward(input_fake)
        fake_loss = self.bce(resp_fake, label)
        return fake_loss

class WriterClaModel(nn.Module):
    def __init__(self, num_writers):
        super(WriterClaModel, self).__init__()
        self.n_layers = 6
        nf = 16
        cnn_f = [Conv2dBlock(1, nf, 7, 1, 3,
                             pad_type='reflect',
                             norm='none',
                             activation='none')]
        for i in range(self.n_layers - 1):
            nf_out = np.min([nf * 2, 1024])
            cnn_f += [ActFirstResBlock(nf, nf, None, 'lrelu', 'none')]
            cnn_f += [ActFirstResBlock(nf, nf_out, None, 'lrelu', 'none')]
            cnn_f += [nn.ReflectionPad2d(1)]
            cnn_f += [nn.AvgPool2d(kernel_size=3, stride=2)]
            nf = np.min([nf * 2, 1024])
        nf_out = np.min([nf * 2, 1024])
        cnn_f += [ActFirstResBlock(nf, nf, None, 'lrelu', 'none')]
        cnn_f += [ActFirstResBlock(nf, nf_out, None, 'lrelu', 'none')]
        cnn_c = [Conv2dBlock(nf_out, num_writers, IMG_HEIGHT//(2**(self.n_layers-1)), IMG_WIDTH//(2**(self.n_layers-1))+1,
                             norm='none',
                             activation='lrelu',
                             activation_first=True)]
        self.cnn_f = nn.Sequential(*cnn_f)
        self.cnn_c = nn.Sequential(*cnn_c)
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, x, y):
        feat = self.cnn_f(x)
        out = self.cnn_c(feat) # b,310,1,1
        loss = self.cross_entropy(out.squeeze(-1).squeeze(-1), y)
        return loss

inception_weights = "/home/woody/iwi5/iwi5333h/model/inception_v3_imagenet1k_v1.pth"
repo_dir  = "/home/woody/iwi5/iwi5333h/facebookresearch_dinov2_main"
ckpt_path = "/home/woody/iwi5/iwi5333h/model/dinov2_vitl14_pretrain.pth"
#ckpt_path2 = "/home/woody/iwi5/iwi5333h/model/dinov2_vits14_pretrain.pth"

class GenModel_FC(nn.Module):
    def __init__(self, text_max_len):
        super(GenModel_FC, self).__init__()
        #self.enc_image = ImageEncoder().to(gpu)
        #self.enc_image = ImageEncoderStyleCNN(in_channels=50, final_size=(8,27)).to(gpu)
        #self.enc_image = ImageEncoderInceptionV3(weight_path=inception_weights).to(gpu)
        self.enc_image = ImageEncoderDINOv2(repo_dir=repo_dir,arch="vitl14",ckpt_path=ckpt_path,in_channels=50,final_size=(8, 27),tap_blocks=[4, 8, 16, 23]).to(gpu)
        #self.enc_image = ImageEncoderEfficientNet(weight_path=efficientnet_weights_path).to(gpu)
        #self.enc_image = ResNet18().to(gpu)
        #self.enc_image = ResNet18(nb_feat=384, in_channels=50).to(gpu)
        #self.enc_image = ConvNeXtTiny(weight_path=convnext_weights, in_channels=50).to(gpu)
        #self.enc_image = ImageEncoderResNet50(weight_path=resnet50_weights_path, in_channels=50).to(gpu)
        self.enc_text = TextEncoder_FC(text_max_len).to(gpu)
        
        self.dec = Decoder().to(gpu)
        self.linear_mix = nn.Linear(1024, 512)
        self.max_conv = nn.MaxPool2d(kernel_size=2, stride=2)

    def assign_adain_params(self,adain_params, results, embed):
        # assign the adain_params to the AdaIN layers in model
        i = 0
        for m in self.dec.modules():
            if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
                mean = adain_params[:, :m.num_features]
                std = adain_params[:, m.num_features:2 * m.num_features]
                m.bias = mean.contiguous().view(-1)
                m.weight = std.contiguous().view(-1)
                m.con = embed
                if (i == 1):
                    m.input = self.max_conv(results[3])
                elif (i == 3):
                    m.input = results[4] #change here for the resnet with few layers
                if adain_params.size(1) > 2 * m.num_features:
                    adain_params = adain_params[:, 2 * m.num_features:]
                i += 1

    def decode(self, content, results, embed, adain_params):
        # decode content and style codes to an image
        self.assign_adain_params(adain_params, results, embed)

        images = self.dec(content)
        return images

    # feat_mix: b,1024,8,27
    def mix(self, results, feat_embed):
        # for i in results:
        #     print(i.size())
        
        feat_mix = torch.cat([results[-1], feat_embed], dim=1)  # b,C,H,W
        f = feat_mix.permute(0, 2, 3, 1)                        # b,H,W,C
        ff = self.linear_mix(f)                                # linear layer
        return ff.permute(0, 3, 1, 2)                           # b,C,H,W
        
        
        
        #feat_mix = torch.cat([results[-1], feat_embed], dim=1) # b,1024,8,27
        #f = feat_mix.permute(0, 2, 3, 1)
        #ff = self.linear_mix(f) # b,8,27,1024->b,8,27,512
        #return ff.permute(0, 3, 1, 2)

class TextEncoder_FC(nn.Module):
    def __init__(self, text_max_len):
        super(TextEncoder_FC, self).__init__()
        embed_size = 64
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.fc = nn.Sequential(
                nn.Linear(text_max_len*embed_size, 1024),
                nn.BatchNorm1d(1024),
                nn.ReLU(inplace=False),
                nn.Linear(1024, 2048),
                nn.BatchNorm1d(2048),
                nn.ReLU(inplace=False),
                nn.Linear(2048, 4096)
                )
        '''embed content force'''
        self.linear = nn.Linear(embed_size, 512)

    def forward(self, x, f_xs_shape):
        xx = self.embed(x) # b,t,embed

        batch_size = xx.shape[0]
        xxx = xx.reshape(batch_size, -1) # b,t*embed
        out = self.fc(xxx)

        '''embed content force'''
        xx_new = self.linear(xx) # b, text_max_len, 512
        ts = xx_new.shape[1]
        height_reps = f_xs_shape[-2]
        
        #width_reps = f_xs_shape[-1] // ts
        width_reps = max(1, f_xs_shape[-1] // ts)
        
        tensor_list = list()
        for i in range(ts):
            text = [xx_new[:, i:i + 1]] # b, text_max_len, 512
            tmp = torch.cat(text * width_reps, dim=1)
            tensor_list.append(tmp)

        padding_reps = f_xs_shape[-1] % ts
        if padding_reps:
            embedded_padding_char = self.embed(torch.full((1, 1), tokens['PAD_TOKEN'], dtype=torch.long).cuda())
            embedded_padding_char = self.linear(embedded_padding_char)
            padding = embedded_padding_char.repeat(batch_size, padding_reps, 1)
            tensor_list.append(padding)

        res = torch.cat(tensor_list, dim=1) # b, text_max_len * width_reps + padding_reps, 512
        res = res.permute(0, 2, 1).unsqueeze(2) # b, 512, 1, text_max_len * width_reps + padding_reps
        final_res = torch.cat([res] * height_reps, dim=2)

        return out, final_res


'''VGG19_IN tro'''
# class ImageEncoder(nn.Module):
#     def __init__(self):
#         super(ImageEncoder, self).__init__()
#         self.model = vgg19_bn(False)
#         self.output_dim = 512
#
#     def forward(self, x):
#         return self.model(x)


class ImageEncoder(nn.Module):
    def __init__(self):
        super(ImageEncoder, self).__init__()
        self.model = vgg19_bn(False)
        self.output_dim = 512
        self.image_encoder_layer = []

        #分割vgg19网络
        enc_layers = list(self.model.features.children())
        # print("enc_layers:"+str(enc_layers))
        enc_1 = nn.DataParallel(nn.Sequential(*enc_layers[:3]).to(gpu))
        enc_2 = nn.DataParallel(nn.Sequential(*enc_layers[3:9]).to(gpu))
        enc_3 = nn.DataParallel(nn.Sequential(*enc_layers[9:16]).to(gpu))
        enc_4 = nn.DataParallel(nn.Sequential(*enc_layers[16:29]).to(gpu))
        enc_5 = nn.DataParallel(nn.Sequential(*enc_layers[29:42]).to(gpu))
        enc_6 = nn.DataParallel(nn.Sequential(*enc_layers[42:]).to(gpu))
        self.image_encoder_layers = [enc_1, enc_2, enc_3, enc_4, enc_5 ,enc_6]

        # print(self.image_encoder_layers)
        ''' 
        input:torch.Size([8, 50, 48, 540])
        results[1]: torch.Size([8, 64, 48, 540])
        results[2]: torch.Size([8, 128, 48, 540])
        results[3]: torch.Size([8, 256, 24, 270])
        results[4]: torch.Size([8, 512, 12, 135])
        results[5]: torch.Size([8, 512, 6, 67])
        results[6]: torch.Size([8, 512, 6, 67])
        '''

    def encode_with_intermediate(self, input_img):
        # 输入图像，将图像经过每一个image_encoder_layers里的vgg层
        # print("input_img:"+ str(input_img.size()))
        results = [input_img]
        for i in range(6):
            func = self.image_encoder_layers[i]
            results.append(func(results[-1]))

        return results[1:]

    def forward(self, x):
        # result = self.model(x)
        results = self.encode_with_intermediate(x)

        return results
        # return cs
        
    
    
#efficientnet_weights_path = "/home/woody/iwi5/iwi5333h/model/efficientnet_v2_s-dd5fe13b.pth"
efficientnet_weights_path = "/home/woody/iwi5/iwi5333h/model/efficientnet_v2_l-59c71312.pth"


class ImageEncoderEfficientNet(nn.Module):
    def __init__(self, weight_path=None, in_channels=50):
        super(ImageEncoderEfficientNet, self).__init__()
        self.output_dim = 512
        self.model = efficientnet_v2_l(weights=None)

        # Load pretrained weights if provided
        if weight_path:
            state_dict = torch.load(weight_path, map_location="cpu")
            self.model.load_state_dict(state_dict)

        # Modify the first conv layer to accept `in_channels` instead of 3
        first_conv = self.model.features[0][0]  # Assuming [Conv2d, BN, SiLU]
        new_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=first_conv.out_channels,
            kernel_size=first_conv.kernel_size,
            stride=first_conv.stride,
            padding=first_conv.padding,
            bias=first_conv.bias is not None
        )

        # Initialize weights smartly from original conv
        with torch.no_grad():
            if first_conv.weight.shape[1] == 3:
                # Copy first 3 channels
                new_conv.weight[:, :3] = first_conv.weight
                # Initialize remaining channels by repeating channel 0 or average
                if in_channels > 3:
                    repeat_tensor = first_conv.weight[:, :1].repeat(1, in_channels - 3, 1, 1)
                    new_conv.weight[:, 3:] = repeat_tensor

        self.model.features[0][0] = new_conv

        self.features = list(self.model.features.children())
        
        # for i, block in enumerate(self.model.features):
        #     if i < 2:
        #         for param in block.parameters():
        #             param.requires_grad = False

        # Helper function to get the last Conv2d out_channels from a block
        def get_out_channels(block):
            for layer in reversed(list(block.modules())):
                if isinstance(layer, nn.Conv2d):
                    return layer.out_channels
            raise ValueError("No Conv2d layer found in block")

        # Reduce selected intermediate outputs to 512 channels
        self.reduce_layers = nn.ModuleList([
            nn.Conv2d(get_out_channels(block), 512, kernel_size=1)
            for i, block in enumerate(self.features)
            #if i in [1, 2, 3, 4, 5] or i == len(self.features) - 1
            if i in [1, 2, 3, 4, 5]
        ])

        self.features = nn.Sequential(*self.features)

    def encode_with_intermediate(self, x):
        results = []
        reduce_idx = 0
        for i, block in enumerate(self.features):
            x = block(x)
            #if i in [1, 2, 3, 4, 5] or i == len(self.features) - 1:
            if i in [1, 2, 3, 4, 5]:
                reduced = self.reduce_layers[reduce_idx](x)
                reduce_idx += 1
                results.append(reduced)

        # Resize final feature map to match [B, 512, 8, 27] like VGG
        results[-1] = F.interpolate(results[-1], size=(8, 27), mode='bilinear', align_corners=False)

        return results[-5:]  # use -6 for adding another block

    def forward(self, x):
        return self.encode_with_intermediate(x)

# resnet18_weights_path = "/home/woody/iwi5/iwi5333h/model/resnet18-f37072fd.pth"
resnet50_weights_path = "/home/woody/iwi5/iwi5333h/model/resnet50-0676ba61.pth"

# #for the generator
class ImageEncoderResNet50(nn.Module):
    def __init__(self, weight_path=None, in_channels=50):
        super(ImageEncoderResNet50, self).__init__()
        self.output_dim = 512

        # Load ResNet50 with no weights initially
        self.model = resnet50(weights=None)

        # Load local pretrained weights if provided
        if weight_path:
            state_dict = torch.load(weight_path, map_location="cpu")
            self.model.load_state_dict(state_dict)

        # Modify the first conv layer to accept custom input channels
        original_conv = self.model.conv1
        new_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=original_conv.bias is not None
        )
        with torch.no_grad():
            new_conv.weight[:, :3] = original_conv.weight
            if in_channels > 3:
                repeat_tensor = original_conv.weight[:, :1].repeat(1, in_channels - 3, 1, 1)
                new_conv.weight[:, 3:] = repeat_tensor
        self.model.conv1 = new_conv

        # Select intermediate feature layers
        return_nodes = {
            'relu': 'feat1',
            'layer1': 'feat2',
            'layer2': 'feat3',
            'layer3': 'feat4',
            'layer4': 'feat5'
        }
        self.extractor = create_feature_extractor(self.model, return_nodes=return_nodes)


        # self.reduce_layers = nn.ModuleList([
        #      nn.Conv2d(64, 512, kernel_size=1),   # relu
        #      nn.Conv2d(64, 512, kernel_size=1),   # layer1
        #      nn.Conv2d(128, 512, kernel_size=1),  # layer2
        #      nn.Conv2d(256, 512, kernel_size=1),  # layer3
        #      nn.Conv2d(512, 512, kernel_size=1),  # layer4
        #         ])
        # use for resnet50
        self.reduce_layers = nn.ModuleList([
            nn.Conv2d(64, 512, kernel_size=1),
            nn.Conv2d(256, 512, kernel_size=1),
            nn.Conv2d(512, 512, kernel_size=1),
            nn.Conv2d(1024, 512, kernel_size=1),
            nn.Conv2d(2048, 512, kernel_size=1),  
        ])

    def encode_with_intermediate(self, x):
        features = self.extractor(x)
        results = []
        for i, key in enumerate(['feat1', 'feat2', 'feat3', 'feat4', 'feat5']): # add feat5 for the layer4
            reduced = self.reduce_layers[i](features[key])
            results.append(reduced)

        # Resize final feature map to match VGG output shape
        results[-1] = F.interpolate(results[-1], size=(8, 27), mode='bilinear', align_corners=False)
        return results

    def forward(self, x):
        return self.encode_with_intermediate(x)




# class ImageEncoderEfficientNet(nn.Module):
#     def __init__(self, weight_path=None):
#         super(ImageEncoderEfficientNet, self).__init__()
#         self.output_dim = 512
#         self.model = efficientnet_v2_l(weights=None)

#         if weight_path:
#             state_dict = torch.load(weight_path, map_location="cpu")
#             self.model.load_state_dict(state_dict)

#         self.features = list(self.model.features.children())

#         # Helper function to get the last Conv2d out_channels from a block
#         def get_out_channels(block):
#             for layer in reversed(list(block.modules())):
#                 if isinstance(layer, nn.Conv2d):
#                     return layer.out_channels
#             raise ValueError("No Conv2d layer found in block")

#         # Reduce all selected intermediate outputs to 512 channels
#         self.reduce_layers = nn.ModuleList([
#             nn.Conv2d(get_out_channels(block), 512, kernel_size=1)
#             for i, block in enumerate(self.features)
#             if i in [1, 2, 3, 4, 5] or i == len(self.features) - 1
#         ])

#         self.features = nn.Sequential(*self.features)

#     def encode_with_intermediate(self, x):
#         results = []
#         reduce_idx = 0
#         for i, block in enumerate(self.features):
#             x = block(x)
#             if i in [1, 2, 3, 4, 5] or i == len(self.features) - 1:
#                 reduced = self.reduce_layers[reduce_idx](x)
#                 reduce_idx += 1
#                 results.append(reduced)

#         # Resize final feature map to match [B, 512, 8, 27] like VGG
#         results[-1] = nn.functional.interpolate(results[-1], size=(8, 27), mode='bilinear', align_corners=False)

#         return results[-6:]  # Keep only last 6 layers like VGG

#     def forward(self, x):
#         return self.encode_with_intermediate(x)



class Decoder(nn.Module):
    def __init__(self, ups=3, n_res=2, dim=512, out_dim=1, res_norm='adain', activ='relu', pad_type='reflect'):
        super(Decoder, self).__init__()

        self.model = []
        self.model += [ResBlocks(n_res, dim, res_norm,
                                 activ, pad_type=pad_type)]
        for i in range(ups):
            self.model += [nn.Upsample(scale_factor=2),
                           Conv2dBlock(dim, dim // 2, 5, 1, 2,
                                       norm='in',
                                       activation=activ,
                                       pad_type=pad_type)]
            dim //= 2
        self.model += [Conv2dBlock(dim, out_dim, 7, 1, 3,
                                   norm='none',
                                   activation='tanh',
                                   pad_type=pad_type)]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)


class RecModel(nn.Module):
    def __init__(self, pretrain=False):
        super(RecModel, self).__init__()
        hidden_size_enc = hidden_size_dec = 512
        embed_size = 60
        INCP_WEIGHTS = "/home/woody/iwi5/iwi5333h/model/inception_v3_imagenet1k_v1.pth"
        #weight_path = "/home/woody/iwi5/iwi5333h/model/efficientnet_v2_l-59c71312.pth"
        #weight_path = "/home/woody/iwi5/iwi5333h/model/vgg19_bn-c79401a0.pth"
        #efficientnet_b7_weights_path = "/home/woody/iwi5/iwi5333h/model/efficientnet_b7_lukemelas-c5b4e57e.pth"
        #self.enc = ResNet50Encoder(hidden_size_enc, IMG_HEIGHT, IMG_WIDTH, True, None, False, weight_path=resnet50_weights_path).to(gpu)
        #self.enc = EfficientNetEncoder(hidden_size_enc, IMG_HEIGHT, IMG_WIDTH, True, None, False, weight_path=efficientnet_weights_path).to(gpu)
        self.enc = rec_encoder(hidden_size_enc, IMG_HEIGHT, IMG_WIDTH, True, None, False).to(gpu)
        #self.enc = RecDecoderDINOv2(hidden_size_enc, IMG_HEIGHT, IMG_WIDTH, bgru=True, step=None, flip=False, repo_dir=repo_dir,ckpt_path=ckpt_path, arch="vits14").to(gpu)
        #self.enc = EncoderInception(hidden_size_enc, IMG_HEIGHT, IMG_WIDTH, True, None, False, weights_path=INCP_WEIGHTS, in_channels=3,output_stride=16,map_location="cpu").to(gpu)
        #self.enc = EncoderResnet(hidden_size_enc, IMG_HEIGHT, IMG_WIDTH, True, None, False).to(gpu)
        #self.enc = EfficientNetB7Encoder(hidden_size_enc, IMG_HEIGHT, IMG_WIDTH,True, None, False, weight_path=efficientnet_b7_weights_path).to(gpu)
        #self.enc = VGG19Encoder(hidden_size_enc, IMG_HEIGHT, IMG_WIDTH,True, None, False, weight_path=weight_path).to(gpu)
        self.dec = rec_decoder(hidden_size_dec, embed_size, vocab_size, rec_attention, None).to(gpu)
        self.seq2seq = rec_seq2seq(self.enc, self.dec, OUTPUT_MAX_LEN, vocab_size).to(gpu)
        if pretrain:
            model_file = 'recognizer/save_weights/seq2seq-72.model_5.79.bak'
            print('Loading RecModel', model_file)
            self.seq2seq.load_state_dict(torch.load(model_file))

    def forward(self, img, label, img_width):
        self.seq2seq.train()
        img = torch.cat([img,img,img], dim=1) # b,1,64,128->b,3,64,128
        output, attn_weights = self.seq2seq(img, label, img_width, teacher_rate=False, train=False, beam_size=3)
        return output.permute(1, 0, 2) # t,b,83->b,t,83


#LOCAL_CKPT = "/home/woody/iwi5/iwi5333h/model/trocr-base-handwritten"
#DEFAULT_CKPT = LOCAL_CKPT  # or "microsoft/trocr-base-handwritten"

# class RecModel(nn.Module):
#     """
#     Wrapper around TrOCRRecModel that:
#       - Returns raw logits in *your* vocab: [B, T, vocab_size]
#       - Keeps TrOCR weights frozen (requires_grad=False)
#       - Forces TrOCR to stay in eval() even when parent calls train()
#     """
#     def __init__(self, pretrain: bool = False, ckpt: str = DEFAULT_CKPT, local_only: bool = True):
#         super().__init__()
#         # TrOCRRecModel itself freezes params and sets eval(); gradients still flow to the image.
#         self.rec = TrOCRRecModel(ckpt=ckpt, local_only=local_only)

#     def forward(self, img: torch.Tensor, label: torch.Tensor, img_width=None) -> torch.Tensor:
#         # img:   [B, 1, H, W]  (grayscale, your pipeline)
#         # label: [B, T]        (your vocab indices incl. <GO>, PAD, etc.)
#         # returns logits: [B, T, vocab_size]
#         return self.rec(img, label, img_width)

#     def train(self, mode: bool = True):
#         """
#         Keep TrOCR frozen/inference-mode regardless of parent mode.
#         This preserves dropout/bn behavior inside TrOCR and avoids accidental unfreezing.
#         """
#         super().train(mode)
#         # Ensure the underlying TrOCR stays eval() no matter what
#         if hasattr(self.rec, "model"):
#             self.rec.model.eval()
#         return self

#     @torch.no_grad()
#     def decode(self, img: torch.Tensor, beam_size: int = 5, max_new_tokens: int = 128):
#         """
#         Optional convenience passthrough for qualitative checks.
#         """
#         if hasattr(self.rec, "decode"):
#             return self.rec.decode(img, beam_size=beam_size, max_new_tokens=max_new_tokens)
#         raise AttributeError("Underlying recognizer does not implement decode().")



class MLP(nn.Module):
    def __init__(self, in_dim=64, out_dim=4096, dim=256, n_blk=3, norm='none', activ='relu'):

        super(MLP, self).__init__()
        self.model = []
        self.model += [LinearBlock(in_dim, dim, norm=norm, activation=activ)]
        for i in range(n_blk - 2):
            self.model += [LinearBlock(dim, dim, norm=norm, activation=activ)]
        self.model += [LinearBlock(dim, out_dim,
                                   norm='none', activation='none')]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x.view(x.size(0), -1))
