import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from load_data import NUM_CHANNEL



model_urls = {
    'vgg11': '/home/woody/iwi5/iwi5333h/model/vgg/vgg11-bbd30ac9.pth',
    'vgg13': '/home/woody/iwi5/iwi5333h/model/vgg/vgg13-c768596a.pth',
    'vgg16': '/home/woody/iwi5/iwi5333h/model/vgg16-397923af.pth',
    'vgg19': '/home/woody/iwi5/iwi5333h/model/vgg19-dcbb9e9d.pth',
    'vgg11_bn': '/home/woody/iwi5/iwi5333h/model/vgg/vgg11_bn-6002323d.pth',
    'vgg13_bn': '/home/woody/iwi5/iwi5333h/model/vgg/vgg13_bn-abd245e5.pth',
    'vgg16_bn': '/home/woody/iwi5/iwi5333h/model/vgg16_bn-6c64b313.pth',
    'vgg19_bn': '/home/woody/iwi5/iwi5333h/model/vgg19_bn-c79401a0.pth',
}

class VGG(nn.Module):
    def __init__(self, features, init_weights=True):
        super(VGG, self).__init__()
        self.features = features
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu') # need to be updated
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = NUM_CHANNEL
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                #layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                layers += [conv2d, nn.InstanceNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512],
    #'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    # b,3,64,512 -> b,512,2,16
    #'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
    # b,3,64,512 -> b,512,4,32
    #'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512],
    # b,3,64,512 -> b,512,8,64
    'E': [64, 64, 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512],
    
}

#main
def vgg19_bn(pretrained=False, **kwargs):
    """VGG 19-layer model (configuration 'E') with batch normalization

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:       
        kwargs['init_weights'] = False
        model = VGG(make_layers(cfg['E'], batch_norm=True), **kwargs)
        model_dict = model.state_dict()
        ## total_dict = model_zoo.load_url(model_urls['vgg19_bn'])
        total_dict = torch.load(model_urls['vgg19_bn'])
        partial_dict = {k: v for k, v in total_dict.items() if k in model_dict}
        model_dict.update(partial_dict)
        #model.load_state_dict(partial_dict)
        model.load_state_dict(model_dict, strict=False)
    else:
        model = VGG(make_layers(cfg['E'], batch_norm=True), **kwargs)
    return model


