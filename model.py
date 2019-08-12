import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import model_zoo
import numpy as np
import functools

from torchvision.models.resnet import ResNet
from torchvision.models.resnet import BasicBlock
from pretrainedmodels.models.torchvision_models import pretrained_settings


def normalize_input(x, mean=None, std=None, input_space='RGB', input_range=None, **kwargs):
    if input_space == 'BGR':
        x = x[..., ::-1].copy()

    if input_range is not None:
        if x.max() > 1 and input_range[1] == 1:
            x = x / 255.

    if mean is not None:
        mean = np.array(mean)
        x = x - mean

    if std is not None:
        std = np.array(std)
        x = x / std

    return x


class ModelBase(nn.Module):
    def __init__(self):
        super().__init__()

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class Conv2dReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0,
                 stride=1, use_batchnorm=True, **batchnorm_params):

        super().__init__()

        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size,
                      stride=stride, padding=padding, bias=not (use_batchnorm)),
            nn.ReLU(inplace=True)
        ]

        if use_batchnorm:
            layers.insert(1, nn.BatchNorm2d(out_channels, **batchnorm_params))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class UNetDecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_batchnorm=True, interpolate="nearest"):
        super().__init__()
        self.interpolate = interpolate

        self.conv_transpose_block = None
        if self.interpolate is None:
            self.conv_transpose_block = nn.ConvTranspose2d(out_channels*2, out_channels*2, kernel_size=2, stride=2)

        layers = []
        layers.append(DoubleConv2DReLU(in_channels, out_channels, kernel_size=3, padding=1, use_batchnorm=use_batchnorm))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        x, skip = x
        if self.conv_transpose_block is not None:
            x = self.conv_transpose_block(x)
        else:
            x = F.interpolate(x, scale_factor=2, mode=self.interpolate)

        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.block(x)
        return x


 # skip: torch.Size([1, 128, 32, 32])
 # upsample: torch.Size([1, 256, 16, 16])

 # output index 2 --> skip
 # output index 3 --> upscale


class DoubleConv2DReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, use_batchnorm):
        super().__init__()
        layers = []
        layers.append(Conv2dReLU(in_channels, out_channels, kernel_size=kernel_size,
                                 padding=padding, use_batchnorm=use_batchnorm))
        layers.append(Conv2dReLU(out_channels, out_channels, kernel_size=kernel_size,
                                 padding=padding, use_batchnorm=use_batchnorm))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        x = self.block(x)
        return x

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_batchnorm=True):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv1 = DoubleConv2DReLU(in_channels, out_channels, kernel_size=3,
                                      padding=1, use_batchnorm=use_batchnorm)

    def forward(self, x):
        y = self.relu(x)
        y = self.bn1(y)
        y = self.conv1(y)

        x = self.bn2(x)

        x = torch.cat([x, y])
        return x

class UNetPlusPlusDecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, use_batchnorm=True,
                 interpolate="nearest", dropout=None):
        super().__init__()
        self.interpolate = interpolate

        self.conv_transpose_block = None
        if self.interpolate is None:
            self.conv_transpose_block = nn.ConvTranspose2d(in_channels, in_channels,
                                                           kernel_size=2, stride=2)

        layers = []
        if dropout is not None:
            layers.append(nn.Dropout(dropout))

        layers.append(DoubleConv2DReLU(in_channels + skip_channels, out_channels,
                                       kernel_size=3, padding=1,
                                       use_batchnorm=use_batchnorm))

        self.block = nn.Sequential(*layers)

    # first upsampling, second skip connections
    def forward(self, x):
        y = x[0]
        skips = x[1:]
        if self.conv_transpose_block is not None:
            y = self.conv_transpose_block(y)
        else:
            y = F.interpolate(y, scale_factor=2, mode=self.interpolate)

        if len(skips) > 0:
            y = torch.cat([y] + skips, dim=1)

        y = self.block(y)
        return y

class UNetPlusPlusDecoder(ModelBase):
    def __init__(self, enc_ch, interpolate, dec_ch=(256, 128, 64, 32, 16),
                 out_channels=1, use_batchnorm=True, dropout=None):
        super().__init__()
        self.depth = 5
        assert(self.depth > 2)

        out_ch = dec_ch

        self.layer31 = UNetPlusPlusDecoderBlock(enc_ch[0],   enc_ch[1], out_ch[0], use_batchnorm=use_batchnorm, interpolate=interpolate, dropout=dropout)
        self.layer22 = UNetPlusPlusDecoderBlock(dec_ch[0], 2*enc_ch[2], out_ch[1], use_batchnorm=use_batchnorm, interpolate=interpolate, dropout=dropout)
        self.layer13 = UNetPlusPlusDecoderBlock(dec_ch[1], 3*enc_ch[3], out_ch[2], use_batchnorm=use_batchnorm, interpolate=interpolate, dropout=dropout)
        self.layer04 = UNetPlusPlusDecoderBlock(dec_ch[2], 4*enc_ch[4], out_ch[3], use_batchnorm=use_batchnorm, interpolate=interpolate, dropout=dropout)

        self.layer21 = UNetPlusPlusDecoderBlock(enc_ch[1],   enc_ch[2], out_ch[1], use_batchnorm=use_batchnorm, interpolate=interpolate, dropout=dropout)
        self.layer12 = UNetPlusPlusDecoderBlock(enc_ch[2], 2*enc_ch[3], out_ch[2], use_batchnorm=use_batchnorm, interpolate=interpolate, dropout=dropout)
        self.layer03 = UNetPlusPlusDecoderBlock(enc_ch[3], 3*enc_ch[4], out_ch[2], use_batchnorm=use_batchnorm, interpolate=interpolate, dropout=dropout)

        self.layer11 = UNetPlusPlusDecoderBlock(enc_ch[2],   enc_ch[3], out_ch[2], use_batchnorm=use_batchnorm, interpolate=interpolate, dropout=dropout)
        self.layer02 = UNetPlusPlusDecoderBlock(enc_ch[3], 2*enc_ch[4], out_ch[2], use_batchnorm=use_batchnorm, interpolate=interpolate, dropout=dropout)

        self.layer01 = UNetPlusPlusDecoderBlock(enc_ch[3], enc_ch[4], out_ch[2], use_batchnorm=use_batchnorm, interpolate=interpolate, dropout=dropout)

        self.prefinal = UNetPlusPlusDecoderBlock(out_ch[3], 0, out_ch[4], use_batchnorm=use_batchnorm, interpolate=interpolate, dropout=dropout)
        self.final_layer = nn.Conv2d(out_ch[4], out_channels, kernel_size=(1, 1))
        self.initialize()

    def forward(self, x):
        y = x[0]
        skips = x[1:]

        y31 = self.layer31([y, skips[0]])

        y21 = self.layer21([skips[0], skips[1]])
        y22 = self.layer22([y31,y21,skips[1]])

        y11 = self.layer11([skips[1], skips[2]])
        y12 = self.layer12([y21, y11, skips[2]])
        y13 = self.layer13([y22,y12,y11,skips[2]])

        y01 = self.layer01([skips[2], skips[3]])
        y02 = self.layer02([y11, y01, skips[3]])
        y03 = self.layer03([y12, y02, y01, skips[3]])
        y04 = self.layer04([y13, y03, y02, y01, skips[3]])

        y = self.prefinal([y04])
        y = self.final_layer(y)

        return y


class UNetDecoder(ModelBase):
    def __init__(self, enc_ch, interpolate, dec_ch=(256, 128, 64, 32, 16),
                 out_channels=1, use_batchnorm=True):
        super().__init__()
        self.depth = 5
        assert(self.depth > 2)

        in_ch = self.layer_channels(enc_ch, dec_ch)
        out_ch = dec_ch

        self.layer = nn.ModuleList()
        for i in range(self.depth):
            self.layer.append(UNetDecoderBlock(in_ch[i], out_ch[i], use_batchnorm=use_batchnorm, interpolate=interpolate))
        self.final_layer = nn.Conv2d(out_ch[4], out_channels, kernel_size=(1, 1))
        self.initialize()

    def layer_channels(self, enc_ch, dec_ch):
        channels = self.depth*[None]

        channels[0] = enc_ch[0] + enc_ch[1]
        for i in range(1,self.depth-1):
            channels[i] = enc_ch[i+1] + dec_ch[i-1]
        channels[self.depth-1] = dec_ch[self.depth-2]
        return channels

    def forward(self, x):
        y = x[0]
        skips = x[1:] + [None]
        for i in range(self.depth):
            y = self.layer[i]([y, skips[i]])
        y = self.final_layer(y)
        return y


class ResNetModel(ResNet):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pretrained = False
        del self.fc

    def forward(self, x):
        x0 = self.conv1(x)
        x0 = self.bn1(x0)
        x0 = self.relu(x0)

        x1 = self.maxpool(x0)
        x1 = self.layer1(x1)

        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        return [x4, x3, x2, x1, x0]

    def load_state_dict(self, state_dict, **kwargs):
        state_dict.pop('fc.bias')
        state_dict.pop('fc.weight')
        super().load_state_dict(state_dict, **kwargs)

    @staticmethod
    def resnet18(pretrained=None):
        block = BasicBlock
        layers = [2, 2, 2, 2]
        out_ch = (512, 256, 128, 64, 64)
        name = "resnet18"
        return ResNetModel._create_resnet(block, layers, out_ch, name, pretrained)

    @staticmethod
    def resnet34(pretrained=None):
        block = BasicBlock
        layers = [3, 4, 6, 3]
        out_ch = (512, 256, 128, 64, 64)
        name = "resnet34"
        return ResNetModel._create_resnet(block, layers, out_ch, name, pretrained)

    @staticmethod
    def resnet50(pretrained=None):
        block = BasicBlock
        layers = [3, 4, 6, 3]
        out_ch = (2048, 1024, 512, 256, 64)
        name = "resnet50"
        return ResNetModel._create_resnet(block, layers, out_ch, name, pretrained)

    @staticmethod
    def input_preprocess_function(name, pretrained='imagenet'):
        settings = pretrained_settings[name]

        input_space = settings[pretrained].get('input_space')
        input_range = settings[pretrained].get('input_range')
        mean = settings[pretrained].get('mean')
        std = settings[pretrained].get('std')

        return functools.partial(normalize_input, mean=mean, std=std, input_space=input_space, input_range=input_range)

    @staticmethod
    def _create_resnet(block, layers, out_ch, name, pretrained):
        resnet = ResNetModel(block=block, layers=layers)
        resnet.out_shapes = out_ch

        if pretrained is not None:
            settings = pretrained_settings[name][pretrained]
            resnet.load_state_dict(model_zoo.load_url(settings['url']))
        return resnet


def ResUNet(resnet, pretrained=None, interpolate="nearest"):
    encoder = getattr(ResNetModel, resnet)(pretrained)
    decoder = UNetDecoder((512,256,128,64,64), interpolate)
    return SegmentationModel(encoder, decoder)

def ResUNetPlusPlus(resnet, pretrained=None, interpolate="nearest"):
    encoder = getattr(ResNetModel, resnet)(pretrained)
    decoder = UNetPlusPlusDecoder((512,256,128,64,64), interpolate)
    return SegmentationModel(encoder, decoder)

class SegmentationModel(ModelBase):

    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.activation = nn.Sigmoid()

    # forward() without final activation --> loss function needs to take care!
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def predict(self, x):
        if self.training:
            self.eval()
        with torch.no_grad():
            x = self.forward(x)
            x = self.activation(x)
        return x
