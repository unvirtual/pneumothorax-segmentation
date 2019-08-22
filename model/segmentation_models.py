import torch
from . import model_base as mb
from . import unet_decoder as unet_decoder
from . import unetpp_decoder as unetpp_decoder
from .resnet import ResNetEncoder
from .efficientnet import EffNetEncoder
import numpy as np

def ResUNet(resnet, pretrained, dropout, interpolate):
    encoder, out_ch = getattr(ResNetEncoder, resnet)(pretrained)
    decoder = unet_decoder.UNetDecoder(out_ch, dropout=dropout, interpolate=interpolate)
    return mb.SegmentationModel(encoder, decoder)

def ResUNetPlusPlus(resnet, pretrained, interpolate, dropout, decoder_type):
    encoder, out_ch = getattr(ResNetEncoder, resnet)(pretrained)
    decoder = unetpp_decoder.UNetPlusPlusDecoder(out_ch, interpolate, dropout=dropout, decoder_type=decoder_type)
    return mb.SegmentationModel(encoder, decoder)

def EffUNet(effnet, pretrained, dropout, interpolate):
    encoder, out_ch = getattr(EffNetEncoder, effnet)(pretrained)
    decoder = unet_decoder.UNetDecoder(out_ch, dropout=dropout, interpolate=interpolate)
    return mb.SegmentationModel(encoder, decoder)

def EffUNetPlusPlus(effnet, pretrained, interpolate, dropout, decoder_type):
    encoder, out_ch = getattr(EffNetEncoder, effnet)(pretrained)
    decoder = unetpp_decoder.UNetPlusPlusDecoder(out_ch, dec_ch=(128,64,32,16,8), interpolate=interpolate, decoder_type=decoder_type, dropout=dropout)
    return mb.SegmentationModel(encoder, decoder)

def n_paramters(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params

if __name__ == "__main__":
#    model = ResUNet("resnet34_encoder", pretrained=None, dropout=None, interpolate="nearest")
#    ll = model(torch.zeros((2,3,128,128)))
#    print("ResUNet: ", ll.shape)
#    print(n_paramters(model))
#    
#    model = ResUNetPlusPlus("resnet34_encoder", pretrained=None, interpolate=None, decoder_type="default", dropout=0.1)
#    ll = model(torch.zeros((2,3,128,128)))
#    print("ResUNetPlusPlus: ", ll.shape)
#    print(n_paramters(model))

#    
#    model = EffUNet("effnet_b4_encoder", dropout=None, pretrained=None, interpolate=None)
#    ll = model(torch.zeros((2,3,128,128)))
#    print("EffB4UNet: ", ll.shape)
#    print(n_paramters(model))

 
    model = EffUNetPlusPlus("effnet_b4_encoder", pretrained=None, interpolate=None, decoder_type="residual", dropout=0.1)
    ll = model(torch.zeros((1,3,256,256)))
    print("EffB4UNetPlusPlus: ", ll.shape)
    print(n_paramters(model))
    
