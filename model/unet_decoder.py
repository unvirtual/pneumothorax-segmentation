import torch
import torch.nn as nn
import torch.nn.functional as F
from . import blocks as blocks
from .model_base import ModelBase 

class UNetDecoder(ModelBase):
    def __init__(self, enc_ch, interpolate, dropout, dec_ch=(256, 128, 64, 32, 16),
                 out_channels=1, use_batchnorm=True):
        super().__init__()
        self.depth = 5
        assert(self.depth > 2)
        
        in_ch, skip_channels = self.layer_channels(enc_ch, dec_ch)
        out_ch = dec_ch

        self.layer = nn.ModuleList()
        # Blocks with skip connections
        for i in range(self.depth-1):
            self.layer.append(UNetDecoderBlock(in_ch[i], skip_channels[i], out_ch[i], dropout=dropout, use_batchnorm=use_batchnorm, interpolate=interpolate))
        # final blocks
        self.layer.append(UNetDecoderBlock(dec_ch[3], 0, out_ch[4], dropout=dropout, use_batchnorm=use_batchnorm, interpolate=interpolate))
        self.final_layer = nn.Conv2d(out_ch[4], out_channels, kernel_size=(1, 1))
        self.initialize()

    def layer_channels(self, enc_ch, dec_ch):
        in_ch = self.depth*[None]
        skip_ch = self.depth*[None]
    
        in_ch[0] = enc_ch[0]        
        for i in range(0,self.depth-1):
            skip_ch[i] = enc_ch[i+1]
            if i > 0:
                in_ch[i] = dec_ch[i-1]
        return in_ch, skip_ch
        
    def forward(self, x):
        y = x[0]
        skips = x[1:] + [None] 
        for i in range(self.depth):
            y = self.layer[i]([y, skips[i]])
        y = self.final_layer(y)
        return y
    
class UNetDecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, dropout, use_batchnorm, interpolate):
        super().__init__()
        self.interpolate = interpolate
        
        self.conv_transpose_block = None
        if self.interpolate is None:
            self.conv_transpose_block = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2)
            
        layers = []
        if dropout is not None:
            layers.append(nn.Dropout(dropout))
        
        layers.append(blocks.DoubleConv2DReLU(in_channels + skip_channels, out_channels, kernel_size=3, padding=1, use_batchnorm=use_batchnorm))
    
        self.block = nn.Sequential(*layers)
        
    # first upsampling, second skip connections
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
