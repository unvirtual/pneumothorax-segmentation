import torch
import torch.nn as nn
import torch.nn.functional as F
from . import blocks as blocks
from .model_base import ModelBase 
import numpy as np

class UNetPlusPlusDecoder(ModelBase):
    def __init__(self, enc_ch, interpolate, dec_ch=(256, 128, 64, 32, 16),
                 out_channels=1, use_batchnorm=True, dropout=None,
                 decoder_type="default"):
        super(UNetPlusPlusDecoder, self).__init__()
        
        self.depth = 4
        assert(self.depth > 2)
        
        if decoder_type == "default":
            self.block_cls = UNetPlusPlusDecoderBlock
        elif decoder_type == "residual":
            self.block_cls = UNetPlusPlusResDecoderBlock
        
        self.layer = nn.ModuleList([]*self.n_layers())
        ins, outs, skips = self.layer_channels(enc_ch, dec_ch)
        
        for i in range(self.depth):
            for j in range(self.depth-i):
                channels = (ins[i,j], skips[i,j], outs[i,j])
                block = self.block_cls(*channels, use_batchnorm=use_batchnorm, 
                                       interpolate=interpolate, dropout=dropout)
                index = self.layer_index(i,j)
                self.layer.insert(index, block)

        self.prefinal = self.block_cls(dec_ch[self.depth-1], 0, 
                                       dec_ch[self.depth], 
                                       use_batchnorm=use_batchnorm, 
                                       interpolate=interpolate, 
                                       dropout=dropout)
        
        self.final_layer = nn.Conv2d(dec_ch[self.depth], 
                                     out_channels, 
                                     kernel_size=(1, 1))
        self.initialize()
         
    def n_layers(self):
        return sum(range(self.depth + 1)) 
    
    def layer_index(self, n,m):
        return sum(range(self.depth + 1 - n, self.depth + 1)) + m
        
    # Assumption: Channels remaoin constant for k=const in out[k,n]
    def layer_channels(self, enc_ch, dec_ch):
        enc_ch = list(reversed(enc_ch))
        dec_ch = list(reversed(dec_ch))
        
        outs  = np.ndarray((self.depth, self.depth)).astype(int)
        ins   = np.ndarray((self.depth, self.depth)).astype(int)
        skips = np.ndarray((self.depth, self.depth)).astype(int)
        
        outs[:] = np.nan
        ins[:] = np.nan
        skips[:] = np.nan
            
        for n in range(self.depth):
            for m in range(0, self.depth - n):
                outs[n,m] = dec_ch[n+1]
                        
        for n in range(self.depth):
            ins[n,0] = enc_ch[n+1]
            if n < self.depth - 1:
                for m in range(1, self.depth):
                    ins[n,m] = outs[n+1,m-1]
                
            for m in range(0, self.depth - n):    
                skips[n,m] = enc_ch[n]
                for k in range(m):
                    skips[n,m] += outs[n,k]                     
        return (ins, outs, skips)  

    def layer_input(self, x, y, n, m):
        # inputs
        if m == 0:
            args = [x[self.depth-n-1]]
        else:
            args = [y[self.layer_index(n+1, m-1)]]    
        # skips
        for k in reversed(range(m)):
            args.append(y[self.layer_index(n, k)])
        args.append(x[self.depth-n])
        return args
            
    def forward(self, x):
        y = [np.nan]*self.n_layers()
        
        for m in range(self.depth):
            for n in range(self.depth-m):
                args = self.layer_input(x, y, n, m)
                index = self.layer_index(n, m)
                y[index] = self.layer[index](args)

        y = self.prefinal([y[self.layer_index(0,3)]])
        y = self.final_layer(y)
        return y

class UNetPlusPlusDecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, interpolate, dropout, use_batchnorm=True):
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
    
class UNetPlusPlusResDecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, interpolate, dropout, use_batchnorm=True):
        super().__init__()
        self.interpolate = interpolate
        
        self.conv_transpose_block = None
        if self.interpolate is None:
            self.conv_transpose_block = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2)
            
        layers = []
        if dropout is not None:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Conv2d(in_channels + skip_channels, out_channels, kernel_size=3, padding=1))
        layers.append(blocks.ResidualBlock(out_channels, kernel_size=3, padding=1, use_batchnorm=use_batchnorm))
        layers.append(nn.ReLU(inplace=True))
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
