import torch
import torch.nn as nn

class Conv2dReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0,
                 stride=1, use_batchnorm=True, activation=True, **batchnorm_params):

        super().__init__()

        layers = []
        layers.append(
            nn.Conv2d(in_channels, out_channels, kernel_size,
                      stride=stride, padding=padding, bias=not (use_batchnorm)) 
        )
        
        if use_batchnorm:
            layers.append(nn.BatchNorm2d(out_channels, **batchnorm_params))
        
        if activation:
            layers.append(nn.ReLU(inplace=True))
            
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)

class DoubleConv2DReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0, stride=1, use_batchnorm=True):
        super().__init__()
        layers = []
        layers.append(Conv2dReLU(in_channels, out_channels, 
                                 kernel_size=kernel_size, padding=padding, 
                                 use_batchnorm=use_batchnorm,
                                 stride=stride))
        layers.append(Conv2dReLU(out_channels, out_channels, 
                                 kernel_size=kernel_size, padding=padding, 
                                 use_batchnorm=use_batchnorm,
                                 stride=stride))
        self.block = nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.block(x)
        return x
 
class ResidualBlock(nn.Module):
    def __init__(self, channels, kernel_size, padding, stride=1, use_batchnorm=True):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm2d(channels)
        self.bn2 = nn.BatchNorm2d(channels)
        self.conv1 = Conv2dReLU(channels, channels, kernel_size=kernel_size, 
                                padding=padding, stride=stride, 
                                use_batchnorm=use_batchnorm)
        self.conv2 = Conv2dReLU(channels, channels, kernel_size=kernel_size, 
                                padding=padding, stride=stride, 
                                use_batchnorm=use_batchnorm, activation=False)
        
    def forward(self, x):
        y = self.relu(x)
        y = self.bn1(y)
        y = self.conv1(y)
        y = self.conv2(y)
        
        x = self.bn2(x)
        
        x = torch.add(x, y)
        return x