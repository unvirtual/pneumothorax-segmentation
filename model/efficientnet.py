import torch.nn as nn
from . import blocks
import functools
from numpy import cumsum
from efficientnet_pytorch import EfficientNet
from efficientnet_pytorch.utils import get_model_params, relu_fn, load_pretrained_weights
from .utils import normalize_input

class EffNetEncoder(EfficientNet):
    def __init__(self, *args, **kwargs):        
        super(EffNetEncoder, self).__init__(*args, **kwargs)
        
        out_ch = 256
        self.middle_block = nn.Sequential(
                 #nn.MaxPool2d(kernel_size=2),
                 nn.Conv2d(1792, out_ch, kernel_size=3, stride=1, padding=1),
                 blocks.ResidualBlock(out_ch, kernel_size=3, padding=1),
                 blocks.ResidualBlock(out_ch, kernel_size=3, padding=1),
                 nn.ReLU(inplace=True)
        )        
        del self._fc
        
    def extract_features(self, inputs):
        """ Returns output of the final convolution layer """

        out_block_ids = []
        for b in self._blocks_args:
            out_block_ids.append(b.num_repeat*2)
        out_block_ids = [x - 1 for x in list(cumsum(out_block_ids))]
            
        # Stem
        x = relu_fn(self._bn0(self._conv_stem(inputs)))
        res = []
        # Blocks
        for idx, block in enumerate(self._blocks):
            drop_connect_rate = self._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self._blocks)
            y = block(x, drop_connect_rate=drop_connect_rate)
            if idx in out_block_ids:
                res.append(y)
            x = y
        x = relu_fn(self._bn1(self._conv_head(x)))
        
        x = self.middle_block(x)
        
        res = res[0:3] + [res[4], x]
        res.reverse()
        return res
        
    def forward(self, x):
        x = self.extract_features(x)
        return x
    
    @classmethod
    def _create_effnet_encoder(cls, name, pretrained):
        cls._check_model_name_is_valid(name)
        
        block_args, global_params = get_model_params(name, None)
        model = cls(block_args, global_params)
        if pretrained is not None:
            load_pretrained_weights(model, name, load_fc=False)
        return model
    
    @staticmethod
    def input_preprocess_function(name):    
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        return functools.partial(normalize_input, mean=mean, std=std)
    
    @classmethod
    def effnet_b4_encoder(cls, pretrained=None):
        name = "efficientnet-b4"
        out_ch = (256,160,56,32,24)
        model = cls._create_effnet_encoder(name, pretrained)
        return (model, out_ch)
    
    
