import functools
from torchvision.models.resnet import ResNet
from torchvision.models.resnet import BasicBlock
from pretrainedmodels.models.torchvision_models import pretrained_settings
from torch.utils import model_zoo

from .utils import normalize_input

class ResNetEncoder(ResNet):
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
    def resnet18_encoder(pretrained=None):
        block = BasicBlock
        layers = [2, 2, 2, 2]
        out_ch = (512, 256, 128, 64, 64)
        name = "resnet18"
        return (ResNetEncoder._create_resnet(block, layers, out_ch, name, pretrained),
                out_ch)
    
    @staticmethod
    def resnet34_encoder(pretrained=None):
        block = BasicBlock
        layers = [3, 4, 6, 3]
        out_ch = (512, 256, 128, 64, 64)
        name = "resnet34"
        return (ResNetEncoder._create_resnet(block, layers, out_ch, name, pretrained),
                out_ch)
    
    @staticmethod
    def resnet50_encoder(pretrained=None):
        block = BasicBlock
        layers = [3, 4, 6, 3]
        out_ch = (2048, 1024, 512, 256, 64)
        name = "resnet50"
        return (ResNetEncoder._create_resnet(block, layers, out_ch, name, pretrained),
                out_ch)
    
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
        resnet = ResNetEncoder(block=block, layers=layers) 
        resnet.out_shapes = out_ch
        
        if pretrained is not None:
            settings = pretrained_settings[name][pretrained]
            resnet.load_state_dict(model_zoo.load_url(settings['url']))
        return resnet
