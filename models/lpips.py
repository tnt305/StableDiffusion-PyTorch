from __future__ import absolute_import
from collections import namedtuple
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable
import numpy as np
import torch.nn
import torchvision

# Taken from https://github.com/richzhang/PerceptualSimilarity/blob/master/lpips/lpips.py

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def spatial_average(in_tens, keepdim=True):
    return in_tens.mean([2, 3], keepdim=keepdim)

class vgg16(torch.nn.Module):
    def __init__(self, requires_grad=False, pretrained=True):
        super(vgg16, self).__init__()
        vgg_pretrained_features = torchvision.models.vgg16(pretrained=pretrained).features
        self.slices = nn.ModuleList()
        slice_indices = [0, 4, 9, 16, 23, 30]
        for i in range(len(slice_indices)-1):
            slice_module = nn.Sequential()
            for j in range(slice_indices[i], slice_indices[i+1]):
                slice_module.add_module(str(j), vgg_pretrained_features[j])
            self.slices.append(slice_module)
        
        # Freeze vgg model
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False
    
    def forward(self, X):
        h_rels = []
        for slice_module in self.slices:
            X = slice_module(X)
            h_rels.append(X)
        vgg_outputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3', 'relu5_3'])
        out = vgg_outputs(*h_rels)
        return out

class ScalingLayer(nn.Module):
    def __init__(self):
        super(ScalingLayer, self).__init__()
        # Imagenet normalization
        self.register_buffer('shift', torch.Tensor([-.030, -.088, -.188])[None, :, None, None])
        self.register_buffer('scale', torch.Tensor([.458, .448, .450])[None, :, None, None])
    
    def forward(self, inp):
        return (inp - self.shift) / self.scale

class LPIPS(nn.Module):
    def __init__(self, net='vgg', version='0.1', use_dropout=True):
        super(LPIPS, self).__init__()
        self.version = version
        self.scaling_layer = ScalingLayer()
        self.chns = [64, 128, 256, 512, 512]
        self.L = len(self.chns)
        self.net = vgg16(pretrained=True, requires_grad=False)
        
        # Add 1x1 convolutional Layers
        self.lin0 = nn.Conv2d(self.chns[0], 1, kernel_size=1, stride=1, padding=0)
        self.lin1 = nn.Conv2d(self.chns[1], 1, kernel_size=1, stride=1, padding=0)
        self.lin2 = nn.Conv2d(self.chns[2], 1, kernel_size=1, stride=1, padding=0)
        self.lin3 = nn.Conv2d(self.chns[3], 1, kernel_size=1, stride=1, padding=0)
        self.lin4 = nn.Conv2d(self.chns[4], 1, kernel_size=1, stride=1, padding=0)
        self.lins = nn.ModuleList([self.lin0, self.lin1, self.lin2, self.lin3, self.lin4])
        
        # Load the weights of trained LPIPS model
        model_dir = os.path.join(os.path.dirname(__file__), 'weights', f'v{version}')
        model_path = os.path.join(model_dir, f'{net}.pth')
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model weights not found at {model_path}")
        
        try:
            self.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
            print('Model loaded successfully from:', model_path)
        except Exception as e:
            raise RuntimeError(f"Error loading model: {e}")
        
        # Freeze all parameters
        self.eval()
        for param in self.parameters():
            param.requires_grad = False
    
    def forward(self, in0, in1, normalize=False):
        if normalize:
            in0 = 2 * in0 - 1
            in1 = 2 * in1 - 1
        
        in0_input, in1_input = self.scaling_layer(in0), self.scaling_layer(in1)
        
        outs0, outs1 = self.net.forward(in0_input), self.net.forward(in1_input)
        feats0, feats1, diffs = {}, {}, {}
        
        for kk in range(self.L):
            feats0[kk], feats1[kk] = torch.nn.functional.normalize(outs0[kk], dim=1), torch.nn.functional.normalize(outs1[kk])
            diffs[kk] = (feats0[kk] - feats1[kk]) ** 2
        
        res = [spatial_average(self.lins[kk](diffs[kk]), keepdim=True) for kk in range(self.L)]
        val = sum(res)
        return val


class ScalingLayer(nn.Module):
    def __init__(self):
        super(ScalingLayer, self).__init__()
        # Imagnet normalization for (0-1)
        # mean = [0.485, 0.456, 0.406]
        # std = [0.229, 0.224, 0.225]
        self.register_buffer('shift', torch.Tensor([-.030, -.088, -.188])[None, :, None, None])
        self.register_buffer('scale', torch.Tensor([.458, .448, .450])[None, :, None, None])
    
    def forward(self, inp):
        return (inp - self.shift) / self.scale


class NetLinLayer(nn.Module):
    ''' A single linear layer which does a 1x1 conv '''
    
    def __init__(self, chn_in, chn_out=1, use_dropout=False):
        super(NetLinLayer, self).__init__()
        
        layers = [nn.Dropout(), ] if (use_dropout) else []
        layers += [nn.Conv2d(chn_in, chn_out, 1, stride=1, padding=0, bias=False), ]
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        out = self.model(x)
        return out
