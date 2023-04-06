"""
Ported to pytorch thanks to [tstandley](https://github.com/tstandley/Xception-PyTorch)

@author: tstandley
Adapted by cadene

Creates an Xception Model as defined in:

Francois Chollet
Xception: Deep Learning with Depthwise Separable Convolutions
https://arxiv.org/pdf/1610.02357.pdf

This weights ported from the Keras implementation. Achieves the following performance on the validation set:

Loss:0.9173 Prec@1:78.892 Prec@5:94.292

REMEMBER to set your image size to 3x299x299 for both test and validation

normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                  std=[0.5, 0.5, 0.5])

The resize parameter of the validation transform should be 333, and make sure to center crop at 299x299
"""
from __future__ import print_function, division, absolute_import
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torch.nn import init

__all__ = ['xception']

pretrained_settings = {
    'xception': {
        'imagenet': {
            'url': 'http://data.lip6.fr/cadene/pretrainedmodels/xception-43020ad28.pth',
            'input_space': 'RGB',
            'input_size': [3, 299, 299],
            'input_range': [0, 1],
            'mean': [0.5, 0.5, 0.5],
            'std': [0.5, 0.5, 0.5],
            'num_classes': 1000,
            'scale': 0.8975 # The resize parameter of the validation transform should be 333, and make sure to center crop at 299x299
        }
    }
}


class GloAN(nn.Module):
    def __init__(self, in_dim=128, embed_dim=64, height=56, width=56):
        super(GloAN, self).__init__()

        self.embed = nn.Linear(in_dim, embed_dim)
        self.dropout = nn.Dropout(0.1)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        # self.pos_embed = nn.Parameter(torch.zeros(1, height*width, embed_dim))  # optional
        self.qkv = nn.Linear(embed_dim, embed_dim*3)
        self.proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.Dropout(0.1)
        )
        self.mlp = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim, embed_dim),
            nn.Dropout(0.1)
        )
        # self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        B, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # shape=(1,16384,128)
        x = self.embed(x)
        x = self.norm1(x)
        # x += self.pos_embed  # optional
        x = self.dropout(x)
        x = self.norm2(x)
        q, k, v = self.qkv(x).chunk(3, dim=-1)   # q.shape=k.shape=v.shape=(1,16384,64)
        attn = q @ k.transpose(-2, -1) * 0.125   # attn.shape=(1,16384,16384)
        attn = attn.softmax(dim=-1)
        x = attn @ v  # shape=(1,3136,2)  # x.shape=(1,16384,64)
        x = self.proj(x)  # x.shape=(1,16384,64)
        x = self.mlp(x).transpose(1, 2).reshape(B, 64, H, W)   # x.shape=(1,64,128,128)

        return torch.mean(x, dim=1, keepdim=True)


class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False):
        super(SeparableConv2d, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(in_channels,out_channels,1,1,0,1,1,bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x


class Block(nn.Module):
    def __init__(self, in_filters, out_filters, reps, strides=1, start_with_relu=True, grow_first=True, padding=1, dilation=1):
        super(Block, self).__init__()

        if out_filters != in_filters or strides!=1:
            self.skip = nn.Conv2d(in_filters, out_filters, 1, stride=strides, bias=False)
            self.skipbn = nn.BatchNorm2d(out_filters)
        else:
            self.skip = None

        rep = []

        filters = in_filters

        if grow_first:
            rep.append(nn.ReLU(inplace=True))
            rep.append(SeparableConv2d(in_filters, out_filters, 3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(out_filters))
            filters = out_filters

        for i in range(reps-1):
            rep.append(nn.ReLU(inplace=True))
            rep.append(SeparableConv2d(filters,filters,3,stride=1,padding=padding,bias=False, dilation=dilation))
            rep.append(nn.BatchNorm2d(filters))

        if not grow_first:
            rep.append(nn.ReLU(inplace=True))
            rep.append(SeparableConv2d(in_filters,out_filters,3,stride=1,padding=1,bias=False))
            rep.append(nn.BatchNorm2d(out_filters))

        if not start_with_relu:
            rep = rep[1:]
        else:
            rep[0] = nn.ReLU(inplace=False)

        if strides != 1:
            rep.append(nn.MaxPool2d(3, strides, 1))
        self.rep = nn.Sequential(*rep)

    def forward(self, inp):
        x = self.rep(inp)

        if self.skip is not None:
            skip = self.skip(inp)
            skip = self.skipbn(skip)
        else:
            skip = inp

        x+=skip
        return x


class Xception(nn.Module):
    """
    Xception optimized for the ImageNet dataset, as specified in
    https://arxiv.org/pdf/1610.02357.pdf
    """
    def __init__(self, num_classes=1000):
        """ Constructor
        Args:
            num_classes: number of classes
        """
        super(Xception, self).__init__()
        self.num_classes = num_classes

        self.conv1 = nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(32,64,3, padding=1,bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)
        # do relu here

        self.block1=Block(64,128,2,2,start_with_relu=False,grow_first=True)
        self.block2=Block(128,256,2,2,start_with_relu=True,grow_first=True)
        self.block3=Block(256,728,2,2,start_with_relu=True,grow_first=True)

        self.block4=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block5=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block6=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block7=Block(728,728,3,1,start_with_relu=True,grow_first=True)

        self.block8=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block9=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block10=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block11=Block(728,728,3,1,start_with_relu=True,grow_first=True)

        self.block12=Block(728,1024,2,1,start_with_relu=True,grow_first=False, padding=2, dilation=2)
        self.conv3 = SeparableConv2d(1024,1536,3,1,1,dilation=1)
        self.bn3 = nn.BatchNorm2d(1536)
        self.relu3 = nn.ReLU(inplace=True)

        self.conv4 = SeparableConv2d(1536,2048,3,1,1, dilation=1)
        self.bn4 = nn.BatchNorm2d(2048)

        self.fc = nn.Linear(2048, num_classes)
        # 512
        self.attn1 = GloAN(in_dim=128, height=128, width=128)
        self.attn2 = GloAN(in_dim=256, height=64, width=64)
        self.attn3 = GloAN(in_dim=728, height=32, width=32)
        self.attn4 = GloAN(in_dim=728, height=32, width=32)
        self.attn5 = GloAN(in_dim=728, height=32, width=32)
        self.attn6 = GloAN(in_dim=728, height=32, width=32)
        self.attn7 = GloAN(in_dim=728, height=32, width=32)
        self.attn8 = GloAN(in_dim=728, height=32, width=32)
        self.attn9 = GloAN(in_dim=728, height=32, width=32)
        self.attn10 = GloAN(in_dim=728, height=32, width=32)
        self.attn11 = GloAN(in_dim=728, height=32, width=32)
        self.attn12 = GloAN(in_dim=1024, height=32, width=32)
        self.attn13 = GloAN(in_dim=2048, height=32, width=32)
        # 224
        # self.attn1 = GloAN(in_dim=128, height=56, width=56)
        # self.attn2 = GloAN(in_dim=256, height=28, width=28)
        # self.attn3 = GloAN(in_dim=728, height=14, width=14)
        # self.attn4 = GloAN(in_dim=728, height=14, width=14)
        # self.attn5 = GloAN(in_dim=728, height=14, width=14)
        # self.attn6 = GloAN(in_dim=728, height=14, width=14)
        # self.attn7 = GloAN(in_dim=728, height=14, width=14)
        # self.attn8 = GloAN(in_dim=728, height=14, width=14)
        # self.attn9 = GloAN(in_dim=728, height=14, width=14)
        # self.attn10 = GloAN(in_dim=728, height=14, width=14)
        # self.attn11 = GloAN(in_dim=728, height=14, width=14)
        # self.attn12 = GloAN(in_dim=1024, height=14, width=14)
        # self.attn13 = GloAN(in_dim=2048, height=14, width=14)
        # #------- init weights --------
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.normal_(0, math.sqrt(2. / n))
        #     elif isinstance(m, nn.BatchNorm2d):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()
        # #-----------------------------

    def features(self, input):   # input.shape=(1,3,512,512)
        x = self.conv1(input)  # x.shape=(1,32,256,256)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)  # x.shape=(1,64,256,256)
        x = self.bn2(x)
        x = self.relu2(x)

        low_level_feature = self.block1(x)
        attn = self.attn1(low_level_feature)
        low_level_feature = low_level_feature + attn * low_level_feature    # shape=(1,128,128,128)     1

        x = self.block2(low_level_feature)   # shape=(1,256,64,64)   2
        attn = self.attn2(x)
        x = x + attn * x

        x = self.block3(x)   # shape=(1,728,32,32)   3
        attn = self.attn3(x)
        x = x + attn * x

        x = self.block4(x)   # shape=(1,728,32,32)  4
        attn = self.attn4(x)
        x = x + attn * x

        x = self.block5(x)  # shape=(1,728,32,32)
        attn = self.attn5(x)
        x = x + attn * x

        x = self.block6(x)  # shape=(1,728,32,32)
        attn = self.attn6(x)
        x = x + attn * x

        x = self.block7(x)  # shape=(1,728,32,32)  5
        attn = self.attn7(x)
        x = x + attn * x

        x = self.block8(x)  # shape=(1,728,32,32)
        attn = self.attn8(x)
        x = x + attn * x

        x = self.block9(x)  # shape=(1,728,32,32)
        attn = self.attn9(x)
        x = x + attn * x

        x = self.block10(x)  # shape=(1,728,32,32)  6
        attn = self.attn10(x)
        x = x + attn * x

        x = self.block11(x)  # shape=(1,728,32,32)
        attn = self.attn11(x)
        x = x + attn * x

        x = self.block12(x)  # shape=(1,1024,32,32)  7
        attn = self.attn12(x)
        x = x + attn * x

        x = self.conv3(x)  # shape=(1,1536,32,32)
        x = self.bn3(x)
        x = self.relu3(x)

        x = self.conv4(x)  # shape=(1,2048,32,32)   # 8
        x = self.bn4(x)
        attn = self.attn13(x)
        x = x + attn * x

        return low_level_feature, x

    def logits(self, features):
        x = nn.ReLU(inplace=True)(features)

        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        x = self.last_linear(x)
        return x

    def forward(self, input):
        low_level_feature, x = self.features(input)
        # x = self.logits(x)
        return low_level_feature, x


def xception_attn(num_classes=1000, pretrained=True):
    model = Xception(num_classes=num_classes)
    if pretrained:
        pre_dict = torch.load('./PTH/xception-43020ad28.pth')
        model.load_state_dict(pre_dict, strict=False)
    return model
    # if pretrained:
    #     settings = pretrained_settings['xception'][pretrained]
    #     assert num_classes == settings['num_classes'], \
    #         "num_classes should be {}, but is {}".format(settings['num_classes'], num_classes)
    #
    #     model = Xception(num_classes=num_classes)
    #     model.load_state_dict(model_zoo.load_url(settings['url']))
    #
    #     model.input_space = settings['input_space']
    #     model.input_size = settings['input_size']
    #     model.input_range = settings['input_range']
    #     model.mean = settings['mean']
    #     model.std = settings['std']
    #
    # # TODO: ugly
    # model.last_linear = model.fc
    # del model.fc
    # return model


if __name__ == '__main__':
    model = xception_attn(pretrained=False)
    data = torch.randn(1, 3, 512, 512)
    low_level_feature, x = model(data)
    print(f'low_level_feature.shape = {low_level_feature.shape}, x.shape = {x.shape}')
