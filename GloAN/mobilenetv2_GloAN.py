import os

import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo

BatchNorm2d = nn.BatchNorm2d

# def conv_bn(inp, oup, stride):
#     return nn.Sequential(
#         nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
#         BatchNorm2d(oup),
#         nn.ReLU6(inplace=True)
#     )
#
# def conv_1x1_bn(inp, oup):
#     return nn.Sequential(
#         nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
#         BatchNorm2d(oup),
#         nn.ReLU6(inplace=True)
#     )
#
# class InvertedResidual(nn.Module):
#     def __init__(self, inp, oup, stride, expand_ratio):
#         super(InvertedResidual, self).__init__()
#         self.stride = stride
#         assert stride in [1, 2]
#
#         hidden_dim = round(inp * expand_ratio)
#         self.use_res_connect = self.stride == 1 and inp == oup
#
#         if expand_ratio == 1:
#             self.conv = nn.Sequential(
#                 # dw
#                 nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
#                 BatchNorm2d(hidden_dim),
#                 nn.ReLU6(inplace=True),
#                 # pw-linear
#                 nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
#                 BatchNorm2d(oup),
#             )
#         else:
#             self.conv = nn.Sequential(
#                 # pw
#                 nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
#                 BatchNorm2d(hidden_dim),
#                 nn.ReLU6(inplace=True),
#                 # dw
#                 nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
#                 BatchNorm2d(hidden_dim),
#                 nn.ReLU6(inplace=True),
#                 # pw-linear
#                 nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
#                 BatchNorm2d(oup),
#             )
#
#     def forward(self, x):
#         if self.use_res_connect:
#             return x + self.conv(x)
#         else:
#             return self.conv(x)
#
#
# class MobileNetV2(nn.Module):
#     def __init__(self, n_class=1000, input_size=224, width_mult=1.):
#         super(MobileNetV2, self).__init__()
#         block = InvertedResidual
#         input_channel = 32
#         last_channel = 1280
#         interverted_residual_setting = [
#             # t, c, n, s
#             [1, 16, 1, 1],
#             [6, 24, 2, 2],
#             [6, 32, 3, 2],
#             [6, 64, 4, 2],
#             [6, 96, 3, 1],
#             [6, 160, 3, 2],
#             [6, 320, 1, 1],
#         ]
#
#         # building first layer
#         assert input_size % 32 == 0
#         input_channel = int(input_channel * width_mult)
#         self.last_channel = int(last_channel * width_mult) if width_mult > 1.0 else last_channel
#         self.features = [conv_bn(3, input_channel, 2)]
#         # building inverted residual blocks
#         for t, c, n, s in interverted_residual_setting:
#             output_channel = int(c * width_mult)
#             for i in range(n):
#                 if i == 0:
#                     self.features.append(block(input_channel, output_channel, s, expand_ratio=t))
#                 else:
#                     self.features.append(block(input_channel, output_channel, 1, expand_ratio=t))
#                 input_channel = output_channel
#         # building last several layers
#         self.features.append(conv_1x1_bn(input_channel, self.last_channel))
#         # make it nn.Sequential
#         self.features = nn.Sequential(*self.features)
#
#         # building classifier
#         self.classifier = nn.Sequential(
#             nn.Dropout(0.2),
#             nn.Linear(self.last_channel, n_class),
#         )
#
#         self._initialize_weights()
#
#     def forward(self, x):
#         x = self.features(x)
#         x = x.mean(3).mean(2)
#         x = self.classifier(x)
#         return x
#
#     def _initialize_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#                 m.weight.data.normal_(0, math.sqrt(2. / n))
#                 if m.bias is not None:
#                     m.bias.data.zero_()
#             elif isinstance(m, BatchNorm2d):
#                 m.weight.data.fill_(1)
#                 m.bias.data.zero_()
#             elif isinstance(m, nn.Linear):
#                 n = m.weight.size(1)
#                 m.weight.data.normal_(0, 0.01)
#                 m.bias.data.zero_()


# mobilenetv2 that can apply transfer learning
def _make_divisible(ch, divisor=8, min_ch=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_ch is None:
        min_ch = divisor
    new_ch = max(min_ch, int(ch + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_ch < 0.9 * ch:
        new_ch += divisor
    return new_ch


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2  # padding = 1 if kernel_size = 3, padding = 0 if kernel_size = 1
        self.stride = stride
        self.kernel_size = kernel_size
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU6(inplace=True)
        )


class InvertedResidual(nn.Module):
    def __init__(self, in_channel, out_channel, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        hidden_channel = in_channel * expand_ratio
        self.stride = stride
        self.use_shortcut = stride == 1 and in_channel == out_channel

        layers = []
        if expand_ratio != 1:
            # 1x1 pointwise conv
            layers.append(ConvBNReLU(in_channel, hidden_channel, kernel_size=1))
        layers.extend([
            # 3x3 depthwise conv
            ConvBNReLU(hidden_channel, hidden_channel, stride=stride, groups=hidden_channel),
            # 1x1 pointwise conv(linear)
            nn.Conv2d(hidden_channel, out_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channel),
        ])

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_shortcut:
            return x + self.conv(x)
        else:
            return self.conv(x)


# all hyperparameters adopt default value
class MobileNetV2(nn.Module):
    def __init__(self, n_class=1000, alpha=1.0, round_nearest=8):  # round_nearest makes sure layer channels are divisible by 8
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        input_channel = _make_divisible(32 * alpha, round_nearest)
        last_channel = _make_divisible(1280 * alpha, round_nearest)

        inverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        features = []
        # conv1 layer
        features.append(ConvBNReLU(3, input_channel, stride=2))
        # building inverted residual residual blockes
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * alpha, round_nearest)  # 这行仅为了使output_channel能整除8
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel
        # building last several layers
        features.append(ConvBNReLU(input_channel, last_channel, 1))
        # combine feature layers
        self.features = nn.Sequential(*features)

        # building classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(last_channel, n_class)
        )

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def load_url(url, model_dir='./model_data', map_location=None):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    filename = url.split('/')[-1]
    cached_file = os.path.join(model_dir, filename)
    if os.path.exists(cached_file):
        return torch.load(cached_file, map_location=map_location)
    else:
        return model_zoo.load_url(url, model_dir=model_dir)


def mobilenetv2(pretrained=True, **kwargs):
    model = MobileNetV2(n_class=1000, **kwargs)
    if pretrained:
        # model.load_state_dict(load_url('http://sceneparsing.csail.mit.edu/model/pretrained_resnet/mobilenet_v2.pth.tar'), strict=False)
        model.load_state_dict(torch.load('./PTH/mobilenet_v2.pth'), strict=False)
    return model


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
        x = self.mlp(x).transpose(2, 1).reshape(B, 64, H, W)   # x.shape=(1,64,128,128)

        return torch.mean(x, dim=1, keepdim=True)


class MobileNetV2_attn(nn.Module):
    def __init__(self, downsample_factor=16, pretrained=True):
        super(MobileNetV2_attn, self).__init__()
        from functools import partial

        model = mobilenetv2(pretrained)
        # print(model)
        self.features = model.features[:-1]

        self.total_idx = len(self.features)  # 18
        self.down_idx = [2, 4, 7, 14]
        # input resolution: 512
        self.attn1 = GloAN(in_dim=24, height=128, width=128)
        self.attn2 = GloAN(in_dim=32, height=64, width=64)
        self.attn3 = GloAN(in_dim=64, height=32, width=32)
        self.attn4 = GloAN(in_dim=320, height=32, width=32)
        # input resolution: 224
        # self.attn1 = GloAN(in_dim=24, height=56, width=56)
        # self.attn2 = GloAN(in_dim=32, height=28, width=28)
        # self.attn3 = GloAN(in_dim=64, height=14, width=14)
        # self.attn4 = GloAN(in_dim=320, height=14, width=14)

        if downsample_factor == 8:
            for i in range(self.down_idx[-2], self.down_idx[-1]):
                self.features[i].apply(
                    partial(self._nostride_dilate, dilate=2)
                )
            for i in range(self.down_idx[-1], self.total_idx):
                self.features[i].apply(
                    partial(self._nostride_dilate, dilate=4)
                )
        elif downsample_factor == 16:
            for i in range(self.down_idx[-1], self.total_idx):  # for i in range(14, 18):
                self.features[i].apply(  # nn.Module.apply(func): 在nn.Module的每个module执行func
                    partial(self._nostride_dilate, dilate=2)  # partial的作用为将函数self._nostride_dilate的dilate参数固定为2
                )


    def _nostride_dilate(self, m, dilate):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate // 2, dilate // 2)
                    m.padding = (dilate // 2, dilate // 2)
            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)

    def forward(self, x):
        # 4x
        x = self.features[:3](x)
        attn = self.attn1(x)
        x = x + attn * x
        # low_level_features
        low_level_features = self.features[3](x)
        # 8x
        x = self.features[4](low_level_features)
        attn = self.attn2(x)
        x = x + attn * x
        # 16x
        x = self.features[5:8](x)
        attn = self.attn3(x)
        x = x + attn * x
        # 32x
        x = self.features[8:](x)
        attn = self.attn4(x)
        x = x + attn * x
        return low_level_features, x
    
    
if __name__ == '__main__':
    net = MobileNetV2_attn(pretrained=False)
    data = torch.randn(1, 3, 512, 512)
    out = net(data)
    print(out[0].shape, out[1].shape)