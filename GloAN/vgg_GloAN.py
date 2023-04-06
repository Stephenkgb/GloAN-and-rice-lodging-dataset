import torch
import torch.nn as nn


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
        x = self.mlp(x).transpose(1, 2).reshape(B, 64, H, W)   # x.shape=(1,64,128,128)

        return torch.mean(x, dim=1, keepdim=True)


class VGG(nn.Module):

    def __init__(self, features, num_classes=1000, init_weights=True):
        super(VGG, self).__init__()
        self.features = features
        # input resolution: 512
        self.attn1 = GloAN(in_dim=128, height=128, width=128)
        self.attn2 = GloAN(in_dim=256, height=64, width=64)
        self.attn3 = GloAN(in_dim=512, height=32, width=32)
        self.attn4 = GloAN(in_dim=512, height=32, width=32)
        # input resolution: 224
        # self.attn1 = GloAN(in_dim=128, height=56, width=56)
        # self.attn2 = GloAN(in_dim=256, height=28, width=28)
        # self.attn3 = GloAN(in_dim=512, height=14, width=14)
        # self.attn4 = GloAN(in_dim=512, height=14, width=14)

        if init_weights:
            self._initialize_weights()

    def forward(self, x):   # x.shape=(1,3,512,512)
        # 4x
        x = self.features[:14](x)  # x.shape = (1,128,128,128)
        attn = self.attn1(x)  # shape=(1,1,128,128)
        low_level_feature = x * attn + x  # shape=(1,128,128,128)
        # 8x
        x = self.features[14:24](low_level_feature)  # x.shape=(1,256,64,64)
        attn = self.attn2(x)   # shape=(1,1,64,64)
        x = x * attn + x  # shape=(1,256,64,64)
        # 16x
        x = self.features[24:34](x)  # x.shape=(1,512,32,32)
        attn = self.attn3(x)  # shape=(1,1,32,32)
        x = x * attn + x  # shape=(1,512,32,32)
        # 16_2
        x = self.features[34:-1](x)  # shape=(1,512,32,32)
        attn = self.attn4(x)  # shape=(1,1,32,32)
        x = x * attn + x  # shape=(1,512,32,32)

        return low_level_feature, x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfgs = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def _vgg(cfg, batch_norm, pretrained, **kwargs):
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfgs[cfg], batch_norm=batch_norm), **kwargs)
    if pretrained:
        state_dict = torch.load('./PTH/vgg16_bn.pth')
        model.load_state_dict(state_dict, strict=False)
    return model


def vgg16_attn(pretrained=False, **kwargs):
    r"""VGG 16-layer model (configuration "D") with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('D', True, pretrained, **kwargs)


if __name__ == '__main__':
    net = vgg16_attn(False)
    data = torch.randn(1, 3, 512, 512)
    low_level_feature, x = net(data)
    print(f'low_level_feature.shape = {low_level_feature.shape}\noutput.shape = {x.shape}')