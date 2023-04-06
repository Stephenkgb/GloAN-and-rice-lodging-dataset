import torch
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
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
        # x += self.pos_embed.squeeze()   # for torchstat
        x = self.dropout(x)
        x = self.norm2(x)
        q, k, v = self.qkv(x).chunk(3, dim=-1)   # q.shape=k.shape=v.shape=(1,16384,64)
        attn = q @ k.transpose(-2, -1) * 0.125   # attn.shape=(1,16384,16384)
        attn = attn.softmax(dim=-1)
        x = attn @ v  # shape=(1,3136,2)  # x.shape=(1,16384,64)
        x = self.proj(x)  # x.shape=(1,16384,64)
        x = self.mlp(x).transpose(2, 1).reshape(B, 64, H, W)   # x.shape=(1,64,128,128)
        # x = self.mlp(x).transpose(0, 1).reshape(B, 64, H, W)   # for torchstat

        return torch.mean(x, dim=1, keepdim=True)


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class ChannelGloAN(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelGloAN, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // 16, 1, bias=False),
                                nn.ReLU(),
                                nn.Conv2d(in_planes // 16, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialGloAN(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialGloAN, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, resolution=128):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.attn = GloAN(in_dim=planes, height=resolution, width=resolution)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        attn = self.attn(out)
        out = out + attn * out

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)

        self.ca = ChannelGloAN(planes * 4)
        self.sa = SpatialGloAN()

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out = self.ca(out) * out
        out = self.sa(out) * out

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        # self.resolution = [128, 64, 32, 16]  # 512
        self.resolution = [56, 28, 14, 7]  # 224
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], resolution=self.resolution[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, resolution=self.resolution[1])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, resolution=self.resolution[2])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, resolution=self.resolution[3])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, resolution=128):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, resolution=resolution))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, resolution=resolution))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)   # shape=(1,64,128,128)

        low_level_feature = self.layer1(x)  # shape=(1,64,128,128)
        x = self.layer2(low_level_feature)  # shape=(1,128,64,64)
        x = self.layer3(x)  # shape=(1,256,32,32)
        output = self.layer4(x)  # shape=(1,512,16,16)

        # x = self.avgpool(x)
        # x = x.view(x.size(0), -1)
        # x = self.fc(x)

        return low_level_feature, output


def resnet18_attn(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        # pretrained_state_dict = model_zoo.load_url(model_urls['resnet18'])
        pretrained_state_dict = torch.load('./PTH/resnet18.pth')
        # now_state_dict = model.state_dict()
        # now_state_dict.update(pretrained_state_dict)
        # model.load_state_dict(now_state_dict)
        model.load_state_dict(pretrained_state_dict, strict=False)
    return model





if __name__ == '__main__':
    net = resnet18_attn(pretrained=False)
    data = torch.randn(1, 3, 512, 512)
    out = net(data)
    print(out[0].shape, out[1].shape)
