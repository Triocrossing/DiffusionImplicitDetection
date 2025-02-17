import torch.nn as nn
import torch.utils.model_zoo as model_zoo

__all__ = ["ResNet", "resnet18", "resnet34", "resnet50", "resnet101", "resnet152"]

model_urls = {
    "resnet18": "https://download.pytorch.org/models/resnet18-5c106cde.pth",
    "resnet34": "https://download.pytorch.org/models/resnet34-333f7ec4.pth",
    "resnet50": "https://download.pytorch.org/models/resnet50-19c8e357.pth",
    "resnet101": "https://download.pytorch.org/models/resnet101-5d3b4d8f.pth",
    "resnet152": "https://download.pytorch.org/models/resnet152-b121ed2d.pth",
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
    )


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False):
        super().__init__()
        self.inplanes = 64
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        layers.extend(block(self.inplanes, planes) for _ in range(1, blocks))
        return nn.Sequential(*layers)

    def forward(self, x, *args):
        if x.shape[0] == 256:
            x = x.squeeze(1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls["resnet18"]))
    return model


def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls["resnet34"]))
    return model


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls["resnet50"]))
    return model


def resnet18feat(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    model.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=1, padding=3, bias=False)

    # Adjust the max pool layer to match the new size
    model.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls["resnet18"]))
    model.fc = nn.Sequential(
        nn.Linear(512, 512),
        nn.Dropout(0.9),
        nn.Linear(512, model.num_classes),
    )
    return model


# WIP
def resnet50feat(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)

    model.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
    # Adjust the max pool layer to match the new size
    model.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls["resnet50"]))

    if True:
        model.fc = nn.Sequential(
            nn.Linear(512 * Bottleneck.expansion, 512 * Bottleneck.expansion),
            nn.Dropout(0.9),
            nn.Linear(512 * Bottleneck.expansion, model.num_classes),
        )
    return model


def resnet50patch(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)

    model.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
    # Adjust the max pool layer to match the new size
    model.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls["resnet50"]))

    if True:
        model.fc = nn.Sequential(
            nn.Linear(512 * Bottleneck.expansion, 512 * Bottleneck.expansion),
            nn.Dropout(0.7),
            nn.Linear(512 * Bottleneck.expansion, model.num_classes),
        )
    return model


def resnet50feattrip(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)

    model.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
    # Adjust the max pool layer to match the new size
    model.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls["resnet50"]))

    if True:
        model.fc = nn.Sequential(
            nn.Linear(512 * Bottleneck.expansion, 512 * Bottleneck.expansion),
            nn.Dropout(0.0),
            nn.Linear(512 * Bottleneck.expansion, 512 * Bottleneck.expansion),
            # nn.Linear(512 * Bottleneck.expansion, model.num_classes),
        )
    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls["resnet101"]))
    return model


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls["resnet152"]))
    return model


class CustomResNet50(nn.Module):
    def __init__(self, original_resnet50, dropout=0.6):
        super(CustomResNet50, self).__init__()
        # Initialize the 3D convolutional layer to process the input tensor
        # This layer will reduce the 10 channels to 3, which ResNet expects
        self.conv3d = nn.Conv3d(
            in_channels=10,
            out_channels=3,
            kernel_size=(3, 3, 3),
            stride=(1, 1, 1),
            padding=(1, 1, 1),
        )

        # Adaptive average pooling to reduce the depth dimension from 12 to 1
        self.pool = nn.AdaptiveAvgPool3d((1, None, None))
        # self.pool = nn.AdaptiveMaxPool3d((1, None, None))

        # Original ResNet-50 model
        self.original_resnet50 = original_resnet50
        print("dropout", dropout)
        if True:
            self.original_resnet50.fc = nn.Sequential(
                nn.Linear(512 * Bottleneck.expansion, 512 * Bottleneck.expansion),
                nn.Dropout(dropout),
                nn.Linear(
                    512 * Bottleneck.expansion, self.original_resnet50.num_classes
                ),
            )

        # Modify the first convolution layer of the ResNet-50 model to accept the output of the 3D conv layer
        # No need to modify if the output of conv3d is set to 3 channels
        # self.original_resnet50.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

    def forward(self, x):
        # Apply 3D convolution and pooling
        x = self.conv3d(x)  # Shape after conv3d: [64, 3, 12, 64, 64]
        x = self.pool(x)  # Shape after pooling: [64, 3, 1, 64, 64]

        # Remove the extra depth dimension
        x = x.squeeze(2)  # Shape: [64, 3, 64, 64]

        # Pass the tensor through the original ResNet-50 model
        x = self.original_resnet50(x)
        return x


def resnet_50_cfg(pretrained=False, dropout=0.6 , **kwargs):
    # Constructs a ResNet-50 model
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls["resnet50"]))

    # Wrap the original ResNet model with the custom module
    model = CustomResNet50(model, dropout)
    return model


def resnet_34_cfg(pretrained=False, **kwargs):
    # Constructs a ResNet-50 model
    model = ResNet(Bottleneck, [3, 3, 3, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls["resnet34"]))

    # Wrap the original ResNet model with the custom module
    model = CustomResNet50(model)
    return model


def resnet_18_cfg(pretrained=False, **kwargs):
    # Constructs a ResNet-50 model
    model = ResNet(Bottleneck, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls["resnet18"]))

    # Wrap the original ResNet model with the custom module
    model = CustomResNet50(model)
    return model


# model_dict = {
#     "resnet18": [resnet18, 512],
#     "resnet34": [resnet34, 512],
#     "resnet50": [resnet50, 2048],
#     "resnet101": [resnet101, 2048],
# }
