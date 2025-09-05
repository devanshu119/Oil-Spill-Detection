# file: models/deeplab_model.py
import torch
import torch.nn as nn
import torchvision.models.segmentation as seg_models


def create_deeplabv3(n_classes: int, n_channels: int = 3):
    """
    Create a DeepLabV3-ResNet50 model with adjustable number of input channels and output classes.
    """
    model = seg_models.deeplabv3_resnet50(pretrained=False, progress=True)

    # adjust input channels
    if n_channels != 3:
        # Replace first conv layer to accept n_channels instead of 3
        old_conv = model.backbone.conv1
        new_conv = nn.Conv2d(n_channels, old_conv.out_channels,
                             kernel_size=old_conv.kernel_size,
                             stride=old_conv.stride,
                             padding=old_conv.padding,
                             bias=old_conv.bias is not None)
        # initialize new conv weights
        if n_channels == 1:
            new_conv.weight.data = old_conv.weight.data.mean(dim=1, keepdim=True)
        else:
            nn.init.kaiming_normal_(new_conv.weight, mode='fan_out', nonlinearity='relu')
        model.backbone.conv1 = new_conv

    # adjust classifier head for n_classes
    model.classifier[4] = nn.Conv2d(256, n_classes, kernel_size=(1, 1))

    # add attributes for compatibility with UNet wrapper
    model.n_classes = n_classes
    model.n_channels = n_channels
    return model
